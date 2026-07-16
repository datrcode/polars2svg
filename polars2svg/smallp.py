import polars as pl
import time
import random

import polars2svg
from polars2svg.export import ExportMixin

#
# Small Multiples
#
class Smallp(ExportMixin):

    _VALID_KWARGS = frozenset({
        'df', 'category_by', 'sm_template',
        'order', 'descending', 'grid_mode', 'wxh', 'insets',
        'include_all', 'collate_remainder', 'use_lazy_execution',
        'sketch_only', 'draw_labels', 'draw_border', 'txt_h', 'cycle_by',
    })

    def __init__(self, *args, **kwargs):
        self.t_start        = time.time()
        self.p2s            = polars2svg.Polars2SVG()
        self.timing_metrics = {}
        self.gatherMetrics(self.__parseInput__, *args, **kwargs)
        self.gatherMetrics(self.__validateInput__)
        if self.df is not None:
            rand_id = random.randint(0, 2**32)  # nosec B311 - non-cryptographic SVG id scoping, see SECURITY.md
            self.gatherMetrics(self.__addColumnsToDataFrame__)
            self.gatherMetrics(self.__computeOrderingStats__)
            self.gatherMetrics(self.__constructGeometry__)
            if not self.sketch_only:
                self.gatherMetrics(self.__renderSVG__, rand_id)
        # trim verbose float tails from the finished SVG (idempotent; no-op on the
        # dataless placeholder) -- see Polars2SVG.roundSvgFloats. Child panels are
        # already rounded (each component rounds in its own __init__); the outer
        # <g transform> coordinates are rounded here.
        self.svg = self.p2s.roundSvgFloats(self.svg)
        self.t_end     = time.time()
        self.t_overall = self.t_end - self.t_start

    def _repr_svg_(self): return self.svg

    #
    # gpuDisplayList() / webgpu() - WebGPU representation composed from each cell
    # component's own display list, translated to its tile position (mirrors the
    # SVG <g transform="translate(...)"> assembly).  Lazy + cached.
    #
    def gpuDisplayList(self):
        if self.df is None or getattr(self, '_render_lu_', None) is None: return None
        if getattr(self, '_gpu_dl_', None) is not None: return self._gpu_dl_
        from polars2svg.p2s_displaylist import DisplayList
        tmpl_w, tmpl_h = self.sm_template.wxh
        _w_, _h_       = self.wxh_actual
        _bg_color_     = self.p2s.colorTyped('background', 'default')
        _label_color_  = self.p2s.colorTyped('label',      'defaultfg')
        _axis_color_   = self.p2s.colorTyped('axis',       'default')
        _dl_ = DisplayList(_w_, _h_, bg=_bg_color_)
        _dl_.rect(0, 0, _w_, _h_, _bg_color_)
        for _tuple_, _xy_ in self.category_to_xy.items():
            _rendering_ = self._render_lu_.get(_tuple_)
            _cell_dl_   = _rendering_.gpuDisplayList() if (_rendering_ is not None and
                                                           getattr(_rendering_, 'gpuDisplayList', None) is not None) else None
            if _cell_dl_ is not None:
                _dl_.extend(_cell_dl_, offset=_xy_)
            if self.draw_border:
                _x0_, _y0_, _x1_, _y1_ = _xy_[0], _xy_[1], _xy_[0] + tmpl_w, _xy_[1] + tmpl_h
                _dl_.line(_x0_, _y0_, _x1_, _y0_, _axis_color_, width=0.5)
                _dl_.line(_x0_, _y1_, _x1_, _y1_, _axis_color_, width=0.5)
                _dl_.line(_x0_, _y0_, _x0_, _y1_, _axis_color_, width=0.5)
                _dl_.line(_x1_, _y0_, _x1_, _y1_, _axis_color_, width=0.5)
            if self.draw_labels:
                _dl_.text(self.p2s, self._label_for_tuple_(_tuple_), _xy_[0] + tmpl_w/2,
                          _xy_[1] + tmpl_h + self.txt_h,
                          color=_label_color_, txt_h=self.txt_h, anchor='middle', svg='')
        self._gpu_dl_ = _dl_
        return _dl_

    def webgpu(self):
        if getattr(self, '_gpu_payload_', None) is not None: return self._gpu_payload_
        _dl_ = self.gpuDisplayList()
        if _dl_ is None: return None
        self._gpu_payload_ = _dl_.webgpu_payload(self.p2s.glyphAtlas())
        return self._gpu_payload_

    def render_with_df(self, df):
        return Smallp(df, self.sm_template,
                      category_by        = self.category_by,
                      cycle_by           = self.cycle_by,
                      wxh                = self.wxh,
                      insets             = self.insets,
                      grid_mode          = self.grid_mode,
                      order              = self.order,
                      descending         = self.descending,
                      include_all        = self.include_all,
                      collate_remainder  = self.collate_remainder,
                      use_lazy_execution = self.use_lazy_execution,
                      sketch_only        = False,
                      draw_labels        = self.draw_labels,
                      draw_border        = self.draw_border,
                      txt_h              = self.txt_h)

    def gatherMetrics(self, callable, *args, **kwargs):
        t0 = time.time()
        _results_ = callable(*args, **kwargs)
        t1 = time.time()
        if callable.__name__ not in self.timing_metrics: self.timing_metrics[callable.__name__] = 0.0
        self.timing_metrics[callable.__name__] += t1 - t0
        return _results_

    def _label_for_tuple_(self, _tuple_):
        if   _tuple_ == '__remainder__': return 'Remainder'
        elif _tuple_ == '__all__':       return 'All'
        else:
            _as_strs_ = [str(x) for x in _tuple_]
            return '|'.join(_as_strs_) if len(_tuple_) > 1 else str(_tuple_[0])

    def _validate_tfield_column_(self, field):
        _column_, _ = self.p2s.tFieldTuple(field)
        _types_          = self.p2s.tFieldAccepts(field)
        if not any(isinstance(self.df.dtypes[self.df.columns.index(_column_)], t) for t in _types_):
            raise ValueError(f'Smallp.__validateInput__(): the column {_column_} is not of type {_types_} for t-field {field}')

    def __parseInput__(self, *args, **kwargs):
        _unknown_ = set(kwargs) - self._VALID_KWARGS
        if _unknown_:
            raise TypeError(f'Smallp: unexpected keyword argument(s): {sorted(_unknown_)}')

        # Single source of truth for smallp's simple kwargs.get()-style parameters
        # (name -> default). df / category_by / sm_template arrive positionally and
        # are handled explicitly, so they are the spec's `extra` names.
        _defaults_ = {
            'order':              self.p2s.ROW_COUNTp,
            'descending':         True,
            'grid_mode':          False,
            'wxh':                (1280, 1024),
            'insets':             (0, 0),
            'include_all':        False,
            'collate_remainder':  True,
            'use_lazy_execution': False,
            'sketch_only':        False,
            'draw_labels':       True,
            'draw_border':        True,
            'txt_h':              12,
            'cycle_by':           None,
        }
        self.p2s.assertParamSpecMatches('Smallp', self._VALID_KWARGS, _defaults_,
                                        extra=('df', 'category_by', 'sm_template'))

        self.df, self.df_orig, self.sm_template, self.category_by = None, None, None, None

        def __isColumn__(df, col):
            # t-field-aware column membership -- delegates to the shared helper so the
            # (isTField -> tFieldTuple -> base-column) logic lives in exactly one place.
            return self.p2s.columnInDataFrame(col, df)

        def __tupleElementsAreColumns__(df, _tuple_):
            return all(__isColumn__(df, _tuple_[i]) for i in range(len(_tuple_)))

        def __elementsAreDataFrames__(_list_):
            return all(isinstance(_list_[i], pl.DataFrame) for i in range(len(_list_)))

        # Extract the dataframe first
        for arg in args:
            if isinstance(arg, pl.DataFrame):
                if self.df is None: self.df = self.df_orig = arg
                else:               raise ValueError('Smallp.__parseInput__(): df already set (2)')
        if 'df' in kwargs:
            if self.df is None: self.df = self.df_orig = kwargs['df']
            else:               raise ValueError('Smallp.__parseInput__(): df already set (3)')
        if self.df is not None and '__p2s_index__' not in self.df.columns:
            self.df = self.df.with_row_index('__p2s_index__')

        # Then the rest of the arguments
        for arg in args:
            if   isinstance(arg, pl.DataFrame):
                pass
            elif self.p2s.isTemplate(arg) and self.sm_template is None:
                self.sm_template = arg
            elif isinstance(arg, str) and __isColumn__(self.df, arg):
                if self.category_by is not None: raise ValueError('Smallp.__parseInput__(): category_by already set (2)')
                self.category_by = arg
            elif isinstance(arg, tuple) and __tupleElementsAreColumns__(self.df, arg):
                if self.category_by is not None: raise ValueError('Smallp.__parseInput__(): category_by already set (3)')
                self.category_by = arg
            elif isinstance(arg, list) and __elementsAreDataFrames__(arg):
                if self.category_by is not None: raise ValueError('Smallp.__parseInput__(): category_by already set (4)')
                self.category_by = arg
            elif isinstance(arg, dict) and __elementsAreDataFrames__(list(arg.values())):
                if self.category_by is not None: raise ValueError('Smallp.__parseInput__(): category_by already set (5)')
                self.category_by = arg
            else:
                raise ValueError('Smallp.__parseInput__(): Unknown argument type: ' + str(type(arg)))

        if self.category_by is not None and 'category_by' in kwargs: raise ValueError('Smallp.__parseInput__(): category_by already set (6)')
        if self.sm_template is not None and 'sm_template'  in kwargs: raise ValueError('Smallp.__parseInput__(): sm_template already set')

        if 'category_by' in kwargs: self.category_by = kwargs['category_by']
        if 'sm_template'  in kwargs: self.sm_template  = kwargs['sm_template']

        # unconditional on purpose: sm_template is a panel template (a resolved
        # sibling component), not a Smallp clone source — smallp itself has no
        # template snapshot to preserve, so current defaults always apply
        kwargs = self.p2s._apply_defaults('smallp', kwargs)
        self.p2s.assignKwargsWithDefaults(self, _defaults_, kwargs)

        # Normalize to a canonical (w, h) tuple; smallp permits one dimension to be
        # None so the missing side can be auto-sized from the panels later.
        self.wxh = self.p2s.normalizeWxh(self.wxh, 'Smallp', allow_none=True)

        if self.sm_template is None: raise ValueError('Smallp.__parseInput__(): No template specified')
        if self.category_by is None and self.cycle_by is None:
            raise ValueError('Smallp.__parseInput__(): Neither category_by nor cycle_by specified')
        if self.category_by is not None and self.cycle_by is not None:
            raise ValueError('Smallp.__parseInput__(): category_by and cycle_by are mutually exclusive')

        # "No data" placeholder for early error visibility -- smallp only assigns
        # self.svg inside __renderSVG__, which __init__ skips when no df is supplied,
        # so without this a dataless build has no .svg at all (AttributeError on
        # repr). A successful render overwrites it. wxh dims may still be None here
        # (auto-sizing resolves later); placeholderSVG falls back to a square.
        _w_, _h_ = self.wxh
        self.svg = self.p2s.placeholderSVG(_w_, _h_)

    def __validateInput__(self):
        self.p2s.checkReservedColumns(self.df, 'Smallp')
        if self.cycle_by is not None:
            if not isinstance(self.cycle_by, dict) or len(self.cycle_by) == 0:
                raise ValueError('Smallp.__validateInput__(): cycle_by must be a non-empty dict')
            _lengths_ = [len(v) for v in self.cycle_by.values()]
            if len(set(_lengths_)) != 1:
                raise ValueError('Smallp.__validateInput__(): cycle_by value lists must all be the same length')

        if self.grid_mode:
            if not isinstance(self.category_by, tuple) or len(self.category_by) != 2:
                raise ValueError('Smallp.__validateInput__(): category_by must be a tuple with exactly two fields when grid_mode is True')

        # wxh was normalized to a canonical (w, h) tuple (one dimension may be None
        # for auto-sizing) in __parseInput__ via self.p2s.normalizeWxh(); no
        # re-validation needed here.

        if self.df is not None:
            if isinstance(self.category_by, str) and self.p2s.isTField(self.category_by, df=self.df):
                self._validate_tfield_column_(self.category_by)
            elif isinstance(self.category_by, tuple):
                for field in self.category_by:
                    if isinstance(field, str) and self.p2s.isTField(field, df=self.df):
                        self._validate_tfield_column_(field)

    def __addColumnsToDataFrame__(self):
        _ops_ = []
        def __addOp__(field):
            self.p2s.warnIfTFieldAliasCollides(field, self.df, 'Smallp')
            _column_, _enum_ = self.p2s.tFieldTuple(field)
            _ops_.append(self.p2s.polarsOperationForEnum(_column_, _enum_).alias(field))

        if isinstance(self.category_by, str) and self.p2s.isTField(self.category_by, df=self.df):
            __addOp__(self.category_by)
        elif isinstance(self.category_by, tuple):
            for field in self.category_by:
                if isinstance(field, str) and self.p2s.isTField(field, df=self.df):
                    __addOp__(field)

        if _ops_:
            if self.use_lazy_execution: self.df = self.df.lazy().with_columns(_ops_).collect()
            else:                       self.df = self.df.with_columns(_ops_)

    def __orderAggExpr__(self):
        if self.order == self.p2s.ROW_COUNTp:
            return pl.len().alias('__order_metric__')
        elif isinstance(self.order, str):
            return pl.col(self.order).sum().alias('__order_metric__')
        elif isinstance(self.order, tuple):
            _field_ = self.order[0]
            if len(self.order) == 1:
                return pl.col(_field_).sum().alias('__order_metric__')
            elif isinstance(self.order[1], str):
                return pl.struct([self.order[0], self.order[1]]).n_unique().alias('__order_metric__')
            elif self.order[1] == self.p2s.SETp:
                return pl.col(_field_).n_unique().alias('__order_metric__')
            elif self.order[1] in self.p2s.statistic_types:
                _stat_map_ = {
                    self.p2s.MINp:    pl.col(_field_).min(),
                    self.p2s.MEDIANp: pl.col(_field_).median(),
                    self.p2s.MEANp:   pl.col(_field_).mean(),
                    self.p2s.MAXp:    pl.col(_field_).max(),
                    self.p2s.STDp:    pl.col(_field_).std(),
                    self.p2s.SUMp:    pl.col(_field_).sum(),
                }
                return _stat_map_[self.order[1]].alias('__order_metric__')
        return pl.len().alias('__order_metric__')

    def __computeOrderingStats__(self):
        if isinstance(self.category_by, (str, tuple)):
            _cat_cols_ = [self.category_by] if isinstance(self.category_by, str) else list(self.category_by)
            _agg_df_   = self.df.group_by(_cat_cols_).agg(self.__orderAggExpr__())
            # Two-level sort matching original: pre-sort by string repr (stable), then sort by metric
            _str_aliases_ = [f'__s_{c}__' for c in _cat_cols_]
            _agg_df_ = (_agg_df_
                .with_columns([pl.col(c).cast(pl.String).alias(a) for c, a in zip(_cat_cols_, _str_aliases_)])
                .sort(_str_aliases_,        descending=False,          maintain_order=True)
                .sort(['__order_metric__'],  descending=self.descending, maintain_order=True)
                .drop(_str_aliases_))
            self._sorted_category_keys_ = [row for row in _agg_df_.select(_cat_cols_).iter_rows()]
        elif isinstance(self.category_by, list):
            self._sorted_category_keys_ = [str(i) for i in range(len(self.category_by))]
        elif isinstance(self.category_by, dict):
            self._sorted_category_keys_ = list(self.category_by.keys())
        elif self.cycle_by is not None:
            _param_names_  = list(self.cycle_by.keys())
            _values_lists_ = list(self.cycle_by.values())
            _n_ = len(_values_lists_[0])
            self._sorted_category_keys_ = [
                tuple(_values_lists_[j][i] for j in range(len(_values_lists_)))
                for i in range(_n_)
            ]
            self._cycle_override_ = {
                key: {_param_names_[j]: key[j] for j in range(len(_param_names_))}
                for key in self._sorted_category_keys_
            }
        else:
            raise ValueError('Smallp.__computeOrderingStats__(): Unknown category_by type: ' + str(type(self.category_by)))
        self._num_categories_ = len(self._sorted_category_keys_)

    def __filterForKey__(self, key):
        if isinstance(self.category_by, str):
            return self.df.filter(pl.col(self.category_by) == key[0])
        else:
            _cat_cols_ = list(self.category_by)
            _struct_   = pl.struct([pl.col(c) for c in _cat_cols_])
            _target_   = {c: v for c, v in zip(_cat_cols_, key)}
            return self.df.filter(_struct_.is_in([_target_]))

    def __constructGeometry__(self):
        _tiles_needed_ = self._num_categories_
        if self.include_all: _tiles_needed_ += 1
        tmpl_w, tmpl_h = self.sm_template.wxh
        tmpl_h_adj = tmpl_h + self.txt_h + 3 if self.draw_labels else tmpl_h
        x_ins,  y_ins  = self.insets
        _w_,    _h_    = self.wxh
        _tiles_down_, _tiles_across_ = None, None
        _w_leftover_, _h_leftover_   = None, None
        if isinstance(_w_, int):
            _tiles_across_ = (_w_ - x_ins) // (tmpl_w + x_ins)
            _w_leftover_   = (_w_ - x_ins) %  (tmpl_w + x_ins)
        if isinstance(_h_, int):
            _tiles_down_   = (_h_ - y_ins) // (tmpl_h_adj + y_ins)
            _h_leftover_   = (_h_ - y_ins) %  (tmpl_h_adj + y_ins)
        if _w_ is None:
            _tiles_across_   = _tiles_needed_ // _tiles_down_
            _tiles_leftover_ = _tiles_needed_ %  _tiles_down_
            if _tiles_leftover_ > 0: _tiles_across_ += 1
            _w_ = _tiles_across_ * (tmpl_w + x_ins) + x_ins
        if _h_ is None:
            _tiles_down_     = _tiles_needed_ // _tiles_across_
            _tiles_leftover_ = _tiles_needed_ %  _tiles_across_
            if _tiles_leftover_ > 0: _tiles_down_ += 1
            _h_ = _tiles_down_ * (tmpl_h_adj + y_ins) + y_ins
        _needs_remainder_ = _tiles_across_ * _tiles_down_ < _tiles_needed_
        self.wxh_actual = (_w_, _h_)

        self.category_to_xy, self.category_to_df = {}, {}
        _x_tile_,   _y_tile_   = 0,     0
        _x_offset_, _y_offset_ = x_ins, y_ins
        def __increment__():
            nonlocal _x_tile_, _x_offset_, _y_tile_, _y_offset_
            _x_tile_   += 1
            _x_offset_ += tmpl_w + x_ins
            if _x_tile_ >= _tiles_across_:
                _x_tile_, _x_offset_ = 0, x_ins
                _y_tile_   += 1
                _y_offset_ += tmpl_h_adj + y_ins

        if self.include_all:
            self.category_to_xy['__all__'] = (_x_offset_, _y_offset_)
            self.category_to_df['__all__'] = self.df
            __increment__()

        _all_offset_   = 1 if self.include_all else 0
        _total_slots_  = _tiles_across_ * _tiles_down_
        _max_visible_  = _total_slots_ - _all_offset_ - (1 if _needs_remainder_ and self.collate_remainder else 0)
        _num_remainder_categories_ = max(0, self._num_categories_ - _max_visible_)

        self.category_by_dict = {}
        if isinstance(self.category_by, (str, tuple)):
            _cat_cols_     = [self.category_by] if isinstance(self.category_by, str) else list(self.category_by)
            _visible_keys_ = self._sorted_category_keys_[:_max_visible_]
            for key in _visible_keys_:
                _sub_df_ = self.__filterForKey__(key)
                self.category_to_xy[key]   = (_x_offset_, _y_offset_)
                self.category_to_df[key]   = _sub_df_
                self.category_by_dict[key] = _sub_df_
                __increment__()
            if _needs_remainder_ and self.collate_remainder:
                if isinstance(self.category_by, str):
                    _visible_raw_  = [k[0] for k in _visible_keys_]
                    _remainder_df_ = self.df.filter(~pl.col(self.category_by).is_in(_visible_raw_))
                else:
                    _struct_          = pl.struct([pl.col(c) for c in _cat_cols_])
                    _visible_structs_ = [{c: v for c, v in zip(_cat_cols_, k)} for k in _visible_keys_]
                    _remainder_df_    = self.df.filter(~_struct_.is_in(_visible_structs_))
                self.category_to_xy['__remainder__'] = (_x_offset_, _y_offset_)
                self.category_to_df['__remainder__'] = _remainder_df_
            else:
                self.category_to_df['__remainder__'] = None
        elif self.cycle_by is not None:
            for key in self._sorted_category_keys_:
                self.category_to_xy[key] = (_x_offset_, _y_offset_)
                self.category_to_df[key] = self.df
                __increment__()
        else:
            # list/dict: pre-split DataFrames — look up by key, concat overflow as before
            _dfs_remainder_ = []
            for key in self._sorted_category_keys_:
                v = self.category_by[int(key)] if isinstance(self.category_by, list) else self.category_by[key]
                if   _x_tile_ == _tiles_across_ - 1 and \
                     _y_tile_ == _tiles_down_   - 1 and \
                     _needs_remainder_              and \
                     self.collate_remainder:
                    self.category_to_xy['__remainder__'] = (_x_offset_, _y_offset_)
                    _dfs_remainder_ = [v]
                elif _x_tile_ < _tiles_across_ and _y_tile_ < _tiles_down_:
                    self.category_to_xy[key]   = (_x_offset_, _y_offset_)
                    self.category_to_df[key]   = v
                    self.category_by_dict[key] = v
                elif self.collate_remainder:
                    _dfs_remainder_.append(v)
                __increment__()
            self.category_to_df['__remainder__'] = pl.concat(_dfs_remainder_) if self.collate_remainder and len(_dfs_remainder_) > 0 else None

        if self.sketch_only:
            self.__printSketchInfo__(_tiles_needed_, _tiles_across_, _tiles_down_, _w_, _h_, _w_leftover_, _h_leftover_, _num_remainder_categories_)
            _bg_color_    = self.p2s.colorTyped('background', 'default')
            _axis_color_  = self.p2s.colorTyped('axis',       'default')
            _label_color_ = self.p2s.colorTyped('label',      'defaultfg')
            _svg_ = [f'<svg x="0" y="0" width="{_w_}" height="{_h_}" xmlns="http://www.w3.org/2000/svg">']
            _svg_.append(f'<rect width="{_w_}" height="{_h_}" x="0" y="0" fill="{_bg_color_}" />')
            for _tuple_, _xy_ in self.category_to_xy.items():
                _label_ = self._label_for_tuple_(_tuple_)
                if self.draw_border:
                    _svg_.append(f'<rect width="{tmpl_w}" height="{tmpl_h}" x="{_xy_[0]}" y="{_xy_[1]}" fill="none" stroke="{_axis_color_}" stroke-width="0.5"/>')
                _svg_.append(self.p2s.svgText(_label_, _xy_[0]+4, _xy_[1]+10, color=_label_color_, txt_h=8))
                _df_ = self.category_to_df[_tuple_]
                _svg_.append(self.p2s.svgText(f'df.len = {len(_df_)}', _xy_[0]+4, _xy_[1]+20, color=_label_color_, txt_h=8))
                if self.draw_labels:
                    _x_ = _xy_[0] + tmpl_w/2
                    _y_ = _xy_[1] + tmpl_h + self.txt_h
                    _svg_.append(self.p2s.svgText(_label_, _x_, _y_, color=_label_color_, txt_h=self.txt_h, anchor='middle'))
            _svg_.append('</svg>')
            self.svg = ''.join(_svg_)

    def __printSketchInfo__(self, _tiles_needed_, _tiles_across_, _tiles_down_, _w_, _h_, _w_leftover_, _h_leftover_, _num_remainder_categories_):
        _tiles_needed_str_ = f'Tiles Needed:    {_tiles_needed_}'
        _tiles_total_str_  = f'Tile Spaces:     {_tiles_across_ * _tiles_down_}'
        _tiles_across_str_ = f'Tiles Across:    {_tiles_across_}'
        _tiles_down_str_   = f'Tiles Down:      {_tiles_down_}'
        _w_str_            = f'Width:           {_w_}'
        _h_str_            = f'Height:          {_h_}'
        _xins_str_         = f'X Inset:         {self.insets[0]}'
        _yins_str_         = f'Y Inset:         {self.insets[1]}'
        _w_leftover_str_   = f'Width Leftover:  {_w_leftover_}'
        _h_leftover_str_   = f'Height Leftover: {_h_leftover_}'
        _remainder_str_    = f'Remainder DFs:   {_num_remainder_categories_}'

        tmpl_w, tmpl_h = self.sm_template.wxh
        tmpl_h_adj = tmpl_h + self.txt_h + 3 if self.draw_labels else tmpl_h

        _width_fit_  = _tiles_across_ * tmpl_w     + (_tiles_across_ + 1) * self.insets[0]
        _height_fit_ = _tiles_down_   * tmpl_h_adj + (_tiles_down_   + 1) * self.insets[1]
        _width_fit_str_  = f'Width Fit:       {_width_fit_}'
        _height_fit_str_ = f'Height Fit:      {_height_fit_}'

        _adj_tile_w_ = int((_w_ - (_tiles_across_ + 1) * self.insets[0]) // _tiles_across_)
        _adj_tile_h_ = int((_h_ - (_tiles_down_   + 1) * self.insets[1]) // _tiles_down_)
        _adj_tile_w_str_ = f'Fit Tile Width:  {_adj_tile_w_}'
        _adj_tile_h_str_ = f'Fit Tile Height: {_adj_tile_h_}'

        # sketch_only=True is an explicit, user-facing diagnostic mode whose whole
        # purpose is to surface these dimensions so the user can pick a wxh — so this
        # prints unconditionally on every call (not routed through the logger, which is
        # off by default). This is the one intentional print() in the package.
        strw = 24
        print('Sketch Info')
        print(f'{_tiles_needed_str_:<{strw}} | {_tiles_total_str_:<{strw}} | {_tiles_across_str_:<{strw}} | {_tiles_down_str_:<{strw}}')
        print(f'{_w_str_:<{strw}} | {_h_str_:<{strw}} | {_xins_str_:<{strw}} | {_yins_str_:<{strw}}')
        print(f'{_w_leftover_str_:<{strw}} | {_h_leftover_str_:<{strw}} | {_remainder_str_:<{strw}} |')
        print(f'{_width_fit_str_:<{strw}} | {_height_fit_str_:<{strw}} | {_adj_tile_w_str_:<{strw}} | {_adj_tile_h_str_:<{strw}}')

    def __renderSVG__(self, rand_id):
        tmpl_w, tmpl_h = self.sm_template.wxh
        _w_,    _h_    = self.wxh_actual
        _bg_color_     = self.p2s.colorTyped('background', 'default')
        _label_color_  = self.p2s.colorTyped('label',      'defaultfg')
        _axis_color_   = self.p2s.colorTyped('axis',       'default')
        _svg_          = [f'<svg id="smallp_{rand_id}" x="0" y="0" width="{_w_}" height="{_h_}" xmlns="http://www.w3.org/2000/svg">']
        _svg_.append(f'<rect width="{_w_}" height="{_h_}" x="0" y="0" fill="{_bg_color_}" />')
        if self.cycle_by is not None:
            _render_lu_ = {
                key: self.sm_template.render_with(self.df, **self._cycle_override_[key])
                for key in self.category_to_xy
            }
        else:
            _render_lu_ = self.sm_template.renderSmallMultiples(self.df, self.category_to_df, '__all__')
        self._render_lu_  = _render_lu_   # retained for the WebGPU composition path
        self._gpu_dl_     = None          # invalidate cached GPU state
        self._gpu_payload_ = None
        for _tuple_, _xy_ in self.category_to_xy.items():
            _label_     = self._label_for_tuple_(_tuple_)
            _rendering_ = _render_lu_[_tuple_]
            _svg_.append(f'<g transform="translate({_xy_[0]},{_xy_[1]})">{_rendering_._repr_svg_()}</g>')
            if self.draw_border:
                _svg_.append(f'<rect width="{tmpl_w}" height="{tmpl_h}" x="{_xy_[0]}" y="{_xy_[1]}" fill="none" stroke="{_axis_color_}" stroke-width="0.5"/>')
            if self.draw_labels:
                _x_ = _xy_[0] + tmpl_w/2
                _y_ = _xy_[1] + tmpl_h + self.txt_h
                _svg_.append(self.p2s.svgText(_label_, _x_, _y_, color=_label_color_, txt_h=self.txt_h, anchor='middle'))
        _svg_.append('</svg>')
        self.svg = ''.join(_svg_)
