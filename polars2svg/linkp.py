import polars as pl
import random
import time
import xml.etree.ElementTree as ET  # nosec B405 - background= shape descriptors are trusted caller config, not untrusted data; see SECURITY.md

import polars2svg
from polars2svg.p2s_displaylist import DisplayList, hexToRGBA, cubicBezierSegmentsTable
from polars2svg.export import ExportMixin
from polars2svg.exceptions import DataError, Polars2SVGError

__name__ = 'linkp'

class LinkP(ExportMixin):

    _VALID_KWARGS = frozenset({
        'template', 'df',
        'relationships', 'pos', 'view_window',
        'color', 'node_color', 'count',
        'node_size', 'node_opacity', 'node_size_range',
        'draw_labels', 'node_labels', 'label_only',
        'label_line_width', 'label_max_lines', 'label_ellipsis',
        'link_size', 'link_shape', 'link_opacity', 'link_size_range',
        'wxh', 'insets', 'bounds_percent', 'use_pos_for_bounds',
        'convex_hull_lu', 'convex_hull_opacity', 'convex_hull_labels', 'convex_hull_stroke_width',
        'background', 'background_label_color', 'background_opacity',
        'background_fill', 'background_stroke_w', 'background_stroke',
        'sm_shared', '_shared_view_x_', '_shared_view_y_',
        'count_range_shared', 'color_stat_range_shared',
        'draw_border', 'txt_h', 'legend',
    })

    #
    # __init__()
    #
    def __init__(self, *args, **kwargs):
        self.t_start        = time.time()
        self.p2s            = polars2svg.Polars2SVG()
        self.timing_metrics = {}
        self.gatherMetrics(self.__parseInput__, *args, **kwargs)
        self.gatherMetrics(self.__validateInput__)
        if self.df is not None:
            rand_id = random.randint(0, 2**32)  # nosec B311 - non-cryptographic SVG id scoping, see SECURITY.md
            self.gatherMetrics(self.__calculateGeometry__)
            self.gatherMetrics(self.__calculateScreenCoordinates__)
            self.gatherMetrics(self.__renderLinks__)
            self.gatherMetrics(self.__renderNodes__)
            self.gatherMetrics(self.__renderSVG__, rand_id)
        self.t_end     = time.time()
        self.t_overall = self.t_end - self.t_start

    def _repr_svg_(self): return self.svg

    #
    # webgpu() - WebGPU payload of the same render, extracted from the retained
    # df_link / df_node tables (no recompute of the polars pipelines).  Curve links
    # are flattened to line segments; collapsed-node clouds approximate as circles.
    # Lazy + cached; invalidated by __renderSVG__.
    #
    def webgpu(self):
        # Honor a pending relayout the same way renderSVG() does, so the
        # interactive GPU path never serves a stale payload after a node move /
        # layout op / zoom (which set _render_invalid_ but don't clear the cache).
        if getattr(self, '_render_invalid_', False): self.renderSVG()
        if getattr(self, '_gpu_payload_', None) is not None: return self._gpu_payload_
        _dl_ = self.gpuDisplayList()
        if _dl_ is None: return None
        self._gpu_payload_ = _dl_.webgpu_payload(self.p2s.glyphAtlas())
        return self._gpu_payload_

    #
    # gpuDisplayList() - the composed backend-neutral display list (also consumed
    # by smallp when this component renders as a cell)
    #
    def gpuDisplayList(self):
        if self.df is None or getattr(self, 'svg', None) is None: return None
        if getattr(self, '_render_invalid_', False): self.renderSVG()
        if getattr(self, '_gpu_dl_', None) is not None: return self._gpu_dl_
        w, h    = self.wxh
        _bg_co_ = self.p2s.colorTyped('background', 'default')
        _dl_    = DisplayList(w, h, bg=_bg_co_)
        _dl_.rect(0, 0, w, h, _bg_co_)
        # Background shapes (recorded during __renderBackground__)
        if getattr(self, '_dl_background_', None) is not None: _dl_.extend(self._dl_background_)
        # Convex hulls (recorded during __renderConvexHull__)
        if getattr(self, '_dl_hull_', None) is not None: _dl_.extend(self._dl_hull_)
        # Links
        _node_size_lu_ = {'small': 1, 'nil': 0.2, 'medium': 3, 'large': 5}
        _lk_w_ = _node_size_lu_.get(self.link_size, 1.0)
        if isinstance(self.link_size, (int, float)): _lk_w_ = float(self.link_size)
        if self.df_link is not None and self.link_size is not None:
            for i in range(len(self.relationships)):
                _fm_sx_, _fm_sy_ = f'__rel{i}_fm_sx__', f'__rel{i}_fm_sy__'
                _to_sx_, _to_sy_ = f'__rel{i}_to_sx__', f'__rel{i}_to_sy__'
                if _fm_sx_ not in self.df_link.columns: continue
                _sub_ = self.df_link.drop_nulls(subset=[_fm_sx_, _to_sx_])
                if len(_sub_) == 0: continue
                _sub_ = _sub_.with_columns(self.p2s.rgbFromHexPolarsOperations('__lc_hex__', '__r_f__', '__g_f__', '__b_f__'))
                if self.link_size == 'vary':
                    _lc_min_, _lc_max_ = self.__countMinMax__(_sub_['__count__'])
                    _sub_  = _sub_.with_columns(self.__interpolatedSizeExpr__(self.link_size_range, _lc_min_, _lc_max_).alias('__w_f__'))
                    _w_arg_ = '__w_f__'
                else:
                    _w_arg_ = _lk_w_
                if self.link_shape == 'curve':
                    _seg_ = cubicBezierSegmentsTable(_sub_, _fm_sx_, _fm_sy_,
                                                     f'__xo0{i}__', f'__yo0{i}__', f'__xo1{i}__', f'__yo1{i}__',
                                                     _to_sx_, _to_sy_)
                    _dl_.lines_table(_seg_, '__bx__', '__by__', '__bx2__', '__by2__',
                                     ('__r_f__', '__g_f__', '__b_f__'), width=_w_arg_,
                                     opacity=self.link_opacity, svg_col=None)
                else:
                    _dl_.lines_table(_sub_, _fm_sx_, _fm_sy_, _to_sx_, _to_sy_,
                                     ('__r_f__', '__g_f__', '__b_f__'), width=_w_arg_,
                                     opacity=self.link_opacity, svg_col=None)
        # Nodes
        _nsz_lu_ = {'small': 3, 'medium': 5, 'large': 7, 'nil': 0.5}
        if self.df_node is not None and self.node_size is not None and len(self.df_node) > 0:
            _dfn_ = self.df_node.with_columns(self.p2s.rgbFromHexPolarsOperations('__nc_hex__', '__r_f__', '__g_f__', '__b_f__'))
            if self.node_size == 'vary':
                _dl_.circles_table(_dfn_, '__sx__', '__sy__', '__sz__',
                                   ('__r_f__', '__g_f__', '__b_f__'), opacity=self.node_opacity,
                                   stroke=hexToRGBA('#000000', self.node_opacity), stroke_w=1.0, svg_col=None)
            else:
                _sz_ = _nsz_lu_.get(self.node_size, self.node_size) if isinstance(self.node_size, str) else float(self.node_size)
                _sw_ = 1.0 if _sz_ > 3 else _sz_ / 2.0
                _singles_ = _dfn_.filter(pl.col('__nodes__') == 1)
                _dl_.circles_table(_singles_, '__sx__', '__sy__', _sz_,
                                   ('__r_f__', '__g_f__', '__b_f__'), opacity=self.node_opacity,
                                   stroke=hexToRGBA('#000000', self.node_opacity), stroke_w=_sw_, svg_col=None)
                # Collapsed nodes (cloud icon in SVG) -- approximated as larger circles on GPU
                _multis_ = _dfn_.filter(pl.col('__nodes__') > 1)
                _dl_.circles_table(_multis_, '__sx__', '__sy__', 6.0,
                                   ('__r_f__', '__g_f__', '__b_f__'), opacity=self.node_opacity,
                                   stroke=hexToRGBA('#000000', self.node_opacity), stroke_w=0.5, svg_col=None)
        # Node labels (info recorded during __renderNodes__)
        for _sx_, _y0_, _lines_ in getattr(self, '_node_label_info_', []):
            for _li_, _line_ in enumerate(_lines_):
                _txt_ = _line_.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&')
                _dl_.text(self.p2s, _txt_, _sx_, _y0_ + _li_ * self.txt_h,
                          txt_h=self.txt_h, anchor='middle', svg='')
        # Legend (recorded during __renderSVG__)
        if getattr(self, '_dl_legend_', None) is not None: _dl_.extend(self._dl_legend_)
        # Border
        if self.draw_border:
            _border_co_ = self.p2s.colorTyped('axis', 'inner')
            _dl_.line(0, 0, w-1, 0, _border_co_, width=1.0)
            _dl_.line(0, h-1, w-1, h-1, _border_co_, width=1.0)
            _dl_.line(0, 0, 0, h-1, _border_co_, width=1.0)
            _dl_.line(w-1, 0, w-1, h-1, _border_co_, width=1.0)
        self._gpu_dl_ = _dl_
        return _dl_

    #
    # gatherMetrics()
    #
    def gatherMetrics(self, callable, *args, **kwargs):
        t0 = time.time()
        _results_ = callable(*args, **kwargs)
        t1 = time.time()
        if callable.__name__ not in self.timing_metrics: self.timing_metrics[callable.__name__] = 0.0
        self.timing_metrics[callable.__name__] += t1 - t0
        return _results_

    #
    # __parseInput__()
    #
    def __parseInput__(self, *args, **kwargs):
        _unknown_ = set(kwargs) - self._VALID_KWARGS
        if _unknown_:
            raise TypeError(f'LinkP: unexpected keyword argument(s): {sorted(_unknown_)}')

        # Single source of truth for every parameter (name -> from-scratch default);
        # drives both the from-scratch assignment and the keyword-override copy below.
        # node_color: None | '#rrggbb' | p2s.COLOR_BY_NODE_NAME | p2s_constant | ('field', p2s_constant) | {node: '#rrggbb'}
        _defaults_ = {
            # Core
            'relationships':          None,
            'pos':                    {},
            'view_window':            None,
            # Color (p2s style)
            'color':                  None,   # None | '#rrggbb' | 'field'
            'node_color':             None,
            # Count
            'count':                  self.p2s.ROW_COUNTp,
            # Node styling
            'node_size':              'medium',
            'node_opacity':           1.0,
            'node_size_range':        (0.3, 4),
            'draw_labels':            False,
            'node_labels':            None,
            'label_only':             set(),
            'label_line_width':       32,
            'label_max_lines':        4,
            'label_ellipsis':         True,
            # Link styling
            'link_size':              'small',
            'link_shape':             'line',
            'link_opacity':           1.0,
            'link_size_range':        (0.25, 4),
            # Geometry (p2s style)
            'wxh':                    (256, 256),
            'insets':                 (3, 3),
            'bounds_percent':         0.05,
            'use_pos_for_bounds':     True,
            # Convex hulls
            'convex_hull_lu':           None,
            'convex_hull_opacity':      0.3,
            'convex_hull_labels':       False,
            'convex_hull_stroke_width': None,
            # Background shapes (p2s naming, matching xyp's background_* convention)
            'background':             None,
            'background_label_color': None,
            'background_opacity':     1.0,
            'background_fill':        None,
            'background_stroke_w':    1.0,
            'background_stroke':      'default',
            # Small multiples
            'sm_shared':              set(),
            '_shared_view_x_':        None,
            '_shared_view_y_':        None,
            'count_range_shared':     None,
            'color_stat_range_shared': None,
            # Context
            'draw_border':            True,
            'txt_h':                  12,
            'legend':                 False,
        }
        self.p2s.assertParamSpecMatches('LinkP', self._VALID_KWARGS, _defaults_)

        self.df, self.df_orig = None, None

        # Template support
        self.template = None
        for i in range(len(args)):
            if isinstance(args[i], LinkP): self.template = args[i]
        if 'template' in kwargs: self.template = kwargs['template']
        if self.template is not None:
            _template_copy_ = self.template
            self.p2s._clone_template_state(self, _template_copy_)
            self.template = _template_copy_
            self._count_min_      = None
            self._count_max_      = None
            self._color_stat_min_ = None
            self._color_stat_max_ = None
        else:
            self.p2s.assignScratchDefaults(self, _defaults_)
            # Internal (non-parameter) state — not part of the kwarg spec
            self._view_window_user_set_ = False
            self._count_min_            = None
            self._count_max_            = None
            self._color_stat_min_       = None
            self._color_stat_max_       = None
            self.color_nodes_final      = {}
            self.view_window_orig       = None
            self._render_invalid_       = False
            # from-scratch builds only — a template clone is an exact snapshot and
            # must not re-apply session defaults (see Polars2SVG._apply_defaults)
            kwargs = self.p2s._apply_defaults('linkp', kwargs)

        # Extract DataFrame
        _new_df_ = None
        for _arg_ in args:
            if isinstance(_arg_, pl.DataFrame):
                if _new_df_ is None: _new_df_ = _arg_
                else:                raise ValueError('LinkP.__parseInput__(): df already set')
        if 'df' in kwargs:
            if _new_df_ is None: _new_df_ = kwargs['df']
            else:                raise ValueError('LinkP.__parseInput__(): df already set')
        if _new_df_ is not None:
            self.df = self.df_orig = _new_df_

        # Infer relationships and pos from positional args
        # - list of tuples (each length >= 2) → relationships
        # - dict with values that are length-2 numeric coordinate pairs → pos
        def _is_pos_dict_(arg):
            if not isinstance(arg, dict) or len(arg) == 0: return False
            for v in arg.values():
                if isinstance(v, (str, dict)) or not hasattr(v, '__len__') or len(v) != 2: return False
                try:    float(v[0]); float(v[1])
                except (TypeError, ValueError): return False
            return True

        _rel_from_pos_ = None
        _pos_from_pos_ = None
        for _arg_ in args:
            if   isinstance(_arg_, pl.DataFrame): pass
            elif isinstance(_arg_, LinkP):        pass
            elif isinstance(_arg_, list) and len(_arg_) > 0 and all(isinstance(t, tuple) and len(t) >= 2 for t in _arg_):
                if _rel_from_pos_ is not None: raise ValueError('LinkP.__parseInput__(): relationships specified twice positionally')
                _rel_from_pos_ = _arg_
            elif _is_pos_dict_(_arg_):
                if _pos_from_pos_ is not None: raise ValueError('LinkP.__parseInput__(): pos specified twice positionally')
                _pos_from_pos_ = _arg_
            else:
                raise ValueError(f'LinkP.__parseInput__(): Unrecognized positional argument type {type(_arg_).__name__}')

        # Track positional origin so a downstream "field not found" can name positional
        # dispatch (list-of-tuples -> relationships) as the likely cause; a relationships=
        # keyword is an explicit assignment, so no dispatch ambiguity to flag.
        self._relationships_from_positional_ = _rel_from_pos_ is not None
        if _rel_from_pos_ is not None:
            if 'relationships' in kwargs: raise ValueError('LinkP.__parseInput__(): relationships specified both positionally and as a keyword')
            self.relationships = _rel_from_pos_
        if _pos_from_pos_ is not None:
            if 'pos' in kwargs: raise ValueError('LinkP.__parseInput__(): pos specified both positionally and as a keyword')
            self.pos = _pos_from_pos_

        # Apply kwargs overrides (view_window carries a side effect, handled below)
        self.p2s.assignKwargOverrides(self, _defaults_, kwargs, skip={'view_window'})
        if 'view_window'            in kwargs:
            self.view_window              = kwargs['view_window']
            self._view_window_user_set_   = kwargs['view_window'] is not None

        # Normalize label_only to a set
        if isinstance(self.label_only, list): self.label_only = set(self.label_only)
        if isinstance(self.label_only, str):  self.label_only = {self.label_only}

        # "No data" placeholder for early error visibility -- only ever seen when
        # no df is supplied (a successful render overwrites self.svg); makes a
        # dropped-df plumbing mistake visible instead of a silently blank canvas.
        self.wxh = self.p2s.normalizeWxh(self.wxh, 'LinkP')
        w, h = self.wxh
        self.svg = self.p2s.placeholderSVG(w, h)

        if self.df is None: return

        # Copy the DataFrame and add the row index if not already present
        self.df = self.df.clone()
        if '__p2s_index__' not in self.df.columns:
            self.df = self.df.with_row_index('__p2s_index__')

        # Expand tuple-based node fields into concatenated columns
        self.relationships_orig = self.relationships
        self.relationships, i = [], 0
        for _edge_ in self.relationships_orig:
            _fm_, _to_ = _edge_[0], _edge_[1]
            new_fm, new_to = _fm_, _to_
            if isinstance(_fm_, tuple):
                new_fm = f'__fm{i}__'
                self.df = self._createConcatColumn_(self.df, _fm_, new_fm)
            if isinstance(_to_, tuple):
                new_to = f'__to{i}__'
                self.df = self._createConcatColumn_(self.df, _to_, new_to)
            if   len(_edge_) == 2: self.relationships.append((new_fm, new_to))
            elif len(_edge_) == 3: self.relationships.append((new_fm, new_to, _edge_[2]))
            else: raise ValueError(f'LinkP: relationship tuples must have 2 or 3 parts, got {_edge_!r}')
            i += 1

        # Classify color modes and pre-build categorical color columns (must happen before group_by)
        self._link_color_mode_ = self.__colorModeInfo__(self.__effectiveColorSpec__('links'))
        self._node_color_mode_ = self.__colorModeInfo__(self.__effectiveColorSpec__('nodes'))
        if self._link_color_mode_['kind'] == 'categorical' and self._link_color_mode_['field']:
            self.df = self.df.with_columns(
                self.p2s.colorizeColumnPolarsOperations(self._link_color_mode_['field']).alias('__lc_cat__')
            )
        elif self._link_color_mode_['kind'] == 'cset' and self._link_color_mode_['field']:
            self.df = self.df.with_columns(
                pl.col(self._link_color_mode_['field']).cast(pl.String).alias('__lc_cat__')
            )
        # Node categorical color is derived in __renderNodes__ after concatenating all edge endpoints,
        # so each node's color reflects field values seen across all its edges (both fm and to sides).

    #
    # _wrap_label_() - word-wrap a label string into lines
    # - breaks at word boundaries; hard-breaks words longer than line_width
    # - truncates to max_lines (pass -1 for unlimited)
    # - appends '…' on the last line when truncated and use_ellipsis is True
    #
    def _wrap_label_(self, text, line_width, max_lines, use_ellipsis):
        words = text.split(' ')
        lines, current = [], ''
        for word in words:
            if not current:
                while len(word) > line_width:
                    lines.append(word[:line_width])
                    word = word[line_width:]
                current = word
            elif len(current) + 1 + len(word) <= line_width:
                current += ' ' + word
            else:
                lines.append(current)
                current = ''
                while len(word) > line_width:
                    lines.append(word[:line_width])
                    word = word[line_width:]
                current = word
        if current:
            lines.append(current)
        if max_lines != -1 and len(lines) > max_lines:
            lines = lines[:max_lines]
            if use_ellipsis:
                last = lines[-1]
                lines[-1] = last[:line_width - 1] + '…' if len(last) >= line_width else last + '…'
        return lines

    #
    # _createConcatColumn_() - concatenate multiple fields into one string column
    #
    def _createConcatColumn_(self, df, fields, new_col):
        _parts_ = []
        for i, f in enumerate(fields):
            if i > 0: _parts_.append(pl.lit('|'))
            _parts_.append(pl.col(f).cast(pl.String))
        return df.with_columns(pl.concat_str(_parts_).alias(new_col))

    #
    # __countAggExpr__() - return the Polars aggregation expression for counting edges
    # - mirrors the identical method in Timep and Histop
    #
    def __countAggExpr__(self):
        if self.count == self.p2s.ROW_COUNTp:
            return pl.len().alias('__count__')
        elif isinstance(self.count, str):
            _is_num_ = self.p2s.numericColumn(self.df, self.count)
            self.p2s.logDtypeKeyedCount('LinkP', self.count, _is_num_)
            if _is_num_: return pl.col(self.count).sum()    .alias('__count__')
            else:        return pl.col(self.count).n_unique().alias('__count__')
        elif isinstance(self.count, tuple):
            _fields_ = [_f_ for _f_ in self.count if isinstance(_f_, str)]
            if self.p2s.SETp in self.count:      return pl.col(_fields_[0]).n_unique().alias('__count__')
            elif len(_fields_) == 1:             return pl.col(_fields_[0]).sum()    .alias('__count__')
            else:                                return pl.struct(_fields_).n_unique().alias('__count__')
        return pl.len().alias('__count__')

    def __countFields__(self):
        if self.count == self.p2s.ROW_COUNTp: return set()
        if isinstance(self.count, str):        return {self.count}
        if isinstance(self.count, tuple):      return {_f_ for _f_ in self.count if isinstance(_f_, str)}
        return set()

    #
    # __effectiveColorSpec__() - resolve the color spec for links or nodes
    #
    def __effectiveColorSpec__(self, target):
        if target == 'links': return self.color
        return self.node_color

    #
    # __colorModeInfo__() - classify a color spec into a mode dict
    # Returns: {'kind': str, 'field': str|None, 'stat': str, 'hex': str|None}
    # Kinds: 'default' | 'fixed_hex' | 'categorical' | 'crow_magnitude' | 'crow_stretched' |
    #        'cset_magnitude' | 'cset_stretched' | 'stat_magnitude' | 'stat_stretched'
    #
    def __colorModeInfo__(self, spec):
        _p2s_ = self.p2s
        _cmag_  = {_p2s_.CMAGNITUDE_SUMp, _p2s_.CMAGNITUDE_MINp, _p2s_.CMAGNITUDE_MEDIANp,
                   _p2s_.CMAGNITUDE_MEANp, _p2s_.CMAGNITUDE_MAXp}
        _cstr_  = {_p2s_.CSTRETCHED_SUMp, _p2s_.CSTRETCHED_MINp, _p2s_.CSTRETCHED_MEDIANp,
                   _p2s_.CSTRETCHED_MEANp, _p2s_.CSTRETCHED_MAXp}
        _smap_  = {
            _p2s_.CMAGNITUDE_SUMp: 'sum',    _p2s_.CSTRETCHED_SUMp: 'sum',
            _p2s_.CMAGNITUDE_MINp: 'min',    _p2s_.CSTRETCHED_MINp: 'min',
            _p2s_.CMAGNITUDE_MEDIANp:'median',_p2s_.CSTRETCHED_MEDIANp:'median',
            _p2s_.CMAGNITUDE_MEANp: 'mean',  _p2s_.CSTRETCHED_MEANp: 'mean',
            _p2s_.CMAGNITUDE_MAXp: 'max',    _p2s_.CSTRETCHED_MAXp: 'max',
            _p2s_.SUMp: 'sum',   _p2s_.MINp: 'min',
            _p2s_.MEDIANp: 'median', _p2s_.MEANp: 'mean',
            _p2s_.MAXp: 'max',   _p2s_.STDp: 'std',
        }
        _info_ = {'kind': 'default', 'field': None, 'stat': 'sum', 'hex': None}
        if spec is None:
            pass
        elif isinstance(spec, self.p2s.HexColorString):
            _info_['kind'] = 'fixed_hex';  _info_['hex'] = spec
        elif spec == _p2s_.CROW_MAGNITUDEp:
            _info_['kind'] = 'crow_magnitude'
        elif spec == _p2s_.CROW_STRETCHEDp:
            _info_['kind'] = 'crow_stretched'
        elif spec == _p2s_.COLOR_BY_NODE_NAME:
            _info_['kind'] = 'categorical'   # field=None → colorize by node name
        elif isinstance(spec, str) and self.df is not None and spec in self.df.columns:
            _is_num_ = self.p2s.numericColumn(self.df, spec)
            self.p2s.logDtypeKeyedColor('LinkP', spec, _is_num_)
            if _is_num_:
                _info_['kind'] = 'stat_magnitude'; _info_['field'] = spec; _info_['stat'] = 'sum'
            else:
                _info_['kind'] = 'cset'; _info_['field'] = spec
        elif isinstance(spec, tuple):
            _strs_  = [f for f in spec if isinstance(f, str)]
            _enums_ = [e for e in spec if not isinstance(e, str)]
            _field_ = _strs_[0] if _strs_ else None
            _enum_  = _enums_[0] if _enums_ else None
            if   _enum_ == _p2s_.CSETp:
                _info_['kind'] = 'cset';           _info_['field'] = _field_
            elif _enum_ == _p2s_.CSET_MAGNITUDEp:
                _info_['kind'] = 'cset_magnitude'; _info_['field'] = _field_
            elif _enum_ == _p2s_.CSET_STRETCHEDp:
                _info_['kind'] = 'cset_stretched'; _info_['field'] = _field_
            elif _enum_ in _cmag_:
                _info_['kind'] = 'stat_magnitude'; _info_['field'] = _field_; _info_['stat'] = _smap_.get(_enum_, 'sum')
            elif _enum_ in _cstr_:
                _info_['kind'] = 'stat_stretched'; _info_['field'] = _field_; _info_['stat'] = _smap_.get(_enum_, 'sum')
            elif _enum_ in _smap_:
                _info_['kind'] = 'stat_magnitude'; _info_['field'] = _field_; _info_['stat'] = _smap_[_enum_]
            elif _field_:
                _info_['kind'] = 'categorical';    _info_['field'] = _field_
        return _info_

    #
    # __colorAggExprs__() - return agg expressions needed by a color mode (added into group_by().agg())
    #
    def __colorAggExprs__(self, mode_info, prefix):
        kind = mode_info['kind']
        if kind in ('categorical', 'cset'):
            return [
                pl.col(f'__{prefix}_cat__').n_unique().alias(f'__{prefix}_nuniq__'),
                pl.col(f'__{prefix}_cat__').first().alias(f'__{prefix}_first__'),
            ]
        elif kind in ('cset_magnitude', 'cset_stretched') and mode_info['field']:
            return [pl.col(mode_info['field']).n_unique().alias(f'__{prefix}_stat__')]
        elif kind in ('stat_magnitude', 'stat_stretched') and mode_info['field']:
            _field_ = mode_info['field']
            _op_ = {
                'sum':    pl.col(_field_).sum(),
                'min':    pl.col(_field_).min(),
                'median': pl.col(_field_).median(),
                'mean':   pl.col(_field_).mean(),
                'max':    pl.col(_field_).max(),
                'std':    pl.col(_field_).std(),
            }.get(mode_info['stat'], pl.col(_field_).sum())
            return [_op_.alias(f'__{prefix}_stat__')]
        elif kind in ('crow_magnitude', 'crow_stretched'):
            return [pl.len().alias(f'__{prefix}_row_count__')]
        return []

    #
    # __applyColorToDF__() - add f'__{prefix}_hex__' column to an aggregated DataFrame
    #
    def __applyColorToDF__(self, df, mode_info, prefix, default_hex):
        kind    = mode_info['kind']
        col_hex = f'__{prefix}_hex__'
        if kind == 'fixed_hex':
            return df.with_columns(pl.lit(mode_info['hex']).alias(col_hex))
        elif kind == 'categorical':
            return df.with_columns(
                pl.when(pl.col(f'__{prefix}_nuniq__') == 1)
                  .then(pl.col(f'__{prefix}_first__'))
                  .otherwise(pl.lit(default_hex))
                  .alias(col_hex)
            )
        elif kind == 'cset':
            df = df.with_columns(
                pl.when(pl.col(f'__{prefix}_nuniq__') == 1)
                  .then(pl.col(f'__{prefix}_first__'))
                  .otherwise(pl.lit(-1))
                  .alias(f'__{prefix}_set_elem__')
            )
            return df.with_columns(
                self.p2s.colorizeColumnPolarsOperations(f'__{prefix}_set_elem__').alias(col_hex)
            )
        elif kind in ('crow_magnitude', 'crow_stretched', 'cset_magnitude', 'cset_stretched',
                      'stat_magnitude', 'stat_stretched'):
            _sc_     = f'__{prefix}_row_count__' if kind in ('crow_magnitude', 'crow_stretched') else f'__{prefix}_stat__'
            _norm_   = f'__{prefix}_norm__'
            _r_, _g_, _b_ = f'__{prefix}_r__', f'__{prefix}_g__', f'__{prefix}_b__'
            # legend-only stat accumulator (kept separate from _color_stat_min_/_max_,
            # which smallp SM_COLOR sharing reads and which stretched modes never touch)
            _lg_min_ = df[_sc_].cast(pl.Float64).min()
            _lg_max_ = df[_sc_].cast(pl.Float64).max()
            if _lg_min_ is not None and (getattr(self, '_legend_stat_min_', None) is None or _lg_min_ < self._legend_stat_min_):
                self._legend_stat_min_ = float(_lg_min_)
            if _lg_max_ is not None and (getattr(self, '_legend_stat_max_', None) is None or _lg_max_ > self._legend_stat_max_):
                self._legend_stat_max_ = float(_lg_max_)
            if kind in ('crow_stretched', 'cset_stretched', 'stat_stretched'):
                _n_unique_ = df[_sc_].n_unique()
                df = df.with_columns(
                    ((pl.col(_sc_).rank('dense') - 1).cast(pl.Float64) / max(_n_unique_ - 1, 1)).alias(_norm_)
                )
            else:
                if self.color_stat_range_shared is not None:
                    _cs_min_ = float(self.color_stat_range_shared[0])
                    _cs_max_ = float(self.color_stat_range_shared[1])
                else:
                    _min_v_ = df[_sc_].cast(pl.Float64).min()
                    _max_v_ = df[_sc_].cast(pl.Float64).max()
                    _cs_min_ = float(_min_v_) if _min_v_ is not None else 0.0
                    _cs_max_ = float(_max_v_) if _max_v_ is not None else 1.0
                if self._color_stat_min_ is None or _cs_min_ < self._color_stat_min_:
                    self._color_stat_min_ = _cs_min_
                if self._color_stat_max_ is None or _cs_max_ > self._color_stat_max_:
                    self._color_stat_max_ = _cs_max_
                df = df.with_columns(
                    ((pl.col(_sc_).cast(pl.Float64) - _cs_min_) /
                     (0.001 + _cs_max_ - _cs_min_))
                    .clip(0.0, 1.0).alias(_norm_)
                )
            df = df.with_columns(
                self.p2s.colorSpectrumPolarsOperations(_norm_, _r_, _g_, _b_)
            ).with_columns(
                self.p2s.hexColorFromRGBTriplesPolarsOperations(_r_, _g_, _b_).alias(col_hex)
            )
            return df
        else:
            return df.with_columns(pl.lit(default_hex).alias(col_hex))

    #
    # __validateColorSpec__() - raise ValueError if a node_color value is not a recognized form
    #
    def __validateColorSpec__(self, spec, param_name, allow_dict=False):
        if spec is None: return
        if isinstance(spec, dict):
            if not allow_dict:
                raise ValueError(f'LinkP.__validateInput__(): {param_name} does not support dict values')
            return
        if isinstance(spec, tuple): return
        _p2s_ = self.p2s
        if spec in (_p2s_.CROW_MAGNITUDEp, _p2s_.CROW_STRETCHEDp, _p2s_.COLOR_BY_NODE_NAME): return
        if isinstance(spec, self.p2s.HexColorString): return
        if isinstance(spec, str):
            if self.df is not None and spec in self.df.columns: return
            raise ValueError(
                f'LinkP.__validateInput__(): {param_name}={spec!r} is not a hex color, '
                f'a recognized constant, or a DataFrame column name'
            )
        raise ValueError(
            f'LinkP.__validateInput__(): {param_name}={spec!r} has unsupported type {type(spec).__name__}'
        )

    #
    # __validateInput__()
    #
    def __validateInput__(self):
        # Normalize legend= eagerly so a bad spec fails fast (raises InvalidSpecError).
        self.legend_spec = self.p2s.legendResolveSpec(self.legend)
        if self.df is None: return
        self.p2s.checkReservedColumns(self.df, 'LinkP')
        if self.relationships is None or len(self.relationships) == 0:
            raise ValueError('LinkP.__validateInput__(): relationships must be specified')
        _rel_hint_ = self.p2s.positionalDispatchHint('LinkP', 'relationships',
                                                     self._relationships_from_positional_)
        for _rel_ in self.relationships:
            for _field_ in _rel_[:2]:
                if _field_ not in self.df.columns:
                    raise ValueError(f'LinkP.__validateInput__(): field "{_field_}" not found in DataFrame{_rel_hint_}')
        for _field_ in self.__countFields__():
            if _field_ not in self.df.columns:
                raise ValueError(f'LinkP.__validateInput__(): count field "{_field_}" not found in DataFrame')
        if self.color == self.p2s.COLOR_BY_NODE_NAME:
            raise ValueError(
                'LinkP.__validateInput__(): color=p2s.COLOR_BY_NODE_NAME is not valid for the '
                'color parameter; use node_color=p2s.COLOR_BY_NODE_NAME instead'
            )
        self.__validateColorSpec__(self.node_color, 'node_color', allow_dict=True)

        # count= is only consumed by 'vary' sizing: fixed node/link sizes ignore it,
        # CROW_* color modes use raw row count (__row_count__), and the fallback
        # layout assigns random positions without regard to edge weights.
        if self.count != self.p2s.ROW_COUNTp and \
           self.node_size != 'vary' and self.link_size != 'vary':
            self.p2s.logger.warning(
                "LinkP: count= is set but has no visible effect at the current settings; "
                "use node_size='vary' and/or link_size='vary' to size nodes/links by count "
                "(CROW_* color modes use raw row count, not count=)"
            )

    # Color-mode kinds that carry data-driven color semantics a legend can describe
    # ('default' / 'fixed_hex' / node-dict overrides do not).
    _LEGENDABLE_KINDS_ = frozenset({'categorical', 'cset', 'crow_magnitude', 'crow_stretched',
                                    'cset_magnitude', 'cset_stretched', 'stat_magnitude', 'stat_stretched'})

    #
    # __legendPrepare__() - resolve legend kind/metadata (the capture hook) and the
    # strip to reserve.  LinkP has two color channels; the legend describes the link
    # channel (color=) when it is legend-able, otherwise the node channel
    # (node_color=).  Colorbar domains are only known after __renderLinks__/
    # __renderNodes__ run (__applyColorToDF__ accumulates _legend_stat_min_/_max_),
    # so the colorbar is finalized in __renderSVG__.  Decision A: a truthy legend
    # with nothing to legend silently reserves nothing.
    #
    def __legendPrepare__(self):
        self.legend_info       = None
        self._legend_region_   = None
        self._legend_reserve_  = (0, 0, 0, 0)
        self._legend_stat_min_ = None
        self._legend_stat_max_ = None
        if self.legend_spec is None or self.df is None or len(self.df) == 0: return
        _mode_, _channel_spec_ = None, None
        if   self._link_color_mode_['kind'] in self._LEGENDABLE_KINDS_:
            _mode_, _channel_spec_ = self._link_color_mode_, self.color
        elif self._node_color_mode_['kind'] in self._LEGENDABLE_KINDS_:
            _mode_, _channel_spec_ = self._node_color_mode_, self.node_color
        if _mode_ is None: return
        _spec_  = self.legend_spec
        _kind_  = 'categorical' if _mode_['kind'] in ('categorical', 'cset') else 'colorbar'
        if   _mode_['kind'] in ('crow_magnitude', 'crow_stretched'):    _title_default_ = 'rows'
        elif _mode_['field'] is not None:                               _title_default_ = _mode_['field']
        elif _channel_spec_ == self.p2s.COLOR_BY_NODE_NAME:             _title_default_ = 'node'
        else:                                                           _title_default_ = ''
        _title_ = _spec_['title'] if _spec_['title'] is not None else _title_default_
        if _kind_ == 'categorical':
            _field_ = _mode_['field']
            if _field_ is None:
                # COLOR_BY_NODE_NAME: entries are the (string-cast) node names, which
                # is exactly what __renderNodes__ colorizes
                _names_ = pl.concat([self.df.select(pl.col(_r_[_j_]).cast(pl.String).alias('__legend_node__'))
                                     for _r_ in self.relationships for _j_ in (0, 1)])
                _vc_ = self.p2s.legendCategoricalValueCounts(_names_, '__legend_node__')
                self.legend_info = self.p2s.legendInfoCategorical(_spec_, _vc_, _title_)
            elif _mode_['kind'] == 'cset':
                # cset colorizes the string-cast values -> the default string-hash matches
                _vc_ = self.p2s.legendCategoricalValueCounts(self.df, _field_)
                self.legend_info = self.p2s.legendInfoCategorical(_spec_, _vc_, _title_)
            else:
                # 'categorical' colorizes the raw (dtype-sensitive) values -- compute the
                # hashes the same way and hand them to the capture as an explicit lookup
                _agg_ = (self.df.group_by(_field_).agg(pl.len().alias('__legend_n__'))
                                .with_columns(self.p2s.colorizeColumnPolarsOperations(_field_).alias('__legend_hex__')))
                _vc_     = [(str(_k_), _n_, _k_) for _k_, _n_ in zip(_agg_[_field_].to_list(),
                                                                     _agg_['__legend_n__'].to_list())]
                _hex_lu_ = {str(_k_): _h_ for _k_, _h_ in zip(_agg_[_field_].to_list(),
                                                              _agg_['__legend_hex__'].to_list())}
                self.legend_info = self.p2s.legendInfoCategorical(_spec_, _vc_, _title_, hex_lu=_hex_lu_)
        else:
            self.legend_info = self.p2s.legendInfoColorbar(_title_)
            self._legend_stretched_ = _mode_['kind'].endswith('stretched')
        _reserve_ = self.p2s.legendReserve(_spec_, self.legend_info, self.txt_h, self.wxh)
        _l_, _r_, _t_, _b_ = _reserve_
        if self.wxh[0] - (_l_ + _r_) < 48 or self.wxh[1] - (_t_ + _b_) < 48:
            self.p2s.logger.warning(f'LinkP.__legendPrepare__(): not enough space for legend (wxh = {self.wxh}); legend dropped')
            self.legend_info = None
            return
        self._legend_reserve_ = _reserve_
        _pos_ = _spec_['pos']
        if   _pos_ == 'right':  self._legend_region_ = (self.wxh[0] - _r_, 0, _r_, self.wxh[1])
        elif _pos_ == 'left':   self._legend_region_ = (0, 0, _l_, self.wxh[1])
        elif _pos_ == 'top':    self._legend_region_ = (0, 0, self.wxh[0], _t_)
        else:                   self._legend_region_ = (0, self.wxh[1] - _b_, self.wxh[0], _b_)

    #
    # __calculateGeometry__()
    # - map node names to world coordinates using pos dict (via replace_strict for O(n) lookup)
    # - compute world bounds and coordinate transform lambdas
    #
    def __calculateGeometry__(self):
        # Legend strip (if any) comes out of wxh first -- the plot region shrinks,
        # the physical output size does not ("reserve from wxh").
        self.__legendPrepare__()
        # Collect all nodes from the data (unique in Polars; only the distinct
        # values cross into Python instead of every row)
        self.all_nodes = set()
        for _rel_ in self.relationships:
            self.all_nodes |= set(self.df[_rel_[0]].drop_nulls().unique().to_list())
            self.all_nodes |= set(self.df[_rel_[1]].drop_nulls().unique().to_list())

        # Assign random positions for any node not in pos
        for _node_ in self.all_nodes - self.pos.keys():
            self.pos[_node_] = (random.random(), random.random())  # nosec B311 - non-cryptographic initial layout jitter

        # Shadow pos as float dicts for replace_strict compatibility
        _xpos_ = {k: float(v[0]) for k, v in self.pos.items()}
        _ypos_ = {k: float(v[1]) for k, v in self.pos.items()}

        # Batch: map all from/to node columns to world x/y coordinates
        _operations_, self.xcols, self.ycols = [], [], []
        for i, _rel_ in enumerate(self.relationships):
            _fmx_, _fmy_ = f'__rel{i}_fm_wx__', f'__rel{i}_fm_wy__'
            _tox_, _toy_ = f'__rel{i}_to_wx__', f'__rel{i}_to_wy__'
            _operations_ += [
                pl.col(_rel_[0]).replace_strict(_xpos_, default=None).alias(_fmx_),
                pl.col(_rel_[0]).replace_strict(_ypos_, default=None).alias(_fmy_),
                pl.col(_rel_[1]).replace_strict(_xpos_, default=None).alias(_tox_),
                pl.col(_rel_[1]).replace_strict(_ypos_, default=None).alias(_toy_),
            ]
            self.xcols += [_fmx_, _tox_]
            self.ycols += [_fmy_, _toy_]
        self.df = self.df.with_columns(*_operations_)

        # Compute world bounds
        self.wx0 = self.df[self.xcols[0]].min()
        self.wy0 = self.df[self.ycols[0]].min()
        self.wx1 = self.df[self.xcols[0]].max()
        self.wy1 = self.df[self.ycols[0]].max()
        for i in range(1, len(self.xcols)):
            _xmin_ = self.df[self.xcols[i]].min()
            _ymin_ = self.df[self.ycols[i]].min()
            _xmax_ = self.df[self.xcols[i]].max()
            _ymax_ = self.df[self.ycols[i]].max()
            if _xmin_ is not None: self.wx0 = min(self.wx0, _xmin_) if self.wx0 is not None else _xmin_
            if _ymin_ is not None: self.wy0 = min(self.wy0, _ymin_) if self.wy0 is not None else _ymin_
            if _xmax_ is not None: self.wx1 = max(self.wx1, _xmax_) if self.wx1 is not None else _xmax_
            if _ymax_ is not None: self.wy1 = max(self.wy1, _ymax_) if self.wy1 is not None else _ymax_

        # Extend bounds to include all pos nodes if requested
        if self.use_pos_for_bounds:
            for _node_, _v_ in self.pos.items():
                _px_, _py_ = float(_v_[0]), float(_v_[1])
                self.wx0 = min(self.wx0, _px_) if self.wx0 is not None else _px_
                self.wy0 = min(self.wy0, _py_) if self.wy0 is not None else _py_
                self.wx1 = max(self.wx1, _px_) if self.wx1 is not None else _px_
                self.wy1 = max(self.wy1, _py_) if self.wy1 is not None else _py_

        # Defaults if still None
        if self.wx0 is None: self.wx0 = 0.0
        if self.wy0 is None: self.wy0 = 0.0
        if self.wx1 is None: self.wx1 = 1.0
        if self.wy1 is None: self.wy1 = 1.0

        # Handle degenerate single-point bounds
        if abs(self.wx1 - self.wx0) < 1e-6: self.wx0, self.wx1 = self.wx0 - 0.5, self.wx1 + 0.5
        if abs(self.wy1 - self.wy0) < 1e-6: self.wy0, self.wy1 = self.wy0 - 0.5, self.wy1 + 0.5

        # Apply bounds_percent padding
        if self.bounds_percent != 0:
            _dx_ = (self.wx1 - self.wx0) * self.bounds_percent
            _dy_ = (self.wy1 - self.wy0) * self.bounds_percent
            self.wx0 -= _dx_; self.wx1 += _dx_
            self.wy0 -= _dy_; self.wy1 += _dy_

        # Apply SM_X / SM_Y shared world bounds (from renderSmallMultiples reference)
        if self._shared_view_x_ is not None:
            self.wx0, self.wx1 = self._shared_view_x_
        if self._shared_view_y_ is not None:
            self.wy0, self.wy1 = self._shared_view_y_

        # Apply view_window override
        if self.view_window is not None:
            self.wx0, self.wy0, self.wx1, self.wy1 = self.view_window
        else:
            self.view_window = (self.wx0, self.wy0, self.wx1, self.wy1)

        # Coordinate transform lambdas (world ↔ screen); the legend reserve offsets
        # the plot region within the full canvas
        _lg_l_, _lg_r_, _lg_t_, _lg_b_ = self._legend_reserve_
        w, h   = self.wxh[0] - _lg_l_ - _lg_r_, self.wxh[1] - _lg_t_ - _lg_b_
        xi, yi = self.insets
        self.xT     = lambda wx: _lg_l_ + xi + (w - 2*xi) * (wx - self.wx0) / (self.wx1 - self.wx0)
        self.yT     = lambda wy: _lg_t_ + h - yi - (h - 2*yi) * (wy - self.wy0) / (self.wy1 - self.wy0)
        self.xT_inv = lambda sx: self.wx0 + (sx - _lg_l_ - xi) * (self.wx1 - self.wx0) / (w - 2*xi)
        self.yT_inv = lambda sy: self.wy0 + (_lg_t_ + h - yi - sy) * (self.wy1 - self.wy0) / (h - 2*yi)

    #
    # __calculateScreenCoordinates__()
    # - batch convert all world coordinate columns to integer screen coordinates
    #
    def __calculateScreenCoordinates__(self):
        _lg_l_, _lg_r_, _lg_t_, _lg_b_ = self._legend_reserve_
        w,  h  = self.wxh[0] - _lg_l_ - _lg_r_, self.wxh[1] - _lg_t_ - _lg_b_
        xi, yi = self.insets
        _operations_ = []
        for i in range(len(self.xcols)):
            _wx_col_ = self.xcols[i]
            _wy_col_ = self.ycols[i]
            _sx_col_ = _wx_col_.replace('wx', 'sx')
            _sy_col_ = _wy_col_.replace('wy', 'sy')
            _operations_ += [
                (pl.lit(_lg_l_ + xi) + pl.lit(w - 2*xi) * (pl.col(_wx_col_) - self.wx0) / (self.wx1 - self.wx0)).cast(pl.Int32).alias(_sx_col_),
                (pl.lit(_lg_t_ + h - yi) - pl.lit(h - 2*yi) * (pl.col(_wy_col_) - self.wy0) / (self.wy1 - self.wy0)).cast(pl.Int32).alias(_sy_col_),
            ]
        self.df = self.df.with_columns(*_operations_)

    def __countMinMax__(self, col):
        if self.count_range_shared is not None:
            return float(self.count_range_shared[0]), float(self.count_range_shared[1])
        lo, hi = col.min(), col.max()
        return (float(lo) if lo is not None else 0.0, float(hi) if hi is not None else 1.0)

    def __interpolatedSizeExpr__(self, size_range, lo, hi):
        return (
            size_range[0] +
            (size_range[1] - size_range[0]) *
            (pl.col('__count__').cast(pl.Float64) - lo) /
            (0.01 + hi - lo)
        )

    #
    # __renderLinks__()
    # - uses Polars group_by + concat_str to build SVG strings without Python row loops
    #
    def __renderLinks__(self):
        _node_size_lu_ = {'small': 1, 'nil': 0.2, 'medium': 3, 'large': 5}
        _sz_           = _node_size_lu_.get(self.link_size, 1.0)
        if isinstance(self.link_size, (int, float)): _sz_ = float(self.link_size)

        _data_co_ = self.p2s.colorTyped('data', 'default')
        _lc_agg_  = self.__colorAggExprs__(self._link_color_mode_, 'lc')

        _lk_sw_attr_      = '' if self.link_size == 'vary' else f' stroke-width="{_sz_}"'
        _link_group_open_ = f'<g fill="none"{_lk_sw_attr_} opacity="{self.link_opacity}">'

        _all_svg_ = set()
        self.df_link = None

        for i, _rel_ in enumerate(self.relationships):
            _fm_sx_, _fm_sy_ = f'__rel{i}_fm_sx__', f'__rel{i}_fm_sy__'
            _to_sx_, _to_sy_ = f'__rel{i}_to_sx__', f'__rel{i}_to_sy__'
            _gb_              = [_fm_sx_, _fm_sy_, _to_sx_, _to_sy_]
            _count_agg_       = self.__countAggExpr__()

            if self.link_shape == 'curve':
                _dx_, _dy_   = f'__dx{i}__', f'__dy{i}__'
                _mag_        = f'__mag{i}__'
                _u_,  _v_    = f'__u{i}__',  f'__v{i}__'
                _pu_, _pv_   = f'__pu{i}__', f'__pv{i}__'
                _xo0_, _yo0_ = f'__xo0{i}__', f'__yo0{i}__'
                _xo1_, _yo1_ = f'__xo1{i}__', f'__yo1{i}__'

                _df_link_ = (
                    self.df.group_by(_gb_)
                           .agg(_count_agg_, *_lc_agg_)
                )
                _df_link_ = self.__applyColorToDF__(_df_link_, self._link_color_mode_, 'lc', _data_co_)
                _df_link_ = (
                    _df_link_
                    .with_columns(
                        (pl.col(_to_sx_) - pl.col(_fm_sx_)).alias(_dx_),
                        (pl.col(_to_sy_) - pl.col(_fm_sy_)).alias(_dy_),
                    )
                    .with_columns(
                        ((pl.col(_dx_)**2 + pl.col(_dy_)**2).sqrt()).alias(_mag_)
                    )
                    .with_columns(
                        (pl.when(pl.col(_mag_) == 0).then(pl.lit(0.0)).otherwise(pl.col(_dx_) / pl.col(_mag_))).alias(_u_),
                        (pl.when(pl.col(_mag_) == 0).then(pl.lit(0.0)).otherwise(pl.col(_dy_) / pl.col(_mag_))).alias(_v_),
                    )
                    .with_columns(
                        (-pl.col(_v_)).alias(_pu_),
                        ( pl.col(_u_)).alias(_pv_),
                    )
                    .with_columns(
                        (pl.col(_fm_sx_) + pl.col(_mag_) * pl.col(_u_) / 3.0 + pl.col(_mag_) * pl.col(_pu_) / 10.0).alias(_xo0_),
                        (pl.col(_fm_sy_) + pl.col(_mag_) * pl.col(_v_) / 3.0 + pl.col(_mag_) * pl.col(_pv_) / 10.0).alias(_yo0_),
                        (pl.col(_to_sx_) - pl.col(_mag_) * pl.col(_u_) / 3.0 + pl.col(_mag_) * pl.col(_pu_) / 10.0).alias(_xo1_),
                        (pl.col(_to_sy_) - pl.col(_mag_) * pl.col(_v_) / 3.0 + pl.col(_mag_) * pl.col(_pv_) / 10.0).alias(_yo1_),
                    )
                )

                if self.link_size == 'vary':
                    _lc_min_, _lc_max_ = self.__countMinMax__(_df_link_['__count__'])
                    _stroke_w_ = self.__interpolatedSizeExpr__(self.link_size_range, _lc_min_, _lc_max_)
                    _str_ops_ = [
                        pl.lit('<path d="M '), pl.col(_fm_sx_), pl.lit(' '), pl.col(_fm_sy_),
                        pl.lit(' C '), pl.col(_xo0_), pl.lit(' '), pl.col(_yo0_),
                        pl.lit(' '), pl.col(_xo1_), pl.lit(' '), pl.col(_yo1_), pl.lit(' '),
                        pl.col(_to_sx_), pl.lit(' '), pl.col(_to_sy_),
                        pl.lit('" stroke="'), pl.col('__lc_hex__'),
                        pl.lit('" stroke-width="'), _stroke_w_,
                        pl.lit('" />'),
                    ]
                else:
                    _str_ops_ = [
                        pl.lit('<path d="M '), pl.col(_fm_sx_), pl.lit(' '), pl.col(_fm_sy_),
                        pl.lit(' C '), pl.col(_xo0_), pl.lit(' '), pl.col(_yo0_),
                        pl.lit(' '), pl.col(_xo1_), pl.lit(' '), pl.col(_yo1_), pl.lit(' '),
                        pl.col(_to_sx_), pl.lit(' '), pl.col(_to_sy_),
                        pl.lit('" stroke="'), pl.col('__lc_hex__'),
                        pl.lit('" />'),
                    ]

            else:  # 'line'
                _df_link_ = (
                    self.df.group_by(_gb_)
                           .agg(_count_agg_, *_lc_agg_)
                )
                _df_link_ = self.__applyColorToDF__(_df_link_, self._link_color_mode_, 'lc', _data_co_)

                if self.link_size == 'vary':
                    _lc_min_, _lc_max_ = self.__countMinMax__(_df_link_['__count__'])
                    _stroke_w_ = self.__interpolatedSizeExpr__(self.link_size_range, _lc_min_, _lc_max_)
                    _str_ops_ = [
                        pl.lit('<line x1="'), pl.col(_fm_sx_), pl.lit('" y1="'), pl.col(_fm_sy_),
                        pl.lit('" x2="'),    pl.col(_to_sx_), pl.lit('" y2="'), pl.col(_to_sy_),
                        pl.lit('" stroke="'), pl.col('__lc_hex__'),
                        pl.lit('" stroke-width="'), _stroke_w_,
                        pl.lit('" />'),
                    ]
                else:
                    _str_ops_ = [
                        pl.lit('<line x1="'), pl.col(_fm_sx_), pl.lit('" y1="'), pl.col(_fm_sy_),
                        pl.lit('" x2="'),    pl.col(_to_sx_), pl.lit('" y2="'), pl.col(_to_sy_),
                        pl.lit('" stroke="'), pl.col('__lc_hex__'),
                        pl.lit('" />'),
                    ]

            _link_col_ = '__link_svg__'
            _df_link_  = _df_link_.with_columns(pl.concat_str(_str_ops_).alias(_link_col_))
            self.df_link = _df_link_ if self.df_link is None else pl.concat([self.df_link, _df_link_], how='diagonal')

            if self.link_size is not None:
                _all_svg_ |= set(_df_link_.drop_nulls(subset=[_link_col_])[_link_col_].unique())

        _sorted_links_        = sorted(_all_svg_)
        self._link_svg_list_  = ([_link_group_open_] + _sorted_links_ + ['</g>']
                                  if _sorted_links_ else [])

    #
    # __renderNodes__()
    # - build a node DataFrame from all relationship endpoints via pl.concat
    # - group by screen coordinates and assemble SVG strings without Python row loops
    #
    def __renderNodes__(self):
        _node_size_lu_ = {'small': 3, 'medium': 5, 'large': 7, 'nil': 0.5}

        # Build node DataFrame by concat of fm/to columns for each relationship
        _nc_extra_ = ([] if '__nc_cat__' not in self.df.columns else [pl.col('__nc_cat__')])
        _nc_stat_f_ = self._node_color_mode_.get('field') \
            if self._node_color_mode_['kind'] in ('stat_magnitude', 'stat_stretched',
                                                   'cset_magnitude', 'cset_stretched') else None
        if _nc_stat_f_ and _nc_stat_f_ in self.df.columns:
            _nc_extra_.append(pl.col(_nc_stat_f_))
        _nc_cat_f_ = self._node_color_mode_.get('field') \
            if self._node_color_mode_['kind'] in ('categorical', 'cset') else None
        if _nc_cat_f_ and _nc_cat_f_ in self.df.columns:
            _nc_extra_.append(pl.col(_nc_cat_f_))
        _existing_extra_names_ = ({_nc_stat_f_} if _nc_stat_f_ else set()) | \
                                 ({_nc_cat_f_} if _nc_cat_f_ else set()) | \
                                 ({'__nc_cat__'} if '__nc_cat__' in self.df.columns else set())
        if self.node_size == 'vary':
            for _cf_ in self.__countFields__():
                if _cf_ in self.df.columns and _cf_ not in _existing_extra_names_:
                    _nc_extra_.append(pl.col(_cf_))
        _dfs_ = []
        for i, _rel_ in enumerate(self.relationships):
            for j in range(2):
                _sxfld_ = f'__rel{i}_{"fm" if j == 0 else "to"}_sx__'
                _syfld_ = f'__rel{i}_{"fm" if j == 0 else "to"}_sy__'
                _nmfld_ = _rel_[j]
                _dfs_.append(
                    self.df.select(
                        pl.col(_sxfld_).alias('__sx__'),
                        pl.col(_syfld_).alias('__sy__'),
                        pl.col(_nmfld_).cast(pl.String).alias('__nm__'),
                        *_nc_extra_,
                    ).drop_nulls(subset=['__sx__', '__sy__', '__nm__'])
                )
        self.df_node = pl.concat(_dfs_, how='diagonal')

        # For categorical node color: derive __nc_cat__ from either the field values or node name.
        if self._node_color_mode_['kind'] == 'categorical':
            _cat_field_ = self._node_color_mode_.get('field')
            if _cat_field_:
                # Field-based: colorize the field value. Nodes with one unique value get that
                # value's hash color; nodes with mixed values get background (n_unique > 1).
                self.df_node = self.df_node.with_columns(
                    self.p2s.colorizeColumnPolarsOperations(_cat_field_).alias('__nc_cat__')
                )
            else:
                # COLOR_BY_NODE_NAME: colorize by node name so each node always gets its own color.
                self.df_node = self.df_node.with_columns(
                    self.p2s.colorizeColumnPolarsOperations('__nm__').alias('__nc_cat__')
                )
        elif self._node_color_mode_['kind'] == 'cset':
            _cat_field_ = self._node_color_mode_.get('field')
            if _cat_field_:
                # Cast to String so colorization uses the string representation, matching histop.
                self.df_node = self.df_node.with_columns(
                    pl.col(_cat_field_).cast(pl.String).alias('__nc_cat__')
                )

        # Group by screen coords: count, unique names, color aggregation
        _nc_agg_ = self.__colorAggExprs__(self._node_color_mode_, 'nc')
        _bg_co_  = self.p2s.colorTyped('background', 'default')
        self.df_node = (
            self.df_node.group_by(['__sx__', '__sy__'])
                        .agg(
                            self.__countAggExpr__() if self.node_size == 'vary' else (pl.len() / 2.0).alias('__count__'),
                            pl.col('__nm__').unique(),
                            *_nc_agg_,
                        )
                        .with_columns(
                            pl.col('__nm__').list.len().alias('__nodes__'),
                            pl.col('__nm__').list.get(0).alias('__first__'),
                        )
        )
        self.df_node = self.__applyColorToDF__(self.df_node, self._node_color_mode_, 'nc', _bg_co_)

        # Node dict override: apply per-node-name hex colors after grouping (keyed on __first__)
        if isinstance(self.node_color, dict):
            _filled_ = {str(k): (v if isinstance(v, self.p2s.HexColorString) else self.p2s.color(v))
                        for k, v in self.node_color.items()}
            self.df_node = self.df_node.with_columns(
                pl.col('__first__').replace_strict(_filled_, default=_bg_co_).alias('__nc_hex__')
            )

        # Build node SVG strings
        _svg_strs_ = []
        if self.node_size is None:
            pass
        elif self.node_size == 'vary':
            _nc_min_, _nc_max_ = self.__countMinMax__(self.df_node['__count__'])
            self.df_node = self.df_node.with_columns(
                self.__interpolatedSizeExpr__(self.node_size_range, _nc_min_, _nc_max_).alias('__sz__')
            )
            _str_op_ = [
                pl.lit('<circle cx="'), pl.col('__sx__'), pl.lit('" cy="'), pl.col('__sy__'),
                pl.lit('" r="'), pl.col('__sz__').round(1),
                pl.lit('" fill="'), pl.col('__nc_hex__'),
                pl.lit('" />'),
            ]
            self.df_node = self.df_node.with_columns(pl.concat_str(_str_op_).alias('__node_svg__'))
            _raw_svgs_ = sorted(self.df_node.drop_nulls(subset=['__node_svg__'])['__node_svg__'].unique())
            _svg_strs_ = ([f'<g stroke="#000000" stroke-width="1" opacity="{self.node_opacity}">']
                          + _raw_svgs_ + ['</g>'] if _raw_svgs_ else [])
        else:
            _sz_ = _node_size_lu_.get(self.node_size, self.node_size) if isinstance(self.node_size, str) else float(self.node_size)
            _sw_ = 1.0 if _sz_ > 3 else _sz_ / 2.0
            # Single nodes (not collapsed)
            _str_op_ = [
                pl.lit('<circle cx="'), pl.col('__sx__'), pl.lit('" cy="'), pl.col('__sy__'),
                pl.lit(f'" r="{_sz_}" fill="'), pl.col('__nc_hex__'),
                pl.lit('" />'),
            ]
            _df_singles_ = self.df_node.filter(pl.col('__nodes__') == 1).with_columns(
                pl.concat_str(_str_op_).alias('__node_svg__')
            )
            _singles_svgs_ = sorted(_df_singles_.drop_nulls(subset=['__node_svg__'])['__node_svg__'].unique())
            _svg_strs_ = ([f'<g stroke="#000000" stroke-width="{_sw_}" opacity="{self.node_opacity}">']
                          + _singles_svgs_ + ['</g>'] if _singles_svgs_ else [])
            # Collapsed nodes (multiple nodes at same pixel): render as cloud symbol
            _str_op_multi_ = [
                pl.lit('<use href="#cloud" x="'), pl.col('__sx__'),
                pl.lit('" y="'), pl.col('__sy__'),
                pl.lit('" fill="'), pl.col('__nc_hex__'),
                pl.lit('" />'),
            ]
            _df_multis_ = self.df_node.filter(pl.col('__nodes__') > 1).with_columns(
                pl.concat_str(_str_op_multi_).alias('__node_svg__')
            )
            _multis_svgs_ = sorted(_df_multis_.drop_nulls(subset=['__node_svg__'])['__node_svg__'].unique())
            if _multis_svgs_:
                _svg_strs_.extend([f'<g stroke-width="0.5" opacity="{self.node_opacity}">']
                                   + _multis_svgs_ + ['</g>'])

        # Labels (deferred to after main node SVG)
        self._node_label_svg_  = []
        self._node_label_info_ = []   # (sx, y0, lines) -- for the WebGPU glyph path
        if self.draw_labels and len(_svg_strs_) > 0:
            _sz_for_label_ = _node_size_lu_.get(self.node_size, self.node_size) \
                if isinstance(self.node_size, str) else (float(self.node_size) if self.node_size is not None else 5)
            _df_labels_ = self.df_node.filter(pl.col('__nodes__') == 1)

            if self.label_only and len(self.label_only) > 0:
                _df_labels_ = _df_labels_.filter(pl.col('__first__').is_in(self.label_only))

            if self.node_labels is not None and len(self.node_labels) > 0:
                _label_map_ = {str(k): str(v) for k, v in self.node_labels.items()}
                _df_labels_ = _df_labels_.with_columns(
                    pl.col('__first__').replace_strict(_label_map_, default=None).alias('__label__')
                ).filter(pl.col('__label__').is_not_null())
            else:
                _df_labels_ = _df_labels_.with_columns(pl.col('__first__').alias('__label__'))

            _df_labels_ = _df_labels_.with_columns(
                pl.col('__label__').str.replace_all('&', '&amp;')
                                   .str.replace_all('<', '&lt;')
                                   .str.replace_all('>', '&gt;')
            )
            if '__sz__' not in _df_labels_.columns:
                _df_labels_ = _df_labels_.with_columns(pl.lit(float(_sz_for_label_)).alias('__sz__'))
            _lbl_set_ = set()
            for _sx_, _sy_, _sz_, _label_ in _df_labels_.select('__sx__', '__sy__', '__sz__', '__label__').iter_rows():
                if not _label_:
                    continue
                _lines_ = self._wrap_label_(_label_, self.label_line_width, self.label_max_lines, self.label_ellipsis)
                if not _lines_:
                    continue
                _y0_ = _sy_ + _sz_ + self.txt_h
                self._node_label_info_.append((_sx_, _y0_, _lines_))
                if len(_lines_) == 1:
                    _lbl_set_.add(
                        f'<text x="{_sx_}" y="{_y0_}" font-size="{self.txt_h}px" text-anchor="middle">{_lines_[0]}</text>'
                    )
                else:
                    _spans_ = [f'<tspan x="{_sx_}" dy="0">{_lines_[0]}</tspan>']
                    for _l_ in _lines_[1:]:
                        _spans_.append(f'<tspan x="{_sx_}" dy="{self.txt_h}">{_l_}</tspan>')
                    _lbl_set_.add(
                        f'<text x="{_sx_}" y="{_y0_}" font-size="{self.txt_h}px" text-anchor="middle">{"".join(_spans_)}</text>'
                    )
            self._node_label_svg_ = sorted(_lbl_set_)

        self._node_svg_list_ = _svg_strs_

        # Track final node colors for interactive queries (nodeColor, nodesWithColor)
        self.color_nodes_final = {}
        for _nm_list_, _hex_ in self.df_node.select('__nm__', '__nc_hex__').iter_rows():
            for _node_ in _nm_list_:
                self.color_nodes_final[_node_] = _hex_

        # Compute instance-level count range for SM_COUNT support in renderSmallMultiples
        _all_counts_ = []
        if self.df_link is not None and '__count__' in self.df_link.columns:
            _all_counts_.append(self.df_link['__count__'].cast(pl.Float64))
        if self.df_node is not None and '__count__' in self.df_node.columns:
            _all_counts_.append(self.df_node['__count__'].cast(pl.Float64))
        if _all_counts_:
            _combined_ = pl.concat(_all_counts_)
            _min_v_ = _combined_.min()
            _max_v_ = _combined_.max()
            self._count_min_ = float(_min_v_) if _min_v_ is not None else 0.0
            self._count_max_ = float(_max_v_) if _max_v_ is not None else 1.0

    #
    # __renderConvexHull__() - render convex hull annotations
    #
    def __renderConvexHull__(self):
        self._dl_hull_ = _dl_ = DisplayList(self.wxh[0], self.wxh[1])
        if not self.convex_hull_lu: return ''
        _svg_ = []
        _pt_lu_ = {}

        _first_value_ = next(iter(self.convex_hull_lu.values()))

        if isinstance(_first_value_, (list, set)):
            for hull_name, node_list in self.convex_hull_lu.items():
                _pts_ = {}
                for _node_ in node_list:
                    if _node_ in self.pos:
                        _pts_[_node_] = (self.xT(self.pos[_node_][0]), self.yT(self.pos[_node_][1]))
                if _pts_: _pt_lu_[hull_name] = _pts_
        else:
            # regex pattern → hull name
            import re as _re_
            for _node_ in self.all_nodes:
                _x_ = self.xT(self.pos[_node_][0]) if _node_ in self.pos else None
                _y_ = self.yT(self.pos[_node_][1]) if _node_ in self.pos else None
                if _x_ is None: continue
                for _pattern_, _hull_name_ in self.convex_hull_lu.items():
                    if _re_.match(_pattern_, str(_node_)):
                        if _hull_name_ not in _pt_lu_: _pt_lu_[_hull_name_] = {}
                        _pt_lu_[_hull_name_][_node_] = (_x_, _y_)

        for hull_name, _pts_ in _pt_lu_.items():
            _color_ = self.p2s.color(hull_name)
            _pts_list_ = list(_pts_.values())
            if len(_pts_list_) == 1:
                _x_, _y_ = _pts_list_[0]
                _svg_.append(f'<circle cx="{_x_}" cy="{_y_}" r="8" fill="{_color_}" fill-opacity="{self.convex_hull_opacity}" />')
                _dl_.circle(_x_, _y_, 8, _color_, opacity=self.convex_hull_opacity)
            elif len(_pts_list_) == 2:
                _x0_, _y0_ = _pts_list_[0]
                _x1_, _y1_ = _pts_list_[1]
                _svg_.append(f'<line x1="{_x0_}" y1="{_y0_}" x2="{_x1_}" y2="{_y1_}" stroke="{_color_}" stroke-width="8" stroke-opacity="{self.convex_hull_opacity}" />')
                _dl_.line(_x0_, _y0_, _x1_, _y1_, _color_, width=8, opacity=self.convex_hull_opacity)
            else:
                # Compute convex hull via gift wrapping
                _hull_pts_ = self._convexHull_(_pts_list_)
                _poly_pts_ = ' '.join(f'{x},{y}' for x, y in _hull_pts_)
                _svg_.append(f'<polygon points="{_poly_pts_}" fill="{_color_}" fill-opacity="{self.convex_hull_opacity}" stroke="none" />')
                _dl_.polygon(_hull_pts_, _color_, opacity=self.convex_hull_opacity)
                if self.convex_hull_stroke_width is not None:
                    _sw_ = self.convex_hull_stroke_width
                    _op_ = min(1.0, self.convex_hull_opacity + 0.2)
                    _svg_.append(f'<polygon points="{_poly_pts_}" fill="none" stroke="{_color_}" stroke-width="{_sw_}" stroke-opacity="{_op_}" />')
                    _closed_ = list(_hull_pts_) + [_hull_pts_[0]]
                    for _j_ in range(len(_closed_) - 1):
                        _dl_.line(_closed_[_j_][0], _closed_[_j_][1], _closed_[_j_+1][0], _closed_[_j_+1][1],
                                  _color_, width=_sw_, opacity=_op_)
            if self.convex_hull_labels:
                _cx_ = sum(p[0] for p in _pts_list_) / len(_pts_list_)
                _cy_ = sum(p[1] for p in _pts_list_) / len(_pts_list_)
                _svg_.append(_dl_.text(self.p2s, hull_name, _cx_, _cy_, txt_h=self.txt_h, anchor='middle'))
        return ''.join(_svg_)

    #
    # _convexHull_() - gift-wrapping convex hull for a list of (x, y) points
    #
    def _convexHull_(self, points):
        if len(points) <= 2: return points
        _pts_ = sorted(set(points))
        if len(_pts_) <= 2: return _pts_
        def _cross_(o, a, b): return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
        _lower_, _upper_ = [], []
        for p in _pts_:
            while len(_lower_) >= 2 and _cross_(_lower_[-2], _lower_[-1], p) <= 0: _lower_.pop()
            _lower_.append(p)
        for p in reversed(_pts_):
            while len(_upper_) >= 2 and _cross_(_upper_[-2], _upper_[-1], p) <= 0: _upper_.pop()
            _upper_.append(p)
        return _lower_[:-1] + _upper_[:-1]

    #
    # __shapelyToSVGPath__() - convert a Shapely geometry to an SVG path description string
    # - background= is the only feature that needs shapely, so the import is
    #   deferred here rather than paid by every linkp render.
    #
    def __shapelyToSVGPath__(self, shape):
        try:
            from shapely.geometry import Polygon, MultiPolygon, LineString, MultiLineString, GeometryCollection
        except ImportError as _e_:
            raise ImportError(
                "background= shapes require the optional 'layouts' dependency (shapely). "
                "Install it with:\n"
                "    pip install polars2svg[layouts]"
            ) from _e_
        if isinstance(shape, MultiPolygon):
            return ' '.join(self.__shapelyToSVGPath__(subpoly) for subpoly in shape.geoms)
        elif isinstance(shape, MultiLineString):
            return ' '.join(self.__shapelyToSVGPath__(subline) for subline in shape.geoms)
        elif isinstance(shape, LineString):
            coords = shape.coords
            path_str = f'M {coords[0][0]} {coords[0][1]}'
            for i in range(1, len(coords)):
                path_str += f' L {coords[i][0]} {coords[i][1]}'
            return path_str
        elif isinstance(shape, Polygon):
            xx, yy = shape.exterior.coords.xy
            path_str = f'M {xx[0]} {yy[0]}'
            for i in range(1, len(xx)):
                path_str += f' L {xx[i]} {yy[i]}'
            for interior in shape.interiors:
                ix, iy = interior.coords.xy
                path_str += f' M {ix[0]} {iy[0]}'
                for i in range(1, len(ix)):
                    path_str += f' L {ix[i]} {iy[i]}'
            return path_str + ' Z'
        elif isinstance(shape, GeometryCollection):
            if len(shape.geoms) > 0:
                raise DataError('LinkP.__shapelyToSVGPath__() - non-empty GeometryCollection not supported')
            return None
        else:
            raise DataError(f'LinkP.__shapelyToSVGPath__() - unsupported type: {type(shape)}')

    #
    # __bgMinsAndMaxes__() - update a bounding box with a new point
    #
    def __bgMinsAndMaxes__(self, x, y, x0, y0, x1, y1):
        if x0 is None:
            return x, y, x, y
        return min(x, x0), min(y, y0), max(x, x1), max(y, y1)

    #
    # __backgroundShapeRenderDetails__() - build SVG fill/stroke attribute string for a background shape
    #
    def __backgroundShapeRenderDetails__(self, name, bg_shape_opacity, bg_shape_fill, bg_shape_stroke_w, bg_shape_stroke):
        svg = ''
        # Fill
        if bg_shape_fill is not None and bg_shape_opacity is not None:
            if isinstance(bg_shape_opacity, dict):
                _opacity = bg_shape_opacity.get(name, 1.0)
            else:
                _opacity = bg_shape_opacity
            svg += f' fill-opacity="{_opacity}"'

            if   isinstance(bg_shape_fill, dict) and name in bg_shape_fill:
                _co = bg_shape_fill[name]
            elif bg_shape_fill == 'vary':
                _co = self.p2s.color(name)
            elif isinstance(bg_shape_fill, self.p2s.HexColorString):
                _co = bg_shape_fill
            else:
                _co = self.p2s.colorTyped('axis', 'inner')
            svg += f' fill="{_co}"'
        else:
            svg += ' fill-opacity="0.0"'

        # Stroke
        if bg_shape_stroke_w is not None and bg_shape_stroke is not None:
            if   bg_shape_stroke == 'vary':
                _co = self.p2s.color(name)
            elif isinstance(bg_shape_stroke, self.p2s.HexColorString):
                _co = bg_shape_stroke
            elif isinstance(bg_shape_stroke, dict) and name in bg_shape_stroke:
                _co = bg_shape_stroke[name]
            else:
                _co = self.p2s.colorTyped('axis', 'inner')

            if isinstance(bg_shape_stroke_w, dict) and name in bg_shape_stroke_w:
                _wi = bg_shape_stroke_w[name]
            else:
                _wi = bg_shape_stroke_w
            svg += f' stroke="{_co}" stroke-width="{_wi}"'
        return svg

    #
    # __backgroundShapeLabel__() - render a centred text label over the shape bounding box
    #
    def __backgroundShapeLabel__(self, name, x0, y0, x1, y1, bg_shape_label_color):
        if bg_shape_label_color is None or x0 is None:
            return ''
        if   isinstance(bg_shape_label_color, dict) and name in bg_shape_label_color:
            _co = bg_shape_label_color[name]
        elif bg_shape_label_color == 'vary':
            _co = self.p2s.color(name)
        elif isinstance(bg_shape_label_color, self.p2s.HexColorString):
            _co = bg_shape_label_color
        else:
            _co = self.p2s.colorTyped('axis', 'inner')
        _cx_ = (x0 + x1) / 2
        _cy_ = self.txt_h / 2 + (y0 + y1) / 2
        return (f'<text x="{_cx_}" y="{_cy_}" text-anchor="middle" '
                f'font-family="{self.p2s.default_font}" fill="{_co}" font-size="{self.txt_h}px">'
                f'{name}</text>')

    #
    # __transformCircleSVG__() - transform a <circle> SVG element into an <ellipse> in screen coordinates
    #
    def __transformCircleSVG__(self, name, shape_desc, bg_shape_label_color, bg_shape_opacity,
                                bg_shape_fill, bg_shape_stroke_w, bg_shape_stroke):
        _root_ = ET.fromstring(shape_desc)  # nosec B314 - trusted caller config (background= shape descriptor), not untrusted data; see SECURITY.md
        cx  = float(_root_.attrib['cx'])
        cy  = float(_root_.attrib['cy'])
        r   = float(_root_.attrib['r'])
        cx_s = self.xT(cx)
        cy_s = self.yT(cy)
        rx_s = abs(self.xT(r + cx) - cx_s)
        ry_s = abs(self.yT(r + cy) - cy_s)
        svg  = f'<ellipse cx="{cx_s}" cy="{cy_s}" rx="{rx_s}" ry="{ry_s}"'
        svg += self.__backgroundShapeRenderDetails__(name, bg_shape_opacity, bg_shape_fill,
                                                     bg_shape_stroke_w, bg_shape_stroke)
        return svg + '/>', self.__backgroundShapeLabel__(name, cx_s - rx_s, cy_s - ry_s,
                                                         cx_s + rx_s, cy_s + ry_s, bg_shape_label_color)

    #
    # __transformPathDescription__() - transform an SVG path description string into screen coordinates
    #
    def __transformPathDescription__(self, name, shape_desc, bg_shape_label_color, bg_shape_opacity,
                                      bg_shape_fill, bg_shape_stroke_w, bg_shape_stroke):
        svg = '<path d="'
        x0, y0, x1, y1 = None, None, None, None
        tokens = ' '.join(shape_desc.split()).split(' ')
        i = 0
        while i < len(tokens):
            if tokens[i] == 'M':
                _x, _y = self.xT(float(tokens[i+1])), self.yT(float(tokens[i+2]))
                svg += f' M {_x} {_y}'
                x0, y0, x1, y1 = self.__bgMinsAndMaxes__(_x, _y, x0, y0, x1, y1)
                i += 3
            elif tokens[i] == 'L':
                _x, _y = self.xT(float(tokens[i+1])), self.yT(float(tokens[i+2]))
                svg += f' L {_x} {_y}'
                x0, y0, x1, y1 = self.__bgMinsAndMaxes__(_x, _y, x0, y0, x1, y1)
                i += 3
            elif tokens[i] == 'C':
                _xcp1, _ycp1 = self.xT(float(tokens[i+1])), self.yT(float(tokens[i+2]))
                _xcp2, _ycp2 = self.xT(float(tokens[i+3])), self.yT(float(tokens[i+4]))
                _x,    _y    = self.xT(float(tokens[i+5])), self.yT(float(tokens[i+6]))
                svg += f' C {_xcp1} {_ycp1} {_xcp2} {_ycp2} {_x} {_y}'
                x0, y0, x1, y1 = self.__bgMinsAndMaxes__(_x,    _y,    x0, y0, x1, y1)
                x0, y0, x1, y1 = self.__bgMinsAndMaxes__(_xcp1, _ycp1, x0, y0, x1, y1)
                x0, y0, x1, y1 = self.__bgMinsAndMaxes__(_xcp2, _ycp2, x0, y0, x1, y1)
                i += 7
            elif tokens[i] == 'Z':
                svg += ' Z'
                i += 1
            else:
                raise Polars2SVGError(f'LinkP.__transformPathDescription__() - unhandled path token "{tokens[i]}"')
        svg += '"'
        svg += self.__backgroundShapeRenderDetails__(name, bg_shape_opacity, bg_shape_fill,
                                                     bg_shape_stroke_w, bg_shape_stroke)
        return svg + '/>', self.__backgroundShapeLabel__(name, x0, y0, x1, y1, bg_shape_label_color)

    #
    # __transformPointsList__() - transform a list of (x, y) tuples into a screen-coordinate SVG path
    #
    def __transformPointsList__(self, name, points_list, bg_shape_label_color, bg_shape_opacity,
                                 bg_shape_fill, bg_shape_stroke_w, bg_shape_stroke):
        _x, _y = self.xT(points_list[0][0]), self.yT(points_list[0][1])
        svg = f'<path d="M {_x} {_y}'
        x0, y0, x1, y1 = _x, _y, _x, _y
        for i in range(1, len(points_list)):
            _x, _y = self.xT(points_list[i][0]), self.yT(points_list[i][1])
            svg += f' L {_x} {_y}'
            x0, y0, x1, y1 = self.__bgMinsAndMaxes__(_x, _y, x0, y0, x1, y1)
        svg += ' Z"'
        svg += self.__backgroundShapeRenderDetails__(name, bg_shape_opacity, bg_shape_fill,
                                                     bg_shape_stroke_w, bg_shape_stroke)
        return svg + '/>', self.__backgroundShapeLabel__(name, x0, y0, x1, y1, bg_shape_label_color)

    #
    # __transformBackgroundShapes__() - dispatch a background shape to the appropriate transform method
    #
    def __transformBackgroundShapes__(self, name, shape_desc, bg_shape_label_color, bg_shape_opacity,
                                       bg_shape_fill, bg_shape_stroke_w, bg_shape_stroke):
        # Convert Shapely geometries to SVG path strings. shapely is an optional
        # 'layouts' dependency: if it isn't installed, shape_desc can't actually
        # be a shapely geometry (the caller couldn't have constructed one), so
        # skip these isinstance checks rather than importing/erroring for a
        # plain string/svg background.
        try:
            from shapely.geometry import Polygon, MultiPolygon, LineString, MultiLineString, GeometryCollection
            _has_shapely_ = True
        except ImportError:
            _has_shapely_ = False

        if _has_shapely_ and isinstance(shape_desc, (Polygon, MultiPolygon)):
            shape_desc = self.__shapelyToSVGPath__(shape_desc)
        if _has_shapely_ and isinstance(shape_desc, (LineString, MultiLineString)):
            shape_desc = self.__shapelyToSVGPath__(shape_desc)
            bg_shape_fill = 'none'
        if _has_shapely_ and isinstance(shape_desc, GeometryCollection):
            if len(shape_desc.geoms) > 0:
                raise DataError('LinkP.__transformBackgroundShapes__() - non-empty GeometryCollection not supported')
            return '', ''

        if isinstance(shape_desc, str):
            if shape_desc.lower().startswith('<circle'):
                return self.__transformCircleSVG__(name, shape_desc, bg_shape_label_color, bg_shape_opacity,
                                                   bg_shape_fill, bg_shape_stroke_w, bg_shape_stroke)
            else:
                return self.__transformPathDescription__(name, shape_desc, bg_shape_label_color, bg_shape_opacity,
                                                         bg_shape_fill, bg_shape_stroke_w, bg_shape_stroke)
        elif isinstance(shape_desc, list):
            return self.__transformPointsList__(name, shape_desc, bg_shape_label_color, bg_shape_opacity,
                                                bg_shape_fill, bg_shape_stroke_w, bg_shape_stroke)
        else:
            raise DataError(f'LinkP.__transformBackgroundShapes__() - unsupported type "{type(shape_desc)}"')

    #
    # __renderBackground__() - render background shapes into self.svg_background and self._dl_background_
    #
    def __renderBackground__(self):
        self.svg_background  = ''
        self._dl_background_ = DisplayList(self.wxh[0], self.wxh[1])
        if self.background is None:
            return
        _shapes_, _labels_ = [], []
        for name, shape_desc in self.background.items():
            _s, _l = self.__transformBackgroundShapes__(
                name, shape_desc,
                self.background_label_color,
                self.background_opacity,
                self.background_fill,
                self.background_stroke_w,
                self.background_stroke)
            _shapes_.append(_s)
            _labels_.append(_l)
            self.__backgroundShapeToDL__(_s, self._dl_background_)
        for _l in _labels_:
            self.__backgroundLabelToDL__(_l, self._dl_background_)
        self.svg_background = ''.join(_shapes_) + ''.join(_labels_)

    #
    # __backgroundShapeToDL__() - GPU geometry for a generated background shape string
    # (ellipse or M/L/C/Z path in screen coordinates); fills are polygon-filled,
    # strokes become line segments, cubic beziers are flattened
    #
    def __backgroundShapeToDL__(self, _s_, dl):
        import re
        if not _s_: return
        def _attr_(name, default=None):
            _m_ = re.search(f'{name}="([^"]*)"', _s_)
            return _m_.group(1) if _m_ else default
        _fill_         = _attr_('fill')
        _fill_opacity_ = float(_attr_('fill-opacity', '1.0'))
        _stroke_       = _attr_('stroke')
        _stroke_w_     = float(_attr_('stroke-width', '1.0'))
        if _s_.startswith('<ellipse'):
            cx, cy = float(_attr_('cx')), float(_attr_('cy'))
            rx, ry = float(_attr_('rx')), float(_attr_('ry'))
            import math as _math_
            _pts_ = [(cx + rx*_math_.cos(2*_math_.pi*i/48), cy + ry*_math_.sin(2*_math_.pi*i/48)) for i in range(48)]
            _subpaths_ = [(_pts_, True)]
        elif _s_.startswith('<path'):
            _d_ = _attr_('d', '')
            _tokens_ = _d_.split()
            _subpaths_, _cur_ = [], []
            i = 0
            while i < len(_tokens_):
                _t_ = _tokens_[i]
                if _t_ == 'M':
                    if len(_cur_) > 1: _subpaths_.append((_cur_, False))
                    _cur_ = [(float(_tokens_[i+1]), float(_tokens_[i+2]))]
                    i += 3
                elif _t_ == 'L':
                    _cur_.append((float(_tokens_[i+1]), float(_tokens_[i+2])))
                    i += 3
                elif _t_ == 'C':
                    if len(_cur_) > 0:
                        _p0_ = _cur_[-1]
                        _p1_ = (float(_tokens_[i+1]), float(_tokens_[i+2]))
                        _p2_ = (float(_tokens_[i+3]), float(_tokens_[i+4]))
                        _p3_ = (float(_tokens_[i+5]), float(_tokens_[i+6]))
                        for k in range(1, 17):
                            t = k / 16.0
                            mt = 1.0 - t
                            _cur_.append((mt*mt*mt*_p0_[0] + 3*mt*mt*t*_p1_[0] + 3*mt*t*t*_p2_[0] + t*t*t*_p3_[0],
                                          mt*mt*mt*_p0_[1] + 3*mt*mt*t*_p1_[1] + 3*mt*t*t*_p2_[1] + t*t*t*_p3_[1]))
                    i += 7
                elif _t_ == 'Z':
                    if len(_cur_) > 1: _subpaths_.append((_cur_, True))
                    _cur_ = []
                    i += 1
                else:
                    i += 1  # unknown token -- skip (svg path stays authoritative)
            if len(_cur_) > 1: _subpaths_.append((_cur_, False))
        else:
            return
        for _pts_, _closed_ in _subpaths_:
            if _fill_ is not None and _fill_ != 'none' and _fill_opacity_ > 0.0 and _closed_ and len(_pts_) >= 3:
                dl.polygon(_pts_, _fill_, opacity=_fill_opacity_)
            if _stroke_ is not None and _stroke_ != 'none':
                _seq_ = _pts_ + [_pts_[0]] if _closed_ else _pts_
                for j in range(len(_seq_) - 1):
                    dl.line(_seq_[j][0], _seq_[j][1], _seq_[j+1][0], _seq_[j+1][1], _stroke_, width=_stroke_w_)

    #
    # __backgroundLabelToDL__() - GPU glyphs for a generated background label string
    #
    def __backgroundLabelToDL__(self, _l_, dl):
        import re
        if not _l_: return
        _m_ = re.search(r'<text x="([^"]*)" y="([^"]*)" text-anchor="middle"[^>]*fill="([^"]*)" font-size="([^"]*)px">([^<]*)</text>', _l_)
        if _m_ is None: return
        _x_, _y_, _co_, _th_, _txt_ = float(_m_.group(1)), float(_m_.group(2)), _m_.group(3), float(_m_.group(4)), _m_.group(5)
        dl.text(self.p2s, _txt_, _x_, _y_, txt_h=_th_, anchor='middle', color=_co_, svg='')

    #
    # __renderSVG__()
    #
    def __renderSVG__(self, rand_id):
        self._gpu_payload_ = self._gpu_dl_ = None   # invalidate GPU state cached from a template
        if self.view_window_orig is None:
            self.view_window_orig = self.view_window
        self._render_invalid_ = False

        w, h = self.wxh
        _bg_co_     = self.p2s.colorTyped('background', 'default')
        _border_co_ = self.p2s.colorTyped('axis', 'inner')

        svg = [f'<svg x="0" y="0" width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">']
        #
        # Cloud Icon in <defs> -- attribution is as follows:
        #
        # Source: https://www.svgrepo.com/svg/520637/cloud
        # License:  CC Attribution License
        # COLLECTION: Xnix Circular Interface Icons
        # AUTHOR: Ankush Syal
        # Modified To Remove Bottom Path (Second Path)
        #
        svg.append(
            '<defs>'
            '<g id="cloud" transform="translate(-50,-25)">'
            '<svg x="0" y="0" width="100px" height="50px" viewBox="-5 -5.5 35 35" xmlns="http://www.w3.org/2000/svg">'
            '<path fill-rule="evenodd" clip-rule="evenodd" '
            'd="M14.091 7.00151C14.9928 6.9746 15.8684 7.30725 16.5249 7.9262C17.1813 8.54515 17.5649 9.39965 '
            '17.591 10.3015C17.5914 10.6221 17.5425 10.9408 17.446 11.2465C18.6091 11.4334 19.4729 12.4239 '
            '19.5 13.6015C19.4586 14.9664 18.32 16.0402 16.955 16.0015H8.045C6.67999 16.0402 5.54137 14.9664 '
            '5.5 13.6015C5.52293 12.4783 6.31258 11.5171 7.41 11.2765C7.41 11.2512 7.41 11.2262 7.41 11.2015C'
            '7.45137 9.83659 8.58999 8.76283 9.955 8.80151C10.2738 8.80108 10.5901 8.85764 10.889 8.96851C'
            '11.4867 7.74927 12.7333 6.98347 14.091 7.00151Z" '
            'stroke="#000000" stroke-linecap="round" stroke-linejoin="round"/>'
            '</svg></g></defs>'
        )
        svg.append(f'<rect x="0" y="0" width="{w}" height="{h}" fill="{_bg_co_}" />')

        # Background shapes (behind hulls, links, and nodes)
        self.__renderBackground__()
        svg.append(self.svg_background)

        # Convex hulls (behind links and nodes)
        svg.append(self.__renderConvexHull__())

        # Links
        svg.extend(self._link_svg_list_)

        # Nodes
        svg.extend(self._node_svg_list_)

        # Deferred: node labels on top
        svg.extend(self._node_label_svg_)

        # Legend (drawn into the strip reserved by __legendPrepare__); the colorbar
        # domain is finalized here -- __applyColorToDF__ has run by now
        self._dl_legend_ = None
        if getattr(self, 'legend_info', None) is not None and self._legend_region_ is not None:
            if self.legend_info.kind == 'colorbar':
                if self.color_stat_range_shared is not None and not getattr(self, '_legend_stretched_', False):
                    _vmin_, _vmax_ = self.color_stat_range_shared
                else:
                    _vmin_, _vmax_ = self._legend_stat_min_, self._legend_stat_max_
                self.p2s.legendInfoColorbarFinalize(self.legend_info, self.legend_spec, _vmin_, _vmax_)
            self._dl_legend_ = self.p2s.legendRenderDL(self.wxh, self._legend_region_, self.legend_spec,
                                                       self.legend_info, self.txt_h)
            svg.append(self._dl_legend_.svg())

        # Border
        if self.draw_border:
            svg.append(f'<rect x="0" y="0" width="{w-1}" height="{h-1}" fill="none" stroke="{_border_co_}" stroke-width="1" />')

        svg.append('</svg>')
        # trim verbose float tails from the finished SVG (idempotent) -- rounded here
        # rather than in __init__ so the interactive renderSVG() re-render path is
        # covered too. See Polars2SVG.roundSvgFloats
        self.svg = self.p2s.roundSvgFloats(''.join(svg))

    #
    # renderSmallMultiples() - smallp integration
    # - each panel gets the same pos/relationships (graph layout) but a different data subset
    #
    def renderSmallMultiples(self, df_all, df_lu, all_key):
        _kwargs_ = {}

        # The template's computed view_window would otherwise be inherited by every sub-panel
        # and force all panels to share the same bounds. Reset it unless the user explicitly
        # set a view_window on the template (in which case they want it shared).
        if not self._view_window_user_set_:
            _kwargs_['view_window'] = None

        _needs_ref_ = (self.p2s.SM_X     in self.sm_shared or
                       self.p2s.SM_Y     in self.sm_shared or
                       self.p2s.SM_COUNT in self.sm_shared or
                       self.p2s.SM_COLOR in self.sm_shared)
        if _needs_ref_:
            _ref_ = LinkP(df=df_all, template=self, **{k: v for k, v in _kwargs_.items()
                                                        if k == 'view_window'})
            if self.p2s.SM_X in self.sm_shared:
                _kwargs_['_shared_view_x_'] = (_ref_.wx0, _ref_.wx1)
            if self.p2s.SM_Y in self.sm_shared:
                _kwargs_['_shared_view_y_'] = (_ref_.wy0, _ref_.wy1)
            if self.p2s.SM_COUNT in self.sm_shared and _ref_._count_min_ is not None:
                _kwargs_['count_range_shared'] = (_ref_._count_min_, _ref_._count_max_)
            if self.p2s.SM_COLOR in self.sm_shared and _ref_._color_stat_min_ is not None:
                _kwargs_['color_stat_range_shared'] = (_ref_._color_stat_min_, _ref_._color_stat_max_)
        return {k: LinkP(df=v, template=self, **_kwargs_) for k, v in df_lu.items()}

    #
    # render_with() - create a new instance with overrides (used by smallp cycle_by mode)
    #
    def render_with(self, df, **overrides):
        return LinkP(df=df, template=self, **overrides)

    # -------------------------------------------------------------------------
    # Interactive methods — called by linkpi() on dfs_layout entries
    # Modeled on RTLink in racetrack_svg_framework/rtsvg/rt_link_mixin.py
    # -------------------------------------------------------------------------

    def invalidateRender(self):
        self._render_invalid_ = True

    def renderSVG(self):
        if self._render_invalid_:
            rand_id = random.randint(0, 2**32)  # nosec B311 - non-cryptographic SVG id scoping, see SECURITY.md
            self.gatherMetrics(self.__calculateGeometry__)
            self.gatherMetrics(self.__calculateScreenCoordinates__)
            self.gatherMetrics(self.__renderLinks__)
            self.gatherMetrics(self.__renderNodes__)
            self.gatherMetrics(self.__renderSVG__, rand_id)
        return self.svg

    def setViewWindow(self, view_window):
        self.view_window = view_window
        self.wx0, self.wy0, self.wx1, self.wy1 = view_window
        self.invalidateRender()

    def getViewWindow(self):
        return self.view_window

    def applyScrollEvent(self, scroll_amount, coordinate=None):
        factor = 1.0 + scroll_amount / 1000.0
        wx0, wy0, wx1, wy1 = self.view_window
        cx = self.xT_inv(coordinate[0]) if coordinate is not None else (wx0 + wx1) / 2
        cy = self.yT_inv(coordinate[1]) if coordinate is not None else (wy0 + wy1) / 2
        self.setViewWindow((cx + (wx0 - cx) * factor, cy + (wy0 - cy) * factor,
                            cx + (wx1 - cx) * factor, cy + (wy1 - cy) * factor))
        return True

    def applyMiddleClick(self, coordinate):
        if self.view_window != self.view_window_orig:
            self.setViewWindow(self.view_window_orig)
            return True
        return False

    def applyMiddleDrag(self, coordinate, delta):
        if self.view_window is not None:
            dwx = self.xT_inv(coordinate[0]) - self.xT_inv(coordinate[0] + delta[0])
            dwy = self.yT_inv(coordinate[1]) - self.yT_inv(coordinate[1] + delta[1])
            wx0, wy0, wx1, wy1 = self.view_window
            self.setViewWindow((wx0 + dwx, wy0 + dwy, wx1 + dwx, wy1 + dwy))
            return True
        return False

    def applyViewConfiguration(self, other):
        other_vw = other.getViewWindow()
        if other_vw != self.getViewWindow():
            self.setViewWindow(other_vw)
            return True
        return False

    def overlappingEntities(self, to_intersect):
        from shapely.geometry import Point
        _str_to_key_ = {str(k): k for k in self.pos.keys()}
        _set_ = set()
        for sx, sy, nm_list in self.df_node.select('__sx__', '__sy__', '__nm__').iter_rows():
            if Point(sx, sy).within(to_intersect):
                for nm in (nm_list if isinstance(nm_list, (list, set)) else [nm_list]):
                    _set_.add(_str_to_key_.get(nm, nm))
        return list(_set_)

    def entitiesAtPoint(self, xy):
        from shapely.geometry import Polygon
        poly = Polygon([(xy[0] - 5, xy[1] - 5), (xy[0] - 5, xy[1] + 5),
                        (xy[0] + 5, xy[1] + 5), (xy[0] + 5, xy[1] - 5)])
        return self.overlappingEntities(poly)

    def nodeColor(self, node):
        return self.color_nodes_final.get(node)

    def nodesWithColor(self, color):
        return {k for k, v in self.color_nodes_final.items() if v == color}

    def nodeShape(self, node):
        return 'circle'

    def nodesWithShape(self, shape):
        return set(self.color_nodes_final.keys()) if shape == 'circle' else set()

    _NODE_PATH_OPS_ = [pl.lit('M '), pl.col('__sx__') - 5, pl.lit(' '), pl.col('__sy__') - 5,
                       pl.lit(' l 10 0 l 0 10 l -10 0 z')]

    def __filterNodesBySelection__(self, my_selection):
        _strs_ = {str(e) for e in my_selection}
        return self.df_node.explode('__nm__').filter(pl.col('__nm__').is_in(_strs_))

    def __createPathDescriptionForAllEntities__(self):
        return ' '.join(
            self.df_node.unique(['__sx__', '__sy__'])
                        .with_columns(pl.concat_str(*self._NODE_PATH_OPS_).alias('__svg__'))
                        ['__svg__']
        )

    def __createPathDescriptionOfSelectedEntities__(self, my_selection=None):
        _fallback_ = 'M -100 -100 l 10 0 l 0 10 l -10 0 l 0 -10 Z'
        if not my_selection:
            return _fallback_
        _df_ = self.__filterNodesBySelection__(my_selection)
        if len(_df_) == 0:
            return _fallback_
        return ' '.join(
            _df_.unique(['__sx__', '__sy__'])
                .with_columns(pl.concat_str(*self._NODE_PATH_OPS_).alias('__svg__'))
                ['__svg__']
        )

    def __moveSelectedEntities__(self, dxy, my_selection=None):
        if not my_selection:
            return {}
        _updated_    = {}
        _str_to_key_ = {str(k): k for k in self.pos.keys()}
        _df_ = self.__filterNodesBySelection__(my_selection)
        for sx, sy, nm in _df_.select('__sx__', '__sy__', '__nm__').iter_rows():
            _key_ = _str_to_key_.get(nm, nm)
            self.pos[_key_] = (self.xT_inv(sx + dxy[0]), self.yT_inv(sy + dxy[1]))
            _updated_[_key_] = self.pos[_key_]
        self.invalidateRender()
        return _updated_

    def labelOnly(self, label_set):
        self.label_only = set(label_set) if label_set else set()
        self.invalidateRender()

    def drawLabels(self, draw_labels):
        self.draw_labels = draw_labels
        self.invalidateRender()
