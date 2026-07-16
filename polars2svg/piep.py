import polars as pl
import time
import random
import zlib
import colorsys
from math import pi, cos, sin, atan2, sqrt, radians

import polars2svg
from polars2svg.p2s_displaylist import DisplayList
from polars2svg.export import ExportMixin

#
# Piechart  (pie / donut / waffle)
#
# Mirrors Histop in parameters and usage: bins become slices, count= sets each slice's
# magnitude (its share of the whole), color= sets each slice's color, and descending sets
# the direction slices are laid out around the ring (by count).  Renders through a DisplayList so the
# same compute drives both the SVG string and the WebGPU payload (for panelize()).
#
class Piep(ExportMixin):

    _VALID_KWARGS = frozenset({
        'template', 'df',
        'bin_by', 'count', 'count_range', 'count_range_shared',
        'color', 'descending', 'style',
        'wxh', 'insets', 'draw_context', 'draw_labels', 'draw_border', 'txt_h',
        'start_angle', 'donut_ratio', 'waffle_n', 'min_slice_deg',
        'sm_shared', 'use_lazy_execution', 'color_stat_range_shared',
        'legend',
        # internal (set by smallp small-multiple sharing) -----------------------
        '_shared_order_', '_base_slices_',
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
            self.gatherMetrics(self.__computeAggregates__)
            self.gatherMetrics(self.__constructGeometry__)
            self.gatherMetrics(self.__renderSVG__, rand_id)
        # trim verbose float tails from the finished SVG (idempotent; no-op on the
        # dataless placeholder) -- see Polars2SVG.roundSvgFloats
        self.svg = self.p2s.roundSvgFloats(self.svg)
        self.t_end     = time.time()
        self.t_overall = self.t_end - self.t_start

    def _repr_svg_(self): return self.svg

    #
    # webgpu() - WebGPU payload of the same render (buffers + manifest); the polars
    # compute is shared with the SVG path -- only the serialization differs
    #
    def webgpu(self):
        if getattr(self, '_dl_', None) is None: return None
        return self._dl_.webgpu_payload(self.p2s.glyphAtlas())

    # gpuDisplayList() - consumed by smallp when this component renders as a cell
    def gpuDisplayList(self):
        return getattr(self, '_dl_', None)

    def gatherMetrics(self, callable, *args, **kwargs):
        t0 = time.time()
        _results_ = callable(*args, **kwargs)
        t1 = time.time()
        if callable.__name__ not in self.timing_metrics: self.timing_metrics[callable.__name__] = 0.0
        self.timing_metrics[callable.__name__] += t1 - t0
        return _results_

    # ── Input parsing ─────────────────────────────────────────────────────────

    def __parseInput__(self, *args, **kwargs):
        _unknown_ = set(kwargs) - self._VALID_KWARGS
        if _unknown_:
            raise TypeError(f'Piep: unexpected keyword argument(s): {sorted(_unknown_)}')

        # Single source of truth for every parameter (name -> from-scratch default);
        # drives both the from-scratch assignment and the keyword-override copy below.
        _defaults_ = {
            'bin_by':                  None,
            'count':                   self.p2s.ROW_COUNTp,
            'count_range':             None,
            'count_range_shared':      None,
            'color':                   None,
            'descending':              True,
            'style':                   self.p2s.PIEp,
            'wxh':                     (160, 160),
            'insets':                  (2, 2),
            'draw_context':            True,
            'draw_labels':             False,
            'draw_border':             True,
            'txt_h':                   12,
            'start_angle':             -90.0,
            'donut_ratio':             0.55,
            'waffle_n':                10,
            'min_slice_deg':           3.0,
            'sm_shared':               set(),
            'use_lazy_execution':      True,
            'color_stat_range_shared': None,
            'legend':                  False,
            '_shared_order_':          None,
            '_base_slices_':           None,
        }
        self.p2s.assertParamSpecMatches('Piep', self._VALID_KWARGS, _defaults_)

        self.df, self.df_orig = None, None

        # Template support
        self.template = None
        for i in range(len(args)):
            if isinstance(args[i], Piep): self.template = args[i]
        if 'template' in kwargs: self.template = kwargs['template']
        if self.template is not None:
            _template_copy_ = self.template
            self.p2s._clone_template_state(self, _template_copy_)
            self.template = _template_copy_
        else:
            self.p2s.assignScratchDefaults(self, _defaults_)
            # from-scratch builds only — a template clone is an exact snapshot and
            # must not re-apply session defaults (see Polars2SVG._apply_defaults)
            kwargs = self.p2s._apply_defaults('piep', kwargs)

        # A template re-render must not inherit stale one-shot sharing state unless
        # the caller explicitly re-supplies it.
        self._shared_order_ = None
        self._base_slices_  = None

        # Extract DataFrame
        _new_df_ = None
        for _arg_ in args:
            if isinstance(_arg_, pl.DataFrame):
                if _new_df_ is None: _new_df_ = _arg_
                else:                raise ValueError('Piep.__parseInput__(): df already set')
        if 'df' in kwargs:
            if _new_df_ is None: _new_df_ = kwargs['df']
            else:                raise ValueError('Piep.__parseInput__(): df already set')
        if _new_df_ is not None:
            self.df = self.df_orig = _new_df_
        if self.df is not None and '__p2s_index__' not in self.df.columns:
            self.df = self.df.with_row_index('__p2s_index__')

        # Extract bin_by from positional args (string or tuple)
        for arg in args:
            if   isinstance(arg, pl.DataFrame): pass
            elif isinstance(arg, Piep):         pass
            elif isinstance(arg, str):
                if self.bin_by is None: self.bin_by = arg
                else:                   raise ValueError('Piep.__parseInput__(): bin_by already set')
            elif isinstance(arg, tuple):
                if self.bin_by is None: self.bin_by = arg
                else:                   raise ValueError('Piep.__parseInput__(): bin_by already set')
            else:
                raise ValueError(f'Piep.__parseInput__(): Unknown argument type: {type(arg)}')

        self.p2s.assignKwargOverrides(self, _defaults_, kwargs)

        # "No data" placeholder for early error visibility -- only ever seen when
        # no df is supplied (a successful render overwrites self.svg); makes a
        # dropped-df plumbing mistake visible instead of a silently blank canvas.
        self.wxh = self.p2s.normalizeWxh(self.wxh, 'Piep')
        w, h = self.wxh
        self.svg = self.p2s.placeholderSVG(w, h)

    def __validateInput__(self):
        # Normalize legend= eagerly so a bad spec fails fast (raises InvalidSpecError).
        self.legend_spec = self.p2s.legendResolveSpec(self.legend)
        if self.df is None: return
        self.p2s.checkReservedColumns(self.df, 'Piep')

        if self.bin_by is None:
            raise ValueError('Piep.__validateInput__(): bin_by must be specified')
        if isinstance(self.bin_by, str):
            self._bin_cols_ = [self.bin_by]
            self._bin_col_  = self.bin_by
        elif isinstance(self.bin_by, tuple):
            self._bin_cols_ = list(self.bin_by)
            self._bin_col_  = '__bin__'
        else:
            raise ValueError(f'Piep.__validateInput__(): bin_by must be str or tuple, got {type(self.bin_by)}')

        for _f_ in self._bin_cols_:
            if _f_ not in self.df.columns:
                raise ValueError(f'Piep.__validateInput__(): bin_by field "{_f_}" not found')

        # Validate count
        if self.count != self.p2s.ROW_COUNTp:
            if isinstance(self.count, str) and not self.p2s.columnInDataFrame(self.count, self.df):
                raise ValueError(f'Piep.__validateInput__(): count field "{self.count}" not found')
            elif isinstance(self.count, tuple):
                for _f_ in self.count:
                    if isinstance(_f_, str) and not self.p2s.columnInDataFrame(_f_, self.df):
                        raise ValueError(f'Piep.__validateInput__(): count field "{_f_}" not found')

        # Resolve + validate color (mirrors xyp's color model)
        self.__resolveColor__()

        # Validate style
        _valid_styles_ = {self.p2s.PIEp, self.p2s.DONUTp, self.p2s.WAFFLEp}
        if self.style not in _valid_styles_:
            raise ValueError(f'Piep.__validateInput__(): style must be one of {_valid_styles_}')

        # Warn on SM_* values that Piep does not support
        _supported_sm_   = {self.p2s.SM_SLICE_ORDERp, self.p2s.SM_PARTOFWHOLEp,
                            self.p2s.SM_COLOR, self.p2s.SM_COUNT}
        _unsupported_sm_ = set(self.sm_shared) - _supported_sm_
        if _unsupported_sm_:
            self.p2s.logger.warning(
                f'Piep: sm_shared contains {_unsupported_sm_} which Piep does not support '
                f'(supported: SM_SLICE_ORDERp, SM_PARTOFWHOLEp, SM_COLOR, SM_COUNT); these values will be ignored'
            )

        # wxh was normalized to a canonical (int, int) tuple in __parseInput__ via
        # self.p2s.normalizeWxh(); no re-validation needed here.
        w, h = self.wxh
        if w < 56 or h < 56:
            self.draw_context = False

    # ── Color specification (mirrors xyp's color= model) ─────────────────────
    #
    # Resolves self.color into an internal mode + parameters:
    #   'none'     — no field; every slice uses the default data colour (xyp default)
    #   'fixed'    — a single '#RRGGBB' applied to every slice
    #   'hexlist'  — an ordered list of hex colours cycled across the slices
    #   'cset'     — categorical: a slice with one unique field value takes that
    #                value's colour, otherwise a shared "set" colour  (xyp CSETp)
    #   'spectrum' — the slice's colour-stat (sum/min/median/mean/max, unique-count,
    #                or raw row count) mapped onto the colour spectrum, normalized
    #                linearly (magnitude) or by rank (stretched)
    #
    def __resolveColor__(self):
        self._color_mode_         = 'none'
        self._color_fields_       = []
        self._color_field_        = None
        self._color_agg_          = None      # sum|min|median|mean|max|setcount|rowcount
        self._color_is_stretched_ = False
        self._fixed_color_        = None
        self._hex_list_           = None

        _MAG_ = {self.p2s.CMAGNITUDE_SUMp: 'sum', self.p2s.CMAGNITUDE_MINp: 'min',
                 self.p2s.CMAGNITUDE_MEDIANp: 'median', self.p2s.CMAGNITUDE_MEANp: 'mean',
                 self.p2s.CMAGNITUDE_MAXp: 'max'}
        _STR_ = {self.p2s.CSTRETCHED_SUMp: 'sum', self.p2s.CSTRETCHED_MINp: 'min',
                 self.p2s.CSTRETCHED_MEDIANp: 'median', self.p2s.CSTRETCHED_MEANp: 'mean',
                 self.p2s.CSTRETCHED_MAXp: 'max'}

        _c_ = self.color

        # ── enum-only (CROW_* need no field) ──────────────────────────────
        if isinstance(_c_, self.p2s.ColorTypeP):
            if _c_ in (self.p2s.CROW_MAGNITUDEp, self.p2s.CROW_STRETCHEDp):
                self._color_mode_         = 'spectrum'
                self._color_agg_          = 'rowcount'
                self._color_is_stretched_ = (_c_ == self.p2s.CROW_STRETCHEDp)
                return
            raise ValueError(f'Piep: color enum {_c_} requires a field (e.g. ("field", {_c_}))')

        if _c_ is None:
            return
        if isinstance(_c_, self.p2s.HexColorString):
            self._color_mode_, self._fixed_color_ = 'fixed', _c_
            return

        # ── list / tuple: split into hex literals, fields, and a colour enum ─
        if isinstance(_c_, (list, tuple)):
            _items_ = list(_c_)
            if len(_items_) > 0 and all(isinstance(i, self.p2s.HexColorString) for i in _items_):
                self._color_mode_, self._hex_list_ = 'hexlist', list(_items_)
                return
            _fields_ = [i for i in _items_ if isinstance(i, str) and not isinstance(i, self.p2s.HexColorString)]
            _enums_  = [i for i in _items_ if isinstance(i, self.p2s.ColorTypeP)]
            if len(_enums_) > 1:
                raise ValueError(f'Piep: color has more than one colour enum ({_enums_})')
            _enum_ = _enums_[0] if _enums_ else None
        elif isinstance(_c_, str):
            _fields_, _enum_ = [_c_], None
        else:
            raise ValueError(f'Piep: unsupported color specification: {_c_!r}')

        # validate fields exist
        for _f_ in _fields_:
            if not self.p2s.columnInDataFrame(_f_, self.df):
                raise ValueError(f'Piep.__resolveColor__(): color field "{_f_}" not found')
        if not _fields_:
            raise ValueError(f'Piep: color enum {_enum_} requires a field')
        self._color_fields_ = _fields_

        # ── CROW enum inside a list ignores the field (xyp parity) ─────────
        if _enum_ in (self.p2s.CROW_MAGNITUDEp, self.p2s.CROW_STRETCHEDp):
            self._color_mode_, self._color_agg_ = 'spectrum', 'rowcount'
            self._color_is_stretched_ = (_enum_ == self.p2s.CROW_STRETCHEDp)
            return

        _multi_    = len(_fields_) > 1
        _cat_field_ = '__color__' if _multi_ else _fields_[0]

        if _enum_ is None:
            # implicit: numeric single field -> magnitude sum; otherwise categorical set
            _is_num_ = (not _multi_) and self.p2s.numericColumn(self.df, _fields_[0])
            if not _multi_:
                self.p2s.logDtypeKeyedColor('Piep', _fields_[0], _is_num_)
            if _is_num_:
                self._color_mode_, self._color_agg_, self._color_field_ = 'spectrum', 'sum', _fields_[0]
            else:
                self._color_mode_, self._color_field_ = 'cset', _cat_field_
        elif _enum_ == self.p2s.CSETp:
            self._color_mode_, self._color_field_ = 'cset', _cat_field_
        elif _enum_ in (self.p2s.CSET_MAGNITUDEp, self.p2s.CSET_STRETCHEDp):
            self._color_mode_, self._color_agg_, self._color_field_ = 'spectrum', 'setcount', _cat_field_
            self._color_is_stretched_ = (_enum_ == self.p2s.CSET_STRETCHEDp)
        elif _enum_ in _MAG_ or _enum_ in _STR_:
            _agg_ = _MAG_.get(_enum_) or _STR_[_enum_]
            if not self.p2s.numericColumn(self.df, _fields_[0]):
                self.p2s.logger.warning(f'Piep: color {_enum_} needs a numeric field; '
                                        f'"{_fields_[0]}" is categorical — colouring by set instead')
                self._color_mode_, self._color_field_ = 'cset', _cat_field_
            else:
                self._color_mode_, self._color_agg_, self._color_field_ = 'spectrum', _agg_, _fields_[0]
                self._color_is_stretched_ = (_enum_ in _STR_)
        else:
            raise ValueError(f'Piep: unsupported color enum {_enum_}')

    def __addColumnsToDataFrame__(self):
        _ops_ = []

        # Multi-field bin: concatenate to '__bin__' (non-printable separator so distinct
        # field tuples can't collide into one slice; shown as '|' via formatMultiFieldValue)
        if isinstance(self.bin_by, tuple):
            _ops_.append(pl.concat_str([pl.col(c).cast(pl.String) for c in self._bin_cols_],
                                       separator=self.p2s.MULTI_FIELD_SEP).alias('__bin__'))

        # Count field t-fields
        if isinstance(self.count, str) and self.p2s.isTField(self.count, df=self.df_orig):
            self.p2s.warnIfTFieldAliasCollides(self.count, self.df_orig, 'Piep')
            _ops_.append(self.p2s.polarsOperationForTField(self.count).alias(self.count))
        elif isinstance(self.count, tuple):
            for _f_ in self.count:
                if isinstance(_f_, str) and self.p2s.isTField(_f_, df=self.df_orig):
                    self.p2s.warnIfTFieldAliasCollides(_f_, self.df_orig, 'Piep')
                    _ops_.append(self.p2s.polarsOperationForTField(_f_).alias(_f_))

        # Color field t-fields
        for _f_ in self._color_fields_:
            if self.p2s.isTField(_f_, df=self.df_orig):
                self.p2s.warnIfTFieldAliasCollides(_f_, self.df_orig, 'Piep')
                _ops_.append(self.p2s.polarsOperationForTField(_f_).alias(_f_))

        # Multi-field categorical colour: concatenate the fields to '__color__'
        # (non-printable separator so distinct field tuples get distinct colors)
        if self._color_field_ == '__color__':
            _ops_.append(pl.concat_str([pl.col(c).cast(pl.String) for c in self._color_fields_],
                                       separator=self.p2s.MULTI_FIELD_SEP).alias('__color__'))

        if len(_ops_) > 0:
            if self.use_lazy_execution: self.df = self.df.lazy().with_columns(_ops_).collect()
            else:                       self.df = self.df.with_columns(_ops_)

    # ── Aggregate expressions (shared with Histop's contract) ────────────────

    def __countAggExpr__(self):
        if self.count == self.p2s.ROW_COUNTp:
            return pl.len().alias('__count__')
        elif isinstance(self.count, str):
            _is_num_ = self.p2s.numericColumn(self.df, self.count)
            self.p2s.logDtypeKeyedCount('Piep', self.count, _is_num_)
            if _is_num_: return pl.col(self.count).sum()    .alias('__count__')
            else:        return pl.col(self.count).n_unique().alias('__count__')
        elif isinstance(self.count, tuple):
            _fields_ = [_f_ for _f_ in self.count if isinstance(_f_, str)]
            if self.p2s.SETp in self.count:
                return pl.col(_fields_[0]).n_unique().alias('__count__')
            elif len(_fields_) == 1:
                return pl.col(_fields_[0]).sum().alias('__count__')
            else:
                return pl.struct(_fields_).n_unique().alias('__count__')
        return pl.len().alias('__count__')

    # ── Aggregation ─────────────────────────────────────────────────────────

    def __computeAggregates__(self):
        # Coloring mode was resolved in __resolveColor__; expose the two flags the
        # rest of the pipeline (spectrum range, smallp sharing) keys off of.
        self._color_is_spectrum_ = (self._color_mode_ == 'spectrum')
        self._color_is_crow_     = (self._color_agg_ == 'rowcount')

        _agg_exprs_ = [self.__countAggExpr__(), pl.len().alias('__row_count__')]
        # Spectrum: aggregate the colour-stat per slice.
        if self._color_mode_ == 'spectrum' and self._color_agg_ not in (None, 'rowcount'):
            _cf_ = self._color_field_
            if   self._color_agg_ == 'setcount': _agg_exprs_.append(pl.col(_cf_).n_unique().cast(pl.Float64).alias('__color_stat__'))
            elif self._color_agg_ == 'sum':      _agg_exprs_.append(pl.col(_cf_).sum()   .alias('__color_stat__'))
            elif self._color_agg_ == 'min':      _agg_exprs_.append(pl.col(_cf_).min()   .alias('__color_stat__'))
            elif self._color_agg_ == 'median':   _agg_exprs_.append(pl.col(_cf_).median().alias('__color_stat__'))
            elif self._color_agg_ == 'mean':     _agg_exprs_.append(pl.col(_cf_).mean()  .alias('__color_stat__'))
            elif self._color_agg_ == 'max':      _agg_exprs_.append(pl.col(_cf_).max()   .alias('__color_stat__'))
        # Categorical set: per slice, the unique-value count and a representative value.
        if self._color_mode_ == 'cset':
            _cf_ = self._color_field_
            _agg_exprs_.append(pl.col(_cf_).cast(pl.String).n_unique().alias('__cset_n__'))
            _agg_exprs_.append(pl.col(_cf_).cast(pl.String).first()   .alias('__cset_first__'))

        if self.use_lazy_execution:
            self.df_agg = self.df.lazy().group_by(self._bin_col_).agg(_agg_exprs_).collect()
        else:
            self.df_agg = self.df.group_by(self._bin_col_).agg(_agg_exprs_)

        if self._color_agg_ == 'rowcount':
            self.df_agg = self.df_agg.with_columns(pl.col('__row_count__').cast(pl.Float64).alias('__color_stat__'))

        # Per-slice categorical colour value (single value, or None → shared set colour)
        self._colorset_lu_ = {}
        if self._color_mode_ == 'cset':
            for _row_ in self.df_agg.iter_rows(named=True):
                self._colorset_lu_[_row_[self._bin_col_]] = (
                    _row_['__cset_first__'] if _row_['__cset_n__'] == 1 else None)

        # ── ORDER OF SLICES ────────────────────────────────────────────────
        # Slices are laid out by their count= magnitude (their share of the whole);
        # descending puts the largest slice first.  The bin value is a stable
        # secondary key so tied counts order deterministically (group_by output
        # order is otherwise arbitrary) — this keeps re-renders identical.
        self._sorted_bins_ = (self.df_agg
            .select([self._bin_col_, pl.col('__count__').alias('__order_metric__')])
            .with_columns(pl.col(self._bin_col_).cast(pl.String).alias('__order_tie__'))
            .sort(['__order_metric__', '__order_tie__'], descending=[self.descending, False])
            [self._bin_col_].to_list())

        # Raw per-bin lookups (count, color-stat), keyed by individual bin value
        _raw_count_ = {row[self._bin_col_]: float(row['__count__'])
                       for row in self.df_agg.iter_rows(named=True)}
        self._colorstat_lu_ = {}
        if '__color_stat__' in self.df_agg.columns:
            self._colorstat_lu_ = {row[self._bin_col_]: (None if row['__color_stat__'] is None
                                                         else float(row['__color_stat__']))
                                   for row in self.df_agg.iter_rows(named=True)}

        _total_ = sum(v for v in _raw_count_.values() if v > 0) or 1.0
        self._whole_total_ = _total_

        # A shared layout (small multiples) fixes both the slice order AND which bins
        # are shown individually vs folded into "(other)": derive it from whichever
        # reference the panel was handed.
        _shared_layout_ = None
        if   self._shared_order_ is not None: _shared_layout_ = list(self._shared_order_)
        elif self._base_slices_  is not None: _shared_layout_ = [s['bin'] for s in self._base_slices_]

        # Original bin values folded into the "(other)" slice; kept so interactive
        # selection (recordsAt / filterByRectangle / filterBySubstring) can expand
        # "(other)" back to the rows it actually represents.
        self._other_members_ = []

        if _shared_layout_ is not None:
            # ── SHARED: honor the reference's visible set + order; fold the rest ──
            _visible_set_ = {b for b in _shared_layout_ if b != '(other)'}
            _other_members_ = [_b_ for _b_ in self._sorted_bins_ if _b_ not in _visible_set_]
            _other_sum_     = sum(_raw_count_.get(_b_, 0.0) for _b_ in _other_members_
                                  if _raw_count_.get(_b_, 0.0) > 0)
            self._count_lu_ = {b: _raw_count_[b] for b in _visible_set_ if b in _raw_count_}
            if '(other)' in _shared_layout_ and _other_sum_ > 0:
                self._count_lu_['(other)']  = _other_sum_
                self._colorstat_lu_.setdefault('(other)', None)
                self._other_members_ = _other_members_
            self._sorted_bins_ = [b for b in _shared_layout_ if b in self._count_lu_]
        else:
            # ── STANDALONE: collapse any slice narrower than min_slice_deg ────────
            self._count_lu_ = dict(_raw_count_)
            _min_deg_ = float(self.min_slice_deg) if self.min_slice_deg else 0.0
            if _min_deg_ > 0.0:
                _keep_, _other_members_, _other_ = [], [], 0.0
                for _b_ in self._sorted_bins_:
                    _c_ = self._count_lu_.get(_b_, 0.0)
                    if _c_ > 0 and (_c_ / _total_) * 360.0 < _min_deg_:
                        _other_ += _c_
                        _other_members_.append(_b_)
                    else:
                        _keep_.append(_b_)
                if _other_ > 0 and len(_keep_) < len(self._sorted_bins_):
                    self._sorted_bins_ = _keep_ + ['(other)']
                    for _b_ in list(self._count_lu_.keys()):
                        if _b_ not in _keep_: del self._count_lu_[_b_]
                    self._count_lu_['(other)']  = _other_
                    self._colorstat_lu_.setdefault('(other)', None)
                    self._other_members_ = _other_members_

        # ── COLOR-STAT RANGE (spectrum coloring) ───────────────────────────
        self._color_stat_min_ = None
        self._color_stat_max_ = None
        if self._color_is_spectrum_:
            if self.color_stat_range_shared is not None:
                self._color_stat_min_, self._color_stat_max_ = self.color_stat_range_shared
            else:
                _vals_ = [v for v in self._colorstat_lu_.values() if v is not None]
                if _vals_:
                    self._color_stat_min_ = round(min(_vals_), 3)
                    self._color_stat_max_ = round(max(_vals_), 3)

        # ── COUNT RANGE (mostly for smallp SM_COUNT parity) ────────────────
        if self.count_range_shared is not None:
            self._count_min_, self._count_max_ = self.count_range_shared
        elif self.count_range is not None:
            self._count_min_, self._count_max_ = self.count_range
        else:
            _cvals_ = [v for v in self._count_lu_.values()]
            self._count_min_ = 0.0
            self._count_max_ = max(_cvals_) if _cvals_ else 1.0

    # ── Color resolution ─────────────────────────────────────────────────────

    def __dataColorShades__(self, hexcolor, n=5):
        '''n shades of hexcolor spread by small (barely perceptible) lightness steps.'''
        try:
            _r_ = int(hexcolor[1:3], 16) / 255.0
            _g_ = int(hexcolor[3:5], 16) / 255.0
            _b_ = int(hexcolor[5:7], 16) / 255.0
        except (ValueError, IndexError):
            return [hexcolor] * n
        _h_, _l_, _s_ = colorsys.rgb_to_hls(_r_, _g_, _b_)
        _step_    = 0.022                                   # ~just-noticeable lightness step
        _offsets_ = [(_i_ - (n - 1) / 2.0) * _step_ for _i_ in range(n)]
        _out_ = []
        for _off_ in _offsets_:
            _ll_ = min(1.0, max(0.0, _l_ + _off_))
            _rr_, _gg_, _bb_ = colorsys.hls_to_rgb(_h_, _ll_, _s_)
            _out_.append('#%02x%02x%02x' % (round(_rr_ * 255), round(_gg_ * 255), round(_bb_ * 255)))
        return _out_

    def __assignShades__(self, bins, shades):
        '''Deterministically assign shades to bins so no two adjacent slices (the ring
        wraps, so first and last are adjacent too) share a shade.  The RNG is seeded from
        the ordered bin values, so identical data + settings always yield the same image.'''
        _n_, _k_ = len(bins), len(shades)
        if _n_ == 0: return {}
        _seed_ = zlib.crc32('|'.join(str(b) for b in bins).encode('utf-8'))
        _rng_  = random.Random(_seed_)  # nosec B311 - deterministic seeded shade assignment, not security sensitive
        _chosen_, _out_ = [], {}
        for _i_, _b_ in enumerate(bins):
            _excluded_ = set()
            if _i_ > 0:                       _excluded_.add(_chosen_[_i_ - 1])
            if _i_ == _n_ - 1 and _n_ > 1:    _excluded_.add(_chosen_[0])   # ring wrap-around
            _options_ = [_j_ for _j_ in range(_k_) if _j_ not in _excluded_] or list(range(_k_))
            _idx_ = _rng_.choice(_options_)
            _chosen_.append(_idx_)
            _out_[_b_] = shades[_idx_]
        return _out_

    def __sliceColors__(self, bins):
        '''Return {bin_value: hex} for a list of bins under the active coloring mode.'''
        _default_ = self.p2s.colorTyped('data', 'default')

        # ── none: outline each slice in one of five barely-distinct shades of
        #    the data colour, so adjacent slices read apart (deterministic) ──
        if self._color_mode_ == 'none':
            return self.__assignShades__(bins, self.__dataColorShades__(_default_, 5))

        # ── fixed: one colour for all slices ───────────────────────────────
        if self._color_mode_ == 'fixed':
            return {b: self._fixed_color_ for b in bins}

        # ── hexlist: cycle the supplied colours across the slices in order ─
        if self._color_mode_ == 'hexlist':
            _hl_ = self._hex_list_
            return {b: _hl_[i % len(_hl_)] for i, b in enumerate(bins)}

        # ── spectrum: normalize the colour-stat and map onto the spectrum ──
        if self._color_mode_ == 'spectrum' and self._color_stat_min_ is not None:
            _cspan_ = max(self._color_stat_max_ - self._color_stat_min_, 1e-9)
            if self._color_is_stretched_:
                _vals_  = [(b, self._colorstat_lu_.get(b)) for b in bins]
                _known_ = sorted([v for _, v in _vals_ if v is not None])
                _rank_  = {v: i for i, v in enumerate(_known_)}
                _n_     = max(len(_known_) - 1, 1)
                _norm_  = {b: (0.0 if v is None else _rank_[v] / _n_) for b, v in _vals_}
            else:
                _norm_  = {}
                for b in bins:
                    v = self._colorstat_lu_.get(b)
                    _norm_[b] = 0.0 if v is None else min(max((v - self._color_stat_min_) / _cspan_, 0.0), 1.0)
            _df_ = pl.DataFrame({'__b__': list(_norm_.keys()), '__norm__': list(_norm_.values())})
            _df_ = (_df_
                    .with_columns(self.p2s.colorSpectrumPolarsOperations('__norm__', '__r__', '__g__', '__b3__'))
                    .with_columns(self.p2s.hexColorFromRGBTriplesPolarsOperations('__r__', '__g__', '__b3__').alias('__hx__')))
            _lu_ = dict(zip(_df_['__b__'].to_list(), _df_['__hx__'].to_list()))
            return {b: _lu_.get(b, _default_) for b in bins}

        # ── cset: single-value slices take that value's colour, mixed slices
        #    share one "set" colour ─────────────────────────────────────────
        if self._color_mode_ == 'cset':
            _singles_ = {self._colorset_lu_.get(b) for b in bins} - {None}
            _cmap_    = self.p2s.colors([str(v) for v in _singles_]) if _singles_ else {}
            _set_col_ = self.p2s.color('(set)')          # one shared colour for mixed slices
            _out_     = {}
            for b in bins:
                _v_ = self._colorset_lu_.get(b)
                if _v_ is None:
                    # '(other)' is synthetic (no aggregated value) → give it its own hash
                    _out_[b] = self.p2s.color('(other)') if b == '(other)' else _set_col_
                else:
                    _out_[b] = _cmap_[str(_v_)]
            return _out_

        return {b: _default_ for b in bins}

    def __colorFor__(self, bin_value, lu):
        return lu.get(bin_value, self.p2s.colorTyped('data', 'default'))

    # ── Geometry ────────────────────────────────────────────────────────────

    #
    # __legendPrepare__() - resolve legend kind/metadata (the capture hook) and the
    # strip to reserve, from the color mode __resolveColor__/__computeAggregates__
    # already resolved.  Decision A: a truthy legend with nothing to legend
    # (color=None / fixed hex / hexlist -- no data-driven color semantics) silently
    # reserves nothing.
    #
    def __legendPrepare__(self):
        self.legend_info      = None
        self._legend_region_  = None
        self._legend_reserve_ = (0, 0, 0, 0)
        if self.legend_spec is None or self.df is None or len(self.df_agg) == 0: return
        if self._color_mode_ not in ('cset', 'spectrum'): return
        _spec_  = self.legend_spec
        _title_default_ = 'rows' if self._color_is_crow_ else '|'.join(self._color_fields_)
        _title_ = _spec_['title'] if _spec_['title'] is not None else _title_default_
        if self._color_mode_ == 'cset':
            _vc_ = self.p2s.legendCategoricalValueCounts(self.df, self._color_field_)
            self.legend_info = self.p2s.legendInfoCategorical(_spec_, _vc_, _title_)
        else:
            self.legend_info = self.p2s.legendInfoColorbar(_title_)
            self.p2s.legendInfoColorbarFinalize(self.legend_info, _spec_,
                                                self._color_stat_min_, self._color_stat_max_)
        _reserve_ = self.p2s.legendReserve(_spec_, self.legend_info, self.txt_h, self.wxh)
        _l_, _r_, _t_, _b_ = _reserve_
        if self.wxh[0] - (_l_ + _r_) < 48 or self.wxh[1] - (_t_ + _b_) < 48:
            self.p2s.logger.warning(f'Piep.__legendPrepare__(): not enough space for legend (wxh = {self.wxh}); legend dropped')
            self.legend_info = None
            return
        self._legend_reserve_ = _reserve_
        _pos_ = _spec_['pos']
        if   _pos_ == 'right':  self._legend_region_ = (self.wxh[0] - _r_, 0, _r_, self.wxh[1])
        elif _pos_ == 'left':   self._legend_region_ = (0, 0, _l_, self.wxh[1])
        elif _pos_ == 'top':    self._legend_region_ = (0, 0, self.wxh[0], _t_)
        else:                   self._legend_region_ = (0, self.wxh[1] - _b_, self.wxh[0], _b_)

    def __constructGeometry__(self):
        w, h         = self.wxh
        # Legend strip (if any) comes out of wxh first -- the plot region shrinks,
        # the physical output size does not ("reserve from wxh").
        self.__legendPrepare__()
        _leg_l_, _leg_r_, _leg_t_, _leg_b_ = self._legend_reserve_
        w -= (_leg_l_ + _leg_r_)
        h -= (_leg_t_ + _leg_b_)
        self._avail_x0_, self._avail_x1_ = _leg_l_, _leg_l_ + w
        self._avail_y0_, self._avail_y1_ = _leg_t_, _leg_t_ + h
        x_ins, y_ins = self.insets
        _label_pad_  = (self.txt_h + 4) if self.draw_context else 0
        _plot_x0_    = _leg_l_ + x_ins
        _plot_y0_    = _leg_t_ + y_ins
        _plot_w_     = w - 2 * x_ins
        _plot_h_     = h - 2 * y_ins - _label_pad_
        self._plot_x0_, self._plot_y0_ = _plot_x0_, _plot_y0_
        self._plot_w_,  self._plot_h_  = _plot_w_,  _plot_h_
        self.cx = _plot_x0_ + _plot_w_ / 2.0
        self.cy = _plot_y0_ + _plot_h_ / 2.0
        self.r  = max(4.0, min(_plot_w_, _plot_h_) / 2.0 - 1.0)
        if self.style == self.p2s.DONUTp:
            self.r_inner = max(0.5, self.r * float(self.donut_ratio))
        else:
            self.r_inner = 0.0

        # Build the ordered slice list with angular extents (degrees).
        self._slices_ = []
        if self._base_slices_ is not None:
            # Part-of-whole: geometry (order/angles) comes from the "all rows" reference.
            _total_ = float(sum(s['count_all'] for s in self._base_slices_)) or 1.0
            for s in self._base_slices_:
                _b_ = s['bin']
                self._slices_.append({
                    'bin':        _b_,
                    'count':      self._count_lu_.get(_b_, 0.0),   # this panel's count for the slice
                    'count_all':  float(s['count_all']),
                    'frac':       float(s['count_all']) / _total_,
                    'a0':         float(s['a0']),
                    'a1':         float(s['a1']),
                })
        else:
            _total_ = sum(self._count_lu_.get(b, 0.0) for b in self._sorted_bins_
                          if self._count_lu_.get(b, 0.0) > 0) or 1.0
            _a_ = float(self.start_angle)
            for _b_ in self._sorted_bins_:
                _c_ = self._count_lu_.get(_b_, 0.0)
                if _c_ <= 0: continue
                _sweep_ = _c_ / _total_ * 360.0
                self._slices_.append({
                    'bin':   _b_,
                    'count': _c_,
                    'frac':  _c_ / _total_,
                    'a0':    _a_,
                    'a1':    _a_ + _sweep_,
                })
                _a_ += _sweep_
        self._slice_total_ = _total_

    # ── Wedge drawing (SVG path + GPU polygon) ───────────────────────────────

    def __wedge__(self, dl, r0, r1, a0_deg, a1_deg, fill=None, opacity=1.0, stroke=None, stroke_w=1.0,
                  stroke_in_svg=True):
        '''Draw an annular wedge. fill=None → outline only; stroke=None → filled only.
        stroke_in_svg=False omits the stroke attributes from the SVG element (so it
        inherits them from an enclosing <g>) while still drawing the GPU stroke lines.'''
        _sweep_ = a1_deg - a0_deg
        if _sweep_ <= 1e-6 or r1 <= r0:    return
        if fill is None and stroke is None: return
        def _strokeAttr_():
            if not stroke_in_svg:   return ''
            if stroke is not None:  return f' stroke="{stroke}" stroke-width="{stroke_w:.2f}"'
            return ' stroke="none"'
        cx, cy = self.cx, self.cy
        if _sweep_ >= 359.999:
            if r0 <= 0.0:
                _fa_ = fill if fill is not None else 'none'
                _fo_ = f' fill-opacity="{opacity:.3f}"' if fill is not None else ''
                dl.circle(cx, cy, r1, fill if fill is not None else 'none',
                          stroke=stroke, stroke_w=(stroke_w if stroke is not None else 0.0), opacity=opacity,
                          svg=f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r1:.2f}" fill="{_fa_}"{_fo_}{_strokeAttr_()} />')
            else:
                self.__wedge__(dl, r0, r1, a0_deg,         a0_deg + 180.0, fill=fill, opacity=opacity, stroke=stroke, stroke_w=stroke_w, stroke_in_svg=stroke_in_svg)
                self.__wedge__(dl, r0, r1, a0_deg + 180.0, a0_deg + 360.0, fill=fill, opacity=opacity, stroke=stroke, stroke_w=stroke_w, stroke_in_svg=stroke_in_svg)
            return
        a0, a1 = radians(a0_deg), radians(a1_deg)
        _n_ = max(2, int(_sweep_ / 6.0) + 1)
        _pts_ = []
        for i in range(_n_ + 1):
            a = a0 + (a1 - a0) * i / _n_
            _pts_.append((cx + r1 * cos(a), cy + r1 * sin(a)))
        if r0 <= 0.0:
            _pts_.append((cx, cy))
        else:
            for i in range(_n_ + 1):
                a = a1 + (a0 - a1) * i / _n_
                _pts_.append((cx + r0 * cos(a), cy + r0 * sin(a)))
        _large_ = 1 if _sweep_ > 180.0 else 0
        _x0o_, _y0o_ = cx + r1 * cos(a0), cy + r1 * sin(a0)
        _x1o_, _y1o_ = cx + r1 * cos(a1), cy + r1 * sin(a1)
        if r0 <= 0.0:
            _d_ = (f'M {cx:.2f} {cy:.2f} L {_x0o_:.2f} {_y0o_:.2f} '
                   f'A {r1:.2f} {r1:.2f} 0 {_large_} 1 {_x1o_:.2f} {_y1o_:.2f} Z')
        else:
            _x0i_, _y0i_ = cx + r0 * cos(a0), cy + r0 * sin(a0)
            _x1i_, _y1i_ = cx + r0 * cos(a1), cy + r0 * sin(a1)
            _d_ = (f'M {_x0o_:.2f} {_y0o_:.2f} A {r1:.2f} {r1:.2f} 0 {_large_} 1 {_x1o_:.2f} {_y1o_:.2f} '
                   f'L {_x1i_:.2f} {_y1i_:.2f} A {r0:.2f} {r0:.2f} 0 {_large_} 0 {_x0i_:.2f} {_y0i_:.2f} Z')
        _fa_ = fill if fill is not None else 'none'
        _fo_ = f' fill-opacity="{opacity:.3f}"' if (fill is not None and opacity < 1.0) else ''
        _svg_ = f'<path d="{_d_}" fill="{_fa_}"{_fo_}{_strokeAttr_()} />'
        _svg_emitted_ = False
        if fill is not None:
            dl.polygon(_pts_, fill, opacity=opacity, svg=_svg_)   # GPU tris + the SVG path
            _svg_emitted_ = True
        if stroke is not None:
            # stroke the wedge boundary as GPU line segments; the SVG path (carrying
            # both fill & stroke attrs) is emitted once — here if fill didn't already
            _m_ = len(_pts_)
            for i in range(_m_):
                _p0_, _p1_ = _pts_[i], _pts_[(i + 1) % _m_]
                dl.line(_p0_[0], _p0_[1], _p1_[0], _p1_[1], stroke, width=stroke_w,
                        svg=('' if _svg_emitted_ else _svg_))
                _svg_emitted_ = True

    # ── Waffle allocation (largest-remainder so cells sum to the grid) ───────

    def __waffleCounts__(self, bins_counts, total_cells):
        _tot_ = sum(c for _, c in bins_counts if c > 0) or 1.0
        _raw_ = [(b, c / _tot_ * total_cells) for b, c in bins_counts if c > 0]
        _base_ = [(b, int(v)) for b, v in _raw_]
        _rem_  = sorted(range(len(_raw_)), key=lambda i: _raw_[i][1] - int(_raw_[i][1]), reverse=True)
        _left_ = total_cells - sum(v for _, v in _base_)
        _counts_ = {b: v for b, v in _base_}
        for i in _rem_:
            if _left_ <= 0: break
            _counts_[_raw_[i][0]] += 1
            _left_ -= 1
        return [(b, _counts_[b]) for b, _ in _raw_]

    # ── Rendering ────────────────────────────────────────────────────────────

    def __renderSVG__(self, rand_id):
        w, h          = self.wxh
        _bg_          = self.p2s.colorTyped('background', 'default')
        _label_color_ = self.p2s.colorTyped('label',      'defaultfg')

        _dl_ = self._dl_ = DisplayList(w, h, bg=_bg_)
        _svg_head_ = f'<svg id="piep_{rand_id}" x="0" y="0" width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">'
        _dl_.rect(0, 0, w, h, _bg_, svg=f'<rect x="0" y="0" width="{w}" height="{h}" fill="{_bg_}" />')

        _part_of_whole_ = self._base_slices_ is not None
        _bins_          = [s['bin'] for s in self._slices_]
        _color_lu_      = self.__sliceColors__(_bins_) if _bins_ else {}
        # Faded opacity for the "all rows" backdrop in part-of-whole mode
        _fade_ = 0.22

        if self.style == self.p2s.WAFFLEp:
            self.__renderWaffle__(_dl_, _color_lu_, _part_of_whole_, _fade_, _label_color_)
        else:
            self.__renderRadial__(_dl_, _color_lu_, _part_of_whole_, _fade_, _label_color_)

        # ── LEGEND (drawn into the strip reserved by __legendPrepare__) ───
        if getattr(self, 'legend_info', None) is not None and self._legend_region_ is not None:
            _dl_.extend(self.p2s.legendRenderDL(self.wxh, self._legend_region_, self.legend_spec,
                                                self.legend_info, self.txt_h), copy_svg=True)

        # ── BORDER ─────────────────────────────────────────────────────────
        if self.draw_border:
            _axis_inner_ = self.p2s.colorTyped('axis', 'inner')
            _border_svg_ = f'<rect x="0" y="0" width="{w-1}" height="{h-1}" fill="none" stroke="{_axis_inner_}" stroke-width="1" />'
            _dl_.line(0, 0, w-1, 0, _axis_inner_, width=1.0, svg=_border_svg_)
            _dl_.line(0, h-1, w-1, h-1, _axis_inner_, width=1.0)
            _dl_.line(0, 0, 0, h-1, _axis_inner_, width=1.0)
            _dl_.line(w-1, 0, w-1, h-1, _axis_inner_, width=1.0)

        self.svg = _svg_head_ + _dl_.svg() + '</svg>'

    def __renderRadial__(self, _dl_, _color_lu_, _part_of_whole_, _fade_, _label_color_):
        r0, r1 = self.r_inner, self.r
        # Every slice gets a thin background-color delineation stroke so adjacent
        # slices read apart regardless of color.  The stroke lives on a wrapping
        # <g> in the SVG (to conserve space); GPU draws it per slice as lines.
        _bg_ = self.p2s.colorTyped('background', 'default')
        _dl_.raw(f'<g stroke="{_bg_}" stroke-width="0.1">')
        for s in self._slices_:
            _c_ = self.__colorFor__(s['bin'], _color_lu_)
            if _part_of_whole_:
                # faded backdrop wedge (the full "all rows" slice)
                self.__wedge__(_dl_, r0, r1, s['a0'], s['a1'], fill=_c_, opacity=_fade_,
                               stroke=_bg_, stroke_w=0.1, stroke_in_svg=False)
                # this panel's share of the slice's total
                _f_ = 0.0 if s['count_all'] <= 0 else min(max(s['count'] / s['count_all'], 0.0), 1.0)
                if _f_ > 0:
                    if r0 > 0.0:   _fill_r_ = r0 + (r1 - r0) * _f_          # donut: fill ring thickness
                    else:          _fill_r_ = r1 * sqrt(_f_)                 # pie: area-proportional radius
                    self.__wedge__(_dl_, r0, _fill_r_, s['a0'], s['a1'], fill=_c_,
                                   stroke=_bg_, stroke_w=0.1, stroke_in_svg=False)
            else:
                self.__wedge__(_dl_, r0, r1, s['a0'], s['a1'], fill=_c_,
                               stroke=_bg_, stroke_w=0.1, stroke_in_svg=False)
        _dl_.raw('</g>')

        if self.draw_labels:
            self.__renderSliceLabels__(_dl_, _label_color_)
        if self.draw_context:
            self.__renderTitle__(_dl_, _label_color_)

    def __renderTitle__(self, _dl_, _label_color_):
        # Bin field name centered below the chart
        _title_ = self.p2s.cropText('|'.join(self._bin_cols_), self.txt_h, self._plot_w_)
        _dl_.text(self.p2s, _title_, self.cx, self._plot_y0_ + self._plot_h_ + self.txt_h,
                  txt_h=self.txt_h, anchor='middle', color=_label_color_)

    def __fitLabel__(self, txt, avail_w):
        '''Crop txt to avail_w; return None if not even one real character fits.'''
        _s_ = self.p2s.formatMultiFieldValue(txt)
        if self.p2s.textLength(_s_, self.txt_h) <= avail_w:
            return _s_
        _c_ = self.p2s.cropText(_s_, self.txt_h, avail_w)
        return None if _c_ in ('', '...') else _c_

    #
    # __renderSliceLabels__() - category labels for pie/donut (draw_labels=True).
    #   • a slice wide enough to hold its cropped name gets the label placed INSIDE.
    #   • remaining slices are labeled OUTSIDE with a leader line when there is margin
    #     around the pie (large insets / oblong wxh); labels are prioritized
    #     largest-slice-first, cropped to the room on their side, with whitespace.
    #
    def __renderSliceLabels__(self, _dl_, _label_color_):
        # outside labels must stay within the plot area (excludes any legend strip)
        w, h    = self._avail_x1_, self._avail_y1_
        _lr_    = (self.r_inner + self.r) / 2.0 if self.style == self.p2s.DONUTp else self.r * 0.58
        _leader_color_ = self.p2s.colorTyped('axis', 'inner')

        _inside_, _outside_ = [], []
        for s in self._slices_:
            _sweep_ = s['a1'] - s['a0']
            if _sweep_ <= 0: continue
            # chord of the wedge at the label radius = the room across the slice
            _chord_ = 2.0 * _lr_ * sin(radians(min(_sweep_, 160.0)) / 2.0)
            if _chord_ >= 1.3 * self.txt_h:
                _lbl_ = self.__fitLabel__(s['bin'], _chord_ * 0.95)
                if _lbl_ is not None:
                    _inside_.append((s, _lbl_))
                    continue
            _outside_.append(s)

        # ── INSIDE labels ─────────────────────────────────────────────────
        for s, _lbl_ in _inside_:
            _am_ = radians((s['a0'] + s['a1']) / 2.0)
            _tx_, _ty_ = self.cx + _lr_ * cos(_am_), self.cy + _lr_ * sin(_am_)
            _dl_.text(self.p2s, _lbl_, _tx_, _ty_ + self.txt_h * 0.35,
                      txt_h=self.txt_h, anchor='middle', color=_label_color_)

        if not _outside_:
            return

        # ── OUTSIDE labels (leader lines), only where there is room ────────
        # Keep a full txt_h of whitespace between the text and every canvas edge.
        _gap_       = float(self.txt_h)
        _lead_      = 8.0
        _line_h_    = self.txt_h + 3.0
        _x_txt_r_   = self.cx + self.r + _lead_ + 2.0     # right-column text start (anchor start)
        _x_txt_l_   = self.cx - self.r - _lead_ - 2.0     # left-column  text end   (anchor end)
        _avail_r_   = (w - _gap_) - _x_txt_r_
        _avail_l_   = _x_txt_l_ - (self._avail_x0_ + _gap_)
        _y_top_     = self._avail_y0_ + _gap_ + self.txt_h
        _y_bot_     = h - _gap_
        _capacity_  = max(0, int((_y_bot_ - _y_top_) / _line_h_))
        _min_room_  = self.txt_h * 1.2   # need at least ~1 char + ellipsis

        # Prioritize by slice size (largest first) since space is limited.
        _ranked_ = sorted(_outside_, key=lambda s: s['count'], reverse=True)

        _right_, _left_ = [], []
        for s in _ranked_:
            _am_ = radians((s['a0'] + s['a1']) / 2.0)
            _to_right_ = cos(_am_) >= 0.0
            if _to_right_ and _avail_r_ >= _min_room_ and len(_right_) < _capacity_:
                _right_.append((s, _am_))
            elif (not _to_right_) and _avail_l_ >= _min_room_ and len(_left_) < _capacity_:
                _left_.append((s, _am_))

        def _place_side_(items, avail_w, x_text, anchor):
            if not items or avail_w < _min_room_: return
            # natural y where each slice's mid-ray meets the circle
            _rows_ = [[s, _am_, self.cy + self.r * sin(_am_)] for s, _am_ in items]
            _rows_.sort(key=lambda r: r[2])
            # top-down declutter, then clamp the tail upward
            _prev_ = _y_top_ - _line_h_
            for _row_ in _rows_:
                _row_[2] = max(_row_[2], _prev_ + _line_h_)
                _prev_   = _row_[2]
            _over_ = _rows_[-1][2] - _y_bot_
            if _over_ > 0:
                _floor_ = _y_bot_
                for _row_ in reversed(_rows_):
                    _row_[2] = min(_row_[2], _floor_)
                    _floor_  = _row_[2] - _line_h_
            for s, _am_, _ly_ in _rows_:
                _lbl_ = self.__fitLabel__(s['bin'], avail_w)
                if _lbl_ is None: continue
                _cxp_ = self.cx + self.r * cos(_am_)
                _cyp_ = self.cy + self.r * sin(_am_)
                _elx_ = x_text + (-2.0 if anchor == 'start' else 2.0)
                _dl_.line(_cxp_, _cyp_, _elx_, _ly_ - self.txt_h * 0.32, _leader_color_, width=0.5,
                          svg=f'<line x1="{_cxp_:.2f}" y1="{_cyp_:.2f}" x2="{_elx_:.2f}" '
                              f'y2="{_ly_ - self.txt_h * 0.32:.2f}" stroke="{_leader_color_}" stroke-width="0.5" />')
                _dl_.text(self.p2s, _lbl_, x_text, _ly_,
                          txt_h=self.txt_h, anchor=anchor, color=_label_color_)

        _place_side_(_right_, _avail_r_, _x_txt_r_, 'start')
        _place_side_(_left_,  _avail_l_, _x_txt_l_, 'end')

    def __renderWaffle__(self, _dl_, _color_lu_, _part_of_whole_, _fade_, _label_color_):
        _n_    = max(1, int(self.waffle_n))
        _cells_ = _n_ * _n_
        _side_ = min(self._plot_w_, self._plot_h_)
        _cell_ = _side_ / _n_
        _ox_   = self.cx - _side_ / 2.0
        _oy_   = self.cy - _side_ / 2.0
        _gap_  = min(1.0, _cell_ * 0.12)

        _bins_counts_ = [(s['bin'], s['count'] if not _part_of_whole_ else s['count_all'])
                         for s in self._slices_]
        _alloc_ = self.__waffleCounts__(_bins_counts_, _cells_)
        # part-of-whole: how many of each slice's cells are "filled" by this panel
        _fill_alloc_ = {}
        if _part_of_whole_:
            for s in self._slices_:
                _f_ = 0.0 if s['count_all'] <= 0 else min(max(s['count'] / s['count_all'], 0.0), 1.0)
                _fill_alloc_[s['bin']] = _f_

        # cell order: bottom-up, left-to-right (row 0 = bottom)
        def _cell_xy_(idx):
            _row_ = idx // _n_
            _col_ = idx % _n_
            _x_ = _ox_ + _col_ * _cell_
            _y_ = _oy_ + (_n_ - 1 - _row_) * _cell_
            return _x_, _y_

        _idx_ = 0
        for _b_, _n_cells_ in _alloc_:
            _c_       = self.__colorFor__(_b_, _color_lu_)
            _n_fill_  = int(round(_fill_alloc_.get(_b_, 1.0) * _n_cells_)) if _part_of_whole_ else _n_cells_
            for _k_ in range(_n_cells_):
                _x_, _y_ = _cell_xy_(_idx_)
                _idx_ += 1
                _rx_, _ry_ = _x_ + _gap_, _y_ + _gap_
                _rw_ = max(1.0, _cell_ - 2 * _gap_)
                if _part_of_whole_:
                    _op_ = 1.0 if _k_ < _n_fill_ else _fade_
                else:
                    _op_ = 1.0
                _dl_.rect(_rx_, _ry_, _rw_, _rw_, _c_, opacity=_op_,
                          svg=f'<rect x="{_rx_:.2f}" y="{_ry_:.2f}" width="{_rw_:.2f}" height="{_rw_:.2f}" '
                              f'fill="{_c_}" fill-opacity="{_op_:.3f}" stroke="none" />')

        if self.draw_context:
            _title_ = self.p2s.cropText('|'.join(self._bin_cols_), self.txt_h, self._plot_w_)
            _dl_.text(self.p2s, _title_, self.cx, self._plot_y0_ + self._plot_h_ + self.txt_h,
                      txt_h=self.txt_h, anchor='middle', color=_label_color_)

    # ── smallp integration ───────────────────────────────────────────────────

    def render_with(self, df, **overrides):
        return Piep(df=df, template=self, **overrides)

    #
    # renderSmallMultiples() - smallp integration
    #   SM_SLICE_ORDERp / SM_COLOR : identical slice order (and therefore stable colors)
    #                                in every panel, taken from the "all rows" reference
    #   SM_PARTOFWHOLEp            : fade the "all rows" chart behind each panel and fill in
    #                                each slice's share (implies a shared order)
    #   SM_COUNT                  : share the spectrum/count normalization range
    #
    def renderSmallMultiples(self, df_all, df_lu, all_key):
        _kwargs_       = {'sm_shared': self.sm_shared}
        _want_order_   = (self.p2s.SM_SLICE_ORDERp in self.sm_shared or
                          self.p2s.SM_COLOR        in self.sm_shared)
        _want_pow_     = self.p2s.SM_PARTOFWHOLEp in self.sm_shared
        _want_count_   = self.p2s.SM_COUNT        in self.sm_shared
        if _want_order_ or _want_pow_ or _want_count_:
            _ref_ = Piep(df=df_all, template=self)
            if _want_count_ and _ref_._color_stat_min_ is not None:
                _kwargs_['color_stat_range_shared'] = (_ref_._color_stat_min_, _ref_._color_stat_max_)
            if _want_count_:
                _kwargs_['count_range_shared'] = (_ref_._count_min_, _ref_._count_max_)
            if _want_pow_:
                _kwargs_['_base_slices_'] = [
                    {'bin': s['bin'], 'count_all': s['count'], 'a0': s['a0'], 'a1': s['a1']}
                    for s in _ref_._slices_
                ]
            elif _want_order_:
                _kwargs_['_shared_order_'] = list(_ref_._sorted_bins_)
        return {k: Piep(df=v, template=self, **_kwargs_) for k, v in df_lu.items()}

    # ── Interactivity (panelize / brushing) ──────────────────────────────────

    def __expandBins__(self, bins):
        '''Expand the synthetic "(other)" slice into the real bin values it folded in.'''
        _out_ = []
        for _b_ in bins:
            if _b_ == '(other)': _out_.extend(self._other_members_)
            else:                _out_.append(_b_)
        return list(dict.fromkeys(_out_))

    def __sliceForBin__(self, bin_value, how='inner'):
        return self.__binsForBins__([bin_value], remove=(how == 'anti'))

    def __dropInternal__(self, df):
        _to_drop_ = [c for c in ['__p2s_index__'] if c in df.columns]
        if self._bin_col_ == '__bin__' and '__bin__' in df.columns:
            _to_drop_.append('__bin__')
        return df.drop(_to_drop_)

    def __binsForBins__(self, bins, remove=False):
        _bins_ = self.__expandBins__(bins)
        if not _bins_:
            return self.__dropInternal__(self.df if remove else self.df.clear())
        _bin_dtype_   = self.df_agg[self._bin_col_].dtype
        _selected_df_ = pl.DataFrame({self._bin_col_: _bins_}, schema={self._bin_col_: _bin_dtype_})
        _how_         = 'anti' if remove else 'inner'
        return self.__dropInternal__(self.df.join(_selected_df_, on=self._bin_col_, how=_how_))

    def __binAtAngleDist__(self, angle_deg, dist):
        '''Return the bin whose wedge/annulus contains (angle_deg, dist), or None.'''
        if self.style == self.p2s.WAFFLEp: return None
        if dist < self.r_inner or dist > self.r: return None
        _a_ = angle_deg
        for s in self._slices_:
            _a0_, _a1_ = s['a0'], s['a1']
            # normalize the test angle into [a0, a0+360)
            _t_ = _a_
            while _t_ < _a0_:        _t_ += 360.0
            while _t_ >= _a0_ + 360: _t_ -= 360.0
            if _a0_ <= _t_ <= _a1_:
                return s['bin']
        return None

    def __binAtWaffleXY__(self, x, y):
        _n_    = max(1, int(self.waffle_n))
        _side_ = min(self._plot_w_, self._plot_h_)
        _cell_ = _side_ / _n_
        _ox_   = self.cx - _side_ / 2.0
        _oy_   = self.cy - _side_ / 2.0
        _col_  = int((x - _ox_) / _cell_)
        _rowv_ = int((y - _oy_) / _cell_)
        if _col_ < 0 or _col_ >= _n_ or _rowv_ < 0 or _rowv_ >= _n_: return None
        _row_  = _n_ - 1 - _rowv_          # invert: row 0 is the bottom
        _idx_  = _row_ * _n_ + _col_
        _bins_counts_ = [(s['bin'], s.get('count_all', s['count'])) for s in self._slices_]
        _alloc_ = self.__waffleCounts__(_bins_counts_, _n_ * _n_)
        _acc_ = 0
        for _b_, _cnt_ in _alloc_:
            if _idx_ < _acc_ + _cnt_: return _b_
            _acc_ += _cnt_
        return None

    def recordsAt(self, xy, shape=None, threshold=2.0):
        '''Records whose slice contains the pixel xy (SELECT_CIRCLEp).'''
        if shape is None: shape = self.p2s.SELECT_CIRCLEp
        if shape != self.p2s.SELECT_CIRCLEp:
            raise ValueError(f'Piep.recordsAt(): only SELECT_CIRCLEp is supported, got {shape}')
        _x_, _y_ = xy
        if self.style == self.p2s.WAFFLEp:
            _bin_ = self.__binAtWaffleXY__(_x_, _y_)
        else:
            _dx_, _dy_ = _x_ - self.cx, _y_ - self.cy
            _dist_     = sqrt(_dx_ * _dx_ + _dy_ * _dy_)
            _ang_      = atan2(_dy_, _dx_) * 180.0 / pi
            _bin_      = self.__binAtAngleDist__(_ang_, _dist_)
        if _bin_ is None:
            return self.__dropInternal__(self.df.clear())
        return self.__sliceForBin__(_bin_)

    def filterByRectangle(self, bounding_box, remove_records=False):
        _x0_, _y0_, _x1_, _y1_ = bounding_box
        if _x0_ > _x1_: _x0_, _x1_ = _x1_, _x0_
        if _y0_ > _y1_: _y0_, _y1_ = _y1_, _y0_
        # A plain click arrives as a zero-area rectangle (x0==x1, y0==y1): treat any
        # rectangle as covering at least the pixel under the cursor.
        _cxr_, _cyr_ = (_x0_ + _x1_) / 2.0, (_y0_ + _y1_) / 2.0

        def _in_rect_(px, py):
            return _x0_ <= px <= _x1_ and _y0_ <= py <= _y1_

        _selected_ = []
        if self.style == self.p2s.WAFFLEp:
            # click / small box: the cell under the rectangle center
            _center_bin_ = self.__binAtWaffleXY__(_cxr_, _cyr_)
            if _center_bin_ is not None:
                _selected_.append(_center_bin_)
            # drag box: any slice with a cell centroid inside the rectangle
            _n_    = max(1, int(self.waffle_n))
            _side_ = min(self._plot_w_, self._plot_h_)
            _cell_ = _side_ / _n_
            _ox_   = self.cx - _side_ / 2.0
            _oy_   = self.cy - _side_ / 2.0
            _bins_counts_ = [(s['bin'], s.get('count_all', s['count'])) for s in self._slices_]
            _alloc_ = self.__waffleCounts__(_bins_counts_, _n_ * _n_)
            _idx_ = 0
            for _b_, _cnt_ in _alloc_:
                _hit_ = False
                for _k_ in range(_cnt_):
                    _row_ = _idx_ // _n_
                    _col_ = _idx_ % _n_
                    _cxp_ = _ox_ + (_col_ + 0.5) * _cell_
                    _cyp_ = _oy_ + (_n_ - 1 - _row_ + 0.5) * _cell_
                    _idx_ += 1
                    if _in_rect_(_cxp_, _cyp_): _hit_ = True
                if _hit_: _selected_.append(_b_)
        else:
            # click / small box: the wedge under the rectangle center
            _dxr_, _dyr_ = _cxr_ - self.cx, _cyr_ - self.cy
            _center_bin_ = self.__binAtAngleDist__(atan2(_dyr_, _dxr_) * 180.0 / pi,
                                                   sqrt(_dxr_ * _dxr_ + _dyr_ * _dyr_))
            if _center_bin_ is not None:
                _selected_.append(_center_bin_)
            # drag box: any wedge whose area is touched by the rectangle.  Sample the
            # wedge across angle x radius and test each sample against the rect.
            _r_lo_ = max(self.r_inner, self.r * 0.08)
            _radii_ = [_r_lo_, (_r_lo_ + self.r) / 2.0, self.r * 0.97]
            for s in self._slices_:
                if s['bin'] in _selected_: continue
                _sweep_ = s['a1'] - s['a0']
                _na_    = max(2, int(_sweep_ / 8.0) + 1)
                _hit_   = False
                for _i_ in range(_na_ + 1):
                    _a_ = radians(s['a0'] + _sweep_ * _i_ / _na_)
                    _ca_, _sa_ = cos(_a_), sin(_a_)
                    for _rr_ in _radii_:
                        if _in_rect_(self.cx + _rr_ * _ca_, self.cy + _rr_ * _sa_):
                            _hit_ = True
                            break
                    if _hit_: break
                if _hit_: _selected_.append(s['bin'])

        _selected_ = list(dict.fromkeys(_selected_))
        if not _selected_:
            return self.__dropInternal__(self.df if remove_records else self.df.clear())
        return self.__binsForBins__(_selected_, remove=remove_records)

    def filterByOval(self, oval, remove_records=False):
        _cx_, _cy_, _rx_, _ry_ = oval
        # A plain click arrives as a zero-radius oval: keep it covering the pixel under the cursor.
        _rx_, _ry_ = max(float(_rx_), 0.5), max(float(_ry_), 0.5)
        # The oval center is the click point (mouse-press seeds the center).
        _cxr_, _cyr_ = _cx_, _cy_

        def _in_ellipse_(px, py):
            return ((px - _cx_) / _rx_) ** 2 + ((py - _cy_) / _ry_) ** 2 <= 1.0

        _selected_ = []
        if self.style == self.p2s.WAFFLEp:
            # click: the cell under the oval center
            _center_bin_ = self.__binAtWaffleXY__(_cxr_, _cyr_)
            if _center_bin_ is not None:
                _selected_.append(_center_bin_)
            # drag: any slice with a cell centroid inside the oval
            _n_    = max(1, int(self.waffle_n))
            _side_ = min(self._plot_w_, self._plot_h_)
            _cell_ = _side_ / _n_
            _ox_   = self.cx - _side_ / 2.0
            _oy_   = self.cy - _side_ / 2.0
            _bins_counts_ = [(s['bin'], s.get('count_all', s['count'])) for s in self._slices_]
            _alloc_ = self.__waffleCounts__(_bins_counts_, _n_ * _n_)
            _idx_ = 0
            for _b_, _cnt_ in _alloc_:
                _hit_ = False
                for _k_ in range(_cnt_):
                    _row_ = _idx_ // _n_
                    _col_ = _idx_ % _n_
                    _cxp_ = _ox_ + (_col_ + 0.5) * _cell_
                    _cyp_ = _oy_ + (_n_ - 1 - _row_ + 0.5) * _cell_
                    _idx_ += 1
                    if _in_ellipse_(_cxp_, _cyp_): _hit_ = True
                if _hit_: _selected_.append(_b_)
        else:
            # click: the wedge under the oval center
            _dxr_, _dyr_ = _cxr_ - self.cx, _cyr_ - self.cy
            _center_bin_ = self.__binAtAngleDist__(atan2(_dyr_, _dxr_) * 180.0 / pi,
                                                   sqrt(_dxr_ * _dxr_ + _dyr_ * _dyr_))
            if _center_bin_ is not None:
                _selected_.append(_center_bin_)
            # drag: any wedge whose area is touched by the oval.  Sample the wedge across
            # angle x radius and test each sample against the ellipse.
            _r_lo_ = max(self.r_inner, self.r * 0.08)
            _radii_ = [_r_lo_, (_r_lo_ + self.r) / 2.0, self.r * 0.97]
            for s in self._slices_:
                if s['bin'] in _selected_: continue
                _sweep_ = s['a1'] - s['a0']
                _na_    = max(2, int(_sweep_ / 8.0) + 1)
                _hit_   = False
                for _i_ in range(_na_ + 1):
                    _a_ = radians(s['a0'] + _sweep_ * _i_ / _na_)
                    _ca_, _sa_ = cos(_a_), sin(_a_)
                    for _rr_ in _radii_:
                        if _in_ellipse_(self.cx + _rr_ * _ca_, self.cy + _rr_ * _sa_):
                            _hit_ = True
                            break
                    if _hit_: break
                if _hit_: _selected_.append(s['bin'])

        _selected_ = list(dict.fromkeys(_selected_))
        if not _selected_:
            return self.__dropInternal__(self.df if remove_records else self.df.clear())
        return self.__binsForBins__(_selected_, remove=remove_records)

    def filterBySubstring(self, substring, remove_bins=False):
        _sub_ = substring.lower()
        # Search every real bin, including those folded into "(other)", so a folded
        # category is still findable by name.
        _all_real_ = [b for b in self._sorted_bins_ if b != '(other)'] + list(self._other_members_)
        # Match against the display form ('|'-joined) so a user's 'A|x' still matches a
        # multi-field bin whose internal key uses the non-printable MULTI_FIELD_SEP.
        _matching_ = [b for b in _all_real_ if _sub_ in self.p2s.formatMultiFieldValue(b).lower()]
        # Typing the aggregate label selects the whole "(other)" slice.
        if '(other)' in self._sorted_bins_ and _sub_ and _sub_ in '(other)':
            _matching_ += list(self._other_members_)
        _matching_ = list(dict.fromkeys(_matching_))
        if not _matching_:
            return self.__dropInternal__(self.df if remove_bins else self.df.clear())
        return self.__binsForBins__(_matching_, remove=remove_bins)
