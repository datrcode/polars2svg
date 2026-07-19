import polars as pl
import time
import random

import polars2svg
from polars2svg.p2s_displaylist import DisplayList
from polars2svg.export import ExportMixin
from polars2svg.p2s_bin_component_mixin import P2SBinComponentMixin

#
# Histogram
#
class Histop(P2SBinComponentMixin, ExportMixin):

    _VALID_KWARGS = frozenset({
        'template', 'df',
        'bin_by', 'count', 'count_range', 'count_range_shared',
        'color', 'order', 'descending', 'style',
        'wxh', 'insets', 'draw_context', 'draw_border', 'draw_labels', 'txt_h', 'bar_h', 'v_gap',
        'draw_distribution', 'distribution', 'distribution_bin_w',
        'sm_shared', 'use_lazy_execution', 'min_bar_w',
        'swarm_max_pts', 'remainder_threshold', 'color_stat_range_shared',
        'legend',
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
            self.gatherMetrics(self.__computeDistribution__)
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

    def __parseInput__(self, *args, **kwargs):
        _unknown_ = set(kwargs) - self._VALID_KWARGS
        if _unknown_:
            raise TypeError(f'Histop: unexpected keyword argument(s): {sorted(_unknown_)}')

        # Single source of truth for every parameter (name -> from-scratch default);
        # drives both the from-scratch assignment and the keyword-override copy below.
        _defaults_ = {
            'bin_by':                  None,
            'count':                   self.p2s.ROW_COUNTp,
            'count_range':             None,
            'count_range_shared':      None,
            'color':                   None,
            'order':                   self.p2s.ROW_COUNTp,
            'descending':              True,
            'style':                   self.p2s.BARCHARTp,
            'wxh':                     (128, 256),
            'insets':                  (2, 2),
            'draw_context':            True,
            'draw_border':             True,
            'draw_labels':             True,
            'txt_h':                   12,
            'bar_h':                   None,
            'v_gap':                   0,
            'draw_distribution':       False,
            'distribution':            True,
            'distribution_bin_w':      10,
            'sm_shared':               set(),
            'use_lazy_execution':      True,
            'min_bar_w':               1.0,
            'swarm_max_pts':           50,
            'remainder_threshold':     3.0,
            'color_stat_range_shared': None,
            'legend':                  False,
        }
        self.p2s.assertParamSpecMatches('Histop', self._VALID_KWARGS, _defaults_)

        self.df, self.df_orig = None, None

        # Template support
        self.template = None
        for i in range(len(args)):
            if isinstance(args[i], Histop): self.template = args[i]
        if 'template' in kwargs: self.template = kwargs['template']
        if self.template is not None:
            _template_copy_ = self.template
            self.p2s._clone_template_state(self, _template_copy_)
            self.template = _template_copy_
        else:
            self.p2s.assignScratchDefaults(self, _defaults_)
            # from-scratch builds only — a template clone is an exact snapshot and
            # must not re-apply session defaults (see Polars2SVG._apply_defaults)
            kwargs = self.p2s._apply_defaults('histop', kwargs)

        # Extract DataFrame
        _new_df_ = None
        for _arg_ in args:
            if isinstance(_arg_, pl.DataFrame):
                if _new_df_ is None: _new_df_ = _arg_
                else:                raise ValueError('Histop.__parseInput__(): df already set')
        if 'df' in kwargs:
            if _new_df_ is None: _new_df_ = kwargs['df']
            else:                raise ValueError('Histop.__parseInput__(): df already set')
        if _new_df_ is not None:
            self.df = self.df_orig = _new_df_
        if self.df is not None and '__p2s_index__' not in self.df.columns:
            self.df = self.df.with_row_index('__p2s_index__')

        # Extract bin_by from positional args (string or tuple)
        for arg in args:
            if   isinstance(arg, pl.DataFrame): pass
            elif isinstance(arg, Histop):       pass
            elif isinstance(arg, str):
                if self.bin_by is None: self.bin_by = arg
                else:                   raise ValueError('Histop.__parseInput__(): bin_by already set')
            elif isinstance(arg, tuple):
                if self.bin_by is None: self.bin_by = arg
                else:                   raise ValueError('Histop.__parseInput__(): bin_by already set')
            else:
                raise ValueError(f'Histop.__parseInput__(): Unknown argument type: {type(arg)}')

        self.p2s.assignKwargOverrides(self, _defaults_, kwargs)

        # "No data" placeholder for early error visibility -- only ever seen when
        # no df is supplied (a successful render overwrites self.svg); makes a
        # dropped-df plumbing mistake visible instead of a silently blank canvas.
        self.wxh = self.p2s.normalizeWxh(self.wxh, 'Histop')
        w, h = self.wxh
        self.svg = self.p2s.placeholderSVG(w, h)

    def __validateInput__(self):
        # Normalize legend= eagerly so a bad spec fails fast (raises InvalidSpecError).
        self.legend_spec = self.p2s.legendResolveSpec(self.legend)
        if self.df is None: return
        self.p2s.checkReservedColumns(self.df, 'Histop')

        # Resolve bin fields
        if self.bin_by is None:
            raise ValueError('Histop.__validateInput__(): bin_by must be specified')
        if isinstance(self.bin_by, str):
            self._bin_cols_ = [self.bin_by]
            self._bin_col_  = self.bin_by
        elif isinstance(self.bin_by, tuple):
            self._bin_cols_ = list(self.bin_by)
            self._bin_col_  = '__bin__'
        else:
            raise ValueError(f'Histop.__validateInput__(): bin_by must be str or tuple, got {type(self.bin_by)}')

        for _f_ in self._bin_cols_:
            if _f_ not in self.df.columns:
                raise ValueError(f'Histop.__validateInput__(): bin_by field "{_f_}" not found')

        # Validate count
        if self.count != self.p2s.ROW_COUNTp:
            if isinstance(self.count, str) and not self.p2s.columnInDataFrame(self.count, self.df):
                raise ValueError(f'Histop.__validateInput__(): count field "{self.count}" not found')
            elif isinstance(self.count, tuple):
                for _f_ in self.count:
                    if isinstance(_f_, str) and not self.p2s.columnInDataFrame(_f_, self.df):
                        raise ValueError(f'Histop.__validateInput__(): count field "{_f_}" not found')

        # Validate color
        if self.color is not None:
            if isinstance(self.color, str) and not self.p2s.columnInDataFrame(self.color, self.df):
                raise ValueError(f'Histop.__validateInput__(): color field "{self.color}" not found')
            elif isinstance(self.color, tuple):
                for _f_ in self.color:
                    if isinstance(_f_, str) and not self.p2s.columnInDataFrame(_f_, self.df):
                        raise ValueError(f'Histop.__validateInput__(): color field "{_f_}" not found')

        # Validate style
        _valid_styles_ = {self.p2s.BARCHARTp, self.p2s.BOXPLOTp, self.p2s.BOXPLOT_W_SWARMp, self.p2s.STACKEDBARp}
        if self.style not in _valid_styles_:
            raise ValueError(f'Histop.__validateInput__(): style must be one of {_valid_styles_}')

        # Warn on SM_* values that Histop does not support
        _unsupported_sm_ = {self.p2s.SM_X, self.p2s.SM_Y} & self.sm_shared
        if _unsupported_sm_:
            self.p2s.logger.warning(
                f'Histop: sm_shared contains {_unsupported_sm_} which Histop does not support '
                f'(supported: SM_COUNT, SM_COLOR); these values will be ignored'
            )

        # wxh was normalized to a canonical (int, int) tuple in __parseInput__ via
        # self.p2s.normalizeWxh(); no re-validation needed here.
        w, h = self.wxh
        if w < 64 or h < 64:
            self.draw_context = False
        x_ins, y_ins = self.insets
        if w - 2 * x_ins < 48: self.insets = (0,            self.insets[1])
        if h - 2 * y_ins < 48: self.insets = (self.insets[0], 0           )

        # Resolve bar_h default (after draw_context may have been overridden by size)
        if self.bar_h is None:
            self.bar_h = self.txt_h + 4 if self.draw_context else 5

    def __addColumnsToDataFrame__(self):
        _ops_ = []

        # Multi-field bin: concatenate to '__bin__' (non-printable separator so distinct
        # field tuples can't collide into one bar; shown as '|' via formatMultiFieldValue)
        if isinstance(self.bin_by, tuple):
            _ops_.append(pl.concat_str(self._bin_cols_, separator=self.p2s.MULTI_FIELD_SEP).alias('__bin__'))

        # Count field t-fields
        if isinstance(self.count, str) and self.p2s.isTField(self.count, df=self.df_orig):
            self.p2s.warnIfTFieldAliasCollides(self.count, self.df_orig, 'Histop')
            _ops_.append(self.p2s.polarsOperationForTField(self.count).alias(self.count))
        elif isinstance(self.count, tuple):
            for _f_ in self.count:
                if isinstance(_f_, str) and self.p2s.isTField(_f_, df=self.df_orig):
                    self.p2s.warnIfTFieldAliasCollides(_f_, self.df_orig, 'Histop')
                    _ops_.append(self.p2s.polarsOperationForTField(_f_).alias(_f_))

        # Color field t-fields
        if isinstance(self.color, str) and self.p2s.isTField(self.color, df=self.df_orig):
            self.p2s.warnIfTFieldAliasCollides(self.color, self.df_orig, 'Histop')
            _ops_.append(self.p2s.polarsOperationForTField(self.color).alias(self.color))
        elif isinstance(self.color, tuple):
            for _f_ in self.color:
                if isinstance(_f_, str) and self.p2s.isTField(_f_, df=self.df_orig):
                    self.p2s.warnIfTFieldAliasCollides(_f_, self.df_orig, 'Histop')
                    _ops_.append(self.p2s.polarsOperationForTField(_f_).alias(_f_))

        if len(_ops_) > 0:
            if self.use_lazy_execution: self.df = self.df.lazy().with_columns(_ops_).collect()
            else:                       self.df = self.df.with_columns(_ops_)

    # ── Count aggregate expression ──────────────────────────────────────────

    def __countAggExpr__(self):
        if self.count == self.p2s.ROW_COUNTp:
            return pl.len().alias('__count__')
        elif isinstance(self.count, str):
            _is_num_ = self.p2s.numericColumn(self.df, self.count)
            self.p2s.logDtypeKeyedCount('Histop', self.count, _is_num_)
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

    def __countFields__(self):
        if self.count == self.p2s.ROW_COUNTp: return set()
        if isinstance(self.count, str):        return {self.count}
        if isinstance(self.count, tuple):      return {_f_ for _f_ in self.count if isinstance(_f_, str)}
        return set()

    def __findNumericCountField__(self):
        if isinstance(self.count, str) and self.p2s.numericColumn(self.df, self.count):
            return self.count
        elif isinstance(self.count, tuple):
            for _f_ in self.count:
                if isinstance(_f_, str) and self.p2s.numericColumn(self.df, _f_):
                    return _f_
        return None

    def __orderAggExpr__(self):
        if self.order == self.p2s.ROW_COUNTp:
            return pl.len().alias('__order_metric__')
        elif isinstance(self.order, str):
            return pl.col(self.order).sum().alias('__order_metric__')
        elif isinstance(self.order, tuple):
            _field_ = self.order[0]
            if len(self.order) == 1:
                return pl.col(_field_).sum().alias('__order_metric__')
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

    # ── Aggregation ─────────────────────────────────────────────────────────

    def __computeAggregates__(self):
        # Determine color field and whether it's categorical
        self._color_field_            = None
        self._color_is_categorical_   = False
        self._color_is_crow_          = False  # CROW_MAGNITUDEp / CROW_STRETCHEDp: color by row count
        self._color_is_stretched_     = False  # rank-based normalization instead of linear min-max
        self._color_is_cset_spectrum_ = False  # stacked bars with per-segment spectrum colors

        _cstretched_types_ = {self.p2s.CSTRETCHED_SUMp, self.p2s.CSTRETCHED_MINp,
                              self.p2s.CSTRETCHED_MEDIANp, self.p2s.CSTRETCHED_MEANp, self.p2s.CSTRETCHED_MAXp}

        if self.color is not None:
            if self.color in {self.p2s.CROW_MAGNITUDEp, self.p2s.CROW_STRETCHEDp}:
                self._color_is_crow_      = True
                self._color_is_stretched_ = (self.color == self.p2s.CROW_STRETCHEDp)
            elif isinstance(self.color, tuple):
                _strs_         = [f for f in self.color if isinstance(f, str)]
                _has_cset_     = self.p2s.CSETp          in self.color
                _has_cset_mag_ = self.p2s.CSET_MAGNITUDEp in self.color
                _has_cset_str_ = self.p2s.CSET_STRETCHEDp in self.color
                _has_cstr_     = bool(set(self.color) & _cstretched_types_)
                if len(_strs_) > 1:
                    self._color_field_          = '__color__'
                    self._color_is_categorical_ = True
                else:
                    self._color_field_ = _strs_[0] if _strs_ else None
                    if _has_cset_mag_ or _has_cset_str_:
                        self._color_is_categorical_   = True
                        self._color_is_cset_spectrum_ = True
                        self._color_is_stretched_     = _has_cset_str_
                    elif _has_cset_:
                        self._color_is_categorical_ = True
                    elif _has_cstr_:
                        self._color_is_stretched_ = True
                    else:
                        if self._color_field_ is not None:
                            _is_num_ = self.p2s.numericColumn(self.df, self._color_field_)
                            # Only the mode chosen purely from dtype is diagnosed; a
                            # magnitude/stretch color enum in the tuple is explicit intent.
                            if not any(isinstance(_e_, self.p2s.ColorTypeP) for _e_ in self.color):
                                self.p2s.logDtypeKeyedColor('Histop', self._color_field_, _is_num_)
                            self._color_is_categorical_ = not _is_num_
                        else:
                            self._color_is_categorical_ = False
            else:
                self._color_field_          = self.color
                _is_num_ = self.p2s.numericColumn(self.df, self.color)
                self.p2s.logDtypeKeyedColor('Histop', self.color, _is_num_)
                self._color_is_categorical_ = not _is_num_

        self._agg_type_      = 'simple'
        self.df_swarm        = None
        self._numeric_field_ = None

        # ── BOXPLOT ───────────────────────────────────────────────────────
        if self.style in {self.p2s.BOXPLOTp, self.p2s.BOXPLOT_W_SWARMp}:
            _nf_ = self.__findNumericCountField__()
            if _nf_ is None:
                self.p2s.logger.warning('Histop: BOXPLOTp requires a numeric count field; falling back to BARCHARTp')
                self.style = self.p2s.BARCHARTp
            else:
                self._numeric_field_ = _nf_
                _bp_agg_ = [
                    pl.col(_nf_).min()          .alias('__box_min__'),
                    pl.col(_nf_).quantile(0.25) .alias('__box_q1__'),
                    pl.col(_nf_).median()       .alias('__box_median__'),
                    pl.col(_nf_).quantile(0.75) .alias('__box_q3__'),
                    pl.col(_nf_).max()          .alias('__box_max__'),
                    pl.len()                    .alias('__count__'),
                ]
                if self.use_lazy_execution:
                    self.df_agg = self.df.lazy().group_by(self._bin_col_).agg(_bp_agg_).collect()
                else:
                    self.df_agg = self.df.group_by(self._bin_col_).agg(_bp_agg_)
                self._agg_type_ = 'boxplot'
                if self.style == self.p2s.BOXPLOT_W_SWARMp:
                    self.df_swarm = (self.df.select([self._bin_col_, _nf_])
                        .with_columns(pl.int_range(pl.len()).over(self._bin_col_).alias('__rank__'))
                        .filter(pl.col('__rank__') < self.swarm_max_pts)
                        .drop('__rank__'))

        # ── STACKED (categorical color only) ──────────────────────────────
        # Numeric color fields use spectrum coloring on the whole bar (simple path).
        # Skip stacked when color field == bin field (one segment per bar).
        if self.style in {self.p2s.BARCHARTp, self.p2s.STACKEDBARp} and self._color_field_ is not None \
                and self._color_is_categorical_ \
                and self._agg_type_ == 'simple' and self._color_field_ != self._bin_col_:
            # Concatenate multiple color fields if needed
            if isinstance(self.color, tuple):
                _strs_ = [f for f in self.color if isinstance(f, str)]
                if len(_strs_) > 1:
                    self.df = self.df.with_columns(pl.concat_str(_strs_, separator=self.p2s.MULTI_FIELD_SEP).alias('__color__'))
            if self.use_lazy_execution:
                self.df_agg = self.df.lazy().group_by([self._bin_col_, self._color_field_]) \
                                            .agg(self.__countAggExpr__()) \
                                            .sort([self._bin_col_, self._color_field_]).collect()
            else:
                self.df_agg = self.df.group_by([self._bin_col_, self._color_field_]) \
                                     .agg(self.__countAggExpr__()) \
                                     .sort([self._bin_col_, self._color_field_])
            self._agg_type_ = 'stacked'

            # Reduce color cardinality: collapse values whose estimated pixel width
            # is below remainder_threshold into an '(other)' bucket.  Mirrors the
            # logic in timep.__computeAggregates2__ (linear lines 415-441, periodic 515-537).
            _est_plot_w_  = float(self.wxh[0])
            _max_bt_      = float(self.df_agg.group_by(self._bin_col_)
                                              .agg(pl.col('__count__').sum().alias('__bt__'))
                                              ['__bt__'].max() or 1.0)
            _color_stats_ = (self.df_agg.group_by(self._color_field_)
                                         .agg(pl.col('__count__').max().alias('__max_in_bin__'))
                                         .with_columns(
                                             (pl.col('__max_in_bin__') / _max_bt_ * _est_plot_w_)
                                             .alias('__est_px__')))
            _visible_     = set(_color_stats_.filter(pl.col('__est_px__') >= self.remainder_threshold)
                                             [self._color_field_].to_list())
            if len(_visible_) < len(_color_stats_):
                _visible_str_ = {str(v) for v in _visible_}
                self.df_agg   = (self.df_agg
                    .with_columns(pl.col(self._color_field_).cast(pl.String))
                    .with_columns(
                        pl.when(pl.col(self._color_field_).is_in(_visible_str_))
                          .then(pl.col(self._color_field_))
                          .otherwise(pl.lit('(other)'))
                          .alias(self._color_field_))
                    .group_by([self._bin_col_, self._color_field_])
                    .agg(pl.col('__count__').sum())
                    .sort([self._bin_col_, self._color_field_]))

        # ── SIMPLE ────────────────────────────────────────────────────────
        if self._agg_type_ == 'simple':
            _agg_exprs_ = [self.__countAggExpr__()]
            _keep_      = {self._bin_col_} | self.__countFields__()
            if self._color_is_crow_:
                _agg_exprs_.append(pl.len().alias('__row_count__'))
            elif self._color_field_ is not None and not self._color_is_categorical_:
                _agg_exprs_.append(self.__colorStatAggExpr__())
                _keep_ |= {self._color_field_}
            _drop_ = [c for c in self.df.columns if c not in _keep_]
            if self.use_lazy_execution:
                self.df_agg = self.df.lazy().drop(_drop_).group_by(self._bin_col_).agg(_agg_exprs_).collect()
            else:
                self.df_agg = self.df.drop(_drop_).group_by(self._bin_col_).agg(_agg_exprs_)
            if self._color_is_crow_:
                self.df_agg = self.df_agg.with_columns(
                    pl.col('__row_count__').cast(pl.Float64).alias('__color_stat__')
                )

        # ── SORT ORDER FOR BINS ────────────────────────────────────────────
        if self._agg_type_ == 'stacked':
            _order_df_ = self.df_agg.group_by(self._bin_col_).agg(pl.col('__count__').sum().alias('__order_metric__'))
        elif self._agg_type_ == 'boxplot':
            _order_df_ = self.df_agg.select([self._bin_col_, pl.col('__count__').alias('__order_metric__')])
        elif self.order == self.p2s.ROW_COUNTp:
            # Sort by the actual bar length (df_agg['__count__']), not raw row count.
            # This ensures set-based counts (n_unique) sort correctly by what the bars show.
            _order_df_ = self.df_agg.select([self._bin_col_, pl.col('__count__').alias('__order_metric__')])
        else:
            _order_df_ = self.df.group_by(self._bin_col_).agg(self.__orderAggExpr__())
        self._sorted_bins_ = _order_df_.sort('__order_metric__', descending=self.descending)[self._bin_col_].to_list()

        # ── COUNT RANGE ───────────────────────────────────────────────────
        if self.count_range_shared is not None:
            self._count_min_, self._count_max_ = self.count_range_shared
        elif self.count_range is not None:
            self._count_min_, self._count_max_ = self.count_range
        else:
            self._count_min_ = 0
            if self._agg_type_ == 'stacked':
                _bin_totals_     = self.df_agg.group_by(self._bin_col_).agg(pl.col('__count__').sum())
                _max_total_      = _bin_totals_['__count__'].max()
                self._count_max_ = _max_total_ if _max_total_ is not None and _max_total_ > 0 else 1
            elif self._agg_type_ == 'boxplot':
                _m_ = self.df_agg['__box_max__'].max() if len(self.df_agg) > 0 else 1
                self._count_max_ = _m_ if _m_ is not None and _m_ > 0 else 1
            else:
                _m_ = self.df_agg['__count__'].max() if len(self.df_agg) > 0 else 1
                self._count_max_ = _m_ if _m_ is not None and _m_ > 0 else 1

        # ── COLOR STAT RANGE (for spectrum coloring) ──────────────────────
        self._color_stat_min_ = None
        self._color_stat_max_ = None
        if (self._color_is_crow_ or
                (self._color_field_ is not None and not self._color_is_categorical_)) \
                and '__color_stat__' in self.df_agg.columns:
            if self.color_stat_range_shared is not None:
                self._color_stat_min_, self._color_stat_max_ = self.color_stat_range_shared
            else:
                _vals_ = self.df_agg.filter(pl.col('__count__') > 0)['__color_stat__'].drop_nulls()
                if len(_vals_) > 0:
                    self._color_stat_min_ = round(float(_vals_.min()), 3)
                    self._color_stat_max_ = round(float(_vals_.max()), 3)

    # ── Geometry ────────────────────────────────────────────────────────────

    #
    # __legendPrepare__() - resolve legend kind/metadata (the capture hook) and the
    # strip to reserve, from the color state __computeAggregates__ already resolved.
    # Histop's whole color domain is known pre-render, so the colorbar is finalized
    # here too (no render-time capture needed).  Decision A: a truthy legend with
    # nothing to legend (color=None / boxplot styles, which ignore color=) silently
    # reserves nothing.
    #
    def __legendPrepare__(self):
        self.legend_info      = None
        self._legend_region_  = None
        self._legend_reserve_ = (0, 0, 0, 0)
        if self.legend_spec is None or self.df is None or len(self.df_agg) == 0: return
        if self.color is None or self._agg_type_ == 'boxplot': return
        _spec_ = self.legend_spec
        if   self._color_is_crow_:            _kind_, _title_default_ = 'colorbar',    'rows'
        elif self._color_is_cset_spectrum_:   _kind_, _title_default_ = 'colorbar',    self.__legendColorFieldName__()
        elif self._color_is_categorical_:     _kind_, _title_default_ = 'categorical', self.__legendColorFieldName__()
        elif self._color_field_ is not None:  _kind_, _title_default_ = 'colorbar',    self.__legendColorFieldName__()
        else:                                 return
        _title_ = _spec_['title'] if _spec_['title'] is not None else _title_default_
        if _kind_ == 'categorical':
            # df_agg carries the rendered categories (post-'(other)' collapse for
            # stacked); weight by the aggregated count so order matches bar area
            _vc_ = self.p2s.legendCategoricalValueCounts(self.df_agg, self._color_field_, weight='__count__')
            self.legend_info = self.p2s.legendInfoCategorical(_spec_, _vc_, _title_)
        else:
            self.legend_info = self.p2s.legendInfoColorbar(_title_)
            if self._color_is_cset_spectrum_:
                # per-segment spectrum: domain = segment count range (mirrors __renderSVG__)
                _vmin_ = round(float(self.df_agg['__count__'].min() or 0), 3)
                _vmax_ = round(float(self.df_agg['__count__'].max() or 1), 3)
            else:
                _vmin_, _vmax_ = self._color_stat_min_, self._color_stat_max_
            self.p2s.legendInfoColorbarFinalize(self.legend_info, _spec_, _vmin_, _vmax_)
        _reserve_ = self.p2s.legendReserve(_spec_, self.legend_info, self.txt_h, self.wxh)
        _l_, _r_, _t_, _b_ = _reserve_
        if self.wxh[0] - (_l_ + _r_) < 48 or self.wxh[1] - (_t_ + _b_) < 48:
            self.p2s.logger.warning(f'Histop.__legendPrepare__(): not enough space for legend (wxh = {self.wxh}); legend dropped')
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
        self._avail_y0_, self._avail_y1_ = _leg_t_, _leg_t_ + h
        x_ins, y_ins = self.insets
        _ctx_h_       = self.txt_h + 2 if self.draw_context else 0
        _right_lbl_w_ = (round(self.txt_h * 0.8) + 4) if self.draw_context else 0

        self._plot_x0_     = _leg_l_ + x_ins
        self._plot_y0_     = _leg_t_ + y_ins + _ctx_h_
        self._plot_w_      = w - 2 * x_ins - _right_lbl_w_
        _chart_right_      = self._plot_x0_ + self._plot_w_
        self._right_lbl_x_ = _chart_right_ + (_right_lbl_w_ - self.txt_h * 0.8) / 2
        self._slot_h_      = self.bar_h + self.v_gap

        _raw_strip_h_        = max(1, self.distribution_bin_w)
        _candidate_strip_y0_ = self._avail_y1_ - y_ins - _raw_strip_h_
        _min_bar_bottom_     = self._plot_y0_ + self._slot_h_ + 4
        if (not self.distribution
                or w < 48 or h < 48
                or self._agg_type_ == 'boxplot'
                or _candidate_strip_y0_ < _min_bar_bottom_):
            self._dist_h_        = 0
            self._dist_strip_y0_ = self._avail_y1_
        else:
            self._dist_h_        = _raw_strip_h_ + y_ins
            self._dist_strip_y0_ = _candidate_strip_y0_

    def __computeDistribution__(self):
        self._dist_stacked_     = {}
        self._dist_bins_lu_     = {}
        self._dist_stacked_max_ = 1
        self._dist_stacked_min_ = 1
        self._dist_n_bins_      = 0
        self._dist_actual_bin_w_ = float(self.distribution_bin_w)
        if self._dist_h_ == 0 or self._agg_type_ not in ('simple', 'stacked') or len(self.df_agg) == 0:
            return
        # Compute actual bin width: vary within ±2 of distribution_bin_w so
        # bins tile the plot width evenly without fixed-integer aliasing.
        _n_bins_ = max(1, round(self._plot_w_ / self.distribution_bin_w))
        _actual_bw_ = self._plot_w_ / _n_bins_
        self._dist_n_bins_       = _n_bins_
        self._dist_actual_bin_w_ = _actual_bw_
        if self._agg_type_ == 'stacked':
            _counts_df_ = self.df_agg.group_by(self._bin_col_).agg(
                pl.col('__count__').sum().alias('__total__')
            )
        else:
            _counts_df_ = self.df_agg.select([
                pl.col(self._bin_col_),
                pl.col('__count__').alias('__total__'),
            ])
        _span_ = max(float(self._count_max_) - float(self._count_min_), 1e-9)
        for _row_ in _counts_df_.iter_rows(named=True):
            _bin_   = _row_[self._bin_col_]
            _total_ = _row_['__total__']
            if _total_ <= 0:
                continue
            _pw_  = self._plot_w_ * (float(_total_) - float(self._count_min_)) / _span_
            _bi_  = min(int(_pw_ / _actual_bw_), _n_bins_ - 1)
            _bi_  = max(0, _bi_)
            self._dist_stacked_[_bi_] = self._dist_stacked_.get(_bi_, 0) + 1
            self._dist_bins_lu_.setdefault(_bi_, []).append(_bin_)
        self._dist_stacked_max_ = max(self._dist_stacked_.values()) if self._dist_stacked_ else 1
        self._dist_stacked_min_ = min(self._dist_stacked_.values()) if self._dist_stacked_ else 1

    # ── Rendering ────────────────────────────────────────────────────────────

    def __renderSVG__(self, rand_id):
        w, h          = self.wxh
        _bg_          = self.p2s.colorTyped('background', 'default')
        _axis_color_  = self.p2s.colorTyped('axis',       'default')
        _axis_inner_  = self.p2s.colorTyped('axis',       'inner')
        _label_color_ = self.p2s.colorTyped('label',      'defaultfg')
        _data_color_  = self.p2s.colorTyped('data',       'default')

        _dl_ = self._dl_ = DisplayList(w, h, bg=_bg_)
        _svg_head_ = f'<svg id="histop_{rand_id}" x="0" y="0" width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">'
        _dl_.rect(0, 0, w, h, _bg_, svg=f'<rect x="0" y="0" width="{w}" height="{h}" fill="{_bg_}" />')

        _n_bins_ = len(self._sorted_bins_)
        _y_v_    = self.v_gap // 2 if self.v_gap > 0 else 0

        # Cull bins whose bar bottom would fall outside the available bar area
        # (_avail_y1_ excludes any bottom legend strip)
        _effective_h_ = self._avail_y1_ - self._dist_h_
        _n_visible_ = 0
        for _i_ in range(_n_bins_):
            if self._plot_y0_ + _i_ * self._slot_h_ + _y_v_ + self.bar_h <= _effective_h_:
                _n_visible_ = _i_ + 1
            else:
                break
        _visible_bins_ = self._sorted_bins_[:_n_visible_]

        # ── RIGHT-SIDE LABEL SIZING DECISIONS ────────────────────────────
        _n_more_  = _n_bins_ - _n_visible_
        _f_std_   = self.txt_h * 0.8
        _rhs_gap_ = 8

        if _n_more_ > 0 and _n_visible_ > 0 and self.draw_context:
            _more_lbl_full_  = f'+{_n_more_} more'
            _more_lbl_short_ = f'+{_n_more_}'
            _L_more_full_    = self.p2s.textLength(_more_lbl_full_, _f_std_)
        else:
            _more_lbl_full_  = ''
            _more_lbl_short_ = ''
            _L_more_full_    = 0.0

        _bin_lbl_   = '|'.join(self._bin_cols_)
        _avail_h_span_ = self._avail_y1_ - self._avail_y0_
        _col_avail_ = _avail_h_span_ - 2 * self.insets[1] - (2 * _L_more_full_ + _rhs_gap_ if _L_more_full_ > 0 else 0)
        _show_col_lbl_ = self.draw_context and _n_visible_ > 0 and _col_avail_ >= _f_std_

        def __countToBarW__(count):
            _span_ = max(float(self._count_max_) - float(self._count_min_), 1e-9)
            return max(0.0, self._plot_w_ * (float(count) - float(self._count_min_)) / _span_)

        # ── BACKGROUND CONTEXT (vertical grid lines behind bars) ──────────
        if self.draw_context and self._plot_w_ > 0 and _n_visible_ > 0:
            _plot_y1_ = self._plot_y0_ + _n_visible_ * self._slot_h_
            for _frac_ in [0.25, 0.5, 0.75, 1.0]:
                _gx_ = self._plot_x0_ + self._plot_w_ * _frac_
                _dl_.line(_gx_, self._plot_y0_, _gx_, _plot_y1_, _axis_inner_, width=0.5,
                          svg=f'<line x1="{_gx_:.1f}" y1="{self._plot_y0_:.1f}" '
                              f'x2="{_gx_:.1f}" y2="{_plot_y1_:.1f}" '
                              f'stroke="{_axis_inner_}" stroke-width="0.5" />')

        # ── RENDER BARS ───────────────────────────────────────────────────
        if _n_visible_ > 0 and len(self.df_agg) > 0:

            # ── SIMPLE ────────────────────────────────────────────────────
            if self._agg_type_ == 'simple':
                _row_lu_        = {row[self._bin_col_]: row for row in self.df_agg.iter_rows(named=True)}
                _use_cat_color_ = self._color_field_ is not None and self._color_is_categorical_

                # Precompute spectrum colour lookup for numeric colour fields (and CROW_*)
                _spectrum_lu_ = {}
                _want_spectrum_ = (
                    self._color_is_crow_ or
                    (self._color_field_ is not None and not self._color_is_categorical_)
                ) and self._color_stat_min_ is not None
                if _want_spectrum_:
                    _df_s_ = self.df_agg.filter(pl.col('__count__') > 0) \
                                        .select([pl.col(self._bin_col_), pl.col('__color_stat__')])
                    if self._color_is_stretched_:
                        _n_ = len(_df_s_)
                        _df_s_ = (_df_s_
                            .sort('__color_stat__')
                            .with_columns(
                                (pl.int_range(_n_).cast(pl.Float64) / max(_n_ - 1, 1))
                                .alias('__norm__')
                            )
                        )
                    else:
                        _cspan_ = max(self._color_stat_max_ - self._color_stat_min_, 1e-9)
                        _df_s_ = _df_s_.with_columns(
                            ((pl.col('__color_stat__').fill_null(self._color_stat_min_) - self._color_stat_min_) / _cspan_)
                            .clip(0.0, 1.0).alias('__norm__')
                        )
                    _df_s_ = (_df_s_
                        .with_columns(self.p2s.colorSpectrumPolarsOperations('__norm__', '__r__', '__g__', '__b__'))
                        .with_columns(
                            self.p2s.hexColorFromRGBTriplesPolarsOperations('__r__', '__g__', '__b__').alias('__hx__')
                        )
                    )
                    _spectrum_lu_ = dict(zip(_df_s_[self._bin_col_].to_list(), _df_s_['__hx__'].to_list()))

                # Batch-resolve categorical bar colors in one Polars pass
                # (replaces a 1-row colorize collect per visible bar)
                _cat_lu_ = {}
                if _use_cat_color_:
                    _cat_vals_ = [str(_row_lu_[_b_].get(self._color_field_, _b_))
                                  for _b_ in _visible_bins_ if _b_ in _row_lu_]
                    _cat_lu_   = self.p2s.colors(_cat_vals_)

                for _i_, _bin_ in enumerate(_visible_bins_):
                    _row_ = _row_lu_.get(_bin_)
                    if _row_ is None: continue
                    _bw_ = __countToBarW__(_row_['__count__'])
                    if 0 < _bw_ < self.min_bar_w: _bw_ = self.min_bar_w
                    if _bw_ <= 0: continue
                    _y_ = self._plot_y0_ + _i_ * self._slot_h_ + _y_v_
                    if _bin_ in _spectrum_lu_:
                        _c_ = _spectrum_lu_[_bin_]
                    elif _use_cat_color_:
                        _cv_ = _row_.get(self._color_field_, _bin_)
                        _c_  = _cat_lu_[str(_cv_)]
                    else:
                        _c_ = _data_color_
                    _dl_.rect(self._plot_x0_, _y_, _bw_, self.bar_h, _c_,
                              svg=f'<rect x="{self._plot_x0_:.1f}" y="{_y_:.1f}" '
                                  f'width="{_bw_:.1f}" height="{self.bar_h:.1f}" '
                                  f'fill="{_c_}" stroke="none" />')

            # ── STACKED ───────────────────────────────────────────────────
            elif self._agg_type_ == 'stacked':
                _color_order_ = self.p2s.colorizeOrder(self.df_agg, '__count__', self._color_field_)
                _ys_         = [self._plot_y0_ + _i_ * self._slot_h_ + _y_v_
                                for _i_ in range(_n_visible_)]
                _bin_dtype_  = self.df_agg[self._bin_col_].dtype
                _y_lookup_   = pl.DataFrame({self._bin_col_: _visible_bins_, '__y__': _ys_},
                                            schema_overrides={self._bin_col_: _bin_dtype_})
                _hexcol_ = None
                _df_render_ = self.df_agg
                if self._color_is_cset_spectrum_:
                    _n_ = len(_df_render_)
                    if self._color_is_stretched_:
                        _n_unique_ = _df_render_['__count__'].n_unique()
                        _df_render_ = (_df_render_
                            .with_columns(
                                (pl.col('__count__').cast(pl.Float64).rank('dense') - 1.0)
                                .truediv(max(_n_unique_ - 1, 1)).alias('__norm__')
                            )
                        )
                    else:
                        _seg_min_ = round(float(_df_render_['__count__'].min() or 0), 3)
                        _seg_max_ = round(float(_df_render_['__count__'].max() or 1), 3)
                        _cspan_   = max(_seg_max_ - _seg_min_, 1e-9)
                        _df_render_ = (_df_render_
                            .with_columns(
                                ((pl.col('__count__').cast(pl.Float64) - _seg_min_) / _cspan_)
                                .clip(0.0, 1.0).alias('__norm__')
                            )
                        )
                    _df_render_ = (_df_render_
                        .with_columns(self.p2s.colorSpectrumPolarsOperations('__norm__', '__r__', '__g__', '__b__'))
                        .with_columns(
                            self.p2s.hexColorFromRGBTriplesPolarsOperations('__r__', '__g__', '__b__').alias('__seg_hex__')
                        )
                    )
                    _hexcol_ = '__seg_hex__'
                _df_visible_ = _df_render_.filter(pl.col(self._bin_col_).is_in(_visible_bins_))
                self.p2s.colorizeAllBarsHorizontal(
                    _df_visible_, self._bin_col_, _y_lookup_,
                    self._plot_x0_, self.bar_h,
                    self._plot_w_, self._count_min_, self._count_max_,
                    self._color_field_, color_order=_color_order_,
                    hexcolor_col=_hexcol_, dl=_dl_
                )

            # ── BOXPLOT ───────────────────────────────────────────────────
            elif self._agg_type_ == 'boxplot':
                _row_lu_    = {row[self._bin_col_]: row for row in self.df_agg.iter_rows(named=True)}
                _cy_offset_ = self.bar_h / 2.0
                _bh_frac_   = max(2.0, self.bar_h * 0.6)

                for _i_, _bin_ in enumerate(_visible_bins_):
                    _row_ = _row_lu_.get(_bin_)
                    if _row_ is None or _row_['__count__'] == 0: continue
                    _cy_       = self._plot_y0_ + _i_ * self._slot_h_ + _cy_offset_ + _y_v_
                    _x_min_    = self._plot_x0_ + __countToBarW__(_row_['__box_min__'])
                    _x_q1_     = self._plot_x0_ + __countToBarW__(_row_['__box_q1__'])
                    _x_median_ = self._plot_x0_ + __countToBarW__(_row_['__box_median__'])
                    _x_q3_     = self._plot_x0_ + __countToBarW__(_row_['__box_q3__'])
                    _x_max_    = self._plot_x0_ + __countToBarW__(_row_['__box_max__'])
                    _dl_.line(_x_min_, _cy_, _x_max_, _cy_, _data_color_, width=1.0,
                              svg=f'<line x1="{_x_min_:.1f}" y1="{_cy_:.1f}" x2="{_x_max_:.1f}" y2="{_cy_:.1f}" '
                                  f'stroke="{_data_color_}" stroke-width="1" />')
                    _box_left_ = min(_x_q1_, _x_q3_)
                    _box_w_    = abs(_x_q3_ - _x_q1_)
                    if _box_w_ > 0:
                        _by0_, _by1_ = _cy_ - _bh_frac_/2, _cy_ + _bh_frac_/2
                        _dl_.rect(_box_left_, _by0_, _box_w_, _bh_frac_, _data_color_, opacity=0.3,
                                  svg=f'<rect x="{_box_left_:.1f}" y="{_cy_ - _bh_frac_/2:.1f}" '
                                      f'width="{_box_w_:.1f}" height="{_bh_frac_:.1f}" '
                                      f'fill="{_data_color_}" fill-opacity="0.3" '
                                      f'stroke="{_data_color_}" stroke-width="0.5" />')
                        # GPU stroke outline of the box (svg above already carries the stroke attr)
                        _dl_.line(_box_left_, _by0_, _box_left_ + _box_w_, _by0_, _data_color_, width=0.5)
                        _dl_.line(_box_left_, _by1_, _box_left_ + _box_w_, _by1_, _data_color_, width=0.5)
                        _dl_.line(_box_left_, _by0_, _box_left_, _by1_, _data_color_, width=0.5)
                        _dl_.line(_box_left_ + _box_w_, _by0_, _box_left_ + _box_w_, _by1_, _data_color_, width=0.5)
                    _dl_.line(_x_median_, _cy_ - _bh_frac_/2, _x_median_, _cy_ + _bh_frac_/2, _data_color_, width=1.5,
                              svg=f'<line x1="{_x_median_:.1f}" y1="{_cy_ - _bh_frac_/2:.1f}" '
                                  f'x2="{_x_median_:.1f}" y2="{_cy_ + _bh_frac_/2:.1f}" '
                                  f'stroke="{_data_color_}" stroke-width="1.5" />')

                if self.style == self.p2s.BOXPLOT_W_SWARMp and self.df_swarm is not None:
                    _dot_r_      = max(1.0, self.bar_h * 0.08)
                    for _i_, _bin_ in enumerate(_visible_bins_):
                        _grp_ = self.df_swarm.filter(pl.col(self._bin_col_) == _bin_)
                        _cy_  = self._plot_y0_ + _i_ * self._slot_h_ + _cy_offset_ + _y_v_
                        for _val_ in _grp_[self._numeric_field_]:
                            _jitter_ = (hash(str(_val_)) % 100 - 50) / 100.0 * self.bar_h * 0.3
                            _py_     = _cy_ + _jitter_
                            _px_     = self._plot_x0_ + __countToBarW__(_val_)
                            _dl_.circle(_px_, _py_, _dot_r_, _data_color_, opacity=0.5,
                                        svg=f'<circle cx="{_px_:.1f}" cy="{_py_:.1f}" r="{_dot_r_:.1f}" '
                                            f'fill="{_data_color_}" fill-opacity="0.5" />')

        # ── BIN LABELS (inside plot, left edge, baseline at bar bottom) ──
        # Per-bin (per-entity) labels -- gated on draw_labels, not draw_context.
        _lbl_max_w_ = self._plot_w_ * 0.5
        for _i_, _bin_ in enumerate(_visible_bins_) if self.draw_labels else []:
            _lbl_ = self.p2s.cropText(self.p2s.formatMultiFieldValue(_bin_), self.txt_h, _lbl_max_w_)
            _ly_  = self._plot_y0_ + _i_ * self._slot_h_ + self.bar_h + _y_v_ - 2
            _dl_.text(self.p2s, _lbl_, self._plot_x0_ + 2, _ly_,
                      txt_h=self.txt_h, anchor='start', color=_label_color_)

        # ── MORE ROWS INDICATOR ───────────────────────────────────────────
        if _n_more_ > 0 and _n_visible_ > 0:
            _more_color_ = self.p2s.colorTyped('indicator', 'more_rows')
            _more_y_ = self._plot_y0_ + _n_visible_ * self._slot_h_ + _y_v_
            _dl_.rect(self._plot_x0_, _more_y_, self._plot_w_, 2, _more_color_,
                      svg=f'<rect x="{self._plot_x0_:.1f}" y="{_more_y_:.1f}" '
                          f'width="{self._plot_w_:.1f}" height="2" '
                          f'fill="{_more_color_}" stroke="none" />')
            if self.draw_context:
                if _show_col_lbl_:
                    _more_lbl_ = _more_lbl_full_
                else:
                    _avail_h_ = _avail_h_span_ - 2 * self.insets[1]
                    if self.p2s.textLength(_more_lbl_full_, _f_std_) <= _avail_h_:
                        _more_lbl_ = _more_lbl_full_
                    elif self.p2s.textLength(_more_lbl_short_, _f_std_) <= _avail_h_:
                        _more_lbl_ = _more_lbl_short_
                    else:
                        _more_lbl_ = None
                if _more_lbl_:
                    _dl_.text(self.p2s, _more_lbl_, self._right_lbl_x_, self._avail_y1_ - self.insets[1],
                              txt_h=_f_std_, anchor='end',
                              color=_more_color_, rotation=90)

        # ── CONTEXT (axis border + count axis labels at top) ──────────────
        if self.draw_context and _n_visible_ > 0:
            _plot_h_ = _n_visible_ * self._slot_h_
            # Axis border (stroke-only rect -> 4 GPU line instances; one svg string)
            _bx0_, _by0_ = self._plot_x0_, self._plot_y0_
            _bx1_, _by1_ = _bx0_ + self._plot_w_, _by0_ + _plot_h_
            _dl_.line(_bx0_, _by0_, _bx1_, _by0_, _axis_color_, width=0.5,
                      svg=f'<rect x="{self._plot_x0_:.1f}" y="{self._plot_y0_:.1f}" '
                          f'width="{self._plot_w_:.1f}" height="{_plot_h_:.1f}" '
                          f'stroke="{_axis_color_}" fill="none" stroke-width="0.5" />')
            _dl_.line(_bx0_, _by1_, _bx1_, _by1_, _axis_color_, width=0.5)
            _dl_.line(_bx0_, _by0_, _bx0_, _by1_, _axis_color_, width=0.5)
            _dl_.line(_bx1_, _by0_, _bx1_, _by1_, _axis_color_, width=0.5)
            # Count labels: min (left), center descriptor, max (right)
            # Center label describes what is being counted
            _ctx_y_ = self._plot_y0_ - 1
            _lbl_h_ = self.txt_h * 0.8
            _min_str_ = self.__formatCount__(self._count_min_)
            _max_str_ = self.__formatCount__(self._count_max_)
            if self.count == self.p2s.ROW_COUNTp:
                _ctr_str_ = 'Rows'
            elif isinstance(self.count, str):
                _ctr_str_ = self.count
            elif isinstance(self.count, tuple):
                _ctr_str_ = '|'.join(f for f in self.count if isinstance(f, str))
            else:
                _ctr_str_ = ''
            self.p2s.svgAxisLabels(
                _min_str_, _ctr_str_, _max_str_,
                self._plot_x0_, self._plot_w_, _ctx_y_, _lbl_h_,
                color_left=_label_color_, color_center=_label_color_, color_right=_label_color_,
                gap=_lbl_h_ * 0.5,
                prefer_center=False,
                dl=_dl_,
            )
            # Bin_by column name — rotated label on the right side, centered vertically
            if _show_col_lbl_:
                _cropped_bin_lbl_ = self.p2s.cropText(_bin_lbl_, _f_std_, _col_avail_)
                _dl_.text(self.p2s, _cropped_bin_lbl_, self._right_lbl_x_, (self._avail_y0_ + self._avail_y1_) / 2,
                          txt_h=_f_std_, anchor='middle',
                          color=_label_color_, rotation=90)

        # ── DISTRIBUTION STRIP ───────────────────────────────────────────
        if self._dist_h_ > 0 and self._dist_stacked_:
            self.__renderDistributionStrip__(_dl_)

        # ── DISTRIBUTION (legacy frequency overlay) ───────────────────────
        if self.draw_distribution:
            self.__renderDistribution__(_dl_, _data_color_, _axis_inner_)

        # ── LEGEND (drawn into the strip reserved by __legendPrepare__) ───
        if getattr(self, 'legend_info', None) is not None and self._legend_region_ is not None:
            _dl_.extend(self.p2s.legendRenderDL(self.wxh, self._legend_region_, self.legend_spec,
                                                self.legend_info, self.txt_h), copy_svg=True)

        # ── BORDER (outer SVG border; distinct from draw_context's plot-region axis border) ──
        if self.draw_border:
            _border_svg_ = f'<rect x="0" y="0" width="{w-1}" height="{h-1}" fill="none" stroke="{_axis_inner_}" stroke-width="1" />'
            _dl_.line(0, 0, w-1, 0, _axis_inner_, width=1.0, svg=_border_svg_)
            _dl_.line(0, h-1, w-1, h-1, _axis_inner_, width=1.0)
            _dl_.line(0, 0, 0, h-1, _axis_inner_, width=1.0)
            _dl_.line(w-1, 0, w-1, h-1, _axis_inner_, width=1.0)

        self.svg = _svg_head_ + _dl_.svg() + '</svg>'

    def __renderDistributionStrip__(self, _dl_):
        _abw_   = self._dist_actual_bin_w_   # float cell width
        _bin_h_ = self.distribution_bin_w    # height stays fixed
        _y0_    = self._dist_strip_y0_
        _bis_   = sorted(self._dist_stacked_.keys())
        _cnts_  = [self._dist_stacked_[_bi_] for _bi_ in _bis_]
        _df_    = pl.DataFrame({'__bi__': _bis_, '__cnt__': _cnts_})
        _norm_lo_   = float(self._dist_stacked_min_)
        _norm_span_ = max(float(self._dist_stacked_max_) - _norm_lo_, 1.0)
        _df_    = _df_.with_columns(
            ((pl.col('__cnt__').cast(pl.Float64) - _norm_lo_) / _norm_span_).clip(0.0, 1.0).alias('__norm__')
        )
        _df_    = (
            _df_
            .with_columns(self.p2s.grayscaleSpectrumPolarsOperations('__norm__', '__r__', '__g__', '__b__'))
            .with_columns(self.p2s.hexColorFromRGBTriplesPolarsOperations('__r__', '__g__', '__b__').alias('__hx__'))
        )
        _gap_ = min(1.0, _abw_ * 0.1)   # 10% gap, at most 1px
        for _row_ in _df_.iter_rows(named=True):
            _rx_ = self._plot_x0_ + _row_['__bi__'] * _abw_ + _gap_
            _ry_ = _y0_ + _gap_
            _rw_ = max(1.0, _abw_ - 2 * _gap_)
            _rh_ = max(1.0, _bin_h_ - 2 * _gap_)
            _dl_.rect(_rx_, _ry_, _rw_, _rh_, _row_['__hx__'],
                      svg=f'<rect x="{_rx_:.1f}" y="{_ry_:.1f}" width="{_rw_:.1f}" '
                          f'height="{_rh_:.1f}" fill="{_row_["__hx__"]}" stroke="none" />')

    def __renderDistribution__(self, _dl_, fill_color, line_color):
        '''Overlay a bucketed frequency distribution of bar lengths below the chart.'''
        if self._agg_type_ not in ('simple', 'stacked') or len(self.df_agg) == 0: return
        _y_ins_   = self.insets[1]
        _dist_h_  = max(20, self.txt_h * 2)
        _dist_y0_ = self._avail_y1_ - _y_ins_ - _dist_h_
        _bar_area_bottom_ = self._plot_y0_ + len(self._sorted_bins_) * self._slot_h_
        if _dist_y0_ < _bar_area_bottom_ + 4: return

        # Gather per-bin counts
        if self._agg_type_ == 'stacked':
            _counts_ = self.df_agg.group_by(self._bin_col_).agg(pl.col('__count__').sum())['__count__'].to_list()
        else:
            _counts_ = self.df_agg['__count__'].to_list()
        if not _counts_: return

        _c_min_, _c_max_ = min(_counts_), max(_counts_)
        if _c_max_ == _c_min_: return

        # Bucket into 10 bins
        _n_buckets_ = 10
        _span_      = _c_max_ - _c_min_
        _buckets_   = [0] * _n_buckets_
        for _c_ in _counts_:
            _bi_ = min(int((_c_ - _c_min_) / _span_ * _n_buckets_), _n_buckets_ - 1)
            _buckets_[_bi_] += 1
        _max_b_ = max(_buckets_) or 1

        _bw_ = self._plot_w_ / _n_buckets_
        for _bi_, _freq_ in enumerate(_buckets_):
            _bh_  = _dist_h_ * _freq_ / _max_b_
            _bx_  = self._plot_x0_ + _bi_ * _bw_
            _by_  = _dist_y0_ + _dist_h_ - _bh_
            _dl_.rect(_bx_, _by_, _bw_, _bh_, fill_color, opacity=0.4,
                      svg=f'<rect x="{_bx_:.1f}" y="{_by_:.1f}" width="{_bw_:.1f}" height="{_bh_:.1f}" '
                          f'fill="{fill_color}" fill-opacity="0.4" stroke="{line_color}" stroke-width="0.3" />')

    def renderSmallMultiples(self, df_all, df_lu, all_key):
        _kwargs_ = {}
        _needs_ref_ = self.p2s.SM_COUNT in self.sm_shared or self.p2s.SM_COLOR in self.sm_shared
        if _needs_ref_:
            _ref_ = Histop(df=df_all, template=self)
            if self.p2s.SM_COUNT in self.sm_shared:
                _kwargs_['count_range_shared'] = (_ref_._count_min_, _ref_._count_max_)
            if self.p2s.SM_COLOR in self.sm_shared and _ref_._color_stat_min_ is not None:
                _kwargs_['color_stat_range_shared'] = (_ref_._color_stat_min_, _ref_._color_stat_max_)
        return {k: Histop(df=v, template=self, **_kwargs_) for k, v in df_lu.items()}

    def render_with(self, df, **overrides):
        return Histop(df=df, template=self, **overrides)

    def filterByRectangle(self, bounding_box, remove_records=False):
        _x0_, _y0_, _x1_, _y1_ = bounding_box
        if _x0_ > _x1_: _x0_, _x1_ = _x1_, _x0_
        if _y0_ > _y1_: _y0_, _y1_ = _y1_, _y0_

        # Replicate render-time culling to identify which bins were actually drawn.
        # Bins beyond _n_visible_ exist in self.df but were never rendered.
        _, h          = self.wxh
        _effective_h_ = h - self._dist_h_
        _y_v_         = self.v_gap // 2 if self.v_gap > 0 else 0
        _n_visible_   = 0
        for _i_ in range(len(self._sorted_bins_)):
            if self._plot_y0_ + _i_ * self._slot_h_ + _y_v_ + self.bar_h <= _effective_h_:
                _n_visible_ = _i_ + 1
            else:
                break
        _visible_bins_ = self._sorted_bins_[:_n_visible_]

        # Per-bin bar pixel bounds
        _span_ = max(float(self._count_max_) - float(self._count_min_), 1e-9)
        def __countToBarW__(v):
            return max(0.0, self._plot_w_ * (float(v) - float(self._count_min_)) / _span_)

        if self._agg_type_ == 'stacked':
            _counts_df_ = self.df_agg.group_by(self._bin_col_).agg(pl.col('__count__').sum())
        elif self._agg_type_ == 'boxplot':
            _counts_df_ = self.df_agg.select([self._bin_col_, '__box_min__', '__box_max__'])
        else:
            _counts_df_ = self.df_agg.select([self._bin_col_, '__count__'])
        _row_lu_ = {row[self._bin_col_]: row for row in _counts_df_.iter_rows(named=True)}

        _selected_bins_ = []
        for _i_, _bin_ in enumerate(_visible_bins_):
            _row_ = _row_lu_.get(_bin_)
            if _row_ is None: continue
            _bar_y0_ = self._plot_y0_ + _i_ * self._slot_h_ + _y_v_
            _bar_y1_ = _bar_y0_ + self.bar_h
            if self._agg_type_ == 'boxplot':
                _bx0_ = self._plot_x0_ + __countToBarW__(_row_['__box_min__'])
                _bx1_ = self._plot_x0_ + __countToBarW__(_row_['__box_max__'])
            else:
                _bx0_ = self._plot_x0_
                _bx1_ = self._plot_x0_ + __countToBarW__(_row_['__count__'])
            if _bx0_ <= _x1_ and _bx1_ >= _x0_ and _bar_y0_ <= _y1_ and _bar_y1_ >= _y0_:
                _selected_bins_.append(_bin_)

        if self._dist_h_ > 0 and self._dist_bins_lu_:
            _strip_y0_ = self._dist_strip_y0_
            _strip_y1_ = self._dist_strip_y0_ + self.distribution_bin_w
            if _y0_ <= _strip_y1_ and _y1_ >= _strip_y0_:
                _x_left_  = _x0_ - self._plot_x0_
                _x_right_ = _x1_ - self._plot_x0_
                _abw_     = self._dist_actual_bin_w_
                for _bi_, _bins_ in self._dist_bins_lu_.items():
                    _cell_x0_ = _bi_ * _abw_
                    _cell_x1_ = (_bi_ + 1) * _abw_
                    if _cell_x0_ <= _x_right_ and _cell_x1_ >= _x_left_:
                        _selected_bins_.extend(_bins_)

        _selected_bins_ = list(dict.fromkeys(_selected_bins_))

        _bin_dtype_   = self.df_agg[self._bin_col_].dtype
        _selected_df_ = pl.DataFrame({self._bin_col_: _selected_bins_},
                                      schema={self._bin_col_: _bin_dtype_})

        # inner: records whose bin was selected by the rectangle
        # anti:  remove selected visible bins; unrendered bins are automatically kept
        #        because _selected_df_ only contains bins from _visible_bins_
        _how_       = 'anti' if remove_records else 'inner'
        _df_result_ = self.df.join(_selected_df_, on=self._bin_col_, how=_how_)
        _to_drop_   = [c for c in ['__p2s_index__'] if c in _df_result_.columns]
        if self._bin_col_ == '__bin__' and '__bin__' in _df_result_.columns:
            _to_drop_.append('__bin__')
        return _df_result_.drop(_to_drop_)

    def filterByOval(self, oval, remove_records=False):
        _cx_, _cy_, _rx_, _ry_ = oval
        # A plain click arrives as a zero-radius oval: keep it covering the pixel under the cursor.
        _rx_, _ry_ = max(float(_rx_), 0.5), max(float(_ry_), 0.5)

        # Exact ellipse-vs-box (axis-aligned) overlap: clamp the oval center to the box,
        # then check that closest point against the ellipse.
        def _ellipse_hits_box_(bx0, by0, bx1, by1):
            _qx_ = min(max(_cx_, bx0), bx1)
            _qy_ = min(max(_cy_, by0), by1)
            return ((_qx_ - _cx_) / _rx_) ** 2 + ((_qy_ - _cy_) / _ry_) ** 2 <= 1.0

        # Replicate render-time culling to identify which bins were actually drawn.
        _, h          = self.wxh
        _effective_h_ = h - self._dist_h_
        _y_v_         = self.v_gap // 2 if self.v_gap > 0 else 0
        _n_visible_   = 0
        for _i_ in range(len(self._sorted_bins_)):
            if self._plot_y0_ + _i_ * self._slot_h_ + _y_v_ + self.bar_h <= _effective_h_:
                _n_visible_ = _i_ + 1
            else:
                break
        _visible_bins_ = self._sorted_bins_[:_n_visible_]

        _span_ = max(float(self._count_max_) - float(self._count_min_), 1e-9)
        def __countToBarW__(v):
            return max(0.0, self._plot_w_ * (float(v) - float(self._count_min_)) / _span_)

        if self._agg_type_ == 'stacked':
            _counts_df_ = self.df_agg.group_by(self._bin_col_).agg(pl.col('__count__').sum())
        elif self._agg_type_ == 'boxplot':
            _counts_df_ = self.df_agg.select([self._bin_col_, '__box_min__', '__box_max__'])
        else:
            _counts_df_ = self.df_agg.select([self._bin_col_, '__count__'])
        _row_lu_ = {row[self._bin_col_]: row for row in _counts_df_.iter_rows(named=True)}

        _selected_bins_ = []
        for _i_, _bin_ in enumerate(_visible_bins_):
            _row_ = _row_lu_.get(_bin_)
            if _row_ is None: continue
            _bar_y0_ = self._plot_y0_ + _i_ * self._slot_h_ + _y_v_
            _bar_y1_ = _bar_y0_ + self.bar_h
            if self._agg_type_ == 'boxplot':
                _bx0_ = self._plot_x0_ + __countToBarW__(_row_['__box_min__'])
                _bx1_ = self._plot_x0_ + __countToBarW__(_row_['__box_max__'])
            else:
                _bx0_ = self._plot_x0_
                _bx1_ = self._plot_x0_ + __countToBarW__(_row_['__count__'])
            if _ellipse_hits_box_(_bx0_, _bar_y0_, _bx1_, _bar_y1_):
                _selected_bins_.append(_bin_)

        if self._dist_h_ > 0 and self._dist_bins_lu_:
            _strip_y0_ = self._dist_strip_y0_
            _strip_y1_ = self._dist_strip_y0_ + self.distribution_bin_w
            _abw_      = self._dist_actual_bin_w_
            for _bi_, _bins_ in self._dist_bins_lu_.items():
                _cell_x0_ = self._plot_x0_ + _bi_ * _abw_
                _cell_x1_ = self._plot_x0_ + (_bi_ + 1) * _abw_
                if _ellipse_hits_box_(_cell_x0_, _strip_y0_, _cell_x1_, _strip_y1_):
                    _selected_bins_.extend(_bins_)

        _selected_bins_ = list(dict.fromkeys(_selected_bins_))

        _bin_dtype_   = self.df_agg[self._bin_col_].dtype
        _selected_df_ = pl.DataFrame({self._bin_col_: _selected_bins_},
                                      schema={self._bin_col_: _bin_dtype_})

        _how_       = 'anti' if remove_records else 'inner'
        _df_result_ = self.df.join(_selected_df_, on=self._bin_col_, how=_how_)
        _to_drop_   = [c for c in ['__p2s_index__'] if c in _df_result_.columns]
        if self._bin_col_ == '__bin__' and '__bin__' in _df_result_.columns:
            _to_drop_.append('__bin__')
        return _df_result_.drop(_to_drop_)

    def filterBySubstring(self, substring, remove_bins=False):
        _sub_ = substring.lower()
        # Match against the display form ('|'-joined) so a user's 'A|x' still matches a
        # multi-field bin whose internal key uses the non-printable MULTI_FIELD_SEP.
        _matching_bins_ = [b for b in self._sorted_bins_
                           if _sub_ in self.p2s.formatMultiFieldValue(b).lower()]
        _bin_dtype_   = self.df_agg[self._bin_col_].dtype
        _selected_df_ = pl.DataFrame({self._bin_col_: _matching_bins_}, schema={self._bin_col_: _bin_dtype_})
        _how_         = 'anti' if remove_bins else 'inner'
        _df_result_   = self.df.join(_selected_df_, on=self._bin_col_, how=_how_)
        _to_drop_     = [c for c in ['__p2s_index__'] if c in _df_result_.columns]
        if self._bin_col_ == '__bin__' and '__bin__' in _df_result_.columns:
            _to_drop_.append('__bin__')
        return _df_result_.drop(_to_drop_)

    def recordsAt(self, xy, shape=None, threshold=2.0):
        """Return the original records whose bin row contains pixel y.

        Only SELECT_HORIZONTALp is supported: the x coordinate and threshold are
        ignored.  Any y within a bin's rendered slot (bar height + gap) selects
        all records belonging to that bin.  Bins that were never rendered because
        they fell outside the SVG height return an empty DataFrame.

        Parameters
        ----------
        xy        : (x, y) pixel coordinate
        shape     : must be p2s.SELECT_HORIZONTALp or None (default)
        threshold : accepted for API compatibility with XYp.recordsAt; unused
        """
        if shape is None: shape = self.p2s.SELECT_HORIZONTALp
        if shape != self.p2s.SELECT_HORIZONTALp:
            raise ValueError(f'Histop.recordsAt(): only SELECT_HORIZONTALp is supported, got {shape}')

        _x_, _y_ = xy

        # Replicate render-time culling to know which bins were actually drawn
        _, h          = self.wxh
        _effective_h_ = h - self._dist_h_
        _y_v_         = self.v_gap // 2 if self.v_gap > 0 else 0
        _n_visible_   = 0
        for _i_ in range(len(self._sorted_bins_)):
            if self._plot_y0_ + _i_ * self._slot_h_ + _y_v_ + self.bar_h <= _effective_h_:
                _n_visible_ = _i_ + 1
            else:
                break

        # Helper: return a correctly-schemed empty DataFrame
        def _empty_():
            _drop_ = [c for c in ['__p2s_index__'] if c in self.df.columns]
            if self._bin_col_ == '__bin__':
                _drop_ += [c for c in ['__bin__'] if c in self.df.columns]
            return self.df.drop(_drop_).clear()

        if self._dist_h_ > 0 and self._dist_bins_lu_:
            _strip_y0_ = self._dist_strip_y0_
            _strip_y1_ = self._dist_strip_y0_ + self.distribution_bin_w
            if _strip_y0_ <= _y_ <= _strip_y1_:
                _x_offset_   = _x_ - self._plot_x0_
                _abw_        = self._dist_actual_bin_w_
                _bi_         = max(0, min(int(_x_offset_ / _abw_), self._dist_n_bins_ - 1))
                _strip_bins_ = self._dist_bins_lu_.get(_bi_, [])
                if not _strip_bins_:
                    return _empty_()
                _bin_dtype_  = self.df_agg[self._bin_col_].dtype
                _selected_   = pl.DataFrame({self._bin_col_: _strip_bins_},
                                            schema={self._bin_col_: _bin_dtype_})
                _df_result_  = self.df.join(_selected_, on=self._bin_col_, how='inner')
                _to_drop_    = [c for c in ['__p2s_index__'] if c in _df_result_.columns]
                if self._bin_col_ == '__bin__' and '__bin__' in _df_result_.columns:
                    _to_drop_.append('__bin__')
                return _df_result_.drop(_to_drop_)

        # Map pixel y to a display index using the full slot height as the hit area.
        # Compute the offset first: y values above _plot_y0_ (negative offset) must
        # yield -1 rather than 0 (int(-0.06) == 0 in Python, which would match
        # the first bin incorrectly).
        _offset_ = _y_ - self._plot_y0_
        if _offset_ < 0 or self._slot_h_ <= 0:
            return _empty_()
        _i_ = int(_offset_ / self._slot_h_)
        if _i_ >= _n_visible_:
            return _empty_()

        _bin_       = self._sorted_bins_[_i_]
        _bin_dtype_ = self.df_agg[self._bin_col_].dtype
        _selected_  = pl.DataFrame({self._bin_col_: [_bin_]},
                                    schema={self._bin_col_: _bin_dtype_})
        _df_result_ = self.df.join(_selected_, on=self._bin_col_, how='inner')
        _to_drop_   = [c for c in ['__p2s_index__'] if c in _df_result_.columns]
        if self._bin_col_ == '__bin__' and '__bin__' in _df_result_.columns:
            _to_drop_.append('__bin__')
        return _df_result_.drop(_to_drop_)
