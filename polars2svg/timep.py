import polars as pl
import polars.selectors as cs
import time
import random

import polars2svg
from polars2svg.p2s_displaylist import DisplayList
from polars2svg.export import ExportMixin

class Timep(ExportMixin):

    _VALID_KWARGS = frozenset({
        'template', 'df',
        'time', 'count', 'count_range', 'count_range_shared',
        'color', 'style',
        'wxh', 'insets', 'draw_context', 'txt_h',
        'sm_shared', 'use_lazy_execution', 'min_bar_w',
        'swarm_max_pts', 'remainder_threshold', 'color_stat_range_shared',
        'date_range_shared', 'min_label_spacing', 'legend',
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
            self.gatherMetrics(self.__computeAggregates2__)
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

    def __parseInput__(self, *args, **kwargs):
        _unknown_ = set(kwargs) - self._VALID_KWARGS
        if _unknown_:
            raise TypeError(f'Timep: unexpected keyword argument(s): {sorted(_unknown_)}')

        # Single source of truth for every parameter (name -> from-scratch default);
        # drives both the from-scratch assignment and the keyword-override copy below.
        _defaults_ = {
            'time':                    None,
            'count':                   self.p2s.ROW_COUNTp,
            'count_range':             None,
            'count_range_shared':      None,
            'color':                   None,
            'style':                   self.p2s.BARCHARTp,
            'wxh':                     (512, 256),
            'insets':                  (2, 2),
            'draw_context':            True,
            'txt_h':                   12,
            'sm_shared':               set(),
            'use_lazy_execution':      True,
            'min_bar_w':               1.0,
            'swarm_max_pts':           50,
            'remainder_threshold':     3.0,
            'color_stat_range_shared': None,
            'date_range_shared':       None,
            'min_label_spacing':       15,
            'legend':                  False,
        }
        self.p2s.assertParamSpecMatches('Timep', self._VALID_KWARGS, _defaults_)

        self.df, self.df_orig = None, None

        # Template support (same pattern as XYp)
        self.template = None
        for i in range(len(args)):
            if isinstance(args[i], Timep): self.template = args[i]
        if 'template' in kwargs: self.template = kwargs['template']
        if self.template is not None:
            _template_copy_ = self.template
            self.p2s._clone_template_state(self, _template_copy_)
            self.template = _template_copy_
        else:
            self.p2s.assignScratchDefaults(self, _defaults_)
            # from-scratch builds only — a template clone is an exact snapshot and
            # must not re-apply session defaults (see Polars2SVG._apply_defaults)
            kwargs = self.p2s._apply_defaults('timep', kwargs)

        # Extract DataFrame
        # Collect any new df from positional args or 'df' kwarg first, so that an
        # explicitly-supplied df always overrides a df inherited from a template.
        # This is required by renderSmallMultiples(), which passes each panel's
        # subset df alongside the template object.
        _new_df_ = None
        for _arg_ in args:
            if isinstance(_arg_, pl.DataFrame):
                if _new_df_ is None: _new_df_ = _arg_
                else:                raise ValueError('Timep.__parseInput__(): df already set (1)')
        if 'df' in kwargs:
            if _new_df_ is None: _new_df_ = kwargs['df']
            else:                raise ValueError('Timep.__parseInput__(): df already set (1)')
        if _new_df_ is not None:
            self.df = self.df_orig = _new_df_   # override any df inherited from template
        if self.df is not None and '__p2s_index__' not in self.df.columns:
            self.df = self.df.with_row_index('__p2s_index__')

        # Extract time field from positional args (string or tuple)
        for arg in args:
            if   isinstance(arg, pl.DataFrame): pass # dataframe already handled
            elif isinstance(arg, Timep):        pass # template already handled
            elif isinstance(arg, str):
                if self.time is None: self.time = arg
                else:                 raise ValueError('Timep.__parseInput__(): time already set')
            elif isinstance(arg, tuple):
                if self.time is None: self.time = arg
                else:                 raise ValueError('Timep.__parseInput__(): time already set')
            else:
                raise ValueError(f'Timep.__parseInput__(): Unknown argument type: {type(arg)}')

        self.p2s.assignKwargOverrides(self, _defaults_, kwargs)

        # "No data" placeholder for early error visibility -- only ever seen when
        # no df is supplied (a successful render overwrites self.svg); makes a
        # dropped-df plumbing mistake visible instead of a silently blank canvas.
        self.wxh = self.p2s.normalizeWxh(self.wxh, 'Timep')
        w, h = self.wxh
        self.svg = self.p2s.placeholderSVG(w, h)

    def __validateInput__(self):
        # Normalize legend= eagerly so a bad spec fails fast (raises InvalidSpecError).
        self.legend_spec = self.p2s.legendResolveSpec(self.legend)
        if self.df is None: return
        self.p2s.checkReservedColumns(self.df, 'Timep')

        # Resolve time field and enum
        if self.time is None:
            _date_cols_ = self.df.select(cs.date() | cs.datetime()).columns
            if len(_date_cols_) == 0:
                raise ValueError('Timep.__validateInput__(): No date/datetime columns found and no time field specified')
            self._time_field_ = _date_cols_[0]
            self._time_enum_  = None
        elif isinstance(self.time, self.p2s.TField):
            self._time_field_ = self.time.column
            self._time_enum_  = self.time.transform
        elif isinstance(self.time, str):
            self._time_field_ = self.time
            self._time_enum_  = None
        elif isinstance(self.time, tuple)  and len(self.time) == 2 and \
             isinstance(self.time[0], str) and \
             (isinstance(self.time[1], self.p2s.TimePeriodicTypeP) or isinstance(self.time[1], self.p2s.TimeLinearTypeP)):
            self._time_field_ = self.time[0]
            self._time_enum_  = self.time[1]
        else:
            raise ValueError(f'Timep.__validateInput__(): time must be str or tuple, got {type(self.time)}')

        if self._time_field_ not in self.df.columns:
            raise ValueError(f'Timep.__validateInput__(): time field "{self._time_field_}" not found in DataFrame')
        if not (self.p2s.dateColumn(self.df, self._time_field_) or self.p2s.dateTimeColumn(self.df, self._time_field_)):
            raise ValueError(f'Timep.__validateInput__(): time field "{self._time_field_}" is not a date/datetime column')

        self._is_periodic_ = isinstance(self._time_enum_, self.p2s.TimePeriodicTypeP)

        # Validate count
        if self.count != self.p2s.ROW_COUNTp:
            if isinstance(self.count, str) and self.p2s.columnInDataFrame(self.count, self.df) == False:
                raise ValueError(f'Timep.__validateInput__(): count field "{self.count}" not found')
            elif isinstance(self.count, tuple):
                for _f_ in self.count:
                    if isinstance(_f_, str) and self.p2s.columnInDataFrame(_f_, self.df) == False:
                        raise ValueError(f'Timep.__validateInput__(): count field "{_f_}" not found')

        # Validate color
        _accept_types_ = {self.p2s.ROW_COUNTp} | self.p2s.color_types | self.p2s.statistic_types
        if self.color is not None:
            if   isinstance(self.color, str) and self.p2s.columnInDataFrame(self.color, self.df):
                pass # okay
            elif isinstance(self.color, tuple):
                for i in range(len(self.color)):
                    _f_ = self.color[i]
                    if   i == len(self.color) - 1 and _f_ in _accept_types_:
                        pass # okay
                    elif isinstance(_f_, str) and self.p2s.columnInDataFrame(_f_, self.df):
                        pass # okay
                    else:
                        raise ValueError(f'Timep.__validateInput__(): color field "{_f_}" not found')

        # Validate style
        _valid_styles_ = {self.p2s.BARCHARTp, self.p2s.BOXPLOTp, self.p2s.BOXPLOT_W_SWARMp, self.p2s.STACKEDBARp}
        if self.style not in _valid_styles_:
            raise ValueError(f'Timep.__validateInput__(): style must be one of {_valid_styles_}')

        # Warn on SM_* values that Timep does not support
        _unsupported_sm_ = {self.p2s.SM_Y} & self.sm_shared
        if _unsupported_sm_:
            self.p2s.logger.warning(
                'Timep: sm_shared contains SM_Y which Timep does not support '
                '(supported: SM_X, SM_COUNT, SM_COLOR); SM_Y will be ignored'
            )

        # wxh was normalized to a canonical (int, int) tuple in __parseInput__ via
        # self.p2s.normalizeWxh(); no re-validation needed here.
        w, h = self.wxh
        if w < 64 or h < 64:
            self.draw_context = False
        x_ins, y_ins = self.insets
        if w - 2 * x_ins < 48: self.insets = (0,             self.insets[1])
        if h - 2 * y_ins < 48: self.insets = (self.insets[0], 0            )

    def __addColumnsToDataFrame__(self):
        _ops_ = []

        # Periodic Time
        if self._is_periodic_:
            _ops_.append(self.p2s.polarsOperationForEnum(self._time_field_, self._time_enum_).alias('__time_bin__'))

        # Count Field
        if   isinstance(self.count, str) and self.p2s.isTField(self.count, df=self.df_orig):
            self.p2s.warnIfTFieldAliasCollides(self.count, self.df_orig, 'Timep')
            _ops_.append(self.p2s.polarsOperationForTField(self.count).alias(self.count))
        elif isinstance(self.count, tuple):
            for _f_ in self.count:
                if isinstance(_f_, str) and self.p2s.isTField(_f_, df=self.df_orig):
                    self.p2s.warnIfTFieldAliasCollides(_f_, self.df_orig, 'Timep')
                    _ops_.append(self.p2s.polarsOperationForTField(_f_).alias(_f_))

        # Color Field
        if isinstance(self.color, str) and self.p2s.isTField(self.color, df=self.df_orig):
            self.p2s.warnIfTFieldAliasCollides(self.color, self.df_orig, 'Timep')
            _ops_.append(self.p2s.polarsOperationForTField(self.color).alias(self.color))
        elif isinstance(self.color, tuple):
            for _f_ in self.color:
                if isinstance(_f_, str) and self.p2s.isTField(_f_, df=self.df_orig):
                    self.p2s.warnIfTFieldAliasCollides(_f_, self.df_orig, 'Timep')
                    _ops_.append(self.p2s.polarsOperationForTField(_f_).alias(_f_))

        # If the operations are non-empty, apply them
        if len(_ops_) > 0:
            if self.use_lazy_execution: self.df = self.df.lazy().with_columns(_ops_).collect()
            else:                       self.df = self.df.with_columns(_ops_)

    # ── Duration mapping (LinearTypeP → polars duration string) ───────────

    def __linearDurMap__(self):
        return {
            self.p2s.LT_Yp:              '1y',
            self.p2s.LT_Y_Qp:            '1q',
            self.p2s.LT_Y_mp:            '1mo',
            self.p2s.LT_Y_m_dp:          '1d',
            self.p2s.LT_Y_m_d_4Hp:       '4h',
            self.p2s.LT_Y_m_d_Hp:        '1h',
            self.p2s.LT_Y_m_d_H_15Mp:    '15m',
            self.p2s.LT_Y_m_d_H_Mp:      '1m',
            self.p2s.LT_Y_m_d_H_M_15Sp:  '15s',
            self.p2s.LT_Y_m_d_H_M_Sp:    '1s',
        }

    def __linearTruncMap__(self):
        '''Like __linearDurMap__ but with dt.truncate-compatible strings (no "1q").'''
        return {
            self.p2s.LT_Yp:              '1y',
            self.p2s.LT_Y_Qp:            '3mo',  # dt.truncate doesn't accept '1q'
            self.p2s.LT_Y_mp:            '1mo',
            self.p2s.LT_Y_m_dp:          '1d',
            self.p2s.LT_Y_m_d_4Hp:       '4h',
            self.p2s.LT_Y_m_d_Hp:        '1h',
            self.p2s.LT_Y_m_d_H_15Mp:    '15m',
            self.p2s.LT_Y_m_d_H_Mp:      '1m',
            self.p2s.LT_Y_m_d_H_M_15Sp:  '15s',
            self.p2s.LT_Y_m_d_H_M_Sp:    '1s',
        }

    def __linearEnumOrder__(self):
        return [
            self.p2s.LT_Yp,
            self.p2s.LT_Y_Qp,
            self.p2s.LT_Y_mp,
            self.p2s.LT_Y_m_dp,
            self.p2s.LT_Y_m_d_4Hp,      # 4-hour bins — between 1d and 1h
            self.p2s.LT_Y_m_d_Hp,
            self.p2s.LT_Y_m_d_H_15Mp,   # 15-minute bins — between 1h and 1m
            self.p2s.LT_Y_m_d_H_Mp,
            self.p2s.LT_Y_m_d_H_M_15Sp, # 15-second bins — between 1m and 1s
            self.p2s.LT_Y_m_d_H_M_Sp,
        ]

    def __dataGranularityCap__(self, sorted_df):
        '''Return the finest allowable TimeLinearTypeP based on actual data granularity.'''
        if self.p2s.dateColumn(sorted_df, self._time_field_):
            return self.p2s.LT_Y_m_dp   # Date columns have no sub-day component
        # Datetime: check time components efficiently
        if len(sorted_df) == 0: return self.p2s.LT_Y_m_dp
        _stats_ = sorted_df.select([
            pl.col(self._time_field_).dt.hour()  .n_unique().alias('__nh__'), # number of hours
            pl.col(self._time_field_).dt.minute().n_unique().alias('__nm__'), # number of minutes
            pl.col(self._time_field_).dt.second().n_unique().alias('__ns__'), # number of seconds
            pl.col(self._time_field_).dt.hour()  .min()     .alias('__h0__'), # hour minimum
            pl.col(self._time_field_).dt.minute().min()     .alias('__m0__'), # minute minimum
            pl.col(self._time_field_).dt.second().min()     .alias('__s0__'), # second minimum
        ]).row(0, named=True)
        _all_same_h_ = _stats_['__nh__'] == 1 and _stats_['__h0__'] == 0
        _all_same_m_ = _stats_['__nm__'] == 1 and _stats_['__m0__'] == 0
        _all_same_s_ = _stats_['__ns__'] == 1 and _stats_['__s0__'] == 0
        if   _all_same_h_ and _all_same_m_ and _all_same_s_: return self.p2s.LT_Y_m_dp
        elif                  _all_same_m_ and _all_same_s_: return self.p2s.LT_Y_m_d_Hp
        elif                                   _all_same_s_: return self.p2s.LT_Y_m_d_H_Mp
        else:                                                return self.p2s.LT_Y_m_d_H_M_Sp

    def __autoResolveLinearEnum2__(self):
        '''Like __autoResolveLinearEnum__ but avoids sort+group_by_dynamic.
        Counts distinct bins via dt.truncate().n_unique() — no sort required.
        Also checks the spine size (min-to-max range ÷ interval) so that sparse
        data with only a few unique timestamps does not produce a spine with far
        more bins than the plot can display.

        Single-pass implementation: the spine-size check needs only the raw
        min/max (scalars), and both spine size and n_unique grow monotonically
        as the granularity gets finer.  So candidates are pre-filtered
        analytically from the span, and the surviving truncations (plus the
        granularity-cap statistics for datetime columns) are evaluated in one
        select over the data instead of one select per enum.'''
        w, _h_       = self.wxh
        x_ins, _     = self.insets
        _axis_w_     = self.txt_h if self.draw_context else 0
        _plot_w_     = w - 2 * x_ins - _axis_w_
        _max_bins_   = max(2, _plot_w_ // 2)
        _trunc_map_  = self.__linearTruncMap__()
        _enum_order_ = self.__linearEnumOrder__()
        _selected_   = self.p2s.LT_Y_mp  # fallback
        # Approximate seconds per truncation string — used to estimate spine length.
        _secs_per_trunc_ = {
            '1y': 365.25*86400, '3mo': 91.3125*86400, '1mo': 30.4375*86400,
            '1d': 86400, '4h': 14400, '1h': 3600, '15m': 900, '1m': 60, '15s': 15, '1s': 1,
        }
        if len(self.df) == 0: return _selected_
        _tf_       = pl.col(self._time_field_)
        _mn_, _mx_ = self.df.select(_tf_.min().alias('__mn__'), _tf_.max().alias('__mx__')).row(0)
        if _mn_ is None or _mx_ is None: return _selected_
        _span_s_   = (_mx_ - _mn_).total_seconds()
        _is_date_  = self.p2s.dateColumn(self.df, self._time_field_)

        # Spine pre-filter: spine size grows monotonically with finer granularity,
        # so everything past the first failing enum can be dropped without a scan.
        # Date columns are additionally capped at daily granularity up front so no
        # sub-day truncation is ever evaluated against a Date column.
        _candidates_ = []
        for _enum_ in _enum_order_:  # coarsest → finest
            if _is_date_ and _enum_order_.index(_enum_) > _enum_order_.index(self.p2s.LT_Y_m_dp): break
            _secs_ = _secs_per_trunc_.get(_trunc_map_[_enum_])
            if _secs_ is not None and int(_span_s_ / _secs_) + 2 > _max_bins_: break
            _candidates_.append(_enum_)
        if not _candidates_: return _selected_

        # One pass: distinct-bin counts for every candidate, plus the time-component
        # statistics that __dataGranularityCap__ would otherwise compute separately.
        _exprs_ = [_tf_.dt.truncate(_trunc_map_[_e_]).n_unique().alias(f'_nu{_i_}_')
                   for _i_, _e_ in enumerate(_candidates_)]
        if not _is_date_:
            _exprs_ += [
                _tf_.dt.hour()  .n_unique().alias('__nh__'),
                _tf_.dt.minute().n_unique().alias('__nm__'),
                _tf_.dt.second().n_unique().alias('__ns__'),
                _tf_.dt.hour()  .min()     .alias('__h0__'),
                _tf_.dt.minute().min()     .alias('__m0__'),
                _tf_.dt.second().min()     .alias('__s0__'),
            ]
        try:
            _stats_ = self.df.select(_exprs_).row(0, named=True)
        except Exception:
            return _selected_

        # Granularity cap (same logic as __dataGranularityCap__)
        if _is_date_:
            _cap_enum_ = self.p2s.LT_Y_m_dp
        else:
            _all_same_h_ = _stats_['__nh__'] == 1 and _stats_['__h0__'] == 0
            _all_same_m_ = _stats_['__nm__'] == 1 and _stats_['__m0__'] == 0
            _all_same_s_ = _stats_['__ns__'] == 1 and _stats_['__s0__'] == 0
            if   _all_same_h_ and _all_same_m_ and _all_same_s_: _cap_enum_ = self.p2s.LT_Y_m_dp
            elif                  _all_same_m_ and _all_same_s_: _cap_enum_ = self.p2s.LT_Y_m_d_Hp
            elif                                   _all_same_s_: _cap_enum_ = self.p2s.LT_Y_m_d_H_Mp
            else:                                                _cap_enum_ = self.p2s.LT_Y_m_d_H_M_Sp
        _cap_idx_ = _enum_order_.index(_cap_enum_)

        # n_unique also grows monotonically with finer granularity → break on first failure.
        for _i_, _enum_ in enumerate(_candidates_):
            if _enum_order_.index(_enum_) > _cap_idx_: break
            _n_ = _stats_[f'_nu{_i_}_']
            if _n_ < 1: continue
            if _n_ <= _max_bins_: _selected_ = _enum_
            else: break
        return _selected_

    # ── Count aggregate expression ─────────────────────────────────────────

    def __countAggExpr__(self):
        if self.count == self.p2s.ROW_COUNTp:
            return pl.len().alias('__count__')
        elif isinstance(self.count, str):
            _is_num_ = self.p2s.numericColumn(self.df, self.count)
            self.p2s.logDtypeKeyedCount('Timep', self.count, _is_num_)
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
        '''Return the set of column names referenced by self.count.'''
        if self.count == self.p2s.ROW_COUNTp: return set()
        if isinstance(self.count, str):        return {self.count}
        if isinstance(self.count, tuple):      return {_f_ for _f_ in self.count if isinstance(_f_, str)}
        return set()

    def __colorStatAggExpr__(self):
        """Aggregation expression for numeric spectrum coloring (default: sum)."""
        _field_ = self._color_field_
        _op_    = pl.col(_field_).sum()
        if isinstance(self.color, tuple):
            for item in self.color:
                if   item in {self.p2s.CMAGNITUDE_MINp,    self.p2s.CSTRETCHED_MINp,    self.p2s.MINp}:    _op_ = pl.col(_field_).min();    break
                elif item in {self.p2s.CMAGNITUDE_MEDIANp, self.p2s.CSTRETCHED_MEDIANp, self.p2s.MEDIANp}: _op_ = pl.col(_field_).median(); break
                elif item in {self.p2s.CMAGNITUDE_MEANp,   self.p2s.CSTRETCHED_MEANp,   self.p2s.MEANp}:   _op_ = pl.col(_field_).mean();   break
                elif item in {self.p2s.CMAGNITUDE_MAXp,    self.p2s.CSTRETCHED_MAXp,    self.p2s.MAXp}:    _op_ = pl.col(_field_).max();    break
                elif item == self.p2s.STDp:                                                                   _op_ = pl.col(_field_).std();    break
        return _op_.alias('__color_stat__')

    def __computeAggregates2__(self):
        '''Faster linear aggregation: truncate + group_by + spine join (O(n) vs O(n log n)).
        Periodic branch is unchanged from __computeAggregates__.
        Boxplot style falls back to sort+group_by_dynamic (fill_null(0) is wrong for stats).'''
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
                                self.p2s.logDtypeKeyedColor('Timep', self._color_field_, _is_num_)
                            self._color_is_categorical_ = not _is_num_
                        else:
                            self._color_is_categorical_ = False
            else:
                self._color_field_          = self.color
                _is_num_ = self.p2s.numericColumn(self.df, self.color)
                self.p2s.logDtypeKeyedColor('Timep', self.color, _is_num_)
                self._color_is_categorical_ = not _is_num_

        self._agg_type_         = 'simple'
        self.df_swarm           = None
        self._numeric_field_    = None
        self._color_categories_ = []

        # ── LINEAR ────────────────────────────────────────────────────────
        if not self._is_periodic_:
            if self._time_enum_ is None:
                self._time_enum_ = self.__autoResolveLinearEnum2__()
            _every_  = self.__linearDurMap__()[self._time_enum_]
            _trunc_  = self.__linearTruncMap__()[self._time_enum_]
            _is_date_ = self.p2s.dateColumn(self.df, self._time_field_)

            # Boxplot — keep sort+group_by_dynamic: fill_null(0) is wrong for stat columns
            if self.style in {self.p2s.BOXPLOTp, self.p2s.BOXPLOT_W_SWARMp}:
                _nf_ = self.__findNumericCountField__()
                if _nf_ is None:
                    self.p2s.logger.warning('Timep: BOXPLOTp requires a numeric count field; falling back to BARCHARTp')
                    self.style = self.p2s.BARCHARTp
                else:
                    self._numeric_field_ = _nf_
                    _sorted_ = self.df.sort(self._time_field_)
                    _keep_ = {self._time_field_, _nf_}
                    _bp_agg_ = [
                        pl.col(_nf_).min()          .alias('__box_min__'),
                        pl.col(_nf_).quantile(0.25) .alias('__box_q1__'),
                        pl.col(_nf_).median()       .alias('__box_median__'),
                        pl.col(_nf_).quantile(0.75) .alias('__box_q3__'),
                        pl.col(_nf_).max()          .alias('__box_max__'),
                        pl.len()                    .alias('__count__'),
                    ]
                    _drop_ = [c for c in _sorted_.columns if c not in _keep_]
                    if self.use_lazy_execution:
                        self.df_agg = _sorted_.lazy().drop(_drop_).group_by_dynamic(self._time_field_, every=_every_).agg(_bp_agg_).collect()
                    else:
                        self.df_agg = _sorted_.drop(_drop_).group_by_dynamic(self._time_field_, every=_every_).agg(_bp_agg_)
                    self._agg_type_ = 'boxplot'
                    if self.style == self.p2s.BOXPLOT_W_SWARMp:
                        if self.use_lazy_execution:
                            self.df_swarm = self.df.lazy().select([self._time_field_, _nf_]).sort(self._time_field_).collect()
                        else:
                            self.df_swarm = self.df.select([self._time_field_, _nf_]).sort(self._time_field_)
                        self.df_swarm = (self.df_swarm
                            .with_columns(pl.int_range(pl.len()).over(self._time_field_).alias('__rank__'))
                            .filter(pl.col('__rank__') < self.swarm_max_pts)
                            .drop('__rank__'))

            # Build datetime spine for non-boxplot styles
            if self._agg_type_ == 'simple':
                if self.date_range_shared is not None:
                    self._date_min_, self._date_max_ = self.date_range_shared
                else:
                    self._date_min_ = self.df[self._time_field_].dt.truncate(_trunc_).min()
                    self._date_max_ = self.df[self._time_field_].dt.truncate(_trunc_).max()
                _min_, _max_ = self._date_min_, self._date_max_
                if _min_ is None or _max_ is None:  # empty df → empty spine → blank chart
                    _spine_ = pl.DataFrame({'__bin__': []}, schema={'__bin__': pl.Date if _is_date_ else self.df.schema[self._time_field_]})
                elif _is_date_:
                    _spine_ = pl.DataFrame({'__bin__': pl.date_range(_min_, _max_, interval=_trunc_, eager=True)})
                else:
                    _tu_    = self.df[self._time_field_].dtype.time_unit
                    _spine_ = pl.DataFrame({'__bin__': pl.datetime_range(_min_, _max_, interval=_trunc_, eager=True, time_unit=_tu_)})

            # Stacked barchart (categorical color only; numeric color uses spectrum on simple bars)
            if self.style in {self.p2s.BARCHARTp, self.p2s.STACKEDBARp} and self._color_field_ is not None \
                    and self._color_is_categorical_ and self._agg_type_ == 'simple':
                if isinstance(self.color, tuple):
                    _strs_ = [f for f in self.color if isinstance(f, str)]
                    if len(_strs_) > 1:
                        self.df = self.df.with_columns(pl.concat_str(_strs_, separator=self.p2s.MULTI_FIELD_SEP).alias('__color__'))
                _keep_ = {self._time_field_, self._color_field_} | self.__countFields__()
                _drop_ = [c for c in self.df.columns if c not in _keep_]
                if self.use_lazy_execution:
                    _partial_ = (self.df.lazy().drop(_drop_)
                                 .with_columns(pl.col(self._time_field_).dt.truncate(_trunc_).alias('__bin__'))
                                 .group_by(['__bin__', self._color_field_])
                                 .agg(self.__countAggExpr__())
                                 .collect())
                else:
                    _partial_ = (self.df.drop(_drop_)
                                 .with_columns(pl.col(self._time_field_).dt.truncate(_trunc_).alias('__bin__'))
                                 .group_by(['__bin__', self._color_field_])
                                 .agg(self.__countAggExpr__()))
                # Merge color categories whose estimated max pixel height is below
                # remainder_threshold into a single '(other)' bucket.  This bounds
                # df_agg to O(bins × visible_colors) regardless of input cardinality.
                _est_plot_h_  = float(self.wxh[1])
                _max_bt_      = float(_partial_.group_by('__bin__')
                                               .agg(pl.col('__count__').sum().alias('__bt__'))
                                               ['__bt__'].max() or 1.0)
                # Use max count of each color in any single bin (not the total across bins).
                # A color that appears in many bins but contributes < remainder_threshold px
                # in each individual bin should be collapsed into the remainder bucket.
                _color_stats_ = (_partial_.group_by(self._color_field_)
                                          .agg(pl.col('__count__').max().alias('__max_in_bin__'))
                                          .with_columns(
                                              (pl.col('__max_in_bin__') / _max_bt_ * _est_plot_h_)
                                              .alias('__est_px__')))
                _visible_     = set(_color_stats_.filter(pl.col('__est_px__') >= self.remainder_threshold)
                                                 [self._color_field_].to_list())
                if len(_visible_) < len(_color_stats_):
                    _visible_str_ = {str(v) for v in _visible_}
                    _partial_     = (_partial_
                        .with_columns(pl.col(self._color_field_).cast(pl.String))
                        .with_columns(
                            pl.when(pl.col(self._color_field_).is_in(_visible_str_))
                              .then(pl.col(self._color_field_))
                              .otherwise(pl.lit('(other)'))
                              .alias(self._color_field_))
                        .group_by(['__bin__', self._color_field_])
                        .agg(pl.col('__count__').sum()))
                # Store full spine for correct bar positioning (empty bins must keep their x slot)
                self._all_stacked_bins_ = _spine_['__bin__'].to_list()
                self.df_agg = (_partial_
                               .rename({'__bin__': self._time_field_})
                               .sort([self._time_field_, self._color_field_]))
                self._color_categories_ = sorted(self.df_agg[self._color_field_].unique().to_list())
                self._agg_type_ = 'stacked'

            # Simple barchart (row count, numeric sum, or numeric spectrum colour)
            if self._agg_type_ == 'simple':
                _agg_exprs_ = [self.__countAggExpr__()]
                _keep_      = {self._time_field_} | self.__countFields__()
                if self._color_is_crow_:
                    _agg_exprs_.append(pl.len().alias('__row_count__'))
                elif self._color_field_ is not None and not self._color_is_categorical_:
                    _agg_exprs_.append(self.__colorStatAggExpr__())
                    _keep_ |= {self._color_field_}
                _drop_ = [c for c in self.df.columns if c not in _keep_]
                if self.use_lazy_execution:
                    _partial_ = (self.df.lazy().drop(_drop_)
                                 .with_columns(pl.col(self._time_field_).dt.truncate(_trunc_).alias('__bin__'))
                                 .group_by('__bin__')
                                 .agg(_agg_exprs_)
                                 .collect())
                else:
                    _partial_ = (self.df.drop(_drop_)
                                 .with_columns(pl.col(self._time_field_).dt.truncate(_trunc_).alias('__bin__'))
                                 .group_by('__bin__')
                                 .agg(_agg_exprs_))
                self.df_agg = (_spine_
                               .join(_partial_, on='__bin__', how='left')
                               .fill_null(0)
                               .rename({'__bin__': self._time_field_}))
                if self._color_is_crow_:
                    self.df_agg = self.df_agg.with_columns(
                        pl.col('__row_count__').cast(pl.Float64).alias('__color_stat__')
                    )

        # ── PERIODIC ────────────────────────────────────────────────── (unchanged)
        else:
            _pmin_, _pmax_ = self.p2s.timePeriodicRange(self._time_enum_)
            _all_bins_     = pl.DataFrame({'__time_bin__': list(range(_pmin_, _pmax_ + 1))},
                                          schema={'__time_bin__': pl.Int64})

            # Boxplot styles
            if self.style in {self.p2s.BOXPLOTp, self.p2s.BOXPLOT_W_SWARMp}:
                _nf_ = self.__findNumericCountField__()
                if _nf_ is None:
                    self.p2s.logger.warning('Timep: BOXPLOTp requires a numeric count field; falling back to BARCHARTp')
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
                        _agg_partial_ = self.df.lazy().group_by('__time_bin__').agg(_bp_agg_).collect()
                    else:
                        _agg_partial_ = self.df.group_by('__time_bin__').agg(_bp_agg_)
                    self.df_agg = _all_bins_.join(_agg_partial_, on='__time_bin__', how='left').fill_null(0)
                    self._agg_type_ = 'boxplot'
                    if self.style == self.p2s.BOXPLOT_W_SWARMp:
                        self.df_swarm = (self.df.select(['__time_bin__', _nf_])
                            .with_columns(pl.int_range(pl.len()).over('__time_bin__').alias('__rank__'))
                            .filter(pl.col('__rank__') < self.swarm_max_pts)
                            .drop('__rank__'))

            # Stacked barchart (categorical color only; numeric color uses spectrum on simple bars)
            if self.style in {self.p2s.BARCHARTp, self.p2s.STACKEDBARp} and self._color_field_ is not None \
                    and self._color_is_categorical_ and self._agg_type_ == 'simple':
                if isinstance(self.color, tuple):
                    _strs_ = [f for f in self.color if isinstance(f, str)]
                    if len(_strs_) > 1:
                        self.df = self.df.with_columns(pl.concat_str(_strs_, separator=self.p2s.MULTI_FIELD_SEP).alias('__color__'))
                if self.use_lazy_execution:
                    self.df_agg = self.df.lazy().group_by(['__time_bin__', self._color_field_]) \
                                                 .agg(self.__countAggExpr__()) \
                                                 .sort(['__time_bin__', self._color_field_]).collect()
                else:
                    self.df_agg = self.df.group_by(['__time_bin__', self._color_field_]) \
                                         .agg(self.__countAggExpr__()) \
                                         .sort(['__time_bin__', self._color_field_])
                _est_plot_h_  = float(self.wxh[1])
                _max_bt_      = float(self.df_agg.group_by('__time_bin__')
                                                  .agg(pl.col('__count__').sum().alias('__bt__'))
                                                  ['__bt__'].max() or 1.0)
                _color_stats_ = (self.df_agg.group_by(self._color_field_)
                                            .agg(pl.col('__count__').max().alias('__max_in_bin__'))
                                            .with_columns(
                                                (pl.col('__max_in_bin__') / _max_bt_ * _est_plot_h_)
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
                        .group_by(['__time_bin__', self._color_field_])
                        .agg(pl.col('__count__').sum())
                        .sort(['__time_bin__', self._color_field_]))
                self._color_categories_ = sorted(self.df_agg[self._color_field_].unique().to_list())
                self._agg_type_ = 'stacked'

            # Simple barchart (row count, numeric sum, or numeric spectrum colour)
            if self._agg_type_ == 'simple':
                _agg_exprs_ = [self.__countAggExpr__()]
                if self._color_is_crow_:
                    _agg_exprs_.append(pl.len().alias('__row_count__'))
                elif self._color_field_ is not None and not self._color_is_categorical_:
                    _agg_exprs_.append(self.__colorStatAggExpr__())
                if self.use_lazy_execution:
                    _agg_partial_ = self.df.lazy().group_by('__time_bin__').agg(_agg_exprs_).collect()
                else:
                    _agg_partial_ = self.df.group_by('__time_bin__').agg(_agg_exprs_)
                self.df_agg = _all_bins_.join(_agg_partial_, on='__time_bin__', how='left').fill_null(0)
                if self._color_is_crow_:
                    self.df_agg = self.df_agg.with_columns(
                        pl.col('__row_count__').cast(pl.Float64).alias('__color_stat__')
                    )

        # ── COUNT RANGE ───────────────────────────────────────────────────
        if self.count_range_shared is not None:
            self._count_min_, self._count_max_ = self.count_range_shared
        elif self.count_range is not None:
            self._count_min_, self._count_max_ = self.count_range
        else:
            self._count_min_ = 0
            if self._agg_type_ == 'stacked':
                _bin_col_   = '__time_bin__' if self._is_periodic_ else self._time_field_
                _totals_    = self.df_agg.group_by(_bin_col_).agg(pl.col('__count__').sum())
                _max_total_ = _totals_['__count__'].max()
                self._count_max_ = _max_total_ if _max_total_ is not None and _max_total_ > 0 else 1
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
                _valid_ser_ = self.df_agg.filter(pl.col('__count__') > 0)['__color_stat__'].drop_nulls()
                if len(_valid_ser_) > 0:
                    self._color_stat_min_ = round(float(_valid_ser_.min()), 3)
                    self._color_stat_max_ = round(float(_valid_ser_.max()), 3)

    def __findNumericCountField__(self):
        if isinstance(self.count, str) and self.p2s.numericColumn(self.df, self.count):
            return self.count
        elif isinstance(self.count, tuple):
            for _f_ in self.count:
                if isinstance(_f_, str) and self.p2s.numericColumn(self.df, _f_):
                    return _f_
        return None

    #
    # __legendPrepare__() - resolve legend kind/metadata (the capture hook) and the
    # strip to reserve, from the color state __computeAggregates2__ already resolved.
    # Mirrors Histop.__legendPrepare__ (same color-state flags, same '__count__'
    # aggregate); Decision A: a truthy legend with nothing to legend (color=None /
    # boxplot styles, which ignore color=) silently reserves nothing.
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
            self.p2s.logger.warning(f'Timep.__legendPrepare__(): not enough space for legend (wxh = {self.wxh}); legend dropped')
            self.legend_info = None
            return
        self._legend_reserve_ = _reserve_
        _pos_ = _spec_['pos']
        if   _pos_ == 'right':  self._legend_region_ = (self.wxh[0] - _r_, 0, _r_, self.wxh[1])
        elif _pos_ == 'left':   self._legend_region_ = (0, 0, _l_, self.wxh[1])
        elif _pos_ == 'top':    self._legend_region_ = (0, 0, self.wxh[0], _t_)
        else:                   self._legend_region_ = (0, self.wxh[1] - _b_, self.wxh[0], _b_)

    def __legendColorFieldName__(self):
        if isinstance(self.color, str):   return self.color
        if isinstance(self.color, tuple): return '|'.join(_f_ for _f_ in self.color if isinstance(_f_, str))
        return ''

    def __constructGeometry__(self):
        w, h         = self.wxh
        # Legend strip (if any) comes out of wxh first -- the plot region shrinks,
        # the physical output size does not ("reserve from wxh").
        self.__legendPrepare__()
        _leg_l_, _leg_r_, _leg_t_, _leg_b_ = self._legend_reserve_
        w -= (_leg_l_ + _leg_r_)
        h -= (_leg_t_ + _leg_b_)
        x_ins, y_ins = self.insets
        _axis_w_     = self.txt_h if self.draw_context else 0

        # Plot width is independent of bottom-label space; compute it first so we can
        # test whether any bottom labels will actually render before reserving the height.
        _plot_w_     = w - x_ins - _axis_w_ - x_ins
        if self.draw_context:
            _lbl_h_  = self.txt_h * 0.8
            _c_w_    = self.p2s.textLength(self.__timeGranularityStr__(), _lbl_h_)
            _axis_h_ = self.txt_h + 4 if _c_w_ <= _plot_w_ else 0
        else:
            _axis_h_ = 0

        self._plot_x0_ = _leg_l_ + x_ins + _axis_w_
        self._plot_y0_ = _leg_t_ + y_ins
        self._plot_w_  = _plot_w_
        self._plot_h_  = h - y_ins - _axis_h_ - y_ins
        self._plot_y1_ = self._plot_y0_ + self._plot_h_

        if self._is_periodic_:
            _pmin_, _pmax_ = self.p2s.timePeriodicRange(self._time_enum_)
            self._bin_min_ = _pmin_
            self._bin_max_ = _pmax_
            self._n_bins_  = _pmax_ - _pmin_ + 1
        else:
            _bin_col_ = self._time_field_
            if self._agg_type_ == 'stacked' and hasattr(self, '_all_stacked_bins_'):
                # _all_stacked_bins_ is the full spine; df_agg only contains bins
                # that have data, so using it would produce a bar_w_raw_ that is
                # too large and push sparse bins off-screen.
                _n_ = len(self._all_stacked_bins_)
            elif self._agg_type_ == 'stacked':
                _n_ = len(self.df_agg[_bin_col_].unique())
            else:
                _n_ = len(self.df_agg)
            self._n_bins_ = max(_n_, 1)

        self._bar_w_raw_ = self._plot_w_ / self._n_bins_
        self._bar_w_     = min(self._bar_w_raw_, max(self.min_bar_w, self._bar_w_raw_ - 1.0))

    def __renderSVG__(self, rand_id):
        w, h          = self.wxh
        _bg_          = self.p2s.colorTyped('background', 'default')
        _axis_color_  = self.p2s.colorTyped('axis',       'default')
        _axis_inner_  = self.p2s.colorTyped('axis',       'inner')
        _label_color_ = self.p2s.colorTyped('label',      'defaultfg')
        _data_color_  = self.p2s.colorTyped('data',       'default')

        _dl_ = self._dl_ = DisplayList(w, h, bg=_bg_)
        _svg_head_ = f'<svg id="timep_{rand_id}" x="0" y="0" width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">'
        _dl_.rect(0, 0, w, h, _bg_, svg=f'<rect x="0" y="0" width="{w}" height="{h}" fill="{_bg_}" />')

        def __binToX__(idx):
            return self._plot_x0_ + idx * self._bar_w_raw_

        def __countToBarH__(count):
            _span_ = max(float(self._count_max_) - float(self._count_min_), 1e-9)
            return max(0.0, self._plot_h_ * (float(count) - float(self._count_min_)) / _span_)

        # ── BACKGROUND CONTEXT (grid lines behind bars) ───────────────────
        if self.draw_context:
            self.__renderTimeContextBG__(_dl_, _axis_inner_)

        # ── RENDER BARS ───────────────────────────────────────────────────
        if len(self.df_agg) > 0:

            # ── SIMPLE ────────────────────────────────────────────────────
            if self._agg_type_ == 'simple':
                # Precompute spectrum colour lookup for numeric colour fields (and CROW_*)
                _spectrum_lu_ = {}
                _bin_col_     = '__time_bin__' if self._is_periodic_ else self._time_field_
                _want_spectrum_ = (
                    self._color_is_crow_ or
                    (self._color_field_ is not None and not self._color_is_categorical_)
                ) and self._color_stat_min_ is not None
                if _want_spectrum_:
                    _df_s_ = self.df_agg.filter(pl.col('__count__') > 0) \
                                        .select([pl.col(_bin_col_), pl.col('__color_stat__')])
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
                    _spectrum_lu_ = dict(zip(_df_s_[_bin_col_].to_list(), _df_s_['__hx__'].to_list()))

                _rows_ = self.df_agg.iter_rows(named=True)

                for _i_, _row_ in enumerate(_rows_):
                    _idx_    = int(_row_['__time_bin__']) - self._bin_min_ if self._is_periodic_ else _i_
                    _x_      = __binToX__(_idx_)
                    _bh_     = __countToBarH__(_row_['__count__'])
                    _bin_key_ = _row_['__time_bin__'] if self._is_periodic_ else _row_[self._time_field_]
                    _c_      = _spectrum_lu_.get(_bin_key_, _data_color_) if _spectrum_lu_ else _data_color_
                    if _bh_ > 0:
                        _dl_.rect(_x_, self._plot_y1_ - _bh_, self._bar_w_, _bh_, _c_,
                                  svg=f'<rect x="{_x_:.1f}" y="{self._plot_y1_ - _bh_:.1f}" '
                                      f'width="{self._bar_w_:.1f}" height="{_bh_:.1f}" fill="{_c_}" />')

            # ── STACKED ───────────────────────────────────────────────────
            elif self._agg_type_ == 'stacked':
                _bin_col_st_  = '__time_bin__' if self._is_periodic_ else self._time_field_
                _color_order_ = self.p2s.colorizeOrder(self.df_agg, '__count__', self._color_field_)
                if self._is_periodic_:
                    _sorted_bins_ = sorted(self.df_agg[_bin_col_st_].unique().to_list())
                    _xs_ = [self._plot_x0_ + (int(_bk_) - self._bin_min_) * self._bar_w_raw_
                            for _bk_ in _sorted_bins_]
                else:
                    _sorted_bins_ = (self._all_stacked_bins_
                                     if hasattr(self, '_all_stacked_bins_')
                                     else sorted(self.df_agg[_bin_col_st_].unique().to_list()))
                    _xs_ = [self._plot_x0_ + _i_ * self._bar_w_raw_
                            for _i_ in range(len(_sorted_bins_))]
                _x_lookup_ = pl.DataFrame({_bin_col_st_: _sorted_bins_, '__x__': _xs_})
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
                self.p2s.colorizeAllBarsVertical(
                    _df_render_, _bin_col_st_, _x_lookup_,
                    self._plot_y1_, self._bar_w_,
                    self._plot_h_, self._count_min_, self._count_max_,
                    self._color_field_, color_order=_color_order_,
                    hexcolor_col=_hexcol_, dl=_dl_
                )

            # ── BOXPLOT ───────────────────────────────────────────────────
            elif self._agg_type_ == 'boxplot':
                _bin_col_   = '__time_bin__' if self._is_periodic_ else self._time_field_
                _cx_offset_ = self._bar_w_raw_ / 2.0
                for _i_, _row_ in enumerate(self.df_agg.iter_rows(named=True)):
                    _idx_ = int(_row_['__time_bin__']) - self._bin_min_ if self._is_periodic_ else _i_
                    _cx_  = __binToX__(_idx_) + _cx_offset_
                    _bw_  = max(2.0, self._bar_w_ * 0.6)
                    _y_min_    = self._plot_y1_ - __countToBarH__(_row_['__box_min__'])
                    _y_q1_     = self._plot_y1_ - __countToBarH__(_row_['__box_q1__'])
                    _y_median_ = self._plot_y1_ - __countToBarH__(_row_['__box_median__'])
                    _y_q3_     = self._plot_y1_ - __countToBarH__(_row_['__box_q3__'])
                    _y_max_    = self._plot_y1_ - __countToBarH__(_row_['__box_max__'])
                    if _row_['__count__'] == 0: continue
                    # Whisker (max to min, top to bottom in SVG)
                    _dl_.line(_cx_, _y_max_, _cx_, _y_min_, _data_color_, width=1.0,
                              svg=f'<line x1="{_cx_:.1f}" y1="{_y_max_:.1f}" x2="{_cx_:.1f}" y2="{_y_min_:.1f}" '
                                  f'stroke="{_data_color_}" stroke-width="1" />')
                    # IQR box
                    _box_top_ = min(_y_q1_, _y_q3_)
                    _box_h_   = abs(_y_q1_ - _y_q3_)
                    if _box_h_ > 0:
                        _bx0_, _bx1_ = _cx_ - _bw_/2, _cx_ + _bw_/2
                        _dl_.rect(_bx0_, _box_top_, _bw_, _box_h_, _data_color_, opacity=0.3,
                                  svg=f'<rect x="{_cx_ - _bw_/2:.1f}" y="{_box_top_:.1f}" '
                                      f'width="{_bw_:.1f}" height="{_box_h_:.1f}" '
                                      f'fill="{_data_color_}" fill-opacity="0.3" '
                                      f'stroke="{_data_color_}" stroke-width="0.5" />')
                        # GPU stroke outline of the box (svg above already carries the stroke attr)
                        _dl_.line(_bx0_, _box_top_, _bx1_, _box_top_, _data_color_, width=0.5)
                        _dl_.line(_bx0_, _box_top_ + _box_h_, _bx1_, _box_top_ + _box_h_, _data_color_, width=0.5)
                        _dl_.line(_bx0_, _box_top_, _bx0_, _box_top_ + _box_h_, _data_color_, width=0.5)
                        _dl_.line(_bx1_, _box_top_, _bx1_, _box_top_ + _box_h_, _data_color_, width=0.5)
                    # Median line
                    _dl_.line(_cx_ - _bw_/2, _y_median_, _cx_ + _bw_/2, _y_median_, _data_color_, width=1.5,
                              svg=f'<line x1="{_cx_ - _bw_/2:.1f}" y1="{_y_median_:.1f}" '
                                  f'x2="{_cx_ + _bw_/2:.1f}" y2="{_y_median_:.1f}" '
                                  f'stroke="{_data_color_}" stroke-width="1.5" />')

                # Swarm overlay
                if self.style == self.p2s.BOXPLOT_W_SWARMp and self.df_swarm is not None:
                    _dot_r_ = max(1.0, self._bar_w_ * 0.08)
                    for _i_, (_bk_, _grp_) in enumerate(
                            self.df_swarm.group_by(_bin_col_, maintain_order=True)):
                        _idx_ = int(_bk_[0]) - self._bin_min_ if self._is_periodic_ else _i_
                        _cx_  = __binToX__(_idx_) + _cx_offset_
                        for _val_ in _grp_[self._numeric_field_]:
                            _jitter_ = (hash(str(_val_)) % 100 - 50) / 100.0 * self._bar_w_ * 0.3
                            _px_     = _cx_ + _jitter_
                            _py_     = self._plot_y1_ - __countToBarH__(_val_)
                            _dl_.circle(_px_, _py_, _dot_r_, _data_color_, opacity=0.5,
                                        svg=f'<circle cx="{_px_:.1f}" cy="{_py_:.1f}" r="{_dot_r_:.1f}" '
                                            f'fill="{_data_color_}" fill-opacity="0.5" />')

        # ── CONTEXT ───────────────────────────────────────────────────────
        if self.draw_context:
            # Axis border (stroke-only rect -> 4 GPU line instances; one svg string)
            _bx0_, _by0_ = self._plot_x0_, self._plot_y0_
            _bx1_, _by1_ = _bx0_ + self._plot_w_, _by0_ + self._plot_h_
            _dl_.line(_bx0_, _by0_, _bx1_, _by0_, _axis_color_, width=0.5,
                      svg=f'<rect x="{self._plot_x0_}" y="{self._plot_y0_}" '
                          f'width="{self._plot_w_}" height="{self._plot_h_}" '
                          f'stroke="{_axis_color_}" fill="none" stroke-width="0.5" />')
            _dl_.line(_bx0_, _by1_, _bx1_, _by1_, _axis_color_, width=0.5)
            _dl_.line(_bx0_, _by0_, _bx0_, _by1_, _axis_color_, width=0.5)
            _dl_.line(_bx1_, _by0_, _bx1_, _by1_, _axis_color_, width=0.5)
            # Y-axis labels
            _lx_  = self._plot_x0_ - 2
            _dl_.text(self.p2s, self.__formatCount__(self._count_max_), _lx_,
                      self._plot_y0_,
                      txt_h=self.txt_h * 0.8, anchor='end', color=_label_color_, rotation=270)
            _dl_.text(self.p2s, self.__formatCount__(self._count_min_), _lx_,
                      self._plot_y1_,
                      txt_h=self.txt_h * 0.8, anchor='start', color=_label_color_, rotation=270)
            self.__renderTimeContext__(_dl_, _axis_inner_, _label_color_)

        # ── LEGEND (drawn into the strip reserved by __legendPrepare__) ───
        if getattr(self, 'legend_info', None) is not None and self._legend_region_ is not None:
            _dl_.extend(self.p2s.legendRenderDL(self.wxh, self._legend_region_, self.legend_spec,
                                                self.legend_info, self.txt_h), copy_svg=True)

        self.svg = _svg_head_ + _dl_.svg() + '</svg>'

    def __formatCount__(self, count):
        if count is None: return '0'
        _v_ = float(count)
        if   _v_ >= 1_000_000: return f'{_v_/1_000_000:.1f}M'
        elif _v_ >= 1_000:     return f'{_v_/1_000:.1f}K'
        elif _v_ == int(_v_):  return str(int(_v_))
        else:                  return f'{_v_:.2g}'

    def __linearFormatStr__(self):
        return {
            self.p2s.LT_Yp:              '%Y',
            self.p2s.LT_Y_Qp:            '%Y-%m',
            self.p2s.LT_Y_mp:            '%Y-%m',
            self.p2s.LT_Y_m_dp:          '%Y-%m-%d',
            self.p2s.LT_Y_m_d_4Hp:       '%Y-%m-%d %H',
            self.p2s.LT_Y_m_d_Hp:        '%Y-%m-%d %H',
            self.p2s.LT_Y_m_d_H_15Mp:    '%Y-%m-%d %H:%M',
            self.p2s.LT_Y_m_d_H_Mp:      '%Y-%m-%d %H:%M',
            self.p2s.LT_Y_m_d_H_M_15Sp:  '%Y-%m-%d %H:%M:%S',
            self.p2s.LT_Y_m_d_H_M_Sp:    '%Y-%m-%d %H:%M:%S',
        }.get(self._time_enum_, '%Y-%m-%d')

    def __linearTopTickLabel__(self, ts):
        '''Short contextual label for a top-axis tick mark.
        Shows only the component that varies at the current granularity;
        higher-order context (date, month, year) is shown at boundary crossings.'''
        e, p = self._time_enum_, self.p2s
        if not hasattr(ts, 'strftime'):
            return str(ts)[:10]
        if e == p.LT_Yp:
            return ts.strftime('%Y')
        elif e in (p.LT_Y_Qp, p.LT_Y_mp):
            return ts.strftime('%Y')                            # marks always land at January
        elif e == p.LT_Y_m_dp:
            if ts.month == 1:
                return ts.strftime("%b '%y")                   # "Jan '23" — year context in January
            return ts.strftime('%b')                           # "Mar"
        elif e in (p.LT_Y_m_d_4Hp, p.LT_Y_m_d_Hp):
            if ts.month == 1 and ts.day == 1:
                return ts.strftime("%b %d '%y")               # "Jan 01 '23" — year context
            return ts.strftime('%b %d')                        # "Mar 15"
        elif e in (p.LT_Y_m_d_H_15Mp, p.LT_Y_m_d_H_Mp):
            if getattr(ts, 'hour', 0) == 0:
                return ts.strftime('%b %d')                    # "Mar 15" at midnight
            return ts.strftime('%Hh')                          # "14h"
        elif e in (p.LT_Y_m_d_H_M_15Sp, p.LT_Y_m_d_H_M_Sp):
            if getattr(ts, 'hour', 0) == 0 and getattr(ts, 'minute', 0) == 0:
                return ts.strftime('%b %d')                    # "Mar 15" at midnight
            if getattr(ts, 'minute', 0) == 0:
                return ts.strftime('%Hh')                      # "14h" at top of hour
            return ts.strftime('%H:%M')                        # "14:30"
        return ts.strftime('%Y-%m-%d')

    def __linearMajorMarkFn__(self):
        '''Return a predicate ts→bool identifying major context boundaries for the linear time enum.
        Mirrors the modulo logic in rt_temporal_barchart_mixin.drawTemporalContext().'''
        e, p = self._time_enum_, self.p2s
        if   e == p.LT_Yp:               return lambda ts: ts.year % 5 == 0
        elif e in (p.LT_Y_Qp,
                   p.LT_Y_mp):           return lambda ts: getattr(ts, 'month', 1) == 1
        elif e == p.LT_Y_m_dp:           return lambda ts: ts.day % 40 == 1
        elif e == p.LT_Y_m_d_4Hp:        return lambda ts: getattr(ts, 'hour', 0) == 0
        elif e == p.LT_Y_m_d_Hp:         return lambda ts: ts.day % 2 == 0 and getattr(ts, 'hour', 0) == 0
        elif e == p.LT_Y_m_d_H_15Mp:     return lambda ts: getattr(ts, 'minute', 0) == 0
        elif e == p.LT_Y_m_d_H_Mp:       return lambda ts: getattr(ts, 'hour', 0) % 2 == 0 and getattr(ts, 'minute', 0) == 0
        elif e == p.LT_Y_m_d_H_M_15Sp:   return lambda ts: getattr(ts, 'second', 0) == 0
        elif e == p.LT_Y_m_d_H_M_Sp:     return lambda ts: getattr(ts, 'second', 0) == 0
        else:                             return lambda ts: False

    def __periodicMajorPeriod__(self):
        '''Return None (every bin is a potential major mark) or an int period P such that
        0-based bin index i is major when i % P == 0.'''
        e, p = self._time_enum_, self.p2s
        if e in {p.PT_Qp, p.PT_mp, p.PT_DoWp, p.PT_dp, p.PT_Hp, p.PT_Mp, p.PT_Sp}:
            return None
        return {
            p.PT_DoYp:     30,
            p.PT_m_dp:     30,
            p.PT_m_d_Hp:   24,
            p.PT_DoW_Hp:   24,
            p.PT_DoW_H_Mp: 24 * 60,
            p.PT_d_Hp:     24,
            p.PT_d_H_Mp:   24 * 60,
            p.PT_H_Mp:     60,
            p.PT_H_M_Sp:   3600,
            p.PT_M_Sp:     60,
        }.get(e, None)

    def __timeGranularityStr__(self):
        '''Human-readable granularity label for the center position below the axis.'''
        if self._is_periodic_:
            return {
                self.p2s.PT_Qp:       'quarter',
                self.p2s.PT_mp:       'month',
                self.p2s.PT_m_dp:     'day of month',
                self.p2s.PT_m_d_Hp:   'month/day/hour',
                self.p2s.PT_DoYp:     'day of year',
                self.p2s.PT_DoWp:     'day of week',
                self.p2s.PT_DoW_Hp:   'day of week / hour',
                self.p2s.PT_DoW_H_Mp: 'day of week / hour / min',
                self.p2s.PT_dp:       'day',
                self.p2s.PT_d_Hp:     'day / hour',
                self.p2s.PT_d_H_Mp:   'day / hour / min',
                self.p2s.PT_Hp:       'hour',
                self.p2s.PT_H_Mp:     'hour / minute',
                self.p2s.PT_H_M_Sp:   'hour / min / sec',
                self.p2s.PT_Mp:       'minute',
                self.p2s.PT_M_Sp:     'minute / second',
                self.p2s.PT_Sp:       'second',
            }.get(self._time_enum_, 'periodic')
        else:
            return {
                self.p2s.LT_Yp:           'yearly',
                self.p2s.LT_Y_Qp:         'quarterly',
                self.p2s.LT_Y_mp:         'monthly',
                self.p2s.LT_Y_m_dp:           'daily',
                self.p2s.LT_Y_m_d_4Hp:        'every 4 hours',
                self.p2s.LT_Y_m_d_Hp:         'hourly',
                self.p2s.LT_Y_m_d_H_15Mp:     'every 15 min',
                self.p2s.LT_Y_m_d_H_Mp:       'by minute',
                self.p2s.LT_Y_m_d_H_M_15Sp:   'every 15 sec',
                self.p2s.LT_Y_m_d_H_M_Sp:     'by second',
            }.get(self._time_enum_, 'by time')

    def __renderTimeContextBG__(self, _dl_, grid_color):
        '''Draw context cues inside the chart at the top, in a light color.
        Major marks: full-height solid line + text label near the top.
        Minor marks: short 5px tick from the top only.
        All elements are drawn before bars so they appear behind them.'''
        _lbl_h_  = self.txt_h * 0.75
        _lbl_y_  = self._plot_y0_ + _lbl_h_          # label baseline, one line-height below top
        _tick_y2_ = self._plot_y0_ + 5               # minor tick: 5px down from top

        _last_lbl_right_ = self._plot_x0_ - self.min_label_spacing

        if self._is_periodic_:
            _period_ = self.__periodicMajorPeriod__()
            _total_  = self._bin_max_ - self._bin_min_ + 1
            for _i_ in range(1, _total_):             # skip i==0 (coincides with left plot border)
                _bv_       = self._bin_min_ + _i_
                _is_major_ = _period_ is None or _i_ % _period_ == 0
                _x_        = self._plot_x0_ + _i_ * self._bar_w_raw_
                if _is_major_:
                    _dl_.line(_x_, self._plot_y0_, _x_, self._plot_y1_, grid_color, width=0.5,
                              svg=f'<line x1="{_x_:.1f}" y1="{self._plot_y0_:.1f}" '
                                  f'x2="{_x_:.1f}" y2="{self._plot_y1_:.1f}" '
                                  f'stroke="{grid_color}" stroke-width="0.5" />')
                    _lbl_   = self.p2s.timePeriodicHumanReadable(_bv_, self._time_enum_)
                    _lbl_x_ = _x_ + 2
                    _est_w_ = len(_lbl_) * _lbl_h_ * 0.6
                    if _lbl_x_ - _last_lbl_right_ >= self.min_label_spacing:
                        _dl_.text(self.p2s, _lbl_, _lbl_x_, _lbl_y_, txt_h=_lbl_h_, anchor='start', color=grid_color)
                        _last_lbl_right_ = _lbl_x_ + _est_w_
                else:
                    _dl_.line(_x_, self._plot_y0_, _x_, _tick_y2_, grid_color, width=0.3,
                              svg=f'<line x1="{_x_:.1f}" y1="{self._plot_y0_:.1f}" '
                                  f'x2="{_x_:.1f}" y2="{_tick_y2_:.1f}" '
                                  f'stroke="{grid_color}" stroke-width="0.3" />')
        else:
            if self._agg_type_ == 'stacked':
                _bins_ = (self._all_stacked_bins_
                          if hasattr(self, '_all_stacked_bins_')
                          else sorted(self.df_agg[self._time_field_].unique().to_list()))
            else:
                _bins_ = self.df_agg[self._time_field_].to_list()
            if not _bins_: return
            _is_major_ = self.__linearMajorMarkFn__()
            for _i_, _ts_ in enumerate(_bins_):
                if _i_ == 0: continue                 # skip i==0 (left plot border)
                _x_        = self._plot_x0_ + _i_ * self._bar_w_raw_
                if _is_major_(_ts_):
                    _dl_.line(_x_, self._plot_y0_, _x_, self._plot_y1_, grid_color, width=0.5,
                              svg=f'<line x1="{_x_:.1f}" y1="{self._plot_y0_:.1f}" '
                                  f'x2="{_x_:.1f}" y2="{self._plot_y1_:.1f}" '
                                  f'stroke="{grid_color}" stroke-width="0.5" />')
                    _lbl_   = self.__linearTopTickLabel__(_ts_)
                    _lbl_x_ = _x_ + 2
                    _est_w_ = len(_lbl_) * _lbl_h_ * 0.6
                    if _lbl_x_ - _last_lbl_right_ >= self.min_label_spacing:
                        _dl_.text(self.p2s, _lbl_, _lbl_x_, _lbl_y_,
                                  txt_h=_lbl_h_, anchor='start', color=grid_color)
                        _last_lbl_right_ = _lbl_x_ + _est_w_
                else:
                    _dl_.line(_x_, self._plot_y0_, _x_, _tick_y2_, grid_color, width=0.3,
                              svg=f'<line x1="{_x_:.1f}" y1="{self._plot_y0_:.1f}" '
                                  f'x2="{_x_:.1f}" y2="{_tick_y2_:.1f}" '
                                  f'stroke="{grid_color}" stroke-width="0.3" />')

    def __renderTimeContext__(self, _dl_, tick_color, label_color):
        _y_lbl_ = self._plot_y1_ + self.txt_h + 2
        _lbl_h_ = self.txt_h * 0.8
        _fmt_   = self.__linearFormatStr__()
        if self._is_periodic_:
            _l_lbl_ = self.p2s.timePeriodicHumanReadable(self._bin_min_, self._time_enum_)
            _r_lbl_ = self.p2s.timePeriodicHumanReadable(self._bin_max_, self._time_enum_)
        else:
            if self._agg_type_ == 'stacked':
                _bins_ = (self._all_stacked_bins_
                          if hasattr(self, '_all_stacked_bins_')
                          else sorted(self.df_agg[self._time_field_].unique().to_list()))
            else:
                _bins_ = self.df_agg[self._time_field_].to_list()
            if not _bins_: return
            _l_ts_, _r_ts_ = _bins_[0], _bins_[-1]
            _l_lbl_ = _l_ts_.strftime(_fmt_) if hasattr(_l_ts_, 'strftime') else str(_l_ts_)[:10]
            _r_lbl_ = _r_ts_.strftime(_fmt_) if hasattr(_r_ts_, 'strftime') else str(_r_ts_)[:10]
        _c_lbl_ = self.__timeGranularityStr__()
        self.p2s.svgAxisLabels(
            _l_lbl_, _c_lbl_, _r_lbl_,
            self._plot_x0_, self._plot_w_, _y_lbl_, _lbl_h_,
            color_left=label_color, color_center=label_color, color_right=label_color,
            prefer_center=True,
            dl=_dl_,
        )

    def renderSmallMultiples(self, df_all, df_lu, all_key):
        _kwargs_ = {}
        _needs_ref_ = (self.p2s.SM_COUNT in self.sm_shared or
                       self.p2s.SM_COLOR  in self.sm_shared or
                       self.p2s.SM_X      in self.sm_shared)
        if _needs_ref_:
            _ref_ = Timep(df=df_all, template=self)
            if self.p2s.SM_X in self.sm_shared and not self._is_periodic_ and hasattr(_ref_, '_date_min_'):
                _kwargs_['date_range_shared'] = (_ref_._date_min_, _ref_._date_max_)
            if self.p2s.SM_COUNT in self.sm_shared:
                _kwargs_['count_range_shared'] = (_ref_._count_min_, _ref_._count_max_)
            if self.p2s.SM_COLOR in self.sm_shared and _ref_._color_stat_min_ is not None:
                _kwargs_['color_stat_range_shared'] = (_ref_._color_stat_min_, _ref_._color_stat_max_)
        return {k: Timep(df=v, template=self, **_kwargs_) for k, v in df_lu.items()}

    def render_with(self, df, **overrides):
        return Timep(df=df, template=self, **overrides)

    def filterByRectangle(self, bounding_box, remove_records=False):
        _x0_, _y0_, _x1_, _y1_ = bounding_box
        if _x0_ > _x1_: _x0_, _x1_ = _x1_, _x0_
        if _y0_ > _y1_: _y0_, _y1_ = _y1_, _y0_
        _span_ = max(float(self._count_max_) - float(self._count_min_), 1.0)
        _join_ = 'anti' if remove_records else 'inner'

        if self._is_periodic_:
            # For stacked, sum counts per bin so bar height reflects total
            if self._agg_type_ == 'stacked':
                _agg_ = self.df_agg.group_by('__time_bin__').agg(pl.col('__count__').sum())
            else:
                _agg_ = self.df_agg
            _top_col_ = '__box_max__' if self._agg_type_ == 'boxplot' else '__count__'
            _selected_ = (
                _agg_
                .with_columns([
                    ((pl.col('__time_bin__').cast(pl.Float64) - self._bin_min_)
                     * self._bar_w_raw_ + self._plot_x0_).alias('__bx0__'),
                    ((pl.col('__time_bin__').cast(pl.Float64) - self._bin_min_ + 1)
                     * self._bar_w_raw_ + self._plot_x0_).alias('__bx1__'),
                    (self._plot_y1_ - self._plot_h_
                     * (pl.col(_top_col_).cast(pl.Float64) - float(self._count_min_))
                     / _span_).alias('__by0__'),
                    pl.lit(float(self._plot_y1_)).alias('__by1__'),
                ])
                .filter(
                    (pl.col('__bx0__') <= _x1_) & (pl.col('__bx1__') >= _x0_) &
                    (pl.col('__by0__') <= _y1_) & (pl.col('__by1__') >= _y0_)
                )
                .select('__time_bin__')
            )
            _df_result_ = self.df.join(_selected_, on='__time_bin__', how=_join_)
            return _df_result_.drop([c for c in ['__p2s_index__', '__time_bin__']
                                     if c in _df_result_.columns])

        else:  # linear
            _trunc_ = self.__linearTruncMap__()[self._time_enum_]
            _top_col_ = '__box_max__' if self._agg_type_ == 'boxplot' else '__count__'
            if self._agg_type_ == 'stacked':
                _sorted_bins_ = (self._all_stacked_bins_
                                 if hasattr(self, '_all_stacked_bins_')
                                 else sorted(self.df_agg[self._time_field_].unique().to_list()))
                _counts_ = (self.df_agg.group_by(self._time_field_)
                                       .agg(pl.col('__count__').sum()))
                _idx_df_ = pl.DataFrame({self._time_field_: _sorted_bins_,
                                         '__idx__': list(range(len(_sorted_bins_)))})
                _agg_indexed_ = _idx_df_.join(_counts_, on=self._time_field_, how='left').fill_null(0)
            else:
                _agg_indexed_ = self.df_agg.with_row_index('__idx__')
            _selected_ = (
                _agg_indexed_
                .with_columns([
                    (pl.col('__idx__').cast(pl.Float64) * self._bar_w_raw_
                     + self._plot_x0_).alias('__bx0__'),
                    ((pl.col('__idx__').cast(pl.Float64) + 1) * self._bar_w_raw_
                     + self._plot_x0_).alias('__bx1__'),
                    (self._plot_y1_ - self._plot_h_
                     * (pl.col(_top_col_).cast(pl.Float64) - float(self._count_min_))
                     / _span_).alias('__by0__'),
                    pl.lit(float(self._plot_y1_)).alias('__by1__'),
                ])
                .filter(
                    (pl.col('__bx0__') <= _x1_) & (pl.col('__bx1__') >= _x0_) &
                    (pl.col('__by0__') <= _y1_) & (pl.col('__by1__') >= _y0_)
                )
                .select(self._time_field_)
            )
            _df_with_bin_ = self.df.with_columns(
                pl.col(self._time_field_).dt.truncate(_trunc_).alias('__bin_key__')
            )
            _selected_renamed_ = _selected_.rename({self._time_field_: '__bin_key__'})
            _df_result_ = _df_with_bin_.join(_selected_renamed_, on='__bin_key__', how=_join_)
            return _df_result_.drop([c for c in ['__p2s_index__', '__bin_key__']
                                     if c in _df_result_.columns])

    def filterByOval(self, oval, remove_records=False):
        _cx_, _cy_, _rx_, _ry_ = oval
        # A plain click arrives as a zero-radius oval: keep it covering the pixel under the cursor.
        _rx_, _ry_ = max(float(_rx_), 0.5), max(float(_ry_), 0.5)
        _span_ = max(float(self._count_max_) - float(self._count_min_), 1.0)
        _join_ = 'anti' if remove_records else 'inner'

        # Exact ellipse-vs-bar (axis-aligned AABB) overlap test: clamp the oval center to the
        # bar's box, then check that closest point against the ellipse.
        _qx_ = pl.min_horizontal(pl.max_horizontal(pl.lit(_cx_), pl.col('__bx0__')), pl.col('__bx1__'))
        _qy_ = pl.min_horizontal(pl.max_horizontal(pl.lit(_cy_), pl.col('__by0__')), pl.col('__by1__'))
        _overlap_ = (((_qx_ - _cx_) / _rx_).pow(2) + ((_qy_ - _cy_) / _ry_).pow(2)) <= 1.0

        if self._is_periodic_:
            if self._agg_type_ == 'stacked':
                _agg_ = self.df_agg.group_by('__time_bin__').agg(pl.col('__count__').sum())
            else:
                _agg_ = self.df_agg
            _top_col_ = '__box_max__' if self._agg_type_ == 'boxplot' else '__count__'
            _selected_ = (
                _agg_
                .with_columns([
                    ((pl.col('__time_bin__').cast(pl.Float64) - self._bin_min_)
                     * self._bar_w_raw_ + self._plot_x0_).alias('__bx0__'),
                    ((pl.col('__time_bin__').cast(pl.Float64) - self._bin_min_ + 1)
                     * self._bar_w_raw_ + self._plot_x0_).alias('__bx1__'),
                    (self._plot_y1_ - self._plot_h_
                     * (pl.col(_top_col_).cast(pl.Float64) - float(self._count_min_))
                     / _span_).alias('__by0__'),
                    pl.lit(float(self._plot_y1_)).alias('__by1__'),
                ])
                .filter(_overlap_)
                .select('__time_bin__')
            )
            _df_result_ = self.df.join(_selected_, on='__time_bin__', how=_join_)
            return _df_result_.drop([c for c in ['__p2s_index__', '__time_bin__']
                                     if c in _df_result_.columns])

        else:  # linear
            _trunc_ = self.__linearTruncMap__()[self._time_enum_]
            _top_col_ = '__box_max__' if self._agg_type_ == 'boxplot' else '__count__'
            if self._agg_type_ == 'stacked':
                _sorted_bins_ = (self._all_stacked_bins_
                                 if hasattr(self, '_all_stacked_bins_')
                                 else sorted(self.df_agg[self._time_field_].unique().to_list()))
                _counts_ = (self.df_agg.group_by(self._time_field_)
                                       .agg(pl.col('__count__').sum()))
                _idx_df_ = pl.DataFrame({self._time_field_: _sorted_bins_,
                                         '__idx__': list(range(len(_sorted_bins_)))})
                _agg_indexed_ = _idx_df_.join(_counts_, on=self._time_field_, how='left').fill_null(0)
            else:
                _agg_indexed_ = self.df_agg.with_row_index('__idx__')
            _selected_ = (
                _agg_indexed_
                .with_columns([
                    (pl.col('__idx__').cast(pl.Float64) * self._bar_w_raw_
                     + self._plot_x0_).alias('__bx0__'),
                    ((pl.col('__idx__').cast(pl.Float64) + 1) * self._bar_w_raw_
                     + self._plot_x0_).alias('__bx1__'),
                    (self._plot_y1_ - self._plot_h_
                     * (pl.col(_top_col_).cast(pl.Float64) - float(self._count_min_))
                     / _span_).alias('__by0__'),
                    pl.lit(float(self._plot_y1_)).alias('__by1__'),
                ])
                .filter(_overlap_)
                .select(self._time_field_)
            )
            _df_with_bin_ = self.df.with_columns(
                pl.col(self._time_field_).dt.truncate(_trunc_).alias('__bin_key__')
            )
            _selected_renamed_ = _selected_.rename({self._time_field_: '__bin_key__'})
            _df_result_ = _df_with_bin_.join(_selected_renamed_, on='__bin_key__', how=_join_)
            return _df_result_.drop([c for c in ['__p2s_index__', '__bin_key__']
                                     if c in _df_result_.columns])

    def recordsAt(self, xy, shape=None, threshold=2.0):
        """Return the original records whose bar column contains pixel x.

        Only SELECT_VERTICALp is supported: the y coordinate and threshold are
        ignored.  Any x within a bin's rendered column selects all records
        belonging to that time bin.

        Parameters
        ----------
        xy        : (x, y) pixel coordinate
        shape     : must be p2s.SELECT_VERTICALp or None (default)
        threshold : accepted for API compatibility with XYp.recordsAt; unused
        """
        if shape is None: shape = self.p2s.SELECT_VERTICALp
        if shape != self.p2s.SELECT_VERTICALp:
            raise ValueError(f'Timep.recordsAt(): only SELECT_VERTICALp is supported, got {shape}')

        _x_, _y_ = xy

        # Map pixel x to a 0-based bin index.
        # Compute the offset first so that x values left of _plot_x0_ (negative
        # offset) yield _idx_ = -1 rather than 0 (int() truncates toward zero,
        # so int(-0.06) == 0, which would incorrectly match the first bin).
        _offset_ = _x_ - self._plot_x0_
        _idx_    = int(_offset_ / self._bar_w_raw_) if (self._bar_w_raw_ > 0 and _offset_ >= 0) else -1

        # Helper: return a correctly-schemed empty DataFrame
        def _empty_():
            _drop_ = [c for c in ['__p2s_index__', '__time_bin__'] if c in self.df.columns]
            return self.df.drop(_drop_).clear()

        if self._is_periodic_:
            _bin_value_ = self._bin_min_ + _idx_
            _pmin_, _pmax_ = self.p2s.timePeriodicRange(self._time_enum_)
            if _idx_ < 0 or _bin_value_ < _pmin_ or _bin_value_ > _pmax_:
                return _empty_()
            _selected_ = pl.DataFrame({'__time_bin__': [_bin_value_]},
                                       schema={'__time_bin__': pl.Int64})
            _df_result_ = self.df.join(_selected_, on='__time_bin__', how='inner')
            return _df_result_.drop([c for c in ['__p2s_index__', '__time_bin__']
                                     if c in _df_result_.columns])

        else:  # linear
            _trunc_ = self.__linearTruncMap__()[self._time_enum_]
            _bins_ = (sorted(self.df_agg[self._time_field_].unique().to_list())
                      if self._agg_type_ == 'stacked'
                      else self.df_agg[self._time_field_].to_list())
            if _idx_ < 0 or _idx_ >= len(_bins_):
                return self.df.drop([c for c in ['__p2s_index__'] if c in self.df.columns]).clear()
            _bin_ts_ = _bins_[_idx_]
            _df_with_bin_ = self.df.with_columns(
                pl.col(self._time_field_).dt.truncate(_trunc_).alias('__bin_key__')
            )
            _df_result_ = _df_with_bin_.filter(pl.col('__bin_key__') == _bin_ts_)
            return _df_result_.drop([c for c in ['__p2s_index__', '__bin_key__']
                                     if c in _df_result_.columns])
