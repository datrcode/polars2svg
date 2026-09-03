from typing import Any, TypedDict, Unpack, cast
import polars as pl
import random
import time

import polars2svg
from polars2svg.export import ExportMixin
from polars2svg.p2s_displaylist import (CLOUD_ICON_H, CLOUD_ICON_RX, CLOUD_ICON_W,
                                        cloudIconDef, pathToDL, roundedRectPoints,
                                        strokePolylineDL)

#
# Implementation of the following:
#
# Y.-H. Kuo, D. Liu, and K.-L. Ma, "SpreadLine: Visualizing Egocentric Dynamic
# Influence," IEEE Transactions on Visualization and Computer Graphics
# (Proc. IEEE VIS 2024), arXiv:2408.08992.
#
# Also based on the racetrack_svg_framework reference implementation
# (rt_spreadlines_mixin.py).
#
# Layout summary
# --------------
# Time flows left → right.  Each timestamp becomes a "bin" — a vertical column
# of packed circles.  A thick horizontal ego line connects all bin centers.
# Circles above the ego line are senders (alter-1 fm, alter-2 fm); circles below
# are receivers (alter-1 to, alter-2 to).
#
# Circles that are new in this bin get a red left-pointing triangle.
# Circles that do not appear after this bin get a blue right-pointing triangle.
# When circles cannot fit individually, they collapse into a cloud summarization.
#
# Same-entity circles in consecutive bins are joined by Bézier cross-connect
# curves.  Nodes that skip bins are routed through labelled channels drawn above
# (fm side) or below (to side) the main body.
#
# Zigzag dashed lines between bins show time periods where the ego is absent.
#


class SpreadLinesPKwargs(TypedDict, total=False):
    """Keyword arguments accepted by ``p2s.spreadlinesp()`` / ``SpreadLinesP(...)``.

    Every key is optional (``total=False``); the set is exactly SpreadLinesP._VALID_KWARGS,
    which is what the constructor validates at runtime -- a name not listed here
    raises TypeError.  Declaring it lets a type checker catch the misspelling
    before the call runs, and gives editors completion over the parameter set.

    Value types are deliberately conservative.  Most parameters are data-drivable
    (they take a literal *or* a column name *or* a ``(field, enum)`` spec), so they
    are typed ``Any`` rather than guessed at; the precise ones were each confirmed
    against how the test suite actually calls them.
    """
    alter_inter_d:           Any
    alter_separation_h:      Any
    anno:                    Any
    channel_inter_d:         Any
    circle_inter_d:          Any
    circle_spacer:           Any
    color_stat_range_shared: Any
    count:                   Any
    count_range_shared:      Any
    df:                      pl.DataFrame | None
    draw_border:             bool
    draw_context:            bool
    draw_labels:             bool
    ego:                     Any
    h_collapsed_sections:    Any
    highlight_nodes:         Any
    legend:                  Any
    max_bin_h:               Any
    max_bin_w:               Any
    max_channel_w:           Any
    max_rings:               Any
    min_channel_w:           Any
    node_color:              Any
    node_labels:             Any
    r_min:                   Any
    r_pref:                  Any
    relationships:           Any
    sm_shared:               set
    template:                'SpreadLinesP | None'
    time:                    Any
    time_order:              Any
    txt_h:                   Any
    wxh:                     Any
    x_ins:                   Any
    y_ins:                   Any


class SpreadLinesP(ExportMixin):

    _VALID_KWARGS = frozenset({
        'template', 'df',
        'relationships', 'ego', 'time',
        'node_color',             # None | p2s.COLOR_BY_NODE_NAME | dict{node→color} | field_name | '#rrggbb'
        'count',
        'anno',                   # {time_val: label} event annotations
        'time_order',             # explicit sorted timestamp list
        'max_rings',              # 1 or 2 (include 2-level alters)
        # bin geometry
        'r_min', 'r_pref',
        'circle_inter_d', 'circle_spacer',
        'alter_inter_d', 'alter_separation_h',
        'max_bin_w', 'max_bin_h',
        'h_collapsed_sections',
        # channel geometry
        'min_channel_w', 'max_channel_w', 'channel_inter_d',
        # output / labels
        'draw_labels',
        'node_labels',
        'wxh',
        'x_ins', 'y_ins',
        'sm_shared',
        'count_range_shared', 'color_stat_range_shared',
        'draw_border', 'draw_context', 'txt_h', 'legend',
        'highlight_nodes',            # set/frozenset of node names to visually highlight
    })

    # ---------------------------------------------------------------------
    # Parameters assigned onto the instance from _defaults_ by
    # Polars2SVG.assignScratchDefaults() -- setattr(), so no checker can see them.
    #
    # Declarations, not assignments: bare annotations populate __annotations__
    # and create no class attribute, so this block is a no-op at runtime.
    #
    # Types are deliberately conservative -- `Any` wherever a parameter is
    # data-drivable (accepts a literal *or* a column name), which is most of
    # them.  The job here is to make the attribute VISIBLE; the precise
    # per-parameter contract lands in the Unpack[TypedDict] work (phase 2),
    # which is where a caller-facing type belongs.
    #
    # tests/test_typing_surface.py::TestComponentAttrDeclarations checks this
    # block against _VALID_KWARGS, so a new parameter fails the suite until it
    # is declared here too.
    # ---------------------------------------------------------------------
    alter_inter_d:           int
    alter_separation_h:      int
    anno:                    dict
    channel_inter_d:         int
    circle_inter_d:          float
    circle_spacer:           int
    color_stat_range_shared: Any
    count:                   Any
    count_range_shared:      Any
    draw_border:             bool
    draw_context:            bool
    draw_labels:             bool
    ego:                     Any
    h_collapsed_sections:    int
    legend:                  bool
    max_bin_h:               int
    max_bin_w:               int
    max_channel_w:           int
    max_rings:               int
    min_channel_w:           int
    node_color:              Any
    node_labels:             Any
    r_min:                   float
    r_pref:                  float
    sm_shared:               set
    time_order:              Any
    txt_h:                   int
    x_ins:                   int
    y_ins:                   int

    # --- state built during __init__/render, not passed in --------------
    # Initialised to None and filled once the frame is resolved, so a checker
    # infers `... | None` and flags every later use.  `Any` because these hold
    # polars frames, display lists and cached statistics whose concrete types
    # this class does not otherwise name.
    _color_stat_max_: Any
    _color_stat_min_: Any
    _count_max_:      Any
    _count_min_:      Any
    _dl_body_:        Any
    _dl_legend_:      Any
    _gpu_dl_:         Any
    _gpu_payload_:    dict | None
    _legend_region_:  tuple | None
    _ts_enum_:        Any
    _ts_field_:       Any
    df:               Any
    df_orig:          pl.DataFrame | None
    legend_info:      Any
    template:         'SpreadLinesP | None'
    vx0:              Any
    vx1:              Any
    vy0:              Any
    vy1:              Any
    svg: str
    wxh: Any

    def __init__(self, *args: Any, **kwargs: Unpack[SpreadLinesPKwargs]) -> None:
        self.t_start        = time.time()
        self.p2s            = polars2svg.Polars2SVG()
        self.timing_metrics: dict = {}
        self.gatherMetrics(self.__parseInput__, *args, **kwargs)
        self.gatherMetrics(self.__validateInput__)
        if self.df is not None:
            rand_id = random.randint(0, 2**32)  # nosec B311 - non-cryptographic SVG id scoping, see SECURITY.md
            self.gatherMetrics(self.__calculateLayout__)
            self.gatherMetrics(self.__renderSVG__, rand_id)
        self.t_end     = time.time()
        self.t_overall = self.t_end - self.t_start

    def _repr_svg_(self) -> str: return self.svg

    #
    # gpuDisplayList() / webgpu() - WebGPU representation of the same render.
    #
    # The body is dual-recorded during __renderSVG__ (self._dl_body_), in world
    # coordinates, and mapped into canvas pixels there once the viewBox is known --
    # so the GPU primitives come from the same numbers the SVG strings do rather than
    # from re-parsing the finished document.  The legend is recorded separately in
    # screen pixels (it is drawn at an inverse scale in the SVG so the root viewBox
    # lands it at true pixel size) and is spliced on top here.
    #
    # Lazy + cached; invalidated by __renderSVG__.
    #
    def gpuDisplayList(self) -> Any:
        if self.df is None or getattr(self, 'svg', None) is None: return None
        if getattr(self, '_gpu_dl_', None) is not None: return self._gpu_dl_
        _body_ = getattr(self, '_dl_body_', None)
        if _body_ is None: return None
        from polars2svg.p2s_displaylist import DisplayList
        w, h = self.wxh
        _dl_ = DisplayList(w, h, bg=self.p2s.colorTyped('background', 'default'))
        _dl_.extend(_body_)
        if getattr(self, '_dl_legend_', None) is not None: _dl_.extend(self._dl_legend_)
        self._gpu_dl_ = _dl_
        return _dl_

    def webgpu(self) -> dict | None:
        if getattr(self, '_gpu_payload_', None) is not None: return self._gpu_payload_
        _dl_ = self.gpuDisplayList()
        if _dl_ is None: return None
        self._gpu_payload_ = _dl_.webgpu_payload(self.p2s.glyphAtlas())
        return self._gpu_payload_

    def gatherMetrics(self, callable: Any, *args: Any, **kwargs: Any) -> int:
        t0 = time.time()
        _results_ = callable(*args, **kwargs)
        t1 = time.time()
        if callable.__name__ not in self.timing_metrics:
            self.timing_metrics[callable.__name__] = 0.0
        self.timing_metrics[callable.__name__] += t1 - t0
        return _results_

    # -------------------------------------------------------------------------
    # __parseInput__
    # -------------------------------------------------------------------------

    def __parseInput__(self, *args: Any, **kwargs: Unpack[SpreadLinesPKwargs]) -> None:
        _unknown_ = set(kwargs) - self._VALID_KWARGS
        if _unknown_:
            raise TypeError(f'SpreadLinesP: unexpected keyword argument(s): {sorted(_unknown_)}')

        # Single source of truth for every parameter (name -> from-scratch default);
        # drives both the from-scratch assignment and the keyword-override copy below.
        _defaults_: dict = {
            'relationships':          None,
            'ego':                    None,
            'time':                   None,   # str | (str, TimeLinearTypeP)
            'node_color':             None,
            'count':                  self.p2s.ROW_COUNTp,
            'anno':                   {},
            'time_order':             None,
            'max_rings':              2,
            # bin geometry (matches racetrack defaults)
            'r_min':                  4.0,
            'r_pref':                 7.0,
            'circle_inter_d':         2.0,
            'circle_spacer':          3,
            'alter_inter_d':          96,
            'alter_separation_h':     32,
            'max_bin_w':              64,
            'max_bin_h':              400,
            'h_collapsed_sections':   16,
            # channel geometry
            'min_channel_w':          8,
            'max_channel_w':          16,
            'channel_inter_d':        4,
            # output
            'draw_labels':            False,
            'node_labels':            None,
            'wxh':                    (800, 400),
            'x_ins':                  32,
            'y_ins':                  8,
            'sm_shared':              set(),
            'count_range_shared':     None,
            'color_stat_range_shared': None,
            'draw_border':            True,
            'draw_context':           True,
            'txt_h':                  12,
            'legend':                 False,
            'highlight_nodes':        frozenset(),
        }
        self.p2s.assertParamSpecMatches('SpreadLinesP', self._VALID_KWARGS, _defaults_)

        self.df, self.df_orig = None, None

        self.template = None
        for i in range(len(args)):
            if isinstance(args[i], SpreadLinesP): self.template = args[i]
        if 'template' in kwargs: self.template = kwargs['template']
        if self.template is not None:
            _tc_ = self.template
            self.p2s._clone_template_state(self, _tc_)
            self.template         = _tc_
            self._count_min_      = None
            self._count_max_      = None
            self._color_stat_min_ = None
            self._color_stat_max_ = None
        else:
            self.p2s.assignScratchDefaults(self, _defaults_)
            # Internal (non-parameter) state — not part of the kwarg spec
            self._ts_field_       = None   # always the plain column name
            self._ts_enum_        = None   # TimeLinearTypeP or None (→ auto)
            self._count_min_      = None
            self._count_max_      = None
            self._color_stat_min_ = None
            self._color_stat_max_ = None
            # from-scratch builds only — a template clone is an exact snapshot and
            # must not re-apply session defaults (see Polars2SVG._apply_defaults)
            kwargs = cast(SpreadLinesPKwargs, self.p2s._apply_defaults('spreadlinesp', kwargs))

        # DataFrame
        _new_df_ = None
        for _arg_ in args:
            if isinstance(_arg_, pl.DataFrame):
                if _new_df_ is None: _new_df_ = _arg_
                else: raise ValueError('SpreadLinesP: df already set')
        if 'df' in kwargs:
            if _new_df_ is None: _new_df_ = kwargs['df']
            else: raise ValueError('SpreadLinesP: df already set')
        if _new_df_ is not None:
            self.df = self.df_orig = _new_df_

        # Relationships from positional
        for _arg_ in args:
            if   isinstance(_arg_, pl.DataFrame):  pass
            elif isinstance(_arg_, SpreadLinesP):   pass
            elif (isinstance(_arg_, list) and len(_arg_) > 0
                  and all(isinstance(t, tuple) and len(t) >= 2 for t in _arg_)):
                if 'relationships' not in kwargs:
                    self.relationships = _arg_
            else:
                raise ValueError(f'SpreadLinesP: unrecognised positional arg {type(_arg_).__name__}')

        # time (parses TField/tuple forms) and highlight_nodes (frozenset coercion)
        # carry special handling, so they are skipped by the spec-driven copy.
        self.p2s.assignKwargOverrides(self, _defaults_, kwargs, skip={'time', 'highlight_nodes'})
        if 'time' in kwargs:
            _time_val_ = kwargs['time']
            if isinstance(_time_val_, self.p2s.TField):
                self.time      = _time_val_.column      # field name
                self._ts_enum_ = _time_val_.transform    # explicit TimeLinearTypeP
            elif isinstance(_time_val_, tuple) and len(_time_val_) == 2:
                self.time      = _time_val_[0]   # field name
                self._ts_enum_ = _time_val_[1]   # explicit TimeLinearTypeP
            else:
                self.time      = _time_val_       # field name only
                self._ts_enum_ = None             # will auto-resolve if date/datetime
        self._ts_field_ = self.time            # always a plain column name
        if 'highlight_nodes'         in kwargs:
            _hn_ = kwargs['highlight_nodes']
            self.highlight_nodes = frozenset(str(n) for n in _hn_) if _hn_ else frozenset()

        # SpreadLinesP has no per-node/per-entity labeling (unlike linkp/chordp/piep) --
        # the circle-packing layout doesn't reserve space for it. draw_context covers
        # the one label this component does draw (the timestamp row along the bottom).
        if self.draw_labels:
            raise NotImplementedError(
                'SpreadLinesP: draw_labels is not implemented (no per-node label layout). '
                'Use draw_context to control the timestamp row instead.')

        # "No data" placeholder for early error visibility -- only ever seen when
        # no df is supplied (a successful render overwrites self.svg); makes a
        # dropped-df plumbing mistake visible instead of a silently blank canvas.
        self.wxh = self.p2s.normalizeWxh(self.wxh, 'SpreadLinesP')
        w, h = self.wxh
        self.svg = self.p2s.placeholderSVG(w, h)

        if self.df is None: return

        self.df = self.df.clone()
        if '__p2s_index__' not in self.df.columns:
            self.df = self.df.with_row_index('__p2s_index__')

        # Expand compound field specs
        self.relationships_orig = self.relationships
        self.relationships, _i_ = [], 0
        for _edge_ in self.relationships_orig:
            _fm_, _to_ = _edge_[0], _edge_[1]
            new_fm, new_to = _fm_, _to_
            if isinstance(_fm_, tuple):
                new_fm = f'__fm{_i_}__'
                self.df = self._createConcatColumn_(self.df, _fm_, new_fm)
            if isinstance(_to_, tuple):
                new_to = f'__to{_i_}__'
                self.df = self._createConcatColumn_(self.df, _to_, new_to)
            if   len(_edge_) == 2: self.relationships.append((new_fm, new_to))
            elif len(_edge_) == 3: self.relationships.append((new_fm, new_to, _edge_[2]))
            else: raise ValueError(f'SpreadLinesP: relationship tuple bad length: {_edge_!r}')
            _i_ += 1

    def _createConcatColumn_(self, df: pl.DataFrame, fields: tuple, new_col: str) -> pl.DataFrame:
        _parts_ = []
        for i, f in enumerate(fields):
            if i > 0: _parts_.append(pl.lit('|'))
            _parts_.append(pl.col(f).cast(pl.String))
        return df.with_columns(pl.concat_str(_parts_).alias(new_col))

    def __countAggExpr__(self) -> pl.Expr:
        if self.count == self.p2s.ROW_COUNTp:
            return pl.len().alias('__count__')
        elif isinstance(self.count, str):
            _is_num_ = self.p2s.numericColumn(self.df, self.count)
            self.p2s.logDtypeKeyedCount('SpreadLinesP', self.count, _is_num_)
            if _is_num_:
                return pl.col(self.count).sum().alias('__count__')
            else:
                return pl.col(self.count).n_unique().alias('__count__')
        elif isinstance(self.count, tuple):
            _fields_ = [_f_ for _f_ in self.count if isinstance(_f_, str)]
            if self.p2s.SETp in self.count:  return pl.col(_fields_[0]).n_unique().alias('__count__')
            elif len(_fields_) == 1:         return pl.col(_fields_[0]).sum().alias('__count__')
            else:                            return pl.struct(_fields_).n_unique().alias('__count__')
        return pl.len().alias('__count__')

    def __countFields__(self) -> set:
        if self.count == self.p2s.ROW_COUNTp: return set()
        if isinstance(self.count, str):        return {self.count}
        if isinstance(self.count, tuple):      return {_f_ for _f_ in self.count if isinstance(_f_, str)}
        return set()

    # -------------------------------------------------------------------------
    # __validateInput__
    # -------------------------------------------------------------------------

    def __validateInput__(self) -> None:
        # Normalize legend= eagerly so a bad spec fails fast (raises InvalidSpecError).
        self.legend_spec = self.p2s.legendResolveSpec(self.legend)
        if self.df is None: return
        self.p2s.checkReservedColumns(self.df, 'SpreadLinesP')
        if self.relationships is None or len(self.relationships) == 0:
            raise ValueError('SpreadLinesP: relationships must be specified')
        if self.ego is None:
            raise ValueError('SpreadLinesP: ego must be specified')
        if self._ts_field_ is None:
            raise ValueError('SpreadLinesP: time (timestamp column) must be specified')
        for _rel_ in self.relationships:
            for _field_ in _rel_[:2]:
                if _field_ not in self.df.columns:
                    raise ValueError(f'SpreadLinesP: field "{_field_}" not in DataFrame')
        if self._ts_field_ not in self.df.columns:
            raise ValueError(f'SpreadLinesP: time field "{self._ts_field_}" not in DataFrame')
        if self._ts_enum_ is not None and not isinstance(self._ts_enum_, self.p2s.TimeLinearTypeP):
            raise ValueError(f'SpreadLinesP: time enum must be a TimeLinearTypeP, got {type(self._ts_enum_).__name__}')

    # -------------------------------------------------------------------------
    # Temporal binning helpers  (mirror timep's approach, tailored for bins)
    # -------------------------------------------------------------------------

    def __linearTruncMap__(self) -> dict:
        """TimeLinearTypeP → dt.truncate()-compatible interval string."""
        p = self.p2s
        return {
            p.LT_Yp:              '1y',
            p.LT_Y_Qp:            '3mo',   # dt.truncate doesn't accept '1q'
            p.LT_Y_mp:            '1mo',
            p.LT_Y_m_dp:          '1d',
            p.LT_Y_m_d_4Hp:       '4h',
            p.LT_Y_m_d_Hp:        '1h',
            p.LT_Y_m_d_H_15Mp:    '15m',
            p.LT_Y_m_d_H_Mp:      '1m',
            p.LT_Y_m_d_H_M_15Sp:  '15s',
            p.LT_Y_m_d_H_M_Sp:    '1s',
        }

    def __linearEnumOrder__(self) -> list:
        """Return enums coarsest → finest."""
        p = self.p2s
        return [
            p.LT_Yp, p.LT_Y_Qp, p.LT_Y_mp,
            p.LT_Y_m_dp,
            p.LT_Y_m_d_4Hp, p.LT_Y_m_d_Hp,
            p.LT_Y_m_d_H_15Mp, p.LT_Y_m_d_H_Mp,
            p.LT_Y_m_d_H_M_15Sp, p.LT_Y_m_d_H_M_Sp,
        ]

    def __dataGranularityCap__(self) -> Any:
        """Return the finest TimeLinearTypeP allowed by actual data precision."""
        p = self.p2s
        if p.dateColumn(self.df, self._ts_field_):
            return p.LT_Y_m_dp          # Date columns: daily is the finest
        if len(self.df) == 0:
            return p.LT_Y_m_dp
        _s_ = self.df.select([
            pl.col(self._ts_field_).dt.hour()  .n_unique().alias('nh'),
            pl.col(self._ts_field_).dt.minute().n_unique().alias('nm'),
            pl.col(self._ts_field_).dt.second().n_unique().alias('ns'),
            pl.col(self._ts_field_).dt.hour()  .min()     .alias('h0'),
            pl.col(self._ts_field_).dt.minute().min()     .alias('m0'),
            pl.col(self._ts_field_).dt.second().min()     .alias('s0'),
        ]).row(0, named=True)
        all_midnight = _s_['nh'] == 1 and _s_['h0'] == 0
        all_on_hour  = _s_['nm'] == 1 and _s_['m0'] == 0
        all_on_min   = _s_['ns'] == 1 and _s_['s0'] == 0
        if all_midnight and all_on_hour and all_on_min: return p.LT_Y_m_dp
        if                  all_on_hour and all_on_min: return p.LT_Y_m_d_Hp
        if                               all_on_min:    return p.LT_Y_m_d_H_Mp
        return p.LT_Y_m_d_H_M_Sp

    def __autoResolveLinearEnum__(self) -> Any:
        """
        Pick the finest granularity where ego-containing bins fit within the
        canvas without overcrowding.

        Unlike timep (which sizes to plot_width // 2), SpreadLinesP uses
        canvas_width / alter_inter_d as its bin budget — each bin column
        needs alter_inter_d pixels of horizontal space.

        Iterates coarse → fine and keeps selecting as long as:
          - the ego appears in ≥ 2 bins at this granularity (enough to show
            temporal change), AND
          - the total spine (min→max / interval) ≤ max_bins (nothing off-screen)

        Falls back to the coarsest enum that produces ≥ 2 ego bins.
        """
        p            = self.p2s
        w, _h_       = self.wxh
        max_bins     = max(2, int(w / self.alter_inter_d))
        trunc_map    = self.__linearTruncMap__()
        enum_order   = self.__linearEnumOrder__()
        cap_enum     = self.__dataGranularityCap__()
        cap_idx      = enum_order.index(cap_enum)

        # Rows that involve the ego — only these matter for bin presence
        _ego_dfs_ = []
        for _rel_ in self.relationships:
            _fm_col_, _to_col_ = _rel_[0], _rel_[1]
            _ego_dfs_.append(
                self.df.filter(
                    pl.col(_fm_col_).cast(pl.String).is_in(self.node_focus) |
                    pl.col(_to_col_).cast(pl.String).is_in(self.node_focus)
                ).select(self._ts_field_)
            )
        if not _ego_dfs_:
            return p.LT_Y_m_dp
        _ego_ts_ = pl.concat(_ego_dfs_)
        if len(_ego_ts_) == 0:
            return p.LT_Y_m_dp

        _selected_ = None
        _fallback_  = None   # coarsest enum with ≥ 2 ego bins

        for _i_, _enum_ in enumerate(enum_order):
            if _i_ > cap_idx:
                break
            _trunc_ = trunc_map[_enum_]
            try:
                _n_ = (_ego_ts_
                       .select(pl.col(self._ts_field_).dt.truncate(_trunc_).n_unique())
                       .item())
            except Exception:  # nosec B112 - best-effort truncation-granularity probe; a failure here just means this granularity is skipped, not silently masking a real error path
                continue

            if _n_ < 1:
                continue

            # Record the coarsest granularity that gives ≥ 2 ego-present bins
            if _n_ >= 2 and _fallback_ is None:
                _fallback_ = _enum_

            # Unlike timep, SpreadLinesP only renders ego-present bins as columns
            # (absent timestamps become zigzag discontinuities), so the canvas
            # budget is based on ego-present bin count, not the full time spine.
            if 2 <= _n_ <= max_bins:
                _selected_ = _enum_   # keeps updating toward finest that fits

        return _selected_ or _fallback_ or p.LT_Y_m_dp

    # -------------------------------------------------------------------------
    # __calculateLayout__  — build per-bin alter sets
    # -------------------------------------------------------------------------

    def __calculateLayout__(self) -> None:
        # ── Normalise ego to a frozenset of strings ────────────────────────────
        if isinstance(self.ego, (list, set)):
            self.node_focus = frozenset(str(n) for n in self.ego)
        else:
            self.node_focus = frozenset({str(self.ego)})
        self.ego_is_set = isinstance(self.ego, (list, set)) and len(self.node_focus) > 1

        # ── Auto-resolve granularity for date/datetime ts columns ─────────────
        # Only runs when ts is a date/datetime column and no enum was supplied.
        # Stores the resolved enum back on self so template reuse preserves it.
        _is_temporal_ = (self.p2s.dateColumn(self.df, self._ts_field_) or
                         self.p2s.dateTimeColumn(self.df, self._ts_field_))
        if _is_temporal_ and self._ts_enum_ is None:
            self._ts_enum_ = self.__autoResolveLinearEnum__()

        # Derive the truncation string (None when ts column is already strings)
        _ts_trunc_ = (self.__linearTruncMap__()[self._ts_enum_]
                      if self._ts_enum_ is not None else None)

        # Label-length map: how many characters of the ISO string to show
        _label_len_map_ = {
            self.p2s.LT_Yp:              4,    # 2023
            self.p2s.LT_Y_Qp:            7,    # 2023-01
            self.p2s.LT_Y_mp:            7,    # 2023-01
            self.p2s.LT_Y_m_dp:         10,    # 2023-01-15
            self.p2s.LT_Y_m_d_4Hp:      13,    # 2023-01-15 04
            self.p2s.LT_Y_m_d_Hp:       13,    # 2023-01-15 04
            self.p2s.LT_Y_m_d_H_15Mp:   16,    # 2023-01-15 04:00
            self.p2s.LT_Y_m_d_H_Mp:     16,    # 2023-01-15 04:00
            self.p2s.LT_Y_m_d_H_M_15Sp: 19,    # 2023-01-15 04:00:00
            self.p2s.LT_Y_m_d_H_M_Sp:   19,    # 2023-01-15 04:00:00
        }
        self._ts_label_len_ = (_label_len_map_.get(self._ts_enum_, 16)
                               if self._ts_enum_ is not None else 10)

        # ── Aggregate edges per (fm, to, ts-bin) ──────────────────────────────
        # Truncate before group_by so rows in the same bin are merged naturally.
        _ts_col_   = self._ts_field_
        _ts_agg_col_ = '__ts_bin__' if _ts_trunc_ is not None else _ts_col_
        if _ts_trunc_ is not None:
            _df_src_ = self.df.with_columns(
                pl.col(_ts_col_).dt.truncate(_ts_trunc_).alias(_ts_agg_col_)
            )
        else:
            _df_src_ = self.df

        _agg_dfs_ = []
        for _rel_ in self.relationships:
            _fm_col_, _to_col_ = _rel_[0], _rel_[1]
            _count_agg_ = self.__countAggExpr__()
            _df_ = (_df_src_
                    .group_by([_fm_col_, _to_col_, _ts_agg_col_])
                    .agg(_count_agg_)
                    .rename({_fm_col_: '__fm__', _to_col_: '__to__', _ts_agg_col_: '__ts__'}))
            _agg_dfs_.append(_df_)
        _df_all_ = pl.concat(_agg_dfs_).with_columns(
            pl.col('__fm__').cast(pl.String),
            pl.col('__to__').cast(pl.String),
            pl.col('__ts__').cast(pl.String),
        )

        # ── Collapse set-ego into a single virtual node ───────────────────────
        # When a set of nodes is the ego, replace each member with the sentinel
        # '__EGO__' in both edge endpoints, drop intra-ego self-loops, and
        # re-aggregate counts so that parallel edges (A→X and B→X) merge.
        if self.ego_is_set:
            _VIRTUAL_EGO_ = '__EGO__'
            _df_all_ = (
                _df_all_
                .with_columns([
                    pl.when(pl.col('__fm__').is_in(self.node_focus))
                      .then(pl.lit(_VIRTUAL_EGO_))
                      .otherwise(pl.col('__fm__'))
                      .alias('__fm__'),
                    pl.when(pl.col('__to__').is_in(self.node_focus))
                      .then(pl.lit(_VIRTUAL_EGO_))
                      .otherwise(pl.col('__to__'))
                      .alias('__to__'),
                ])
                .filter(~((pl.col('__fm__') == _VIRTUAL_EGO_) & (pl.col('__to__') == _VIRTUAL_EGO_)))
                .group_by(['__fm__', '__to__', '__ts__'])
                .agg(pl.col('__count__').sum())
            )
            self.node_focus = frozenset({_VIRTUAL_EGO_})

        # ── Sorted timestamps ──────────────────────────────────────────────────
        if self.time_order is not None:
            _ts_list_ = [str(t) for t in self.time_order]
        else:
            _ts_list_ = sorted(_df_all_['__ts__'].unique().to_list())
        self.ts_list = _ts_list_

        # Pre-partition _df_all_ by timestamp: one pass builds a dict so each
        # per-bin access is an O(1) lookup instead of an O(|_df_all_|) scan.
        _ts_map_ = {}
        for _sub_ in _df_all_.partition_by('__ts__', maintain_order=False):
            _ts_map_[str(_sub_['__ts__'][0])] = _sub_

        # Identify timestamps where the ego has at least one non-self edge.
        # This is the set we actually need to iterate over; all others are
        # discontinuities.  One vectorized filter replaces O(N_ts) per-row
        # Python filter calls in the loop below.
        _ego_present_ts_ = set(
            _df_all_.filter(
                (pl.col('__fm__').is_in(self.node_focus) & ~pl.col('__to__').is_in(self.node_focus)) |
                (pl.col('__to__').is_in(self.node_focus) & ~pl.col('__fm__').is_in(self.node_focus))
            )['__ts__'].to_list()
        )

        # Build an index so gap sizes between ego-present bins can be computed
        # without iterating the full _ts_list_.
        _ts_index_ = {ts: i for i, ts in enumerate(_ts_list_)}
        # Only the ego-present entries need to be visited, in sorted order.
        _ego_ts_ordered_ = [ts for ts in _ts_list_ if ts in _ego_present_ts_]

        # ── Bin data structures ────────────────────────────────────────────────
        self.bin_to_timestamps             = {}   # bin_idx → ts_str
        self.bin_to_alter1s: dict                = {}   # bin_idx → {'fm': set, 'to': set}
        self.bin_to_alter2s: dict                = {}   # bin_idx → {'fm': set, 'to': set}
        self.bin_to_focal_nodes_present: dict    = {}   # bin_idx → set of ego nodes present
        self.discontinuity_count_after_bin: dict = {}   # bin_idx → count of absent timestamps
        self.bin_to_node_weights           = {}   # bin_idx → {node_str: weight}  (empty when count=ROW_COUNTp)

        _bin_        = 0
        _prev_idx_   = -1   # position in _ts_list_ of the most recent ego bin

        for _ts_str_ in _ego_ts_ordered_:
            _current_idx_ = _ts_index_[_ts_str_]

            # Count how many timestamps were skipped since the previous ego bin.
            if _bin_ > 0:
                _gap_ = _current_idx_ - _prev_idx_ - 1
                if _gap_ > 0:
                    _prev_ = _bin_ - 1
                    self.discontinuity_count_after_bin[_prev_] = (
                        self.discontinuity_count_after_bin.get(_prev_, 0) + _gap_
                    )

            _k_df_ = _ts_map_.get(_ts_str_)
            if _k_df_ is None:
                # time_order entry absent from data — skip (counted via gap above)
                _prev_idx_ = _current_idx_
                continue

            # Edges where ego is the FROM (ego sends → alter-1 'to')
            _fm_is_focus_ = _k_df_.filter(
                pl.col('__fm__').is_in(self.node_focus) &
                ~pl.col('__to__').is_in(self.node_focus)
            )
            # Edges where ego is the TO (alter → ego → alter-1 'fm')
            _to_is_focus_ = _k_df_.filter(
                pl.col('__to__').is_in(self.node_focus) &
                ~pl.col('__fm__').is_in(self.node_focus)
            )
            # Self-edges within ego set
            _fm_to_conn_ = _k_df_.filter(
                pl.col('__to__').is_in(self.node_focus) &
                pl.col('__fm__').is_in(self.node_focus)
            )

            self.bin_to_timestamps         [_bin_] = _ts_str_
            self.bin_to_alter1s            [_bin_] = {'fm': set(), 'to': set()}
            self.bin_to_alter2s            [_bin_] = {'fm': set(), 'to': set()}
            self.bin_to_focal_nodes_present[_bin_] = set()

            # Alter-1 'to' side: ego → these nodes
            if len(_fm_is_focus_) > 0:
                _to_set_ = set(_fm_is_focus_['__to__'].to_list())
                self.bin_to_alter1s[_bin_]['to'] |= _to_set_
                if self.max_rings >= 2:
                    _a2_ = _k_df_.filter(
                        (pl.col('__to__').is_in(_to_set_) | pl.col('__fm__').is_in(_to_set_)) &
                        ~pl.col('__fm__').is_in(self.node_focus) &
                        ~pl.col('__to__').is_in(self.node_focus)
                    )
                    self.bin_to_alter2s[_bin_]['to'] |= (
                        (set(_a2_['__fm__'].to_list()) | set(_a2_['__to__'].to_list())) - _to_set_
                    )

            # Alter-1 'fm' side: these nodes → ego
            if len(_to_is_focus_) > 0:
                _fm_set_ = set(_to_is_focus_['__fm__'].to_list())
                self.bin_to_alter1s[_bin_]['fm'] |= _fm_set_
                if self.max_rings >= 2:
                    _a2_ = _k_df_.filter(
                        (pl.col('__to__').is_in(_fm_set_) | pl.col('__fm__').is_in(_fm_set_)) &
                        ~pl.col('__fm__').is_in(self.node_focus) &
                        ~pl.col('__to__').is_in(self.node_focus)
                    )
                    self.bin_to_alter2s[_bin_]['fm'] |= (
                        (set(_a2_['__fm__'].to_list()) | set(_a2_['__to__'].to_list())) - _fm_set_
                    )

            # Focal nodes present in this bin
            self.bin_to_focal_nodes_present[_bin_] |= (
                set(_fm_is_focus_['__fm__'].to_list()) |
                set(_to_is_focus_['__to__'].to_list()) |
                set(_fm_to_conn_['__fm__'].to_list()) |
                set(_fm_to_conn_['__to__'].to_list())
            )

            # Per-node edge-weight lookup for count-based sort (skipped for ROW_COUNTp)
            if self.count != self.p2s.ROW_COUNTp:
                _wfm_ = (_k_df_.group_by('__fm__')
                                .agg(pl.col('__count__').sum())
                                .rename({'__fm__': '_nd_'}))
                _wto_ = (_k_df_.group_by('__to__')
                                .agg(pl.col('__count__').sum())
                                .rename({'__to__': '_nd_'}))
                _wall_ = (pl.concat([_wfm_, _wto_])
                            .group_by('_nd_')
                            .agg(pl.col('__count__').sum()))
                self.bin_to_node_weights[_bin_] = dict(
                    zip(_wall_['_nd_'].to_list(), _wall_['__count__'].to_list())
                )

            _prev_idx_ = _current_idx_
            _bin_ += 1

        # ── Deduplicate alters (fm side wins for bidirectionals) ───────────────
        for _b_ in self.bin_to_alter1s:
            a1fm = self.bin_to_alter1s[_b_]['fm']
            a1to = self.bin_to_alter1s[_b_]['to']
            a2fm = self.bin_to_alter2s[_b_]['fm']
            a2to = self.bin_to_alter2s[_b_]['to']
            a1to -= a1fm
            a2fm -= (a1fm | a1to)
            a2to -= (a1fm | a1to | a2fm)
            a1fm -= self.node_focus
            a1to -= self.node_focus
            a2fm -= self.node_focus
            a2to -= self.node_focus

        # ── Pre-build node color lookup ────────────────────────────────────────
        # Resolved here in batch (p2s.colors() is one Polars pass) so __nodeColor__
        # never falls back to per-node 1-row p2s.color() collects during render.
        self._node_color_lu_ = {}
        if (isinstance(self.node_color, str) and
                not isinstance(self.node_color, self.p2s.HexColorString) and
                self.node_color in self.df.columns):
            _nc_fld_ = self.node_color
            for _rel_ in self.relationships:
                for _col_ in (_rel_[0], _rel_[1]):
                    _df_nv_      = self.df.group_by(_col_).agg(pl.col(_nc_fld_).first())
                    _vals_       = [str(v) for v in _df_nv_[_nc_fld_].to_list()]
                    _val_colors_ = self.p2s.colors(_vals_)
                    for _nm_, _val_ in zip(_df_nv_[_col_].to_list(), _vals_):
                        self._node_color_lu_[str(_nm_)] = _val_colors_[_val_]
        elif self.node_color is None or self.node_color == self.p2s.COLOR_BY_NODE_NAME:
            # Default mode colors each node by its name — batch all names seen in
            # the bins (plus the focal nodes) into a single colorize pass.
            _all_names_ = {str(_n_) for _n_ in self.node_focus}
            for _b_ in self.bin_to_alter1s:
                _all_names_ |= {str(_n_) for _n_ in self._nodesInBin_(_b_)}
            for _b_ in self.bin_to_focal_nodes_present:
                _all_names_ |= {str(_n_) for _n_ in self.bin_to_focal_nodes_present[_b_]}
            self._node_color_lu_ = self.p2s.colors(_all_names_)

    # -------------------------------------------------------------------------
    # Helpers: node count in bin, node existence across bins
    # -------------------------------------------------------------------------

    def _nodesInBin_(self, b: int) -> set:
        s = set()
        if b in self.bin_to_alter1s:
            s |= self.bin_to_alter1s[b].get('fm', set())
            s |= self.bin_to_alter1s[b].get('to', set())
        if b in self.bin_to_alter2s:
            s |= self.bin_to_alter2s[b].get('fm', set())
            s |= self.bin_to_alter2s[b].get('to', set())
        return s

    def _nodesExistInOtherBins_(self, b: Any) -> set:
        me = self._nodesInBin_(b)
        others = set()
        for ob in list(self.bin_to_alter1s.keys()) + list(self.bin_to_alter2s.keys()):
            if ob != b:
                others |= self._nodesInBin_(ob)
        return me & others

    # -------------------------------------------------------------------------
    # Node color
    # -------------------------------------------------------------------------

    def __nodeColor__(self, node: str) -> str:
        _ns_ = str(node)
        if self.node_color is None or self.node_color == self.p2s.COLOR_BY_NODE_NAME:
            _c_ = self._node_color_lu_.get(_ns_)
            return _c_ if _c_ is not None else self.p2s.color(_ns_)
        elif isinstance(self.node_color, self.p2s.HexColorString):
            return self.node_color
        elif isinstance(self.node_color, dict):
            _v_ = self.node_color.get(_ns_)
            if _v_ is None: return self.p2s.colorTyped('axis', 'default')
            return _v_ if isinstance(_v_, self.p2s.HexColorString) else self.p2s.color(str(_v_))
        elif _ns_ in self._node_color_lu_:
            return self._node_color_lu_[_ns_]
        else:
            return self.p2s.color(_ns_)

    # -------------------------------------------------------------------------
    # packable() — fit node circles into available vertical space
    #   mul=-1 grows upward from y, mul=+1 grows downward from y
    #   Returns (node_to_xy, left_overs, out_of) or (None, None, None)
    # -------------------------------------------------------------------------

    def packable(self, nodes: list, x: float, y: float, y_max: float, w_max: int, mul: int,
                 r_min: float, r_pref: float, circle_inter_d: float, circle_spacer: int) -> tuple:
        node_to_xy = {}
        h          = abs(y - y_max)
        n          = len(nodes)
        left_overs = 0
        out_of     = n
        if n == 0:
            return None, None, None

        # Single-strand attempt
        r = (h - (n - 1) * circle_inter_d) / n / 2.0
        if r >= r_min:
            r = min(r, r_pref)
            for _ni_ in range(n):
                _node_             = nodes[-(  _ni_ + 1)]
                node_to_xy[_node_] = (x, y + mul * r, r)
                y                 += mul * (2 * r + circle_inter_d)
        else:
            # Multi-strand attempt
            m_max = w_max / (2 * r_min + circle_spacer)
            for m in range(2, int(m_max) + 1):
                r = (h - (n // m) * circle_inter_d) / (n // m) / 2.0
                if r < r_min: continue
                r    = min(r, r_pref)
                tw   = m * (2 * r) + (m - 1) * circle_spacer
                if tw > w_max: continue
                nodes_per_col = n // m
                left_overs    = n - nodes_per_col * m
                out_of        = nodes_per_col
                if left_overs > 0: m += 1
                tw = m * (2 * r) + (m - 1) * circle_spacer
                _cols_: list
                _col_: list
                _cols_, _col_ = [], []
                for _node_ in nodes:
                    col_i = len(_col_)
                    _x_c_ = x - tw / 2.0 + len(_cols_) * (2 * r + circle_spacer) + r
                    _y_r_ = y + mul * r + mul * col_i * (2 * r + circle_inter_d)
                    _col_.append((_x_c_, _y_r_))
                    if len(_col_) >= nodes_per_col:
                        _cols_.append(_col_); _col_ = []
                if _col_: _cols_.append(_col_)
                _xi_, _yi_ = 0, 0
                for _ni_ in range(n):
                    _node_ = nodes[n - 1 - _ni_] if mul == -1 else nodes[_ni_]
                    if _yi_ >= len(_cols_[_xi_]): _yi_, _xi_ = _yi_ + 1, 0
                    _xy_               = _cols_[_xi_][_yi_]
                    node_to_xy[_node_] = (_xy_[0], _xy_[1], r)
                    _xi_              += 1
                    if _xi_ >= len(_cols_): _yi_, _xi_ = _yi_ + 1, 0
                break

        if not node_to_xy: return None, None, None
        return node_to_xy, left_overs, out_of

    # -------------------------------------------------------------------------
    # GPU recording helpers
    #
    # The component records its own primitives as it renders (rather than parsing its
    # finished SVG back), so both representations come from one set of numbers.  Two
    # shapes need help because the display list has no direct equivalent:
    #   - a *stroked* rounded rect: rect() is fill-only, so the outline is a polyline
    #     through roundedRectPoints() -- four straight edges would cut the corners off
    #   - a path 'd' string: handed to the shared pathToDL(), the same flattener
    #     svgToDisplayList() uses, so a curve cannot land in two different places
    # -------------------------------------------------------------------------

    def __roundedRectToDL__(self, dl: Any, x: float, y: float, w: float, h: float, rx: float, fill: str | None = None, fill_opacity: float = 1.0,
                            stroke: str | None = None, stroke_w: float = 0.0, stroke_opacity: float = 1.0, scissor: tuple | None = None) -> None:
        if fill is not None and fill != 'none':
            dl.rect(x, y, w, h, fill, rx=rx, opacity=fill_opacity, svg='', scissor=scissor)
        if stroke is not None and stroke != 'none' and stroke_w > 0:
            _pts_ = roundedRectPoints(x, y, w, h, rx)
            strokePolylineDL(dl, _pts_ + [_pts_[0]], stroke, width=stroke_w,
                             opacity=stroke_opacity, scissor=scissor)

    # The selection ring around a collapsed ego: the SVG strokes the cloud outline,
    # so on the GPU it rings the cloud's rounded-rect stand-in, 2px outside the
    # silhouette.  scissor reproduces the partial-selection clip window.
    def __selectionRingToDL__(self, dl: Any, x: float, y: float, color: str, scissor: tuple | None = None) -> None:
        self.__roundedRectToDL__(dl, x - CLOUD_ICON_W / 2.0 - 2, y - CLOUD_ICON_H / 2.0 - 2,
                                 CLOUD_ICON_W + 4, CLOUD_ICON_H + 4, CLOUD_ICON_RX + 2,
                                 stroke=color, stroke_w=2.0, scissor=scissor)

    # -------------------------------------------------------------------------
    # renderAlter() — render circles (or clouds) for one alter group
    # -------------------------------------------------------------------------

    # dl records the same primitives this method emits as SVG (world coordinates --
    # __renderSVG__ maps the whole body into canvas pixels once the viewBox is known).
    def renderAlter(self, nodes: list, befores: set, afters: set, x: float, y: float, y_max: float, w_max: int, mul: int,
                    r_min: float, r_pref: float, circle_inter_d: float, circle_spacer: int,
                    h_collapsed_sections: int, _bin_: int, _alter_: int, _alter_side_: str,
                    node_weights: dict | None = None, dl: Any = None) -> tuple:
        xmin, ymin, xmax, ymax = x, y, x, y
        node_to_xyrepstat = {}
        svg               = []

        def _nodeState_(seen_before: bool, seen_after: bool) -> str:
            if   seen_before and seen_after: return 'continuous'
            elif seen_before:                return 'stopped'
            elif seen_after:                 return 'started'
            else:                            return 'isolated'

        def _triangle_(tx: float, ty: float, r: float, s: float, d: int) -> str:
            nonlocal xmin, ymin, xmax, ymax
            p0 = (tx + d * (r / 2.0),  ty)
            p1 = (tx + d * (r + s),    ty + r)
            p2 = (tx + d * (r + s),    ty - r)
            for _pt_ in [p0, p1, p2]:
                xmin, ymin = min(xmin, _pt_[0]), min(ymin, _pt_[1])
                xmax, ymax = max(xmax, _pt_[0]), max(ymax, _pt_[1])
            _co_  = '#ff0000' if d == 1 else '#0000ff'  # red=new(left), blue=ending(right)
            _path_= f'M {p0[0]:.1f} {p0[1]:.1f} L {p1[0]:.1f} {p1[1]:.1f} L {p2[0]:.1f} {p2[1]:.1f} Z'
            if dl is not None: dl.polygon([p0, p1, p2], _co_, svg='')
            return f'<path d="{_path_}" stroke="none" fill="{_co_}" />'

        def _cloud_triangle_(tx: float, ty: float, offset: int, s: int, d: int) -> str:
            nonlocal xmin, ymin, xmax, ymax
            p0 = (tx + d * offset,        ty)
            p1 = (tx + d * (offset + s),  ty + s)
            p2 = (tx + d * (offset + s),  ty - s)
            for _pt_ in [p0, p1, p2]:
                xmin, ymin = min(xmin, _pt_[0]), min(ymin, _pt_[1])
                xmax, ymax = max(xmax, _pt_[0]), max(ymax, _pt_[1])
            _co_  = '#d3494e' if d == 1 else '#658cbb'
            _path_= f'M {p0[0]:.1f} {p0[1]:.1f} L {p1[0]:.1f} {p1[1]:.1f} L {p2[0]:.1f} {p2[1]:.1f} Z'
            if dl is not None: dl.polygon([p0, p1, p2], _co_, svg='')
            return f'<path d="{_path_}" stroke="none" fill="{_co_}" />'

        def _place_(n2xy: dict) -> None:
            nonlocal xmin, ymin, xmax, ymax, svg
            for _node_, _xyr_ in n2xy.items():
                _co_  = self.__nodeColor__(_node_)
                _hl_  = str(_node_) in self.highlight_nodes
                svg.append(
                    f'<circle cx="{_xyr_[0]:.1f}" cy="{_xyr_[1]:.1f}" r="{_xyr_[2]:.1f}"'
                    f' stroke="{_co_}" stroke-width="{"2.50" if _hl_ else "1.25"}"'
                    f' fill="{_co_}" fill-opacity="{"0.80" if _hl_ else "0.25"}"/>'
                )
                if dl is not None:
                    # the ring is opaque and only the fill is translucent, which is why
                    # circle() takes the two alphas separately
                    dl.circle(_xyr_[0], _xyr_[1], _xyr_[2], _co_, stroke=_co_,
                              stroke_w=(2.50 if _hl_ else 1.25),
                              opacity=(0.80 if _hl_ else 0.25), stroke_opacity=1.0, svg='')
                xmin = min(xmin, _xyr_[0] - _xyr_[2])
                ymin = min(ymin, _xyr_[1] - _xyr_[2])
                xmax = max(xmax, _xyr_[0] + _xyr_[2])
                ymax = max(ymax, _xyr_[1] + _xyr_[2])
                # started/stopped triangles
                if _node_ not in befores:
                    svg.append(_triangle_(_xyr_[0], _xyr_[1], _xyr_[2], circle_spacer / 2, -1))
                if _node_ not in afters:
                    svg.append(_triangle_(_xyr_[0], _xyr_[1], _xyr_[2], circle_spacer / 2,  1))
                _xyrepstat_ = (_xyr_[0], _xyr_[1], 'single',
                               _nodeState_(_node_ in befores, _node_ in afters),
                               _bin_, _alter_, _alter_side_, _xyr_[2])
                node_to_xyrepstat[_node_] = _xyrepstat_
                self.bin_to_node_to_xyrepstat[_bin_][_node_] = _xyrepstat_

        def _cloud_(n: int, y_cloud: float, ltriangle: bool, rtriangle: bool, nodes_in_cloud: list) -> None:
            nonlocal xmin, ymin, xmax, ymax, svg
            _cloud_co_ = self.p2s.colorTyped('axis', 'default')
            svg.append(
                f'<rect x="{x - 16:.1f}" y="{y_cloud - 8:.1f}" width="32" height="16"'
                f' rx="8" fill="{_cloud_co_}" fill-opacity="0.25"'
                f' stroke="{_cloud_co_}" stroke-width="1"/>'
            )
            if dl is not None:
                self.__roundedRectToDL__(dl, x - 16, y_cloud - 8, 32, 16, 8,
                                         fill=_cloud_co_, fill_opacity=0.25,
                                         stroke=_cloud_co_, stroke_w=1.0)
            if ltriangle: svg.append(_cloud_triangle_(x, y_cloud, 16, 6, -1))
            if rtriangle: svg.append(_cloud_triangle_(x, y_cloud, 16, 6,  1))
            _txt_co_ = self.p2s.colorTyped('label', 'defaultfg')
            svg.append(
                f'<text x="{x:.1f}" y="{y_cloud + self.txt_h * 0.38:.1f}"'
                f' font-size="{self.txt_h}px" text-anchor="middle" fill="{_txt_co_}">{self.p2s.svgEscape(str(n))}</text>'
            )
            if dl is not None:
                dl.text(self.p2s, str(n), x, y_cloud + self.txt_h * 0.38, txt_h=self.txt_h,
                        anchor='middle', color=_txt_co_, svg='')
            xmin = min(xmin, x - 22)
            ymin = min(ymin, y_cloud - 8)
            xmax = max(xmax, x + 22)
            ymax = max(ymax, y_cloud + 8)
            for _node_ in nodes_in_cloud:
                _xyrepstat_ = (x, y_cloud, 'cloud',
                               _nodeState_(not ltriangle, not rtriangle),
                               _bin_, _alter_, _alter_side_, None)
                node_to_xyrepstat[_node_] = _xyrepstat_
                self.bin_to_node_to_xyrepstat[_bin_][_node_] = _xyrepstat_

        if nodes:
            # Sort into 4 state buckets
            nodes_sorter     = []
            nodes_isolated   = []
            nodes_started    = []
            nodes_stopped    = []
            nodes_continuous = []
            for _node_ in nodes:
                _wt_ = node_weights.get(_node_, 0) if node_weights else None
                if   _node_ in befores and _node_ in afters:
                    _s_ = 3; nodes_continuous.append(_node_)
                elif _node_ in befores:
                    _s_ = 2; nodes_stopped.append(_node_)
                elif _node_ in afters:
                    _s_ = 1; nodes_started.append(_node_)
                else:
                    _s_ = 0; nodes_isolated.append(_node_)
                # With weights: (state, weight_asc, name) — ascending weight puts the
                # heaviest node last; packable() reverses so it sits closest to the ego.
                # Without weights: (state, name) — alphabetical within each state group.
                nodes_sorter.append((_s_, _wt_, _node_) if _wt_ is not None
                                    else (_s_, _node_))
            nodes_sorter  = sorted(nodes_sorter)
            nodes_ordered = [p[-1] for p in nodes_sorter]

            # Cascading fallback: try to show as many individuals as possible
            n2xy, _, _ = self.packable(nodes_ordered, x, y, y_max, w_max, mul,
                                        r_min, r_pref, circle_inter_d, circle_spacer)
            if n2xy is not None:
                _place_(n2xy)
            else:
                _ta_ = h_collapsed_sections if mul == 1 else -h_collapsed_sections
                n2xy, _, _ = self.packable(
                    nodes_started + nodes_stopped + nodes_continuous,
                    x, y, y_max - _ta_, w_max, mul, r_min, r_pref, circle_inter_d, circle_spacer)
                if n2xy is not None:
                    _place_(n2xy)
                    _yo_ = ymax if mul == 1 else ymin
                    if nodes_isolated:
                        _cloud_(len(nodes_isolated), _yo_ + mul * 0.5 * h_collapsed_sections,
                                True, True, nodes_isolated)
                else:
                    _ta2_ = 2 * h_collapsed_sections if mul == 1 else -2 * h_collapsed_sections
                    n2xy, _, _ = self.packable(
                        nodes_started + nodes_continuous, x, y, y_max - _ta2_, w_max, mul,
                        r_min, r_pref, circle_inter_d, circle_spacer)
                    if n2xy is not None:
                        _place_(n2xy)
                        _yo_ = ymax if mul == 1 else ymin
                        if nodes_stopped:
                            _cloud_(len(nodes_stopped), _yo_ + mul * 0.5 * h_collapsed_sections,
                                    False, True, nodes_stopped)
                            _yo_ = ymax if mul == 1 else ymin
                        if nodes_isolated:
                            _cloud_(len(nodes_isolated), _yo_ + mul * 0.5 * h_collapsed_sections,
                                    True, True, nodes_isolated)
                    else:
                        # Everything collapses to clouds
                        _yo_ = ymax if mul == 1 else ymin
                        for _grp_, _lt_, _rt_ in [
                            (nodes_continuous, False, False),
                            (nodes_started,    True,  False),
                            (nodes_stopped,    False, True ),
                            (nodes_isolated,   True,  True ),
                        ]:
                            if _grp_:
                                _cloud_(len(_grp_), _yo_ + mul * 0.5 * h_collapsed_sections,
                                        _lt_, _rt_, _grp_)
                                _yo_ = ymax if mul == 1 else ymin

        xmin -= r_pref; ymin -= r_pref
        xmax += r_pref; ymax += r_pref
        return ''.join(svg), (xmin, ymin, xmax, ymax), node_to_xyrepstat

    # -------------------------------------------------------------------------
    # _cloud_outline_d_() — cloud path in outer SVG coords, centred at (cx,cy)
    # The inner SVG has width="100px" height="50px" viewBox="-5 -5.5 35 35".
    # preserveAspectRatio="xMidYMid meet" fits the square viewBox into 100×50
    # using the smaller dimension (height=50), so scale=50/35 for both axes
    # with a 25px centering offset in x.
    #   translate(-50,-25) + centering(+25,0) + scale(50/35)
    #   => dx = (10*vx - 125)/7,  dy = (10*vy - 120)/7
    # -------------------------------------------------------------------------

    @staticmethod
    def _cloud_outline_d_(cx: float, cy: float) -> str:
        x, y = cx, cy
        return (
            f'M {x+2.27:.2f} {y-7.14:.2f}'
            f' C {x+3.56:.2f} {y-7.18:.2f} {x+4.81:.2f} {y-6.70:.2f} {x+5.75:.2f} {y-5.82:.2f}'
            f' C {x+6.69:.2f} {y-4.93:.2f} {x+7.24:.2f} {y-3.71:.2f} {x+7.27:.2f} {y-2.43:.2f}'
            f' C {x+7.27:.2f} {y-1.97:.2f} {x+7.20:.2f} {y-1.51:.2f} {x+7.07:.2f} {y-1.08:.2f}'
            f' C {x+8.73:.2f} {y-0.81:.2f} {x+9.96:.2f} {y+0.61:.2f} {x+10.00:.2f} {y+2.29:.2f}'
            f' C {x+9.94:.2f} {y+4.24:.2f} {x+8.31:.2f} {y+5.77:.2f} {x+6.36:.2f} {y+5.72:.2f}'
            f' L {x-6.36:.2f} {y+5.72:.2f}'
            f' C {x-8.31:.2f} {y+5.77:.2f} {x-9.94:.2f} {y+4.24:.2f} {x-10.00:.2f} {y+2.29:.2f}'
            f' C {x-9.97:.2f} {y+0.68:.2f} {x-8.84:.2f} {y-0.69:.2f} {x-7.27:.2f} {y-1.03:.2f}'
            f' C {x-7.27:.2f} {y-1.07:.2f} {x-7.27:.2f} {y-1.10:.2f} {x-7.27:.2f} {y-1.14:.2f}'
            f' C {x-7.21:.2f} {y-3.09:.2f} {x-5.59:.2f} {y-4.62:.2f} {x-3.64:.2f} {y-4.57:.2f}'
            f' C {x-3.18:.2f} {y-4.57:.2f} {x-2.73:.2f} {y-4.49:.2f} {x-2.30:.2f} {y-4.33:.2f}'
            f' C {x-1.45:.2f} {y-6.07:.2f} {x+0.33:.2f} {y-7.17:.2f} {x+2.27:.2f} {y-7.14:.2f} Z'
        )

    # -------------------------------------------------------------------------
    # svgCrossConnect() — Bézier curve joining two points across bins
    # -------------------------------------------------------------------------

    def svgCrossConnect(self, x0: float, y0: float, x1: float, y1: float,
                        launch: Any = None, shift0: Any = None, shift1: Any = None,
                        color: str = '#000000', width: float = 1.0, dl: Any = None) -> str:
        if launch is None: launch = (x1 - x0) * 0.1
        if shift0 is None: shift0 = 0
        if shift1 is None: shift1 = 0
        xm = (x0 + x1) / 2.0
        _d_ = (f'M {x0:.1f} {y0:.1f} L {x0+launch:.1f} {y0:.1f}'
               f' C {xm+shift0:.1f} {y0:.1f} {xm-shift1:.1f} {y1:.1f}'
               f' {x1-launch:.1f} {y1:.1f} L {x1:.1f} {y1:.1f}')
        if dl is not None: pathToDL(dl, _d_, stroke=color, width=width)
        return (f'<path d="{_d_}"'
                f' stroke="{color}" stroke-width="{width}" fill="none" />')

    # -------------------------------------------------------------------------
    # bubbleNumberOnLine() — channel label pill
    # -------------------------------------------------------------------------

    def bubbleNumberOnLine(self, x0: float, x1: float, y: float, txt: str,
                           color: str = '#c0c0c0', width: float = 2.0, dl: Any = None) -> str:
        _txt_h_  = self.txt_h
        _txt_w_  = len(str(txt)) * _txt_h_ * 0.62
        xm       = (x0 + x1) / 2.0
        x0_1     = xm - 0.75 * _txt_w_
        x0_2     = x0_1 - _txt_w_ / 2.0
        x1_1     = xm + 0.75 * _txt_w_
        x1_2     = x1_1 + _txt_w_ / 2.0
        y_top    = y - _txt_h_ / 2.0 - 2
        y_bot    = y + _txt_h_ / 2.0 + 2
        p = [
            f'M {x0:.1f} {y:.1f} L {x0_2:.1f} {y:.1f}',
            f'C {x0_1:.1f} {y:.1f} {x0_2:.1f} {y_top:.1f} {x0_1:.1f} {y_top:.1f}',
            f'L {x1_1:.1f} {y_top:.1f}',
            f'C {x1_2:.1f} {y_top:.1f} {x1_1:.1f} {y:.1f} {x1_2:.1f} {y:.1f}',
            f'L {x1:.1f} {y:.1f} L {x1_2:.1f} {y:.1f}',
            f'C {x1_1:.1f} {y:.1f} {x1_2:.1f} {y_bot:.1f} {x1_1:.1f} {y_bot:.1f}',
            f'L {x0_1:.1f} {y_bot:.1f}',
            f'C {x0_2:.1f} {y_bot:.1f} {x0_1:.1f} {y:.1f} {x0_2:.1f} {y:.1f}',
            f'L {x0:.1f} {y:.1f}',
        ]
        _txt_co_ = self.p2s.colorTyped('label', 'defaultfg')
        _d_      = ' '.join(p)
        if dl is not None:
            pathToDL(dl, _d_, fill=color, stroke=color, width=width)
            dl.text(self.p2s, str(txt), xm, y + _txt_h_ * 0.38, txt_h=_txt_h_,
                    anchor='middle', color=_txt_co_, svg='')
        return (f'<path d="{_d_}" stroke="{color}" stroke-width="{width}" fill="{color}"/>'
                f'<text x="{xm:.1f}" y="{y + _txt_h_ * 0.38:.1f}"'
                f' font-size="{_txt_h_}px" text-anchor="middle" fill="{_txt_co_}">{self.p2s.svgEscape(str(txt))}</text>')

    # -------------------------------------------------------------------------
    # _smoothedPath_() — closed polygon with rounded corners
    # -------------------------------------------------------------------------

    def _smoothedPath_(self, points: list, corner_r: float = 8.0) -> str:
        """
        Convert a closed polygon (list of (x,y) tuples) into an SVG path
        with rounded corners.

        At every corner the sharp turn is replaced by:
            L pre_corner  C corner corner  post_corner
        where pre_corner and post_corner are corner_r pixels before/after
        the corner along the adjacent edges, capped at half each edge's
        length so they never overshoot the midpoint.
        """
        n = len(points)
        if n < 2:
            return ''

        def _unit_(a: tuple, b: tuple) -> tuple:
            dx, dy = b[0] - a[0], b[1] - a[1]
            d = (dx * dx + dy * dy) ** 0.5
            return (dx / d, dy / d) if d > 1e-9 else (1.0, 0.0)

        def _dist_(a: tuple, b: tuple) -> float:
            return ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5

        def _offset_(pt: tuple, uv: tuple, r: float) -> tuple:
            return (pt[0] + uv[0] * r, pt[1] + uv[1] * r)

        def _corner_(i: int) -> tuple:
            """(pre_corner, post_corner, corner_pt) for vertex i of the polygon."""
            p_prev = points[(i - 1) % n]
            p_curr = points[i]
            p_next = points[(i + 1) % n]
            uv_in  = _unit_(p_prev, p_curr)
            uv_out = _unit_(p_curr, p_next)
            r_in   = min(corner_r, _dist_(p_prev, p_curr) / 2.0)
            r_out  = min(corner_r, _dist_(p_curr, p_next) / 2.0)
            pre    = _offset_(p_curr, (-uv_in[0],  -uv_in[1]),  r_in)
            post   = _offset_(p_curr, ( uv_out[0],  uv_out[1]), r_out)
            return pre, post, p_curr

        # Start at the exit of corner 0 (just past the first corner)
        _, post0, _ = _corner_(0)
        parts = [f'M {post0[0]:.2f} {post0[1]:.2f}']

        # Corners 1 … n-1
        for i in range(1, n):
            pre, post, corner = _corner_(i)
            parts.append(f'L {pre[0]:.2f} {pre[1]:.2f}')
            parts.append(
                f'C {corner[0]:.2f} {corner[1]:.2f}'
                f' {corner[0]:.2f} {corner[1]:.2f}'
                f' {post[0]:.2f} {post[1]:.2f}'
            )

        # Explicit close: re-enter corner 0 then end at the M-point
        pre0, _, corner0 = _corner_(0)
        parts.append(f'L {pre0[0]:.2f} {pre0[1]:.2f}')
        parts.append(
            f'C {corner0[0]:.2f} {corner0[1]:.2f}'
            f' {corner0[0]:.2f} {corner0[1]:.2f}'
            f' {post0[0]:.2f} {post0[1]:.2f}'
        )
        parts.append('Z')
        return ' '.join(parts)

    # -------------------------------------------------------------------------
    # _binOutlinePoints_() — polygon corners for the metro-map bin outline
    # -------------------------------------------------------------------------

    def _binOutlinePoints_(self, x: float, y: float, w2: float, ind: float, r_pref: float,
                           a1fm: tuple | None, a2fm: tuple | None, a1to: tuple | None, a2to: tuple | None) -> list:
        """
        Return the clockwise corner sequence (list of (x,y) tuples) that
        describes the metro-map bin boundary.  Starting from the left edge
        at ego-line height, the polygon traces: down through the to-side
        alters (below ego), back up the right side, up through the fm-side
        alters (above ego), and back to the start.

        Indented bridges at the 1↔2-level boundaries give the characteristic
        metro-map notch shape.

        a1fm/a2fm  — (xmin,ymin,xmax,ymax) bounds for alter-1/2 fm, or None
        a1to/a2to  — same for alter-1/2 to side
        """
        pts = [(x - w2, y)]                     # left edge at ego centre

        # ── Below ego: to side ───────────────────────────────────────────────
        if a1to is not None:
            pts.append((x - w2, a1to[3]))       # left down to alter-1 bottom
            if a2to is not None:
                pts += [
                    (x - w2 + ind, a1to[3]),    # indent: narrow bridge begins
                    (x - w2 + ind, a2to[1]),    # bridge down to alter-2 top
                    (x - w2,       a2to[1]),    # expand back out
                    (x - w2,       a2to[3]),    # down to alter-2 bottom
                    (x + w2,       a2to[3]),    # across bottom
                    (x + w2,       a2to[1]),    # up to alter-2 top
                    (x + w2 - ind, a2to[1]),    # indent: bridge begins
                    (x + w2 - ind, a1to[3]),    # bridge up to alter-1 bottom
                    (x + w2,       a1to[3]),    # expand back out
                ]
            else:
                pts.append((x + w2, a1to[3]))  # simple right corner
        else:
            pts += [(x - w2, y + r_pref),       # no alters: small ego cap
                    (x + w2, y + r_pref)]
        pts.append((x + w2, y))                 # right edge back to ego centre

        # ── Above ego: fm side ───────────────────────────────────────────────
        if a1fm is not None:
            pts.append((x + w2, a1fm[1]))       # right up to alter-1 top
            if a2fm is not None:
                pts += [
                    (x + w2 - ind, a1fm[1]),    # indent: bridge begins
                    (x + w2 - ind, a2fm[3]),    # bridge up to alter-2 bottom
                    (x + w2,       a2fm[3]),    # expand back out
                    (x + w2,       a2fm[1]),    # up to alter-2 top
                    (x - w2,       a2fm[1]),    # across top
                    (x - w2,       a2fm[3]),    # down to alter-2 bottom
                    (x - w2 + ind, a2fm[3]),    # indent: bridge begins
                    (x - w2 + ind, a1fm[1]),    # bridge down to alter-1 top
                    (x - w2,       a1fm[1]),    # expand back out
                ]
            else:
                pts.append((x - w2, a1fm[1]))  # simple left corner
        else:
            pts += [(x + w2, y - r_pref),       # no alters: small ego cap
                    (x - w2, y - r_pref)]

        # _smoothedPath_ closes back to pts[0] automatically
        return pts

    # -------------------------------------------------------------------------
    # renderBin() — render a full temporal bin column
    # -------------------------------------------------------------------------

    def renderBin(self, b: int, x: float, y: float, max_w: int, max_h: int, dl: Any = None) -> tuple:
        r_min                = self.r_min
        r_pref               = self.r_pref
        circle_inter_d       = self.circle_inter_d
        circle_spacer        = self.circle_spacer
        alter_separation_h   = self.alter_separation_h
        h_collapsed_sections = self.h_collapsed_sections

        self.bin_to_node_to_xyrepstat[b] = {}

        _bins_ordered_ = sorted(self.bin_to_timestamps.keys())
        _befores_ = set()
        _afters_  = set()
        for _i_ in _bins_ordered_:
            if _i_ < b: _befores_ |= self._nodesInBin_(_i_)
            if _i_ > b: _afters_  |= self._nodesInBin_(_i_)

        node_2_xyrs = {}
        svg         = []

        # ── Focal node ────────────────────────────────────────────────────────
        _n_focal_     = len(self.bin_to_focal_nodes_present[b])
        _ego_co_      = self.p2s.colorTyped('data', 'default')
        _axis_co_     = self.p2s.colorTyped('axis', 'default')
        _sel_co_      = self.p2s.colorTyped('selection', 'default')
        _focal_nodes_ = self.bin_to_focal_nodes_present[b]
        if self.ego_is_set:
            # When ego_is_set, the layout replaced real ego names with '__EGO__', so
            # bin_to_focal_nodes_present contains {'__EGO__'} — not the actual names.
            # Use self.ego (the original parameter) to check highlight_nodes, which
            # always holds real node names (from linkp or from click translation).
            _real_ego_    = frozenset(str(n) for n in self.ego)
            _n_ego_total_ = len(_real_ego_)
            _n_sel_focal_ = sum(1 for n in _real_ego_ if n in self.highlight_nodes)
            svg.append(
                f'<use href="#cloud" x="{x:.1f}" y="{y:.1f}"'
                f' fill="{_ego_co_}" stroke="{_axis_co_}" stroke-width="0.5"/>'
            )
            if dl is not None:
                # The cloud glyph has no GPU primitive; both this component and linkp
                # approximate it with the shared CLOUD_ICON_* rounded rect.
                self.__roundedRectToDL__(dl, x - CLOUD_ICON_W / 2.0, y - CLOUD_ICON_H / 2.0,
                                         CLOUD_ICON_W, CLOUD_ICON_H, CLOUD_ICON_RX,
                                         fill=_ego_co_, stroke=_axis_co_, stroke_w=0.5)
            if _n_sel_focal_ == _n_ego_total_ and _n_ego_total_ > 0:
                svg.append(
                    f'<use href="#cloud_outline" x="{x:.1f}" y="{y:.1f}"'
                    f' fill="none" stroke="{_sel_co_}" stroke-width="2.0"/>'
                )
                if dl is not None:
                    self.__selectionRingToDL__(dl, x, y, _sel_co_)
            elif _n_sel_focal_ > 0:
                _clip_id_ = f'ccl_{b}'
                svg.append(
                    f'<clipPath id="{_clip_id_}" clipPathUnits="userSpaceOnUse">'
                    f'<rect x="{x:.1f}" y="{y-10:.1f}" width="15" height="20"/>'
                    f'</clipPath>'
                    f'<path d="{self._cloud_outline_d_(x, y)}"'
                    f' fill="none" stroke="{_sel_co_}" stroke-width="2.0"'
                    f' clip-path="url(#{_clip_id_})"/>'
                )
                if dl is not None:
                    # partial selection: the SVG shows the ring through a clip window,
                    # which the display list expresses as a per-op scissor
                    self.__selectionRingToDL__(dl, x, y, _sel_co_,
                                               scissor=(x, y - 10, 15, 20))
        elif _n_focal_ == 1:
            _n_sel_focal_ = sum(1 for n in _focal_nodes_ if n in self.highlight_nodes)
            svg.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r_pref}"'
                f' stroke="{_axis_co_}" stroke-width="0.4" fill="{_ego_co_}"/>'
            )
            if dl is not None:
                dl.circle(x, y, r_pref, _ego_co_, stroke=_axis_co_, stroke_w=0.4, svg='')
            if _n_sel_focal_ == 1:
                svg.append(
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r_pref + 2:.1f}"'
                    f' fill="none" stroke="{_sel_co_}" stroke-width="2.0"/>'
                )
                if dl is not None:
                    dl.circle(x, y, r_pref + 2, 'none', stroke=_sel_co_, stroke_w=2.0, svg='')
        else:
            svg.append(
                f'<rect x="{x-12:.1f}" y="{y-8:.1f}" width="24" height="16" rx="8"'
                f' fill="{_ego_co_}" stroke="{_axis_co_}" stroke-width="0.4"/>'
            )
            _txt_co_ = self.p2s.colorTyped('label', 'defaultfg')
            svg.append(
                f'<text x="{x:.1f}" y="{y + self.txt_h * 0.38:.1f}"'
                f' font-size="{self.txt_h}px" text-anchor="middle" fill="{_txt_co_}">{self.p2s.svgEscape(str(_n_focal_))}</text>'
            )
            if dl is not None:
                self.__roundedRectToDL__(dl, x - 12, y - 8, 24, 16, 8,
                                         fill=_ego_co_, stroke=_axis_co_, stroke_w=0.4)
                dl.text(self.p2s, str(_n_focal_), x, y + self.txt_h * 0.38,
                        txt_h=self.txt_h, anchor='middle', color=_txt_co_, svg='')
        _xyrepstat_ = (x, y, 'cloud' if self.ego_is_set else ('single' if _n_focal_ == 1 else 'cloud'),
                       'continuous', b, None, None, r_pref)
        for _fn_ in self.bin_to_focal_nodes_present[b]:
            self.bin_to_node_to_xyrepstat[b][_fn_] = _xyrepstat_

        max_alter_h = max_h / 5.0
        _wts_       = self.bin_to_node_weights.get(b) or None   # None when count=ROW_COUNTp

        # ── Alter-1 fm (senders → ego, above ego line) ───────────────────────
        a1fm_bounds = None
        if b in self.bin_to_alter1s and self.bin_to_alter1s[b]['fm']:
            _svg_, _bnd_, _n2xyrs_ = self.renderAlter(
                sorted(self.bin_to_alter1s[b]['fm']), _befores_, _afters_,
                x, y - r_pref - 2 * circle_inter_d,
                y - r_pref - max_alter_h, max_w, -1,
                r_min, r_pref, circle_inter_d, circle_spacer, h_collapsed_sections,
                b, 1, 'fm', node_weights=_wts_, dl=dl)
            svg.append(_svg_); node_2_xyrs.update(_n2xyrs_)
            a1fm_bounds = _bnd_
            _prev_bnd_  = _bnd_
        else:
            _prev_bnd_ = (x - r_pref, y - r_pref - 2 * circle_inter_d - 5,
                          x + r_pref, y - r_pref - 2 * circle_inter_d)

        # ── Alter-2 fm ────────────────────────────────────────────────────────
        a2fm_bounds = None
        if (self.max_rings >= 2 and b in self.bin_to_alter2s and
                self.bin_to_alter2s[b]['fm']):
            _svg_, _bnd_, _n2xyrs_ = self.renderAlter(
                sorted(self.bin_to_alter2s[b]['fm']), _befores_, _afters_,
                x, _prev_bnd_[1] - alter_separation_h,
                _prev_bnd_[1] - alter_separation_h - max_alter_h, max_w, -1,
                r_min, r_pref, circle_inter_d, circle_spacer, h_collapsed_sections,
                b, 2, 'fm', node_weights=_wts_, dl=dl)
            svg.append(_svg_); node_2_xyrs.update(_n2xyrs_)
            a2fm_bounds = _bnd_

        # ── Alter-1 to (ego → receivers, below ego line) ─────────────────────
        a1to_bounds = None
        if b in self.bin_to_alter1s and self.bin_to_alter1s[b]['to']:
            _svg_, _bnd_, _n2xyrs_ = self.renderAlter(
                sorted(self.bin_to_alter1s[b]['to']), _befores_, _afters_,
                x, y + r_pref + 2 * circle_inter_d,
                y + r_pref + 2 * circle_inter_d + max_alter_h, max_w, 1,
                r_min, r_pref, circle_inter_d, circle_spacer, h_collapsed_sections,
                b, 1, 'to', node_weights=_wts_, dl=dl)
            svg.append(_svg_); node_2_xyrs.update(_n2xyrs_)
            a1to_bounds = _bnd_
            _prev_bnd_  = _bnd_
        else:
            _prev_bnd_ = (x - r_pref, y + r_pref + 2 * circle_inter_d,
                          x + r_pref, y + r_pref + 2 * circle_inter_d + 5)

        # ── Alter-2 to ────────────────────────────────────────────────────────
        a2to_bounds = None
        if (self.max_rings >= 2 and b in self.bin_to_alter2s and
                self.bin_to_alter2s[b]['to']):
            _svg_, _bnd_, _n2xyrs_ = self.renderAlter(
                sorted(self.bin_to_alter2s[b]['to']), _befores_, _afters_,
                x, _prev_bnd_[3] + alter_separation_h,
                _prev_bnd_[3] + alter_separation_h + max_alter_h, max_w, 1,
                r_min, r_pref, circle_inter_d, circle_spacer, h_collapsed_sections,
                b, 2, 'to', node_weights=_wts_, dl=dl)
            svg.append(_svg_); node_2_xyrs.update(_n2xyrs_)
            a2to_bounds = _bnd_

        # ── Bin outline path — rounded metro-map boundary ─────────────────────
        _w_  = 3 * r_pref
        for _bnd_ in [a1fm_bounds, a1to_bounds, a2fm_bounds, a2to_bounds]:
            if _bnd_ is not None:
                _w_ = max(_w_, _bnd_[2] - _bnd_[0])
        _w2_  = _w_ / 2.0
        _ind_ = max(r_pref, _w_ / 4.0)

        _pts_ = self._binOutlinePoints_(
            x, y, _w2_, _ind_, r_pref,
            a1fm_bounds, a2fm_bounds, a1to_bounds, a2to_bounds,
        )
        _outline_d_ = self._smoothedPath_(_pts_, corner_r=r_pref * 0.8)
        svg.append(
            f'<path d="{_outline_d_}"'
            f' stroke="{_axis_co_}" stroke-width="2.0" fill="none"/>'
        )
        if dl is not None:
            pathToDL(dl, _outline_d_, stroke=_axis_co_, width=2.0)

        # Bounds from the polygon corners (no string parsing needed)
        bx0 = min(pt[0] for pt in _pts_)
        by0 = min(pt[1] for pt in _pts_)
        bx1 = max(pt[0] for pt in _pts_)
        by1 = max(pt[1] for pt in _pts_)
        return ''.join(svg), (bx0, by0, bx1, by1), node_2_xyrs

    # -------------------------------------------------------------------------
    # __renderSVG__
    # -------------------------------------------------------------------------

    #
    # __legendPrepare__() - resolve legend kind/metadata (the capture hook).
    # SpreadLinesP colors nodes categorically only (by node name -- the default and
    # COLOR_BY_NODE_NAME -- or by a field's per-node value), so the legend is always
    # a categorical swatch list.  Decision A: fixed hex / explicit dict node colors
    # carry no data-driven semantics, so a truthy legend silently reserves nothing.
    #
    def __legendPrepare__(self) -> None:
        self.legend_info      = None
        self._legend_region_  = None
        self._legend_reserve_ = (0, 0, 0, 0)
        self._dl_legend_      = None
        if self.legend_spec is None or self.df is None or len(self.df) == 0: return
        if isinstance(self.node_color, self.p2s.HexColorString) or isinstance(self.node_color, dict): return
        _spec_ = self.legend_spec
        if isinstance(self.node_color, str) and self.node_color in self.df.columns:
            _title_default_ = self.node_color
            _vc_ = self.p2s.legendCategoricalValueCounts(self.df, self.node_color)
        else:  # None / COLOR_BY_NODE_NAME -> colored by node name
            _title_default_ = 'node'
            _names_ = pl.concat([self.df.select(pl.col(_r_[_j_]).cast(pl.String).alias('__legend_node__'))
                                 for _r_ in self.relationships for _j_ in (0, 1)])
            _vc_ = self.p2s.legendCategoricalValueCounts(_names_, '__legend_node__')
        _title_ = _spec_['title'] if _spec_['title'] is not None else _title_default_
        self.legend_info = self.p2s.legendInfoCategorical(_spec_, _vc_, _title_)
        _reserve_ = self.p2s.legendReserve(_spec_, self.legend_info, self.txt_h, self.wxh)
        _l_, _r_, _t_, _b_ = _reserve_
        if self.wxh[0] - (_l_ + _r_) < 48 or self.wxh[1] - (_t_ + _b_) < 48:
            self.p2s.logger.warning(f'SpreadLinesP.__legendPrepare__(): not enough space for legend (wxh = {self.wxh}); legend dropped')
            self.legend_info = None
            return
        self._legend_reserve_ = _reserve_
        # the screen region is derived in __renderSVG__ from the final viewBox
        # transform (the content is letterbox-fitted, so the strip position depends
        # on the fitted bounds)

    def __renderSVG__(self, rand_id: int) -> None:
        self._gpu_dl_ = self._gpu_payload_ = None   # invalidate cached GPU state
        self._dl_body_ = None
        self.__legendPrepare__()
        w, h        = self.wxh
        _bg_co_     = self.p2s.colorTyped('background', 'default')
        _axis_co_   = self.p2s.colorTyped('axis', 'default')
        _border_co_ = self.p2s.colorTyped('axis', 'inner')
        _txt_co_    = self.p2s.colorTyped('label', 'defaultfg')
        _data_co_   = self.p2s.colorTyped('data', 'default')

        alter_inter_d   = self.alter_inter_d
        max_bin_w       = self.max_bin_w
        max_bin_h       = self.max_bin_h
        min_channel_w   = self.min_channel_w
        max_channel_w   = self.max_channel_w
        channel_inter_d = self.channel_inter_d

        svg: list = []

        # ── GPU recording layers ───────────────────────────────────────────────
        # Everything below is emitted in *world* coordinates and mapped into canvas
        # pixels at the end, once the viewBox is known (it is sized from the bins this
        # method is about to place, so the transform cannot be known up front).
        #
        # The SVG list is assembled with insert(0) for the strokes that belong under
        # the bins, which the display list has no equivalent for -- so each z-layer
        # records separately and they are composed bottom-to-top at the end.  The two
        # insert(0) groups end up in reverse emission order in the SVG, so their
        # per-item lists are extended in reverse to match.
        from polars2svg.p2s_displaylist import DisplayList
        _mkdl_          = lambda: DisplayList(w, h, bg=_bg_co_)
        _dl_ego_line_   = _mkdl_()      # bottom
        _dl_zigzags_    = []            # per-item, reversed at compose time
        _dl_connects_   = []            # per-item, reversed at compose time
        _dl_bins_       = _mkdl_()
        _dl_channels_   = _mkdl_()
        _dl_anno_       = _mkdl_()
        _dl_ts_labels_  = _mkdl_()      # top

        # ── Reset per-render state ─────────────────────────────────────────────
        self.vx0 = self.vy0 = self.vx1 = self.vy1 = None
        self.bin_to_bounds            = {}
        self.bin_to_node_to_xyrepstat: dict = {}

        if not self.bin_to_timestamps:
            # No data — blank
            svg.insert(0, f'<svg x="0" y="0" width="{w}" height="{h}" font-family="{self.p2s.default_font}" xmlns="http://www.w3.org/2000/svg">')
            svg.insert(1, f'<rect x="0" y="0" width="{w}" height="{h}" fill="{_bg_co_}"/>')
            svg.append('</svg>')
            self.svg = ''.join(svg)
            _dl_ = _mkdl_()
            _dl_.rect(0, 0, w, h, _bg_co_, svg='')
            self._dl_body_ = _dl_       # already canvas-space: no viewBox on this branch
            return

        # ── Render each bin ────────────────────────────────────────────────────
        _bins_ = sorted(self.bin_to_timestamps.keys())
        bin_to_n2xyrs = {}
        bx = alter_inter_d
        by = h / 2.0

        for _b_ in _bins_:
            _svg_, _bnd_, _n2xyrs_ = self.renderBin(_b_, bx, by, max_bin_w, max_bin_h,
                                                    dl=_dl_bins_)
            bin_to_n2xyrs    [_b_] = _n2xyrs_
            self.bin_to_bounds[_b_] = _bnd_
            svg.append(_svg_)
            xmin, ymin, xmax, ymax = _bnd_
            bx = xmax + alter_inter_d
            if self.vx0 is None:
                self.vx0, self.vy0, self.vx1, self.vy1 = _bnd_
            self.vx0 = min(self.vx0, _bnd_[0] - alter_inter_d / 3.0)
            self.vy0 = min(self.vy0, _bnd_[1] - 3 * channel_inter_d)
            self.vx1 = max(self.vx1, _bnd_[2] + alter_inter_d / 3.0)
            self.vy1 = max(self.vy1, _bnd_[3] + 3 * channel_inter_d)

        # ── Channel allocation ─────────────────────────────────────────────────
        bin_to_nodes_to_channel: dict          = {}
        max_n_channel, min_n_channel     = 0, int(1e9)
        tuple_to_channel_geom: dict            = {}
        channel_tuples                   = []

        for _fm_to_ in ['to', 'fm']:
            for i in range(len(_bins_) - 1, 1, -1):
                _b_    = _bins_[i]
                _nodes_ = set()
                if _b_ in self.bin_to_alter1s and _fm_to_ in self.bin_to_alter1s[_b_]:
                    _nodes_ |= self.bin_to_alter1s[_b_][_fm_to_]
                if _b_ in self.bin_to_alter2s and _fm_to_ in self.bin_to_alter2s[_b_]:
                    _nodes_ |= self.bin_to_alter2s[_b_][_fm_to_]

                _nodes_ -= self._nodesInBin_(_bins_[i - 1])

                if _fm_to_ == 'fm':
                    y_clear = self.bin_to_bounds[_bins_[i-1]][1] - max_channel_w - channel_inter_d
                else:
                    y_clear = self.bin_to_bounds[_bins_[i-1]][3] + max_channel_w + channel_inter_d

                _befores_ = set()
                for j in range(i): _befores_ |= self._nodesInBin_(_bins_[j])
                _nodes_ &= _befores_
                n_in_ch   = len(_nodes_)
                max_n_channel = max(max_n_channel, n_in_ch)
                min_n_channel = min(min_n_channel, n_in_ch)

                if n_in_ch > 0:
                    _saving_ = []
                    for j in range(i - 2, -1, -1):
                        _here_      = _bins_[j]
                        _here_nds_  = self._nodesInBin_(_here_)
                        for _nd_ in sorted(_nodes_ & _here_nds_):
                            _saving_.append((_b_, _here_, _nd_))
                        _nodes_ -= _here_nds_
                        if not _nodes_: break
                        if _fm_to_ == 'fm':
                            y_clear = min(y_clear,
                                          self.bin_to_bounds[_here_][1] - max_channel_w - channel_inter_d)
                        else:
                            y_clear = max(y_clear,
                                          self.bin_to_bounds[_here_][3] + max_channel_w + channel_inter_d)
                    _ct_ = (_here_, _b_, y_clear, n_in_ch, _fm_to_)
                    channel_tuples.append(_ct_)
                    for _tb_, _fb_, _nd_ in _saving_:
                        bin_to_nodes_to_channel.setdefault(_fb_, {}).setdefault(_tb_, {})[_nd_] = _ct_

        # Sort and render channels
        _channel_max_y_ = 0.0
        channel_tuples.sort(key=lambda ct: ct[2])
        for i, _ct_ in enumerate(channel_tuples):
            _start_, _end_, _y_, _n_, _fm_to_ = _ct_
            _div_ = max_n_channel - min_n_channel
            _ch_h_ = (min_channel_w if _div_ == 0 else
                      (_n_ - min_n_channel) / _div_ * (max_channel_w - min_channel_w) + min_channel_w)
            _ch_w_ = (self.bin_to_bounds[_end_][0] - self.bin_to_bounds[_start_][2] -
                      1.5 * alter_inter_d)
            _ch_x_ = self.bin_to_bounds[_start_][2] + alter_inter_d

            # Bump y to avoid overlap
            ok = False
            while not ok:
                ok = True
                for _other_ct_, _og_ in tuple_to_channel_geom.items():
                    if (_og_[0] < _ch_x_ + _ch_w_ and _og_[0] + _og_[2] > _ch_x_ and
                            _og_[1] - channel_inter_d < _y_ + _ch_h_ + channel_inter_d and
                            _og_[1] + _og_[3] + channel_inter_d > _y_ - channel_inter_d):
                        ok = False
                        break
                if not ok:
                    _y_ += channel_inter_d if _fm_to_ == 'to' else -channel_inter_d

            self.vy0 = min(self.vy0, _y_ - 3 * channel_inter_d)
            self.vy1 = max(self.vy1, _y_ + _ch_h_ + 3 * channel_inter_d)
            tuple_to_channel_geom[_ct_] = (_ch_x_, _y_, _ch_w_, _ch_h_)
            svg.append(self.bubbleNumberOnLine(_ch_x_, _ch_x_ + _ch_w_,
                                               _y_ + _ch_h_ / 2.0, str(_n_),
                                               color=_axis_co_, width=2.0,
                                               dl=_dl_channels_))
            _channel_max_y_ = max(_y_ + _ch_h_ + self.txt_h, _channel_max_y_)

        # ── Direct connects and channel end-connectors ─────────────────────────
        # Each connect goes to its own DisplayList so the compose step can replay them
        # in the order svg.insert(0) leaves them in (newest underneath).
        def _connect_(*args: Any, **kwargs: Any) -> str:
            _d_ = _mkdl_()
            _dl_connects_.append(_d_)
            return self.svgCrossConnect(*args, dl=_d_, **kwargs)

        for i in range(len(_bins_) - 1):
            _b0_, _b1_ = _bins_[i], _bins_[i + 1]
            _bnd0_     = self.bin_to_bounds[_b0_]
            _bnd1_     = self.bin_to_bounds[_b1_]
            _drawn_    = set()

            # Direct connects
            _shared_ = (bin_to_n2xyrs.get(_b0_, {}).keys() &
                        bin_to_n2xyrs.get(_b1_, {}).keys())
            for _nd_ in sorted(_shared_):
                _xyrs0_ = bin_to_n2xyrs[_b0_][_nd_]
                _xyrs1_ = bin_to_n2xyrs[_b1_][_nd_]
                _key_   = (_bnd0_[2], _xyrs0_[1], _bnd1_[0], _xyrs1_[1])
                if _key_ not in _drawn_:
                    _co_ = self.__nodeColor__(_nd_)
                    svg.insert(0, _connect_(
                        _bnd0_[2], _xyrs0_[1], _bnd1_[0], _xyrs1_[1],
                        color=_co_, width=1.5))
                    _drawn_.add(_key_)

            # Channel connectors from each bin
            if _b0_ in bin_to_nodes_to_channel:
                for _b_n_ in bin_to_nodes_to_channel[_b0_]:
                    for _nd_ in bin_to_nodes_to_channel[_b0_][_b_n_]:
                        _xyrs_   = bin_to_n2xyrs.get(_b0_, {}).get(_nd_)
                        if _xyrs_ is None: continue
                        _ct_     = bin_to_nodes_to_channel[_b0_][_b_n_][_nd_]
                        _cg_     = tuple_to_channel_geom.get(_ct_)
                        if _cg_ is None: continue
                        _cmid_   = _cg_[1] + _cg_[3] / 2.0
                        _hway_   = max(_bnd1_[0], _cg_[0])
                        _k1_     = (_bnd0_[2], _xyrs_[1], _cg_[0], _cmid_)
                        if _k1_ not in _drawn_:
                            svg.insert(0, _connect_(
                                _bnd0_[2], _xyrs_[1], _hway_, _cmid_,
                                color=_axis_co_, width=2.0))
                            _drawn_.add(_k1_)
                        # End connector
                        if _b_n_ in bin_to_n2xyrs:
                            _xyrs_e_ = bin_to_n2xyrs[_b_n_].get(_nd_)
                            if _xyrs_e_ is not None:
                                _bnd_n_ = self.bin_to_bounds[_b_n_]
                                _k2_ = (_bnd_n_[0], _xyrs_e_[1], _cg_[0] + _cg_[2], _cmid_)
                                if _k2_ not in _drawn_:
                                    svg.insert(0, _connect_(
                                        _bnd_n_[0], _xyrs_e_[1],
                                        _cg_[0] + _cg_[2], _cmid_,
                                        color=_axis_co_, width=2.0))
                                    _drawn_.add(_k2_)

        # ── Discontinuity zigzag lines ─────────────────────────────────────────
        # discontinuity_count_after_bin[b] = missing timestamps AFTER bin b.
        # Draw the zigzag in the gap between bin b's right edge and the next
        # valid bin's left edge (exact midpoint).
        _hrun_ = self.r_pref * 1.25
        _vrun_ = self.r_pref * 2.0
        _ctx_co_ = '#a0a0a0'
        _sorted_bounds_bins_ = sorted(self.bin_to_bounds.keys())
        _next_bin_lu_ = {_sorted_bounds_bins_[i]: _sorted_bounds_bins_[i + 1]
                         for i in range(len(_sorted_bounds_bins_) - 1)}
        for _b_ in self.discontinuity_count_after_bin:
            if _b_ not in self.bin_to_bounds: continue
            _next_b_ = _next_bin_lu_.get(_b_)
            if _next_b_ is None: continue          # gap after the last bin — nothing between
            _bnd_    = self.bin_to_bounds[_b_]
            _bnd_nx_ = self.bin_to_bounds[_next_b_]
            _zx_     = (_bnd_[2] + _bnd_nx_[0]) / 2.0
            _zy_  = self.vy0
            _d_   = [f'M {_zx_ - _hrun_:.1f} {_zy_:.1f}']
            _zy_ += _vrun_
            _turns_ = int(1 + (self.vy1 - self.vy0) / _vrun_)
            for _t_ in range(_turns_):
                _d_.append(f'L {(_zx_ + _hrun_ if _t_ % 2 == 0 else _zx_ - _hrun_):.1f} {_zy_:.1f}')
                _zy_ += _vrun_
            svg.insert(0,
                f'<path d="{" ".join(_d_)}" stroke="{_ctx_co_}" stroke-width="0.5"'
                f' fill="none" stroke-dasharray="3 3"/>')
            _dlz_ = _mkdl_()
            pathToDL(_dlz_, ' '.join(_d_), stroke=_ctx_co_, width=0.5, dash=(3.0, 3.0))
            _dl_zigzags_.append(_dlz_)

        # ── Ego horizontal line ────────────────────────────────────────────────
        if _bins_:
            _x_l_ = self.bin_to_bounds[_bins_[0]][0]
            _x_r_ = self.bin_to_bounds[_bins_[-1]][2]
            svg.insert(0,
                f'<line x1="{_x_l_:.1f}" y1="{by:.1f}" x2="{_x_r_:.1f}" y2="{by:.1f}"'
                f' stroke="{_data_co_}" stroke-width="3.0"/>')
            _dl_ego_line_.line(_x_l_, by, _x_r_, by, _data_co_, width=3.0, svg='')

        # ── Anno event lines ───────────────────────────────────────────────────
        _anno_ts_to_bx_ = {}
        for _b_, _ts_ in self.bin_to_timestamps.items():
            if _ts_ in self.anno:
                _anno_ts_to_bx_[_ts_] = (self.bin_to_bounds[_b_][0] + self.bin_to_bounds[_b_][2]) / 2.0
        for _ts_v_, _label_ in self.anno.items():
            _ts_s_ = str(_ts_v_)
            if _ts_s_ not in _anno_ts_to_bx_: continue
            _ax_ = _anno_ts_to_bx_[_ts_s_]
            svg.append(
                f'<line x1="{_ax_:.1f}" y1="{self.vy0:.1f}"'
                f' x2="{_ax_:.1f}" y2="{self.vy1:.1f}"'
                f' stroke="{_axis_co_}" stroke-width="1.5" stroke-dasharray="6,3" opacity="0.7"/>'
            )
            _dl_anno_.line(_ax_, self.vy0, _ax_, self.vy1, _axis_co_, width=1.5,
                           dash=(6.0, 3.0), opacity=0.7, svg='')
            svg.append(
                f'<text x="{_ax_ + 3:.1f}" y="{self.vy0 + self.txt_h:.1f}"'
                f' font-size="{self.txt_h}px" fill="{_txt_co_}">{self.p2s.svgEscape(str(_label_))}</text>'
            )
            _dl_anno_.text(self.p2s, str(_label_), _ax_ + 3, self.vy0 + self.txt_h,
                           txt_h=self.txt_h, anchor='start', color=_txt_co_, svg='')

        # ── Timestamp labels ───────────────────────────────────────────────────
        _channel_max_y_ = self.vy1
        if self.draw_context:
            for _b_ in _bins_:
                _bnd_ = self.bin_to_bounds[_b_]
                _channel_max_y_ = max(_bnd_[3] + self.txt_h, _channel_max_y_)
            for _b_ in self.bin_to_timestamps:
                if _b_ not in self.bin_to_bounds: continue
                _bnd_ = self.bin_to_bounds[_b_]
                _lx_  = (_bnd_[0] + _bnd_[2]) / 2.0
                _ts_  = str(self.bin_to_timestamps[_b_])
                # Untrusted timestamp data.  Sliced first, escaped second -- the order
                # svgEscape()'s header requires: a cut taken after escaping can land inside
                # an entity.
                _lbl_ = self.p2s.svgEscape(_ts_[:self._ts_label_len_])
                svg.append(
                    f'<text x="{_lx_:.1f}" y="{_channel_max_y_:.1f}"'
                    f' font-size="{self.txt_h}px" text-anchor="middle" fill="{_txt_co_}">'
                    f'{_lbl_}</text>'
                )
                _dl_ts_labels_.text(self.p2s, _ts_[:self._ts_label_len_], _lx_,
                                    _channel_max_y_, txt_h=self.txt_h, anchor='middle',
                                    color=_txt_co_, svg='')
            self.vy1 = _channel_max_y_ + self.txt_h

        # ── SVG header/footer with viewBox ─────────────────────────────────────
        if self.vx0 is None: self.vx0 = 0.0; self.vy0 = 0.0; self.vx1 = 1.0; self.vy1 = 1.0
        # Legend reserve: extend the world-space viewBox so the reserved strip maps
        # to exactly the requested pixels once the box is letterbox-fitted to wxh --
        # the content then scales into the remaining plot region ("reserve from wxh").
        _leg_l_, _leg_r_, _leg_t_, _leg_b_ = getattr(self, '_legend_reserve_', (0, 0, 0, 0))
        if getattr(self, 'legend_info', None) is not None and (_leg_l_ or _leg_r_ or _leg_t_ or _leg_b_):
            _cvw_, _cvh_ = self.vx1 - self.vx0, self.vy1 - self.vy0
            _s_fit_ = min((w - _leg_l_ - _leg_r_) / _cvw_, (h - _leg_t_ - _leg_b_) / _cvh_)
            if _s_fit_ > 0:
                self.vx0 -= _leg_l_ / _s_fit_;  self.vx1 += _leg_r_ / _s_fit_
                self.vy0 -= _leg_t_ / _s_fit_;  self.vy1 += _leg_b_ / _s_fit_
                # Pad the viewBox to the canvas aspect ratio (same scale, centering
                # made explicit): renderers that clip strictly to the viewBox rect
                # (svglib) would otherwise drop legend content drawn in the
                # letterboxed band outside it.
                _bvw_, _bvh_ = self.vx1 - self.vx0, self.vy1 - self.vy0
                _s_box_ = min(w / _bvw_, h / _bvh_)
                _dx_ = (w / _s_box_ - _bvw_) / 2.0
                _dy_ = (h / _s_box_ - _bvh_) / 2.0
                self.vx0 -= _dx_;  self.vx1 += _dx_
                self.vy0 -= _dy_;  self.vy1 += _dy_
            else:
                self.legend_info = None
        _vw_ = self.vx1 - self.vx0
        _vh_ = self.vy1 - self.vy0
        svg.insert(0, (f'<svg x="0" y="0" width="{w}" height="{h}"'
                       f' viewBox="{self.vx0:.1f} {self.vy0:.1f} {_vw_:.1f} {_vh_:.1f}"'
                       f' font-family="{self.p2s.default_font}"'
                       f' xmlns="http://www.w3.org/2000/svg">'))
        svg.insert(1, (f'<rect x="{self.vx0:.1f}" y="{self.vy0:.1f}"'
                       f' width="{_vw_:.1f}" height="{_vh_:.1f}" fill="{_bg_co_}"/>'))
        if self.ego_is_set:
            # Cloud symbols -- the shared definition beside CLOUD_ICON_* in
            # p2s_displaylist, the same one linkp emits.  #cloud_outline is the
            # unstroked variant, drawn under the ego node's own outline.
            svg.insert(2, '<defs>' + cloudIconDef() +
                          cloudIconDef('cloud_outline', stroke=None) + '</defs>')
        if self.draw_border:
            _bc_ = _border_co_
            svg.append(f'<rect x="{self.vx0:.1f}" y="{self.vy0:.1f}"'
                       f' width="{_vw_ - 1:.1f}" height="{_vh_ - 1:.1f}"'
                       f' fill="none" stroke="{_bc_}" stroke-width="1"/>')
        # ── Legend ─────────────────────────────────────────────────────────────
        # The SVG copy is drawn in *world* coordinates (scale=1/s), so it needs no
        # transform group (svglib misorders chained <g> transforms) and comes out at
        # true pixel size after the root viewBox mapping.  The GPU copy is recorded in
        # screen pixels instead and is spliced on top in gpuDisplayList(), after the
        # body has had applyTransform() run over it -- it must not be transformed twice.
        if getattr(self, 'legend_info', None) is not None and (_leg_l_ or _leg_r_ or _leg_t_ or _leg_b_):
            _s_  = min(w / _vw_, h / _vh_)
            _tx_ = (w - _vw_ * _s_) / 2.0 - self.vx0 * _s_
            _ty_ = (h - _vh_ * _s_) / 2.0 - self.vy0 * _s_
            if   _leg_r_: _region_ = (self.vx1 * _s_ + _tx_ - _leg_r_, 0, _leg_r_, h)
            elif _leg_l_: _region_ = (self.vx0 * _s_ + _tx_,           0, _leg_l_, h)
            elif _leg_t_: _region_ = (0, self.vy0 * _s_ + _ty_,           w, _leg_t_)
            else:         _region_ = (0, self.vy1 * _s_ + _ty_ - _leg_b_, w, _leg_b_)
            self._legend_region_ = _region_
            self._dl_legend_ = self.p2s.legendRenderDL(self.wxh, _region_, self.legend_spec,
                                                       self.legend_info, self.txt_h)
            _region_world_ = ((_region_[0] - _tx_) / _s_, (_region_[1] - _ty_) / _s_,
                              _region_[2] / _s_,          _region_[3] / _s_)
            _dl_world_ = self.p2s.legendRenderDL(self.wxh, _region_world_, self.legend_spec,
                                                 self.legend_info, self.txt_h, scale=1.0 / _s_)
            svg.append(_dl_world_.svg())
        svg.append('</svg>')
        self.svg = ''.join(svg)

        # ── Compose the GPU body ───────────────────────────────────────────────
        # Bottom-to-top in the order the SVG list ended up in: the ego line and the
        # zigzags/connects were insert(0)-ed under the bins (newest first, hence the
        # reversed replay), everything after the bins was appended over them.
        _dl_body_ = _mkdl_()
        _dl_body_.rect(self.vx0, self.vy0, _vw_, _vh_, _bg_co_, svg='')
        _dl_body_.extend(_dl_ego_line_)
        for _d_ in reversed(_dl_zigzags_):  _dl_body_.extend(_d_)
        for _d_ in reversed(_dl_connects_): _dl_body_.extend(_d_)
        _dl_body_.extend(_dl_bins_)
        _dl_body_.extend(_dl_channels_)
        _dl_body_.extend(_dl_anno_)
        _dl_body_.extend(_dl_ts_labels_)
        if self.draw_border:
            strokePolylineDL(_dl_body_,
                             [(self.vx0, self.vy0), (self.vx0 + _vw_ - 1, self.vy0),
                              (self.vx0 + _vw_ - 1, self.vy0 + _vh_ - 1),
                              (self.vx0, self.vy0 + _vh_ - 1), (self.vx0, self.vy0)],
                             _border_co_, width=1.0)
        # World -> canvas, the same mapping the root viewBox expresses.  The legend is
        # already in screen pixels, so it is spliced in afterwards (gpuDisplayList()).
        _s_fit_ = min(w / _vw_, h / _vh_)
        _dl_body_.applyTransform(_s_fit_,
                                 (w - _vw_ * _s_fit_) / 2.0 - self.vx0 * _s_fit_,
                                 (h - _vh_ * _s_fit_) / 2.0 - self.vy0 * _s_fit_)
        self._dl_body_ = _dl_body_

    # -------------------------------------------------------------------------
    # Small multiples / render_with
    # -------------------------------------------------------------------------

    def renderSmallMultiples(self, df_all: Any, df_lu: dict, all_key: Any) -> dict:
        return {k: SpreadLinesP(df=v, template=self) for k, v in df_lu.items()}

    def render_with(self, df: pl.DataFrame, **overrides: Any) -> Any:
        # `overrides` cannot be Unpack[SpreadLinesPKwargs]: PEP 692 rejects a TypedDict
        # that repeats a named parameter, and `df` is both.
        return SpreadLinesP(df=df, template=self, **overrides)
