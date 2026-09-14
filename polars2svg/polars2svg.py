import polars as pl
import polars.selectors as cs
import logging
import copy

from collections.abc import Mapping
from typing import Any, Unpack, TYPE_CHECKING
import re

from .exceptions            import Polars2SVGError, InvalidSpecError
from .p2s_colors_mixin      import P2SColorsMixin
from .p2s_geometry_mixin    import P2SGeometryMixin
from .p2s_graph_mixin       import P2SGraphMixin
from .p2s_polars_mixin      import P2SPolarsMixin
from .p2s_render_mixin      import P2SRenderMixin
from .p2s_text_mixin        import P2STextMixin
from .p2s_time_mixin        import P2STimeMixin
from .p2s_interactive_mixin import P2SInteractiveMixin
from .p2s_legend_mixin      import P2SLegendMixin
from .p2s_background_mixin  import BackgroundShape as _BackgroundShape_, INHERIT as _INHERIT_
from . import p2s_enums as _enums_
from .xyp                import XYp, XYpKwargs
from .smallp             import Smallp, SmallpKwargs
from .timep              import Timep, TimepKwargs
from .histop             import Histop, HistopKwargs
from .piep               import Piep, PiepKwargs
from .linkp              import LinkP, LinkPKwargs
from .spreadlinesp       import SpreadLinesP, SpreadLinesPKwargs
from .tile               import Tile, TileKwargs

# ChP (chordp.py) needs scipy for its core node-ordering algorithm (unlike the
# other components, this isn't an optional feature within chordp — every chord
# render uses it), so scipy is an optional 'layouts' dependency rather than
# core. Importing ChP eagerly here would make `import polars2svg` itself
# require scipy; instead it's imported lazily inside chordp()/isTemplate(),
# and only for static type checking here.
if TYPE_CHECKING:
    from .chordp import ChP, ChPKwargs

#
# _copy_mutable_containers_() - structural copy for template cloning: dict / set / list
# containers are copied (recursively for dict values and list items; set elements are
# hashable and therefore leaves), everything else is shared by reference.  The memo is
# keyed by id() so aliased containers stay aliased in the copy and self-referential
# containers terminate; callers must keep the originals alive while it runs (the
# template's __dict__ does).
#
def _copy_mutable_containers_(_value_: Any, _memo_: dict) -> Any:
    _vid_ = id(_value_)
    if _vid_ in _memo_: return _memo_[_vid_]
    if isinstance(_value_, dict):
        _copied_: Any = copy.copy(_value_)   # copy.copy preserves dict/set/list subclasses
        _memo_[_vid_] = _copied_
        for _k_ in _copied_: _copied_[_k_] = _copy_mutable_containers_(_copied_[_k_], _memo_)
        return _copied_
    if isinstance(_value_, list):
        _copied_ = copy.copy(_value_)
        _memo_[_vid_] = _copied_
        for _i_ in range(len(_copied_)): _copied_[_i_] = _copy_mutable_containers_(_copied_[_i_], _memo_)
        return _copied_
    if isinstance(_value_, set):
        _copied_ = copy.copy(_value_)
        _memo_[_vid_] = _copied_
        return _copied_
    return _value_


#
# OnceFilter - lets each distinct log message through exactly once.  Module-level
# (rather than local to __init__) so init code can recognize and strip stale
# instances from the shared 'polars2svg_logger' -- matched by class name, since a
# module reload would make isinstance() miss filters added by the old module object.
#
class OnceFilter(logging.Filter):
    def __init__(self, name: str = '', seen: set | None = None) -> None:
        super().__init__(name)
        # seen= carries the already-emitted messages over from the filter this one
        # replaces.  The install runs on every Polars2SVG.__init__, so starting from an
        # empty set would re-open every "warn once" message each time an instance is
        # built -- which used to be invisible only because there was never a second one.
        self.seen_messages: set = set() if seen is None else seen
    def filter(self, record: Any) -> bool:
        if record.msg not in self.seen_messages:
            self.seen_messages.add(record.msg)
            return True
        return False


class Polars2SVG(P2SColorsMixin,
                 P2SGeometryMixin,
                 P2SGraphMixin,
                 P2SPolarsMixin,
                 P2SRenderMixin,
                 P2STextMixin,
                 P2STimeMixin,
                 P2SInteractiveMixin,
                 P2SLegendMixin):
    '''
    Polars2SVG — render polars DataFrames directly to SVG.

    The entry point to the library. Construct one instance and call a component
    factory method on it; each returns a component object that carries the rendered
    SVG string in its ``.svg`` attribute and displays itself inline in Jupyter via
    ``_repr_svg_``::

        import polars as pl
        from polars2svg import Polars2SVG

        p2s = Polars2SVG()
        df  = pl.DataFrame({'x': [1, 2, 3], 'y': [3, 1, 4], 'group': ['a', 'b', 'a']})

        chart = p2s.xyp(df, 'x', 'y', color='group', wxh=(400, 300))
        open('scatter.svg', 'w').write(chart.svg)   # or just `chart` in a notebook

    Construction
    ------------
    ``Polars2SVG()`` takes no arguments and returns a **new, independent instance**.
    Configuration lives on the instance, not in the process: ``set_defaults()``,
    ``reset_defaults()`` and ``setColorOverrides()`` apply to every figure built from
    *that* instance and to no other. Configure one instance once and everything you
    render from it shares a style; build a second instance when you want a second style
    (or a second user's session) that cannot be reached by the first.

    Each factory method hands the component the instance it was called on (``p2s=``), so
    a component always resolves its defaults and color overrides against its own caller.
    Constructing a component directly (``XYp(df, ...)``) passes no instance and gets a
    fresh, unconfigured one — pass ``p2s=`` to share a configured instance with it.
    The instance exposes every enum member as an attribute by name, so
    ``p2s.ROW_COUNTp``, ``p2s.CSETp``, ``p2s.BARCHARTp`` etc. are all available
    without importing the enum classes.

    Component factory methods
    -------------------------
    ``xyp``          — scatter / distribution plot (x and y numeric or categorical)
    ``histop``       — horizontal histogram, one bar per category/bin
    ``timep``        — temporal bar chart (linear or periodic time modes)
    ``piep``         — pie / donut / waffle chart
    ``linkp``        — node-link graph / network with pluggable layouts
    ``chordp``       — chord diagram of weighted flows around a circle
    ``spreadlinesp`` — egocentric influence-over-time (SpreadLine) layout
    ``smallp``       — small multiples / trellis of one template component
    ``tile``         — compose finished renderings into a strip or a grid (no DataFrame)
    Interactive, cross-linked variants share the same signatures via ``xypi``,
    ``histopi``, ``timepi``, ``linkpi``, ``smallpi`` and are composed into a
    dashboard with ``panelize()``.

    Two parameters recur across the counting components and are **orthogonal**:
    ``count=`` controls size/magnitude, ``color=`` controls color. See the per-method
    docstrings and the project ``CLAUDE.md`` for the authoritative ``count=``/``color=``
    semantics tables.

    Global defaults
    ---------------
    ``set_defaults(**kwargs)`` sets defaults for all components; ``set_defaults(component,
    **kwargs)`` sets per-component defaults. Explicit kwargs at call time always win.

    Transformation fields
    ---------------------
    ``tField(column, enum)`` builds a time-transformation field (see ``TField``) for
    binning a timestamp column by period (month, day-of-week, …) in ``timep``/``smallp``.
    '''
    # _COMPONENT_KWARGS_ / _VALID_COMPONENTS_ are properties rather than plain
    # class attributes so that resolving ChP._VALID_KWARGS (which requires the
    # optional 'layouts' extra) is deferred to first access instead of paying
    # for it at `import polars2svg` time. 'chordp' is simply absent from both
    # when scipy isn't installed.
    @property
    def _COMPONENT_KWARGS_(self) -> dict:
        _kwargs_ = {'histop':       Histop._VALID_KWARGS,
                    'timep':        Timep._VALID_KWARGS,
                    'xyp':          XYp._VALID_KWARGS,
                    'linkp':        LinkP._VALID_KWARGS,
                    'smallp':       Smallp._VALID_KWARGS,
                    'spreadlinesp': SpreadLinesP._VALID_KWARGS,
                    'piep':         Piep._VALID_KWARGS}
        try:
            from .chordp import ChP
            _kwargs_['chordp'] = ChP._VALID_KWARGS
        except ImportError:
            pass
        return _kwargs_

    @property
    def _VALID_COMPONENTS_(self) -> frozenset:
        return frozenset(self._COMPONENT_KWARGS_)

    # Separator used to concatenate multi-field bin / color values into a single
    # internal grouping-key string ('__bin__' / '__color__'). It must never appear
    # in real data, otherwise two distinct field tuples can collapse into the same
    # key: with a printable '|', ('a|b', 'c') and ('a', 'b|c') both become 'a|b|c'
    # and merge into one bar/slice/segment. ASCII US (Unit Separator, 0x1f) is the
    # purpose-built, non-printable choice. It is stripped back to a visible '|' at
    # every text-display site via formatMultiFieldValue(), so labels are unchanged.
    MULTI_FIELD_SEP      = '\x1f'

    # Prefix of the node name given to a null relationship endpoint that
    # linkp(null_nodes=True) materializes. The prefix is followed by the name of the
    # entity on the other end, so every entity gets its OWN null partner: two records
    # with a missing endpoint are not asserted to point at the same thing. A single
    # shared null node would instead make every such entity one connected component --
    # neighbor expansion from it would select all of them, community detection would
    # group them, and force layout would ball them up. Like MULTI_FIELD_SEP this is
    # built from the non-printable ASCII US (0x1f) rather than a readable 'None'/'null'
    # /'', all of which are plausible values in real data and would collide with it.
    # nullNodeDisplay() restores a readable '(null)' at every text-display site.
    NULL_NODE_PREFIX     = '\x1fNULL\x1f'

    # The enumerations live in p2s_enums.py so that the mixins -- every one of which
    # is upstream of this module in the import graph -- can name their types instead
    # of declaring them `Any`.  Re-exposed here as class attributes so p2s.ColorTypeP
    # and Polars2SVG.ColorTypeP keep resolving exactly as they did when these were
    # nested class definitions.  Annotations elsewhere in this file must use the
    # module-qualified `_enums_.X` spelling: mypy rejects a class *variable* as a type.
    FieldTypeP        = _enums_.FieldTypeP
    StatisticP        = _enums_.StatisticP
    ColorTypeP        = _enums_.ColorTypeP
    TimeLinearTypeP   = _enums_.TimeLinearTypeP
    TimePeriodicTypeP = _enums_.TimePeriodicTypeP

    # enum <-> suffix lookup tables -- class attributes (rather than built in __init__) so
    # TField.__new__ can resolve a suffix without needing a Polars2SVG instance.
    _ENUM_TO_SUFFIX_ = {
        TimeLinearTypeP.LT_Yp:              'Yp',
        TimeLinearTypeP.LT_Y_Qp:            'Y_Qp',
        TimeLinearTypeP.LT_Y_mp:            'Y_mp',
        TimeLinearTypeP.LT_Y_m_dp:          'Y_m_dp',
        TimeLinearTypeP.LT_Y_m_d_4Hp:       'Y_m_d_4Hp',
        TimeLinearTypeP.LT_Y_m_d_Hp:        'Y_m_d_Hp',
        TimeLinearTypeP.LT_Y_m_d_H_15Mp:    'Y_m_d_H_15Mp',
        TimeLinearTypeP.LT_Y_m_d_H_Mp:      'Y_m_d_H_Mp',
        TimeLinearTypeP.LT_Y_m_d_H_M_15Sp:  'Y_m_d_H_M_15Sp',
        TimeLinearTypeP.LT_Y_m_d_H_M_Sp:    'Y_m_d_H_M_Sp',
        TimePeriodicTypeP.PT_Qp:           'Qp',
        TimePeriodicTypeP.PT_mp:           'mp',
        TimePeriodicTypeP.PT_m_dp:         'm_dp',
        TimePeriodicTypeP.PT_m_d_Hp:       'm_d_Hp',
        TimePeriodicTypeP.PT_DoYp:         'DoYp',
        TimePeriodicTypeP.PT_DoWp:         'DoWp',
        TimePeriodicTypeP.PT_DoW_Hp:       'DoW_Hp',
        TimePeriodicTypeP.PT_DoW_H_Mp:     'DoW_H_Mp',
        TimePeriodicTypeP.PT_dp:           'dp',
        TimePeriodicTypeP.PT_d_Hp:         'd_Hp',
        TimePeriodicTypeP.PT_d_H_Mp:       'd_H_Mp',
        TimePeriodicTypeP.PT_Hp:           'Hp',
        TimePeriodicTypeP.PT_H_Mp:         'H_Mp',
        TimePeriodicTypeP.PT_H_M_Sp:       'H_M_Sp',
        TimePeriodicTypeP.PT_Mp:           'Mp',
        TimePeriodicTypeP.PT_M_Sp:         'M_Sp',
        TimePeriodicTypeP.PT_Sp:           'Sp',
    }
    _SUFFIX_TO_ENUM_: dict = {}
    for _k_, _v_ in _ENUM_TO_SUFFIX_.items():
        if _v_ in _SUFFIX_TO_ENUM_: raise Polars2SVGError(f'polars2svg.Polars2SVG() - Collision between enum {_k_} and {_SUFFIX_TO_ENUM_[_v_]}')
        _SUFFIX_TO_ENUM_[_v_] = _k_
    del _k_, _v_

    #
    # Background records (see bgShape() below).  Exposed on the class so the p2s.INHERIT /
    # p2s.BackgroundShape spelling the docstrings use is real -- matching how HexColorString
    # and the render enums are reached -- and re-exported from polars2svg/__init__.py for
    # `from polars2svg import INHERIT`.  Imported under private aliases because binding the
    # public names here shadows them for the rest of the class body, and mypy will not accept
    # a class variable as the return annotation on bgShape().
    #
    INHERIT         = _INHERIT_
    BackgroundShape = _BackgroundShape_

    #
    # TField - typed replacement for the magic 'column|suffix' t-field string.
    # - subclasses str so its value *is* the legacy alias ('column|suffix'): every
    #   downstream consumer that keys on the string (df[...], .alias(...), 'x in
    #   df.columns', axis-label f-strings, isinstance(x, str) dispatch) keeps working
    #   unchanged, while callers can also do isinstance(x, TField) / x.column / x.transform.
    #
    class TField(str):
        '''A typed time-transformation field: pairs a timestamp ``column`` with a
        ``transform`` (a ``TimeLinearTypeP`` or ``TimePeriodicTypeP`` member) so a
        component bins that column by period instead of using it raw.

        Construct one with ``p2s.tField(column, enum)`` rather than calling this
        directly. ``TField`` subclasses ``str`` and its value *is* the legacy
        ``'column|suffix'`` alias, so it works anywhere a plain column string does
        (DataFrame lookups, tuples, sets, f-strings) while never being mistaken for a
        real column. Exposes ``.column`` and ``.transform`` attributes; instances are
        immutable.

        Example::

            tf = p2s.tField('timestamp', p2s.PT_DoWp)   # bin by day-of-week
            p2s.timep(df, tf)
        '''
        __slots__ = ('column', 'transform')
        column:    str   # the real column name
        transform: _enums_.TimeLinearTypeP | _enums_.TimePeriodicTypeP
        def __new__(cls, column: str, transform: _enums_.TimeLinearTypeP | _enums_.TimePeriodicTypeP) -> Any:
            if not isinstance(column, str): raise TypeError(f'polars2svg.TField(): column must be a string, got {type(column)}')
            if transform not in Polars2SVG._ENUM_TO_SUFFIX_: raise InvalidSpecError(f'polars2svg.tField(): unknown enumeration {transform}')
            _self_ = super().__new__(cls, column + '|' + Polars2SVG._ENUM_TO_SUFFIX_[transform])
            str.__setattr__(_self_, 'column',    column)
            str.__setattr__(_self_, 'transform', transform)
            return _self_
        def __setattr__(self, k: str, v: Any) -> None: raise AttributeError('TField is immutable')
        @property
        def alias(self) -> str: return str(self)
        def __repr__(self) -> str: return f'TField({self.column!r}, {self.transform})'

    # The thirteen classes that replaced the RenderEnumsP grab-bag, plus the tuple
    # and union alias that still mean "any render enum".
    RowCountP              = _enums_.RowCountP
    DistributionPlacementP = _enums_.DistributionPlacementP
    DistributionScaleP     = _enums_.DistributionScaleP
    LineWidthP             = _enums_.LineWidthP
    LineStyleP             = _enums_.LineStyleP
    LineColorP             = _enums_.LineColorP
    LineOpacityP           = _enums_.LineOpacityP
    SmallMultipleP         = _enums_.SmallMultipleP
    BarStyleP              = _enums_.BarStyleP
    SelectShapeP           = _enums_.SelectShapeP
    NodeColorP             = _enums_.NodeColorP
    PieStyleP              = _enums_.PieStyleP
    OrderBucketP           = _enums_.OrderBucketP
    RENDER_ENUM_CLASSES    = _enums_.RENDER_ENUM_CLASSES
    RenderEnum             = _enums_.RenderEnum

    # ---------------------------------------------------------------------
    # Enum members bound onto the instance by the setattr() loops in __init__.
    #
    # Declarations, not assignments: a bare annotation populates __annotations__
    # and creates no class attribute, so this block is a no-op at runtime.  It
    # exists so type checkers can see the p2s.SCALARp / p2s.PT_DoWp / p2s.BARCHARTp
    # attributes that __init__ binds dynamically -- without it mypy reports 637
    # false "Polars2SVG has no attribute" errors across the package.
    #
    # tests/test_typing_surface.py::TestEnumMemberDeclarations asserts this block
    # matches the loops exactly, so a member added to an enum above fails the
    # suite until it is declared here too.
    # ---------------------------------------------------------------------
    # FieldTypeP
    SCALARp: _enums_.FieldTypeP
    SETp:    _enums_.FieldTypeP

    # StatisticP
    MINp:    _enums_.StatisticP
    MEDIANp: _enums_.StatisticP
    MEANp:   _enums_.StatisticP
    MAXp:    _enums_.StatisticP
    STDp:    _enums_.StatisticP
    SUMp:    _enums_.StatisticP

    # ColorTypeP
    CSETp:              _enums_.ColorTypeP
    CSET_MAGNITUDEp:    _enums_.ColorTypeP
    CSET_STRETCHEDp:    _enums_.ColorTypeP
    CROW_MAGNITUDEp:    _enums_.ColorTypeP
    CROW_STRETCHEDp:    _enums_.ColorTypeP
    CMAGNITUDE_SUMp:    _enums_.ColorTypeP
    CMAGNITUDE_MINp:    _enums_.ColorTypeP
    CMAGNITUDE_MEDIANp: _enums_.ColorTypeP
    CMAGNITUDE_MEANp:   _enums_.ColorTypeP
    CMAGNITUDE_MAXp:    _enums_.ColorTypeP
    CSTRETCHED_SUMp:    _enums_.ColorTypeP
    CSTRETCHED_MINp:    _enums_.ColorTypeP
    CSTRETCHED_MEDIANp: _enums_.ColorTypeP
    CSTRETCHED_MEANp:   _enums_.ColorTypeP
    CSTRETCHED_MAXp:    _enums_.ColorTypeP

    # TimeLinearTypeP
    LT_Yp:             _enums_.TimeLinearTypeP
    LT_Y_Qp:           _enums_.TimeLinearTypeP
    LT_Y_mp:           _enums_.TimeLinearTypeP
    LT_Y_m_dp:         _enums_.TimeLinearTypeP
    LT_Y_m_d_Hp:       _enums_.TimeLinearTypeP
    LT_Y_m_d_H_Mp:     _enums_.TimeLinearTypeP
    LT_Y_m_d_H_M_Sp:   _enums_.TimeLinearTypeP
    LT_Y_m_d_4Hp:      _enums_.TimeLinearTypeP
    LT_Y_m_d_H_15Mp:   _enums_.TimeLinearTypeP
    LT_Y_m_d_H_M_15Sp: _enums_.TimeLinearTypeP

    # TimePeriodicTypeP
    PT_Qp:       _enums_.TimePeriodicTypeP
    PT_mp:       _enums_.TimePeriodicTypeP
    PT_m_dp:     _enums_.TimePeriodicTypeP
    PT_m_d_Hp:   _enums_.TimePeriodicTypeP
    PT_DoYp:     _enums_.TimePeriodicTypeP
    PT_DoWp:     _enums_.TimePeriodicTypeP
    PT_DoW_Hp:   _enums_.TimePeriodicTypeP
    PT_DoW_H_Mp: _enums_.TimePeriodicTypeP
    PT_dp:       _enums_.TimePeriodicTypeP
    PT_d_Hp:     _enums_.TimePeriodicTypeP
    PT_d_H_Mp:   _enums_.TimePeriodicTypeP
    PT_Hp:       _enums_.TimePeriodicTypeP
    PT_H_Mp:     _enums_.TimePeriodicTypeP
    PT_H_M_Sp:   _enums_.TimePeriodicTypeP
    PT_Mp:       _enums_.TimePeriodicTypeP
    PT_M_Sp:     _enums_.TimePeriodicTypeP
    PT_Sp:       _enums_.TimePeriodicTypeP

    # The render enums (RowCountP .. OrderBucketP)
    ROW_COUNTp:                          _enums_.RowCountP
    DISTRIBUTION_INSIDEp:                _enums_.DistributionPlacementP
    DISTRIBUTION_OUTSIDEp:               _enums_.DistributionPlacementP
    DISTRIBUTION_AUTOBINp:               _enums_.DistributionPlacementP
    DISTRIBUTION_COLOR_MIN_TO_COLOR_MAX: _enums_.DistributionScaleP
    DISTRIBUTION_ZERO_TO_COLOR_MAX:      _enums_.DistributionScaleP
    DISTRIBUTION_ALL_MIN_TO_ALL_MAX:     _enums_.DistributionScaleP
    DISTRIBUTION_ZERO_TO_ALL_MAX:        _enums_.DistributionScaleP
    LINEWIDTH_DOTSIZE_MEAN:              _enums_.LineWidthP
    LINEWIDTH_DOTSIZE_VARIABLE:          _enums_.LineWidthP
    LINEWIDTH_DOTSIZE_SPECIFIED:         _enums_.LineWidthP
    LINESTYLE_SOLID:                     _enums_.LineStyleP
    LINESTYLE_DOTTED:                    _enums_.LineStyleP
    LINESTYLE_SPECIFIED:                 _enums_.LineStyleP
    LINECOLOR_GROUPBY:                   _enums_.LineColorP
    LINECOLOR_FIELD:                     _enums_.LineColorP
    LINECOLOR_SPECIFIED:                 _enums_.LineColorP
    LINEOPACITY_FIELD_MEAN:              _enums_.LineOpacityP
    LINEOPACITY_FIELD_VARIABLE:          _enums_.LineOpacityP
    LINEOPACITY_100:                     _enums_.LineOpacityP
    LINEOPACITY_75:                      _enums_.LineOpacityP
    LINEOPACITY_50:                      _enums_.LineOpacityP
    LINEOPACITY_25:                      _enums_.LineOpacityP
    LINEOPACITY_10:                      _enums_.LineOpacityP
    SM_X:                                _enums_.SmallMultipleP
    SM_Y:                                _enums_.SmallMultipleP
    SM_COUNT:                            _enums_.SmallMultipleP
    SM_COLOR:                            _enums_.SmallMultipleP
    BARCHARTp:                           _enums_.BarStyleP
    BOXPLOTp:                            _enums_.BarStyleP
    BOXPLOT_W_SWARMp:                    _enums_.BarStyleP
    STACKEDBARp:                         _enums_.BarStyleP
    SELECT_CIRCLEp:                      _enums_.SelectShapeP
    SELECT_HORIZONTALp:                  _enums_.SelectShapeP
    SELECT_VERTICALp:                    _enums_.SelectShapeP
    COLOR_BY_NODE_NAME:                  _enums_.NodeColorP
    PIEp:                                _enums_.PieStyleP
    DONUTp:                              _enums_.PieStyleP
    WAFFLEp:                             _enums_.PieStyleP
    SM_SLICE_ORDERp:                     _enums_.SmallMultipleP
    SM_PARTOFWHOLEp:                     _enums_.SmallMultipleP
    REMAINDERp:                          _enums_.OrderBucketP

    def __init__(self) -> None:
        # Every Polars2SVG() builds its own instance, so everything below is per-instance
        # state that runs exactly once for it.  The one thing that is NOT per-instance is
        # the 'polars2svg_logger' further down -- that logger is module-global and shared
        # by every instance, which is why its filter needs the care it gets there.
        self._global_defaults: dict    = {}
        self._component_defaults: dict = {}

        # Assign all enum members as instance attributes by name.  The registry in
        # p2s_enums.py is the single source of truth for which classes take part;
        # the loop variable is shared across all of them, so it is declared rather
        # than inferred.
        _m_: Any
        for _enum_cls_ in _enums_.ENUM_CLASSES:
            for _m_ in _enum_cls_: setattr(self, _m_.name, _m_)

        # Label rendered for the REMAINDERp bucket (arc name / axis tick).  A listed
        # order= value equal to this string collides with the bucket and raises --
        # see ChP.__resolveOrder__() / XYp.__resolveOrder__().
        self.REMAINDER_LABEL     = 'remainder'

        self.statistic_types     = set(self.StatisticP)
        self.color_types         = set(self.ColorTypeP)
        self.time_linear_types   = set(self.TimeLinearTypeP)
        self.time_periodic_types = set(self.TimePeriodicTypeP)

        self.periodic_ranges = {}
        self.periodic_ranges[self.PT_Qp          ] = (1,     4)          # quarters ... i.e., 3 months
        self.periodic_ranges[self.PT_mp          ] = (1,     12)         # months
        self.periodic_ranges[self.PT_m_dp        ] = (1,     366)        # days
        self.periodic_ranges[self.PT_m_d_Hp      ] = (24,    367*24-1)   # hours
        self.periodic_ranges[self.PT_DoYp        ] = (1,     366)        # days
        self.periodic_ranges[self.PT_DoWp        ] = (1,     7)          # days
        self.periodic_ranges[self.PT_DoW_Hp      ] = (24,    24*8-1)     # hours
        self.periodic_ranges[self.PT_DoW_H_Mp    ] = (24*60, 24*60*8-1)  # minutes
        self.periodic_ranges[self.PT_dp          ] = (1,     31)         # days
        self.periodic_ranges[self.PT_d_Hp        ] = (24,    24*32-1)    # hours
        self.periodic_ranges[self.PT_d_H_Mp      ] = (24*60, 32*24*60-1) # minutes
        self.periodic_ranges[self.PT_Hp          ] = (0,     23)         # hours
        self.periodic_ranges[self.PT_H_Mp        ] = (0,     24*60-1)    # minutes
        self.periodic_ranges[self.PT_H_M_Sp      ] = (0,     24*60*60-1) # seconds
        self.periodic_ranges[self.PT_Mp          ] = (0,     59)         # minutes
        self.periodic_ranges[self.PT_M_Sp        ] = (0,     60*60-1)    # seconds
        self.periodic_ranges[self.PT_Sp          ] = (0,     59)         # seconds

        # Kept as instance sets for back-compat; the classes are now the source of truth.
        self.distribution_types = set(_enums_.DistributionPlacementP) | set(_enums_.DistributionScaleP)

        self.line_types = (set(_enums_.LineWidthP) | set(_enums_.LineStyleP)
                           | set(_enums_.LineColorP) | set(_enums_.LineOpacityP))

        self.all_enums   = {self.SCALARp,
                            self.SETp,
                            self.ROW_COUNTp,
                           } | self.statistic_types     \
                             | self.color_types         \
                             | self.distribution_types  \
                             | self.line_types

        # enum lookup tables (built once at class-definition time as _ENUM_TO_SUFFIX_ /
        # _SUFFIX_TO_ENUM_ -- kept here as instance aliases for back-compat)
        self.enum_to_suffix = self._ENUM_TO_SUFFIX_
        self.suffix_to_enum = self._SUFFIX_TO_ENUM_

        # Setup the logging / use a filter to only show a message once.  Instances are
        # per-caller but this logger is not -- logging.getLogger() returns the same
        # module-global object to all of them -- so without care every Polars2SVG() would
        # stack another OnceFilter onto it.  Strip stale instances so exactly one is ever
        # installed, and carry their seen set into the replacement so "once" means once
        # per process rather than once per instance.
        self.logger = logging.getLogger('polars2svg_logger')
        _seen_: set = set()
        for _filter_ in [f for f in self.logger.filters if type(f).__name__ == 'OnceFilter']:
            _seen_ |= getattr(_filter_, 'seen_messages', set())
            self.logger.removeFilter(_filter_)
        self.logger.addFilter(OnceFilter(seen=_seen_))

        # Initialize the mixins
        self.__p2s_colors_mixin_init__()
        self.__p2s_geometry_mixin_init__()
        self.__p2s_graph_mixin_init__()
        self.__p2s_polars_mixin_init__()
        self.__p2s_render_mixin_init__()
        self.__p2s_text_mixin_init__()
        self.__p2s_time_mixin_init__()
        self.__p2s_interactive_mixin_init__()
        self.__p2s_legend_mixin_init__()

    # ------------------------------------------------------------------
    # Global configuration defaults
    # ------------------------------------------------------------------

    def set_defaults(self, *args: Any, **kwargs: Any) -> None:
        '''set_defaults(**kwargs) — set global defaults for all components.
        set_defaults(component, **kwargs) — set per-component defaults.
        Per-component defaults override global defaults; explicit kwargs always win.
        Kwargs are validated eagerly: per-component defaults must be accepted by that
        component; global defaults must be accepted by at least one component.
        Templates: defaults are resolved once, when a component is built from scratch.
        A component built from a template= is an exact snapshot of the template's
        resolved state — defaults set or changed after the template was created do not
        apply to its clones (explicit kwargs at clone time still win). smallp's
        sm_template is a panel template, not a clone source, so smallp itself always
        picks up current defaults.'''
        if args and isinstance(args[0], str):
            component = args[0]
            if component not in self._VALID_COMPONENTS_:
                raise ValueError(f'Polars2SVG.set_defaults(): unknown component "{component}". '
                                 f'Valid components: {sorted(self._VALID_COMPONENTS_)}')
            _unknown_ = set(kwargs) - self._COMPONENT_KWARGS_[component]
            if _unknown_:
                raise TypeError(f'Polars2SVG.set_defaults(): unexpected keyword argument(s) for "{component}": '
                                f'{sorted(_unknown_)}. Valid kwargs: {sorted(self._COMPONENT_KWARGS_[component])}')
            self._component_defaults.setdefault(component, {}).update(kwargs)
        else:
            _all_kwargs_ = frozenset().union(*self._COMPONENT_KWARGS_.values())
            _unknown_    = set(kwargs) - _all_kwargs_
            if _unknown_:
                raise TypeError(f'Polars2SVG.set_defaults(): keyword argument(s) not accepted by any component: '
                                f'{sorted(_unknown_)}')
            self._global_defaults.update(kwargs)

    def reset_defaults(self, component: str | None = None) -> None:
        '''reset_defaults() — clear all global and component defaults.
        reset_defaults(component) — clear defaults for a single component only.'''
        if component is None:
            self._global_defaults.clear()
            self._component_defaults.clear()
        elif component in self._VALID_COMPONENTS_:
            self._component_defaults.pop(component, None)
        else:
            raise ValueError(f'Polars2SVG.reset_defaults(): unknown component "{component}". '
                             f'Valid components: {sorted(self._VALID_COMPONENTS_)}')

    def get_defaults(self) -> dict:
        '''Return a snapshot of current global and per-component defaults.'''
        return {'_global': dict(self._global_defaults),
                **{k: dict(v) for k, v in self._component_defaults.items()}}

    def _apply_defaults(self, component_name: str, kwargs: Mapping[str, Any]) -> dict:
        '''Merge global defaults, then component defaults, then explicit kwargs.
        Explicit kwargs always win; hardcoded component defaults remain the last fallback.
        Global defaults valid for only some components are filtered to what this
        component accepts (a global bin_by= applies to histop/piep but never leaks into xyp).
        Components call this only when building from scratch — never when cloning from a
        template=. A template clone is an exact snapshot of the template's resolved state
        (defaults were already baked in when the template was constructed); re-applying
        current defaults over that snapshot would break template reproducibility. See
        tests/test_template_defaults.py for the precedence contract:
        explicit kwargs > template snapshot > defaults-at-template-creation.'''
        _valid_ = self._COMPONENT_KWARGS_.get(component_name)
        merged  = {}
        for _key_, _value_ in self._global_defaults.items():
            if _valid_ is None or _key_ in _valid_: merged[_key_] = _value_
        merged.update(self._component_defaults.get(component_name, {}))
        merged.update(kwargs)
        return merged

    # Per-instance lifecycle state: freshly initialized by every component's own
    # __init__ before __parseInput__ runs, and never part of the template snapshot.
    # 'p2s' is here for the same reason -- it is the clone's wiring to the instance that
    # built it, not resolved render state.  Without the skip the __dict__ copy below
    # would silently replace it with the template's instance, so a clone built by one
    # instance from another's template would resolve defaults and color overrides
    # against the template's owner rather than its own caller.
    _TEMPLATE_CLONE_SKIP_ = frozenset({'timing_metrics', 't_start', 't_end', 't_overall', 'p2s'})

    def _clone_template_state(self, target: Any, template: Any) -> None:
        '''Copy a template component's resolved state onto a fresh clone.

        Replaces the old `target.__dict__.update(template.__dict__)` pattern, which
        shared every mutable attribute between template and clone: the clone's fresh
        timing_metrics dict was replaced by the template's (so clone renders
        accumulated timing into the template), t_start was clobbered (so the clone's
        t_overall spanned the template's lifetime too), and any in-place mutation of
        a shared container (sm_shared, background dicts, pos) corrupted the template
        and every sibling clone.

        Mutable containers (dict / set / list) are copied recursively; leaf objects
        (DataFrames, shapely geometry, enums, ...) are shared by reference — they are
        treated as immutable by the framework. Aliasing among containers is preserved
        (two attributes referencing one dict reference one copied dict). Attributes in
        _TEMPLATE_CLONE_SKIP_ are per-instance and keep the clone's own fresh values.'''
        _memo_: dict = {}
        for _key_, _value_ in template.__dict__.items():
            if _key_ in self._TEMPLATE_CLONE_SKIP_: continue
            target.__dict__[_key_] = _copy_mutable_containers_(_value_, _memo_)

    # ── Shared __parseInput__ helpers ─────────────────────────────────────────
    # Every component's __parseInput__ used to hand-copy the same two loops: a
    # from-scratch default block (`self.x = <default>`) and a keyword-override
    # block (`if 'x' in kwargs: self.x = kwargs['x']`), one line per parameter in
    # each. Because the two blocks listed the parameters independently they drifted
    # (e.g. xyp accepted `use_lazy_execution` but never read it from the merged
    # kwargs; the hex-color handling diverged the same way). These helpers make a
    # single per-component `defaults` mapping (name -> from-scratch value) the sole
    # source of truth for BOTH phases, so a declared-but-forgotten parameter is
    # structurally impossible.

    # Component names whose param spec has already been checked this process; the
    # equality check is invariant per class, so it only needs to run once.
    _PARAM_SPEC_VERIFIED_: set = set()

    def assertParamSpecMatches(self, component_name: str, valid_kwargs: frozenset, defaults: dict, extra: tuple = ('df', 'template')) -> None:
        '''Structural drift guard: the parameter names in `defaults` plus `extra`
        (the handful of args handled outside the spec — `df`, `template`, and any
        component-specific positionals) must exactly equal `_VALID_KWARGS`. Because
        `defaults` drives both default assignment (`assignScratchDefaults`) and the
        keyword-override copy (`assignKwargOverrides`), this equality makes both
        "kwarg accepted but never assigned" and "attribute assigned but not accepted"
        impossible to introduce silently. Runs once per component per process.'''
        if component_name in self._PARAM_SPEC_VERIFIED_: return
        _spec_ = set(defaults) | set(extra)
        _valid_ = set(valid_kwargs)
        if _spec_ != _valid_:
            raise RuntimeError(
                f'{component_name}: parameter-spec drift — '
                f'in _VALID_KWARGS only: {sorted(_valid_ - _spec_)}; '
                f'in defaults/extra only: {sorted(_spec_ - _valid_)}')
        self._PARAM_SPEC_VERIFIED_.add(component_name)

    def assignScratchDefaults(self, target: Any, defaults: dict) -> None:
        '''From-scratch (non-template) default assignment: write every
        `name -> value` in `defaults` onto `target`. Paired with
        `assignKwargOverrides`, which reads the same mapping.'''
        for _name_, _value_ in defaults.items():
            setattr(target, _name_, _value_)

    def assignKwargOverrides(self, target: Any, defaults: dict, kwargs: Mapping[str, Any],
                             skip: set | tuple = ()) -> None:
        '''Copy any supplied keyword argument over the current attribute value for
        every parameter named in `defaults` (the same mapping `assignScratchDefaults`
        uses). Names in `skip` are handled explicitly by the caller (type coercion or
        a side effect) and are left untouched here.'''
        for _name_ in defaults:
            if _name_ in skip: continue
            if _name_ in kwargs: setattr(target, _name_, kwargs[_name_])

    def assignKwargsWithDefaults(self, target: Any, defaults: dict, kwargs: Mapping[str, Any]) -> None:
        '''Combined `kwargs.get(name, default)` assignment for components that have
        no from-scratch/template split (smallp): write `kwargs.get(name, default)`
        for every `name -> default` in `defaults`. Same single-source-of-truth
        guarantee as the scratch/override pair.'''
        for _name_, _default_ in defaults.items():
            setattr(target, _name_, kwargs.get(_name_, _default_))

    def webgpuHTML(self, component: Any, border: str = '1px solid #ccc') -> str:
        '''
        webgpuHTML(component)

        Render a component's WebGPU representation as a self-contained HTML string
        (canvas + inline runtime + buffers).  The component must support webgpu() --
        all eight static components do.  Display in a notebook with IPython.display.HTML.
        '''
        if getattr(component, 'webgpu', None) is None:
            raise ValueError(f'webgpuHTML(): component {type(component).__name__} has no webgpu() representation')
        from polars2svg.p2s_webgpu_runtime import standalone_html
        _payload_ = component.webgpu()
        if _payload_ is None:
            raise ValueError('webgpuHTML(): component has no rendered content')
        return standalone_html(_payload_, border=border)

    def xyp(self, *args: Any, **kwargs: Unpack[XYpKwargs]) -> XYp:
        '''
        xyp(polars.DataFrame, x, y, ...)

        A scatter / distribution plot. Each row becomes a dot at (x, y); x and y may be
        numeric or categorical (set-based). Optional distributions, connecting lines,
        and shapely background shapes layer on top. dot color/size/opacity, and line
        style are all independently data-drivable.

        Example::

            p2s.xyp(df, 'x', 'y', color='group', dot_size=6, wxh=(400, 300))

        x|y = 'field'
            = (field-name, sub-field-name, ...)
            = ['field', 'field2', ...]
            = [(field-name, sub-field-name, ...), (field-name2, sub-field-name2, ...), ...]

        For single field names (i.e., non-tuples), the field data-types should match (the frames will be vstacked).

        By default, tuples will be converted into structs, sorted, and assigned an integer value.

        Additionally, the x or y fields can be modified by the following enumerations:
        - polars2svg.SCALARp  # field contents will be treated as a scalar (default for ints and floats)
        - polars2svg.SETp     # field contents will be treated as a categorical datatype (default for strings or any other non-numeric type)
          (this value only needs to be specified once)
          e.g., ['field1', 'field2', polars2svg.SETp]
                ('field1', polars2svg.SETp)
        - these two enumerations should be specified in the list of fields
          e.g., ['field1', 'field2', polars2svg.SETp]
          e.g., ('field1', polars2svg.SETp)           # for a single field only
          e.g., ('field1', 'field2', polars2svg.SETp) # for a single tuple only
        - they should not be specified as follows: [('field1', polars2svg.SETp), ('field2', polars2svg.SETp)]

        If either x or y is a list, then the length of the field list must match the length of other arguments
        - The exception is that the other arguments can be a single value (e.g., one field-name)
        - If they are single values, then those values will be repeated to match the length of the field list

        template       = None                              # another XYp instance; copies all settings, then applies any overrides

        === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< === %<

        Order for categorical (set-based) axes can be set using the following variables:

        x_order|y_order

        if the axis consists of multiple fields, then the order should contain tuples of the field values.

        The order variables can either be a list of values or a dictionary that maps field values to a numeric value.

        Any field value not included in the order will be assigned a value of maximum plus one.

        x_order|y_order = 'spectral'                     # spectral (Fiedler) seriation of a categorical axis

        With 'spectral', the axis categories are ordered by the Fiedler vector (the
        second-smallest eigenvector of the graph Laplacian) of a category x category
        affinity matrix, so similar categories land adjacent and block structure lines
        up along the axis.  The axis must be categorical (string / struct field, or a
        SETp-tagged field) or a ValueError is raised.  The ordering is defined up to
        reflection.  It is tuned by:

            spectral_by         = None                   # signal column(s) defining category similarity;
                                                         #   None = the opposite axis's column(s).  Two
                                                         #   categories are similar when they co-occur with
                                                         #   the same partners.  e.g. spectral_by='color'
            spectral_weight     = None                   # numeric column summed into each contingency cell
                                                         #   (None = raw co-occurrence row counts)
            spectral_similarity = 'cosine'               # 'cosine' | 'linear' | 'correlation'
            spectral_normalize  = True                   # symmetric-normalized Laplacian

        Under small multiples, a 'spectral' order on a SHARED axis (SM_X / SM_Y in
        sm_shared) is computed ONCE over the full dataset and applied identically to
        every tile, so panels stay comparable; an unshared 'spectral' axis is seriated
        independently per tile.

        === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< === %<

        color = None                                    # default — data colour for all dots
              = polars2svg.CROW_MAGNITUDEp | polars2svg.CROW_STRETCHEDp
              = 'field'
              = ('field', COLOR_ENUM)
              = ['field', 'field2', ...]
              = '#RRGGBB'
              = ['#RRGGBB', '#RRGGBB', ...]

        The following enumerations can be used with a field(s) to modify the color:
            - CSETp               # the field will be treated as a categorical field & if there's a unique value
                                  # at that pixel, it will be assigned that color
            - CSET_MAGNITUDEp     # the number of set elements at a pixel will be used within a spectrum
            - CSET_STRETCHEDp     # the number of set elements will be sorted monotonically and stretched across a spectrum
            - CMAGNITUDE_SUMp     # for numeric fields, the sum of values at a pixel will be used within a spectrum
            - CMAGNITUDE_MINp     # ... min value ...
            - CMAGNITUDE_MEDIANp  # ... median value ...
            - CMAGNITUDE_MEANp    # ... mean value ...
            - CMAGNITUDE_MAXp     # ... max value ...
            - CSTRETCHED_SUMp     # for numeric fields, the sum of values will be sorted monotonically and stretched across a spectrum
            - CSTRETCHED_MINp     # ... min value ...
            - CSTRETCHED_MEDIANp  # ... median value ...
            - CSTRETCHED_MEANp    # ... mean value ...
            - CSTRETCHED_MAXp     # ... max value ...

        The specific spectrum used may be modified via the p2s.spectrum_palette variable.

        === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< === %<

        dot size is specified as follows:

        dot_size = 1                                    # default — fixed pixel size (integer triggers grid pipeline)
                 = int | float
                 = polars2svg.ROW_COUNTp
                 = 'field'
                 = (field-name, sub-field-name, ...)
                 = ['field', 'field2', ...] # may be modified by polars2svg.SETp
                 = [(field-name, sub-field-name, ...), (field-name2, sub-field-name2, ...), ...]
                 = [int | float, int | float, ...]

        When the dot size is specified as an integer (vs a float), then the grid is constructed with squares
        of that specific size.  This may result in a extra (unused) space at the end of the grid.

        When the dot_size is specified by a field, then the following parameter controls the range:

        dot_size_range = (0.5, 4.0)                     # (min, max) radius when dot_size is field-driven

        === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< === %<

        dot opacity is specified as follows:

        opacity = None                                  # default — no opacity variation (all dots fully opaque)
                = polars2svg.ROW_COUNTp
                = float
                = 'field'
                = ('field-name', 'sub-field-name', ...)
                = ['field', 'field2', ...] # may be modified by polars2svg.SETp
                = [(field-name, sub-field-name, ...), (field-name2, sub-field-name2, ...), ...]
                = [float, float, ...]

        When the opacity is specified by a variable, then the following parameter controls the opacity range:

        opacity_range  = (0.5, 1.0)                     # (min, max) opacity when opacity is field-driven

        === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< === %<

        line

        individual lines are specified as follows with the renderer performing a group by operation on the line:

        line = 'field'
             = ('field', sub-field-name, ...)
             = ['field', 'field2', ...]
             = [('field', 'sub-field-name', ...), ('field2', 'sub-field-name2', ...), ...]

        By default the order of the line will be determined by the x-axis.  A custom order may be specified
        using the line_order_by parameter:

        line_order_by = 'field'
                      = ('field', 'sub-field-name', ...)
                      = ['field', 'field2', ...]

        Line width may be specified using the following enumerations:

        LINEWIDTH_DOTSIZE_MEAN           # line width will be the size field used for dots
        LINEWIDTH_DOTSIZE_VARIABLE       # line width will vary per point using the dot size field
        integer or floating-point number # (i.e., LINEWIDTH_DOTSIZE_SPECIFIED)

        Line style may be specified using the following enumerations:

        LINESTYLE_SOLID
        LINESTYLE_DOTTED
        list of integers                # (i.e., LINESTYLE_SPECIFIED)

        Line color may be specified using the following enumerations:

        LINECOLOR_GROUPBY               # line color will be the group-by key
        LINECOLOR_FIELD                 # line color will be the color field used for dots
        hex-color string                # (e.g., '#RRGGBB') (i.e., LINECOLOR_SPECIFIED)

        Line opacity may be specified using the following enumerations:

        LINEOPACITY_FIELD_MEAN          # line opacity will be the mean of the opacity field
        LINEOPACITY_FIELD_VARIABLE      # line opacity will be the opacity field used for dots
        LINEOPACITY_100
        LINEOPACITY_75
        LINEOPACITY_50
        LINEOPACITY_25
        LINEOPACITY_10

        === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< === %<

        distributions are specified as follows:

        x_distributions|y_distributions = polars2svg.ROW_COUNTp
                                        = (field-name, sub-field-name, ...)
                                        = ['field', 'field2', ...]
                                        = [(field-name, sub-field-name, ...), (field-name2, sub-field-name2, ...), ...]

        a single integer may be supplied within the list (or the tuple for a single field) that controls the number of bins
        - by default, if no integer is supplied, then the number of bins will be calculated automatically

        a single floating point value may be supplied within the list (or the tuple for a single field) that controls the
        height of the rendering as a percentage of the chart size

        hex colors may be included at either the list-level or within each tuple to force a specific color

        two enumerations control whether the distributions are rendered inside or outside the chart
        - polars2svg.DISTRIBUTION_INSIDEp  # for inside, either dot_size should be None or a transparent opacity should be used
        - polars2svg.DISTRIBUTION_OUTSIDEp

        lastly, the following rendering hints are available for controlling the relative height of the distributions:
        - polars2svg.DISTRIBUTION_COLOR_MIN_TO_COLOR_MAX # for multiple colors, the min and the max is based on the specific color
        - polars2svg.DISTRIBUTION_ZERO_TO_COLOR_MAX      # same but minimum is zero
        - polars2svg.DISTRIBUTION_ALL_MIN_TO_ALL_MAX     # minimum of all colors and maximum of all colors
        - polars2svg.DISTRIBUTION_ZERO_TO_ALL_MAX        # same but minimum is zero

        === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< === %<

        Render Options

        wxh            = (width, height)
        insets         = (x-inset, y-inset)   # if the plot is too small, the insets won't be drawn
        draw_context   = True (default) | False  # if the plot is too small, the context won't be drawn
        draw_border    = True (default) | False  # draw a rectangular border around the SVG
        txt_h          = label text height
        x_range        = (min, max)           # clip the x axis to this window (default: the data extent)
        y_range        = (min, max)           # clip the y axis to this window (default: the data extent)
        aspect         = None (default)       # each axis independently fills its own pixel extent
                       = 'equal'              # one world unit is the same length on both axes
                       = 'geo'                # x is degrees longitude, y is degrees latitude
                       = number               # pixels-per-y-unit / pixels-per-x-unit (matplotlib's set_aspect)

          Without aspect=, a dataset whose shape does not match the width/height ratio of
          the component is stretched to fit it -- which distorts geospatial data.  aspect=
          pins the ratio between the two axis scales by *widening* the over-magnified axis
          about its center, so the plot gains empty margin rather than losing any data;
          x_range/y_range become a floor for the visible window rather than a ceiling, and
          rows that the widening brings into view are drawn instead of leaving an empty
          band inside the frame.

          'geo' is a plate carree centered on the data: a degree of longitude covers
          cos(latitude) as much ground as a degree of latitude, so it is given cos(lat) as
          many pixels.  It is the usual good-enough correction for a regional extent, not a
          reprojection -- it will not stay true across a latitude span wide enough for the
          curvature to matter.

          Requires numeric x and y axes.  A categorical axis draws its grid lines from the
          category count and the temporal renderers work in world-unit datetimes, so neither
          would follow a widened window; both raise instead of rendering a plot whose dots
          and grid lines disagree.
        background             = None                       # no background (default)
                               = {key: shape_desc, ...}     # bare descriptor; inherits every background_* below
                               = {key: record, ...}         # record; carries its own appearance
                                                            # the two forms mix freely in one dict

          shape_desc           = shapely Polygon | MultiPolygon | LineString | MultiLineString
                               = [(x,y), ...]               # list of world-coordinate pairs
                               = '<circle cx=... r=... />'  # SVG circle string
                               = 'M x y L x y C ... Z'      # SVG path description (M/L/C/Z)

          record               = p2s.bgShape(shape_desc, fill=..., stroke=..., ...)
                               = {'shape': shape_desc, 'fill': ..., ...}   # same fields, plain dict

            shape              = shape_desc                 # required
            fill               = p2s.INHERIT (default)      # take background_fill
                               = None                       # explicitly UNFILLED -- emits fill="none"
                               = '#RRGGBB' | 'vary'
            fill_opacity       = p2s.INHERIT | None | float
            stroke             = p2s.INHERIT | None | '#RRGGBB' | 'vary'
            stroke_opacity     = p2s.INHERIT | None | float # NEW -- no sidecar equivalent
            stroke_width       = p2s.INHERIT | None | float
            dash               = p2s.INHERIT | None | '4 2' # NEW -- stroke-dasharray
            stroke_linecap     = p2s.INHERIT | None | 'butt' | 'round' | 'square'   # NEW, SVG only
            stroke_linejoin    = p2s.INHERIT | None | 'miter' | 'round' | 'bevel'   # NEW, SVG only
            label              = p2s.INHERIT                # the dict key (default)
                               = None                       # no label, even when background_label_color is set
                               = 'text'                     # any string; decouples the label from the key
            label_color        = p2s.INHERIT | None | '#RRGGBB' | 'vary'

        background_fill        = None                       # no fill (transparent)
                               = '#RRGGBB'                  # fixed hex colour for all shapes
                               = 'vary'                     # auto-assign a unique colour per key
                               = {key: '#RRGGBB', ...}      # per-shape colour lookup
        background_opacity     = 1.0 | None | {key: float, ...}
        background_stroke      = 'default' | None | '#RRGGBB' | 'vary' | {key: '#RRGGBB', ...}
        background_stroke_w    = 1.0 | {key: float, ...}
        background_label_color = None | '#RRGGBB' | 'vary' | {key: '#RRGGBB', ...}

        p2s.INHERIT vs None on a record: INHERIT defers to the background_* parameter
        above; None means explicitly off / omit the attribute. A bare shape_desc is a
        record whose every field is INHERIT, which is why the two forms mix.
        Note stroke_width=None on a RECORD omits the attribute (SVG's default applies);
        background_stroke_w=None as a PARAMETER still means "no stroke at all". Say
        stroke=None on the record for that.

        Shapes are drawn in dict insertion order -- all shapes, then all labels -- so a
        later entry lands on top of an earlier one. This is contract, not incidental.
        stroke_linecap / stroke_linejoin apply to the SVG only; the GPU display list
        has no cap/join (see PLANNING.md §8).

        Supported shapely types: Polygon, MultiPolygon, LineString, MultiLineString.
        LineString / MultiLineString shapes are unfilled (an open subpath has no interior
        worth painting -- SVG would close it implicitly in order to fill it) unless the
        record sets fill= explicitly, in which case the caller outranks the coercion.
        Both x and y axes must be scalars for background shapes to be transformed correctly.

        sm_shared      = {p2s.SM_X | p2s.SM_Y | p2s.SM_COLOR | p2s.SM_COUNT} # shared attributes w/in small multiples
        use_lazy_execution = True (default) | False

        The following parameters fix axis extents or normalise values across small-multiple panels:

        x_range                       = (min, max)   # fix the x axis extent
        y_range                       = (min, max)   # fix the y axis extent
        x_shared_label_range          = (min, max)   # shared axis label range for small multiples
        y_shared_label_range          = (min, max)
        color_magnitude_min           = float        # normalise CMAGNITUDE color across panels
        color_magnitude_max           = float
        color_stretched_global_values = list         # normalise CSTRETCHED color across panels
        dot_size_global_min           = float        # normalise dot size across panels
        dot_size_global_max           = float


        legend = False (default)                        # no legend -- output identical to pre-legend renders
               = True                                   # same as 'right'
               = 'right' | 'left' | 'top' | 'bottom'    # position of the legend strip
               = {'pos': ..., 'title': ..., 'fmt': ..., 'max_items': ..., 'order': ...}

        The legend kind is auto-selected from the resolved color mode: a categorical
        swatch list for CSETp / bare-categorical color, a colorbar for the spectrum
        modes (CMAGNITUDE_* / CSTRETCHED_* / CROW_* / CSET_MAGNITUDE*). The strip is
        reserved FROM wxh -- the plot region shrinks; the physical output size does
        not change (allocate a larger wxh if the plot needs the space). A truthy
        legend with nothing to legend (e.g. a flat / literal hex color) is silently
        omitted. The captured scale/category metadata is exposed as .legend_info
        (a polars2svg.LegendInfo). Also settable globally via set_defaults(legend=...).
        v1 scope is the color encoding only: dot_size= / opacity= carry no size legend yet.

        '''
        return XYp(*args, p2s=self, **kwargs)

    def histop(self, *args: Any, **kwargs: Unpack[HistopKwargs]) -> Histop:
        '''
        histop(polars.DataFrame, bin_by, ...)

        A horizontal histogram: one bar per category/bin of ``bin_by``. Bar length is
        set by ``count=`` (the primary size knob — row count by default) and bar color
        by the orthogonal ``color=``; bars sort by ``order=``. Also renders as boxplot,
        swarm, or stacked-bar via ``style=``.

        Example::

            p2s.histop(df, 'category', count='bytes', color=p2s.CROW_MAGNITUDEp)

        bin_by         = 'field'                                   # (can be specified as a string / i.e., not a keyword argument)
                       = ('field1', 'field2', ...)                 # multi-field bins are joined with '|' for display

        template       = None                                      # another Histop instance; copies all settings, then applies any overrides

        === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< === %<

        count          = p2s.ROW_COUNTp                            # default
                       = 'field'                                   # non-numeric fields will be counted via set operations
                       = ('field', p2s.SETp)                       # treat the field as categorical (set-based counting)
                       = ('field1', 'field2', ...)                 # fields will be concatenated for counting

        count_range        = (min, max)                            # use a specified range for bar lengths (useful for small multiples)
        count_range_shared = (min, max)                            # shared range across small multiples (set automatically by smallp)

        color          = None                                      # default — fixed color
                       = 'field'                                   # numeric → whole-bar spectrum (sum); string → stacked, categorical hash
                       = p2s.CROW_MAGNITUDEp                      # whole-bar spectrum by raw row count, independent of count= (linear normalization)
                       = p2s.CROW_STRETCHEDp                      # whole-bar spectrum by raw row count, independent of count= (rank normalization)
                       = ('field', p2s.CSETp)                     # stacked, categorical hash colors (even if field is numeric)
                       = ('field', p2s.CSET_MAGNITUDEp)           # stacked, each segment spectrum-colored by its count (linear)
                       = ('field', p2s.CSET_STRETCHEDp)           # stacked, each segment spectrum-colored by its count (rank)
                       = ('field', p2s.CMAGNITUDE_SUMp | CMAGNITUDE_MINp | CMAGNITUDE_MEDIANp | CMAGNITUDE_MEANp | CMAGNITUDE_MAXp)
                                                                   # whole-bar spectrum using that statistic (linear normalization)
                       = ('field', p2s.CSTRETCHED_SUMp | CSTRETCHED_MINp | CSTRETCHED_MEDIANp | CSTRETCHED_MEANp | CSTRETCHED_MAXp)
                                                                   # whole-bar spectrum using that statistic (rank normalization)
                       = ('field', <statistic enum>)               # whole-bar spectrum using MINp/MEDIANp/MEANp/MAXp/STDp (linear)
                       = ('field1', 'field2', ...)                 # fields concatenated → stacked, categorical hash

        color_stat_range_shared = (min, max)                       # shared spectrum range across small multiples (set automatically by smallp)

        order          = p2s.ROW_COUNTp                            # default — sort bars by count, descending
                       = 'field'                                   # sort by the sum of this field
                       = ('field', p2s.SETp)                       # sort by the number of unique values
                       = ('field', <statistic enum>)               # sort by the specified statistic
        descending     = True (default) | False

        style          = p2s.BARCHARTp                             # default
                       = p2s.BOXPLOTp
                       = p2s.BOXPLOT_W_SWARMp
                       = p2s.STACKEDBARp

        === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< === %<

        wxh                = (128, 256)                             # default canvas size (width, height)
        insets             = (x-inset, y-inset)                    # if the plot is too small, the insets won't be drawn
        draw_context       = True (default) | False                # if the plot is too small, the context won't be drawn
                                                                    # (grid lines, count axis, "+N more" indicator)
        draw_labels        = True (default) | False                # per-bin label (the bin's own value/category), drawn
                                                                    # inside the plot at the bar's left edge -- independent
                                                                    # of draw_context; survives draw_context=False
        draw_border        = True (default) | False                # draw a rectangular border around the SVG
        txt_h              = 12                                    # label text height in pixels
        bar_h              = height of each bar in pixels (default matches txt_h)
        v_gap              = vertical gap between bars in pixels (default 0)
        draw_distribution  = True | False (default)                # overlay a legacy frequency curve on the bar chart
        distribution       = True (default) | False                # show the distribution strip below the bars
        distribution_bin_w = height of each strip cell in pixels (default 10)
        sm_shared          = {p2s.SM_COLOR | p2s.SM_COUNT}        # shared attributes w/in small multiples
        use_lazy_execution = True (default) | False
        min_bar_w          = minimum bar width in pixels (default 1.0)
        swarm_max_pts      = max swarm points per bin for BOXPLOT_W_SWARMp (default 50)
        remainder_threshold = min estimated pixel width for a color segment to be shown individually
                             (default 3.0); segments below this are collapsed into an "(other)" bucket

        legend = False (default)                        # no legend -- output identical to pre-legend renders
               = True                                   # same as 'right'
               = 'right' | 'left' | 'top' | 'bottom'    # position of the legend strip
               = {'pos': ..., 'title': ..., 'fmt': ..., 'max_items': ..., 'order': ...}

        The legend kind is auto-selected from the resolved color mode: a categorical
        swatch list for CSETp / bare-categorical color, a colorbar for the spectrum
        modes (CMAGNITUDE_* / CSTRETCHED_* / CROW_* / CSET_MAGNITUDE*). The strip is
        reserved FROM wxh -- the plot region shrinks; the physical output size does
        not change (allocate a larger wxh if the plot needs the space). A truthy
        legend with nothing to legend (e.g. a flat / literal hex color) is silently
        omitted. The captured scale/category metadata is exposed as .legend_info
        (a polars2svg.LegendInfo). Also settable globally via set_defaults(legend=...).
        v1 scope is the color encoding only: bar length (count=) is covered by the count axis labels, not the legend.

        '''
        return Histop(*args, p2s=self, **kwargs)

    def piep(self, *args: Any, **kwargs: Unpack[PiepKwargs]) -> Piep:
        '''
        piep(polars.DataFrame, bin_by, ...)

        A pie / donut / waffle chart.  Mirrors histop() in parameters and usage: bins become
        slices, count= sets each slice's magnitude (its share of the whole), and color= sets
        each slice's color.

        Example::

            p2s.piep(df, 'category', style=p2s.DONUTp, draw_labels=True)

        bin_by         = 'field'                                   # (can be specified as a positional string)
                       = ('field1', 'field2', ...)                 # multi-field slices joined with '|'

        template       = None                                      # another Piep instance; copies all settings, then applies overrides

        === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< === %<

        count          = p2s.ROW_COUNTp                            # default — slice share = row count
                       = 'field'                                   # numeric → sum; non-numeric → n_unique (set-based)
                       = ('field', p2s.SETp)                       # treat the field as categorical (set-based counting)
                       = ('field1', 'field2', ...)                 # fields concatenated → n_unique

        color          = None                                      # default — each slice is filled and stroked with one of five
                                                                   #   barely-distinct shades of the data color, assigned so adjacent
                                                                   #   slices differ (deterministic per data+settings)
                       = 'field'                                   # numeric → per-slice spectrum (sum); categorical → per-slice
                                                                   #   value color (one value → that color; mixed → a set color)
                       = ('field', COLOR_ENUM)                     # see the enums below
                       = ['field', 'field2', ...]                  # multiple categorical fields, concatenated (set coloring)
                       = p2s.CROW_MAGNITUDEp | p2s.CROW_STRETCHEDp # per-slice spectrum by raw row count (linear | rank)
                       = '#RRGGBB'                                 # one fixed color for every slice
                       = ['#RRGGBB', '#RRGGBB', ...]               # cycled across the slices (largest-first order)

        # COLOR_ENUM (used with a field) mirrors xyp:
        #   CSETp                                 categorical: one value → that color, mixed slice → a shared set color
        #   CSET_MAGNITUDEp | CSET_STRETCHEDp     spectrum by the number of unique values in the slice (linear | rank)
        #   CMAGNITUDE_SUMp|MINp|MEDIANp|MEANp|MAXp     numeric field → spectrum by that statistic (linear)
        #   CSTRETCHED_SUMp|MINp|MEDIANp|MEANp|MAXp     numeric field → spectrum by that statistic (rank)
        # To color each slice by its own category, pass color=<bin_by field>.
        # The spectrum palette is p2s.spectrum_palette.

        descending     = True (default) | False                    # slices are ordered by count=; True puts the largest first

        style          = p2s.PIEp                                  # default
                       = p2s.DONUTp
                       = p2s.WAFFLEp

        === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< === %<

        wxh                = (160, 160)                             # default canvas size (width, height)
        insets             = (x-inset, y-inset)
        draw_context       = True (default) | False                # bin-field title beneath the chart; suppressed when too small
        draw_labels         = False (default) | True                # category labels: placed inside a slice when it is large
                                                                   # enough, otherwise outside with a leader line when there is
                                                                   # margin around the pie (large insets / oblong wxh); labels
                                                                   # are cropped and prioritized largest-slice-first
        draw_border        = True (default) | False                # draw a rectangular border around the SVG
        txt_h              = 12                                    # label text height in pixels
        start_angle        = -90.0                                 # degrees; -90 = 12 o'clock, sweeping clockwise
        donut_ratio        = 0.55                                  # inner/outer radius for DONUTp
        waffle_n           = 10                                    # WAFFLEp grid is waffle_n x waffle_n cells
        min_slice_deg      = 3.0                                   # slices narrower than this many degrees are
                                                                   # folded into an "(other)" slice holding their
                                                                   # combined total (0 disables); in small multiples
                                                                   # the "all rows" reference decides the fold so every
                                                                   # panel shows the same slices
        count_range_shared = (min, max)                            # whole-total, shared across small multiples (set by smallp)
        color_stat_range_shared = (min, max)                       # shared spectrum range across small multiples (set by smallp)
        sm_shared          = {p2s.SM_SLICE_ORDERp | p2s.SM_PARTOFWHOLEp | p2s.SM_COLOR}
                             # SM_SLICE_ORDERp  → identical slice order & colors in every panel
                             # SM_PARTOFWHOLEp  → fade the "all rows" chart behind each panel and fill in this
                             #                    panel's share of every slice (implies SM_SLICE_ORDERp)
        use_lazy_execution = True (default) | False

        legend = False (default)                        # no legend -- output identical to pre-legend renders
               = True                                   # same as 'right'
               = 'right' | 'left' | 'top' | 'bottom'    # position of the legend strip
               = {'pos': ..., 'title': ..., 'fmt': ..., 'max_items': ..., 'order': ...}

        The legend kind is auto-selected from the resolved color mode: a categorical
        swatch list for CSETp / bare-categorical color, a colorbar for the spectrum
        modes (CMAGNITUDE_* / CSTRETCHED_* / CROW_* / CSET_MAGNITUDE*). The strip is
        reserved FROM wxh -- the plot region shrinks; the physical output size does
        not change (allocate a larger wxh if the plot needs the space). A truthy
        legend with nothing to legend (e.g. a flat / literal hex color) is silently
        omitted. The captured scale/category metadata is exposed as .legend_info
        (a polars2svg.LegendInfo). Also settable globally via set_defaults(legend=...).
        v1 scope is the color encoding only: slice angle (count=) is conveyed by the slices themselves, not the legend.

        '''
        return Piep(*args, p2s=self, **kwargs)

    def linkp(self, *args: Any, **kwargs: Unpack[LinkPKwargs]) -> LinkP:
        '''
        linkp(polars.DataFrame, relationships, pos, ...)

        A node-link graph / network. Each ``relationships`` pair contributes an edge and
        its endpoint nodes; nodes are placed by ``pos=`` (a networkx-style dict) or given
        random positions. Node and link size, color, opacity, shape, labels, convex hulls,
        and shapely backgrounds are all configurable.

        Note ``count=`` only drives geometry once you opt into ``node_size='vary'`` /
        ``link_size='vary'``; at the default fixed sizes it has no visible effect (a
        one-time warning fires if you set it anyway).

        Example::

            pos = networkx.spring_layout(g)
            p2s.linkp(df, [('src', 'dst')], pos, color='dept', node_size='vary')

        Layout provenance: ``LandmarkMDSLayout`` implements de Silva & Tenenbaum (2003)
        Landmark MDS; ``PivotMDSLayout`` implements Brandes & Pich (2007) Pivot MDS;
        ``TFDPLayout`` implements Zhong et al. (2023) t-FDP. See each class's module header
        for the full citation. (``PolarsForceDirectedLayout`` and ``ConveyProximityLayout``,
        which implemented Cohen (1997) "Drawing Graphs to Convey Proximity", were deprecated
        in 0.2.0 and removed in 0.3.0 — see the CHANGELOG for replacements.)

        relationships and pos may be passed as positional arguments (in any order after df) or
        as keyword arguments — but not both ways for the same parameter.

        Auto-inference rules for positional args:
          list of tuples (each length >= 2)   → relationships
          dict with {node: [x, y]} values     → pos

        relationships  = [('from_field', 'to_field')]
                       = [('from_field', 'to_field', 'predicate_field')]
                       = [(('f0','f1'), ('f2','f3'))]      # tuple fields are concatenated with '|'

        pos            = {node_name: [x, y], ...}          # networkx-style position dict
                                                            # nodes absent from pos get random positions

        null_nodes     = False                             # default — a row with one null endpoint draws only the
                                                            # entity beside it, as an unconnected dot
                       = True                              # draw the missing endpoint too, as that entity's OWN null
                                                            # node (p2s.NULL_NODE_PREFIX + entity, labeled '(null)'),
                                                            # so the row becomes a visible stub edge. Per-entity, not
                                                            # one shared node: records with a missing endpoint are not
                                                            # connected to each other. Rows with BOTH endpoints null
                                                            # are left alone.

        template       = None                              # another LinkP instance; copies all settings, then applies any overrides

        === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< ===

        color          = None                              # default — data color for links and nodes
                       = '#rrggbb'                         # fixed hex constant (both links and nodes)
                       = 'field'                           # categorical hash by field (applies to both)
                       = p2s.CROW_MAGNITUDEp               # spectrum by raw edge/row count, independent of count= (linear normalization)
                       = p2s.CROW_STRETCHEDp               # spectrum by raw edge/row count, independent of count= (rank normalization)
                       = ('field', p2s.CSETp)              # categorical, even if field is numeric
                       = ('field', p2s.CSET_MAGNITUDEp)   # spectrum by unique-value count (linear)
                       = ('field', p2s.CSET_STRETCHEDp)   # spectrum by unique-value count (rank)
                       = ('field', p2s.CMAGNITUDE_SUMp | CMAGNITUDE_MINp | CMAGNITUDE_MEDIANp | CMAGNITUDE_MEANp | CMAGNITUDE_MAXp)
                       = ('field', p2s.CSTRETCHED_SUMp | CSTRETCHED_MINp | CSTRETCHED_MEDIANp | CSTRETCHED_MEANp | CSTRETCHED_MAXp)

        node_color     = None                                                   # node gets no color (default)
                       = p2s.COLOR_BY_NODE_NAME                                 # node gets the color of the color hash of the node name
                       = '#rrggbb'                                              # fixed hex constant for all nodes
                       = {node_name: '#rrggbb', ...}                            # per-node hex dict (takes full priority)
                       = 'field'                                                 # if field is non-numeric, same as ('field', p2s.CSETp)
                                                                                 # ... otherwise (field is numeric), same as ('field', p2s.CMAGNITUDE_SUMp)
                       = p2s.CROW_MAGNITUDEp                                                    # raw row count, independent of count=
                       = p2s.CROW_STRETCHEDp                                                    # raw row count, independent of count=
                       = ('field', p2s.CSETp)                                    # categorical on field
                       = ('field', p2s.CSET_MAGNITUDEp | CSET_STRETCHEDp)        # categorical size on field
                       = ('field', p2s.CMAGNITUDE_SUMp | ... | CSTRETCHED_MAXp)  # numeric field calculations

        === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< ===

        count          = p2s.ROW_COUNTp                   # default — count rows
                       = 'field'                           # numeric → sum; non-numeric → count-distinct
                       = ('field', p2s.SETp)               # force count-distinct
                       = ('field1', 'field2', ...)         # concatenate fields, then count-distinct

        === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< ===

        node_size      = 'medium'                          # 'nil' | 'small' | 'medium' | 'large' | 'vary' | None | float
        node_opacity   = 1.0
        node_size_range = (0.3, 4)                         # (min, max) radius for node_size='vary'
        draw_node_labels = False                           # when True, draw text labels on non-collapsed nodes
        node_labels    = None                              # {node_name: label_str}; a node absent from the dict is not labeled
        label_only     = set()                             # restrict labels to these NAMES (list or str also accepted); if
                                                            # empty, everything is labeled.  Gates both channels: a node by its
                                                            # name, an edge by its label value (a '*' edge survives if any of
                                                            # the values behind the '*' is in the set).  Tested against the raw
                                                            # value, before node_labels / link_labels rename it for display.

        link_size      = 'small'                           # 'nil' | 'small' | 'medium' | 'large' | 'vary' | None | float
        link_shape     = 'line'                            # 'line' | 'curve' | 'flowmap' (force-directed OD flow layout, Jenny et al. 2017)
        link_arrows    = False                             # draw arrowheads at link destinations
        draw_link_labels = False                           # label each edge with the relationship tuple's third
                                                            # element -- or, for a two-part tuple, with the field
                                                            # driving color=.  'line' rotates the text onto the edge,
                                                            # 'curve' runs it along the curve; 'flowmap' is not
                                                            # labeled.  An edge whose rows disagree is labeled '*'.
                                                            # Both directions of a bidirectional pair are labeled, on
                                                            # opposite sides of the edge.  Independent of
                                                            # draw_node_labels.
        link_labels    = None                              # {label_value: label_str} -- the link-side node_labels; an edge
                                                            # whose value is absent from the dict is not labeled
        link_opacity   = 1.0
        link_size_range = (0.25, 4)                        # (min, max) stroke width for link_size='vary'

        time           = None                              # timing marks OFF (default). When set, draws short colored ticks
                                                            # along each edge: position + spectrum color encode each record's
                                                            # timestamp (normalized over the whole df's min/max), and the side
                                                            # of the edge + a slight slant encode the direction of activity.
                       = 'field'                           # a timestamp/date column (mirrors timep's time field)
                       = ('field', p2s.LT_Y_m_d_Hp | ...)  # a linear granularity (TimeLinearTypeP)
                       = ('field', p2s.PT_Hp | ...)        # a periodicity (TimePeriodicTypeP)
                       = p2s.tField('field', enum)         # the typed equivalent
        timing_marks_length = 3.0                          # tick length in PIXELS (frame-size independent)
        timing_marks_spacing = 1.0                         # min spacing between marks in PIXELS (decimation
                                                            # resolution): marks closer than this along an edge
                                                            # merge into one; larger = sparser. 1.0 = per-pixel
                                                            # default, clamped to >= 1 (sub-pixel is meaningless)

        === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< ===

        wxh            = (width, height)
        insets         = (x-inset, y-inset)
        bounds_percent = 0.05                              # padding around graph extent
        view_window    = None                              # (wx0, wy0, wx1, wy1) world-coord viewport
        use_pos_for_bounds = False                         # False: bounds fit only nodes in the dataframe;
                                                           # True: also stretch bounds to every key in pos,
                                                           # even nodes not drawn at this stack level

        draw_border    = True
        txt_h          = 12

        === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< ===

        convex_hull_lu           = None                    # {regex: hull_name} or {name: [node_list]}
        convex_hull_opacity      = 0.3
        convex_hull_labels       = False
        convex_hull_stroke_width = None

        background             = None                       # no background (default)
                               = {key: shape_desc, ...}     # bare descriptor; inherits every background_* below
                               = {key: record, ...}         # record; carries its own appearance
                                                            # the two forms mix freely in one dict

          shape_desc           = shapely Polygon | MultiPolygon | LineString | MultiLineString
                               = [(x,y), ...]               # list of world-coordinate pairs
                               = '<circle cx=... r=... />'  # SVG circle string
                               = 'M x y L x y C ... Z'      # SVG path description (M/L/C/Z)

          record               = p2s.bgShape(shape_desc, fill=..., stroke=..., ...)
                               = {'shape': shape_desc, 'fill': ..., ...}   # same fields, plain dict

            shape              = shape_desc                 # required
            fill               = p2s.INHERIT (default)      # take background_fill
                               = None                       # explicitly UNFILLED -- emits fill="none"
                               = '#RRGGBB' | 'vary'
            fill_opacity       = p2s.INHERIT | None | float
            stroke             = p2s.INHERIT | None | '#RRGGBB' | 'vary'
            stroke_opacity     = p2s.INHERIT | None | float # NEW -- no sidecar equivalent
            stroke_width       = p2s.INHERIT | None | float
            dash               = p2s.INHERIT | None | '4 2' # NEW -- stroke-dasharray
            stroke_linecap     = p2s.INHERIT | None | 'butt' | 'round' | 'square'   # NEW, SVG only
            stroke_linejoin    = p2s.INHERIT | None | 'miter' | 'round' | 'bevel'   # NEW, SVG only
            label              = p2s.INHERIT                # the dict key (default)
                               = None                       # no label, even when background_label_color is set
                               = 'text'                     # any string; decouples the label from the key
            label_color        = p2s.INHERIT | None | '#RRGGBB' | 'vary'

        background_fill        = None                       # no fill (transparent)
                               = '#RRGGBB'                  # fixed hex colour for all shapes
                               = 'vary'                     # auto-assign a unique colour per key
                               = {key: '#RRGGBB', ...}      # per-shape colour lookup
        background_opacity     = 1.0 | None | {key: float, ...}
        background_stroke      = 'default' | None | '#RRGGBB' | 'vary' | {key: '#RRGGBB', ...}
        background_stroke_w    = 1.0 | {key: float, ...}
        background_label_color = None | '#RRGGBB' | 'vary' | {key: '#RRGGBB', ...}

        p2s.INHERIT vs None on a record: INHERIT defers to the background_* parameter
        above; None means explicitly off / omit the attribute. A bare shape_desc is a
        record whose every field is INHERIT, which is why the two forms mix.
        Note stroke_width=None on a RECORD omits the attribute (SVG's default applies);
        background_stroke_w=None as a PARAMETER still means "no stroke at all". Say
        stroke=None on the record for that.

        Shapes are drawn in dict insertion order -- all shapes, then all labels -- so a
        later entry lands on top of an earlier one. This is contract, not incidental.
        stroke_linecap / stroke_linejoin apply to the SVG only; the GPU display list
        has no cap/join (see PLANNING.md §8).

        sm_shared      = set()                             # shared attributes within small multiples
                       = {p2s.SM_X}                        # share X world-coordinate range across panels**
                       = {p2s.SM_Y}                        # share Y world-coordinate range across panels**
                       = {p2s.SM_X, p2s.SM_Y}              # share full coordinate space (identical graph layout bounds)**
                       = {p2s.SM_COUNT}                    # share count normalization for link_size='vary' / node_size='vary'
                       = {p2s.SM_COLOR}                    # share color-stat range for magnitude-mode coloring

                       ** - note that "use_pos_for_bounds" when True will override SM_X and SM_Y

        legend = False (default)                        # no legend -- output identical to pre-legend renders
               = True                                   # same as 'right'
               = 'right' | 'left' | 'top' | 'bottom'    # position of the legend strip
               = {'pos': ..., 'title': ..., 'fmt': ..., 'max_items': ..., 'order': ...}

        The legend kind is auto-selected from the resolved color mode: a categorical
        swatch list for CSETp / bare-categorical color, a colorbar for the spectrum
        modes (CMAGNITUDE_* / CSTRETCHED_* / CROW_* / CSET_MAGNITUDE*). The strip is
        reserved FROM wxh -- the plot region shrinks; the physical output size does
        not change (allocate a larger wxh if the plot needs the space). A truthy
        legend with nothing to legend (e.g. a flat / literal hex color) is silently
        omitted. The captured scale/category metadata is exposed as .legend_info
        (a polars2svg.LegendInfo). Also settable globally via set_defaults(legend=...).
        v1 scope is the color encoding only: describes color= (links) when data-driven, else node_color=; 'vary' node/link sizing has no size legend yet.

        '''
        return LinkP(*args, p2s=self, **kwargs)

    def chordp(self, *args: Any, **kwargs: 'Unpack[ChPKwargs]') -> 'ChP':
        '''
        chordp(polars.DataFrame, relationships, ...)

        Chord diagram with curve or bundled link shapes.
        relationships may be passed as a positional argument (after df) or as a keyword argument.
        The bundled link shapes implement hierarchical edge bundling (Holten, IEEE TVCG 2006).

        Nodes are arranged as arcs around a circle and edges drawn as chords between them.
        By default the node order is derived by hierarchically clustering the edge weights
        (which ``count=`` feeds), so ``count=`` subtly reshuffles the ring; pin it with
        ``order=``. Arc/ribbon *geometry* only scales with count once ``node_size='vary'``
        / ``link_size='vary'`` (else a one-time warning fires if count is set).

        Example::

            p2s.chordp(df, [('src', 'dst')], color='src', node_size='vary')

        relationships  = [('from_field', 'to_field')]
                       = [(('f0','f1'), ('f2','f3'))]      # tuple fields are concatenated with '|'

        template       = None                              # another ChP instance; copies all settings, then applies any overrides

        === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< ===

        count          = p2s.ROW_COUNTp                   # default — count rows
                       = 'field'                           # numeric → sum; non-numeric → count-distinct
                       = ('field', p2s.SETp)               # force count-distinct
                       = ('field1', 'field2', ...)         # concatenate fields, then count-distinct

        === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< ===

        color          = None                              # default data color for all links
                       = '#rrggbb'                         # fixed hex constant for all links
                       = 'src'                             # each link inherits its source node's color
                       = 'dst'                             # each link inherits its destination node's color
                       = 'field'                           # string field → CSETp; numeric → CMAGNITUDE_SUMp
                       = ('field', p2s.CSETp)              # same spec forms as node_color below
                       = p2s.CROW_MAGNITUDEp               # color by raw row count per edge → spectrum
                       = p2s.CROW_STRETCHEDp               # color by raw row count rank → spectrum

        node_color     = None                              # nodes colored by hash of node name (default)
                       = '#rrggbb'                         # fixed hex constant for all nodes
                       = p2s.COLOR_BY_NODE_NAME            # hash each node's own name to a distinct color
                       = 'field'                           # string field → CSETp (hash); numeric field → CMAGNITUDE_SUMp
                       = ('field', p2s.CSETp)              # categorical: one color per unique value; multiset → sentinel
                       = ('field', p2s.CSET_MAGNITUDEp)   # categorical by count-distinct magnitude
                       = ('field', p2s.CSET_STRETCHEDp)   # categorical by count-distinct rank
                       = ('field', p2s.CMAGNITUDE_SUMp)    # numeric field aggregated per node → spectrum
                       = ('field', p2s.CMAGNITUDE_MINp)    #   (also MEDIANp, MEANp, MAXp, CSTRETCHED_* variants)
                       = p2s.CROW_MAGNITUDEp               # color by raw row count per node → spectrum
                       = p2s.CROW_STRETCHEDp               # color by raw row count rank → spectrum

        node_size      = 'medium'                          # 'small' | 'medium' | 'large' | 'vary' (arc height proportional to count)
        node_size_range = (0.3, 4.0)                       # (min_h, max_h) arc height range in pixels (for node_size='vary')
        node_gap       = 2                                 # gap between adjacent node arcs in pixels
        node_opacity   = 1.0                               # opacity for all node arcs (0.0–1.0)
        node_selection = set()                             # highlight these node values on the outer ring; others rendered smaller

        node_labels    = None                              # {node_value: label_str} — display labels for nodes
        label_only     = set()                             # restrict labels to these node values
        order          = None                              # explicit node ordering list; None → auto via hierarchical clustering

        === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< ===

        link_shape     = 'curve'                           # 'curve'    — bezier curves with arc-length-aware attachment
                       = 'bundled'                         # 'bundled'  — edges routed through a geometric skeleton
        link_size      = 'small'                           # 'small' | 'medium' | 'large' | 'vary'
        link_size_range = (0.25, 4.0)                      # (min, max) stroke-width in pixels
        link_opacity   = 1.0                               # opacity for all link paths (0.0–1.0)

        bundle_strength = 0.85                             # bundled routing tension: 0.0 = straight chord, 1.0 = full skeleton
        bundle_rings   = 4                                 # hex mesh density: hex edge = r / bundle_rings
        skeleton_algorithm = 'hexagonal'                   # 'hexagonal' | 'radial' | 'kmeans'

        === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< ===

        wxh            = (width, height)                   # default (256, 256)
        insets         = (x-inset, y-inset)                # default (3, 3)

        draw_labels    = False                             # when True, render text labels for each node
        label_style    = 'radial'                          # 'radial'   — text radiates outward from arc midpoint
                       = 'circular'                        # 'circular' — text follows the outer arc curve via <textPath>
        txt_h          = 12                                # font size in pixels for node labels
        txt_offset     = 0                                 # extra gap in pixels between arc outer edge and label

        draw_border    = True                              # draw a rectangular border around the SVG

        === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< ===

        sm_shared      = set()                             # small-multiples sharing flags (use with p2s.smallp)
                       = {p2s.SM_X}                        # share node order and arc positions across panels
                       = {p2s.SM_Y}                        # share bundled-edge routing skeleton (link_shape='bundled' only)
                       = {p2s.SM_COUNT}                    # share count normalization range
                       = {p2s.SM_COLOR}                    # share color-stat normalization range

        legend = False (default)                        # no legend -- output identical to pre-legend renders
               = True                                   # same as 'right'
               = 'right' | 'left' | 'top' | 'bottom'    # position of the legend strip
               = {'pos': ..., 'title': ..., 'fmt': ..., 'max_items': ..., 'order': ...}

        The legend kind is auto-selected from the resolved color mode: a categorical
        swatch list for CSETp / bare-categorical color, a colorbar for the spectrum
        modes (CMAGNITUDE_* / CSTRETCHED_* / CROW_* / CSET_MAGNITUDE*). The strip is
        reserved FROM wxh -- the plot region shrinks; the physical output size does
        not change (allocate a larger wxh if the plot needs the space). A truthy
        legend with nothing to legend (e.g. a flat / literal hex color) is silently
        omitted. The captured scale/category metadata is exposed as .legend_info
        (a polars2svg.LegendInfo). Also settable globally via set_defaults(legend=...).
        v1 scope is the color encoding only: describes color= (links) when data-driven, else node_color=; 'vary' node/link sizing has no size legend yet.

        '''
        try:
            from .chordp import ChP
        except ImportError as _e_:
            raise ImportError(
                "chordp requires the optional 'layouts' dependencies "
                "(scipy, and networkx for some layouts). Install them with:\n"
                "    pip install polars2svg[layouts]"
            ) from _e_
        return ChP(*args, p2s=self, **kwargs)

    def timep(self, *args: Any, **kwargs: Unpack[TimepKwargs]) -> Timep:
        '''
        timep(polars.DataFrame, time, ...)

        A temporal bar chart: the ``time`` column is binned along the x-axis and each
        bin becomes a vertical bar. Time is either *linear* (chronological, auto- or
        ``TimeLinearTypeP``-resolved) or *periodic* (folded into a repeating cycle via
        ``TimePeriodicTypeP`` — day-of-week, month, hour, …). Bar height is ``count=``
        (row count by default), bar color the orthogonal ``color=``. Also boxplot /
        swarm / stacked-bar via ``style=``.

        Example::

            p2s.timep(df, p2s.tField('timestamp', p2s.PT_DoWp))    # counts by day-of-week

        time           = 'field'                                   # (can be specified as a string / i.e., not a keyword argument)
                                                                   # defaults to linear time with an automatic resolution chosen
                       = ('field', TimePeriodicTypeP)              # periodic time will be used
                       = ('field', TimeLinearTypeP)                # the specific linear time will be forced

        (if no field is provided, the time field will be determined automatically)

        template       = None                                      # another Timep instance; copies all settings, then applies any overrides

        === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< === %<

        count          = p2s.ROW_COUNTp                            # default
                       = 'field'                                   # non-numeric fields will be counted via set operations
                       = ('field', p2s.SETp)                       # treat the field as categorical (set-based counting)
                       = ('field1', 'field2', ...)                 # fields will be concatenated for counting

        count_range        = (min, max)                            # use a specified range for the mins and maxes (useful for small multiples)
        count_range_shared = (min, max)                            # shared range across small multiples (set automatically by smallp)

        color          = None                                      # default — fixed color
                       = 'field'                                   # numeric → whole-bar spectrum (sum); string → stacked, categorical hash
                       = p2s.CROW_MAGNITUDEp                      # whole-bar spectrum by raw row count, independent of count= (linear normalization)
                       = p2s.CROW_STRETCHEDp                      # whole-bar spectrum by raw row count, independent of count= (rank normalization)
                       = ('field', p2s.CSETp)                     # stacked, categorical hash colors (even if field is numeric)
                       = ('field', p2s.CSET_MAGNITUDEp)           # stacked, each segment spectrum-colored by its count (linear)
                       = ('field', p2s.CSET_STRETCHEDp)           # stacked, each segment spectrum-colored by its count (rank)
                       = ('field', p2s.CMAGNITUDE_SUMp | CMAGNITUDE_MINp | CMAGNITUDE_MEDIANp | CMAGNITUDE_MEANp | CMAGNITUDE_MAXp)
                                                                   # whole-bar spectrum using that statistic (linear normalization)
                       = ('field', p2s.CSTRETCHED_SUMp | CSTRETCHED_MINp | CSTRETCHED_MEDIANp | CSTRETCHED_MEANp | CSTRETCHED_MAXp)
                                                                   # whole-bar spectrum using that statistic (rank normalization)
                       = ('field', <statistic enum>)               # whole-bar spectrum using MINp/MEDIANp/MEANp/MAXp/STDp (linear)
                       = ('field1', 'field2', ...)                 # fields concatenated → stacked, categorical hash

        color_stat_range_shared = (min, max)                       # shared spectrum range across small multiples (set automatically by smallp)

        style          = p2s.BARCHARTp                             # default
                       = p2s.BOXPLOTp
                       = p2s.BOXPLOT_W_SWARMp
                       = p2s.STACKEDBARp

        === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< === %<

        wxh                = (512, 256)                             # default canvas size (width, height)
        insets             = (x-inset, y-inset)                    # if the plot is too small, the insets won't be drawn
        draw_context       = True (default) | False                # if the plot is too small, the context won't be drawn
        draw_border        = True (default) | False                # draw a rectangular border around the SVG
        txt_h              = 12                                    # label text height in pixels
        sm_shared          = {p2s.SM_X | p2s.SM_COLOR | p2s.SM_COUNT}  # shared attributes w/in small multiples
        use_lazy_execution = True (default) | False
        min_bar_w          = minimum bar width in pixels (default 1.0)
        swarm_max_pts      = max swarm points per bin for BOXPLOT_W_SWARMp (default 50)
        remainder_threshold = min estimated pixel height for a color segment to be shown individually
                             (default 3.0); segments below this are collapsed into an "(other)" bucket
        date_range_shared  = (min_date, max_date)                  # shared x-axis date range for small multiples (set automatically by smallp w/ SM_X)
        min_label_spacing  = minimum pixel gap between adjacent tick labels on the time axis (default 15)

        legend = False (default)                        # no legend -- output identical to pre-legend renders
               = True                                   # same as 'right'
               = 'right' | 'left' | 'top' | 'bottom'    # position of the legend strip
               = {'pos': ..., 'title': ..., 'fmt': ..., 'max_items': ..., 'order': ...}

        The legend kind is auto-selected from the resolved color mode: a categorical
        swatch list for CSETp / bare-categorical color, a colorbar for the spectrum
        modes (CMAGNITUDE_* / CSTRETCHED_* / CROW_* / CSET_MAGNITUDE*). The strip is
        reserved FROM wxh -- the plot region shrinks; the physical output size does
        not change (allocate a larger wxh if the plot needs the space). A truthy
        legend with nothing to legend (e.g. a flat / literal hex color) is silently
        omitted. The captured scale/category metadata is exposed as .legend_info
        (a polars2svg.LegendInfo). Also settable globally via set_defaults(legend=...).
        v1 scope is the color encoding only: bar height (count=) is covered by the count axis labels, not the legend.

        '''
        return Timep(*args, p2s=self, **kwargs)

    def smallp(self, *args: Any, **kwargs: Unpack[SmallpKwargs]) -> Smallp:
        '''
        smallp(polars.DataFrame, sm_template, category_by, ...)

        Create small multiples (trellis / facet) views of a DataFrame.  Each panel is rendered
        using sm_template as a blueprint.  Panels are produced either by splitting the data on
        category_by values, or by cycling template parameters with cycle_by.

        Example::

            tmpl = p2s.xyp(x='x', y='y', sm_shared={p2s.SM_X, p2s.SM_Y})  # dataless template
            p2s.smallp(df, tmpl, 'region')                               # one panel per region

        sm_template        = XYp | Timep | Histop             # see the sm_shared variable w/in the template

        --- split mode (mutually exclusive with cycle_by) ---

        category_by        = 'field'
                           = ('field1', 'field2', ...)
                           = [dataframe1, dataframe2, ...]
                           = {label1: dataframe1, label2: dataframe2, ...}

        === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< === %<

        order              = ROW_COUNTp
                           = 'field'                          # default is summation
                           = ('field', SETp)                  # count by the size of the set
                           = ('field', <statistic enum>)      # use the specified statistic
                           = ('field1', 'field2', ...)        # count by the size of the set
        descending         = True (default) | False           # default is True
        include_all        = True           | False (default) # show the "all" small multiple
        use_lazy_execution = True | False (default)
        sketch_only        = True           | False (default) # produce a sketch that indicates what it will look like
        collate_remainder  = True (default) | False           # for the ones that won't fit, collate them into a single small multiple

        === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< === %<

        grid_mode          = True | False (default)           # when the category_by is a tuple of two fields,
                                                              # organize the small multiples as a grid

        --- cycle mode (mutually exclusive with category_by) ---

        cycle_by           = {param: [val1, val2, ...], ...}  # each panel shows the full dataset rendered
                                                              # with a different value for the named template
                                                              # parameter(s); all value lists must be the
                                                              # same length; panels appear in list order.

        # cycle the color field across column names (works for XYp, Timep, Histop):
        #   cycle_by = {'color': ['country', 'region', 'city']}

        # cycle the x axis across column names (XYp):
        #   cycle_by = {'x': ['revenue', 'cost', 'profit']}

        # cycle time periods using tField strings (Timep):
        #   cycle_by = {'time': ['date|mp', 'date|DoWp', 'date|Qp']}

        # cycle multiple parameters in lockstep (zipped, not crossed):
        #   cycle_by = {'x': ['revenue', 'cost'], 'y': ['margin', 'profit']}

        # cycle_by works with LinkP too; any LinkP parameter can be a key:
        #   cycle_by = {'color':      ['country', 'region']}          # vary node/link color field
        #   cycle_by = {'node_color': ['#e74c3c', '#3498db']}         # vary fixed node color
        #   cycle_by = {'link_size':  ['weight', 'volume']}           # vary link-width field
        #   cycle_by = {'view_window': [(-1,-1,1,1), (-2,-2,2,2)]}    # vary viewport
        #   cycle_by = {'color': ['country','region'],
        #               'link_size': ['weight', 'volume']}             # multiple params in lockstep

        === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< === %<

        wxh                = (width, height)
                           = (width, None)                    # default w/ a width of 1280
                           = (None,  height)
        insets             = (x-inset, y-inset)               # if the plot is too small, the insets won't be drawn
        draw_labels        = True (default) | False           # centered label (the tile's category/entity) below each tile
        draw_border        = True (default) | False           # rectangular outline around each tile
        txt_h              = label text height

        # Note: draw_context (axes) is a per-template concern -- set it on sm_template
        # (e.g. p2s.xyp(..., draw_context=False)) before passing it in, not on smallp() itself.

        '''
        return Smallp(*args, p2s=self, **kwargs)

    def spreadlinesp(self, *args: Any, **kwargs: Unpack[SpreadLinesPKwargs]) -> SpreadLinesP:
        '''
        spreadlinesp(polars.DataFrame, relationships, ego, time, ...)

        Visualizes egocentric dynamic influence over time (SpreadLine layout,
        Kuo et al., arXiv:2408.08992).
        Time flows left → right; each timestamp becomes a vertical bin of packed
        circles.  Circles above the ego line are senders; circles below are receivers.

        Example::

            p2s.spreadlinesp(df, [('src', 'dst')], ego='alice', time='timestamp')

        relationships  = [('from_field', 'to_field')]

        ego            = 'node_name'                           # single ego node
                       = {'node_a', 'node_b', ...}             # set ego (collapsed to virtual __EGO__)

        time           = 'field'                               # auto-resolves granularity
                       = ('field', TimeLinearTypeP)            # explicit linear granularity

        node_color     = None                                  # hash by node name (default)
                       = p2s.COLOR_BY_NODE_NAME                # same as None but explicit
                       = '#rrggbb'                             # fixed hex for all nodes
                       = {node_name: '#rrggbb', ...}           # per-node dict
                       = 'field'                               # categorical hash by field value

        count          = p2s.ROW_COUNTp                        # default — count rows
                       = 'field'                               # numeric → sum; non-numeric → count-distinct
                       = ('field', p2s.SETp)                   # force count-distinct

        template       = None                                  # another SpreadLinesP instance; copies all settings, then applies overrides

        wxh            = (800, 400)                            # default canvas size (width, height)
        draw_context   = True (default) | False                # timestamp row along the bottom of each bin
        draw_border    = True (default) | False                # draw a rectangular border around the SVG
        draw_labels    = False (default)                       # per-node labels are not supported by the packed-circle
                                                                # layout; setting True raises NotImplementedError
        txt_h          = 12                                    # label text height in pixels

        legend = False (default)                        # no legend -- output identical to pre-legend renders
               = True                                   # same as 'right'
               = 'right' | 'left' | 'top' | 'bottom'    # position of the legend strip
               = {'pos': ..., 'title': ..., 'fmt': ..., 'max_items': ..., 'order': ...}

        The legend kind is auto-selected from the resolved color mode: a categorical
        swatch list for CSETp / bare-categorical color, a colorbar for the spectrum
        modes (CMAGNITUDE_* / CSTRETCHED_* / CROW_* / CSET_MAGNITUDE*). The strip is
        reserved FROM wxh -- the plot region shrinks; the physical output size does
        not change (allocate a larger wxh if the plot needs the space). A truthy
        legend with nothing to legend (e.g. a flat / literal hex color) is silently
        omitted. The captured scale/category metadata is exposed as .legend_info
        (a polars2svg.LegendInfo). Also settable globally via set_defaults(legend=...).
        v1 scope is the color encoding only: node_color= only (a field or by-name); circle size has no legend.

        '''
        return SpreadLinesP(*args, p2s=self, **kwargs)

    def tile(self, *args: Any, **kwargs: Unpack[TileKwargs]) -> Tile:
        '''
        tile(svg_list, ...)

        Lay already-rendered SVGs out as one SVG.  Unlike every other method here it takes
        renderings rather than a DataFrame: any component, any SVG string, or another
        tile() -- mixed freely.  A single rendering may be passed on its own (it is treated
        as a one-element list), which is the way to drop one component into a fixed-size
        viewport.

        Example::

            p2s.tile([chart_a, chart_b])                            # side by side
            p2s.tile([chart_a, chart_b], per_row=1)                 # stacked
            p2s.tile(charts, per_row=4, spacer=(8, 24))             # 4-across grid
            p2s.tile(charts, per_row=4, wxh=(640, 480))             # ... scaled to fit 640x480

        svg_list       = [rendering, ...]                          # (can be specified as a positional list)
                       = rendering                                 # a lone rendering == a one-element list
                                                                   #   each element: a component, an SVG string, or a Tile

        === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< === %< === %<

        per_row        = None (default)                            # a single row -- every tile side by side
                       = <n>                                       # a grid, filled row by row, n tiles per row
                                                                   #   (the last row may be short)
                       = 1                                         # a single column -- every tile stacked
                                                                   #   (rtsvg's tile(horz=False))

        spacer         = 0 (default)                               # that much separation both horizontally and
                                                                   #   vertically
                       = (horizontal, vertical)                    # separate gaps; a layout with one row has no
                                                                   #   vertical gaps to draw, and per_row=1 no
                                                                   #   horizontal ones

        wxh            = None (default)                            # canvas is exactly as large as the tiling
                       = (width, height)                           # scale the whole tiling to fit this size --
                                                                   #   uniformly, centered, so nothing is cropped or
                                                                   #   distorted; bg_color fills what the aspect
                                                                   #   difference leaves over
                       = (width, None) | (None, height)            # ... the other side follows the aspect ratio

        bg_color       = None (default)                            # the background color
                       = '#RRGGBB'                                 # any SVG color; shows through the spacer gaps and
                                                                   #   the viewport margins

        Tiles are top-aligned within their row and rows are left-aligned, so a row is as
        tall as its tallest tile and the canvas as wide as its widest row.  The rendered
        size is .wxh_actual and the unscaled size of the tiling itself is .content_wxh.

        '''
        return Tile(*args, p2s=self, **kwargs)

    def __allhex__(self, s: str) -> bool:
        return all(c in '0123456789abcdefABCDEF' for c in s)

    #
    # isTemplate() - is the argument the template for a small multiple?
    # - add other types as they become available
    #
    def isTemplate(self, _template_: Any) -> bool:
        _types_ = [XYp, Timep, Histop, LinkP, SpreadLinesP, Piep]
        try:
            from .chordp import ChP
            _types_.append(ChP)
        except ImportError:
            pass
        return isinstance(_template_, tuple(_types_))


    #
    # columnInDataFrame() - check if a column is in the dataframe
    # - considers transformation fields: a t-field (TField or legacy 'col|suffix')
    #   is present iff its base column is present, so it validates like a real column.
    # - this is the single source of truth for t-field-aware membership. Every
    #   user-field check on a spec that the component may resolve as a t-field routes
    #   through here (histop/piep/timep count & color, smallp's __isColumn__, xyp's
    #   transform builder). Checks that intentionally do NOT accept t-fields keep a
    #   plain `x in df.columns` on purpose: histop/piep `bin_by` (no bin transform is
    #   applied, so accepting a t-field would validate then crash at aggregation) and
    #   the graph components' relationship/field checks (node identity, not time).
    #
    def columnInDataFrame(self, column: Any, df: pl.DataFrame) -> bool:
        if self.isTField(column, df=df): column, _ = self.tFieldTuple(column)
        return column in df.columns

    #
    # Reserved internal column namespace
    # - the framework aliases dunder-style working columns ('__count__', '__bin__',
    #   '__row_count__', ...) into DataFrames during aggregation and rendering; a user
    #   column with one of those names silently produces wrong aggregates instead of an
    #   error, so the entire '__name__' pattern is reserved for the framework.
    # - the persisted subset below is tolerated on input: the framework writes these
    #   columns into self.df in place, so they legitimately re-enter components when a
    #   DataFrame round-trips (smallp panel dfs, interactive drill-down stack pushes,
    #   plot.df reuse). Each one is deterministically overwritten (aliased with_columns)
    #   or framework-managed before it is read, so tolerating them cannot corrupt
    #   aggregates the way an arbitrary reserved-name collision can.
    #
    _RESERVED_COLUMN_RE_    = re.compile(r'^__.+__$')
    _PERSISTED_COLUMNS_     = {'__p2s_index__',   # row index added by every component
                               '__bin__',         # histop/piep multi-field bin key
                               '__color__',       # histop/timep/piep multi-field color key
                               '__time_bin__',    # timep periodic time bin
                               '__lc_cat__'}      # linkp/chordp categorical link color
    _PERSISTED_COLUMNS_RE_  = re.compile(r'^__(?:rel\d+_(?:fm|to)_[ws][xy]|(?:fm|to)\d+)__$')
                              # linkp per-relationship world/screen positions; linkp/chordp/
                              # spreadlinesp concatenated tuple-endpoint columns

    #
    # checkReservedColumns() - raise if a dataframe uses the reserved '__name__' namespace
    # - call from each component's __validateInput__ so collisions error instead of
    #   silently producing wrong aggregates
    #
    def checkReservedColumns(self, df: pl.DataFrame | None, component_name: str) -> None:
        if df is None: return
        _bad_ = [c for c in df.columns
                 if self._RESERVED_COLUMN_RE_.match(c)
                 and c not in self._PERSISTED_COLUMNS_
                 and not self._PERSISTED_COLUMNS_RE_.match(c)]
        if _bad_:
            raise ValueError(
                f'{component_name}.__validateInput__(): DataFrame column(s) {_bad_} collide with the '
                f'reserved "__name__" namespace used for framework-internal columns; please rename them'
            )

    #
    # positionalDispatchHint() - a suffix appended to a "column not found" error when
    # the offending value was inferred from a *positional* argument (xyp's x/y, linkp's
    # relationships). Positional dispatch keys on Python type/shape, so an argument-order
    # mistake (or a value meant for another parameter, e.g. a color) is silently routed
    # to x/y/relationships and only surfaces downstream as a bare "column not found".
    # This makes the message name the likely root cause and point at the keyword form,
    # turning a confusing failure into a self-explaining one. Returns '' when the value
    # was supplied by keyword (the assignment was explicit, so no dispatch ambiguity).
    #
    def positionalDispatchHint(self, component: str, param_name: str, from_positional: bool) -> str:
        if not from_positional: return ''
        return (f' (note: {param_name} was inferred from a positional argument to {component}; '
                f'if this is an argument-order mistake or a value meant for another parameter, '
                f'pass it explicitly as {param_name}=... )')

    #
    # placeholderSVG() - the fallback canvas a component paints before (or in place
    # of) a real render: a background-filled rect plus a centered diagnostic message.
    #
    # Each component builds this unconditionally during __parseInput__ for "early
    # error visibility"; a successful render overwrites self.svg, so the message is
    # only ever *seen* when the render is skipped -- i.e. when no DataFrame was
    # supplied. Painting a visible "no data" message (rather than a silently blank
    # canvas) turns a plumbing mistake -- a df that never reached the component --
    # into something the user can see, while still letting dataless template
    # construction succeed: a template just reprs as "no data" until it's cloned
    # with data. No warning is emitted -- dataless construction is a legitimate,
    # common pattern (template building), so a warn-once here would be a false
    # positive; the visible message is diagnostic without being noisy.
    #
    # `notes` draws optional extra lines below the message (xyp uses it to echo the
    # x/y specs). None dimensions fall back to a small square so the canvas is
    # always well-formed even before auto-sizing has resolved.
    #
    def placeholderSVG(self, w: int | None, h: int | None, message: str = 'no data - no DataFrame supplied', notes: list | tuple = ()) -> str:
        w = w if isinstance(w, (int, float)) else 256
        h = h if isinstance(h, (int, float)) else 256
        _bg_ = self.colorTyped('background', 'default')
        _parts_ = [f'<svg x="0" y="0" width="{w}" height="{h}" font-family="{self.default_font}" xmlns="http://www.w3.org/2000/svg">',
                   f'<rect x="0" y="0" width="{w}" height="{h}" fill="{_bg_}" />']
        if message:
            _parts_.append(self.svgText(message, w / 2, h / 2, txt_h=12, anchor='middle'))
        _y_ = h / 2 + 18
        for _note_ in notes:
            _parts_.append(self.svgText(_note_, w / 2, _y_, txt_h=10, anchor='middle'))
            _y_ += 14
        _parts_.append('</svg>')
        return ''.join(_parts_)

    #
    # normalizeWxh() - the single, shared canvas-size validator/normalizer.
    #
    # Historically wxh validation was inconsistent: histop/piep/timep required a
    # *tuple of exactly two ints* (a list [128, 256] or floats raised), smallp had
    # its own None-tolerant check, and chordp/linkp/spreadlinesp/xyp validated
    # nothing at all (a bad wxh only surfaced as a cryptic unpack error deep in a
    # render method). This helper normalizes every component: it accepts any
    # 2-element sequence of numbers (tuple or list), coerces each element to int,
    # and returns a canonical (w, h) tuple used everywhere downstream. Bad input
    # raises a clear ValueError naming the component.
    #
    # allow_none=True (smallp only) permits exactly one dimension to be None so the
    # missing side can be auto-sized later; the None passes through uncoerced. A
    # bool is rejected even though it's an int subclass -- wxh=(True, 256) is a
    # mistake, not a 1-pixel canvas.
    #
    def normalizeWxh(self, wxh: Any, component_name: str, allow_none: bool = False) -> tuple:
        _hint_ = ('a 2-sequence of numbers, one of which may be None' if allow_none
                  else 'a 2-sequence of numbers')
        if isinstance(wxh, (str, bytes)) or not isinstance(wxh, (tuple, list)):
            raise ValueError(f'{component_name}.__validateInput__(): wxh must be {_hint_}, '
                             f'got {type(wxh).__name__} {wxh!r}')
        if len(wxh) != 2:
            raise ValueError(f'{component_name}.__validateInput__(): wxh must have exactly two '
                             f'elements, got {len(wxh)} ({wxh!r})')
        _out_: list = []
        for _i_, _v_ in enumerate(wxh):
            _side_ = 'width' if _i_ == 0 else 'height'
            if _v_ is None:
                if not allow_none:
                    raise ValueError(f'{component_name}.__validateInput__(): wxh {_side_} must be '
                                     f'a number, got None')
                _out_.append(None)
            elif isinstance(_v_, bool) or not isinstance(_v_, (int, float)):
                raise ValueError(f'{component_name}.__validateInput__(): wxh {_side_} must be '
                                 f'a number, got {type(_v_).__name__} {_v_!r}')
            else:
                _out_.append(int(_v_))
        if allow_none and _out_[0] is None and _out_[1] is None:
            raise ValueError(f'{component_name}.__validateInput__(): wxh cannot have both '
                             f'dimensions None -- at least one must be a number')
        return tuple(_out_)

    #
    # rejectBoolParam() - a display-name map is not an on/off flag.
    #
    # draw_node_labels= turns node labels on; node_labels= supplies the
    # {name: display_str} map.  The two names differ by a prefix, so the value meant for
    # one lands on the other -- and a bool there used to fail in whichever way the
    # parameter happened to be consumed downstream: stored and never looked at with the
    # draw flag off, or `object of type 'bool' has no len()` from the middle of the render
    # with it on.  Neither names the parameter, and neither names the fix.  (It is not a
    # hypothetical mistake: a linkp test in test_edge_case_inputs.py made it, rendered no
    # labels at all, and sat there asserting nothing about them.)
    #
    # linkp had been catching exactly this for link_labels= since edge labels landed; the
    # guard just never reached the parameters beside it.  This is that guard, in one place,
    # for every component that has a label map or a label filter.
    #
    # Bool only, deliberately.  These parameters each accept several shapes (label_only=
    # takes a set, a list, or a single name), so full type validation would be a larger and
    # riskier change; a bool is the one wrong value that is a predictable confusion rather
    # than a typo.  normalizeWxh() singles bools out from the other ints for the same
    # reason.
    #
    def rejectBoolParam(self, value: Any, component_name: str, param: str, shape: str, flag: str | None = None) -> None:
        if not isinstance(value, bool): return
        _use_ = f'; use {flag}= for that' if flag is not None else ''
        raise TypeError(f'{component_name}: {param}= is {shape}, not an on/off flag'
                        f'{_use_} (got {value!r})')

    #
    # formatMultiFieldValue() - render a bin/color grouping-key value for display.
    # Multi-field keys are joined internally with the non-printable MULTI_FIELD_SEP
    # (see the constant's note) to avoid collisions; this restores the visible '|'
    # separator for any text drawn to the user. A single-field value never contains
    # the separator, so this is a no-op there.
    #
    def formatMultiFieldValue(self, value: Any) -> str:
        return str(value).replace(self.MULTI_FIELD_SEP, '|')

    #
    # nullNode() / isNullNode() / nullNodeDisplay() - the private null partner of an
    # entity, materialized by linkp(null_nodes=True). See NULL_NODE_PREFIX for why the
    # partner is per-entity rather than one shared node. nullNodeDisplay() is the
    # NULL_NODE_PREFIX counterpart of formatMultiFieldValue(): the internal name is
    # non-printable, so anything drawn as text goes through it first.
    #
    def nullNode(self, entity: str) -> str:
        return f'{self.NULL_NODE_PREFIX}{entity}'

    def isNullNode(self, value: str) -> bool:
        return isinstance(value, str) and value.startswith(self.NULL_NODE_PREFIX)

    def nullNodeDisplay(self, value: str) -> str:
        return '(null)' if self.isNullNode(value) else value

    #
    # bgShape() - create a background record
    #
    def bgShape(self, shape: list | str, **fields: Any) -> _BackgroundShape_:
        '''Build a background entry that carries its own appearance.

        ``shape`` is any descriptor ``background=`` already accepts (a shapely geometry,
        a list of ``(x, y)`` world coordinates, an ``'<circle .../>'`` string, or an
        ``M/L/C/Z`` path string). Every appearance field defaults to ``p2s.INHERIT``,
        meaning "take the component's ``background_*`` parameter", so a bare descriptor
        and a record mix freely inside one ``background=`` dict::

            p2s.linkp(df, rels, pos=pos, background={
                'dmz':    subnet_polygon,                                  # bare -- inherits
                'flow 1': p2s.bgShape(curves, fill=None, stroke='#4060a0',
                                      stroke_opacity=0.35, label=None),
            })

        ``None`` on a field means *explicitly off*, which the ``background_*``
        parameters cannot express: ``fill=None`` leaves the shape unfilled (rather than
        filling it with the axis colour), ``stroke=None`` leaves it unstroked, and
        ``label=None`` suppresses the drawn label. ``stroke_opacity=``, ``dash=``,
        ``stroke_linecap=``, ``stroke_linejoin=`` and ``label=`` have no ``background_*``
        equivalent at all.

        Fields: ``fill``, ``fill_opacity``, ``stroke``, ``stroke_opacity``,
        ``stroke_width``, ``dash``, ``stroke_linecap``, ``stroke_linejoin``, ``label``,
        ``label_color``. Colour fields take ``'#rrggbb'`` or ``'vary'`` (hash the entry's
        key). The returned record is immutable.
        '''
        return _BackgroundShape_(shape, **fields)

    #
    # tField() - create a transformation field
    #
    def tField(self, column: str, _enum_: '_enums_.TimeLinearTypeP | _enums_.TimePeriodicTypeP') -> 'Polars2SVG.TField':
        '''Build a time-transformation field pairing a timestamp ``column`` with a
        binning ``_enum_`` (a ``TimeLinearTypeP`` or ``TimePeriodicTypeP`` member).

        Returns a :class:`TField` — the preferred, typed replacement for the legacy
        ``'column|suffix'`` string. Pass it anywhere a field name is accepted (chiefly
        ``timep``'s ``time=`` and ``smallp``'s ``cycle_by``)::

            p2s.timep(df, p2s.tField('timestamp', p2s.PT_mp))       # bin by month (periodic)
            p2s.timep(df, p2s.tField('timestamp', p2s.LT_Y_m_dp))   # bin by calendar day (linear)
        '''
        return self.TField(column, _enum_)

    #
    # isTField() - check if a column is a transformation field
    # - a TField instance is always a t-field.
    # - a plain 'column|suffix' string is DEPRECATED: it's only recognized as a t-field
    #   when the literal string isn't itself a real column in df (so a real column named
    #   e.g. 'price|mp' is never hijacked into a month transform). When df is None (e.g.
    #   a dataless template) there's no schema to consult, so a recognized suffix parses
    #   as a t-field; the guard re-evaluates once a df is supplied, since
    #   __validateInput__ re-runs on every clone/render.
    # - each accepted legacy string emits a one-time (per-process, via the logger's
    #   OnceFilter) deprecation warning pointing at p2s.tField().
    #
    def isTField(self, column: Any, df: pl.DataFrame | None = None) -> bool:
        if isinstance(column, self.TField): return True
        if not isinstance(column, str) or '|' not in column: return False
        _suffix_ = column[column.rindex('|')+1:]
        if _suffix_ not in self.suffix_to_enum: return False
        if df is not None and column in df.columns: return False
        _base_ = column[:column.rindex('|')]
        self.logger.warning(
            f"polars2svg: implicit t-field string '{column}' is deprecated; "
            f"use p2s.tField('{_base_}', p2s.{self.suffix_to_enum[_suffix_].name}) instead"
        )
        return True

    #
    # tFieldTuple() - split a transformation field into (column, enum)
    #
    def tFieldTuple(self, tfield: Any) -> tuple:
        if isinstance(tfield, self.TField): return tfield.column, tfield.transform
        if not isinstance(tfield, str) or '|' not in tfield or tfield[tfield.rindex('|')+1:] not in self.suffix_to_enum:
            raise InvalidSpecError(f'XYp.tFieldTuple(): column is not a t-field {tfield=}')
        i = tfield.rindex('|')
        _suffix_ = tfield[i+1:]
        column   = tfield[:i]
        return column, self.suffix_to_enum[_suffix_]

    #
    # polarsOperationForTField() - return a polars expression for a tField string
    #
    def polarsOperationForTField(self, tfield: Any) -> pl.Expr:
        _column_, _enum_ = self.tFieldTuple(tfield)
        return self.polarsOperationForEnum(_column_, _enum_)

    #
    # tFieldAccepts() - return what column types (as a set) a transformation field accepts
    #
    def tFieldAccepts(self, tfield: Any) -> set:
        _column_, _enum_ = self.tFieldTuple(tfield)
        if isinstance(_enum_, self.TimeLinearTypeP) or isinstance(_enum_, self.TimePeriodicTypeP):
            return {pl.Date, pl.Datetime}
        raise InvalidSpecError(f'XYp.tFieldAccepts(): unknown enumeration {_enum_}')

    #
    # warnIfTFieldAliasCollides() - an explicit TField's derived-column alias ('column|suffix')
    # can itself be a real column in df. The transform still wins (the caller applies it
    # regardless), but a one-time warning flags that the real column is being shadowed.
    #
    # df is Optional because callers pass component .df, which is None on a template
    # instance -- the body has always guarded for it.  The annotation said otherwise
    # until T7 typed the components' df and the mismatch became visible.
    def warnIfTFieldAliasCollides(self, tfield: Any, df: pl.DataFrame | None, component_name: str) -> None:
        if df is not None and isinstance(tfield, self.TField) and str(tfield) in df.columns:
            self.logger.warning(f'{component_name}: column {str(tfield)!r} is shadowed by the derived t-field column for {tfield!r}')

    #
    # numericColumn - check if a column is numeric (integer or float)
    #
    def numericColumn(self, df: pl.DataFrame, column: Any) -> bool:
        return column in df.select(cs.integer()).columns or column in df.select(cs.float()).columns

    #
    # Dtype-keyed inference logging -------------------------------------------
    #
    # Several parameters (`count=`, `color=`) infer their meaning from the
    # *dtype* of a bare field spec: a numeric column aggregates by sum / colors
    # by magnitude spectrum, a non-numeric column aggregates by distinct-count /
    # colors categorically.  That inference is silent, so an upstream schema
    # change (string IDs becoming integer IDs) flips the interpretation with no
    # signal.  The explicit enums are the way to pin intent -- (field, SCALARp)/
    # (field, SETp) for count, (field, CSETp)/(field, CMAGNITUDE_SUMp) for color
    # -- and these helpers make the *inferred* choice diagnosable: a one-time
    # (per distinct message, via the logger OnceFilter) INFO log naming the field,
    # the interpretation picked, and the enum that would override it.  INFO is off
    # by default so normal use stays quiet; enable it (logging.getLogger(
    # 'polars2svg_logger').setLevel(logging.INFO) + a handler) to see the choices.
    #
    def logDtypeKeyedCount(self, component_name: str, field: Any, is_numeric: bool) -> None:
        if is_numeric:
            self.logger.info(f"{component_name}: count={field!r} is numeric -> sum(); "
                             f"pass ('{field}', p2s.SETp) to force distinct-count")
        else:
            self.logger.info(f"{component_name}: count={field!r} is non-numeric -> n_unique() (distinct-count); "
                             f"pass ('{field}', p2s.SCALARp) to force sum")

    def logDtypeKeyedColor(self, component_name: str, field: Any, is_numeric: bool) -> None:
        if is_numeric:
            self.logger.info(f"{component_name}: color={field!r} is numeric -> magnitude spectrum; "
                             f"pass ('{field}', p2s.CSETp) to force categorical")
        else:
            self.logger.info(f"{component_name}: color={field!r} is non-numeric -> categorical; "
                             f"a numeric spectrum needs a numeric field (e.g. ('{field}', p2s.CMAGNITUDE_SUMp))")

    #
    # dateColumn - check if a column is a date
    #
    def dateColumn(self, df: pl.DataFrame, column: str) -> bool:
        return column in df.select(cs.date()).columns

    #
    # timeColumn - check if a column is a time
    #
    def timeColumn(self, df: pl.DataFrame, column: str) -> bool:
        return column in df.select(cs.time()).columns

    #
    # datetimeColumn - check if a column is a datetime
    #
    def dateTimeColumn(self, df: pl.DataFrame, column: str) -> bool:
        return column in df.select(cs.datetime()).columns
