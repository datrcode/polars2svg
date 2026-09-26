#
# p2s_enums.py - the package's enumerations, and the parameter-spec unions
# built out of them.
#
# They used to be nested inside Polars2SVG, which put them downstream of every
# mixin in the import graph: a mixin could not name the type of an enum it reads
# off `self`, so its host-attribute declarations said `Any`.  This module imports
# nothing from the package, so every mixin can import it without a cycle, and
# Polars2SVG re-exposes each class as a class attribute -- p2s.ColorTypeP and
# Polars2SVG.ColorTypeP still resolve exactly as before.
#
# Members are bound onto the instance by name (p2s.SCALARp, p2s.BARCHARTp, ...)
# by the ENUM_CLASSES loop in Polars2SVG.__init__.
#
from enum import Enum


class P2SEnum(Enum):
    '''Member-less base class for every polars2svg enumeration.

    Member-less on purpose: an Enum that defines members cannot be subclassed, so
    this stays empty and the concrete enums below derive from it.  That gives the
    package one ``isinstance(x, P2SEnum)`` that recognises a member of any of them,
    which is what a single grab-bag enum would otherwise be needed for.
    '''
    pass


class FieldTypeP(P2SEnum):
    '''How a ``count=``/``order=`` field is aggregated. Pair with a field to
    override the dtype-keyed default: ``('field', p2s.SCALARp)`` forces ``sum``,
    ``('field', p2s.SETp)`` forces distinct-count (``n_unique``). Members are
    exposed on the instance as ``p2s.SCALARp`` / ``p2s.SETp``.'''
    SCALARp = 1 # Treat a field as a scalar -- e.g., (1 + 1 + 2 + 2) = 4
    SETp    = 2 # Treat a field as a set -- e.g., (1 + 1 + 2 + 2) = set(1,2) = len(set(1,2)) = 2

class StatisticP(P2SEnum):
    '''Aggregation statistic for a numeric field, used in ``('field', <stat>)``
    specs for ``order=`` (histop) and magnitude coloring. Exposed on the instance
    as ``p2s.MINp``, ``p2s.MEANp``, etc.'''
    MINp    = 1
    MEDIANp = 2
    MEANp   = 3
    MAXp    = 4
    STDp    = 5
    SUMp    = 6

class ColorTypeP(P2SEnum):
    '''Color-encoding modes, used as ``color=('field', <enum>)`` (or bare, e.g.
    ``color=p2s.CROW_MAGNITUDEp``). The ``C``-prefix distinguishes them from the
    counting enums. Broadly: ``CSET*`` treat the field categorically; ``CMAGNITUDE_*``
    / ``CSTRETCHED_*`` map a numeric statistic onto ``p2s.spectrum_palette`` (linear
    vs. rank-equalized); ``CROW_*`` color by raw row count (``pl.len()``), independent
    of ``count=``. Exposed on the instance by name, e.g. ``p2s.CSETp``.'''
    CSETp              =  1 # if set_size == 1, color == color(object-in-set) else generate set color                 (xyp, timep)
    CSET_MAGNITUDEp    =  2 # count the items in the set, scale across a spectrum                                     (xyp)
    CSET_STRETCHEDp    =  3 # count the items in the set, give that value an equal amount of the spectrum             (xyp)
    CROW_MAGNITUDEp    =  4 # count the number of rows at that pixel, scale across a spectrum                         (xyp)
    CROW_STRETCHEDp    =  5 # count the number of rows at that pixel, give that value an equal amount of the spectrum (xyp)
    CMAGNITUDE_SUMp    =  6 # sum a field (numeric field), scale across a spectrum                                    (xyp)
    CMAGNITUDE_MINp    =  7 # min a field (numeric field), scale across a spectrum                                    (xyp)
    CMAGNITUDE_MEDIANp =  8 # median a field (numeric field), scale across a spectrum                                 (xyp)
    CMAGNITUDE_MEANp   =  9 # mean a field (numeric field), scale across a spectrum                                   (xyp)
    CMAGNITUDE_MAXp    = 10 # max a field (numeric field), scale across a spectrum                                    (xyp)
    CSTRETCHED_SUMp    = 11 # sum a field (numeric field), give that value an equal amount of the spectrum            (xyp)
    CSTRETCHED_MINp    = 12 # min a field (numeric field), scale across a spectrum                                    (xyp)
    CSTRETCHED_MEDIANp = 13 # median a field (numeric field), scale across a spectrum                                 (xyp)
    CSTRETCHED_MEANp   = 14 # mean a field (numeric field), scale across a spectrum                                   (xyp)
    CSTRETCHED_MAXp    = 15 # max a field (numeric field), scale across a spectrum                                    (xyp)

class TimeLinearTypeP(P2SEnum):
    '''Linear (monotonic) time-binning resolutions — each ``LT_*`` member bins a
    timestamp down to a calendar granularity (year, month, day, 4-hour, …) while
    preserving chronological order. Used via ``p2s.tField(col, p2s.LT_Y_mp)`` or a
    ``('field', <enum>)`` time spec in ``timep``. Contrast ``TimePeriodicTypeP``,
    which folds time into a repeating cycle.'''
    LT_Yp                = 1
    LT_Y_Qp              = 2
    LT_Y_mp              = 3
    LT_Y_m_dp            = 4
    LT_Y_m_d_Hp          = 5
    LT_Y_m_d_H_Mp        = 6
    LT_Y_m_d_H_M_Sp      = 7
    LT_Y_m_d_4Hp         = 8   # 4-hour bins
    LT_Y_m_d_H_15Mp      = 9   # 15-minute bins
    LT_Y_m_d_H_M_15Sp   = 10   # 15-second bins

class TimePeriodicTypeP(P2SEnum):
    '''Periodic (cyclic) time-binning resolutions — each ``PT_*`` member folds a
    timestamp into a repeating cycle (quarter, month, day-of-week, hour, …), so all
    Mondays or all Januaries collapse into one bin. Used via
    ``p2s.tField(col, p2s.PT_DoWp)`` or a ``('field', <enum>)`` time spec in ``timep``.
    Contrast ``TimeLinearTypeP``, which keeps chronological order.'''
    PT_Qp       = 1   # Quarter
    PT_mp       = 2   # Month
    PT_m_dp     = 3   # Month Day      (note that this uses a leap year to determine the number of days)
    PT_m_d_Hp   = 4   # Month Day Hour (note that this uses a leap year to determine the number of days)
    PT_DoYp     = 5   # Day of Year    (note that this does *NOT* use a leap year to determine the number of days)
    PT_DoWp     = 6   # Day of Week
    PT_DoW_Hp   = 7   # Day of Week Hour
    PT_DoW_H_Mp = 8   # Day of Week Hour Minute
    PT_dp       = 9   # Day (of Month)
    PT_d_Hp     = 10  # Day (of Month) Hour
    PT_d_H_Mp   = 11  # Day (of Month) Hour Minute
    PT_Hp       = 12  # Hour
    PT_H_Mp     = 13  # Hour Minute
    PT_H_M_Sp   = 14  # Hour Minute Second
    PT_Mp       = 15  # Minute
    PT_M_Sp     = 16  # Minute Second
    PT_Sp       = 17  # Second


#
# The render-option enums.  These were one 42-member `RenderEnumsP` grab-bag; they
# are now one class per decision, so a parameter that accepts exactly one of them
# can say so.  The member *names* are unchanged and still reach callers as
# p2s.BARCHARTp / p2s.SM_COLOR / ..., so this is not a caller-visible change.
#
# Values restart at 1 in each class.  Nothing reads `.value` -- members are
# compared by identity -- and members of different Enum classes never compare
# equal, so the reuse is safe.
#


class RowCountP(P2SEnum):
    '''The "count by raw row count" sentinel -- the default for ``count=`` and
    ``order=``.  Exposed as ``p2s.ROW_COUNTp``.'''
    ROW_COUNTp = 1 # for certain transformations, treat the row count as the parameter


class DistributionPlacementP(P2SEnum):
    '''Where an xyp marginal distribution is drawn, and how it is binned.
    Exposed as ``p2s.DISTRIBUTION_INSIDEp`` etc.'''
    DISTRIBUTION_INSIDEp  = 1 # xy default (doesn't require specification)
    DISTRIBUTION_OUTSIDEp = 2
    DISTRIBUTION_AUTOBINp = 3 # xy default (doesn't require specification)


class DistributionScaleP(P2SEnum):
    '''What an xyp distribution's magnitude axis is scaled against.  Exactly one
    may appear in a ``*_distributions=`` spec.'''
    DISTRIBUTION_COLOR_MIN_TO_COLOR_MAX = 1
    DISTRIBUTION_ZERO_TO_COLOR_MAX      = 2 # xy default
    DISTRIBUTION_ALL_MIN_TO_ALL_MAX     = 3
    DISTRIBUTION_ZERO_TO_ALL_MAX        = 4


class LineWidthP(P2SEnum):
    '''How an xyp ``line=`` width is determined.'''
    LINEWIDTH_DOTSIZE_MEAN      = 1
    LINEWIDTH_DOTSIZE_VARIABLE  = 2
    LINEWIDTH_DOTSIZE_SPECIFIED = 3 # xy default


class LineStyleP(P2SEnum):
    '''How an xyp ``line=`` dash pattern is determined.'''
    LINESTYLE_SOLID     = 1 # xy default
    LINESTYLE_DOTTED    = 2
    LINESTYLE_SPECIFIED = 3


class LineColorP(P2SEnum):
    '''How an xyp ``line=`` color is determined.'''
    LINECOLOR_GROUPBY   = 1 # xy default
    LINECOLOR_FIELD     = 2
    LINECOLOR_SPECIFIED = 3


class LineOpacityP(P2SEnum):
    '''How an xyp ``line=`` opacity is determined.'''
    LINEOPACITY_FIELD_MEAN     = 1
    LINEOPACITY_FIELD_VARIABLE = 2
    LINEOPACITY_100            = 3 # xy default
    LINEOPACITY_75             = 4
    LINEOPACITY_50             = 5
    LINEOPACITY_25             = 6
    LINEOPACITY_10             = 7


class SmallMultipleP(P2SEnum):
    '''Attributes a small-multiple grid shares across its panels, passed as a set
    to ``sm_shared=``.  The ``SM_X``/``SM_Y``/``SM_COUNT``/``SM_COLOR`` members
    apply to the axis-based components; ``SM_SLICE_ORDERp``/``SM_PARTOFWHOLEp``
    are piep's.  One class because one parameter takes all of them.'''
    SM_X            = 1
    SM_Y            = 2
    SM_COUNT        = 3
    SM_COLOR        = 4
    SM_SLICE_ORDERp = 5 # keep the same slice order & colors across panels
    SM_PARTOFWHOLEp = 6 # fade the "all rows" chart behind, fill each slice's share


class BarStyleP(P2SEnum):
    '''Bar rendering style for temporal barcharts and histograms (timep, histop),
    passed as ``style=``.'''
    BARCHARTp        = 1
    BOXPLOTp         = 2
    BOXPLOT_W_SWARMp = 3
    STACKEDBARp      = 4


class SelectShapeP(P2SEnum):
    '''Shape of an interactive node selection.'''
    SELECT_CIRCLEp     = 1
    SELECT_HORIZONTALp = 2
    SELECT_VERTICALp   = 3


class NodeColorP(P2SEnum):
    '''Node coloring mode for the graph components.'''
    COLOR_BY_NODE_NAME = 1


class PieStyleP(P2SEnum):
    '''Piechart rendering style (piep), passed as ``style=``.'''
    PIEp    = 1
    DONUTp  = 2
    WAFFLEp = 3


class OrderBucketP(P2SEnum):
    '''Placeholder bucket for a partial ``order=`` (chordp ``order=``, xyp
    ``x_order=``/``y_order=``).'''
    REMAINDERp = 1 # placeholder in order= -- values absent from order= merge into
                   # one bucket at the sentinel's position.  Without it, unlisted
                   # values are appended in sorted order and keep their identity.


class OrderKeyP(P2SEnum):
    '''A sort-key ``order=`` (histop) that is neither ``p2s.ROW_COUNTp`` nor a field.
    Exposed as ``p2s.LABELp``.'''
    LABELp = 1 # order by the labels themselves: numbers as numbers, text alphabetically.
               # An enum, not a string, because a string order= names the column to sum.


#
# RENDER_ENUM_CLASSES / RenderEnum - what `RenderEnumsP` used to mean.  The tuple
# is for isinstance(); the union alias is the same thing spelled as a type, and
# isinstance() accepts it too.  Keep the two in step.
#
RENDER_ENUM_CLASSES: tuple[type[Enum], ...] = (
    RowCountP, DistributionPlacementP, DistributionScaleP,
    LineWidthP, LineStyleP, LineColorP, LineOpacityP,
    SmallMultipleP, BarStyleP, SelectShapeP, NodeColorP, PieStyleP, OrderBucketP,
    OrderKeyP,
)

RenderEnum = (RowCountP | DistributionPlacementP | DistributionScaleP
              | LineWidthP | LineStyleP | LineColorP | LineOpacityP
              | SmallMultipleP | BarStyleP | SelectShapeP | NodeColorP
              | PieStyleP | OrderBucketP | OrderKeyP)


#
# ENUM_CLASSES - the registry Polars2SVG.__init__ walks to bind every member onto
# the instance by name.  Listing a class here is the only step needed to expose
# its members as p2s.<MEMBER>; tests/test_typing_surface.py checks the result
# against the declaration block in polars2svg.py, so a class added here without a
# matching declaration fails the suite.
#
ENUM_CLASSES: tuple[type[Enum], ...] = (
    FieldTypeP,
    StatisticP,
    ColorTypeP,
    TimeLinearTypeP,
    TimePeriodicTypeP,
) + RENDER_ENUM_CLASSES


#
# CountSpec / ColorSpec - the count= and color= contracts, transcribed from the
# aggregation rule in parameter_conventions.md.  `TField` subclasses `str`, so
# `str` already covers a t-field.
#
CountSpec = RowCountP | str | tuple[str | FieldTypeP, ...]

ColorSpec = (ColorTypeP | RowCountP | str | dict[str, str]
             | tuple[str | ColorTypeP | FieldTypeP | RowCountP, ...] | None)

#
# ResolvedColorSpec - what a render helper receives *after* a component's
# __parseInput__ has resolved color= down to a column name.  Narrower than
# ColorSpec on purpose: by this point it is never a bare enum, a dict or None,
# and typing it ColorSpec instead makes the helpers fail to type-check (they
# append it to a list of column names).  The two are genuinely different
# contracts that used to share one `Any`.
#
ResolvedColorSpec = str | tuple[str | ColorTypeP | FieldTypeP | RowCountP, ...]
