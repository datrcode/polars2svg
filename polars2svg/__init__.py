from importlib.metadata import version as _pkg_version, PackageNotFoundError
try:
    __version__ = _pkg_version('polars2svg')
except PackageNotFoundError:
    __version__ = '0.0.0.dev0'

# Every relative import below uses the redundant `X as X` form.  That is not a
# typo and it is not something to tidy away.  This package ships py.typed, so
# the typing spec's library-interface rule applies: a name imported into a
# module is PRIVATE unless it is re-exported explicitly.  Without the aliases,
# `from polars2svg import Polars2SVG` is an ERROR under pyright/Pylance
# (reportPrivateImportUsage, its default mode -- what VS Code runs) and under
# mypy --strict -- for every user who has a type checker on, and first of all
# for the <Component>Kwargs TypedDicts below, which exist only for type
# checking.
#
# __all__ is the other spelling of this and is NOT usable here: the four groups
# at the foot are guarded by try/except ImportError and must stay absent when
# their extra is not installed (test_package_exports.py pins that), which a
# static __all__ would break, and a conditional `__all__ +=` is not reliably
# honoured by static checkers.  The alias form needs no list and works per-name
# inside a try.
#
# See 20260915_fable_code_audit.md H1.  Two things guard it: [tool.mypy]
# no_implicit_reexport, and test_init_reexports_are_explicit in
# tests/test_security_automation.py.
from .exceptions                   import (Polars2SVGError  as Polars2SVGError,
                                           InvalidSpecError as InvalidSpecError,
                                           DataError        as DataError)
from .polars2svg                   import Polars2SVG as Polars2SVG
from .p2s_legend_mixin             import LegendInfo as LegendInfo
from .p2s_background_mixin         import (BackgroundShape as BackgroundShape,
                                           INHERIT         as INHERIT)
TField = Polars2SVG.TField
from .layout_protocol              import LayoutAlgorithm as LayoutAlgorithm
# Profile A (SECURITY.md) output contract.  Public because it is the
# enforcement point: an appliance calls assertOutputContract() on what it is
# about to serve.  Reports, never rewrites -- it is not a sanitizer.
from .svg_contract                 import (ALLOWED_ATTRIBUTES   as ALLOWED_ATTRIBUTES,
                                           ALLOWED_ELEMENTS     as ALLOWED_ELEMENTS,
                                           OutputContractError  as OutputContractError,
                                           Violation            as Violation,
                                           assertOutputContract as assertOutputContract,
                                           checkOutputContract  as checkOutputContract)

# Per-component keyword-argument TypedDicts.  Each factory method is typed
# `**kwargs: Unpack[<Component>Kwargs]`, so a checker flags a misspelled
# parameter at the call site and editors complete the parameter set.  Exported
# so callers can annotate their own kwargs dicts:
#
#     opts: p2s.XYpKwargs = {'dot_size': 6, 'wxh': (400, 300)}
#     p2s.xyp(df, x='a', y='b', **opts)
from .xyp                          import XYpKwargs as XYpKwargs
from .smallp                       import SmallpKwargs as SmallpKwargs
from .timep                        import TimepKwargs as TimepKwargs
from .histop                       import HistopKwargs as HistopKwargs
from .piep                         import PiepKwargs as PiepKwargs
from .linkp                        import LinkPKwargs as LinkPKwargs
from .spreadlinesp                 import SpreadLinesPKwargs as SpreadLinesPKwargs
from .tile                         import TileKwargs as TileKwargs
# Background producer, not a layout -- numpy + polars only, so unlike the graph
# layouts below it carries no optional-extra guard.
from .flow_field_background        import FlowFieldBackground as FlowFieldBackground
from .laguerre_voronoi             import (laguerre_voronoi as laguerre_voronoi,
                                           QuadTree         as QuadTree)

# These standalone layout classes need networkx (and, for the MDS pair,
# scipy/scikit-learn too) — an optional 'layouts' extra, not a core dependency.
# Guarded the same way TFDPLayout already was, so `import polars2svg` succeeds
# without them installed; the names are simply absent when they're missing.
# ChPKwargs lives behind the same guard as ChP itself: chordp's node ordering
# needs scipy, so importing it here eagerly would make `import polars2svg`
# require the 'layouts' extra.
try:
    from .chordp import ChPKwargs as ChPKwargs
except ImportError:
    pass

try:
    from .mds_at_scale import (LandmarkMDSLayout as LandmarkMDSLayout,
                               PivotMDSLayout    as PivotMDSLayout)
except ImportError:
    pass

try:
    from .tfdp_layout import (TFDPLayout  as TFDPLayout,
                              gpu_backend as gpu_backend)
except ImportError:
    pass

try:
    from .ncp_layout import (NCPLayout                     as NCPLayout,
                             NeighborhoodPreservingPacking as NeighborhoodPreservingPacking)
except ImportError:
    pass
