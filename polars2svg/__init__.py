from importlib.metadata import version as _pkg_version, PackageNotFoundError
try:
    __version__ = _pkg_version('polars2svg')
except PackageNotFoundError:
    __version__ = '0.0.0.dev0'

from .exceptions                   import Polars2SVGError, InvalidSpecError, DataError
from .polars2svg                   import Polars2SVG
from .p2s_legend_mixin             import LegendInfo
from .p2s_background_mixin         import BackgroundShape, INHERIT
TField = Polars2SVG.TField
from .layout_protocol              import LayoutAlgorithm

# Per-component keyword-argument TypedDicts.  Each factory method is typed
# `**kwargs: Unpack[<Component>Kwargs]`, so a checker flags a misspelled
# parameter at the call site and editors complete the parameter set.  Exported
# so callers can annotate their own kwargs dicts:
#
#     opts: p2s.XYpKwargs = {'dot_size': 6, 'wxh': (400, 300)}
#     p2s.xyp(df, x='a', y='b', **opts)
from .xyp                          import XYpKwargs
from .smallp                       import SmallpKwargs
from .timep                        import TimepKwargs
from .histop                       import HistopKwargs
from .piep                         import PiepKwargs
from .linkp                        import LinkPKwargs
from .spreadlinesp                 import SpreadLinesPKwargs
from .tile                         import TileKwargs
# Background producer, not a layout -- numpy + polars only, so unlike the graph
# layouts below it carries no optional-extra guard.
from .flow_field_background        import FlowFieldBackground
from .laguerre_voronoi             import laguerre_voronoi, QuadTree

# These standalone layout classes need networkx (and, for the MDS pair,
# scipy/scikit-learn too) — an optional 'layouts' extra, not a core dependency.
# Guarded the same way TFDPLayout already was, so `import polars2svg` succeeds
# without them installed; the names are simply absent when they're missing.
# ChPKwargs lives behind the same guard as ChP itself: chordp's node ordering
# needs scipy, so importing it here eagerly would make `import polars2svg`
# require the 'layouts' extra.
try:
    from .chordp import ChPKwargs
except ImportError:
    pass

try:
    from .mds_at_scale import LandmarkMDSLayout, PivotMDSLayout
except ImportError:
    pass

try:
    from .tfdp_layout import TFDPLayout, gpu_backend
except ImportError:
    pass

try:
    from .ncp_layout import NCPLayout, NeighborhoodPreservingPacking
except ImportError:
    pass
