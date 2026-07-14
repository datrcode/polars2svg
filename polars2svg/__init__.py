from importlib.metadata import version as _pkg_version, PackageNotFoundError
try:
    __version__ = _pkg_version('polars2svg')
except PackageNotFoundError:
    __version__ = '0.0.0.dev0'

from .exceptions                   import Polars2SVGError, InvalidSpecError, DataError
from .polars2svg                   import Polars2SVG
from .p2s_legend_mixin             import LegendInfo
TField = Polars2SVG.TField
from .layout_protocol              import LayoutAlgorithm
from .laguerre_voronoi             import laguerre_voronoi, QuadTree

# These standalone layout classes need networkx (and, for the MDS pair,
# scipy/scikit-learn too) — an optional 'layouts' extra, not a core dependency.
# Guarded the same way TFDPLayout already was, so `import polars2svg` succeeds
# without them installed; the names are simply absent when they're missing.
try:
    from .polars_force_directed_layout import PolarsForceDirectedLayout
except ImportError:
    pass

try:
    from .convey_proximity_layout import ConveyProximityLayout
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
