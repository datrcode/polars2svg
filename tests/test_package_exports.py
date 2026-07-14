import unittest

import polars2svg
from polars2svg import (
    Polars2SVG,
    LayoutAlgorithm,
    PolarsForceDirectedLayout,
    ConveyProximityLayout,
    LandmarkMDSLayout,
    PivotMDSLayout,
)


class TestPackageExports(unittest.TestCase):

    def test_polars2svg_exported(self):
        self.assertIs(polars2svg.Polars2SVG, Polars2SVG)

    def test_layout_algorithm_protocol_exported(self):
        self.assertIs(polars2svg.LayoutAlgorithm, LayoutAlgorithm)

    def test_polars_force_directed_exported(self):
        self.assertIs(polars2svg.PolarsForceDirectedLayout, PolarsForceDirectedLayout)

    def test_convey_proximity_exported(self):
        self.assertIs(polars2svg.ConveyProximityLayout, ConveyProximityLayout)

    def test_landmark_mds_exported(self):
        self.assertIs(polars2svg.LandmarkMDSLayout, LandmarkMDSLayout)

    def test_pivot_mds_exported(self):
        self.assertIs(polars2svg.PivotMDSLayout, PivotMDSLayout)

    def test_tfdp_exported_when_mlx_available(self):
        # If TFDPLayout was importable, it should be on the package namespace.
        try:
            from polars2svg.tfdp_layout import TFDPLayout
            _available = True
        except ImportError:
            _available = False

        if _available:
            self.assertIs(polars2svg.TFDPLayout, TFDPLayout)
        else:
            self.assertFalse(hasattr(polars2svg, 'TFDPLayout'))


if __name__ == '__main__':
    unittest.main()
