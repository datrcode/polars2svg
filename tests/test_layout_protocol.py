import unittest
import networkx as nx

from polars2svg.layout_protocol              import LayoutAlgorithm
from polars2svg.polars_force_directed_layout import PolarsForceDirectedLayout
from polars2svg.convey_proximity_layout      import ConveyProximityLayout
from polars2svg.mds_at_scale                 import LandmarkMDSLayout, PivotMDSLayout

try:
    from polars2svg.tfdp_layout import TFDPLayout
    _TFDP_AVAILABLE = True
except ImportError:
    _TFDP_AVAILABLE = False


def _g():
    return nx.cycle_graph(5)


class TestLayoutAlgorithmProtocol(unittest.TestCase):

    def test_polars_force_directed_satisfies_protocol(self):
        layout = PolarsForceDirectedLayout(_g(), iterations=3)
        self.assertIsInstance(layout, LayoutAlgorithm)

    def test_convey_proximity_satisfies_protocol(self):
        layout = ConveyProximityLayout(_g(), iterations_min=2)
        self.assertIsInstance(layout, LayoutAlgorithm)

    def test_landmark_mds_satisfies_protocol(self):
        layout = LandmarkMDSLayout(_g())
        self.assertIsInstance(layout, LayoutAlgorithm)

    def test_pivot_mds_satisfies_protocol(self):
        layout = PivotMDSLayout(_g())
        self.assertIsInstance(layout, LayoutAlgorithm)

    @unittest.skipUnless(_TFDP_AVAILABLE, 'mlx not installed')
    def test_tfdp_satisfies_protocol(self):
        layout = TFDPLayout(_g(), max_iter=3, seed=0)
        self.assertIsInstance(layout, LayoutAlgorithm)

    def test_object_without_results_does_not_satisfy_protocol(self):
        class NoResults:
            pass
        self.assertNotIsInstance(NoResults(), LayoutAlgorithm)

    def test_object_with_results_satisfies_protocol(self):
        class HasResults:
            def results(self):
                return {}
        self.assertIsInstance(HasResults(), LayoutAlgorithm)

    def test_results_returns_dict(self):
        layout = PolarsForceDirectedLayout(_g(), iterations=3)
        self.assertIsInstance(layout.results(), dict)

    def test_results_keys_are_all_nodes(self):
        g = _g()
        layout = PolarsForceDirectedLayout(g, iterations=3)
        self.assertEqual(set(layout.results().keys()), set(g.nodes()))


if __name__ == '__main__':
    unittest.main()
