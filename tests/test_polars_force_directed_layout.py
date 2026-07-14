import unittest
import networkx as nx

from polars2svg.polars_force_directed_layout import PolarsForceDirectedLayout


def _cycle(n=5):
    return nx.cycle_graph(n)


def _path(n=4):
    return nx.path_graph(n)


class TestPFDLResultsContract(unittest.TestCase):

    def test_results_is_dict(self):
        r = PolarsForceDirectedLayout(_cycle(), iterations=5).results()
        self.assertIsInstance(r, dict)

    def test_results_contains_all_nodes(self):
        g = _cycle(6)
        r = PolarsForceDirectedLayout(g, iterations=5).results()
        self.assertEqual(set(r.keys()), set(g.nodes()))

    def test_results_values_are_float_pairs(self):
        r = PolarsForceDirectedLayout(_path(4), iterations=5).results()
        for v in r.values():
            self.assertEqual(len(v), 2)
            self.assertIsInstance(float(v[0]), float)
            self.assertIsInstance(float(v[1]), float)

    def test_two_node_graph(self):
        g = nx.path_graph(2)
        r = PolarsForceDirectedLayout(g, iterations=5).results()
        self.assertEqual(set(r.keys()), {0, 1})

    def test_single_node_graph_degenerate(self):
        g = nx.Graph()
        g.add_node('a')
        r = PolarsForceDirectedLayout(g, iterations=5).results()
        self.assertIn('a', r)


class TestPFDLStaticNodes(unittest.TestCase):

    def test_static_node_position_preserved(self):
        g = _path(4)
        initial_pos = {0: (0.0, 0.0), 1: (0.5, 0.5), 2: (1.0, 0.0), 3: (1.5, 0.5)}
        r = PolarsForceDirectedLayout(g, pos=dict(initial_pos), static_nodes={0}, iterations=20).results()
        self.assertAlmostEqual(r[0][0], 0.0, places=10)
        self.assertAlmostEqual(r[0][1], 0.0, places=10)

    def test_static_node_unchanged_while_others_move(self):
        g = _cycle(5)
        initial_pos = {i: (float(i), 0.0) for i in g.nodes()}
        r = PolarsForceDirectedLayout(g, pos=dict(initial_pos), static_nodes={0}, iterations=15).results()
        self.assertAlmostEqual(r[0][0], 0.0, places=10)
        self.assertAlmostEqual(r[0][1], 0.0, places=10)

    def test_multiple_static_nodes(self):
        g = _path(5)
        initial_pos = {i: (float(i), 0.0) for i in g.nodes()}
        r = PolarsForceDirectedLayout(g, pos=dict(initial_pos), static_nodes={0, 4}, iterations=15).results()
        self.assertAlmostEqual(r[0][0], 0.0, places=10)
        self.assertAlmostEqual(r[4][0], 4.0, places=10)

    def test_integer_pos_values_accepted(self):
        # pos values from linkp can be int (e.g. [0, 0]) when the user passes integer
        # coordinates — Polars strict mode rejects mixed int/float lists; ensure floats
        # are coerced before DataFrame construction.
        g = _path(3)
        mixed_pos = {0: [0, 0], 1: [1, 0], 2: [0, 1]}  # int coords, like a user might pass
        r = PolarsForceDirectedLayout(g, pos=mixed_pos, static_nodes={0}, iterations=5).results()
        self.assertEqual(set(r.keys()), set(g.nodes()))
        self.assertAlmostEqual(r[0][0], 0.0, places=10)
        self.assertAlmostEqual(r[0][1], 0.0, places=10)


class TestPFDLDegenerate(unittest.TestCase):

    def test_all_nodes_same_position_returns_early(self):
        g = _path(3)
        pos = {i: (0.5, 0.5) for i in g.nodes()}
        layout = PolarsForceDirectedLayout(g, pos=pos, iterations=20)
        r = layout.results()
        self.assertEqual(set(r.keys()), set(g.nodes()))

    def test_degenerate_result_still_has_all_nodes(self):
        g = _cycle(4)
        pos = {i: (0.0, 0.0) for i in g.nodes()}
        r = PolarsForceDirectedLayout(g, pos=pos).results()
        self.assertEqual(len(r), 4)


class TestPFDLParameters(unittest.TestCase):

    def test_k_equal_one(self):
        r = PolarsForceDirectedLayout(_cycle(5), k=1, iterations=10).results()
        self.assertEqual(len(r), 5)

    def test_k_equal_two(self):
        r = PolarsForceDirectedLayout(_cycle(5), k=2, iterations=10).results()
        self.assertEqual(len(r), 5)

    def test_custom_iterations(self):
        r = PolarsForceDirectedLayout(_path(4), iterations=3).results()
        self.assertEqual(len(r), 4)

    def test_custom_distances(self):
        g = _path(3)
        dists = {i: {j: abs(i - j) for j in g.nodes() if j != i} for i in g.nodes()}
        r = PolarsForceDirectedLayout(g, distances=dists, iterations=5).results()
        self.assertEqual(set(r.keys()), set(g.nodes()))

    def test_stress_threshold_none(self):
        r = PolarsForceDirectedLayout(_cycle(4), stress_threshold=None, iterations=5).results()
        self.assertEqual(len(r), 4)

    def test_initial_positions_respected(self):
        g = _path(3)
        pos = {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (2.0, 0.0)}
        layout = PolarsForceDirectedLayout(g, pos=dict(pos), iterations=5)
        r = layout.results()
        self.assertEqual(set(r.keys()), {0, 1, 2})


class TestPFDLStress(unittest.TestCase):

    def test_stress_returns_float(self):
        layout = PolarsForceDirectedLayout(_cycle(5), iterations=10)
        s = layout.stress()
        self.assertIsInstance(float(s), float)

    def test_stress_vector_returns_list(self):
        layout = PolarsForceDirectedLayout(_path(4), iterations=5)
        sv = layout.stressVector()
        self.assertIsInstance(sv, list)

    def test_stress_vector_nonempty(self):
        layout = PolarsForceDirectedLayout(_cycle(4), iterations=5)
        self.assertGreater(len(layout.stressVector()), 0)

    def test_stress_vector_all_floats(self):
        layout = PolarsForceDirectedLayout(_path(4), iterations=5)
        for v in layout.stressVector():
            self.assertIsInstance(float(v), float)


if __name__ == '__main__':
    unittest.main()
