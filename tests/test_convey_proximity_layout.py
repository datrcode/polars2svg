import unittest
import networkx as nx

from polars2svg.convey_proximity_layout import ConveyProximityLayout


def _path(n=4):
    return nx.path_graph(n)


def _cycle(n=5):
    return nx.cycle_graph(n)


class TestCPLResultsContract(unittest.TestCase):

    def test_results_is_dict(self):
        r = ConveyProximityLayout(_path(4)).results()
        self.assertIsInstance(r, dict)

    def test_results_contains_all_nodes(self):
        g = _path(5)
        r = ConveyProximityLayout(g).results()
        self.assertEqual(set(r.keys()), set(g.nodes()))

    def test_results_values_are_float_pairs(self):
        r = ConveyProximityLayout(_path(4)).results()
        for v in r.values():
            self.assertEqual(len(v), 2)
            self.assertIsInstance(float(v[0]), float)
            self.assertIsInstance(float(v[1]), float)

    def test_two_node_graph(self):
        g = nx.path_graph(2)
        r = ConveyProximityLayout(g).results()
        self.assertEqual(set(r.keys()), {0, 1})

    def test_cycle_graph(self):
        g = _cycle(6)
        r = ConveyProximityLayout(g).results()
        self.assertEqual(len(r), 6)


class TestCPLDistanceModes(unittest.TestCase):

    def test_dijkstra_distances(self):
        g = _path(4)
        r = ConveyProximityLayout(g, use_resistive_distances=False).results()
        self.assertEqual(set(r.keys()), set(g.nodes()))

    def test_resistive_distances_default(self):
        g = _path(4)
        r = ConveyProximityLayout(g).results()
        self.assertEqual(len(r), 4)

    def test_custom_distances(self):
        g = _path(3)
        dists = {i: {j: abs(i - j) for j in g.nodes() if j != i} for i in g.nodes()}
        r = ConveyProximityLayout(g, distances=dists).results()
        self.assertEqual(set(r.keys()), set(g.nodes()))


class TestCPLStressOutput(unittest.TestCase):

    def test_stress_returns_positive_float(self):
        layout = ConveyProximityLayout(_path(4))
        s = layout.stress()
        self.assertIsInstance(float(s), float)
        self.assertGreater(float(s), 0.0)

    def test_stress_df_has_required_columns(self):
        layout = ConveyProximityLayout(_path(4))
        cols = set(layout.stress_df.columns)
        for col in ('stress', 'trial', 'round', 'best_flag'):
            self.assertIn(col, cols)

    def test_stress_df_has_i_global_column(self):
        layout = ConveyProximityLayout(_path(4))
        self.assertIn('i_global', layout.stress_df.columns)

    def test_stress_df_nonempty(self):
        layout = ConveyProximityLayout(_path(4))
        self.assertGreater(len(layout.stress_df), 0)

    def test_best_flag_column_is_bool(self):
        import polars as pl
        layout = ConveyProximityLayout(_path(4))
        self.assertEqual(layout.stress_df['best_flag'].dtype, pl.Boolean)

    def test_best_flag_has_at_least_one_true(self):
        layout = ConveyProximityLayout(_path(4))
        self.assertTrue(layout.stress_df['best_flag'].any())


class TestCPLVerticesAdded(unittest.TestCase):

    def test_vertices_added_is_list_of_sets(self):
        layout = ConveyProximityLayout(_path(4))
        self.assertIsInstance(layout.vertices_added, list)
        for s in layout.vertices_added:
            self.assertIsInstance(s, set)

    def test_all_nodes_covered_in_vertices_added(self):
        g = _path(5)
        layout = ConveyProximityLayout(g)
        covered = set()
        for s in layout.vertices_added:
            covered |= s
        self.assertEqual(covered, set(g.nodes()))

    def test_small_graph_single_round(self):
        g = _path(4)
        layout = ConveyProximityLayout(g)
        self.assertGreaterEqual(len(layout.vertices_added), 1)


if __name__ == '__main__':
    unittest.main()
