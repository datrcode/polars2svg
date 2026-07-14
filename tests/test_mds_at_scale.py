import unittest
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix

from polars2svg.mds_at_scale import LandmarkMDSLayout, PivotMDSLayout, _tileSideBySide_


def _cycle(n=6):
    return nx.cycle_graph(n)


def _path(n=5):
    return nx.path_graph(n)


def _disconnected():
    g = nx.Graph()
    g.add_edges_from([(0, 1), (1, 2)])
    g.add_edges_from([(10, 11), (11, 12)])
    return g


class TestTileSideBySide(unittest.TestCase):

    def test_all_nodes_present(self):
        g = _disconnected()
        pos = {n: (float(n), 0.0) for n in g.nodes()}
        result = _tileSideBySide_(g, pos)
        self.assertEqual(set(result.keys()), set(g.nodes()))

    def test_result_values_are_pairs(self):
        g = _disconnected()
        pos = {n: (float(n % 3), 0.0) for n in g.nodes()}
        result = _tileSideBySide_(g, pos)
        for v in result.values():
            self.assertEqual(len(v), 2)

    def test_components_do_not_overlap(self):
        g = _disconnected()
        pos = {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (2.0, 0.0),
               10: (0.0, 0.0), 11: (1.0, 0.0), 12: (2.0, 0.0)}
        result = _tileSideBySide_(g, pos)
        comp_a = [0, 1, 2]
        comp_b = [10, 11, 12]
        max_x_a = max(result[n][0] for n in comp_a)
        min_x_b = min(result[n][0] for n in comp_b)
        self.assertLess(max_x_a, min_x_b)

    def test_y_coordinates_preserved(self):
        g = _disconnected()
        pos = {0: (0.0, 1.5), 1: (1.0, 2.5), 2: (2.0, 3.5),
               10: (0.0, -1.0), 11: (1.0, -2.0), 12: (2.0, -3.0)}
        result = _tileSideBySide_(g, pos)
        self.assertAlmostEqual(result[0][1], 1.5)
        self.assertAlmostEqual(result[1][1], 2.5)
        self.assertAlmostEqual(result[10][1], -1.0)

    def test_single_component_normalizes_x_to_zero(self):
        g = _path(3)
        pos = {0: (5.0, 0.0), 1: (6.0, 0.0), 2: (7.0, 0.0)}
        result = _tileSideBySide_(g, pos)
        min_x = min(result[n][0] for n in g.nodes())
        self.assertAlmostEqual(min_x, 0.0)


class TestLandmarkMDSLayout(unittest.TestCase):

    def test_all_nodes_in_results_default(self):
        g = _cycle(6)
        r = LandmarkMDSLayout(g).results()
        self.assertEqual(set(r.keys()), set(g.nodes()))

    def test_result_values_have_length_two(self):
        g = _cycle(6)
        r = LandmarkMDSLayout(g).results()
        for v in r.values():
            self.assertEqual(len(v), 2)

    def test_invalid_type_raises_value_error(self):
        with self.assertRaises(ValueError):
            LandmarkMDSLayout([[0, 1], [1, 0]])

    def test_num_landmarks_parameter(self):
        g = _cycle(8)
        r = LandmarkMDSLayout(g, num_landmarks=3).results()
        self.assertEqual(set(r.keys()), set(g.nodes()))

    def test_explicit_landmarks_list(self):
        g = _cycle(6)
        r = LandmarkMDSLayout(g, landmarks=[0, 1, 2]).results()
        self.assertEqual(set(r.keys()), set(g.nodes()))

    def test_landmark_pos_returns_landmark_nodes_only(self):
        g = _cycle(6)
        lm_pos = {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (0.5, 1.0)}
        r = LandmarkMDSLayout(g, landmark_pos=lm_pos).results()
        # Only landmark nodes are returned when landmark_pos is provided
        self.assertEqual(set(r.keys()), {0, 1, 2})

    def test_scipy_sparse_input(self):
        g = _cycle(5)
        adj = nx.to_scipy_sparse_array(g, format='csr')
        r = LandmarkMDSLayout(adj).results()
        self.assertEqual(len(r), 5)

    def test_numpy_array_input(self):
        g = _cycle(4)
        adj = np.array(nx.to_numpy_array(g))
        r = LandmarkMDSLayout(adj).results()
        self.assertEqual(len(r), 4)

    def test_result_values_are_numpy_arrays(self):
        g = _cycle(6)
        r = LandmarkMDSLayout(g).results()
        for v in r.values():
            self.assertIsInstance(v, np.ndarray)


class TestLandmarkMDSDisconnected(unittest.TestCase):

    def test_disconnected_without_rt_self_all_nodes_present(self):
        g = _disconnected()
        r = LandmarkMDSLayout(g).results()
        self.assertEqual(set(r.keys()), set(g.nodes()))

    def test_disconnected_result_values_have_length_two(self):
        g = _disconnected()
        r = LandmarkMDSLayout(g).results()
        for v in r.values():
            self.assertEqual(len(v), 2)


class TestPivotMDSLayout(unittest.TestCase):

    def test_all_nodes_in_results(self):
        g = _cycle(6)
        r = PivotMDSLayout(g).results()
        self.assertEqual(set(r.keys()), set(g.nodes()))

    def test_result_values_have_length_two(self):
        g = _cycle(6)
        r = PivotMDSLayout(g).results()
        for v in r.values():
            self.assertEqual(len(v), 2)

    def test_invalid_type_raises_value_error(self):
        with self.assertRaises(ValueError):
            PivotMDSLayout([[0, 1], [1, 0]])

    def test_num_pivots_parameter(self):
        g = _cycle(8)
        r = PivotMDSLayout(g, num_pivots=3).results()
        self.assertEqual(set(r.keys()), set(g.nodes()))

    def test_scipy_sparse_input(self):
        g = _cycle(5)
        adj = nx.to_scipy_sparse_array(g, format='csr')
        r = PivotMDSLayout(adj).results()
        self.assertEqual(len(r), 5)

    def test_numpy_array_input(self):
        g = _cycle(4)
        adj = np.array(nx.to_numpy_array(g))
        r = PivotMDSLayout(adj).results()
        self.assertEqual(len(r), 4)

    def test_result_values_are_numpy_arrays(self):
        g = _cycle(6)
        r = PivotMDSLayout(g).results()
        for v in r.values():
            self.assertIsInstance(v, np.ndarray)


class TestPivotMDSDisconnected(unittest.TestCase):

    def test_disconnected_without_rt_self_all_nodes_present(self):
        g = _disconnected()
        r = PivotMDSLayout(g).results()
        self.assertEqual(set(r.keys()), set(g.nodes()))

    def test_disconnected_result_values_have_length_two(self):
        g = _disconnected()
        r = PivotMDSLayout(g).results()
        for v in r.values():
            self.assertEqual(len(v), 2)


if __name__ == '__main__':
    unittest.main()
