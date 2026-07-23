import math
import unittest

import networkx as nx
import numpy as np

from polars2svg.layout_protocol import LayoutAlgorithm
from polars2svg.ncp_layout import (NCPLayout, NeighborhoodPreservingPacking,
                                    _node_size_weights)


def _weighted_graph(n=24, m=60, seed=0):
    rng = nx.utils.create_random_state(seed)
    g = nx.gnm_random_graph(n, m, seed=seed)
    for u, v in g.edges():
        g[u][v]['weight'] = float(rng.randint(1, 50))
    return g


def _spread_pos(g, seed=0, span=1000.0):
    rng = np.random.default_rng(seed)
    return {n: (float(rng.uniform(0, span)), float(rng.uniform(0, span))) for n in g.nodes()}


def _hull_area(pos_dict):
    from scipy.spatial import ConvexHull
    return float(ConvexHull(np.array(list(pos_dict.values()))).volume)


class TestNCPResultsContract(unittest.TestCase):
    def setUp(self):
        self.g = _weighted_graph()
        self.pos = _spread_pos(self.g)

    def _fast(self, **kw):
        return NCPLayout(self.g, pos=self.pos, force_iterations=150, **kw)

    def test_results_is_dict(self):
        self.assertIsInstance(self._fast().results(), dict)

    def test_satisfies_layout_protocol(self):
        self.assertIsInstance(self._fast(), LayoutAlgorithm)

    def test_packs_all_nodes_without_selection(self):
        r = self._fast().results()
        self.assertEqual(set(r.keys()), set(self.g.nodes()))

    def test_values_are_float_pairs(self):
        for v in self._fast().results().values():
            self.assertEqual(len(v), 2)
            self.assertIsInstance(float(v[0]), float)
            self.assertIsInstance(float(v[1]), float)

    def test_no_output_overlap(self):
        # A valid packing satisfies the non-overlap constraint.
        self.assertLess(self._fast().packing.overlaps(), 1e-6)

    def test_compacts_a_spread_out_layout(self):
        r = self._fast().results()
        self.assertLess(_hull_area(r), _hull_area(self.pos))

    def test_deterministic(self):
        # No RNG in the pipeline: identical inputs -> identical positions.
        a = self._fast().results()
        b = self._fast().results()
        for n in a:
            self.assertEqual(a[n], b[n])


class TestNCPSelection(unittest.TestCase):
    def setUp(self):
        self.g = _weighted_graph()
        self.pos = _spread_pos(self.g)

    def test_selection_packs_only_selected(self):
        sel = set(list(self.g.nodes())[:8])
        r = NCPLayout(self.g, pos=self.pos, selection=sel, force_iterations=150).results()
        self.assertEqual(set(r.keys()), sel)

    def test_empty_selection_packs_all(self):
        r = NCPLayout(self.g, pos=self.pos, selection=set(), force_iterations=150).results()
        self.assertEqual(set(r.keys()), set(self.g.nodes()))

    def test_unpositioned_nodes_are_skipped(self):
        pos = dict(self.pos)
        dropped = list(self.g.nodes())[0]
        del pos[dropped]
        r = NCPLayout(self.g, pos=pos, force_iterations=150).results()
        self.assertNotIn(dropped, r)


class TestNCPNodeSizing(unittest.TestCase):
    """Requirement 5: radius weight = log(count); count is the node's flow
    volume (weighted degree), or the neighbour count when unweighted."""

    def test_weighted_degree_drives_the_weight(self):
        g = nx.Graph()
        g.add_edge('a', 'b', weight=10)
        g.add_edge('a', 'c', weight=20)   # a's count = 30
        g.add_edge('b', 'c', weight=5)    # b: 15, c: 25
        w = _node_size_weights(g, ['a', 'b', 'c'])
        self.assertAlmostEqual(w[0], math.log(30))
        self.assertAlmostEqual(w[1], math.log(15))
        self.assertAlmostEqual(w[2], math.log(25))

    def test_falls_back_to_neighbor_count_when_unweighted(self):
        g = nx.star_graph(4)              # centre degree 4, leaves degree 1
        w = _node_size_weights(g, list(g.nodes()))
        self.assertAlmostEqual(w[0], math.log(4))
        self.assertTrue(np.all(w[1:] == 0.0))   # log(1) == 0 for the leaves

    def test_bigger_count_yields_a_bigger_circle(self):
        g = nx.star_graph(6)
        for u, v in g.edges():
            g[u][v]['weight'] = 1.0
        pos = {n: (float(i), float((i * 7) % 5)) for i, n in enumerate(g.nodes())}
        pack = NCPLayout(g, pos=pos, force_iterations=150).packing
        # the hub (index 0) carries all the flow and must be the largest circle
        self.assertEqual(int(np.argmax(pack.radii)), 0)


class TestNCPDegenerateInputs(unittest.TestCase):
    def test_empty_graph(self):
        self.assertEqual(NCPLayout(nx.Graph(), pos={}).results(), {})

    def test_two_nodes_are_left_in_place(self):
        g = nx.path_graph(2)
        pos = {0: (1.0, 2.0), 1: (3.0, 4.0)}
        r = NCPLayout(g, pos=pos).results()
        self.assertEqual(r, {0: (1.0, 2.0), 1: (3.0, 4.0)})


class TestNCPFitModes(unittest.TestCase):
    def setUp(self):
        self.g = _weighted_graph()
        self.pos = _spread_pos(self.g)

    def test_preserve_radii_shrinks_the_extent(self):
        pack = NCPLayout(self.g, pos=self.pos, fit_mode='preserve_radii',
                         force_iterations=150).packing
        span_in = float(np.ptp(pack.positions_in, axis=0).max())
        span_out = float(np.ptp(pack.positions, axis=0).max())
        self.assertLess(span_out, span_in)

    def test_fill_keeps_radii_relations(self):
        pack = NCPLayout(self.g, pos=self.pos, fit_mode='fill',
                         force_iterations=150).packing
        self.assertLess(pack.overlaps(), 1e-6)


class TestNeighborhoodPreservingPackingCore(unittest.TestCase):
    def test_rejects_bad_radii(self):
        pos = np.random.default_rng(0).uniform(0, 10, (5, 2))
        with self.assertRaises(ValueError):
            NeighborhoodPreservingPacking(pos, radii=np.array([1, 2, 3]))
        with self.assertRaises(ValueError):
            NeighborhoodPreservingPacking(pos, radii=np.array([1, 2, 3, 4, -1]))

    def test_scale_invariance(self):
        # Same packing (up to similarity) whether the input is small or large.
        rng = np.random.default_rng(0)
        pos = rng.uniform(0, 100, (40, 2))
        radii = rng.uniform(2, 9, 40)
        a = NeighborhoodPreservingPacking(pos, radii=radii, force_iterations=200)
        b = NeighborhoodPreservingPacking(pos * 3 + 500, radii=radii * 3, force_iterations=200)
        ca = a.positions - a.positions.mean(axis=0)
        cb = (b.positions - b.positions.mean(axis=0)) / 3.0
        self.assertLess(float(np.abs(ca - cb).max()) / a.radii.mean(), 1e-6)


if __name__ == '__main__':
    unittest.main()
