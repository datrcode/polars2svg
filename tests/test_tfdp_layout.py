import math
import unittest
import networkx as nx

# TFDPLayout requires mlx, an optional extra: polars2svg[mlx] on Apple silicon
# (Metal), polars2svg[mlx-cuda] on Linux + NVIDIA. Guard the import so this module
# still collects where mlx is absent (e.g. the clean-room CI); the tests then skip
# rather than erroring at collection.
try:
    import mlx.core as mx
    from polars2svg.tfdp_layout import TFDPLayout, gpu_backend
    _HAS_MLX = True
except ImportError:
    mx = None
    TFDPLayout = gpu_backend = None
    _HAS_MLX = False

_requires_mlx = unittest.skipUnless(
    _HAS_MLX, 'mlx not installed (polars2svg[mlx] / polars2svg[mlx-cuda])')


def _cycle(n=6):
    return nx.cycle_graph(n)


def _path(n=4):
    return nx.path_graph(n)


def _disconnected():
    g = nx.Graph()
    g.add_edges_from([(0, 1), (1, 2)])   # component A
    g.add_edges_from([(3, 4)])            # component B
    return g


def _digraph(n=5):
    return nx.cycle_graph(n, create_using=nx.DiGraph)


@_requires_mlx
class TestTFDPLayoutContract(unittest.TestCase):

    def test_results_is_dict(self):
        r = TFDPLayout(_cycle(), max_iter=5, seed=0).results()
        self.assertIsInstance(r, dict)

    def test_results_contains_all_nodes(self):
        g = _cycle(6)
        r = TFDPLayout(g, max_iter=5, seed=0).results()
        self.assertEqual(set(r.keys()), set(g.nodes()))

    def test_results_values_are_float_pairs(self):
        r = TFDPLayout(_path(4), max_iter=5, seed=0).results()
        for v in r.values():
            self.assertEqual(len(v), 2)
            self.assertIsInstance(float(v[0]), float)
            self.assertIsInstance(float(v[1]), float)

    def test_two_node_graph(self):
        g = nx.path_graph(2)
        r = TFDPLayout(g, max_iter=5, seed=0).results()
        self.assertEqual(set(r.keys()), {0, 1})

    def test_single_node_graph(self):
        g = nx.Graph()
        g.add_node('a')
        r = TFDPLayout(g, max_iter=5, seed=0).results()
        self.assertIn('a', r)
        self.assertEqual(len(r['a']), 2)

    def test_elapsed_is_float(self):
        layout = TFDPLayout(_cycle(), max_iter=5, seed=0)
        self.assertIsInstance(layout.elapsed, float)
        self.assertGreaterEqual(layout.elapsed, 0.0)

    def test_directed_graph(self):
        g = _digraph(5)
        r = TFDPLayout(g, max_iter=5, seed=0).results()
        self.assertEqual(set(r.keys()), set(g.nodes()))


@_requires_mlx
class TestTFDPLayoutWarmStart(unittest.TestCase):

    def test_pos_kwarg_accepted(self):
        g = _path(4)
        initial_pos = {i: (float(i), 0.0) for i in g.nodes()}
        r = TFDPLayout(g, pos=initial_pos, max_iter=5, seed=0).results()
        self.assertEqual(set(r.keys()), set(g.nodes()))

    def test_pin_background_freezes_non_focal_nodes(self):
        g = _cycle(6)
        initial_pos = {i: (float(i), 0.0) for i in g.nodes()}
        focal = {0, 1}
        r = TFDPLayout(g, pos=initial_pos, selection=focal,
                       pin_background=True, max_iter=20, seed=0).results()
        # Non-focal nodes should remain at their initial positions
        for node in g.nodes():
            if node not in focal:
                self.assertAlmostEqual(r[node][0], initial_pos[node][0], places=5)
                self.assertAlmostEqual(r[node][1], initial_pos[node][1], places=5)


@_requires_mlx
class TestTFDPLayoutAlgorithms(unittest.TestCase):

    def test_algo_exact(self):
        g = _cycle(8)
        r = TFDPLayout(g, algo='exact', max_iter=5, seed=0).results()
        self.assertEqual(set(r.keys()), set(g.nodes()))

    def test_algo_rvs(self):
        g = _cycle(8)
        r = TFDPLayout(g, algo='rvs', rvs_k=4, max_iter=5, seed=0).results()
        self.assertEqual(set(r.keys()), set(g.nodes()))

    def test_algo_invalid_raises(self):
        with self.assertRaises(ValueError):
            TFDPLayout(_cycle(), algo='bogus', max_iter=2, seed=0)

    def test_combine_false(self):
        g = _path(4)
        r = TFDPLayout(g, combine=False, max_iter=5, seed=0).results()
        self.assertEqual(set(r.keys()), set(g.nodes()))


@_requires_mlx
class TestTFDPLayoutDisconnected(unittest.TestCase):

    def test_disconnected_graph_all_nodes_present(self):
        g = _disconnected()
        r = TFDPLayout(g, max_iter=5, seed=0).results()
        self.assertEqual(set(r.keys()), set(g.nodes()))

    def test_disconnected_graph_values_are_float_pairs(self):
        g = _disconnected()
        r = TFDPLayout(g, max_iter=5, seed=0).results()
        for v in r.values():
            self.assertEqual(len(v), 2)
            self.assertIsInstance(float(v[0]), float)
            self.assertIsInstance(float(v[1]), float)


def _mean_edge_length(g, pos):
    lengths = [math.dist(pos[u], pos[v]) for u, v in g.edges()]
    return sum(lengths) / len(lengths)


@_requires_mlx
class TestTFDPLayoutCorrectness(unittest.TestCase):
    """The layout must be right, not merely non-crashing.

    Every other test here asserts the *contract* (keys present, values are float
    pairs) — a run that returned all-NaN would satisfy every one of them. These
    assert the numbers mean something, which is what actually has to hold when the
    kernels run on a backend (CUDA) other than the one they were written against.
    """

    def test_positions_are_finite(self):
        r = TFDPLayout(nx.karate_club_graph(), max_iter=50, seed=0).results()
        for x, y in r.values():
            self.assertTrue(math.isfinite(x) and math.isfinite(y))

    def test_seed_is_reproducible(self):
        # GPU tolerance: Metal/CUDA reduction order is not deterministic run-to-run,
        # so same-seed layouts drift up to ~1e-05 over 25 iterations (measured on
        # Metal). places=4 gives headroom; the CPU device is bit-exact.
        g  = nx.karate_club_graph()
        r1 = TFDPLayout(g, max_iter=25, seed=7).results()
        r2 = TFDPLayout(g, max_iter=25, seed=7).results()
        for node in r1:
            self.assertAlmostEqual(r1[node][0], r2[node][0], places=4)
            self.assertAlmostEqual(r1[node][1], r2[node][1], places=4)

    def test_layout_converges(self):
        # The integrator should contract the graph toward its equilibrium, not
        # diverge. Mean edge length after many steps must beat the near-init state.
        g     = nx.karate_club_graph()
        early = _mean_edge_length(g, TFDPLayout(g, max_iter=1,   seed=0).results())
        late  = _mean_edge_length(g, TFDPLayout(g, max_iter=300, seed=0).results())
        self.assertLess(late, early)

    def test_gpu_matches_cpu_reference(self):
        # The strongest available signal that the GPU kernels (Metal or CUDA)
        # compute the same thing as the reference path. Skips when there is no GPU,
        # since then both runs would trivially be the same device.
        if gpu_backend() == 'cpu':
            self.skipTest('no GPU backend — nothing to cross-check against')
        g   = nx.karate_club_graph()
        gpu = TFDPLayout(g, max_iter=50, seed=0).results()
        cpu = TFDPLayout(g, max_iter=50, seed=0, device=mx.cpu).results()
        for node in gpu:
            self.assertAlmostEqual(gpu[node][0], cpu[node][0], places=3)
            self.assertAlmostEqual(gpu[node][1], cpu[node][1], places=3)


if __name__ == '__main__':
    unittest.main()
