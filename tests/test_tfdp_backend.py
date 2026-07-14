import unittest

# These tests exercise the individual mlx.core ops that tfdp_layout.py depends on,
# directly on whatever device MLX resolves to (Metal / CUDA / CPU). The point is
# blame localization: MLX's CUDA backend is newer than its Metal one, so if a single
# op (scatter-add, keyed RNG) has a backend gap, it should fail *here* by name rather
# than surface as a mystery NaN 300 steps into a layout loop.
try:
    import mlx.core as mx
    from polars2svg.tfdp_layout import _default_device, gpu_backend
    _HAS_MLX = True
except ImportError:
    mx = None
    _default_device = gpu_backend = None
    _HAS_MLX = False

_requires_mlx = unittest.skipUnless(
    _HAS_MLX, 'mlx not installed (polars2svg[mlx] / polars2svg[mlx-cuda])')


@_requires_mlx
class TestBackendResolution(unittest.TestCase):
    def test_gpu_backend_is_known(self):
        self.assertIn(gpu_backend(), ('metal', 'cuda', 'cpu'))

    def test_device_is_cached(self):
        self.assertIs(_default_device(), _default_device())


@_requires_mlx
class TestTFDPOpsOnDevice(unittest.TestCase):
    """Every mlx.core op tfdp_layout.py uses, run on the resolved device."""

    def setUp(self):
        self.device = _default_device()

    def test_elementwise_and_reductions(self):
        # The t-kernel: d2 = sum(diff*diff, -1); q = 1/(1+d2)
        with mx.stream(self.device):
            diff = mx.array([[3.0, 4.0], [0.0, 0.0]])
            d2   = mx.sum(diff * diff, axis=-1)
            q    = 1.0 / (1.0 + d2)
            mx.eval(d2, q)
        self.assertAlmostEqual(float(d2[0]), 25.0, places=5)
        self.assertAlmostEqual(float(q[1]),   1.0, places=5)

    def test_sqrt_zeros_eye_where(self):
        with mx.stream(self.device):
            s   = mx.sqrt(mx.array([9.0, 1e-12]))
            z   = mx.zeros((3, 2))
            eye = mx.eye(3)
            w   = mx.where(mx.array([True, False]), 1.0, 5.0)
            mx.eval(s, z, eye, w)
        self.assertAlmostEqual(float(s[0]), 3.0, places=5)
        self.assertEqual(z.shape, (3, 2))
        self.assertAlmostEqual(float(eye[1, 1]), 1.0, places=5)
        self.assertAlmostEqual(float(eye[0, 1]), 0.0, places=5)
        self.assertAlmostEqual(float(w[0]), 1.0, places=5)
        self.assertAlmostEqual(float(w[1]), 5.0, places=5)

    def test_broadcast_all_pairs_diff(self):
        # _repulsive_exact materializes an (n, n, 2) tensor this way.
        with mx.stream(self.device):
            pos  = mx.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
            diff = pos[:, None, :] - pos[None, :, :]
            mx.eval(diff)
        self.assertEqual(diff.shape, (3, 3, 2))
        self.assertAlmostEqual(float(diff[0, 1, 0]), -1.0, places=5)
        self.assertAlmostEqual(float(diff[2, 0, 1]),  2.0, places=5)

    def test_fancy_index_gather(self):
        # pos[edge_src] and pos[idx] in _attractive_exact / _repulsive_rvs.
        with mx.stream(self.device):
            pos = mx.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
            idx = mx.array([2, 0, 2])
            g   = pos[idx]
            mx.eval(g)
        self.assertEqual(g.shape, (3, 2))
        self.assertAlmostEqual(float(g[0, 0]), 2.0, places=5)
        self.assertAlmostEqual(float(g[1, 0]), 0.0, places=5)

    def test_scatter_add_accumulates_duplicate_indices(self):
        # _attractive_exact:112-113. Using .at[].add() rather than []= is the whole
        # point: a node touched by several edges must accumulate every edge's force.
        # A backend that silently implemented this as an overwrite would produce a
        # plausible-looking but wrong layout, so assert the accumulation directly.
        with mx.stream(self.device):
            acc = mx.zeros((3, 2))
            idx = mx.array([0, 0, 2])
            val = mx.array([[1.0, 1.0], [2.0, 2.0], [5.0, 5.0]])
            out = acc.at[idx].add(val)
            mx.eval(out)
        self.assertAlmostEqual(float(out[0, 0]), 3.0, places=5)  # 1 + 2, not 2
        self.assertAlmostEqual(float(out[1, 0]), 0.0, places=5)  # untouched
        self.assertAlmostEqual(float(out[2, 0]), 5.0, places=5)

    def test_keyed_rng(self):
        # _repulsive_rvs:87 — mx.random.randint(key=...) on the algo='rvs' path.
        with mx.stream(self.device):
            key = mx.random.key(0)
            k1, k2 = mx.random.split(key)
            a = mx.random.randint(0, 10, shape=(4, 3), key=k1)
            b = mx.random.randint(0, 10, shape=(4, 3), key=k1)
            c = mx.random.randint(0, 10, shape=(4, 3), key=k2)
            mx.eval(a, b, c)
        self.assertEqual(a.shape, (4, 3))
        self.assertTrue(bool(mx.all(a >= 0)) and bool(mx.all(a < 10)))
        # Same key -> same draw (the layout's seed= reproducibility rests on this).
        self.assertTrue(bool(mx.all(a == b)))
        # Split keys -> independent streams.
        self.assertFalse(bool(mx.all(a == c)))


if __name__ == '__main__':
    unittest.main()
