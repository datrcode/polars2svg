import math
import unittest
from unittest import mock

import numpy as np

import polars2svg.od_flow_layout as odmod
from polars2svg.od_flow_layout import ODFlowLayout

_HAS_MLX     = odmod.mx is not None
_HAS_MLX_GPU = _HAS_MLX and (odmod._default_device() == odmod.mx.gpu)


def _mid(f):
    return ((f[0] + f[2]) / 2.0, (f[1] + f[3]) / 2.0)


def _hub_flows():
    # Hub-and-spoke with a crossing vertical, screen-coordinate scale
    return [
        (50, 200, 350, 200),
        (50, 200, 200, 60),
        (50, 200, 200, 340),
        (350, 200, 200, 60),
        (350, 200, 200, 340),
        (200, 60, 200, 340),
    ]


class TestODFlowLayoutContract(unittest.TestCase):

    def test_results_one_cp_per_flow(self):
        flows = _hub_flows()
        cps = ODFlowLayout(flows, iterations=10).results()
        self.assertEqual(len(cps), len(flows))
        for cx, cy in cps:
            self.assertTrue(math.isfinite(cx) and math.isfinite(cy))

    def test_deterministic(self):
        flows = _hub_flows()
        a = ODFlowLayout(flows).results()
        b = ODFlowLayout(flows).results()
        self.assertEqual(a, b)

    def test_single_flow_stays_straight(self):
        # No other flows or unconnected nodes -> nothing curves the flow
        flows = [(10, 10, 100, 100)]
        cps = ODFlowLayout(flows, iterations=10).results()
        self.assertEqual(cps[0], _mid(flows[0]))

    def test_zero_length_flow_is_inert(self):
        flows = [(50, 50, 50, 50), (10, 10, 100, 10)]
        cps = ODFlowLayout(flows, iterations=10).results()
        self.assertEqual(cps[0], (50.0, 50.0))
        for cx, cy in cps:
            self.assertTrue(math.isfinite(cx) and math.isfinite(cy))

    def test_empty_flows(self):
        self.assertEqual(ODFlowLayout([]).results(), [])


class TestODFlowLayoutBehavior(unittest.TestCase):

    def test_flows_curve_apart(self):
        # Two flows sharing both endpoints' neighborhood: repulsion must bow at
        # least one of them away from its straight baseline
        flows = [(0, 100, 300, 100), (0, 110, 300, 110)]
        cps = ODFlowLayout(flows, iterations=50).results()
        _bow_ = max(math.hypot(cp[0] - _mid(f)[0], cp[1] - _mid(f)[1])
                    for f, cp in zip(flows, cps))
        self.assertGreater(_bow_, 1.0)

    def test_control_point_inside_constraint_rectangle(self):
        # Section 3.2.1: cp constrained to the flow-aligned rectangle
        flows = _hub_flows()
        layout = ODFlowLayout(flows)
        for f, cp in zip(flows, layout.results()):
            b = math.hypot(f[2] - f[0], f[3] - f[1])
            if b < 1e-9: continue
            ex, ey = (f[2] - f[0]) / b, (f[3] - f[1]) / b
            rx, ry = cp[0] - f[0], cp[1] - f[1]
            lx = rx * ex + ry * ey
            ly = ry * ex - rx * ey
            self.assertGreaterEqual(lx, -1e-6)
            self.assertLessEqual(lx, b + 1e-6)
            self.assertLessEqual(abs(ly), layout.rect_pct * b / 2.0 + 1e-6)

    def test_control_point_inside_canvas(self):
        canvas = (0.0, 0.0, 400.0, 400.0)
        for cx, cy in ODFlowLayout(_hub_flows(), canvas=canvas).results():
            self.assertTrue(canvas[0] <= cx <= canvas[2])
            self.assertTrue(canvas[1] <= cy <= canvas[3])

    def test_moved_off_node_flow_clears_obstacle(self):
        # The long horizontal flow passes straight through an unconnected node
        # at (200, 200); the layout should bend or move flows so pinned flows
        # keep the minimum obstacle clearance
        flows = [(50, 200, 350, 200), (200, 200, 200, 60), (200, 200, 200, 340)]
        layout = ODFlowLayout(flows)
        for f in layout._pinned_:
            self.assertTrue(layout._clearOfObstacles_(f, layout.cps[f]))

    def test_arrow_obstacles_included_when_enabled(self):
        flows = _hub_flows()
        layout = ODFlowLayout(flows, arrows=True, arrow_radius=8.0)
        # flows with unshared destinations see other flows' arrowheads as obstacles
        self.assertGreater(max(len(layout._arrowObstacles_(f)) for f in layout.active), 0)
        for f in layout._pinned_:
            self.assertTrue(layout._clearOfObstacles_(f, layout.cps[f]))

    def test_arrow_obstacles_deterministic(self):
        flows = _hub_flows()
        a = ODFlowLayout(flows, arrows=True, arrow_radius=8.0).results()
        b = ODFlowLayout(flows, arrows=True, arrow_radius=8.0).results()
        self.assertEqual(a, b)

    def test_arrows_disabled_yields_no_arrow_obstacles(self):
        layout = ODFlowLayout(_hub_flows(), iterations=5)
        for f in layout.active:
            self.assertEqual(layout._arrowObstacles_(f), [])


#
# The endpoint-blocked shortcut (PLANNING.md G3): a flow with an unconnected node within
# clearance of one of its own endpoints fails _moveOffObstacles_() without the scan.  It
# must change nothing -- not a control point, not which flows are pinned.
#
def _scattered_flows():
    _rng_ = np.random.default_rng(7)
    return [tuple(map(float, r)) for r in _rng_.uniform((0, 0, 0, 0), (300, 200, 300, 200), size=(30, 4))]


def _no_shortcut(self):
    return np.zeros(len(self.flows), dtype=bool)


class TestEndpointBlockedShortcut(unittest.TestCase):

    def _layout(self, **kwargs):
        return ODFlowLayout(_scattered_flows(), canvas=(0.0, 0.0, 300.0, 200.0), iterations=40,
                            backend='numpy', **kwargs)

    def test_the_shortcut_changes_nothing(self):
        for kwargs in ({}, dict(arrows=True, arrow_radius=3.0)):
            with self.subTest(**kwargs):
                fast = self._layout(**kwargs)
                with mock.patch.object(ODFlowLayout, '_endpointBlockedFlows_', _no_shortcut):
                    full = self._layout(**kwargs)
                self.assertGreater(int(fast._endpoint_blocked_.sum()), 0, 'no flow took the shortcut')
                self.assertGreater(len(fast._pinned_), 0, 'no flow was moved off an obstacle')
                self.assertEqual(fast.results(), full.results())
                self.assertEqual(fast._pinned_, full._pinned_)

    def test_the_full_scan_cannot_move_a_blocked_flow(self):
        layout = self._layout()
        blocked = np.nonzero(layout._endpoint_blocked_)[0]
        self.assertGreater(len(blocked), 0)
        layout._endpoint_blocked_[:] = False
        for f in blocked:
            cp = layout.cps[f]
            self.assertFalse(layout._moveOffObstacles_(int(f)))
            self.assertEqual(layout.cps[f], cp)
            self.assertNotIn(int(f), layout._pinned_)

    def test_only_an_unconnected_node_inside_the_clearance_blocks(self):
        # clearance is node_radius + min_obstacle_dist = 5 + 4 = 9 px
        def blocked(flows):
            return ODFlowLayout(flows, iterations=0)._endpoint_blocked_.tolist()
        # a node 5 px from (0, 0) blocks both flows: each has the other's endpoint that near
        self.assertEqual(blocked([(0, 0, 100, 0), (5, 0, 5, 100)]), [True, True])
        # a shared node is the flow's own endpoint, not an obstacle
        self.assertEqual(blocked([(0, 0, 100, 0), (0, 0, 0, 100)]), [False, False])
        # exactly 9 px away is clear: the scan rejects only distances below the clearance
        self.assertEqual(blocked([(0, 0, 100, 0), (9, 0, 9, 100)]), [False, False])
        # a node near the far endpoint counts too
        self.assertEqual(blocked([(0, 0, 100, 0), (100, 6, 100, 100)]), [True, True])


#
# The move-off clearance test checks one curve sample at a time and drops rejected
# candidates as it goes (PLANNING.md G3 step 2).  It must agree with checking all 25
# samples of every candidate against every obstacle.
#
class TestClearCandidatesPruning(unittest.TestCase):

    def _brute(self, layout, f, cands, tests):
        out = []
        for cp in cands:
            pts = layout._sampleArr_(f, cp=(float(cp[0]), float(cp[1])))     # (25, 2)
            ok = True
            for obs, mn in tests:
                dd = np.hypot(pts[:, 0][:, None] - obs[None, :, 0], pts[:, 1][:, None] - obs[None, :, 1])
                ok &= bool(dd.min() >= mn)
            out.append(ok)
        return out

    def test_pruning_agrees_with_checking_every_sample(self):
        layout = ODFlowLayout(_scattered_flows(), canvas=(0.0, 0.0, 300.0, 200.0), iterations=1,
                              backend='numpy', arrows=True, arrow_radius=3.0)
        rng = np.random.default_rng(3)
        cands = rng.uniform((0, 0), (300, 200), size=(400, 2))
        seen = set()
        for f in layout.active:
            nodes  = np.asarray(layout._obstacles_(f), dtype=np.float64)
            arrows = np.asarray(layout._arrowObstacles_(f), dtype=np.float64).reshape(-1, 2)
            tests  = [(o, m) for o, m in ((nodes,  layout.node_radius  + layout.min_obstacle_dist),
                                          (arrows, layout.arrow_radius + layout.min_obstacle_dist)) if len(o) > 0]
            got = layout._clearCandidates_(f, cands, tests).tolist()
            self.assertEqual(got, self._brute(layout, f, cands, tests), f'flow {f}')
            seen.update(got)
        self.assertEqual(seen, {True, False}, 'the candidates should include clear and rejected ones')

    def test_exactly_the_clearance_away_is_clear(self):
        # A straight flow (0,0)->(240,0) with its control point at the midpoint: sample 12
        # (t = 0.5) is exactly (120, 0), so an obstacle at (120, 9) is exactly 9 px away
        layout = ODFlowLayout([(0, 0, 240, 0)], iterations=1, backend='numpy')
        cand   = np.array([[120.0, 0.0]])
        at     = [(np.array([[120.0, 9.0]]),  9.0)]
        inside = [(np.array([[120.0, 8.99]]), 9.0)]
        self.assertTrue(layout._clearCandidates_(0, cand, at)[0])
        self.assertFalse(layout._clearCandidates_(0, cand, inside)[0])

    def test_no_obstacles_clears_every_candidate(self):
        layout = ODFlowLayout(_hub_flows(), iterations=1, backend='numpy')
        self.assertTrue(layout._clearCandidates_(0, np.array([[1.0, 2.0], [3.0, 4.0]]), []).all())


class TestODFlowLayoutBackends(unittest.TestCase):

    def _run_forced(self, use_mlx, flows, **kw):
        # backend= is the supported way to pin this; it used to need a monkeypatch of the
        # module-level mlx handle, which meant the thing under test was the patch rather
        # than the contract callers actually have.
        return ODFlowLayout(flows, backend='mlx' if use_mlx else 'numpy', **kw).results()

    def test_numpy_fallback_when_mlx_absent(self):
        # With mlx forced absent the module still lays out correctly on NumPy
        flows = _hub_flows()
        cps = self._run_forced(False, flows, iterations=20)
        self.assertEqual(len(cps), len(flows))
        for cx, cy in cps:
            self.assertTrue(math.isfinite(cx) and math.isfinite(cy))

    def test_backend_numpy_is_available_everywhere(self):
        # The point of pinning: this runs identically whether or not mlx is installed, so
        # a stored expectation stays a property of the code rather than of the machine.
        _a_ = ODFlowLayout(_hub_flows(), iterations=15, backend='numpy').results()
        _b_ = ODFlowLayout(_hub_flows(), iterations=15, backend='numpy').results()
        self.assertEqual(_a_, _b_)

    def test_backend_rejects_an_unknown_name(self):
        with self.assertRaises(ValueError):
            ODFlowLayout(_hub_flows(), iterations=3, backend='cuda')

    @unittest.skipUnless(not _HAS_MLX, 'only meaningful when mlx is absent')
    def test_backend_mlx_raises_when_mlx_is_absent(self):
        # A clear error beats silently falling back: a caller who asked for the GPU path
        # and got the CPU one would draw the wrong conclusion from the timings.
        with self.assertRaises(ValueError):
            ODFlowLayout(_hub_flows(), iterations=3, backend='mlx')

    @unittest.skipUnless(_HAS_MLX_GPU, 'requires mlx with a usable GPU')
    def test_backend_auto_takes_mlx_when_a_gpu_is_present(self):
        self.assertTrue(ODFlowLayout(_hub_flows(), iterations=3)._use_mlx_)

    @unittest.skipUnless(_HAS_MLX_GPU, 'requires mlx with a usable GPU')
    def test_mlx_numpy_parity(self):
        # On a stable fixture the float32 GPU path tracks the float64 NumPy path
        # to well under a pixel (chaotic inputs can diverge at discrete
        # intersection/pinning branches; the fixture avoids those bifurcations).
        flows = _hub_flows()
        np_cps  = self._run_forced(False, flows)
        mlx_cps = self._run_forced(True,  flows)
        for (nx, ny), (mx_, my) in zip(np_cps, mlx_cps):
            self.assertLess(math.hypot(nx - mx_, ny - my), 1e-2)
        # both paths must satisfy the flow-aligned constraint rectangle
        layout = ODFlowLayout(flows)
        for cps in (np_cps, mlx_cps):
            for f, cp in zip(flows, cps):
                b = math.hypot(f[2] - f[0], f[3] - f[1])
                if b < 1e-9: continue
                ex, ey = (f[2] - f[0]) / b, (f[3] - f[1]) / b
                rx, ry = cp[0] - f[0], cp[1] - f[1]
                lx, ly = rx * ex + ry * ey, ry * ex - rx * ey
                self.assertGreaterEqual(lx, -1e-6)
                self.assertLessEqual(lx, b + 1e-6)
                self.assertLessEqual(abs(ly), layout.rect_pct * b / 2.0 + 1e-6)


if __name__ == '__main__':
    unittest.main()
