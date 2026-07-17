import math
import unittest

import polars2svg.od_flow_layout as odmod
from polars2svg.od_flow_layout import ODFlowLayout

_HAS_MLX_GPU = (odmod.mx is not None) and (odmod._default_device() == odmod.mx.gpu)


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


class TestODFlowLayoutBackends(unittest.TestCase):

    def _run_forced(self, use_mlx, flows, **kw):
        # Force the backend by toggling the module-level mlx handle
        _saved_ = odmod.mx
        if not use_mlx:
            odmod.mx = None
        try:
            return ODFlowLayout(flows, **kw).results()
        finally:
            odmod.mx = _saved_

    def test_numpy_fallback_when_mlx_absent(self):
        # With mlx forced absent the module still lays out correctly on NumPy
        flows = _hub_flows()
        cps = self._run_forced(False, flows, iterations=20)
        self.assertEqual(len(cps), len(flows))
        for cx, cy in cps:
            self.assertTrue(math.isfinite(cx) and math.isfinite(cy))

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
