import unittest
import polars as pl
from polars2svg import Polars2SVG


def _make_xyp(p2s, wxh=(256, 256)):
    df = pl.DataFrame({
        'x': [0.0,  25.0,  50.0,  75.0, 100.0],
        'y': [0.0,  25.0,  50.0,  75.0, 100.0],
    })
    return p2s.xyp(df, 'x', 'y', wxh=wxh)


class TestXYpCoordinateTransforms(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def setUp(self):
        self.xyp = _make_xyp(self.p2s)

    def test_wx_to_sx_round_trip(self):
        for wx in [0.0, 25.0, 50.0, 75.0, 100.0]:
            sx = self.xyp.wxToSx(wx)
            recovered = self.xyp.sxToWx(sx)
            self.assertAlmostEqual(recovered, wx, places=6,
                                   msg=f'Round-trip failed for wx={wx}')

    def test_wy_to_sy_round_trip(self):
        for wy in [0.0, 25.0, 50.0, 75.0, 100.0]:
            sy = self.xyp.wyToSy(wy)
            recovered = self.xyp.syToWy(sy)
            self.assertAlmostEqual(recovered, wy, places=6,
                                   msg=f'Round-trip failed for wy={wy}')

    def test_wx_to_sx_returns_float(self):
        sx = self.xyp.wxToSx(50.0)
        self.assertIsInstance(sx, float)

    def test_wy_to_sy_returns_float(self):
        sy = self.xyp.wyToSy(50.0)
        self.assertIsInstance(sy, float)

    def test_sx_to_wx_returns_float(self):
        wx = self.xyp.sxToWx(128.0)
        self.assertIsInstance(wx, float)

    def test_sy_to_wy_returns_float(self):
        wy = self.xyp.syToWy(128.0)
        self.assertIsInstance(wy, float)

    def test_wx_ordering_preserved(self):
        sx_lo = self.xyp.wxToSx(10.0)
        sx_hi = self.xyp.wxToSx(90.0)
        self.assertLess(sx_lo, sx_hi)

    def test_wy_ordering_inverted(self):
        # Screen y increases downward; higher world y → lower screen y
        sy_lo_world = self.xyp.wyToSy(10.0)
        sy_hi_world = self.xyp.wyToSy(90.0)
        self.assertGreater(sy_lo_world, sy_hi_world)

    def test_midpoint_maps_near_center(self):
        _xorigin_, _xmin_, _dx_, _width_ = self.xyp.x_transform_vars
        _yorigin_, _ymin_, _dy_, _height_ = self.xyp.y_transform_vars
        wx_mid = _xmin_ + _dx_ / 2.0
        wy_mid = _ymin_ + _dy_ / 2.0
        sx_mid = self.xyp.wxToSx(wx_mid)
        sy_mid = self.xyp.wyToSy(wy_mid)
        # Should be near the center of the plot area (within 5 pixels)
        expected_sx = _xorigin_ + _width_ / 2.0
        expected_sy = _yorigin_ - _height_ / 2.0
        self.assertAlmostEqual(sx_mid, expected_sx, delta=5.0)
        self.assertAlmostEqual(sy_mid, expected_sy, delta=5.0)

    def test_transforms_vary_with_wxh(self):
        xyp_small = _make_xyp(self.p2s, wxh=(128, 128))
        xyp_large = _make_xyp(self.p2s, wxh=(512, 512))
        sx_small = xyp_small.wxToSx(50.0)
        sx_large = xyp_large.wxToSx(50.0)
        # Pixel position should differ for different widget sizes
        self.assertNotAlmostEqual(sx_small, sx_large, places=1)


if __name__ == '__main__':
    unittest.main()
