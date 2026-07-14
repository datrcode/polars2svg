import unittest

import polars as pl

from polars2svg.udist_scatterplots_via_sectors_tile_opt import UDistScatterPlotsViaSectorsTileOpt as UDist


_XS_ = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.5]
_YS_ = [0.5, 0.1, 0.8, 0.3, 0.6, 0.2, 0.9, 0.4, 0.7, 0.3]
_N_  = len(_XS_)


class TestUDistException(unittest.TestCase):

    def test_invalid_num_of_tiles_raises(self):
        # 7 is not in [16, 32, 64, 128, 256, 512, 1024]; parquet filter gives 0 rows
        with self.assertRaises(Exception) as ctx:
            UDist(_XS_, _YS_, num_of_tiles=7, iterations=1)
        self.assertIn('No xo/yo sector data found', str(ctx.exception))


class TestUDistResults(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.udist = UDist(_XS_, _YS_, iterations=1)

    def test_results_length_matches_input(self):
        x_out, y_out = self.udist.results()
        self.assertEqual(len(x_out), _N_)
        self.assertEqual(len(y_out), _N_)

    def test_results_returns_polars_series(self):
        x_out, y_out = self.udist.results()
        self.assertIsInstance(x_out, pl.Series)
        self.assertIsInstance(y_out, pl.Series)

    def test_results_values_are_float(self):
        x_out, y_out = self.udist.results()
        self.assertTrue(x_out.dtype.is_float())
        self.assertTrue(y_out.dtype.is_float())

    def test_results_in_unit_range(self):
        x_out, y_out = self.udist.results()
        self.assertGreaterEqual(float(x_out.min()), 0.0)
        self.assertLessEqual(float(x_out.max()), 1.0)
        self.assertGreaterEqual(float(y_out.min()), 0.0)
        self.assertLessEqual(float(y_out.max()), 1.0)


class TestUDistStaticPoints(unittest.TestCase):

    def test_all_static_points_same_result_regardless_of_iterations(self):
        # With all points static, no movement occurs; normalization is idempotent
        # so extra iterations must produce identical results
        static = [1] * _N_
        u1 = UDist(_XS_, _YS_, static_points=static, iterations=1)
        u4 = UDist(_XS_, _YS_, static_points=static, iterations=4)
        x1, y1 = u1.results()
        x4, y4 = u4.results()
        for i in range(_N_):
            self.assertAlmostEqual(float(x1[i]), float(x4[i]), places=5)
            self.assertAlmostEqual(float(y1[i]), float(y4[i]), places=5)


class TestUDistPublicAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.udist = UDist(_XS_, _YS_, iterations=1)

    def test_repr_svg_returns_svg_string(self):
        svg = self.udist._repr_svg_()
        self.assertIsInstance(svg, str)
        self.assertTrue(svg.startswith('<svg'))
        self.assertIn('</svg>', svg)

    def test_svg_animation_returns_animated_svg(self):
        svg = self.udist.svgAnimation()
        self.assertIsInstance(svg, str)
        self.assertTrue(svg.startswith('<svg'))
        self.assertIn('<animate', svg)

    def test_tile_bounds_x0_less_than_x1(self):
        x0, y0, x1, y1 = self.udist.tileBounds(0, 0)
        self.assertLess(x0, x1)

    def test_tile_bounds_y0_less_than_y1(self):
        x0, y0, x1, y1 = self.udist.tileBounds(0, 0)
        self.assertLess(y0, y1)

    def test_tile_bounds_covers_unit_square(self):
        # All tiles should tile [0, 1] x [0, 1] with no gaps
        n = self.udist.num_of_tiles
        x0, y0, x1, y1 = self.udist.tileBounds(0, 0)
        self.assertAlmostEqual(x0, 0.0)
        self.assertAlmostEqual(y0, 0.0)
        last_x0, last_y0, last_x1, last_y1 = self.udist.tileBounds(n - 1, n - 1)
        self.assertAlmostEqual(last_x1, 1.0)
        self.assertAlmostEqual(last_y1, 1.0)

    def test_with_explicit_colors(self):
        colors = ['#ff1234'] * _N_
        u = UDist(_XS_, _YS_, colors=colors, iterations=1)
        svg = u._repr_svg_()
        self.assertIn('#ff1234', svg)

    def test_with_explicit_weights(self):
        weights = [float(i + 1) for i in range(_N_)]
        u = UDist(_XS_, _YS_, weights=weights, iterations=1)
        x_out, y_out = u.results()
        self.assertEqual(len(x_out), _N_)

    def test_with_decay_rate(self):
        u = UDist(_XS_, _YS_, decay_rate=0.5, iterations=2)
        x_out, y_out = u.results()
        self.assertEqual(len(x_out), _N_)


if __name__ == '__main__':
    unittest.main()
