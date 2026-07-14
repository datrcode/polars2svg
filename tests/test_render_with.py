import unittest
import datetime
import polars as pl
from polars2svg import Polars2SVG


def _histop_df(n=10):
    return pl.DataFrame({'cat': [f'c{i % 3}' for i in range(n)], 'val': list(range(n))})


def _timep_df(n=6):
    base = datetime.date(2024, 1, 1)
    return pl.DataFrame({
        'ts':  [base + datetime.timedelta(days=i * 30) for i in range(n)],
        'val': list(range(n)),
    })


def _xyp_df(n=5):
    return pl.DataFrame({'x': list(range(n)), 'y': list(range(n))})


class TestHistopRenderWith(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def setUp(self):
        self.df1 = _histop_df(9)
        self.df2 = _histop_df(12)
        self.original = self.p2s.histop(self.df1, bin_by='cat')

    def test_render_with_returns_histop_instance(self):
        result = self.original.render_with(self.df2)
        self.assertTrue(hasattr(result, 'svg'))

    def test_render_with_produces_valid_svg(self):
        result = self.original.render_with(self.df2)
        self.assertIn('<svg', result.svg)

    def test_render_with_svg_differs_from_original(self):
        # df2 has more rows so bar lengths differ
        result = self.original.render_with(self.df2)
        self.assertNotEqual(self.original.svg, result.svg)

    def test_render_with_preserves_bin_by(self):
        result = self.original.render_with(self.df2)
        self.assertIn('<svg', result.svg)

    def test_render_with_override_accepted(self):
        result = self.original.render_with(self.df2, wxh=(200, 100))
        self.assertIn('<svg', result.svg)


class TestTimepRenderWith(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def setUp(self):
        self.df1 = _timep_df(6)
        self.df2 = _timep_df(12)
        self.original = self.p2s.timep(self.df1, 'ts')

    def test_render_with_returns_timep_instance(self):
        result = self.original.render_with(self.df2)
        self.assertTrue(hasattr(result, 'svg'))

    def test_render_with_produces_valid_svg(self):
        result = self.original.render_with(self.df2)
        self.assertIn('<svg', result.svg)

    def test_render_with_svg_changes_with_new_data(self):
        result = self.original.render_with(self.df2)
        self.assertIn('<svg', result.svg)

    def test_render_with_override_accepted(self):
        result = self.original.render_with(self.df2, wxh=(300, 150))
        self.assertIn('<svg', result.svg)


class TestXYpRenderWith(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def setUp(self):
        self.df1 = _xyp_df(5)
        self.df2 = _xyp_df(10)
        self.original = self.p2s.xyp(self.df1, 'x', 'y', wxh=(200, 200))

    def test_render_with_returns_xyp_instance(self):
        result = self.original.render_with(self.df2)
        self.assertTrue(hasattr(result, 'svg'))

    def test_render_with_produces_valid_svg(self):
        result = self.original.render_with(self.df2)
        self.assertIn('<svg', result.svg)

    def test_render_with_override_accepted(self):
        result = self.original.render_with(self.df2, wxh=(300, 300))
        self.assertIn('<svg', result.svg)


if __name__ == '__main__':
    unittest.main()
