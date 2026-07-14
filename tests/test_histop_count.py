import re
import unittest
import polars as pl
from polars2svg import Polars2SVG
from histop_dataframes import makeHistoDf


class TestHistopCount(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()
        self.df  = makeHistoDf(n=200)

    # ── count variants ────────────────────────────────────────────────────────

    def test_count_default_row_count(self):
        '''Default count (omitted) → ROW_COUNTp: bars show number of rows per bin.'''
        self.p2s.histop(self.df, 'cat')

    def test_count_explicit_row_count(self):
        self.p2s.histop(self.df, 'cat', count=self.p2s.ROW_COUNTp)

    def test_count_numeric_int_field(self):
        '''Numeric (Int32) field → sum per bin.'''
        self.p2s.histop(self.df, 'cat', count='value')

    def test_count_numeric_float_field(self):
        '''Float64 field → sum per bin.'''
        self.p2s.histop(self.df, 'cat', count='score')

    def test_count_categorical_field_nunique(self):
        '''Non-numeric (Utf8) field → n_unique per bin.'''
        self.p2s.histop(self.df, 'cat', count='group')

    def test_count_set_tuple(self):
        '''(field, SETp) → treat field as categorical (n_unique).'''
        self.p2s.histop(self.df, 'cat', count=('value', self.p2s.SETp))

    def test_count_multi_field_struct(self):
        '''(field1, field2) → struct n_unique per bin.'''
        self.p2s.histop(self.df, 'cat', count=('group', 'value'))

    # ── agg_type is 'simple' for all non-color count variants ────────────────

    def test_count_row_count_agg_type_simple(self):
        t = self.p2s.histop(self.df, 'cat')
        self.assertEqual(t._agg_type_, 'simple')

    def test_count_numeric_agg_type_simple(self):
        t = self.p2s.histop(self.df, 'cat', count='value')
        self.assertEqual(t._agg_type_, 'simple')

    # ── count_range ───────────────────────────────────────────────────────────

    def test_count_range_sets_bounds(self):
        t = self.p2s.histop(self.df, 'cat', count_range=(0, 500))
        self.assertEqual(t._count_min_, 0)
        self.assertEqual(t._count_max_, 500)

    def test_count_range_non_zero_min(self):
        t = self.p2s.histop(self.df, 'cat', count_range=(10, 200))
        self.assertEqual(t._count_min_, 10)
        self.assertEqual(t._count_max_, 200)

    # ── count_range_shared ────────────────────────────────────────────────────

    def test_count_range_shared_sets_bounds(self):
        t = self.p2s.histop(self.df, 'cat', count_range_shared=(5, 250))
        self.assertEqual(t._count_min_, 5)
        self.assertEqual(t._count_max_, 250)

    def test_count_range_shared_takes_precedence_over_count_range(self):
        '''count_range_shared beats count_range when both are supplied.'''
        t = self.p2s.histop(self.df, 'cat',
                            count_range=(0, 100), count_range_shared=(10, 500))
        self.assertEqual(t._count_min_, 10)
        self.assertEqual(t._count_max_, 500)

    # ── auto count range ──────────────────────────────────────────────────────

    def test_auto_count_max_positive(self):
        '''Without an explicit range the computed max is positive.'''
        t = self.p2s.histop(self.df, 'cat')
        self.assertGreater(t._count_max_, 0)
        self.assertEqual(t._count_min_, 0)

    def test_auto_count_max_stacked(self):
        '''Stacked agg_type computes max from per-bin totals.'''
        t = self.p2s.histop(self.df, 'cat', color='group')
        self.assertGreater(t._count_max_, 0)

    def test_auto_count_max_boxplot(self):
        '''Boxplot agg_type uses max of __box_max__ column.'''
        t = self.p2s.histop(self.df, 'cat', style=self.p2s.BOXPLOTp, count='value')
        self.assertGreater(t._count_max_, 0)


    # ── sub-1.0 count values fill full plot width ─────────────────────────────

    def test_sub_one_count_simple_bar_fills_plot_width(self):
        '''Simple bar whose sum is 0.001 must span the full plot width.'''
        df = pl.DataFrame({'bin': ['a', 'a', 'b'], 'val': [0.0006, 0.0004, 0.0001]})
        t   = self.p2s.histop(df, 'bin', count='val', wxh=(256, 64))
        svg = t._repr_svg_()
        bar_widths = [float(w) for w in re.findall(r'<rect[^>]+width="([^"]+)"', svg)
                      if float(w) < 256]  # exclude the outer SVG viewport rect
        self.assertTrue(any(abs(bw - t._plot_w_) < 1.0 for bw in bar_widths),
                        f'Expected a bar ≈{t._plot_w_}px wide, got widths: {bar_widths}')

    def test_sub_one_count_stacked_bar_fills_plot_width(self):
        '''Stacked bars whose bin total is 0.001 must span the full plot width.'''
        df = pl.DataFrame({'bin': ['a', 'a'], 'val': [0.0006, 0.0004], 'grp': ['x', 'y']})
        t   = self.p2s.histop(df, 'bin', count='val', color='grp', wxh=(256, 64))
        svg = t._repr_svg_()
        # Keep only filled data-segment rects (stroke="none", fill not white/grey)
        segment_widths = [float(m.group(1))
                          for m in re.finditer(r'<rect[^>]+width="([^"]+)"[^>]+stroke="none"', svg)
                          if m.group(0).find('#ffffff') == -1 and m.group(0).find('#cccccc') == -1]
        total_width = sum(segment_widths)
        self.assertAlmostEqual(total_width, t._plot_w_, delta=1.0,
                               msg=f'Stacked segments sum {total_width:.1f} ≠ plot_w {t._plot_w_:.1f}')


if __name__ == '__main__':
    unittest.main()
