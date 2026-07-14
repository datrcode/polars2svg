import re
import unittest
import polars as pl
from polars2svg import Polars2SVG
from timep_dataframes import makeTimeDf


class TestTimepCount(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()
        self.df  = makeTimeDf(n=200, year=(2020, 2024), month=(1, 12))

    def _both_modes(self, **extra):
        '''Run once with auto-linear and once with a periodic enum.'''
        self.p2s.timep(self.df, 'ts',                   **extra)
        self.p2s.timep(self.df, ('ts', self.p2s.PT_mp), **extra)

    # ── count variants ────────────────────────────────────────────────────────

    def test_count_default_row_count(self):
        '''Default count (omitted) → ROW_COUNTp.'''
        self._both_modes()

    def test_count_explicit_row_count(self):
        self._both_modes(count=self.p2s.ROW_COUNTp)

    def test_count_numeric_int_field(self):
        '''Numeric (Int32) field → sum per bin.'''
        self._both_modes(count='value')

    def test_count_numeric_float_field(self):
        '''Float field → sum per bin.'''
        self._both_modes(count='numeric')

    def test_count_categorical_field_nunique(self):
        '''Non-numeric (Utf8) field → n_unique per bin.'''
        self._both_modes(count='category')

    def test_count_set_tuple_single_field(self):
        '''(field, SETp) → n_unique per bin, explicit set semantics.'''
        self._both_modes(count=('category', self.p2s.SETp))

    def test_count_multi_field_struct_nunique(self):
        '''(field1, field2) → struct n_unique per bin.'''
        self._both_modes(count=('category', 'value'))

    # ── count_range and count_range_shared ────────────────────────────────────

    def test_count_range_sets_y_axis_bounds(self):
        t = self.p2s.timep(self.df, 'ts', count_range=(0, 1000))
        self.assertEqual(t._count_min_, 0)
        self.assertEqual(t._count_max_, 1000)

    def test_count_range_periodic(self):
        t = self.p2s.timep(self.df, ('ts', self.p2s.PT_mp), count_range=(0, 500))
        self.assertEqual(t._count_min_, 0)
        self.assertEqual(t._count_max_, 500)

    def test_count_range_shared_sets_y_axis_bounds(self):
        t = self.p2s.timep(self.df, 'ts', count_range_shared=(5, 250))
        self.assertEqual(t._count_min_, 5)
        self.assertEqual(t._count_max_, 250)

    def test_count_range_shared_takes_precedence_over_count_range(self):
        '''count_range_shared beats count_range when both are supplied.'''
        t = self.p2s.timep(self.df, 'ts',
                           count_range=(0, 100), count_range_shared=(10, 500))
        self.assertEqual(t._count_min_, 10)
        self.assertEqual(t._count_max_, 500)

    def test_auto_count_max_positive(self):
        '''Without an explicit range the computed max is positive.'''
        t = self.p2s.timep(self.df, 'ts')
        self.assertGreater(t._count_max_, 0)
        self.assertEqual(t._count_min_, 0)

    # ── sub-1.0 count values fill full plot height ────────────────────────────

    def test_sub_one_count_simple_bar_fills_plot_height(self):
        '''Simple bar whose max sum is 0.01 must span the full plot height.'''
        df = pl.DataFrame({
            'ts':  ['2024-01-01', '2024-01-01', '2024-02-01'],
            'val': [0.007, 0.003, 0.001],
        }).with_columns(pl.col('ts').str.to_date().cast(pl.Datetime))
        t   = self.p2s.timep(df, 'ts', count='val', wxh=(256, 128))
        svg = t._repr_svg_()
        # Only solid-filled data rects: non-white/grey hex fill and no stroke attribute
        bar_heights = [float(re.search(r'height="([^"]+)"', m.group(0)).group(1))
                       for m in re.finditer(r'<rect\b[^>]*/>', svg)
                       if re.search(r'fill="#[0-9a-fA-F]{6}"', m.group(0))
                       and 'stroke=' not in m.group(0)
                       and '#ffffff' not in m.group(0)
                       and '#cccccc' not in m.group(0)]
        self.assertTrue(any(abs(bh - t._plot_h_) < 1.0 for bh in bar_heights),
                        f'Expected a bar ≈{t._plot_h_:.1f}px tall, got heights: {bar_heights}')

    def test_sub_one_count_stacked_bar_fills_plot_height(self):
        '''Stacked bars whose bin total is 0.01 must span the full plot height.'''
        df = pl.DataFrame({
            'ts':  ['2024-01-01', '2024-01-01'],
            'val': [0.007, 0.003],
            'grp': ['x', 'y'],
        }).with_columns(pl.col('ts').str.to_date().cast(pl.Datetime))
        t   = self.p2s.timep(df, 'ts', count='val', color='grp', wxh=(256, 128))
        svg = t._repr_svg_()
        # Keep only filled data-segment rects (stroke="none", fill not white/grey)
        segment_heights = [float(m.group(1))
                           for m in re.finditer(r'<rect[^>]+height="([^"]+)"[^>]+stroke="none"', svg)
                           if m.group(0).find('#ffffff') == -1 and m.group(0).find('#cccccc') == -1]
        total_height = sum(segment_heights)
        self.assertAlmostEqual(total_height, t._plot_h_, delta=1.0,
                               msg=f'Stacked segments sum {total_height:.1f} ≠ plot_h {t._plot_h_:.1f}')


if __name__ == '__main__':
    unittest.main()
