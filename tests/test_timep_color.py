import unittest
import polars as pl
from polars2svg import Polars2SVG
from timep_dataframes import makeTimeDf


class TestTimepColor(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()
        self.df  = makeTimeDf(n=200, year=(2020, 2024), month=(1, 12))

    def _both_modes(self, **extra):
        '''Run with auto-linear and with a periodic enum.'''
        self.p2s.timep(self.df, 'ts',                   **extra)
        self.p2s.timep(self.df, ('ts', self.p2s.PT_mp), **extra)

    # ── no color (default) ───────────────────────────────────────────────────

    def test_no_color_default(self):
        '''Default: color=None, all bars use the default data color.'''
        self._both_modes()

    def test_no_color_explicit_none(self):
        self._both_modes(color=None)

    # ── categorical color ────────────────────────────────────────────────────

    def test_color_categorical_string_field(self):
        '''String column → color treated as categorical → stacked-bar coloring.'''
        self._both_modes(color='category')

    def test_color_cset_tuple(self):
        self._both_modes(color=('category', self.p2s.CSETp))

    def test_color_categorical_with_stackedbar_style(self):
        '''Categorical color + STACKEDBARp → stacked bars.'''
        self._both_modes(color='category', style=self.p2s.STACKEDBARp)

    def test_color_categorical_agg_type_is_stacked(self):
        t = self.p2s.timep(self.df, 'ts', color='category')
        self.assertEqual(t._agg_type_, 'stacked')

    # ── numeric color (spectrum) ─────────────────────────────────────────────

    def test_color_numeric_int_field(self):
        '''Int32 color field → spectrum coloring on whole bar (not stacked).'''
        self._both_modes(color='value')

    def test_color_numeric_float_field(self):
        '''Float64 color field → spectrum coloring on whole bar (not stacked).'''
        self._both_modes(color='numeric')

    def test_color_numeric_agg_type_is_simple(self):
        '''Numeric color stays in simple path.'''
        t = self.p2s.timep(self.df, 'ts', color='value')
        self.assertEqual(t._agg_type_, 'simple')

    def test_color_numeric_color_stat_in_df_agg(self):
        '''Numeric color → df_agg contains __color_stat__ column.'''
        t = self.p2s.timep(self.df, 'ts', color='value')
        self.assertIn('__color_stat__', t.df_agg.columns)

    def test_color_numeric_stat_range_computed(self):
        '''Numeric color → _color_stat_min_ and _color_stat_max_ are set.'''
        t = self.p2s.timep(self.df, 'ts', color='value')
        self.assertIsNotNone(t._color_stat_min_)
        self.assertIsNotNone(t._color_stat_max_)
        self.assertLessEqual(t._color_stat_min_, t._color_stat_max_)

    def test_color_numeric_cmagnitude_mean(self):
        '''(field, CMAGNITUDE_MEANp) → mean stat, spectrum coloring.'''
        t = self.p2s.timep(self.df, 'ts', color=('value', self.p2s.CMAGNITUDE_MEANp))
        self.assertEqual(t._agg_type_, 'simple')
        self.assertIn('__color_stat__', t.df_agg.columns)

    def test_color_numeric_cmagnitude_max(self):
        t = self.p2s.timep(self.df, 'ts', color=('value', self.p2s.CMAGNITUDE_MAXp))
        self.assertEqual(t._agg_type_, 'simple')

    def test_color_numeric_statistic_mean_tuple(self):
        '''(field, MEANp) also accepted.'''
        t = self.p2s.timep(self.df, 'ts', color=('value', self.p2s.MEANp))
        self.assertEqual(t._agg_type_, 'simple')

    def test_color_cset_numeric_is_stacked(self):
        '''(numeric-field, CSETp) → treated as categorical → stacked.'''
        t = self.p2s.timep(self.df, 'ts', color=('value', self.p2s.CSETp))
        self.assertEqual(t._agg_type_, 'stacked')

    def test_color_numeric_periodic(self):
        '''Numeric color in periodic mode → simple, spectrum.'''
        t = self.p2s.timep(self.df, ('ts', self.p2s.PT_mp), color='value')
        self.assertEqual(t._agg_type_, 'simple')
        self.assertIn('__color_stat__', t.df_agg.columns)

    # ── combined color + count ────────────────────────────────────────────────

    def test_color_categorical_with_numeric_count(self):
        '''Categorical color + numeric count field.'''
        self._both_modes(color='category', count='value')

    def test_color_numeric_with_numeric_count(self):
        '''Numeric color + numeric count field → renders.'''
        self._both_modes(color='value', count='numeric')


if __name__ == '__main__':
    unittest.main()
