import unittest
import polars as pl
from polars2svg import Polars2SVG
from histop_dataframes import makeHistoDf


class TestHistopColor(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()
        self.df  = makeHistoDf(n=200)

    # ── no color (default) ───────────────────────────────────────────────────

    def test_no_color_default(self):
        '''Default: color=None, all bars use the default data color.'''
        self.p2s.histop(self.df, 'cat')

    def test_no_color_explicit_none(self):
        self.p2s.histop(self.df, 'cat', color=None)

    def test_no_color_agg_type_simple(self):
        t = self.p2s.histop(self.df, 'cat')
        self.assertEqual(t._agg_type_, 'simple')

    # ── numeric color (spectrum) ─────────────────────────────────────────────

    def test_color_numeric_int_field(self):
        '''Numeric (Int32) color field → spectrum coloring on whole bar.'''
        self.p2s.histop(self.df, 'cat', color='value')

    def test_color_numeric_float_field(self):
        '''Float64 color field → spectrum coloring on whole bar.'''
        self.p2s.histop(self.df, 'cat', color='score')

    def test_color_numeric_agg_type_simple(self):
        '''Numeric color stays in simple path (spectrum, not stacked).'''
        t = self.p2s.histop(self.df, 'cat', color='value')
        self.assertEqual(t._agg_type_, 'simple')

    def test_color_numeric_not_categorical(self):
        t = self.p2s.histop(self.df, 'cat', color='value')
        self.assertFalse(t._color_is_categorical_)

    def test_color_numeric_color_stat_in_df_agg(self):
        '''Numeric color → df_agg contains __color_stat__ column.'''
        t = self.p2s.histop(self.df, 'cat', color='value')
        self.assertIn('__color_stat__', t.df_agg.columns)

    def test_color_numeric_stat_range_computed(self):
        '''Numeric color → _color_stat_min_ and _color_stat_max_ are set.'''
        t = self.p2s.histop(self.df, 'cat', color='value')
        self.assertIsNotNone(t._color_stat_min_)
        self.assertIsNotNone(t._color_stat_max_)
        self.assertLessEqual(t._color_stat_min_, t._color_stat_max_)

    def test_color_numeric_cmagnitude_mean(self):
        '''(field, CMAGNITUDE_MEANp) → mean statistic, spectrum coloring.'''
        t = self.p2s.histop(self.df, 'cat', color=('value', self.p2s.CMAGNITUDE_MEANp))
        self.assertEqual(t._agg_type_, 'simple')
        self.assertIn('__color_stat__', t.df_agg.columns)

    def test_color_numeric_cmagnitude_max(self):
        t = self.p2s.histop(self.df, 'cat', color=('value', self.p2s.CMAGNITUDE_MAXp))
        self.assertEqual(t._agg_type_, 'simple')

    def test_color_numeric_statistic_mean(self):
        '''(field, MEANp) → mean statistic, spectrum coloring.'''
        t = self.p2s.histop(self.df, 'cat', color=('value', self.p2s.MEANp))
        self.assertEqual(t._agg_type_, 'simple')

    def test_color_numeric_with_numeric_count(self):
        '''Numeric color + numeric count field → renders.'''
        self.p2s.histop(self.df, 'cat', color='score', count='value')

    # ── categorical color ────────────────────────────────────────────────────

    def test_color_categorical_string_field(self):
        '''String color field with different values than bin → stacked bars.'''
        self.p2s.histop(self.df, 'cat', color='group')

    def test_color_categorical_agg_type_stacked(self):
        t = self.p2s.histop(self.df, 'cat', color='group')
        self.assertEqual(t._agg_type_, 'stacked')

    def test_color_categorical_is_categorical_flag(self):
        t = self.p2s.histop(self.df, 'cat', color='group')
        self.assertTrue(t._color_is_categorical_)

    def test_color_cset_tuple(self):
        '''(field, CSETp) → treat field as categorical even if numeric → stacked.'''
        self.p2s.histop(self.df, 'cat', color=('value', self.p2s.CSETp))

    def test_color_cset_tuple_agg_type_stacked(self):
        t = self.p2s.histop(self.df, 'cat', color=('value', self.p2s.CSETp))
        self.assertEqual(t._agg_type_, 'stacked')

    def test_color_categorical_with_stackedbar_style(self):
        '''Categorical color + explicit STACKEDBARp → stacked bars.'''
        self.p2s.histop(self.df, 'cat', color='group', style=self.p2s.STACKEDBARp)

    # ── color == bin field ────────────────────────────────────────────────────

    def test_color_same_as_bin_field_renders(self):
        '''color=bin_by (categorical): each bar gets a distinct hash color.'''
        self.p2s.histop(self.df, 'cat', color='cat')

    def test_color_same_as_bin_agg_type_simple(self):
        '''When color==bin there is one color per bar; stays in simple path.'''
        t = self.p2s.histop(self.df, 'cat', color='cat')
        self.assertEqual(t._agg_type_, 'simple')

    def test_color_numeric_same_as_bin_uses_spectrum(self):
        '''Numeric color==bin → spectrum coloring, __color_stat__ present.'''
        t = self.p2s.histop(self.df, 'value', color='value')
        self.assertEqual(t._agg_type_, 'simple')
        self.assertIn('__color_stat__', t.df_agg.columns)

    # ── combined color + count ────────────────────────────────────────────────

    def test_color_categorical_with_numeric_count(self):
        '''Categorical color + numeric count field.'''
        self.p2s.histop(self.df, 'cat', color='group', count='value')


if __name__ == '__main__':
    unittest.main()
