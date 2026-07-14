"""Extended histop tests targeting uncovered validation, count/order expressions,
lazy vs eager paths, narrow-widget label fitting, and stacked distribution rendering.
"""
import unittest
import polars as pl
from polars2svg import Polars2SVG
from histop_dataframes import makeHistoDf, makeOrderedHistoDf


class TestHistopValidationErrors(unittest.TestCase):
    '''Error-path tests for __parseInput__ and __validateInput__.'''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()
        self.df  = makeHistoDf(n=50)

    # ── __parseInput__ duplicate-df errors (lines 72, 75) ────────────────────

    def test_df_passed_twice_positional_raises(self):
        '''Two positional DataFrame args → ValueError.'''
        with self.assertRaises(ValueError):
            self.p2s.histop(self.df, self.df, 'cat')

    def test_df_passed_positional_and_keyword_raises(self):
        '''df as positional AND as keyword → ValueError.'''
        with self.assertRaises(ValueError):
            self.p2s.histop(self.df, 'cat', df=self.df)

    # ── __parseInput__ bin_by already-set / unknown type (lines 87, 90, 92) ──

    def test_two_string_bin_by_args_raises(self):
        '''Two positional string args → bin_by already set ValueError.'''
        with self.assertRaises(ValueError):
            self.p2s.histop(self.df, 'cat', 'group')

    def test_string_then_tuple_bin_by_raises(self):
        '''string then tuple → bin_by already set ValueError.'''
        with self.assertRaises(ValueError):
            self.p2s.histop(self.df, 'cat', ('cat', 'group'))

    def test_unknown_positional_arg_type_raises(self):
        '''An unrecognised positional arg (e.g. an int) → ValueError.'''
        with self.assertRaises(ValueError):
            self.p2s.histop(self.df, 42)

    # ── __validateInput__ (lines 124, 132, 136, 141, 145, 150, 154, 159, 166) ─

    def test_no_bin_by_raises(self):
        '''Missing bin_by → ValueError.'''
        with self.assertRaises(ValueError):
            self.p2s.histop(self.df, bin_by=None)

    def test_bin_by_invalid_type_raises(self):
        '''bin_by set to an integer → ValueError (not str or tuple).'''
        with self.assertRaises(ValueError):
            self.p2s.histop(self.df, bin_by=99)

    def test_bin_by_missing_column_raises(self):
        '''bin_by names a column not in the DataFrame → ValueError.'''
        with self.assertRaises(ValueError):
            self.p2s.histop(self.df, 'nonexistent_col')

    def test_count_missing_column_raises(self):
        '''count names a column not in the DataFrame → ValueError.'''
        with self.assertRaises(ValueError):
            self.p2s.histop(self.df, 'cat', count='no_such_column')

    def test_count_tuple_missing_field_raises(self):
        '''count tuple with a missing column → ValueError.'''
        with self.assertRaises(ValueError):
            self.p2s.histop(self.df, 'cat', count=('value', 'no_such'))

    def test_color_missing_column_raises(self):
        '''color names a column not in the DataFrame → ValueError.'''
        with self.assertRaises(ValueError):
            self.p2s.histop(self.df, 'cat', color='no_such_color')

    def test_color_tuple_missing_field_raises(self):
        '''color tuple with a missing column → ValueError.'''
        with self.assertRaises(ValueError):
            self.p2s.histop(self.df, 'cat', color=('group', 'no_such'))

    def test_invalid_style_raises(self):
        '''Unrecognised style enum → ValueError.'''
        with self.assertRaises(ValueError):
            self.p2s.histop(self.df, 'cat', style=object())

    def test_invalid_wxh_raises(self):
        '''wxh that is not a 2-tuple → ValueError or TypeError.'''
        with self.assertRaises((ValueError, TypeError)):
            self.p2s.histop(self.df, 'cat', wxh=256)


class TestHistopCountExpressions(unittest.TestCase):
    '''Tests for __countAggExpr__ / __countFields__ / __findNumericCountField__ tuple paths.'''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()
        self.df  = makeHistoDf(n=100)

    def test_count_single_numeric_tuple(self):
        '''count=('value',) — length-1 tuple with numeric field → sum (line 208).'''
        t = self.p2s.histop(self.df, 'cat', count=('value',))
        self.assertEqual(t._agg_type_, 'simple')
        self.assertIn('<svg', t._repr_svg_())

    def test_count_struct_n_unique_tuple(self):
        '''count=('cat','group') — multi-field tuple → struct n_unique (line 210).'''
        t = self.p2s.histop(self.df, 'group', count=('cat', 'value'))
        self.assertIn('<svg', t._repr_svg_())

    def test_count_fallback_returns_row_count(self):
        '''Explicit ROW_COUNTp count always produces a valid histogram.'''
        t = self.p2s.histop(self.df, 'cat', count=self.p2s.ROW_COUNTp)
        self.assertGreater(t._count_max_, 0)


class TestHistopOrderExpressions(unittest.TestCase):
    '''Tests for __orderAggExpr__ tuple variants (lines 230, 236, 249).'''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()
        self.df  = makeHistoDf(n=100)

    def test_order_row_count(self):
        '''order=ROW_COUNTp → bars ordered by row count (line 230).'''
        t = self.p2s.histop(self.df, 'cat', order=self.p2s.ROW_COUNTp)
        self.assertGreater(len(t._sorted_bins_), 0)

    def test_order_single_field_tuple(self):
        '''order=('value',) — single-field tuple → sum ordering (line 236).'''
        t = self.p2s.histop(self.df, 'cat', order=('value',))
        self.assertGreater(len(t._sorted_bins_), 0)

    def test_order_set_tuple(self):
        '''order=('group', SETp) → n_unique ordering.'''
        t = self.p2s.histop(self.df, 'cat', order=('group', self.p2s.SETp))
        self.assertGreater(len(t._sorted_bins_), 0)

    def test_order_statistic_min(self):
        '''order=('value', MINp) → min-based ordering.'''
        t = self.p2s.histop(self.df, 'cat', order=('value', self.p2s.MINp))
        self.assertGreater(len(t._sorted_bins_), 0)

    def test_order_statistic_mean(self):
        '''order=('value', MEANp) → mean-based ordering.'''
        t = self.p2s.histop(self.df, 'cat', order=('value', self.p2s.MEANp))
        self.assertGreater(len(t._sorted_bins_), 0)


class TestHistopNonLazyPaths(unittest.TestCase):
    '''Tests for non-lazy (eager) aggregation paths.'''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()
        self.df  = makeHistoDf(n=100)

    def test_non_lazy_stacked(self):
        '''use_lazy_execution=False with categorical color → stacked bars (line 317).'''
        t = self.p2s.histop(self.df, 'cat', color='group', use_lazy_execution=False)
        self.assertEqual(t._agg_type_, 'stacked')
        self.assertIn('<svg', t._repr_svg_())

    def test_non_lazy_boxplot(self):
        '''use_lazy_execution=False with BOXPLOTp (line 298).'''
        t = self.p2s.histop(self.df, 'cat', style=self.p2s.BOXPLOTp,
                            count='value', use_lazy_execution=False)
        self.assertEqual(t._agg_type_, 'boxplot')
        self.assertIn('<svg', t._repr_svg_())


class TestHistopMultiStringColor(unittest.TestCase):
    '''Tests for color=tuple-with-multiple-string-fields (lines 263-264).'''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()
        self.df  = makeHistoDf(n=100)

    def test_color_two_string_fields_creates_concat_color(self):
        '''color=('cat','group') → __color__ concatenation column (lines 263-264).'''
        # 'cat' and 'group' are both categorical — multi-string tuple path
        t = self.p2s.histop(self.df, 'group', color=('cat', 'group'))
        self.assertEqual(t._color_field_, '__color__')
        self.assertIn('<svg', t._repr_svg_())


class TestHistopNarrowWidgetLabels(unittest.TestCase):
    '''Tests for label-fitting logic in __renderSVG__ (lines 557-563).'''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()
        self.df  = makeHistoDf(n=200)

    def test_narrow_widget_renders_without_error(self):
        '''Very narrow widget triggers the min_fits_ and max-only label paths.'''
        # Width=40 is so narrow only max label (or none) fits
        t = self.p2s.histop(self.df, 'cat', wxh=(40, 300), draw_context=True)
        self.assertIn('<svg', t._repr_svg_())

    def test_medium_widget_renders_without_error(self):
        '''Medium-width widget: min+max fit but center label doesn't.'''
        t = self.p2s.histop(self.df, 'cat', wxh=(80, 300), draw_context=True)
        self.assertIn('<svg', t._repr_svg_())

    def test_tuple_count_label_in_context(self):
        '''Tuple count → ctr_str is the joined field names (line 538 path).'''
        t = self.p2s.histop(self.df, 'cat', count=('group', 'value'),
                            wxh=(256, 300), draw_context=True)
        self.assertIn('<svg', t._repr_svg_())


class TestHistopDistributionStacked(unittest.TestCase):
    '''draw_distribution=True with a stacked chart (line 594).'''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()
        self.df  = makeHistoDf(n=200)

    def test_distribution_with_stacked_color(self):
        '''draw_distribution=True + categorical color → stacked + distribution path.'''
        t = self.p2s.histop(self.df, 'cat', color='group',
                            draw_distribution=True, wxh=(256, 800))
        self.assertIn('<svg', t._repr_svg_())


class TestSwarmMaxPtsHistop(unittest.TestCase):
    """swarm_max_pts caps the number of jitter dots rendered per bin."""

    def setUp(self):
        self.p2s = Polars2SVG()
        import random
        rng  = random.Random(0)
        rows = {'cat': [], 'value': []}
        for cat in 'ABCDEFGHIJ':
            for _ in range(100):
                rows['cat'].append(cat)
                rows['value'].append(rng.randint(1, 200))
        self.df_large = pl.DataFrame(rows)

    def test_swarm_max_pts_default_is_50(self):
        t = self.p2s.histop(self.df_large, 'cat', style=self.p2s.BOXPLOT_W_SWARMp, count='value')
        self.assertEqual(t.swarm_max_pts, 50)

    def test_swarm_max_pts_accepted_as_kwarg(self):
        t = self.p2s.histop(self.df_large, 'cat',
                            style=self.p2s.BOXPLOT_W_SWARMp, count='value', swarm_max_pts=10)
        self.assertEqual(t.swarm_max_pts, 10)

    def test_swarm_points_capped_at_default(self):
        """df_swarm must have at most swarm_max_pts rows per bin."""
        t = self.p2s.histop(self.df_large, 'cat', style=self.p2s.BOXPLOT_W_SWARMp, count='value')
        self.assertIsNotNone(t.df_swarm)
        for (cat,), grp in t.df_swarm.group_by('cat', maintain_order=False):
            self.assertLessEqual(len(grp), 50,
                                 f'bin "{cat}" has {len(grp)} swarm points, expected ≤ 50')

    def test_swarm_points_capped_at_custom_value(self):
        cap = 5
        t = self.p2s.histop(self.df_large, 'cat',
                            style=self.p2s.BOXPLOT_W_SWARMp, count='value', swarm_max_pts=cap)
        self.assertIsNotNone(t.df_swarm)
        for (cat,), grp in t.df_swarm.group_by('cat', maintain_order=False):
            self.assertLessEqual(len(grp), cap,
                                 f'bin "{cat}" has {len(grp)} swarm points, expected ≤ {cap}')

    def test_swarm_max_pts_1_renders_without_error(self):
        """Edge case: cap of 1 must still render valid SVG."""
        t = self.p2s.histop(self.df_large, 'cat',
                            style=self.p2s.BOXPLOT_W_SWARMp, count='value', swarm_max_pts=1)
        self.assertIn('<svg', t._repr_svg_())

    def test_swarm_max_pts_preserved_through_template(self):
        """A template created with a custom cap should pass it through."""
        tmpl = self.p2s.histop(self.df_large, 'cat',
                               style=self.p2s.BOXPLOT_W_SWARMp, count='value', swarm_max_pts=7)
        t = self.p2s.histop(self.df_large, template=tmpl)
        self.assertEqual(t.swarm_max_pts, 7)
        for (cat,), grp in t.df_swarm.group_by('cat', maintain_order=False):
            self.assertLessEqual(len(grp), 7)

    def test_timep_swarm_max_pts_still_works(self):
        """Regression: Timep swarm capping must continue to function."""
        from timep_dataframes import makeTimeDf
        df = makeTimeDf(n=500, year=(2020, 2022), month=(1, 12))
        t  = self.p2s.timep(df, 'ts', style=self.p2s.BOXPLOT_W_SWARMp, count='value', swarm_max_pts=10)
        self.assertIsNotNone(t.df_swarm)
        self.assertLessEqual(len(t.df_swarm), t.df_swarm['ts'].n_unique() * 10)


if __name__ == '__main__':
    unittest.main()
