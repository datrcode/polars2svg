import unittest
import polars as pl
from polars2svg import Polars2SVG
from histop_dataframes import makeHistoDf
from svg_test_utils import assert_valid_svg, assert_timing_metrics_populated, capture_log_warnings


class TestHistopBasic(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def setUp(self):
        self.df = makeHistoDf(n=100)

    # ── bin_by specification ──────────────────────────────────────────────────

    def test_bin_by_positional_string(self):
        '''bin_by as the first positional string arg.'''
        self.p2s.histop(self.df, 'cat')

    def test_bin_by_keyword(self):
        '''bin_by as an explicit keyword argument.'''
        self.p2s.histop(self.df, bin_by='cat')

    def test_df_as_keyword_arg(self):
        '''df= may be supplied as a keyword argument.'''
        self.p2s.histop(df=self.df, bin_by='cat')

    def test_bin_by_tuple_two_fields(self):
        '''bin_by as a tuple of two field names; bins are joined with "|".'''
        self.p2s.histop(self.df, ('cat', 'group'))

    def test_bin_by_tuple_keyword(self):
        '''bin_by tuple supplied as a keyword argument.'''
        self.p2s.histop(self.df, bin_by=('cat', 'group'))

    # ── SVG output ────────────────────────────────────────────────────────────

    def test_repr_svg_returns_valid_svg_string(self):
        t = self.p2s.histop(self.df, 'cat')
        assert_valid_svg(self, t._repr_svg_())

    def test_svg_contains_rects(self):
        '''A non-empty DataFrame must produce at least one <rect> bar.'''
        t = self.p2s.histop(self.df, 'cat')
        self.assertIn('<rect', t._repr_svg_())

    def test_svg_dimensions_match_wxh(self):
        '''The root <svg> element must carry the requested width and height.'''
        t = self.p2s.histop(self.df, 'cat', wxh=(300, 400))
        svg = t._repr_svg_()
        self.assertIn('width="300"',  svg)
        self.assertIn('height="400"', svg)

    # ── edge-case DataFrames ──────────────────────────────────────────────────

    def test_single_bin(self):
        '''DataFrame with only one unique bin value renders without error.'''
        df = pl.DataFrame({'cat': ['A', 'A', 'A'], 'value': [1, 2, 3]})
        self.p2s.histop(df, 'cat')

    def test_single_row(self):
        df = self.df.head(1)
        self.p2s.histop(df, 'cat')

    def test_empty_df_returns_blank_svg(self):
        '''An empty DataFrame should not crash; it returns a blank SVG.'''
        df_empty = self.df.clear()
        t = self.p2s.histop(df_empty, 'cat')
        self.assertIn('<svg', t._repr_svg_())

    # ── render options ────────────────────────────────────────────────────────

    def test_various_wxh(self):
        for w, h in [(128, 256), (256, 512), (512, 1024)]:
            self.p2s.histop(self.df, 'cat', wxh=(w, h))

    def test_draw_context_true(self):
        self.p2s.histop(self.df, 'cat', draw_context=True)

    def test_draw_context_false(self):
        self.p2s.histop(self.df, 'cat', draw_context=False)

    def test_draw_context_false_no_axis_elements(self):
        '''Disabling context means no vertical grid lines; SVG is still valid.'''
        t = self.p2s.histop(self.df, 'cat', draw_context=False)
        self.assertIn('<svg', t._repr_svg_())

    def test_draw_labels_false_no_bin_labels(self):
        '''Bin labels (per-bin entity labels) are suppressed when draw_labels=False;
        they are independent of draw_context (which covers grid lines / count axis).
        Isolate via draw_context=False so no axis <text> is present either way.'''
        t_lbl   = self.p2s.histop(self.df, 'cat', draw_context=False, draw_labels=True)
        t_nolbl = self.p2s.histop(self.df, 'cat', draw_context=False, draw_labels=False)
        self.assertIn('<text', t_lbl._repr_svg_())
        self.assertNotIn('<text', t_nolbl._repr_svg_())

    def test_draw_context_false_bin_labels_still_shown(self):
        '''draw_labels defaults True on histop, so bin labels survive draw_context=False.'''
        t_noctx = self.p2s.histop(self.df, 'cat', draw_context=False)
        self.assertIn('<text', t_noctx._repr_svg_())

    def test_custom_txt_h(self):
        for txt_h in [8, 10, 12, 16]:
            self.p2s.histop(self.df, 'cat', txt_h=txt_h)

    def test_custom_bar_h(self):
        '''Explicit bar_h overrides the txt_h default.'''
        t = self.p2s.histop(self.df, 'cat', bar_h=20)
        self.assertEqual(t.bar_h, 20)

    def test_bar_h_defaults_to_txt_h_plus_4(self):
        '''When bar_h is not given and draw_context=True, bar_h defaults to txt_h + 4.'''
        t = self.p2s.histop(self.df, 'cat', txt_h=14)
        self.assertEqual(t.bar_h, 18)

    def test_bar_h_defaults_to_5_when_no_context(self):
        '''When bar_h is not given and draw_context=False, bar_h defaults to 5.'''
        t = self.p2s.histop(self.df, 'cat', draw_context=False)
        self.assertEqual(t.bar_h, 5)

    def test_custom_v_gap(self):
        self.p2s.histop(self.df, 'cat', v_gap=4)

    def test_custom_insets(self):
        for insets in [(0, 0), (2, 2), (5, 10)]:
            self.p2s.histop(self.df, 'cat', insets=insets)

    def test_draw_distribution_tall_widget(self):
        '''draw_distribution=True with enough vertical space renders without error.'''
        self.p2s.histop(self.df, 'cat', draw_distribution=True, wxh=(256, 600))

    def test_lazy_and_eager_both_render(self):
        t_lazy  = self.p2s.histop(self.df, 'cat', use_lazy_execution=True)
        t_eager = self.p2s.histop(self.df, 'cat', use_lazy_execution=False)
        self.assertIn('<svg', t_lazy._repr_svg_())
        self.assertIn('<svg', t_eager._repr_svg_())

    # ── geometry sanity ───────────────────────────────────────────────────────

    def test_plot_region_within_widget_bounds(self):
        w, h = 256, 512
        t = self.p2s.histop(self.df, 'cat', wxh=(w, h))
        self.assertGreaterEqual(t._plot_x0_, 0)
        self.assertGreaterEqual(t._plot_y0_, 0)
        self.assertGreater(t._plot_w_, 0)
        self.assertLessEqual(t._plot_x0_ + t._plot_w_, w)

    def test_sorted_bins_populated(self):
        t = self.p2s.histop(self.df, 'cat')
        self.assertGreater(len(t._sorted_bins_), 0)

    def test_slot_h_equals_bar_h_plus_v_gap(self):
        t = self.p2s.histop(self.df, 'cat', bar_h=15, v_gap=3)
        self.assertEqual(t._slot_h_, 15 + 3)

    def test_bars_culled_when_svg_too_short(self):
        '''When bar_h * n_bins exceeds the SVG height, only fitting bars are rendered.'''
        import polars as pl
        # 20 distinct bins, bar_h=20, v_gap=0 → needs 400px; use only 150px height
        df = pl.DataFrame({'cat': [str(i % 20) for i in range(200)], 'value': list(range(200))})
        t  = self.p2s.histop(df, 'cat', bar_h=20, v_gap=0, wxh=(256, 150), draw_context=False)
        # All 20 bins exist in sorted_bins but not all should be rendered
        self.assertEqual(len(t._sorted_bins_), 20)
        # Count <rect> elements in SVG — should be fewer than 20
        import re
        n_rects = len(re.findall(r'<rect\b', t._repr_svg_()))
        # Background rect + rendered bars; rendered bars must be < 20
        self.assertLess(n_rects - 1, 20)  # subtract the background rect

    def test_timing_metrics_populated(self):
        t = self.p2s.histop(self.df, 'cat')
        assert_timing_metrics_populated(self, t,
            ('__parseInput__', '__validateInput__', '__renderSVG__'))


class TestHistopWxhValidation(unittest.TestCase):
    """wxh accepts any 2-sequence of numbers and coerces floats to int
    (shared Polars2SVG.normalizeWxh); see tests/test_wxh_normalization.py."""

    def setUp(self):
        self.p2s = Polars2SVG()
        self.df  = makeHistoDf(n=50)

    def test_wxh_float_width_coerced(self):
        h = self.p2s.histop(self.df, 'cat', wxh=(256.9, 128))
        self.assertEqual(h.wxh, (256, 128))

    def test_wxh_float_height_coerced(self):
        h = self.p2s.histop(self.df, 'cat', wxh=(256, 128.5))
        self.assertEqual(h.wxh, (256, 128))

    def test_wxh_list_coerced_to_tuple(self):
        h = self.p2s.histop(self.df, 'cat', wxh=[256, 128])
        self.assertEqual(h.wxh, (256, 128))

    def test_wxh_bad_still_raises(self):
        with self.assertRaises(ValueError):
            self.p2s.histop(self.df, 'cat', wxh=(256, 'x'))

    def test_wxh_int_int_ok(self):
        self.p2s.histop(self.df, 'cat', wxh=(256, 128))


class TestHistopSmSharedWarnings(unittest.TestCase):
    """Unsupported SM_* values log a warning; supported values do not."""

    def setUp(self):
        self.p2s = Polars2SVG()
        self.df  = makeHistoDf(n=100)

    def test_sm_x_warns(self):
        records = capture_log_warnings(
            lambda: self.p2s.histop(self.df, 'cat', sm_shared={self.p2s.SM_X})
        )
        self.assertTrue(any('SM_X' in r.getMessage() or 'sm_shared' in r.getMessage()
                            for r in records))

    def test_sm_y_warns(self):
        records = capture_log_warnings(
            lambda: self.p2s.histop(self.df, 'cat', sm_shared={self.p2s.SM_Y})
        )
        self.assertTrue(any('SM_Y' in r.getMessage() or 'sm_shared' in r.getMessage()
                            for r in records))

    def test_sm_count_no_warning(self):
        """SM_COUNT is supported by Histop — no warning expected."""
        records = capture_log_warnings(
            lambda: self.p2s.histop(self.df, 'cat', sm_shared={self.p2s.SM_COUNT})
        )
        self.assertEqual([r for r in records if 'sm_shared' in r.getMessage()], [])

    def test_sm_color_no_warning(self):
        records = capture_log_warnings(
            lambda: self.p2s.histop(self.df, 'cat', sm_shared={self.p2s.SM_COLOR})
        )
        self.assertEqual([r for r in records if 'sm_shared' in r.getMessage()], [])


if __name__ == '__main__':
    unittest.main()
