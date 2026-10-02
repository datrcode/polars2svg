import re
import unittest
import polars as pl
from polars2svg import Polars2SVG
from histop_dataframes import makeHistoDf, histopBars
from svg_test_utils import assert_valid_svg, assert_timing_metrics_populated, capture_log_warnings, normalize_svg



class TestHistopBasic(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def setUp(self):
        self.df = makeHistoDf(n=100)

    # Every bar is its bin's row count: one bar per bin, largest first and ties A to Z,
    # the longest spanning the plot and the rest in proportion.  `keys` is each row's bin.
    def assertBarsAreRowCounts(self, h, keys: pl.Series) -> None:
        _want_ = dict(keys.value_counts().iter_rows())
        _bars_ = histopBars(h)
        self.assertEqual([_lbl_ for _lbl_, *_ in _bars_], sorted(_want_, key=lambda k: (-_want_[k], k)),
                         'one bar per bin, largest first, ties in label order')
        _top_ = max(_want_.values())
        for _label_, _x_, _y_, _w_ in _bars_:
            self.assertAlmostEqual(_w_, h._plot_w_ * _want_[_label_] / _top_, delta=0.06,
                                   msg=f'bar {_label_!r} is not as long as its {_want_[_label_]} rows')

    # Two calls that mean the same thing render identically
    def assertSameRender(self, a, b) -> None:
        self.assertEqual(normalize_svg(a._repr_svg_()), normalize_svg(b._repr_svg_()))

    # ── bin_by specification ──────────────────────────────────────────────────

    def test_bin_by_positional_string(self):
        '''bin_by as the first positional string arg.'''
        self.assertBarsAreRowCounts(self.p2s.histop(self.df, 'cat'), self.df['cat'])

    def test_bin_by_keyword(self):
        '''bin_by as an explicit keyword argument.'''
        self.assertSameRender(self.p2s.histop(self.df, bin_by='cat'), self.p2s.histop(self.df, 'cat'))

    def test_df_as_keyword_arg(self):
        '''df= may be supplied as a keyword argument.'''
        self.assertSameRender(self.p2s.histop(df=self.df, bin_by='cat'), self.p2s.histop(self.df, 'cat'))

    def test_bin_by_tuple_two_fields(self):
        '''bin_by as a tuple of two field names; bins are joined with "|".'''
        self.assertBarsAreRowCounts(self.p2s.histop(self.df, ('cat', 'group')),
                                    self.df.select(pl.concat_str('cat', 'group', separator='|'))['cat'])

    def test_bin_by_tuple_keyword(self):
        '''bin_by tuple supplied as a keyword argument.'''
        self.assertSameRender(self.p2s.histop(self.df, bin_by=('cat', 'group')), self.p2s.histop(self.df, ('cat', 'group')))

    def test_tied_bins_are_in_label_order(self):
        '''Bins with equal counts sort by their own value, every time.  Their order used
        to be group_by's: three bins of two rows came out in all six orders over thirty
        identical calls.'''
        df = pl.DataFrame({'cat': list('ccaabbd')})
        for _descending_, _want_ in ((True, ['a', 'b', 'c', 'd']), (False, ['d', 'a', 'b', 'c'])):
            with self.subTest(descending=_descending_):
                _orders_ = {tuple(_lbl_ for _lbl_, *_ in histopBars(self.p2s.histop(df, 'cat', descending=_descending_)))
                            for _ in range(20)}
                self.assertEqual(_orders_, {tuple(_want_)})

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
        '''One unique bin value: one full-width bar of three rows.'''
        df = pl.DataFrame({'cat': ['A', 'A', 'A'], 'value': [1, 2, 3]})
        self.assertBarsAreRowCounts(self.p2s.histop(df, 'cat'), df['cat'])

    def test_single_row(self):
        df = self.df.head(1)
        self.assertBarsAreRowCounts(self.p2s.histop(df, 'cat'), df['cat'])

    def test_empty_df_returns_blank_svg(self):
        '''An empty DataFrame should not crash; it returns a blank SVG.'''
        df_empty = self.df.clear()
        t = self.p2s.histop(df_empty, 'cat')
        self.assertIn('<svg', t._repr_svg_())

    # ── render options ────────────────────────────────────────────────────────

    def test_various_wxh(self):
        for w, h in [(128, 256), (256, 512), (512, 1024)]:
            with self.subTest(wxh=(w, h)):
                _h_ = self.p2s.histop(self.df, 'cat', wxh=(w, h))
                self.assertIn(f'width="{w}" height="{h}"', _h_._repr_svg_())
                self.assertBarsAreRowCounts(_h_, self.df['cat'])

    def test_draw_context_true(self):
        # The default: the count axis -- grid lines, the 'Rows' title, the largest count
        _h_ = self.p2s.histop(self.df, 'cat', draw_context=True)
        self.assertSameRender(_h_, self.p2s.histop(self.df, 'cat'))
        _svg_ = _h_._repr_svg_()
        self.assertIn('<line', _svg_)
        self.assertIn('>Rows</text>', _svg_)
        self.assertIn(f'>{self.df["cat"].value_counts()["count"].max()}</text>', _svg_)

    def test_draw_context_false(self):
        # No count axis at all, and the bars are unchanged in proportion
        _h_ = self.p2s.histop(self.df, 'cat', draw_context=False)
        _svg_ = _h_._repr_svg_()
        self.assertNotIn('<line', _svg_)
        self.assertNotIn('>Rows</text>', _svg_)
        self.assertBarsAreRowCounts(_h_, self.df['cat'])

    def test_draw_context_false_no_axis_elements(self):
        '''Disabling context means no vertical grid lines; SVG is still valid.'''
        t = self.p2s.histop(self.df, 'cat', draw_context=False)
        self.assertIn('<svg', t._repr_svg_())
        self.assertNotIn('<line', t._repr_svg_())

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

    # ── bin label width budget ───────────────────────────────────────────────
    # The bin label is drawn inside the plot at the left edge, so the whole bar row
    # is its budget.  It used to be cropped at half the plot width, which truncated
    # names that had ample room to render in full.

    def test_bin_label_uses_full_plot_width(self):
        '''A bin label wider than half the row -- but comfortably inside the whole
        row -- renders untruncated.'''
        _txt_h_, _w_ = 12, 256
        _plot_w_ = _w_ - 2 * 2                      # insets; no right strip w/o context
        _name_   = 'ww'
        while self.p2s.textLength(_name_, _txt_h_) <= _plot_w_ * 0.6:
            _name_ += 'w'
        self.assertGreater(self.p2s.textLength(_name_, _txt_h_), _plot_w_ * 0.5)
        self.assertLess(self.p2s.textLength(_name_, _txt_h_), _plot_w_ - 4)
        _df_  = pl.DataFrame({'cat': [_name_, 'b']})
        _svg_ = self.p2s.histop(_df_, 'cat', wxh=(_w_, 128), txt_h=_txt_h_,
                                draw_context=False)._repr_svg_()
        self.assertIn(f'>{_name_}</text>', _svg_)
        self.assertNotIn('...', _svg_)

    def test_bin_label_crop_stays_inside_plot(self):
        '''A label too long for the row is cropped so that the appended ellipsis
        still lands inside the plot rather than overhanging its right edge.'''
        _txt_h_ = 12
        _df_    = pl.DataFrame({'cat': ['a really really long bin name ' * 3, 'b']})
        for _w_ in (64, 96, 128, 192, 256, 384):
            with self.subTest(w=_w_):
                _t_     = self.p2s.histop(_df_, 'cat', wxh=(_w_, 128), txt_h=_txt_h_,
                                          draw_context=False)
                _svg_   = _t_._repr_svg_()
                _right_ = _t_._plot_x0_ + _t_._plot_w_
                _labels_ = re.findall(r'<text x="([\d.]+)"[^>]*>([^<]*)</text>', _svg_)
                self.assertTrue(_labels_, 'no bin labels rendered')
                self.assertTrue(any(_l_.endswith('...') for _, _l_ in _labels_),
                                'expected the long bin name to be cropped')
                for _x_, _lbl_ in _labels_:
                    self.assertLessEqual(float(_x_) + self.p2s.textLength(_lbl_, _txt_h_),
                                         _right_)

    def test_custom_txt_h(self):
        # The bin labels are drawn at txt_h, and the bars are txt_h + 4 tall to hold them
        for txt_h in [8, 10, 12, 16]:
            with self.subTest(txt_h=txt_h):
                _h_ = self.p2s.histop(self.df, 'cat', txt_h=txt_h)
                self.assertEqual(_h_.bar_h, txt_h + 4)
                self.assertBarsAreRowCounts(_h_, self.df['cat'])   # which finds the labels by their font-size

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
        # Bar rows are bar_h + v_gap apart
        _h_ = self.p2s.histop(self.df, 'cat', v_gap=4)
        _ys_ = [y for _, _, y, _ in histopBars(_h_)]
        self.assertEqual({round(b - a, 3) for a, b in zip(_ys_, _ys_[1:])}, {_h_.bar_h + 4})

    def test_custom_insets(self):
        # insets move the plot: the bars start insets[0] in, and insets[1] further down
        _top0_ = histopBars(self.p2s.histop(self.df, 'cat', insets=(0, 0)))[0][2]
        for insets in [(0, 0), (2, 2), (5, 10)]:
            with self.subTest(insets=insets):
                _h_    = self.p2s.histop(self.df, 'cat', insets=insets)
                _bars_ = histopBars(_h_)
                self.assertEqual({x for _, x, _, _ in _bars_}, {float(insets[0])})
                self.assertEqual(_bars_[0][2], _top0_ + insets[1])
                self.assertBarsAreRowCounts(_h_, self.df['cat'])

    def test_draw_distribution_tall_widget(self):
        '''draw_distribution=True draws a histogram of the bar lengths below the bars:
        how many bins' counts fall in each tenth of [smallest count, largest count].'''
        _h_ = self.p2s.histop(self.df, 'cat', draw_distribution=True, wxh=(256, 600))
        _svg_ = _h_._repr_svg_()
        _heights_ = [float(hh) for hh in re.findall(r'<rect x="[\d.]+" y="[\d.]+" width="[\d.]+" height="([\d.]+)" fill="#[0-9a-fA-F]{6}" fill-opacity="0.4"', _svg_)]
        _counts_ = self.df['cat'].value_counts()['count'].to_list()
        _lo_, _span_ = min(_counts_), max(_counts_) - min(_counts_)
        _freq_ = [0] * 10
        for _c_ in _counts_: _freq_[min(int((_c_ - _lo_) / _span_ * 10), 9)] += 1
        self.assertEqual(len(_heights_), 10)
        self.assertEqual([round(hh / max(_heights_), 3) for hh in _heights_], [round(f / max(_freq_), 3) for f in _freq_])
        self.assertNotIn('fill-opacity="0.4"', self.p2s.histop(self.df, 'cat', wxh=(256, 600))._repr_svg_())

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
        self.assertEqual(self.p2s.histop(self.df, 'cat', wxh=(256, 128)).wxh, (256, 128))


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
