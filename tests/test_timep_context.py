import re
import unittest
import polars as pl
from polars2svg import Polars2SVG
from timep_dataframes import makeTimeDf, TimepAssertions, timepColumns
from svg_test_utils import normalize_svg


class TestTimepContext(TimepAssertions, unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()
        self.df  = makeTimeDf(n=200, year=(2020, 2024), month=(1, 12), seed=9)

    # With context: a frame round the plot and axis labels, drawn at 0.75 and 0.8 of txt_h.
    # Without: no frame, no labels, no lines, and the plot takes the canvas less its insets.
    # Either way every column is its bin's row count.
    def assertContext(self, t, drawn: bool) -> None:
        _svg_ = t._repr_svg_()
        _plot_, _frame_, _ = timepColumns(t)
        if drawn:
            self.assertIsNotNone(_frame_, 'no frame round the plot')
            _sizes_ = {round(float(f) / t.txt_h, 3) for f in re.findall(r'font-size="([\d.]+)px"', _svg_)}
            self.assertTrue(_sizes_ and _sizes_ <= {0.75, 0.8}, f'axis labels at {_sizes_} of txt_h')
        else:
            self.assertIsNone(_frame_)
            self.assertNotIn('<text', _svg_)
            self.assertNotIn('<line', _svg_)
            _xi_, _yi_ = t.insets
            self.assertEqual(_plot_, (_xi_, _yi_, t.wxh[0] - 2 * _xi_, t.wxh[1] - 2 * _yi_))
        self.assertColumnsShow(t, self.df, 'ts', pl.len())

    # ── draw_context ──────────────────────────────────────────────────────────

    def test_draw_context_true(self):
        _t_ = self.p2s.timep(self.df, 'ts', draw_context=True)
        self.assertEqual(normalize_svg(_t_.svg), normalize_svg(self.p2s.timep(self.df, 'ts').svg), 'context is the default')
        self.assertContext(_t_, drawn=True)

    def test_draw_context_false(self):
        self.assertContext(self.p2s.timep(self.df, 'ts', draw_context=False), drawn=False)

    def test_draw_context_both_produce_valid_svg(self):
        t_ctx   = self.p2s.timep(self.df, 'ts', wxh=(256, 128), draw_context=True)
        t_noctx = self.p2s.timep(self.df, 'ts', wxh=(256, 128), draw_context=False)
        self.assertIn('<svg', t_ctx._repr_svg_())
        self.assertIn('<svg', t_noctx._repr_svg_())

    def test_draw_context_periodic(self):
        for _drawn_ in (True, False):
            with self.subTest(draw_context=_drawn_):
                self.assertContext(self.p2s.timep(self.df, ('ts', self.p2s.PT_mp), draw_context=_drawn_), drawn=_drawn_)

    # ── width sweep ───────────────────────────────────────────────────────────

    def test_width_sweep_linear(self):
        '''Auto-granularity adjusts across a range of widget widths -- and at every width,
        at whatever level it settles on, the columns are the rows.'''
        for w in range(64, 1024 + 64, 64):
            with self.subTest(w=w):
                _t_ = self.p2s.timep(self.df, 'ts', wxh=(w, 128))
                self.assertEqual(_t_.wxh, (w, 128))
                self.assertColumnsShow(_t_, self.df, 'ts', pl.len())

    def test_width_sweep_periodic(self):
        for w in range(64, 1024 + 64, 64):
            with self.subTest(w=w):
                self.assertColumnsShow(self.p2s.timep(self.df, ('ts', self.p2s.PT_mp), wxh=(w, 128)), self.df, 'ts', pl.len())

    def test_height_variation(self):
        # 32px is too short for context: it drops, and the columns take the height
        for h in [32, 64, 128, 256, 512]:
            with self.subTest(h=h):
                _t_ = self.p2s.timep(self.df, 'ts', wxh=(512, h))
                self.assertContext(_t_, drawn=_t_.draw_context)
                self.assertEqual(_t_.draw_context, h > 32)

    # ── insets ────────────────────────────────────────────────────────────────

    def test_nondefault_insets(self):
        # insets move the plot in from every edge by exactly that much
        _x0_, _y0_, _w0_, _h0_ = timepColumns(self.p2s.timep(self.df, 'ts', insets=(0, 0)))[1]
        for insets in [(0, 0), (1, 1), (5, 5), (10, 10)]:
            with self.subTest(insets=insets):
                _t_ = self.p2s.timep(self.df, 'ts', insets=insets)
                _xi_, _yi_ = insets
                self.assertEqual(timepColumns(_t_)[1], (_x0_ + _xi_, _y0_ + _yi_, _w0_ - 2 * _xi_, _h0_ - 2 * _yi_))
                self.assertColumnsShow(_t_, self.df, 'ts', pl.len())

    # ── txt_h ─────────────────────────────────────────────────────────────────

    def test_nondefault_txt_h(self):
        for txt_h in [8, 10, 12, 14, 16]:
            with self.subTest(txt_h=txt_h):
                self.assertContext(self.p2s.timep(self.df, 'ts', txt_h=txt_h), drawn=True)

    # ── min_bar_w ─────────────────────────────────────────────────────────────

    def test_min_bar_w_respected(self):
        '''_bar_w_ must not cause bars to overlap; min_bar_w is honoured when slot allows.'''
        df = makeTimeDf(n=500, year=(2020, 2024), month=(1, 12), day=(1, 28))
        for min_bar_w in [0.5, 1.0, 3.0, 5.0, 10.0]:
            t = self.p2s.timep(df, 'ts', min_bar_w=min_bar_w, wxh=(256, 128))
            self.assertLessEqual(t._bar_w_, t._bar_w_raw_,
                                 msg=f"bar_w exceeds slot (overlap) with min_bar_w={min_bar_w}")
            self.assertGreaterEqual(t._bar_w_, min(min_bar_w, t._bar_w_raw_),
                                    msg=f"bar_w below expected floor with min_bar_w={min_bar_w}")

    # ── lazy vs eager execution ───────────────────────────────────────────────

    def test_lazy_and_eager_both_render(self):
        '''Both lazy and eager execution paths produce valid SVGs.'''
        t_lazy  = self.p2s.timep(self.df, 'ts', use_lazy_execution=True,  wxh=(256, 128))
        t_eager = self.p2s.timep(self.df, 'ts', use_lazy_execution=False, wxh=(256, 128))
        self.assertIn('<svg', t_lazy._repr_svg_())
        self.assertIn('<svg', t_eager._repr_svg_())

    def test_lazy_and_eager_periodic(self):
        t_lazy  = self.p2s.timep(self.df, ('ts', self.p2s.PT_mp),
                                 use_lazy_execution=True,  wxh=(256, 128))
        t_eager = self.p2s.timep(self.df, ('ts', self.p2s.PT_mp),
                                 use_lazy_execution=False, wxh=(256, 128))
        self.assertIn('<svg', t_lazy._repr_svg_())
        self.assertIn('<svg', t_eager._repr_svg_())

    # ── small size ────────────────────────────────────────────────────────────

    def test_small_width_disables_context(self):
        t = self.p2s.timep(self.df, 'ts', wxh=(32, 128))
        self.assertFalse(t.draw_context)

    def test_small_height_disables_context(self):
        t = self.p2s.timep(self.df, 'ts', wxh=(512, 32))
        self.assertFalse(t.draw_context)

    def test_small_insets_zeroized(self):
        t = self.p2s.timep(self.df, 'ts', wxh=(20, 20), insets=(10, 10))
        self.assertEqual(t.insets, (0, 0))

    def test_tiny_width_sweep_renders(self):
        for w in [8, 16, 32, 48, 56]:
            t = self.p2s.timep(self.df, 'ts', wxh=(w, 128))
            self.assertIn('<svg', t._repr_svg_())

    def test_label_trimming_narrow(self):
        self.assertIn('<svg', self.p2s.timep(self.df, 'ts', wxh=(60,  128))._repr_svg_())
        self.assertIn('<svg', self.p2s.timep(self.df, 'ts', wxh=(8,   128))._repr_svg_())

    def test_bottom_label_space_reclaimed_when_no_labels_fit(self):
        '''When no bottom labels render the height space must be reclaimed for the plot.'''
        h = 128
        t_wide   = self.p2s.timep(self.df, 'ts', wxh=(512, h))   # labels definitely render
        t_narrow = self.p2s.timep(self.df, 'ts', wxh=(64,  h))   # labels may not render
        # If narrow omits bottom labels, _plot_h_ must be >= wide _plot_h_ (same total h)
        if t_narrow._plot_h_ > t_wide._plot_h_:
            pass  # space was reclaimed — correct
        else:
            # Both may render labels at 64px; ensure at minimum no space is wasted
            # (plot height must use all available pixels minus insets)
            x_ins, y_ins = t_narrow.insets
            _axis_h_ = t_narrow.txt_h + 4 if t_narrow.draw_context else 0
            expected_h = h - 2 * y_ins - _axis_h_
            self.assertEqual(t_narrow._plot_h_, expected_h)

    # ── geometry sanity ───────────────────────────────────────────────────────

    def test_plot_region_within_widget_bounds(self):
        '''Plot region x0/y0 and derived sizes must be non-negative and fit within wxh.'''
        w, h = 512, 256
        t    = self.p2s.timep(self.df, 'ts', wxh=(w, h))
        self.assertGreaterEqual(t._plot_x0_, 0)
        self.assertGreaterEqual(t._plot_y0_, 0)
        self.assertGreater(t._plot_w_, 0)
        self.assertGreater(t._plot_h_, 0)
        self.assertLessEqual(t._plot_x0_ + t._plot_w_, w)
        self.assertLessEqual(t._plot_y0_ + t._plot_h_, h)

    def test_n_bins_positive(self):
        t = self.p2s.timep(self.df, 'ts')
        self.assertGreater(t._n_bins_, 0)


if __name__ == '__main__':
    unittest.main()
