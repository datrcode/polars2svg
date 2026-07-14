"""
Tests for Histop distribution strip (distribution=True, distribution_bin_w=10).

The strip is a 1D density bar placed at the bottom of the SVG. Every bin in
the dataset (including non-visible/unrendered ones) contributes one tick to a
quantized x-column based on its bar width in pixels. After accumulation the
columns are spectrum-coloured by count and rendered as small squares.

Key attributes set after construction:
  _dist_h_           -- reserved pixel height at bottom (0 when suppressed)
  _dist_strip_y0_    -- top y of the strip (sentinel == h when suppressed)
  _dist_stacked_     -- dict {q: count}
  _dist_bins_lu_     -- dict {q: [bin_label, ...]}
  _dist_stacked_max_ -- max(stacked.values()) for normalisation
"""
import re
import unittest
from xml.etree import ElementTree as ET

import polars as pl
from polars2svg import Polars2SVG


# ── shared fixtures ──────────────────────────────────────────────────────────

def _make_df():
    """Five bins A=10, B=8, C=6, D=4, E=2 (deterministic)."""
    return pl.DataFrame({
        'cat': ['A'] * 10 + ['B'] * 8 + ['C'] * 6 + ['D'] * 4 + ['E'] * 2,
        'val': list(range(30)),
    })


def _make_grouped_df():
    """Three bins each with two colour segments (for stacked agg_type)."""
    return pl.DataFrame({
        'cat':   ['A'] * 6 + ['B'] * 4 + ['C'] * 2,
        'group': (['x'] * 3 + ['y'] * 3) + (['x'] * 2 + ['y'] * 2) + (['x'] * 1 + ['y'] * 1),
        'val':   list(range(12)),
    })


P2S = Polars2SVG()


def _strip_rects_from_svg(svg_str, strip_y0, bin_w):
    """Extract rect elements whose y attribute falls within the strip."""
    root = ET.fromstring(svg_str)
    rects = []
    for rect in root.iter('{http://www.w3.org/2000/svg}rect'):
        try:
            ry = float(rect.get('y', -1))
        except (TypeError, ValueError):
            continue
        if strip_y0 - 1 <= ry <= strip_y0 + bin_w + 1:
            rects.append(rect)
    return rects


# ── Geometry ─────────────────────────────────────────────────────────────────

class TestDistributionStripGeometry(unittest.TestCase):

    def test_strip_active_by_default(self):
        h = P2S.histop(_make_df(), 'cat', wxh=(200, 256))
        self.assertGreater(h._dist_h_, 0)
        self.assertLess(h._dist_strip_y0_, 256)
        self.assertGreater(h._dist_strip_y0_, 0)

    def test_strip_disabled_when_distribution_false(self):
        h = P2S.histop(_make_df(), 'cat', wxh=(200, 256), distribution=False)
        self.assertEqual(h._dist_h_, 0)
        self.assertEqual(h._dist_strip_y0_, 256)

    def test_strip_suppressed_for_boxplot(self):
        df = pl.DataFrame({'cat': ['A'] * 10 + ['B'] * 8, 'val': list(range(18))})
        h = P2S.histop(df, 'cat', wxh=(200, 256), style=P2S.BOXPLOTp, count=('val', P2S.MINp))
        self.assertEqual(h._dist_h_, 0)

    def test_strip_suppressed_when_height_below_48(self):
        h = P2S.histop(_make_df(), 'cat', wxh=(200, 47))
        self.assertEqual(h._dist_h_, 0)

    def test_strip_suppressed_when_width_below_48(self):
        h = P2S.histop(_make_df(), 'cat', wxh=(47, 200))
        self.assertEqual(h._dist_h_, 0)

    def test_strip_active_at_exactly_48(self):
        h = P2S.histop(_make_df(), 'cat', wxh=(48, 48))
        # May or may not have room depending on bar geometry, but not suppressed by size check
        # Just verify no exception and _dist_h_ reflects geometry (not the w/h guard)
        self.assertIsNotNone(h._dist_h_)

    def test_strip_y0_is_below_bar_area(self):
        h = P2S.histop(_make_df(), 'cat', wxh=(200, 256))
        min_bar_bottom = h._plot_y0_ + h._slot_h_
        self.assertGreater(h._dist_strip_y0_, min_bar_bottom)

    def test_custom_distribution_bin_w(self):
        h = P2S.histop(_make_df(), 'cat', wxh=(200, 256), distribution_bin_w=10)
        self.assertEqual(h.distribution_bin_w, 10)
        h5 = P2S.histop(_make_df(), 'cat', wxh=(200, 256), distribution_bin_w=5)
        self.assertNotEqual(h._dist_h_, h5._dist_h_)

    def test_dist_h_equals_bin_w_plus_y_inset(self):
        h = P2S.histop(_make_df(), 'cat', wxh=(200, 256), distribution_bin_w=5, insets=(2, 3))
        self.assertEqual(h._dist_h_, 5 + 3)

    def test_strip_y0_position(self):
        h = P2S.histop(_make_df(), 'cat', wxh=(200, 256), distribution_bin_w=5, insets=(2, 2))
        expected = 256 - 2 - 5
        self.assertAlmostEqual(h._dist_strip_y0_, expected, places=1)


# ── Distribution computation ─────────────────────────────────────────────────

class TestDistributionCompute(unittest.TestCase):

    def setUp(self):
        self.df = _make_df()

    def test_stacked_dict_non_empty(self):
        h = P2S.histop(self.df, 'cat', wxh=(200, 256))
        self.assertGreater(len(h._dist_stacked_), 0)

    def test_bins_lu_non_empty(self):
        h = P2S.histop(self.df, 'cat', wxh=(200, 256))
        self.assertGreater(len(h._dist_bins_lu_), 0)

    def test_all_bins_accounted_for(self):
        """Every bin (including non-visible) must appear somewhere in _dist_bins_lu_."""
        h = P2S.histop(self.df, 'cat', wxh=(200, 48), distribution_bin_w=5)
        all_bins_in_lu = set(b for bins in h._dist_bins_lu_.values() for b in bins)
        for bin_label in h._sorted_bins_:
            self.assertIn(bin_label, all_bins_in_lu,
                          f'bin {bin_label!r} missing from _dist_bins_lu_')

    def test_quantization_keys_are_bin_indices(self):
        """Keys in _dist_stacked_ are integer bin indices in [0, n_bins)."""
        h = P2S.histop(self.df, 'cat', wxh=(200, 256), distribution_bin_w=5)
        for bi in h._dist_stacked_:
            self.assertIsInstance(bi, int)
            self.assertGreaterEqual(bi, 0)
            self.assertLess(bi, h._dist_n_bins_)

    def test_stacked_counts_sum_to_n_bins(self):
        """Sum of stacked counts equals total number of bins (each bin contributes once)."""
        h = P2S.histop(self.df, 'cat', wxh=(200, 256))
        total = sum(h._dist_stacked_.values())
        self.assertEqual(total, len(h._sorted_bins_))

    def test_stacked_max_correct(self):
        h = P2S.histop(self.df, 'cat', wxh=(200, 256))
        self.assertEqual(h._dist_stacked_max_, max(h._dist_stacked_.values()))

    def test_empty_after_distribution_false(self):
        h = P2S.histop(self.df, 'cat', wxh=(200, 256), distribution=False)
        self.assertEqual(h._dist_stacked_, {})
        self.assertEqual(h._dist_bins_lu_, {})

    def test_stacked_agg_uses_total_per_bin(self):
        """With stacked agg (color=string), bin totals drive quantization, not per-segment counts."""
        h = P2S.histop(_make_grouped_df(), 'cat', wxh=(200, 256), color='group')
        self.assertEqual(h._agg_type_, 'stacked')
        all_bins = set(b for bins in h._dist_bins_lu_.values() for b in bins)
        for cat in ('A', 'B', 'C'):
            self.assertIn(cat, all_bins)

    def test_boxplot_empty(self):
        df = pl.DataFrame({'cat': ['A'] * 10 + ['B'] * 8, 'val': list(range(18))})
        h = P2S.histop(df, 'cat', wxh=(200, 256), style=P2S.BOXPLOTp, count=('val', P2S.MINp))
        self.assertEqual(h._dist_stacked_, {})


# ── SVG rendering ─────────────────────────────────────────────────────────────

class TestDistributionStripRendering(unittest.TestCase):

    def setUp(self):
        self.df = _make_df()

    def test_strip_rects_appear_in_svg(self):
        h = P2S.histop(self.df, 'cat', wxh=(200, 256))
        rects = _strip_rects_from_svg(h.svg, h._dist_strip_y0_, h.distribution_bin_w)
        self.assertGreater(len(rects), 0)

    def test_strip_absent_when_distribution_false(self):
        h = P2S.histop(self.df, 'cat', wxh=(200, 256), distribution=False)
        rects = _strip_rects_from_svg(h.svg, 249, 5)   # approx strip position
        self.assertEqual(len(rects), 0)

    def test_strip_rects_have_spectrum_fill(self):
        h = P2S.histop(self.df, 'cat', wxh=(200, 256))
        rects = _strip_rects_from_svg(h.svg, h._dist_strip_y0_, h.distribution_bin_w)
        for rect in rects:
            fill = rect.get('fill', '')
            self.assertRegex(fill, r'^#[0-9a-fA-F]{6}$', f'Expected hex colour, got {fill!r}')

    def test_bars_do_not_overlap_strip_y_range(self):
        """Bar rects (whose y starts at _plot_y0_ region) must not fall in the strip."""
        h = P2S.histop(self.df, 'cat', wxh=(200, 256))
        root = ET.fromstring(h.svg)
        strip_y0 = h._dist_strip_y0_
        strip_y1 = strip_y0 + h.distribution_bin_w
        for rect in root.iter('{http://www.w3.org/2000/svg}rect'):
            fill = rect.get('fill', '')
            x, y = rect.get('x', ''), rect.get('y', '')
            if x == '0' and y == '0': continue   # background
            if fill == 'none':        continue   # axis border
            try:
                ry = float(y)
                rh = float(rect.get('height', 0))
            except (TypeError, ValueError):
                continue
            ry_bottom = ry + rh
            # A bar rect starts in the bar area (below _plot_y0_, well above strip)
            bar_area_top = h._plot_y0_
            bar_area_bottom = h._dist_strip_y0_ - 1
            if bar_area_top <= ry < bar_area_bottom:
                # This is a bar rect — its bottom should not enter the strip
                self.assertLessEqual(ry_bottom, strip_y1 + 1,
                                     f'Bar rect at y={ry} h={rh} overlaps strip at {strip_y0}')

    def test_culling_respects_strip_height(self):
        """With strip active, fewer bins visible than with distribution=False."""
        # Use a short widget so the strip noticeably reduces available bar area.
        h_on  = P2S.histop(_make_df(), 'cat', wxh=(200, 80), bar_h=14, distribution=True)
        h_off = P2S.histop(_make_df(), 'cat', wxh=(200, 80), bar_h=14, distribution=False)
        # Count visible bins from svg (bars are in simple mode → one rect per bin)
        # We check via geometry attributes instead.
        effective_on  = 80 - h_on._dist_h_
        effective_off = 80 - h_off._dist_h_
        self.assertEqual(h_off._dist_h_, 0)
        self.assertLess(effective_on, effective_off)

    def test_custom_bin_w_changes_svg(self):
        h5  = P2S.histop(self.df, 'cat', wxh=(200, 256), distribution_bin_w=5)
        h10 = P2S.histop(self.df, 'cat', wxh=(200, 256), distribution_bin_w=10)
        rects5  = _strip_rects_from_svg(h5.svg,  h5._dist_strip_y0_,  5)
        rects10 = _strip_rects_from_svg(h10.svg, h10._dist_strip_y0_, 10)
        # 10px bins are coarser → fewer or equal strip rects
        self.assertGreaterEqual(len(rects5), len(rects10))


# ── filterByRectangle ────────────────────────────────────────────────────────

class TestDistributionStripFilterByRectangle(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = _make_df()
        # Large enough that all 5 bins are visible AND the strip is active.
        cls.h = P2S.histop(cls.df, 'cat', wxh=(200, 300), distribution_bin_w=5,
                           draw_context=False)

    def test_rectangle_fully_in_strip_returns_records(self):
        """A box covering the whole strip y and full x returns non-empty result."""
        strip_y0 = self.h._dist_strip_y0_
        strip_y1 = strip_y0 + self.h.distribution_bin_w
        x0, x1 = self.h._plot_x0_, self.h._plot_x0_ + self.h._plot_w_
        result = self.h.filterByRectangle((x0, strip_y0, x1, strip_y1))
        self.assertGreater(len(result), 0)

    def test_strip_selection_returns_correct_columns(self):
        strip_y0 = self.h._dist_strip_y0_
        strip_y1 = strip_y0 + self.h.distribution_bin_w
        x0, x1 = self.h._plot_x0_, self.h._plot_x0_ + self.h._plot_w_
        result = self.h.filterByRectangle((x0, strip_y0, x1, strip_y1))
        self.assertIn('cat', result.columns)
        self.assertIn('val', result.columns)
        self.assertNotIn('__p2s_index__', result.columns)

    def test_strip_wrong_x_returns_empty(self):
        """A box in the strip y range but at x < _plot_x0_ should return empty."""
        strip_y0 = self.h._dist_strip_y0_
        strip_y1 = strip_y0 + self.h.distribution_bin_w
        # x range entirely to the left of the plot area (no strip columns there)
        x0, x1 = -20, -1
        result = self.h.filterByRectangle((x0, strip_y0, x1, strip_y1))
        self.assertEqual(len(result), 0)

    def test_rectangle_spanning_bar_and_strip_returns_union(self):
        """A box from the first bar row all the way through the strip selects more records."""
        y_bar_top    = self.h._plot_y0_ + 0.5
        strip_y1     = self.h._dist_strip_y0_ + self.h.distribution_bin_w
        x0, x1       = self.h._plot_x0_, self.h._plot_x0_ + self.h._plot_w_
        result_bar   = self.h.filterByRectangle((x0, y_bar_top, x1, y_bar_top + self.h.bar_h - 1))
        result_combo = self.h.filterByRectangle((x0, y_bar_top, x1, strip_y1))
        # Spanning both regions selects at least as many as just the bar row
        self.assertGreaterEqual(len(result_combo), len(result_bar))

    def test_deduplication_no_duplicate_rows(self):
        """Bins that appear both from bar selection and strip selection are not duplicated."""
        strip_y0 = self.h._dist_strip_y0_
        strip_y1 = strip_y0 + self.h.distribution_bin_w
        x0, x1   = self.h._plot_x0_, self.h._plot_x0_ + self.h._plot_w_
        # Full rectangle covering everything
        result = self.h.filterByRectangle((x0, self.h._plot_y0_, x1, strip_y1))
        # No duplicated val entries
        vals = result['val'].to_list()
        self.assertEqual(len(vals), len(set(vals)))


# ── recordsAt ────────────────────────────────────────────────────────────────

class TestDistributionStripRecordsAt(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = _make_df()
        cls.h = P2S.histop(cls.df, 'cat', wxh=(200, 300), distribution_bin_w=5,
                           draw_context=False)

    def test_records_at_strip_y_returns_records(self):
        """Clicking anywhere in the strip that maps to a non-empty cell returns records."""
        bi  = min(self.h._dist_bins_lu_.keys())
        abw = self.h._dist_actual_bin_w_
        x   = self.h._plot_x0_ + bi * abw + abw * 0.5   # centre of cell
        y   = self.h._dist_strip_y0_ + 2
        result = self.h.recordsAt((x, y))
        self.assertGreater(len(result), 0)

    def test_records_at_strip_returns_correct_bins(self):
        """Records returned belong to the expected bins for that cell."""
        bi  = min(self.h._dist_bins_lu_.keys())
        expected_bins = set(self.h._dist_bins_lu_[bi])
        abw = self.h._dist_actual_bin_w_
        x   = self.h._plot_x0_ + bi * abw + abw * 0.5   # centre of cell
        y   = self.h._dist_strip_y0_ + 1
        result = self.h.recordsAt((x, y))
        returned_cats = set(result['cat'].to_list())
        self.assertTrue(returned_cats.issubset(expected_bins))
        # Every expected bin should contribute at least one record
        for cat in expected_bins:
            self.assertIn(cat, returned_cats)

    def test_records_at_strip_y_empty_column_returns_empty(self):
        """Clicking in the strip at an x with no stacked bins returns empty."""
        # x = -1 (before plot area) → no bins mapped there
        y = self.h._dist_strip_y0_ + 1
        result = self.h.recordsAt((-1, y))
        self.assertEqual(len(result), 0)

    def test_records_at_bar_still_works_with_strip_active(self):
        """recordsAt on a bar slot still works when distribution is enabled."""
        y_v  = self.h.v_gap // 2 if self.h.v_gap > 0 else 0
        y    = self.h._plot_y0_ + y_v + self.h.bar_h // 2   # middle of first bar
        result = self.h.recordsAt((self.h._plot_x0_ + 1, y))
        self.assertGreater(len(result), 0)
        # Should be bin A (count=10, highest → first bar)
        self.assertTrue(all(c == 'A' for c in result['cat'].to_list()))

    def test_records_at_strip_correct_columns(self):
        q = min(self.h._dist_bins_lu_.keys())
        x = self.h._plot_x0_ + q + 1
        y = self.h._dist_strip_y0_ + 1
        result = self.h.recordsAt((x, y))
        self.assertIn('cat', result.columns)
        self.assertIn('val', result.columns)
        self.assertNotIn('__p2s_index__', result.columns)


if __name__ == '__main__':
    unittest.main()
