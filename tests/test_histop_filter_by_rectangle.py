"""
Tests for Histop.filterByRectangle(bounding_box, remove_records=False).

Strategy
--------
Histop renders bins as HORIZONTAL bars (x = count, y = bin position).  Bins
are sorted by count (descending by default) and only as many bars as fit
within the SVG height are actually drawn — the rest are "unrendered".

Key geometry (all fields are public after construction):
  _plot_x0_   left edge of the plot area
  _plot_y0_   top edge of the plot area
  _plot_w_    width of the plot area (= full bar width at count_max)
  _slot_h_    = bar_h + v_gap (height of one bin's vertical slot)
  bar_h       rendered bar height in pixels
  _sorted_bins_  ordered list of bin labels (display order, top to bottom)

Bar i (0-based in _sorted_bins_) occupies:
  x: [_plot_x0_, _plot_x0_ + countToBarW(count_i)]
  y: [_plot_y0_ + i*_slot_h_ + y_v,
      _plot_y0_ + i*_slot_h_ + y_v + bar_h]

Helper _bar_bbox(histop, display_index) computes the full bounding box of bar
`display_index`, using the full plot width so that any non-zero bar is hit.

Critical behaviour for remove_records=True
-------------------------------------------
When `remove_records=True`, the rectangle identifies visible bars to exclude.
Bins beyond the rendered range (_sorted_bins_[_n_visible_:]) exist in self.df
but were NEVER drawn.  These "unrendered" records must NOT be removed — only
the selected visible bars are excluded.
"""
import unittest
import polars as pl
from polars2svg import Polars2SVG


def _n_visible(histop):
    """Recompute the number of visible bins, mirroring __renderSVG__ culling."""
    _, h          = histop.wxh
    effective_h   = h - histop._dist_h_
    y_v           = histop.v_gap // 2 if histop.v_gap > 0 else 0
    n             = 0
    for i in range(len(histop._sorted_bins_)):
        if histop._plot_y0_ + i * histop._slot_h_ + y_v + histop.bar_h <= effective_h:
            n = i + 1
        else:
            break
    return n


_INSET = 0.5   # half-pixel inset keeps bboxes from touching adjacent bar edges


def _bar_bbox(histop, display_index):
    """Full bounding box (x0,y0,x1,y1) for the bar at display_index.

    A half-pixel inset on the y edges prevents the inclusive overlap test from
    accidentally selecting the adjacent bar when bars share an exact boundary
    (i.e. when v_gap=0 and bars are packed tightly).
    """
    y_v  = histop.v_gap // 2 if histop.v_gap > 0 else 0
    x0   = histop._plot_x0_
    x1   = histop._plot_x0_ + histop._plot_w_    # full width — hits any bar
    y0   = histop._plot_y0_ + display_index * histop._slot_h_ + y_v + _INSET
    y1   = y0 + histop.bar_h - 2 * _INSET
    return (x0, y0, x1, y1)


class TestHistopFilterByRectangleSimple(unittest.TestCase):
    """Simple (non-stacked) bar chart with a single bin_by field."""

    @classmethod
    def setUpClass(cls):
        cls.p2s = Polars2SVG()
        # Deterministic counts: A=10, B=6, C=3.
        # Descending sort → display order: A (idx 0), B (idx 1), C (idx 2).
        cls.df = pl.DataFrame({
            'cat': ['A'] * 10 + ['B'] * 6 + ['C'] * 3,
            'val': list(range(19)),
        })
        # Tall enough to show all 3 bins.
        cls.histop = cls.p2s.histop(
            cls.df, 'cat', wxh=(256, 512), bar_h=16, v_gap=0,
            draw_context=False
        )

    # ── basic selection ───────────────────────────────────────────────────────

    def test_select_first_bin_returns_A_records(self):
        """Box around the top bar (A, 10 rows) returns exactly those 10 records."""
        result = self.histop.filterByRectangle(_bar_bbox(self.histop, 0))
        self.assertEqual(len(result), 10)
        self.assertTrue(all(v == 'A' for v in result['cat'].to_list()))

    def test_select_second_bin_returns_B_records(self):
        result = self.histop.filterByRectangle(_bar_bbox(self.histop, 1))
        self.assertEqual(len(result), 6)
        self.assertTrue(all(v == 'B' for v in result['cat'].to_list()))

    def test_select_third_bin_returns_C_records(self):
        result = self.histop.filterByRectangle(_bar_bbox(self.histop, 2))
        self.assertEqual(len(result), 3)
        self.assertTrue(all(v == 'C' for v in result['cat'].to_list()))

    def test_box_spanning_all_bars_returns_full_df(self):
        """A box covering the whole SVG returns all records."""
        w, h = self.histop.wxh
        result = self.histop.filterByRectangle((0, 0, w, h))
        self.assertEqual(len(result), len(self.df))

    def test_box_right_of_shortest_bar_misses_C(self):
        """A very narrow box that only starts past C's bar width misses C.
        C has 3 rows; its bar is 30% of the plot width (3/10).
        A box starting at 40% of the plot width should miss C's bar entirely."""
        x_start = self.histop._plot_x0_ + self.histop._plot_w_ * 0.40
        x_end   = self.histop._plot_x0_ + self.histop._plot_w_
        y0, y1  = self.histop._plot_y0_, self.histop._plot_y0_ + self.histop.bar_h * 3
        result  = self.histop.filterByRectangle((x_start, y0, x_end, y1))
        self.assertNotIn('C', result['cat'].to_list())

    def test_box_outside_plot_returns_empty(self):
        """A box above the plot area (negative y) returns nothing."""
        result = self.histop.filterByRectangle((-10, -10, -1, -1))
        self.assertEqual(len(result), 0)

    # ── remove_records ────────────────────────────────────────────────────────

    def test_remove_records_false_is_default(self):
        bbox = _bar_bbox(self.histop, 0)
        r1 = self.histop.filterByRectangle(bbox)
        r2 = self.histop.filterByRectangle(bbox, remove_records=False)
        self.assertEqual(sorted(r1['val'].to_list()),
                         sorted(r2['val'].to_list()))

    def test_remove_records_true_excludes_selected_bin(self):
        """remove_records=True on bar A removes A, keeps B and C."""
        result = self.histop.filterByRectangle(
            _bar_bbox(self.histop, 0), remove_records=True
        )
        cats = result['cat'].to_list()
        self.assertNotIn('A', cats)
        self.assertIn('B', cats)
        self.assertIn('C', cats)
        self.assertEqual(len(result), 9)   # 6 + 3

    def test_inside_plus_outside_covers_full_df(self):
        """Selected + complement reconstruct the complete DataFrame."""
        bbox    = _bar_bbox(self.histop, 1)
        inside  = self.histop.filterByRectangle(bbox)
        outside = self.histop.filterByRectangle(bbox, remove_records=True)
        combined = sorted(inside['val'].to_list() + outside['val'].to_list())
        self.assertEqual(combined, sorted(self.df['val'].to_list()))

    # ── bounding box normalisation ─────────────────────────────────────────────

    def test_inverted_bbox_normalised(self):
        """x0>x1 and y0>y1 are swapped before filtering."""
        x0, y0, x1, y1 = _bar_bbox(self.histop, 0)
        normal   = self.histop.filterByRectangle((x0, y0, x1, y1))
        inverted = self.histop.filterByRectangle((x1, y1, x0, y0))
        self.assertEqual(sorted(normal['val'].to_list()),
                         sorted(inverted['val'].to_list()))

    # ── result schema ─────────────────────────────────────────────────────────

    def test_result_has_original_columns(self):
        w, h = self.histop.wxh
        result = self.histop.filterByRectangle((0, 0, w, h))
        self.assertIn('cat', result.columns)
        self.assertIn('val', result.columns)

    def test_result_drops_p2s_index(self):
        w, h = self.histop.wxh
        result = self.histop.filterByRectangle((0, 0, w, h))
        self.assertNotIn('__p2s_index__', result.columns)


class TestHistopFilterByRectangleUnrenderedRecords(unittest.TestCase):
    """
    The critical remove_records=True invariant: records belonging to bins
    that were never rendered (they fell outside the SVG height) must survive
    the anti-join regardless of the selected rectangle.
    """

    @classmethod
    def setUpClass(cls):
        cls.p2s = Polars2SVG()
        # Five bins with counts 10, 8, 6, 4, 2.
        # Descending sort → display order: A, B, C, D, E.
        cls.df = pl.DataFrame({
            'cat': ['A']*10 + ['B']*8 + ['C']*6 + ['D']*4 + ['E']*2,
            'val': list(range(30)),
        })
        # bar_h=20, v_gap=0, draw_context=False → plot_y0 = insets[1] = 2.
        # Slot 0: y [2, 22], slot 1: [22, 42], slot 2: [42, 62].
        # With h=65: slots 0-2 fit (y1=62 ≤ 65), slot 3 would need y1=82 > 65.
        # → 3 visible bins (A, B, C); D and E are unrendered.
        cls.histop = cls.p2s.histop(
            cls.df, 'cat',
            wxh=(256, 65), bar_h=20, v_gap=0,
            insets=(2, 2), draw_context=False, distribution=False
        )

    def test_setup_sanity_three_visible_bins(self):
        """Verify the fixture actually produces exactly 3 visible bins."""
        self.assertEqual(_n_visible(self.histop), 3)
        self.assertEqual(len(self.histop._sorted_bins_), 5)

    def test_unrendered_bins_present_in_df(self):
        """D and E must exist in self.df even though their bars weren't drawn."""
        all_cats = set(self.histop.df['cat'].to_list())
        self.assertIn('D', all_cats)
        self.assertIn('E', all_cats)

    def test_remove_records_keeps_unrendered_D_and_E(self):
        """Selecting visible bar A and removing it must NOT remove D or E records."""
        result = self.histop.filterByRectangle(
            _bar_bbox(self.histop, 0), remove_records=True   # remove A
        )
        cats = result['cat'].to_list()
        self.assertNotIn('A', cats)          # A was selected → removed
        self.assertIn('D', cats)             # D unrendered → preserved
        self.assertIn('E', cats)             # E unrendered → preserved
        # Total: B(8) + C(6) + D(4) + E(2) = 20
        self.assertEqual(len(result), 20)

    def test_remove_all_visible_keeps_only_unrendered(self):
        """Selecting all three visible bars returns only D and E records."""
        # Box covering the full y range of all visible bars, inset by 0.5px at
        # the bottom so the first unrendered bin (D) is not accidentally touched.
        y_top    = self.histop._plot_y0_ + _INSET
        y_bottom = self.histop._plot_y0_ + 3 * self.histop._slot_h_ - _INSET
        x0       = self.histop._plot_x0_
        x1       = self.histop._plot_x0_ + self.histop._plot_w_
        result   = self.histop.filterByRectangle(
            (x0, y_top, x1, y_bottom), remove_records=True
        )
        cats = set(result['cat'].to_list())
        self.assertNotIn('A', cats)
        self.assertNotIn('B', cats)
        self.assertNotIn('C', cats)
        self.assertIn('D', cats)
        self.assertIn('E', cats)
        self.assertEqual(len(result), 6)   # D(4) + E(2)

    def test_remove_one_visible_preserves_other_visible_too(self):
        """remove_records on A preserves B and C (visible-but-not-selected) as well."""
        result = self.histop.filterByRectangle(
            _bar_bbox(self.histop, 0), remove_records=True   # remove A
        )
        cats = result['cat'].to_list()
        self.assertIn('B', cats)
        self.assertIn('C', cats)

    def test_no_remove_on_box_missing_all_bars_returns_everything_with_remove(self):
        """A box that misses all bars with remove_records=True returns all rows."""
        result = self.histop.filterByRectangle(
            (-10, -10, -1, -1), remove_records=True
        )
        self.assertEqual(len(result), len(self.df))


class TestHistopFilterByRectangleStacked(unittest.TestCase):
    """Stacked bar chart — categorical color field."""

    @classmethod
    def setUpClass(cls):
        cls.p2s = Polars2SVG()
        # Two bins: A has red×4 + blue×2, B has red×1 + blue×3.
        cls.df = pl.DataFrame({
            'cat':   ['A']*6  + ['B']*4,
            'color': ['red','red','red','red','blue','blue',
                      'red','blue','blue','blue'],
            'val':   list(range(10)),
        })
        cls.histop = cls.p2s.histop(
            cls.df, 'cat', color='color',
            style=cls.p2s.STACKEDBARp, wxh=(256, 512),
            bar_h=16, v_gap=0, draw_context=False
        )

    def test_select_first_bin_returns_all_colors_for_that_bin(self):
        """Selecting bin A's bar returns both red and blue records for A."""
        result = self.histop.filterByRectangle(_bar_bbox(self.histop, 0))
        self.assertEqual(len(result), 6)
        colors = set(result['color'].to_list())
        self.assertIn('red',  colors)
        self.assertIn('blue', colors)
        cats = result['cat'].to_list()
        self.assertTrue(all(c == 'A' for c in cats))

    def test_stacked_remove_records_excludes_bin(self):
        """remove_records=True on A removes all 6 A records."""
        result = self.histop.filterByRectangle(
            _bar_bbox(self.histop, 0), remove_records=True
        )
        self.assertEqual(len(result), 4)
        self.assertNotIn('A', result['cat'].to_list())

    def test_stacked_inside_plus_outside_covers_full_df(self):
        bbox    = _bar_bbox(self.histop, 0)
        inside  = self.histop.filterByRectangle(bbox)
        outside = self.histop.filterByRectangle(bbox, remove_records=True)
        combined = sorted(inside['val'].to_list() + outside['val'].to_list())
        self.assertEqual(combined, sorted(self.df['val'].to_list()))


class TestHistopFilterByRectangleMultiFieldBin(unittest.TestCase):
    """bin_by as a tuple: Histop creates an internal __bin__ column."""

    @classmethod
    def setUpClass(cls):
        cls.p2s = Polars2SVG()
        cls.df = pl.DataFrame({
            'cat':   ['A', 'A', 'B', 'B', 'B'],
            'group': ['x', 'y', 'x', 'x', 'y'],
            'val':   [1,   2,   3,   4,   5  ],
        })
        cls.histop = cls.p2s.histop(
            cls.df, ('cat', 'group'), wxh=(256, 512),
            bar_h=16, v_gap=0, draw_context=False
        )

    def test_result_does_not_contain_internal_bin_column(self):
        """The synthetic __bin__ column must be dropped from the result."""
        w, h = self.histop.wxh
        result = self.histop.filterByRectangle((0, 0, w, h))
        self.assertNotIn('__bin__', result.columns)

    def test_result_contains_original_columns(self):
        w, h = self.histop.wxh
        result = self.histop.filterByRectangle((0, 0, w, h))
        self.assertIn('cat',   result.columns)
        self.assertIn('group', result.columns)
        self.assertIn('val',   result.columns)

    def test_selection_returns_records_from_selected_bin(self):
        """Selecting the top bar returns records from the most frequent bin."""
        result = self.histop.filterByRectangle(_bar_bbox(self.histop, 0))
        # Top bin by row count is 'B|x' (2 records) or similar; just verify
        # we get a non-empty DataFrame with original columns intact.
        self.assertGreater(len(result), 0)
        self.assertIn('cat', result.columns)


def _bar_slot_y(histop, display_index):
    """Y pixel at the vertical centre of the slot at display_index."""
    y_v = histop.v_gap // 2 if histop.v_gap > 0 else 0
    return histop._plot_y0_ + display_index * histop._slot_h_ + y_v + histop.bar_h / 2


class TestHistopRecordsAtSimple(unittest.TestCase):
    """recordsAt() on a simple (non-stacked) bar chart."""

    @classmethod
    def setUpClass(cls):
        cls.p2s = Polars2SVG()
        # Counts: A=10, B=6, C=3 → descending display order: A(0), B(1), C(2)
        cls.df = pl.DataFrame({
            'cat': ['A'] * 10 + ['B'] * 6 + ['C'] * 3,
            'val': list(range(19)),
        })
        cls.histop = cls.p2s.histop(
            cls.df, 'cat', wxh=(256, 512), bar_h=16, v_gap=0,
            draw_context=False
        )

    # ── basic hit-testing ─────────────────────────────────────────────────────

    def test_y_in_first_slot_returns_A_records(self):
        """y inside the first bar slot returns all 10 A records."""
        result = self.histop.recordsAt((0, _bar_slot_y(self.histop, 0)))
        self.assertEqual(len(result), 10)
        self.assertTrue(all(c == 'A' for c in result['cat'].to_list()))

    def test_y_in_second_slot_returns_B_records(self):
        result = self.histop.recordsAt((0, _bar_slot_y(self.histop, 1)))
        self.assertEqual(len(result), 6)
        self.assertTrue(all(c == 'B' for c in result['cat'].to_list()))

    def test_y_in_third_slot_returns_C_records(self):
        result = self.histop.recordsAt((0, _bar_slot_y(self.histop, 2)))
        self.assertEqual(len(result), 3)
        self.assertTrue(all(c == 'C' for c in result['cat'].to_list()))

    def test_x_coordinate_is_ignored(self):
        """The x value has no effect — any x selects the same bin row."""
        y = _bar_slot_y(self.histop, 0)
        r_left  = self.histop.recordsAt((0,    y))
        r_right = self.histop.recordsAt((9999, y))
        self.assertEqual(sorted(r_left['val'].to_list()),
                         sorted(r_right['val'].to_list()))

    # ── out-of-bounds y ───────────────────────────────────────────────────────

    def test_y_above_plot_area_returns_empty(self):
        """y above the plot area returns an empty DataFrame."""
        result = self.histop.recordsAt((0, self.histop._plot_y0_ - 5))
        self.assertEqual(len(result), 0)

    def test_y_below_last_visible_bar_returns_empty(self):
        """y below all rendered bars returns an empty DataFrame."""
        _, h = self.histop.wxh
        result = self.histop.recordsAt((0, h + 10))
        self.assertEqual(len(result), 0)

    def test_empty_result_has_correct_schema(self):
        """Empty result keeps original columns, drops internal ones."""
        result = self.histop.recordsAt((0, self.histop._plot_y0_ - 5))
        self.assertIn('cat', result.columns)
        self.assertIn('val', result.columns)
        self.assertNotIn('__p2s_index__', result.columns)

    # ── default and explicit shape ────────────────────────────────────────────

    def test_default_shape_is_select_horizontal(self):
        """Calling without shape= uses SELECT_HORIZONTALp by default."""
        y = _bar_slot_y(self.histop, 0)
        r_default  = self.histop.recordsAt((0, y))
        r_explicit = self.histop.recordsAt((0, y), shape=self.p2s.SELECT_HORIZONTALp)
        self.assertEqual(sorted(r_default['val'].to_list()),
                         sorted(r_explicit['val'].to_list()))

    def test_unsupported_shape_raises(self):
        """Passing SELECT_CIRCLEp or SELECT_VERTICALp raises ValueError."""
        y = _bar_slot_y(self.histop, 0)
        with self.assertRaises(ValueError):
            self.histop.recordsAt((0, y), shape=self.p2s.SELECT_CIRCLEp)
        with self.assertRaises(ValueError):
            self.histop.recordsAt((0, y), shape=self.p2s.SELECT_VERTICALp)

    # ── result schema ─────────────────────────────────────────────────────────

    def test_result_has_original_columns(self):
        result = self.histop.recordsAt((0, _bar_slot_y(self.histop, 0)))
        self.assertIn('cat', result.columns)
        self.assertIn('val', result.columns)

    def test_result_drops_p2s_index(self):
        result = self.histop.recordsAt((0, _bar_slot_y(self.histop, 0)))
        self.assertNotIn('__p2s_index__', result.columns)


class TestHistopRecordsAtUnrendered(unittest.TestCase):
    """recordsAt() never returns records from unrendered bins."""

    @classmethod
    def setUpClass(cls):
        cls.p2s = Polars2SVG()
        # Same fixture as TestHistopFilterByRectangleUnrenderedRecords:
        # 5 bins, only 3 visible (A, B, C); D and E are unrendered.
        cls.df = pl.DataFrame({
            'cat': ['A']*10 + ['B']*8 + ['C']*6 + ['D']*4 + ['E']*2,
            'val': list(range(30)),
        })
        cls.histop = cls.p2s.histop(
            cls.df, 'cat',
            wxh=(256, 65), bar_h=20, v_gap=0,
            insets=(2, 2), draw_context=False, distribution=False
        )

    def test_y_past_visible_area_returns_empty(self):
        """A y coordinate past the last visible bar returns empty, not D or E."""
        _, h = self.histop.wxh
        result = self.histop.recordsAt((0, h - 1))
        self.assertEqual(len(result), 0)

    def test_visible_bins_are_accessible(self):
        """The three rendered bins can be reached by their slot y values."""
        for _i_, expected_cat in enumerate(['A', 'B', 'C']):
            result = self.histop.recordsAt((0, _bar_slot_y(self.histop, _i_)))
            self.assertGreater(len(result), 0)
            self.assertTrue(all(c == expected_cat for c in result['cat'].to_list()))


class TestHistopRecordsAtStacked(unittest.TestCase):
    """recordsAt() on a stacked chart returns all color groups for the bin."""

    @classmethod
    def setUpClass(cls):
        cls.p2s = Polars2SVG()
        cls.df = pl.DataFrame({
            'cat':   ['A']*6  + ['B']*4,
            'color': ['red','red','red','red','blue','blue',
                      'red','blue','blue','blue'],
            'val':   list(range(10)),
        })
        cls.histop = cls.p2s.histop(
            cls.df, 'cat', color='color',
            style=cls.p2s.STACKEDBARp, wxh=(256, 512),
            bar_h=16, v_gap=0, draw_context=False
        )

    def test_y_in_first_bin_returns_all_colors(self):
        """y in the A row returns all 6 A records (both red and blue)."""
        result = self.histop.recordsAt((0, _bar_slot_y(self.histop, 0)))
        self.assertEqual(len(result), 6)
        colors = set(result['color'].to_list())
        self.assertIn('red',  colors)
        self.assertIn('blue', colors)


class TestHistopRecordsAtMultiFieldBin(unittest.TestCase):
    """recordsAt() with tuple bin_by drops the internal __bin__ column."""

    @classmethod
    def setUpClass(cls):
        cls.p2s = Polars2SVG()
        cls.df = pl.DataFrame({
            'cat':   ['A', 'A', 'B', 'B', 'B'],
            'group': ['x', 'y', 'x', 'x', 'y'],
            'val':   [1,   2,   3,   4,   5  ],
        })
        cls.histop = cls.p2s.histop(
            cls.df, ('cat', 'group'), wxh=(256, 512),
            bar_h=16, v_gap=0, draw_context=False
        )

    def test_result_drops_internal_bin_column(self):
        result = self.histop.recordsAt((0, _bar_slot_y(self.histop, 0)))
        self.assertNotIn('__bin__', result.columns)

    def test_result_contains_original_columns(self):
        result = self.histop.recordsAt((0, _bar_slot_y(self.histop, 0)))
        self.assertIn('cat',   result.columns)
        self.assertIn('group', result.columns)
        self.assertIn('val',   result.columns)


if __name__ == '__main__':
    unittest.main()
