"""
Tests for the "None" bin — the bar produced by rows whose bin_by value is null.

polars' group_by gives null its own group, so a null bin_by value renders as a
normal bar labelled 'None'.  Selecting it, however, goes through a join on the
bin column, and polars joins treat null as *unequal to itself* by default.  That
made the "None" bar inert in the interactive histogram (histopi): clicking or
dragging over it selected nothing (inner join → 0 rows), and filtering it out
removed nothing (anti join → every null row kept).  Every bin→record path
therefore joins with nulls_equal=True.

Covered here for each entry point the interactive controller calls:
  filterByRectangle()  drag select        (applyDragOp)
  filterByOval()       drag select, oval  (applyDragOp)
  recordsAt()          brush              (applyBrushOp → _doBrushAt)
  filterBySubstring()  '/' search         (applySearchOp)
"""
import unittest
import polars as pl
from polars2svg import Polars2SVG


def _bar_bbox(histop, display_index):
    """Full-width bounding box of the bar at display_index, inset half a pixel on
    the y edges so the inclusive overlap test cannot reach the adjacent bar."""
    _inset = 0.5
    y_v = histop.v_gap // 2 if histop.v_gap > 0 else 0
    y0  = histop._plot_y0_ + display_index * histop._slot_h_ + y_v + _inset
    return (histop._plot_x0_, y0,
            histop._plot_x0_ + histop._plot_w_, y0 + histop.bar_h - 2 * _inset)


def _bar_slot_y(histop, display_index):
    """A y coordinate inside the slot of the bar at display_index."""
    return histop._plot_y0_ + display_index * histop._slot_h_ + 1


class TestHistopNullBinSingleField(unittest.TestCase):
    """bin_by is a single column holding nulls."""

    @classmethod
    def setUpClass(cls):
        cls.p2s = Polars2SVG()
        # Counts are distinct so the descending sort is deterministic:
        # None=5, 'a'=3, 'b'=2  →  display order None (0), a (1), b (2).
        cls.df = pl.DataFrame({
            'cat': [None] * 5 + ['a'] * 3 + ['b'] * 2,
            'val': list(range(10)),
        })
        cls.h = cls.p2s.histop(cls.df, 'cat', wxh=(256, 512), bar_h=16, v_gap=0,
                               draw_context=False)

    # ── the bar exists and is drawn first ─────────────────────────────────────

    def test_null_is_its_own_bin(self):
        self.assertIn(None, self.h._sorted_bins_)

    def test_null_bin_is_the_largest(self):
        self.assertEqual(self.h._sorted_bins_[0], None)

    def test_null_bin_renders_as_none_label(self):
        self.assertIn('None', self.h._repr_svg_())

    # ── filterByRectangle ─────────────────────────────────────────────────────

    def test_rectangle_selects_null_bin(self):
        result = self.h.filterByRectangle(_bar_bbox(self.h, 0))
        self.assertEqual(len(result), 5)
        self.assertTrue(all(v is None for v in result['cat'].to_list()))

    def test_rectangle_removes_null_bin(self):
        result = self.h.filterByRectangle(_bar_bbox(self.h, 0), remove_records=True)
        self.assertEqual(len(result), 5)
        self.assertNotIn(None, result['cat'].to_list())

    def test_rectangle_on_other_bin_keeps_null_rows(self):
        """Removing a non-null bin must not disturb the null rows."""
        result = self.h.filterByRectangle(_bar_bbox(self.h, 1), remove_records=True)
        self.assertEqual(result['cat'].null_count(), 5)
        self.assertNotIn('a', result['cat'].to_list())

    def test_rectangle_select_and_remove_partition_the_frame(self):
        keep   = self.h.filterByRectangle(_bar_bbox(self.h, 0))
        remove = self.h.filterByRectangle(_bar_bbox(self.h, 0), remove_records=True)
        self.assertEqual(len(keep) + len(remove), len(self.df))

    # ── filterByOval ──────────────────────────────────────────────────────────

    def test_oval_selects_null_bin(self):
        x0, y0, x1, y1 = _bar_bbox(self.h, 0)
        result = self.h.filterByOval(((x0 + x1) / 2, (y0 + y1) / 2,
                                      (x1 - x0) / 2, (y1 - y0) / 2))
        self.assertEqual(len(result), 5)
        self.assertTrue(all(v is None for v in result['cat'].to_list()))

    def test_oval_removes_null_bin(self):
        x0, y0, x1, y1 = _bar_bbox(self.h, 0)
        result = self.h.filterByOval(((x0 + x1) / 2, (y0 + y1) / 2,
                                      (x1 - x0) / 2, (y1 - y0) / 2),
                                     remove_records=True)
        self.assertEqual(len(result), 5)
        self.assertNotIn(None, result['cat'].to_list())

    # ── recordsAt (brush) ─────────────────────────────────────────────────────

    def test_records_at_null_bin(self):
        result = self.h.recordsAt((self.h._plot_x0_ + 2, _bar_slot_y(self.h, 0)))
        self.assertEqual(len(result), 5)
        self.assertTrue(all(v is None for v in result['cat'].to_list()))

    def test_records_at_non_null_bin_unaffected(self):
        result = self.h.recordsAt((self.h._plot_x0_ + 2, _bar_slot_y(self.h, 1)))
        self.assertEqual(sorted(result['cat'].to_list()), ['a'] * 3)

    # ── filterBySubstring ('/' search) ────────────────────────────────────────

    def test_substring_none_selects_null_bin(self):
        """The bar is labelled 'None', so '/none' has to match it."""
        result = self.h.filterBySubstring('none')
        self.assertEqual(len(result), 5)
        self.assertTrue(all(v is None for v in result['cat'].to_list()))

    def test_substring_none_remove_drops_null_bin(self):
        result = self.h.filterBySubstring('none', remove_bins=True)
        self.assertEqual(len(result), 5)
        self.assertNotIn(None, result['cat'].to_list())

    def test_substring_other_bin_keeps_null_rows(self):
        result = self.h.filterBySubstring('a', remove_bins=True)
        self.assertEqual(result['cat'].null_count(), 5)

    # ── result hygiene ────────────────────────────────────────────────────────

    def test_null_selection_drops_p2s_index(self):
        result = self.h.filterByRectangle(_bar_bbox(self.h, 0))
        self.assertNotIn('__p2s_index__', result.columns)

    def test_null_selection_keeps_original_columns(self):
        result = self.h.filterByRectangle(_bar_bbox(self.h, 0))
        self.assertEqual(result.columns, self.df.columns)


class TestHistopNullBinNumeric(unittest.TestCase):
    """The bin column need not be a string for nulls to appear."""

    @classmethod
    def setUpClass(cls):
        cls.p2s = Polars2SVG()
        cls.df = pl.DataFrame({
            'code': [None, None, None, 1, 1, 2],
            'val':  list(range(6)),
        }, schema={'code': pl.Int64, 'val': pl.Int64})
        cls.h = cls.p2s.histop(cls.df, 'code', wxh=(256, 512), bar_h=16, v_gap=0,
                               draw_context=False)

    def test_rectangle_selects_null_bin(self):
        idx = self.h._sorted_bins_.index(None)
        result = self.h.filterByRectangle(_bar_bbox(self.h, idx))
        self.assertEqual(len(result), 3)
        self.assertEqual(result['code'].null_count(), 3)

    def test_rectangle_removes_null_bin(self):
        idx = self.h._sorted_bins_.index(None)
        result = self.h.filterByRectangle(_bar_bbox(self.h, idx), remove_records=True)
        self.assertEqual(len(result), 3)
        self.assertEqual(result['code'].null_count(), 0)


class TestHistopNullBinMultiField(unittest.TestCase):
    """Tuple bin_by concatenates into '__bin__'; pl.concat_str yields null when
    any part is null, so those rows fold into one 'None' bar."""

    @classmethod
    def setUpClass(cls):
        cls.p2s = Polars2SVG()
        cls.df = pl.DataFrame({
            'cat':   ['A',  'A',  'A',  None, None, 'B'],
            'group': ['x',  'x',  'x',  'y',  'z',  None],
            'val':   [1,    2,    3,    4,    5,    6],
        })
        cls.h = cls.p2s.histop(cls.df, ('cat', 'group'), wxh=(256, 512), bar_h=16,
                               v_gap=0, draw_context=False)

    def test_null_bin_present(self):
        self.assertIn(None, self.h._sorted_bins_)

    def test_rectangle_selects_all_rows_with_a_null_part(self):
        idx = self.h._sorted_bins_.index(None)
        result = self.h.filterByRectangle(_bar_bbox(self.h, idx))
        self.assertEqual(sorted(result['val'].to_list()), [4, 5, 6])

    def test_rectangle_removes_all_rows_with_a_null_part(self):
        idx = self.h._sorted_bins_.index(None)
        result = self.h.filterByRectangle(_bar_bbox(self.h, idx), remove_records=True)
        self.assertEqual(sorted(result['val'].to_list()), [1, 2, 3])

    def test_records_at_null_bin(self):
        idx = self.h._sorted_bins_.index(None)
        result = self.h.recordsAt((self.h._plot_x0_ + 2, _bar_slot_y(self.h, idx)))
        self.assertEqual(sorted(result['val'].to_list()), [4, 5, 6])

    def test_internal_bin_column_dropped(self):
        idx = self.h._sorted_bins_.index(None)
        result = self.h.filterByRectangle(_bar_bbox(self.h, idx))
        self.assertNotIn('__bin__', result.columns)


class TestHistopNullBinDistributionStrip(unittest.TestCase):
    """The distribution strip quantizes bins by bar length; a null bin lands in a
    strip cell like any other and must be selectable from there too."""

    @classmethod
    def setUpClass(cls):
        cls.p2s = Polars2SVG()
        cls.df = pl.DataFrame({
            'cat': [None] * 5 + ['a'] * 3 + ['b'] * 2,
            'val': list(range(10)),
        })
        cls.h = cls.p2s.histop(cls.df, 'cat', wxh=(256, 512), bar_h=16, v_gap=0,
                               draw_context=False)

    def _strip_cell_for_null(self):
        for _bi_, _bins_ in self.h._dist_bins_lu_.items():
            if None in _bins_:
                return _bi_, _bins_
        return None, None

    def test_strip_is_active(self):
        self.assertGreater(self.h._dist_h_, 0)
        self.assertTrue(self.h._dist_bins_lu_)

    def test_null_bin_is_in_the_strip(self):
        _bi_, _ = self._strip_cell_for_null()
        self.assertIsNotNone(_bi_)

    def test_records_at_strip_cell_includes_null_rows(self):
        _bi_, _bins_ = self._strip_cell_for_null()
        _x_ = self.h._plot_x0_ + (_bi_ + 0.5) * self.h._dist_actual_bin_w_
        _y_ = self.h._dist_strip_y0_ + 1
        result = self.h.recordsAt((_x_, _y_))
        # every row whose bin shares that strip cell, nulls included
        _expected_ = sum(5 if b is None else (3 if b == 'a' else 2) for b in _bins_)
        self.assertEqual(len(result), _expected_)
        self.assertEqual(result['cat'].null_count(), 5)

    def test_rectangle_over_strip_cell_includes_null_rows(self):
        _bi_, _ = self._strip_cell_for_null()
        _abw_ = self.h._dist_actual_bin_w_
        _x0_  = self.h._plot_x0_ + _bi_ * _abw_ + 0.5
        _x1_  = self.h._plot_x0_ + (_bi_ + 1) * _abw_ - 0.5
        _y0_  = self.h._dist_strip_y0_ + 0.5
        _y1_  = _y0_ + self.h.distribution_bin_w - 1.0
        result = self.h.filterByRectangle((_x0_, _y0_, _x1_, _y1_))
        self.assertEqual(result['cat'].null_count(), 5)


if __name__ == '__main__':
    unittest.main()
