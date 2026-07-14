"""
Tests for Timep.filterByRectangle(bounding_box, remove_records=False).

Strategy
--------
Timep renders horizontal time bins as vertical bars.  Each bin `i` occupies
pixel x-range [plot_x0 + i*bar_w_raw, plot_x0 + (i+1)*bar_w_raw] and
y-range [plot_y1 - bar_h, plot_y1].  All geometry fields are public after
construction, so tests derive bounding boxes directly from them rather than
hardcoding magic pixel numbers.

Three modes are covered:
  linear   – bins follow calendar time in chronological order
  periodic – bins are cyclic (e.g. month-of-year 1..12)
  stacked  – categorical color stacks within each time bin

Helper: _bin_bbox(timep, bin_index)
  Returns a bounding box (x0,y0,x1,y1) that snugly covers the bar at
  position `bin_index` (0-based position in df_agg / sorted display order).
"""
import unittest
import datetime
import polars as pl
from polars2svg import Polars2SVG


_INSET = 0.5   # half-pixel inset keeps bboxes from touching adjacent bin edges


def _bin_bbox(timep, bin_index):
    """Pixel bounding box for the bar at display position bin_index.

    A half-pixel inset on the x edges prevents the inclusive overlap test from
    accidentally selecting the adjacent bin when bins share an exact boundary.
    """
    x0 = timep._plot_x0_ + bin_index       * timep._bar_w_raw_ + _INSET
    x1 = timep._plot_x0_ + (bin_index + 1) * timep._bar_w_raw_ - _INSET
    y0 = timep._plot_y0_
    y1 = timep._plot_y1_
    return (x0, y0, x1, y1)


def _periodic_bin_bbox(timep, bin_value):
    """Pixel bounding box for a periodic bin identified by its integer value."""
    idx = bin_value - timep._bin_min_
    return _bin_bbox(timep, idx)


class TestTimepFilterByRectangleLinear(unittest.TestCase):
    """Linear (calendar-ordered) time axis."""

    @classmethod
    def setUpClass(cls):
        cls.p2s = Polars2SVG()
        # Three calendar months, each with a known, distinct record count:
        #   Jan 2024 → 5 records  (val 0-4)
        #   Feb 2024 → 3 records  (val 5-7)
        #   Mar 2024 → 7 records  (val 8-14)
        dates = (
            [datetime.date(2024, 1, 15)] * 5 +
            [datetime.date(2024, 2, 15)] * 3 +
            [datetime.date(2024, 3, 15)] * 7
        )
        cls.df = pl.DataFrame({
            'ts':  dates,
            'val': list(range(15)),
        })
        # Force monthly bins so there are exactly 3 bins (Jan, Feb, Mar)
        cls.timep = cls.p2s.timep(
            cls.df, ('ts', cls.p2s.LT_Y_mp), wxh=(512, 256)
        )

    # ── basic selection ───────────────────────────────────────────────────────

    def test_select_first_bin_returns_jan_records(self):
        """Box around bin 0 (January) returns the 5 January records."""
        result = self.timep.filterByRectangle(_bin_bbox(self.timep, 0))
        self.assertEqual(len(result), 5)
        months = result['ts'].dt.month().to_list()
        self.assertTrue(all(m == 1 for m in months))

    def test_select_second_bin_returns_feb_records(self):
        """Box around bin 1 (February) returns the 3 February records."""
        result = self.timep.filterByRectangle(_bin_bbox(self.timep, 1))
        self.assertEqual(len(result), 3)
        months = result['ts'].dt.month().to_list()
        self.assertTrue(all(m == 2 for m in months))

    def test_select_third_bin_returns_mar_records(self):
        """Box around bin 2 (March) returns the 7 March records."""
        result = self.timep.filterByRectangle(_bin_bbox(self.timep, 2))
        self.assertEqual(len(result), 7)
        months = result['ts'].dt.month().to_list()
        self.assertTrue(all(m == 3 for m in months))

    def test_box_covering_all_bins_returns_full_df(self):
        """A box spanning the whole SVG returns every record."""
        w, h = self.timep.wxh
        result = self.timep.filterByRectangle((0, 0, w, h))
        self.assertEqual(len(result), len(self.df))

    def test_box_outside_plot_returns_empty(self):
        """A box to the left of the plot area (before any bars) returns nothing."""
        result = self.timep.filterByRectangle((-100, -100, -1, -1))
        self.assertEqual(len(result), 0)

    # ── remove_records ────────────────────────────────────────────────────────

    def test_remove_records_false_is_default(self):
        bbox = _bin_bbox(self.timep, 0)
        r_default  = self.timep.filterByRectangle(bbox)
        r_explicit = self.timep.filterByRectangle(bbox, remove_records=False)
        self.assertEqual(sorted(r_default['val'].to_list()),
                         sorted(r_explicit['val'].to_list()))

    def test_remove_records_true_returns_complement(self):
        """remove_records=True on Jan bin returns Feb + Mar records."""
        result = self.timep.filterByRectangle(
            _bin_bbox(self.timep, 0), remove_records=True
        )
        self.assertEqual(len(result), 10)   # 3 Feb + 7 Mar
        months = set(result['ts'].dt.month().to_list())
        self.assertNotIn(1, months)         # no January

    def test_inside_plus_outside_covers_all(self):
        """Selected + complement together reconstruct the full DataFrame."""
        bbox    = _bin_bbox(self.timep, 1)
        inside  = self.timep.filterByRectangle(bbox)
        outside = self.timep.filterByRectangle(bbox, remove_records=True)
        combined = sorted(inside['val'].to_list() + outside['val'].to_list())
        self.assertEqual(combined, sorted(self.df['val'].to_list()))

    # ── bounding box normalisation ─────────────────────────────────────────────

    def test_inverted_bbox_gives_same_result(self):
        """x0>x1 and y0>y1 are both normalised before filtering."""
        x0, y0, x1, y1 = _bin_bbox(self.timep, 0)
        normal   = self.timep.filterByRectangle((x0, y0, x1, y1))
        inverted = self.timep.filterByRectangle((x1, y1, x0, y0))
        self.assertEqual(sorted(normal['val'].to_list()),
                         sorted(inverted['val'].to_list()))

    # ── result schema ─────────────────────────────────────────────────────────

    def test_result_has_original_columns(self):
        w, h = self.timep.wxh
        result = self.timep.filterByRectangle((0, 0, w, h))
        self.assertIn('ts',  result.columns)
        self.assertIn('val', result.columns)

    def test_result_drops_p2s_index(self):
        w, h = self.timep.wxh
        result = self.timep.filterByRectangle((0, 0, w, h))
        self.assertNotIn('__p2s_index__', result.columns)

    def test_result_drops_bin_key(self):
        """Temporary __bin_key__ column must not appear in the result."""
        w, h = self.timep.wxh
        result = self.timep.filterByRectangle((0, 0, w, h))
        self.assertNotIn('__bin_key__', result.columns)


class TestTimepFilterByRectanglePeriodic(unittest.TestCase):
    """Periodic (cyclic) time axis — month-of-year (PT_mp)."""

    @classmethod
    def setUpClass(cls):
        cls.p2s = Polars2SVG()
        # Two years of data: Jan and Mar each appear in both years.
        # Under PT_mp (month-of-year), each calendar month is one bin.
        dates = (
            [datetime.date(2023, 1, 10)] * 4 +   # January (bin value 1)
            [datetime.date(2024, 1, 20)] * 2 +   # January again → same bin
            [datetime.date(2023, 3, 5)]  * 3     # March (bin value 3)
        )
        cls.df = pl.DataFrame({
            'ts':  dates,
            'val': list(range(9)),
        })
        cls.timep = cls.p2s.timep(
            cls.df, ('ts', cls.p2s.PT_mp), wxh=(512, 256)
        )

    def test_select_january_bin(self):
        """Box over the January periodic bin returns all 6 January records."""
        result = self.timep.filterByRectangle(
            _periodic_bin_bbox(self.timep, 1)   # 1 = January
        )
        self.assertEqual(len(result), 6)
        months = result['ts'].dt.month().to_list()
        self.assertTrue(all(m == 1 for m in months))

    def test_select_march_bin(self):
        """Box over the March periodic bin returns the 3 March records."""
        result = self.timep.filterByRectangle(
            _periodic_bin_bbox(self.timep, 3)   # 3 = March
        )
        self.assertEqual(len(result), 3)
        months = result['ts'].dt.month().to_list()
        self.assertTrue(all(m == 3 for m in months))

    def test_remove_records_true_periodic(self):
        """remove_records=True on January returns March records only."""
        result = self.timep.filterByRectangle(
            _periodic_bin_bbox(self.timep, 1), remove_records=True
        )
        self.assertEqual(len(result), 3)
        months = set(result['ts'].dt.month().to_list())
        self.assertNotIn(1, months)

    def test_result_drops_time_bin_column(self):
        """Internal __time_bin__ column must not appear in the result."""
        w, h = self.timep.wxh
        result = self.timep.filterByRectangle((0, 0, w, h))
        self.assertNotIn('__time_bin__', result.columns)


class TestTimepFilterByRectangleStacked(unittest.TestCase):
    """Stacked bar chart — categorical color field produces stacked segments."""

    @classmethod
    def setUpClass(cls):
        cls.p2s = Polars2SVG()
        # Two months, two color categories.
        # Jan: 4 rows split as 3×'red' + 1×'blue'
        # Feb: 2 rows split as 1×'red' + 1×'blue'
        dates = (
            [datetime.date(2024, 1, 15)] * 4 +
            [datetime.date(2024, 2, 15)] * 2
        )
        colors = ['red', 'red', 'red', 'blue', 'red', 'blue']
        cls.df = pl.DataFrame({
            'ts':    dates,
            'color': colors,
            'val':   list(range(6)),
        })
        cls.timep = cls.p2s.timep(
            cls.df, ('ts', cls.p2s.LT_Y_mp),
            color='color', style=cls.p2s.STACKEDBARp, wxh=(512, 256)
        )

    def test_select_jan_bin_returns_all_color_groups(self):
        """Selecting the Jan bar returns records for all color categories in that bin."""
        result = self.timep.filterByRectangle(_bin_bbox(self.timep, 0))
        self.assertEqual(len(result), 4)
        colors = set(result['color'].to_list())
        self.assertIn('red',  colors)
        self.assertIn('blue', colors)

    def test_select_feb_bin_returns_feb_records(self):
        """Box around Feb bar returns only the 2 February records."""
        result = self.timep.filterByRectangle(_bin_bbox(self.timep, 1))
        self.assertEqual(len(result), 2)
        months = result['ts'].dt.month().to_list()
        self.assertTrue(all(m == 2 for m in months))

    def test_stacked_remove_records_complement(self):
        """remove_records=True on Jan returns Feb records only."""
        result = self.timep.filterByRectangle(
            _bin_bbox(self.timep, 0), remove_records=True
        )
        self.assertEqual(len(result), 2)
        months = set(result['ts'].dt.month().to_list())
        self.assertNotIn(1, months)


def _bin_center_x(timep, bin_index):
    """X pixel at the centre of the bar at display position bin_index."""
    return timep._plot_x0_ + (bin_index + 0.5) * timep._bar_w_raw_


def _periodic_bin_center_x(timep, bin_value):
    """X pixel at the centre of the periodic bin identified by bin_value."""
    idx = bin_value - timep._bin_min_
    return _bin_center_x(timep, idx)


class TestTimepRecordsAtLinear(unittest.TestCase):
    """recordsAt() on a linear time axis."""

    @classmethod
    def setUpClass(cls):
        cls.p2s = Polars2SVG()
        dates = (
            [datetime.date(2024, 1, 15)] * 5 +
            [datetime.date(2024, 2, 15)] * 3 +
            [datetime.date(2024, 3, 15)] * 7
        )
        cls.df = pl.DataFrame({'ts': dates, 'val': list(range(15))})
        cls.timep = cls.p2s.timep(cls.df, ('ts', cls.p2s.LT_Y_mp), wxh=(512, 256))

    # ── basic hit-testing ─────────────────────────────────────────────────────

    def test_center_of_first_bin_returns_jan_records(self):
        """x at bin 0 centre returns all 5 January records."""
        x = _bin_center_x(self.timep, 0)
        result = self.timep.recordsAt((x, 0))
        self.assertEqual(len(result), 5)
        self.assertTrue(all(m == 1 for m in result['ts'].dt.month().to_list()))

    def test_center_of_second_bin_returns_feb_records(self):
        x = _bin_center_x(self.timep, 1)
        result = self.timep.recordsAt((x, 0))
        self.assertEqual(len(result), 3)
        self.assertTrue(all(m == 2 for m in result['ts'].dt.month().to_list()))

    def test_center_of_third_bin_returns_mar_records(self):
        x = _bin_center_x(self.timep, 2)
        result = self.timep.recordsAt((x, 0))
        self.assertEqual(len(result), 7)
        self.assertTrue(all(m == 3 for m in result['ts'].dt.month().to_list()))

    def test_y_coordinate_is_ignored(self):
        """The y value has no effect — any y within or outside the chart works."""
        x = _bin_center_x(self.timep, 0)
        r_inside  = self.timep.recordsAt((x, self.timep._plot_y1_ / 2))
        r_outside = self.timep.recordsAt((x, 9999))
        self.assertEqual(sorted(r_inside['val'].to_list()),
                         sorted(r_outside['val'].to_list()))

    # ── out-of-bounds x ───────────────────────────────────────────────────────

    def test_x_before_plot_area_returns_empty(self):
        """x to the left of the plot area returns an empty DataFrame."""
        result = self.timep.recordsAt((self.timep._plot_x0_ - 10, 0))
        self.assertEqual(len(result), 0)

    def test_x_after_last_bin_returns_empty(self):
        """x beyond the last bar returns an empty DataFrame."""
        x_past = self.timep._plot_x0_ + self.timep._n_bins_ * self.timep._bar_w_raw_ + 10
        result = self.timep.recordsAt((x_past, 0))
        self.assertEqual(len(result), 0)

    def test_empty_result_has_correct_schema(self):
        """Empty result preserves original column names (no internal columns)."""
        result = self.timep.recordsAt((self.timep._plot_x0_ - 10, 0))
        self.assertIn('ts',  result.columns)
        self.assertIn('val', result.columns)
        self.assertNotIn('__p2s_index__', result.columns)
        self.assertNotIn('__bin_key__',   result.columns)

    # ── default and explicit shape ────────────────────────────────────────────

    def test_default_shape_is_select_vertical(self):
        """Calling without shape= uses SELECT_VERTICALp by default."""
        x = _bin_center_x(self.timep, 0)
        r_default  = self.timep.recordsAt((x, 0))
        r_explicit = self.timep.recordsAt((x, 0), shape=self.p2s.SELECT_VERTICALp)
        self.assertEqual(sorted(r_default['val'].to_list()),
                         sorted(r_explicit['val'].to_list()))

    def test_unsupported_shape_raises(self):
        """Passing SELECT_CIRCLEp or SELECT_HORIZONTALp raises ValueError."""
        x = _bin_center_x(self.timep, 0)
        with self.assertRaises(ValueError):
            self.timep.recordsAt((x, 0), shape=self.p2s.SELECT_CIRCLEp)
        with self.assertRaises(ValueError):
            self.timep.recordsAt((x, 0), shape=self.p2s.SELECT_HORIZONTALp)

    # ── result schema ─────────────────────────────────────────────────────────

    def test_result_has_original_columns(self):
        x = _bin_center_x(self.timep, 0)
        result = self.timep.recordsAt((x, 0))
        self.assertIn('ts',  result.columns)
        self.assertIn('val', result.columns)

    def test_result_drops_p2s_index(self):
        x = _bin_center_x(self.timep, 0)
        result = self.timep.recordsAt((x, 0))
        self.assertNotIn('__p2s_index__', result.columns)

    def test_result_drops_bin_key(self):
        x = _bin_center_x(self.timep, 0)
        result = self.timep.recordsAt((x, 0))
        self.assertNotIn('__bin_key__', result.columns)


class TestTimepRecordsAtPeriodic(unittest.TestCase):
    """recordsAt() on a periodic (cyclic) time axis."""

    @classmethod
    def setUpClass(cls):
        cls.p2s = Polars2SVG()
        dates = (
            [datetime.date(2023, 1, 10)] * 4 +
            [datetime.date(2024, 1, 20)] * 2 +
            [datetime.date(2023, 3,  5)] * 3
        )
        cls.df = pl.DataFrame({'ts': dates, 'val': list(range(9))})
        cls.timep = cls.p2s.timep(cls.df, ('ts', cls.p2s.PT_mp), wxh=(512, 256))

    def test_x_at_january_bin_returns_all_january_records(self):
        """Six January records (two years) are all returned by January bin x."""
        x = _periodic_bin_center_x(self.timep, 1)   # 1 = January
        result = self.timep.recordsAt((x, 0))
        self.assertEqual(len(result), 6)
        self.assertTrue(all(m == 1 for m in result['ts'].dt.month().to_list()))

    def test_x_at_march_bin_returns_march_records(self):
        x = _periodic_bin_center_x(self.timep, 3)   # 3 = March
        result = self.timep.recordsAt((x, 0))
        self.assertEqual(len(result), 3)
        self.assertTrue(all(m == 3 for m in result['ts'].dt.month().to_list()))

    def test_x_outside_plot_returns_empty(self):
        result = self.timep.recordsAt((self.timep._plot_x0_ - 5, 0))
        self.assertEqual(len(result), 0)

    def test_result_drops_time_bin_column(self):
        """__time_bin__ must not appear in the result."""
        x = _periodic_bin_center_x(self.timep, 1)
        result = self.timep.recordsAt((x, 0))
        self.assertNotIn('__time_bin__', result.columns)


class TestTimepRecordsAtStacked(unittest.TestCase):
    """recordsAt() on a stacked bar chart returns all color groups for the bin."""

    @classmethod
    def setUpClass(cls):
        cls.p2s = Polars2SVG()
        dates  = [datetime.date(2024, 1, 15)] * 4 + [datetime.date(2024, 2, 15)] * 2
        colors = ['red', 'red', 'red', 'blue', 'red', 'blue']
        cls.df = pl.DataFrame({'ts': dates, 'color': colors, 'val': list(range(6))})
        cls.timep = cls.p2s.timep(
            cls.df, ('ts', cls.p2s.LT_Y_mp),
            color='color', style=cls.p2s.STACKEDBARp, wxh=(512, 256)
        )

    def test_x_at_jan_returns_all_color_groups(self):
        """x inside the Jan column returns both red and blue records for Jan."""
        x = _bin_center_x(self.timep, 0)
        result = self.timep.recordsAt((x, 0))
        self.assertEqual(len(result), 4)
        self.assertIn('red',  result['color'].to_list())
        self.assertIn('blue', result['color'].to_list())


if __name__ == '__main__':
    unittest.main()
