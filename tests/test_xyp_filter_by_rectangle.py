"""
Tests for XYp.filterByRectangle(bounding_box, remove_records=False).

Strategy
--------
XYp stores per-point pixel coordinates in self.df_flat (__xpx__, __ypx__).
Each test constructs a small, deterministic scatter plot, reads the actual
pixel coordinates from df_flat for specific records (identified by
__p2s_index__), then builds bounding boxes from those coordinates.
This avoids re-deriving the coordinate transform and tests the method
against what the renderer actually computed.

Geometry reminder (scatter / simple XYp)
-----------------------------------------
  __xpx__  – x pixel coordinate of the rendered dot
  __ypx__  – y pixel coordinate of the rendered dot
  __p2s_index__ – row index added to self.df; matches across df and df_flat
"""
import unittest
import polars as pl
from polars2svg import Polars2SVG


def _pixel_of(xyp, p2s_index):
    """Return (xpx, ypx) of the record with the given __p2s_index__."""
    row = (xyp.df_flat
              .filter(pl.col('__p2s_index__') == p2s_index)
              .select(['__xpx__', '__ypx__'])
              .row(0))
    return row[0], row[1]


class TestXYpFilterByRectangle(unittest.TestCase):

    def setUp(self):
        self.p2s = Polars2SVG()
        # Five points spread far enough apart in data space that each maps to
        # a clearly distinct pixel position.  val is unique per row so we can
        # assert the exact record came back.
        self.df = pl.DataFrame({
            'x':   [0.0,  25.0,  50.0,  75.0, 100.0],
            'y':   [0.0,  25.0,  50.0,  75.0, 100.0],
            'val': [10,   20,    30,    40,    50   ],
        })
        self.xyp = self.p2s.xyp(self.df, 'x', 'y', wxh=(256, 256))

    # ── basic selection ───────────────────────────────────────────────────────

    def test_box_around_single_point_returns_that_record(self):
        """A tight box around one dot returns exactly one record."""
        px, py = _pixel_of(self.xyp, 0)   # val == 10
        eps = 3.0
        result = self.xyp.filterByRectangle((px - eps, py - eps, px + eps, py + eps))
        self.assertEqual(len(result), 1)
        self.assertEqual(result['val'][0], 10)

    def test_box_around_middle_point(self):
        """Box centered on the middle point (p2s_index=2, val=30)."""
        px, py = _pixel_of(self.xyp, 2)
        eps = 3.0
        result = self.xyp.filterByRectangle((px - eps, py - eps, px + eps, py + eps))
        self.assertEqual(len(result), 1)
        self.assertEqual(result['val'][0], 30)

    def test_box_covering_two_points(self):
        """A box that spans two adjacent points returns exactly two records."""
        px0, py0 = _pixel_of(self.xyp, 0)   # val 10
        px1, py1 = _pixel_of(self.xyp, 1)   # val 20
        eps = 3.0
        x_lo = min(px0, px1) - eps
        x_hi = max(px0, px1) + eps
        y_lo = min(py0, py1) - eps
        y_hi = max(py0, py1) + eps
        result = self.xyp.filterByRectangle((x_lo, y_lo, x_hi, y_hi))
        self.assertEqual(len(result), 2)
        self.assertIn(10, result['val'].to_list())
        self.assertIn(20, result['val'].to_list())

    def test_box_covering_all_points_returns_full_df(self):
        """A box covering the entire SVG canvas returns every record."""
        w, h = self.xyp.wxh
        result = self.xyp.filterByRectangle((0, 0, w, h))
        self.assertEqual(len(result), len(self.df))

    def test_box_outside_plot_returns_empty(self):
        """A box placed in the top-left corner before any dots returns nothing."""
        result = self.xyp.filterByRectangle((-10, -10, -1, -1))
        self.assertEqual(len(result), 0)

    # ── remove_records ────────────────────────────────────────────────────────

    def test_remove_records_false_is_default_behaviour(self):
        """remove_records=False (default) and explicit False give same result."""
        px, py = _pixel_of(self.xyp, 0)
        eps = 3.0
        bbox = (px - eps, py - eps, px + eps, py + eps)
        r_default  = self.xyp.filterByRectangle(bbox)
        r_explicit = self.xyp.filterByRectangle(bbox, remove_records=False)
        self.assertEqual(sorted(r_default['val'].to_list()),
                         sorted(r_explicit['val'].to_list()))

    def test_remove_records_true_returns_complement(self):
        """remove_records=True returns all records NOT hit by the box."""
        px, py = _pixel_of(self.xyp, 0)   # val 10
        eps = 3.0
        bbox = (px - eps, py - eps, px + eps, py + eps)
        result = self.xyp.filterByRectangle(bbox, remove_records=True)
        self.assertEqual(len(result), 4)
        self.assertNotIn(10, result['val'].to_list())

    def test_remove_records_true_plus_false_cover_all_records(self):
        """Inside + outside together reconstruct the full original DataFrame."""
        px, py = _pixel_of(self.xyp, 2)   # val 30
        eps = 3.0
        bbox   = (px - eps, py - eps, px + eps, py + eps)
        inside  = self.xyp.filterByRectangle(bbox)
        outside = self.xyp.filterByRectangle(bbox, remove_records=True)
        combined = sorted(inside['val'].to_list() + outside['val'].to_list())
        self.assertEqual(combined, sorted(self.df['val'].to_list()))

    def test_box_outside_with_remove_records_returns_everything(self):
        """A box that misses all dots and remove_records=True returns all rows."""
        result = self.xyp.filterByRectangle((-10, -10, -1, -1), remove_records=True)
        self.assertEqual(len(result), len(self.df))

    # ── bounding box normalisation ─────────────────────────────────────────────

    def test_inverted_x_coords_normalised(self):
        """Passing x0 > x1 should be normalised and return the same result."""
        px, py = _pixel_of(self.xyp, 0)
        eps = 3.0
        normal   = self.xyp.filterByRectangle((px - eps, py - eps, px + eps, py + eps))
        inverted = self.xyp.filterByRectangle((px + eps, py - eps, px - eps, py + eps))
        self.assertEqual(sorted(normal['val'].to_list()),
                         sorted(inverted['val'].to_list()))

    def test_inverted_y_coords_normalised(self):
        """Passing y0 > y1 should be normalised and return the same result."""
        px, py = _pixel_of(self.xyp, 0)
        eps = 3.0
        normal   = self.xyp.filterByRectangle((px - eps, py - eps, px + eps, py + eps))
        inverted = self.xyp.filterByRectangle((px - eps, py + eps, px + eps, py - eps))
        self.assertEqual(sorted(normal['val'].to_list()),
                         sorted(inverted['val'].to_list()))

    # ── result schema ─────────────────────────────────────────────────────────

    def test_result_contains_original_columns(self):
        """Returned DataFrame preserves the original user-supplied columns."""
        w, h = self.xyp.wxh
        result = self.xyp.filterByRectangle((0, 0, w, h))
        for col in ['x', 'y', 'val']:
            self.assertIn(col, result.columns)

    def test_result_drops_p2s_index(self):
        """__p2s_index__ is an internal column and must not appear in the result."""
        w, h = self.xyp.wxh
        result = self.xyp.filterByRectangle((0, 0, w, h))
        self.assertNotIn('__p2s_index__', result.columns)

    def test_result_has_no_pixel_columns(self):
        """Pixel-space columns (__xpx__, __ypx__) must not bleed into the result."""
        w, h = self.xyp.wxh
        result = self.xyp.filterByRectangle((0, 0, w, h))
        self.assertNotIn('__xpx__', result.columns)
        self.assertNotIn('__ypx__', result.columns)

    # ── colour-encoded scatter ────────────────────────────────────────────────

    def test_works_with_categorical_color_encoding(self):
        """filterByRectangle still works when a categorical color field is used."""
        df = pl.DataFrame({
            'x':   [0.0, 50.0, 100.0],
            'y':   [0.0, 50.0, 100.0],
            'grp': ['a',  'b',   'a'],
            'val': [1,    2,     3  ],
        })
        xyp = self.p2s.xyp(df, 'x', 'y', color='grp', wxh=(256, 256))
        px, py = _pixel_of(xyp, 1)   # val 2, group 'b'
        eps = 3.0
        result = xyp.filterByRectangle((px - eps, py - eps, px + eps, py + eps))
        self.assertEqual(len(result), 1)
        self.assertEqual(result['val'][0], 2)

    def test_works_with_numeric_color_encoding(self):
        """filterByRectangle works when color is mapped to a numeric spectrum."""
        df = pl.DataFrame({
            'x':     [0.0, 50.0, 100.0],
            'y':     [0.0, 50.0, 100.0],
            'score': [1.0,  5.0,  10.0],
        })
        xyp = self.p2s.xyp(df, 'x', 'y', color='score', wxh=(256, 256))
        px, py = _pixel_of(xyp, 0)
        eps = 3.0
        result = xyp.filterByRectangle((px - eps, py - eps, px + eps, py + eps))
        self.assertEqual(len(result), 1)


if __name__ == '__main__':
    unittest.main()
