"""
Tests for the oval/ellipse selection shape: <Plot>.filterByOval(oval, remove_records=False).

`oval` is (cx, cy, rx, ry): the mouse-press point is the ellipse *center* and the
drag distance sets the radii. Membership is the standard point-in-ellipse test
    ((px-cx)/rx)^2 + ((py-cy)/ry)^2 <= 1
for point-like reps (xyp dots, chordp node midpoints, piep cell/wedge samples), and an
exact ellipse-vs-axis-aligned-box overlap test for the bar plots (timep, histop).

These mirror the existing filterByRectangle test suites but exercise the ellipse path,
including the radius floor that keeps a click grabbing the item under the cursor.
"""
import unittest
import datetime
from math import cos, sin

import polars as pl
from polars2svg import Polars2SVG


def _pixel_of(xyp, p2s_index):
    row = (xyp.df_flat
              .filter(pl.col('__p2s_index__') == p2s_index)
              .select(['__xpx__', '__ypx__'])
              .row(0))
    return row[0], row[1]


class TestXYpFilterByOval(unittest.TestCase):

    def setUp(self):
        self.p2s = Polars2SVG()
        self.df = pl.DataFrame({
            'x':   [0.0,  25.0,  50.0,  75.0, 100.0],
            'y':   [0.0,  25.0,  50.0,  75.0, 100.0],
            'val': [10,   20,    30,    40,    50   ],
        })
        self.xyp = self.p2s.xyp(self.df, 'x', 'y', wxh=(256, 256))

    def test_small_oval_around_single_point(self):
        px, py = _pixel_of(self.xyp, 2)
        result = self.xyp.filterByOval((px, py, 3.0, 3.0))
        self.assertEqual(len(result), 1)
        self.assertEqual(result['val'][0], 30)

    def test_click_zero_radius_still_grabs_point_under_cursor(self):
        """A click (rx=ry=0) is floored to cover the pixel under the cursor."""
        px, py = _pixel_of(self.xyp, 1)
        result = self.xyp.filterByOval((px, py, 0, 0))
        self.assertIn(20, result['val'].to_list())

    def test_oval_covering_all_points(self):
        w, h = self.xyp.wxh
        result = self.xyp.filterByOval((w / 2, h / 2, w, h))
        self.assertEqual(len(result), len(self.df))

    def test_oval_outside_returns_empty(self):
        result = self.xyp.filterByOval((-50, -50, 5, 5))
        self.assertEqual(len(result), 0)

    def test_corner_outside_ellipse_but_inside_bbox(self):
        """A point near an ellipse corner is excluded even though a same-extent box keeps it.

        Places a probe point just outside the ellipse boundary along the 45° diagonal,
        where the inscribed ellipse excludes it but the bounding rectangle would not.
        """
        # Point 0 sits at one pixel corner; build an oval centered on point 2 whose
        # radii reach point 0/point 4 only along the axes, not the diagonal.
        px2, py2 = _pixel_of(self.xyp, 2)
        px0, py0 = _pixel_of(self.xyp, 0)
        rx = abs(px2 - px0) + 1.0
        ry = abs(py2 - py0) + 1.0
        oval_result = set(self.xyp.filterByOval((px2, py2, rx, ry))['val'].to_list())
        rect_result = set(self.xyp.filterByRectangle(
            (px2 - rx, py2 - ry, px2 + rx, py2 + ry))['val'].to_list())
        # The diagonal corner points (10, 50) fall inside the box but outside the ellipse.
        self.assertTrue(oval_result.issubset(rect_result))
        self.assertNotIn(10, oval_result)
        self.assertNotIn(50, oval_result)
        self.assertIn(10, rect_result)

    def test_remove_records_true_returns_complement(self):
        px, py = _pixel_of(self.xyp, 0)
        oval = (px, py, 3.0, 3.0)
        result = self.xyp.filterByOval(oval, remove_records=True)
        self.assertEqual(len(result), 4)
        self.assertNotIn(10, result['val'].to_list())

    def test_inside_plus_outside_reconstructs_full_df(self):
        px, py = _pixel_of(self.xyp, 2)
        oval = (px, py, 3.0, 3.0)
        inside  = self.xyp.filterByOval(oval)
        outside = self.xyp.filterByOval(oval, remove_records=True)
        combined = sorted(inside['val'].to_list() + outside['val'].to_list())
        self.assertEqual(combined, sorted(self.df['val'].to_list()))

    def test_result_drops_internal_columns(self):
        w, h = self.xyp.wxh
        result = self.xyp.filterByOval((w / 2, h / 2, w, h))
        for col in ['__p2s_index__', '__xpx__', '__ypx__']:
            self.assertNotIn(col, result.columns)
        for col in ['x', 'y', 'val']:
            self.assertIn(col, result.columns)


class TestTimepFilterByOval(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.p2s = Polars2SVG()
        dates = ([datetime.date(2024, 1, 15)] * 5 +
                 [datetime.date(2024, 2, 15)] * 3 +
                 [datetime.date(2024, 3, 15)] * 7)
        cls.df = pl.DataFrame({'when': dates, 'val': list(range(len(dates)))})
        cls.tp = cls.p2s.timep(cls.df, 'when')

    def _bar_center(self, bin_index):
        x = self.tp._plot_x0_ + (bin_index + 0.5) * self.tp._bar_w_raw_
        y = self.tp._plot_y1_ - 2.0   # just inside the bar, above the baseline
        return x, y

    def test_oval_over_first_bar_selects_it(self):
        x, y = self._bar_center(0)
        result = self.tp.filterByOval((x, y, self.tp._bar_w_raw_ / 2 - 0.5, 4.0))
        self.assertEqual(len(result), 5)   # Jan has 5 records

    def test_oval_covering_whole_plot_returns_all(self):
        w, h = self.tp.wxh
        result = self.tp.filterByOval((w / 2, h / 2, w, h))
        self.assertEqual(len(result), len(self.df))

    def test_oval_off_plot_returns_empty(self):
        result = self.tp.filterByOval((-50, -50, 5, 5))
        self.assertEqual(len(result), 0)

    def test_remove_records_complement(self):
        w, h = self.tp.wxh
        inside  = self.tp.filterByOval((w / 2, h / 2, w, h))
        outside = self.tp.filterByOval((w / 2, h / 2, w, h), remove_records=True)
        self.assertEqual(len(inside), len(self.df))
        self.assertEqual(len(outside), 0)


class TestHistopFilterByOval(unittest.TestCase):

    def setUp(self):
        self.p2s = Polars2SVG()
        self.df = pl.DataFrame({
            'cat': ['a'] * 6 + ['b'] * 4 + ['c'] * 2,
            'val': list(range(12)),
        })
        self.hp = self.p2s.histop(self.df, 'cat')

    def test_oval_covering_whole_plot_returns_all(self):
        w, h = self.hp.wxh
        result = self.hp.filterByOval((w / 2, h / 2, w, h))
        self.assertEqual(len(result), len(self.df))

    def test_oval_off_plot_returns_empty(self):
        result = self.hp.filterByOval((-50, -50, 5, 5))
        self.assertEqual(len(result), 0)

    def test_remove_records_complement(self):
        w, h = self.hp.wxh
        outside = self.hp.filterByOval((w / 2, h / 2, w, h), remove_records=True)
        self.assertEqual(len(outside), 0)

    def test_result_drops_p2s_index(self):
        w, h = self.hp.wxh
        result = self.hp.filterByOval((w / 2, h / 2, w, h))
        self.assertNotIn('__p2s_index__', result.columns)


class TestChordpFilterByOval(unittest.TestCase):

    def setUp(self):
        self.p2s = Polars2SVG()
        self.df = pl.DataFrame({
            'fm':     ['a', 'b', 'c', 'a', 'b', 'd'],
            'to':     ['b', 'c', 'a', 'c', 'a', 'a'],
            'weight': [1,   3,   2,   1,   4,   2  ],
        })
        self.ch = self.p2s.chordp(df=self.df, relationships=[('fm', 'to')])

    def _node_midpoint(self, node):
        row = self.ch.df_node.filter(pl.col('__nm__').cast(pl.String) == str(node))
        amr = row['__amr__'][0]
        r_mid = (self.ch.r + self.ch.r_inner) / 2.0
        return (self.ch.cx + r_mid * cos(amr), self.ch.cy + r_mid * sin(amr))

    def test_oval_covering_whole_plot_returns_all(self):
        w, h = self.ch.wxh
        result = self.ch.filterByOval((w / 2, h / 2, w, h))
        self.assertEqual(len(result), len(self.df))

    def test_oval_off_plot_returns_empty(self):
        result = self.ch.filterByOval((-500, -500, 5, 5))
        self.assertEqual(len(result), 0)

    def test_small_oval_on_node_selects_its_edges(self):
        mx, my = self._node_midpoint('a')
        result = self.ch.filterByOval((mx, my, 4.0, 4.0))
        # every returned edge should touch node 'a'
        for fm, to in zip(result['fm'].to_list(), result['to'].to_list()):
            self.assertTrue('a' in (fm, to))

    def test_remove_records_complement(self):
        w, h = self.ch.wxh
        outside = self.ch.filterByOval((w / 2, h / 2, w, h), remove_records=True)
        self.assertEqual(len(outside), 0)


class TestPiepFilterByOval(unittest.TestCase):

    def setUp(self):
        self.p2s = Polars2SVG()
        self.df = pl.DataFrame({'cat': ['a'] * 6 + ['b'] * 3 + ['c'] * 2})

    def test_pie_oval_covering_whole_plot_returns_all(self):
        pp = self.p2s.piep(self.df, 'cat')
        w, h = pp.wxh
        result = pp.filterByOval((w / 2, h / 2, w, h))
        self.assertEqual(len(result), len(self.df))

    def test_pie_oval_off_plot_returns_empty(self):
        pp = self.p2s.piep(self.df, 'cat')
        result = pp.filterByOval((-500, -500, 5, 5))
        self.assertEqual(len(result), 0)

    def test_waffle_oval_covering_whole_plot_returns_all(self):
        pp = self.p2s.piep(self.df, 'cat', style=self.p2s.WAFFLEp, waffle_n=10)
        w, h = pp.wxh
        result = pp.filterByOval((w / 2, h / 2, w, h))
        self.assertEqual(len(result), len(self.df))

    def test_remove_records_complement(self):
        pp = self.p2s.piep(self.df, 'cat')
        w, h = pp.wxh
        outside = pp.filterByOval((w / 2, h / 2, w, h), remove_records=True)
        self.assertEqual(len(outside), 0)


if __name__ == '__main__':
    unittest.main()
