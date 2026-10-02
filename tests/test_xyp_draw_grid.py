"""
Tests for xyp's draw_grid=: the gridlines inside the plot, apart from the axes.

draw_context=False could only drop the axes and the gridlines together (user feedback
2026-09-27: "a way to not draw the lines internal but still draw the axes").  draw_grid=False
drops the lines inside the plot and the small labels drawn along them, and keeps the plot
outline, the two end labels and the axis name on each axis.
"""
import datetime as dt
import re
import unittest

import polars as pl

from polars2svg import Polars2SVG
from svg_test_utils import normalize_svg


def _frame_():
    return pl.DataFrame({
        'n':    [1.0, 2.5, 4.0, 7.5, 9.0, 3.0, 6.0, 8.0],
        'v':    [3.0, 1.0, 4.0, 1.5, 5.0, 9.0, 2.0, 6.0],
        'cat':  ['ant', 'bee', 'cat', 'dog', 'eel', 'fox', 'gnu', 'hen'],
        'when': [dt.datetime(2026, 1, 1) + dt.timedelta(days=9 * i) for i in range(8)],
    })


class TestXYpDrawGrid(unittest.TestCase):

    def setUp(self):
        self.p2s   = Polars2SVG()
        self.df    = _frame_()

    def _xyp_(self, x, **kw):
        if x == 'periodic': x = self.p2s.tField('when', self.p2s.PT_mp)
        _x_ = self.p2s.xyp(self.df, x, 'v', wxh=(256, 160), **kw)
        _x_._repr_svg_()
        return _x_

    # The outline is a <rect>, so every <line> in the context is a gridline or a tick on one
    # (a periodic axis draws its majors in the axis colour, not the inner one).
    def _gridlines_(self, xyp):
        return len(re.findall(r'<line ', xyp.svg_context))

    def _end_and_name_labels_(self, xyp):
        return re.findall(r'font-size="12px"[^>]*>([^<]+)<', xyp.svg_context)

    def test_off_drops_the_gridlines_and_keeps_the_axes(self):
        for _x_ in ('n', 'cat', 'when', 'periodic'):
            with self.subTest(x=str(_x_)):
                _on_, _off_ = self._xyp_(_x_), self._xyp_(_x_, draw_grid=False)
                self.assertGreater(self._gridlines_(_on_), 0, 'the fixture should draw gridlines')
                self.assertEqual(self._gridlines_(_off_), 0)
                # the outline, the end labels and the axis names are all still there
                self.assertIn('stroke-width="0.25"', _off_.svg_context)
                self.assertEqual(self._end_and_name_labels_(_off_), self._end_and_name_labels_(_on_))
                self.assertEqual(len(self._end_and_name_labels_(_off_)), 6)

    def test_the_plot_itself_is_unchanged(self):
        _on_, _off_ = self._xyp_('n'), self._xyp_('n', draw_grid=False)
        self.assertEqual(_on_.plot_origin, _off_.plot_origin)
        self.assertEqual(_on_.plot_size, _off_.plot_size)
        self.assertTrue(_on_.df_pixels.sort('__xpx__', '__ypx__').equals(_off_.df_pixels.sort('__xpx__', '__ypx__')))

    def test_without_context_it_changes_nothing(self):
        for _x_ in ('n', 'cat'):
            with self.subTest(x=_x_):
                self.assertEqual(normalize_svg(self._xyp_(_x_, draw_context=False).svg),
                                 normalize_svg(self._xyp_(_x_, draw_context=False, draw_grid=False).svg))

    def test_the_default_is_on(self):
        self.assertEqual(normalize_svg(self._xyp_('n').svg), normalize_svg(self._xyp_('n', draw_grid=True).svg))

    def test_a_template_carries_it(self):
        _t_ = self._xyp_('n', draw_grid=False)
        self.assertEqual(self._gridlines_(self.p2s.xyp(df=self.df, template=_t_)), 0)


if __name__ == '__main__':
    unittest.main()
