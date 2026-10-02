"""
Tests that several distribution fields do not repeat the rows the marks read.

xyp flattens one copy of the frame per distribution field when x / y name only one
column: x_distributions=['a', 'b'] made two copies of every row.  The distributions need
both -- each copy carries its own field -- but the dots, lines, colour and size sums,
recordsAt() and the other axis's distribution all read the same frame, and every row
counted once per copy: a row-count colour of 2 read as 4, each line was drawn twice,
recordsAt() returned every record twice, and y_distributions=p2s.ROW_COUNTp counted each
row twice.  The copies a distribution alone needs are now dropped once it is computed.
"""
import re
import unittest

import numpy as np
import polars as pl

from polars2svg import Polars2SVG


def _frame_(n=200, seed=1):
    _rng_ = np.random.default_rng(seed)
    return pl.DataFrame({'x': _rng_.integers(0, 50, n), 'y': _rng_.integers(0, 50, n),
                         'x2': _rng_.integers(0, 50, n),
                         'a': _rng_.integers(0, 9, n), 'b': _rng_.integers(0, 9, n), 'c': _rng_.integers(0, 9, n),
                         'g': [str(_i_) for _i_ in _rng_.integers(0, 3, n)]})


class TestXYpDistributionCopies(unittest.TestCase):

    def setUp(self):
        self.p2s = Polars2SVG()
        self.df  = _frame_()

    def _xyp_(self, **kw):
        _x_ = self.p2s.xyp(self.df, **{'x': 'x', 'y': 'y', 'wxh': (300, 300), **kw})
        _x_._repr_svg_()
        return _x_

    def _pixels_(self, xyp):
        return xyp.df_pixels.select('__xpx__', '__ypx__', '__color_sum__', '__dot_size_sum__').sort('__xpx__', '__ypx__')

    # Same layout on both sides (one field vs two), so the pixels are directly comparable.
    def test_dot_sums_do_not_repeat(self):
        _kw_  = dict(color=self.p2s.CROW_MAGNITUDEp, dot_size='a')
        _one_ = self._xyp_(x_distributions='a', **_kw_)
        _two_ = self._xyp_(x_distributions=['a', 'b'], **_kw_)
        self.assertEqual(len(_two_.df_flat), len(self.df))
        self.assertTrue(self._pixels_(_two_).equals(self._pixels_(_one_)))
        self.assertNotIn('__copy__', _two_.df_flat.columns)

    def test_each_line_is_drawn_once(self):
        _paths_ = lambda x: len(re.findall('<path d=', x.svg_lines))  # noqa: E731
        self.assertEqual(_paths_(self._xyp_(line='g', x_distributions=['a', 'b', 'c'])),
                         _paths_(self._xyp_(line='g', x_distributions='a')))
        self.assertEqual(_paths_(self._xyp_(line='g', x_distributions='a')), 3)

    def test_records_are_returned_once(self):
        _x_ = self._xyp_(x_distributions=['a', 'b'])
        _r_ = _x_.recordsAt((150, 150), shape=self.p2s.SELECT_CIRCLEp, threshold=40)
        self.assertGreater(len(_r_), 0)
        self.assertEqual(len(_r_), len(_r_.unique()))
        _f_ = _x_.filterByRectangle((0, 0, 300, 300))
        self.assertEqual(len(_f_), len(_f_.unique()))

    # Each field's bars are exactly what that field alone draws.
    def test_each_field_keeps_its_own_distribution(self):
        _two_ = self._xyp_(x_distributions=['a', 'b'])
        _k_   = ['__xi_bin__', '__xi_total__']
        for _field_ in ('a', 'b'):
            with self.subTest(field=_field_):
                _mine_ = _two_.df_x_distribution.filter(pl.col('__xdists_color__') == self.p2s.color((_field_,)))
                _solo_ = self._xyp_(x_distributions=_field_).df_x_distribution
                self.assertTrue(_mine_.select(_k_).sort('__xi_bin__').equals(_solo_.select(_k_).sort('__xi_bin__')))

    # The other axis -- counting rows, or one field -- reads each row once.
    def test_the_other_axis_counts_each_row_once(self):
        for _y_spec_ in (self.p2s.ROW_COUNTp, 'c'):
            with self.subTest(y=_y_spec_):
                _two_ = self._xyp_(x_distributions=['a', 'b'], y_distributions=_y_spec_)
                _one_ = self._xyp_(x_distributions='a',        y_distributions=_y_spec_)
                self.assertEqual(_two_.df_y_distribution['__yi_total__'].sum(),
                                 _one_.df_y_distribution['__yi_total__'].sum())
        _rows_ = self._xyp_(x_distributions=['a', 'b'], y_distributions=self.p2s.ROW_COUNTp)
        self.assertEqual(_rows_.df_y_distribution['__yi_total__'].sum(), len(self.df))

    def test_both_axes_with_two_fields(self):
        _x_ = self._xyp_(x_distributions=['a', 'b'], y_distributions=['b', 'c'])
        self.assertEqual(len(_x_.df_flat), len(self.df))
        self.assertEqual(_x_.df_y_distribution['__ydists_color__'].n_unique(), 2)

    # Two x columns are two sets of marks, not a repeat: both copies stay.
    def test_two_x_columns_keep_both_copies(self):
        _x_ = self._xyp_(x=['x', 'x2'], x_distributions='a')
        self.assertEqual(len(_x_.df_flat), 2 * len(self.df))
        self.assertNotIn('__copy__', _x_.df_flat.columns)


if __name__ == '__main__':
    unittest.main()
