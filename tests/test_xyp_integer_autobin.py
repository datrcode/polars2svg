"""
Tests for xyp's automatic distribution bins on an axis of whole numbers.

The automatic bin count is pixel-derived (the plot extent / 6), and on an axis of whole
numbers a bin width that is not itself a whole number leaves some bins holding no integer
and others holding two: a smooth distribution draws as a comb of spikes and gaps (user
feedback 2026-09-27, books_read_per_year in the millionaire-habits survey -- 27 values in
82 bins, 55 of them empty).  Autobin now gives each bin a whole number of integers -- one
per bin when they fit -- and draws each bar under the values it counts.
"""
import unittest

import polars as pl

from polars2svg import Polars2SVG
from polars2svg.xyp import _intBins_


class TestIntBins(unittest.TestCase):

    def test_one_integer_per_bin_when_they_fit(self):
        self.assertEqual(_intBins_(0, 26, 82), (0, 1, 27))

    def test_the_fewest_integers_per_bin_that_fit(self):
        _first_, _per_, _n_ = _intBins_(0, 999, 42)
        self.assertEqual((_first_, _per_), (0, 24))
        self.assertLessEqual(_n_, 42)
        self.assertGreater(_n_ * _per_, 999)
        self.assertEqual(_intBins_(0, 999, 42), (0, 24, 42))

    def test_a_window_with_fractional_ends(self):
        self.assertEqual(_intBins_(0.5, 10.5, 50), (1, 1, 10))   # the integers 1..10

    def test_a_single_value(self):
        self.assertEqual(_intBins_(7, 7, 30), (7, 1, 1))


class TestXYpIntegerAutobin(unittest.TestCase):

    def setUp(self):
        self.p2s = Polars2SVG()

    def _xyp_(self, df, x='v', **kw):
        _x_ = self.p2s.xyp(df, x, 'y', dot_size=None, wxh=(512, 120),
                           x_distributions=[self.p2s.ROW_COUNTp, self.p2s.DISTRIBUTION_INSIDEp], **kw)
        _x_._repr_svg_()
        return _x_

    def _counts_(self, xyp):
        return xyp.df_x_distribution.sort('__xi_bin__')['__xi_total__'].to_list()

    def _survey_(self, dtype=pl.Int64):
        # a bell over 0..26, every value present: what a count answered in a survey looks like
        _vals_ = [_v_ for _v_ in range(27) for _ in range(1 + min(_v_, 26 - _v_))]
        return pl.DataFrame({'v': pl.Series(_vals_).cast(dtype), 'y': [1.0] * len(_vals_)})

    # The regression: every bin holds exactly one value, so none is empty.
    def test_every_bin_holds_one_value(self):
        _x_ = self._xyp_(self._survey_())
        self.assertEqual(_x_.x_distributions_clean['bins'], [27])
        self.assertEqual(self._counts_(_x_), [1.0 + min(_v_, 26 - _v_) for _v_ in range(27)])

    # A survey count read from CSV is often a float column of whole numbers.
    def test_a_float_column_of_whole_numbers_counts_as_integers(self):
        _int_, _flt_ = self._xyp_(self._survey_()), self._xyp_(self._survey_(pl.Float64))
        self.assertEqual(_flt_.x_distributions_clean['bins'], [27])
        self.assertEqual(self._counts_(_flt_), self._counts_(_int_))

    # Each bar sits under the dots it counts: value v spans v-1/2 .. v+1/2 of the axis, so the
    # two end bars are half bars.
    def test_each_bar_sits_under_its_value(self):
        _x_  = self._xyp_(self._survey_())
        _lo_, _hi_ = _x_.x_effective_range
        _d_  = _x_.df_x_distribution.sort('__xi_bin__')
        for _v_, _mn_, _mx_ in zip(range(27), _d_['__xdists_xi_min__'], _d_['__xdists_xi_max__']):
            _centre_ = (_v_ - _lo_) / (_hi_ - _lo_)
            with self.subTest(value=_v_):
                self.assertLessEqual(_mn_, _centre_)
                self.assertGreaterEqual(_mx_, _centre_)
        self.assertAlmostEqual(_d_['__xdists_xi_min__'][0], 0.0)
        self.assertAlmostEqual(_d_['__xdists_xi_max__'][26], 1.0)

    def test_many_integers_share_bins_evenly(self):
        _df_ = pl.DataFrame({'v': list(range(1000)), 'y': [1.0] * 1000})
        _x_  = self._xyp_(_df_)
        _n_  = _x_.x_distributions_clean['bins'][0]
        _counts_ = self._counts_(_x_)
        self.assertLessEqual(_n_, 512 // 6)
        self.assertEqual(len(set(_counts_[:-1])), 1, 'every bin but the last holds the same count of integers')
        self.assertEqual(sum(_counts_), 1000)

    def test_a_categorical_axis_gets_one_bar_per_category(self):
        _df_ = pl.DataFrame({'c': ['ant', 'bee', 'cat', 'dog', 'eel'] * 3 + ['ant'], 'y': [1.0] * 16})
        _x_  = self._xyp_(_df_, x='c')
        self.assertEqual(_x_.x_distributions_clean['bins'], [5])
        self.assertEqual(self._counts_(_x_), [4.0, 3.0, 3.0, 3.0, 3.0])

    # What does not change.
    def test_fractional_values_keep_the_pixel_count(self):
        _df_ = pl.DataFrame({'v': [_i_ * 0.37 for _i_ in range(60)], 'y': [1.0] * 60})
        _x_  = self._xyp_(_df_)
        self.assertEqual(_x_.x_distributions_clean['bins'], [_x_.plot_size[0] // 6])

    def test_an_explicit_bin_count_is_kept(self):
        _x_ = self.p2s.xyp(self._survey_(), 'v', 'y', dot_size=None, wxh=(512, 120),
                           x_distributions=[self.p2s.ROW_COUNTp, 10])
        _x_._repr_svg_()
        self.assertEqual(_x_.x_distributions_clean['bins'], [10])
        self.assertEqual(len(self._counts_(_x_)), 10)

    # xyp widens a one-value axis to (v, v+1), so the window holds two integers and the
    # second has no rows -- and the first bar is the half bar under the dots at the edge.
    def test_a_single_value(self):
        _x_ = self._xyp_(pl.DataFrame({'v': [4, 4, 4], 'y': [1.0, 2.0, 3.0]}))
        self.assertEqual(_x_.x_effective_range, (4, 5))
        self.assertEqual(self._counts_(_x_), [3.0, 0.0])
        self.assertEqual(_x_.df_x_distribution.sort('__xi_bin__')['__xdists_xi_max__'][0], 0.5)


if __name__ == '__main__':
    unittest.main()
