"""
Tests for what x_distributions= / y_distributions= accept.

Two shorthands spell out to an ordinary spec -- True is p2s.ROW_COUNTp and an int n is
[p2s.ROW_COUNTp, n] -- and False means None.  A spec that parses but cannot mean anything
is rejected with a message naming the parameter and what is wrong with it; several of
these used to surface as "'int' object is not iterable", a ZeroDivisionError, or polars
failing to find the internal column __xdists__.
"""
import unittest

import polars as pl

from polars2svg import Polars2SVG
from polars2svg.interactive_render_rows import XYpRenderRows
from svg_test_utils import normalize_svg


class TestXYpDistributionSpecs(unittest.TestCase):

    def setUp(self):
        self.p2s = Polars2SVG()
        self.df  = pl.DataFrame({'x': [1, 2, 3, 4, 5, 6, 7, 8], 'y': [3, 1, 4, 1, 5, 9, 2, 6],
                                 'a': list('aabbccdd'), 'n': [1, 2, 3, 4, 5, 6, 7, 8]})

    def _xyp_(self, **kw):
        _x_ = self.p2s.xyp(self.df, 'x', 'y', wxh=(200, 160), **kw)
        _x_._repr_svg_()
        return _x_

    def _svg_(self, **kw):
        return normalize_svg(self._xyp_(**kw).svg)

    # ── the shorthands ───────────────────────────────────────────────────────

    def test_true_is_row_count(self):
        _x_ = self._xyp_(x_distributions=True, y_distributions=True)
        self.assertIs(_x_.x_distributions, self.p2s.ROW_COUNTp)
        self.assertEqual(normalize_svg(_x_.svg),
                         self._svg_(x_distributions=self.p2s.ROW_COUNTp, y_distributions=self.p2s.ROW_COUNTp))

    def test_int_is_row_count_in_that_many_bins(self):
        _x_ = self._xyp_(x_distributions=10, y_distributions=3)
        self.assertEqual(_x_.x_distributions, [self.p2s.ROW_COUNTp, 10])
        self.assertEqual(_x_.x_distributions_clean['bins'], [10])
        self.assertEqual(_x_.y_distributions_clean['bins'], [3])
        self.assertEqual(normalize_svg(_x_.svg),
                         self._svg_(x_distributions=[self.p2s.ROW_COUNTp, 10], y_distributions=[self.p2s.ROW_COUNTp, 3]))

    def test_false_is_none(self):
        _x_ = self._xyp_(x_distributions=False)
        self.assertIsNone(_x_.x_distributions)
        self.assertIsNone(_x_.x_distributions_clean)
        self.assertEqual(normalize_svg(_x_.svg), self._svg_())

    # A template holds the spelled-out spec, so a re-render and xypi's panel read that.
    def test_template_and_panel_see_the_spelled_out_spec(self):
        _t_ = self._xyp_(x_distributions=10)
        self.assertEqual(normalize_svg(self.p2s.xyp(df=self.df, template=_t_).svg), normalize_svg(_t_.svg))
        _s_ = XYpRenderRows(_t_).initial_settings()
        self.assertEqual(_s_['distributions'], 'x')
        self.assertEqual(_s_['x_bins'], 'as built')
        self.assertEqual(XYpRenderRows(self._xyp_(x_distributions=True)).initial_settings()['x_bins'], 'auto')

    # The existing forms are unchanged.
    def test_existing_forms(self):
        for _spec_ in (self.p2s.ROW_COUNTp, 'n', ['n', self.p2s.SETp], ('a', self.p2s.DISTRIBUTION_OUTSIDEp, 16),
                       [self.p2s.ROW_COUNTp, 12, 0.5, '#ff0000', self.p2s.DISTRIBUTION_INSIDEp], [('a', 'n')],
                       [self.p2s.ROW_COUNTp, 1.0]):
            with self.subTest(spec=_spec_):
                self.assertIsNotNone(self._xyp_(x_distributions=_spec_).x_distributions_clean)

    # ── rejections ───────────────────────────────────────────────────────────

    def assertRejected(self, spec, error, pattern, axis='x'):
        with self.assertRaisesRegex(error, pattern):
            self._xyp_(**{f'{axis}_distributions': spec})

    def test_not_a_spec(self):
        self.assertRejected(0.3, TypeError, r'x_distributions=0\.3 is not a distribution spec')
        self.assertRejected({}, TypeError, r'y_distributions=\{\} is not a distribution spec', axis='y')

    def test_nothing_to_measure(self):
        for _spec_ in ([10], [], '#ff0000', self.p2s.DISTRIBUTION_INSIDEp, self.p2s.SETp):
            with self.subTest(spec=_spec_):
                self.assertRejected(_spec_, ValueError, 'nothing to measure')

    def test_row_count_and_a_column_are_ambiguous(self):
        self.assertRejected([self.p2s.ROW_COUNTp, 'n'], ValueError, 'ambiguous')

    def test_bin_count(self):
        self.assertRejected(0, ValueError, r'x_distributions=0: the bin count must be at least 1')
        self.assertRejected([self.p2s.ROW_COUNTp, -2], ValueError, 'the bin count must be at least 1')
        self.assertRejected([self.p2s.ROW_COUNTp, True], TypeError, 'give the bin count as an int')

    def test_height(self):
        for _h_ in (0.0, 1.5, -0.2):
            with self.subTest(height=_h_):
                self.assertRejected([self.p2s.ROW_COUNTp, _h_], ValueError, r'the height is a fraction in \(0, 1\]')

    def test_foreign_setting(self):
        self.assertRejected([self.p2s.ROW_COUNTp, self.p2s.LINEWIDTH_DOTSIZE_MEAN], ValueError,
                            'LINEWIDTH_DOTSIZE_MEAN is not a distribution setting')

    def test_missing_column_names_the_parameter(self):
        self.assertRejected('nope', ValueError, r'nope \(y_distributions\)', axis='y')


if __name__ == '__main__':
    unittest.main()
