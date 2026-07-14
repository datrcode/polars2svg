#
# test_legend_golden.py
#
# Golden-file regression tests for the legend= feature: categorical swatch lists and colorbars on the two reference components
# (xyp, histop), covering both a vertical strip ('right') and a horizontal one
# ('bottom').  Existing component goldens are untouched -- legend=False is the
# default and renders byte-identically to the pre-legend output.
#
# First run (or UPDATE_GOLDEN=1): golden files are written, tests pass.
# Subsequent runs: SVG must match the golden exactly.
#
import unittest

import polars as pl

from polars2svg import Polars2SVG

from svg_test_utils import assert_svg_matches_golden, assert_image_matches_golden

_XY_DF_ = pl.DataFrame({
    'x':   [1, 2, 3, 4, 5, 6, 7, 8] * 4,
    'y':   [2, 4, 1, 8, 5, 7, 3, 6] * 4,
    'cat': ['a', 'b', 'c', 'a', 'b', 'c', 'd', 'e'] * 4,
    'val': [1.0, 2.5, 3.2, 0.5, 4.4, 2.2, 1.1, 9.9] * 4,
})

# Distinct per-bin row counts: histop breaks count ties arbitrarily (group_by
# order), so equal-frequency bins would make the golden nondeterministic.
_BAR_DF_ = pl.DataFrame({
    'bin': ['alpha'] * 8 + ['beta'] * 6 + ['gamma'] * 4 + ['delta'] * 2,
    'grp': (['x', 'x', 'y', 'z'] * 5),
    'val': ([5.0, 2.0, 8.0, 1.0] * 5),
})


class TestLegendGolden(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def test_xyp_categorical_right(self):
        _r_ = self.p2s.xyp(_XY_DF_, 'x', 'y', color='cat', legend='right', wxh=(256, 192))
        assert_svg_matches_golden(_r_.svg, 'legend_xyp_categorical_right')
        assert_image_matches_golden(_r_.svg, 'legend_xyp_categorical_right')

    def test_xyp_colorbar_left(self):
        _r_ = self.p2s.xyp(_XY_DF_, 'x', 'y', color='val', legend='left', wxh=(256, 192))
        assert_svg_matches_golden(_r_.svg, 'legend_xyp_colorbar_left')
        assert_image_matches_golden(_r_.svg, 'legend_xyp_colorbar_left')

    def test_xyp_colorbar_bottom(self):
        _r_ = self.p2s.xyp(_XY_DF_, 'x', 'y', color=self.p2s.CROW_MAGNITUDEp,
                           legend='bottom', wxh=(256, 192))
        assert_svg_matches_golden(_r_.svg, 'legend_xyp_colorbar_bottom')
        assert_image_matches_golden(_r_.svg, 'legend_xyp_colorbar_bottom')

    def test_xyp_dict_spec_top(self):
        _r_ = self.p2s.xyp(_XY_DF_, 'x', 'y', color='cat',
                           legend={'pos': 'top', 'title': 'Category', 'max_items': 3},
                           wxh=(256, 192))
        assert_svg_matches_golden(_r_.svg, 'legend_xyp_dict_top')
        assert_image_matches_golden(_r_.svg, 'legend_xyp_dict_top')

    def test_histop_stacked_categorical_bottom(self):
        _r_ = self.p2s.histop(_BAR_DF_, 'bin', color='grp', legend='bottom', wxh=(192, 256))
        assert_svg_matches_golden(_r_.svg, 'legend_histop_categorical_bottom')
        assert_image_matches_golden(_r_.svg, 'legend_histop_categorical_bottom')

    def test_histop_colorbar_right(self):
        _r_ = self.p2s.histop(_BAR_DF_, 'bin', color='val', legend='right', wxh=(256, 256))
        assert_svg_matches_golden(_r_.svg, 'legend_histop_colorbar_right')
        assert_image_matches_golden(_r_.svg, 'legend_histop_colorbar_right')


if __name__ == '__main__':
    unittest.main()
