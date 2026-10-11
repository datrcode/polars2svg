import unittest
import polars as pl
from polars2svg import Polars2SVG

from svg_test_utils import assert_svg_matches_golden, assert_image_matches_golden


# DT's shelter example (2026-10-11).  Largest count per slice across the shelters:
# dog 8 (b), cat 6 (c), ferret 2 (b) -- the SM_COUNT base without an "all" panel.
# All rows: cat 14, dog 14, ferret 2 -- the base with one.
_SHELTERS_ = pl.DataFrame({
    'shelter': ['shelter a', 'shelter a', 'shelter b', 'shelter b', 'shelter b', 'shelter c'],
    'type':    ['cat',       'dog',       'cat',       'dog',       'ferret',    'cat'],
    'number':  [5,           6,           3,           8,           2,           6],
})


class TestPiepSmallpGolden(unittest.TestCase):
    '''Golden files for piep small multiples under sm_shared={SM_COUNT}: one faded base pie
    in every panel, each panel filling its share of every slice.  UPDATE_GOLDEN=1 rewrites.
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def _render_(self, style, include_all, wxh, **kwargs):
        tmpl = self.p2s.piep(_SHELTERS_, 'type', count='number', color='type', style=style,
                             sm_shared={self.p2s.SM_COUNT}, **kwargs)
        return self.p2s.smallp(_SHELTERS_, 'shelter', tmpl, include_all=include_all, wxh=wxh)

    def test_sm_count_pie(self):
        _r_ = self._render_(self.p2s.PIEp, False, (520, 180))
        assert_svg_matches_golden(_r_.svg, 'piep_smallp_sm_count')
        assert_image_matches_golden(_r_.svg, 'piep_smallp_sm_count')

    def test_sm_count_pie_include_all(self):
        # the notebook's call: with the "all" panel, the base is all rows
        _r_ = self._render_(self.p2s.PIEp, True, (800, 175), legend=True)
        assert_svg_matches_golden(_r_.svg, 'piep_smallp_sm_count_include_all')
        assert_image_matches_golden(_r_.svg, 'piep_smallp_sm_count_include_all')

    def test_sm_count_donut(self):
        _r_ = self._render_(self.p2s.DONUTp, False, (520, 180))
        assert_svg_matches_golden(_r_.svg, 'piep_smallp_sm_count_donut')
        assert_image_matches_golden(_r_.svg, 'piep_smallp_sm_count_donut')


if __name__ == '__main__':
    unittest.main()
