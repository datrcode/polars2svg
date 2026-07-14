import unittest
import polars as pl
from polars2svg import Polars2SVG

from svg_test_utils import assert_svg_matches_golden, assert_image_matches_golden


_DF_ = pl.DataFrame({
    'fm':   ['a', 'b', 'c', 'a', 'd', 'b', 'c', 'a'],
    'to':   ['b', 'a', 'a', 'c', 'a', 'c', 'b', 'd'],
    'time': [1,   1,   1,   2,   2,   2,   3,   3  ],
    'role': ['x', 'y', 'y', 'x', 'y', 'y', 'x', 'x'],
    'w':    [3,   1,   2,   4,   1,   2,   3,   1  ],
})
_RELS_ = [('fm', 'to')]


class TestSpreadLinesPGolden(unittest.TestCase):
    '''Golden-file regression tests for SpreadLinesP SVG output.

    First run (or UPDATE_GOLDEN=1): golden files are written, tests pass.
    Subsequent runs: SVG must match the golden exactly.
    To regenerate after an intentional visual change: UPDATE_GOLDEN=1 pytest
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def test_basic_ego_a(self):
        sp = self.p2s.spreadlinesp(_DF_, _RELS_, ego='a', time='time', wxh=(700, 300))
        assert_svg_matches_golden(sp.svg, 'spreadlinesp_basic_ego_a')
        assert_image_matches_golden(sp.svg, 'spreadlinesp_basic_ego_a')

    def test_node_color_field(self):
        sp = self.p2s.spreadlinesp(_DF_, _RELS_, ego='a', time='time',
                                   node_color='role', wxh=(700, 300))
        assert_svg_matches_golden(sp.svg, 'spreadlinesp_node_color_field')
        assert_image_matches_golden(sp.svg, 'spreadlinesp_node_color_field')

    def test_count_field(self):
        sp = self.p2s.spreadlinesp(_DF_, _RELS_, ego='a', time='time',
                                   count='w', wxh=(700, 300))
        assert_svg_matches_golden(sp.svg, 'spreadlinesp_count_field')
        assert_image_matches_golden(sp.svg, 'spreadlinesp_count_field')

    def test_highlight_nodes(self):
        sp = self.p2s.spreadlinesp(_DF_, _RELS_, ego='a', time='time',
                                   highlight_nodes={'b', 'c'}, wxh=(700, 300))
        assert_svg_matches_golden(sp.svg, 'spreadlinesp_highlight_nodes')
        assert_image_matches_golden(sp.svg, 'spreadlinesp_highlight_nodes')


if __name__ == '__main__':
    unittest.main()
