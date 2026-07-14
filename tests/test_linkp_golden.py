import unittest
import polars as pl
from polars2svg import Polars2SVG

from svg_test_utils import assert_svg_matches_golden, assert_image_matches_golden


_DF_ = pl.DataFrame({
    'fm':      ['a',     'b',     'c',     'd',     'b'],
    'to':      ['b',     'c',     'd',     'a',     'a'],
    'category':['cat_x', 'cat_y', 'cat_y', 'cat_x', 'cat_x'],
    'cat_n':   [10,      12,      12,      10,      10],
    'count':   [2.0,     5.0,     10.0,    0.1,     0.5],
})
_REL_ = [('fm', 'to')]
_POS_ = {'a': (0.0, 0.5), 'b': (0.5, 0.0), 'c': (1.0, 0.5), 'd': (0.5, 1.0)}


def _params(**extra):
    return dict(df=_DF_, relationships=_REL_, pos=_POS_,
                wxh=(96, 96), link_shape='curve', draw_labels=True,
                insets=(16, 16), **extra)


class TestLinkPNodeColorGolden(unittest.TestCase):
    '''Golden-file regression tests for LinkP node_color SVG output.

    First run (or UPDATE_GOLDEN=1): golden files are written, tests pass.
    Subsequent runs: SVG must match the golden exactly.
    To regenerate after an intentional visual change: UPDATE_GOLDEN=1 pytest
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    # --- cell 9b4c3ea9 ---

    def test_node_color_none(self):
        lp = self.p2s.linkp(**_params(node_color=None))
        assert_svg_matches_golden(lp.svg, 'linkp_node_color_none')
        assert_image_matches_golden(lp.svg, 'linkp_node_color_none')

    # --- cell 424d568d ---

    def test_node_color_hex(self):
        lp = self.p2s.linkp(**_params(node_color='#ff0000'))
        assert_svg_matches_golden(lp.svg, 'linkp_node_color_hex')
        assert_image_matches_golden(lp.svg, 'linkp_node_color_hex')

    # --- cell 5f5c2877 ---

    def test_node_color_dict_single(self):
        lp = self.p2s.linkp(**_params(node_color={'a': '#ff0000'}))
        assert_svg_matches_golden(lp.svg, 'linkp_node_color_dict_single')
        assert_image_matches_golden(lp.svg, 'linkp_node_color_dict_single')

    def test_node_color_dict_two(self):
        lp = self.p2s.linkp(**_params(node_color={'b': '#ff0000', 'd': '#00ff00'}))
        assert_svg_matches_golden(lp.svg, 'linkp_node_color_dict_two')
        assert_image_matches_golden(lp.svg, 'linkp_node_color_dict_two')

    def test_node_color_dict_three(self):
        lp = self.p2s.linkp(**_params(node_color={'c': '#ff0000', 'd': '#999'}))
        assert_svg_matches_golden(lp.svg, 'linkp_node_color_dict_three')
        assert_image_matches_golden(lp.svg, 'linkp_node_color_dict_three')

    def test_node_color_dict_unknown_key(self):
        lp = self.p2s.linkp(**_params(node_color={'c': '#ff0000', 'd': '#999', 'z': '#fff'}))
        assert_svg_matches_golden(lp.svg, 'linkp_node_color_dict_unknown_key')
        assert_image_matches_golden(lp.svg, 'linkp_node_color_dict_unknown_key')


if __name__ == '__main__':
    unittest.main()
