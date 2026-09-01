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
# 'c' and 'd' share a position, so they collapse to one screen pixel -> cloud icon
_POS_COLLAPSED_ = {'a': (0.0, 0.5), 'b': (0.5, 0.0), 'c': (1.0, 0.5), 'd': (1.0, 0.5)}


def _params(pos=_POS_, **extra):
    return dict(df=_DF_, relationships=_REL_, pos=pos,
                wxh=(96, 96), link_shape='curve', draw_node_labels=True,
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


class TestLinkPCollapsedNodeGolden(unittest.TestCase):
    '''Golden-file regression test for collapsed nodes (PLANNING.md S5).

    Two nodes share a world position, so they land on one screen pixel and render
    as the shared cloud icon instead of two circles.  This is the only golden that
    carries the `<defs>` cloud block -- every other linkp golden must not have it.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def test_collapsed_nodes(self):
        lp = self.p2s.linkp(**_params(pos=_POS_COLLAPSED_))
        self.assertIn('<use href="#cloud"', lp.svg)   # the golden is only meaningful if a cloud is drawn
        assert_svg_matches_golden(lp.svg, 'linkp_collapsed_nodes')
        assert_image_matches_golden(lp.svg, 'linkp_collapsed_nodes')


class TestLinkPBackgroundGolden(unittest.TestCase):
    '''Golden-file regression test for background records (PLANNING.md §9.1).

    Mixes a bare descriptor (inherits every background_* parameter) with records that
    say things the parameters cannot: fill off, a translucent dashed stroke, and a
    drawn label that differs from the dict key.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def test_background_records(self):
        lp = self.p2s.linkp(**_params(
            background={
                'zone':        [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)],
                'flow 1':      self.p2s.bgShape('M 0.1 0.9 L 0.5 0.3 L 0.9 0.9',
                                                fill=None, stroke='#204080',
                                                stroke_opacity=0.35, stroke_width=3.0, dash='4 2'),
                'flow 1 head': self.p2s.bgShape('<circle cx="0.9" cy="0.9" r="0.06" />',
                                                fill='#204080', fill_opacity=0.6,
                                                stroke=None, label='head'),
            },
            background_fill='#aabbcc', background_opacity=0.35,
            background_label_color='#303030'))
        assert_svg_matches_golden(lp.svg, 'linkp_background_records')
        assert_image_matches_golden(lp.svg, 'linkp_background_records')


class TestLinkPFlowFieldBackgroundGolden(unittest.TestCase):
    '''Golden-file regression test for FlowFieldBackground cells.

    The producer's records reach linkp with no background_* arguments at all, so
    this pins the whole path: layer decomposition -> world-coordinate glyph paths
    -> record appearance -> rendered SVG.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def _cells(self, **kw):
        from polars2svg import FlowFieldBackground
        return FlowFieldBackground(_DF_, _REL_, pos=_POS_, grid_res=12, **kw).cells()

    def test_flow_field_arrows(self):
        lp = self.p2s.linkp(**_params(background=self._cells(k_layers=2)))
        assert_svg_matches_golden(lp.svg, 'linkp_flow_field_arrows')
        assert_image_matches_golden(lp.svg, 'linkp_flow_field_arrows')

    def test_flow_field_streamlines(self):
        lp = self.p2s.linkp(**_params(background=self._cells(k_layers=2, glyph='streamline')))
        assert_svg_matches_golden(lp.svg, 'linkp_flow_field_streamlines')
        assert_image_matches_golden(lp.svg, 'linkp_flow_field_streamlines')


if __name__ == '__main__':
    unittest.main()
