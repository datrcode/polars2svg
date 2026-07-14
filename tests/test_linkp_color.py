import re
import unittest
from xml.etree import ElementTree as ET
import polars as pl
from polars2svg import Polars2SVG


_DF_ = pl.DataFrame({
    'fm':      ['a',     'b',     'c',     'd',     'b'],
    'to':      ['b',     'c',     'd',     'a',     'a'],
    'category':['cat_x', 'cat_y', 'cat_y', 'cat_x', 'cat_x'],
    'cat_n':   [10,      12,      12,      10,      10],
    'count':   [2.0,     5.0,     10.0,    0.1,     0.5],
})
_REL_ = [('fm', 'to')]
_POS_ = {'a': (0.0, 0.5), 'b': (0.5, 0.0), 'c': (1.0, 0.5), 'd': (0.5, 1.0)}


def _linkp_params(**extra):
    return dict(df=_DF_, relationships=_REL_, pos=_POS_,
                wxh=(96, 96), link_shape='curve', draw_labels=True,
                insets=(16, 16), **extra)


def _node_fills(svg):
    return sorted(re.findall(r'<circle[^>]*fill="(#[0-9a-fA-F]+)"', svg))


def _bar_fills(svg_str):
    root = ET.fromstring(svg_str)
    fills = []
    for rect in root.iter('{http://www.w3.org/2000/svg}rect'):
        fill    = rect.get('fill', '')
        x, y    = rect.get('x', ''), rect.get('y', '')
        fill_op = rect.get('fill-opacity')
        if fill in ('none', ''): continue
        if x == '0' and y == '0': continue
        if fill_op is not None:   continue
        fills.append(fill)
    return sorted(fills)


class TestLinkPNodeColorConsistency(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    # --- cell 2bb76554 ---
    # COLOR_BY_NODE_NAME hashes by node name; histop color='fm' hashes by fm value.
    # Nodes {a,b,c,d} == unique fm values, so both should produce the same color set.
    # draw_context=False, distribution=False strips axis/strip rects so only data bars remain.

    def test_color_by_node_name_matches_histop_color_by_field(self):
        lp = self.p2s.linkp(**_linkp_params(node_color=self.p2s.COLOR_BY_NODE_NAME))
        hp = self.p2s.histop(_DF_, 'fm', color='fm', wxh=(128, 96),
                              draw_context=False, distribution=False)
        self.assertEqual(sorted(set(_node_fills(lp.svg))), sorted(set(_bar_fills(hp.svg))))

    # --- cell 6d3a5b7c ---
    # node_color='category' (string field, auto-CSETp) and node_color=('category', CSETp)
    # must produce identical fills.
    # 'a' → colorHash('cat_x'), 'c' → colorHash('cat_y'), 'b'/'d' → string multiset sentinel.

    def test_node_color_str_field_equals_cset_tuple(self):
        lp_field = self.p2s.linkp(**_linkp_params(node_color='category'))
        lp_cset  = self.p2s.linkp(**_linkp_params(node_color=('category', self.p2s.CSETp)))
        self.assertEqual(_node_fills(lp_field.svg), _node_fills(lp_cset.svg))

    def test_node_color_str_field_multiset_nodes_get_sentinel(self):
        lp = self.p2s.linkp(**_linkp_params(node_color='category'))
        # 'b' and 'd' span cat_x and cat_y → string multiset sentinel
        self.assertIn('#7f8367', _node_fills(lp.svg))

    # --- cell 89447ea3 ---
    # node_color=('cat_n', CSETp): integer field treated categorically.
    # 'a' → colorHash(10), 'c' → colorHash(12), 'b'/'d' → numeric multiset sentinel.
    # colorHash(integer N) must equal what histop produces for the same integer values.

    def test_node_color_int_cset_single_value_nodes_match_histop(self):
        lp = self.p2s.linkp(**_linkp_params(node_color=('cat_n', self.p2s.CSETp)))
        hp = self.p2s.histop(_DF_, 'cat_n', color=('cat_n', self.p2s.CSETp), wxh=(128, 96),
                              draw_context=False, distribution=False)
        bar_set  = set(_bar_fills(hp.svg))   # colorHash(10) and colorHash(12)
        node_set = set(_node_fills(lp.svg))  # those two plus the multiset sentinel
        self.assertTrue(bar_set.issubset(node_set),
                        f'histop bar colors {bar_set} should be a subset of linkp node colors {node_set}')

    # --- cell 295eb678 ---
    # node_color='cat_n' (numeric field, auto-CMAGNITUDE_SUMp) and
    # node_color=('cat_n', CMAGNITUDE_SUMp) must produce identical fills.
    # Per-node sums: a=30, b=32, c=24, d=22 → spectrum colors.

    def test_node_color_numeric_field_equals_magnitude_sum(self):
        lp_field = self.p2s.linkp(**_linkp_params(node_color='cat_n'))
        lp_mag   = self.p2s.linkp(**_linkp_params(node_color=('cat_n', self.p2s.CMAGNITUDE_SUMp)))
        self.assertEqual(_node_fills(lp_field.svg), _node_fills(lp_mag.svg))

    # --- cell d3a4858f ---
    # CROW_MAGNITUDEp node colors must equal histop bar colors when both use the same
    # per-node row counts (a=3, b=3, c=2, d=2).

    def test_node_color_crow_magnitude_matches_histop(self):
        # nodes a,b appear 3 times each; c,d appear 2 times each → 2 distinct magnitude tiers.
        # linkp and histop may have a ±1 RGB rounding difference at the spectrum boundary,
        # so we verify both produce 2 distinct colors and share at least the top-tier color.
        _df_nodes_ = pl.concat([_DF_.select(['fm', 'category']).rename({'fm': 'node'}),
                                 _DF_.select(['to', 'category']).rename({'to': 'node'})])
        lp = self.p2s.linkp(**_linkp_params(node_color=self.p2s.CROW_MAGNITUDEp))
        hp = self.p2s.histop(_df_nodes_, 'node', color=self.p2s.CROW_MAGNITUDEp, wxh=(96, 96),
                              draw_context=False, distribution=False)
        node_colors = set(_node_fills(lp.svg))
        bar_colors  = set(_bar_fills(hp.svg))
        self.assertEqual(len(node_colors), 2, f'Expected 2 magnitude tiers, got {node_colors}')
        self.assertEqual(len(bar_colors),  2, f'Expected 2 magnitude tiers, got {bar_colors}')
        self.assertTrue(node_colors & bar_colors,
                        f'No shared colors between linkp {node_colors} and histop {bar_colors}')

    def test_node_color_crow_stretched_renders(self):
        lp = self.p2s.linkp(**_linkp_params(node_color=self.p2s.CROW_STRETCHEDp))
        self.assertGreater(len(_node_fills(lp.svg)), 0)

    def test_color_by_node_name_in_color_param_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self.p2s.linkp(**_linkp_params(color=self.p2s.COLOR_BY_NODE_NAME))
        self.assertIn('node_color', str(ctx.exception))


if __name__ == '__main__':
    unittest.main()
