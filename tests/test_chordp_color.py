import re
import unittest
import polars as pl
from polars2svg import Polars2SVG

# Same DF used in test_linkp_color.py so cross-component comparisons are valid
_DF_ = pl.DataFrame({
    'fm':      ['a',     'b',     'c',     'd',     'b'],
    'to':      ['b',     'c',     'd',     'a',     'a'],
    'category':['cat_x', 'cat_y', 'cat_y', 'cat_x', 'cat_x'],
    'cat_n':   [10,      12,      12,      10,      10],
    'count':   [2.0,     5.0,     10.0,    0.1,     0.5],
})
_REL_ = [('fm', 'to')]
_POS_ = {'a': (0.0, 0.5), 'b': (0.5, 0.0), 'c': (1.0, 0.5), 'd': (0.5, 1.0)}


def _cp(**extra):
    p2s = Polars2SVG()
    return p2s.chordp(df=_DF_, relationships=_REL_, **extra)


def _lp(**extra):
    p2s = Polars2SVG()
    return p2s.linkp(df=_DF_, relationships=_REL_, pos=_POS_,
                     wxh=(96, 96), link_shape='curve', insets=(16, 16), **extra)


def _node_fills_linkp(svg):
    return sorted(re.findall(r'<circle[^>]*fill="(#[0-9a-fA-F]+)"', svg))


class TestChordPColorModes(unittest.TestCase):
    """Smoke test: every color mode renders without error and produces an SVG."""

    def setUp(self):
        self.p2s = Polars2SVG()

    def _ok(self, cp):
        self.assertIn('<svg', cp.svg)
        self.assertIn('<path', cp.svg)

    # --- node_color modes ---

    def test_node_color_none(self):
        self._ok(_cp(node_color=None))

    def test_node_color_hex(self):
        cp = _cp(node_color='#ff0000')
        self.assertIn('#ff0000', cp.svg)

    def test_node_color_color_by_node_name(self):
        self._ok(_cp(node_color=self.p2s.COLOR_BY_NODE_NAME))

    def test_node_color_str_field(self):
        self._ok(_cp(node_color='category'))

    def test_node_color_cset_tuple(self):
        self._ok(_cp(node_color=('category', self.p2s.CSETp)))

    def test_node_color_numeric_field(self):
        self._ok(_cp(node_color='cat_n'))

    def test_node_color_magnitude_sum(self):
        self._ok(_cp(node_color=('cat_n', self.p2s.CMAGNITUDE_SUMp)))

    def test_node_color_magnitude_min(self):
        self._ok(_cp(node_color=('cat_n', self.p2s.CMAGNITUDE_MINp)))

    def test_node_color_magnitude_max(self):
        self._ok(_cp(node_color=('cat_n', self.p2s.CMAGNITUDE_MAXp)))

    def test_node_color_magnitude_mean(self):
        self._ok(_cp(node_color=('cat_n', self.p2s.CMAGNITUDE_MEANp)))

    def test_node_color_magnitude_median(self):
        self._ok(_cp(node_color=('cat_n', self.p2s.CMAGNITUDE_MEDIANp)))

    def test_node_color_stretched_sum(self):
        self._ok(_cp(node_color=('cat_n', self.p2s.CSTRETCHED_SUMp)))

    def test_node_color_crow_magnitude(self):
        self._ok(_cp(node_color=self.p2s.CROW_MAGNITUDEp))

    def test_node_color_crow_stretched(self):
        self._ok(_cp(node_color=self.p2s.CROW_STRETCHEDp))

    def test_node_color_cset_magnitude(self):
        self._ok(_cp(node_color=('category', self.p2s.CSET_MAGNITUDEp)))

    def test_node_color_cset_stretched(self):
        self._ok(_cp(node_color=('category', self.p2s.CSET_STRETCHEDp)))

    def test_node_color_dict(self):
        cp = _cp(node_color={'a': '#ff0000', 'b': '#00ff00'})
        self.assertIn('#ff0000', cp.svg)
        self.assertIn('#00ff00', cp.svg)

    # --- color (link) modes ---

    def test_link_color_none(self):
        self._ok(_cp(color=None))

    def test_link_color_hex(self):
        cp = _cp(color='#00ff00')
        self.assertIn('#00ff00', cp.svg)

    def test_link_color_src(self):
        self._ok(_cp(color='src'))

    def test_link_color_dst(self):
        self._ok(_cp(color='dst'))

    def test_link_color_str_field(self):
        self._ok(_cp(color='category'))

    def test_link_color_cset_tuple(self):
        self._ok(_cp(color=('category', self.p2s.CSETp)))

    def test_link_color_numeric_field(self):
        self._ok(_cp(color='cat_n'))

    def test_link_color_crow_magnitude(self):
        self._ok(_cp(color=self.p2s.CROW_MAGNITUDEp))

    def test_link_color_crow_stretched(self):
        self._ok(_cp(color=self.p2s.CROW_STRETCHEDp))

    def test_link_color_magnitude_sum(self):
        self._ok(_cp(color=('cat_n', self.p2s.CMAGNITUDE_SUMp)))


class TestChordPColorConsistency(unittest.TestCase):
    """Verify that color modes produce correct SVG output and color_nodes_final values."""

    def setUp(self):
        self.p2s = Polars2SVG()

    def test_node_color_fixed_hex_appears_in_svg(self):
        cp = _cp(node_color='#ab1234')
        self.assertIn('#ab1234', cp.svg)

    def test_node_color_by_name_distinct_colors(self):
        # Each node gets a unique hash color when COLOR_BY_NODE_NAME is used
        cp = _cp(node_color=self.p2s.COLOR_BY_NODE_NAME)
        colors = set(cp.color_nodes_final.values())
        self.assertEqual(len(colors), len(cp.color_nodes_final),
                         f'Expected distinct colors per node, got {cp.color_nodes_final}')

    def test_node_color_by_name_covers_all_nodes(self):
        cp = _cp(node_color=self.p2s.COLOR_BY_NODE_NAME)
        self.assertEqual(set(cp.color_nodes_final.keys()), {'a', 'b', 'c', 'd'})

    def test_link_color_hex_in_svg(self):
        cp = _cp(color='#aabbcc')
        self.assertIn('#aabbcc', cp.svg)

    def test_link_color_field_renders_without_error(self):
        cp = _cp(color='category')
        self.assertIn('<path', cp.svg)

    def test_invalid_node_color_raises(self):
        with self.assertRaises(ValueError):
            _cp(node_color='nonexistent_field')

    def test_color_by_node_name_raises_for_color_param(self):
        # color=p2s.COLOR_BY_NODE_NAME is invalid (only valid for node_color)
        with self.assertRaises(ValueError):
            _cp(color=self.p2s.COLOR_BY_NODE_NAME)


class TestChordPLinkPColorInterchangeable(unittest.TestCase):
    """Cross-component: same color spec → same node color results in chordp and linkp."""

    def setUp(self):
        self.p2s = Polars2SVG()

    def test_color_by_node_name_same_color_set(self):
        # Both components hash node names the same way → identical color sets
        cp = _cp(node_color=self.p2s.COLOR_BY_NODE_NAME)
        lp = _lp(node_color=self.p2s.COLOR_BY_NODE_NAME)
        chord_colors = set(cp.color_nodes_final.values())
        linkp_colors = set(_node_fills_linkp(lp.svg))
        self.assertEqual(chord_colors, linkp_colors,
                         f'chordp colors {chord_colors} != linkp colors {linkp_colors}')

    def test_color_by_node_name_node_count(self):
        cp = _cp(node_color=self.p2s.COLOR_BY_NODE_NAME)
        lp = _lp(node_color=self.p2s.COLOR_BY_NODE_NAME)
        self.assertEqual(len(cp.color_nodes_final), len(_node_fills_linkp(lp.svg)))


if __name__ == '__main__':
    unittest.main()
