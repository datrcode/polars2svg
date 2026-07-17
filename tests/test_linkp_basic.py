import unittest
import polars as pl
from polars2svg import Polars2SVG
from svg_test_utils import assert_timing_metrics_populated


def _make_df():
    return pl.DataFrame({
        'fm':       ['a', 'b', 'c', 'a', 'b'],
        'to':       ['b', 'c', 'a', 'c', 'a'],
        'category': ['x', 'y', 'x', 'y', 'x'],
        'weight':   [1,   3,   2,   1,   4  ],
    })

def _make_pos():
    return {'a': [0, 0], 'b': [1, 0], 'c': [0.5, 0.866]}

def _rels():
    return [('fm', 'to')]


class TestLinkPBasic(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def test_basic_render(self):
        lp = self.p2s.linkp(_make_df(), relationships=_rels(), pos=_make_pos())
        self.assertIn('<circle', lp.svg)
        self.assertIn('<line',   lp.svg)
        self.assertGreater(len(lp.svg), 200)

    def test_color_field(self):
        lp = self.p2s.linkp(_make_df(), relationships=_rels(), pos=_make_pos(), color='category')
        self.assertIn('<circle', lp.svg)
        self.assertIn('<line',   lp.svg)

    def test_node_color_fixed(self):
        lp = self.p2s.linkp(_make_df(), relationships=_rels(), pos=_make_pos(),
                            node_color='#ff0000')
        self.assertIn('#ff0000', lp.svg)

    def test_color_fixed_for_links(self):
        lp = self.p2s.linkp(_make_df(), relationships=_rels(), pos=_make_pos(),
                            color='#00ff00')
        self.assertIn('#00ff00', lp.svg)

    def test_link_size_vary_with_count(self):
        lp = self.p2s.linkp(_make_df(), relationships=_rels(), pos=_make_pos(),
                            link_size='vary', count='weight')
        self.assertIn('<line', lp.svg)

    def test_link_size_vary_count_set(self):
        lp = self.p2s.linkp(_make_df(), relationships=_rels(), pos=_make_pos(),
                            link_size='vary', count=('category', self.p2s.SETp))
        self.assertIn('<line', lp.svg)

    def test_link_size_vary_count_row_count(self):
        lp = self.p2s.linkp(_make_df(), relationships=_rels(), pos=_make_pos(),
                            link_size='vary', count=self.p2s.ROW_COUNTp)
        self.assertIn('<line', lp.svg)

    def test_node_size_vary(self):
        lp = self.p2s.linkp(_make_df(), relationships=_rels(), pos=_make_pos(),
                            node_size='vary')
        self.assertIn('<circle', lp.svg)

    def test_link_shape_curve(self):
        lp = self.p2s.linkp(_make_df(), relationships=_rels(), pos=_make_pos(),
                            link_shape='curve')
        self.assertIn('<path', lp.svg)
        self.assertNotIn('<line', lp.svg)

    def test_link_shape_flowmap(self):
        lp = self.p2s.linkp(_make_df(), relationships=_rels(), pos=_make_pos(),
                            link_shape='flowmap')
        self.assertIn('<path', lp.svg)
        self.assertNotIn('<line', lp.svg)

    def test_link_shape_flowmap_deterministic(self):
        # compare the link path geometry only (the full SVG embeds a random id)
        import re
        _mk_ = lambda: sorted(re.findall(r'<path d="[^"]*"',
                                         self.p2s.linkp(_make_df(), relationships=_rels(),
                                                        pos=_make_pos(),
                                                        link_shape='flowmap').svg))
        self.assertEqual(_mk_(), _mk_())

    def test_link_shape_flowmap_multiple_relationships(self):
        df = pl.DataFrame({'fm': ['a', 'b'], 'to': ['b', 'c'], 'p': ['c', 'a'], 'q': ['d', 'e']})
        pos = {'a': [0, 0], 'b': [1, 0], 'c': [0.5, 0.866], 'd': [0.2, 0.5], 'e': [0.8, 0.5]}
        lp = self.p2s.linkp(df, relationships=[('fm', 'to'), ('p', 'q')], pos=pos,
                            link_shape='flowmap')
        self.assertIn('<path', lp.svg)

    def test_link_shape_flowmap_link_size_vary(self):
        lp = self.p2s.linkp(_make_df(), relationships=_rels(), pos=_make_pos(),
                            link_shape='flowmap', link_size='vary')
        self.assertIn('stroke-width=', lp.svg)
        self.assertIn('<path', lp.svg)

    def test_link_arrows_off_by_default(self):
        lp = self.p2s.linkp(_make_df(), relationships=_rels(), pos=_make_pos())
        self.assertNotIn('<polygon', lp.svg)

    def test_link_arrows_all_shapes(self):
        for shape in ('line', 'curve', 'flowmap'):
            lp = self.p2s.linkp(_make_df(), relationships=_rels(), pos=_make_pos(),
                                link_shape=shape, link_arrows=True)
            self.assertIn('<polygon', lp.svg, msg=f'link_shape={shape!r}')

    def test_link_arrows_with_vary_size(self):
        lp = self.p2s.linkp(_make_df(), relationships=_rels(), pos=_make_pos(),
                            link_arrows=True, link_size='vary')
        self.assertIn('<polygon', lp.svg)

    def test_link_arrows_zero_length_link_safe(self):
        # b->b collapses to a point: no arrow for it, but the render survives
        df  = pl.DataFrame({'fm': ['a', 'a', 'b'], 'to': ['b', 'c', 'b']})
        pos = {'a': [0, 0], 'b': [1, 0], 'c': [0.5, 0.9]}
        lp  = self.p2s.linkp(df, relationships=[('fm', 'to')], pos=pos, link_arrows=True)
        self.assertEqual(lp.svg.count('<polygon'), 2)

    def test_draw_labels(self):
        lp = self.p2s.linkp(_make_df(), relationships=_rels(), pos=_make_pos(),
                            draw_labels=True)
        self.assertIn('<text', lp.svg)

    def test_node_labels_dict(self):
        lp = self.p2s.linkp(_make_df(), relationships=_rels(), pos=_make_pos(),
                            draw_labels=True, node_labels={'a': 'Alice', 'b': 'Bob', 'c': 'Carol'})
        self.assertIn('Alice', lp.svg)

    def test_multiple_relationships(self):
        df = pl.DataFrame({'fm': ['a', 'b'], 'to': ['b', 'c'], 'p': ['c', 'a'], 'q': ['d', 'e']})
        pos = {'a': [0, 0], 'b': [1, 0], 'c': [0.5, 0.866], 'd': [0.2, 0.5], 'e': [0.8, 0.5]}
        lp = self.p2s.linkp(df, relationships=[('fm', 'to'), ('p', 'q')], pos=pos)
        self.assertIn('<line', lp.svg)

    def test_wxh_parameter(self):
        lp = self.p2s.linkp(_make_df(), relationships=_rels(), pos=_make_pos(),
                            wxh=(512, 384))
        self.assertIn('width="512"', lp.svg)
        self.assertIn('height="384"', lp.svg)

    def test_no_pos_random_placement(self):
        lp = self.p2s.linkp(_make_df(), relationships=_rels(), pos={})
        self.assertIn('<circle', lp.svg)

    def test_color_hex_literal(self):
        lp = self.p2s.linkp(_make_df(), relationships=_rels(), pos=_make_pos(),
                            color='#3355aa')
        self.assertIn('#3355aa', lp.svg)

    def test_node_size_small_medium_large(self):
        for sz in ('small', 'medium', 'large'):
            lp = self.p2s.linkp(_make_df(), relationships=_rels(), pos=_make_pos(), node_size=sz)
            self.assertIn('<circle', lp.svg, msg=f'node_size={sz!r}')

    def test_collapsed_nodes_render_as_cloud(self):
        # Two nodes sharing identical pos collapse to the same screen pixel → cloud icon
        df = pl.DataFrame({'fm': ['a', 'a'], 'to': ['b', 'c']})
        pos = {'a': [0.0, 0.0], 'b': [1.0, 0.0], 'c': [1.0, 0.0]}
        lp = self.p2s.linkp(df, relationships=[('fm', 'to')], pos=pos)
        self.assertIn('href="#cloud"', lp.svg)
        self.assertIn('id="cloud"',   lp.svg)

    def test_link_shape_line_none(self):
        lp = self.p2s.linkp(_make_df(), relationships=_rels(), pos=_make_pos(),
                            node_size=None, link_size=None)
        # With no node or link sizes, SVG should still be valid
        self.assertIn('<svg', lp.svg)

    def test_timing_metrics_populated(self):
        lp = self.p2s.linkp(_make_df(), relationships=_rels(), pos=_make_pos())
        assert_timing_metrics_populated(self, lp, (
            '__parseInput__', '__calculateGeometry__',
            '__calculateScreenCoordinates__', '__renderLinks__',
            '__renderNodes__', '__renderSVG__',
        ))

    def test_repr_svg(self):
        lp = self.p2s.linkp(_make_df(), relationships=_rels(), pos=_make_pos())
        self.assertEqual(lp._repr_svg_(), lp.svg)

    def test_tuple_node_fields(self):
        df = pl.DataFrame({
            'src_a': ['x', 'y'], 'src_b': ['1', '2'],
            'dst_a': ['y', 'z'], 'dst_b': ['2', '3'],
        })
        pos = {'x|1': [0, 0], 'y|2': [1, 0], 'z|3': [0.5, 1]}
        lp = self.p2s.linkp(df, relationships=[(('src_a', 'src_b'), ('dst_a', 'dst_b'))], pos=pos)
        self.assertIn('<line', lp.svg)

    def test_convex_hull_list(self):
        lp = self.p2s.linkp(_make_df(), relationships=_rels(), pos=_make_pos(),
                            convex_hull_lu={'group1': ['a', 'b'], 'group2': ['c']})
        self.assertIn('<svg', lp.svg)

    def test_smallp_integration(self):
        from polars2svg.linkp import LinkP
        df  = _make_df()
        pos = _make_pos()
        template = self.p2s.linkp(df, relationships=_rels(), pos=pos, node_size='medium')
        sm = self.p2s.smallp(df, template, 'category')
        self.assertIn('<svg', sm.svg)

    def test_render_with(self):
        df  = _make_df()
        pos = _make_pos()
        template = self.p2s.linkp(df, relationships=_rels(), pos=pos)
        subset   = df.filter(pl.col('category') == 'x')
        lp2      = template.render_with(subset)
        self.assertIn('<svg', lp2.svg)

    def test_df_as_keyword(self):
        lp = self.p2s.linkp(df=_make_df(), relationships=_rels(), pos=_make_pos())
        self.assertIn('<circle', lp.svg)

    # --- Positional inference tests ---

    def test_positional_relationships_and_pos(self):
        lp = self.p2s.linkp(_make_df(), _rels(), _make_pos())
        self.assertIn('<circle', lp.svg)
        self.assertIn('<line',   lp.svg)

    def test_positional_relationships_keyword_pos(self):
        lp = self.p2s.linkp(_make_df(), _rels(), pos=_make_pos())
        self.assertIn('<circle', lp.svg)

    def test_positional_pos_keyword_relationships(self):
        lp = self.p2s.linkp(_make_df(), _make_pos(), relationships=_rels())
        self.assertIn('<circle', lp.svg)

    def test_positional_pos_only(self):
        lp = self.p2s.linkp(_make_df(), _make_pos(), relationships=_rels())
        self.assertIn('<circle', lp.svg)

    def test_positional_order_pos_before_rels(self):
        lp = self.p2s.linkp(_make_df(), _make_pos(), _rels())
        self.assertIn('<circle', lp.svg)

    def test_positional_duplicate_relationships_raises(self):
        with self.assertRaises(ValueError):
            self.p2s.linkp(_make_df(), _rels(), relationships=_rels(), pos=_make_pos())

    def test_positional_duplicate_pos_raises(self):
        with self.assertRaises(ValueError):
            self.p2s.linkp(_make_df(), _make_pos(), pos=_make_pos(), relationships=_rels())

    def test_positional_with_extra_kwargs(self):
        lp = self.p2s.linkp(_make_df(), _rels(), _make_pos(),
                            color='category', node_size='large')
        self.assertIn('<circle', lp.svg)

    # --- Full color schema tests ---

    def test_color_crow_magnitude(self):
        lp = self.p2s.linkp(_make_df(), relationships=_rels(), pos=_make_pos(),
                            color=self.p2s.CROW_MAGNITUDEp)
        self.assertIn('<line', lp.svg)
        self.assertIn('<circle', lp.svg)

    def test_color_crow_stretched(self):
        lp = self.p2s.linkp(_make_df(), relationships=_rels(), pos=_make_pos(),
                            color=self.p2s.CROW_STRETCHEDp)
        self.assertIn('<line', lp.svg)

    def test_color_crow_magnitude_links(self):
        lp = self.p2s.linkp(_make_df(), relationships=_rels(), pos=_make_pos(),
                            color=self.p2s.CROW_MAGNITUDEp)
        self.assertIn('<line', lp.svg)

    def test_color_crow_stretched_links(self):
        lp = self.p2s.linkp(_make_df(), relationships=_rels(), pos=_make_pos(),
                            color=self.p2s.CROW_STRETCHEDp)
        self.assertIn('<line', lp.svg)

    def test_node_color_crow_magnitude(self):
        lp = self.p2s.linkp(_make_df(), relationships=_rels(), pos=_make_pos(),
                            node_color=self.p2s.CROW_MAGNITUDEp)
        self.assertIn('<circle', lp.svg)

    def test_color_cset_links(self):
        lp = self.p2s.linkp(_make_df(), relationships=_rels(), pos=_make_pos(),
                            color=('category', self.p2s.CSETp))
        self.assertIn('<line', lp.svg)

    def test_color_cset_magnitude_links(self):
        lp = self.p2s.linkp(_make_df(), relationships=_rels(), pos=_make_pos(),
                            color=('category', self.p2s.CSET_MAGNITUDEp))
        self.assertIn('<line', lp.svg)

    def test_color_cset_stretched_links(self):
        lp = self.p2s.linkp(_make_df(), relationships=_rels(), pos=_make_pos(),
                            color=('category', self.p2s.CSET_STRETCHEDp))
        self.assertIn('<line', lp.svg)

    def test_color_stat_magnitude_links(self):
        lp = self.p2s.linkp(_make_df(), relationships=_rels(), pos=_make_pos(),
                            color=('weight', self.p2s.CMAGNITUDE_SUMp))
        self.assertIn('<line', lp.svg)

    def test_color_stat_stretched_links(self):
        lp = self.p2s.linkp(_make_df(), relationships=_rels(), pos=_make_pos(),
                            color=('weight', self.p2s.CSTRETCHED_MAXp))
        self.assertIn('<line', lp.svg)

    def test_node_color_stat_magnitude(self):
        lp = self.p2s.linkp(_make_df(), relationships=_rels(), pos=_make_pos(),
                            node_color=('weight', self.p2s.CMAGNITUDE_MEANp))
        self.assertIn('<circle', lp.svg)

    def test_node_color_dict(self):
        lp = self.p2s.linkp(_make_df(), relationships=_rels(), pos=_make_pos(),
                            node_color={'a': '#ff0000', 'b': '#00ff00', 'c': '#0000ff'})
        self.assertIn('<circle', lp.svg)
        # At least one of the dict colors should appear
        has_color = any(c in lp.svg for c in ('#ff0000', '#00ff00', '#0000ff'))
        self.assertTrue(has_color)

    def test_node_color_categorical_field(self):
        # Nodes with mixed field values get the string multiset sentinel color (same as CSETp).
        import re
        lp = self.p2s.linkp(_make_df(), relationships=_rels(), pos=_make_pos(),
                            color='category', node_color='fm')
        multiset_color = '#7f8367'
        fills = re.findall(r'<circle[^>]*fill="(#[0-9a-f]+)"', lp.svg)
        self.assertGreater(len(fills), 0)
        # In _make_df(), every node appears on both fm and to sides with different fm values → all multiset
        self.assertTrue(all(f == multiset_color for f in fills),
                        f'Expected all nodes to get multiset sentinel {multiset_color}, got {fills}')

    def test_node_color_categorical_tuple(self):
        # ('fm', CSETp) with mixed-value nodes → all get the string multiset sentinel color.
        import re
        lp = self.p2s.linkp(_make_df(), relationships=_rels(), pos=_make_pos(),
                            node_color=('fm', self.p2s.CSETp))
        multiset_color = '#7f8367'
        fills = re.findall(r'<circle[^>]*fill="(#[0-9a-f]+)"', lp.svg)
        self.assertGreater(len(fills), 0)
        self.assertTrue(all(f == multiset_color for f in fills),
                        f'Expected all nodes to get multiset sentinel {multiset_color}, got {fills}')

    def test_node_color_by_node_name(self):
        import re
        lp = self.p2s.linkp(_make_df(), relationships=_rels(), pos=_make_pos(),
                            node_color=self.p2s.COLOR_BY_NODE_NAME)
        bg = lp.p2s.colorTyped('background', 'default')
        fills = re.findall(r'<circle[^>]*fill="(#[0-9a-f]+)"', lp.svg)
        self.assertGreater(len(fills), 0)
        self.assertTrue(all(f != bg for f in fills),
                        'Expected all nodes to get their node-name hash color')

    def test_node_color_crow_stretched(self):
        lp = self.p2s.linkp(_make_df(), relationships=_rels(), pos=_make_pos(),
                            node_color=self.p2s.CROW_STRETCHEDp)
        self.assertIn('<circle', lp.svg)

    def test_node_color_numeric_field_auto(self):
        lp = self.p2s.linkp(_make_df(), relationships=_rels(), pos=_make_pos(),
                            node_color='weight')
        self.assertIn('<circle', lp.svg)
        self.assertEqual(lp._node_color_mode_['kind'], 'stat_magnitude')
        self.assertEqual(lp._node_color_mode_['field'], 'weight')

    def test_color_and_node_color_override(self):
        # node_color hex takes priority over color field
        lp = self.p2s.linkp(_make_df(), relationships=_rels(), pos=_make_pos(),
                            color='category', node_color='#aabbcc')
        self.assertIn('#aabbcc', lp.svg)

    def test_color_stat_min_links(self):
        lp = self.p2s.linkp(_make_df(), relationships=_rels(), pos=_make_pos(),
                            color=('weight', self.p2s.CMAGNITUDE_MINp))
        self.assertIn('<line', lp.svg)

    def test_color_stat_median_links(self):
        lp = self.p2s.linkp(_make_df(), relationships=_rels(), pos=_make_pos(),
                            color=('weight', self.p2s.CMAGNITUDE_MEDIANp))
        self.assertIn('<line', lp.svg)

    def test_color_stat_mean_links(self):
        lp = self.p2s.linkp(_make_df(), relationships=_rels(), pos=_make_pos(),
                            color=('weight', self.p2s.CMAGNITUDE_MEANp))
        self.assertIn('<line', lp.svg)

    def test_color_stat_max_links(self):
        lp = self.p2s.linkp(_make_df(), relationships=_rels(), pos=_make_pos(),
                            color=('weight', self.p2s.CMAGNITUDE_MAXp))
        self.assertIn('<line', lp.svg)

    def test_color_stretched_sum_links(self):
        lp = self.p2s.linkp(_make_df(), relationships=_rels(), pos=_make_pos(),
                            color=('weight', self.p2s.CSTRETCHED_SUMp))
        self.assertIn('<line', lp.svg)

    def test_curve_with_crow_color(self):
        lp = self.p2s.linkp(_make_df(), relationships=_rels(), pos=_make_pos(),
                            link_shape='curve', color=self.p2s.CROW_MAGNITUDEp)
        self.assertIn('<path', lp.svg)

    def test_node_color_vary_raises(self):
        with self.assertRaises(ValueError):
            self.p2s.linkp(_make_df(), relationships=_rels(), pos=_make_pos(),
                           color='category', node_color='vary')

    # --- numpy / array-like pos dict tests ---

    def test_positional_numpy_pos_three_args(self):
        import numpy as np
        pos = {'a': np.array([0.0, 0.0]), 'b': np.array([1.0, 0.0]), 'c': np.array([0.5, 0.866])}
        lp = self.p2s.linkp(_make_df(), _rels(), pos)
        self.assertIn('<circle', lp.svg)
        self.assertIn('<line',   lp.svg)

    def test_positional_numpy_pos_reversed_order(self):
        import numpy as np
        pos = {'a': np.array([0.0, 0.0]), 'b': np.array([1.0, 0.0]), 'c': np.array([0.5, 0.866])}
        lp = self.p2s.linkp(_make_df(), pos, _rels())
        self.assertIn('<circle', lp.svg)
        self.assertIn('<line',   lp.svg)

    def test_keyword_numpy_pos(self):
        import numpy as np
        pos = {'a': np.array([0.0, 0.0]), 'b': np.array([1.0, 0.0]), 'c': np.array([0.5, 0.866])}
        lp = self.p2s.linkp(_make_df(), relationships=_rels(), pos=pos)
        self.assertIn('<circle', lp.svg)
        self.assertIn('<line',   lp.svg)

    def test_positional_tuple_pos_values(self):
        pos = {'a': (0.0, 0.0), 'b': (1.0, 0.0), 'c': (0.5, 0.866)}
        lp = self.p2s.linkp(_make_df(), _rels(), pos)
        self.assertIn('<circle', lp.svg)
        self.assertIn('<line',   lp.svg)

    def test_positional_unrecognized_type_raises(self):
        with self.assertRaises(ValueError):
            self.p2s.linkp(_make_df(), _rels(), 'not_a_valid_arg')


if __name__ == '__main__':
    unittest.main()
