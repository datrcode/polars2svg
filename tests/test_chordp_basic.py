import unittest
import polars as pl
from polars2svg import Polars2SVG
from svg_test_utils import assert_timing_metrics_populated

_DF_ = pl.DataFrame({
    'fm':       ['a', 'b', 'c', 'a', 'b', 'd'],
    'to':       ['b', 'c', 'a', 'c', 'a', 'a'],
    'category': ['x', 'y', 'x', 'y', 'x', 'y'],
    'weight':   [1,   3,   2,   1,   4,   2  ],
})
_RELS_ = [('fm', 'to')]


def _p(**extra):
    p2s = Polars2SVG()
    return p2s.chordp(df=_DF_, relationships=_RELS_, **extra)


class TestChordPBasic(unittest.TestCase):

    def setUp(self):
        self.p2s = Polars2SVG()

    # --- basic sanity ---

    def test_basic_render(self):
        cp = _p()
        self.assertIn('<svg', cp.svg)
        self.assertIn('<path', cp.svg)
        self.assertGreater(len(cp.svg), 200)

    def test_repr_svg(self):
        cp = _p()
        self.assertEqual(cp._repr_svg_(), cp.svg)

    def test_df_as_keyword(self):
        cp = self.p2s.chordp(df=_DF_, relationships=_RELS_)
        self.assertIn('<svg', cp.svg)

    def test_timing_metrics_populated(self):
        cp = _p()
        assert_timing_metrics_populated(self, cp, (
            '__parseInput__', '__calculateOrder__', '__calculateGeometry__',
            '__renderLinks__', '__renderNodes__', '__renderSVG__',
        ))

    def test_t_overall_positive(self):
        cp = _p()
        self.assertGreater(cp.t_overall, 0.0)

    # --- positional args ---

    def test_positional_relationships(self):
        cp = self.p2s.chordp(_DF_, _RELS_)
        self.assertIn('<svg', cp.svg)

    def test_positional_df_and_rels_with_kwarg(self):
        cp = self.p2s.chordp(_DF_, _RELS_, wxh=(128, 128))
        self.assertIn('width="128"', cp.svg)

    def test_duplicate_df_raises(self):
        with self.assertRaises(Exception):
            self.p2s.chordp(_DF_, df=_DF_, relationships=_RELS_)

    # --- geometry ---

    def test_wxh_custom(self):
        cp = _p(wxh=(512, 512))
        self.assertIn('width="512"', cp.svg)

    def test_default_wxh(self):
        cp = _p()
        self.assertIn('width="256"', cp.svg)

    def test_insets(self):
        cp = _p(insets=(10, 10))
        self.assertIn('<path', cp.svg)

    # --- node params ---

    def test_node_color_hex(self):
        cp = _p(node_color='#ff0000')
        self.assertIn('#ff0000', cp.svg)

    def test_node_color_none(self):
        cp = _p(node_color=None)
        self.assertIn('<path', cp.svg)

    def test_node_color_vary(self):
        cp = _p(node_color='category')
        self.assertIn('<path', cp.svg)

    def test_node_color_by_name(self):
        p2s = Polars2SVG()
        cp = p2s.chordp(df=_DF_, relationships=_RELS_, node_color=p2s.COLOR_BY_NODE_NAME)
        self.assertIn('<path', cp.svg)

    def test_node_size_small(self):
        cp = _p(node_size='small')
        self.assertIn('<path', cp.svg)

    def test_node_size_medium(self):
        cp = _p(node_size='medium')
        self.assertIn('<path', cp.svg)

    def test_node_size_large(self):
        cp = _p(node_size='large')
        self.assertIn('<path', cp.svg)

    def test_node_size_vary(self):
        cp = _p(node_size='vary')
        self.assertIn('<path', cp.svg)

    def test_node_size_range(self):
        cp = _p(node_size='vary', node_size_range=(1.0, 8.0))
        self.assertIn('<path', cp.svg)

    def test_node_gap(self):
        cp = _p(node_gap=10)
        self.assertIn('<path', cp.svg)

    def test_node_opacity(self):
        cp = _p(node_opacity=0.5)
        self.assertIn('0.5', cp.svg)

    def test_node_selection(self):
        cp = _p(node_selection={'a'})
        self.assertIn('<path', cp.svg)

    def test_node_labels_draw_labels(self):
        cp = _p(draw_labels=True, node_labels={'a': 'Node A', 'b': 'Node B'})
        self.assertIn('Node A', cp.svg)
        self.assertIn('Node B', cp.svg)

    def test_draw_labels_radial(self):
        cp = _p(draw_labels=True, label_style='radial')
        self.assertIn('<text', cp.svg)

    def test_draw_labels_circular(self):
        cp = _p(draw_labels=True, label_style='circular')
        self.assertIn('textPath', cp.svg)

    def test_label_only(self):
        cp = _p(draw_labels=True, label_only={'a'})
        self.assertIn('<text', cp.svg)

    def test_order_explicit(self):
        cp = _p(order=['a', 'b', 'c', 'd'])
        self.assertIn('<path', cp.svg)

    def test_order_stored(self):
        cp = _p(order=['d', 'c', 'b', 'a'])
        self.assertEqual(cp.order, ['d', 'c', 'b', 'a'])

    # --- link params ---

    def test_link_shape_curve(self):
        cp = _p(link_shape='curve')
        self.assertIn('<path', cp.svg)

    def test_link_shape_bundled(self):
        cp = _p(link_shape='bundled')
        self.assertIn('<path', cp.svg)

    def test_link_color_hex(self):
        cp = _p(color='#00ff00')
        self.assertIn('#00ff00', cp.svg)

    def test_link_color_src(self):
        cp = _p(color='src')
        self.assertIn('<path', cp.svg)

    def test_link_color_dst(self):
        cp = _p(color='dst')
        self.assertIn('<path', cp.svg)

    def test_link_color_vary(self):
        cp = _p(color='category')
        self.assertIn('<path', cp.svg)

    def test_link_opacity(self):
        cp = _p(link_opacity=0.3)
        self.assertIn('0.3', cp.svg)

    def test_link_size(self):
        cp = _p(link_size='large')
        self.assertIn('<path', cp.svg)

    def test_link_size_range(self):
        cp = _p(link_size_range=(1.0, 6.0))
        self.assertIn('<path', cp.svg)

    # --- bundled-specific params ---

    def test_bundle_strength(self):
        cp = _p(link_shape='bundled', bundle_strength=0.5)
        self.assertIn('<path', cp.svg)

    def test_bundle_rings(self):
        cp = _p(link_shape='bundled', bundle_rings=3)
        self.assertIn('<path', cp.svg)

    def test_skeleton_algorithm_hexagonal(self):
        cp = _p(link_shape='bundled', skeleton_algorithm='hexagonal')
        self.assertIn('<path', cp.svg)

    def test_skeleton_algorithm_radial(self):
        cp = _p(link_shape='bundled', skeleton_algorithm='radial')
        self.assertIn('<path', cp.svg)

    def test_skeleton_algorithm_kmeans(self):
        cp = _p(link_shape='bundled', skeleton_algorithm='kmeans')
        self.assertIn('<path', cp.svg)

    # --- count ---

    def test_count_row_count(self):
        p2s = Polars2SVG()
        cp = p2s.chordp(df=_DF_, relationships=_RELS_, count=p2s.ROW_COUNTp)
        self.assertIn('<path', cp.svg)

    def test_count_field_numeric(self):
        cp = _p(count='weight')
        self.assertIn('<path', cp.svg)

    def test_count_field_set(self):
        p2s = Polars2SVG()
        cp = p2s.chordp(df=_DF_, relationships=_RELS_, count=('category', p2s.SETp))
        self.assertIn('<path', cp.svg)

    # --- display flags ---

    def test_draw_border_false(self):
        cp_with    = _p(draw_border=True)
        cp_without = _p(draw_border=False)
        self.assertGreater(len(cp_with.svg), len(cp_without.svg))

    def test_txt_h(self):
        cp = _p(draw_labels=True, txt_h=16)
        self.assertIn('16px', cp.svg)

    def test_txt_offset(self):
        cp = _p(draw_labels=True, txt_offset=5)
        self.assertIn('<text', cp.svg)

    # --- template / render_with ---

    def test_template_clone(self):
        p2s = Polars2SVG()
        tmpl = p2s.chordp(df=_DF_, relationships=_RELS_, wxh=(200, 200))
        cp   = p2s.chordp(template=tmpl, df=_DF_, relationships=_RELS_)
        self.assertIn('width="200"', cp.svg)

    def test_render_with(self):
        p2s = Polars2SVG()
        tmpl = p2s.chordp(df=_DF_, relationships=_RELS_, wxh=(200, 200))
        cp   = tmpl.render_with(_DF_)
        self.assertIn('width="200"', cp.svg)

    # --- small multiples ---

    def test_smallp_integration(self):
        p2s  = Polars2SVG()
        tmpl = p2s.chordp(df=_DF_, relationships=_RELS_, wxh=(128, 128))
        sm   = p2s.smallp(_DF_, tmpl, 'category')
        self.assertIn('<svg', sm.svg)

    def test_sm_x_shared_order(self):
        p2s  = Polars2SVG()
        tmpl = p2s.chordp(df=_DF_, relationships=_RELS_, wxh=(128, 128),
                          sm_shared={p2s.SM_X})
        sm   = p2s.smallp(_DF_, tmpl, 'category')
        self.assertIn('<svg', sm.svg)

    def test_sm_y_shared_skeleton(self):
        p2s  = Polars2SVG()
        tmpl = p2s.chordp(df=_DF_, relationships=_RELS_, wxh=(128, 128),
                          link_shape='bundled', sm_shared={p2s.SM_Y})
        sm   = p2s.smallp(_DF_, tmpl, 'category')
        self.assertIn('<svg', sm.svg)

    def test_sm_x_and_y_together(self):
        p2s  = Polars2SVG()
        tmpl = p2s.chordp(df=_DF_, relationships=_RELS_, wxh=(128, 128),
                          link_shape='bundled', sm_shared={p2s.SM_X, p2s.SM_Y})
        sm   = p2s.smallp(_DF_, tmpl, 'category')
        self.assertIn('<svg', sm.svg)

    # --- color_nodes_final ---

    def test_color_nodes_final_populated(self):
        cp = _p()
        self.assertIsInstance(cp.color_nodes_final, dict)
        self.assertGreater(len(cp.color_nodes_final), 0)

    def test_color_nodes_final_hex_values(self):
        cp = _p()
        for v in cp.color_nodes_final.values():
            self.assertTrue(v.startswith('#'), f'{v!r} is not a hex color')

    # --- validation ---

    def test_no_df_gives_blank_svg(self):
        p2s = Polars2SVG()
        cp  = p2s.chordp(relationships=_RELS_)
        self.assertIn('<svg', cp.svg)

    def test_missing_relationships_raises(self):
        with self.assertRaises(Exception):
            _p(relationships=None)

    def test_unknown_field_raises(self):
        with self.assertRaises(Exception):
            _p(relationships=[('no_such_field', 'to')])

    def test_invalid_label_style_raises(self):
        with self.assertRaises(Exception):
            _p(label_style='banana')


if __name__ == '__main__':
    unittest.main()
