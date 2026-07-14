import unittest
import polars as pl
from polars2svg import Polars2SVG
from svg_test_utils import assert_timing_metrics_populated, assert_valid_svg


def _make_df():
    return pl.DataFrame({
        'fm':   ['a', 'b', 'c', 'a', 'd', 'b'],
        'to':   ['b', 'a', 'a', 'c', 'a', 'c'],
        'time': [1,   1,   1,   2,   2,   3  ],
        'w':    [3,   1,   2,   4,   1,   2  ],
    })

def _rels():
    return [('fm', 'to')]


class TestSpreadLinesPBasic(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def test_basic_render(self):
        sp = self.p2s.spreadlinesp(_make_df(), _rels(), ego='a', time='time')
        assert_valid_svg(self, sp.svg)
        self.assertIn('<circle', sp.svg)

    def test_timing_metrics_populated(self):
        sp = self.p2s.spreadlinesp(_make_df(), _rels(), ego='a', time='time')
        assert_timing_metrics_populated(self, sp, ('__parseInput__', '__calculateLayout__', '__renderSVG__'))

    def test_wxh_respected(self):
        sp = self.p2s.spreadlinesp(_make_df(), _rels(), ego='a', time='time', wxh=(600, 300))
        self.assertEqual(sp.wxh, (600, 300))

    def test_count_field(self):
        sp = self.p2s.spreadlinesp(_make_df(), _rels(), ego='a', time='time', count='w')
        assert_valid_svg(self, sp.svg)

    def test_node_color_hex(self):
        sp = self.p2s.spreadlinesp(_make_df(), _rels(), ego='a', time='time', node_color='#ff0000')
        self.assertIn('#ff0000', sp.svg)

    def test_node_color_by_node_name(self):
        sp = self.p2s.spreadlinesp(_make_df(), _rels(), ego='a', time='time',
                                   node_color=self.p2s.COLOR_BY_NODE_NAME)
        assert_valid_svg(self, sp.svg)

    def test_node_color_dict(self):
        sp = self.p2s.spreadlinesp(_make_df(), _rels(), ego='a', time='time',
                                   node_color={'b': '#00ff00', 'c': '#0000ff'})
        self.assertIn('#00ff00', sp.svg)

    def test_set_ego(self):
        sp = self.p2s.spreadlinesp(_make_df(), _rels(), ego={'a', 'b'}, time='time')
        assert_valid_svg(self, sp.svg)

    def test_repr_svg(self):
        sp = self.p2s.spreadlinesp(_make_df(), _rels(), ego='a', time='time')
        self.assertEqual(sp._repr_svg_(), sp.svg)

    def test_no_df_constructor_does_not_crash(self):
        tmpl = self.p2s.spreadlinesp(_rels(), ego='a', time='time')
        self.assertIsNone(tmpl.df)

    def test_render_with_applies_template(self):
        tmpl = self.p2s.spreadlinesp(_rels(), ego='a', time='time')
        sp   = tmpl.render_with(_make_df())
        assert_valid_svg(self, sp.svg)

    def test_render_with_overrides_ego(self):
        tmpl = self.p2s.spreadlinesp(_rels(), ego='a', time='time')
        sp   = tmpl.render_with(_make_df(), ego='b')
        assert_valid_svg(self, sp.svg)

    def test_unknown_kwarg_raises(self):
        with self.assertRaises(TypeError):
            self.p2s.spreadlinesp(_make_df(), _rels(), ego='a', time='time',
                                  _nonexistent_param_=True)

    def test_anno_does_not_crash(self):
        sp = self.p2s.spreadlinesp(_make_df(), _rels(), ego='a', time='time', anno={1: 'Event A'})
        assert_valid_svg(self, sp.svg)

    def test_highlight_nodes(self):
        sp = self.p2s.spreadlinesp(_make_df(), _rels(), ego='a', time='time',
                                   highlight_nodes={'b'})
        assert_valid_svg(self, sp.svg)

    def test_max_rings_1(self):
        sp = self.p2s.spreadlinesp(_make_df(), _rels(), ego='a', time='time', max_rings=1)
        assert_valid_svg(self, sp.svg)

    def test_max_rings_2(self):
        sp = self.p2s.spreadlinesp(_make_df(), _rels(), ego='a', time='time', max_rings=2)
        assert_valid_svg(self, sp.svg)


if __name__ == '__main__':
    unittest.main()
