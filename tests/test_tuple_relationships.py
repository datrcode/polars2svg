"""Tuple relationship endpoints across the template-clone and interactive paths.

A relationship endpoint may be a tuple of columns -- [(('src_ip','src_port'), 'dst_ip')] --
which the component expands into a synthetic '|'-joined '__fm{i}__' / '__to{i}__' column
and then REWRITES its own `relationships` to name.  That rewritten spec is valid only
against the component's private render frame (`self.df`); the pristine spec is kept as
`relationships_orig`.

Two things used to pair the rewritten spec with a frame that never had the synthetic
column -- the caller's `df_orig`, or a fresh df handed to a template clone -- so a graph
that rendered perfectly on its own died with `ColumnNotFoundError: __fm0__` the moment it
was put in a panelize(), and render_with() raised '"__fm0__" not found in DataFrame'.
All-string relationships were unaffected, which is why it went unnoticed.
"""
import asyncio
import datetime
import unittest

import polars as pl

from polars2svg import Polars2SVG
from polars2svg.spreadlinepi import _filter_out_nodes


def _make_df():
    # a-b has 3 rows, b-c / c-a one each, far-a two: 7 rows, 4 nodes once (fm, port)
    # is joined.  'port' varies within a node so the joined names are distinguishable.
    return pl.DataFrame({
        'fm':       ['a', 'a', 'a', 'b',   'c', 'far', 'far'],
        'port':     ['1', '1', '1', '2',   '3', '4',   '4'  ],
        'to':       ['b', 'b', 'b', 'c',   'a', 'a',   'a'  ],
        'category': ['x', 'y', 'x', 'y',   'x', 'z',   'z'  ],
    })


# The joined node names the render frame uses; the graph must agree letter for letter.
_FM_NODES_ = {'a|1', 'b|2', 'c|3', 'far|4'}
_TO_NODES_ = {'a', 'b', 'c'}


class TestLinkPTupleEndpointsInteractive(unittest.TestCase):
    """linkpi()/panelize() work against df_orig and the stack frames derived from it,
    so they need the pristine spec -- relationships_orig, not relationships."""

    @classmethod
    def setUpClass(cls):
        cls.p2s = Polars2SVG()

    def setUp(self):
        self.df   = _make_df()
        self.lp   = self.p2s.linkp(self.df, [(('fm', 'port'), 'to')], wxh=(512, 512))
        self.lp._repr_svg_()
        self.ctrl = self.p2s.linkpi(self.lp)

    def test_linkp_keeps_both_specs(self):
        self.assertEqual(self.lp.relationships_orig, [(('fm', 'port'), 'to')])
        self.assertEqual(self.lp.relationships,      [('__fm0__', 'to')])
        # the synthetic column lives on the render frame only -- this is the whole trap
        self.assertIn('__fm0__',    self.lp.df.columns)
        self.assertNotIn('__fm0__', self.lp.df_orig.columns)

    def test_panelize_accepts_a_tuple_endpoint_linkp(self):
        # The reported failure: renders standalone, ColumnNotFoundError inside panelize().
        self.assertIsNotNone(self.p2s.panelize([[self.lp]]))

    def test_graph_node_names_match_the_rendered_ones(self):
        # Both sides join with '|', so a selection made on screen resolves in the graph.
        self.assertEqual(set(self.ctrl.graphs[0].nodes()), _FM_NODES_ | _TO_NODES_)

    def test_ln_params_carries_the_pristine_spec(self):
        # Every consumer of this entry pairs it with a df_orig-shaped frame.
        self.assertEqual(self.ctrl.ln_params['relationships'], [(('fm', 'port'), 'to')])

    def test_extract_nodes_joins_tuple_endpoints(self):
        # Previously returned an empty set -- no exception, just every node looking absent.
        self.assertEqual(self.ctrl._extractNodes_(self.df), _FM_NODES_ | _TO_NODES_)

    def test_push_stack_rebuilds_the_graph(self):
        _sub_ = self.df.filter(pl.col('category') != 'z')
        self.ctrl.pushStack(_sub_)
        self.assertEqual(self.ctrl.df_level, 1)
        self.assertEqual(set(self.ctrl.graphs[1].nodes()), {'a|1', 'b|2', 'c|3', 'a', 'b', 'c'})

    def test_push_selected_filters_out_the_selection(self):
        # 'x': drop the selected nodes and push what is left (filterDataFrameByGraph +
        # _extractNodes_, both of which took the rewritten spec).
        self.ctrl.selected_entities = {'far|4'}
        self.assertTrue(self.ctrl.apply_push_selected())
        self.assertNotIn('far|4', set(self.ctrl.graphs[self.ctrl.df_level].nodes()))
        self.assertEqual(len(self.ctrl.dfs[self.ctrl.df_level]), 5)

    def test_collapse_edges_and_unfilter_round_trip(self):
        self.assertTrue(self.ctrl.apply_collapse_edges())
        self.assertEqual(len(self.ctrl.dfs[self.ctrl.df_level]), 4)   # one row per edge
        self.assertTrue(self.ctrl.apply_edge_unfilter())
        self.assertEqual(len(self.ctrl.dfs[self.ctrl.df_level]), 7)   # all rows back

    def test_node_expansion(self):
        self.assertTrue(self.ctrl.apply_collapse_edges())
        self.assertTrue(self.ctrl.apply_node_expansion())

    def test_replace_base_dataframe(self):
        _sub_ = self.df.filter(pl.col('category') != 'z')
        asyncio.run(self.ctrl.replaceBaseDataframe(_sub_))
        self.assertEqual(self.ctrl.df_level, 0)
        self.assertEqual(set(self.ctrl.graphs[0].nodes()), {'a|1', 'b|2', 'c|3', 'a', 'b', 'c'})

    def test_string_relationships_are_untouched(self):
        _lp_ = self.p2s.linkp(self.df, [('fm', 'to')], wxh=(512, 512))
        _lp_._repr_svg_()
        _ctrl_ = self.p2s.linkpi(_lp_)
        self.assertEqual(_lp_.relationships_orig, [('fm', 'to')])
        self.assertEqual(_lp_.relationships,      [('fm', 'to')])
        self.assertEqual(_ctrl_.ln_params['relationships'], [('fm', 'to')])
        self.assertEqual(set(_ctrl_.graphs[0].nodes()), {'a', 'b', 'c', 'far'})


class TestTemplateCloneTupleEndpoints(unittest.TestCase):
    """A clone inherits the template's already-rewritten relationships, so it has to
    re-expand from relationships_orig or its fresh df never gets the synthetic column."""

    @classmethod
    def setUpClass(cls):
        cls.p2s = Polars2SVG()

    def setUp(self):
        self.df = _make_df()

    def _assert_clone_renders(self, comp):
        _clone_ = comp.render_with(self.df)
        self.assertIn('<svg', _clone_._repr_svg_())
        # the clone re-derives both specs from the pristine one, so it is not one
        # rewrite deeper than its template
        self.assertEqual(_clone_.relationships_orig, comp.relationships_orig)
        self.assertEqual(_clone_.relationships,      comp.relationships)
        return _clone_

    def test_linkp_clone(self):
        _lp_ = self.p2s.linkp(self.df, [(('fm', 'port'), 'to')], wxh=(512, 512))
        _lp_._repr_svg_()
        _clone_ = self._assert_clone_renders(_lp_)
        self.assertIn('__fm0__', _clone_.df.columns)

    def test_linkp_clone_with_tuple_on_the_to_side(self):
        _lp_ = self.p2s.linkp(self.df, [('to', ('fm', 'port'))], wxh=(512, 512))
        _lp_._repr_svg_()
        self._assert_clone_renders(_lp_)

    def test_linkp_clone_with_a_link_label_field(self):
        _lp_ = self.p2s.linkp(self.df, [(('fm', 'port'), 'to', 'category')], wxh=(512, 512))
        _lp_._repr_svg_()
        self._assert_clone_renders(_lp_)

    def test_chordp_clone(self):
        _ch_ = self.p2s.chordp(self.df, [(('fm', 'port'), 'to')], wxh=(512, 512))
        _ch_._repr_svg_()
        self._assert_clone_renders(_ch_)

    def test_spreadlinesp_clone(self):
        _df_ = self.df.with_columns(
            pl.Series('time', [datetime.datetime(2024, 1, 1 + i) for i in range(len(self.df))])
        )
        _sl_ = self.p2s.spreadlinesp(_df_, [(('fm', 'port'), 'to')], ego='a|1', time='time')
        _sl_._repr_svg_()
        _clone_ = _sl_.render_with(_df_)
        self.assertIn('<svg', _clone_._repr_svg_())
        self.assertEqual(_clone_.relationships_orig, _sl_.relationships_orig)
        self.assertEqual(_clone_.relationships,      _sl_.relationships)

    def test_clone_honors_an_explicit_relationships_override(self):
        # An override arrives pristine; the restore must not clobber it with the
        # template's spec.
        _lp_ = self.p2s.linkp(self.df, [(('fm', 'port'), 'to')], wxh=(512, 512))
        _lp_._repr_svg_()
        _clone_ = _lp_.render_with(self.df, relationships=[('fm', 'to')])
        self.assertIn('<svg', _clone_._repr_svg_())
        self.assertEqual(_clone_.relationships_orig, [('fm', 'to')])
        self.assertEqual(_clone_.relationships,      [('fm', 'to')])

    def test_null_nodes_clone_still_fills(self):
        # null_nodes= reassigns df_orig to the FILLED frame, which already carries
        # __fm0__; re-expanding it from ('fm','port') recomputes the column, and the
        # fill is deterministic per row, so the null partners survive the round trip.
        _df_ = pl.DataFrame({'fm': ['a', 'b', None], 'port': ['1', '2', '3'],
                             'to': ['b', None, 'c']})
        _lp_ = self.p2s.linkp(_df_, [(('fm', 'port'), 'to')], wxh=(512, 512), null_nodes=True)
        _lp_._repr_svg_()
        _clone_ = _lp_.render_with(_lp_.df_orig)
        self.assertIn('<svg', _clone_._repr_svg_())
        self.assertEqual(_clone_.df['__fm0__'].to_list(), _lp_.df['__fm0__'].to_list())
        self.assertEqual(_clone_.df['to'].to_list(),      _lp_.df['to'].to_list())
        self.assertIsNotNone(self.p2s.panelize([[_lp_]]))


class TestSpreadLinePiTupleEndpoints(unittest.TestCase):
    """_filter_out_nodes() is the spreadlines twin of the linkpi defect: rewritten
    relationships against _spread_.df_orig."""

    @classmethod
    def setUpClass(cls):
        cls.p2s = Polars2SVG()

    def setUp(self):
        self.df = _make_df().with_columns(
            pl.Series('time', [datetime.datetime(2024, 1, 1 + i) for i in range(7)])
        )
        self.spread = self.p2s.spreadlinesp(self.df, [(('fm', 'port'), 'to')],
                                            ego='a|1', time='time')
        self.spread._repr_svg_()

    def test_filter_out_nodes_drops_the_selection(self):
        _out_ = _filter_out_nodes(self.spread, self.spread.df_orig, {'far|4'})
        self.assertEqual(len(_out_), 5)
        # the '|'-joined column is scratch -- it must not reach the pushed frame
        self.assertEqual(_out_.columns, self.spread.df_orig.columns)

    def test_filter_out_nodes_on_a_plain_endpoint(self):
        # 'c' is a to-side name only ('c|3' is what the fm side calls that row), so the
        # single to=='c' row goes and the c|3->a row stays -- the two sides are joined
        # differently and must not be conflated.
        _out_ = _filter_out_nodes(self.spread, self.spread.df_orig, {'c'})
        self.assertEqual(len(_out_), 6)

    def test_filter_out_nodes_string_relationships_unchanged(self):
        _spread_ = self.p2s.spreadlinesp(self.df, [('fm', 'to')], ego='a', time='time')
        _spread_._repr_svg_()
        _out_ = _filter_out_nodes(_spread_, _spread_.df_orig, {'far'})
        self.assertEqual(len(_out_), 5)
        self.assertEqual(_out_.columns, _spread_.df_orig.columns)


if __name__ == '__main__':
    unittest.main()
