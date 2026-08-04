#
# Null relationship endpoints.
#
# A row whose fm/to pair is only half populated used to produce an entity that linkp drew
# and hit-tested but createNetworkXGraph() never added, because its row filter drops the
# whole row when either endpoint is null. Every graph-derived interactive op then disagreed
# with the view: 'x' (remove selected) raised NetworkXError on the first such entity and
# aborted the filter, and 'q' (invert selection) could never reach one.
#
# Two behaviors are covered here:
#   default          -- those entities are isolated nodes of the graph; nothing new is drawn
#   null_nodes=True  -- each one gets its OWN null partner node and a visible stub edge
#
import unittest
import asyncio
import polars as pl
from polars2svg import Polars2SVG


def _df_half_edges():
    # a-b and b-c are ordinary edges; the two @ entities only ever appear opposite a null.
    return pl.DataFrame({
        'fm': ['a', 'b', 'restore2k@ayna.com', 'seggers@rothers.com'],
        'to': ['b', 'c', None,                 None],
    })


def _pos():
    return {'a': (0.0, 0.0), 'b': (1.0, 0.0), 'c': (0.5, 0.9),
            'restore2k@ayna.com': (0.3, 0.4), 'seggers@rothers.com': (0.7, 0.4)}


_RELS_ = [('fm', 'to')]
_WXH_  = (512, 512)


class TestNullEndpointsDefault(unittest.TestCase):
    def setUp(self):
        self.p2s  = Polars2SVG()
        self.lp   = self.p2s.linkp(_df_half_edges(), _RELS_, dict(_pos()), wxh=_WXH_)
        self.ctrl = self.p2s.linkpi(self.lp)
        self.lp.renderSVG()

    def _press(self, key):
        self.ctrl.key_op_finished = key
        asyncio.run(self.ctrl.applyKeyOp(None))

    def test_half_edge_entity_is_a_graph_node(self):
        self.assertIn('restore2k@ayna.com', self.ctrl.graphs[0])

    def test_every_drawn_node_is_a_graph_node(self):
        # The invariant the interactive ops rely on.
        self.assertEqual(set(self.lp.all_nodes) - set(self.ctrl.graphs[0].nodes()), set())

    def test_half_edge_entity_is_isolated(self):
        self.assertEqual(self.ctrl.graphs[0].degree('restore2k@ayna.com'), 0)

    def test_real_edges_are_unaffected(self):
        self.assertTrue(self.ctrl.graphs[0].has_edge('a', 'b'))
        self.assertTrue(self.ctrl.graphs[0].has_edge('b', 'c'))

    def test_no_sentinel_leaks_into_the_default_graph(self):
        for _node_ in self.ctrl.graphs[0].nodes():
            self.assertFalse(self.p2s.isNullNode(_node_))

    def test_drag_select_all_then_push_does_not_raise(self):
        # The reported crash: select everything, press 'x'.
        self.ctrl.apply_drag_select(0, 0, _WXH_[0], _WXH_[1])
        self.assertIn('restore2k@ayna.com', self.ctrl.selected_entities)
        self._press('x')   # NetworkXError before the fix

    def test_pushing_only_the_half_edge_entities_keeps_the_real_edges(self):
        self.ctrl.selected_entities = {'restore2k@ayna.com', 'seggers@rothers.com'}
        self.assertTrue(self.ctrl.apply_push_selected())
        self.assertEqual(self.ctrl.df_level, 1)
        self.assertTrue(self.ctrl.graphs[1].has_edge('a', 'b'))
        self.assertNotIn('restore2k@ayna.com', self.ctrl.graphs[1])

    def test_invert_selection_reaches_the_half_edge_entities(self):
        self.ctrl.selected_entities = {'a'}
        self._press('q')
        self.assertIn('restore2k@ayna.com', self.ctrl.selected_entities)
        self.assertNotIn('a', self.ctrl.selected_entities)

    def test_expand_from_an_isolated_entity_does_not_raise(self):
        self.ctrl.selected_entities = {'restore2k@ayna.com'}
        self._press('e')
        self.assertEqual(self.ctrl.selected_entities, {'restore2k@ayna.com'})

    def test_three_part_relationship_with_a_null_label_field(self):
        # flattenTuple() puts the label field in the row filter too, so a null there used to
        # strand BOTH endpoints outside the graph even though the edge is fully populated.
        _df_ = pl.DataFrame({'fm': ['a', 'x'], 'to': ['b', 'y'], 'lbl': ['sent', None]})
        _g_  = self.p2s.createNetworkXGraph(_df_, [('fm', 'to', 'lbl')])
        self.assertIn('x', _g_)
        self.assertIn('y', _g_)
        self.assertTrue(_g_.has_edge('a', 'b'))

    def test_rows_with_both_endpoints_null_add_nothing(self):
        _df_ = pl.DataFrame({'fm': ['a', None], 'to': ['b', None]})
        self.assertEqual(set(self.p2s.createNetworkXGraph(_df_, _RELS_).nodes()), {'a', 'b'})


class TestNullEndpointsMaterialized(unittest.TestCase):
    def setUp(self):
        self.p2s  = Polars2SVG()
        self.lp   = self.p2s.linkp(_df_half_edges(), _RELS_, dict(_pos()), wxh=_WXH_,
                                   null_nodes=True)
        self.ctrl = self.p2s.linkpi(self.lp)
        self.lp.renderSVG()

    def test_each_entity_gets_its_own_null_partner(self):
        # The design decision: private, not one shared hub -- two records with a missing
        # endpoint are not asserted to point at the same thing.
        _g_ = self.ctrl.graphs[0]
        self.assertTrue(_g_.has_edge('restore2k@ayna.com', self.p2s.nullNode('restore2k@ayna.com')))
        self.assertTrue(_g_.has_edge('seggers@rothers.com', self.p2s.nullNode('seggers@rothers.com')))
        self.assertNotEqual(self.p2s.nullNode('restore2k@ayna.com'),
                            self.p2s.nullNode('seggers@rothers.com'))

    def test_null_partners_are_degree_one(self):
        # A shared sentinel would show up here as a hub.
        for _node_ in self.ctrl.graphs[0].nodes():
            if self.p2s.isNullNode(_node_):
                self.assertEqual(self.ctrl.graphs[0].degree(_node_), 1)

    def test_the_two_entities_are_not_connected_to_each_other(self):
        import networkx as nx
        self.assertFalse(nx.has_path(self.ctrl.graphs[0],
                                     'restore2k@ayna.com', 'seggers@rothers.com'))

    def test_every_drawn_node_is_a_graph_node(self):
        self.assertEqual(set(self.lp.all_nodes) - set(self.ctrl.graphs[0].nodes()), set())

    def test_null_partner_is_drawn(self):
        self.assertIn(self.p2s.nullNode('restore2k@ayna.com'), self.lp.all_nodes)

    def test_stub_survives_a_push(self):
        # filterDataFrameByGraph() keeps only rows that are edges, so a stub row only
        # survives a filter because the fill made it a real edge.
        self.ctrl.selected_entities = {'c'}
        self.assertTrue(self.ctrl.apply_push_selected())
        self.assertIn('restore2k@ayna.com', self.ctrl.graphs[self.ctrl.df_level])
        self.assertTrue(self.ctrl.graphs[self.ctrl.df_level].has_edge(
            'restore2k@ayna.com', self.p2s.nullNode('restore2k@ayna.com')))

    def test_removing_the_entity_removes_its_null_partner_too(self):
        self.ctrl.selected_entities = {'restore2k@ayna.com'}
        self.assertTrue(self.ctrl.apply_push_selected())
        _g_ = self.ctrl.graphs[self.ctrl.df_level]
        self.assertNotIn(self.p2s.nullNode('restore2k@ayna.com'), _g_)

    def test_label_shows_null_not_the_sentinel(self):
        _lp_ = self.p2s.linkp(_df_half_edges(), _RELS_, dict(_pos()), wxh=_WXH_,
                              null_nodes=True, draw_node_labels=True)
        _svg_ = _lp_.renderSVG()
        self.assertIn('(null)', _svg_)
        self.assertNotIn(self.p2s.NULL_NODE_PREFIX, _svg_)

    def test_both_null_row_is_left_alone(self):
        _df_ = pl.DataFrame({'fm': ['a', None], 'to': ['b', None]})
        _out_ = self.p2s.nullFillEndpoints(_df_, _RELS_)
        self.assertIsNone(_out_['fm'][1])
        self.assertIsNone(_out_['to'][1])

    def test_fill_is_idempotent(self):
        _once_  = self.p2s.nullFillEndpoints(_df_half_edges(), _RELS_)
        _twice_ = self.p2s.nullFillEndpoints(_once_, _RELS_)
        self.assertTrue(_once_.equals(_twice_))

    def test_fill_leaves_a_frame_without_nulls_untouched(self):
        _df_ = pl.DataFrame({'fm': [1, 2], 'to': [2, 3]})   # dtype preserved: no nulls
        _out_ = self.p2s.nullFillEndpoints(_df_, _RELS_)
        self.assertEqual(_out_['fm'].dtype, pl.Int64)


class TestSelectionStaysASubsetOfItsLevel(unittest.TestCase):
    # The other way the selection and the graph diverged: level changes that did not
    # re-intersect. 'F' pushes a superset, so popping off it lands on a graph that never
    # had the nodes selected up there.
    def setUp(self):
        self.p2s  = Polars2SVG()
        _df_      = pl.DataFrame({'fm': ['a', 'b', 'c', 'd'], 'to': ['b', 'c', 'd', 'e']})
        _pos_     = {_n_: (i * 0.2, 0.5) for i, _n_ in enumerate('abcde')}
        self.lp   = self.p2s.linkp(_df_, _RELS_, _pos_, wxh=_WXH_)
        self.ctrl = self.p2s.linkpi(self.lp)

    def test_pop_re_intersects_the_selection(self):
        self.ctrl.selected_entities = {'e'}
        self.assertTrue(self.ctrl.apply_push_selected())   # level 1, without 'e'
        self.ctrl.selected_entities = set()
        self.ctrl.apply_node_expansion()                   # level 2, 'e' back
        self.assertIn('e', self.ctrl.graphs[self.ctrl.df_level])
        self.ctrl.selected_entities = {'e'}
        self.ctrl.apply_pop()                              # level 1 again
        self.assertNotIn('e', self.ctrl.selected_entities)

    def test_stack_position_re_intersects_the_selection(self):
        self.ctrl.selected_entities = {'e'}
        self.assertTrue(self.ctrl.apply_push_selected())
        self.ctrl.setStackPostion(0)
        self.ctrl.selected_entities = {'e'}
        self.ctrl.setStackPostion(1)
        self.assertNotIn('e', self.ctrl.selected_entities)

    def test_push_tolerates_a_selection_the_graph_never_had(self):
        self.ctrl.selected_entities = {'a', 'not-a-node-at-all'}
        self.assertTrue(self.ctrl.apply_push_selected())
        self.assertNotIn('a', self.ctrl.graphs[self.ctrl.df_level])


if __name__ == '__main__':
    unittest.main()
