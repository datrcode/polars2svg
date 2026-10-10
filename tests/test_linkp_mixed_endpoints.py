#
# linkp with endpoint columns of different types (PLANNING.md §5 C-linkp-mixed-endpoint-dtypes).
#
# A string column linked to an integer one used to raise inside __calculateGeometry__: the
# node positions were one replace_strict literal, and polars cannot build one from a mix of
# str and int keys.  A graph whose node keys mix types now looks its positions up by node
# name as a string, the name __nm__ draws it by; a same-type graph keeps the raw lookup.
#
# Every test reads the geometry back -- a render that silently drops a link is the failure
# this file exists to catch, and it would still produce an SVG.
#
import datetime as dt
import json
import os
import tempfile
import unittest
from unittest import mock

import networkx as nx
import polars as pl

from polars2svg import Polars2SVG
from polars2svg.linkp import LinkP


def _mixed_df():
    return pl.DataFrame({
        'h':   ['a', 'b', 'c', 'a'],
        'p':   ['x', 'y', 'x', 'y'],
        'd':   [2, 3, 1, 3],
        'n':   [5, 1, 2, 7],
        'lbl': ['l1', 'l2', 'l3', 'l4'],
        'ts':  [dt.datetime(2026, 1, 1, _h_) for _h_ in (1, 2, 3, 4)],
    })


def _coord_cols(lp):
    return [_c_ for _c_ in lp.df.columns if _c_.startswith('__rel') and _c_.endswith(('_wx__', '_wy__'))]


def _nodes(lp):
    return sorted(lp.df_node['__nm__'].explode().unique().to_list())


class _MixedEndpointsBase(unittest.TestCase):
    def setUp(self):
        self.p2s = Polars2SVG()

    def assertEveryEndpointPlaced(self, lp):
        '''Every row's from and to endpoint has world coordinates, so every link is drawn.'''
        _cols_ = _coord_cols(lp)
        self.assertEqual(len(_cols_), 4 * len(lp.relationships))
        _nulls_ = lp.df.select(pl.col(_cols_).is_null().sum()).row(0, named=True)
        self.assertEqual({_c_: _n_ for _c_, _n_ in _nulls_.items() if _n_}, {})

    def endpointXY(self, lp, rel, end, value):
        '''World (x, y) of the rows whose relationship `rel` endpoint `end` ('fm'/'to') is `value`.'''
        _col_ = lp.relationships[rel][0 if end == 'fm' else 1]
        _xy_ = (lp.df.filter(pl.col(_col_) == value)
                     .select(f'__rel{rel}_{end}_wx__', f'__rel{rel}_{end}_wy__').unique())
        self.assertEqual(len(_xy_), 1, f'{value!r} placed at more than one position')
        return _xy_.row(0)


class TestMixedEndpointsRender(_MixedEndpointsBase):
    def test_string_against_integer_column(self):
        lp = self.p2s.linkp(_mixed_df(), [('h', 'd')])
        self.assertEveryEndpointPlaced(lp)
        self.assertEqual(_nodes(lp), ['1', '2', '3', 'a', 'b', 'c'])

    def test_tuple_endpoint_against_integer_column(self):
        lp = self.p2s.linkp(_mixed_df(), [(('h', 'p'), 'd')])
        self.assertEveryEndpointPlaced(lp)
        self.assertEqual(_nodes(lp), ['1', '2', '3', 'a|x', 'a|y', 'b|y', 'c|x'])

    def test_two_relationships_each_of_one_type(self):
        lp = self.p2s.linkp(_mixed_df(), [('h', 'p'), ('d', 'n')])
        self.assertEveryEndpointPlaced(lp)
        self.assertEqual(_nodes(lp), ['1', '2', '3', '5', '7', 'a', 'b', 'c', 'x', 'y'])

    def test_every_feature_on_mixed_endpoints(self):
        _cases_ = {
            'vary sizes, count=':           dict(node_size='vary', link_size='vary', count='n'),
            'node-name colour, legend':     dict(node_color=self.p2s.COLOR_BY_NODE_NAME, legend=True),
            'field colour, legend':         dict(color='lbl', legend=True),
            'curve with arrows':            dict(link_shape='curve', link_arrows=True),
            'flowmap':                      dict(link_shape='flowmap', flowmap_backend='numpy'),
            'timing marks':                 dict(time='ts'),
            'node labels':                  dict(draw_node_labels=True),
        }
        for _name_, _kw_ in _cases_.items():
            with self.subTest(_name_):
                lp = self.p2s.linkp(_mixed_df(), [('h', 'd')], **_kw_)
                self.assertEveryEndpointPlaced(lp)
                self.assertEqual(_nodes(lp), ['1', '2', '3', 'a', 'b', 'c'])
                self.assertEqual(lp.svg.count('<circle'), 6)

    def test_link_labels_on_mixed_endpoints(self):
        lp = self.p2s.linkp(_mixed_df(), [('h', 'd', 'lbl')], draw_link_labels=True)
        self.assertEveryEndpointPlaced(lp)
        for _lbl_ in ('l1', 'l2', 'l3', 'l4'):
            self.assertIn(f'>{_lbl_}<', lp.svg)

    def test_null_nodes_on_mixed_endpoints(self):
        _df_ = _mixed_df().with_columns(pl.when(pl.col('d') == 1).then(None).otherwise(pl.col('d')).alias('d'))
        lp = self.p2s.linkp(_df_, [('h', 'd')], null_nodes=True)
        self.assertEveryEndpointPlaced(lp)
        self.assertEqual(len(_nodes(lp)), 6)

    def test_render_with_on_mixed_endpoints(self):
        lp = self.p2s.linkp(_mixed_df(), [('h', 'd')])
        _clone_ = lp.render_with(_mixed_df().head(2))
        self.assertEveryEndpointPlaced(_clone_)
        self.assertEqual(_nodes(_clone_), ['2', '3', 'a', 'b'])

    def test_convex_hulls_keyed_by_raw_node(self):
        _pos_ = {'a': (0, 0), 'b': (1, 0), 'c': (0, 1), 1: (1, 1), 2: (.3, .6), 3: (.7, .2)}
        _cases_ = [({'g': ['a', 'b', 'c']}, 1),
                   ({'g': [1, 2, 3]}, 1),
                   ({'^[a-c]$': 'letters', '^[0-9]$': 'digits'}, 2)]
        for _lu_, _hulls_ in _cases_:
            with self.subTest(_lu_):
                lp = self.p2s.linkp(_mixed_df(), [('h', 'd')], pos=dict(_pos_), convex_hull_lu=_lu_)
                self.assertEqual(lp.svg.count('<polygon'), _hulls_)


class TestMixedEndpointsPositions(_MixedEndpointsBase):
    def test_float_and_integer_endpoints_are_both_placed(self):
        # Python makes 1 and 1.0 one key, but they cast to '1' and '1.0'.  Keyed by str() of
        # the node set, one of them would find no position and its link would vanish.
        _df_ = pl.DataFrame({'h': ['a', 'b'], 'f': [1.0, 2.0], 'i': [1, 5]})
        lp = self.p2s.linkp(_df_, [('h', 'f'), ('h', 'i')])
        self.assertEveryEndpointPlaced(lp)
        self.assertEqual(_nodes(lp), ['1', '1.0', '2.0', '5', 'a', 'b'])

    def test_string_two_and_integer_two_are_one_node(self):
        _df_ = pl.DataFrame({'s': ['2', '9'], 'i': [2, 3]})
        lp = self.p2s.linkp(_df_, [('s', 'i')])
        self.assertEveryEndpointPlaced(lp)
        self.assertEqual(_nodes(lp), ['2', '3', '9'])
        self.assertEqual(self.endpointXY(lp, 0, 'fm', '2'), self.endpointXY(lp, 0, 'to', 2))

    def test_user_positions_with_raw_keys_are_honoured(self):
        _pos_ = {'a': (0.0, 0.0), 'b': (1.0, 0.0), 'c': (0.0, 1.0), 1: (1.0, 1.0), 2: (0.25, 0.75), 3: (0.75, 0.25)}
        lp = self.p2s.linkp(_mixed_df(), [('h', 'd')], pos=dict(_pos_))
        self.assertEveryEndpointPlaced(lp)
        for _v_ in ('a', 'b', 'c'):
            self.assertEqual(self.endpointXY(lp, 0, 'fm', _v_), _pos_[_v_])
        for _v_ in (1, 2, 3):
            self.assertEqual(self.endpointXY(lp, 0, 'to', _v_), _pos_[_v_])

    def test_new_nodes_get_positions_under_their_raw_keys(self):
        _pos_ = {'a': (0.0, 0.0)}
        lp = self.p2s.linkp(_mixed_df(), [('h', 'd')], pos=_pos_)
        self.assertEqual(set(lp.pos), {'a', 'b', 'c', 1, 2, 3})
        self.assertEqual(lp.pos['a'], (0.0, 0.0))

    def test_networkx_layout_fed_back_as_pos(self):
        _df_ = _mixed_df()
        _pos_ = nx.spring_layout(self.p2s.createNetworkXGraph(_df_, [('h', 'd')]), seed=1)
        lp = self.p2s.linkp(_df_, [('h', 'd')], pos=dict(_pos_))
        self.assertEveryEndpointPlaced(lp)
        for _v_ in (1, 2, 3):
            self.assertEqual(self.endpointXY(lp, 0, 'to', _v_), tuple(float(_c_) for _c_ in _pos_[_v_]))

    def test_saved_positions_of_an_integer_graph_reload_in_place(self):
        # A positions file's keys are strings, so an integer graph's own file mixes '1'
        # with 1 once loaded; it used to raise instead of drawing the saved layout.
        _df_ = pl.DataFrame({'a': [1, 2, 3], 'b': [2, 3, 1]})
        lp = self.p2s.linkp(_df_, [('a', 'b')])
        with tempfile.TemporaryDirectory() as _d_:
            _fn_ = os.path.join(_d_, 'pos.json')
            self.p2s.savePositions(_fn_, lp)
            with open(_fn_) as _f_:
                self.assertEqual(set(json.load(_f_)), {'1', '2', '3'})
            _pos_ = self.p2s.loadPositions(_fn_)
        lp2 = self.p2s.linkp(_df_, [('a', 'b')], pos=_pos_)
        self.assertEveryEndpointPlaced(lp2)
        for _v_ in (1, 2, 3):
            self.assertEqual(self.endpointXY(lp2, 0, 'fm', _v_), tuple(float(_c_) for _c_ in lp.pos[_v_]))
        self.assertEqual(set(lp2.pos), {'1', '2', '3'})


class TestLinkpiOnMixedEndpoints(_MixedEndpointsBase):
    '''linkpi on a mixed graph: the keys a selection, a drag and a push hand back keep the
    types the frame holds, and the selection narrows to the right rows (step 4).'''

    def setUp(self):
        super().setUp()
        self.df = pl.DataFrame({'h': ['a', 'b', 'c', 'a'], 'd': [2, 3, 1, 3], 'k': [10, 20, 30, 40]})
        self.lp = self.p2s.linkp(self.df, [('h', 'd')], wxh=(256, 256))
        self.v  = self.p2s.linkpi(self.lp)

    def test_a_drag_over_everything_selects_raw_keys(self):
        self.v.apply_drag_select(-10, -10, 300, 300)
        self.assertEqual(self.v.selected_entities, {'a', 'b', 'c', 1, 2, 3})

    def test_a_node_selection_narrows_to_incident_rows(self):
        _cases_ = [({2}, [10]), ({3}, [20, 40]), ({'a'}, [10, 40]), ({'c', 2}, [10, 30])]
        for _sel_, _rows_ in _cases_:
            with self.subTest(_sel_):
                self.v.selected_entities = set(_sel_)
                self.assertEqual(self.v.mvc.selectedDataFrame()['k'].to_list(), _rows_)

    def test_a_dragged_integer_node_moves_under_its_raw_key(self):
        _before_, _other_ = self.endpointXY(self.lp, 0, 'to', 3), self.endpointXY(self.lp, 0, 'to', 2)
        _one_px_ = abs(self.lp.yT_inv(1) - self.lp.yT_inv(0))
        self.v.selected_entities = {3}
        _updated_ = self.v.apply_move_selected(40, 0)
        self.assertEqual(set(_updated_), {3})
        self.assertNotIn('3', self.lp.pos)
        self.lp.renderSVG()
        _after_ = self.endpointXY(self.lp, 0, 'to', 3)
        self.assertGreater(_after_[0], _before_[0])
        # a drag goes through the node's pixel position and back (yT_inv), so y keeps to
        # within a pixel -- under 1 px on integer and mixed graphs alike (30 drags each)
        self.assertLess(abs(_after_[1] - _before_[1]), _one_px_)
        self.assertEqual(self.endpointXY(self.lp, 0, 'to', 2), _other_)

    def test_pushing_a_selection_away_removes_its_rows(self):
        # the 'x' key: the view's own stack moves now; the controller hears it on the event
        # loop, so read the level the view pushed
        self.v.selected_entities = {3}
        self.assertTrue(self.v.apply_push_selected())
        self.assertEqual(self.v.df_level, 1)
        self.assertEqual(sorted(self.v.dfs[1]['k'].to_list()), [10, 30])
        self.assertEqual(set(self.v.graphs[1].nodes()), {'a', 'c', 1, 2})
        self.assertEveryEndpointPlaced(self.v.dfs_layout[1])


class TestSameTypeGraphsKeepTheRawLookup(_MixedEndpointsBase):
    def setUp(self):
        super().setUp()
        _helper_ = LinkP.__positionsByNodeString__
        _patch_ = mock.patch.object(LinkP, '__positionsByNodeString__', autospec=True, side_effect=_helper_)
        self.by_string = _patch_.start()
        self.addCleanup(_patch_.stop)

    def test_integer_graph(self):
        _df_ = pl.DataFrame({'a': [1, 2, 3], 'b': [2, 3, 1]})
        _pos_ = {1: (0.0, 0.0), 2: (1.0, 0.0)}
        lp = self.p2s.linkp(_df_, [('a', 'b')], pos=_pos_)
        self.assertEveryEndpointPlaced(lp)
        self.assertEqual(self.by_string.call_count, 0)
        self.assertEqual({type(_k_) for _k_ in lp.pos}, {int})
        self.assertEqual(self.endpointXY(lp, 0, 'fm', 2), (1.0, 0.0))

    def test_string_graph(self):
        _df_ = pl.DataFrame({'a': ['x', 'y'], 'b': ['y', 'z']})
        lp = self.p2s.linkp(_df_, [('a', 'b')])
        self.assertEqual(self.by_string.call_count, 0)
        self.assertEveryEndpointPlaced(lp)
        self.assertEqual({type(_k_) for _k_ in lp.pos}, {str})
        self.assertEqual(_nodes(lp), ['x', 'y', 'z'])

    def test_a_mixed_graph_does_take_the_string_path(self):
        # the control for the two above: the spy does see the string path when it runs
        lp = self.p2s.linkp(_mixed_df(), [('h', 'd')])
        self.assertEveryEndpointPlaced(lp)
        self.assertEqual(self.by_string.call_count, 1)


if __name__ == '__main__':
    unittest.main()
