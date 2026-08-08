import unittest
import polars as pl
from polars2svg import Polars2SVG

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# 5 nodes (a..e) in a ring, plus one a->c chord
_DF_ = pl.DataFrame({'fm': ['a', 'b', 'c', 'd', 'e', 'a'],
                     'to': ['b', 'c', 'd', 'e', 'a', 'c']})

_WXH_ = (400, 400)


class TestChordPPartialOrder(unittest.TestCase):
    '''A partial order= used to drop every unlisted node -- its arc and its ribbons
    vanished silently.  Default is now to append them; p2s.REMAINDERp merges them.'''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def _chordp(self, **kwargs):
        return self.p2s.chordp(_DF_, [('fm', 'to')], wxh=_WXH_, **kwargs)

    # -- default: append ----------------------------------------------------

    def test_partial_order_appends_unlisted_nodes(self):
        cp = self._chordp(order=['a', 'b', 'c'])
        self.assertEqual(cp.order, ['a', 'b', 'c', 'd', 'e'])
        self.assertEqual(cp.df_node['__nm__'].to_list(), ['a', 'b', 'c', 'd', 'e'])

    def test_partial_order_keeps_listed_nodes_first(self):
        cp = self._chordp(order=['e', 'c'])
        self.assertEqual(cp.order[:2], ['e', 'c'])
        self.assertEqual(sorted(cp.order[2:]), ['a', 'b', 'd'])

    def test_partial_order_hides_nothing(self):
        '''The regression: a partial order rendered fewer ribbons than a full one.'''
        full    = self._chordp()
        partial = self._chordp(order=['a', 'b', 'c'])
        self.assertEqual(partial._repr_svg_().count('<path'),
                         full._repr_svg_().count('<path'))

    def test_partial_order_appends_under_vary(self):
        cp = self._chordp(order=['a', 'b'], node_size='vary')
        self.assertEqual(sorted(cp.df_node['__nm__'].to_list()), ['a', 'b', 'c', 'd', 'e'])

    def test_order_may_name_absent_nodes(self):
        '''Listing a node that is not in the data still reserves its arc (unchanged).'''
        cp = self._chordp(order=['a', 'zz', 'b'])
        self.assertIn('zz', cp.df_node['__nm__'].to_list())

    # -- REMAINDERp: merge --------------------------------------------------

    def test_remainder_merges_unlisted_into_one_arc(self):
        cp = self._chordp(order=['a', 'b', 'c', self.p2s.REMAINDERp])
        self.assertEqual(cp.order, ['a', 'b', 'c', 'remainder'])
        self.assertEqual(cp.df_node['__nm__'].to_list(), ['a', 'b', 'c', 'remainder'])

    def test_remainder_honours_sentinel_position(self):
        cp = self._chordp(order=[self.p2s.REMAINDERp, 'a', 'b'])
        self.assertEqual(cp.order[0], 'remainder')
        self.assertEqual(cp.order[1:], ['a', 'b'])

    def test_remainder_alone_collapses_everything(self):
        cp = self._chordp(order=[self.p2s.REMAINDERp])
        self.assertEqual(cp.order, ['remainder'])

    def test_remainder_conserves_edge_weight(self):
        '''Merging must move weight onto the bucket, never discard it.'''
        full = self._chordp(node_size='vary')
        rem  = self._chordp(node_size='vary', order=['a', self.p2s.REMAINDERp])
        self.assertEqual(rem.df_edge_weights['__count__'].sum(),
                         full.df_edge_weights['__count__'].sum())

    def test_remainder_internal_edges_become_self_loops(self):
        '''b->c, c->d and d->e are wholly inside the remainder once only 'a' is listed.'''
        cp    = self._chordp(order=['a', self.p2s.REMAINDERp])
        _rows_ = set(zip(cp.df_edge_weights['__fm__'].to_list(),
                         cp.df_edge_weights['__to__'].to_list()))
        self.assertIn(('remainder', 'remainder'), _rows_)
        cp._repr_svg_()  # a self-loop must not blow up the ribbon geometry

    def test_remainder_with_no_unlisted_nodes_drops_the_sentinel(self):
        cp = self._chordp(order=['a', 'b', 'c', 'd', 'e', self.p2s.REMAINDERp])
        self.assertEqual(cp.order, ['a', 'b', 'c', 'd', 'e'])

    def test_remainder_casts_integer_node_ids(self):
        '''One name has to stand for many nodes, so integer ids become strings.'''
        df = pl.DataFrame({'fm': [1, 2, 3, 4], 'to': [2, 3, 4, 1]})
        cp = self.p2s.chordp(df, [('fm', 'to')], wxh=_WXH_, order=[1, 2, self.p2s.REMAINDERp])
        self.assertEqual(cp.order, ['1', '2', 'remainder'])
        self.assertEqual(cp.df_node['__nm__'].to_list(), ['1', '2', 'remainder'])

    def test_remainder_label_collision_raises(self):
        df = pl.DataFrame({'fm': ['remainder', 'b'], 'to': ['b', 'c']})
        with self.assertRaises(ValueError):
            self.p2s.chordp(df, [('fm', 'to')], wxh=_WXH_,
                            order=['remainder', self.p2s.REMAINDERp])

    # -- arc spacing --------------------------------------------------------

    def test_arcs_and_gaps_fill_the_circle(self):
        '''Gap allocation counted nodes_all while the arcs came from df_node.  The two
        diverge whenever order= names a node absent from the data ('zz' below): the
        circle was then divided into too few slices and the arcs ran past 360 degrees.'''
        # Every case needs >1 arc: the gap is measured between consecutive arcs, and a
        # lone arc still carries its gap with no second arc to read it from.
        for _order_ in (None, ['a', 'b', 'c'], ['a', 'zz', 'b'],
                        ['a', 'zz', self.p2s.REMAINDERp]):
            cp = self._chordp() if _order_ is None else self._chordp(order=_order_)
            _arcs_ = list(cp.node_to_arc.values())
            _span_ = sum(_a1_ - _a0_ for _a0_, _a1_ in _arcs_)
            _gap_  = _arcs_[1][0] - _arcs_[0][1]
            self.assertAlmostEqual(_span_ + _gap_ * len(_arcs_), 360.0, places=6,
                                   msg=f'order={_order_} did not tile the circle')

    def test_remainder_arc_is_not_narrower_than_a_plain_arc(self):
        '''Four arcs from a bucket must be spaced like four plain arcs.'''
        bucket = self._chordp(order=['a', 'b', 'c', self.p2s.REMAINDERp])
        plain  = self.p2s.chordp(pl.DataFrame({'fm': ['a', 'b'], 'to': ['c', 'd']}),
                                 [('fm', 'to')], wxh=_WXH_, order=['a', 'b', 'c', 'd'])
        _b_ = [a1 - a0 for a0, a1 in bucket.node_to_arc.values()]
        _p_ = [a1 - a0 for a0, a1 in plain.node_to_arc.values()]
        self.assertAlmostEqual(max(_b_), max(_p_), places=6)


if __name__ == '__main__':
    unittest.main()
