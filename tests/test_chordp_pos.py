import unittest
import polars as pl
import numpy as np
from polars2svg import Polars2SVG
from polars2svg.chordp import _pos_components_, _pos_to_order_angle_, _pos_to_order_pca_

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# Connected chain: a – b – c – d
_DF_CONN_ = pl.DataFrame({'fm': ['a', 'b', 'c'], 'to': ['b', 'c', 'd']})
_RELS_    = [('fm', 'to')]

# Disconnected: component {a,b} and component {c,d}, no bridge between them
_DF_DISC_ = pl.DataFrame({'fm': ['a', 'c'], 'to': ['b', 'd']})

# Cardinal-point layout — centroid is (0, 0), so angles equal raw atan2 values:
#   south=-π/2, east=0, north=+π/2, west=+π
# Expected angle order: ['south', 'east', 'north', 'west']
_POS_CARDINAL_ = {
    'east':  ( 1.0,  0.0),
    'north': ( 0.0,  1.0),
    'west':  (-1.0,  0.0),
    'south': ( 0.0, -1.0),
}
_EXPECTED_ANGLE_ORDER_ = ['south', 'east', 'north', 'west']

# Linear layout — PC1 is clearly the x-axis.
# Tiny y-offsets break perfect degeneracy so SVD gives a stable sign.
_POS_LINEAR_ = {
    'leftmost':  (-3.0, 0.01),
    'left':      (-1.0, 0.01),
    'right':     ( 1.0, 0.01),
    'rightmost': ( 3.0, 0.01),
}

# Two-cluster disconnected layout for component-grouping tests.
# Cluster 1 (right): a and b near (2, 0)
# Cluster 2 (left):  c and d near (-2, 0)
_POS_DISC_ = {
    'a': ( 2.0,  0.1),
    'b': ( 2.0, -0.1),
    'c': (-2.0,  0.1),
    'd': (-2.0, -0.1),
}

# Connected-graph pos used in ChP integration tests.
# Same cardinal layout as _POS_CARDINAL_ but with node names matching _DF_CONN_.
# Angles from centroid (0,0): d=-π/2, a=0, b=+π/2, c=+π
# Expected angle order: ['d', 'a', 'b', 'c']
_POS_CONN_ = {
    'a': ( 1.0,  0.0),
    'b': ( 0.0,  1.0),
    'c': (-1.0,  0.0),
    'd': ( 0.0, -1.0),
}
_EXPECTED_CONN_ORDER_ = ['d', 'a', 'b', 'c']


# ---------------------------------------------------------------------------
# _pos_components_
# ---------------------------------------------------------------------------

class TestPosComponents(unittest.TestCase):
    """Unit tests for _pos_components_(): derives connected components from an edge DataFrame."""

    def _edge_df(self, pairs):
        fm, to = zip(*pairs)
        return pl.DataFrame({'__fm__': list(fm), '__to__': list(to)})

    def test_connected_graph_is_one_component(self):
        df = self._edge_df([('a', 'b'), ('b', 'c'), ('c', 'd')])
        comps = _pos_components_(df)
        self.assertEqual(len(comps), 1)
        self.assertEqual(comps[0], {'a', 'b', 'c', 'd'})

    def test_disconnected_graph_two_components(self):
        df = self._edge_df([('a', 'b'), ('c', 'd')])
        comps = _pos_components_(df)
        self.assertEqual(len(comps), 2)
        comp_sets = sorted(comps, key=lambda s: min(s))
        self.assertEqual(comp_sets[0], {'a', 'b'})
        self.assertEqual(comp_sets[1], {'c', 'd'})

    def test_three_components(self):
        df = self._edge_df([('a', 'b'), ('c', 'd'), ('e', 'f')])
        comps = _pos_components_(df)
        self.assertEqual(len(comps), 3)

    def test_all_nodes_covered(self):
        df = self._edge_df([('a', 'b'), ('c', 'd'), ('e', 'f')])
        comps = _pos_components_(df)
        self.assertEqual(set().union(*comps), {'a', 'b', 'c', 'd', 'e', 'f'})

    def test_components_are_disjoint(self):
        df = self._edge_df([('a', 'b'), ('c', 'd')])
        c1, c2 = _pos_components_(df)
        self.assertTrue(c1.isdisjoint(c2))

    def test_empty_df_returns_empty(self):
        df = pl.DataFrame({'__fm__': pl.Series([], dtype=pl.String),
                           '__to__': pl.Series([], dtype=pl.String)})
        self.assertEqual(_pos_components_(df), [])


# ---------------------------------------------------------------------------
# _pos_to_order_angle_
# ---------------------------------------------------------------------------

class TestPosToOrderAngle(unittest.TestCase):

    # --- basics ---

    def test_empty_pos_returns_empty(self):
        self.assertEqual(_pos_to_order_angle_({}), [])

    def test_single_node_returns_that_node(self):
        self.assertEqual(_pos_to_order_angle_({'a': (1.0, 0.0)}), ['a'])

    def test_output_contains_all_pos_keys(self):
        order = _pos_to_order_angle_(_POS_CARDINAL_)
        self.assertEqual(set(order), set(_POS_CARDINAL_))

    def test_output_length_equals_pos_size(self):
        order = _pos_to_order_angle_(_POS_CARDINAL_)
        self.assertEqual(len(order), len(_POS_CARDINAL_))

    # --- ordering correctness ---

    def test_cardinal_points_known_order(self):
        order = _pos_to_order_angle_(_POS_CARDINAL_)
        self.assertEqual(order, _EXPECTED_ANGLE_ORDER_)

    def test_numpy_array_values_accepted(self):
        pos = {n: np.array(v) for n, v in _POS_CARDINAL_.items()}
        order = _pos_to_order_angle_(pos)
        self.assertEqual(order, _EXPECTED_ANGLE_ORDER_)

    # --- components=None degeneracy ---

    def test_none_components_same_as_omitted(self):
        self.assertEqual(
            _pos_to_order_angle_(_POS_CARDINAL_, components=None),
            _pos_to_order_angle_(_POS_CARDINAL_),
        )

    def test_single_component_same_as_none(self):
        all_nodes = set(_POS_DISC_)
        self.assertEqual(
            _pos_to_order_angle_(_POS_DISC_, components=[all_nodes]),
            _pos_to_order_angle_(_POS_DISC_),
        )

    # --- disconnected / component grouping ---

    def test_disconnected_components_are_contiguous(self):
        comps = [{'a', 'b'}, {'c', 'd'}]
        order = _pos_to_order_angle_(_POS_DISC_, components=comps)
        idx = {n: i for i, n in enumerate(order)}
        self.assertEqual(abs(idx['a'] - idx['b']), 1)
        self.assertEqual(abs(idx['c'] - idx['d']), 1)

    def test_disconnected_all_nodes_present(self):
        comps = [{'a', 'b'}, {'c', 'd'}]
        order = _pos_to_order_angle_(_POS_DISC_, components=comps)
        self.assertEqual(set(order), {'a', 'b', 'c', 'd'})
        self.assertEqual(len(order), 4)

    def test_disconnected_no_interleaving(self):
        comps = [{'a', 'b'}, {'c', 'd'}]
        order = _pos_to_order_angle_(_POS_DISC_, components=comps)
        # The two components must appear as two contiguous blocks
        first_half  = set(order[:2])
        second_half = set(order[2:])
        self.assertTrue(
            first_half in ({'a', 'b'}, {'c', 'd'}) and
            second_half in ({'a', 'b'}, {'c', 'd'}) and
            first_half != second_half
        )


# ---------------------------------------------------------------------------
# _pos_to_order_pca_
# ---------------------------------------------------------------------------

class TestPosToOrderPca(unittest.TestCase):

    # --- basics ---

    def test_empty_pos_returns_empty(self):
        self.assertEqual(_pos_to_order_pca_({}), [])

    def test_single_node_returns_that_node(self):
        self.assertEqual(_pos_to_order_pca_({'a': (1.0, 0.0)}), ['a'])

    def test_output_contains_all_pos_keys(self):
        order = _pos_to_order_pca_(_POS_LINEAR_)
        self.assertEqual(set(order), set(_POS_LINEAR_))

    def test_output_length_equals_pos_size(self):
        order = _pos_to_order_pca_(_POS_LINEAR_)
        self.assertEqual(len(order), len(_POS_LINEAR_))

    # --- ordering correctness ---

    def test_linear_layout_extremes_at_ends(self):
        # PC1 direction may be ± so we only assert the extremes are at the ends
        order = _pos_to_order_pca_(_POS_LINEAR_)
        self.assertIn(order[0],  {'leftmost', 'rightmost'})
        self.assertIn(order[-1], {'leftmost', 'rightmost'})
        self.assertNotEqual(order[0], order[-1])

    def test_linear_layout_inner_nodes_between_extremes(self):
        order = _pos_to_order_pca_(_POS_LINEAR_)
        self.assertEqual(set(order[1:-1]), {'left', 'right'})

    def test_numpy_array_values_accepted(self):
        pos = {n: np.array(v) for n, v in _POS_LINEAR_.items()}
        order = _pos_to_order_pca_(pos)
        self.assertIn(order[0],  {'leftmost', 'rightmost'})
        self.assertIn(order[-1], {'leftmost', 'rightmost'})

    # --- components=None degeneracy ---

    def test_none_components_same_as_omitted(self):
        self.assertEqual(
            _pos_to_order_pca_(_POS_LINEAR_, components=None),
            _pos_to_order_pca_(_POS_LINEAR_),
        )

    def test_single_component_same_as_none(self):
        all_nodes = set(_POS_DISC_)
        self.assertEqual(
            _pos_to_order_pca_(_POS_DISC_, components=[all_nodes]),
            _pos_to_order_pca_(_POS_DISC_),
        )

    # --- disconnected / component grouping ---

    def test_disconnected_components_are_contiguous(self):
        comps = [{'a', 'b'}, {'c', 'd'}]
        order = _pos_to_order_pca_(_POS_DISC_, components=comps)
        idx = {n: i for i, n in enumerate(order)}
        self.assertEqual(abs(idx['a'] - idx['b']), 1)
        self.assertEqual(abs(idx['c'] - idx['d']), 1)

    def test_disconnected_all_nodes_present(self):
        comps = [{'a', 'b'}, {'c', 'd'}]
        order = _pos_to_order_pca_(_POS_DISC_, components=comps)
        self.assertEqual(set(order), {'a', 'b', 'c', 'd'})
        self.assertEqual(len(order), 4)

    def test_disconnected_no_interleaving(self):
        comps = [{'a', 'b'}, {'c', 'd'}]
        order = _pos_to_order_pca_(_POS_DISC_, components=comps)
        first_half  = set(order[:2])
        second_half = set(order[2:])
        self.assertTrue(
            first_half in ({'a', 'b'}, {'c', 'd'}) and
            second_half in ({'a', 'b'}, {'c', 'd'}) and
            first_half != second_half
        )


# ---------------------------------------------------------------------------
# ChP integration: pos= parameter
# ---------------------------------------------------------------------------

class TestChordPPos(unittest.TestCase):

    def setUp(self):
        self.p2s = Polars2SVG()

    def _chordp(self, **kwargs):
        return self.p2s.chordp(df=_DF_CONN_, relationships=_RELS_, **kwargs)

    # --- parameter acceptance ---

    def test_pos_kwarg_renders(self):
        cp = self._chordp(pos=_POS_CONN_)
        self.assertIn('<svg', cp.svg)
        self.assertIn('<path', cp.svg)

    def test_pos_positional_arg_renders(self):
        cp = self.p2s.chordp(_DF_CONN_, _RELS_, _POS_CONN_)
        self.assertIn('<svg', cp.svg)

    def test_pos_positional_and_kwarg_raises(self):
        with self.assertRaises(Exception):
            self.p2s.chordp(_DF_CONN_, _RELS_, _POS_CONN_, pos=_POS_CONN_)

    def test_pos_default_is_empty_dict(self):
        cp = self._chordp()
        self.assertEqual(cp.pos, {})

    def test_pos_stored_on_instance(self):
        cp = self._chordp(pos=_POS_CONN_)
        self.assertEqual(cp.pos, _POS_CONN_)

    # --- ordering ---

    def test_pos_all_nodes_in_order(self):
        cp = self._chordp(pos=_POS_CONN_)
        self.assertEqual(set(cp.order), {'a', 'b', 'c', 'd'})

    def test_pos_produces_angle_order(self):
        cp = self._chordp(pos=_POS_CONN_)
        self.assertEqual(cp.order, _EXPECTED_CONN_ORDER_)

    def test_explicit_order_overrides_pos(self):
        explicit = ['c', 'a', 'd', 'b']
        cp = self._chordp(pos=_POS_CONN_, order=explicit)
        self.assertEqual(cp.order, explicit)

    def test_no_pos_uses_leaf_walk(self):
        cp_default = self._chordp()
        cp_pos     = self._chordp(pos=_POS_CONN_)
        # The two methods are not guaranteed to agree; just confirm pos changes the order
        self.assertNotEqual(cp_pos.order, cp_default.order)

    # --- partial pos: missing data nodes appended ---

    def test_node_absent_from_pos_appended_at_end(self):
        # pos covers a, b, c only — d is in the data but not in pos
        partial_pos = {k: v for k, v in _POS_CONN_.items() if k != 'd'}
        cp = self._chordp(pos=partial_pos)
        self.assertIn('d', cp.order)
        last_pos_node_idx = max(cp.order.index(n) for n in partial_pos)
        self.assertGreater(cp.order.index('d'), last_pos_node_idx)

    # --- numpy array values (mirrors nx.spring_layout output) ---

    def test_pos_with_numpy_arrays_renders(self):
        pos_np = {n: np.array(v) for n, v in _POS_CONN_.items()}
        cp = self._chordp(pos=pos_np)
        self.assertIn('<svg', cp.svg)
        self.assertEqual(cp.order, _EXPECTED_CONN_ORDER_)

    # --- disconnected graph ---

    def test_pos_disconnected_renders(self):
        cp = self.p2s.chordp(df=_DF_DISC_, relationships=_RELS_, pos=_POS_DISC_)
        self.assertIn('<svg', cp.svg)

    def test_pos_disconnected_components_contiguous(self):
        cp = self.p2s.chordp(df=_DF_DISC_, relationships=_RELS_, pos=_POS_DISC_)
        idx = {n: i for i, n in enumerate(cp.order)}
        self.assertEqual(abs(idx['a'] - idx['b']), 1)
        self.assertEqual(abs(idx['c'] - idx['d']), 1)

    # --- SM_X takes precedence over pos ---

    def test_sm_x_overrides_pos(self):
        tmpl = self.p2s.chordp(df=_DF_CONN_, relationships=_RELS_,
                               pos=_POS_CONN_, wxh=(128, 128),
                               sm_shared={self.p2s.SM_X})
        sm = self.p2s.smallp(_DF_CONN_, tmpl, 'fm')
        self.assertIn('<svg', sm.svg)


if __name__ == '__main__':
    unittest.main()
