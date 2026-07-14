"""spreadlinepi interactive wrapper: hit-test helpers, selection set-ops,
drag/click selection, key operations ('x', 'X', 'c'), selection broadcast,
and stack display() cache management.

Param watchers for drag/key ops are detached in the fixtures so the async
callbacks can be driven directly and deterministically with asyncio.run().
"""
import asyncio
import datetime
import types
import unittest

import polars as pl

from polars2svg import Polars2SVG
from polars2svg.spreadlinepi import (
    spreadlinepi,
    _to_viewbox_coords,
    _nodes_at_xy,
    _nodes_in_rect,
    _expand_ego,
    _apply_set_op,
    _filter_out_nodes,
)


def _make_df():
    return pl.DataFrame({
        'fm':   ['a', 'a', 'b', 'c', 'a'],
        'to':   ['b', 'c', 'c', 'd', 'd'],
        'time': [datetime.datetime(2024, 1, d) for d in (1, 2, 3, 4, 5)],
    })


def _vb_to_px(spread, vx, vy):
    """Inverse of _to_viewbox_coords: viewBox coordinates -> pixel coordinates."""
    w, h  = spread.wxh
    vw    = spread.vx1 - spread.vx0
    vh    = spread.vy1 - spread.vy0
    scale = min(w / vw, h / vh)
    ox    = (w - vw * scale) / 2
    oy    = (h - vh * scale) / 2
    return ox + (vx - spread.vx0) * scale, oy + (vy - spread.vy0) * scale


def _node_xy(spread, node):
    for n2xyrs in spread.bin_to_node_to_xyrepstat.values():
        if node in n2xyrs:
            return n2xyrs[node][0], n2xyrs[node][1]
    raise KeyError(node)


class _SpreadLinePiBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p2s = Polars2SVG()

    def _make(self, **spread_kwargs):
        spread = self.p2s.spreadlinesp(_make_df(), [('fm', 'to')],
                                       ego='a', time='time', **spread_kwargs)
        view = spreadlinepi(spread)
        for _name_ in ('drag_op_finished', 'key_op_finished'):
            for _w_ in list(view.param.watchers.get(_name_, {}).get('value', [])):
                view.param.unwatch(_w_)
        return spread, view

    def _drag(self, view, x0, y0, x1, y1, shift=False, ctrl=False):
        view.drag_x0, view.drag_y0 = int(x0), int(y0)
        view.drag_x1, view.drag_y1 = int(x1), int(y1)
        view.shiftkey, view.ctrlkey = shift, ctrl
        asyncio.run(view.applyDragOp(None))

    def _click_node(self, view, node, shift=False, ctrl=False):
        px, py = _vb_to_px(view._spread_, *_node_xy(view._spread_, node))
        self._drag(view, px, py, px, py, shift=shift, ctrl=ctrl)

    def _key(self, view, key):
        async def _run_():
            view.key_op_finished = key
            await view.applyKeyOp(None)
        asyncio.run(_run_())


# ─────────────────────────────────────────────────────────────────────────────
# Pure helper functions
# ─────────────────────────────────────────────────────────────────────────────

class TestApplySetOp(unittest.TestCase):

    def test_replace(self):
        self.assertEqual(_apply_set_op({'a'}, {'b'}, False, False), {'b'})

    def test_ctrl_adds(self):
        self.assertEqual(_apply_set_op({'a'}, {'b'}, False, True), {'a', 'b'})

    def test_shift_subtracts(self):
        self.assertEqual(_apply_set_op({'a', 'b'}, {'b'}, True, False), {'a'})

    def test_ctrl_shift_intersects(self):
        self.assertEqual(_apply_set_op({'a', 'b'}, {'b', 'c'}, True, True), {'b'})


class TestViewboxCoords(_SpreadLinePiBase):

    def test_degenerate_viewbox_returns_input(self):
        _fake_ = types.SimpleNamespace(wxh=(100, 100), vx0=5.0, vx1=5.0, vy0=0.0, vy1=10.0)
        self.assertEqual(_to_viewbox_coords(_fake_, 7.0, 9.0), (7.0, 9.0))

    def test_roundtrip_through_letterboxing(self):
        spread, _ = self._make()
        vx, vy = _node_xy(spread, 'a')
        px, py = _vb_to_px(spread, vx, vy)
        rx, ry = _to_viewbox_coords(spread, px, py)
        self.assertAlmostEqual(rx, vx, places=6)
        self.assertAlmostEqual(ry, vy, places=6)

    def test_viewbox_center_maps_to_pixel_center(self):
        spread, _ = self._make()
        w, h = spread.wxh
        cx, cy = _to_viewbox_coords(spread, w / 2, h / 2)
        self.assertAlmostEqual(cx, (spread.vx0 + spread.vx1) / 2, places=6)
        self.assertAlmostEqual(cy, (spread.vy0 + spread.vy1) / 2, places=6)


class TestHitTesting(_SpreadLinePiBase):

    def test_nodes_at_xy_hits_node(self):
        spread, _ = self._make()
        vx, vy = _node_xy(spread, 'b')
        self.assertIn('b', _nodes_at_xy(spread, vx, vy))

    def test_nodes_at_xy_misses_far_away(self):
        spread, _ = self._make()
        self.assertEqual(_nodes_at_xy(spread, spread.vx1 + 500, spread.vy1 + 500), set())

    def test_nodes_in_rect_full_viewbox_selects_all(self):
        spread, _ = self._make()
        _all_ = _nodes_in_rect(spread, spread.vx0, spread.vy0, spread.vx1, spread.vy1)
        self.assertEqual(_all_, {'a', 'b', 'c', 'd'})

    def test_nodes_in_rect_handles_inverted_corners(self):
        spread, _ = self._make()
        _all_ = _nodes_in_rect(spread, spread.vx1, spread.vy1, spread.vx0, spread.vy0)
        self.assertEqual(_all_, {'a', 'b', 'c', 'd'})

    def test_nodes_in_rect_boundary_overlap_counts(self):
        # rect ending exactly at a node's left edge minus epsilon excludes;
        # touching the circle includes
        spread, _ = self._make()
        vx, vy = _node_xy(spread, 'd')
        r = spread.r_pref
        self.assertIn('d', _nodes_in_rect(spread, vx - r, vy - r, vx - r + 0.1, vy + r))
        self.assertNotIn('d', _nodes_in_rect(spread, vx - 3 * r, vy - 3 * r,
                                             vx - 2 * r, vy - 2 * r))


class TestExpandEgoAndFilter(_SpreadLinePiBase):

    def test_expand_ego_noop_when_not_set(self):
        spread, _ = self._make()
        self.assertFalse(spread.ego_is_set)
        self.assertEqual(_expand_ego(spread, {'__EGO__', 'z'}), {'__EGO__', 'z'})

    def test_expand_ego_replaces_token_for_set_ego(self):
        spread = self.p2s.spreadlinesp(_make_df(), [('fm', 'to')],
                                       ego={'a', 'b'}, time='time')
        self.assertTrue(spread.ego_is_set)
        self.assertEqual(_expand_ego(spread, {'__EGO__', 'z'}), {'a', 'b', 'z'})

    def test_filter_out_nodes_removes_touching_rows(self):
        spread, _ = self._make()
        out = _filter_out_nodes(spread, _make_df(), {'d'})
        self.assertEqual(len(out), 3)
        for _col_ in ('fm', 'to'):
            self.assertNotIn('d', out[_col_].to_list())

    def test_filter_out_nodes_empty_set_keeps_all(self):
        spread, _ = self._make()
        self.assertEqual(len(_filter_out_nodes(spread, _make_df(), set())), 5)


# ─────────────────────────────────────────────────────────────────────────────
# Drag / click selection
# ─────────────────────────────────────────────────────────────────────────────

class TestDragSelection(_SpreadLinePiBase):

    def test_click_node_selects_it(self):
        _, view = self._make()
        self._click_node(view, 'b')
        self.assertEqual(view.selected_entities, {'b'})

    def test_click_empty_space_clears_selection(self):
        _, view = self._make()
        self._click_node(view, 'b')
        self._drag(view, 1, 1, 2, 2)
        self.assertEqual(view.selected_entities, set())

    def test_ctrl_click_adds_to_selection(self):
        _, view = self._make()
        self._click_node(view, 'b')
        self._click_node(view, 'c', ctrl=True)
        self.assertEqual(view.selected_entities, {'b', 'c'})

    def test_shift_click_subtracts_from_selection(self):
        _, view = self._make()
        spread = view._spread_
        w, h = spread.wxh
        self._drag(view, 0, 0, w, h)   # select all
        self._click_node(view, 'b', shift=True)
        self.assertEqual(view.selected_entities, {'a', 'c', 'd'})

    def test_full_canvas_drag_selects_all_nodes(self):
        _, view = self._make()
        w, h = view._spread_.wxh
        self._drag(view, 0, 0, w, h)
        self.assertEqual(view.selected_entities, {'a', 'b', 'c', 'd'})

    def test_drag_updates_rendered_view(self):
        _, view = self._make()
        _before_ = view.mod_inner
        w, h = view._spread_.wxh
        self._drag(view, 0, 0, w, h)
        self.assertNotEqual(view.mod_inner, _before_)

    def test_selection_broadcast_to_peer_views(self):
        _, view = self._make()

        class _Peer_:
            def __init__(self):
                self.received = []
            async def receiveSelection(self, entities):
                self.received.append(set(entities))

        peer = _Peer_()
        view.mvc.view_refs[id(peer)] = peer
        self._click_node(view, 'c')
        self.assertEqual(peer.received, [{'c'}])


# ─────────────────────────────────────────────────────────────────────────────
# Key operations
# ─────────────────────────────────────────────────────────────────────────────

class TestKeyOperations(_SpreadLinePiBase):

    def test_x_removes_selected_nodes_and_pushes_stack(self):
        _, view = self._make()
        view.selected_entities = {'d'}
        self._key(view, 'x')
        s = view.mvc.stacks['default']
        self.assertEqual(len(s['dfs']), 2)
        self.assertEqual(s['index'], 1)
        self.assertEqual(len(s['dfs'][1]), 3)
        self.assertEqual(view.selected_entities, set())

    def test_x_with_no_selection_is_noop(self):
        _, view = self._make()
        self._key(view, 'x')
        self.assertEqual(len(view.mvc.stacks['default']['dfs']), 1)

    def test_x_that_would_empty_df_is_noop(self):
        _, view = self._make()
        view.selected_entities = {'a', 'b', 'c', 'd'}
        self._key(view, 'x')
        self.assertEqual(len(view.mvc.stacks['default']['dfs']), 1)

    def test_capital_x_clears_selection_and_pops_stack(self):
        _, view = self._make()
        view.selected_entities = {'d'}
        self._key(view, 'x')
        self.assertEqual(view.mvc.stacks['default']['index'], 1)
        self._key(view, 'X')
        self.assertEqual(view.mvc.stacks['default']['index'], 0)
        self.assertEqual(view.selected_entities, set())

    def test_c_single_selection_becomes_new_ego(self):
        _, view = self._make()
        view.selected_entities = {'b'}
        self._key(view, 'c')
        self.assertEqual(view._spread_.ego, 'b')
        self.assertFalse(view._spread_.ego_is_set)

    def test_c_multi_selection_becomes_ego_set(self):
        _, view = self._make()
        view.selected_entities = {'b', 'c'}
        self._key(view, 'c')
        self.assertEqual(view._spread_.ego, {'b', 'c'})
        self.assertTrue(view._spread_.ego_is_set)

    def test_c_with_no_selection_is_noop(self):
        _, view = self._make()
        _ego_before_ = view._spread_.ego
        self._key(view, 'c')
        self.assertEqual(view._spread_.ego, _ego_before_)

    def test_unknown_key_is_noop(self):
        _, view = self._make()
        _before_ = view.mod_inner
        self._key(view, 'q')
        self.assertEqual(view.mod_inner, _before_)


# ─────────────────────────────────────────────────────────────────────────────
# MVC callbacks: receiveSelection / display
# ─────────────────────────────────────────────────────────────────────────────

class TestMvcCallbacks(_SpreadLinePiBase):

    def test_receiveSelection_sets_entities_and_rerenders(self):
        _, view = self._make()
        _before_ = view.mod_inner
        asyncio.run(view.receiveSelection(['b', 'c']))
        self.assertEqual(view.selected_entities, {'b', 'c'})
        self.assertNotEqual(view.mod_inner, _before_)

    def test_display_caches_rendered_frames_and_prunes(self):
        _, view = self._make()
        base = view._spread_.df_orig
        df2  = base.head(3)
        asyncio.run(view.display(df2, [base, df2], 1))
        self.assertIn(id(df2), view._cache_)
        asyncio.run(view.display(base, [base], 0))
        self.assertNotIn(id(df2), view._cache_)
        self.assertIn(id(base), view._cache_)

    def test_display_reuses_cached_spread(self):
        _, view = self._make()
        base = view._spread_.df_orig
        df2  = base.head(3)
        asyncio.run(view.display(df2, [base, df2], 1))
        _spread2_ = view._cache_[id(df2)]
        asyncio.run(view.display(df2, [base, df2], 1))
        self.assertIs(view._cache_[id(df2)], _spread2_)


if __name__ == '__main__':
    unittest.main()
