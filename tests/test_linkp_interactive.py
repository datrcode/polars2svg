import asyncio
import unittest
import polars as pl
from polars2svg import Polars2SVG
from polars2svg.interactive_controller import linkpi, InteractionController


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


class TestLinkPInteractive(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def setUp(self):
        self.lp = self.p2s.linkp(_make_df(), relationships=_rels(), pos=_make_pos())

    # -------------------------------------------------------------------------
    # invalidateRender()
    # -------------------------------------------------------------------------

    def test_invalidate_render_sets_flag(self):
        self.assertFalse(self.lp._render_invalid_)
        self.lp.invalidateRender()
        self.assertTrue(self.lp._render_invalid_)

    # -------------------------------------------------------------------------
    # renderSVG()
    # -------------------------------------------------------------------------

    def test_renderSVG_returns_svg_string(self):
        svg = self.lp.renderSVG()
        self.assertIsInstance(svg, str)
        self.assertIn('<svg', svg)

    def test_renderSVG_clears_invalid_flag(self):
        self.lp.invalidateRender()
        self.assertTrue(self.lp._render_invalid_)
        self.lp.renderSVG()
        self.assertFalse(self.lp._render_invalid_)

    def test_renderSVG_does_not_rerender_when_valid(self):
        svg1 = self.lp.renderSVG()
        svg2 = self.lp.renderSVG()
        self.assertIs(svg1, svg2)

    # -------------------------------------------------------------------------
    # setViewWindow() / getViewWindow()
    # -------------------------------------------------------------------------

    def test_setViewWindow_stores_tuple(self):
        vw = (-1.0, -2.0, 3.0, 4.0)
        self.lp.setViewWindow(vw)
        self.assertEqual(self.lp.getViewWindow(), vw)

    def test_setViewWindow_invalidates_render(self):
        self.lp.setViewWindow((-1.0, -1.0, 2.0, 2.0))
        self.assertTrue(self.lp._render_invalid_)

    def test_getViewWindow_returns_4tuple(self):
        vw = self.lp.getViewWindow()
        self.assertIsInstance(vw, tuple)
        self.assertEqual(len(vw), 4)

    # -------------------------------------------------------------------------
    # applyScrollEvent()
    # -------------------------------------------------------------------------

    def test_applyScrollEvent_returns_true(self):
        result = self.lp.applyScrollEvent(100)
        self.assertTrue(result)

    def test_applyScrollEvent_zoom_in_shrinks_window(self):
        wx0, wy0, wx1, wy1 = self.lp.getViewWindow()
        orig_width = wx1 - wx0
        self.lp.applyScrollEvent(200)  # factor 1.2 → zoom in
        nwx0, nwy0, nwx1, nwy1 = self.lp.getViewWindow()
        self.assertGreater(nwx1 - nwx0, orig_width)  # positive scroll = zoom out (factor > 1)

    def test_applyScrollEvent_zoom_out_shrinks_factor(self):
        wx0, wy0, wx1, wy1 = self.lp.getViewWindow()
        orig_width = wx1 - wx0
        self.lp.applyScrollEvent(-200)  # factor 0.8 → zoom in
        nwx0, nwy0, nwx1, nwy1 = self.lp.getViewWindow()
        self.assertLess(nwx1 - nwx0, orig_width)

    def test_applyScrollEvent_with_coordinate_centers_on_point(self):
        # The center of the new view window should be the inverse-transform of the coordinate
        sx, sy = 128.0, 128.0
        cx_world = self.lp.xT_inv(sx)
        cy_world = self.lp.yT_inv(sy)
        self.lp.applyScrollEvent(500, coordinate=[sx, sy])
        nwx0, nwy0, nwx1, nwy1 = self.lp.getViewWindow()
        mid_x = (nwx0 + nwx1) / 2
        mid_y = (nwy0 + nwy1) / 2
        self.assertAlmostEqual(mid_x, cx_world, places=6)
        self.assertAlmostEqual(mid_y, cy_world, places=6)

    def test_applyScrollEvent_without_coordinate_centers_on_midpoint(self):
        wx0, wy0, wx1, wy1 = self.lp.getViewWindow()
        mid_x = (wx0 + wx1) / 2
        mid_y = (wy0 + wy1) / 2
        self.lp.applyScrollEvent(500)
        nwx0, nwy0, nwx1, nwy1 = self.lp.getViewWindow()
        self.assertAlmostEqual((nwx0 + nwx1) / 2, mid_x, places=6)
        self.assertAlmostEqual((nwy0 + nwy1) / 2, mid_y, places=6)

    # -------------------------------------------------------------------------
    # applyMiddleClick()
    # -------------------------------------------------------------------------

    def test_applyMiddleClick_returns_false_at_original(self):
        result = self.lp.applyMiddleClick([128, 128])
        self.assertFalse(result)

    def test_applyMiddleClick_resets_view_returns_true(self):
        orig_vw = self.lp.getViewWindow()
        self.lp.applyScrollEvent(500)
        self.assertNotEqual(self.lp.getViewWindow(), orig_vw)
        result = self.lp.applyMiddleClick([128, 128])
        self.assertTrue(result)
        self.assertEqual(self.lp.getViewWindow(), orig_vw)

    # -------------------------------------------------------------------------
    # applyMiddleDrag()
    # -------------------------------------------------------------------------

    def test_applyMiddleDrag_returns_true(self):
        result = self.lp.applyMiddleDrag([128, 128], [10, 0])
        self.assertTrue(result)

    def test_applyMiddleDrag_shifts_view(self):
        wx0, wy0, wx1, wy1 = self.lp.getViewWindow()
        sx, sy = 128.0, 128.0
        dx, dy = 20.0, 0.0
        dwx = self.lp.xT_inv(sx) - self.lp.xT_inv(sx + dx)
        dwy = self.lp.yT_inv(sy) - self.lp.yT_inv(sy + dy)
        self.lp.applyMiddleDrag([sx, sy], [dx, dy])
        nwx0, nwy0, nwx1, nwy1 = self.lp.getViewWindow()
        self.assertAlmostEqual(nwx0, wx0 + dwx, places=6)
        self.assertAlmostEqual(nwy0, wy0 + dwy, places=6)
        self.assertAlmostEqual(nwx1, wx1 + dwx, places=6)
        self.assertAlmostEqual(nwy1, wy1 + dwy, places=6)

    # -------------------------------------------------------------------------
    # applyViewConfiguration()
    # -------------------------------------------------------------------------

    def test_applyViewConfiguration_syncs_window(self):
        lp2 = self.p2s.linkp(_make_df(), relationships=_rels(), pos=_make_pos())
        self.lp.applyScrollEvent(500)
        new_vw = self.lp.getViewWindow()
        result = lp2.applyViewConfiguration(self.lp)
        self.assertTrue(result)
        self.assertEqual(lp2.getViewWindow(), new_vw)

    def test_applyViewConfiguration_returns_false_when_same(self):
        lp2 = self.p2s.linkp(_make_df(), relationships=_rels(), pos=_make_pos())
        lp2.setViewWindow(self.lp.getViewWindow())
        lp2.renderSVG()  # clear invalid flag
        result = lp2.applyViewConfiguration(self.lp)
        self.assertFalse(result)

    # -------------------------------------------------------------------------
    # nodeColor() / nodesWithColor()
    # -------------------------------------------------------------------------

    def test_nodeColor_returns_hex_for_known_nodes(self):
        for node in ('a', 'b', 'c'):
            color = self.lp.nodeColor(node)
            self.assertIsNotNone(color, f'nodeColor({node!r}) returned None')
            self.assertRegex(color, r'^#[0-9a-fA-F]{6}$',
                             f'nodeColor({node!r}) = {color!r} is not a hex color')

    def test_nodeColor_returns_none_for_unknown(self):
        self.assertIsNone(self.lp.nodeColor('zzz_not_a_node'))

    def test_nodesWithColor_roundtrip(self):
        for node in ('a', 'b', 'c'):
            color = self.lp.nodeColor(node)
            self.assertIn(node, self.lp.nodesWithColor(color))

    # -------------------------------------------------------------------------
    # nodeShape() / nodesWithShape()
    # -------------------------------------------------------------------------

    def test_nodeShape_always_circle(self):
        for node in ('a', 'b', 'c', 'anything'):
            self.assertEqual(self.lp.nodeShape(node), 'circle')

    def test_nodesWithShape_circle_returns_all_nodes(self):
        expected = set(self.lp.color_nodes_final.keys())
        self.assertEqual(self.lp.nodesWithShape('circle'), expected)

    def test_nodesWithShape_other_returns_empty(self):
        self.assertEqual(self.lp.nodesWithShape('square'), set())
        self.assertEqual(self.lp.nodesWithShape('triangle'), set())

    # -------------------------------------------------------------------------
    # overlappingEntities() / entitiesAtPoint()
    # -------------------------------------------------------------------------

    def test_overlappingEntities_large_polygon_finds_all_nodes(self):
        from shapely.geometry import Polygon
        w, h = self.lp.wxh
        poly = Polygon([(0, 0), (0, h), (w, h), (w, 0)])
        found = set(self.lp.overlappingEntities(poly))
        self.assertEqual(found, {'a', 'b', 'c'})

    def test_overlappingEntities_tiny_polygon_finds_zero(self):
        from shapely.geometry import Polygon
        poly = Polygon([(-1000, -1000), (-1000, -990), (-990, -990), (-990, -1000)])
        self.assertEqual(self.lp.overlappingEntities(poly), [])

    def test_entitiesAtPoint_finds_node_at_screen_coord(self):
        sx = self.lp.xT(0.0)   # screen x for world x=0 (node 'a')
        sy = self.lp.yT(0.0)   # screen y for world y=0 (node 'a')
        found = self.lp.entitiesAtPoint([sx, sy])
        self.assertIn('a', found)

    def test_entitiesAtPoint_miss_returns_empty(self):
        found = self.lp.entitiesAtPoint([-5000, -5000])
        self.assertEqual(found, [])

    # -------------------------------------------------------------------------
    # __createPathDescriptionForAllEntities__()
    # -------------------------------------------------------------------------

    def test_path_all_entities_is_nonempty_string(self):
        path = self.lp.__createPathDescriptionForAllEntities__()
        self.assertIsInstance(path, str)
        self.assertGreater(len(path), 0)

    def test_path_all_entities_contains_M_commands(self):
        path = self.lp.__createPathDescriptionForAllEntities__()
        m_count = path.count('M ')
        self.assertGreaterEqual(m_count, 3)  # one per distinct node position

    # -------------------------------------------------------------------------
    # __createPathDescriptionOfSelectedEntities__()
    # -------------------------------------------------------------------------

    _FALLBACK_ = 'M -100 -100'

    def test_path_selected_none_returns_fallback(self):
        path = self.lp.__createPathDescriptionOfSelectedEntities__(None)
        self.assertIn(self._FALLBACK_, path)

    def test_path_selected_empty_list_returns_fallback(self):
        path = self.lp.__createPathDescriptionOfSelectedEntities__([])
        self.assertIn(self._FALLBACK_, path)

    def test_path_selected_empty_set_returns_fallback(self):
        path = self.lp.__createPathDescriptionOfSelectedEntities__(set())
        self.assertIn(self._FALLBACK_, path)

    def test_path_selected_unknown_nodes_returns_fallback(self):
        path = self.lp.__createPathDescriptionOfSelectedEntities__({'zzz_not_a_node'})
        self.assertIn(self._FALLBACK_, path)

    def test_path_selected_valid_nodes_returns_M_path(self):
        path = self.lp.__createPathDescriptionOfSelectedEntities__({'a'})
        self.assertNotIn(self._FALLBACK_, path)
        self.assertIn('M ', path)

    # -------------------------------------------------------------------------
    # __moveSelectedEntities__()
    # -------------------------------------------------------------------------

    def test_move_none_selection_no_change(self):
        orig_pos = dict(self.lp.pos)
        self.lp.__moveSelectedEntities__((10, 0), None)
        self.assertEqual(self.lp.pos, orig_pos)
        self.assertFalse(self.lp._render_invalid_)

    def test_move_empty_selection_no_change(self):
        orig_pos = dict(self.lp.pos)
        self.lp.__moveSelectedEntities__((10, 0), set())
        self.assertEqual(self.lp.pos, orig_pos)
        self.assertFalse(self.lp._render_invalid_)

    def test_move_updates_pos(self):
        orig_x = self.lp.pos['a'][0]
        self.lp.__moveSelectedEntities__((20, 0), {'a'})
        new_x = self.lp.pos['a'][0]
        self.assertNotAlmostEqual(new_x, orig_x, places=6)

    def test_move_invalidates_render(self):
        self.lp.__moveSelectedEntities__((10, 0), {'a'})
        self.assertTrue(self.lp._render_invalid_)

    def test_move_none_selection_returns_empty_dict(self):
        result = self.lp.__moveSelectedEntities__((10, 0), None)
        self.assertEqual(result, {})

    def test_move_empty_selection_returns_empty_dict(self):
        result = self.lp.__moveSelectedEntities__((10, 0), set())
        self.assertEqual(result, {})

    def test_move_returns_updated_positions(self):
        result = self.lp.__moveSelectedEntities__((20, 0), {'a'})
        self.assertIn('a', result)
        wx, wy = result['a']
        self.assertAlmostEqual(wx, self.lp.pos['a'][0], places=6)
        self.assertAlmostEqual(wy, self.lp.pos['a'][1], places=6)

    def test_move_return_dict_does_not_include_unmoved_nodes(self):
        result = self.lp.__moveSelectedEntities__((20, 0), {'a'})
        self.assertNotIn('b', result)
        self.assertNotIn('c', result)

    def test_move_position_propagates_to_other_layout_level(self):
        # Simulate what applyMoveOp does: move a node in one level and propagate
        # the updated pos to a second LinkP instance (representing another stack level).
        lp2 = self.p2s.linkp(_make_df(), relationships=_rels(), pos=dict(self.lp.pos))
        orig_pos_a = lp2.pos['a']

        _updated_pos_ = self.lp.__moveSelectedEntities__((30, 0), {'a'})

        # Propagation logic (mirrors applyMoveOp in interactive_controller.py)
        _changed_ = False
        for key, new_pos in _updated_pos_.items():
            if key in lp2.pos:
                lp2.pos[key] = new_pos
                _changed_ = True
        if _changed_:
            lp2.invalidateRender()

        self.assertNotEqual(lp2.pos['a'], orig_pos_a,
                            'other level pos should be updated after propagation')
        self.assertTrue(lp2._render_invalid_,
                        'other level should be invalidated after propagation')

    def test_move_propagation_leaves_unchanged_nodes_intact(self):
        lp2 = self.p2s.linkp(_make_df(), relationships=_rels(), pos=dict(self.lp.pos))
        orig_b = lp2.pos['b']
        orig_c = lp2.pos['c']

        _updated_pos_ = self.lp.__moveSelectedEntities__((30, 0), {'a'})

        for key, new_pos in _updated_pos_.items():
            if key in lp2.pos:
                lp2.pos[key] = new_pos

        self.assertEqual(lp2.pos['b'], orig_b)
        self.assertEqual(lp2.pos['c'], orig_c)

    def test_move_unselected_path_propagates_to_other_layout_level(self):
        # Simulates the unselectedMoveOp path: node is selected then moved in one call.
        # Before the fix, this path called __moveSelectedEntities__ without capturing
        # the return value, so positions were never propagated to other stack levels.
        lp2 = self.p2s.linkp(_make_df(), relationships=_rels(), pos=dict(self.lp.pos))
        orig_pos_a = lp2.pos['a']

        # unselectedMoveOp selects the node just before moving — simulate that
        _updated_pos_ = self.lp.__moveSelectedEntities__((30, 0), {'a'})

        # Propagation logic (mirrors unselectedMoveOp in interactive_controller.py)
        for i, layout in enumerate([lp2]):
            for _key_, _new_pos_ in _updated_pos_.items():
                if _key_ in layout.pos:
                    layout.pos[_key_] = _new_pos_
            layout.invalidateRender()

        self.assertNotEqual(lp2.pos['a'], orig_pos_a,
                            'unselectedMoveOp path must propagate pos to other levels')
        self.assertTrue(lp2._render_invalid_)

    # -------------------------------------------------------------------------
    # labelOnly()
    # -------------------------------------------------------------------------

    def test_labelOnly_sets_label_only(self):
        self.lp.labelOnly({'a'})
        self.assertEqual(self.lp.label_only, {'a'})

    def test_labelOnly_none_clears_to_empty_set(self):
        self.lp.labelOnly({'a', 'b'})
        self.lp.labelOnly(None)
        self.assertEqual(self.lp.label_only, set())

    def test_labelOnly_invalidates_render(self):
        self.lp.labelOnly({'a'})
        self.assertTrue(self.lp._render_invalid_)

    # -------------------------------------------------------------------------
    # drawLabels()
    # -------------------------------------------------------------------------

    def test_drawLabels_sets_true(self):
        self.lp.drawLabels(True)
        self.assertTrue(self.lp.draw_labels)

    def test_drawLabels_sets_false(self):
        self.lp.drawLabels(True)
        self.lp.renderSVG()
        self.lp.drawLabels(False)
        self.assertFalse(self.lp.draw_labels)

    def test_drawLabels_invalidates_render(self):
        self.lp.drawLabels(True)
        self.assertTrue(self.lp._render_invalid_)

    # -------------------------------------------------------------------------
    # Group A — overlappingEntities coordinate-space consistency
    # -------------------------------------------------------------------------

    def test_overlappingEntities_uses_actual_df_node_coords(self):
        # Build a rect from the actual stored __sx__/__sy__ values and confirm
        # all three nodes are found.  If this fails, overlappingEntities is broken
        # independently of any import issue.
        from shapely.geometry import Polygon
        xs = self.lp.df_node['__sx__'].to_list()
        ys = self.lp.df_node['__sy__'].to_list()
        x0, y0 = min(xs) - 1, min(ys) - 1
        x1, y1 = max(xs) + 1, max(ys) + 1
        poly   = Polygon([(x0, y0), (x0, y1), (x1, y1), (x1, y0)])
        result = set(self.lp.overlappingEntities(poly))
        self.assertSetEqual(result, {'a', 'b', 'c'})

    def test_overlappingEntities_empty_when_rect_misses(self):
        from shapely.geometry import Polygon
        poly = Polygon([(-1000, -1000), (-1000, -990), (-990, -990), (-990, -1000)])
        self.assertEqual(self.lp.overlappingEntities(poly), [])

    def test_xT_yT_match_df_node_screen_coords(self):
        # The transform lambdas and the stored df_node screen coords must agree.
        # A mismatch would mean applyDragOp rectangles and df_node live in
        # different coordinate spaces.
        row_a = self.lp.df_node.filter(pl.col('__first__') == 'a')
        self.assertEqual(len(row_a), 1, 'expected exactly one df_node row for node a')
        expected_sx = round(self.lp.xT(self.lp.pos['a'][0]))
        expected_sy = round(self.lp.yT(self.lp.pos['a'][1]))
        self.assertAlmostEqual(row_a['__sx__'][0], expected_sx, delta=1)
        self.assertAlmostEqual(row_a['__sy__'][0], expected_sy, delta=1)

    # -------------------------------------------------------------------------
    # Group B — applyDragOp Python logic (isolated from Panel)
    # -------------------------------------------------------------------------

    def test_applyDragOp_body_selects_nodes(self):
        # Replicates the core of applyDragOp using a full-canvas rectangle.
        # Passing here + B2 failing = the *only* problem is the missing Polygon import.
        from shapely.geometry import Polygon
        lp      = self.lp
        w, h    = lp.wxh
        x0, y0, x1, y1 = 0, 0, w, h
        rect    = Polygon([(x0, y0), (x0, y1), (x1, y1), (x1, y0)])
        found   = set(lp.overlappingEntities(rect))
        self.assertSetEqual(found, {'a', 'b', 'c'})

    def test_required_names_imported_in_interactive_controller(self):
        # Checks that every name used in async callbacks is actually imported.
        # A missing import causes a silent NameError that swallows the callback.
        import ast
        import pathlib
        import polars2svg.interactive_controller as _ic_mod
        ic_path = pathlib.Path(_ic_mod.__file__)
        tree     = ast.parse(ic_path.read_text())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    imported.add(alias.asname or alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imported.add(alias.asname or alias.name)
        required = {
            'Polygon': 'applyDragOp uses Polygon()',
            'copy':    '__cacheNodePositions__ uses copy.deepcopy()',
            'sqrt':    'applyLayoutInteraction uses sqrt()',
            'time':    'setAnimation uses time.sleep()',
        }
        for name, reason in required.items():
            self.assertIn(name, imported,
                          f'{name} must be imported in interactive_controller.py ({reason})')

    # -------------------------------------------------------------------------
    # Group C — __moveSelectedEntities__ + re-render
    # -------------------------------------------------------------------------

    def test_move_selected_updates_df_node_after_rerender(self):
        # After __moveSelectedEntities__ + renderSVG(), the df_node __sx__ for
        # the moved node must reflect the new position.
        lp       = self.lp
        sx_before = lp.df_node.explode('__nm__').filter(
            pl.col('__nm__') == 'a')['__sx__'][0]
        lp.__moveSelectedEntities__((20, 0), {'a'})
        lp.renderSVG()
        sx_after  = lp.df_node.explode('__nm__').filter(
            pl.col('__nm__') == 'a')['__sx__'][0]
        self.assertNotAlmostEqual(
            sx_after, sx_before, delta=1,
            msg='After move + re-render, node a screen x must change',
        )

    def test_move_selected_svg_reflects_new_cx(self):
        # The rendered SVG must contain a circle whose cx matches the post-move
        # screen position of node 'a'.
        lp = self.lp
        lp.__moveSelectedEntities__((30, 0), {'a'})
        svg    = lp.renderSVG()
        new_sx = lp.df_node.explode('__nm__').filter(
            pl.col('__nm__') == 'a')['__sx__'][0]
        self.assertIn(f'cx="{new_sx}"', svg)


class TestLinkPInteractiveIntegerNodes(unittest.TestCase):
    """Regression tests: node IDs that are integers (Int64) must survive drag/move.

    When __nm__ is cast to String in df_node (line 778 of linkp.py), overlappingEntities
    returned string node IDs.  __moveSelectedEntities__ then wrote those strings as new
    keys in pos, creating a mixed Int64+String dict.  The next __calculateGeometry__ call
    passed that dict to replace_strict, which tried to build a Series from the mixed-type
    keys and raised:
        TypeError: unexpected value while building Series of type Int64;
                   found value of type String: "103244"
    """

    def setUp(self):
        self.p2s = Polars2SVG()
        self.df = pl.DataFrame({
            'fm': [103244, 103245, 103246, 103244],
            'to': [103245, 103246, 103244, 103246],
        })
        self.pos = {103244: (0.0, 0.0), 103245: (1.0, 0.0), 103246: (0.5, 0.866)}
        self.lp  = self.p2s.linkp(self.df, relationships=[('fm', 'to')], pos=self.pos)

    def test_pos_keys_remain_integers_after_move(self):
        self.lp.__moveSelectedEntities__((20, 0), {103244})
        for k in self.lp.pos.keys():
            self.assertIsInstance(k, int, f'pos key {k!r} should be int, got {type(k).__name__}')

    def test_rerender_after_move_does_not_raise(self):
        # This is the exact bug: re-rendering after a move crashed with TypeError.
        self.lp.__moveSelectedEntities__((20, 0), {103244})
        try:
            svg = self.lp.renderSVG()
        except TypeError as e:
            self.fail(f'renderSVG() raised TypeError after move: {e}')
        self.assertIn('<svg', svg)

    def test_node_actually_moves(self):
        orig_x = self.lp.pos[103244][0]
        self.lp.__moveSelectedEntities__((30, 0), {103244})
        new_x  = self.lp.pos[103244][0]
        self.assertNotAlmostEqual(new_x, orig_x, places=6,
                                  msg='Node 103244 world-x must change after move')

    def test_overlappingEntities_returns_integer_ids(self):
        from shapely.geometry import Polygon
        xs = self.lp.df_node['__sx__'].to_list()
        ys = self.lp.df_node['__sy__'].to_list()
        poly = Polygon([(min(xs) - 1, min(ys) - 1), (min(xs) - 1, max(ys) + 1),
                        (max(xs) + 1, max(ys) + 1), (max(xs) + 1, min(ys) - 1)])
        found = self.lp.overlappingEntities(poly)
        for entity in found:
            self.assertIsInstance(entity, int,
                                  f'overlappingEntities returned {entity!r} (type {type(entity).__name__}), expected int')

    def test_selected_path_with_integer_ids_does_not_raise(self):
        try:
            path = self.lp.__createPathDescriptionOfSelectedEntities__({103244})
        except TypeError as e:
            self.fail(f'__createPathDescriptionOfSelectedEntities__ raised TypeError: {e}')
        self.assertIn('M ', path)

    def test_nodeColor_resolves_integer_node_ids(self):
        # color_nodes_final is keyed by the stringified __nm__ names while the hit-test
        # hands back the original int ids -- nodeColor() has to bridge the two.
        self.lp.renderSVG()
        for _node_ in self.pos.keys():
            self.assertIsNotNone(self.lp.nodeColor(_node_),
                                 f'nodeColor({_node_!r}) returned None for an int node id')

    def test_color_query_roundtrip_from_hit_test_with_integer_ids(self):
        # The 'z'-key chain: entitiesAtPoint -> nodeColor -> nodesWithColor.  A None from
        # nodeColor() silently collapsed this to an empty selection.
        _lp_ = self.p2s.linkp(self.df, relationships=[('fm', 'to')], pos=self.pos,
                              node_color={'103244': '#2166ac', '103245': '#d6604d',
                                          '103246': '#2166ac'})
        _lp_.renderSVG()
        _row_  = _lp_.df_node.explode('__nm__').filter(pl.col('__nm__') == '103244')
        _found_ = set()
        for _e_ in _lp_.entitiesAtPoint((_row_['__sx__'][0], _row_['__sy__'][0])):
            _found_ |= set(_lp_.nodesWithColor(_lp_.nodeColor(_e_)))
        self.assertEqual(_found_, {'103244', '103246'})


class TestLinkPTKeyCollapse(unittest.TestCase):
    """Regression tests: the 't'-key collapse must always land nodes at the
    target position and must always call invalidateRender().

    Bug: the 'collapse to single point' branch contained dead code:
        xy = _ln_.pos[_entity_]   # unused — raised KeyError if entity absent
        _ln_.pos[_entity_] = (target_x, target_y)
    A KeyError stopped the loop before invalidateRender() was called, leaving
    the SVG stale.  Nodes appeared to vanish after panning/zooming away.
    The bug was non-deterministic because Python set iteration is unordered.

    Fix: removed the dead pos read; pos is now written directly.
    These tests exercise the fixed logic via helpers that replicate the
    interactive_controller 't'-key handler at the LinkP level.
    """

    def setUp(self):
        self.p2s = Polars2SVG()
        # Triangle {a, b, c} plus 'd' connected to 'a' — lets "e" expansion
        # add a node that differs from the initially selected set.
        df = pl.DataFrame({
            'fm': ['a', 'b', 'c', 'a'],
            'to': ['b', 'c', 'a', 'd'],
        })
        self.pos = {'a': (0.0, 0.0), 'b': (1.0, 0.0), 'c': (0.5, 0.866), 'd': (1.5, 0.5)}
        self.lp  = self.p2s.linkp(df, relationships=[('fm', 'to')], pos=dict(self.pos))

    def _collapse_to(self, entities, sx, sy):
        """Replicate the fixed 't'-key single-point collapse at the LinkP level."""
        _target_wx_ = self.lp.xT_inv(sx)
        _target_wy_ = self.lp.yT_inv(sy)
        for _entity_ in entities:
            self.lp.pos[_entity_] = (_target_wx_, _target_wy_)
        self.lp.invalidateRender()

    # -------------------------------------------------------------------------

    def test_collapse_entity_absent_from_pos_does_not_raise(self):
        # The old code read  xy = pos[entity]  before writing — that line raised
        # KeyError when the entity was missing.  The new code writes directly.
        del self.lp.pos['d']
        try:
            self._collapse_to({'a', 'b', 'c', 'd'}, 128, 128)
        except KeyError as e:
            self.fail(f'Collapsing with one entity absent from pos raised KeyError: {e}')

    def test_collapse_all_entities_reach_target(self):
        target_sx, target_sy = 128.0, 128.0
        target_wx = self.lp.xT_inv(target_sx)
        target_wy = self.lp.yT_inv(target_sy)
        selection = {'a', 'b', 'c', 'd'}
        self._collapse_to(selection, target_sx, target_sy)
        for entity in selection:
            wx, wy = self.lp.pos[entity]
            self.assertAlmostEqual(wx, target_wx, places=6,
                                   msg=f'node {entity!r} wx should be at target after collapse')
            self.assertAlmostEqual(wy, target_wy, places=6,
                                   msg=f'node {entity!r} wy should be at target after collapse')

    def test_collapse_invalidates_render(self):
        self._collapse_to({'a', 'b', 'c'}, 128.0, 128.0)
        self.assertTrue(self.lp._render_invalid_,
                        'invalidateRender() must be called after collapse')

    def test_second_collapse_after_expansion_lands_at_new_target(self):
        # Reproduces the reported bug scenario:
        # 1. Collapse {a, b, c} to P1
        # 2. Simulate 'e' expansion — selection grows to {a, b, c, d}
        # 3. Collapse {a, b, c, d} to P2 — must not raise, must land at P2
        self._collapse_to({'a', 'b', 'c'}, 100.0, 100.0)
        self.lp.renderSVG()  # clear the flag, simulating __refreshView__ after first 't'

        target_sx2, target_sy2 = 200.0, 150.0
        target_wx2 = self.lp.xT_inv(target_sx2)
        target_wy2 = self.lp.yT_inv(target_sy2)
        selection_after_e = {'a', 'b', 'c', 'd'}

        try:
            self._collapse_to(selection_after_e, target_sx2, target_sy2)
        except KeyError as e:
            self.fail(f'Second collapse after expansion raised KeyError: {e}')

        self.assertTrue(self.lp._render_invalid_,
                        'Render must be invalidated after second collapse')
        for entity in selection_after_e:
            wx, wy = self.lp.pos[entity]
            self.assertAlmostEqual(wx, target_wx2, places=6,
                                   msg=f'node {entity!r} wx must be at P2 after second collapse')
            self.assertAlmostEqual(wy, target_wy2, places=6,
                                   msg=f'node {entity!r} wy must be at P2 after second collapse')

    def test_second_collapse_rerenders_without_error(self):
        self._collapse_to({'a', 'b', 'c'}, 100.0, 100.0)
        self.lp.renderSVG()
        self._collapse_to({'a', 'b', 'c', 'd'}, 200.0, 150.0)
        try:
            svg = self.lp.renderSVG()
        except Exception as e:
            self.fail(f'renderSVG() after second collapse raised: {e}')
        self.assertIn('<svg', svg)


# ---------------------------------------------------------------------------
# replaceBaseDataframe() — LINKPI Panel wrapper
# ---------------------------------------------------------------------------

try:
    import panel as pn  # noqa: F401
    from panel.reactive import ReactiveHTML  # noqa: F401
    _PANEL_AVAILABLE_ = True
except ImportError:
    _PANEL_AVAILABLE_ = False


@unittest.skipUnless(_PANEL_AVAILABLE_, 'panel not installed')
class TestReplaceBaseDataframe(unittest.TestCase):
    """Tests for LINKPI.replaceBaseDataframe() — the method that swaps the base
    dataframe, resets the internal stack, and preserves node positions."""

    def setUp(self):
        self.p2s = Polars2SVG()
        self.df  = _make_df()
        self.pos = _make_pos()
        self.lp  = self.p2s.linkp(self.df, relationships=_rels(), pos=self.pos)
        self.ctrl = self.p2s.linkpi(self.lp)

    # ── stack reset ──────────────────────────────────────────────────────────

    def test_dfs_reset_to_single_entry(self):
        new_df = _make_df()
        asyncio.run(self.ctrl.replaceBaseDataframe(new_df))
        self.assertEqual(len(self.ctrl.dfs), 1)
        self.assertIs(self.ctrl.dfs[0], new_df)

    def test_dfs_layout_reset_to_single_entry(self):
        new_df = _make_df()
        asyncio.run(self.ctrl.replaceBaseDataframe(new_df))
        self.assertEqual(len(self.ctrl.dfs_layout), 1)

    def test_graphs_reset_to_single_entry(self):
        new_df = _make_df()
        asyncio.run(self.ctrl.replaceBaseDataframe(new_df))
        self.assertEqual(len(self.ctrl.graphs), 1)

    def test_df_level_reset_to_zero(self):
        new_df = _make_df()
        asyncio.run(self.ctrl.replaceBaseDataframe(new_df))
        self.assertEqual(self.ctrl.df_level, 0)

    def test_previous_layouts_cleared(self):
        self.ctrl.previous_layouts.append({'a': (0.1, 0.2)})
        new_df = _make_df()
        asyncio.run(self.ctrl.replaceBaseDataframe(new_df))
        self.assertEqual(self.ctrl.previous_layouts, [])

    def test_selected_entities_cleared(self):
        self.ctrl.selected_entities = {'a', 'b'}
        new_df = _make_df()
        asyncio.run(self.ctrl.replaceBaseDataframe(new_df))
        self.assertEqual(self.ctrl.selected_entities, set())

    # ── position preservation ────────────────────────────────────────────────

    def test_existing_node_positions_preserved(self):
        # Move 'a' to a known spot, then replace the dataframe with the same nodes
        self.ctrl.dfs_layout[0].pos['a'] = (0.42, 0.77)
        new_df = _make_df()  # has same nodes a, b, c
        asyncio.run(self.ctrl.replaceBaseDataframe(new_df))
        pos_a = self.ctrl.dfs_layout[0].pos['a']
        self.assertAlmostEqual(pos_a[0], 0.42, places=6)
        self.assertAlmostEqual(pos_a[1], 0.77, places=6)

    def test_new_node_receives_a_position(self):
        # Add a new node 'd' that wasn't in the original dataframe
        new_df = pl.DataFrame({
            'fm':       ['a', 'b', 'c', 'd'],
            'to':       ['b', 'c', 'a', 'a'],
            'category': ['x', 'y', 'x', 'y'],
            'weight':   [1,   3,   2,   1  ],
        })
        asyncio.run(self.ctrl.replaceBaseDataframe(new_df))
        self.assertIn('d', self.ctrl.dfs_layout[0].pos)
        pos_d = self.ctrl.dfs_layout[0].pos['d']
        self.assertIsNotNone(pos_d[0])
        self.assertIsNotNone(pos_d[1])

    def test_removed_node_not_in_new_layout(self):
        # Remove node 'c' — it should not be forced into the new graph
        new_df = pl.DataFrame({
            'fm':       ['a', 'b'],
            'to':       ['b', 'a'],
            'category': ['x', 'y'],
            'weight':   [1,   3  ],
        })
        asyncio.run(self.ctrl.replaceBaseDataframe(new_df))
        # The graph should only have 'a' and 'b'
        self.assertNotIn('c', self.ctrl.graphs[0].nodes())

    # ── view refresh ─────────────────────────────────────────────────────────

    def test_mod_inner_updated_after_replace(self):
        original_inner = self.ctrl.mod_inner
        new_df = _make_df()
        asyncio.run(self.ctrl.replaceBaseDataframe(new_df))
        # mod_inner should have been refreshed (not guaranteed identical, but set)
        self.assertIsNotNone(self.ctrl.mod_inner)

    # ── deep-stack reset ─────────────────────────────────────────────────────

    def test_reset_from_mid_stack_collapses_to_single_level(self):
        # Push a filtered frame, then replace — should collapse to one level
        filtered_df = self.df.filter(pl.col('fm') == 'a')
        self.ctrl.pushStack(filtered_df)
        self.assertEqual(self.ctrl.df_level, 1)

        new_df = _make_df()
        asyncio.run(self.ctrl.replaceBaseDataframe(new_df))
        self.assertEqual(self.ctrl.df_level, 0)
        self.assertEqual(len(self.ctrl.dfs), 1)

    def test_positions_taken_from_deepest_level_when_mid_stack(self):
        # After a push, move a node at level 1, then replace — position should carry over
        filtered_df = self.df.filter(pl.col('fm') == 'a')
        self.ctrl.pushStack(filtered_df)
        self.ctrl.dfs_layout[self.ctrl.df_level].pos['a'] = (0.99, 0.88)

        new_df = _make_df()
        asyncio.run(self.ctrl.replaceBaseDataframe(new_df))
        pos_a = self.ctrl.dfs_layout[0].pos['a']
        self.assertAlmostEqual(pos_a[0], 0.99, places=6)
        self.assertAlmostEqual(pos_a[1], 0.88, places=6)


@unittest.skipUnless(_PANEL_AVAILABLE_, 'panel not installed')
class TestCommunityDetection(unittest.TestCase):
    """Tests for the 'd' key — louvain community detection colored via node_color."""

    def setUp(self):
        self.p2s = Polars2SVG()
        # Two triangles joined by a single bridge edge (a1-b1): louvain separates them.
        self.df = pl.DataFrame({
            'fm': ['a1', 'a2', 'a3', 'b1', 'b2', 'b3', 'a1'],
            'to': ['a2', 'a3', 'a1', 'b2', 'b3', 'b1', 'b1'],
        })
        self.lp   = self.p2s.linkp(self.df, relationships=[('fm', 'to')])
        self.ctrl = self.p2s.linkpi(self.lp)

    def _press(self, key):
        self.ctrl.key_op_finished = key
        asyncio.run(self.ctrl.applyKeyOp(None))

    # ── apply_community_detection() ──────────────────────────────────────────

    def test_every_node_gets_a_color(self):
        _nc_ = self.ctrl.apply_community_detection()
        self.assertEqual(set(_nc_.keys()), set(self.ctrl.graphs[self.ctrl.df_level].nodes()))

    def test_two_cliques_yield_two_colors(self):
        _nc_ = self.ctrl.apply_community_detection()
        self.assertEqual(len(set(_nc_.values())), 2)

    def test_clique_members_share_one_color(self):
        _nc_ = self.ctrl.apply_community_detection()
        self.assertEqual(_nc_['a1'], _nc_['a2'])
        self.assertEqual(_nc_['a2'], _nc_['a3'])
        self.assertNotEqual(_nc_['a1'], _nc_['b2'])

    def test_colors_are_hex_strings(self):
        for _hex_ in self.ctrl.apply_community_detection().values():
            self.assertRegex(_hex_, r'^#[0-9a-fA-F]{6}$')

    def test_repeat_run_is_stable(self):
        # Colors hash off each community's canonical member, so a re-run must not reshuffle.
        self.assertEqual(self.ctrl.apply_community_detection(),
                         self.ctrl.apply_community_detection())

    def test_coincident_nodes_land_in_same_community(self):
        # Exactly-coincident nodes are merged before detection (as the layout ops do),
        # so a node stacked on top of another always takes that node's color.
        _ln_ = self.ctrl.dfs_layout[self.ctrl.df_level]
        _ln_.pos['a1'] = _ln_.pos['b1']
        _nc_ = self.ctrl.apply_community_detection()
        self.assertEqual(_nc_['a1'], _nc_['b1'])

    def test_sets_community_colors_attribute(self):
        _nc_ = self.ctrl.apply_community_detection()
        self.assertEqual(self.ctrl.community_colors, _nc_)

    # ── the 'd' / shift-d key ops ────────────────────────────────────────────

    def test_d_pushes_node_color_to_every_stack_level(self):
        self.ctrl.selected_entities = {'b3'}
        self.ctrl.apply_push_selected()
        self._press('d')
        for _layout_ in self.ctrl.dfs_layout:
            self.assertEqual(_layout_.node_color, self.ctrl.community_colors)

    def test_shift_d_restores_original_node_color(self):
        self._press('d')
        self.assertIsInstance(self.ctrl.dfs_layout[0].node_color, dict)
        self._press('D')
        self.assertEqual(self.ctrl.dfs_layout[0].node_color, self.ctrl._orig_node_color_)
        self.assertIsNone(self.ctrl.community_colors)

    def test_popped_stack_nodes_are_absent_from_the_color_dict(self):
        # Detect at a deeper level, then pop: the nodes only present at the shallower
        # level have no entry, so LinkP paints them the background color ("colorless").
        self.ctrl.selected_entities = {'b3'}
        self.assertTrue(self.ctrl.apply_push_selected())
        self._press('d')
        self.ctrl.apply_pop()
        self.assertEqual(self.ctrl.df_level, 0)
        self.assertNotIn('b3', self.ctrl.dfs_layout[0].node_color)

    def test_popped_stack_still_renders(self):
        self.ctrl.selected_entities = {'b3'}
        self.ctrl.apply_push_selected()
        self._press('d')
        self.ctrl.apply_pop()
        self.assertGreater(len(self.ctrl.dfs_layout[0].renderSVG()), 0)

    def test_d_does_not_move_nodes(self):
        _before_ = {k: tuple(v) for k, v in self.ctrl.dfs_layout[0].pos.items()}
        self._press('d')
        _after_ = {k: tuple(v) for k, v in self.ctrl.dfs_layout[0].pos.items()}
        self.assertEqual(_before_, _after_)


@unittest.skipUnless(_PANEL_AVAILABLE_, 'panel not installed')
class TestStickyLabelsAcrossStack(unittest.TestCase):
    """Tests for the 's' family — sticky labels (label_only) and label-mode
    (draw_labels) must be applied to EVERY stack layer, including dfs_layout[0]
    (the template future pushed layers are cloned from), so the label state is
    consistent as the user navigates and grows the stack. Regression: these ops
    used to write only self.dfs_layout[self.df_level]."""

    def setUp(self):
        self.p2s = Polars2SVG()
        self.df  = pl.DataFrame({
            'fm': ['a1', 'a2', 'a3', 'b1', 'b2', 'b3', 'a1'],
            'to': ['a2', 'a3', 'a1', 'b2', 'b3', 'b1', 'b1'],
        })
        self.lp   = self.p2s.linkp(self.df, relationships=[('fm', 'to')])
        self.ctrl = self.p2s.linkpi(self.lp)

    def _press(self, key, shift=False, ctrl=False):
        self.ctrl.shiftkey        = shift
        self.ctrl.ctrlkey         = ctrl
        self.ctrl.key_op_finished = key
        asyncio.run(self.ctrl.applyKeyOp(None))

    # ── sticky-label set (s / shift-s / ctrl-s) ──────────────────────────────

    def test_s_sets_sticky_labels_on_every_existing_level(self):
        self.ctrl.selected_entities = {'b3'}
        self.ctrl.apply_push_selected()                 # level 1
        self.ctrl.label_mode = 'sticky labels'
        self.ctrl.selected_entities = {'a1', 'a2'}
        self._press('s')                                # replace sticky set
        self.assertEqual(self.ctrl.sticky_labels, {'a1', 'a2'})
        for _layout_ in self.ctrl.dfs_layout:
            self.assertTrue(_layout_.draw_labels)
            self.assertEqual(_layout_.label_only, {'a1', 'a2'})

    def test_setting_labels_at_deep_level_updates_the_template(self):
        # Regression: at level 1 the old handler touched only dfs_layout[1], so
        # dfs_layout[0] (the template) never learned the sticky set.
        self.ctrl.selected_entities = {'b3'}
        self.ctrl.apply_push_selected()                 # level 1
        self.assertEqual(self.ctrl.df_level, 1)
        self.ctrl.label_mode = 'sticky labels'
        self.ctrl.selected_entities = {'a1', 'a2'}
        self._press('s')
        self.assertEqual(self.ctrl.dfs_layout[0].label_only, {'a1', 'a2'})
        self.assertTrue(self.ctrl.dfs_layout[0].draw_labels)

    def test_new_layer_pushed_after_setting_labels_inherits_them(self):
        # The key end-to-end guarantee: grow the stack AFTER choosing sticky
        # labels and the fresh (cloned) layer still carries them.
        self.ctrl.selected_entities = {'a3'}
        self.ctrl.apply_push_selected()                 # level 1 (so we set at level 1)
        self.ctrl.label_mode = 'sticky labels'
        self.ctrl.selected_entities = {'a1'}
        self._press('s')
        self.ctrl.selected_entities = {'b3'}
        self.ctrl.apply_push_selected()                 # level 2, cloned from template
        _top_ = self.ctrl.dfs_layout[self.ctrl.df_level]
        self.assertTrue(_top_.draw_labels)
        self.assertEqual(_top_.label_only, {'a1'})

    def test_ctrl_s_adds_and_shift_s_removes_across_stack(self):
        self.ctrl.selected_entities = {'b3'}
        self.ctrl.apply_push_selected()                 # level 1
        self.ctrl.label_mode = 'sticky labels'
        self.ctrl.selected_entities = {'a1'}
        self._press('s')                                # {a1}
        self.ctrl.selected_entities = {'a2'}
        self._press('s', ctrl=True)                     # add -> {a1, a2}
        self.assertEqual(self.ctrl.sticky_labels, {'a1', 'a2'})
        self.ctrl.selected_entities = {'a1'}
        self._press('s', shift=True)                    # remove -> {a2}
        self.assertEqual(self.ctrl.sticky_labels, {'a2'})
        for _layout_ in self.ctrl.dfs_layout:
            self.assertEqual(_layout_.label_only, {'a2'})

    # ── label-visibility mode (ctrl-shift-s) ─────────────────────────────────

    def test_ctrl_shift_s_cycles_mode_on_every_level(self):
        self.ctrl.selected_entities = {'b3'}
        self.ctrl.apply_push_selected()                 # level 1
        self.ctrl.label_mode = 'all labels'
        self._press('S', shift=True, ctrl=True)         # all -> sticky
        self.assertEqual(self.ctrl.label_mode, 'sticky labels')
        for _layout_ in self.ctrl.dfs_layout:
            self.assertTrue(_layout_.draw_labels)
        self._press('S', shift=True, ctrl=True)         # sticky -> no labels
        self.assertEqual(self.ctrl.label_mode, 'no labels')
        for _layout_ in self.ctrl.dfs_layout:
            self.assertFalse(_layout_.draw_labels)
        self._press('S', shift=True, ctrl=True)         # no labels -> all
        self.assertEqual(self.ctrl.label_mode, 'all labels')
        for _layout_ in self.ctrl.dfs_layout:
            self.assertTrue(_layout_.draw_labels)
            self.assertEqual(_layout_.label_only, set())

    def test_all_labels_mode_clears_label_only_everywhere(self):
        self.ctrl.label_mode = 'sticky labels'
        self.ctrl.selected_entities = {'a1'}
        self._press('s')
        self.ctrl.selected_entities = {'b3'}
        self.ctrl.apply_push_selected()                 # level 1, inherits sticky
        # ctrl-shift-s: sticky -> no labels -> all labels
        self._press('S', shift=True, ctrl=True)         # -> no labels
        self._press('S', shift=True, ctrl=True)         # -> all labels
        for _layout_ in self.ctrl.dfs_layout:
            self.assertTrue(_layout_.draw_labels)
            self.assertEqual(_layout_.label_only, set())


@unittest.skipUnless(_PANEL_AVAILABLE_, 'panel not installed')
class TestCKeyRecenterAfterPush(unittest.TestCase):
    """Regression: after removing a node with 'x', pressing 'c' (no modifier, no
    selection) must recenter/rescale on the remaining (rendered) nodes only.

    The removed node's stale position survives in the pushed level's pos dict
    (__renderView__ copies the parent pos wholesale). With the old
    use_pos_for_bounds=True default, __calculateGeometry__ stretched the bounds
    back out to that removed node, so 'c' refit a window that still enclosed it.
    use_pos_for_bounds now defaults to False, so the fit ignores off-df nodes."""

    def setUp(self):
        self.p2s = Polars2SVG()
        # a-b-c triangle plus a far outlier 'far' edged to a. 'far' is the node
        # we remove; its position is 100x beyond the triangle's extent.
        self.df  = pl.DataFrame({'fm': ['a', 'b', 'c', 'far'],
                                 'to': ['b', 'c', 'a', 'a']})
        self.pos = {'a': (0.0, 0.0), 'b': (1.0, 0.0), 'c': (0.5, 0.866),
                    'far': (100.0, 100.0)}
        self.lp   = self.p2s.linkp(self.df, relationships=[('fm', 'to')], pos=self.pos)
        self.ctrl = self.p2s.linkpi(self.lp)

    def _press(self, key):
        self.ctrl.key_op_finished = key
        asyncio.run(self.ctrl.applyKeyOp(None))

    def _remove_far(self):
        self.ctrl.selected_entities = {'far'}
        self.assertTrue(self.ctrl.apply_push_selected())   # 'x'
        self.assertEqual(self.ctrl.df_level, 1)
        # pushStack intersects the selection with the surviving nodes, so 'far'
        # has already dropped out — 'c' sees an empty selection.
        self.assertEqual(self.ctrl.selected_entities, set())

    def test_push_leaves_removed_node_in_pos(self):
        # Precondition the fix hinges on: 'far' is gone from the graph/df but its
        # position lingers in the pushed level's pos dict.
        self._remove_far()
        _ln_ = self.ctrl.dfs_layout[self.ctrl.df_level]
        self.assertNotIn('far', self.ctrl.graphs[self.ctrl.df_level].nodes())
        self.assertIn('far', _ln_.pos)

    def test_c_after_push_fits_only_visible_nodes(self):
        self._remove_far()
        self._press('c')
        wx0, wy0, wx1, wy1 = self.ctrl.dfs_layout[self.ctrl.df_level].view_window
        # a,b,c span x in [0,1], y in [0,0.866]; the removed 'far' at (100,100)
        # must not stretch the window. Ceilings are generous vs. bounds_percent.
        self.assertLess(wx1, 10.0)
        self.assertLess(wy1, 10.0)

    def test_use_pos_for_bounds_true_still_includes_removed_node(self):
        # Opt-in contrast: with the flag on, the stale pos DOES drag the bounds
        # out to the removed node — exactly what the new default avoids.
        self._remove_far()
        _ln_ = self.ctrl.dfs_layout[self.ctrl.df_level]
        _ln_.use_pos_for_bounds = True
        self._press('c')
        wx0, wy0, wx1, wy1 = _ln_.view_window
        self.assertGreater(wx1, 50.0)
        self.assertGreater(wy1, 50.0)


def _node_screen_xy(lp):
    """Map node name -> (sx, sy) screen coords from a rendered LinkP."""
    lp.renderSVG()
    out = {}
    for sx, sy, nm in lp.df_node.select('__sx__', '__sy__', '__nm__').iter_rows():
        for n in (nm if isinstance(nm, (list, set)) else [nm]):
            out[str(n)] = (sx, sy)
    return out


class TestLinkPRecordsAt(unittest.TestCase):
    """LinkP.recordsAt() — the spatial hit-test that feeds brush propagation."""

    def setUp(self):
        self.p2s = Polars2SVG()
        # a->b appears twice; b->c and c->a once each -> lets us prove directionality
        self.df = pl.DataFrame({
            'fm':  ['a', 'b', 'c', 'a'],
            'to':  ['b', 'c', 'a', 'b'],
            'amt': [ 1,   2,   3,   4 ],
        })
        self.pos = {'a': [0, 0], 'b': [1, 0], 'c': [1, 1]}
        self.lp  = self.p2s.linkp(self.df, relationships=[('fm', 'to')], pos=self.pos)
        self.scr = _node_screen_xy(self.lp)

    def test_node_hit_returns_rows_where_node_is_source_or_dest(self):
        sx, sy = self.scr['a']
        out = self.lp.recordsAt((sx, sy), threshold=6)
        # every row touching 'a': a->b (x2), c->a
        rows = set(out.select('fm', 'to').iter_rows())
        self.assertEqual(rows, {('a', 'b'), ('c', 'a')})

    def test_edge_hit_is_directional(self):
        (ax, ay), (bx, by) = self.scr['a'], self.scr['b']
        mx, my = (ax + bx) // 2, (ay + by) // 2
        out = self.lp.recordsAt((mx, my), threshold=4)
        # only a->b rows (the drawn direction); not b->c or c->a
        self.assertEqual(set(out['to'].to_list()), {'b'})
        self.assertEqual(set(out['fm'].to_list()), {'a'})
        self.assertEqual(len(out), 2)  # both a->b rows

    def test_far_away_returns_empty_with_schema(self):
        out = self.lp.recordsAt((-50, -50), threshold=3)
        self.assertEqual(len(out), 0)
        self.assertEqual(out.columns, self.df.columns)

    def test_result_has_only_original_columns(self):
        sx, sy = self.scr['a']
        out = self.lp.recordsAt((sx, sy), threshold=6)
        self.assertEqual(out.columns, self.df.columns)

    def test_default_shape_matches_circle(self):
        sx, sy = self.scr['a']
        self.assertEqual(
            self.lp.recordsAt((sx, sy), threshold=6).sort('amt').rows(),
            self.lp.recordsAt((sx, sy), shape=self.p2s.SELECT_CIRCLEp, threshold=6).sort('amt').rows(),
        )

    def test_band_shapes_raise(self):
        with self.assertRaises(ValueError):
            self.lp.recordsAt((0, 0), shape=self.p2s.SELECT_HORIZONTALp)
        with self.assertRaises(ValueError):
            self.lp.recordsAt((0, 0), shape=self.p2s.SELECT_VERTICALp)


class TestLinkPInteractiveBrush(unittest.TestCase):
    """Brush round-trip: a linkp source broadcasts a subset to a linkp peer on the
    same stack, and the peer re-renders it (and reverts on clear)."""

    def setUp(self):
        self.p2s = Polars2SVG()
        self.df  = pl.DataFrame({
            'fm':  ['a', 'b', 'c', 'a'],
            'to':  ['b', 'c', 'a', 'b'],
            'amt': [ 1,   2,   3,   4 ],
        })
        self.pos = {'a': [0, 0], 'b': [1, 0], 'c': [1, 1]}
        self.mvc = InteractionController()
        self.mvc.addStack('default', self.df)
        self.src  = linkpi(self.p2s.linkp(self.df, relationships=[('fm', 'to')], pos=self.pos), mvc=self.mvc)
        self.peer = linkpi(self.p2s.linkp(self.df, relationships=[('fm', 'to')], pos=self.pos), mvc=self.mvc)
        self.mvc.view_stack[id(self.src)]  = 'default'
        self.mvc.view_stack[id(self.peer)] = 'default'
        self.scr = _node_screen_xy(self.src.dfs_layout[0])

    def _edge_midpoint(self, u, v):
        (ux, uy), (vx, vy) = self.scr[u], self.scr[v]
        return (ux + vx) // 2, (uy + vy) // 2

    def test_peer_rerenders_subset_on_edge_brush(self):
        full = self.peer.mod_inner
        self.src.brush_state = 1
        mx, my = self._edge_midpoint('a', 'b')
        asyncio.run(self.src._doBrushAt((mx, my), 1))
        self.assertTrue(self.peer._brush_active_)
        self.assertNotEqual(self.peer.mod_inner, full)
        # edge brush -> fewer links drawn than the full graph
        self.assertLess(self.peer.mod_inner.count('<line'), full.count('<line'))

    def test_source_does_not_brush_itself(self):
        self.src.brush_state = 1
        mx, my = self._edge_midpoint('a', 'b')
        asyncio.run(self.src._doBrushAt((mx, my), 1))
        self.assertFalse(self.src._brush_active_)

    def test_brush_clear_reverts_peer(self):
        full = self.peer.mod_inner
        self.src.brush_state = 1
        mx, my = self._edge_midpoint('a', 'b')
        asyncio.run(self.src._doBrushAt((mx, my), 1))
        self.assertTrue(self.peer._brush_active_)
        asyncio.run(self.mvc.brushClear(self.src))
        self.assertFalse(self.peer._brush_active_)
        self.assertEqual(self.peer.mod_inner, full)

    def test_empty_hit_clears_rather_than_brushes(self):
        self.src.brush_state = 1
        asyncio.run(self.src._doBrushAt((-50, -50), 1))
        self.assertFalse(self.peer._brush_active_)

    def test_brush_state_zero_broadcasts_clear(self):
        # turning the brush off (state 0) should revert peers, not brush them
        self.src.brush_state = 1
        mx, my = self._edge_midpoint('a', 'b')
        asyncio.run(self.src._doBrushAt((mx, my), 1))
        self.assertTrue(self.peer._brush_active_)
        self.src.brush_state = 0
        self.src.x_mouse, self.src.y_mouse = mx, my
        asyncio.run(self.src.applyBrushOp(None))
        self.assertFalse(self.peer._brush_active_)


@unittest.skipUnless(_PANEL_AVAILABLE_, 'panel not installed')
class TestEdgeUnfilter(unittest.TestCase):
    """Tests for the 'f' key — edge unfilter: ADD base-dataframe rows lying on the
    currently-visible edges on top of the current view (the rest of the view is kept).
    With a selection, scope to the subgraph induced by the selected nodes first."""

    def setUp(self):
        self.p2s = Polars2SVG()
        # a-b has 3 rows, b-c/c-a have 1 each, far-a (peripheral) has 2 rows: 7 total.
        self.df = pl.DataFrame({
            'fm':       ['a', 'a', 'a', 'b', 'c', 'far', 'far'],
            'to':       ['b', 'b', 'b', 'c', 'a', 'a',   'a'  ],
            'category': ['x', 'y', 'x', 'y', 'x', 'z',   'z'  ],
        })
        self.lp   = self.p2s.linkp(self.df, relationships=[('fm', 'to')])
        self.ctrl = self.p2s.linkpi(self.lp)

    def _press(self, key):
        self.ctrl.key_op_finished = key
        asyncio.run(self.ctrl.applyKeyOp(None))

    def _len_here(self):
        return len(self.ctrl.dfs[self.ctrl.df_level])

    def _edge_count(self, fm, to):
        _df_ = self.ctrl.dfs[self.ctrl.df_level]
        return _df_.filter((pl.col('fm') == fm) & (pl.col('to') == to)).height

    # ── no selection ─────────────────────────────────────────────────────────

    def test_refills_thinned_edges(self):
        # Collapse every edge to one row (4 rows), then edge-unfilter restores all 7.
        self.assertTrue(self.ctrl.apply_collapse_edges())
        self.assertEqual(self._len_here(), 4)
        self.assertTrue(self.ctrl.apply_edge_unfilter())
        self.assertEqual(self.ctrl.df_level, 2)
        self.assertEqual(self._len_here(), 7)

    def test_no_op_when_nothing_filtered(self):
        # At the base with every row present there is nothing to add back.
        self.assertFalse(self.ctrl.apply_edge_unfilter())
        self.assertEqual(self.ctrl.df_level, 0)
        self.assertEqual(len(self.ctrl.dfs), 1)

    # ── with a selection: ADDITIVE (rest of the view is preserved) ────────────

    def test_selection_refills_only_that_edge_and_keeps_the_rest(self):
        # Collapse thins every edge to one row (a-b now shows 1 of 3). Selecting the
        # neighbors a & b and edge-unfiltering refills a-b to its full 3 rows while
        # leaving the b-c / c-a / far-a rows untouched -- an additive operation.
        self.assertTrue(self.ctrl.apply_collapse_edges())     # level 1, 4 rows
        self.assertEqual(self._edge_count('a', 'b'), 1)
        self.ctrl.selected_entities = {'a', 'b'}
        self.assertTrue(self.ctrl.apply_edge_unfilter())      # level 2
        self.assertEqual(self._len_here(), 6)                 # 3 kept (b-c,c-a,far-a) + 3 a-b
        self.assertEqual(self._edge_count('a', 'b'), 3)       # a-b fully restored
        self.assertEqual(set(self.ctrl.graphs[self.ctrl.df_level].nodes()),
                         {'a', 'b', 'c', 'far'})              # rest of the graph preserved

    # ── key dispatch + discoverability ───────────────────────────────────────

    def test_f_key_dispatches_edge_unfilter(self):
        # Drive both steps through the key handler so the mvc stack stays in sync:
        # collapse (ctrl-shift-x) thins the edges, then 'f' restores every row.
        self._press('ctrl_shift_x')
        self.assertEqual(self.ctrl.df_level, 1)
        self._press('f')
        self.assertEqual(self.ctrl.df_level, 2)
        self.assertEqual(self._len_here(), 7)

    def test_keydown_js_captures_f_and_F(self):
        kd = type(self.ctrl)._scripts['myOnKeyDown']
        self.assertIn("data.key_op_finished = 'f'", kd)
        self.assertIn("data.key_op_finished = 'F'", kd)

    def test_help_text_advertises_f_and_F(self):
        self.assertIn('edge unfilter',  self.ctrl._keyboard_commands_)
        self.assertIn('node expansion', self.ctrl._keyboard_commands_)


@unittest.skipUnless(_PANEL_AVAILABLE_, 'panel not installed')
class TestNodeExpansion(unittest.TestCase):
    """Tests for the 'F' key — node expansion: ADD base-dataframe rows incident to the
    currently-visible nodes (source or destination) on top of the current view, which
    can pull previously-filtered neighbors back in. With a selection, scope to the
    selected nodes first; every other visible row is preserved."""

    def setUp(self):
        self.p2s = Polars2SVG()
        self.df = pl.DataFrame({
            'fm':       ['a', 'a', 'a', 'b', 'c', 'far', 'far'],
            'to':       ['b', 'b', 'b', 'c', 'a', 'a',   'a'  ],
            'category': ['x', 'y', 'x', 'y', 'x', 'z',   'z'  ],
        })
        self.lp   = self.p2s.linkp(self.df, relationships=[('fm', 'to')])
        self.ctrl = self.p2s.linkpi(self.lp)

    def _press(self, key):
        self.ctrl.key_op_finished = key
        asyncio.run(self.ctrl.applyKeyOp(None))

    def _len_here(self):
        return len(self.ctrl.dfs[self.ctrl.df_level])

    def _edge_count(self, fm, to):
        _df_ = self.ctrl.dfs[self.ctrl.df_level]
        return _df_.filter((pl.col('fm') == fm) & (pl.col('to') == to)).height

    # ── with a selection: ADDITIVE (rest of the view is preserved) ────────────

    def test_selection_expands_selected_and_keeps_the_rest(self):
        # Remove 'far' via 'x' (level 1, 5 rows: a-b x3, b-c, c-a; far hidden). Then
        # select the visible node 'a' and node-expand: 'a's incident base rows return
        # -- including far-a, so 'far' comes back -- while b-c (not incident to a) is
        # kept. Additive: nothing already visible is dropped.
        self.ctrl.selected_entities = {'far'}
        self.assertTrue(self.ctrl.apply_push_selected())      # level 1
        self.assertEqual(self._len_here(), 5)
        self.ctrl.selected_entities = {'a'}
        self.assertTrue(self.ctrl.apply_node_expansion())     # level 2
        self.assertEqual(self._len_here(), 7)
        self.assertIn('far', self.ctrl.graphs[self.ctrl.df_level].nodes())  # neighbor pulled back
        self.assertEqual(self._edge_count('b', 'c'), 1)       # non-incident row preserved

    # ── no selection: reaches into the base to restore filtered neighbors ─────

    def test_restores_a_removed_neighbor(self):
        # Remove 'far' via 'x' (down to 5 rows), then node-expand from the visible core
        # {a,b,c}: 'far' and its 2 rows return because far-a is incident to a.
        self.ctrl.selected_entities = {'far'}
        self.assertTrue(self.ctrl.apply_push_selected())
        self.assertEqual(self._len_here(), 5)
        self.assertEqual(self.ctrl.selected_entities, set())   # 'far' left the graph
        self.assertTrue(self.ctrl.apply_node_expansion())
        self.assertEqual(self._len_here(), 7)
        self.assertIn('far', self.ctrl.graphs[self.ctrl.df_level].nodes())

    def test_no_op_on_full_graph(self):
        # Nothing has been filtered, so expanding from every visible node adds nothing.
        self.assertFalse(self.ctrl.apply_node_expansion())
        self.assertEqual(self.ctrl.df_level, 0)

    # ── key dispatch ─────────────────────────────────────────────────────────

    def test_F_key_dispatches_node_expansion(self):
        # Remove 'far' (mvc-synced), then 'F' brings it back on top of the visible core.
        self.ctrl.selected_entities = {'far'}
        self._press('x')
        self.assertEqual(self.ctrl.df_level, 1)
        self._press('F')
        self.assertEqual(self.ctrl.df_level, 2)
        self.assertEqual(self._len_here(), 7)
        self.assertIn('far', self.ctrl.graphs[self.ctrl.df_level].nodes())


@unittest.skipUnless(_PANEL_AVAILABLE_, 'panel not installed')
class TestZKeyColorSelection(unittest.TestCase):
    """Regression tests: 'z' selects every node sharing the color under the mouse.

    With integer node ids the hit-test returns ints while color_nodes_final is keyed by
    the stringified names, so nodeColor() returned None and 'z' silently selected
    nothing -- the case hit by panelize([[linkp]]) over an int-id graph."""

    def setUp(self):
        self.p2s    = Polars2SVG()
        self.df     = pl.DataFrame({'fm': [0, 1, 2, 3], 'to': [1, 2, 3, 0]})
        self.pos    = {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (1.0, 1.0), 3: (0.0, 1.0)}
        self.colors = {'0': '#2166ac', '1': '#d6604d', '2': '#2166ac', '3': '#1a9850'}
        self.lp     = self.p2s.linkp(self.df, relationships=[('fm', 'to')], pos=self.pos,
                                     node_color=self.colors)
        self.lp.renderSVG()
        self.ctrl   = self.p2s.linkpi(self.lp)

    def _press_z_over(self, node):
        _row_ = self.lp.df_node.explode('__nm__').filter(pl.col('__nm__') == str(node))
        self.ctrl.x_mouse, self.ctrl.y_mouse = _row_['__sx__'][0], _row_['__sy__'][0]
        self.ctrl.key_op_finished = 'z'
        asyncio.run(self.ctrl.applyKeyOp(None))
        return set(self.ctrl.selected_entities)

    def test_selects_every_node_sharing_the_color(self):
        self.assertEqual(self._press_z_over(0), {0, 2})

    def test_selects_just_the_node_when_the_color_is_unique(self):
        self.assertEqual(self._press_z_over(1), {1})

    def test_selection_keeps_the_original_node_id_type(self):
        _selected_ = self._press_z_over(3)
        self.assertEqual(_selected_, {3})
        for _node_ in _selected_:
            self.assertIsInstance(_node_, int,
                                  f'selection holds {_node_!r} ({type(_node_).__name__}), expected int')


if __name__ == '__main__':
    unittest.main()
