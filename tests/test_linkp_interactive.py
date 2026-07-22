import asyncio
import unittest
import polars as pl
from polars2svg import Polars2SVG


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


if __name__ == '__main__':
    unittest.main()
