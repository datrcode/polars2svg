import unittest
import polars as pl
from polars2svg import Polars2SVG


_DF_ = pl.DataFrame({
    'src': ['abc', 'def', 'ghi', 'jkl', 'cat', 'dog'],
    'dst': ['def', 'ghi', 'jkl', 'abc', 'abc', 'abc'],
})
_POS_ = {
    'ghi': (0.7127330398521416, 0.16147226170840212),
    'jkl': (0.18301128026917612, 0.16147226170840212),
    'def': (0.7127330398521416, 0.40937323255213387),
    'abc': (0.18301128026917612, 0.40937323255213387),
    'cat': (0.18301128026917612, 0.4814274939738414),
    'dog': (0.034882399169988164, 0.40937323255213387),
}
_RELS_ = [('src', 'dst')]
_WXH_  = (512, 512)


def _make_widget(p2s):
    lp = p2s.linkp(_DF_, _RELS_, dict(_POS_), wxh=_WXH_)
    return p2s.linkpi(lp)


def _drag_select(widget, nodes):
    """Simulate a drag-selection bounding the screen positions of the given nodes."""
    lp = widget.dfs_layout[widget.df_level]
    sxs  = [lp.xT(lp.pos[n][0]) for n in nodes if n in lp.pos]
    sys_ = [lp.yT(lp.pos[n][1]) for n in nodes if n in lp.pos]
    margin = 5
    widget.apply_drag_select(
        min(sxs) - margin, min(sys_) - margin,
        max(sxs) + margin, max(sys_) + margin,
    )


def _graph_nodes(widget):
    """Return nodes visible in the current stack level (full-canvas drag select)."""
    lp = widget.dfs_layout[widget.df_level]
    w, h = lp.wxh
    widget.selected_entities = set()
    widget.apply_drag_select(0, 0, w, h)
    nodes = set(widget.selected_entities)
    widget.selected_entities = set()
    return nodes


class TestStackBasedNodePositioning(unittest.TestCase):

    def setUp(self):
        self.p2s    = Polars2SVG()
        self.widget = _make_widget(self.p2s)

    # -------------------------------------------------------------------------
    # Drag-selection smoke tests
    # -------------------------------------------------------------------------

    def test_drag_select_ghi_finds_ghi(self):
        _drag_select(self.widget, {'ghi'})
        self.assertIn('ghi', self.widget.selected_entities,
                      'drag rect around ghi must select ghi')

    def test_drag_select_jkl_finds_jkl(self):
        _drag_select(self.widget, {'jkl'})
        self.assertIn('jkl', self.widget.selected_entities)

    def test_drag_select_cat_dog_abc_finds_all_three(self):
        _drag_select(self.widget, {'cat', 'dog', 'abc'})
        for node in ('cat', 'dog', 'abc'):
            self.assertIn(node, self.widget.selected_entities,
                          f'drag rect around cat/dog/abc must select {node}')

    # -------------------------------------------------------------------------
    # Stack push smoke tests
    # -------------------------------------------------------------------------

    def test_push_removes_ghi_from_level1(self):
        w = self.widget
        _drag_select(w, {'ghi'})
        w.apply_push_selected()
        self.assertEqual(w.df_level, 1)
        self.assertNotIn('ghi', _graph_nodes(w),
                         'ghi must be absent from the rendered graph at level 1')

    def test_push_twice_removes_ghi_and_jkl(self):
        w = self.widget
        _drag_select(w, {'ghi'})
        w.apply_push_selected()
        _drag_select(w, {'jkl'})
        w.apply_push_selected()
        self.assertEqual(w.df_level, 2)
        nodes = _graph_nodes(w)
        self.assertNotIn('ghi', nodes)
        self.assertNotIn('jkl', nodes)

    # -------------------------------------------------------------------------
    # Propagation tests
    # -------------------------------------------------------------------------

    def test_move_propagates_to_all_stack_levels(self):
        w = self.widget
        _drag_select(w, {'ghi'}); w.apply_push_selected()
        _drag_select(w, {'jkl'}); w.apply_push_selected()

        orig_cat_0 = w.dfs_layout[0].pos['cat']
        orig_abc_0 = w.dfs_layout[0].pos['abc']

        _drag_select(w, {'cat', 'dog', 'abc'})
        w.apply_move_selected(10, 5)

        self.assertNotEqual(w.dfs_layout[0].pos['cat'], orig_cat_0,
                            'cat pos at level 0 must change after move+propagate')
        self.assertNotEqual(w.dfs_layout[0].pos['abc'], orig_abc_0,
                            'abc pos at level 0 must change after move+propagate')
        self.assertNotEqual(w.dfs_layout[1].pos['cat'], orig_cat_0,
                            'cat pos at level 1 must change after move+propagate')

    def test_positions_persist_after_first_pop(self):
        w = self.widget
        _drag_select(w, {'ghi'}); w.apply_push_selected()
        _drag_select(w, {'jkl'}); w.apply_push_selected()
        _drag_select(w, {'cat', 'dog', 'abc'})
        w.apply_move_selected(10, 5)
        moved_cat = w.dfs_layout[w.df_level].pos['cat']
        moved_abc = w.dfs_layout[w.df_level].pos['abc']
        moved_dog = w.dfs_layout[w.df_level].pos['dog']

        w.apply_pop()

        self.assertEqual(w.df_level, 1)
        self.assertEqual(w.dfs_layout[1].pos['cat'], moved_cat)
        self.assertEqual(w.dfs_layout[1].pos['abc'], moved_abc)
        self.assertEqual(w.dfs_layout[1].pos['dog'], moved_dog)

    def test_positions_persist_after_second_pop(self):
        w = self.widget
        _drag_select(w, {'ghi'}); w.apply_push_selected()
        _drag_select(w, {'jkl'}); w.apply_push_selected()
        _drag_select(w, {'cat', 'dog', 'abc'})
        w.apply_move_selected(10, 5)
        moved_cat = w.dfs_layout[w.df_level].pos['cat']
        moved_abc = w.dfs_layout[w.df_level].pos['abc']
        moved_dog = w.dfs_layout[w.df_level].pos['dog']

        w.apply_pop()
        w.apply_pop()

        self.assertEqual(w.df_level, 0)
        self.assertEqual(w.dfs_layout[0].pos['cat'], moved_cat)
        self.assertEqual(w.dfs_layout[0].pos['abc'], moved_abc)
        self.assertEqual(w.dfs_layout[0].pos['dog'], moved_dog)

    # -------------------------------------------------------------------------
    # Integration: full scenario (a) through (j)
    # -------------------------------------------------------------------------

    def test_full_scenario_a_through_j(self):
        w = self.widget

        # (a) graph already created in setUp

        # (b) select 'ghi' via mouse drag
        _drag_select(w, {'ghi'})
        self.assertIn('ghi', w.selected_entities, '(b) drag rect must contain ghi')

        # (c) remove 'ghi' via 'x'
        w.apply_push_selected()
        self.assertEqual(w.df_level, 1, '(c) must be at level 1 after first push')
        self.assertNotIn('ghi', _graph_nodes(w),
                         '(c) ghi must be absent from rendered graph at level 1')

        # (d) select 'jkl' via mouse drag
        _drag_select(w, {'jkl'})
        self.assertIn('jkl', w.selected_entities, '(d) drag rect must contain jkl')

        # (e) remove 'jkl' via 'x'
        w.apply_push_selected()
        self.assertEqual(w.df_level, 2, '(e) must be at level 2 after second push')
        self.assertNotIn('jkl', _graph_nodes(w),
                         '(e) jkl must be absent from rendered graph at level 2')

        # (f) select 'cat', 'dog', 'abc' via mouse drag
        _drag_select(w, {'cat', 'dog', 'abc'})
        for node in ('cat', 'dog', 'abc'):
            self.assertIn(node, w.selected_entities, f'(f) drag rect must contain {node}')

        # (g) move those nodes 10 px right, 5 px down
        w.apply_move_selected(10, 5)
        moved_cat = w.dfs_layout[w.df_level].pos['cat']
        moved_abc = w.dfs_layout[w.df_level].pos['abc']
        moved_dog = w.dfs_layout[w.df_level].pos['dog']

        # (h) pop 'X' → level 2 → 1
        w.apply_pop()
        self.assertEqual(w.df_level, 1, '(h) first pop must land at level 1')

        # (i) verify nodes are still moved at level 1
        self.assertEqual(w.dfs_layout[1].pos['cat'], moved_cat,
                         '(i) cat pos at level 1 must match moved pos')
        self.assertEqual(w.dfs_layout[1].pos['abc'], moved_abc,
                         '(i) abc pos at level 1 must match moved pos')
        self.assertEqual(w.dfs_layout[1].pos['dog'], moved_dog,
                         '(i) dog pos at level 1 must match moved pos')

        # (j) pop 'X' → level 1 → 0; verify nodes are still moved
        w.apply_pop()
        self.assertEqual(w.df_level, 0, '(j) second pop must land at level 0')
        self.assertEqual(w.dfs_layout[0].pos['cat'], moved_cat,
                         '(j) cat pos at level 0 must match moved pos')
        self.assertEqual(w.dfs_layout[0].pos['abc'], moved_abc,
                         '(j) abc pos at level 0 must match moved pos')
        self.assertEqual(w.dfs_layout[0].pos['dog'], moved_dog,
                         '(j) dog pos at level 0 must match moved pos')

    # -------------------------------------------------------------------------
    # Integration: 't' collapse persists through three pops
    # -------------------------------------------------------------------------

    def test_t_collapse_persists_through_three_pops(self):
        w = self.widget

        # (1)/(2) select "def", remove via 'x'
        _drag_select(w, {'def'})
        self.assertIn('def', w.selected_entities, 'drag rect must contain def')
        w.apply_push_selected()
        self.assertNotIn('def', _graph_nodes(w),
                         'def must be absent from rendered graph at level 1')

        # (3)/(4) select "ghi", remove via 'x'
        _drag_select(w, {'ghi'})
        self.assertIn('ghi', w.selected_entities)
        w.apply_push_selected()
        self.assertNotIn('ghi', _graph_nodes(w),
                         'ghi must be absent from rendered graph at level 2')

        # (5)/(6) select "jkl", remove via 'x'
        _drag_select(w, {'jkl'})
        self.assertIn('jkl', w.selected_entities)
        w.apply_push_selected()
        self.assertNotIn('jkl', _graph_nodes(w),
                         'jkl must be absent from rendered graph at level 3')
        self.assertEqual(w.df_level, 3)

        # (7) select "abc" via drag
        _drag_select(w, {'abc'})
        self.assertIn('abc', w.selected_entities)

        # (8)/(9) press 't' — collapse abc to mouse position (20, 20)
        mouse_sx, mouse_sy = 20.0, 20.0
        w.apply_collapse_to(mouse_sx, mouse_sy)
        lp3 = w.dfs_layout[3]
        target_wx = lp3.xT_inv(mouse_sx)
        target_wy = lp3.yT_inv(mouse_sy)
        self.assertAlmostEqual(lp3.pos['abc'][0], target_wx, places=9,
                               msg='abc must be at target x after collapse')
        self.assertAlmostEqual(lp3.pos['abc'][1], target_wy, places=9,
                               msg='abc must be at target y after collapse')

        # (10) pop 'X' three times — back to level 0; verify abc is still at target
        w.apply_pop()
        w.apply_pop()
        w.apply_pop()
        self.assertEqual(w.df_level, 0, 'three pops must return to level 0')
        for level in range(4):
            self.assertAlmostEqual(
                w.dfs_layout[level].pos['abc'][0], target_wx, places=9,
                msg=f'abc world-x must equal target after pops (level {level})',
            )
            self.assertAlmostEqual(
                w.dfs_layout[level].pos['abc'][1], target_wy, places=9,
                msg=f'abc world-y must equal target after pops (level {level})',
            )


    # -------------------------------------------------------------------------
    # 'u' key: undo propagates back to all stack levels
    # -------------------------------------------------------------------------

    def test_undo_propagates_to_all_levels(self):
        w = self.widget

        # (a)/(b) select 'def', push 'x' → level 1
        _drag_select(w, {'def'}); w.apply_push_selected()
        self.assertEqual(w.df_level, 1)
        orig_abc = w.dfs_layout[1].pos['abc']

        # (c)/(d) select 'abc', move 5 right and 10 down
        _drag_select(w, {'abc'})
        self.assertIn('abc', w.selected_entities, 'drag rect must contain abc')
        w.apply_move_selected(5, 10)
        moved_abc = w.dfs_layout[1].pos['abc']
        self.assertNotEqual(moved_abc, orig_abc, 'abc must have moved')
        self.assertEqual(w.dfs_layout[0].pos['abc'], moved_abc,
                         'move must propagate to level 0')

        # (e) press 'u' — undo the move; abc returns to its pre-move position
        w.apply_undo()

        # (f) press 'X' — pop back to level 0
        w.apply_pop()
        self.assertEqual(w.df_level, 0)

        # (g) verify all stack positions are the same
        self.assertEqual(w.dfs_layout[0].pos['abc'], orig_abc,
                         'abc must be at original pos at level 0 after undo + pop')
        self.assertEqual(w.dfs_layout[0].pos['abc'], w.dfs_layout[1].pos['abc'],
                         'abc pos must be consistent across all stack levels')

    # -------------------------------------------------------------------------
    # 'y' key: line layout propagates through three pops
    # -------------------------------------------------------------------------

    def test_line_layout_propagates_through_three_pops(self):
        w = self.widget

        # Push def, ghi, jkl → land at level 3
        _drag_select(w, {'def'}); w.apply_push_selected()
        _drag_select(w, {'ghi'}); w.apply_push_selected()
        _drag_select(w, {'jkl'}); w.apply_push_selected()
        self.assertEqual(w.df_level, 3)

        # Select abc, cat, dog at level 3
        _drag_select(w, {'abc', 'cat', 'dog'})
        for node in ('abc', 'cat', 'dog'):
            self.assertIn(node, w.selected_entities, f'drag rect must contain {node}')

        # Apply line layout along a diagonal across the canvas
        w.apply_layout_interaction(50.0, 50.0, 450.0, 450.0, 'line')
        layout_abc = w.dfs_layout[3].pos['abc']
        layout_cat = w.dfs_layout[3].pos['cat']
        layout_dog = w.dfs_layout[3].pos['dog']

        # Pop three times → level 0; positions must survive at every level
        w.apply_pop()
        w.apply_pop()
        w.apply_pop()
        self.assertEqual(w.df_level, 0)
        for level in range(4):
            self.assertEqual(w.dfs_layout[level].pos['abc'], layout_abc,
                             f'abc pos must match layout result at level {level}')
            self.assertEqual(w.dfs_layout[level].pos['cat'], layout_cat,
                             f'cat pos must match layout result at level {level}')
            self.assertEqual(w.dfs_layout[level].pos['dog'], layout_dog,
                             f'dog pos must match layout result at level {level}')

    # -------------------------------------------------------------------------
    # 'g' key: grid layout propagates through three pops
    # -------------------------------------------------------------------------

    def test_grid_layout_propagates_through_three_pops(self):
        w = self.widget

        # Push def, ghi, jkl → land at level 3
        _drag_select(w, {'def'}); w.apply_push_selected()
        _drag_select(w, {'ghi'}); w.apply_push_selected()
        _drag_select(w, {'jkl'}); w.apply_push_selected()
        self.assertEqual(w.df_level, 3)

        # Select abc, cat, dog at level 3
        _drag_select(w, {'abc', 'cat', 'dog'})
        for node in ('abc', 'cat', 'dog'):
            self.assertIn(node, w.selected_entities, f'drag rect must contain {node}')

        # Apply grid layout within a bounding rectangle on the canvas
        w.apply_layout_interaction(50.0, 50.0, 450.0, 450.0, 'grid')
        layout_abc = w.dfs_layout[3].pos['abc']
        layout_cat = w.dfs_layout[3].pos['cat']
        layout_dog = w.dfs_layout[3].pos['dog']

        # Pop three times → level 0; positions must survive at every level
        w.apply_pop()
        w.apply_pop()
        w.apply_pop()
        self.assertEqual(w.df_level, 0)
        for level in range(4):
            self.assertEqual(w.dfs_layout[level].pos['abc'], layout_abc,
                             f'abc pos must match layout result at level {level}')
            self.assertEqual(w.dfs_layout[level].pos['cat'], layout_cat,
                             f'cat pos must match layout result at level {level}')
            self.assertEqual(w.dfs_layout[level].pos['dog'], layout_dog,
                             f'dog pos must match layout result at level {level}')


# Multi-row-per-edge fixture: several edges carry more than one row so that
# collapse-to-one-row is observable.  Unique edges = 5, total rows = 8.
_DF_MULTI_ = pl.DataFrame({
    'src': ['abc', 'abc', 'abc', 'def', 'ghi', 'ghi', 'cat', 'dog'],
    'dst': ['def', 'def', 'def', 'ghi', 'jkl', 'jkl', 'abc', 'abc'],
})


def _make_widget_multi(p2s):
    lp = p2s.linkp(_DF_MULTI_, _RELS_, dict(_POS_), wxh=_WXH_)
    return p2s.linkpi(lp)


def _edge_row_counts(widget):
    """Map of (src, dst) -> number of rows at the current stack level."""
    df = widget.dfs[widget.df_level]
    counts = {}
    for src, dst in df.select('src', 'dst').iter_rows():
        counts[(src, dst)] = counts.get((src, dst), 0) + 1
    return counts


class TestCollapseEdgesToOneRow(unittest.TestCase):

    def setUp(self):
        self.p2s    = Polars2SVG()
        self.widget = _make_widget_multi(self.p2s)

    def test_collapse_all_keeps_one_row_per_edge(self):
        w = self.widget
        before_nodes = _graph_nodes(w)
        before_edges = set(_edge_row_counts(w).keys())

        self.assertTrue(w.apply_collapse_edges())
        self.assertEqual(w.df_level, 1)

        counts = _edge_row_counts(w)
        self.assertEqual(set(counts.keys()), before_edges,
                         'collapse must preserve the set of edges')
        for edge, n in counts.items():
            self.assertEqual(n, 1, f'edge {edge} must collapse to a single row')
        self.assertEqual(_graph_nodes(w), before_nodes,
                         'collapse must leave the rendered node set unchanged')

    def test_collapse_with_selection_only_touches_adjacent_edges(self):
        w = self.widget
        _drag_select(w, {'ghi'})
        self.assertIn('ghi', w.selected_entities)

        self.assertTrue(w.apply_collapse_edges())
        self.assertEqual(w.df_level, 1)

        counts = _edge_row_counts(w)
        # Edges adjacent to ghi collapse to one row.
        self.assertEqual(counts[('def', 'ghi')], 1)
        self.assertEqual(counts[('ghi', 'jkl')], 1)
        # Edges not adjacent to ghi retain all of their rows.
        self.assertEqual(counts[('abc', 'def')], 3)
        self.assertEqual(counts[('cat', 'abc')], 1)
        self.assertEqual(counts[('dog', 'abc')], 1)

    def test_collapse_is_noop_when_already_one_row_per_edge(self):
        # _DF_ already has exactly one row per edge.
        w = _make_widget(self.p2s)
        self.assertFalse(w.apply_collapse_edges())
        self.assertEqual(w.df_level, 0)


if __name__ == '__main__':
    unittest.main()
