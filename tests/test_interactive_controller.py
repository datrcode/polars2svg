import asyncio
import unittest
from datetime import datetime

import polars as pl

from polars2svg import Polars2SVG
from polars2svg.interactive_controller import (
    InteractionController,
    _collect_leaves,
    _build_sketch_html,
    _sketch_leaf_html,
    _sketch_placeholder_html,
)

try:
    import panel as pn
    from panel.reactive import ReactiveHTML
    PANEL_AVAILABLE = True
except ImportError:
    PANEL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_df():
    return pl.DataFrame({
        'x':   [1.0, 2.0, 3.0, 4.0, 5.0],
        'y':   [2.0, 4.0, 1.0, 3.0, 5.0],
        'cat': ['a', 'b', 'a', 'b', 'a'],
        'ts':  [datetime(2024, 1, d) for d in range(1, 6)],
    })

def _make_link_df():
    return pl.DataFrame({'fm': ['a', 'b', 'c'], 'to': ['b', 'c', 'a']})

def _make_pos():
    return {'a': [0.0, 0.0], 'b': [1.0, 0.0], 'c': [0.5, 0.866]}


# ---------------------------------------------------------------------------
# MockView — minimal stand-in for a Panel reactive widget
# ---------------------------------------------------------------------------

class MockView:
    def __init__(self):
        self.display_calls   = []
        self.selection_calls = []

    async def display(self, df, dfs, index):
        self.display_calls.append({'df': df, 'dfs': dfs, 'index': index})

    async def receiveSelection(self, entities):
        self.selection_calls.append(entities)


# ===========================================================================
# Tier 1: InteractionController — pure Python, no Panel required
# ===========================================================================

class TestInteractionController(unittest.TestCase):

    def setUp(self):
        self.df  = pl.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        self.df2 = self.df.filter(pl.col('a') > 1)
        self.mvc = InteractionController()
        self.mvc.addStack('default', self.df)

    def _registered_view(self, stack='default'):
        v = MockView()
        self.mvc.link(v, [], on='stack', stack=stack)
        return v

    # ── addStack / stackTopDataFrame / stackCurrentDataFrame ─────────────────

    def test_addStack_creates_stack_entry(self):
        self.assertIn('default', self.mvc.stacks)

    def test_stackTopDataFrame_returns_original_df(self):
        v = self._registered_view()
        self.assertIs(self.mvc.stackTopDataFrame(v), self.df)

    def test_stackCurrentDataFrame_initially_equals_top(self):
        v = self._registered_view()
        self.assertIs(self.mvc.stackCurrentDataFrame(v), self.df)

    # ── pushStack ─────────────────────────────────────────────────────────────

    def test_pushStack_increments_index(self):
        v = self._registered_view()
        asyncio.run(self.mvc.pushStack(v, self.df2))
        self.assertEqual(self.mvc.stacks['default']['index'], 1)

    def test_pushStack_calls_display_on_caller(self):
        v = self._registered_view()
        asyncio.run(self.mvc.pushStack(v, self.df2))
        self.assertEqual(len(v.display_calls), 1)
        self.assertEqual(v.display_calls[0]['index'], 1)

    def test_pushStack_notifies_linked_peer(self):
        v1 = self._registered_view()
        v2 = MockView()
        self.mvc.link(v1, [v2], on='stack', stack='default')
        asyncio.run(self.mvc.pushStack(v1, self.df2))
        self.assertEqual(len(v2.display_calls), 1)

    def test_pushStack_updates_stackCurrentDataFrame(self):
        v = self._registered_view()
        asyncio.run(self.mvc.pushStack(v, self.df2))
        self.assertIs(self.mvc.stackCurrentDataFrame(v), self.df2)

    def test_pushStack_mid_history_truncates_forward(self):
        v   = self._registered_view()
        df3 = self.df.filter(pl.col('a') == 1)
        asyncio.run(self.mvc.pushStack(v, self.df2))
        asyncio.run(self.mvc.popStack(v))           # back to index 0
        asyncio.run(self.mvc.pushStack(v, df3))     # new branch from index 0
        s = self.mvc.stacks['default']
        self.assertEqual(len(s['dfs']), 2)          # [df, df3]; df2 gone
        self.assertIs(s['dfs'][1], df3)

    # ── popStack ──────────────────────────────────────────────────────────────

    def test_popStack_decrements_index(self):
        v = self._registered_view()
        asyncio.run(self.mvc.pushStack(v, self.df2))
        asyncio.run(self.mvc.popStack(v))
        self.assertEqual(self.mvc.stacks['default']['index'], 0)

    def test_popStack_calls_display_on_caller(self):
        v = self._registered_view()
        asyncio.run(self.mvc.pushStack(v, self.df2))
        v.display_calls.clear()
        asyncio.run(self.mvc.popStack(v))
        self.assertEqual(len(v.display_calls), 1)
        self.assertEqual(v.display_calls[0]['index'], 0)

    def test_popStack_at_zero_is_noop(self):
        v = self._registered_view()
        asyncio.run(self.mvc.popStack(v))
        self.assertEqual(len(v.display_calls), 0)
        self.assertEqual(self.mvc.stacks['default']['index'], 0)

    def test_popStack_notifies_linked_peer(self):
        v1 = self._registered_view()
        v2 = MockView()
        self.mvc.link(v1, [v2], on='stack', stack='default')
        asyncio.run(self.mvc.pushStack(v1, self.df2))
        v2.display_calls.clear()
        asyncio.run(self.mvc.popStack(v1))
        self.assertEqual(len(v2.display_calls), 1)

    # ── setStackIndex ─────────────────────────────────────────────────────────

    def test_setStackIndex_jumps_to_index(self):
        v   = self._registered_view()
        df3 = self.df.filter(pl.col('a') == 3)
        asyncio.run(self.mvc.pushStack(v, self.df2))
        asyncio.run(self.mvc.pushStack(v, df3))
        asyncio.run(self.mvc.setStackIndex(v, 0))
        self.assertEqual(self.mvc.stacks['default']['index'], 0)

    def test_setStackIndex_out_of_bounds_is_noop(self):
        v = self._registered_view()
        asyncio.run(self.mvc.setStackIndex(v, 99))
        self.assertEqual(len(v.display_calls), 0)

    # ── brushUpdate / brushClear ──────────────────────────────────────────────

    def test_brushUpdate_notifies_same_stack_peer(self):
        v1 = self._registered_view()
        v2 = MockView()
        self.mvc.link(v2, [], on='stack', stack='default')
        asyncio.run(self.mvc.brushUpdate(v1, self.df2))
        self.assertEqual(len(v2.display_calls), 1)

    def test_brushUpdate_does_not_notify_caller(self):
        v1 = self._registered_view()
        v2 = MockView()
        self.mvc.link(v2, [], on='stack', stack='default')
        asyncio.run(self.mvc.brushUpdate(v1, self.df2))
        self.assertEqual(len(v1.display_calls), 0)

    def test_brushClear_reverts_peer_to_current_stack_df(self):
        v1 = self._registered_view()
        v2 = MockView()
        self.mvc.link(v2, [], on='stack', stack='default')
        asyncio.run(self.mvc.pushStack(v1, self.df2))
        v2.display_calls.clear()
        asyncio.run(self.mvc.brushClear(v1))
        self.assertEqual(len(v2.display_calls), 1)
        self.assertIs(v2.display_calls[0]['df'], self.df2)

    # ── selectionUpdate / selectionClear ─────────────────────────────────────

    def test_selectionUpdate_routes_to_selection_linked_views(self):
        v1       = self._registered_view()
        v2       = MockView()
        entities = {'node_a', 'node_b'}
        self.mvc.link(v1, [v2], on='selection')
        asyncio.run(self.mvc.selectionUpdate(v1, entities))
        self.assertEqual(len(v2.selection_calls), 1)
        self.assertEqual(v2.selection_calls[0], entities)

    def test_selectionUpdate_skips_views_without_receiveSelection(self):
        v1 = self._registered_view()
        class NoSelView: pass
        v_no = NoSelView()
        self.mvc.link(v1, [v_no], on='selection')
        asyncio.run(self.mvc.selectionUpdate(v1, {'x'}))  # must not raise

    def test_selectionClear_sends_empty_set(self):
        v1 = self._registered_view()
        v2 = MockView()
        self.mvc.link(v1, [v2], on='selection')
        asyncio.run(self.mvc.selectionClear(v1))
        self.assertEqual(v2.selection_calls[0], set())

    # ── subtractCurrentStackFromTop ───────────────────────────────────────────

    def test_subtract_pushes_anti_join_of_current_from_top(self):
        v = self._registered_view()
        asyncio.run(self.mvc.pushStack(v, self.df2))   # df2 = rows where a > 1
        v.display_calls.clear()
        asyncio.run(self.mvc.subtractCurrentStackFromTop(v))
        # top (all 3 rows) minus current (2 rows where a>1) = 1 row where a==1
        result_df = v.display_calls[-1]['df']
        self.assertEqual(len(result_df), 1)
        self.assertEqual(result_df['a'][0], 1)

    def test_subtract_when_top_equals_current_is_noop(self):
        v = self._registered_view()
        asyncio.run(self.mvc.subtractCurrentStackFromTop(v))
        self.assertEqual(len(v.display_calls), 0)


# ===========================================================================
# Tier 1: _collect_leaves — pure Python helper
# ===========================================================================

class TestCollectLeaves(unittest.TestCase):

    def test_flat_list(self):
        self.assertEqual(_collect_leaves(['a', 'b', 'c']), ['a', 'b', 'c'])

    def test_one_level_nested(self):
        self.assertEqual(_collect_leaves([['a', 'b'], 'c']), ['a', 'b', 'c'])

    def test_deep_nested(self):
        self.assertEqual(_collect_leaves([[['a'], 'b'], ['c']]), ['a', 'b', 'c'])

    def test_single_item(self):
        self.assertEqual(_collect_leaves(['x']), ['x'])

    def test_empty(self):
        self.assertEqual(_collect_leaves([]), [])


# ===========================================================================
# Tier 1: _build_sketch_html — pure Python helper
# ===========================================================================

class TestBuildSketchHtml(unittest.TestCase):

    class FakePlot:
        def _repr_svg_(self):
            return '<svg id="test"/>'

    def setUp(self):
        self.p = self.FakePlot()

    def test_output_contains_flex(self):
        html = _build_sketch_html([self.p])
        self.assertIn('display:flex', html)

    def test_column_orientation(self):
        html = _build_sketch_html([self.p], orientation='column')
        self.assertIn('flex-direction:column', html)

    def test_row_orientation(self):
        html = _build_sketch_html([self.p], orientation='row')
        self.assertIn('flex-direction:row', html)

    def test_svg_content_included(self):
        html = _build_sketch_html([self.p])
        self.assertIn('<svg id="test"/>', html)

    def test_two_plots_both_included(self):
        html = _build_sketch_html([self.p, self.p])
        self.assertEqual(html.count('<svg id="test"/>'), 2)


# ===========================================================================
# Tier 1: _sketch_leaf_html resolution order — pure Python, no Panel required
#
# Interactive-only leaves (e.g. stack_controli) carry no static _repr_svg_;
# the sketch path resolves them through, in order: webgpu() → sketchHtml()
# snapshot → _repr_svg_() → labeled placeholder. These fakes exercise each tier
# without constructing a real Panel ReactiveHTML widget.
# ===========================================================================

class TestSketchLeafResolution(unittest.TestCase):

    class StaticLeaf:                       # static plot component
        def _repr_svg_(self):
            return '<svg id="static"/>'

    class SnapshotLeaf:                     # interactive-only widget with a snapshot
        wxh          = (160, 256)
        sketch_label = 'Snapshot'
        def sketchHtml(self, use_webgpu=False):
            return '<svg id="snapshot"/>'

    class DeferringLeaf:                    # sketchHtml() present but defers (None)
        wxh          = (120, 80)
        sketch_label = 'Deferring'
        def sketchHtml(self, use_webgpu=False):
            return None
        def _repr_svg_(self):
            return '<svg id="deferred"/>'

    class GpuLeaf:                          # leaf that renders via webgpu()
        wxh          = (200, 100)
        sketch_label = 'Gpu'
        def webgpu(self):
            return {'wxh': (200, 100), 'marker': 'gpu-canvas'}
        def sketchHtml(self, use_webgpu=False):
            return '<svg id="snapshot-not-gpu"/>'

    class BareLeaf:                         # no snapshot, no _repr_svg_ → placeholder
        wxh          = (120, 80)
        sketch_label = 'BareWidget'

    def test_static_leaf_uses_repr_svg(self):
        self.assertEqual(_sketch_leaf_html(self.StaticLeaf(), False), '<svg id="static"/>')

    def test_snapshot_leaf_uses_sketchHtml(self):
        self.assertEqual(_sketch_leaf_html(self.SnapshotLeaf(), False), '<svg id="snapshot"/>')

    def test_deferring_sketchHtml_falls_through_to_repr_svg(self):
        self.assertEqual(_sketch_leaf_html(self.DeferringLeaf(), False), '<svg id="deferred"/>')

    def test_bare_leaf_falls_back_to_placeholder(self):
        html = _sketch_leaf_html(self.BareLeaf(), False)
        self.assertIn('BareWidget', html)
        self.assertIn('interactive', html)
        self.assertIn('<svg', html)

    def test_placeholder_uses_widget_dimensions(self):
        html = _sketch_placeholder_html(self.BareLeaf())
        self.assertIn('width="120"', html)
        self.assertIn('height="80"', html)

    def test_placeholder_defaults_when_no_wxh(self):
        class NoSize:
            pass
        html = _sketch_placeholder_html(NoSize())
        self.assertIn('NoSize', html)          # falls back to class name as label
        self.assertIn('<svg', html)

    # ── webgpu tier: works the same for any leaf that exposes webgpu() ──────────

    def test_webgpu_render_used_when_requested_and_available(self):
        html = _sketch_leaf_html(self.GpuLeaf(), True)
        self.assertNotIn('snapshot-not-gpu', html)   # GPU tier wins over sketchHtml
        self.assertIn('gpu-canvas', html)            # payload made it into the render
        self.assertIn('<canvas', html)

    def test_webgpu_ignored_when_not_requested(self):
        # use_webgpu=False → GPU leaf still resolves via its sketchHtml() snapshot
        self.assertEqual(_sketch_leaf_html(self.GpuLeaf(), False), '<svg id="snapshot-not-gpu"/>')

    def test_webgpu_flag_harmless_for_non_gpu_leaf(self):
        # A snapshot-only widget ignores the flag and still returns its snapshot.
        self.assertEqual(_sketch_leaf_html(self.SnapshotLeaf(), True), '<svg id="snapshot"/>')


# ===========================================================================
# Tier 2: Panel-required — factory methods and layout construction
# ===========================================================================

@unittest.skipUnless(PANEL_AVAILABLE, 'panel not installed')
class TestP2SInteractiveMethods(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.p2s      = Polars2SVG()
        cls.df       = _make_df()
        cls.ldf      = _make_link_df()
        cls.pos      = _make_pos()
        cls.xyp_obj    = cls.p2s.xyp(cls.df, 'x', 'y')
        cls.histop_obj = cls.p2s.histop(cls.df, 'cat')
        cls.timep_obj  = cls.p2s.timep(cls.df, 'ts')
        cls.linkp_obj  = cls.p2s.linkp(cls.ldf, relationships=[('fm', 'to')], pos=cls.pos)

    # ── interactiveController ─────────────────────────────────────────────────

    def test_interactiveController_returns_instance(self):
        mvc = self.p2s.interactiveController()
        self.assertIsInstance(mvc, InteractionController)

    def test_interactiveController_returns_new_instance_each_call(self):
        self.assertIsNot(self.p2s.interactiveController(), self.p2s.interactiveController())

    # ── xypi ─────────────────────────────────────────────────────────────────

    def test_xypi_returns_reactive_html(self):
        self.assertIsInstance(self.p2s.xypi(self.xyp_obj), ReactiveHTML)

    def test_xypi_class_name_is_XYPI(self):
        self.assertEqual(type(self.p2s.xypi(self.xyp_obj)).__name__, 'XYPI')

    # ── histopi ───────────────────────────────────────────────────────────────

    def test_histopi_returns_reactive_html(self):
        self.assertIsInstance(self.p2s.histopi(self.histop_obj), ReactiveHTML)

    def test_histopi_class_name_is_HISTOPI(self):
        self.assertEqual(type(self.p2s.histopi(self.histop_obj)).__name__, 'HISTOPI')

    # ── timepi ────────────────────────────────────────────────────────────────

    def test_timepi_returns_reactive_html(self):
        self.assertIsInstance(self.p2s.timepi(self.timep_obj), ReactiveHTML)

    def test_timepi_class_name_is_TIMEPI(self):
        self.assertEqual(type(self.p2s.timepi(self.timep_obj)).__name__, 'TIMEPI')

    # ── linkpi ────────────────────────────────────────────────────────────────

    def test_linkpi_returns_reactive_html(self):
        self.assertIsInstance(self.p2s.linkpi(self.linkp_obj), ReactiveHTML)

    def test_linkpi_class_name_is_LINKPI(self):
        self.assertEqual(type(self.p2s.linkpi(self.linkp_obj)).__name__, 'LINKPI')

    # ── neighborhood layout operations ────────────────────────────────────────

    def test_neighborhood_layouts_registered(self):
        ctrl = self.p2s.linkpi(self.linkp_obj)
        for op in (ctrl.NEIGHBORHOOD_SPATIAL, ctrl.NEIGHBORHOOD_GRAPH):
            self.assertIn(op, ctrl.layout_operations)
            self.assertIn(op, ctrl._layout_registry)

    def test_neighborhood_spatial_op_applies_and_sets_background(self):
        ctrl = self.p2s.linkpi(self.linkp_obj)
        ln   = ctrl.dfs_layout[ctrl.df_level]
        g    = ctrl.graphs[ctrl.df_level]
        ok   = ctrl.__layoutOperation__(ctrl.NEIGHBORHOOD_SPATIAL, ln, g, set())
        self.assertTrue(ok)
        # Background is either a {label: shape} dict, or None when no clusters form.
        self.assertTrue(ctrl.layout_background is None or isinstance(ctrl.layout_background, dict))

    def test_neighborhood_graph_op_repositions_all_nodes(self):
        ctrl = self.p2s.linkpi(self.linkp_obj)
        ln   = ctrl.dfs_layout[ctrl.df_level]
        g    = ctrl.graphs[ctrl.df_level]
        ok   = ctrl.__layoutOperation__(ctrl.NEIGHBORHOOD_GRAPH, ln, g, set())
        self.assertTrue(ok)
        for n in g.nodes():
            self.assertIn(n, ln.pos)

    def test_neighborhood_ops_skip_when_selection_present(self):
        # Both neighborhood layouts are global re-layouts -> no-op with a selection.
        ctrl = self.p2s.linkpi(self.linkp_obj)
        ln   = ctrl.dfs_layout[ctrl.df_level]
        g    = ctrl.graphs[ctrl.df_level]
        self.assertFalse(ctrl.__layoutOperation__(ctrl.NEIGHBORHOOD_SPATIAL, ln, g, {'a'}))
        self.assertFalse(ctrl.__layoutOperation__(ctrl.NEIGHBORHOOD_GRAPH, ln, g, {'a'}))

    # ── collapsed-node contraction (exact xy match -> one representative) ──────

    def test_contraction_none_when_all_positions_distinct(self):
        # The triangle's three nodes sit at distinct locations -> no contraction.
        ctrl = self.p2s.linkpi(self.linkp_obj)
        ln   = ctrl.dfs_layout[ctrl.df_level]
        g    = ctrl.graphs[ctrl.df_level]
        # The shared linkp_obj's pos may have been mutated by another test, so
        # pin distinct locations here to isolate the "nothing collapses" case.
        ln.pos['a'], ln.pos['b'], ln.pos['c'] = (0.0, 0.0), (1.0, 0.0), (0.5, 0.866)
        self.assertIsNone(ctrl.__contractCollapsedGraph__(ln, g, set()))

    def test_contraction_merges_coincident_nodes_and_edges(self):
        # Stack 'b' exactly on top of 'a'. The contracted graph should hold one
        # fewer node, route 'a'/'b' edges through a single representative, and
        # drop the now-internal a-b edge.
        ctrl = self.p2s.linkpi(self.linkp_obj)
        ln   = ctrl.dfs_layout[ctrl.df_level]
        g    = ctrl.graphs[ctrl.df_level]
        ln.pos['b'] = (float(ln.pos['a'][0]), float(ln.pos['a'][1]))

        g_c, pos_c, sel_c, members = ctrl.__contractCollapsedGraph__(ln, g, {'b'})

        self.assertEqual(g_c.number_of_nodes(), g.number_of_nodes() - 1)
        # One representative covers both 'a' and 'b'.
        rep = next(r for r, m in members.items() if set(m) == {'a', 'b'})
        self.assertIn(rep, pos_c)
        # The a-b edge is internal to the group and must not appear as a self-loop.
        self.assertFalse(g_c.has_edge(rep, rep))
        # 'c' (connected to both a and b in the triangle) is still linked to the rep.
        self.assertTrue(g_c.has_edge(rep, 'c'))
        # Selecting a member selects the representative.
        self.assertIn(rep, sel_c)

    def test_contraction_sums_parallel_edge_weights(self):
        # Build a graph where two edges collapse onto the same rep-pair so their
        # weights are summed: a-c and b-c with a,b coincident -> rep-c weight 2.
        ctrl = self.p2s.linkpi(self.linkp_obj)
        ln   = ctrl.dfs_layout[ctrl.df_level]
        g    = ctrl.graphs[ctrl.df_level]
        ln.pos['b'] = (float(ln.pos['a'][0]), float(ln.pos['a'][1]))

        g_c, _, _, members = ctrl.__contractCollapsedGraph__(ln, g, set())
        rep = next(r for r, m in members.items() if set(m) == {'a', 'b'})
        # Triangle edges a-c and b-c each have weight 1 -> merged weight 2.
        self.assertEqual(g_c[rep]['c']['weight'], 2)

    def test_layout_keeps_collapsed_nodes_coincident(self):
        # After a full layout op, the two stacked nodes are placed as one and so
        # remain at an identical location (the group moved together).
        ctrl = self.p2s.linkpi(self.linkp_obj)
        ln   = ctrl.dfs_layout[ctrl.df_level]
        g    = ctrl.graphs[ctrl.df_level]
        ln.pos['b'] = (float(ln.pos['a'][0]), float(ln.pos['a'][1]))

        ok = ctrl.__layoutOperation__(ctrl.SPRING_NX, ln, g, set())
        self.assertTrue(ok)
        for n in g.nodes():
            self.assertIn(n, ln.pos)
        self.assertEqual(ln.pos['a'], ln.pos['b'])

    # ── panelizeSketch ────────────────────────────────────────────────────────

    def test_panelizeSketch_returns_html_pane(self):
        result = self.p2s.panelizeSketch([[self.xyp_obj]])
        self.assertEqual(type(result).__name__, 'HTML')

    def test_panelizeSketch_accepts_multi_row_layout(self):
        result = self.p2s.panelizeSketch([[self.xyp_obj], [self.histop_obj]])
        self.assertEqual(type(result).__name__, 'HTML')

    # xyp_obj is 256x256; stack_controli's default wxh (160, 256) is too short
    # to hold that icon plus the MLX/CUDA header and two skip labels, so these
    # give it a taller widget explicitly.
    _STACK_WXH_ = (160, 340)

    def test_sketch_includes_interactive_only_stack_control(self):
        # stack_controli is interactive-only (no static twin / _repr_svg_); the
        # sketch path must still represent it rather than raising.
        sc   = self.p2s.stack_controli(self.xyp_obj, wxh=self._STACK_WXH_)
        html = _build_sketch_html([[self.xyp_obj, sc]])
        self.assertIn('<svg', html)
        self.assertNotIn('interactive</text>', html)   # used live snapshot, not placeholder

    def test_stack_control_sketchHtml_returns_current_frame(self):
        sc = self.p2s.stack_controli(self.xyp_obj, wxh=self._STACK_WXH_)
        self.assertEqual(sc.sketchHtml(), sc.mod_inner)

    def test_panelizeSketch_with_interactive_only_leaf_returns_pane(self):
        sc     = self.p2s.stack_controli(self.xyp_obj, wxh=self._STACK_WXH_)
        result = self.p2s.panelizeSketch([[self.xyp_obj, sc]])
        self.assertEqual(type(result).__name__, 'HTML')

    # ── panelize ─────────────────────────────────────────────────────────────

    def test_panelize_returns_column(self):
        t  = self.p2s.timep(self.df, 'ts')
        xy = self.p2s.xyp(self.df, 'x', 'y')
        result = self.p2s.panelize([[t, xy]])
        self.assertEqual(type(result).__name__, 'Column')

    def test_panelize_two_row_layout_returns_column(self):
        t  = self.p2s.timep(self.df, 'ts')
        xy = self.p2s.xyp(self.df, 'x', 'y')
        result = self.p2s.panelize([[t], [xy]])
        self.assertEqual(type(result).__name__, 'Column')

    def test_panelize_accepts_pre_existing_reactive_html(self):
        ti = self.p2s.timepi(self.p2s.timep(self.df, 'ts'))
        xi = self.p2s.xypi(self.p2s.xyp(self.df, 'x', 'y'))
        result = self.p2s.panelize([[ti], [xi]])
        self.assertEqual(type(result).__name__, 'Column')

    def test_panelize_assigns_shared_mvc_to_all_views(self):
        ti = self.p2s.timepi(self.p2s.timep(self.df, 'ts'))
        xi = self.p2s.xypi(self.p2s.xyp(self.df, 'x', 'y'))
        self.p2s.panelize([[ti], [xi]])
        self.assertIsInstance(ti.mvc, InteractionController)
        self.assertIs(ti.mvc, xi.mvc)

    def test_panelize_mvc_stack_initialized_with_initial_df(self):
        ti = self.p2s.timepi(self.p2s.timep(self.df, 'ts'))
        xi = self.p2s.xypi(self.p2s.xyp(self.df, 'x', 'y'))
        self.p2s.panelize([[ti], [xi]])
        mvc = ti.mvc
        self.assertIn('default', mvc.stacks)
        self.assertGreater(len(mvc.stacks['default']['dfs']), 0)


@unittest.skipUnless(PANEL_AVAILABLE, 'panel not installed')
class TestLinkpiSelectEntitiesIntegerNodes(unittest.TestCase):
    """Regression: substring/exact search ignored node_labels when node IDs are integers.

    The guard  `if _node_ in all_nodes`  compared string label-keys (e.g. '10')
    against a set of integer graph nodes ({10, 20}), so the membership test was
    always False and no label-based match was ever added to the result.

    Fix: normalise via  str_to_node = {str(n): n for n in all_nodes}  so that
    '10' resolves to integer 10, and the integer is added to selected_entities.
    """

    def _make_ctrl(self, node_labels=None, int_nodes=True):
        p2s = Polars2SVG()
        if int_nodes:
            df = pl.DataFrame({'fm': [10, 20], 'to': [20, 10]})
        else:
            df = pl.DataFrame({'fm': ['a', 'b'], 'to': ['b', 'a']})
        lp = p2s.linkp(df, relationships=[('fm', 'to')], node_labels=node_labels)
        return p2s.linkpi(lp)

    # ── substring, integer nodes ──────────────────────────────────────────────

    def test_substring_finds_int_node_by_label(self):
        ctrl = self._make_ctrl({'10': 'bar', '20': 'foo'})
        ctrl.selectEntities('bar', method='substring')
        self.assertIn(10, ctrl.selected_entities)
        self.assertNotIn(20, ctrl.selected_entities)

    def test_substring_finds_both_int_nodes_when_both_labels_match(self):
        ctrl = self._make_ctrl({'10': 'xyz', '20': 'xyz'})
        ctrl.selectEntities('xyz', method='substring')
        self.assertIn(10, ctrl.selected_entities)
        self.assertIn(20, ctrl.selected_entities)

    def test_substring_case_insensitive_finds_int_node(self):
        ctrl = self._make_ctrl({'10': 'Bar', '20': 'Foo'})
        ctrl.selectEntities('BAR', method='substring', ignore_case=True)
        self.assertIn(10, ctrl.selected_entities)
        self.assertNotIn(20, ctrl.selected_entities)

    def test_substring_case_sensitive_finds_int_node(self):
        ctrl = self._make_ctrl({'10': 'Bar', '20': 'Foo'})
        ctrl.selectEntities('Bar', method='substring', ignore_case=False)
        self.assertIn(10, ctrl.selected_entities)

    def test_substring_case_sensitive_no_match_on_wrong_case(self):
        ctrl = self._make_ctrl({'10': 'Bar', '20': 'Foo'})
        ctrl.selectEntities('bar', method='substring', ignore_case=False)
        self.assertNotIn(10, ctrl.selected_entities)

    def test_substring_partial_label_finds_int_node(self):
        ctrl = self._make_ctrl({'10': 'bar', '20': 'foo'})
        ctrl.selectEntities('ba', method='substring')
        self.assertIn(10, ctrl.selected_entities)

    def test_substring_no_match_returns_empty(self):
        ctrl = self._make_ctrl({'10': 'bar', '20': 'foo'})
        ctrl.selectEntities('zzz', method='substring')
        self.assertEqual(len(ctrl.selected_entities), 0)

    def test_substring_selected_entities_are_integers(self):
        ctrl = self._make_ctrl({'10': 'bar', '20': 'foo'})
        ctrl.selectEntities('bar', method='substring')
        for node in ctrl.selected_entities:
            self.assertIsInstance(node, int,
                                  f'selected entity {node!r} should be int, got {type(node).__name__}')

    # ── exact, integer nodes ──────────────────────────────────────────────────

    def test_exact_finds_int_node_by_label(self):
        ctrl = self._make_ctrl({'10': 'bar', '20': 'foo'})
        ctrl.selectEntities('bar', method='exact')
        self.assertIn(10, ctrl.selected_entities)
        self.assertNotIn(20, ctrl.selected_entities)

    def test_exact_case_insensitive_finds_int_node(self):
        ctrl = self._make_ctrl({'10': 'Bar', '20': 'Foo'})
        ctrl.selectEntities('bar', method='exact', ignore_case=True)
        self.assertIn(10, ctrl.selected_entities)

    def test_exact_selected_entities_are_integers(self):
        ctrl = self._make_ctrl({'10': 'bar', '20': 'foo'})
        ctrl.selectEntities('bar', method='exact')
        for node in ctrl.selected_entities:
            self.assertIsInstance(node, int,
                                  f'selected entity {node!r} should be int, got {type(node).__name__}')

    # ── regression guard: string nodes still work ─────────────────────────────

    def test_substring_string_nodes_still_work(self):
        ctrl = self._make_ctrl({'a': 'alpha', 'b': 'beta'}, int_nodes=False)
        ctrl.selectEntities('alpha', method='substring')
        self.assertIn('a', ctrl.selected_entities)
        self.assertNotIn('b', ctrl.selected_entities)

    def test_exact_string_nodes_still_work(self):
        ctrl = self._make_ctrl({'a': 'alpha', 'b': 'beta'}, int_nodes=False)
        ctrl.selectEntities('alpha', method='exact')
        self.assertIn('a', ctrl.selected_entities)
        self.assertNotIn('b', ctrl.selected_entities)


# ---------------------------------------------------------------------------
# replaceStack() tests
# ---------------------------------------------------------------------------

class TestReplaceStack(unittest.TestCase):

    def _make_mvc_with_view(self):
        mvc  = InteractionController()
        df   = _make_df()
        mvc.addStack('default', df)

        class FakeView:
            def __init__(self):
                self.display_calls = []
            async def display(self, df, dfs, dfs_index):
                self.display_calls.append((df, dfs, dfs_index))

        view = FakeView()
        mvc.view_stack[id(view)] = 'default'
        mvc.view_refs[id(view)]  = view
        return mvc, view, df

    def test_replaceStack_resets_stack_to_single_entry(self):
        mvc, view, df = self._make_mvc_with_view()
        new_df = _make_df()
        asyncio.run(mvc.pushStack(view, new_df))
        self.assertEqual(mvc.stacks['default']['index'], 1)

        replace_df = _make_df()
        asyncio.run(mvc.replaceStack(view, replace_df))

        s = mvc.stacks['default']
        self.assertEqual(len(s['dfs']), 1)
        self.assertIs(s['dfs'][0], replace_df)
        self.assertEqual(s['index'], 0)

    def test_replaceStack_calls_display_on_regular_view(self):
        mvc, view, df = self._make_mvc_with_view()
        new_df = _make_df()
        asyncio.run(mvc.replaceStack(view, new_df))

        self.assertEqual(len(view.display_calls), 1)
        called_df, called_dfs, called_index = view.display_calls[0]
        self.assertIs(called_df, new_df)
        self.assertEqual(called_dfs, [new_df])
        self.assertEqual(called_index, 0)

    def test_replaceStack_calls_replaceBaseDataframe_on_linkpi_like_view(self):
        mvc  = InteractionController()
        df   = _make_df()
        mvc.addStack('default', df)

        class FakeLinkpiView:
            def __init__(self):
                self.replace_calls = []
            async def replaceBaseDataframe(self, df):
                self.replace_calls.append(df)

        view = FakeLinkpiView()
        mvc.view_stack[id(view)] = 'default'
        mvc.view_refs[id(view)]  = view

        new_df = _make_df()
        asyncio.run(mvc.replaceStack(view, new_df))

        self.assertEqual(len(view.replace_calls), 1)
        self.assertIs(view.replace_calls[0], new_df)

    def test_replaceStack_caller_not_in_view_stack_is_noop(self):
        mvc = InteractionController()
        mvc.addStack('default', _make_df())

        class FakeView:
            async def display(self, df, dfs, dfs_index):
                pass

        unregistered = FakeView()
        new_df = _make_df()
        # should not raise
        asyncio.run(mvc.replaceStack(unregistered, new_df))
        self.assertEqual(mvc.stacks['default']['index'], 0)

    def test_replaceStack_notifies_all_views_on_stack(self):
        mvc = InteractionController()
        df  = _make_df()
        mvc.addStack('default', df)

        calls = {}

        class FakeView:
            def __init__(self, name):
                self.name = name
                calls[name] = []
            async def display(self, df, dfs, dfs_index):
                calls[self.name].append(df)

        v1, v2, v3 = FakeView('v1'), FakeView('v2'), FakeView('v3')
        for v in (v1, v2, v3):
            mvc.view_stack[id(v)] = 'default'
            mvc.view_refs[id(v)]  = v

        new_df = _make_df()
        asyncio.run(mvc.replaceStack(v1, new_df))

        for name in ('v1', 'v2', 'v3'):
            self.assertEqual(len(calls[name]), 1, f'{name} should have been notified')
            self.assertIs(calls[name][0], new_df)


@unittest.skipUnless(PANEL_AVAILABLE, 'panel not installed')
class TestLINKPILayoutRegistry(unittest.TestCase):
    """Verify that the layout registry is built correctly on LINKPI instances."""

    @classmethod
    def setUpClass(cls):
        from polars2svg import Polars2SVG
        from polars2svg.interactive_controller import linkpi, _TFDP_AVAILABLE
        cls._tfdp_available = _TFDP_AVAILABLE
        p2s     = Polars2SVG()
        ldf     = _make_link_df()
        pos     = _make_pos()
        linkp   = p2s.linkp(ldf, relationships=[('fm', 'to')], pos=pos)
        cls.linkpi_instance = linkpi(linkp)

    def test_layout_registry_is_dict(self):
        self.assertIsInstance(self.linkpi_instance._layout_registry, dict)

    def test_registry_entries_are_callable(self):
        for key, handler in self.linkpi_instance._layout_registry.items():
            self.assertTrue(callable(handler), f'handler for {key!r} is not callable')

    def test_registry_covers_all_layout_operations(self):
        registry_keys = set(self.linkpi_instance._layout_registry.keys())
        for op in self.linkpi_instance.layout_operations:
            self.assertIn(op, registry_keys, f'{op!r} missing from registry')

    def test_tfdp_in_registry_iff_available(self):
        tfdp_key = self.linkpi_instance.TFDP_LAYOUT
        if self._tfdp_available:
            self.assertIn(tfdp_key, self.linkpi_instance._layout_registry)
        else:
            self.assertNotIn(tfdp_key, self.linkpi_instance._layout_registry)

    def test_tfdp_in_layout_operations_iff_available(self):
        tfdp_key = self.linkpi_instance.TFDP_LAYOUT
        if self._tfdp_available:
            self.assertIn(tfdp_key, self.linkpi_instance.layout_operations)
        else:
            self.assertNotIn(tfdp_key, self.linkpi_instance.layout_operations)

    def test_ncp_in_registry_iff_available(self):
        from polars2svg.interactive_controller import _NCP_AVAILABLE
        ncp_key = self.linkpi_instance.NCP_PACK
        if _NCP_AVAILABLE:
            self.assertIn(ncp_key, self.linkpi_instance._layout_registry)
            self.assertIn(ncp_key, self.linkpi_instance.layout_operations)
        else:
            self.assertNotIn(ncp_key, self.linkpi_instance._layout_registry)

    def test_ncp_handler_packs_visible_graph(self):
        from polars2svg.interactive_controller import _NCP_AVAILABLE
        if not _NCP_AVAILABLE:
            self.skipTest('ncp_layout not importable')
        ln      = self.linkpi_instance.dfs_layout[0]
        g       = self.linkpi_instance.graphs[0]
        handler = self.linkpi_instance._layout_registry[self.linkpi_instance.NCP_PACK]
        result  = handler(ln, g, set())
        self.assertIsInstance(result, dict)
        # every positioned node in the visible graph is packed
        self.assertEqual(set(result.keys()),
                         {n for n in g.nodes() if n in ln.pos})

    def test_spring_nx_handler_returns_dict_for_empty_selection(self):
        import networkx as nx
        ln      = self.linkpi_instance.dfs_layout[0]
        g       = self.linkpi_instance.graphs[0]
        handler = self.linkpi_instance._layout_registry[self.linkpi_instance.SPRING_NX]
        result  = handler(ln, g, set())
        self.assertIsInstance(result, dict)

    def test_spring_nx_handler_returns_none_with_selection(self):
        ln      = self.linkpi_instance.dfs_layout[0]
        g       = self.linkpi_instance.graphs[0]
        handler = self.linkpi_instance._layout_registry[self.linkpi_instance.SPRING_NX]
        result  = handler(ln, g, {'a'})
        self.assertIsNone(result)


@unittest.skipUnless(PANEL_AVAILABLE, 'panel not installed')
class TestLINKPIBackgroundCycling(unittest.TestCase):
    """Verify the 'b' background-cycle and the capture of layout-provided backgrounds."""

    def _make_ctrl(self):
        from polars2svg.interactive_controller import linkpi
        p2s   = Polars2SVG()
        linkp = p2s.linkp(_make_link_df(), relationships=[('fm', 'to')], pos=_make_pos())
        return linkpi(linkp)

    def _make_donut_ctrl(self):
        # Tree with a leaf-parent ('a' -> a1..a4) so the donut layout yields wedge cells.
        from polars2svg.interactive_controller import linkpi
        p2s = Polars2SVG()
        df  = pl.DataFrame({'fm': ['h', 'h', 'a', 'a', 'a', 'a', 'h'],
                            'to': ['a', 'b', 'a1', 'a2', 'a3', 'a4', 'c']})
        nodes = sorted(set(df['fm']) | set(df['to']))
        pos   = {n: [float(i % 3), float(i // 3)] for i, n in enumerate(nodes)}
        linkp = p2s.linkp(df, relationships=[('fm', 'to')], pos=pos)
        return linkpi(linkp)

    def _press_b(self, ctrl):
        async def _go():
            ctrl.key_op_finished = 'b'
            await ctrl.applyKeyOp(None)
        asyncio.run(_go())

    # ── initial state ───────────────────────────────────────────────────────
    def test_initial_background_state_is_zero(self):
        ctrl = self._make_ctrl()
        self.assertEqual(ctrl.background_state, 0)
        self.assertIsNone(ctrl.layout_background)

    def test_background_state_label_values(self):
        ctrl = self._make_ctrl()
        labels = []
        for s in (0, 1, 2):
            ctrl.background_state = s
            labels.append(ctrl.__backgroundStateLabel__())
        self.assertEqual(labels, ['no background', 'background', 'background + labels'])

    # ── registry handlers return backgrounds ─────────────────────────────────
    def test_circle_pack_handler_returns_pos_and_background(self):
        ctrl = self._make_ctrl()
        ln, g = ctrl.dfs_layout[0], ctrl.graphs[0]
        result = ctrl._layout_registry[ctrl.CIRCLE_PACK](ln, g, set())
        self.assertIsInstance(result, tuple)
        pos, shapes = result
        self.assertIsInstance(pos, dict)
        self.assertIsInstance(shapes, dict)
        self.assertGreater(len(shapes), 0)

    def test_donut_handler_returns_pos_and_cells(self):
        ctrl = self._make_donut_ctrl()
        ln, g = ctrl.dfs_layout[0], ctrl.graphs[0]
        result = ctrl._layout_registry[ctrl.HYPERTREE_DONUT](ln, g, set())
        self.assertIsInstance(result, tuple)
        pos, cells = result
        self.assertIsInstance(pos, dict)
        self.assertGreater(len(cells), 0)

    # ── __layoutOperation__ captures / clears the background ──────────────────
    def test_layout_operation_captures_background(self):
        ctrl = self._make_ctrl()
        ln, g = ctrl.dfs_layout[0], ctrl.graphs[0]
        ok = ctrl.__layoutOperation__(ctrl.CIRCLE_PACK, ln, g, set())
        self.assertTrue(ok)
        self.assertIsNotNone(ctrl.layout_background)
        self.assertGreater(len(ctrl.layout_background), 0)

    def test_layout_operation_clears_stale_background(self):
        ctrl = self._make_ctrl()
        ln, g = ctrl.dfs_layout[0], ctrl.graphs[0]
        ctrl.__layoutOperation__(ctrl.CIRCLE_PACK, ln, g, set())
        self.assertIsNotNone(ctrl.layout_background)
        # spring nx provides no background -> the stale one is cleared
        ctrl.__layoutOperation__(ctrl.SPRING_NX, ln, g, set())
        self.assertIsNone(ctrl.layout_background)

    # ── __applyBackgroundState__ drives the LinkP params ──────────────────────
    def test_apply_background_state_sets_and_clears_linkp(self):
        ctrl = self._make_ctrl()
        ln = ctrl.dfs_layout[0]
        ctrl.__layoutOperation__(ctrl.CIRCLE_PACK, ln, ctrl.graphs[0], set())

        ctrl.background_state = 1
        ctrl.__applyBackgroundState__(refresh=False)
        self.assertIsNotNone(ln.background)
        self.assertIsNone(ln.background_label_color)

        ctrl.background_state = 2
        ctrl.__applyBackgroundState__(refresh=False)
        self.assertIsNotNone(ln.background)
        self.assertIsNotNone(ln.background_label_color)

        ctrl.background_state = 0
        ctrl.__applyBackgroundState__(refresh=False)
        self.assertIsNone(ln.background)
        self.assertIsNone(ln.background_label_color)

    def test_apply_background_state_draws_nothing_without_layout_background(self):
        ctrl = self._make_ctrl()
        ln = ctrl.dfs_layout[0]
        ctrl.background_state = 1   # no layout background captured yet
        ctrl.__applyBackgroundState__(refresh=False)
        self.assertIsNone(ln.background)

    # ── the 'b' key cycles through the three states ───────────────────────────
    def test_b_key_cycles_three_states(self):
        ctrl = self._make_ctrl()
        ctrl.__layoutOperation__(ctrl.CIRCLE_PACK, ctrl.dfs_layout[0], ctrl.graphs[0], set())
        self.assertEqual(ctrl.background_state, 0)
        self._press_b(ctrl); self.assertEqual(ctrl.background_state, 1)
        self._press_b(ctrl); self.assertEqual(ctrl.background_state, 2)
        self._press_b(ctrl); self.assertEqual(ctrl.background_state, 0)

    def test_info_str_reports_background_state(self):
        ctrl = self._make_ctrl()
        ctrl.background_state = 2
        ctrl.__refreshView__(comp=False, all_ents=False, sel_ents=False)
        self.assertIn('background + labels', ctrl.info_str)


@unittest.skipUnless(PANEL_AVAILABLE, 'panel not installed')
class TestLINKPIPickerMenu(unittest.TestCase):
    """Verify the shift-W / shift-G picker-menu wiring: the module-level menu
    constants, the Python lists derived from them, the layout_mode /
    layout_operation commit path from JS, and the removal of the old blind
    cycling key ops."""

    _MENU_NAV_KEYS_ = {'j', 'k', 'W', 'G'}

    def _make_ctrl(self):
        from polars2svg.interactive_controller import linkpi
        p2s   = Polars2SVG()
        linkp = p2s.linkp(_make_link_df(), relationships=[('fm', 'to')], pos=_make_pos())
        return linkpi(linkp)

    def _press_key(self, ctrl, key):
        async def _go():
            ctrl.key_op_finished = key
            await ctrl.applyKeyOp(None)
        asyncio.run(_go())

    # ── menu constants ────────────────────────────────────────────────────────
    def test_mnemonics_unique_and_single_char(self):
        from polars2svg.interactive_controller import _LAYOUT_MODE_MENU_, _LAYOUT_OP_MENU_
        for menu in (_LAYOUT_MODE_MENU_, _LAYOUT_OP_MENU_):
            mnemonics = [m for m, _ in menu]
            self.assertEqual(len(mnemonics), len(set(mnemonics)))
            for m in mnemonics:
                self.assertEqual(len(m), 1)

    def test_mnemonics_avoid_menu_navigation_keys(self):
        from polars2svg.interactive_controller import _LAYOUT_MODE_MENU_, _LAYOUT_OP_MENU_
        for menu in (_LAYOUT_MODE_MENU_, _LAYOUT_OP_MENU_):
            for m, _ in menu:
                self.assertNotIn(m, self._MENU_NAV_KEYS_)

    def test_layout_lists_derive_from_menu_constants(self):
        from polars2svg.interactive_controller import _LAYOUT_MODE_MENU_, _LAYOUT_OP_MENU_
        ctrl = self._make_ctrl()
        self.assertEqual(ctrl.layout_modes,      [label for _, label in _LAYOUT_MODE_MENU_])
        self.assertEqual(ctrl.layout_operations, [label for _, label in _LAYOUT_OP_MENU_])

    def test_tfdp_in_op_menu_iff_available(self):
        from polars2svg.interactive_controller import _LAYOUT_OP_MENU_, _TFDP_AVAILABLE
        labels = [label for _, label in _LAYOUT_OP_MENU_]
        if _TFDP_AVAILABLE: self.assertIn('t-fdp', labels)
        else:               self.assertNotIn('t-fdp', labels)

    # ── the JS commit path: setting the params refreshes the info line ───────
    def test_layout_operation_commit_updates_info_str(self):
        ctrl = self._make_ctrl()
        ctrl.layout_operation = 'pivot mds'
        self.assertIn('pivot mds', ctrl.info_str)

    def test_layout_mode_commit_updates_info_str(self):
        ctrl = self._make_ctrl()
        ctrl.layout_mode = 'sunflower'
        self.assertIn('sunflower', ctrl.info_str)

    # ── old blind-cycling key ops are gone ────────────────────────────────────
    def test_old_cycling_key_ops_are_noops(self):
        ctrl = self._make_ctrl()
        op_before, mode_before = ctrl.layout_operation, ctrl.layout_mode
        for key in ('W', 'ctrl_shift_w', 'G', 'ctrl_shift_g'):
            self._press_key(ctrl, key)
            self.assertEqual(ctrl.layout_operation, op_before)
            self.assertEqual(ctrl.layout_mode,      mode_before)
            self.assertEqual(ctrl.key_op_finished,  '')

    # ── template / script wiring ──────────────────────────────────────────────
    def test_template_contains_picker_overlay(self):
        cls = type(self._make_ctrl())
        self.assertIn('pickermenu', cls._template)

    def test_render_script_seeds_menu_items(self):
        from polars2svg.interactive_controller import _LAYOUT_MODE_MENU_, _LAYOUT_OP_MENU_
        cls = type(self._make_ctrl())
        render = cls._scripts['render']
        self.assertIn('state.menu_items', render)
        for _, label in _LAYOUT_OP_MENU_ + _LAYOUT_MODE_MENU_:
            self.assertIn(label, render)

    def test_menu_scripts_exist(self):
        cls = type(self._make_ctrl())
        for script in ('menuOpen', 'menuRender', 'menuCommit', 'menuClose', 'menuArmTimer'):
            self.assertIn(script, cls._scripts)

    def test_keyboard_help_mentions_picker(self):
        cls = type(self._make_ctrl())
        self.assertIn('picker', cls._keyboard_commands_)


@unittest.skipUnless(PANEL_AVAILABLE, 'panel not installed')
class TestLINKPISizeCycleMenus(unittest.TestCase):
    """Verify the shift-L / shift-O / shift-P size & opacity cycle pickers and
    the 'l' link-shape picker: default selections, the JS commit path onto the
    LinkP, the hardcoded-number rule for size menus, and template wiring."""

    def _make_ctrl(self, **link_kwargs):
        from polars2svg.interactive_controller import linkpi
        p2s   = Polars2SVG()
        linkp = p2s.linkp(_make_link_df(), relationships=[('fm', 'to')],
                          pos=_make_pos(), **link_kwargs)
        return linkpi(linkp)

    def _press_key(self, ctrl, key):
        async def _go():
            ctrl.key_op_finished = key
            await ctrl.applyKeyOp(None)
        asyncio.run(_go())

    # ── default selections mirror the LinkP's constructor values ──────────────
    def test_default_choices_match_linkp(self):
        ctrl = self._make_ctrl()
        ln   = ctrl.dfs_layout[ctrl.df_level]
        self.assertEqual(ctrl.link_size_choice,    str(ln.link_size))
        self.assertEqual(ctrl.node_size_choice,    str(ln.node_size))
        self.assertEqual(ctrl.link_opacity_choice, str(int(round(ln.link_opacity * 100))))

    # ── the JS commit path: setting a *_choice param pushes onto the LinkP ─────
    def test_link_size_choice_named_commit(self):
        ctrl = self._make_ctrl()
        ctrl.link_size_choice = 'large'
        for ln in ctrl.dfs_layout:
            self.assertEqual(ln.link_size, 'large')

    def test_node_size_choice_vary_commit(self):
        ctrl = self._make_ctrl()
        ctrl.node_size_choice = 'vary'
        for ln in ctrl.dfs_layout:
            self.assertEqual(ln.node_size, 'vary')

    def test_link_opacity_choice_commit_converts_to_fraction(self):
        ctrl = self._make_ctrl()
        ctrl.link_opacity_choice = '40'
        for ln in ctrl.dfs_layout:
            self.assertAlmostEqual(ln.link_opacity, 0.4)

    def test_numeric_size_label_commit_converts_to_float(self):
        ctrl = self._make_ctrl()
        ctrl.link_size_choice = '2'
        for ln in ctrl.dfs_layout:
            self.assertEqual(ln.link_size, 2.0)

    def test_none_label_commits_real_none_and_renders(self):
        # 'none' round-trips to a real None (links / nodes not drawn) and the
        # view still re-renders to a valid SVG string.
        ctrl = self._make_ctrl()
        ctrl.link_size_choice = 'none'
        ctrl.node_size_choice = 'none'
        for ln in ctrl.dfs_layout:
            self.assertIsNone(ln.link_size)
            self.assertIsNone(ln.node_size)
        self.assertTrue(ctrl.dfs_layout[ctrl.df_level].renderSVG().startswith('<svg'))

    def test_none_size_is_current_when_linkp_created_with_none(self):
        ctrl = self._make_ctrl(link_size=None, node_size=None)
        self.assertEqual(ctrl.link_size_choice, 'none')
        self.assertEqual(ctrl.node_size_choice, 'none')

    # ── hardcoded-number rule: only a user-supplied number joins the cycle ─────
    def test_hardcoded_number_becomes_current_and_is_in_menu(self):
        ctrl   = self._make_ctrl(link_size=2, node_size=4.5)
        self.assertEqual(ctrl.link_size_choice, '2')
        self.assertEqual(ctrl.node_size_choice, '4.5')
        render = type(ctrl)._scripts['render']
        self.assertIn('"link_size"', render)
        # the hardcoded values appear as menu labels
        self.assertIn('"2"', render)
        self.assertIn('"4.5"', render)

    def test_named_sizes_do_not_inject_arbitrary_numbers(self):
        # With named sizes only, the size menus carry no numeric labels.
        import json, re
        ctrl   = self._make_ctrl()  # defaults: link 'small', node 'medium'
        render = type(ctrl)._scripts['render']
        m      = re.search(r'state\.menu_items = (\{.*?\});', render, re.S)
        self.assertIsNotNone(m)
        items  = json.loads(m.group(1))
        for kind in ('link_size', 'node_size'):
            labels = [lbl for _, lbl in items[kind]]
            self.assertEqual(labels, ['none', 'nil', 'small', 'medium', 'large', 'vary'])

    def test_opacity_menu_is_ten_to_hundred_grid(self):
        import json, re
        ctrl   = self._make_ctrl()
        render = type(ctrl)._scripts['render']
        items  = json.loads(re.search(r'state\.menu_items = (\{.*?\});', render, re.S).group(1))
        labels = [lbl for _, lbl in items['link_opacity']]
        self.assertEqual(labels, [str(p) for p in range(10, 101, 10)])

    # ── 'l' opens the link-shape picker (line | curve | flowmap) ──────────────
    def test_link_shape_choice_commit(self):
        ctrl = self._make_ctrl(link_shape='line')
        for shape in ('curve', 'flowmap', 'line'):
            ctrl.link_shape_choice = shape
            for ln in ctrl.dfs_layout:
                self.assertEqual(ln.link_shape, shape)

    def test_link_shape_default_choice_matches_linkp(self):
        ctrl = self._make_ctrl(link_shape='curve')
        self.assertEqual(ctrl.link_shape_choice, 'curve')

    def test_link_shape_menu_lists_all_shapes(self):
        import json, re
        ctrl   = self._make_ctrl()
        render = type(ctrl)._scripts['render']
        items  = json.loads(re.search(r'state\.menu_items = (\{.*?\});', render, re.S).group(1))
        labels = [lbl for _, lbl in items['link_shape']]
        self.assertEqual(labels, ['line', 'curve', 'flowmap'])

    # ── 'a' toggles link arrows on and off ────────────────────────────────────
    def test_a_key_toggles_link_arrows(self):
        ctrl = self._make_ctrl()
        self._press_key(ctrl, 'a')
        for ln in ctrl.dfs_layout:
            self.assertTrue(ln.link_arrows)
        self._press_key(ctrl, 'a')
        for ln in ctrl.dfs_layout:
            self.assertFalse(ln.link_arrows)

    def test_keyboard_help_mentions_link_arrows(self):
        self.assertIn('link arrows', type(self._make_ctrl())._keyboard_commands_)

    def test_old_l_key_op_is_noop(self):
        # 'l' now opens the picker menu in JS; the Python key op no longer
        # blind-toggles the shape
        ctrl   = self._make_ctrl(link_shape='line')
        before = ctrl.dfs_layout[ctrl.df_level].link_shape
        self._press_key(ctrl, 'l')
        self.assertEqual(ctrl.dfs_layout[ctrl.df_level].link_shape, before)

    # ── template / script wiring ──────────────────────────────────────────────
    def test_render_script_seeds_new_menu_kinds(self):
        render = type(self._make_ctrl())._scripts['render']
        for kind in ('"link_size"', '"link_opacity"', '"node_size"', '"link_shape"'):
            self.assertIn(kind, render)

    def test_commit_script_handles_new_kinds(self):
        commit = type(self._make_ctrl())._scripts['menuCommit']
        for field in ('link_size_choice', 'link_opacity_choice', 'node_size_choice', 'link_shape_choice'):
            self.assertIn(field, commit)

    def test_keyboard_help_mentions_size_cycles(self):
        cmds = type(self._make_ctrl())._keyboard_commands_
        self.assertIn('shift-l', cmds)
        self.assertIn('shift-o', cmds)
        self.assertIn('shift-p', cmds)
        self.assertIn('link shape', cmds)


@unittest.skipUnless(PANEL_AVAILABLE, 'panel not installed')
class TestLINKPICopyToClipboard(unittest.TestCase):
    """ctrl-C copies the current selection to the clipboard via 'pyperclip',
    a required top-level import in interactive_controller.py (part of the
    `interactive` extra, same as panel/param)."""

    def _make_ctrl(self):
        from polars2svg.interactive_controller import linkpi
        p2s   = Polars2SVG()
        linkp = p2s.linkp(_make_link_df(), relationships=[('fm', 'to')], pos=_make_pos())
        return linkpi(linkp)

    def _press_ctrl_c(self, ctrl):
        async def _go():
            ctrl.ctrlkey          = True
            ctrl.key_op_finished  = 'c'
            await ctrl.applyKeyOp(None)
        asyncio.run(_go())

    def test_no_error_when_nothing_selected(self):
        # The clipboard path is only entered when there's a selection; with
        # none, ctrl-C should be a no-op regardless of pyperclip availability.
        ctrl = self._make_ctrl()
        ctrl.selected_entities = set()
        self._press_ctrl_c(ctrl)  # should not raise


@unittest.skipUnless(PANEL_AVAILABLE, 'panel not installed')
class TestLINKPITimingMarksCycle(unittest.TestCase):
    """The 'a' key cycles arrows x timing marks when a time field is available, and
    toggles arrows only otherwise."""

    def _ctrl(self, df, **kw):
        from polars2svg.interactive_controller import linkpi
        p2s   = Polars2SVG()
        linkp = p2s.linkp(df, relationships=[('fm', 'to')], pos=_make_pos(), **kw)
        return linkpi(linkp)

    def _df_one_ts(self):
        return pl.DataFrame({'fm': ['a', 'b', 'c'], 'to': ['b', 'c', 'a'],
                             'ts': [datetime(2024, 1, d) for d in (1, 2, 3)]})

    def _df_two_ts(self):
        return pl.DataFrame({'fm': ['a', 'b', 'c'], 'to': ['b', 'c', 'a'],
                             'ts':  [datetime(2024, 1, d) for d in (1, 2, 3)],
                             'ts2': [datetime(2024, 2, d) for d in (1, 2, 3)]})

    def _press_a(self, ctrl):
        async def _go():
            ctrl.key_op_finished = 'a'
            await ctrl.applyKeyOp(None)
        asyncio.run(_go())

    def _state(self, ctrl):
        ln = ctrl.dfs_layout[0]
        return (bool(ln.link_arrows), getattr(ln, '_time_field_', None) is not None)

    # ── availability detection ──────────────────────────────────────────────
    def test_auto_detect_single_date_column(self):
        self.assertEqual(self._ctrl(self._df_one_ts())._timing_time_, 'ts')

    def test_ambiguous_two_date_columns_unavailable(self):
        self.assertIsNone(self._ctrl(self._df_two_ts())._timing_time_)

    def test_no_date_columns_unavailable(self):
        self.assertIsNone(self._ctrl(_make_link_df())._timing_time_)

    def test_explicit_time_used_over_autodetect(self):
        # an explicit time= wins even when the data has several date columns
        self.assertEqual(self._ctrl(self._df_two_ts(), time='ts2')._timing_time_, 'ts2')

    # ── the four-state cycle ────────────────────────────────────────────────
    def test_full_cycle(self):
        ctrl = self._ctrl(self._df_one_ts())
        self.assertEqual(self._state(ctrl), (False, False))                       # initial
        self._press_a(ctrl); self.assertEqual(self._state(ctrl), (True, False))   # arrows
        self._press_a(ctrl); self.assertEqual(self._state(ctrl), (True, True))    # arrows + marks
        self._press_a(ctrl); self.assertEqual(self._state(ctrl), (False, True))   # marks
        self._press_a(ctrl); self.assertEqual(self._state(ctrl), (False, False))  # wrap

    def test_marks_appear_in_svg_when_on(self):
        ctrl = self._ctrl(self._df_one_ts())
        self._press_a(ctrl)                                     # arrows, no marks
        self.assertNotIn('stroke-width="1.5"', ctrl.mod_inner)
        self._press_a(ctrl)                                     # arrows + marks
        self.assertIn('stroke-width="1.5"', ctrl.mod_inner)

    def test_marks_start_on_when_time_configured(self):
        ctrl = self._ctrl(self._df_one_ts(), time='ts')
        self.assertEqual(self._state(ctrl), (False, True))     # constructed with marks on

    # ── arrows-only fallback ────────────────────────────────────────────────
    def test_arrows_only_toggle_without_time(self):
        ctrl = self._ctrl(_make_link_df())
        self.assertEqual(self._state(ctrl), (False, False))
        self._press_a(ctrl); self.assertEqual(self._state(ctrl), (True, False))
        self._press_a(ctrl); self.assertEqual(self._state(ctrl), (False, False))  # never enables marks
        self.assertNotIn('stroke-width="1.5"', ctrl.mod_inner)


if __name__ == '__main__':
    unittest.main()
