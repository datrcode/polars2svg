import asyncio
import logging
import unittest
from datetime import datetime, timedelta

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


class _UnfilteredLoggerMixin:
    """Strips the process-global OnceFilter for the duration of each test.

    Polars2SVG installs a per-message OnceFilter on 'polars2svg_logger', so the second
    and later emissions of one message are dropped -- across tests, since the logger is
    process-global. Any test that asserts on a warning another test also triggers has to
    take the filter off first."""

    def setUp(self):
        super().setUp()
        self._logger        = logging.getLogger('polars2svg_logger')
        self._saved_filters = list(self._logger.filters)
        for _f_ in self._saved_filters:
            self._logger.removeFilter(_f_)

    def tearDown(self):
        for _f_ in list(self._logger.filters):
            self._logger.removeFilter(_f_)
        for _f_ in self._saved_filters:
            self._logger.addFilter(_f_)
        super().tearDown()


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

    def test_display_identity_guard_rerenders_on_id_collision(self):
        # The render cache keys on id(df); a dropped frame's id can be reused by a
        # later dataframe. Entries are (df, plot) and hits are identity-guarded, so
        # a slot holding another object's render must be re-rendered, not served.
        xi = self.p2s.xypi(self.p2s.xyp(self.df, 'x', 'y'))
        df_orig = next(iter(xi._cache_.values()))[0]
        df2 = df_orig.head(3)
        asyncio.run(xi.display(df2, [df_orig, df2], 1))
        xi._cache_[id(df2)] = (object(), 'STALE_PLOT')     # simulate an id() collision
        asyncio.run(xi.display(df2, [df_orig, df2], 1))
        self.assertIs(xi._cache_[id(df2)][0], df2)          # slot rebound to the real df
        self.assertNotEqual(xi._cache_[id(df2)][1], 'STALE_PLOT')

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

    def test_linkpi_display_reconciles_internal_stack_on_collapse(self):
        # A stack-control 'collapse' replaces the MVC stack [base, A, B, C] with
        # [base, C]; index 1 now holds C, not A. LINKPI keeps its own level-indexed
        # stack, so display must reconcile it rather than walk to its stale level 1.
        ldf  = pl.DataFrame({'fm': ['a', 'b', 'c', 'd', 'e'],
                             'to': ['b', 'c', 'd', 'e', 'a']})
        ctrl = self.p2s.linkpi(self.p2s.linkp(ldf, relationships=[('fm', 'to')]))
        base = ctrl.dfs[0]
        A, B, C = base.head(4), base.head(3), base.head(2)
        asyncio.run(ctrl.display(A, [base, A], 1))
        asyncio.run(ctrl.display(B, [base, A, B], 2))
        asyncio.run(ctrl.display(C, [base, A, B, C], 3))
        self.assertEqual(ctrl.df_level, 3)
        asyncio.run(ctrl.display(C, [base, C], 1))                  # collapse
        self.assertEqual(len(ctrl.dfs), 2)
        self.assertIs(ctrl.dfs[0], base)
        self.assertIs(ctrl.dfs[ctrl.df_level], C)                  # shows C, not the stale A
        self.assertIsNot(ctrl.dfs[ctrl.df_level], A)
        self.assertIs(ctrl.dfs_layout[ctrl.df_level].df_orig, C)   # C's layout reused (positions kept)

    def test_linkpi_display_normal_navigation_still_uses_level_walk(self):
        # Append/truncate navigation is identity-prefix compatible, so it keeps
        # using the cheap level-walk (reconciliation only kicks in on divergence).
        ldf  = pl.DataFrame({'fm': ['a', 'b', 'c', 'd', 'e'],
                             'to': ['b', 'c', 'd', 'e', 'a']})
        ctrl = self.p2s.linkpi(self.p2s.linkp(ldf, relationships=[('fm', 'to')]))
        base = ctrl.dfs[0]
        A = base.head(3)
        asyncio.run(ctrl.display(A, [base, A], 1))
        self.assertIs(ctrl.dfs[ctrl.df_level], A)
        asyncio.run(ctrl.display(base, [base, A], 0))              # pop back to base
        self.assertEqual(ctrl.df_level, 0)
        self.assertIs(ctrl.dfs[ctrl.df_level], base)
        self.assertEqual(len(ctrl.dfs), 2)                        # internal stack intact

    # ── neighborhood layout operations ────────────────────────────────────────

    def test_neighborhood_graph_layout_registered(self):
        # 'graph' mode repositions nodes, so its cells ARE its output and it stays a
        # layout operation.  'spatial' mode moves nothing and is a background
        # operation instead -- see test_neighborhood_spatial_is_a_background_op.
        ctrl = self.p2s.linkpi(self.linkp_obj)
        self.assertIn(ctrl.NEIGHBORHOOD_GRAPH, ctrl.layout_operations)
        self.assertIn(ctrl.NEIGHBORHOOD_GRAPH, ctrl._layout_registry)

    def test_neighborhood_spatial_is_a_background_op_not_a_layout(self):
        ctrl = self.p2s.linkpi(self.linkp_obj)
        self.assertNotIn(ctrl.NEIGHBORHOOD_SPATIAL, ctrl.layout_operations)
        self.assertNotIn(ctrl.NEIGHBORHOOD_SPATIAL, ctrl._layout_registry)
        self.assertIn(ctrl.NEIGHBORHOOD_SPATIAL, ctrl.background_operations)
        self.assertIn(ctrl.NEIGHBORHOOD_SPATIAL, ctrl._background_registry)

    def test_neighborhood_spatial_sets_a_contextual_background(self):
        ctrl   = self.p2s.linkpi(self.linkp_obj)
        ln     = ctrl.dfs_layout[ctrl.df_level]
        before = {k: tuple(v) for k, v in ln.pos.items()}
        undo   = len(ctrl.previous_layouts)
        self.assertTrue(ctrl.applyBackgroundOperation(ctrl.NEIGHBORHOOD_SPATIAL))
        # Background is either a {label: shape} dict, or None when no clusters form.
        self.assertTrue(ctrl.layout_background is None or isinstance(ctrl.layout_background, dict))
        if ctrl.layout_background is not None:
            self.assertEqual(ctrl.background_provenance, 'context')
        # It clusters the layout already on screen: nothing moves, so it is not an
        # undo step either.
        self.assertEqual({k: tuple(v) for k, v in ln.pos.items()}, before)
        self.assertEqual(len(ctrl.previous_layouts), undo)

    def test_neighborhood_graph_op_repositions_all_nodes(self):
        ctrl = self.p2s.linkpi(self.linkp_obj)
        ln   = ctrl.dfs_layout[ctrl.df_level]
        g    = ctrl.graphs[ctrl.df_level]
        ok   = ctrl.__layoutOperation__(ctrl.NEIGHBORHOOD_GRAPH, ln, g, set())
        self.assertTrue(ok)
        for n in g.nodes():
            self.assertIn(n, ln.pos)

    def test_neighborhood_graph_op_skips_when_selection_present(self):
        # 'graph' mode is a global re-layout -> no-op with a selection.  'spatial' is
        # no longer a layout op at all, and as a background it describes the whole
        # embedding, so it runs regardless of the selection.
        ctrl = self.p2s.linkpi(self.linkp_obj)
        ln   = ctrl.dfs_layout[ctrl.df_level]
        g    = ctrl.graphs[ctrl.df_level]
        self.assertFalse(ctrl.__layoutOperation__(ctrl.NEIGHBORHOOD_GRAPH, ln, g, {'a'}))
        ctrl.selected_entities = {'a'}
        self.assertTrue(ctrl.applyBackgroundOperation(ctrl.NEIGHBORHOOD_SPATIAL))

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

    def test_registry_entries_are_records_with_a_callable_handler(self):
        from polars2svg.interactive_treatments import RegistryEntry
        for key, entry in self.linkpi_instance._layout_registry.items():
            self.assertIsInstance(entry, RegistryEntry, f'{key!r} is not a RegistryEntry')
            self.assertTrue(callable(entry.handler), f'handler for {key!r} is not callable')

    def test_every_layout_entry_declares_a_treatment(self):
        # The point of the record: a layout added later cannot be silently unguarded.
        # Whoever adds it has to say whether it can be interrupted, whether it is safe to
        # move to a subprocess, and what its quality-for-time knobs are.
        from polars2svg.interactive_treatments import Treatment
        for key, entry in self.linkpi_instance._layout_registry.items():
            self.assertIsInstance(entry.treatment, Treatment, f'{key!r} declares no treatment')
            self.assertIsInstance(entry.treatment.truncatable, bool)
            self.assertIsInstance(entry.treatment.killable,    bool)
            self.assertIsInstance(entry.treatment.levers,      tuple)

    def test_every_background_entry_declares_a_treatment(self):
        from polars2svg.interactive_treatments import RegistryEntry, Treatment
        for key, entry in self.linkpi_instance._background_registry.items():
            self.assertIsInstance(entry, RegistryEntry, f'{key!r} is not a RegistryEntry')
            self.assertIsInstance(entry.treatment, Treatment, f'{key!r} declares no treatment')

    def test_flow_field_backgrounds_are_not_declared_killable(self):
        # They stash the producer instance on the controller for summary()/tests, which
        # would not survive a process hop -- so they must never be declared safe to move
        # into one.  Asserted rather than left to a comment because the failure would be
        # silent state loss rather than a crash.
        for _label_, _entry_ in self.linkpi_instance._background_registry.items():
            if _label_.startswith('flow field'):
                self.assertFalse(_entry_.treatment.killable,
                                 f'{_label_} stashes controller state; it is not killable')

    def test_declared_levers_name_real_parameters(self):
        # A lever that is not actually a parameter of the underlying callable is a lie the
        # T2 work would later trip over.
        import inspect, networkx as nx
        _known_ = {
            'spring nx': nx.spring_layout,
        }
        for _label_, _fn_ in _known_.items():
            _entry_ = self.linkpi_instance._layout_registry.get(_label_)
            if _entry_ is None:
                continue
            _params_ = set(inspect.signature(_fn_).parameters)
            for _lever_ in _entry_.treatment.levers:
                self.assertIn(_lever_, _params_,
                              f'{_label_} declares lever {_lever_!r}, not a parameter of {_fn_}')

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
        handler = self.linkpi_instance._layout_registry[self.linkpi_instance.NCP_PACK].handler
        result  = handler(ln, g, set())
        self.assertIsInstance(result, dict)
        # every positioned node in the visible graph is packed
        self.assertEqual(set(result.keys()),
                         {n for n in g.nodes() if n in ln.pos})

    def test_spring_nx_handler_returns_dict_for_empty_selection(self):
        import networkx as nx
        ln      = self.linkpi_instance.dfs_layout[0]
        g       = self.linkpi_instance.graphs[0]
        handler = self.linkpi_instance._layout_registry[self.linkpi_instance.SPRING_NX].handler
        result  = handler(ln, g, set())
        self.assertIsInstance(result, dict)

    def test_spring_nx_handler_moves_selection_and_pins_the_rest(self):
        # 0.2.0: spring nx absorbed the job the removed PolarsForceDirectedLayout existed
        # for -- with a selection it lays out those nodes and pins the rest via networkx's
        # fixed=, which also suppresses the rescale that would otherwise move them.
        ln      = self.linkpi_instance.dfs_layout[0]
        g       = self.linkpi_instance.graphs[0]
        handler = self.linkpi_instance._layout_registry[self.linkpi_instance.SPRING_NX].handler
        before  = {_n_: tuple(ln.pos[_n_]) for _n_ in g.nodes()}
        result  = handler(ln, g, {'a'})
        self.assertIsInstance(result, dict)
        for _n_ in g.nodes():
            if _n_ == 'a':
                continue
            self.assertAlmostEqual(result[_n_][0], before[_n_][0], places=6,
                                   msg=f'{_n_} should have been pinned')
            self.assertAlmostEqual(result[_n_][1], before[_n_][1], places=6,
                                   msg=f'{_n_} should have been pinned')

    def test_spring_nx_handler_declines_when_selection_is_not_in_this_graph(self):
        # A selection can carry edge entities and nodes from other stack levels; when none
        # of it is in this graph there is nothing to move, and laying everything out would
        # be a surprise.
        ln      = self.linkpi_instance.dfs_layout[0]
        g       = self.linkpi_instance.graphs[0]
        handler = self.linkpi_instance._layout_registry[self.linkpi_instance.SPRING_NX].handler
        self.assertIsNone(handler(ln, g, {'not-a-node-in-this-graph'}))

    def test_removed_layout_operations_are_absent(self):
        # 0.2.0 removed both from linkpi (CHANGELOG). They were the two slowest operations
        # in the registry and neither was the best implementation of what it did.
        from polars2svg.interactive_controller import _LAYOUT_OP_MENU_
        _labels_ = [_l_ for _, _l_ in _LAYOUT_OP_MENU_]
        for _gone_ in ('force directed', 'convey proximity'):
            self.assertNotIn(_gone_, self.linkpi_instance._layout_registry)
            self.assertNotIn(_gone_, _labels_)
            self.assertNotIn(_gone_, self.linkpi_instance.layout_operations)


@unittest.skipUnless(PANEL_AVAILABLE, 'panel not installed')
class TestLINKPIConfirmGate(unittest.TestCase):
    """T4c: an operation declared to need asking about at this size asks once, then runs.

    The thresholds that ship are far above anything a test graph reaches, so these lower
    them on the entry under test rather than building a six-thousand-node graph.  That is
    also the honest shape of the feature: the threshold is a declaration, so a test that
    declares a different one is exercising the mechanism, not faking it.
    """

    def _ctrl(self):
        from polars2svg import Polars2SVG
        from polars2svg.interactive_controller import linkpi
        p2s = Polars2SVG()
        return linkpi(p2s.linkp(_make_link_df(), relationships=[('fm', 'to')], pos=_make_pos()))

    def _lower_threshold(self, ctrl, op, limit=1):
        from polars2svg.interactive_treatments import RegistryEntry, Treatment
        _h_ = ctrl._layout_registry[op].handler
        ctrl._layout_registry[op] = RegistryEntry(_h_, Treatment(confirm_above=limit))
        return ctrl

    def test_below_threshold_runs_without_asking(self):
        ctrl = self._ctrl()
        self.assertTrue(ctrl.apply_layout_operation(ctrl.SPRING_NX))
        self.assertIsNone(ctrl._confirm_armed_)

    def test_over_threshold_refuses_the_first_time(self):
        ctrl = self._lower_threshold(self._ctrl(), 'spring nx')
        before = dict(ctrl.dfs_layout[0].pos)
        self.assertEqual(ctrl.apply_layout_operation(ctrl.SPRING_NX), {})
        self.assertEqual(dict(ctrl.dfs_layout[0].pos), before)
        self.assertEqual(ctrl._confirm_armed_, 'spring nx')

    def test_repeating_the_same_operation_runs_it(self):
        ctrl = self._lower_threshold(self._ctrl(), 'spring nx')
        ctrl.apply_layout_operation(ctrl.SPRING_NX)
        self.assertTrue(ctrl.apply_layout_operation(ctrl.SPRING_NX))
        self.assertIsNone(ctrl._confirm_armed_, 'the confirmation should be spent')

    def test_a_different_operation_disarms(self):
        # A confirmation armed for one operation must never be redeemable by another.
        ctrl = self._lower_threshold(self._ctrl(), 'spring nx')
        ctrl.apply_layout_operation(ctrl.SPRING_NX)
        self.assertEqual(ctrl._confirm_armed_, 'spring nx')
        ctrl.apply_layout_operation(ctrl.HYPERTREE)
        self.assertIsNone(ctrl._confirm_armed_)

    def test_a_refused_operation_is_not_an_undo_step(self):
        ctrl = self._lower_threshold(self._ctrl(), 'spring nx')
        ctrl.apply_layout_operation(ctrl.SPRING_NX)
        self.assertEqual(len(ctrl.previous_layouts), 0)

    def test_a_declined_operation_is_not_an_undo_step(self):
        # Same hygiene for the older path: pivot mds declines outright with a selection.
        ctrl = self._ctrl()
        ctrl.selected_entities = {'a'}
        ctrl.apply_layout_operation(ctrl.PIVOT_MDS)
        self.assertEqual(len(ctrl.previous_layouts), 0)

    def test_a_real_layout_is_still_an_undo_step(self):
        ctrl = self._ctrl()
        ctrl.apply_layout_operation(ctrl.SPRING_NX)
        self.assertEqual(len(ctrl.previous_layouts), 1)

    def test_the_message_names_the_size_and_never_a_duration(self):
        # D6: the gate has no honest seconds estimate to quote, so it must not invent one.
        ctrl = self._lower_threshold(self._ctrl(), 'spring nx')
        ctrl.apply_layout_operation(ctrl.SPRING_NX)
        _msg_ = ctrl.animation_inner + ' ' + (ctrl._last_cost_note_ or '')
        self.assertIn('nodes', _msg_)
        for _duration_word_ in ('second', 'minute', 'hour', 'estimated', '~'):
            self.assertNotIn(_duration_word_, _msg_)

    def test_the_note_reaches_info_str(self):
        ctrl = self._lower_threshold(self._ctrl(), 'spring nx')
        ctrl.apply_layout_operation(ctrl.SPRING_NX)
        ctrl.__refreshView__(comp=False, all_ents=False, sel_ents=False)
        self.assertIn('awaiting confirm', ctrl.info_str)

    def test_spring_nx_spends_full_iterations_on_a_small_graph(self):
        import networkx as nx, unittest.mock as mock
        ctrl = self._ctrl()
        with mock.patch.object(nx, 'spring_layout', wraps=nx.spring_layout) as _sl_:
            ctrl.apply_layout_operation(ctrl.SPRING_NX)
        self.assertEqual(_sl_.call_args.kwargs['iterations'], 50)
        self.assertIsNone(ctrl._last_cost_note_)

    def test_spring_nx_spends_fewer_iterations_on_a_large_graph_and_says_so(self):
        # T2: networkx exposes no per-iteration hook, so the only lever is chosen before
        # the call.  Faked via the node count rather than by building a real 4000-node
        # graph -- the handler reads g.number_of_nodes() and nothing else about size.
        import networkx as nx, unittest.mock as mock
        ctrl = self._ctrl()
        _g_  = ctrl.graphs[ctrl.df_level]
        with mock.patch.object(_g_, 'number_of_nodes', return_value=4000), \
             mock.patch.object(nx, 'spring_layout', wraps=nx.spring_layout) as _sl_:
            ctrl.apply_layout_operation(ctrl.SPRING_NX)
        self.assertEqual(_sl_.call_args.kwargs['iterations'], 25)
        self.assertIn('25 of 50 iterations', ctrl._last_cost_note_)

    def test_spring_nx_iterations_never_fall_below_the_floor(self):
        # Also documents how T4c and T2 compose: at this size the gate asks first, so the
        # lever is only reached on the confirming repeat.  Squeezing iterations is not a
        # substitute for asking -- it is what happens once the user has said yes.
        import networkx as nx, unittest.mock as mock
        ctrl = self._ctrl()
        _g_  = ctrl.graphs[ctrl.df_level]
        with mock.patch.object(_g_, 'number_of_nodes', return_value=10_000_000), \
             mock.patch.object(nx, 'spring_layout', wraps=nx.spring_layout) as _sl_:
            ctrl.apply_layout_operation(ctrl.SPRING_NX)
            self.assertIsNone(_sl_.call_args, 'the gate should have refused the first time')
            ctrl.apply_layout_operation(ctrl.SPRING_NX)
        self.assertEqual(_sl_.call_args.kwargs['iterations'], 10)

    def test_spring_nx_note_does_not_clear_another_operations_note(self):
        ctrl = self._ctrl()
        ctrl._last_cost_note_ = "link_shape='flowmap': 500 edges, awaiting confirm"
        ctrl.apply_layout_operation(ctrl.SPRING_NX)
        self.assertIn('flowmap', ctrl._last_cost_note_)

    def test_community_detection_passes_the_gate_at_test_sizes(self):
        # Wired to the gate but declared cheap, so it must not start asking.
        ctrl = self._ctrl()
        self.assertIsNotNone(ctrl.apply_community_detection())
        self.assertIsNone(ctrl._confirm_armed_)


@unittest.skipUnless(PANEL_AVAILABLE, 'panel not installed')
class TestLINKPIOffLoopExecution(unittest.TestCase):
    """T5a: heavy work runs on a worker thread so the widget keeps repainting.

    This does not make anything interruptible -- Python cannot kill a thread parked in
    numpy or networkx.  What it buys is that a long operation looks like a long operation
    rather than like a hang, which is a different property and worth its own tests.
    """

    def _ctrl(self):
        from polars2svg import Polars2SVG
        from polars2svg.interactive_controller import linkpi
        p2s = Polars2SVG()
        return linkpi(p2s.linkp(_make_link_df(), relationships=[('fm', 'to')], pos=_make_pos()))

    def test_the_event_loop_keeps_running_during_a_slow_operation(self):
        import time
        ctrl  = self._ctrl()
        ticks = []

        def _slow_():
            time.sleep(0.35)
            return {'a': (9.0, 9.0)}

        async def _drive_():
            async def _ticker_():
                _end_ = time.time() + 0.35
                while time.time() < _end_:
                    ticks.append(1)
                    await asyncio.sleep(0.005)
            ctrl.apply_layout_operation = _slow_
            ctrl.key_op_finished = 'w'
            await asyncio.gather(ctrl.applyKeyOp(None), _ticker_())

        asyncio.run(_drive_())
        # Run inline on the loop this would be 0; the exact count is machine-dependent, so
        # the assertion is only that the loop was not starved.
        self.assertGreater(len(ticks), 10, 'the event loop was blocked by the operation')

    def test_setAnimation_from_a_worker_thread_is_deferred_then_flushed(self):
        # D3: params are written only on the loop.  The confirm gate calls setAnimation
        # from inside a helper that now runs off-loop, so the queue is what keeps that
        # from being a cross-thread param write.
        ctrl = self._ctrl()
        _seen_ = {}

        def _work_():
            ctrl.setAnimation('<text>from the worker</text>')
            _seen_['during'] = ctrl.animation_inner
            return True

        asyncio.run(ctrl._run_offloop_(_work_))
        self.assertNotIn('from the worker', _seen_['during'], 'should not write mid-flight')
        self.assertIn('from the worker', ctrl.animation_inner, 'should flush afterwards')

    def test_offloop_depth_returns_to_zero_even_when_the_work_raises(self):
        ctrl = self._ctrl()

        def _boom_():
            raise ValueError('boom')

        with self.assertRaises(ValueError):
            asyncio.run(ctrl._run_offloop_(_boom_))
        self.assertEqual(ctrl._offloop_depth_, 0,
                         'a leaked depth would silently swallow every later setAnimation')

    def test_a_note_is_painted_before_the_work_starts(self):
        ctrl = self._ctrl()
        _seen_ = {}

        def _work_():
            _seen_['during'] = ctrl.animation_inner
            return True

        asyncio.run(ctrl._run_offloop_(_work_, note='working...'))
        self.assertIn('working...', _seen_['during'],
                      'the indicator must be up before the thing it describes runs')

    def test_escape_while_nothing_is_running_does_not_arm_a_cancel(self):
        # Otherwise an idle Escape would silently cancel whatever the user asked for next.
        ctrl = self._ctrl()
        asyncio.run(ctrl.applyCancel(None))
        self.assertFalse(ctrl._cancel_requested_)

    def test_escape_during_an_operation_sets_the_flag_the_worker_polls(self):
        ctrl  = self._ctrl()
        _seen_ = {}

        def _work_():
            for _ in range(200):
                if ctrl._cancelRequested_():
                    _seen_['stopped'] = True
                    return True
                time.sleep(0.005)
            return False

        async def _drive_():
            async def _cancel_():
                await asyncio.sleep(0.05)
                await ctrl.applyCancel(None)
            _res_, _ = await asyncio.gather(ctrl._run_offloop_(_work_), _cancel_())
            return _res_

        import time
        self.assertTrue(asyncio.run(_drive_()))
        self.assertTrue(_seen_.get('stopped'), 'the worker never saw the cancel')

    def test_each_operation_starts_with_the_cancel_flag_clear(self):
        # Armed at the start rather than cleared at the end, so a stale cancel cannot
        # carry into the next operation.
        ctrl = self._ctrl()
        ctrl._cancel_requested_ = True
        asyncio.run(ctrl._run_offloop_(lambda: None))
        self.assertFalse(ctrl._cancel_requested_)

    def test_refresh_offloop_still_updates_the_view(self):
        ctrl = self._ctrl()
        ctrl.mod_inner = ''
        asyncio.run(ctrl._refreshViewOffloop_())
        self.assertIn('<svg', ctrl.mod_inner)


@unittest.skipUnless(PANEL_AVAILABLE, 'panel not installed')
class TestLINKPIRejectWhileBusy(unittest.TestCase):
    """D4: a user action arriving mid-operation is dropped, not banked.

    The lock is held across an await that can last minutes; queueing would replay a burst
    of stale operations against a view that has since moved.
    """

    def _ctrl(self):
        from polars2svg import Polars2SVG
        from polars2svg.interactive_controller import linkpi
        p2s = Polars2SVG()
        return linkpi(p2s.linkp(_make_link_df(), relationships=[('fm', 'to')], pos=_make_pos()))

    def _while_locked(self, ctrl, coro_fn):
        async def _drive_():
            async def _hold_():
                async with ctrl.lock:
                    await asyncio.sleep(0.15)
            async def _act_():
                await asyncio.sleep(0.02)
                await coro_fn()
            await asyncio.gather(_hold_(), _act_())
        asyncio.run(_drive_())

    def test_a_key_press_while_busy_is_dropped(self):
        # Counted through a spy rather than by watching positions: assigning a trigger
        # param fires its watcher immediately, so the operation would already have run
        # once before the lock was ever taken.  The question here is only whether the
        # call made while the lock IS held gets through.
        ctrl   = self._ctrl()
        _calls_ = []
        ctrl.apply_layout_operation = lambda *a, **k: (_calls_.append(1), {})[1]
        ctrl.key_op_finished = 'w'
        _baseline_ = len(_calls_)
        self._while_locked(ctrl, lambda: ctrl.applyKeyOp(None))
        self.assertEqual(len(_calls_), _baseline_, 'the busy guard let the operation run')
        self.assertIn('busy', ctrl.animation_inner)

    def test_every_dropped_callback_clears_its_own_trigger(self):
        # The trigger params fire on CHANGE.  A dropped event that left its trigger set
        # would make the NEXT press of the same key silent -- the failure would look like
        # a broken keyboard, long after the operation that caused it.
        for _trigger_, _value_, _call_ in (
            ('key_op_finished',             'q',   'applyKeyOp'),
            ('drag_op_finished',            True,  'applyDragOp'),
            ('move_op_finished',            True,  'applyMoveOp'),
            ('wheel_op_finished',           True,  'applyWheelOp'),
            ('middle_op_finished',          True,  'applyMiddleOp'),
            ('unselected_move_op_finished', True,  'unselectedMoveOp'),
            ('search_op_finished',          True,  'applySearchOp'),
            ('brush_leave_done',            True,  'applyBrushLeave'),
        ):
            with self.subTest(callback=_call_):
                ctrl = self._ctrl()
                setattr(ctrl, _trigger_, _value_)
                self._while_locked(ctrl, lambda c=ctrl, n=_call_: getattr(c, n)(None))
                _left_ = getattr(ctrl, _trigger_)
                self.assertIn(_left_, ('', False, 0),
                              f'{_call_} dropped the event but left {_trigger_}={_left_!r}')

    def test_the_mvc_path_is_not_dropped_while_busy(self):
        # Peer-driven, not user-driven: dropping one would leave this view showing
        # something the other views are not.
        ctrl = self._ctrl()
        _df2_ = _make_link_df()
        async def _drive_():
            async def _hold_():
                async with ctrl.lock:
                    await asyncio.sleep(0.1)
            async def _peer_():
                await asyncio.sleep(0.02)
                await ctrl.display(_df2_, [_df2_], 0)
            await asyncio.gather(_hold_(), _peer_())
        asyncio.run(_drive_())
        self.assertNotIn('busy', ctrl.animation_inner)


@unittest.skipUnless(PANEL_AVAILABLE, 'panel not installed')
class TestLINKPIFlowmapConfirm(unittest.TestCase):
    """Switching link_shape to 'flowmap' asks first once the graph is big enough."""

    def _ctrl(self, confirm_above=1):
        from polars2svg import Polars2SVG
        from polars2svg.interactive_controller import linkpi
        import polars2svg.interactive_controller as _ic_
        from polars2svg.interactive_treatments import Treatment
        p2s  = Polars2SVG()
        ctrl = linkpi(p2s.linkp(_make_link_df(), relationships=[('fm', 'to')], pos=_make_pos()))
        _orig_ = _ic_.FLOWMAP
        _ic_.FLOWMAP = Treatment(truncatable=True, killable=True, confirm_above=confirm_above)
        self.addCleanup(setattr, _ic_, 'FLOWMAP', _orig_)
        return ctrl

    def _shape(self, ctrl):
        return ctrl.dfs_layout[ctrl.df_level].link_shape

    def _commit(self, ctrl, value):
        """Drive the picker watcher directly instead of assigning the param.

        applySizeChoice became a coroutine in the off-loop work, so assigning
        ``link_shape_choice`` now goes through param's async executor -- which runs the
        coroutine inline when no event loop is active but *defers* it to a task when one
        is.  Whether one is active depends on what else the suite has already done, so
        assigning the param made these tests pass alone and fail in a full run.  Calling
        the watcher explicitly is deterministic either way.
        """
        from types import SimpleNamespace
        asyncio.run(ctrl.applySizeChoice(
            SimpleNamespace(name='link_shape_choice', new=value)))

    def test_switching_to_flowmap_asks_first(self):
        ctrl = self._ctrl()
        self._commit(ctrl, 'flowmap')
        self.assertNotEqual(self._shape(ctrl), 'flowmap', 'should not have applied yet')
        self.assertEqual(ctrl._confirm_armed_, "link_shape='flowmap'")

    def test_the_picker_is_put_back_so_a_repeat_can_fire(self):
        # param watchers fire on change; leaving the picker reading 'flowmap' would make
        # re-picking it silent and the confirmation unreachable.
        ctrl = self._ctrl()
        self._commit(ctrl, 'flowmap')
        self.assertNotEqual(ctrl.link_shape_choice, 'flowmap')

    def test_repeating_applies_flowmap(self):
        ctrl = self._ctrl()
        self._commit(ctrl, 'flowmap')
        self._commit(ctrl, 'flowmap')
        self.assertEqual(self._shape(ctrl), 'flowmap')

    def test_below_threshold_applies_immediately(self):
        ctrl = self._ctrl(confirm_above=10_000)
        self._commit(ctrl, 'flowmap')
        self.assertEqual(self._shape(ctrl), 'flowmap')
        self.assertIsNone(ctrl._confirm_armed_)

    def test_other_shapes_are_never_gated(self):
        ctrl = self._ctrl()
        self._commit(ctrl, 'curve')
        self.assertEqual(self._shape(ctrl), 'curve')
        self.assertIsNone(ctrl._confirm_armed_)


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
        result = ctrl._layout_registry[ctrl.CIRCLE_PACK].handler(ln, g, set())
        self.assertIsInstance(result, tuple)
        pos, shapes = result
        self.assertIsInstance(pos, dict)
        self.assertIsInstance(shapes, dict)
        self.assertGreater(len(shapes), 0)

    def test_donut_handler_returns_pos_and_cells(self):
        ctrl = self._make_donut_ctrl()
        ln, g = ctrl.dfs_layout[0], ctrl.graphs[0]
        result = ctrl._layout_registry[ctrl.HYPERTREE_DONUT].handler(ln, g, set())
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

    # ── background + visibility propagate across the whole stack ──────────────
    def _stacked_ctrl_with_background(self):
        # Capture a real layout background at the base level, then grow the stack.
        ctrl = self._make_ctrl()
        ctrl.__layoutOperation__(ctrl.CIRCLE_PACK, ctrl.dfs_layout[0], ctrl.graphs[0], set())
        ctrl.pushStack(_make_link_df().filter(pl.col('fm') == 'a'))
        self.assertEqual(len(ctrl.dfs_layout), 2)
        return ctrl

    def test_background_reaches_every_existing_level(self):
        ctrl = self._stacked_ctrl_with_background()
        ctrl.background_state = 1
        ctrl.__applyBackgroundState__(refresh=False)
        for _layout_ in ctrl.dfs_layout:
            self.assertIsNotNone(_layout_.background)
            self.assertIsNone(_layout_.background_label_color)

    def test_background_labels_state_reaches_every_existing_level(self):
        ctrl = self._stacked_ctrl_with_background()
        ctrl.background_state = 2
        ctrl.__applyBackgroundState__(refresh=False)
        for _layout_ in ctrl.dfs_layout:
            self.assertIsNotNone(_layout_.background)
            self.assertIsNotNone(_layout_.background_label_color)

    def test_background_cleared_on_every_level(self):
        ctrl = self._stacked_ctrl_with_background()
        ctrl.background_state = 1
        ctrl.__applyBackgroundState__(refresh=False)
        ctrl.background_state = 0
        ctrl.__applyBackgroundState__(refresh=False)
        for _layout_ in ctrl.dfs_layout:
            self.assertIsNone(_layout_.background)
            self.assertIsNone(_layout_.background_label_color)

    def test_new_layer_pushed_after_enabling_background_inherits_it(self):
        # The future-layers guarantee: enable background, THEN grow the stack; the
        # fresh (cloned) layer still carries both the background and its visibility.
        ctrl = self._make_ctrl()
        ctrl.__layoutOperation__(ctrl.CIRCLE_PACK, ctrl.dfs_layout[0], ctrl.graphs[0], set())
        ctrl.background_state = 2
        ctrl.__applyBackgroundState__(refresh=False)
        ctrl.pushStack(_make_link_df().filter(pl.col('fm') == 'a'))
        _top_ = ctrl.dfs_layout[ctrl.df_level]
        self.assertIsNotNone(_top_.background)
        self.assertIsNotNone(_top_.background_label_color)

    def test_b_key_cycle_at_deep_level_reaches_the_base_level(self):
        # End-to-end via the actual 'b' key handler at a pushed level.
        ctrl = self._stacked_ctrl_with_background()
        self.assertEqual(ctrl.df_level, 1)
        self._press_b(ctrl)                       # state 0 -> 1
        self.assertEqual(ctrl.background_state, 1)
        self.assertIsNotNone(ctrl.dfs_layout[0].background)
        self.assertIsNotNone(ctrl.dfs_layout[1].background)


@unittest.skipUnless(PANEL_AVAILABLE, 'panel not installed')
class TestLINKPILayoutStackPropagation(unittest.TestCase):
    """A layout run on one stack level has to reach every other level: each level
    keeps its own copy of pos (__renderView__ copies it at push time), so without
    an explicit push, navigating the stack drops back to the pre-layout
    arrangement."""

    def _make_ctrl_with_stack(self):
        from polars2svg.interactive_controller import linkpi
        p2s   = Polars2SVG()
        linkp = p2s.linkp(_make_link_df(), relationships=[('fm', 'to')], pos=_make_pos())
        ctrl  = linkpi(linkp)
        # Filter down to the a-b edge & push it: level 0 = full graph, level 1 = subset.
        ctrl.pushStack(_make_link_df().filter(pl.col('fm') == 'a'))
        return ctrl

    def test_stack_is_two_levels_deep(self):
        ctrl = self._make_ctrl_with_stack()
        self.assertEqual(ctrl.df_level, 1)
        self.assertEqual(len(ctrl.dfs_layout), 2)

    def test_layout_on_pushed_level_reaches_the_base_level(self):
        ctrl   = self._make_ctrl_with_stack()
        before = dict(ctrl.dfs_layout[0].pos)
        moved  = ctrl.apply_layout_operation(ctrl.SPRING_NX)
        self.assertTrue(moved)
        _ln_top_, _ln_base_ = ctrl.dfs_layout[1], ctrl.dfs_layout[0]
        self.assertNotEqual(before, dict(_ln_base_.pos))     # the base level moved
        for _node_ in ctrl.graphs[1].nodes():
            self.assertEqual(_ln_base_.pos[_node_], _ln_top_.pos[_node_])

    def test_layout_propagates_the_recomputed_view_window(self):
        # The layout rescales the world; a level left on the old window would be
        # looking at empty space after the positions arrive.
        ctrl = self._make_ctrl_with_stack()
        self.assertTrue(ctrl.apply_layout_operation(ctrl.SPRING_NX))
        self.assertEqual(ctrl.dfs_layout[0].getViewWindow(),
                         ctrl.dfs_layout[1].getViewWindow())

    def test_layout_does_not_add_nodes_to_a_level_that_lacks_them(self):
        # 'c' is outside the pushed subset's graph, but it is in every level's pos
        # (copied at push time) -- propagation updates in place, never introduces
        # a node the level didn't already carry.
        ctrl  = self._make_ctrl_with_stack()
        del ctrl.dfs_layout[0].pos['c']
        ctrl.apply_layout_operation(ctrl.SPRING_NX)
        self.assertNotIn('c', ctrl.dfs_layout[0].pos)

    def test_declined_layout_leaves_every_level_untouched(self):
        # pivot mds is a global re-layout -> a no-op with a selection in place.  (This used
        # spring nx until 0.2.0, when spring nx gained a selection branch and stopped
        # declining -- see test_spring_nx_handler_moves_selection_and_pins_the_rest.)
        ctrl   = self._make_ctrl_with_stack()
        ctrl.selected_entities = {'a'}
        before = [dict(ln.pos) for ln in ctrl.dfs_layout]
        self.assertEqual(ctrl.apply_layout_operation(ctrl.PIVOT_MDS), {})
        self.assertEqual([dict(ln.pos) for ln in ctrl.dfs_layout], before)

    def test_w_key_propagates_across_the_stack(self):
        # Same path through the real key handler rather than the sync helper.
        ctrl = self._make_ctrl_with_stack()
        ctrl.layout_operation  = ctrl.SPRING_NX
        before = dict(ctrl.dfs_layout[0].pos)
        ctrl.key_op_finished = 'w'
        asyncio.run(ctrl.applyKeyOp(None))
        self.assertNotEqual(before, dict(ctrl.dfs_layout[0].pos))
        for _node_ in ctrl.graphs[1].nodes():
            self.assertEqual(ctrl.dfs_layout[0].pos[_node_], ctrl.dfs_layout[1].pos[_node_])


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
class TestLINKPICircleByColorLayout(unittest.TestCase):
    """The 'circle (color)' layout mode: same circular drag shape as 'circle',
    but the nodes are grouped by color around the circumference."""

    # Six nodes in a ring, two colors alternating around it -- so the color
    # grouping has to actually re-order them to make each color contiguous.
    _NODES_  = ['n0', 'n1', 'n2', 'n3', 'n4', 'n5']
    _COLORS_ = {'n0': '#ff0000', 'n2': '#ff0000', 'n4': '#ff0000',
                'n1': '#00ff00', 'n3': '#00ff00', 'n5': '#00ff00'}

    def _make_ctrl(self, node_color=None):
        import math
        from polars2svg.interactive_controller import linkpi
        p2s = Polars2SVG()
        df  = pl.DataFrame({'fm': self._NODES_,
                            'to': self._NODES_[1:] + self._NODES_[:1]})
        pos = {n: [math.cos(i * math.pi / 3), math.sin(i * math.pi / 3)]
               for i, n in enumerate(self._NODES_)}
        linkp = p2s.linkp(df, relationships=[('fm', 'to')], pos=pos,
                          node_color=(self._COLORS_ if node_color is None else node_color))
        ctrl  = linkpi(linkp)
        ctrl.selected_entities = set(self._NODES_)
        return ctrl

    def _ring_(self, ctrl, updated):
        """Colors in circumference order, starting from an arbitrary node."""
        import math
        _ln_ = ctrl.dfs_layout[ctrl.df_level]
        cx, cy = _ln_.xT_inv(200.0), _ln_.yT_inv(200.0)
        ordered = sorted(updated, key=lambda n: math.atan2(updated[n][1] - cy, updated[n][0] - cx))
        return [self._COLORS_.get(n) for n in ordered]

    def test_mode_is_in_the_picker_menu(self):
        from polars2svg.interactive_controller import _LAYOUT_MODE_MENU_
        self.assertIn('circle (color)', [label for _, label in _LAYOUT_MODE_MENU_])

    def test_constant_matches_menu_label(self):
        ctrl = self._make_ctrl()
        self.assertEqual(ctrl.CIRCLE_BY_COLOR, 'circle (color)')
        self.assertIn(ctrl.CIRCLE_BY_COLOR, ctrl.layout_modes)

    def test_uses_the_circular_drag_shape_in_js(self):
        # The JS shape preview must treat it exactly like 'circle' (layoutcircle),
        # not fall through to the rectangle branch.
        cls   = type(self._make_ctrl())
        shape = cls._scripts['myUpdateLayoutOp']
        self.assertIn('state.layout_op_shape == "circle (color)"', shape)
        _circle_branch_ = shape.split('reset_circle = false;')[0]
        self.assertIn('"circle (color)"', _circle_branch_)

    def test_groups_colors_contiguously_on_the_circle(self):
        ctrl    = self._make_ctrl()
        updated = ctrl.apply_layout_interaction(200.0, 200.0, 300.0, 200.0, 'circle (color)')
        self.assertEqual(set(updated), set(self._NODES_))
        ring = self._ring_(ctrl, updated)
        switches = sum(1 for i in range(len(ring)) if ring[i] != ring[i-1])
        self.assertEqual(switches, 2, f'each color must form one arc: {ring}')

    def test_all_nodes_land_on_the_dragged_circle(self):
        import math
        ctrl    = self._make_ctrl()
        updated = ctrl.apply_layout_interaction(200.0, 200.0, 300.0, 200.0, 'circle (color)')
        _ln_    = ctrl.dfs_layout[ctrl.df_level]
        cx, cy  = _ln_.xT_inv(200.0), _ln_.yT_inv(200.0)
        r       = math.hypot(_ln_.xT_inv(300.0) - cx, _ln_.yT_inv(200.0) - cy)
        for n in self._NODES_:
            self.assertAlmostEqual(math.hypot(updated[n][0] - cx, updated[n][1] - cy), r, places=6)

    def test_uncolored_nodes_fall_back_to_plain_circle(self):
        # A single node color for everything -> no grouping to do; the result is
        # the plain circle layout (evenly spaced, no color gaps).
        import math
        ctrl    = self._make_ctrl(node_color='#4988b6')
        updated = ctrl.apply_layout_interaction(200.0, 200.0, 300.0, 200.0, 'circle (color)')
        _ln_    = ctrl.dfs_layout[ctrl.df_level]
        cx, cy  = _ln_.xT_inv(200.0), _ln_.yT_inv(200.0)
        angles  = sorted(math.atan2(updated[n][1] - cy, updated[n][0] - cx) for n in self._NODES_)
        for i in range(len(angles)):
            _step_ = (angles[i] - angles[i-1]) % (2 * math.pi)
            self.assertAlmostEqual(_step_, 2 * math.pi / len(self._NODES_), places=6)

    def test_propagates_positions_across_the_stack(self):
        ctrl = self._make_ctrl()
        ctrl.selected_entities = {'n5'}
        self.assertTrue(ctrl.apply_push_selected())
        ctrl.selected_entities = set(self._NODES_)
        updated = ctrl.apply_layout_interaction(200.0, 200.0, 300.0, 200.0, 'circle (color)')
        for level in range(len(ctrl.dfs_layout)):
            for n in self._NODES_:
                if n in ctrl.dfs_layout[level].pos:
                    self.assertEqual(ctrl.dfs_layout[level].pos[n], updated[n])


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

    def _menu_items(self, ctrl):
        import json, re
        render = type(ctrl)._scripts['render']
        return json.loads(re.search(r'state\.menu_items = (\{.*?\});', render, re.S).group(1))

    def test_link_shape_menu_lists_all_shapes(self):
        # Items are [mnemonic, value, display, guarded]; the VALUE is what round-trips to
        # Python and must stay the bare label, or it would stop matching the registry key.
        _items_ = self._menu_items(self._make_ctrl())['link_shape']
        self.assertEqual([_i_[1] for _i_ in _items_], ['line', 'curve', 'flowmap'])

    def test_an_operation_that_will_ask_says_so_in_the_menu(self):
        # Threshold read from the declaration rather than hardcoded: sibling tests patch
        # these module-level Treatments, and pytest-randomly means execution order is not
        # fixed, so a literal here would be an order-dependent failure waiting to happen.
        from polars2svg.interactive_treatments import SPRING_NX
        _by_value_ = {_i_[1]: _i_ for _i_ in self._menu_items(self._make_ctrl())['operation']}
        _spring_ = _by_value_['spring nx']
        self.assertIn(f'asks >{SPRING_NX.confirm_above:,} nodes', _spring_[2])
        self.assertTrue(_spring_[3], 'it should be guarded against accidental commit')

    def test_flowmap_says_so_in_the_shape_menu(self):
        from polars2svg.interactive_treatments import FLOWMAP
        _by_value_ = {_i_[1]: _i_ for _i_ in self._menu_items(self._make_ctrl())['link_shape']}
        self.assertIn(f'asks >{FLOWMAP.confirm_above:,} edges', _by_value_['flowmap'][2])
        self.assertTrue(_by_value_['flowmap'][3])

    def test_a_cheap_operation_is_neither_annotated_nor_guarded(self):
        # Annotating everything would make the annotation worth nothing.
        _by_value_ = {_i_[1]: _i_ for _i_ in self._menu_items(self._make_ctrl())['operation']}
        _hyper_ = _by_value_['hyper tree']
        self.assertEqual(_hyper_[2], 'hyper tree')
        self.assertFalse(_hyper_[3])

    def test_line_and_curve_are_not_guarded(self):
        _by_value_ = {_i_[1]: _i_ for _i_ in self._menu_items(self._make_ctrl())['link_shape']}
        for _cheap_ in ('line', 'curve'):
            self.assertFalse(_by_value_[_cheap_][3])

    def test_guarded_items_are_exempt_from_both_no_enter_commit_paths(self):
        # The two ways this menu used to commit without an Enter: the single-character
        # mnemonic, and the 2.5s inactivity timeout.  Asserted against the emitted JS
        # because that is where the behaviour lives.
        _scripts_ = type(self._make_ctrl())._scripts
        _keydown_ = _scripts_['myOnKeyDown']
        _at_ = _keydown_.index('_items_[_i_][0] === event.key')
        self.assertIn('_items_[_i_][3]', _keydown_[_at_:_at_ + 800],
                      'mnemonic commit does not check the guard flag')
        self.assertIn('_sel_[3]', _scripts_['menuArmTimer'],
                      'the inactivity timeout does not check the guard flag')

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
class TestLINKPITimingSpacingPicker(unittest.TestCase):
    """The shift-a / ctrl-a timing-mark spacing picker: default selection, the pixel
    grid, the JS commit path onto the LinkP's timing_marks_spacing (in pixels), a
    user spacing outside the grid, and that a coarser choice re-renders fewer marks."""

    def _make_ctrl(self, **link_kwargs):
        from polars2svg.interactive_controller import linkpi
        p2s   = Polars2SVG()
        linkp = p2s.linkp(_make_link_df(), relationships=[('fm', 'to')],
                          pos=_make_pos(), **link_kwargs)
        return linkpi(linkp)

    def _make_time_ctrl(self, n=2000, **link_kwargs):
        from polars2svg.interactive_controller import linkpi
        base  = datetime(2024, 1, 1)
        df    = pl.DataFrame({'fm': ['a'] * n, 'to': ['b'] * n,
                              'ts': [base + timedelta(seconds=17 * k) for k in range(n)]})
        p2s   = Polars2SVG()
        linkp = p2s.linkp(df, relationships=[('fm', 'to')], pos={'a': (0.0, 0.0), 'b': (1.0, 0.0)},
                          time='ts', wxh=(512, 512), **link_kwargs)
        return linkpi(linkp)

    def _menu_items(self, ctrl, kind):
        import json, re
        render = type(ctrl)._scripts['render']
        items  = json.loads(re.search(r'state\.menu_items = (\{.*?\});', render, re.S).group(1))
        return items[kind]

    # ── default selection + pixel grid ────────────────────────────────────────
    def test_default_choice_is_one_pixel(self):
        self.assertEqual(self._make_ctrl().timing_spacing_choice, '1')

    def test_menu_is_pixel_grid(self):
        labels = [lbl for _, lbl in self._menu_items(self._make_ctrl(), 'timing_spacing')]
        self.assertEqual(labels, ['1', '2', '4', '8', '16', '32'])

    def test_custom_spacing_becomes_current_and_in_menu(self):
        ctrl = self._make_ctrl(timing_marks_spacing=10)
        self.assertEqual(ctrl.timing_spacing_choice, '10')
        self.assertIn('10', [lbl for _, lbl in self._menu_items(ctrl, 'timing_spacing')])

    # ── the JS commit path: setting the choice pushes pixels onto the LinkP ────
    def test_choice_commit_pushes_pixels_onto_linkp(self):
        ctrl = self._make_ctrl()
        ctrl.timing_spacing_choice = '8'
        for ln in ctrl.dfs_layout:
            self.assertEqual(ln.timing_marks_spacing, 8.0)

    def test_coarser_choice_rerenders_fewer_marks(self):
        ctrl = self._make_time_ctrl()
        fine = ctrl.mod_inner.count('stroke-width="1.5"')   # timing-mark signature
        ctrl.timing_spacing_choice = '16'
        coarse = ctrl.mod_inner.count('stroke-width="1.5"')
        self.assertGreater(fine, 0)
        self.assertLess(coarse, fine)

    # ── template / script wiring ──────────────────────────────────────────────
    def test_render_script_seeds_timing_spacing_kind(self):
        self.assertIn('"timing_spacing"', type(self._make_ctrl())._scripts['render'])

    def test_commit_script_handles_timing_spacing(self):
        self.assertIn('timing_spacing_choice', type(self._make_ctrl())._scripts['menuCommit'])

    def test_keyboard_help_mentions_spacing_picker(self):
        cmds = type(self._make_ctrl())._keyboard_commands_
        self.assertIn('shift-a', cmds)
        self.assertIn('spacing', cmds)


@unittest.skipUnless(PANEL_AVAILABLE, 'panel not installed')
class TestLINKPICopyToClipboard(_UnfilteredLoggerMixin, unittest.TestCase):
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

    def test_copies_the_selection(self):
        from unittest.mock import patch
        from polars2svg import interactive_controller as ic
        ctrl = self._make_ctrl()
        ctrl.selected_entities = {'a', 'b'}
        _copied_ = []
        with patch.object(ic.pyperclip, 'copy', _copied_.append):
            self._press_ctrl_c(ctrl)
        self.assertEqual(len(_copied_), 1)
        self.assertEqual(set(_copied_[0].split('\n')), {'a', 'b'})

    def test_unavailable_clipboard_does_not_raise(self):
        # A headless kernel (remote JupyterHub, served Panel app) has no copy
        # mechanism and pyperclip raises.  Inside an async watcher that leaves the
        # widget wedged, so the failure is logged and the keystroke survives it.
        from unittest.mock import patch
        from polars2svg import interactive_controller as ic

        def _boom(text):
            raise RuntimeError('Pyperclip could not find a copy/paste mechanism')

        ctrl = self._make_ctrl()
        ctrl.selected_entities = {'a', 'b'}
        with patch.object(ic.pyperclip, 'copy', _boom):
            with self.assertLogs('polars2svg_logger', level='WARNING') as _log_:
                self._press_ctrl_c(ctrl)          # must not raise
        self.assertIn('clipboard unavailable', '\n'.join(_log_.output))

    def test_unavailable_clipboard_logs_the_names(self):
        # The selection is the thing the user asked for; if it cannot reach the
        # clipboard it has to remain recoverable somewhere.
        from unittest.mock import patch
        from polars2svg import interactive_controller as ic

        def _boom(text):
            raise RuntimeError('no mechanism')

        ctrl = self._make_ctrl()
        ctrl.selected_entities = {'a', 'b'}
        with patch.object(ic.pyperclip, 'copy', _boom):
            with self.assertLogs('polars2svg_logger', level='WARNING') as _log_:
                self._press_ctrl_c(ctrl)
        _text_ = '\n'.join(_log_.output)
        self.assertIn('a', _text_)
        self.assertIn('b', _text_)

    def test_returns_false_when_clipboard_unavailable(self):
        from unittest.mock import patch
        from polars2svg import interactive_controller as ic

        def _boom(text):
            raise RuntimeError('no mechanism')

        ctrl = self._make_ctrl()
        with patch.object(ic.pyperclip, 'copy', _boom):
            with self.assertLogs('polars2svg_logger', level='WARNING'):
                self.assertFalse(ctrl._copyToClipboard_(['a']))

    def test_non_string_node_names_are_stringified(self):
        # Node names need not be strings (integer node ids are a supported case),
        # and ''.join() over them would raise before pyperclip ever saw them.
        from unittest.mock import patch
        from polars2svg import interactive_controller as ic
        ctrl = self._make_ctrl()
        _copied_ = []
        with patch.object(ic.pyperclip, 'copy', _copied_.append):
            self.assertTrue(ctrl._copyToClipboard_([1, 2]))
        self.assertEqual(_copied_[0], '1\n2')


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


class TestPanelizePayloadGuard(_UnfilteredLoggerMixin, unittest.TestCase):
    '''panelize() warns (with the measured MB) when the composed SVG document would
    exceed the Bokeh WebSocket message limit -- the size condition behind the browser's
    "SyntaxError: Unexpected end of JSON input". The guard operates on the embedded
    ReactiveHTML _template, so it is exercised here with lightweight fakes rather than a
    netflow-sized render.'''

    class _FakeView:
        def __init__(self, template):
            self._template = template

    def _view(self, nbytes):
        return self._FakeView('x' * nbytes)

    def test_estimate_sums_template_bytes(self):
        from polars2svg import interactive_controller as ic
        self.assertEqual(ic._estimate_panel_payload_bytes_([self._view(1000), self._view(2000)]), 3000)

    def test_estimate_ignores_views_without_template(self):
        from polars2svg import interactive_controller as ic
        self.assertEqual(ic._estimate_panel_payload_bytes_([object()]), 0)

    def test_warns_over_default_limit(self):
        from polars2svg import interactive_controller as ic
        with self.assertLogs('polars2svg_logger', level='WARNING') as cm:
            ic._warnOversizePanelPayload_([self._view(25 * 1024 * 1024)])   # 25 MB > 20 MB default
        _msg_ = '\n'.join(cm.output)
        self.assertIn('Unexpected end of JSON input', _msg_)
        self.assertIn('websocket_max_message_size', _msg_)
        self.assertIn('25.0 MB', _msg_)                                     # measured size surfaced

    def test_quiet_under_default_limit(self):
        from polars2svg import interactive_controller as ic
        with self.assertNoLogs('polars2svg_logger', level='WARNING'):
            ic._warnOversizePanelPayload_([self._view(1 * 1024 * 1024)])    # 1 MB

    def test_custom_limit_suppresses_warning(self):
        from polars2svg import interactive_controller as ic
        with self.assertNoLogs('polars2svg_logger', level='WARNING'):
            ic._warnOversizePanelPayload_([self._view(25 * 1024 * 1024)],
                                          websocket_max_message_size=200 * 1024 * 1024)

    def test_lower_custom_limit_triggers_warning(self):
        from polars2svg import interactive_controller as ic
        with self.assertLogs('polars2svg_logger', level='WARNING'):
            ic._warnOversizePanelPayload_([self._view(5 * 1024 * 1024)],    # 5 MB
                                          websocket_max_message_size=4 * 1024 * 1024)


@unittest.skipUnless(PANEL_AVAILABLE, 'panel not installed')
class TestLINKPIRegexSearchBounds(_UnfilteredLoggerMixin, unittest.TestCase):
    """The '/.../' search form compiles a pattern typed in the browser and runs it
    over every node.  It is bounded by a pattern-length cap and a whole-scan deadline
    so that a catastrophically backtracking pattern cannot wedge the event loop."""

    def _make_ctrl(self):
        from polars2svg.interactive_controller import linkpi
        p2s   = Polars2SVG()
        linkp = p2s.linkp(_make_link_df(), relationships=[('fm', 'to')], pos=_make_pos())
        return linkpi(linkp)

    def _nodes(self, ctrl):
        return set(ctrl.graphs[ctrl.df_level].nodes())

    def test_defaults_are_the_module_constants(self):
        from polars2svg import interactive_controller as ic
        ctrl = self._make_ctrl()
        self.assertEqual(ctrl.regex_max_pattern,  ic._REGEX_MAX_PATTERN_)
        self.assertEqual(ctrl.regex_match_budget, ic._REGEX_MATCH_BUDGET_S_)

    def test_ordinary_pattern_still_matches(self):
        ctrl = self._make_ctrl()
        self.assertEqual(ctrl._matchNodesByRegex_('^a$', self._nodes(ctrl)), {'a'})

    def test_ordinary_pattern_leaves_no_cost_note(self):
        ctrl = self._make_ctrl()
        ctrl._matchNodesByRegex_('^a$', self._nodes(ctrl))
        self.assertIsNone(ctrl._last_cost_note_)

    def test_over_long_pattern_is_refused(self):
        ctrl = self._make_ctrl()
        ctrl.regex_max_pattern = 8
        with self.assertLogs('polars2svg_logger', level='WARNING'):
            _set_ = ctrl._matchNodesByRegex_('a' * 64, self._nodes(ctrl))
        self.assertEqual(_set_, set())
        self.assertIn('regex search:', ctrl._last_cost_note_)

    def test_pattern_at_the_cap_is_still_compiled(self):
        ctrl = self._make_ctrl()
        ctrl.regex_max_pattern = 2
        self.assertEqual(ctrl._matchNodesByRegex_('^a', self._nodes(ctrl)), {'a'})

    def test_expired_budget_abandons_the_scan(self):
        # A zero budget expires before the first subject, which is the same code path a
        # runaway pattern reaches after the first few -- the deadline is checked between
        # subjects, because re.search() cannot be interrupted once it is inside a match.
        ctrl = self._make_ctrl()
        ctrl.regex_match_budget = 0.0
        with self.assertLogs('polars2svg_logger', level='WARNING'):
            _set_ = ctrl._matchNodesByRegex_('.', self._nodes(ctrl))
        self.assertEqual(_set_, set())
        self.assertIn('time budget', ctrl._last_cost_note_)

    def test_stale_note_is_cleared_by_a_clean_scan(self):
        ctrl = self._make_ctrl()
        ctrl.regex_match_budget = 0.0
        with self.assertLogs('polars2svg_logger', level='WARNING'):
            ctrl._matchNodesByRegex_('.', self._nodes(ctrl))
        ctrl.regex_match_budget = 2.0
        ctrl._matchNodesByRegex_('^a$', self._nodes(ctrl))
        self.assertIsNone(ctrl._last_cost_note_)

    def test_another_operations_note_is_left_alone(self):
        ctrl = self._make_ctrl()
        ctrl._last_cost_note_ = 'spring nx: 40 of 200 iterations'
        ctrl._matchNodesByRegex_('^a$', self._nodes(ctrl))
        self.assertEqual(ctrl._last_cost_note_, 'spring nx: 40 of 200 iterations')

    def test_invalid_pattern_still_contributes_nothing(self):
        ctrl = self._make_ctrl()
        self.assertEqual(ctrl._matchNodesByRegex_('(unbalanced', self._nodes(ctrl)), set())

    def test_search_op_reaches_the_bound(self):
        # End to end from the param the browser writes.
        ctrl = self._make_ctrl()
        ctrl.regex_match_budget = 0.0
        ctrl.search_str = '/./'
        with self.assertLogs('polars2svg_logger', level='WARNING'):
            asyncio.run(ctrl.applySearchOp(None))
        self.assertEqual(ctrl.selected_entities, set())


class TestControllerStackDepthLimit(_UnfilteredLoggerMixin, unittest.TestCase):
    """InteractionController.pushStack() refuses at max_stack_depth instead of
    growing forever -- every level retains a whole DataFrame and levels are pushed
    one per keystroke."""

    def setUp(self):
        super().setUp()
        self.df  = pl.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        self.mvc = InteractionController()
        self.mvc.addStack('default', self.df)
        self.view = MockView()
        self.mvc.link(self.view, [], on='stack', stack='default')

    def _push(self, n):
        return pl.DataFrame({'a': [n], 'b': [n]})

    def test_default_limit_is_the_module_constant(self):
        from polars2svg import interactive_controller as ic
        self.assertEqual(self.mvc.max_stack_depth, ic._MAX_STACK_DEPTH_)

    def test_push_under_the_limit_returns_true(self):
        self.assertTrue(asyncio.run(self.mvc.pushStack(self.view, self._push(1))))

    def test_stack_stops_growing_at_the_limit(self):
        self.mvc.max_stack_depth = 3
        for i in range(10):
            asyncio.run(self.mvc.pushStack(self.view, self._push(i)))
        self.assertEqual(len(self.mvc.stacks['default']['dfs']), 3)

    def test_refused_push_returns_false(self):
        self.mvc.max_stack_depth = 2
        asyncio.run(self.mvc.pushStack(self.view, self._push(1)))
        with self.assertLogs('polars2svg_logger', level='WARNING'):
            self.assertFalse(asyncio.run(self.mvc.pushStack(self.view, self._push(2))))

    def test_refused_push_leaves_the_index_alone(self):
        self.mvc.max_stack_depth = 2
        asyncio.run(self.mvc.pushStack(self.view, self._push(1)))
        with self.assertLogs('polars2svg_logger', level='WARNING'):
            asyncio.run(self.mvc.pushStack(self.view, self._push(2)))
        self.assertEqual(self.mvc.stacks['default']['index'], 1)

    def test_refused_push_does_not_notify_views(self):
        self.mvc.max_stack_depth = 2
        asyncio.run(self.mvc.pushStack(self.view, self._push(1)))
        _before_ = len(self.view.display_calls)
        with self.assertLogs('polars2svg_logger', level='WARNING'):
            asyncio.run(self.mvc.pushStack(self.view, self._push(2)))
        self.assertEqual(len(self.view.display_calls), _before_)

    def test_popping_makes_room_again(self):
        self.mvc.max_stack_depth = 2
        asyncio.run(self.mvc.pushStack(self.view, self._push(1)))
        with self.assertLogs('polars2svg_logger', level='WARNING'):
            asyncio.run(self.mvc.pushStack(self.view, self._push(2)))
        asyncio.run(self.mvc.popStack(self.view))
        self.assertTrue(asyncio.run(self.mvc.pushStack(self.view, self._push(3))))

    def test_base_level_survives(self):
        # The bound refuses the new level rather than rolling the oldest one off:
        # dfs[0] is the unfiltered base every widening works against.
        self.mvc.max_stack_depth = 2
        asyncio.run(self.mvc.pushStack(self.view, self._push(1)))
        with self.assertLogs('polars2svg_logger', level='WARNING'):
            asyncio.run(self.mvc.pushStack(self.view, self._push(2)))
        self.assertIs(self.mvc.stackTopDataFrame(self.view), self.df)


@unittest.skipUnless(PANEL_AVAILABLE, 'panel not installed')
class TestLINKPIStackDepthLimit(_UnfilteredLoggerMixin, unittest.TestCase):
    """linkpi's own stack (dfs / dfs_layout / graphs) is bounded the same way, with
    one exemption: display()'s replay of the authoritative MVC stack."""

    def _make_ctrl(self):
        from polars2svg.interactive_controller import linkpi
        p2s   = Polars2SVG()
        linkp = p2s.linkp(_make_link_df(), relationships=[('fm', 'to')], pos=_make_pos())
        return linkpi(linkp)

    def _level(self, ctrl, keep):
        return ctrl.dfs[0].filter(pl.col('fm').is_in(keep))

    def test_default_limit_is_the_module_constant(self):
        from polars2svg import interactive_controller as ic
        self.assertEqual(self._make_ctrl().max_stack_depth, ic._MAX_STACK_DEPTH_)

    def test_push_under_the_limit_returns_true(self):
        ctrl = self._make_ctrl()
        self.assertTrue(ctrl.pushStack(self._level(ctrl, ['a', 'b'])))

    def test_push_at_the_limit_returns_false(self):
        ctrl = self._make_ctrl()
        ctrl.max_stack_depth = 2
        ctrl.pushStack(self._level(ctrl, ['a', 'b']))
        self.assertFalse(ctrl.pushStack(self._level(ctrl, ['a'])))

    def test_stack_stops_growing_at_the_limit(self):
        ctrl = self._make_ctrl()
        ctrl.max_stack_depth = 2
        for _keep_ in (['a', 'b'], ['a'], ['b'], ['c']):
            ctrl.pushStack(self._level(ctrl, _keep_))
        self.assertEqual(len(ctrl.dfs), 2)
        self.assertEqual(ctrl.df_level, 1)

    def test_refusal_leaves_a_cost_note(self):
        ctrl = self._make_ctrl()
        ctrl.max_stack_depth = 1
        ctrl.pushStack(self._level(ctrl, ['a', 'b']))
        self.assertIn('level limit', ctrl._last_cost_note_)

    def test_apply_push_selected_reports_the_refusal(self):
        # The 'x' key's helper promises in its docstring that it pushed; at the limit
        # it has to say so rather than return an unconditional True.
        ctrl = self._make_ctrl()
        ctrl.max_stack_depth = 1
        ctrl.selected_entities = {'a'}
        self.assertFalse(ctrl.apply_push_selected())

    def test_popping_makes_room_again(self):
        ctrl = self._make_ctrl()
        ctrl.max_stack_depth = 2
        ctrl.pushStack(self._level(ctrl, ['a', 'b']))
        self.assertFalse(ctrl.pushStack(self._level(ctrl, ['a'])))
        ctrl.popStack()
        self.assertTrue(ctrl.pushStack(self._level(ctrl, ['b'])))

    def test_replay_path_is_exempt(self):
        ctrl = self._make_ctrl()
        ctrl.max_stack_depth = 1
        self.assertTrue(ctrl.pushStack(self._level(ctrl, ['a', 'b']), enforce_limit=False))

    def test_display_walk_terminates_past_the_limit(self):
        # The regression this exemption exists for: display() advances by CALLING
        # pushStack, so a refusal inside that loop never terminates.  Run it off the
        # main thread so a regression fails the test instead of hanging the suite.
        import threading
        ctrl = self._make_ctrl()
        ctrl.max_stack_depth = 1
        _dfs_ = [ctrl.dfs[0],
                 self._level(ctrl, ['a', 'b']),
                 self._level(ctrl, ['a']),
                 self._level(ctrl, ['b'])]
        _done_ = []

        def _run():
            asyncio.run(ctrl.display(_dfs_[3], _dfs_, 3))
            _done_.append(True)

        _t_ = threading.Thread(target=_run, daemon=True)
        _t_.start()
        _t_.join(timeout=60)
        self.assertTrue(_done_, "display()'s level walk did not terminate against the "
                                "stack depth limit")
        self.assertEqual(ctrl.df_level, 3)


if __name__ == '__main__':
    unittest.main()
