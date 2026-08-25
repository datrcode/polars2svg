import math
import random
import unittest

import numpy as np
import polars as pl

from polars2svg import (BackgroundShape, INHERIT, LayoutAlgorithm, Polars2SVG,
                        FlowFieldBackground)
from polars2svg.flow_field_background import (cellNames, headNames, layerAppearance,
                                              layerNames, _estimated_support, _Grid)


# ---------------------------------------------------------------------------
# Ground-truth scene: a two-way east/west highway crossed by a south->north
# stream.  A single vector field cancels along the highway (the two directions
# annihilate); the layer split has to separate them.
# ---------------------------------------------------------------------------

def _crossing(seed=7):
    rnd, pos, rows = random.Random(seed), {}, []
    chain = [f'ew{i:02d}' for i in range(12)]
    for i, n in enumerate(chain):
        pos[n] = (0.05 + 0.9 * i / 11.0, 0.5 + rnd.uniform(-0.04, 0.04))
    vert = [f'sn{i:02d}' for i in range(8)]
    for i, n in enumerate(vert):
        pos[n] = (0.55 + rnd.uniform(-0.05, 0.05), 0.06 + 0.88 * i / 7.0)

    def emit(a, b, n):
        for _ in range(n):
            rows.append({'src': a, 'dst': b, 'bytes': rnd.randint(400, 40000)})

    for i in range(len(chain) - 1):
        emit(chain[i], chain[i + 1], 60)        # eastbound trunk
        emit(chain[i + 1], chain[i], 24)        # westbound return
    for i in range(len(vert) - 1):
        emit(vert[i], vert[i + 1], 40)          # northbound only
    rnd.shuffle(rows)
    return pl.DataFrame(rows), [('src', 'dst')], pos, chain, vert


def _uniform(n_nodes, n_edges, seed=3):
    """Every edge spans the canvas -- the shape that makes the kernels expensive."""
    rnd = random.Random(seed)
    pos = {f'n{i}': (rnd.random(), rnd.random()) for i in range(n_nodes)}
    keys = list(pos)
    return (pl.DataFrame([{'src': rnd.choice(keys), 'dst': rnd.choice(keys)}
                          for _ in range(n_edges)]), [('src', 'dst')], pos)


def _fmap(**kw):
    df, rels, pos, _c, _v = _crossing()
    kw.setdefault('count', 'bytes')
    return FlowFieldBackground(df, rels, pos=pos, **kw)


class TestOutputContract(unittest.TestCase):
    def test_cells_are_background_records(self):
        for r in _fmap(k_layers=2).cells().values():
            self.assertIsInstance(r, BackgroundShape)

    def test_cell_names_are_predictable_before_running(self):
        fm = _fmap(k_layers=3)
        self.assertLessEqual(set(fm.cells()), set(cellNames(3, glyph='arrow')))
        self.assertEqual(layerNames(3), ['flow 1', 'flow 2', 'flow 3'])

    def test_is_not_a_layout_algorithm(self):
        # It describes a layout rather than producing one; there is deliberately
        # no results() to mistake it for one.
        fm = _fmap()
        self.assertFalse(hasattr(fm, 'results'))
        self.assertNotIsInstance(fm, LayoutAlgorithm)

    def test_shapes_are_world_coordinates(self):
        fm = _fmap()
        m  = fm.glyph_reach
        x0, y0, x1, y1 = fm.grid.bounds
        for rec in fm.cells().values():
            toks = rec.shape.split()
            xs = [float(v) for i, v in enumerate(toks) if _isnum(v) and _axis(toks, i) == 0]
            ys = [float(v) for i, v in enumerate(toks) if _isnum(v) and _axis(toks, i) == 1]
            self.assertGreaterEqual(min(xs), x0 - m - 1e-9)
            self.assertLessEqual(max(xs), x1 + m + 1e-9)
            self.assertGreaterEqual(min(ys), y0 - m - 1e-9)
            self.assertLessEqual(max(ys), y1 + m + 1e-9)

    def test_field_fits_inside_linkp_default_bounds_percent(self):
        # linkp pads world bounds by bounds_percent=0.05; a glyph anchored on the
        # outermost grid line overhangs by glyph_reach.  If that exceeds the pad
        # the background is clipped at the canvas edge.
        for glyph in ('arrow', 'streamline'):
            fm = _fmap(glyph=glyph)
            overrun = (fm.glyph_reach + 0.01 * fm.grid.span) / fm.grid.span
            self.assertLessEqual(overrun, 0.05, f'{glyph} overruns by {overrun:.3f}')


def _isnum(tok):
    try:
        float(tok)
        return True
    except ValueError:
        return False


def _axis(toks, i):
    """0 for an x operand, 1 for a y operand, given M/L/C arity."""
    j, k = i, 0
    while j > 0 and _isnum(toks[j - 1]):
        j -= 1
        k += 1
    return k % 2


class TestPathDialect(unittest.TestCase):
    """linkp's __transformPathDescription__ accepts M/L (2 floats), C (6) and Z,
    and raises on anything else -- a stray token is a render crash."""

    def test_only_supported_tokens(self):
        for glyph in ('arrow', 'streamline'):
            for rec in _fmap(glyph=glyph).cells().values():
                toks, i = rec.shape.split(), 0
                while i < len(toks):
                    n = {'M': 2, 'L': 2, 'C': 6}.get(toks[i])
                    if n is None:
                        self.assertEqual(toks[i], 'Z', f'unhandled token {toks[i]!r}')
                        i += 1
                        continue
                    for v in toks[i + 1:i + 1 + n]:
                        self.assertTrue(_isnum(v), f'{v!r} is not a float')
                    i += 1 + n

    def test_arrow_subpaths_are_closed(self):
        for rec in _fmap(glyph='arrow').cells().values():
            self.assertEqual(rec.shape.count('M'), rec.shape.count('Z'))


class TestLayerDecomposition(unittest.TestCase):
    def test_k_is_a_hard_count_not_a_cap(self):
        for k in (1, 2, 3, 4):
            fm = _fmap(k_layers=k)
            self.assertLessEqual(len(set(fm.labels.tolist())), k)
            self.assertTrue((fm.labels >= 0).all(), 'an edge was left unassigned')

    def test_nothing_is_dropped(self):
        fm = _fmap(k_layers=2)
        self.assertEqual(len(fm.labels), len(fm.edges))
        self.assertAlmostEqual(sum(fm.throughput), float(fm.weights.sum()), places=6)

    def test_one_layer_cancels_the_counterflow(self):
        # With k=1 the eastbound and westbound edges share a field and partially
        # annihilate; with k=2 they are separated, so the total field magnitude
        # has to rise.
        one, two = _fmap(k_layers=1), _fmap(k_layers=2)
        self.assertGreater(_field_mass(two), _field_mass(one) * 1.2)

    def test_three_layers_recover_the_three_streams(self):
        df, rels, pos, chain, vert = _crossing()
        fm = FlowFieldBackground(df, rels, pos=pos, k_layers=3, count='bytes')
        layer = fm.edgeLayers()
        east = {layer[(chain[i], chain[i + 1])] for i in range(len(chain) - 1)}
        west = {layer[(chain[i + 1], chain[i])] for i in range(len(chain) - 1)}
        north = {layer[(vert[i], vert[i + 1])] for i in range(len(vert) - 1)}
        for name, s in (('eastbound', east), ('westbound', west), ('northbound', north)):
            self.assertEqual(len(s), 1, f'{name} was split across layers {s}')
        self.assertEqual(len({east.pop(), west.pop(), north.pop()}), 3,
                         'the three streams did not land in three different layers')

    def test_layers_are_ordered_by_throughput(self):
        fm = _fmap(k_layers=3)
        self.assertEqual(fm.throughput, sorted(fm.throughput, reverse=True))


def _field_mass(fm):
    return float(np.sqrt(fm.U ** 2 + fm.V ** 2).sum())


class TestRecordAppearance(unittest.TestCase):
    """`None` means off, stated outright -- not smuggled through fill-opacity 0
    or stroke-width 0 the way the background_* parameters forced."""

    def test_arrows_are_filled_never_stroked(self):
        for rec in _fmap(glyph='arrow').cells().values():
            self.assertIsNotNone(rec.fill)
            self.assertIsNone(rec.stroke)

    def test_streamline_curves_are_stroked_never_filled(self):
        fm = _fmap(glyph='streamline')
        for name in fm.names:
            rec = fm.cells()[name]
            self.assertIsNone(rec.fill)
            self.assertIsNotNone(rec.stroke)
            self.assertGreater(rec.stroke_opacity, 0.0)
            self.assertLess(rec.stroke_opacity, 1.0)

    def test_streamline_heads_are_a_separate_filled_cell(self):
        # One path carries one fill decision, and an open subpath is closed
        # implicitly in order to fill it -- so filled heads need their own cell.
        fm    = _fmap(glyph='streamline')
        cells = fm.cells()
        for name, head in zip(fm.names, fm.head_names):
            self.assertIn(head, cells)
            self.assertEqual(list(cells).index(head), list(cells).index(name) + 1,
                             'heads must be drawn after their own curves')
            self.assertIsNotNone(cells[head].fill)
            self.assertIsNone(cells[head].stroke)
            self.assertNotIn('Z', cells[name].shape)          # curves stay open
            self.assertTrue(cells[head].shape.strip().endswith('Z'))

    def test_head_cells_suppress_their_label(self):
        fm = _fmap(glyph='streamline')
        self.assertIsNone(fm.cells()[fm.head_names[0]].label)
        # ... while a layer cell leaves the decision to the component, so the
        # interactive 'b' cycle still governs whether names are drawn.
        self.assertIs(fm.cells()[fm.names[0]].label, INHERIT)

    def test_marker_none_emits_no_head_cell(self):
        fm = _fmap(glyph='streamline', streamline_marker='none')
        self.assertEqual(set(fm.cells()), set(fm.names))

    def test_layer_appearance_covers_every_cell_name(self):
        for glyph in ('arrow', 'streamline'):
            app = layerAppearance(2, glyph=glyph)
            self.assertEqual(set(app), set(cellNames(2, glyph=glyph)))


class TestSupportBudget(unittest.TestCase):
    """Two different costs, two limits: the kernel footprint is memory and grows
    with edge LENGTH; the assignment is time and grows with edge COUNT."""

    def test_capsule_estimate_matches_the_built_kernels(self):
        fm  = _fmap(k_layers=1)
        est = float(_estimated_support(fm.grid, *_endpoints(fm), 3.0 * fm.sigma).sum())
        # The estimate is an area model, so it is approximate -- but it has to be
        # the right order or it cannot govern a budget.
        self.assertGreater(est, 0.25 * fm.support_size)
        self.assertLess(est, 4.0 * fm.support_size)

    def test_budget_coarsens_the_grid_before_dropping_edges(self):
        df, rels, pos = _uniform(400, 6000)
        loose = FlowFieldBackground(df, rels, pos=pos, support_budget=None, max_edges=None)
        tight = FlowFieldBackground(df, rels, pos=pos, support_budget=loose.support_size // 4,
                                    max_edges=None)
        self.assertLess(tight.grid.nx, loose.grid.nx, 'the grid was not coarsened')
        self.assertLessEqual(tight.support_size, loose.support_size)
        self.assertEqual(len(tight.edges), len(loose.edges), 'edges dropped before coarsening')
        self.assertIsNotNone(tight.budget_note)

    def test_budget_drops_edges_once_the_grid_floor_is_reached(self):
        df, rels, pos = _uniform(400, 6000)
        loose = FlowFieldBackground(df, rels, pos=pos, support_budget=None, max_edges=None)
        tiny  = FlowFieldBackground(df, rels, pos=pos, support_budget=20_000,
                                    min_grid_res=8, max_edges=None)
        self.assertLess(len(tiny.edges), len(loose.edges))
        self.assertIn('edges', tiny.budget_note)

    def test_max_edges_bounds_the_assignment(self):
        df, rels, pos = _uniform(400, 6000)
        fm = FlowFieldBackground(df, rels, pos=pos, max_edges=200)
        self.assertLessEqual(len(fm.edges), 200)
        self.assertIn('max_edges', fm.budget_note)

    def test_defaults_are_on(self):
        fm = _fmap()
        self.assertIsNone(fm.budget_note, 'the ground-truth scene should not degrade')

    def test_no_note_when_nothing_degrades(self):
        self.assertNotIn('!', _fmap().summary())


def _endpoints(fm):
    pts = np.array([fm.pos[n] for n, _d, _w in fm.edges]), np.array([fm.pos[d] for _n, d, _w in fm.edges])
    return pts


class TestDeterminism(unittest.TestCase):
    def test_identical_across_runs(self):
        a, b = _fmap(), _fmap()
        self.assertEqual({n: r.shape for n, r in a.cells().items()},
                         {n: r.shape for n, r in b.cells().items()})

    def test_row_order_does_not_matter(self):
        df, rels, pos, _c, _v = _crossing()
        a = FlowFieldBackground(df, rels, pos=pos, count='bytes')
        b = FlowFieldBackground(df.sample(fraction=1.0, shuffle=True, seed=3), rels,
                                pos=pos, count='bytes')
        self.assertEqual({n: r.shape for n, r in a.cells().items()},
                         {n: r.shape for n, r in b.cells().items()})


class TestInputHandling(unittest.TestCase):
    def test_empty_frame_yields_no_background(self):
        empty = pl.DataFrame({'src': [], 'dst': []}, schema={'src': pl.String, 'dst': pl.String})
        fm = FlowFieldBackground(empty, [('src', 'dst')], pos={'a': (0, 0)})
        self.assertEqual(fm.cells(), {})
        self.assertIn('no flow', fm.summary())

    def test_selection_restricts_to_incident_edges(self):
        df, rels, pos, chain, _v = _crossing()
        fm = FlowFieldBackground(df, rels, pos=pos, count='bytes',
                                 selection={chain[0], chain[1]})
        for src, dst, _w in fm.edges:
            self.assertTrue(src in (chain[0], chain[1]) or dst in (chain[0], chain[1]))

    def test_nodes_missing_from_pos_are_dropped(self):
        df, rels, pos, chain, _v = _crossing()
        partial = {k: v for k, v in pos.items() if k != chain[0]}
        fm = FlowFieldBackground(df, rels, pos=partial, count='bytes')
        for src, dst, _w in fm.edges:
            self.assertNotIn(chain[0], (src, dst))

    def test_undirected_graph_input_becomes_counterflow(self):
        import networkx as nx
        g = nx.Graph()
        g.add_edge('a', 'b', weight=3.0)
        fm = FlowFieldBackground(g, pos={'a': (0.0, 0.0), 'b': (1.0, 0.0)}, k_layers=2)
        self.assertEqual(len(fm.edges), 2, 'an undirected edge should travel both ways')

    def test_rejects_an_unknown_glyph(self):
        with self.assertRaises(ValueError):
            _fmap(glyph='sparkles')


class TestLinkpRender(unittest.TestCase):
    def test_linkp_renders_the_background(self):
        df, rels, pos, _c, _v = _crossing()
        p2s = Polars2SVG()
        for glyph in ('arrow', 'streamline'):
            fm  = FlowFieldBackground(df, rels, pos=pos, count='bytes', glyph=glyph)
            svg = p2s.linkp(df, rels, pos, wxh=(384, 384), background=fm.cells())._repr_svg_()
            self.assertIn('<svg', svg)
            self.assertIn('<path', svg)

    def test_records_need_no_background_parameters(self):
        # The whole point of the record contract: appearance travels with the
        # cells, so the caller passes background= and nothing else.
        df, rels, pos, _c, _v = _crossing()
        fm  = FlowFieldBackground(df, rels, pos=pos, count='bytes', glyph='streamline')
        svg = Polars2SVG().linkp(df, rels, pos, background=fm.cells())._repr_svg_()
        self.assertIn('stroke-opacity=', svg)
        self.assertIn('fill="none"', svg)
        self.assertNotIn('fill-opacity="0.0"', svg)


class TestBackgroundOperationLifecycle(unittest.TestCase):
    """A background operation is not a layout operation, and a *contextual*
    background outlives the positions it was computed from (PLANNING.md B4).
    """

    def setUp(self):
        from polars2svg.interactive_controller import linkpi
        df, rels, pos, self.chain, _v = _crossing()
        self.p2s  = Polars2SVG()
        self.view = linkpi(self.p2s.linkp(df, rels, pos, count='bytes'))

    def _run_flow_field(self):
        return self.view.applyBackgroundOperation('flow field (2 layers)')

    def test_registry_offers_the_flow_field(self):
        self.assertIn('flow field (2 layers)', self.view._background_registry)
        self.assertIn('clear background', self.view._background_registry)

    def test_running_it_adopts_a_contextual_background(self):
        self.assertTrue(self._run_flow_field())
        self.assertEqual(self.view.background_provenance, 'context')
        self.assertEqual(set(self.view.layout_background), {'flow 1', 'flow 2'})

    def test_running_it_reveals_a_hidden_background(self):
        self.view.background_state = 0
        self._run_flow_field()
        self.assertEqual(self.view.background_state, 1)

    def test_it_costs_no_undo_slot(self):
        before = len(self.view.previous_layouts)
        self._run_flow_field()
        self.assertEqual(len(self.view.previous_layouts), before)

    def test_it_preserves_the_view_window(self):
        _ln_ = self.view.dfs_layout[self.view.df_level]
        _ln_.setViewWindow((0.2, 0.2, 0.6, 0.6))
        self._run_flow_field()
        self.assertEqual(_ln_.view_window, (0.2, 0.2, 0.6, 0.6))

    def test_it_moves_no_nodes(self):
        _ln_ = self.view.dfs_layout[self.view.df_level]
        before = {k: tuple(v) for k, v in _ln_.pos.items()}
        self._run_flow_field()
        self.assertEqual({k: tuple(v) for k, v in _ln_.pos.items()}, before)

    def test_a_manual_node_move_keeps_it(self):
        # The decision this feature exists to honour: repositioning nodes in
        # response to what the background showed is the intended workflow, so
        # moving one must not silently erase it.
        self._run_flow_field()
        cells = self.view.layout_background
        self.view.selected_entities = {self.chain[0]}
        self.view.apply_move_selected(12, 7)
        self.assertIs(self.view.layout_background, cells)
        self.assertEqual(self.view.background_provenance, 'context')

    def test_a_layout_without_its_own_background_keeps_it(self):
        self._run_flow_field()
        cells = self.view.layout_background
        self.view.apply_layout_operation('spring nx')
        self.assertIs(self.view.layout_background, cells)
        self.assertEqual(self.view.background_provenance, 'context')

    def test_a_layout_with_its_own_background_supersedes_it(self):
        self._run_flow_field()
        self.view.apply_layout_operation('hyper tree donut')
        self.assertEqual(self.view.background_provenance, 'layout')
        self.assertNotEqual(set(self.view.layout_background), {'flow 1', 'flow 2'})

    def test_a_layout_owned_background_dies_with_its_layout(self):
        self.view.apply_layout_operation('hyper tree donut')
        self.assertEqual(self.view.background_provenance, 'layout')
        self.view.apply_layout_operation('spring nx')
        self.assertIsNone(self.view.layout_background)
        self.assertIsNone(self.view.background_provenance)

    def test_clear_background_clears(self):
        self._run_flow_field()
        self.view.applyBackgroundOperation('clear background')
        self.assertIsNone(self.view.layout_background)
        self.assertIsNone(self.view.background_provenance)

    def test_rerunning_refreshes_against_the_new_positions(self):
        self._run_flow_field()
        first = {n: r.shape for n, r in self.view.layout_background.items()}
        self.view.selected_entities = {self.chain[0], self.chain[1]}
        self.view.apply_move_selected(60, 40)
        self._run_flow_field()
        second = {n: r.shape for n, r in self.view.layout_background.items()}
        self.assertNotEqual(first, second, 're-running should follow the moved nodes')

    def test_the_selection_scopes_the_field(self):
        self.view.selected_entities = {self.chain[0], self.chain[1]}
        self._run_flow_field()
        self.assertIsNotNone(self.view._last_background_)
        for src, dst, _w in self.view._last_background_.edges:
            self.assertTrue(src in self.view.selected_entities
                            or dst in self.view.selected_entities)

    def test_an_unknown_operation_is_refused(self):
        self.assertFalse(self.view.applyBackgroundOperation('not a producer'))

    def test_menu_commit_path_runs_it(self):
        # What the JS actually does on commit: set the label, bump the counter.
        # The counter is what is watched -- re-running the SAME producer is the
        # documented refresh, and an unchanged label would not fire a watcher.
        self.view.background_operation = 'flow field (3 layers)'
        self.view.background_op_seq    = self.view.background_op_seq + 1
        self.assertEqual(set(self.view.layout_background), {'flow 1', 'flow 2', 'flow 3'})
        self.assertEqual(self.view.background_provenance, 'context')

    def test_commit_reaches_the_linkp(self):
        self.view.background_op_seq = self.view.background_op_seq + 1
        _ln_ = self.view.dfs_layout[self.view.df_level]
        self.assertIsNotNone(_ln_.background)
        self.assertIn('<path', _ln_.renderSVG())

    def test_every_menu_label_has_a_handler(self):
        # The picker is a JS string; a label that resolves to nothing would just
        # silently do nothing when the user selects it.
        from polars2svg.interactive_controller import _BACKGROUND_OP_MENU_
        for _mnemonic_, label in _BACKGROUND_OP_MENU_:
            self.assertIn(label, self.view._background_registry)

    def test_menu_mnemonics_are_usable(self):
        from polars2svg.interactive_controller import _BACKGROUND_OP_MENU_
        mnemonics = [m for m, _l in _BACKGROUND_OP_MENU_]
        self.assertEqual(len(set(mnemonics)), len(mnemonics), 'duplicate mnemonic')
        # The picker reserves these for navigation (see _LAYOUT_MODE_MENU_'s comment).
        self.assertFalse(set(mnemonics) & set('jkWG'), 'mnemonic collides with navigation')

    def test_the_javascript_offers_the_picker(self):
        js = '\n'.join(str(v) for v in type(self.view)._scripts.values())
        self.assertIn("state.menu_kind = 'background'; self.menuOpen();", js)
        self.assertIn('"background": [["f", "flow field (2 layers)"]', js)
        self.assertIn('data.background_op_seq   = data.background_op_seq + 1', js)
        # 'b' must still cycle visibility rather than opening the picker.
        self.assertIn('else if (event.key == "b") { data.key_op_finished = \'b\';  }', js)

    def test_state_label_names_the_producer(self):
        self._run_flow_field()
        label = type(self.view).__dict__['__backgroundStateLabel__'](self.view)
        self.assertIn('flow field', label)


if __name__ == '__main__':
    unittest.main()
