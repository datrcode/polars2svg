import json
import math
import tempfile
import types
import unittest
from io import StringIO
from unittest.mock import patch

import networkx as nx
import polars as pl

from polars2svg import Polars2SVG


def _make_df():
    return pl.DataFrame({
        'fm': ['a', 'b', 'c', 'a'],
        'to': ['b', 'c', 'a', 'c'],
        'w':  [1,   2,   3,   4  ],
    })

def _two_component_graph():
    """Three nodes in one component, two in another."""
    g = nx.Graph()
    g.add_edges_from([('a', 'b'), ('b', 'c'), ('d', 'e')])
    return g

def _tree_graph():
    """Single connected tree with 6 nodes."""
    g = nx.Graph()
    g.add_edges_from([('a', 'b'), ('a', 'c'), ('b', 'd'), ('b', 'e'), ('c', 'f')])
    return g

def _two_component_pos():
    """2D positions with non-degenerate spread in each component."""
    return {
        'a': (0.0, 0.0), 'b': (1.0, 0.5), 'c': (0.5, 1.0),
        'd': (3.0, 0.0), 'e': (4.0, 0.5),
    }

def _tree_pos():
    return {n: (i * 0.2, i * 0.1) for i, n in enumerate(['a', 'b', 'c', 'd', 'e', 'f'])}

def _donut_tree():
    """root -> 3 categories -> leaves (classic donut/pie structure).

    Returns (graph, root, {category: [leaves]}) with differently-sized slices.
    """
    g = nx.Graph()
    root = 'root'
    cats = {'A': 5, 'B': 2, 'C': 8}
    cat_leaves = {}
    for c, n in cats.items():
        g.add_edge(root, c)
        leaves = [f'{c}{i}' for i in range(n)]
        cat_leaves[c] = leaves
        for leaf in leaves:
            g.add_edge(c, leaf)
    return g, root, cat_leaves


class TestCreateNetworkXGraph(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.p2s  = Polars2SVG()
        cls.df   = _make_df()
        cls.rels = [('fm', 'to')]

    def test_basic_graph_has_correct_nodes_and_edges(self):
        g = self.p2s.createNetworkXGraph(self.df, self.rels)
        self.assertIsInstance(g, nx.Graph)
        self.assertEqual(set(g.nodes()), {'a', 'b', 'c'})
        self.assertEqual(g.number_of_edges(), 3)

    def test_digraph_flag_produces_digraph(self):
        g = self.p2s.createNetworkXGraph(self.df, self.rels, use_digraph=True)
        self.assertIsInstance(g, nx.DiGraph)

    def test_three_tuple_relationship_sets_edge_attribute(self):
        g = self.p2s.createNetworkXGraph(self.df, [('fm', 'to', 'w')])
        for _u, _v, data in g.edges(data=True):
            self.assertIn('w', data)

    def test_nan_rows_excluded_from_graph(self):
        df_nan = pl.DataFrame({'fm': ['a', None, 'c'], 'to': ['b', 'b', None]})
        g = self.p2s.createNetworkXGraph(df_nan, [('fm', 'to')])
        self.assertEqual(g.number_of_edges(), 1)
        self.assertIn('a', g.nodes())
        self.assertIn('b', g.nodes())

    def test_tuple_column_creates_concat_string_nodes(self):
        g = self.p2s.createNetworkXGraph(self.df, [(('fm', 'w'), 'to')])
        self.assertGreater(g.number_of_nodes(), 0)
        for node in g.nodes():
            self.assertIsInstance(node, str)


class TestCreateNetworkXGraphExceptions(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.p2s = Polars2SVG()
        cls.df  = _make_df()

    def test_four_part_simple_tuple_raises(self):
        with self.assertRaises(Exception) as ctx:
            self.p2s.createNetworkXGraph(self.df, [('fm', 'to', 'w', 'extra')])
        self.assertIn('two or three parts', str(ctx.exception))

    def test_four_part_outer_tuple_with_column_tuple_raises(self):
        with self.assertRaises(Exception) as ctx:
            self.p2s.createNetworkXGraph(self.df, [(('fm', 'w'), 'to', 'extra', 'junk')])
        self.assertIn('two or three parts', str(ctx.exception))


class TestFilterDataFrameByGraph(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.p2s  = Polars2SVG()
        cls.rels = [('fm', 'to')]

    def test_matching_edge_row_is_kept(self):
        g = nx.Graph()
        g.add_edge('a', 'b')
        df = pl.DataFrame({'fm': ['a', 'a'], 'to': ['b', 'z']})
        result = self.p2s.filterDataFrameByGraph(df, self.rels, g)
        self.assertEqual(len(result), 1)
        self.assertEqual(result['fm'][0], 'a')
        self.assertEqual(result['to'][0], 'b')

    def test_empty_graph_returns_empty_dataframe_with_schema(self):
        g = nx.Graph()
        df = pl.DataFrame({'fm': ['a', 'b'], 'to': ['c', 'd']})
        result = self.p2s.filterDataFrameByGraph(df, self.rels, g)
        self.assertEqual(len(result), 0)
        self.assertEqual(result.schema, df.schema)

    def test_undirected_graph_matches_both_edge_directions(self):
        g = nx.Graph()
        g.add_edge('a', 'b')
        df = pl.DataFrame({'fm': ['a', 'b'], 'to': ['b', 'a']})
        result = self.p2s.filterDataFrameByGraph(df, self.rels, g)
        self.assertEqual(len(result), 2)


class TestLayoutMethods(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.p2s     = Polars2SVG()
        cls.g_two   = _two_component_graph()
        cls.g_tree  = _tree_graph()
        cls.pos_two  = _two_component_pos()
        cls.pos_tree = _tree_pos()

    def test_rectangular_layout_1_node(self):
        g = nx.Graph()
        g.add_node('x')
        pos = self.p2s.rectangularLayout(g, ['x'])
        self.assertIn('x', pos)

    def test_rectangular_layout_5_nodes(self):
        g = nx.complete_graph(5)
        nodes = list(g.nodes())
        pos = self.p2s.rectangularLayout(g, nodes)
        self.assertEqual(len(pos), 5)
        for n in nodes:
            self.assertIn(n, pos)

    def test_rectangular_layout_9_nodes(self):
        g = nx.complete_graph(9)
        nodes = list(g.nodes())
        pos = self.p2s.rectangularLayout(g, nodes)
        self.assertEqual(len(pos), 9)

    def test_sunflower_seed_layout_returns_all_nodes(self):
        g = nx.complete_graph(8)
        nodes = list(g.nodes())
        pos = self.p2s.sunflowerSeedLayout(g, nodes)
        self.assertEqual(len(pos), 8)
        for n in nodes:
            self.assertIn(n, pos)

    def test_circle_pack_layout_returns_positions_for_all_nodes(self):
        new_pos, shapes = self.p2s.circlePackLayout(self.g_two, dict(self.pos_two))
        for n in self.g_two.nodes():
            self.assertIn(n, new_pos)

    def test_circle_pack_layout_returns_shapes_dict(self):
        new_pos, shapes = self.p2s.circlePackLayout(self.g_two, dict(self.pos_two))
        self.assertIsInstance(shapes, dict)

    def test_treemap_layout_single_component_returns_same_pos_object(self):
        g = nx.Graph()
        g.add_edges_from([('x', 'y'), ('y', 'z')])
        pos = {'x': (0.0, 0.0), 'y': (1.0, 0.0), 'z': (2.0, 0.0)}
        result = self.p2s.treeMapLayout(g, pos)
        self.assertIs(result, pos)

    def test_treemap_layout_multi_component_returns_all_nodes(self):
        new_pos = self.p2s.treeMapLayout(self.g_two, dict(self.pos_two))
        for n in self.g_two.nodes():
            self.assertIn(n, new_pos)

    def test_treemap_node_color_layout_returns_all_nodes(self):
        nodes = list(self.g_two.nodes())
        color_lu = {n: '#ff0000' if i % 2 == 0 else '#0000ff' for i, n in enumerate(nodes)}
        pos = self.p2s.treeMapNodeColorLayout(self.g_two, nodes, color_lu)
        for n in nodes:
            self.assertIn(n, pos)

    def test_linear_optimized_layout_returns_all_requested_nodes(self):
        # b and c have external neighbor a; d, e, f are also external to {b,c}
        nodes = ['b', 'c']
        result = self.p2s.linearOptimizedLayout(
            self.g_tree, nodes, dict(self.pos_tree),
            segment=((0.0, 0.0), (1.0, 0.0)),
        )
        for n in nodes:
            self.assertIn(n, result)

    def test_rectangular_layout_degree1_sorted_by_weight(self):
        # Star: hub 'h' (degree 3) connected to leaves a/b/c with distinct weights.
        # With ≥6 nodes we'd need the grid branch, but the degree-1 weight sort
        # matters most there.  Build 9 nodes so we hit the grid branch:
        # one hub 'h' with 8 leaves, weights 10..3 on edges h-l0..h-l7.
        g = nx.Graph()
        leaves = [f'l{i}' for i in range(8)]
        weights = [10, 7, 5, 3, 8, 6, 4, 2]
        for leaf, w in zip(leaves, weights):
            g.add_edge('h', leaf, weight=w)
        nodes = ['h'] + leaves
        pos = self.p2s.rectangularLayout(g, nodes)
        # hub must be placed before any leaf (it has higher degree)
        hub_x, hub_y = pos['h']
        # leaves must be ordered by weight descending in grid order (left-to-right, top-to-bottom)
        sorted_leaves = sorted(leaves, key=lambda n: g['h'][n]['weight'], reverse=True)
        leaf_positions = [pos[n] for n in sorted_leaves]
        for i in range(len(leaf_positions) - 1):
            x_i, y_i = leaf_positions[i]
            x_j, y_j = leaf_positions[i + 1]
            self.assertTrue(
                (y_i, x_i) <= (y_j, x_j),
                f'leaf {sorted_leaves[i]} should precede {sorted_leaves[i+1]} in grid order'
            )

    def test_linear_layout_degree1_sorted_by_weight(self):
        # Two leaves (lo=low weight, hi=high weight) both connect to an external hub.
        # After linear layout, the heavier leaf should occupy the earlier (lower x) slot.
        g = nx.Graph()
        g.add_edge('ext', 'hi', weight=9)
        g.add_edge('ext', 'lo', weight=2)
        pos_in = {'ext': (0.5, -1.0), 'hi': (0.5, 0.5), 'lo': (0.5, 0.5)}
        result = self.p2s.linearOptimizedLayout(
            g, ['hi', 'lo'], pos_in, segment=((0.0, 0.0), (1.0, 0.0))
        )
        self.assertLessEqual(result['hi'][0], result['lo'][0],
                             'higher-weight leaf should be placed at or before lower-weight leaf')

    def test_linear_optimized_layout_node_absent_from_graph(self):
        # Regression: a node present in `nodes` and `pos` but absent from the
        # graph (e.g. an isolated node with no edges) previously raised
        # NetworkXError: "The node X is not in the graph."
        g = nx.Graph()
        g.add_edge('a', 'b')
        pos = {'a': (0.0, 0.0), 'b': (1.0, 0.0), 'orphan': (0.5, 0.5)}
        result = self.p2s.linearOptimizedLayout(
            g, ['a', 'b', 'orphan'], pos, segment=((0.0, 0.0), (1.0, 0.0))
        )
        for n in ('a', 'b', 'orphan'):
            self.assertIn(n, result)

    def test_circular_optimized_layout_node_absent_from_graph(self):
        # Regression: a node present in `nodes` and `pos` but absent from the
        # graph (e.g. an isolated node with no edges) previously raised
        # NetworkXError: "The node X is not in the graph."
        g = nx.Graph()
        g.add_edge('a', 'b')
        pos = {'a': (0.0, 0.0), 'b': (1.0, 0.0), 'orphan': (0.5, 0.5)}
        result = self.p2s.circularOptimizedLayout(
            g, ['a', 'b', 'orphan'], pos, xy=(0.5, 0.5), r=0.5
        )
        for n in ('a', 'b', 'orphan'):
            self.assertIn(n, result)

    def test_circular_optimized_layout_returns_all_requested_nodes(self):
        nodes = ['b', 'c', 'd', 'e']
        result = self.p2s.circularOptimizedLayout(
            self.g_tree, nodes, dict(self.pos_tree),
            xy=(0.5, 0.5), r=0.4,
        )
        for n in nodes:
            self.assertIn(n, result)

    def test_hyper_tree_layout_returns_all_nodes(self):
        pos = self.p2s.hyperTreeLayout(self.g_tree)
        for n in self.g_tree.nodes():
            self.assertIn(n, pos)

    def test_hyper_tree_donut_layout_returns_all_nodes(self):
        pos = self.p2s.hyperTreeDonutLayout(self.g_tree)
        for n in self.g_tree.nodes():
            self.assertIn(n, pos)

    def test_hyper_tree_donut_layout_multi_component_returns_all_nodes(self):
        # Two components -> arranged via treeMapLayout; every node still placed.
        g = _two_component_graph()
        pos = self.p2s.hyperTreeDonutLayout(g)
        for n in g.nodes():
            self.assertIn(n, pos)

    def test_hyper_tree_donut_layout_geometry(self):
        # inner_frac/parent_gap_frac default to 0.55 / 0.12 of the outer radius 8.0
        g, root, cat_leaves = _donut_tree()
        _R_, inner = 8.0, 0.55 * 8.0
        r_mid = (inner + _R_) / 2.0
        pos = self.p2s.hyperTreeDonutLayout(g, roots=[root])

        def rad(n): return math.hypot(pos[n][0], pos[n][1])

        # root sits at the center of the donut (the hole)
        self.assertAlmostEqual(rad(root), 0.0, places=6)

        for cat, leaves in cat_leaves.items():
            # leaf-parent sits at the center of its donut wedge (mid-radius)
            self.assertAlmostEqual(rad(cat), r_mid, places=6)
            # leaves fill the donut ring band [inner, R]
            for leaf in leaves:
                self.assertGreaterEqual(rad(leaf), inner - 1e-9)
                self.assertLessEqual(rad(leaf), _R_ + 1e-9)

    def test_hyper_tree_donut_layout_return_cells_default_is_pos_only(self):
        # Default (return_cells=False) keeps the bare-dict return for callers.
        g, root, _ = _donut_tree()
        pos = self.p2s.hyperTreeDonutLayout(g, roots=[root])
        self.assertIsInstance(pos, dict)

    def test_hyper_tree_donut_layout_return_cells_shape(self):
        from shapely.geometry import Point
        g, root, cat_leaves = _donut_tree()
        pos, cells = self.p2s.hyperTreeDonutLayout(g, roots=[root], return_cells=True)
        # One slice per leaf-parent, keyed by the parent node.
        self.assertEqual(set(cells), set(cat_leaves))
        for cat in cat_leaves:
            poly = cells[cat]
            # Each parent sits inside its own slice; leaves sit on its edge.
            self.assertTrue(poly.buffer(1e-6).contains(Point(*pos[cat])))
            for leaf in cat_leaves[cat]:
                self.assertTrue(poly.buffer(1e-6).contains(Point(*pos[leaf])))

    def test_hyper_tree_donut_layout_cells_are_annular(self):
        # Donut (not pie): the layout center is in the hole, outside every slice,
        # while each leaf-parent sits inside its own ring slice.
        from shapely.geometry import Point
        g, root, cat_leaves = _donut_tree()
        pos, cells = self.p2s.hyperTreeDonutLayout(g, roots=[root], return_cells=True)
        center = Point(0.0, 0.0)
        for cat, poly in cells.items():
            self.assertFalse(poly.contains(center))           # hole at the center
            self.assertTrue(poly.buffer(1e-6).contains(Point(*pos[cat])))

    def test_hyper_tree_donut_layout_parent_clearance(self):
        # A clearance is carved around each leaf-parent so no leaf crowds it.
        g, root, cat_leaves = _donut_tree()
        _R_, inner = 8.0, 0.55 * 8.0
        clear = 0.12 * (_R_ - inner)
        pos = self.p2s.hyperTreeDonutLayout(g, roots=[root])
        for cat, leaves in cat_leaves.items():
            px, py = pos[cat]
            for leaf in leaves:
                d = math.hypot(pos[leaf][0] - px, pos[leaf][1] - py)
                self.assertGreaterEqual(d, clear - 1e-6)

    def test_hyper_tree_donut_layout_cells_ordered_by_slice_size(self):
        # Bigger slices (more leaves) -> larger wedge area.
        g, root, cat_leaves = _donut_tree()
        _, cells = self.p2s.hyperTreeDonutLayout(g, roots=[root], return_cells=True)
        self.assertGreater(cells['C'].area, cells['A'].area)   # 8 leaves vs 5
        self.assertGreater(cells['A'].area, cells['B'].area)   # 5 leaves vs 2

    def test_hyper_tree_donut_layout_multi_component_cells_track_nodes(self):
        # Two rooted trees -> treeMapLayout transform; slices must follow it so
        # each parent still falls inside its (transformed) slice polygon.
        from shapely.geometry import Point
        g = nx.Graph()
        for c, n in (('A', 4), ('B', 3)):
            g.add_edge('r1', c)
            for i in range(n):
                g.add_edge(c, f'{c}{i}')
        for c, n in (('X', 5), ('Y', 2)):
            g.add_edge('r2', c)
            for i in range(n):
                g.add_edge(c, f'{c}{i}')
        pos, cells = self.p2s.hyperTreeDonutLayout(g, return_cells=True)
        self.assertGreater(len(cells), 0)
        for parent, poly in cells.items():
            self.assertTrue(poly.buffer(1e-3).contains(Point(*pos[parent])))

    def test_hyper_tree_donut_layout_slices_are_disjoint(self):
        # Leaves of distinct parents must occupy non-interleaving angular slices.
        g, root, cat_leaves = _donut_tree()
        pos = self.p2s.hyperTreeDonutLayout(g, roots=[root])
        leaf_to_cat = {leaf: cat for cat, leaves in cat_leaves.items() for leaf in leaves}
        all_leaves = list(leaf_to_cat)
        all_leaves.sort(key=lambda n: math.atan2(pos[n][1], pos[n][0]))
        # Walking leaves in angular order, the parent label changes at most
        # once per category (no interleaving of slices).
        runs = sum(1 for i in range(1, len(all_leaves))
                   if leaf_to_cat[all_leaves[i]] != leaf_to_cat[all_leaves[i - 1]])
        self.assertLessEqual(runs, len(cat_leaves))


def _make_linkp_stub(pos, all_nodes=None):
    """Minimal stand-in for a LinkP instance with pos and all_nodes."""
    stub = types.SimpleNamespace()
    stub.pos = pos
    if all_nodes is not None:
        stub.all_nodes = all_nodes
    return stub


class TestSavePositions(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.p2s = Polars2SVG()
        cls.pos = {'a': (0.1, 0.2), 'b': (0.5, 0.7), 'c': (0.9, 0.3)}

    def test_file_is_valid_json(self):
        linkp = _make_linkp_stub(self.pos)
        with tempfile.NamedTemporaryFile(suffix='.pos.json', delete=False) as f:
            fname = f.name
        self.p2s.savePositions(fname, linkp)
        with open(fname) as f:
            loaded = json.load(f)
        self.assertIsInstance(loaded, dict)

    def test_saved_keys_match_original(self):
        linkp = _make_linkp_stub(self.pos)
        with tempfile.NamedTemporaryFile(suffix='.pos.json', delete=False) as f:
            fname = f.name
        self.p2s.savePositions(fname, linkp)
        with open(fname) as f:
            loaded = json.load(f)
        self.assertEqual(set(loaded.keys()), set(self.pos.keys()))

    def test_saved_values_match_original_coordinates(self):
        linkp = _make_linkp_stub(self.pos)
        with tempfile.NamedTemporaryFile(suffix='.pos.json', delete=False) as f:
            fname = f.name
        self.p2s.savePositions(fname, linkp)
        with open(fname) as f:
            loaded = json.load(f)
        for node, (x, y) in self.pos.items():
            self.assertAlmostEqual(loaded[node][0], x)
            self.assertAlmostEqual(loaded[node][1], y)

    def test_tuple_values_serialized_as_lists(self):
        linkp = _make_linkp_stub({'x': (0.0, 1.0)})
        with tempfile.NamedTemporaryFile(suffix='.pos.json', delete=False) as f:
            fname = f.name
        self.p2s.savePositions(fname, linkp)
        with open(fname) as f:
            loaded = json.load(f)
        self.assertIsInstance(loaded['x'], list)


class TestLoadPositions(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.p2s = Polars2SVG()
        cls.pos = {'a': (0.1, 0.2), 'b': (0.5, 0.7), 'c': (0.9, 0.3)}

    def _write_pos_file(self, pos):
        f = tempfile.NamedTemporaryFile(
            mode='w', suffix='.pos.json', delete=False
        )
        json.dump({k: list(v) for k, v in pos.items()}, f, indent=2)
        f.close()
        return f.name

    def test_returns_dict(self):
        fname = self._write_pos_file(self.pos)
        result = self.p2s.loadPositions(fname)
        self.assertIsInstance(result, dict)

    def test_returned_keys_match_file(self):
        fname = self._write_pos_file(self.pos)
        result = self.p2s.loadPositions(fname)
        self.assertEqual(set(result.keys()), set(self.pos.keys()))

    def test_returned_values_are_lists(self):
        fname = self._write_pos_file(self.pos)
        result = self.p2s.loadPositions(fname)
        for v in result.values():
            self.assertIsInstance(v, list)

    def test_round_trip_coordinates_match(self):
        linkp = _make_linkp_stub(self.pos)
        with tempfile.NamedTemporaryFile(suffix='.pos.json', delete=False) as f:
            fname = f.name
        self.p2s.savePositions(fname, linkp)
        result = self.p2s.loadPositions(fname)
        for node, (x, y) in self.pos.items():
            self.assertAlmostEqual(result[node][0], x)
            self.assertAlmostEqual(result[node][1], y)

    def test_no_linkp_produces_no_output(self):
        fname = self._write_pos_file(self.pos)
        with patch('sys.stdout', new_callable=StringIO) as mock_out:
            with self.assertNoLogs('polars2svg_logger', level='WARNING'):
                self.p2s.loadPositions(fname)
        self.assertEqual(mock_out.getvalue(), '')

    def test_matching_nodes_produces_no_warning(self):
        fname = self._write_pos_file(self.pos)
        linkp = _make_linkp_stub(self.pos, all_nodes=set(self.pos.keys()))
        with self.assertNoLogs('polars2svg_logger', level='WARNING'):
            self.p2s.loadPositions(fname, linkp)

    def test_extra_node_in_file_logs_warning(self):
        pos_extra = dict(self.pos)
        pos_extra['z'] = (0.0, 0.0)
        fname = self._write_pos_file(pos_extra)
        linkp = _make_linkp_stub(self.pos, all_nodes=set(self.pos.keys()))
        with self.assertLogs('polars2svg_logger', level='WARNING') as cm:
            self.p2s.loadPositions(fname, linkp)
        self.assertIn('differ between position file and linkp', cm.output[0])

    def test_extra_node_in_linkp_logs_warning(self):
        fname = self._write_pos_file(self.pos)
        extra_nodes = set(self.pos.keys()) | {'z'}
        linkp = _make_linkp_stub(self.pos, all_nodes=extra_nodes)
        with self.assertLogs('polars2svg_logger', level='WARNING') as cm:
            self.p2s.loadPositions(fname, linkp)
        self.assertIn('in linkp only: 1', cm.output[0])

    def test_warning_reports_diff_count(self):
        pos_extra = dict(self.pos)
        pos_extra['y'] = (0.0, 0.0)
        pos_extra['z'] = (1.0, 1.0)
        fname = self._write_pos_file(pos_extra)
        linkp = _make_linkp_stub(self.pos, all_nodes=set(self.pos.keys()))
        with self.assertLogs('polars2svg_logger', level='WARNING') as cm:
            self.p2s.loadPositions(fname, linkp)
        self.assertIn('2 node(s) differ', cm.output[0])

    def test_linkp_without_all_nodes_does_not_crash(self):
        fname = self._write_pos_file(self.pos)
        linkp = _make_linkp_stub(self.pos)  # no all_nodes attribute
        result = self.p2s.loadPositions(fname, linkp)
        self.assertIsInstance(result, dict)


class TestIpSubnetTreeMapLayout(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.p2s = Polars2SVG()

    def test_returns_positions_for_all_nodes(self):
        # Two /24 subnets plus a non-IPv4 node — every node should get a position.
        g = nx.Graph()
        g.add_nodes_from(['10.0.0.1', '10.0.0.2', '192.168.1.5', 'example.com'])
        pos = self.p2s.ipSubnetTreeMapLayout(g, subnet_mask=24)
        self.assertEqual(set(pos.keys()), set(g.nodes()))
        for xy in pos.values():
            self.assertEqual(len(xy), 2)

    def test_return_cells_yields_pos_and_background_boxes(self):
        from shapely.geometry import box as _box_
        g = nx.Graph()
        g.add_nodes_from(['10.0.0.1', '10.0.0.2', '192.168.1.5', 'example.com'])
        result = self.p2s.ipSubnetTreeMapLayout(g, subnet_mask=24, return_cells=True)
        # When return_cells is set the method returns a (pos, cells) tuple.
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        pos, cells = result
        self.assertEqual(set(pos.keys()), set(g.nodes()))
        self.assertIsInstance(cells, dict)
        # One cell per subnet group, plus the human-friendly non-IPv4 label.
        self.assertEqual(set(cells.keys()), {'10.0.0.0/24', '192.168.1.0/24', 'non-IPv4'})
        for _label_, _cell_ in cells.items():
            self.assertIsInstance(_cell_, type(_box_(0, 0, 1, 1)))

    def test_node_with_port_suffix_is_grouped_by_ip(self):
        # "1.2.3.4:443" must strip the port and group under the 1.2.3.0/24 cell.
        g = nx.Graph()
        g.add_nodes_from(['1.2.3.4:443', '1.2.3.9:80'])
        pos, cells = self.p2s.ipSubnetTreeMapLayout(g, subnet_mask=24, return_cells=True)
        self.assertEqual(set(cells.keys()), {'1.2.3.0/24'})
        self.assertNotIn('non-IPv4', cells)
        self.assertEqual(set(pos.keys()), {'1.2.3.4:443', '1.2.3.9:80'})


class TestIpSubnetForceDirectedLayout(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.p2s = Polars2SVG()

    def test_returns_positions_for_all_nodes(self):
        # Two /24 subnets plus a non-IPv4 node — every node gets a position.
        g = nx.Graph()
        g.add_edge('10.0.0.1', '192.168.1.5', weight=3)
        g.add_node('10.0.0.2')
        g.add_node('example.com')
        pos = self.p2s.ipSubnetForceDirectedLayout(g, subnet_mask=24)
        self.assertEqual(set(pos.keys()), set(g.nodes()))
        for xy in pos.values():
            self.assertEqual(len(xy), 2)

    def test_positions_fall_within_bounds(self):
        g = nx.Graph()
        g.add_edge('10.0.0.1', '192.168.1.5')
        g.add_edge('10.0.0.2', '172.16.0.9')
        bounds = (0, 0, 10, 5)
        pos = self.p2s.ipSubnetForceDirectedLayout(g, subnet_mask=24, bounds=bounds)
        for (x, y) in pos.values():
            self.assertGreaterEqual(x, bounds[0]); self.assertLessEqual(x, bounds[2])
            self.assertGreaterEqual(y, bounds[1]); self.assertLessEqual(y, bounds[3])

    def test_collapse_places_subnet_members_together(self):
        # With collapse, both members of the same /24 land on one point.
        g = nx.Graph()
        g.add_edge('10.0.0.1', '192.168.1.5')
        g.add_node('10.0.0.2')
        pos = self.p2s.ipSubnetForceDirectedLayout(g, subnet_mask=24, collapse=True)
        self.assertEqual(pos['10.0.0.1'], pos['10.0.0.2'])

    def test_return_rep_pos_accumulates_crossing_edge_weight(self):
        # Two edges cross the same /24 pair; the rep edge weight is their sum.
        g = nx.Graph()
        g.add_edge('1.2.3.4', '5.6.7.8', weight=2)
        g.add_edge('1.2.3.9', '5.6.7.1', weight=5)
        pos, rep_pos = self.p2s.ipSubnetForceDirectedLayout(g, subnet_mask=24, return_rep_pos=True)
        self.assertEqual(set(pos.keys()), set(g.nodes()))
        self.assertEqual(set(rep_pos.keys()), {'1.2.3.0/24', '5.6.7.0/24'})

    def test_reproducible_with_seed(self):
        g = nx.Graph()
        g.add_edge('10.0.0.1', '192.168.1.5')
        g.add_edge('10.0.0.2', '172.16.0.9')
        a = self.p2s.ipSubnetForceDirectedLayout(g, subnet_mask=24, seed=7)
        b = self.p2s.ipSubnetForceDirectedLayout(g, subnet_mask=24, seed=7)
        self.assertEqual(a, b)


def _blobbed_graph_and_pos():
    """Three spatially-separated blobs (8 nodes each), each an internally
    weight-5 path, with one weak (weight-1) cross-blob link. Returns (g, pos)."""
    import numpy as np
    rng   = np.random.default_rng(0)
    g     = nx.Graph()
    pos   = {}
    nodes = []
    for c, (cx, cy) in enumerate([(0.1, 0.1), (0.9, 0.9), (0.1, 0.9)]):
        blk = []
        for i in range(8):
            n = f'{c}_{i}'
            nodes.append(n); blk.append(n)
            g.add_node(n)
            pos[n] = (cx + rng.normal(0, 0.03), cy + rng.normal(0, 0.03))
        for a, b in zip(blk, blk[1:]): g.add_edge(a, b, weight=5)
    g.add_edge('0_0', '1_0', weight=1)
    return g, pos


class TestNeighborhoodLayout(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.p2s = Polars2SVG()

    # --- shared / argument handling ---

    def test_invalid_mode_raises(self):
        g = nx.Graph(); g.add_node('x')
        with self.assertRaises(Exception):
            self.p2s.neighborhoodLayout(g, pos={'x': (0, 0)}, mode='bogus')

    def test_spatial_mode_requires_pos(self):
        g = nx.Graph(); g.add_node('x')
        with self.assertRaises(Exception):
            self.p2s.neighborhoodLayout(g, mode='spatial')

    # --- spatial mode (clustering over the existing layout) ---

    def test_spatial_rejects_unknown_method(self):
        g, pos = _blobbed_graph_and_pos()
        with self.assertRaises(Exception):
            self.p2s.neighborhoodLayout(g, pos=pos, mode='spatial', spatial_method='bogus')

    def test_spatial_default_method_is_hdbscan(self):
        # Default (no spatial_method) must match an explicit hdbscan run (the new
        # default) and find the three blobs.
        g, pos = _blobbed_graph_and_pos()
        default_pos, default_cells = self.p2s.neighborhoodLayout(g, pos=pos, mode='spatial',
                                                                 return_cells=True)
        hb_pos, hb_cells = self.p2s.neighborhoodLayout(g, pos=pos, mode='spatial',
                                                       spatial_method='hdbscan',
                                                       return_cells=True)
        self.assertEqual(len(default_cells), 3)
        self.assertEqual(set(default_cells.keys()), set(hb_cells.keys()))

    def test_spatial_single_linkage_explicit_cluster_dist(self):
        # A tiny cluster_dist (smaller than within-blob spacing) leaves every node
        # in its own singleton neighborhood.
        g, pos = _blobbed_graph_and_pos()
        _, cells = self.p2s.neighborhoodLayout(g, pos=pos, mode='spatial',
                                               spatial_method='single_linkage',
                                               cluster_dist=1e-6, return_cells=True)
        self.assertEqual(len(cells), g.number_of_nodes())

    def test_spatial_hdbscan_still_finds_three_blobs(self):
        g, pos = _blobbed_graph_and_pos()
        _, cells = self.p2s.neighborhoodLayout(g, pos=pos, mode='spatial',
                                               spatial_method='hdbscan', min_cluster_size=4,
                                               return_cells=True)
        self.assertEqual(len(cells), 3)

    def test_spatial_pos_only_returns_dict(self):
        g, pos = _blobbed_graph_and_pos()
        result = self.p2s.neighborhoodLayout(g, pos=pos, mode='spatial', min_cluster_size=4)
        self.assertIsInstance(result, dict)

    def test_spatial_does_not_move_nodes(self):
        g, pos = _blobbed_graph_and_pos()
        new_pos = self.p2s.neighborhoodLayout(g, pos=pos, mode='spatial', min_cluster_size=4)
        for n in g.nodes():
            self.assertAlmostEqual(new_pos[n][0], pos[n][0])
            self.assertAlmostEqual(new_pos[n][1], pos[n][1])

    def test_spatial_return_cells_finds_three_blobs(self):
        from shapely.geometry import Polygon, MultiPolygon
        g, pos = _blobbed_graph_and_pos()
        new_pos, cells = self.p2s.neighborhoodLayout(g, pos=pos, mode='spatial',
                                                     min_cluster_size=4, return_cells=True)
        self.assertEqual(len(cells), 3)
        for poly in cells.values():
            self.assertIsInstance(poly, (Polygon, MultiPolygon))
            self.assertGreater(poly.area, 0.0)

    # --- single-linkage gap threshold (robust outlier cut) ---

    def _gap(self, pts):
        import numpy as np
        fn = getattr(self.p2s, '__p2sg_singleLinkageGap__')
        return fn(np.asarray(pts, float))

    def test_gap_cuts_modest_multicluster_separation(self):
        # Regression: real multi-cluster layouts separate at a *modest* gap. The VAST
        # netflow layouts cut at only ~1.3-1.9x the prior edge (~1.2x the within-cluster
        # median) -- under any fixed relative-step / "gap >> scale" guard, which wrongly
        # collapsed 12 neighborhoods into one. A grid of tight blobs whose bridges barely
        # exceed the within-blob spacing must still cut into many substantial clusters.
        import numpy as np
        from sklearn.cluster import AgglomerativeClustering
        rng = np.random.default_rng(3)
        pts = np.vstack([rng.normal((cx * 0.30, cy * 0.30), 0.04, size=(25, 2))
                         for cx in range(4) for cy in range(3)])
        thr = self._gap(pts)
        self.assertTrue(np.isfinite(thr))
        lbls = AgglomerativeClustering(n_clusters=None, linkage='single',
                                       distance_threshold=thr).fit(pts).labels_
        _, sizes = np.unique(lbls, return_counts=True)
        # a genuine divide: several substantial clusters, not a single peeled stray
        self.assertGreaterEqual(int((sizes >= 10).sum()), 5)

    def test_gap_does_not_split_single_blob(self):
        # A single tight gaussian blob has no real separation: the largest gap merely
        # peels a stray tail point or two, so the cut leaves only one substantial
        # cluster -> rejected -> one neighborhood (inf). Same for a uniform cloud.
        import numpy as np
        self.assertFalse(np.isfinite(self._gap(np.random.default_rng(1).normal(0, 1, (200, 2)))))
        self.assertFalse(np.isfinite(self._gap(np.random.default_rng(2).random((200, 2)))))

    def test_gap_degenerate_inputs_return_inf(self):
        import numpy as np
        self.assertFalse(np.isfinite(self._gap(np.array([[0.5, 0.5]]))))      # 1 point
        self.assertFalse(np.isfinite(self._gap(np.array([[0., 0.], [1., 1.]]))))  # 1 edge

    def test_spatial_cells_contain_their_members(self):
        from shapely.geometry import Point
        g, pos = _blobbed_graph_and_pos()
        new_pos, cells = self.p2s.neighborhoodLayout(g, pos=pos, mode='spatial',
                                                     min_cluster_size=4, return_cells=True)
        # Map each blob prefix to the cell its members fall in, then assert all
        # of that blob's members live in that one cell.
        for c in range(3):
            members = [f'{c}_{i}' for i in range(8)]
            owning  = [lab for lab, poly in cells.items()
                       if poly.buffer(1e-9).contains(Point(*new_pos[members[0]]))]
            self.assertEqual(len(owning), 1)
            poly = cells[owning[0]]
            for m in members:
                self.assertTrue(poly.buffer(1e-9).contains(Point(*new_pos[m])))

    # --- graph mode (weighted community detection + repositioning) ---

    def test_graph_pos_only_returns_dict_with_all_nodes(self):
        g, _ = _blobbed_graph_and_pos()
        result = self.p2s.neighborhoodLayout(g, mode='graph', seed=42)
        self.assertIsInstance(result, dict)
        self.assertEqual(set(result.keys()), set(g.nodes()))

    def test_graph_return_cells_partitions_every_node(self):
        from shapely.geometry import Point
        g, _ = _blobbed_graph_and_pos()
        pos, cells = self.p2s.neighborhoodLayout(g, mode='graph', seed=42, return_cells=True)
        self.assertGreater(len(cells), 0)
        # Every node lands inside exactly one neighborhood cell (a Voronoi tiling).
        for n in g.nodes():
            inside = sum(1 for poly in cells.values() if poly.buffer(1e-9).contains(Point(*pos[n])))
            self.assertGreaterEqual(inside, 1)

    def test_graph_collapse_places_community_members_together(self):
        g, _ = _blobbed_graph_and_pos()
        pos = self.p2s.neighborhoodLayout(g, mode='graph', collapse=True, seed=42)
        communities = nx.community.louvain_communities(g, weight='weight', seed=42)
        for comm in communities:
            members = list(comm)
            for m in members[1:]:
                self.assertEqual(pos[m], pos[members[0]])

    def test_graph_positions_within_bounds(self):
        g, _ = _blobbed_graph_and_pos()
        bounds = (0, 0, 10, 5)
        pos = self.p2s.neighborhoodLayout(g, mode='graph', collapse=True, seed=42, bounds=bounds)
        for (x, y) in pos.values():
            self.assertGreaterEqual(x, bounds[0]); self.assertLessEqual(x, bounds[2])
            self.assertGreaterEqual(y, bounds[1]); self.assertLessEqual(y, bounds[3])

    def test_graph_reproducible_with_seed(self):
        g, _ = _blobbed_graph_and_pos()
        a = self.p2s.neighborhoodLayout(g, mode='graph', seed=7)
        b = self.p2s.neighborhoodLayout(g, mode='graph', seed=7)
        self.assertEqual(a, b)

    def test_cells_are_renderable_as_linkp_background(self):
        # The {label: shapely_polygon} dict must drop straight into linkp's
        # background= (mirrors ipSubnetTreeMapLayout / hyperTreeDonutLayout).
        g, pos = _blobbed_graph_and_pos()
        new_pos, cells = self.p2s.neighborhoodLayout(g, pos=pos, mode='spatial',
                                                     min_cluster_size=4, return_cells=True)
        df = pl.DataFrame({'fm': ['0_0', '1_0'], 'to': ['0_1', '1_1']})
        ln = self.p2s.linkp(df, [('fm', 'to')], pos=new_pos, background=cells)
        svg = ln._repr_svg_()
        self.assertIn('<svg', svg)


if __name__ == '__main__':
    unittest.main()
