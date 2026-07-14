import math
import unittest

from polars2svg import laguerre_voronoi, QuadTree
from polars2svg.laguerre_voronoi import _clip_half_plane


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _area(poly):
    """Signed area via the shoelace formula (positive = CCW)."""
    n = len(poly)
    s = sum(
        poly[i][0] * poly[(i + 1) % n][1] - poly[(i + 1) % n][0] * poly[i][1]
        for i in range(n)
    )
    return s / 2.0


def _abs_area(poly):
    return abs(_area(poly))


def _is_convex(poly):
    """Return True if the polygon vertices form a convex polygon (any winding)."""
    n = len(poly)
    if n < 3:
        return False
    sign = None
    for i in range(n):
        ox, oy = poly[i]
        ax, ay = poly[(i + 1) % n]
        bx, by = poly[(i + 2) % n]
        cross = (ax - ox) * (by - oy) - (ay - oy) * (bx - ox)
        if abs(cross) < 1e-12:
            continue
        s = 1 if cross > 0 else -1
        if sign is None:
            sign = s
        elif sign != s:
            return False
    return True


# ---------------------------------------------------------------------------
# _clip_half_plane tests
# ---------------------------------------------------------------------------

class TestClipHalfPlane(unittest.TestCase):

    def test_empty_input_returns_empty(self):
        self.assertEqual(_clip_half_plane([], 1, 0, 0), [])

    def test_all_inside_unchanged(self):
        # Unit square, clip to x <= 2 — all vertices inside.
        sq = [(0, 0), (1, 0), (1, 1), (0, 1)]
        result = _clip_half_plane(sq, 1.0, 0.0, 2.0)
        self.assertEqual(len(result), 4)
        for v in result:
            self.assertLessEqual(v[0], 2.0 + 1e-12)

    def test_all_outside_returns_empty(self):
        # Unit square, clip to x <= -1 — all vertices outside.
        sq = [(0, 0), (1, 0), (1, 1), (0, 1)]
        result = _clip_half_plane(sq, 1.0, 0.0, -1.0)
        self.assertEqual(result, [])

    def test_vertical_cut_half_square(self):
        # Unit square clipped to x <= 0.5 → left half, 4 vertices.
        sq = [(0, 0), (1, 0), (1, 1), (0, 1)]
        result = _clip_half_plane(sq, 1.0, 0.0, 0.5)
        self.assertEqual(len(result), 4)
        for v in result:
            self.assertLessEqual(v[0], 0.5 + 1e-12)
        self.assertAlmostEqual(_abs_area(result), 0.5, places=10)

    def test_horizontal_cut_half_square(self):
        # Unit square clipped to y <= 0.5 → bottom half.
        sq = [(0, 0), (1, 0), (1, 1), (0, 1)]
        result = _clip_half_plane(sq, 0.0, 1.0, 0.5)
        self.assertEqual(len(result), 4)
        self.assertAlmostEqual(_abs_area(result), 0.5, places=10)

    def test_diagonal_cut_triangle(self):
        # Unit square clipped to x + y <= 1 → lower-left triangle.
        # Boundary vertices are kept (val == c counts as inside), which can produce
        # duplicate points via Sutherland-Hodgman; check area rather than vertex count.
        sq = [(0, 0), (1, 0), (1, 1), (0, 1)]
        result = _clip_half_plane(sq, 1.0, 1.0, 1.0)
        self.assertTrue(len(result) >= 3)
        self.assertAlmostEqual(_abs_area(result), 0.5, places=10)

    def test_single_vertex_on_boundary(self):
        # Triangle clipped to x + y <= 0: only the origin vertex (0+0=0) is on the
        # boundary.  Sutherland-Hodgman emits duplicate coincident points; the
        # resulting degenerate polygon has area ≈ 0.
        tri = [(0, 0), (1, 0), (0, 1)]
        result = _clip_half_plane(tri, 1.0, 1.0, 0.0)
        self.assertTrue(len(result) >= 1)
        self.assertAlmostEqual(_abs_area(result), 0.0, places=10)

    def test_boundary_vertex_kept(self):
        # Vertex exactly on the boundary (val == c) is treated as inside.
        sq = [(0, 0), (1, 0), (1, 1), (0, 1)]
        result = _clip_half_plane(sq, 1.0, 0.0, 1.0)
        self.assertEqual(len(result), 4)


# ---------------------------------------------------------------------------
# QuadTree tests
# ---------------------------------------------------------------------------

class TestQuadTree(unittest.TestCase):

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            QuadTree([])

    def test_two_points_nearest(self):
        qt = QuadTree([(0.0, 0.0), (3.0, 4.0)])
        d, idx = qt.nearest(0.0, 0.0)
        self.assertAlmostEqual(d, 5.0, places=10)
        self.assertEqual(idx, 1)

    def test_nearest_excludes_self(self):
        # Point at (0,0) — nearest is (1,0), not itself.
        qt = QuadTree([(0.0, 0.0), (1.0, 0.0), (10.0, 0.0)])
        d, idx = qt.nearest(0.0, 0.0)
        self.assertAlmostEqual(d, 1.0, places=10)
        self.assertEqual(idx, 1)

    def test_nearest_returns_closest(self):
        pts = [(float(i), 0.0) for i in range(10)]
        qt = QuadTree(pts)
        d, idx = qt.nearest(5.0, 0.0)
        self.assertAlmostEqual(d, 1.0, places=10)
        self.assertIn(idx, (4, 6))

    def test_large_point_set(self):
        import random
        rng = random.Random(0)
        pts = [(rng.uniform(-100, 100), rng.uniform(-100, 100)) for _ in range(500)]
        qt = QuadTree(pts)
        qx, qy = pts[0]
        d_qt, idx_qt = qt.nearest(qx, qy)
        # Brute-force nearest (excluding the query point itself).
        d_bf = min(math.hypot(qx - px, qy - py) for px, py in pts[1:])
        self.assertAlmostEqual(d_qt, d_bf, places=6)

    def test_collinear_points(self):
        # All points on x-axis.
        pts = [(float(i), 0.0) for i in range(5)]
        qt = QuadTree(pts)
        d, idx = qt.nearest(2.0, 0.0)
        self.assertAlmostEqual(d, 1.0, places=10)

    def test_single_point_no_neighbor(self):
        # With only one stored point, nearest returns (inf, -1).
        qt = QuadTree([(1.0, 2.0)])
        d, idx = qt.nearest(1.0, 2.0)
        self.assertEqual(d, math.inf)
        self.assertEqual(idx, -1)


# ---------------------------------------------------------------------------
# laguerre_voronoi — basic structure tests
# ---------------------------------------------------------------------------

class TestLaguerreVoronoiStructure(unittest.TestCase):

    def test_empty_input(self):
        self.assertEqual(laguerre_voronoi({}), {})

    def test_single_node_returns_one_cell(self):
        cells = laguerre_voronoi({"a": (0.0, 0.0)})
        self.assertIn("a", cells)
        self.assertIsNotNone(cells["a"])

    def test_single_node_cell_has_four_vertices(self):
        # Single node → no half-plane clipping → full bounding box = 4 vertices.
        cells = laguerre_voronoi({"a": (0.0, 0.0)})
        self.assertEqual(len(cells["a"]), 4)

    def test_output_keys_match_input_keys(self):
        nodes = {"x": (0.0, 0.0), "y": (1.0, 0.0), "z": (0.5, 1.0)}
        cells = laguerre_voronoi(nodes)
        self.assertEqual(set(cells.keys()), set(nodes.keys()))

    def test_integer_tuple_keys(self):
        grid = {(i, j): (float(i), float(j)) for i in range(3) for j in range(3)}
        cells = laguerre_voronoi(grid)
        self.assertEqual(set(cells.keys()), set(grid.keys()))

    def test_cells_are_convex_polygons(self):
        nodes = {"a": (0.0, 0.0), "b": (1.0, 0.0), "c": (0.5, 1.0)}
        cells = laguerre_voronoi(nodes)
        for key, poly in cells.items():
            if poly is not None:
                self.assertTrue(_is_convex(poly), f"Cell {key!r} is not convex")

    def test_nonempty_cells_have_at_least_three_vertices(self):
        nodes = {i: (float(i), 0.0) for i in range(5)}
        cells = laguerre_voronoi(nodes)
        for key, poly in cells.items():
            if poly is not None:
                self.assertGreaterEqual(len(poly), 3, f"Cell {key!r} has < 3 vertices")


# ---------------------------------------------------------------------------
# laguerre_voronoi — two-node cases
# ---------------------------------------------------------------------------

class TestLaguerreVoronoiTwoNodes(unittest.TestCase):

    def test_two_equal_radius_nodes_both_have_cells(self):
        cells = laguerre_voronoi({"L": (0.0, 0.0, 1.0), "R": (2.0, 0.0, 1.0)})
        self.assertIsNotNone(cells["L"])
        self.assertIsNotNone(cells["R"])

    def test_two_equal_radius_nodes_symmetric_areas(self):
        cells = laguerre_voronoi({"L": (0.0, 0.0, 1.0), "R": (2.0, 0.0, 1.0)})
        area_L = _abs_area(cells["L"])
        area_R = _abs_area(cells["R"])
        self.assertAlmostEqual(area_L, area_R, places=6)

    def test_two_equal_auto_radius_nodes_both_have_cells(self):
        # Auto-radius: each gets r = 0.5 (half the distance of 1.0).
        cells = laguerre_voronoi({"L": (0.0, 0.0), "R": (1.0, 0.0)})
        self.assertIsNotNone(cells["L"])
        self.assertIsNotNone(cells["R"])

    def test_two_nodes_areas_sum_to_bbox(self):
        nodes = {"L": (0.0, 0.0, 1.0), "R": (2.0, 0.0, 1.0)}
        cells = laguerre_voronoi(nodes, bbox_pad=0.1)
        total_cell = sum(_abs_area(p) for p in cells.values() if p is not None)
        # Compute bbox manually: xmin=-1, xmax=3, ymin=-1, ymax=1 → span 4×2
        # padded by 0.1: bx0=-1.4, bx1=3.4, by0=-1.2, by1=1.2
        bbox_area = (3.4 - (-1.4)) * (1.2 - (-1.2))
        self.assertAlmostEqual(total_cell, bbox_area, places=6)

    def test_dominant_large_circle_eliminates_small(self):
        # Large circle at origin with r=3 should dominate small circle at (1,0,0.1).
        cells = laguerre_voronoi({"big": (0.0, 0.0, 3.0), "small": (1.0, 0.0, 0.1)})
        self.assertIsNotNone(cells["big"])
        self.assertIsNone(cells["small"])

    def test_large_circle_gets_full_bbox_when_dominates(self):
        cells = laguerre_voronoi({"big": (0.0, 0.0, 3.0), "small": (1.0, 0.0, 0.1)},
                                 bbox_pad=0.1)
        # The big circle's cell should cover the entire bbox since small is eliminated.
        # Compute expected bbox area.
        # circles: big=(0,0,3), small=(1,0,0.1)
        # xmin=min(-3, 0.9)=-3, xmax=max(3,1.1)=3, ymin=-3, ymax=3 → span 6x6
        # padded: bx0=-3.6, bx1=3.6, by0=-3.6, by1=3.6
        bbox_area = 7.2 * 7.2
        self.assertAlmostEqual(_abs_area(cells["big"]), bbox_area, places=6)


# ---------------------------------------------------------------------------
# laguerre_voronoi — multi-node cases
# ---------------------------------------------------------------------------

class TestLaguerreVoronoiMultiNode(unittest.TestCase):

    def test_triangle_equal_circles_all_nonempty(self):
        tri = {
            "A": (0.0,  1.0, 0.5),
            "B": (-1.0, -0.5, 0.5),
            "C": (1.0, -0.5, 0.5),
        }
        cells = laguerre_voronoi(tri)
        for key, poly in cells.items():
            self.assertIsNotNone(poly, f"Cell {key!r} unexpectedly empty")
            self.assertGreaterEqual(len(poly), 3)

    def test_triangle_total_area_equals_bbox(self):
        tri = {
            "A": (0.0,  1.0, 0.5),
            "B": (-1.0, -0.5, 0.5),
            "C": (1.0, -0.5, 0.5),
        }
        cells = laguerre_voronoi(tri, bbox_pad=0.1)
        total = sum(_abs_area(p) for p in cells.values() if p is not None)
        # All three cells non-empty → their areas must tile the bbox exactly.
        # Compute bbox: x in [-1.5, 1.5], y in [-1.0, 1.5]; span 3.0 x 2.5
        # padded 0.1: bx0=-1.8, bx1=1.8, by0=-1.25, by1=1.75
        bbox_area = 3.6 * 3.0
        self.assertAlmostEqual(total, bbox_area, places=6)

    def test_grid_all_cells_nonempty(self):
        grid = {(i, j): (float(i), float(j)) for i in range(3) for j in range(3)}
        cells = laguerre_voronoi(grid)
        for key, poly in cells.items():
            self.assertIsNotNone(poly, f"Grid cell {key!r} unexpectedly empty")

    def test_grid_total_area_equals_bbox(self):
        grid = {(i, j): (float(i), float(j)) for i in range(3) for j in range(3)}
        cells = laguerre_voronoi(grid, bbox_pad=0.1)
        total = sum(_abs_area(p) for p in cells.values() if p is not None)
        # Auto-radius for grid points on unit grid: r = 0.5 each.
        # xmin=-0.5, xmax=2.5, ymin=-0.5, ymax=2.5; span 3 x 3
        # padded 0.1: bx0=-0.8, bx1=2.8, by0=-0.8, by1=2.8 → 3.6 x 3.6
        bbox_area = 3.6 * 3.6
        self.assertAlmostEqual(total, bbox_area, places=5)

    def test_five_nodes_collinear(self):
        nodes = {i: (float(i), 0.0, 0.5) for i in range(5)}
        cells = laguerre_voronoi(nodes)
        non_empty = [k for k, v in cells.items() if v is not None]
        self.assertEqual(len(non_empty), 5)

    def test_explicit_radii_affect_cell_sizes(self):
        # Node with larger radius gets a larger cell.
        nodes = {"big": (0.0, 0.0, 2.0), "small": (3.0, 0.0, 0.5)}
        cells = laguerre_voronoi(nodes)
        self.assertIsNotNone(cells["big"])
        self.assertIsNotNone(cells["small"])
        self.assertGreater(_abs_area(cells["big"]), _abs_area(cells["small"]))

    def test_all_cells_convex(self):
        grid = {(i, j): (float(i), float(j)) for i in range(4) for j in range(4)}
        cells = laguerre_voronoi(grid)
        for key, poly in cells.items():
            if poly is not None:
                self.assertTrue(_is_convex(poly), f"Cell {key!r} is not convex")


# ---------------------------------------------------------------------------
# laguerre_voronoi — bbox_pad behaviour
# ---------------------------------------------------------------------------

class TestLaguerreVoronoiBboxPad(unittest.TestCase):

    def test_larger_pad_gives_larger_cells(self):
        nodes = {"a": (0.0, 0.0, 1.0), "b": (2.0, 0.0, 1.0)}
        cells_small = laguerre_voronoi(nodes, bbox_pad=0.0)
        cells_large = laguerre_voronoi(nodes, bbox_pad=1.0)
        area_small = sum(_abs_area(p) for p in cells_small.values() if p)
        area_large = sum(_abs_area(p) for p in cells_large.values() if p)
        self.assertGreater(area_large, area_small)

    def test_zero_pad_cells_still_valid(self):
        nodes = {"a": (0.0, 0.0, 1.0), "b": (2.0, 0.0, 1.0)}
        cells = laguerre_voronoi(nodes, bbox_pad=0.0)
        for poly in cells.values():
            if poly is not None:
                self.assertGreaterEqual(len(poly), 3)


# ---------------------------------------------------------------------------
# Package-level export tests
# ---------------------------------------------------------------------------

class TestPackageExports(unittest.TestCase):

    def test_laguerre_voronoi_importable_from_package(self):
        import polars2svg
        self.assertTrue(callable(polars2svg.laguerre_voronoi))

    def test_quadtree_importable_from_package(self):
        import polars2svg
        self.assertIs(polars2svg.QuadTree, QuadTree)

    def test_laguerre_voronoi_function_is_same_object(self):
        import polars2svg
        self.assertIs(polars2svg.laguerre_voronoi, laguerre_voronoi)


if __name__ == '__main__':
    unittest.main()
