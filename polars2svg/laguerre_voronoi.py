"""
Laguerre-Voronoi (Power Diagram) for circles in the plane.

Reference: Imai, Iri, Murota (1985) "Voronoi Diagram in the Laguerre Geometry
and its Applications", SIAM J. Comput., 14(1), pp. 93-105.

The Laguerre distance from a point P = (x, y) to a circle C = (cx, cy, r) is:

    d_L^2(C, P) = (x - cx)^2 + (y - cy)^2 - r^2

The Voronoi (Laguerre) cell of circle i is:

    V(C_i) = { P in R^2 : d_L^2(C_i, P) <= d_L^2(C_j, P)  for all j }

The boundary between cells i and j is the *radical axis* -- a straight line --
so every cell is a convex polygon.  We compute it by intersecting n-1
half-planes with a bounding rectangle (Sutherland-Hodgman clipping).

Radius assignment (when none is supplied): a quad-tree finds each point's
nearest neighbour; the radius is set to half that distance.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

Key      = object
Point2D  = Tuple[float, float]
NodeSpec = Union[Tuple[float, float], Tuple[float, float, float]]


# ---------------------------------------------------------------------------
# Quad-tree for nearest-neighbour search
# ---------------------------------------------------------------------------

class _QTNode:
    __slots__ = ("cx", "cy", "hw", "hh", "pts", "children")

    def __init__(self, cx: float, cy: float, hw: float, hh: float) -> None:
        self.cx = cx;  self.cy = cy
        self.hw = hw;  self.hh = hh
        self.pts: list = []
        self.children: Optional[tuple] = None

    def _contains(self, x: float, y: float) -> bool:
        return (self.cx - self.hw <= x <= self.cx + self.hw and
                self.cy - self.hh <= y <= self.cy + self.hh)

    def _intersects_circle(self, qx: float, qy: float, r: float) -> bool:
        dx = abs(qx - self.cx)
        dy = abs(qy - self.cy)
        if dx > self.hw + r or dy > self.hh + r:
            return False
        if dx <= self.hw or dy <= self.hh:
            return True
        return (dx - self.hw) ** 2 + (dy - self.hh) ** 2 <= r * r

    def _subdivide(self) -> None:
        hw2, hh2 = self.hw * 0.5, self.hh * 0.5
        cx, cy = self.cx, self.cy
        self.children = (
            _QTNode(cx - hw2, cy + hh2, hw2, hh2),  # NW
            _QTNode(cx + hw2, cy + hh2, hw2, hh2),  # NE
            _QTNode(cx - hw2, cy - hh2, hw2, hh2),  # SW
            _QTNode(cx + hw2, cy - hh2, hw2, hh2),  # SE
        )

    def insert(self, x: float, y: float, idx: int, capacity: int = 8) -> bool:
        if not self._contains(x, y):
            return False
        if self.children is None:
            self.pts.append((x, y, idx))
            if len(self.pts) > capacity:
                self._subdivide()
                old, self.pts = self.pts, []
                for p in old:
                    for ch in self.children:  # type: ignore[union-attr]
                        if ch.insert(p[0], p[1], p[2], capacity):
                            break
        else:
            for ch in self.children:  # type: ignore[union-attr]
                if ch.insert(x, y, idx, capacity):
                    break
        return True

    def _min_dist2_to_child(self, ch: "_QTNode", qx: float, qy: float) -> float:
        dx = max(0.0, abs(qx - ch.cx) - ch.hw)
        dy = max(0.0, abs(qy - ch.cy) - ch.hh)
        return dx * dx + dy * dy

    def query_nearest(
        self, qx: float, qy: float, best_d: float, best_idx: int
    ) -> Tuple[float, int]:
        if not self._intersects_circle(qx, qy, best_d):
            return best_d, best_idx
        if self.children is None:
            for px, py, idx in self.pts:
                d = math.hypot(qx - px, qy - py)
                if d > 0.0 and d < best_d:
                    best_d, best_idx = d, idx
        else:
            # Visit closest child first to prune early
            ordered = sorted(
                self.children,  # type: ignore[union-attr]
                key=lambda c: self._min_dist2_to_child(c, qx, qy),
            )
            for ch in ordered:
                best_d, best_idx = ch.query_nearest(qx, qy, best_d, best_idx)
        return best_d, best_idx


class QuadTree:
    """2-D quad-tree supporting nearest-neighbour queries (d > 0, excludes self)."""

    def __init__(self, points: List[Point2D]) -> None:
        if not points:
            raise ValueError("QuadTree requires at least one point")
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        cx = (min(xs) + max(xs)) * 0.5
        cy = (min(ys) + max(ys)) * 0.5
        hw = (max(xs) - min(xs)) * 0.5 + 1e-9
        hh = (max(ys) - min(ys)) * 0.5 + 1e-9
        self._root = _QTNode(cx, cy, hw, hh)
        for i, (x, y) in enumerate(points):
            self._root.insert(x, y, i)

    def nearest(self, x: float, y: float) -> Tuple[float, int]:
        """Return (distance, index) of the nearest stored point with distance > 0."""
        d, idx = self._root.query_nearest(x, y, math.inf, -1)
        return d, idx


# ---------------------------------------------------------------------------
# Sutherland-Hodgman half-plane clipping
# ---------------------------------------------------------------------------

def _clip_half_plane(
    poly: List[Point2D], a: float, b: float, c: float
) -> List[Point2D]:
    """
    Clip a convex polygon to the closed half-plane  a*x + b*y <= c.

    Uses the Sutherland-Hodgman algorithm.  Works for any winding order;
    the output preserves the input winding.
    """
    if not poly:
        return poly
    result: List[Point2D] = []
    n = len(poly)
    for i in range(n):
        cur = poly[i]
        nxt = poly[(i + 1) % n]
        val_cur = a * cur[0] + b * cur[1]
        val_nxt = a * nxt[0] + b * nxt[1]
        inside_cur = val_cur <= c
        inside_nxt = val_nxt <= c
        if inside_cur:
            result.append(cur)
        if inside_cur != inside_nxt:
            # Edge crosses the boundary; add the intersection point.
            denom = val_nxt - val_cur          # != 0 because sides differ
            t = (c - val_cur) / denom
            result.append((
                cur[0] + t * (nxt[0] - cur[0]),
                cur[1] + t * (nxt[1] - cur[1]),
            ))
    return result


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def laguerre_voronoi(
    nodes: Dict[Key, NodeSpec],
    *,
    bbox_pad: float = 0.1,
) -> Dict[Key, Optional[List[Point2D]]]:
    """
    Compute the Laguerre-Voronoi diagram (power diagram) for a set of circles.

    Parameters
    ----------
    nodes
        Mapping from an arbitrary hashable node key to either ``(x, y)`` or
        ``(x, y, r)``.  When only ``(x, y)`` is given, the radius is set to
        half the distance to the nearest other node (found via a quad-tree).
    bbox_pad
        Fractional padding on each side of the data bounding box used to clip
        cells that extend to infinity.  Default 0.1 (10 % of the data span).

    Returns
    -------
    dict
        Maps each node key to a list of ``(x, y)`` vertices forming the
        (clipped) convex Voronoi cell, in counter-clockwise order.
        Returns ``None`` for a site whose cell is empty -- this happens when
        the site's circle is entirely dominated by a larger overlapping circle.

    Algorithm
    ---------
    For each site i, the cell is the intersection of n-1 half-planes:

        d_L^2(C_i, P) <= d_L^2(C_j, P)
        <=>  2(xj-xi)*x + 2(yj-yi)*y  <=  (xj^2+yj^2-rj^2) - (xi^2+yi^2-ri^2)

    We start with the bounding rectangle and clip it with each half-plane in
    turn (Sutherland-Hodgman).  Complexity is O(n^2) per cell, O(n^3) total.
    """
    keys = list(nodes.keys())
    n = len(keys)
    if n == 0:
        return {}

    # ------------------------------------------------------------------
    # 1. Resolve radii
    # ------------------------------------------------------------------
    raw: List[tuple] = [nodes[k] for k in keys]
    pts_xy: List[Point2D] = [(float(t[0]), float(t[1])) for t in raw]

    circles: List[Tuple[float, float, float]] = []

    if n == 1:
        circles.append((pts_xy[0][0], pts_xy[0][1], 0.0))
    else:
        qt = QuadTree(pts_xy)
        for i, t in enumerate(raw):
            x, y = float(t[0]), float(t[1])
            if len(t) >= 3:
                circles.append((x, y, float(t[2])))
            else:
                dist, _ = qt.nearest(x, y)
                circles.append((x, y, dist / 2.0))

    # ------------------------------------------------------------------
    # 2. Build clipping bounding box (padded)
    # ------------------------------------------------------------------
    xmin = min(c[0] - c[2] for c in circles)
    xmax = max(c[0] + c[2] for c in circles)
    ymin = min(c[1] - c[2] for c in circles)
    ymax = max(c[1] + c[2] for c in circles)
    span_x = max(xmax - xmin, 1.0)
    span_y = max(ymax - ymin, 1.0)
    bx0 = xmin - span_x * bbox_pad
    bx1 = xmax + span_x * bbox_pad
    by0 = ymin - span_y * bbox_pad
    by1 = ymax + span_y * bbox_pad

    # CCW bounding-box polygon
    bbox_poly: List[Point2D] = [
        (bx0, by0), (bx1, by0), (bx1, by1), (bx0, by1)
    ]

    # ------------------------------------------------------------------
    # 3. Compute each Laguerre cell by half-plane intersection
    # ------------------------------------------------------------------
    result: Dict[Key, Optional[List[Point2D]]] = {}

    for i, key in enumerate(keys):
        xi, yi, ri = circles[i]
        # Lifted height of site i: wi = xi^2 + yi^2 - ri^2
        wi = xi * xi + yi * yi - ri * ri

        poly = list(bbox_poly)

        for j in range(n):
            if j == i:
                continue
            xj, yj, rj = circles[j]
            wj = xj * xj + yj * yj - rj * rj

            # Keep points P where d_L^2(C_i, P) <= d_L^2(C_j, P):
            #   2(xj - xi)*x + 2(yj - yi)*y  <=  wj - wi
            a = 2.0 * (xj - xi)
            b = 2.0 * (yj - yi)
            c = wj - wi

            poly = _clip_half_plane(poly, a, b, c)
            if not poly:
                break   # cell is empty

        result[key] = poly if poly else None

    return result
