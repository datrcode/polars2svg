# Implementation of:
#
#   NCP: Neighborhood-Preserving Non-Uniform Circle Packing for Visualization
#   Duan Li, Jun Yuan, Xinyuan Guo, Xiting Wang, Yang Liu, Weikai Yang, Shixia Liu
#   Computational Visual Media, 2026 (arXiv:2602.00668)
#
# Used here as a layout *compaction* pass: given an existing 2D layout (spring,
# force-directed, ...) and a per-node size, it packs the nodes into
# non-overlapping circles that reclaim wasted space while preserving the
# spatial neighbourhoods the input layout established. The neighbourhood
# structure preserved is the Delaunay triangulation of the *current positions*
# (paper Sec. 4.3), so the graph's own edges do not constrain the packing --
# only the visual arrangement does.
#
# MLX was measured and rejected for this algorithm (opposite of od_flow_layout):
# the hot arrays are far below the GPU break-even and the 1250 force iterations
# are sequentially dependent. Pure NumPy + SciPy.

from __future__ import annotations

import math

import numpy as np
from scipy.spatial import ConvexHull, Delaunay, QhullError, cKDTree

EPS = 1e-12


# ===========================================================================
# Geometry -- convex polygon clipping / centroids, circle hull, pockets (F_v)
# ===========================================================================

def _clip_halfplane_batch(poly, count, a, c, eps=1e-12):
    """Sutherland-Hodgman for every power cell at once (paper Sec. 4.4).

    Cells are held as a padded ``(n, V, 2)`` buffer with ``count[i]`` live
    vertices in ``poly[i, :count[i]]``. Clipping a convex polygon by one
    half-plane yields at most ``count + 1`` vertices, so a buffer sized
    ``4 + max_degree`` survives the whole cell construction.
    """
    n, cap, _ = poly.shape
    idx = np.arange(cap)
    rows = np.arange(n)[:, None]

    live = idx[None, :] < count[:, None]
    d = np.einsum('nvk,nk->nv', poly, a) - c[:, None]
    inside = (d <= eps) & live

    nxt = (idx[None, :] + 1) % np.maximum(count, 1)[:, None]
    p_next = poly[rows, nxt]
    d_next = d[rows, nxt]
    inside_next = inside[rows, nxt]

    denom = d - d_next
    t = np.divide(d, denom, out=np.zeros_like(d), where=np.abs(denom) > 1e-300)
    inter = poly + t[:, :, None] * (p_next - poly)

    # interleave "keep this vertex" and "emit this crossing" so the cyclic
    # vertex order is preserved by construction
    src = np.empty((n, 2 * cap, 2))
    src[:, 0::2] = poly
    src[:, 1::2] = inter
    valid = np.empty((n, 2 * cap), dtype=bool)
    valid[:, 0::2] = inside
    valid[:, 1::2] = (inside != inside_next) & live

    dest = np.cumsum(valid, axis=1) - 1
    out = np.zeros_like(poly)
    r, k = np.nonzero(valid)
    dd = dest[r, k]
    fits = dd < cap
    out[r[fits], dd[fits]] = src[r[fits], k[fits]]
    return out, np.minimum(valid.sum(axis=1), cap)


def _polygon_centroid_batch(poly, count):
    """Area centroids of a padded cell buffer; vertex mean where degenerate."""
    n, cap, _ = poly.shape
    idx = np.arange(cap)
    rows = np.arange(n)[:, None]
    live = idx[None, :] < count[:, None]
    nxt = (idx[None, :] + 1) % np.maximum(count, 1)[:, None]

    x, y = poly[:, :, 0], poly[:, :, 1]
    xn, yn = x[rows, nxt], y[rows, nxt]
    cross = np.where(live, x * yn - xn * y, 0.0)

    area2 = cross.sum(axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        cx = (np.where(live, (x + xn) * cross, 0.0).sum(axis=1)) / (3.0 * area2)
        cy = (np.where(live, (y + yn) * cross, 0.0).sum(axis=1)) / (3.0 * area2)

    live_n = np.maximum(count, 1)
    mx = np.where(live, x, 0.0).sum(axis=1) / live_n
    my = np.where(live, y, 0.0).sum(axis=1) / live_n

    ok = (count >= 3) & (np.abs(area2) > 1e-15) & np.isfinite(cx) & np.isfinite(cy)
    return np.column_stack([np.where(ok, cx, mx), np.where(ok, cy, my)]), count > 0


def _bbox_polygon(centers, radii, pad=1.0):
    """CCW rectangle enclosing every circle with ``pad`` times the max radius."""
    lo = (centers - radii[:, None]).min(axis=0) - pad * float(radii.max())
    hi = (centers + radii[:, None]).max(axis=0) + pad * float(radii.max())
    return np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])


def _circle_hull(centers, radii, samples=48):
    """Convex hull of the union of circles: (CCW vertices, owning circle ids)."""
    n = len(centers)
    ang = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
    ring = np.column_stack([np.cos(ang), np.sin(ang)])
    pts = (centers[:, None, :] + radii[:, None, None] * ring[None, :, :])
    pts = pts.reshape(-1, 2)
    owner = np.repeat(np.arange(n), samples)
    hull = ConvexHull(pts)
    order = hull.vertices
    return pts[order], owner[order]


def _hull_circle_cycle(centers, radii, samples=48):
    """The circles touching the convex hull, in CCW order: (ids, touch points)."""
    verts, owners = _circle_hull(centers, radii, samples=samples)
    ids, touch = [], []
    start = 0
    m = len(owners)
    if m > 1 and owners[0] == owners[-1]:
        k = 0
        while k < m and owners[k] == owners[0]:
            k += 1
        if k < m:
            verts, owners = np.roll(verts, -k, axis=0), np.roll(owners, -k)
    for i in range(1, m + 1):
        if i == m or owners[i] != owners[start]:
            ids.append(int(owners[start]))
            touch.append(verts[start:i].mean(axis=0))
            start = i
    return ids, np.asarray(touch, dtype=float)


def _hull_pockets(centers, radii, samples=48):
    """Concavities along the circle hull -- the input to F_v (paper Fig. 8).

    Returns a list of ``(i, M, gap)``: the circle to move, the chord midpoint
    between two hull-adjacent circles, and how far ``c_i`` falls short of it.
    """
    ids, touch = _hull_circle_cycle(centers, radii, samples=samples)
    if len(ids) < 3:
        return []

    on_hull = set(ids)
    out, seen = [], set()
    for t in range(len(ids)):
        j, k = ids[t], ids[(t + 1) % len(ids)]
        a, b = touch[t], touch[(t + 1) % len(ids)]
        chord = b - a
        length = float(np.hypot(*chord))
        if length < 1e-12:
            continue
        u = chord / length
        nrm = np.array([u[1], -u[0]])        # outward for a CCW hull

        rel = centers - a
        proj = rel @ u
        depth = rel @ nrm + radii
        cand = np.where((proj > 0.0) & (proj < length))[0]
        cand = [c for c in cand if c != j and c != k and c not in on_hull]
        if not cand:
            continue
        i = int(max(cand, key=lambda c: depth[c]))
        if i in seen:
            continue
        seen.add(i)

        m = 0.5 * (a + b)
        gap = abs(float(np.hypot(*(m - centers[i]))) - float(radii[i]))
        out.append((i, m, gap))
    return out


# ===========================================================================
# Power diagram / regular triangulation (paper Sec. 4.4)
# ===========================================================================

def _regular_triangulation(centers, radii):
    """Simplices (m, 3) of the weighted Delaunay (regular) triangulation."""
    if len(centers) < 3:
        return np.zeros((0, 3), dtype=int)
    lifted = np.column_stack([centers, (centers ** 2).sum(axis=1) - radii ** 2])
    try:
        hull = ConvexHull(lifted)
    except QhullError:
        return np.zeros((0, 3), dtype=int)
    lower = hull.equations[:, 2] < -1e-12
    return hull.simplices[lower]


def _triangulation_edges(simplices):
    """Unique undirected edges (k, 2) of a triangulation, sorted."""
    if len(simplices) == 0:
        return np.zeros((0, 2), dtype=int)
    e = np.vstack([simplices[:, [0, 1]], simplices[:, [1, 2]], simplices[:, [0, 2]]])
    e = np.sort(e, axis=1)
    return np.unique(e, axis=0)


def _neighbor_lists(edges, n):
    nbrs = [[] for _ in range(n)]
    for a, b in edges:
        nbrs[a].append(b)
        nbrs[b].append(a)
    return [np.asarray(sorted(v), dtype=int) for v in nbrs]


def _power_cells(centers, radii, nbrs, region=None):
    """Clip ``region`` by each site's power half-planes, all sites at once.

    Returns the padded ``(n, V, 2)`` vertex buffer and per-cell vertex count.
    """
    n = len(centers)
    if region is None:
        region = _bbox_polygon(centers, radii)
    lift = (centers ** 2).sum(axis=1) - radii ** 2

    degree = np.array([len(v) for v in nbrs], dtype=int) if n else np.zeros(0, int)
    max_degree = int(degree.max()) if n and len(degree) else 0
    cap = len(region) + max_degree

    table = np.full((n, max(max_degree, 1)), -1, dtype=int)
    for i, v in enumerate(nbrs):
        table[i, :len(v)] = v

    poly = np.zeros((n, cap, 2))
    poly[:, :len(region)] = region
    count = np.full(n, len(region), dtype=int)

    for k in range(max_degree):
        j = table[:, k]
        has = j >= 0
        jj = np.where(has, j, 0)
        a = np.where(has[:, None], 2.0 * (centers[jj] - centers), 0.0)
        c = np.where(has, lift[jj] - lift, 1.0)   # 0.x <= 1 is vacuous
        poly, count = _clip_halfplane_batch(poly, count, a, c)
    return poly, count


def _cell_centroids(cells, centers):
    """Centroid of each power cell (max-inscribed-circle stand-in, Sec. 4.4)."""
    poly, count = cells
    cent, ok = _polygon_centroid_batch(poly, count)
    return np.where(ok[:, None], cent, centers)


def _max_scale(centers, weights, hint=None, chunk=512):
    """Largest ``s`` with ``r_i = s * w_i`` satisfying the non-overlap constraint.

    ``s = min_{i<j} ||p_i - p_j|| / (w_i + w_j)``. A ``hint`` enables an exact
    KD-tree fast path (falls back to the O(n^2) scan when its certificate
    fails), which is what keeps the 1250-iteration force loop affordable.
    """
    n = len(centers)
    if n < 2:
        return float('inf')

    w_max = float(np.max(weights))
    if hint is not None and np.isfinite(hint) and hint > 0 and w_max > 0:
        radius = 2.0 * w_max * float(hint) * 1.25
        pairs = cKDTree(centers).query_pairs(radius, output_type='ndarray')
        if len(pairs) > 0:
            a, b = pairs[:, 0], pairs[:, 1]
            denom = weights[a] + weights[b]
            ok = denom > 0
            if ok.any():
                d = np.linalg.norm(centers[a[ok]] - centers[b[ok]], axis=1)
                m = float(np.min(d / denom[ok]))
                if 2.0 * w_max * m <= radius:
                    return m

    best = np.inf
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        block = centers[start:stop]
        d = np.linalg.norm(block[:, None, :] - centers[None, :, :], axis=2)
        w = weights[start:stop, None] + weights[None, :]
        idx = np.arange(start, stop)
        d[np.arange(stop - start), idx] = np.inf
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(w > 0, d / np.where(w > 0, w, 1.0), np.inf)
        best = min(best, float(np.nanmin(ratio)))
    return best


def _weighted_circumcenter(centers, radii, tri):
    """Power-diagram vertex of three weighted sites (Fig. 6). None if collinear."""
    a, b, c = (int(t) for t in tri)
    lift = (centers ** 2).sum(axis=1) - radii ** 2
    m = 2.0 * np.array([centers[b] - centers[a], centers[c] - centers[a]])
    rhs = np.array([lift[b] - lift[a], lift[c] - lift[a]])
    det = m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]
    if abs(det) < 1e-15:
        return None
    return np.linalg.solve(m, rhs)


# ===========================================================================
# Stage 1 -- neighbourhood-preserving planar graph initialization (Sec. 4.3)
# ===========================================================================

def _initial_planar_graph(positions, jitter=1e-9, seed=0):
    """Delaunay triangulation of the input layout -> maximal planar graph."""
    positions = np.asarray(positions, dtype=float)
    n = len(positions)
    if n < 3:
        edges = np.array([[0, 1]], dtype=int) if n == 2 else np.zeros((0, 2), int)
        return np.zeros((0, 3), int), edges

    scale = float(np.ptp(positions, axis=0).max()) or 1.0
    try:
        tri = Delaunay(positions)
    except QhullError:
        rng = np.random.default_rng(seed)
        tri = Delaunay(positions + rng.normal(0.0, jitter * scale, positions.shape))
    return tri.simplices, _triangulation_edges(tri.simplices)


def _normalize_weights(weights, n, floor=0.05):
    """Coerce caller weights into strictly positive radii multipliers in (0, 1]."""
    if weights is None:
        return np.ones(n, dtype=float)
    w = np.asarray(weights, dtype=float).ravel()
    if len(w) != n:
        raise ValueError(f'weights has length {len(w)}, expected {n}')
    if not np.all(np.isfinite(w)):
        raise ValueError('weights must be finite')
    if np.any(w < 0):
        raise ValueError('weights must be non-negative')
    hi = float(w.max())
    if hi <= 0:
        return np.ones(n, dtype=float)
    return np.maximum(w, floor * hi) / hi


def _missing_edges(target_edges, current_edges):
    """Target edges absent from the current triangulation (Fig. 6, step 1)."""
    if len(target_edges) == 0:
        return np.zeros((0, 2), dtype=int)
    have = set(map(tuple, np.asarray(current_edges, dtype=int)))
    keep = [tuple(e) for e in np.asarray(target_edges, dtype=int)
            if tuple(e) not in have]
    return np.asarray(keep, dtype=int).reshape(-1, 2)


# ===========================================================================
# Objective and gradient (paper Eqs. 1, 3, 4)
# ===========================================================================

def _f_neighborhood(positions, edges, s):
    if len(edges) == 0:
        return 0.0
    d = np.linalg.norm(positions[edges[:, 0]] - positions[edges[:, 1]], axis=1)
    return 2.0 * float(d.sum()) / s


def _f_compactness(positions, center, s):
    return float(np.linalg.norm(positions - center, axis=1).sum()) / s


def _f_convexity(positions, radii, s, pockets=None):
    if pockets is None:
        pockets = _hull_pockets(positions, radii)
    return float(sum(gap for _i, _m, gap in pockets)) / s


def _objective_total(positions, radii, edges, center, s, alpha=0.2, beta=1.0,
                     convexity=True, pockets=None):
    """Weighted sum of the three objectives; ``beta=0`` gives Eq. 3."""
    val = _f_neighborhood(positions, edges, s) + alpha * _f_compactness(positions, center, s)
    if convexity and beta:
        val += beta * _f_convexity(positions, radii, s, pockets=pockets)
    return val


def _gradient(positions, radii, edges, center, s, alpha=0.2, beta=1.0,
              convexity=True, pockets=None):
    """d(F'_p + alpha F_c + beta F_v) / dp, holding ``s`` fixed."""
    g = np.zeros_like(positions)

    if len(edges) > 0:
        a, b = edges[:, 0], edges[:, 1]
        delta = positions[a] - positions[b]
        norm = np.linalg.norm(delta, axis=1)[:, None]
        unit = delta / np.maximum(norm, EPS)
        np.add.at(g, a, 2.0 * unit / s)
        np.add.at(g, b, -2.0 * unit / s)

    if alpha:
        delta = positions - center
        norm = np.linalg.norm(delta, axis=1)[:, None]
        g += alpha * (delta / np.maximum(norm, EPS)) / s

    if convexity and beta:
        if pockets is None:
            pockets = _hull_pockets(positions, radii)
        for i, m, _gap in pockets:
            delta = positions[i] - m
            norm = float(np.hypot(*delta))
            if norm > EPS:
                g[i] += beta * (delta / norm) / s
    return g


# ===========================================================================
# Stage 2 -- power-diagram-based planar graph layout (Sec. 4.4, Eq. 3)
# ===========================================================================

def _stage2_energy(pos, w, edges, alpha, hint=None):
    s = _max_scale(pos, w, hint=hint)
    center = pos.mean(axis=0)
    return _objective_total(pos, w * s, edges, center, s, alpha=alpha, convexity=False)


def _stage2_line_search(pos, disp, w, edges, alpha, step0, step_min, hint=None):
    """Move along ``disp``, halving the cap until the objective improves."""
    base = _stage2_energy(pos, w, edges, alpha, hint=hint)
    norm = np.linalg.norm(disp, axis=1)[:, None]
    direction = disp / np.maximum(norm, 1e-12)

    step = step0
    while step >= step_min:
        cand = pos + direction * np.minimum(norm, step)
        if _stage2_energy(cand, w, edges, alpha, hint=hint) < base:
            return cand, _max_scale(cand, w, hint=hint), True
        step *= 0.5
    return pos, _max_scale(pos, w, hint=hint), False


def _stage2_centroid_targets(pos, radii):
    simplices = _regular_triangulation(pos, radii)
    if len(simplices) == 0:
        return pos.copy()
    nbrs = _neighbor_lists(_triangulation_edges(simplices), len(pos))
    return _cell_centroids(_power_cells(pos, radii, nbrs), pos)


def _stage2_side(a, b, p):
    return (b[..., 0] - a[..., 0]) * (p[..., 1] - a[..., 1]) - \
           (b[..., 1] - a[..., 1]) * (p[..., 0] - a[..., 0])


def _stage2_crossing_edge(pos, edges, i, j):
    """The triangulation edge crossing segment ``ij``, nearest to ``i``."""
    a, b = pos[i], pos[j]
    p, q = pos[edges[:, 0]], pos[edges[:, 1]]
    shared = (edges[:, 0] == i) | (edges[:, 1] == i) | \
             (edges[:, 0] == j) | (edges[:, 1] == j)
    d1 = _stage2_side(a, b, p)
    d2 = _stage2_side(a, b, q)
    d3 = _stage2_side(p, q, a)
    d4 = _stage2_side(p, q, b)
    cross = (~shared) & (d1 * d2 < 0) & (d3 * d4 < 0)
    idx = np.where(cross)[0]
    if len(idx) == 0:
        return None
    mid = 0.5 * (p[idx] + q[idx])
    return tuple(int(v) for v in edges[idx[np.argmin(np.linalg.norm(mid - a, axis=1))]])


def _stage2_repair_targets(pos, radii, target_edges):
    """Displacements that restore lost target edges (Fig. 6)."""
    simplices = _regular_triangulation(pos, radii)
    if len(simplices) == 0:
        return None, 0
    current = _triangulation_edges(simplices)
    lost = _missing_edges(target_edges, current)
    if len(lost) == 0:
        return None, 0

    disp = np.zeros_like(pos)
    hit = False
    for i, j in lost:
        blocker = _stage2_crossing_edge(pos, current, i, j)
        if blocker is None:
            disp[i] += pos[j] - pos[i]
            hit = True
            continue
        k, l = blocker
        c = _weighted_circumcenter(pos, radii, (j, k, l))
        if c is not None:
            disp[i] += c - pos[i]
            hit = True
    return (disp if hit else None), len(lost)


def _stage2_optimize(positions, weights, edges, alpha=0.2, iterations=60,
                     move_frac=0.35, min_frac=1e-3, repair=True, tol=1e-6):
    """Run Stage 2 (compactness + neighbourhood repair). Returns (pos, s, report)."""
    pos = np.array(positions, dtype=float, copy=True)
    w = np.asarray(weights, dtype=float)
    s = _max_scale(pos, w)

    unit = 2.0 * float((w * s).mean())
    step0 = move_frac * unit
    step_min = max(min_frac * unit, 1e-12)

    report = {'iterations': 0, 'centroid_moves': 0, 'repairs': 0,
              'start_scale': s, 'edges_lost': 0}
    prev = _stage2_energy(pos, w, edges, alpha, hint=s)

    for it in range(iterations):
        moved = False

        target = _stage2_centroid_targets(pos, w * s)
        pos, s, ok = _stage2_line_search(pos, target - pos, w, edges, alpha,
                                         step0, step_min, hint=s)
        if ok:
            report['centroid_moves'] += 1
            moved = True

        if repair:
            disp, lost = _stage2_repair_targets(pos, w * s, edges)
            report['edges_lost'] = lost
            if disp is not None:
                pos, s, ok = _stage2_line_search(pos, disp, w, edges, alpha,
                                                 step0, step_min, hint=s)
                if ok:
                    report['repairs'] += 1
                    moved = True

        report['iterations'] = it + 1
        cur = _stage2_energy(pos, w, edges, alpha, hint=s)
        if not moved or abs(prev - cur) < tol * max(abs(prev), 1.0):
            prev = cur
            break
        prev = cur

    report['end_scale'] = s
    report['objective'] = prev
    return pos, s, report


# ===========================================================================
# Stage 3 -- force-directed refinement (Sec. 4.5, Eq. 4)
# ===========================================================================

def _safe_pockets(pos, radii):
    try:
        return _hull_pockets(pos, radii)
    except Exception:                       # noqa: BLE001 - degenerate hulls
        return []


def _overlap_pairs(pos, radii, cache, pad=0.10):
    """Candidate overlapping pairs, reusing the KD-tree across iterations.

    The cache is reused only while a movement certificate holds
    (``reach + 2 * max_move <= build_radius``), which keeps it a strict
    superset of the true contact set -- a speedup, not an approximation.
    """
    reach = 2.0 * float(radii.max())
    if cache is not None:
        moved = float(np.linalg.norm(pos - cache['origin'], axis=1).max())
        if reach + 2.0 * moved <= cache['radius']:
            return cache['pairs'], cache
    radius = reach * (1.0 + pad)
    pairs = cKDTree(pos).query_pairs(radius, output_type='ndarray')
    return pairs, {'pairs': pairs, 'origin': np.array(pos, copy=True),
                   'radius': radius}


def _resolve_overlaps(pos, radii, passes=6, tol=1e-7, cache=None):
    """Project onto the non-overlap constraint via Gauss-Seidel pair separation.

    Returns ``(positions, cache)``; pass the cache back in on the next call.
    """
    if len(pos) < 2:
        return pos, cache
    pairs, cache = _overlap_pairs(pos, radii, cache)
    if len(pairs) == 0:
        return pos, cache

    a, b = pairs[:, 0], pairs[:, 1]
    need = radii[a] + radii[b]
    limit = tol * max(float(radii.mean()), 1e-12)
    pos = np.array(pos, dtype=float, copy=True)
    for _ in range(passes):
        delta = pos[a] - pos[b]
        dist = np.linalg.norm(delta, axis=1)
        overlap = need - dist
        act = overlap > limit
        if not act.any():
            break
        unit = delta[act] / np.maximum(dist[act], 1e-12)[:, None]
        push = 0.5 * overlap[act][:, None] * unit
        np.add.at(pos, a[act], push)
        np.subtract.at(pos, b[act], push)
    return pos, cache


def _stage3_refine(positions, weights, edges, alpha=0.2, beta=1.0, iterations=1250,
                   step_frac=0.05, pocket_every=25, overlap_passes=6, seed_scale=None):
    """Run Stage 3 (adds convexity). Returns (pos, s, report).

    ``s`` may only grow, never shrink: recomputing it fresh from the projection
    residual each iteration is a ratchet that compounds away the Stage 2 gain.
    """
    pos = np.array(positions, dtype=float, copy=True)
    w = np.asarray(weights, dtype=float)
    s = float(seed_scale) if seed_scale else _max_scale(pos, w)

    step0 = step_frac * 2.0 * float((w * s).mean())
    pockets = None
    pair_cache = None
    report = {'iterations': iterations, 'start_scale': s}

    for it in range(iterations):
        radii = w * s
        if it % pocket_every == 0 and beta:
            pockets = _safe_pockets(pos, radii)

        center = pos.mean(axis=0)
        g = _gradient(pos, radii, edges, center, s, alpha=alpha,
                      beta=beta, convexity=bool(beta), pockets=pockets)

        mag = np.linalg.norm(g, axis=1)[:, None]
        direction = -g / np.maximum(mag, 1e-12)
        speed = step0 * (1.0 - it / float(iterations))       # linear annealing
        pos = pos + direction * np.minimum(mag * s, 1.0) * speed

        pos, pair_cache = _resolve_overlaps(pos, radii, passes=overlap_passes,
                                            cache=pair_cache)
        s = max(s, _max_scale(pos, w, hint=s))

    # settle the constraint hard once at the end, then read s off the geometry
    pos, _ = _resolve_overlaps(pos, w * s, passes=200, tol=1e-9)
    s = _max_scale(pos, w)

    report['end_scale'] = s
    report['objective'] = _objective_total(pos, w * s, edges, pos.mean(axis=0), s,
                                           alpha=alpha, beta=beta, convexity=bool(beta))
    return pos, s, report


# ===========================================================================
# The packing core (paper Fig. 5 pipeline)
# ===========================================================================

def _default_radius(positions):
    span = float(np.ptp(positions, axis=0).max()) or 1.0
    return 0.5 * span / max(np.sqrt(len(positions)), 1.0)


class NeighborhoodPreservingPacking(object):
    """NCP: neighbourhood-preserving non-uniform circle packing (Li et al. 2026).

    Screen-space geometry in, screen-space geometry out. Give ``positions``
    (the layout to compact) and either ``radii`` (absolute pixels) or
    ``weights`` (relative sizes); ``.positions`` / ``.radii`` hold the result.

    ``fit_mode='preserve_radii'`` keeps the requested radii and returns a
    smaller layout centred on the input; ``'fill'`` keeps the input bounding
    box and returns larger radii.
    """

    def __init__(self, positions, radii=None, weights=None,
                 alpha=0.2, beta=1.0,
                 power_iterations=60, force_iterations=1250,
                 move_frac=0.35, step_frac=0.05, pocket_every=25,
                 overlap_passes=6,
                 repair=True, fit_mode='preserve_radii', bounds=None, padding=0.0):
        self.positions_in = np.asarray(positions, dtype=float).reshape(-1, 2)
        n = len(self.positions_in)
        if n == 0:
            raise ValueError('positions is empty')

        if radii is not None:
            r_in = np.asarray(radii, dtype=float).ravel()
            if len(r_in) != n:
                raise ValueError(f'radii has length {len(r_in)}, expected {n}')
            if not np.all(np.isfinite(r_in)) or np.any(r_in <= 0):
                raise ValueError('radii must be finite and positive')
            self.radii_in = r_in
            self.w = r_in / float(r_in.max())
        else:
            self.w = _normalize_weights(weights, n)
            self.radii_in = self.w * _default_radius(self.positions_in)

        self.alpha, self.beta = float(alpha), float(beta)
        self.fit_mode, self.bounds, self.padding = fit_mode, bounds, float(padding)

        # Stage 1 -- neighbourhood planar graph from the current positions
        self.simplices, self.edges = _initial_planar_graph(self.positions_in)

        if n < 4 or len(self.edges) == 0:
            self.positions_raw = self.positions_in.copy()
            self.scale = _max_scale(self.positions_in, self.w)
            self.report = {'skipped': True, 'power': {}, 'forces': {}}
        else:
            # Stage 2 -- power-diagram compaction
            pos, s, rep_power = _stage2_optimize(
                self.positions_in, self.w, self.edges, alpha=self.alpha,
                iterations=power_iterations, move_frac=move_frac, repair=repair)
            # Stage 3 -- force-directed refinement (adds convexity)
            pos, s, rep_force = _stage3_refine(
                pos, self.w, self.edges, alpha=self.alpha, beta=self.beta,
                iterations=force_iterations, step_frac=step_frac,
                pocket_every=pocket_every, overlap_passes=overlap_passes,
                seed_scale=s)
            self.positions_raw = pos
            self.scale = s
            self.report = {'skipped': False, 'power': rep_power, 'forces': rep_force}

        self.radii_raw = self.w * self.scale
        self.positions, self.radii = self._fit()

    def _fit(self):
        pos, r = self.positions_raw, self.radii_raw
        if self.fit_mode == 'fill':
            lo, hi = _target_box(self.bounds, self.positions_in, self.radii_in, self.padding)
            plo = (pos - r[:, None]).min(axis=0)
            phi = (pos + r[:, None]).max(axis=0)
            span = np.maximum(phi - plo, 1e-12)
            k = float(np.min((hi - lo) / span))
            offset = lo + 0.5 * ((hi - lo) - k * span) - k * plo
            return pos * k + offset, r * k

        if self.fit_mode != 'preserve_radii':
            raise ValueError(f"unknown fit_mode {self.fit_mode!r}")
        k = float(self.radii_in.max()) / float(r.max()) if r.max() > 0 else 1.0
        out = pos * k
        return out + (self.positions_in.mean(axis=0) - out.mean(axis=0)), self.radii_in

    def overlaps(self):
        """Largest non-overlap violation in the result; ~0 for a valid packing."""
        return _max_overlap(self.positions, self.radii)


def _max_overlap(positions, radii):
    if len(positions) < 2:
        return 0.0
    d = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)
    need = radii[:, None] + radii[None, :]
    np.fill_diagonal(need, 0.0)
    return float(np.max(need - d))


def _target_box(bounds, positions, radii, padding):
    if bounds is not None:
        lo = np.array([bounds[0], bounds[1]], dtype=float)
        hi = np.array([bounds[2], bounds[3]], dtype=float)
    else:
        lo = (positions - radii[:, None]).min(axis=0)
        hi = (positions + radii[:, None]).max(axis=0)
    return lo + padding, hi - padding


# ===========================================================================
# Graph-facing wrapper -- the linkp layout interface
# ===========================================================================

def _node_size_weights(g, nodes):
    """Per-node radius weight = log(count), count = the node's flow volume.

    ``count`` is the node's total incident edge weight (the ``__count__`` the
    graph carries, i.e. how many flows it sends/receives). When the graph has
    no edge weights at all, it falls back to the number of neighbours, per the
    layout's spec. Counts are read from the full visible graph so a node's size
    reflects its intrinsic volume even when only a subset is being packed.
    """
    w = np.empty(len(nodes), dtype=float)
    for i, node in enumerate(nodes):
        total, any_weight, degree = 0.0, False, 0
        for _u, _v, data in g.edges(node, data=True):
            degree += 1
            _wt_ = data.get('weight')
            if _wt_ is not None:
                total += float(_wt_)
                any_weight = True
        count = total if (any_weight and total > 0) else float(degree)
        w[i] = math.log(count) if count > 1 else 0.0
    return w


class NCPLayout(object):
    """Neighbourhood-preserving circle packing as a linkp layout operation.

    Compacts an existing layout: it reads the current node positions, gives
    each node a circle sized by its flow volume (``log(count)``, or neighbour
    count when the graph is unweighted), and packs them into a tight,
    non-overlapping arrangement that keeps the input's spatial neighbourhoods.

    Satisfies the ``LayoutAlgorithm`` protocol: ``.results()`` returns a
    ``{node: (x, y)}`` dict, restricted to the packed nodes so the caller can
    leave everything else in place.

    Parameters
    ----------
    g : networkx.Graph
        The currently visible graph.
    pos : dict
        ``{node: (x, y)}`` current positions -- the layout being compacted.
    selection : iterable, optional
        When non-empty, only these nodes are packed (others are untouched);
        when empty or None, every positioned node in ``g`` is packed.
    force_iterations, power_iterations, alpha, beta, fit_mode : see
        :class:`NeighborhoodPreservingPacking`.
    """

    def __init__(self, g, *, pos=None, selection=None,
                 alpha=0.2, beta=1.0, power_iterations=60, force_iterations=1250,
                 fit_mode='preserve_radii', **kwargs):
        pos = pos or {}
        _sel_ = set(selection) if selection else set()

        # (3)/(4): selection restricts the packed set; otherwise pack all.
        # (1): the caller passes the currently visible graph as ``g``.
        _candidates_ = (n for n in g.nodes() if n in _sel_) if _sel_ else g.nodes()
        nodes = [n for n in _candidates_ if n in pos]

        self.resulting_positions = {}
        if len(nodes) == 0:
            return
        if len(nodes) < 3:
            # too few to triangulate/pack: leave them exactly where they are
            self.resulting_positions = {n: (float(pos[n][0]), float(pos[n][1])) for n in nodes}
            return

        positions = np.array([pos[n] for n in nodes], dtype=float)
        weights   = _node_size_weights(g, nodes)   # (5): radius ~ log(count)

        packing = NeighborhoodPreservingPacking(
            positions, weights=weights, alpha=alpha, beta=beta,
            power_iterations=power_iterations, force_iterations=force_iterations,
            fit_mode=fit_mode, **kwargs)
        out = packing.positions
        self.packing = packing
        self.resulting_positions = {
            n: (float(out[i, 0]), float(out[i, 1])) for i, n in enumerate(nodes)
        }

    def results(self) -> dict:
        """Return {node: (x, y)} for the packed nodes."""
        return self.resulting_positions
