# Layered flow-field background for an existing node layout.
#
# A *background producer*, not a layout: it moves no nodes.  Given the positions
# a linkp is already drawing and the individual edge records behind it, it
# returns `{name: BackgroundShape}` cells -- one per flow layer, in world
# coordinates -- that linkp draws underneath the links to show which way traffic
# is actually moving.  (`od_flow_layout.py` is the unrelated Jenny et al. flow
# map, which *is* a layout.)
#
# The core idea:
#
#   A vector field holds ONE vector per point, so where opposing or crossing
#   flows overlap they cancel -- two hosts talking both ways yield a zero field
#   along the edge between them.  The fix is to allow K vectors per point by
#   splitting the edges into K layers that are each internally coherent, then
#   drawing one field per layer.  K is `k_layers=` (2 by default; 1 collapses
#   to the plain net field, and 3+ separates a third crossing stream).
#
# Every edge lands in exactly one of the K layers -- nothing is dropped -- via
# matching pursuit over coherent flow structures, so K is a hard count rather
# than a cap (see `_assign_layers`).
#
# Each edge deposits onto the grid points within `kernel_cutoff * sigma` of its
# segment, stored sparsely (`_sparse_kernels`).  That footprint, summed over the
# edges, is what the algorithm costs -- roughly 6M cell-entries per second at 16
# bytes each -- and it grows with edge LENGTH as well as edge count, so a layout
# whose edges span the canvas costs several times one whose edges are local.
# `support_budget=` bounds it: the grid is coarsened first (detail is lost,
# every edge is kept) and only then are the lightest edges dropped.  Without it
# a dense graph of long edges will exhaust memory rather than run slowly.
#
# Output coordinates are WORLD coordinates (the same space as `pos`), because
# that is what linkp's `background=` transform expects -- so the background pans
# and zooms with the view for free.
#
# NumPy and Polars only -- both core dependencies, so unlike the graph layouts
# this needs no optional extra.  The hot loop is per-edge over a local support
# of a few hundred grid points, far below the GPU break-even; the same call made
# for ncp_layout.

from __future__ import annotations

import math

import numpy as np

from polars2svg.p2s_background_mixin import BackgroundShape, INHERIT

EPS = 1e-12

# Default ceiling on the total kernel footprint, in grid cells.  6M cells is
# ~96MB and ~1s -- comfortable for a background a user re-runs by hand, and far
# enough below the point where a dense graph of long edges exhausts memory.
DEFAULT_SUPPORT_BUDGET = 6_000_000

# Default ceiling on aggregated edges.  Two different things cost money here and
# a single budget cannot bound both: the kernel footprint is memory and scales
# with edge LENGTH (support_budget), while the layer assignment is time and
# scales with edge COUNT -- about 33us per edge at k_layers=2, measured, because
# each edge is visited once per layer per pass.  25k edges is roughly a second,
# and is already far past what linkp can usefully draw as a node-link diagram.
DEFAULT_MAX_EDGES = 25_000

# A layer whose field over a group's support is this small relative to the
# strongest layer there counts as untouched by that group (see _assign_layers).
FREE_LAYER_REL = 1e-6

# Fallback palette for layerAppearance(); layer 1 (the dominant flow) first.
_PALETTE = ('#2b6ca3', '#c8642a', '#4f9d5b', '#8a5fa8', '#b0913b', '#a34f5e')


# ===========================================================================
# Grid
# ===========================================================================

class _Grid(object):
    """Uniform grid with square cells over the layout's bounding box.

    Flat index of cell (ix, iy) is ``iy * nx + ix`` -- row-major with y outer,
    matching a ``(ny, nx)`` reshape.
    """

    def __init__(self, positions, grid_res, pad):
        lo, hi = positions.min(axis=0), positions.max(axis=0)
        ext    = hi - lo
        # A degenerate axis (every node on one line) still needs a width.
        span   = max(float(ext[0]), float(ext[1]))
        if span <= 0.0:
            span = 1.0
        ext = np.maximum(ext, 0.02 * span)
        lo, hi = lo - pad * ext, hi + pad * ext

        w, h = float(hi[0] - lo[0]), float(hi[1] - lo[1])
        if w >= h:
            self.nx = int(max(2, grid_res))
            self.ny = int(max(2, round(grid_res * h / w)))
        else:
            self.ny = int(max(2, grid_res))
            self.nx = int(max(2, round(grid_res * w / h)))

        self.gx     = np.linspace(lo[0], hi[0], self.nx)
        self.gy     = np.linspace(lo[1], hi[1], self.ny)
        self.bounds = (float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1]))
        self.cell   = min(w / max(self.nx - 1, 1), h / max(self.ny - 1, 1))
        self.size   = self.nx * self.ny
        self.span   = max(w, h)


# ===========================================================================
# Sparse deposition kernels
# ===========================================================================

def _sparse_kernels(grid, A, B, sigma, cutoff):
    """Per-edge Gaussian-on-distance-to-segment support.

    Returns ``(starts, idx, val)`` in CSR form: edge ``e`` deposits onto grid
    points ``idx[starts[e]:starts[e+1]]`` with weights ``val[...]``.  Only
    points within ``cutoff * sigma`` of the segment are stored (at cutoff 3.0
    the discarded tail is below 1.1% of the peak), which is what keeps this
    usable at netflow edge counts -- the dense (E, G) matrix the retired
    prototype built is ~1000x larger at grid_res=48.
    """
    E      = len(A)
    reach  = cutoff * sigma
    reach2 = reach * reach
    two_s2 = 2.0 * sigma * sigma
    gx, gy = grid.gx, grid.gy
    nx     = grid.nx

    AB   = B - A
    seg2 = (AB ** 2).sum(axis=1)

    starts = np.zeros(E + 1, dtype=np.int64)
    idx_parts, val_parts = [], []

    for e in range(E):
        ax, ay = A[e]
        abx, aby = AB[e]

        ix0 = int(np.searchsorted(gx, min(ax, ax + abx) - reach, 'left'))
        ix1 = int(np.searchsorted(gx, max(ax, ax + abx) + reach, 'right'))
        iy0 = int(np.searchsorted(gy, min(ay, ay + aby) - reach, 'left'))
        iy1 = int(np.searchsorted(gy, max(ay, ay + aby) + reach, 'right'))
        if ix0 >= ix1 or iy0 >= iy1:
            starts[e + 1] = starts[e]
            continue

        apx = gx[ix0:ix1][None, :] - ax          # (1, bx) -> broadcasts to (by, bx)
        apy = gy[iy0:iy1][:, None] - ay          # (by, 1)
        if seg2[e] > EPS:
            t = np.clip((apx * abx + apy * aby) / seg2[e], 0.0, 1.0)
        else:
            t = np.zeros((iy1 - iy0, ix1 - ix0))
        dx = apx - t * abx
        dy = apy - t * aby
        d2 = dx * dx + dy * dy

        keep = d2 <= reach2
        if not keep.any():
            starts[e + 1] = starts[e]
            continue

        flat = (np.arange(iy0, iy1)[:, None] * nx + np.arange(ix0, ix1)[None, :])
        idx_parts.append(flat[keep].astype(np.int64))
        val_parts.append(np.exp(-d2[keep] / two_s2))
        starts[e + 1] = starts[e] + int(keep.sum())

    if idx_parts:
        idx = np.concatenate(idx_parts)
        val = np.concatenate(val_parts)
    else:
        idx = np.zeros(0, dtype=np.int64)
        val = np.zeros(0)
    return starts, idx, val


def _estimated_support(grid, A, B, reach):
    """Grid cells the kernels will occupy, without building them.

    An edge's support is a capsule: every point within ``reach`` of the segment,
    i.e. a ``L x 2*reach`` rectangle plus a disc of radius ``reach``.  Divided by
    the cell area that is the entry count, and summing over the edges gives the
    memory and the running time before either is committed.
    """
    L = np.sqrt(((B - A) ** 2).sum(axis=1))
    area = L * (2.0 * reach) + math.pi * reach * reach
    return area / (grid.cell * grid.cell)


def _fit_to_budget(positions, A, B, weights, grid_res, pad, sigma_frac, sigma,
                   cutoff, budget, min_grid_res):
    """Choose a grid (and possibly an edge subset) whose support fits ``budget``.

    Two-stage degradation, in the order that costs the least information:

      1. **Coarsen the grid.**  Support scales as ``1 / cell^2``, so dropping the
         resolution buys back quadratically.  Every edge is kept; what is lost is
         spatial detail, which is the cheaper thing to lose in a background.
      2. **Drop the lightest edges.**  Only once the grid is at ``min_grid_res``.
         Edges are kept heaviest-first until the running support estimate fills
         the budget, so what goes is the tail that was contributing least.

    Returns ``(grid, sigma, keep_index_or_None, note)`` -- ``note`` is None when
    nothing was degraded, otherwise a sentence for ``summary()``.
    """
    grid  = _Grid(positions, grid_res, pad)
    sig   = float(sigma) if sigma is not None else sigma_frac * grid.span
    if not budget or len(A) == 0:
        return grid, sig, None, None

    est = float(_estimated_support(grid, A, B, cutoff * sig).sum())
    if est <= budget:
        return grid, sig, None, None

    # 1. coarsen -- support ~ 1/cell^2 ~ grid_res^2
    res = max(int(min_grid_res), int(grid_res * math.sqrt(budget / est)))
    if res < grid_res:
        coarse = _Grid(positions, res, pad)
        sig_c  = float(sigma) if sigma is not None else sigma_frac * coarse.span
        est_c  = float(_estimated_support(coarse, A, B, cutoff * sig_c).sum())
        note   = (f'support budget: grid coarsened {grid_res} -> {res} '
                  f'({est / 1e6:.1f}M -> {est_c / 1e6:.1f}M cells)')
        grid, sig, est = coarse, sig_c, est_c
        if est <= budget:
            return grid, sig, None, note
    else:
        note = f'support budget: {est / 1e6:.1f}M cells over budget'

    # 2. keep the heaviest edges whose cumulative support fits
    per   = _estimated_support(grid, A, B, cutoff * sig)
    order = np.argsort(-weights, kind='stable')
    keep  = order[np.cumsum(per[order]) <= budget]
    if len(keep) == 0:
        keep = order[:1]
    keep.sort()
    note += f'; kept the heaviest {len(keep)} of {len(A)} edges'
    return grid, sig, keep, note


# ===========================================================================
# K-layer assignment
# ===========================================================================

def _edge_components(ends, dirs, cos_thr=0.5):
    """Union-find the edges into coherent flow structures.

    Two edges join when they share an endpoint AND their directions agree
    (``cos > cos_thr``) -- so a chain of hops going the same way is one
    structure, while the return path through the same nodes is a different one.
    A structure is what gets placed in a layer, which is what keeps a stream
    from being torn in half (the fragmentation the retired greedy variant
    suffered from).

    The retired ``edge_components`` compared every pair of edges incident on a
    node, which is quadratic in the degree -- fatal at a netflow hub.  Here each
    node's incident edges are swept in angle order and only angular neighbours
    are tested; union-find closes the same transitive structures at
    ``O(d log d)``.
    """
    E = len(ends)
    parent = np.arange(E)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    ang = np.arctan2(dirs[:, 1], dirs[:, 0])
    max_gap = math.acos(max(-1.0, min(1.0, cos_thr)))

    incident = {}
    for e, (u, v) in enumerate(ends):
        incident.setdefault(int(u), []).append(e)
        incident.setdefault(int(v), []).append(e)

    for es in incident.values():
        if len(es) < 2:
            continue
        es = sorted(es, key=lambda e: ang[e])
        for i in range(len(es)):
            a, b = es[i], es[(i + 1) % len(es)]
            gap = abs(ang[b] - ang[a])
            gap = min(gap, 2.0 * math.pi - gap)
            if gap <= max_gap:
                union(a, b)

    comps = {}
    for e in range(E):
        comps.setdefault(int(find(e)), []).append(e)
    return [np.array(c, dtype=np.int64) for c in comps.values()]


def _assign_layers(k_layers, weights, ends, dirs, starts, idx, val, grid_size,
                   tau=0.2, refine_passes=1, atomic_components=True,
                   cos_thr=0.5, max_component_frac=0.55):
    """Split the edges into exactly ``k_layers`` internally coherent layers.

    Matching pursuit over flow structures (``_edge_components``) in descending
    throughput order.  A structure is scored against every layer's current
    field, over the supports of its own edges:

        score_k = sum_e sum_g w_e K_e(g) (F_k(g) . d_e)
                  ------------------------------------
                  sum_e sum_g w_e K_e(g) |F_k(g)|

    which is the throughput-weighted mean cosine between the structure and what
    the layer already holds there.  A layer the structure does not touch at all
    scores exactly ``tau``: an untouched layer therefore beats a layer the
    structure would fight (cos < tau) and loses to one it agrees with
    (cos > tau).  So the dominant flow keeps consolidating into layer 0 and the
    later layers fill with the counter-flows and crossings layer 0 cannot
    represent.  Ties go to the lowest-index layer.

    Nothing is ever dropped -- with all K layers in conflict the least-bad one
    wins, which is what makes K a hard parameter rather than a cap.
    ``refine_passes`` re-runs each structure's decision against the finished
    fields (its own contribution removed first), repairing choices made before
    the competing layer existed.

    ``max_component_frac`` is the safety valve on the clustering: transitivity
    can chain a whole graph into one structure, and a structure holding more
    than this share of the throughput is broken back into single edges rather
    than swallowing a layer whole.

    Returns ``(label, U, V)`` with ``U``/``V`` shaped ``(k_layers, grid_size)``.
    """
    E = len(weights)
    U = np.zeros((k_layers, grid_size))
    V = np.zeros((k_layers, grid_size))
    label = np.full(E, -1, dtype=np.int32)

    if atomic_components and E > 1:
        groups = _edge_components(ends, dirs, cos_thr)
        total  = float(weights.sum())
        capped = []
        for c in groups:
            if len(c) > 1 and float(weights[c].sum()) > max_component_frac * total:
                capped += [np.array([e], dtype=np.int64) for e in c]
            else:
                capped.append(c)
        groups = capped
    else:
        groups = [np.array([e], dtype=np.int64) for e in range(E)]

    # Heaviest structure first, and inside one, heaviest edge first.
    groups = [g[np.argsort(-weights[g], kind='stable')] for g in groups]
    groups.sort(key=lambda g: (-float(weights[g].sum()), int(g[0])))

    def _deposit(group, k, sign):
        for e in group:
            s, t = starts[e], starts[e + 1]
            if s == t:
                continue
            gi, gv = idx[s:t], val[s:t]
            U[k, gi] += sign * weights[e] * gv * dirs[e, 0]
            V[k, gi] += sign * weights[e] * gv * dirs[e, 1]

    def _place(group, remove=False):
        if remove:
            _deposit(group, int(label[group[0]]), -1.0)

        # num/den per layer, plus that layer's peak field magnitude over this
        # group's support.  The peak is what decides "untouched": removing a
        # group leaves float residue rather than an exact zero, and num/den of
        # pure residue is a meaningless finite number that would otherwise beat
        # the free-layer score and pin the group where it already was.
        nums, dens, peaks = [], [], []
        for k in range(k_layers):
            num = den = peak = 0.0
            for e in group:
                s, t = starts[e], starts[e + 1]
                if s == t:
                    continue
                gi, gv = idx[s:t], val[s:t]
                uk, vk = U[k, gi], V[k, gi]
                mag    = np.sqrt(uk * uk + vk * vk)
                w      = weights[e] * gv
                num += float((w * (uk * dirs[e, 0] + vk * dirs[e, 1])).sum())
                den += float((w * mag).sum())
                peak = max(peak, float(mag.max()))
            nums.append(num); dens.append(den); peaks.append(peak)

        peak_all = max(peaks)
        best_k, best_score = 0, -np.inf
        for k in range(k_layers):
            free  = peak_all <= EPS or peaks[k] <= FREE_LAYER_REL * peak_all
            score = tau if (free or dens[k] <= EPS) else nums[k] / dens[k]
            if score > best_score:
                best_k, best_score = k, score

        _deposit(group, best_k, 1.0)
        label[group] = best_k
        return best_k

    for group in groups:
        _place(group)
    for _ in range(int(refine_passes)):
        for group in groups:
            _place(group, remove=True)

    # Order layers by throughput so 'flow 1' is always the dominant one.
    tp   = np.array([weights[label == k].sum() for k in range(k_layers)])
    perm = np.argsort(-tp, kind='stable')
    if not np.array_equal(perm, np.arange(k_layers)):
        rank = np.empty(k_layers, dtype=np.int32)
        rank[perm] = np.arange(k_layers, dtype=np.int32)
        label = rank[label]
        U, V = U[perm], V[perm]
    return label, U, V


# ===========================================================================
# Glyphs -- world-coordinate SVG path descriptors
# ===========================================================================

def _fmt(v):
    return f'{v:.6g}'


def _arrow_subpath(x0, y0, dx, dy, length, width_scale=1.0,
                   head_frac=0.42, shaft_frac=0.15, head_frac_w=0.42):
    """One closed arrow polygon whose TAIL sits on (x0, y0), pointing (dx, dy).

    Tail-anchored rather than centred so that every layer's arrow for a given
    grid cell leaves from the same point: they then fan out from one origin like
    a wind rose instead of being drawn through each other.  What tells them
    apart is ``width_scale`` -- the layer furthest back is drawn widest and each
    layer above it narrows, so the ones on top stay visible against the ones
    beneath (see ``_layer_width_scales``).

    Drawn as a filled dart -- its record asks for ``fill=<colour>,
    stroke=None`` -- though it still reads correctly as a stroked outline if a
    caller overrides that.
    """
    nx_, ny_ = -dy, dx                       # unit normal
    # Head length tracks the width ramp only halfway: scaled fully, a narrow
    # arrow keeps a long head and reads as a needle.
    hl = length * head_frac * (0.5 + 0.5 * width_scale)
    sw = length * shaft_frac  * 0.5 * width_scale
    hw = length * head_frac_w * 0.5 * width_scale
    tipx, tipy = x0 + length * dx, y0 + length * dy
    nekx, neky = tipx - hl * dx,   tipy - hl * dy

    pts = ((x0   + sw * nx_, y0   + sw * ny_),
           (nekx + sw * nx_, neky + sw * ny_),
           (nekx + hw * nx_, neky + hw * ny_),
           (tipx,            tipy),
           (nekx - hw * nx_, neky - hw * ny_),
           (nekx - sw * nx_, neky - sw * ny_),
           (x0   - sw * nx_, y0   - sw * ny_))
    out = [f'M {_fmt(pts[0][0])} {_fmt(pts[0][1])}']
    out += [f'L {_fmt(x)} {_fmt(y)}' for x, y in pts[1:]]
    out.append('Z')
    return ' '.join(out)


def _layer_width_scales(k_layers, falloff):
    """Arrow width factor per layer: widest at the back, narrowing forward.

    linkp draws the background cells in dict order, so layer 0 lands furthest
    back and layer K-1 sits on top.  A geometric ramp keeps every layer above
    the first legible against it without any of them vanishing.
    """
    return [falloff ** k for k in range(k_layers)]


def _arrow_glyphs(grid, U, V, gmax, stride, min_magnitude, arrow_scale,
                  width_scale=1.0, min_len_frac=0.35):
    """Arrow field: one glyph per `stride`-th grid cell above the threshold.

    Returns ``(path, reach)`` -- the path descriptor and the furthest any glyph
    extends from its cell, which is what the caller needs to know to keep the
    field inside linkp's plot area.
    """
    nx_, ny_ = grid.nx, grid.ny
    u = U.reshape(ny_, nx_)
    v = V.reshape(ny_, nx_)
    mag = np.sqrt(u * u + v * v)

    full  = grid.cell * stride * arrow_scale
    subs  = []
    reach = 0.0
    for iy in range(stride // 2, ny_, stride):
        for ix in range(stride // 2, nx_, stride):
            m = mag[iy, ix]
            if m <= EPS or (m / gmax) < min_magnitude:
                continue
            f = math.sqrt(m / gmax)                       # perceptual length ramp
            length = full * (min_len_frac + (1.0 - min_len_frac) * f)
            reach  = max(reach, length)
            subs.append(_arrow_subpath(grid.gx[ix], grid.gy[iy],
                                       u[iy, ix] / m, v[iy, ix] / m, length,
                                       width_scale=width_scale))
    return ' '.join(subs), reach


def _circle_subpath(cx, cy, r):
    """A closed circle as four cubic Beziers -- round at any zoom, and in the
    M/L/C/Z dialect linkp's background transform accepts (a real <circle> is a
    whole-shape descriptor, so it cannot be mixed into a multi-glyph path)."""
    k = 0.5522847498 * r
    return (f'M {_fmt(cx + r)} {_fmt(cy)} '
            f'C {_fmt(cx + r)} {_fmt(cy + k)} {_fmt(cx + k)} {_fmt(cy + r)} {_fmt(cx)} {_fmt(cy + r)} '
            f'C {_fmt(cx - k)} {_fmt(cy + r)} {_fmt(cx - r)} {_fmt(cy + k)} {_fmt(cx - r)} {_fmt(cy)} '
            f'C {_fmt(cx - r)} {_fmt(cy - k)} {_fmt(cx - k)} {_fmt(cy - r)} {_fmt(cx)} {_fmt(cy - r)} '
            f'C {_fmt(cx + k)} {_fmt(cy - r)} {_fmt(cx + r)} {_fmt(cy - k)} {_fmt(cx + r)} {_fmt(cy)} Z')


def _diamond_subpath(cx, cy, r):
    """A closed diamond -- the straight-line alternative to _circle_subpath."""
    return (f'M {_fmt(cx + r)} {_fmt(cy)} L {_fmt(cx)} {_fmt(cy + r)} '
            f'L {_fmt(cx - r)} {_fmt(cy)} L {_fmt(cx)} {_fmt(cy - r)} Z')


def _sample_field(u, v, grid, x, y):
    """Bilinear sample of a (ny, nx) field at world (x, y); (0,0) outside."""
    fx = (x - grid.gx[0]) / (grid.gx[-1] - grid.gx[0]) * (grid.nx - 1)
    fy = (y - grid.gy[0]) / (grid.gy[-1] - grid.gy[0]) * (grid.ny - 1)
    if fx < 0 or fy < 0 or fx > grid.nx - 1 or fy > grid.ny - 1:
        return 0.0, 0.0
    ix, iy = int(min(fx, grid.nx - 2)), int(min(fy, grid.ny - 2))
    tx, ty = fx - ix, fy - iy
    w00, w10 = (1 - tx) * (1 - ty), tx * (1 - ty)
    w01, w11 = (1 - tx) * ty,       tx * ty
    su = (w00 * u[iy, ix] + w10 * u[iy, ix + 1] + w01 * u[iy + 1, ix] + w11 * u[iy + 1, ix + 1])
    sv = (w00 * v[iy, ix] + w10 * v[iy, ix + 1] + w01 * v[iy + 1, ix] + w11 * v[iy + 1, ix + 1])
    return float(su), float(sv)


def _streamline_glyphs(grid, U, V, gmax, min_magnitude, spacing=1.6,
                       max_steps=60, step_frac=0.5, min_steps=6,
                       marker='circle', marker_size=0.1):
    """Evenly-spaced streamlines (Jobard & Lefebvre).

    Seed on the strongest cells first, integrate RK2 in BOTH directions from
    the seed, and stop on entering a cell another streamline already owns --
    which is what keeps the spacing even instead of clumping on the hot spots.

    The downstream end carries a small **circle** (or diamond), not an
    arrowhead: an arrowhead on a background curve reads as graph structure --
    a directed edge with a real arrow on it -- and competes with the node-link
    drawing it sits under.  A dot is the head, so direction still reads, and
    nothing about it looks like an edge.

    Open subpaths: meant for a stroked background (linkp's default
    ``fill=None`` on their record), not a filled one.
    """
    nx_, ny_ = grid.nx, grid.ny
    u = U.reshape(ny_, nx_)
    v = V.reshape(ny_, nx_)
    mag = np.sqrt(u * u + v * v)

    occ_res  = max(1.0 / max(spacing, 0.1), 0.05)
    onx, ony = max(2, int(nx_ * occ_res)), max(2, int(ny_ * occ_res))
    occupied = np.zeros((ony, onx), dtype=bool)

    def _occ_cell(x, y):
        cx = int((x - grid.bounds[0]) / max(grid.bounds[2] - grid.bounds[0], EPS) * (onx - 1))
        cy = int((y - grid.bounds[1]) / max(grid.bounds[3] - grid.bounds[1], EPS) * (ony - 1))
        return min(max(cy, 0), ony - 1), min(max(cx, 0), onx - 1)

    step = grid.cell * step_frac

    def _trace(x, y, sign, owned):
        """RK2 along (or against) the field until it fades or hits a taken cell."""
        pts = []
        for _ in range(max_steps):
            su, sv = _sample_field(u, v, grid, x, y)
            m = math.hypot(su, sv)
            if m <= EPS or (m / gmax) < min_magnitude:
                break
            mx = x + sign * 0.5 * step * su / m
            my = y + sign * 0.5 * step * sv / m
            mu, mv = _sample_field(u, v, grid, mx, my)
            mm = math.hypot(mu, mv)
            if mm <= EPS:
                break
            x, y = x + sign * step * mu / mm, y + sign * step * mv / mm
            c = _occ_cell(x, y)
            if occupied[c]:
                break
            owned.add(c)
            pts.append((x, y))
        return pts

    seeds = [(mag[iy, ix], iy, ix)
             for iy in range(ny_) for ix in range(nx_)
             if mag[iy, ix] > EPS and (mag[iy, ix] / gmax) >= min_magnitude]
    seeds.sort(key=lambda s: (-s[0], s[1], s[2]))

    line_subs, head_subs = [], []
    for _m, iy, ix in seeds:
        x, y = float(grid.gx[ix]), float(grid.gy[iy])
        c = _occ_cell(x, y)
        if occupied[c]:
            continue
        owned = {c}
        fwd  = _trace(x, y,  1.0, owned)
        back = _trace(x, y, -1.0, owned)
        pts  = back[::-1] + [(x, y)] + fwd
        if len(pts) < min_steps or not fwd:
            continue
        for cy, cx in owned:
            occupied[cy, cx] = True

        d = [f'M {_fmt(pts[0][0])} {_fmt(pts[0][1])}']
        d += [f'L {_fmt(px)} {_fmt(py)}' for px, py in pts[1:]]
        line_subs.append(' '.join(d))

        # Marker at the downstream end: that is the head, and where the line
        # stops is where the flow is going.  It goes in its OWN cell, because a
        # cell is one path and a path carries one fill decision -- filling the
        # cell that holds the curves would fill the curves too (SVG closes an
        # open subpath implicitly to fill it).
        r = grid.cell * marker_size
        if marker == 'circle':
            head_subs.append(_circle_subpath(pts[-1][0], pts[-1][1], r))
        elif marker == 'diamond':
            head_subs.append(_diamond_subpath(pts[-1][0], pts[-1][1], r))

    # A streamline can land at most one integration step outside the grid
    # (_sample_field then returns zero and the trace stops), and the marker sits
    # on that last point -- so that sum is how far the layer really reaches.
    return (' '.join(line_subs), ' '.join(head_subs),
            grid.cell * (marker_size + step_frac))


# ===========================================================================
# Naming / styling helpers (usable before the layout runs)
# ===========================================================================

def layerNames(k_layers, prefix='flow'):
    """One name per flow layer, strongest first.

    Deterministic and independent of the data, so the palette can be inspected
    or overridden (via :func:`layerAppearance`) before any layout has been run.
    """
    return [f'{prefix} {i + 1}' for i in range(int(k_layers))]


def headNames(k_layers, prefix='flow'):
    """The companion cell holding each streamline layer's head markers.

    Heads are a separate background cell so they can be *filled* while the
    curves stay stroke-only -- one cell is one path, and one path gets one fill
    decision.  Arrow glyphs need no such split.
    """
    return [f'{n} heads' for n in layerNames(k_layers, prefix)]


def cellNames(k_layers, prefix='flow', glyph='arrow'):
    """Every background cell name, in the order they are drawn (back first)."""
    if glyph == 'arrow':
        return layerNames(k_layers, prefix)
    out = []
    for name, head in zip(layerNames(k_layers, prefix), headNames(k_layers, prefix)):
        out += [name, head]          # a layer's heads sit above its own curves
    return out


def layerAppearance(k_layers, prefix='flow', colors=None, glyph='arrow',
                    opacity=0.6, opacity_falloff=0.0, stroke_w=1.1):
    """``{cell_name: {BackgroundShape field: value}}`` -- how each cell paints.

    This is what :class:`FlowFieldBackground` stamps onto its records, and it is callable
    before any layout runs (the names are data-independent), so a caller can see
    the palette without computing a field.

    Every field that matters is stated rather than inherited, and ``None`` is
    used for its real meaning -- *off* -- which is the whole point of the
    record contract (PLANNING.md B1):

      * arrows are closed darts: filled, never stroked;
      * streamline curves are open: stroked at ``stroke_opacity``, never filled
        (an open subpath is closed implicitly in order to fill it);
      * streamline heads are closed circles: filled, never stroked, and
        ``label=None`` because ``'flow 1 heads'`` is an implementation artifact
        that has no business being drawn on the canvas.

    Layer cells state ``label=INHERIT`` explicitly rather than relying on the
    field default: it is the same value, but next to a head cell saying
    ``label=None`` the difference is the whole point -- INHERIT leaves the
    interactive ``b`` cycle in charge of whether layer names are drawn, None
    takes the decision away.

    ``opacity_falloff`` is 0 by default: for arrows the width ramp
    (``arrow_width_falloff``) already encodes the front-to-back ordering, and
    fading on top of it makes the topmost layer disappear.
    """
    names  = layerNames(k_layers, prefix)
    colors = list(colors) if colors else [_PALETTE[i % len(_PALETTE)] for i in range(len(names))]
    fade   = [max(0.15, opacity - i * opacity_falloff) for i in range(len(names))]

    out = {}
    if glyph == 'arrow':
        for name, color, alpha in zip(names, colors, fade):
            out[name] = dict(fill=color, fill_opacity=alpha, stroke=None, label=INHERIT)
        return out

    for name, head, color, alpha in zip(names, headNames(k_layers, prefix), colors, fade):
        out[name] = dict(fill=None, stroke=color, stroke_opacity=alpha,
                         stroke_width=stroke_w,
                         stroke_linecap='round', stroke_linejoin='round',
                         label=INHERIT)
        out[head] = dict(fill=color, fill_opacity=alpha, stroke=None, label=None)
    return out


# ===========================================================================
# FlowFieldBackground
# ===========================================================================

class FlowFieldBackground(object):
    """Layered flow-field background for an existing node layout.

    Moves no nodes and returns none: :meth:`cells` is the whole output, a
    ``{name: BackgroundShape}`` dict in world coordinates that goes straight to
    ``background=``::

        ffb = p2s.FlowFieldBackground(df, [('src', 'dst')], pos=pos, k_layers=2)
        p2s.linkp(df, [('src', 'dst')], pos, background=ffb.cells())

    Each cell carries its own appearance, so no ``background_*`` arguments
    accompany it.  Deliberately **not** a ``LayoutAlgorithm``: it describes a
    layout rather than producing one, and there is no ``results()`` to mistake
    for one.  In ``linkpi`` it is a *background operation* (shift-b), which
    leaves positions, undo history and the view window alone -- and, being
    contextual rather than owned by a layout, it survives node drags and later
    layouts until something else supersedes it.

    Parameters
    ----------
    df : polars.DataFrame | networkx.Graph
        The individual edge records linkp is rendering (one row per flow /
        event), or -- for convenience at the registry call site -- an already
        built graph, whose ``weight`` edge attribute is then the flow volume.
    relationships : list
        linkp's ``relationships=``: ``[(from_field, to_field)]`` (a third
        element is ignored here).  Either field may be a tuple of columns,
        which is concatenated with ``|`` exactly as ``createConcatColumn()``
        does.  Required for the DataFrame form.
    pos : dict
        ``{node: (x, y)}`` -- the layout the flow map describes.  Nodes missing
        from ``pos`` drop out of the field.
    k_layers : int
        How many flow layers to split the edges into.  1 = the plain net field
        (counter-flows cancel); 2 = dominant flow + what fights it; 3+ peels a
        further crossing stream.  Every edge lands in exactly one layer.
    count : str | None
        Flow volume per aggregated edge, following linkp's ``count=``: a
        numeric column name sums it, a non-numeric column name counts its
        distinct values, and anything else (``None``, or ``p2s.ROW_COUNTp``
        forwarded straight off a LinkP) is the row count.
    selection : iterable | None
        When non-empty, only edges incident on these nodes contribute.
    log_weights : bool
        Compress the weight range with ``log1p`` before deposition (default
        True).  Netflow volumes span six orders of magnitude; unlogged, one
        trunk owns the entire field.
    grid_res : int
        Cells along the longer axis of the layout's bounding box.
    sigma, sigma_frac : float
        Deposition radius, absolute or as a fraction of the longer axis.
    pad : float
        Grid padding beyond the node bounding box, as a fraction of its extent.
        A glyph anchored on the outermost grid line overhangs it by up to
        ``glyph_reach``, so keep ``pad`` plus that overhang under linkp's
        ``bounds_percent`` (0.05 by default) and it lands inside linkp's own
        margin instead of being clipped at the canvas edge.
    kernel_cutoff : float
        Support radius in sigmas (3.0 keeps >98.9% of each kernel's mass).
    tau : float
        Coherence threshold -- see ``_assign_layers``.
    refine_passes : int
        Re-assignment sweeps after the first pass.
    atomic_components : bool
        Place whole flow structures (chains of same-direction edges through
        shared nodes) rather than individual edges, so a stream cannot be torn
        between two layers.  ``False`` assigns edge by edge.
    cos_thr : float
        How closely two edges through a shared node must agree in direction to
        belong to the same structure.
    glyph : {'arrow', 'streamline'}
        Arrows are filled darts on a decimated grid, all of a cell's layers
        leaving from the same point; streamlines are evenly-spaced integral
        curves with a small circle marking the downstream end (stroke them).
    glyph_stride : int
        Draw an arrow every Nth grid cell.
    min_magnitude : float
        Skip anywhere below this fraction of the strongest cell across all
        layers, so glyphs appear only where there is flow to describe.
    arrow_scale, streamline_spacing, streamline_steps : float
        Glyph sizing knobs.  ``arrow_scale`` is the arrow length as a fraction
        of ``glyph_stride`` cells, measured forward from the cell.
    arrow_width_falloff : float
        Geometric width ramp across the layers: layer 0 is drawn furthest back
        and widest, and each layer above it is this much narrower again, so the
        near layers stay legible on top of the far ones.
    streamline_marker : {'circle', 'diamond', 'none'}
        What marks a streamline's downstream end.  Deliberately not an
        arrowhead -- see ``_streamline_glyphs``.
    streamline_marker_size : float
        Marker radius in grid cells.  The marker is filled, so it reads at a
        much smaller size than an outlined one did.
    name_prefix : str
        Cell names are ``f'{name_prefix} {i+1}'``.
    max_edges : int | None
        Ceiling on aggregated edges, applied before anything else -- the time
        half of the guard, since assignment visits every edge once per layer per
        pass.  ``None`` removes it.
    support_budget : int | None
        Ceiling on the total kernel footprint, in grid cells (each costs 16
        bytes and about 0.17 microseconds).  Over it, the grid is coarsened
        toward ``min_grid_res`` first and the lightest edges dropped only if
        that is not enough; :meth:`summary` reports whatever happened.  ``None``
        disables the guard -- appropriate for a one-off render, not for anything
        a user can invoke on arbitrary data, since cost grows with edge length
        as well as edge count.
    min_grid_res : int
        Floor for the coarsening stage.
    """

    def __init__(self, df=None, relationships=None, *, pos=None, k_layers=2,
                 count=None, selection=None, log_weights=True,
                 grid_res=48, sigma=None, sigma_frac=0.05, pad=0.01,
                 kernel_cutoff=3.0, tau=0.2, refine_passes=1,
                 atomic_components=True, cos_thr=0.5,
                 glyph='arrow', glyph_stride=2, min_magnitude=0.06,
                 arrow_scale=0.7, arrow_width_falloff=0.7,
                 streamline_spacing=1.6, streamline_steps=60,
                 streamline_marker='circle', streamline_marker_size=0.1,
                 colors=None, opacity=0.6, opacity_falloff=0.0, stroke_w=1.1,
                 name_prefix='flow', max_edges=DEFAULT_MAX_EDGES,
                 support_budget=DEFAULT_SUPPORT_BUDGET, min_grid_res=16):
        self.pos          = {k: (float(v[0]), float(v[1])) for k, v in (pos or {}).items()}
        self.k_layers     = max(1, int(k_layers))
        self.glyph        = glyph
        self.name_prefix  = name_prefix
        self.names        = layerNames(self.k_layers, name_prefix)
        self.head_names   = headNames(self.k_layers, name_prefix)
        self.appearance   = layerAppearance(self.k_layers, name_prefix, colors=colors,
                                            glyph=glyph, opacity=opacity,
                                            opacity_falloff=opacity_falloff,
                                            stroke_w=stroke_w)
        self.grid         = None
        self.labels       = None
        self.glyph_reach  = 0.0
        self.dirs         = None
        self._kernels     = None
        self.U = self.V   = None
        self.edges        = []          # [(src, dst, weight)] after aggregation
        self._cells       = {}
        self.support_size = 0
        self.budget_note  = None        # set when a budget forced a degradation
        self._notes_      = []

        if glyph not in ('arrow', 'streamline'):
            raise ValueError(f"FlowFieldBackground: glyph must be 'arrow' or 'streamline', got {glyph!r}")
        if streamline_marker not in ('circle', 'diamond', 'none'):
            raise ValueError("FlowFieldBackground: streamline_marker must be 'circle', "
                             f"'diamond' or 'none', got {streamline_marker!r}")

        nodes, ends, weights = self.__gatherEdges__(df, relationships, count, selection)
        if len(ends) == 0:
            return

        if max_edges is not None and len(ends) > int(max_edges):
            self._notes_.append(f'max_edges: kept the heaviest {int(max_edges)} '
                                f'of {len(ends)} aggregated edges')
            keep    = np.argsort(-weights, kind='stable')[:int(max_edges)]
            keep.sort()
            ends    = ends[keep]
            weights = weights[keep]

        positions = np.array([self.pos[n] for n in nodes], dtype=float)
        A, B      = positions[ends[:, 0]], positions[ends[:, 1]]

        # Direction of travel per edge; zero-length edges deposit nothing.
        AB   = B - A
        seg  = np.sqrt((AB ** 2).sum(axis=1))
        live = seg > EPS
        if not live.any():
            return
        A, B, weights, ends = A[live], B[live], weights[live], ends[live]
        dirs = AB[live] / seg[live][:, None]
        self.dirs  = dirs
        self.edges = [(nodes[int(i)], nodes[int(j)], float(w))
                      for (i, j), w in zip(ends, weights)]

        self.weights = np.log1p(weights) if log_weights else weights.astype(float)

        # Size the grid (and, if it comes to it, the edge set) so the kernels
        # cannot outgrow the budget -- see _fit_to_budget.
        self.grid, self.sigma, _keep_, _note_ = _fit_to_budget(
            positions, A, B, self.weights, grid_res, pad, sigma_frac, sigma,
            kernel_cutoff, support_budget, min_grid_res)
        if _keep_ is not None:
            A, B, dirs   = A[_keep_], B[_keep_], dirs[_keep_]
            ends         = ends[_keep_]
            self.weights = self.weights[_keep_]
            self.dirs    = dirs
            self.edges   = [self.edges[int(i)] for i in _keep_]
        if _note_:
            self._notes_.append(_note_)
        self.budget_note = '; '.join(self._notes_) if self._notes_ else None

        # Kept on the instance: tensorField() needs the per-edge kernels, not
        # the already-summed layer fields.
        self._kernels = _sparse_kernels(self.grid, A, B, self.sigma, kernel_cutoff)
        starts, idx, val  = self._kernels
        self.support_size = int(len(idx))

        self.labels, self.U, self.V = _assign_layers(
            self.k_layers, self.weights, ends, dirs, starts, idx, val, self.grid.size,
            tau=tau, refine_passes=refine_passes,
            atomic_components=atomic_components, cos_thr=cos_thr)

        self.throughput = [float(self.weights[self.labels == k].sum())
                           for k in range(self.k_layers)]
        self._cells = self.__buildCells__(glyph_stride, min_magnitude, arrow_scale,
                                          arrow_width_falloff, streamline_spacing,
                                          streamline_steps, streamline_marker,
                                          streamline_marker_size)

    # -----------------------------------------------------------------------
    # Input
    # -----------------------------------------------------------------------

    def __gatherEdges__(self, df, relationships, count, selection):
        """-> (nodes, (E,2) endpoint indices, (E,) weights) for positioned nodes."""
        if df is None:
            return [], np.zeros((0, 2), dtype=int), np.zeros(0)
        if hasattr(df, 'edges') and hasattr(df, 'nodes'):
            pairs = self.__pairsFromGraph__(df)
        else:
            pairs = self.__pairsFromDataFrame__(df, relationships, count)

        _sel_ = set(selection) if selection else set()
        agg   = {}
        for src, dst, w in pairs:
            if src == dst or src not in self.pos or dst not in self.pos:
                continue
            if _sel_ and src not in _sel_ and dst not in _sel_:
                continue
            agg[(src, dst)] = agg.get((src, dst), 0.0) + float(w)
        if not agg:
            return [], np.zeros((0, 2), dtype=int), np.zeros(0)

        # Sorted node order keeps the whole pipeline reproducible run to run.
        nodes  = sorted({n for pair in agg for n in pair}, key=lambda n: (str(type(n)), str(n)))
        lookup = {n: i for i, n in enumerate(nodes)}
        keys   = sorted(agg, key=lambda p: (str(p[0]), str(p[1])))
        ends   = np.array([[lookup[s], lookup[d]] for s, d in keys], dtype=int)
        w      = np.array([agg[k] for k in keys], dtype=float)
        return nodes, ends, w

    def __pairsFromGraph__(self, g):
        """networkx graph -> [(src, dst, weight)].  An undirected graph has no
        direction to show, so each edge is emitted both ways -- which lands as
        counter-flow and is exactly what the layering separates."""
        out = []
        for u, v, data in g.edges(data=True):
            w = float(data.get('weight', 1.0))
            out.append((u, v, w))
            if not g.is_directed():
                out.append((v, u, w))
        return out

    def __pairsFromDataFrame__(self, df, relationships, count):
        """Polars frame of individual edge records -> [(src, dst, weight)].

        Mirrors createNetworkXGraph()'s aggregation: group by the endpoint
        columns of each relationship and reduce with linkp's count= semantics.
        """
        import polars as pl

        if not relationships:
            raise ValueError('FlowFieldBackground: relationships= is required for a DataFrame')

        out = []
        for i, rel in enumerate(relationships):
            fm, to = rel[0], rel[1]
            frame  = df
            if isinstance(fm, tuple):
                frame = self.__concatColumn__(frame, fm, f'__fm{i}__'); fm = f'__fm{i}__'
            if isinstance(to, tuple):
                frame = self.__concatColumn__(frame, to, f'__to{i}__'); to = f'__to{i}__'
            for _f_ in (fm, to):
                if _f_ not in frame.columns:
                    raise ValueError(f'FlowFieldBackground: field "{_f_}" not found in DataFrame')

            frame = frame.filter(pl.col(fm).is_not_null() & pl.col(to).is_not_null())
            # linkp's count= semantics.  Anything that is not a column name --
            # None, or p2s.ROW_COUNTp straight off a LinkP -- is the row count,
            # so a registry lambda can forward ``ln.count`` unexamined.
            if not isinstance(count, str):
                agg = pl.len().alias('__count__')
            elif count not in frame.columns:
                raise ValueError(f'FlowFieldBackground: count column "{count}" not found in DataFrame')
            elif frame[count].dtype.is_numeric():
                agg = pl.col(count).sum().alias('__count__')
            else:
                agg = pl.col(count).n_unique().alias('__count__')
            grouped = frame.group_by([fm, to]).agg(agg)
            out += [(row[0], row[1], float(row[2]))
                    for row in grouped.select(fm, to, '__count__').iter_rows()]
        return out

    def __concatColumn__(self, df, columns, new_column):
        import polars as pl
        parts = [pl.col(c) if df[c].dtype == pl.String else pl.col(c).cast(pl.String)
                 for c in columns]
        return df.with_columns(pl.concat_str(parts, separator='|').alias(new_column))

    # -----------------------------------------------------------------------
    # Output
    # -----------------------------------------------------------------------

    def __buildCells__(self, glyph_stride, min_magnitude, arrow_scale,
                       arrow_width_falloff, streamline_spacing, streamline_steps,
                       streamline_marker, streamline_marker_size):
        mags = np.sqrt(self.U * self.U + self.V * self.V)
        gmax = float(mags.max()) if mags.size else 0.0
        self.glyph_reach = 0.0
        if gmax <= EPS:
            return {}
        widths = _layer_width_scales(self.k_layers, arrow_width_falloff)
        cells  = {}
        for k, (name, head) in enumerate(zip(self.names, self.head_names)):
            if self.glyph == 'arrow':
                d, reach = _arrow_glyphs(self.grid, self.U[k], self.V[k], gmax,
                                         max(1, int(glyph_stride)), min_magnitude,
                                         arrow_scale, width_scale=widths[k])
                heads = ''
            else:
                d, heads, reach = _streamline_glyphs(
                    self.grid, self.U[k], self.V[k], gmax, min_magnitude,
                    spacing=streamline_spacing, max_steps=int(streamline_steps),
                    marker=streamline_marker, marker_size=streamline_marker_size)
            self.glyph_reach = max(self.glyph_reach, reach)
            # Insertion order is draw order: this layer's curves, then its
            # heads, then the next layer on top of both.  Each cell is a record
            # carrying its own appearance, so nothing about how it paints has to
            # be arranged by the caller (PLANNING.md B1).
            if d:
                cells[name] = BackgroundShape(d, **self.appearance[name])
            if heads:
                cells[head] = BackgroundShape(heads, **self.appearance[head])
        return cells

    def cells(self):
        """``{name: BackgroundShape}`` in world coordinates, in draw order.

        Each record holds an ``M/L/C/Z`` path in the same space as ``pos`` plus
        the appearance it wants, so it can be handed to ``background=`` with no
        accompanying ``background_*`` arguments at all::

            p2s.linkp(df, rels, pos, background=fm.cells())

        One entry per flow layer for ``glyph='arrow'``; for ``'streamline'`` a
        second entry per layer carries the filled head markers -- see
        :func:`cellNames`.  The paths themselves are ``record.shape``.
        """
        return dict(self._cells)

    def cellNames(self):
        """Every cell name this instance can emit, in draw order."""
        return cellNames(self.k_layers, self.name_prefix, self.glyph)

    def edgeLayers(self):
        """``{(src, dst): layer_index}`` -- which layer each aggregated edge
        landed in, for inspection or for colouring the links to match."""
        if self.labels is None:
            return {}
        return {(s, d): int(k) for (s, d, _w), k in zip(self.edges, self.labels)}

    def tensorField(self):
        """Structure tensor ``(a, b, c)`` per grid cell, ``T = [[a,b],[b,c]]``.

        The retired prototype's other answer to multi-valued flow, kept here
        because it is the one representation that never cancels: it is built
        from the per-edge deposits, where opposing flows ADD
        (``d (x) d == (-d) (x) (-d)``), so the leading eigenvector is the
        dominant flow AXIS, its eigenvalue is throughput, and the anisotropy
        ``(l1-l2)/(l1+l2)`` separates a clean corridor from churn.  Unused by
        the glyph path -- an axis has no arrowhead -- but it is the honest
        summary of "how much traffic passes through here, along what line", and
        it is non-zero exactly where the k=1 net field has cancelled to nothing.
        """
        if self.grid is None or self._kernels is None:
            return None
        starts, idx, val = self._kernels
        a = np.zeros(self.grid.size)
        b = np.zeros(self.grid.size)
        c = np.zeros(self.grid.size)
        for e in range(len(self.weights)):
            s, t = starts[e], starts[e + 1]
            if s == t:
                continue
            gi, w  = idx[s:t], self.weights[e] * val[s:t]
            dx, dy = self.dirs[e]
            a[gi] += w * dx * dx
            b[gi] += w * dx * dy
            c[gi] += w * dy * dy
        shape = (self.grid.ny, self.grid.nx)
        return a.reshape(shape), b.reshape(shape), c.reshape(shape)

    def __glyphCount__(self, name):
        """Subpaths drawn for one cell -- each glyph starts with an M."""
        _rec_ = self._cells.get(name)
        return 0 if _rec_ is None else _rec_.shape.count('M')

    def summary(self):
        """One-line-per-layer description of the decomposition."""
        if self.labels is None:
            return 'FlowFieldBackground: no flow (nothing to describe)'
        total = sum(self.throughput) or 1.0
        rows  = [f'FlowFieldBackground: {len(self.edges)} aggregated edges, '
                 f'{self.grid.nx}x{self.grid.ny} grid, sigma={self.sigma:.4g}, '
                 f'{self.support_size} kernel entries']
        if self.budget_note is not None:
            rows.append(f'  ! {self.budget_note}')
        for k, name in enumerate(self.names):
            n = int((self.labels == k).sum())
            rows.append(f'  {name}: {n} edges, {100.0 * self.throughput[k] / total:.1f}% of throughput, '
                        f'{self.__glyphCount__(name)} glyphs')
        return '\n'.join(rows)

    def __repr__(self):
        return (f'FlowFieldBackground(k_layers={self.k_layers}, edges={len(self.edges)}, '
                f'glyph={self.glyph!r})')
