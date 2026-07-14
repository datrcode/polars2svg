"""
t-FDP: Force-Directed Graph Layout via Student's t-Distribution
MLX implementation — runs on Apple silicon (Metal) or NVIDIA (CUDA)

Implementation of the following:

F. Zhong, M. Xue, J. Zhang, F. Zhang, R. Ban, O. Deussen, and Y. Wang,
"Force-Directed Graph Layouts Revisited: A New Force Based on the
t-Distribution," IEEE Transactions on Visualization and Computer Graphics,
2023, arXiv:2303.03964 (https://arxiv.org/abs/2303.03964).
"""

from __future__ import annotations

import logging
import time
from typing import Optional

try:
    import mlx.core as mx
except ImportError as _mlx_err:
    raise ImportError(
        "mlx is required for TFDPLayout. Install it with one of:\n"
        "  pip install polars2svg[mlx]       # Apple silicon (Metal), or CPU elsewhere\n"
        "  pip install polars2svg[mlx-cuda]  # Linux + NVIDIA (CUDA 12)"
    ) from _mlx_err

import numpy as np
import scipy.sparse as sp
import networkx as nx
from scipy.sparse.csgraph import shortest_path

from .mds_at_scale import _tileSideBySide_

logger = logging.getLogger('polars2svg_logger')


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------
#
# mx.gpu is Metal on Apple silicon and CUDA on a Linux mlx[cuda*] build — the kernels
# below are backend-agnostic mlx.core and do not care which. But mx.gpu is not always
# usable (the plain Linux mlx wheel has no GPU backend at all), so probe it once rather
# than assume it. Cached because the probe costs a device init plus a kernel compile.
#
# The probe must do real arithmetic, not just allocate. MLX's CUDA backend JIT-compiles
# its kernels with NVRTC against the *system* CUDA headers found via CUDA_HOME/CUDA_PATH
# (or /usr/local/cuda), so a toolkit/wheel mismatch — e.g. the cuda12 wheel's NVRTC 12.9
# fed CUDA 13's headers — fails at the first *compiled* kernel with a wall of nvcc syntax
# errors. Allocation ops like mx.zeros() are memsets and can sail right past that, which
# would leave the probe reporting a healthy GPU that then explodes mid-layout. Multiply
# and reduce so something actually gets compiled.

_DEVICE_CACHE: mx.Device | None = None


def _default_device() -> mx.Device:
    """Resolve mx.gpu (Metal or CUDA) once, falling back to mx.cpu if unusable."""
    global _DEVICE_CACHE
    if _DEVICE_CACHE is None:
        try:
            with mx.stream(mx.gpu):
                _probe = mx.array([1.0, 2.0])
                mx.eval(mx.sum(_probe * _probe))
            _DEVICE_CACHE = mx.gpu
        except Exception as err:  # noqa: BLE001 - any backend failure means "no GPU"
            logger.warning(
                'mlx GPU backend unavailable — TFDPLayout falling back to CPU (slower). '
                'For NVIDIA GPUs install polars2svg[mlx-cuda] (CUDA 12) or '
                'polars2svg[mlx-cuda13] (CUDA 13); the extra must match the CUDA toolkit '
                'headers on the host, and CUDA_HOME must point at it. Cause: %s', err)
            _DEVICE_CACHE = mx.cpu
    return _DEVICE_CACHE


def gpu_backend() -> str:
    """Return the backend TFDPLayout will use: 'metal', 'cuda', or 'cpu'."""
    if _default_device() == mx.cpu:
        return 'cpu'
    # Metal builds expose mx.metal.is_available(); CUDA builds do not.
    _metal = getattr(mx, 'metal', None)
    if _metal is not None and _metal.is_available():
        return 'metal'
    return 'cuda'


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _scale_by_edge(pos: np.ndarray, indptr: np.ndarray,
                   indices: np.ndarray) -> float:
    """Return a scale factor so the mean edge length ≈ 1."""
    src = np.repeat(np.arange(len(indptr) - 1), np.diff(indptr))
    if len(src) == 0:
        return 1.0
    diffs = pos[src] - pos[indices]
    edge_len = np.sqrt((diffs ** 2).sum(axis=1)).mean()
    return 1.0 / (edge_len + 1e-8)


def _pivot_mds(adj: sp.csr_matrix, n_pivots: int = 50, dim: int = 2,
               seed=None) -> np.ndarray:
    from sklearn.utils.extmath import randomized_svd
    rng = np.random.default_rng(seed)
    n = adj.shape[0]
    pivots = rng.choice(n, size=min(n_pivots, n), replace=False)
    D = shortest_path(adj, method='D', directed=False,
                      indices=pivots, unweighted=True)
    D2 = D ** 2
    C = -0.5 * (D2 - D2.mean(1, keepdims=True)
                   - D2.mean(0, keepdims=True)
                   + D2.mean())
    _, s, Vt = randomized_svd(C, n_components=dim, random_state=seed)
    return (Vt[:dim].T * np.sqrt(np.maximum(s[:dim], 0))).astype(np.float32)


def _t_kernel(diff: mx.array) -> tuple[mx.array, mx.array]:
    d2 = mx.sum(diff * diff, axis=-1)
    q  = 1.0 / (1.0 + d2)
    return q, q[..., None] * diff


def _repulsive_exact(pos: mx.array, gamma: float,
                     node_weights: mx.array | None = None) -> mx.array:
    diff = pos[:, None, :] - pos[None, :, :]
    _q, qd = _t_kernel(diff)
    eye = mx.eye(pos.shape[0])
    qd  = qd * (1.0 - eye)[..., None]
    if node_weights is not None:
        w = node_weights[:, None] * node_weights[None, :]
        return -gamma * mx.sum(qd * w[..., None], axis=1)
    return -gamma * mx.sum(qd, axis=1)


def _repulsive_rvs(pos: mx.array, gamma: float,
                   k: int, key: mx.array,
                   node_weights: mx.array | None = None) -> mx.array:
    n   = pos.shape[0]
    idx = mx.random.randint(0, n, shape=(n, k), key=key)
    diff = pos[:, None, :] - pos[idx]
    d2   = mx.sum(diff * diff, axis=-1)
    qd   = (1.0 / (1.0 + d2))[..., None] * diff
    if node_weights is not None:
        qd = qd * (node_weights[:, None] * node_weights[idx])[..., None]
    return -gamma * (float(n) / float(k)) * mx.sum(qd, axis=1)


def _attractive_exact(pos: mx.array,
                      edge_src: mx.array, edge_tgt: mx.array,
                      alpha: float, beta: float, combine: bool,
                      focal_edge_mask: mx.array | None = None,
                      focal_attraction_scale: float = 1.0) -> mx.array:
    diff = pos[edge_src] - pos[edge_tgt]
    d2   = mx.sum(diff * diff, axis=-1)
    d    = mx.sqrt(d2 + 1e-12)
    edge_force = -alpha * d[..., None] * diff
    if combine:
        edge_force = edge_force + (-beta * (1.0 / (1.0 + d2))[..., None] * diff)
    if focal_edge_mask is not None:
        scale = mx.where(focal_edge_mask, focal_attraction_scale, 1.0)
        edge_force = edge_force * scale[..., None]
    n     = pos.shape[0]
    F_att = mx.zeros((n, 2))
    F_att = F_att.at[edge_src].add(edge_force)
    F_att = F_att.at[edge_tgt].add(-edge_force)
    return F_att


def _cooling_schedule(step: int, max_iter: int, lr_init: float = 0.1) -> float:
    return lr_init * (1.0 - step / max_iter)


# ---------------------------------------------------------------------------
# Core simulation (sparse-matrix API)
# ---------------------------------------------------------------------------

def _tfdp_layout_core(
    graph: sp.csr_matrix,
    *,
    init: str | np.ndarray = "pmds",
    algo: str = "exact",
    alpha: float = 0.1,
    beta: float  = 8.0,
    gamma: float = 2.0,
    max_iter: int = 300,
    combine: bool = True,
    rvs_k: int = 64,
    lr: float = 0.1,
    seed: Optional[int] = None,
    verbose: bool = False,
    device: mx.Device | None = None,
    selection: set | None = None,
    focal_attraction_scale: float = 5.0,
    bg_repulsion_scale: float = 3.0,
    pin_background: bool = False,
) -> tuple[np.ndarray, float]:
    if device is None:
        device = _default_device()

    n = graph.shape[0]
    assert graph.shape == (n, n), "graph must be square"  # nosec B101 - internal shape invariant, not a security boundary; caller-facing validation happens upstream

    # Warm the device context (Metal / CUDA / CPU)
    with mx.stream(device):
        mx.eval(mx.zeros((1,)))

    rng = np.random.default_rng(seed)

    if isinstance(init, np.ndarray):
        pos_np = init.astype(np.float32)
    elif init == "pmds":
        pos_np = _pivot_mds(graph, n_pivots=min(100, n), seed=seed)
        scale  = _scale_by_edge(pos_np, graph.indptr, graph.indices)
        pos_np = pos_np * scale * 2.0 + (0.01 * rng.standard_normal((n, 2)).astype(np.float32))
    elif init == "random":
        pos_np = rng.standard_normal((n, 2)).astype(np.float32)
    else:
        raise ValueError(f"Unknown init: {init!r}")

    graph_csr = graph.tocsr()
    src_np = np.repeat(np.arange(n, dtype=np.int32), np.diff(graph_csr.indptr))
    tgt_np = graph_csr.indices.astype(np.int32)
    mask   = src_np < tgt_np
    src_np, tgt_np = src_np[mask], tgt_np[mask]

    with mx.stream(device):
        pos        = mx.array(pos_np)
        edge_src   = mx.array(src_np)
        edge_tgt   = mx.array(tgt_np)
        velocity   = mx.zeros((n, 2))
        mx_rng_key = mx.random.key(seed if seed is not None else 0)
    mx.eval(pos, edge_src, edge_tgt, velocity)

    node_weights = focal_edge_mask = pin_mask = init_pos = None
    if selection is not None:
        sel = set(selection)
        weights_np = np.where(
            np.array([i in sel for i in range(n)], dtype=bool),
            1.0, bg_repulsion_scale
        ).astype(np.float32)
        focal_np = (np.array([s in sel for s in src_np], dtype=bool) |
                    np.array([t in sel for t in tgt_np], dtype=bool))
        with mx.stream(device):
            node_weights    = mx.array(weights_np)
            focal_edge_mask = mx.array(focal_np)
        mx.eval(node_weights, focal_edge_mask)
        if pin_background:
            pin_np = np.array([i not in sel for i in range(n)], dtype=bool)
            with mx.stream(device):
                pin_mask = mx.array(pin_np)
                init_pos = mx.array(pos_np)
            mx.eval(pin_mask, init_pos)

    t0 = time.perf_counter()
    for step in range(max_iter):
        step_lr = _cooling_schedule(step, max_iter, lr)
        with mx.stream(device):
            if algo == "exact":
                F_rep = _repulsive_exact(pos, gamma, node_weights)
            elif algo == "rvs":
                mx_rng_key, subkey = mx.random.split(mx_rng_key)
                F_rep = _repulsive_rvs(pos, gamma, rvs_k, subkey, node_weights)
            else:
                raise ValueError(f"Unknown algo: {algo!r}")
            F_att    = _attractive_exact(pos, edge_src, edge_tgt,
                                         alpha, beta, combine,
                                         focal_edge_mask, focal_attraction_scale)
            F_total  = F_rep + F_att
            velocity = 0.9 * velocity + step_lr * F_total
            v_norm   = mx.sqrt(mx.sum(velocity * velocity, axis=-1, keepdims=True))
            velocity = mx.where(v_norm > 1.0, velocity / (v_norm + 1e-8), velocity)
            pos      = pos + velocity
            if pin_mask is not None:
                velocity = mx.where(pin_mask[:, None], 0.0, velocity)
                pos      = mx.where(pin_mask[:, None], init_pos, pos)
        if verbose and (step % 50 == 0 or step == max_iter - 1):
            mx.eval(pos)
            logger.info(f"  step {step:4d}/{max_iter}  lr={step_lr:.4f}  "
                        f"elapsed={time.perf_counter()-t0:.2f}s")

    mx.eval(pos)
    return np.array(pos), time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Public class — framework interface
# ---------------------------------------------------------------------------

class TFDPLayout(object):
    """
    t-FDP layout algorithm (Zhong et al., IEEE TVCG 2023).

    Replaces classical electric repulsion with a bounded Student's
    t-distribution force kernel, giving smooth and collision-free layouts
    that run entirely on the MLX GPU — Metal on Apple silicon, CUDA on NVIDIA
    (see gpu_backend()). Falls back to the CPU device if neither is available.

    Parameters
    ----------
    g : NetworkX graph
    pos : dict {node: (x, y)}, optional
        Warm-start positions. If None, Pivot-MDS initialisation is used.
        Mirrors the `pos=` convention of PolarsForceDirectedLayout.
    selection : set of node IDs, optional
        Focal nodes for local refinement. Attractive forces on edges
        touching these nodes are boosted; repulsion between non-focal
        pairs is scaled down, creating a fisheye-like expansion.
    pin_background : bool
        When True (and selection is set), non-focal nodes are held fixed
        at their pos= positions each step — only the focal neighbourhood
        is refined.
    algo : "exact" | "rvs"
        "exact"  — O(n²) per step, best quality (n ≤ ~2 000)
        "rvs"    — O(n·k) random-vertex sampling (n ≤ ~50 000)
    alpha, beta, gamma : float
        Spring weight, t-force attractive weight, repulsive scale.
    max_iter : int
    combine : bool
        Mix spring + t-force attraction (recommended).
    rvs_k : int
        Samples per node in RVS mode.
    lr : float
        Initial learning rate (linearly decays to 0).
    focal_attraction_scale : float
        Force multiplier on edges touching focal nodes.
    bg_repulsion_scale : float
        Repulsion multiplier between non-focal node pairs.
    seed, verbose, device : optional
    """

    def __init__(self, g, *, pos=None, selection=None, pin_background=False,
                 algo='exact', alpha=0.1, beta=8.0, gamma=2.0,
                 max_iter=300, combine=True, rvs_k=64, lr=0.1,
                 focal_attraction_scale=5.0, bg_repulsion_scale=3.0,
                 seed=None, verbose=False, device=None) -> None:

        # Determine connectivity (handle directed and undirected graphs)
        if isinstance(g, nx.DiGraph):
            _is_connected_ = nx.is_weakly_connected(g)
        elif isinstance(g, nx.Graph):
            _is_connected_ = nx.is_connected(g)
        else:
            _is_connected_ = True  # sparse matrix — assume connected

        # Handle disconnected graphs: lay out each component, then tile
        if not _is_connected_:
            g_und = g.to_undirected() if isinstance(g, nx.DiGraph) else g
            components = list(nx.connected_components(g_und))
            merged_pos = {}
            total_elapsed = 0.0
            for comp_nodes in components:
                sub = g.subgraph(comp_nodes)
                sub_pos = ({n: pos[n] for n in comp_nodes if n in pos}
                           if pos is not None else None) or None
                sub_sel = ({n for n in selection if n in comp_nodes}
                           if selection is not None else None)
                sub_layout = TFDPLayout(
                    sub, pos=sub_pos, selection=sub_sel,
                    pin_background=pin_background, algo=algo,
                    alpha=alpha, beta=beta, gamma=gamma, max_iter=max_iter,
                    combine=combine, rvs_k=rvs_k, lr=lr,
                    focal_attraction_scale=focal_attraction_scale,
                    bg_repulsion_scale=bg_repulsion_scale,
                    seed=seed, verbose=verbose, device=device)
                merged_pos |= sub_layout.results()
                total_elapsed += sub_layout.elapsed
            self.resulting_positions = _tileSideBySide_(g_und, merged_pos)
            self.elapsed = total_elapsed
            return

        # Build node ordering and scipy sparse adjacency matrix
        nodes    = list(g.nodes())
        node_idx = {v: i for i, v in enumerate(nodes)}
        n        = len(nodes)

        rows, cols = [], []
        for u, v in g.edges():
            i, j = node_idx[u], node_idx[v]
            rows += [i, j]; cols += [j, i]
        data = np.ones(len(rows), dtype=np.float32) if rows else np.empty(0, dtype=np.float32)
        adj  = sp.csr_matrix((data, (rows, cols)) if rows else
                             (data, (np.empty(0, int), np.empty(0, int))), shape=(n, n))

        # Convert pos dict → numpy init array
        init: str | np.ndarray = "pmds"
        if pos is not None:
            init = np.array(
                [pos[v] if v in pos else (0.0, 0.0) for v in nodes],
                dtype=np.float32)

        # Convert selection node IDs → integer indices
        sel_indices = (
            {node_idx[v] for v in selection if v in node_idx}
            if selection is not None else None
        )

        pos_arr, elapsed = _tfdp_layout_core(
            adj, init=init, algo=algo, alpha=alpha, beta=beta,
            gamma=gamma, max_iter=max_iter, combine=combine,
            rvs_k=rvs_k, lr=lr, seed=seed, verbose=verbose,
            device=device, selection=sel_indices,
            focal_attraction_scale=focal_attraction_scale,
            bg_repulsion_scale=bg_repulsion_scale,
            pin_background=pin_background,
        )

        self.resulting_positions = {
            nodes[i]: (float(pos_arr[i, 0]), float(pos_arr[i, 1]))
            for i in range(n)
        }
        self.elapsed = elapsed

    def results(self) -> dict:
        """Return {node: (x, y)} position dict."""
        return self.resulting_positions
