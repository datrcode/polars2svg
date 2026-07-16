# The bundled link shapes implement hierarchical edge bundling from:
#
# D. Holten, "Hierarchical Edge Bundles: Visualization of Adjacency Relations in
# Hierarchical Data," IEEE Transactions on Visualization and Computer Graphics,
# vol. 12, no. 5, pp. 741-748, Sept.-Oct. 2006, doi: 10.1109/TVCG.2006.147.
#
# Inline "Holten (2006)" comments below (bundle_strength β, Formula 1 blending in
# _svg_cubic_bspline_) refer to this paper.

import polars as pl
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path, laplacian as graph_laplacian, minimum_spanning_tree, depth_first_order, connected_components as _scipy_connected_components_
from scipy.sparse.linalg import eigsh
from math import pi, cos, sin, sqrt, atan2
import html

import random
import time

import polars2svg
from polars2svg.p2s_displaylist import (DisplayList, hexToRGBA,
                                        cubicBezierSegmentsTable, flattenPathD)
from polars2svg.export import ExportMixin


def _mst_order(nodes: list, n: int, A_weight) -> list:
    """
    MST depth-first walk ordering. O(E log E), no convergence issues.
    Builds the minimum spanning tree on inverted weights (so max-weight edges
    are preferred), then traverses with DFS to group connected nodes together.
    """
    A_dist = A_weight.copy()
    A_dist.data = 1.0 / (A_dist.data + 1e-9)
    mst = minimum_spanning_tree(A_dist)
    mst_sym = mst + mst.T
    order_idx = list(depth_first_order(mst_sym, i_start=0, return_predecessors=False))
    visited = set(order_idx)
    # Append any nodes unreachable from node 0 (disconnected components)
    order_idx += [i for i in range(n) if i not in visited]
    return [nodes[i] for i in order_idx]


def _spectral_order(nodes: list, n: int, A_weight) -> list:
    """
    Fiedler-vector ordering for large graphs. Uses shift-invert mode (sigma=1e-6)
    so ARPACK finds eigenvalues near zero of the Laplacian efficiently — converges
    in O(1) iterations vs thousands with plain which='SM'. Falls back to MST-DFS
    on degenerate graphs where eigsh still fails.
    """
    L = graph_laplacian(A_weight, normed=False)
    try:
        # sigma shifts the spectrum so the two smallest Laplacian eigenvalues
        # become the two largest of (L - sigma*I)^{-1}, which ARPACK finds fast.
        _, vecs = eigsh(L, k=2, sigma=1e-6, which='LM', tol=1e-5)
        fiedler = vecs[:, 1]
        return [nodes[i] for i in np.argsort(fiedler)]
    except Exception:
        return _mst_order(nodes, n, A_weight)


def leafWalkFromEdges(
    df: pl.DataFrame,
    source_col: str = "__fm__",
    target_col: str = "__to__",
    weight_col: str = "__count__",
    method: str = "average",
    spectral_threshold: int = 1000,
) -> list:
    """
    Given a Polars DataFrame of edges (source, target, weight),
    derive a node ordering that clusters strongly-connected nodes together.

    For N ≤ spectral_threshold nodes: builds a sparse distance matrix via
    scipy Dijkstra, then runs hierarchical clustering (exact, same quality
    as before but ~10–50× faster than NetworkX for large N).

    For N > spectral_threshold: uses the Fiedler vector of the graph
    Laplacian (spectral ordering), which is O(N log N) vs O(N² log N).

    Parameters
    ----------
    df : pl.DataFrame
        Edge list with source, target, and weight columns.
    source_col, target_col, weight_col : str
        Column names in df.
    method : str
        Linkage method passed to scipy: 'average', 'single', 'complete', 'ward', etc.
    spectral_threshold : int
        Switch to Fiedler-vector ordering when unique node count exceeds this.

    Returns
    -------
    list
        Node labels in a clustering-friendly order.
    """
    _eps = 1e-9
    edges = df.select([source_col, target_col, weight_col]).rows()

    # Collect unique nodes and build integer index
    node_set = set()
    for u, v, _ in edges:
        node_set.add(u); node_set.add(v)
    nodes = sorted(node_set)
    n = len(nodes)
    if n <= 1:  # linkage needs at least a 2x2 distance matrix
        return nodes
    idx = {node: i for i, node in enumerate(nodes)}

    # Deduplicate undirected pairs: for each (u,v) keep the maximum original
    # weight (= minimum inverted distance = strongest connection wins).
    edge_dict: dict = {}
    for u, v, w in edges:
        i, j = idx[u], idx[v]
        key = (min(i, j), max(i, j))
        inv_w = 1.0 / (w + _eps)
        if key not in edge_dict or inv_w < edge_dict[key]:
            edge_dict[key] = inv_w

    ri, ci, data_d, data_w = [], [], [], []
    for (i, j), d in edge_dict.items():
        ri  += [i, j];  ci     += [j, i]
        data_d += [d, d];  data_w += [1.0 / d, 1.0 / d]

    A_dist   = csr_matrix((data_d, (ri, ci)), shape=(n, n))
    A_weight = csr_matrix((data_w, (ri, ci)), shape=(n, n))

    # Large graphs: approximate ordering via Fiedler vector (O(N log N))
    if n > spectral_threshold:
        return _spectral_order(nodes, n, A_weight)

    # Small/medium graphs: exact hierarchical clustering via scipy sparse APSP
    dist_matrix = shortest_path(A_dist, method='D', directed=False)

    # Handle disconnected nodes (inf → large finite value)
    finite_vals = dist_matrix[np.isfinite(dist_matrix)]
    finite_max = finite_vals.max() if finite_vals.size else 1.0
    dist_matrix = np.where(np.isinf(dist_matrix), finite_max * 2, dist_matrix)

    # Symmetrize: per-row Dijkstra runs can accumulate floating-point drift
    dist_matrix = np.minimum(dist_matrix, dist_matrix.T)

    condensed = squareform(dist_matrix)
    Z = linkage(condensed, method=method)
    dend = dendrogram(Z, labels=nodes, no_plot=True)
    return dend["ivl"]

def _pos_components_(df: pl.DataFrame, source_col: str = "__fm__", target_col: str = "__to__") -> list:
    """Return a list of sets of node names, one per connected component, derived from an edge DataFrame."""
    edges = df.select([source_col, target_col]).rows()
    node_set = set()
    for u, v in edges:
        node_set.add(u); node_set.add(v)
    nodes = sorted(node_set)
    n = len(nodes)
    if n == 0:
        return []
    idx = {node: i for i, node in enumerate(nodes)}
    rows = [idx[u] for u, v in edges] + [idx[v] for u, v in edges]
    cols = [idx[v] for u, v in edges] + [idx[u] for u, v in edges]
    A = csr_matrix(([1] * len(rows), (rows, cols)), shape=(n, n))
    n_comps, labels = _scipy_connected_components_(A, directed=False)
    return [set(nodes[i] for i in range(n) if labels[i] == c) for c in range(n_comps)]


def _pos_to_order_angle_(pos: dict, components: list = None) -> list:
    """Sort nodes by polar angle from centroid, respecting connected components.

    For disconnected graphs each component is sorted by angle from its own
    local centroid, and components are sequenced by the angle of their
    centroid from the global centroid.  This keeps each component contiguous
    on the ring and prevents the -π/+π wrap from splitting a cluster.

    For a connected graph (components=None or a single component) this
    degenerates to the original single-centroid sort.
    """
    all_nodes = [n for n in pos if pos[n] is not None]
    if not all_nodes:
        return []

    if components is None or len(components) <= 1:
        groups = [all_nodes]
    else:
        gcx = sum(float(pos[n][0]) for n in all_nodes) / len(all_nodes)
        gcy = sum(float(pos[n][1]) for n in all_nodes) / len(all_nodes)

        def _comp_angle_(comp):
            ns = [n for n in comp if n in pos and pos[n] is not None]
            if not ns:
                return 0.0
            cx = sum(float(pos[n][0]) for n in ns) / len(ns)
            cy = sum(float(pos[n][1]) for n in ns) / len(ns)
            return atan2(cy - gcy, cx - gcx)

        groups = [list(c) for c in sorted(components, key=_comp_angle_)]

    result = []
    for group in groups:
        nodes = [n for n in group if n in pos and pos[n] is not None]
        if len(nodes) <= 1:
            result.extend(nodes)
            continue
        cx = sum(float(pos[n][0]) for n in nodes) / len(nodes)
        cy = sum(float(pos[n][1]) for n in nodes) / len(nodes)
        result.extend(sorted(nodes, key=lambda n: atan2(float(pos[n][1]) - cy, float(pos[n][0]) - cx)))

    return result


def _pos_to_order_pca_(pos: dict, components: list = None) -> list:
    """Sort nodes by PC1 projection, respecting connected components.

    For disconnected graphs, components are ordered by their mean projection
    onto the global PC1, then nodes within each component are sorted by their
    local PC1.  This preserves internal structure that would otherwise be
    compressed when PC1 is dominated by between-component separation.

    For a connected graph (components=None or a single component) this
    degenerates to the original global PC1 sort.
    """
    all_nodes = [n for n in pos if pos[n] is not None]
    if not all_nodes:
        return []

    X_all = np.array([[float(pos[n][0]), float(pos[n][1])] for n in all_nodes])
    global_mean = X_all.mean(axis=0)

    if components is None or len(components) <= 1:
        groups = [all_nodes]
    else:
        X_c = X_all - global_mean
        _, _, Vt = np.linalg.svd(X_c, full_matrices=False)
        global_pc1 = Vt[0]
        node_idx = {n: i for i, n in enumerate(all_nodes)}

        def _comp_pc1_mean_(comp):
            ns = [n for n in comp if n in node_idx]
            if not ns:
                return 0.0
            rows = X_c[[node_idx[n] for n in ns]]
            return float((rows @ global_pc1).mean())

        groups = [list(c) for c in sorted(components, key=_comp_pc1_mean_)]

    result = []
    for group in groups:
        nodes = [n for n in group if n in pos and pos[n] is not None]
        if len(nodes) <= 1:
            result.extend(nodes)
            continue
        X = np.array([[float(pos[n][0]), float(pos[n][1])] for n in nodes])
        X -= X.mean(axis=0)
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        projections = X @ Vt[0]
        result.extend([nodes[i] for i in np.argsort(projections)])

    return result


class ChP(ExportMixin):

    _VALID_KWARGS = frozenset({
        'template', 'df',
        'relationships', 'order', 'pos',
        'color', 'node_color', 'count',
        'node_size', 'node_opacity', 'node_size_range',
        'node_gap', 'draw_labels', 'label_style', 'txt_offset',
        'node_labels', 'label_only', 'node_selection',
        'link_size', 'link_shape', 'link_opacity', 'link_size_range',
        'bundle_strength', 'bundle_rings', 'skeleton_algorithm', 'render_skeleton',
        'wxh', 'insets', 'bounds_percent',
        'sm_shared', '_shared_view_x_', '_shared_view_y_',
        'count_range_shared', 'color_stat_range_shared',
        'draw_border', 'txt_h', 'legend',
    })

    #
    # __init__()
    #
    def __init__(self, *args, **kwargs):
        self.t_start        = time.time()
        self.p2s            = polars2svg.Polars2SVG()
        self.timing_metrics = {}
        self.gatherMetrics(self.__parseInput__, *args, **kwargs)
        self.gatherMetrics(self.__validateInput__)
        if self.df is not None:
            rand_id = random.randint(0, 2**32)  # nosec B311 - non-cryptographic SVG id scoping, see SECURITY.md
            self.gatherMetrics(self.__calculateOrder__)
            self.gatherMetrics(self.__calculateGeometry__)
            self.gatherMetrics(self.__renderLinks__)
            self.gatherMetrics(self.__renderNodes__)
            self.gatherMetrics(self.__renderSVG__, rand_id)
        # trim verbose float tails from the finished SVG (idempotent; no-op on the
        # dataless placeholder) -- see Polars2SVG.roundSvgFloats
        self.svg = self.p2s.roundSvgFloats(self.svg)
        self.t_end     = time.time()
        self.t_overall = self.t_end - self.t_start

    def _repr_svg_(self): return self.svg

    #
    # webgpu() - WebGPU payload of the same render, extracted from the retained
    # df_link / df_node tables.  Curve links flatten to segments via the shared
    # bezier table; bundled links flatten from their generated path strings; node
    # arcs become annular quad strips; labels render as glyphs (circular labels
    # are placed per-glyph along the arc).  Lazy + cached.
    #
    def webgpu(self):
        if getattr(self, '_gpu_payload_', None) is not None: return self._gpu_payload_
        _dl_ = self.gpuDisplayList()
        if _dl_ is None: return None
        self._gpu_payload_ = _dl_.webgpu_payload(self.p2s.glyphAtlas())
        return self._gpu_payload_

    #
    # gpuDisplayList() - the composed backend-neutral display list (also consumed
    # by smallp when this component renders as a cell)
    #
    def gpuDisplayList(self):
        if self.df is None or getattr(self, 'svg', None) is None: return None
        if getattr(self, '_gpu_dl_', None) is not None: return self._gpu_dl_
        import re
        w, h    = self.wxh
        _bg_co_ = self.p2s.colorTyped('background', 'default')
        _dl_    = DisplayList(w, h, bg=_bg_co_)
        _dl_.rect(0, 0, w, h, _bg_co_)
        _cx_, _cy_ = float(self.cx), float(self.cy)

        # ── Skeleton background ─────────────────────────────────────────────
        if self.render_skeleton:
            G = self._build_skeleton_()
            if G is not None and G.number_of_edges() > 0:
                _skel_co_ = self.p2s.colorTyped('axis', 'inner')
                for (x0, y0), (x1, y1) in G.edges():
                    _dl_.line(x0, y0, x1, y1, _skel_co_, width=0.5, opacity=0.35)

        # ── Links ───────────────────────────────────────────────────────────
        _node_size_lu_ = {'small': 1, 'nil': 0.2, 'medium': 3, 'large': 5}
        _lk_w_ = _node_size_lu_.get(self.link_size, 1.0)
        if isinstance(self.link_size, (int, float)): _lk_w_ = float(self.link_size)
        if self.df_link is not None and self.link_size is not None and len(self.df_link) > 0:
            if self.link_shape in ('line', 'curve'):
                _sub_ = self.df_link.drop_nulls(subset=['__fm_x__', '__to_x__'])
                if len(_sub_) > 0:
                    _sub_ = _sub_.with_columns(self.p2s.rgbFromHexPolarsOperations('__lc_hex__', '__r_f__', '__g_f__', '__b_f__'))
                    if self.link_size == 'vary':
                        if self.count_range_shared is not None:
                            _lc_min_, _lc_max_ = float(self.count_range_shared[0]), float(self.count_range_shared[1])
                        else:
                            _mn_, _mx_ = _sub_['__count__'].min(), _sub_['__count__'].max()
                            _lc_min_ = float(_mn_) if _mn_ is not None else 0.0
                            _lc_max_ = float(_mx_) if _mx_ is not None else 1.0
                        _sub_ = _sub_.with_columns(
                            (pl.lit(float(self.link_size_range[0])) +
                             pl.lit(float(self.link_size_range[1] - self.link_size_range[0])) *
                             (pl.col('__count__').cast(pl.Float64) - _lc_min_) /
                             (0.01 + _lc_max_ - _lc_min_)).alias('__w_f__'))
                        _w_arg_ = '__w_f__'
                    else:
                        _w_arg_ = _lk_w_
                    _seg_ = cubicBezierSegmentsTable(_sub_, '__fm_x__', '__fm_y__',
                                                     '__cp1_x__', '__cp1_y__', '__cp2_x__', '__cp2_y__',
                                                     '__to_x__', '__to_y__')
                    _dl_.lines_table(_seg_, '__bx__', '__by__', '__bx2__', '__by2__',
                                     ('__r_f__', '__g_f__', '__b_f__'), width=_w_arg_,
                                     opacity=self.link_opacity, svg_col=None)
                    # Arrowheads: one triangle per link, per-vertex color from the link hex
                    _arr_ = _sub_.select(['__to_x__', '__to_y__', '__arr_lx__', '__arr_ly__',
                                          '__arr_rx__', '__arr_ry__', '__r_f__', '__g_f__', '__b_f__']).to_numpy()
                    if len(_arr_) > 0:
                        _xy_   = _arr_[:, 0:6].reshape(-1, 2)
                        _rgb_  = np.repeat(_arr_[:, 6:9], 3, axis=0)
                        _rgba_ = np.hstack([_rgb_, np.full((len(_rgb_), 1), self.link_opacity)])
                        _dl_.tris(_xy_.flatten().tolist(), list(range(len(_xy_))), _rgba_)
            elif self.link_shape == 'bundled':
                # Parse the generated path+polygon strings (bundled routing is a per-row UDF)
                for _s_ in self._link_svg_list_:
                    _pm_ = re.search(r'<path d="([^"]*)"[^>]*stroke="([^"]*)" stroke-width="([^"]*)"', _s_)
                    if _pm_ is not None:
                        _d_, _co_, _sw_ = _pm_.group(1), _pm_.group(2), float(_pm_.group(3))
                        for _pts_, _closed_ in flattenPathD(_d_):
                            for _j_ in range(len(_pts_) - 1):
                                _dl_.line(_pts_[_j_][0], _pts_[_j_][1], _pts_[_j_+1][0], _pts_[_j_+1][1],
                                          _co_, width=_sw_, opacity=self.link_opacity)
                    _gm_ = re.search(r'<polygon points="([^"]*)" fill="([^"]*)"', _s_)
                    if _gm_ is not None:
                        _pts_ = [tuple(float(v) for v in _pt_.split(',')) for _pt_ in _gm_.group(1).split()]
                        _dl_.polygon(_pts_, _gm_.group(2), opacity=self.link_opacity)

        # ── Nodes: annular sector quad strips ───────────────────────────────
        if self.df_node is not None and len(self.df_node) > 0:
            for _row_ in self.df_node.select(['__a0r__', '__a1r__', '__r__', '__ri__', '__nc_hex__']).iter_rows(named=True):
                _a0_, _a1_ = float(_row_['__a0r__']), float(_row_['__a1r__'])
                _r_, _ri_  = float(_row_['__r__']), float(_row_['__ri__'])
                _n_ = max(2, min(64, int((_a1_ - _a0_) * 180.0 / pi)))
                _xy_, _idx_ = [], []
                for _k_ in range(_n_ + 1):
                    _th_ = _a0_ + (_a1_ - _a0_) * _k_ / _n_
                    _xy_.extend([_cx_ + _r_  * cos(_th_), _cy_ + _r_  * sin(_th_),
                                 _cx_ + _ri_ * cos(_th_), _cy_ + _ri_ * sin(_th_)])
                    if _k_ < _n_:
                        _b_ = 2 * _k_
                        _idx_.extend([_b_, _b_+1, _b_+2, _b_+1, _b_+3, _b_+2])
                _dl_.tris(_xy_, _idx_, hexToRGBA(_row_['__nc_hex__'], self.node_opacity))

        # ── Labels ──────────────────────────────────────────────────────────
        if self.draw_labels and self.df_node is not None and len(self.df_node) > 0:
            _label_co_   = self.p2s.colorTyped('label', 'defaultfg')
            _label_map_  = {str(k): str(v) for k, v in (self.node_labels or {}).items()}
            _label_only_ = set(str(x) for x in self.label_only) if self.label_only else set()
            _bshift_     = 0.35 * self.txt_h   # approximates dominant-baseline="central"
            for _row_ in self.df_node.select(['__nm__', '__a0__', '__a1__', '__r__']).iter_rows(named=True):
                _nm_ = str(_row_['__nm__'])
                if _label_only_ and _nm_ not in _label_only_: continue
                _label_ = _label_map_.get(_nm_, _nm_)
                _a0_, _a1_, _r_ = float(_row_['__a0__']), float(_row_['__a1__']), float(_row_['__r__'])
                if self.label_style == 'circular':
                    # per-glyph placement along the arc (textPath startOffset=50%, anchor=middle)
                    _ro_   = _r_ + self.txt_offset + 2
                    _arcL_ = _ro_ * (_a1_ - _a0_) * pi / 180.0
                    _txtW_ = self.p2s.textLength(_label_, self.txt_h)
                    _s0_   = _arcL_ / 2.0 - _txtW_ / 2.0
                    for _ci_, _ch_ in enumerate(_label_):
                        _adv0_ = self.p2s.textLength(_label_[:_ci_],   self.txt_h)
                        _adv1_ = self.p2s.textLength(_label_[:_ci_+1], self.txt_h)
                        _mid_  = _s0_ + (_adv0_ + _adv1_) / 2.0
                        _th_   = _a0_ * pi / 180.0 + _mid_ / _ro_
                        _gx_   = _cx_ + _ro_ * cos(_th_)
                        _gy_   = _cy_ + _ro_ * sin(_th_)
                        _dl_.text(self.p2s, _ch_, _gx_, _gy_, txt_h=self.txt_h, anchor='middle',
                                  color=_label_co_, rotation=_th_ * 180.0 / pi + 90.0, svg='')
                else:  # 'radial'
                    _a_mid_   = (_a0_ + _a1_) / 2.0
                    _a_mid_r_ = _a_mid_ * pi / 180.0
                    _offset_  = _r_ + self.txt_offset + 2
                    _xt_      = _cx_ + _offset_ * cos(_a_mid_r_)
                    _yt_      = _cy_ + _offset_ * sin(_a_mid_r_)
                    if 90.0 <= _a_mid_ < 270.0: _rot_, _anchor_ = _a_mid_ - 180.0, 'end'
                    else:                       _rot_, _anchor_ = _a_mid_, 'start'
                    _dl_.text(self.p2s, _label_, _xt_, _yt_, txt_h=self.txt_h, anchor=_anchor_,
                              color=_label_co_, rotation=_rot_, baseline_shift=_bshift_, svg='')

        # ── Legend (recorded during __renderSVG__) ──────────────────────────
        if getattr(self, '_dl_legend_', None) is not None: _dl_.extend(self._dl_legend_)
        # ── Border ──────────────────────────────────────────────────────────
        if self.draw_border:
            _border_co_ = self.p2s.colorTyped('axis', 'inner')
            _dl_.line(0, 0, w-1, 0, _border_co_, width=1.0)
            _dl_.line(0, h-1, w-1, h-1, _border_co_, width=1.0)
            _dl_.line(0, 0, 0, h-1, _border_co_, width=1.0)
            _dl_.line(w-1, 0, w-1, h-1, _border_co_, width=1.0)
        self._gpu_dl_ = _dl_
        return _dl_

    #
    # gatherMetrics()
    #
    def gatherMetrics(self, callable, *args, **kwargs):
        t0 = time.time()
        _results_ = callable(*args, **kwargs)
        t1 = time.time()
        if callable.__name__ not in self.timing_metrics: self.timing_metrics[callable.__name__] = 0.0
        self.timing_metrics[callable.__name__] += t1 - t0
        return _results_

    #
    # __parseInput__()
    #
    def __parseInput__(self, *args, **kwargs):
        _unknown_ = set(kwargs) - self._VALID_KWARGS
        if _unknown_:
            raise TypeError(f'ChP: unexpected keyword argument(s): {sorted(_unknown_)}')

        # Single source of truth for every parameter (name -> from-scratch default);
        # drives both the from-scratch assignment and the keyword-override copy below.
        # node_color: None | '#rrggbb' | p2s.COLOR_BY_NODE_NAME | p2s_constant | ('field', p2s_constant) | {node: '#rrggbb'}
        _defaults_ = {
            # Core
            'relationships':          None,
            'order':                  None,
            'pos':                    {},
            # Color (p2s style)
            'color':                  None,   # None | '#rrggbb' | 'field'
            'node_color':             None,
            # Count
            'count':                  self.p2s.ROW_COUNTp,
            # Node styling
            'node_size':              'medium',
            'node_opacity':           1.0,
            'node_size_range':        (0.3, 4),
            'node_gap':               2,
            'draw_labels':            False,
            'label_style':            'radial',
            'txt_offset':             0,
            'node_labels':            None,
            'label_only':             set(),
            'node_selection':         set(),
            # Link styling
            'link_size':              'small',
            'link_shape':             'curve',      # 'bundled'
            'link_opacity':           1.0,
            'link_size_range':        (0.25, 4),
            'bundle_strength':        0.85,         # β in Holten (2006): 0 = straight chord, 1 = full skeleton routing
            'bundle_rings':           4,            # controls hex mesh density: hex edge = r / bundle_rings
            'skeleton_algorithm':     'hexagonal',  # routing graph algorithm; 'hexagonal' is the only current option
            'render_skeleton':        False,        # render skeleton as background when True
            # Geometry (p2s style)
            'wxh':                    (256, 256),
            'insets':                 (3, 3),
            'bounds_percent':         0.05,
            # Small multiples
            'sm_shared':              set(),
            '_shared_view_x_':        None,
            '_shared_view_y_':        None,
            'count_range_shared':     None,
            'color_stat_range_shared': None,
            # Context
            'draw_border':            True,
            'txt_h':                  12,
            'legend':                 False,
        }
        self.p2s.assertParamSpecMatches('ChP', self._VALID_KWARGS, _defaults_)

        self.df, self.df_orig = None, None

        # Template support
        self.template = None
        for i in range(len(args)):
            if isinstance(args[i], ChP): self.template = args[i]
        if 'template' in kwargs: self.template = kwargs['template']
        if self.template is not None:
            _template_copy_ = self.template
            self.p2s._clone_template_state(self, _template_copy_)
            self.template = _template_copy_
            self._count_min_        = None
            self._count_max_        = None
            self._color_stat_min_   = None
            self._color_stat_max_   = None
            self._bundled_skeleton_ = None  # geometry differs from template; must rebuild
        else:
            self.p2s.assignScratchDefaults(self, _defaults_)
            # Internal (non-parameter) state — not part of the kwarg spec
            self._bundled_skeleton_ = None        # cached networkx.Graph built by _build_skeleton_()
            self._count_min_        = None
            self._count_max_        = None
            self._color_stat_min_   = None
            self._color_stat_max_   = None
            self.color_nodes_final  = {}
            self._render_invalid_   = False
            # from-scratch builds only — a template clone is an exact snapshot and
            # must not re-apply session defaults (see Polars2SVG._apply_defaults)
            kwargs = self.p2s._apply_defaults('chordp', kwargs)

        # Extract DataFrame
        _new_df_ = None
        for _arg_ in args:
            if isinstance(_arg_, pl.DataFrame):
                if _new_df_ is None: _new_df_ = _arg_
                else:                raise ValueError('ChP.__parseInput__(): df already set')
        if 'df' in kwargs:
            if _new_df_ is None: _new_df_ = kwargs['df']
            else:                raise ValueError('ChP.__parseInput__(): df already set')
        if _new_df_ is not None:
            self.df = self.df_orig = _new_df_

        def _is_pos_dict_(arg):
            return (isinstance(arg, dict) and len(arg) > 0 and
                    all(isinstance(v, (tuple, list, np.ndarray)) and len(v) >= 2 for v in arg.values()))

        _rel_from_pos_ = None
        _pos_from_pos_ = None
        for _arg_ in args:
            if   isinstance(_arg_, pl.DataFrame): pass
            elif isinstance(_arg_, ChP):          pass
            elif isinstance(_arg_, list) and len(_arg_) > 0 and all(isinstance(t, tuple) and len(t) >= 2 for t in _arg_):
                if _rel_from_pos_ is not None: raise ValueError('ChP.__parseInput__(): relationships specified twice positionally')
                _rel_from_pos_ = _arg_
            elif _is_pos_dict_(_arg_):
                if _pos_from_pos_ is not None: raise ValueError('ChP.__parseInput__(): pos specified twice positionally')
                _pos_from_pos_ = _arg_
            else:
                raise ValueError(f'ChP.__parseInput__(): Unrecognized positional argument type {type(_arg_).__name__}')

        if _rel_from_pos_ is not None:
            if 'relationships' in kwargs: raise ValueError('ChP.__parseInput__(): relationships specified both positionally and as a keyword')
            self.relationships = _rel_from_pos_
        if _pos_from_pos_ is not None:
            if 'pos' in kwargs: raise ValueError('ChP.__parseInput__(): pos specified both positionally and as a keyword')
            self.pos = _pos_from_pos_

        # Apply kwargs overrides. bundle_strength/bundle_rings/render_skeleton coerce
        # their type and _shared_view_y_ carries a side effect, so those are handled
        # explicitly below and skipped by the spec-driven copy.
        self.p2s.assignKwargOverrides(self, _defaults_, kwargs,
                                      skip={'bundle_strength', 'bundle_rings', 'render_skeleton', '_shared_view_y_'})
        if 'bundle_strength'        in kwargs: self.bundle_strength        = float(kwargs['bundle_strength'])
        if 'bundle_rings'           in kwargs: self.bundle_rings           = int(kwargs['bundle_rings'])
        if 'render_skeleton'        in kwargs: self.render_skeleton        = bool(kwargs['render_skeleton'])
        if '_shared_view_y_'        in kwargs:
            self._shared_view_y_     = kwargs['_shared_view_y_']
            self._bundled_skeleton_  = self._shared_view_y_  # pre-seed cache; survives template reset

        # Normalize label_only to a set
        if isinstance(self.label_only, list): self.label_only = set(self.label_only)
        if isinstance(self.label_only, str):  self.label_only = {self.label_only}

        # "No data" placeholder for early error visibility -- only ever seen when
        # no df is supplied (a successful render overwrites self.svg); makes a
        # dropped-df plumbing mistake visible instead of a silently blank canvas.
        self.wxh = self.p2s.normalizeWxh(self.wxh, 'ChordP')
        w, h = self.wxh
        self.svg = self.p2s.placeholderSVG(w, h)

        if self.df is None: return

        # Copy the DataFrame and add the row index if not already present
        self.df = self.df.clone()
        if '__p2s_index__' not in self.df.columns:
            self.df = self.df.with_row_index('__p2s_index__')

        # Expand tuple-based node fields into concatenated columns
        self.relationships_orig = self.relationships
        self.relationships, i = [], 0
        for _edge_ in self.relationships_orig:
            _fm_, _to_ = _edge_[0], _edge_[1]
            new_fm, new_to = _fm_, _to_
            if isinstance(_fm_, tuple):
                new_fm = f'__fm{i}__'
                self.df = self._createConcatColumn_(self.df, _fm_, new_fm)
            if isinstance(_to_, tuple):
                new_to = f'__to{i}__'
                self.df = self._createConcatColumn_(self.df, _to_, new_to)
            if   len(_edge_) == 2: self.relationships.append((new_fm, new_to))
            elif len(_edge_) == 3: self.relationships.append((new_fm, new_to, _edge_[2]))
            else: raise ValueError(f'LinkP: relationship tuples must have 2 or 3 parts, got {_edge_!r}')
            i += 1

        # Classify color modes and pre-build categorical color columns (must happen before group_by)
        self._link_color_mode_ = self.__colorModeInfo__(self.__effectiveColorSpec__('links'))
        self._node_color_mode_ = self.__colorModeInfo__(self.__effectiveColorSpec__('nodes'))
        if self._link_color_mode_['kind'] == 'categorical' and self._link_color_mode_['field']:
            self.df = self.df.with_columns(
                self.p2s.colorizeColumnPolarsOperations(self._link_color_mode_['field']).alias('__lc_cat__')
            )
        elif self._link_color_mode_['kind'] == 'cset' and self._link_color_mode_['field']:
            self.df = self.df.with_columns(
                pl.col(self._link_color_mode_['field']).cast(pl.String).alias('__lc_cat__')
            )
        # Node categorical color is derived in __renderNodes__ after concatenating all edge endpoints,
        # so each node's color reflects field values seen across all its edges (both fm and to sides).

    #
    # _createConcatColumn_() - concatenate multiple fields into one string column
    #
    def _createConcatColumn_(self, df, fields, new_col):
        _parts_ = []
        for i, f in enumerate(fields):
            if i > 0: _parts_.append(pl.lit('|'))
            _parts_.append(pl.col(f).cast(pl.String))
        return df.with_columns(pl.concat_str(_parts_).alias(new_col))

    #
    # __countAggExpr__() - return the Polars aggregation expression for counting edges
    # - mirrors the identical method in Timep and Histop
    #
    def __countAggExpr__(self):
        if self.count == self.p2s.ROW_COUNTp:
            return pl.len().alias('__count__')
        elif isinstance(self.count, str):
            _is_num_ = self.p2s.numericColumn(self.df, self.count)
            self.p2s.logDtypeKeyedCount('Chordp', self.count, _is_num_)
            if _is_num_: return pl.col(self.count).sum()    .alias('__count__')
            else:        return pl.col(self.count).n_unique().alias('__count__')
        elif isinstance(self.count, tuple):
            _fields_ = [_f_ for _f_ in self.count if isinstance(_f_, str)]
            if self.p2s.SETp in self.count:      return pl.col(_fields_[0]).n_unique().alias('__count__')
            elif len(_fields_) == 1:             return pl.col(_fields_[0]).sum()    .alias('__count__')
            else:                                return pl.struct(_fields_).n_unique().alias('__count__')
        return pl.len().alias('__count__')

    def __countFields__(self):
        if self.count == self.p2s.ROW_COUNTp: return set()
        if isinstance(self.count, str):        return {self.count}
        if isinstance(self.count, tuple):      return {_f_ for _f_ in self.count if isinstance(_f_, str)}
        return set()

    #
    # __effectiveColorSpec__() - resolve the color spec for links or nodes
    #
    def __effectiveColorSpec__(self, target):
        if target == 'links': return self.color
        return self.node_color

    #
    # __colorModeInfo__() - classify a color spec into a mode dict
    # Returns: {'kind': str, 'field': str|None, 'stat': str, 'hex': str|None}
    # Kinds: 'default' | 'fixed_hex' | 'categorical' | 'crow_magnitude' | 'crow_stretched' |
    #        'cset_magnitude' | 'cset_stretched' | 'stat_magnitude' | 'stat_stretched'
    #
    def __colorModeInfo__(self, spec):
        _p2s_ = self.p2s
        _cmag_  = {_p2s_.CMAGNITUDE_SUMp, _p2s_.CMAGNITUDE_MINp, _p2s_.CMAGNITUDE_MEDIANp,
                   _p2s_.CMAGNITUDE_MEANp, _p2s_.CMAGNITUDE_MAXp}
        _cstr_  = {_p2s_.CSTRETCHED_SUMp, _p2s_.CSTRETCHED_MINp, _p2s_.CSTRETCHED_MEDIANp,
                   _p2s_.CSTRETCHED_MEANp, _p2s_.CSTRETCHED_MAXp}
        _smap_  = {
            _p2s_.CMAGNITUDE_SUMp: 'sum',    _p2s_.CSTRETCHED_SUMp: 'sum',
            _p2s_.CMAGNITUDE_MINp: 'min',    _p2s_.CSTRETCHED_MINp: 'min',
            _p2s_.CMAGNITUDE_MEDIANp:'median',_p2s_.CSTRETCHED_MEDIANp:'median',
            _p2s_.CMAGNITUDE_MEANp: 'mean',  _p2s_.CSTRETCHED_MEANp: 'mean',
            _p2s_.CMAGNITUDE_MAXp: 'max',    _p2s_.CSTRETCHED_MAXp: 'max',
            _p2s_.SUMp: 'sum',   _p2s_.MINp: 'min',
            _p2s_.MEDIANp: 'median', _p2s_.MEANp: 'mean',
            _p2s_.MAXp: 'max',   _p2s_.STDp: 'std',
        }
        _info_ = {'kind': 'default', 'field': None, 'stat': 'sum', 'hex': None}
        if spec is None:
            pass
        elif isinstance(spec, self.p2s.HexColorString):
            _info_['kind'] = 'fixed_hex';  _info_['hex'] = spec
        elif spec == _p2s_.CROW_MAGNITUDEp:
            _info_['kind'] = 'crow_magnitude'
        elif spec == _p2s_.CROW_STRETCHEDp:
            _info_['kind'] = 'crow_stretched'
        elif spec == _p2s_.COLOR_BY_NODE_NAME:
            _info_['kind'] = 'categorical'   # field=None → colorize by node name
        elif isinstance(spec, str) and self.df is not None and spec in self.df.columns:
            _is_num_ = self.p2s.numericColumn(self.df, spec)
            self.p2s.logDtypeKeyedColor('Chordp', spec, _is_num_)
            if _is_num_:
                _info_['kind'] = 'stat_magnitude'; _info_['field'] = spec; _info_['stat'] = 'sum'
            else:
                _info_['kind'] = 'cset'; _info_['field'] = spec
        elif isinstance(spec, tuple):
            _strs_  = [f for f in spec if isinstance(f, str)]
            _enums_ = [e for e in spec if not isinstance(e, str)]
            _field_ = _strs_[0] if _strs_ else None
            _enum_  = _enums_[0] if _enums_ else None
            if   _enum_ == _p2s_.CSETp:
                _info_['kind'] = 'cset';           _info_['field'] = _field_
            elif _enum_ == _p2s_.CSET_MAGNITUDEp:
                _info_['kind'] = 'cset_magnitude'; _info_['field'] = _field_
            elif _enum_ == _p2s_.CSET_STRETCHEDp:
                _info_['kind'] = 'cset_stretched'; _info_['field'] = _field_
            elif _enum_ in _cmag_:
                _info_['kind'] = 'stat_magnitude'; _info_['field'] = _field_; _info_['stat'] = _smap_.get(_enum_, 'sum')
            elif _enum_ in _cstr_:
                _info_['kind'] = 'stat_stretched'; _info_['field'] = _field_; _info_['stat'] = _smap_.get(_enum_, 'sum')
            elif _enum_ in _smap_:
                _info_['kind'] = 'stat_magnitude'; _info_['field'] = _field_; _info_['stat'] = _smap_[_enum_]
            elif _field_:
                _info_['kind'] = 'categorical';    _info_['field'] = _field_
        return _info_

    #
    # __colorAggExprs__() - return agg expressions needed by a color mode (added into group_by().agg())
    #
    def __colorAggExprs__(self, mode_info, prefix):
        kind = mode_info['kind']
        if kind in ('categorical', 'cset'):
            return [
                pl.col(f'__{prefix}_cat__').n_unique().alias(f'__{prefix}_nuniq__'),
                pl.col(f'__{prefix}_cat__').first().alias(f'__{prefix}_first__'),
            ]
        elif kind in ('cset_magnitude', 'cset_stretched') and mode_info['field']:
            return [pl.col(mode_info['field']).n_unique().alias(f'__{prefix}_stat__')]
        elif kind in ('stat_magnitude', 'stat_stretched') and mode_info['field']:
            _field_ = mode_info['field']
            _op_ = {
                'sum':    pl.col(_field_).sum(),
                'min':    pl.col(_field_).min(),
                'median': pl.col(_field_).median(),
                'mean':   pl.col(_field_).mean(),
                'max':    pl.col(_field_).max(),
                'std':    pl.col(_field_).std(),
            }.get(mode_info['stat'], pl.col(_field_).sum())
            return [_op_.alias(f'__{prefix}_stat__')]
        elif kind in ('crow_magnitude', 'crow_stretched'):
            return [pl.len().alias(f'__{prefix}_row_count__')]
        return []

    #
    # __applyColorToDF__() - add f'__{prefix}_hex__' column to an aggregated DataFrame
    #
    def __applyColorToDF__(self, df, mode_info, prefix, default_hex):
        kind    = mode_info['kind']
        col_hex = f'__{prefix}_hex__'
        if kind == 'fixed_hex':
            return df.with_columns(pl.lit(mode_info['hex']).alias(col_hex))
        elif kind == 'categorical':
            return df.with_columns(
                pl.when(pl.col(f'__{prefix}_nuniq__') == 1)
                  .then(pl.col(f'__{prefix}_first__'))
                  .otherwise(pl.lit(default_hex))
                  .alias(col_hex)
            )
        elif kind == 'cset':
            df = df.with_columns(
                pl.when(pl.col(f'__{prefix}_nuniq__') == 1)
                  .then(pl.col(f'__{prefix}_first__'))
                  .otherwise(pl.lit(-1))
                  .alias(f'__{prefix}_set_elem__')
            )
            return df.with_columns(
                self.p2s.colorizeColumnPolarsOperations(f'__{prefix}_set_elem__').alias(col_hex)
            )
        elif kind in ('crow_magnitude', 'crow_stretched', 'cset_magnitude', 'cset_stretched',
                      'stat_magnitude', 'stat_stretched'):
            _sc_     = f'__{prefix}_row_count__' if kind in ('crow_magnitude', 'crow_stretched') else f'__{prefix}_stat__'
            _norm_   = f'__{prefix}_norm__'
            _r_, _g_, _b_ = f'__{prefix}_r__', f'__{prefix}_g__', f'__{prefix}_b__'
            # legend-only stat accumulator (kept separate from _color_stat_min_/_max_,
            # which smallp SM_COLOR sharing reads and which stretched modes never touch)
            _lg_min_ = df[_sc_].cast(pl.Float64).min()
            _lg_max_ = df[_sc_].cast(pl.Float64).max()
            if _lg_min_ is not None and (getattr(self, '_legend_stat_min_', None) is None or _lg_min_ < self._legend_stat_min_):
                self._legend_stat_min_ = float(_lg_min_)
            if _lg_max_ is not None and (getattr(self, '_legend_stat_max_', None) is None or _lg_max_ > self._legend_stat_max_):
                self._legend_stat_max_ = float(_lg_max_)
            if kind in ('crow_stretched', 'cset_stretched', 'stat_stretched'):
                _n_unique_ = df[_sc_].n_unique()
                df = df.with_columns(
                    ((pl.col(_sc_).rank('dense') - 1).cast(pl.Float64) / max(_n_unique_ - 1, 1)).alias(_norm_)
                )
            else:
                if self.color_stat_range_shared is not None:
                    _cs_min_ = float(self.color_stat_range_shared[0])
                    _cs_max_ = float(self.color_stat_range_shared[1])
                else:
                    _min_v_ = df[_sc_].cast(pl.Float64).min()
                    _max_v_ = df[_sc_].cast(pl.Float64).max()
                    _cs_min_ = float(_min_v_) if _min_v_ is not None else 0.0
                    _cs_max_ = float(_max_v_) if _max_v_ is not None else 1.0
                if self._color_stat_min_ is None or _cs_min_ < self._color_stat_min_:
                    self._color_stat_min_ = _cs_min_
                if self._color_stat_max_ is None or _cs_max_ > self._color_stat_max_:
                    self._color_stat_max_ = _cs_max_
                df = df.with_columns(
                    ((pl.col(_sc_).cast(pl.Float64) - _cs_min_) /
                     (0.001 + _cs_max_ - _cs_min_))
                    .clip(0.0, 1.0).alias(_norm_)
                )
            df = df.with_columns(
                self.p2s.colorSpectrumPolarsOperations(_norm_, _r_, _g_, _b_)
            ).with_columns(
                self.p2s.hexColorFromRGBTriplesPolarsOperations(_r_, _g_, _b_).alias(col_hex)
            )
            return df
        else:
            return df.with_columns(pl.lit(default_hex).alias(col_hex))

    #
    # __validateColorSpec__() - raise ValueError if a node_color value is not a recognized form
    #
    def __validateColorSpec__(self, spec, param_name, allow_dict=False):
        if spec is None: return
        if isinstance(spec, dict):
            if not allow_dict:
                raise ValueError(f'LinkP.__validateInput__(): {param_name} does not support dict values')
            return
        if isinstance(spec, tuple): return
        _p2s_ = self.p2s
        if spec in (_p2s_.CROW_MAGNITUDEp, _p2s_.CROW_STRETCHEDp, _p2s_.COLOR_BY_NODE_NAME): return
        if isinstance(spec, self.p2s.HexColorString): return
        if isinstance(spec, str):
            if self.df is not None and spec in self.df.columns: return
            raise ValueError(
                f'LinkP.__validateInput__(): {param_name}={spec!r} is not a hex color, '
                f'a recognized constant, or a DataFrame column name'
            )
        raise ValueError(
            f'LinkP.__validateInput__(): {param_name}={spec!r} has unsupported type {type(spec).__name__}'
        )

    #
    # __validateInput__()
    #
    def __validateInput__(self):
        # Normalize legend= eagerly so a bad spec fails fast (raises InvalidSpecError).
        self.legend_spec = self.p2s.legendResolveSpec(self.legend)
        if self.df is None: return
        self.p2s.checkReservedColumns(self.df, 'ChP')
        if self.relationships is None or len(self.relationships) == 0:
            raise ValueError('LinkP.__validateInput__(): relationships must be specified')
        for _rel_ in self.relationships:
            for _field_ in _rel_[:2]:
                if _field_ not in self.df.columns:
                    raise ValueError(f'LinkP.__validateInput__(): field "{_field_}" not found in DataFrame')
        for _field_ in self.__countFields__():
            if _field_ not in self.df.columns:
                raise ValueError(f'LinkP.__validateInput__(): count field "{_field_}" not found in DataFrame')
        if self.color == self.p2s.COLOR_BY_NODE_NAME:
            raise ValueError(
                'LinkP.__validateInput__(): color=p2s.COLOR_BY_NODE_NAME is not valid for the '
                'color parameter; use node_color=p2s.COLOR_BY_NODE_NAME instead'
            )
        self.__validateColorSpec__(self.node_color, 'node_color', allow_dict=True)
        if self.label_style not in ('radial', 'circular'):
            raise ValueError(f'ChP: label_style must be "radial" or "circular", got {self.label_style!r}')

        # count= is consumed by 'vary' sizing or -- when the node order is derived
        # from the data -- by the edge-weight clustering that picks the order
        # (leafWalkFromEdges).  With the order pinned (order= / pos= / a shared
        # small-multiples view) and fixed sizes, nothing reads it.
        if self.count != self.p2s.ROW_COUNTp and \
           self.node_size != 'vary' and self.link_size != 'vary' and \
           self._shared_view_x_ is None and \
           (self.order is not None or bool(self.pos)):
            self.p2s.logger.warning(
                "ChP: count= is set but has no visible effect at the current settings; "
                "with order= or pos= supplied, count is only consumed when "
                "node_size='vary' and/or link_size='vary' "
                "(CROW_* color modes use raw row count, not count=)"
            )

    #
    # __calculateOrder__()
    # - if the order wasn't specified by the caller, determine that here
    #
    def __calculateOrder__(self):
        # Collect all nodes from the data & compute their edge weights
        _dfs_ = []
        _node_series_ = []
        if self.count is None or self.count == self.p2s.ROW_COUNTp:
            for _rel_ in self.relationships:
                _node_series_ += [self.df[_rel_[0]].drop_nulls(), self.df[_rel_[1]].drop_nulls()]
                _norm_names_ = {_rel_[0]:'__fm__', _rel_[1]:'__to__', 'len':'__count__'}
                _df_ = self.df.group_by(_rel_).len().rename(_norm_names_)
                _dfs_.append(_df_)
        else:
            for _rel_ in self.relationships:
                _node_series_ += [self.df[_rel_[0]].drop_nulls(), self.df[_rel_[1]].drop_nulls()]
                _count_agg_ = self.__countAggExpr__()
                _df_ = (self.df.group_by(_rel_)
                               .agg(_count_agg_)
                               .rename({_rel_[0]: '__fm__', _rel_[1]: '__to__'}))
                _dfs_.append(_df_)
        # Collect unique nodes via Polars (avoids materialising large Python sets)
        self.nodes_all = set(pl.concat(_node_series_).unique().to_list())
        # Concat the separate dataframes together (there's usually only one)
        self.df_edge_weights = pl.concat(_dfs_)
        # Determine node ordering
        if self._shared_view_x_ is not None:
            # SM_X: use the order from the reference panel so all panels are consistent
            self.order = list(self._shared_view_x_)
            # Include every node in the shared order even if absent from this panel's data
            self.nodes_all = self.nodes_all | set(self.order)
        elif self.order is None:
            if self.pos:
                _components_ = _pos_components_(self.df_edge_weights)
                self.order = _pos_to_order_angle_(self.pos, _components_)
                # Append any data nodes absent from pos at the end
                _pos_set_ = set(self.order)
                self.order += [n for n in sorted(self.nodes_all) if n not in _pos_set_]
            else:
                self.order = leafWalkFromEdges(self.df_edge_weights)

    #
    # __calculateGeometry_NONPOLARS__()
    #
    def __calculateGeometry_NONPOLARS__(self):
        w,  h  = self.wxh
        xi, yi = self.insets
        # Compute the node sizes
        if self.node_size == 'vary':
            self.df_node = pl.concat([self.df_edge_weights.select(['__fm__', '__count__']).rename({'__fm__':'__nm__'}),
                                      self.df_edge_weights.select(['__to__', '__count__']).rename({'__to__':'__nm__'})]) \
                             .group_by('__nm__') \
                             .agg(pl.col('__count__').sum())
            # Inner join: nodes absent from self.order are dropped from the render
            # (a partial order= list therefore hides the unlisted nodes).
            order_df = pl.DataFrame({"__nm__": self.order, "__order__": range(len(self.order))})
            self.df_node = self.df_node.join(order_df, on="__nm__").sort("__order__").drop("__order__")
        else:
            self.df_node = pl.DataFrame({'__nm__':self.order}).with_columns(pl.lit(1.0).alias('__count__'))
        # Calculate their arc position
        _node_gap_          = self.node_gap
        _nodes_len_         = len(self.nodes_all)
        _node_h_            = self.node_size_range[1] # node_size_range doubles as (min arc width, node height)
        _node_w_min_        = self.node_size_range[0]
        _node_space_needed_ = self.df_node['__count__'].sum()
        _r_outer_           = min(w-2*xi, h-2*yi)
        _circum_            = 2.0 * pi * _r_outer_
        _node_selection_    = self.node_selection

        # Derive correct center and radius (_r_outer_ as pre-computed was diameter, not radius)
        self.cx      = w / 2.0
        self.cy      = h / 2.0
        self.r       = min(w - 2*xi, h - 2*yi) / 2.0
        self.r_inner = max(self.r - _node_h_, 0.5)
        _circum_     = 2.0 * pi * self.r

        # Two-tier radii: selected nodes on the outer ring, non-selected on the inner ring.
        # Arc angle geometry is always computed from the outer circumference.
        # The inner tier sits one node_h gap below the outer ring's inner edge.
        if _node_selection_:
            self.r_sel        = self.r
            self.r_inner_sel  = self.r_inner
            self.r_nonsel     = max(self.r_inner - _node_h_, 0.5)
            self.r_inner_nonsel = max(self.r_nonsel - _node_h_, 0.5)
        else:
            self.r_sel = self.r_nonsel = self.r
            self.r_inner_sel = self.r_inner_nonsel = self.r_inner

        # Fit node_gap and node_w_min within the circumference
        if _nodes_len_ > 0:
            if _nodes_len_ * (_node_w_min_ + _node_gap_) > _circum_:
                # Step 1: shrink gap to make room for minimum node widths
                _node_gap_ = max(0.0, (_circum_ - _node_w_min_ * _nodes_len_) / _nodes_len_)
            if _nodes_len_ * (_node_w_min_ + _node_gap_) > _circum_:
                # Step 2: gap is already 0 and still doesn't fit — shrink node_w_min
                _node_gap_   = 0.0
                _node_w_min_ = _circum_ / _nodes_len_

        # Convert pixel gap to degrees and compute available arc for nodes
        _gap_deg_     = (_node_gap_ / _circum_) * 360.0 if _circum_ > 0 else 0.0
        _avail_deg_   = max(360.0 - _gap_deg_ * _nodes_len_, 0.001)
        _total_count_ = float(_node_space_needed_) if _node_space_needed_ else 1.0

        # Assign arc angle ranges in df_node order (already sorted by self.order)
        self.node_to_arc = {}
        _angle_ = 0.0
        for _nm_, _count_ in self.df_node.select(['__nm__', '__count__']).iter_rows():
            _node_deg_ = (float(_count_) / _total_count_ * _avail_deg_
                          if self.node_size == 'vary'
                          else _avail_deg_ / _nodes_len_)
            self.node_to_arc[str(_nm_)] = (_angle_, _angle_ + _node_deg_)
            _angle_ += _node_deg_ + _gap_deg_

        # Sketch the SVG (for prototyping)
        _bg_  = self.p2s.colorTyped('background', 'default')
        _svg_ = [f'<svg x="0" y="0" width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">',
                 f'<rect x="0" y="0" width="{w}" height="{h}" fill="{_bg_}" />']

        for _nm_, (_a0_, _a1_) in self.node_to_arc.items():
            _a0r_, _a1r_ = _a0_ * pi / 180.0, _a1_ * pi / 180.0
            _cx_, _cy_ = self.cx, self.cy
            _selected_ = _nm_ in _node_selection_
            _r_  = self.r_sel        if _selected_ else self.r_nonsel
            _ri_ = self.r_inner_sel  if _selected_ else self.r_inner_nonsel
            _x0o_ = _cx_ + _r_  * cos(_a0r_);  _y0o_ = _cy_ + _r_  * sin(_a0r_)
            _x1o_ = _cx_ + _r_  * cos(_a1r_);  _y1o_ = _cy_ + _r_  * sin(_a1r_)
            _x0i_ = _cx_ + _ri_ * cos(_a0r_);  _y0i_ = _cy_ + _ri_ * sin(_a0r_)
            _x1i_ = _cx_ + _ri_ * cos(_a1r_);  _y1i_ = _cy_ + _ri_ * sin(_a1r_)
            _large_ = 1 if (_a1_ - _a0_) > 180.0 else 0
            _path_ = (f'M {_x0o_:.2f} {_y0o_:.2f} A {_r_:.2f} {_r_:.2f} 0 {_large_} 1 {_x1o_:.2f} {_y1o_:.2f} '
                      f'L {_x1i_:.2f} {_y1i_:.2f} A {_ri_:.2f} {_ri_:.2f} 0 {_large_} 0 {_x0i_:.2f} {_y0i_:.2f} Z')
            _co_ = (self.node_color if isinstance(self.node_color, self.p2s.HexColorString)
                    else self.p2s.color(str(_nm_)))
            _svg_.append(f'<path d="{_path_}" fill="{_co_}" stroke="{_co_}" stroke-width="0.8" />')
        
        _svg_.append('</svg>')
        self.svg = ''.join(_svg_)

    #
    # __calculateGeometry__()
    # - all arc assignment and SVG path generation use Polars operations (no Python row loops)
    # - df_node retains geometry columns (__a0__, __a1__, __a0r__, __a1r__, __amr__, __r__, __ri__)
    #   for downstream link geometry: join df_edge on __fm__/__to__ → __nm__ to get arc endpoints
    #
    # Color-mode kinds that carry data-driven color semantics a legend can describe
    # ('default' / 'fixed_hex' / node-dict overrides do not).
    _LEGENDABLE_KINDS_ = frozenset({'categorical', 'cset', 'crow_magnitude', 'crow_stretched',
                                    'cset_magnitude', 'cset_stretched', 'stat_magnitude', 'stat_stretched'})

    #
    # __legendPrepare__() - resolve legend kind/metadata (the capture hook) and the
    # strip to reserve.  Mirrors LinkP.__legendPrepare__: the legend describes the
    # link channel (color=) when it is legend-able, otherwise the node channel
    # (node_color=); colorbar domains are finalized in __renderSVG__ from the
    # _legend_stat_min_/_max_ accumulator.  Decision A: a truthy legend with
    # nothing to legend silently reserves nothing.
    #
    def __legendPrepare__(self):
        self.legend_info       = None
        self._legend_region_   = None
        self._legend_reserve_  = (0, 0, 0, 0)
        self._legend_stat_min_ = None
        self._legend_stat_max_ = None
        if self.legend_spec is None or self.df is None or len(self.df) == 0: return
        _mode_, _channel_spec_ = None, None
        if   self._link_color_mode_['kind'] in self._LEGENDABLE_KINDS_:
            _mode_, _channel_spec_ = self._link_color_mode_, self.color
        elif self._node_color_mode_['kind'] in self._LEGENDABLE_KINDS_:
            _mode_, _channel_spec_ = self._node_color_mode_, self.node_color
        if _mode_ is None: return
        _spec_  = self.legend_spec
        _kind_  = 'categorical' if _mode_['kind'] in ('categorical', 'cset') else 'colorbar'
        if   _mode_['kind'] in ('crow_magnitude', 'crow_stretched'):    _title_default_ = 'rows'
        elif _mode_['field'] is not None:                               _title_default_ = _mode_['field']
        elif _channel_spec_ == self.p2s.COLOR_BY_NODE_NAME:             _title_default_ = 'node'
        else:                                                           _title_default_ = ''
        _title_ = _spec_['title'] if _spec_['title'] is not None else _title_default_
        if _kind_ == 'categorical':
            _field_ = _mode_['field']
            if _field_ is None:
                # COLOR_BY_NODE_NAME: entries are the (string-cast) node names
                _names_ = pl.concat([self.df.select(pl.col(_r_[_j_]).cast(pl.String).alias('__legend_node__'))
                                     for _r_ in self.relationships for _j_ in (0, 1)])
                _vc_ = self.p2s.legendCategoricalValueCounts(_names_, '__legend_node__')
                self.legend_info = self.p2s.legendInfoCategorical(_spec_, _vc_, _title_)
            elif _mode_['kind'] == 'cset':
                _vc_ = self.p2s.legendCategoricalValueCounts(self.df, _field_)
                self.legend_info = self.p2s.legendInfoCategorical(_spec_, _vc_, _title_)
            else:
                _agg_ = (self.df.group_by(_field_).agg(pl.len().alias('__legend_n__'))
                                .with_columns(self.p2s.colorizeColumnPolarsOperations(_field_).alias('__legend_hex__')))
                _vc_     = [(str(_k_), _n_, _k_) for _k_, _n_ in zip(_agg_[_field_].to_list(),
                                                                     _agg_['__legend_n__'].to_list())]
                _hex_lu_ = {str(_k_): _h_ for _k_, _h_ in zip(_agg_[_field_].to_list(),
                                                              _agg_['__legend_hex__'].to_list())}
                self.legend_info = self.p2s.legendInfoCategorical(_spec_, _vc_, _title_, hex_lu=_hex_lu_)
        else:
            self.legend_info = self.p2s.legendInfoColorbar(_title_)
            self._legend_stretched_ = _mode_['kind'].endswith('stretched')
        _reserve_ = self.p2s.legendReserve(_spec_, self.legend_info, self.txt_h, self.wxh)
        _l_, _r_, _t_, _b_ = _reserve_
        if self.wxh[0] - (_l_ + _r_) < 48 or self.wxh[1] - (_t_ + _b_) < 48:
            self.p2s.logger.warning(f'ChP.__legendPrepare__(): not enough space for legend (wxh = {self.wxh}); legend dropped')
            self.legend_info = None
            return
        self._legend_reserve_ = _reserve_
        _pos_ = _spec_['pos']
        if   _pos_ == 'right':  self._legend_region_ = (self.wxh[0] - _r_, 0, _r_, self.wxh[1])
        elif _pos_ == 'left':   self._legend_region_ = (0, 0, _l_, self.wxh[1])
        elif _pos_ == 'top':    self._legend_region_ = (0, 0, self.wxh[0], _t_)
        else:                   self._legend_region_ = (0, self.wxh[1] - _b_, self.wxh[0], _b_)

    def __calculateGeometry__(self):
        # Legend strip (if any) comes out of wxh first -- the plot region shrinks,
        # the physical output size does not ("reserve from wxh").
        self.__legendPrepare__()
        _leg_l_, _leg_r_, _leg_t_, _leg_b_ = self._legend_reserve_
        w,  h  = self.wxh[0] - _leg_l_ - _leg_r_, self.wxh[1] - _leg_t_ - _leg_b_
        xi, yi = self.insets

        # ── 1. Build df_node with node name + count ───────────────────────────
        if self.node_size == 'vary':
            self.df_node = (
                pl.concat([
                    self.df_edge_weights.select(pl.col('__fm__').alias('__nm__'), pl.col('__count__')),
                    self.df_edge_weights.select(pl.col('__to__').alias('__nm__'), pl.col('__count__')),
                ])
                .group_by('__nm__').agg(pl.col('__count__').sum())
            )
            order_df = pl.DataFrame({'__nm__': self.order, '__order__': range(len(self.order))})
            # Use right join so nodes in the shared order but absent from this panel still appear
            self.df_node = (self.df_node.join(order_df, on='__nm__', how='right')
                                        .with_columns(pl.col('__count__').fill_null(0.0))
                                        .sort('__order__').drop('__order__'))
        else:
            self.df_node = (pl.DataFrame({'__nm__': self.order})
                              .with_columns(pl.lit(1.0).alias('__count__')))

        # ── 2. Scalar geometry constants ──────────────────────────────────────
        _node_gap_          = self.node_gap
        _nodes_len_         = len(self.nodes_all)
        _node_h_            = self.node_size_range[1]
        _node_w_min_        = self.node_size_range[0]
        _node_space_needed_ = self.df_node['__count__'].sum()
        _node_selection_    = self.node_selection

        self.cx      = _leg_l_ + w / 2.0
        self.cy      = _leg_t_ + h / 2.0
        self.r       = min(w - 2*xi, h - 2*yi) / 2.0
        self.r_inner = max(self.r - _node_h_, 0.5)
        _circum_     = 2.0 * pi * self.r

        if _node_selection_:
            self.r_sel          = self.r
            self.r_inner_sel    = self.r_inner
            self.r_nonsel       = max(self.r_inner - _node_h_, 0.5)
            self.r_inner_nonsel = max(self.r_nonsel - _node_h_, 0.5)
        else:
            self.r_sel = self.r_nonsel = self.r
            self.r_inner_sel = self.r_inner_nonsel = self.r_inner

        if _nodes_len_ > 0:
            if _nodes_len_ * (_node_w_min_ + _node_gap_) > _circum_:
                _node_gap_ = max(0.0, (_circum_ - _node_w_min_ * _nodes_len_) / _nodes_len_)
            if _nodes_len_ * (_node_w_min_ + _node_gap_) > _circum_:
                _node_gap_   = 0.0
                _node_w_min_ = _circum_ / _nodes_len_

        _gap_deg_     = (_node_gap_ / _circum_) * 360.0 if _circum_ > 0 else 0.0
        _avail_deg_   = max(360.0 - _gap_deg_ * _nodes_len_, 0.001)
        _total_count_ = float(_node_space_needed_) if _node_space_needed_ else 1.0
        _to_rad_      = pi / 180.0

        # ── 3. Arc angles via Polars cumsum (no Python row loop) ──────────────
        # __arc_deg__: angular span of each node
        # __a0__: start angle (exclusive prefix sum of arc+gap, fill 0)
        # __a1__: end angle = a0 + arc_deg
        # __a0r__, __a1r__, __amr__: radians for trig (amr = midpoint, used by link geometry)
        _arc_expr_ = (
            pl.col('__count__').cast(pl.Float64) / _total_count_ * _avail_deg_
            if self.node_size == 'vary'
            else pl.lit(_avail_deg_ / max(_nodes_len_, 1))
        )
        lf = (
            self.df_node.lazy()
            .with_columns(_arc_expr_.alias('__arc_deg__'))
            .with_columns(
                ((pl.col('__arc_deg__') + _gap_deg_).cum_sum().shift(1, fill_value=0.0))
                .alias('__a0__')
            )
            .with_columns((pl.col('__a0__') + pl.col('__arc_deg__')).alias('__a1__'))
            .with_columns(
                (pl.col('__a0__') * _to_rad_).alias('__a0r__'),
                (pl.col('__a1__') * _to_rad_).alias('__a1r__'),
            )
            .with_columns(
                ((pl.col('__a0r__') + pl.col('__a1r__')) * 0.5).alias('__amr__')
            )
        )

        # Per-node outer/inner radii (two-tier when node_selection is active)
        if _node_selection_:
            _sel_set_ = {str(n) for n in _node_selection_}
            _is_sel_  = pl.col('__nm__').cast(pl.String).is_in(_sel_set_)
            lf = lf.with_columns(
                pl.when(_is_sel_).then(pl.lit(self.r_sel))        .otherwise(pl.lit(self.r_nonsel))       .alias('__r__'),
                pl.when(_is_sel_).then(pl.lit(self.r_inner_sel))  .otherwise(pl.lit(self.r_inner_nonsel)) .alias('__ri__'),
            )
        else:
            lf = lf.with_columns(pl.lit(self.r).alias('__r__'), pl.lit(self.r_inner).alias('__ri__'))

        # Eight arc-corner coordinates (outer start/end, inner start/end)
        lf = (
            lf
            .with_columns(
                (self.cx + pl.col('__r__')  * pl.col('__a0r__').cos()).alias('__x0o__'),
                (self.cy + pl.col('__r__')  * pl.col('__a0r__').sin()).alias('__y0o__'),
                (self.cx + pl.col('__r__')  * pl.col('__a1r__').cos()).alias('__x1o__'),
                (self.cy + pl.col('__r__')  * pl.col('__a1r__').sin()).alias('__y1o__'),
                (self.cx + pl.col('__ri__') * pl.col('__a0r__').cos()).alias('__x0i__'),
                (self.cy + pl.col('__ri__') * pl.col('__a0r__').sin()).alias('__y0i__'),
                (self.cx + pl.col('__ri__') * pl.col('__a1r__').cos()).alias('__x1i__'),
                (self.cy + pl.col('__ri__') * pl.col('__a1r__').sin()).alias('__y1i__'),
            )
            .with_columns(
                pl.when((pl.col('__a1__') - pl.col('__a0__')) > 180.0)
                  .then(pl.lit(1)).otherwise(pl.lit(0)).cast(pl.String)
                  .alias('__large__')
            )
        )
        self.df_node = lf.collect()

        # ── 4. node_to_arc dict (backward-compat; avoids row iteration) ───────
        self.node_to_arc = dict(zip(
            self.df_node['__nm__'].cast(pl.String).to_list(),
            zip(self.df_node['__a0__'].to_list(), self.df_node['__a1__'].to_list()),
        ))

        # ── 4a. node_outer_attach: screen coords of outer-rim midpoint per node ──
        # Maps node name → (x, y) at (cx + r·cos(amr), cy + r·sin(amr)).
        # Exposed so callers (e.g. LinkChordP) can use these as edge attachment points.
        self.node_outer_attach = {
            str(nm): (self.cx + self.r * cos(amr), self.cy + self.r * sin(amr))
            for nm, amr in self.df_node.select(['__nm__', '__amr__']).iter_rows()
        }

        # ── 5. Node color column ──────────────────────────────────────────────
        if isinstance(self.node_color, self.p2s.HexColorString):
            _co_expr_ = pl.lit(self.node_color)
        else:
            _co_expr_ = self.p2s.colorizeColumnPolarsOperations('__nm__')
        self.df_node = self.df_node.with_columns(_co_expr_.alias('__nc_hex__'))

        # ── 6. SVG path strings via concat_str (no Python row loop) ───────────
        def _r2_(c): return pl.col(c).round(2)
        _svg_ops_ = [
            pl.lit('<path d="M '), _r2_('__x0o__'), pl.lit(' '), _r2_('__y0o__'),
            pl.lit(' A '),         _r2_('__r__'),   pl.lit(' '), _r2_('__r__'),
            pl.lit(' 0 '), pl.col('__large__'), pl.lit(' 1 '),
            _r2_('__x1o__'), pl.lit(' '), _r2_('__y1o__'),
            pl.lit(' L '), _r2_('__x1i__'), pl.lit(' '), _r2_('__y1i__'),
            pl.lit(' A '), _r2_('__ri__'),  pl.lit(' '), _r2_('__ri__'),
            pl.lit(' 0 '), pl.col('__large__'), pl.lit(' 0 '),
            _r2_('__x0i__'), pl.lit(' '), _r2_('__y0i__'),
            pl.lit('" fill="'),        pl.col('__nc_hex__'),
            pl.lit('" stroke="'),      pl.col('__nc_hex__'),
            pl.lit('" stroke-width="0.8" />'),
        ]
        self.df_node = self.df_node.with_columns(
            pl.concat_str(_svg_ops_).alias('__node_svg__')
        )

        _bg_ = self.p2s.colorTyped('background', 'default')
        self.svg = ''.join([
            f'<svg x="0" y="0" width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">',
            f'<rect x="0" y="0" width="{w}" height="{h}" fill="{_bg_}" />',
            *self.df_node['__node_svg__'].to_list(),
            '</svg>',
        ])

    #
    # __renderLinks__()
    # - uses Polars group_by + concat_str to build SVG strings without Python row loops
    # - computes link attachment points from df_node arc geometry
    #
    def __renderLinks__(self):
        _WIDE_THRESH_ = 6.0   # px arc-length at inner radius to treat a node as "wide"
        _INSET_PX_    = 2.0   # px inset from each edge of a wide node
        _TENSION_     = 1.0   # max Bezier pull toward center (reached when nodes are opposite)
        _ARROW_LEN_   = 5.0   # arrowhead length, px
        _ARROW_W_     = 2.5   # arrowhead half-width, px
        _TWO_PI_      = 2.0 * pi

        _node_size_lu_ = {'small': 1, 'nil': 0.2, 'medium': 3, 'large': 5}
        _sz_           = _node_size_lu_.get(self.link_size, 1.0)
        if isinstance(self.link_size, (int, float)): _sz_ = float(self.link_size)

        _data_co_ = self.p2s.colorTyped('data', 'default')
        _lc_agg_  = self.__colorAggExprs__(self._link_color_mode_, 'lc')

        # Geometry lookup: pull only the columns needed for attachment point math
        _geom_cols_ = ['__nm__', '__a0r__', '__a1r__', '__amr__', '__r__', '__ri__', '__arc_deg__']
        _df_geom_   = self.df_node.select(_geom_cols_).with_columns(
            pl.col('__nm__').cast(pl.String)
        )
        _df_geom_fm_ = _df_geom_.rename({
            '__nm__':      '__fm_nm__',
            '__a0r__':     '__fm_a0r__',
            '__a1r__':     '__fm_a1r__',
            '__amr__':     '__fm_amr__',
            '__r__':       '__fm_r__',
            '__ri__':      '__fm_ri__',
            '__arc_deg__': '__fm_arc_deg__',
        })
        _df_geom_to_ = _df_geom_.rename({
            '__nm__':      '__to_nm__',
            '__a0r__':     '__to_a0r__',
            '__a1r__':     '__to_a1r__',
            '__amr__':     '__to_amr__',
            '__r__':       '__to_r__',
            '__ri__':      '__to_ri__',
            '__arc_deg__': '__to_arc_deg__',
        })

        _cx_ = float(self.cx)
        _cy_ = float(self.cy)
        _all_svg_ = set()
        self.df_link = None

        for _rel_ in self.relationships:
            _fm_fld_ = _rel_[0]
            _to_fld_ = _rel_[1]
            _count_agg_ = self.__countAggExpr__()

            # ── Step A: aggregate raw data by (fm, to) node pairs ────────────
            _df_link_ = (
                self.df
                .group_by([_fm_fld_, _to_fld_])
                .agg(_count_agg_, *_lc_agg_)
            )
            _df_link_ = self.__applyColorToDF__(_df_link_, self._link_color_mode_, 'lc', _data_co_)
            _df_link_ = _df_link_.with_columns(
                pl.col(_fm_fld_).cast(pl.String).alias('__fm_nm__'),
                pl.col(_to_fld_).cast(pl.String).alias('__to_nm__'),
            ).filter(pl.col('__fm_nm__') != pl.col('__to_nm__'))

            # ── Step B: join arc geometry for fm and to endpoints ─────────────
            _df_link_ = (
                _df_link_
                .join(_df_geom_fm_, on='__fm_nm__', how='left')
                .join(_df_geom_to_, on='__to_nm__', how='left')
            )

            # ── Steps C–D: arc lengths and relative angles ────────────────────
            _df_link_ = (
                _df_link_
                # C: arc length at inner radius (px) — determines wide/narrow
                .with_columns(
                    (pl.col('__fm_arc_deg__') * (pi / 180.0) * pl.col('__fm_ri__')).alias('__fm_arc_len_px__'),
                    (pl.col('__to_arc_deg__') * (pi / 180.0) * pl.col('__to_ri__')).alias('__to_arc_len_px__'),
                )
                # D: relative angles — how far is the destination CW from the source
                .with_columns(
                    ((pl.col('__to_amr__') - pl.col('__fm_amr__') + _TWO_PI_) % _TWO_PI_).alias('__theta_fm_to__'),
                    ((pl.col('__fm_amr__') - pl.col('__to_amr__') + _TWO_PI_) % _TWO_PI_).alias('__theta_to_fm__'),
                )
            )

            # ── Step D+: rank all neighbors of each node by theta ─────────────
            # Union both edge directions so each node sees its full neighborhood
            # (both outgoing and incoming), then rank by CW angle.
            # Deduplication handles bidirectional edges (A→B and B→A both present).
            _all_nbrs_ = (
                pl.concat([
                    _df_link_.select(
                        pl.col('__fm_nm__').alias('__node__'),
                        pl.col('__to_nm__').alias('__neighbor__'),
                        pl.col('__theta_fm_to__').alias('__theta__'),
                    ),
                    _df_link_.select(
                        pl.col('__to_nm__').alias('__node__'),
                        pl.col('__fm_nm__').alias('__neighbor__'),
                        pl.col('__theta_to_fm__').alias('__theta__'),
                    ),
                ])
                .unique(subset=['__node__', '__neighbor__'])
                .with_columns(
                    (pl.col('__theta__').rank(method='ordinal').over('__node__') - 1)
                      .cast(pl.Float64).alias('__rank__'),
                    pl.col('__node__').count().over('__node__')
                      .cast(pl.Float64).alias('__n_nbrs__'),
                )
            )
            _fm_ranks_ = _all_nbrs_.select(
                pl.col('__node__').alias('__fm_nm__'),
                pl.col('__neighbor__').alias('__to_nm__'),
                pl.col('__rank__').alias('__fm_rank__'),
                pl.col('__n_nbrs__').alias('__fm_n_nbrs__'),
            )
            _to_ranks_ = _all_nbrs_.select(
                pl.col('__node__').alias('__to_nm__'),
                pl.col('__neighbor__').alias('__fm_nm__'),
                pl.col('__rank__').alias('__to_rank__'),
                pl.col('__n_nbrs__').alias('__to_n_nbrs__'),
            )

            # ── Steps E–F: attachment angles and screen coords ────────────────
            # E uses rank-based t = (n - rank - 0.5) / n so all neighbors are
            # spread evenly across the full node arc in CW-angle order:
            #   rank=0 (most-CW neighbor) → t≈1.0 → CW end of node arc
            #   rank=n-1 (most-CCW neighbor) → t≈0.0 → CCW end of node arc
            _df_link_ = (
                _df_link_
                .join(_fm_ranks_, on=['__fm_nm__', '__to_nm__'], how='left')
                .join(_to_ranks_, on=['__to_nm__', '__fm_nm__'], how='left')
                # E: rank-based attachment angle
                .with_columns(
                    pl.when(pl.col('__fm_arc_len_px__') > _WIDE_THRESH_)
                      .then(
                          pl.col('__fm_a0r__')
                          + pl.lit(_INSET_PX_) / pl.col('__fm_ri__')
                          + ((pl.col('__fm_n_nbrs__') - pl.col('__fm_rank__') - 0.5) / pl.col('__fm_n_nbrs__'))
                          * (pl.col('__fm_a1r__') - pl.col('__fm_a0r__')
                             - pl.lit(2.0 * _INSET_PX_) / pl.col('__fm_ri__'))
                      )
                      .otherwise(pl.col('__fm_amr__'))
                      .alias('__fm_attach__'),
                    pl.when(pl.col('__to_arc_len_px__') > _WIDE_THRESH_)
                      .then(
                          pl.col('__to_a0r__')
                          + pl.lit(_INSET_PX_) / pl.col('__to_ri__')
                          + ((pl.col('__to_n_nbrs__') - pl.col('__to_rank__') - 0.5) / pl.col('__to_n_nbrs__'))
                          * (pl.col('__to_a1r__') - pl.col('__to_a0r__')
                             - pl.lit(2.0 * _INSET_PX_) / pl.col('__to_ri__'))
                      )
                      .otherwise(pl.col('__to_amr__'))
                      .alias('__to_attach__'),
                )
                # F: screen coordinates on the inner arc
                .with_columns(
                    (pl.lit(_cx_) + pl.col('__fm_ri__') * pl.col('__fm_attach__').cos()).alias('__fm_x__'),
                    (pl.lit(_cy_) + pl.col('__fm_ri__') * pl.col('__fm_attach__').sin()).alias('__fm_y__'),
                    (pl.lit(_cx_) + pl.col('__to_ri__') * pl.col('__to_attach__').cos()).alias('__to_x__'),
                    (pl.lit(_cy_) + pl.col('__to_ri__') * pl.col('__to_attach__').sin()).alias('__to_y__'),
                )
            )

            # ── Step G: Bezier control points pulled toward center ─────────────
            # Tension scales linearly with angular distance: 0 for adjacent nodes
            # (nearly straight chord) up to _TENSION_ for directly-opposite nodes.
            _ang_dist_ = pl.min_horizontal(
                pl.col('__theta_fm_to__'),
                pl.lit(_TWO_PI_) - pl.col('__theta_fm_to__'),
            )
            _tension_ = pl.lit(_TENSION_) * _ang_dist_ / pl.lit(pi)
            _df_link_ = _df_link_.with_columns(
                (pl.col('__fm_x__') + _tension_ * (pl.lit(_cx_) - pl.col('__fm_x__'))).alias('__cp1_x__'),
                (pl.col('__fm_y__') + _tension_ * (pl.lit(_cy_) - pl.col('__fm_y__'))).alias('__cp1_y__'),
                (pl.col('__to_x__') + _tension_ * (pl.lit(_cx_) - pl.col('__to_x__'))).alias('__cp2_x__'),
                (pl.col('__to_y__') + _tension_ * (pl.lit(_cy_) - pl.col('__to_y__'))).alias('__cp2_y__'),
            )

            # ── Step H: arrowhead at destination ──────────────────────────────
            # Direction at Bezier end = cp2 → to_point
            _df_link_ = (
                _df_link_
                .with_columns(
                    (pl.col('__to_x__') - pl.col('__cp2_x__')).alias('__arr_dx__'),
                    (pl.col('__to_y__') - pl.col('__cp2_y__')).alias('__arr_dy__'),
                )
                .with_columns(
                    ((pl.col('__arr_dx__')**2 + pl.col('__arr_dy__')**2).sqrt()).alias('__arr_mag__'),
                )
                .with_columns(
                    pl.when(pl.col('__arr_mag__') < 1e-9)
                      .then(pl.lit(1.0))
                      .otherwise(pl.col('__arr_dx__') / pl.col('__arr_mag__'))
                      .alias('__arr_nx__'),
                    pl.when(pl.col('__arr_mag__') < 1e-9)
                      .then(pl.lit(0.0))
                      .otherwise(pl.col('__arr_dy__') / pl.col('__arr_mag__'))
                      .alias('__arr_ny__'),
                )
                # Arrow base center (tip is to_x/to_y)
                .with_columns(
                    (pl.col('__to_x__') - pl.col('__arr_nx__') * pl.lit(_ARROW_LEN_)).alias('__arr_bx__'),
                    (pl.col('__to_y__') - pl.col('__arr_ny__') * pl.lit(_ARROW_LEN_)).alias('__arr_by__'),
                )
                # Left and right base corners (perpendicular = (-ny, nx))
                .with_columns(
                    (pl.col('__arr_bx__') + (-pl.col('__arr_ny__')) * pl.lit(_ARROW_W_)).alias('__arr_lx__'),
                    (pl.col('__arr_by__') + ( pl.col('__arr_nx__')) * pl.lit(_ARROW_W_)).alias('__arr_ly__'),
                    (pl.col('__arr_bx__') - (-pl.col('__arr_ny__')) * pl.lit(_ARROW_W_)).alias('__arr_rx__'),
                    (pl.col('__arr_by__') - ( pl.col('__arr_nx__')) * pl.lit(_ARROW_W_)).alias('__arr_ry__'),
                )
            )

            # ── Step I: stroke width ───────────────────────────────────────────
            if self.link_size == 'vary':
                if self.count_range_shared is not None:
                    _lc_min_ = float(self.count_range_shared[0])
                    _lc_max_ = float(self.count_range_shared[1])
                else:
                    _min_v_ = _df_link_['__count__'].min()
                    _max_v_ = _df_link_['__count__'].max()
                    _lc_min_ = float(_min_v_) if _min_v_ is not None else 0.0
                    _lc_max_ = float(_max_v_) if _max_v_ is not None else 1.0
                _stroke_w_ = (
                    pl.lit(float(self.link_size_range[0])) +
                    pl.lit(float(self.link_size_range[1] - self.link_size_range[0])) *
                    (pl.col('__count__').cast(pl.Float64) - _lc_min_) /
                    (0.01 + _lc_max_ - _lc_min_)
                )
            else:
                _stroke_w_ = pl.lit(_sz_)

            # ── Step J: dispatch to shape renderer ────────────────────────────
            if self.link_shape in ('line', 'curve'):
                _df_link_ = self._renderLinkShape_curve_(_df_link_, _stroke_w_)
            elif self.link_shape == 'bundled':
                _df_link_ = self._renderLinkShape_bundled_(_df_link_, _stroke_w_)

            _link_col_ = '__link_svg__'
            self.df_link = (
                _df_link_ if self.df_link is None
                else pl.concat([self.df_link, _df_link_], how='diagonal')
            )

            if self.link_size is not None:
                _all_svg_ |= set(_df_link_.drop_nulls(subset=[_link_col_])[_link_col_].unique())

        self._link_svg_list_ = sorted(_all_svg_)

    #
    # _renderLinkShape_curve_()
    # - builds __link_svg__ column for the 'line'/'curve' chord shape
    # - expects df_link to already have: __fm_x/y__, __to_x/y__, __cp1/2_x/y__,
    #   __arr_{lx,ly,rx,ry,to_x,to_y}__, __lc_hex__, and a stroke_w_expr
    #
    def _renderLinkShape_curve_(self, df_link, stroke_w_expr):
        _r2_ = lambda c: pl.col(c).round(2)
        _path_ops_ = [
            pl.lit('<path d="M '), _r2_('__fm_x__'), pl.lit(' '), _r2_('__fm_y__'),
            pl.lit(' C '),         _r2_('__cp1_x__'), pl.lit(' '), _r2_('__cp1_y__'),
            pl.lit(' '),           _r2_('__cp2_x__'), pl.lit(' '), _r2_('__cp2_y__'),
            pl.lit(' '),           _r2_('__to_x__'),  pl.lit(' '), _r2_('__to_y__'),
            pl.lit('" fill="none" stroke="'), pl.col('__lc_hex__'),
            pl.lit('" stroke-width="'), stroke_w_expr.round(3),
            pl.lit(f'" opacity="{self.link_opacity}" />'),
        ]
        _arrow_ops_ = [
            pl.lit('<polygon points="'),
            _r2_('__to_x__'),   pl.lit(','), _r2_('__to_y__'),   pl.lit(' '),
            _r2_('__arr_lx__'), pl.lit(','), _r2_('__arr_ly__'), pl.lit(' '),
            _r2_('__arr_rx__'), pl.lit(','), _r2_('__arr_ry__'),
            pl.lit('" fill="'), pl.col('__lc_hex__'),
            pl.lit(f'" opacity="{self.link_opacity}" />'),
        ]
        return df_link.with_columns(
            (pl.concat_str(_path_ops_) + pl.concat_str(_arrow_ops_)).alias('__link_svg__')
        )

    #
    # _build_skeleton_()
    # - returns cached routing skeleton, building it on first call via skeleton_algorithm dispatch
    # - contract: every skeleton builder returns a networkx.Graph with 'weight' on every edge
    # - adding a new algorithm: implement _build_<name>_skeleton_() and add a branch below
    #
    def _build_skeleton_(self):
        if self._bundled_skeleton_ is not None:
            return self._bundled_skeleton_
        if   self.skeleton_algorithm == 'hexagonal':
            G = self._build_hex_skeleton_()
        elif self.skeleton_algorithm == 'radial':
            G = self._build_radial_skeleton_()
        elif self.skeleton_algorithm == 'kmeans':
            G = self._build_kmeans_skeleton_()
        else:
            raise ValueError(f'ChP: unknown skeleton_algorithm {self.skeleton_algorithm!r}')
        self._bundled_skeleton_ = G
        return G

    #
    # _build_hex_skeleton_()
    # - hexagonal mesh routing graph inside the chord circle (pure builder, no cache)
    # - ported from rtsvg __renderEdges_createSkeletonHexagonal__; no rtsvg import needed
    # - hex edge length = self.r / bundle_rings; only edges with both endpoints inside
    #   self.r_inner * 0.9 are included; hex-center spokes add extra connectivity
    #
    def _build_hex_skeleton_(self):
        import networkx as nx
        from math import sqrt

        cx, cy  = self.cx, self.cy
        r_div   = self.bundle_rings
        hx_e    = self.r / r_div
        hx_h    = sqrt(hx_e**2 - (hx_e / 2)**2)
        adj_r   = self.r_inner

        def _rnd(pts):
            return [(round(x, 3), round(y, 3)) for x, y in pts]

        def _hx_corners(x, y):
            return _rnd([
                (x - hx_e,     y        ),
                (x - hx_e/2,   y + hx_h ),
                (x + hx_e/2,   y + hx_h ),
                (x + hx_e,     y        ),
                (x + hx_e/2,   y - hx_h ),
                (x - hx_e/2,   y - hx_h ),
                (x - hx_e,     y        ),  # close
            ])

        def _dist2(ax, ay, bx, by):
            return (ax - bx)**2 + (ay - by)**2

        G    = nx.Graph()
        seen = set()

        for j in range(r_div):
            y_shift = hx_h if (j % 2) == 0 else 0.0
            for i in range(r_div):
                for sx in [-1, 1]:
                    for sy in [-1, 1]:
                        hx = cx + sx * hx_e * 1.5 * j
                        hy = cy + y_shift + sy * 2 * hx_h * i
                        corners = _hx_corners(hx, hy)
                        hx_ctr  = (round(hx, 3), round(hy, 3))

                        for p0, p1 in zip(corners, corners[1:]):
                            if (_dist2(*p0, cx, cy) < (adj_r * 0.9)**2 and
                                    _dist2(*p1, cx, cy) < (adj_r * 0.9)**2):
                                edge = (p0, p1)
                                if edge not in seen:
                                    G.add_edge(p0, p1, weight=sqrt(_dist2(*p0, *p1)))
                                    seen.add(edge)
                                    seen.add((p1, p0))

                        for pt in corners:
                            if _dist2(*pt, cx, cy) < (adj_r * 0.9)**2:
                                edge = (pt, hx_ctr)
                                if edge not in seen:
                                    G.add_edge(pt, hx_ctr, weight=sqrt(_dist2(*pt, *hx_ctr)))
                                    seen.add(edge)
                                    seen.add((hx_ctr, pt))

        return G

    #
    # _build_radial_skeleton_()
    # - pure-geometric concentric ring skeleton; no data dependency
    # - bundle_rings rings, node count per ring proportional to circumference
    # - circumferential edges on each ring; each node connects to 2 nearest nodes on the
    #   adjacent inner ring; innermost ring connects to center
    #
    def _build_radial_skeleton_(self):
        import networkx as nx
        from math import cos, sin, pi, sqrt

        cx, cy  = self.cx, self.cy
        N       = self.bundle_rings
        r_inner = self.r_inner
        G       = nx.Graph()

        def _dist(a, b):
            return sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

        def _add_edge(a, b):
            G.add_edge(a, b, weight=_dist(a, b))

        rings = []
        for i in range(N):
            r_ring = r_inner * (N - i) / (N + 1)     # evenly spaced from ≈r_inner down toward center
            K      = max(4, round(2 * pi * (N - i)))  # node count scales with ring circumference
            ring   = []
            for j in range(K):
                a  = 2 * pi * j / K
                pt = (round(cx + r_ring * cos(a), 3), round(cy + r_ring * sin(a), 3))
                ring.append(pt)
            rings.append(ring)
            for j in range(K):                        # circumferential edges
                _add_edge(ring[j], ring[(j + 1) % K])

        # Radial edges: each outer-ring node → 2 nearest inner-ring nodes
        for i in range(N - 1):
            outer, inner = rings[i], rings[i + 1]
            for pt in outer:
                nearest = sorted(inner, key=lambda q: (pt[0]-q[0])**2 + (pt[1]-q[1])**2)[:2]
                for q in nearest:
                    _add_edge(pt, q)

        # Innermost ring → center
        ctr = (round(cx, 3), round(cy, 3))
        for pt in rings[-1]:
            _add_edge(pt, ctr)

        return G

    #
    # _build_kmeans_skeleton_()
    # - data-adaptive three-ring hierarchy built by successive k-means compression
    #   of node arc-midpoint positions (ported from rtsvg __renderEdges_createSkeletonKMeans__)
    # - stage 1: node midpoints → outer ring (3r/4); stage 2: outer → middle (r/2);
    #   stage 3: middle → inner (r/4); inner ring connects to center
    # - intra-cluster full connectivity at each stage; circumferential skip connections
    #   at rings 2 and 3
    #
    def _build_kmeans_skeleton_(self):
        import networkx as nx
        from math import cos, sin, atan2, sqrt

        cx, cy  = self.cx, self.cy
        r       = self.r
        r_inner = self.r_inner
        N       = self.bundle_rings
        G       = nx.Graph()

        def _dist(a, b):
            return sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

        def _add_edge(a, b):
            G.add_edge(a, b, weight=_dist(a, b))

        def _circular_mean_angle_(pts):
            """Mean angle of a cluster via unit-vector averaging (handles wrap-around)."""
            sx = sum(cos(atan2(p[1] - cy, p[0] - cx)) for p in pts) / len(pts)
            sy = sum(sin(atan2(p[1] - cy, p[0] - cx)) for p in pts) / len(pts)
            return atan2(sy, sx)

        def _kmeans_(pts, k, iterations=20):
            """Lloyd's algorithm seeded from input points; returns {center_idx: [pts]}."""
            import random
            pts = list(pts)
            n   = len(pts)
            if n == 0:
                return {}
            if n <= k:
                return {i: [pts[i]] for i in range(n)}
            rng        = random.Random(42)  # nosec B311 - fixed-seed deterministic clustering, not security sensitive
            centroids  = rng.sample(pts, k)
            assign     = {}
            for _ in range(iterations):
                assign = [[] for _ in range(k)]
                for pt in pts:
                    best = min(range(k), key=lambda i: (pt[0]-centroids[i][0])**2+(pt[1]-centroids[i][1])**2)
                    assign[best].append(pt)
                new_c = []
                for i in range(k):
                    if assign[i]:
                        mx = sum(p[0] for p in assign[i]) / len(assign[i])
                        my = sum(p[1] for p in assign[i]) / len(assign[i])
                        new_c.append((mx, my))
                    else:
                        new_c.append(centroids[i])
                moved     = sum((new_c[i][0]-centroids[i][0])**2+(new_c[i][1]-centroids[i][1])**2 for i in range(k))
                centroids = new_c
                if moved < 1e-10:
                    break
            return {i: assign[i] for i in range(k) if assign[i]}

        # Seed positions: node arc midpoints projected onto inner radius
        angles_rad = self.df_node['__amr__'].to_list()
        if len(angles_rad) < 3:
            return G

        attach_pts = list({(round(cx + r_inner * cos(a), 3), round(cy + r_inner * sin(a), 3))
                           for a in angles_rad})

        radii    = [3.0 * r / 4.0,  r / 2.0,  r / 4.0]
        k_values = [max(3, min(len(attach_pts), N * 4)),
                    max(2, min(len(attach_pts), N * 2)),
                    max(1, min(len(attach_pts), N))]

        # ── Stage 1: attach_pts → outer ring ──────────────────────────────────
        ring1 = []
        assign1 = _kmeans_(attach_pts, k_values[0])
        for cluster_pts in assign1.values():
            ang  = _circular_mean_angle_(cluster_pts)
            rpt  = (round(cx + radii[0] * cos(ang), 3), round(cy + radii[0] * sin(ang), 3))
            ring1.append(rpt)

        if not ring1:
            return G

        # ── Stage 2: outer ring → middle ring; full intra-cluster + ring edges ─
        ring2 = []
        assign2 = _kmeans_(ring1, k_values[1])
        for cluster_pts in assign2.values():
            ang  = _circular_mean_angle_(cluster_pts)
            rpt  = (round(cx + radii[1] * cos(ang), 3), round(cy + radii[1] * sin(ang), 3))
            ring2.append(rpt)
            for pt in cluster_pts:
                _add_edge(pt, rpt)               # ring-1 → ring-2
            for i, pt in enumerate(cluster_pts): # full intra-cluster ring-1 connectivity
                for other in cluster_pts[i + 1:]:
                    _add_edge(pt, other)

        ring2.sort(key=lambda p: atan2(p[1] - cy, p[0] - cx))
        n2 = len(ring2)
        for i in range(n2):                      # circumferential: adjacent + skip-1 + skip-2
            _add_edge(ring2[i], ring2[(i + 1) % n2])
            _add_edge(ring2[i], ring2[(i + 2) % n2])
            _add_edge(ring2[i], ring2[(i + 3) % n2])

        # ── Stage 3: middle ring → inner ring; inner ring → center ────────────
        ring3 = []
        assign3 = _kmeans_(ring2, k_values[2])
        for cluster_pts in assign3.values():
            ang  = _circular_mean_angle_(cluster_pts)
            rpt  = (round(cx + radii[2] * cos(ang), 3), round(cy + radii[2] * sin(ang), 3))
            ring3.append(rpt)
            for pt in cluster_pts:
                _add_edge(pt, rpt)               # ring-2 → ring-3

        ctr = (round(cx, 3), round(cy, 3))
        ring3.sort(key=lambda p: atan2(p[1] - cy, p[0] - cx))
        n3 = len(ring3)
        for i in range(n3):
            _add_edge(ring3[i], ring3[(i + 1) % n3])
            _add_edge(ring3[i], ctr)             # inner ring → center

        return G

    #
    # _svg_cubic_bspline_()
    # - applies Holten (2006) Formula 1 to skeleton waypoints, then renders as piecewise
    #   cubic B-spline SVG path string (ported from rtsvg svgPathCubicBSpline)
    # - beta=0 → straight chord, beta=1 → full skeleton routing
    # - requires len(pts) >= 2; degenerate n=2 returns a straight line segment
    #
    def _svg_cubic_bspline_(self, pts, beta):
        n = len(pts)
        if n < 2:
            return ''
        if n == 2:
            return f'M {pts[0][0]:.2f} {pts[0][1]:.2f} L {pts[1][0]:.2f} {pts[1][1]:.2f}'

        # Holten (2006) Formula 1: q_i = β·p_i + (1−β)·(p_0 + t_i·(p_{n−1} − p_0))
        def _cp_(i):
            t  = i / (n - 1)
            dx = pts[0][0] + t * (pts[-1][0] - pts[0][0])
            dy = pts[0][1] + t * (pts[-1][1] - pts[0][1])
            return (beta * pts[i][0] + (1 - beta) * dx,
                    beta * pts[i][1] + (1 - beta) * dy)

        cps = [_cp_(i) for i in range(n)]
        svg = [f'M {cps[0][0]:.2f} {cps[0][1]:.2f}']

        # Fold the first 3 control points into a cubic Bézier
        p0, p1, p2 = cps[0], cps[1], cps[2]
        pt_end   = ((p0[0] + 4*p1[0] + p2[0]) / 6,  (p0[1] + 4*p1[1] + p2[1]) / 6)
        pt_end_d = ((2*p1[0] + p2[0]) / 3,            (2*p1[1] + p2[1]) / 3)
        pt_mid_e = (pt_end[0] + (pt_end[0] - pt_end_d[0]), pt_end[1] + (pt_end[1] - pt_end_d[1]))
        pt_mid_b = ((p0[0] + pt_mid_e[0]) / 2,        (p0[1] + pt_mid_e[1]) / 2)
        svg.append(f'C {pt_mid_b[0]:.2f} {pt_mid_b[1]:.2f} {pt_mid_e[0]:.2f} {pt_mid_e[1]:.2f} {pt_end[0]:.2f} {pt_end[1]:.2f}')

        # Uniform cubic B-spline segments (Wikipedia B-spline basis, as in rtsvg)
        for i in range(n - 3):
            p1 = ((2*cps[i+1][0] +   cps[i+2][0]) / 3, (2*cps[i+1][1] +   cps[i+2][1]) / 3)
            p2 = ((  cps[i+1][0] + 2*cps[i+2][0]) / 3, (  cps[i+1][1] + 2*cps[i+2][1]) / 3)
            p3 = ((cps[i+1][0] + 4*cps[i+2][0] + cps[i+3][0]) / 6,
                  (cps[i+1][1] + 4*cps[i+2][1] + cps[i+3][1]) / 6)
            svg.append(f'C {p1[0]:.2f} {p1[1]:.2f} {p2[0]:.2f} {p2[1]:.2f} {p3[0]:.2f} {p3[1]:.2f}')

        # Fold the last 3 control points into a cubic Bézier
        p0, p1, p2 = cps[-3], cps[-2], cps[-1]
        pt_beg   = ((p0[0] + 4*p1[0] + p2[0]) / 6,  (p0[1] + 4*p1[1] + p2[1]) / 6)
        pt_beg_d = ((p0[0] + 2*p1[0]) / 3,            (p0[1] + 2*p1[1]) / 3)
        pt_mid_b = (pt_beg[0] + (pt_beg[0] - pt_beg_d[0]), pt_beg[1] + (pt_beg[1] - pt_beg_d[1]))
        pt_mid_e = ((p2[0] + pt_mid_b[0]) / 2,        (p2[1] + pt_mid_b[1]) / 2)
        svg.append(f'C {pt_mid_b[0]:.2f} {pt_mid_b[1]:.2f} {pt_mid_e[0]:.2f} {pt_mid_e[1]:.2f} {p2[0]:.2f} {p2[1]:.2f}')

        return ' '.join(svg)

    #
    # _renderLinkShape_bundled_()
    # - builds __link_svg__ for 'bundled' shape via per-row UDF (map_elements)
    # - routes each edge through the hexagonal skeleton via NetworkX Dijkstra
    # - Holten β-blending + B-spline smoothing applied in _svg_cubic_bspline_()
    # - arrowhead direction from the last blended skeleton segment
    #
    def _renderLinkShape_bundled_(self, df_link, stroke_w_expr):
        import networkx as nx
        from math import sqrt
        _ARROW_LEN_ = 5.0
        _ARROW_W_   = 2.5
        beta        = self.bundle_strength
        opacity     = self.link_opacity

        skeleton    = self._build_skeleton_()
        skel_nodes  = list(skeleton.nodes())

        def _nearest(x, y):
            best, bd = None, float('inf')
            for nx_, ny_ in skel_nodes:
                d = (nx_ - x)**2 + (ny_ - y)**2
                if d < bd:
                    bd, best = d, (nx_, ny_)
            return best

        df = df_link.with_columns(stroke_w_expr.alias('__sw__'))

        def _make_svg_(row):
            fm_x  = row['__fm_x__'];  fm_y = row['__fm_y__']
            to_x  = row['__to_x__'];  to_y = row['__to_y__']
            color = row['__lc_hex__']
            sw    = row['__sw__']

            # Route through skeleton: fm_attach → nearest entry → Dijkstra → nearest exit → to_attach
            entry = _nearest(fm_x, fm_y)
            exit_ = _nearest(to_x, to_y)

            if entry is None or exit_ is None:
                skel_path = []
            elif entry == exit_:
                skel_path = [entry]
            else:
                try:
                    skel_path = nx.shortest_path(skeleton, entry, exit_, weight='weight')
                except nx.NetworkXNoPath:
                    skel_path = [entry, exit_]

            waypts = [(fm_x, fm_y)] + skel_path + [(to_x, to_y)]
            path_d = self._svg_cubic_bspline_(waypts, beta)

            # Arrowhead: direction from last blended waypoint before to_attach → to_attach
            n = len(waypts)
            t = (n - 2) / (n - 1)
            pre_x = beta * waypts[-2][0] + (1 - beta) * (fm_x + t * (to_x - fm_x))
            pre_y = beta * waypts[-2][1] + (1 - beta) * (fm_y + t * (to_y - fm_y))
            dx, dy  = to_x - pre_x, to_y - pre_y
            mag     = sqrt(dx*dx + dy*dy) or 1e-9
            nx_, ny_ = dx / mag, dy / mag
            bx = to_x - nx_ * _ARROW_LEN_;  by = to_y - ny_ * _ARROW_LEN_
            lx = bx + (-ny_) * _ARROW_W_;   ly = by +   nx_ * _ARROW_W_
            rx = bx - (-ny_) * _ARROW_W_;   ry = by -   nx_ * _ARROW_W_

            path_svg  = (f'<path d="{path_d}" fill="none" stroke="{color}" '
                         f'stroke-width="{sw:.3f}" opacity="{opacity}" />')
            arrow_svg = (f'<polygon points="{to_x:.2f},{to_y:.2f} '
                         f'{lx:.2f},{ly:.2f} {rx:.2f},{ry:.2f}" '
                         f'fill="{color}" opacity="{opacity}" />')
            return path_svg + arrow_svg

        return df.with_columns(
            pl.struct(['__fm_x__', '__fm_y__', '__to_x__', '__to_y__',
                       '__lc_hex__', '__sw__']).map_elements(
                _make_svg_, return_dtype=pl.String
            ).alias('__link_svg__')
        )

    #
    # __renderNodes__()
    # - uses df_node arc geometry already computed by __calculateGeometry__
    # - applies proper node_color through _node_color_mode_
    # - generates radial or circular labels controlled by label_style
    #
    def __renderNodes__(self):
        # ── 1. Node color ──────────────────────────────────────────────────────
        _bg_co_   = self.p2s.colorTyped('background', 'default')
        _data_co_ = self.p2s.colorTyped('data', 'default')
        _nc_      = self._node_color_mode_

        if isinstance(self.node_color, dict):
            _filled_ = {str(k): (v if isinstance(v, self.p2s.HexColorString) else self.p2s.color(v))
                        for k, v in self.node_color.items()}
            _co_expr_ = pl.col('__nm__').cast(pl.String).replace_strict(_filled_, default=_bg_co_)
        elif _nc_['kind'] == 'fixed_hex':
            _co_expr_ = pl.lit(_nc_['hex'])
        elif _nc_['kind'] == 'categorical':   # COLOR_BY_NODE_NAME or tuple('field', …)
            _co_expr_ = self.p2s.colorizeColumnPolarsOperations('__nm__')
        else:                                  # 'default' and stat/crow modes: data default
            _co_expr_ = pl.lit(_data_co_)

        self.df_node = self.df_node.with_columns(_co_expr_.alias('__nc_hex__'))

        # ── 2. Build closed annulus-sector SVG paths ───────────────────────────
        def _r2_(c): return pl.col(c).round(2)
        _svg_ops_ = [
            pl.lit('<path d="M '), _r2_('__x0o__'), pl.lit(' '), _r2_('__y0o__'),
            pl.lit(' A '),         _r2_('__r__'),   pl.lit(' '), _r2_('__r__'),
            pl.lit(' 0 '), pl.col('__large__'), pl.lit(' 1 '),
            _r2_('__x1o__'), pl.lit(' '), _r2_('__y1o__'),
            pl.lit(' L '), _r2_('__x1i__'), pl.lit(' '), _r2_('__y1i__'),
            pl.lit(' A '), _r2_('__ri__'),  pl.lit(' '), _r2_('__ri__'),
            pl.lit(' 0 '), pl.col('__large__'), pl.lit(' 0 '),
            _r2_('__x0i__'), pl.lit(' '), _r2_('__y0i__'),
            pl.lit(' Z" fill="'),    pl.col('__nc_hex__'),
            pl.lit('" stroke="'),     pl.col('__nc_hex__'),
            pl.lit(f'" stroke-width="0.8" opacity="{self.node_opacity}" />'),
        ]
        self.df_node = self.df_node.with_columns(
            pl.concat_str(_svg_ops_).alias('__node_svg__')
        )
        self._node_svg_list_ = self.df_node['__node_svg__'].drop_nulls().to_list()

        # ── 3. Labels ──────────────────────────────────────────────────────────
        self._node_label_svg_ = []
        if self.draw_labels:
            _label_co_  = self.p2s.colorTyped('label', 'defaultfg')
            _ff_        = 'sans-serif'
            _rand_id_   = random.randint(0, 2**32)  # nosec B311 - non-cryptographic SVG id scoping, see SECURITY.md
            _label_map_ = {str(k): str(v) for k, v in (self.node_labels or {}).items()}
            _label_only_ = set(str(x) for x in self.label_only) if self.label_only else set()
            _rows_ = self.df_node.select(['__nm__', '__a0__', '__a1__', '__r__']).to_dicts()

            if self.label_style == 'circular':
                _defs_svgs_ = ['<defs>']
                _text_svgs_ = []
                for _idx_, _row_ in enumerate(_rows_):
                    _nm_ = str(_row_['__nm__'])
                    if _label_only_ and _nm_ not in _label_only_:
                        continue
                    _a0_    = float(_row_['__a0__'])
                    _a1_    = float(_row_['__a1__'])
                    _r_     = float(_row_['__r__'])
                    _ro_    = _r_ + self.txt_offset + 2
                    _a0r_   = _a0_ * pi / 180.0
                    _a1r_   = _a1_ * pi / 180.0
                    _xs_    = self.cx + _ro_ * cos(_a0r_)
                    _ys_    = self.cy + _ro_ * sin(_a0r_)
                    _xe_    = self.cx + _ro_ * cos(_a1r_)
                    _ye_    = self.cy + _ro_ * sin(_a1r_)
                    _large_ = 1 if (_a1_ - _a0_) > 180.0 else 0
                    _arc_id_ = f'chp_{_rand_id_}_arc_{_idx_}'
                    _defs_svgs_.append(
                        f'<path id="{_arc_id_}" d="M {_xs_:.2f} {_ys_:.2f} '
                        f'A {_ro_:.2f} {_ro_:.2f} 0 {_large_} 1 {_xe_:.2f} {_ye_:.2f}" '
                        f'fill="none" stroke="none" />'
                    )
                    _label_ = html.escape(_label_map_.get(_nm_, _nm_))
                    _text_svgs_.append(
                        f'<text font-family="{_ff_}" font-size="{self.txt_h}px" fill="{_label_co_}">'
                        f'<textPath href="#{_arc_id_}" startOffset="50%" text-anchor="middle">'
                        f'{_label_}</textPath></text>'
                    )
                _defs_svgs_.append('</defs>')
                self._node_label_svg_ = _defs_svgs_ + _text_svgs_

            else:  # 'radial' (default)
                for _row_ in _rows_:
                    _nm_      = str(_row_['__nm__'])
                    if _label_only_ and _nm_ not in _label_only_:
                        continue
                    _a0_      = float(_row_['__a0__'])
                    _a1_      = float(_row_['__a1__'])
                    _r_       = float(_row_['__r__'])
                    _a_mid_   = (_a0_ + _a1_) / 2.0
                    _a_mid_r_ = _a_mid_ * pi / 180.0
                    _offset_  = _r_ + self.txt_offset + 2
                    _xt_      = self.cx + _offset_ * cos(_a_mid_r_)
                    _yt_      = self.cy + _offset_ * sin(_a_mid_r_)
                    if 90.0 <= _a_mid_ < 270.0:
                        _rot_, _anchor_ = _a_mid_ - 180.0, 'end'
                    else:
                        _rot_, _anchor_ = _a_mid_, 'start'
                    _label_ = html.escape(_label_map_.get(_nm_, _nm_))
                    self._node_label_svg_.append(
                        f'<text x="{_xt_:.2f}" y="{_yt_:.2f}" font-family="{_ff_}" '
                        f'font-size="{self.txt_h}px" text-anchor="{_anchor_}" '
                        f'dominant-baseline="central" fill="{_label_co_}" '
                        f'transform="rotate({_rot_:.2f},{_xt_:.2f},{_yt_:.2f})">'
                        f'{_label_}</text>'
                    )

        # ── 4. Track node colors for interactive queries ───────────────────────
        self.color_nodes_final = {}
        for _nm_, _hex_ in self.df_node.select('__nm__', '__nc_hex__').iter_rows():
            self.color_nodes_final[str(_nm_)] = _hex_

        # ── 5. Compute instance-level count range for renderSmallMultiples ─────
        _all_counts_ = []
        if self.df_link is not None and '__count__' in self.df_link.columns:
            _all_counts_.append(self.df_link['__count__'].cast(pl.Float64))
        if self.df_node is not None and '__count__' in self.df_node.columns:
            _all_counts_.append(self.df_node['__count__'].cast(pl.Float64))
        if _all_counts_:
            _combined_ = pl.concat(_all_counts_)
            _min_v_ = _combined_.min()
            _max_v_ = _combined_.max()
            self._count_min_ = float(_min_v_) if _min_v_ is not None else 0.0
            self._count_max_ = float(_max_v_) if _max_v_ is not None else 1.0

    #
    # __renderSVG__()
    #
    def __renderSVG__(self, rand_id):
        self._gpu_payload_ = self._gpu_dl_ = None   # invalidate GPU state cached from a template
        self._render_invalid_ = False

        w, h = self.wxh
        _bg_co_     = self.p2s.colorTyped('background', 'default')
        _border_co_ = self.p2s.colorTyped('axis', 'inner')

        svg = [f'<svg x="0" y="0" width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">']
        svg.append(f'<rect x="0" y="0" width="{w}" height="{h}" fill="{_bg_co_}" />')

        # Skeleton background (rendered before links so it appears underneath)
        if self.render_skeleton:
            G = self._build_skeleton_()
            if G is not None and G.number_of_edges() > 0:
                _skel_co_ = self.p2s.colorTyped('axis', 'inner')
                for (x0, y0), (x1, y1) in G.edges():
                    svg.append(
                        f'<line x1="{x0:.2f}" y1="{y0:.2f}" x2="{x1:.2f}" y2="{y1:.2f}" '
                        f'stroke="{_skel_co_}" stroke-width="0.5" opacity="0.35" />'
                    )

        # Links
        svg.extend(self._link_svg_list_)

        # Nodes
        svg.extend(self._node_svg_list_)

        # Deferred: node labels on top
        svg.extend(self._node_label_svg_)

        # Legend (drawn into the strip reserved by __legendPrepare__); the colorbar
        # domain is finalized here -- __applyColorToDF__ has run by now
        self._dl_legend_ = None
        if getattr(self, 'legend_info', None) is not None and self._legend_region_ is not None:
            if self.legend_info.kind == 'colorbar':
                if self.color_stat_range_shared is not None and not getattr(self, '_legend_stretched_', False):
                    _vmin_, _vmax_ = self.color_stat_range_shared
                else:
                    _vmin_, _vmax_ = self._legend_stat_min_, self._legend_stat_max_
                self.p2s.legendInfoColorbarFinalize(self.legend_info, self.legend_spec, _vmin_, _vmax_)
            self._dl_legend_ = self.p2s.legendRenderDL(self.wxh, self._legend_region_, self.legend_spec,
                                                       self.legend_info, self.txt_h)
            svg.append(self._dl_legend_.svg())

        # Border
        if self.draw_border: svg.append(f'<rect x="0" y="0" width="{w-1}" height="{h-1}" fill="none" stroke="{_border_co_}" stroke-width="1" />')

        svg.append('</svg>')
        self.svg = ''.join(svg)

    #
    # render_with() - create a new instance with overrides (used by smallp cycle_by mode)
    #
    def recordsAt(self, xy, shape=None, threshold=2.0):
        if shape is None: shape = self.p2s.SELECT_CIRCLEp
        if shape != self.p2s.SELECT_CIRCLEp:
            raise ValueError(f'ChP.recordsAt(): only SELECT_CIRCLEp is supported, got {shape}')
        _x_, _y_ = xy
        _dx_, _dy_ = _x_ - self.cx, _y_ - self.cy
        _dist_ = sqrt(_dx_*_dx_ + _dy_*_dy_)
        if _dist_ < self.r_inner - threshold or _dist_ > self.r + threshold:
            return self.df.head(0)
        _angle_ = atan2(_dy_, _dx_)
        if _angle_ < 0: _angle_ += 2.0 * pi
        _nodes_ = set(
            self.df_node
            .filter((pl.col('__a0r__') <= _angle_) & (pl.col('__a1r__') >= _angle_))
            ['__nm__'].cast(pl.String).to_list()
        )
        if not _nodes_: return self.df.head(0)
        _mask_ = pl.lit(False)
        for _fm_, _to_ in self.relationships:
            _mask_ = _mask_ | pl.col(_fm_).cast(pl.String).is_in(_nodes_) | pl.col(_to_).cast(pl.String).is_in(_nodes_)
        return self.df.filter(_mask_)

    def filterByRectangle(self, bounding_box, remove_records=False):
        _x0_, _y0_, _x1_, _y1_ = bounding_box
        if _x0_ > _x1_: _x0_, _x1_ = _x1_, _x0_
        if _y0_ > _y1_: _y0_, _y1_ = _y1_, _y0_
        _r_mid_ = (self.r + self.r_inner) / 2.0
        _nodes_ = set(
            self.df_node
            .with_columns(
                (self.cx + _r_mid_ * pl.col('__amr__').cos()).alias('__mx__'),
                (self.cy + _r_mid_ * pl.col('__amr__').sin()).alias('__my__'),
            )
            .filter(
                (pl.col('__mx__') >= _x0_) & (pl.col('__mx__') <= _x1_) &
                (pl.col('__my__') >= _y0_) & (pl.col('__my__') <= _y1_)
            )
            ['__nm__'].cast(pl.String).to_list()
        )
        if not _nodes_:
            return self.df.head(0) if not remove_records else self.df
        _masks_ = []
        for _fm_, _to_ in self.relationships:
            _masks_.append(
                pl.col(_fm_).cast(pl.String).is_in(_nodes_) &
                pl.col(_to_).cast(pl.String).is_in(_nodes_)
            )
        _mask_ = _masks_[0]
        for _m_ in _masks_[1:]: _mask_ = _mask_ | _m_
        if remove_records: _mask_ = ~_mask_
        return self.df.filter(_mask_)

    def filterByOval(self, oval, remove_records=False):
        _cx_, _cy_, _rx_, _ry_ = oval
        # A plain click arrives as a zero-radius oval: keep it covering the pixel under the cursor.
        _rx_, _ry_ = max(float(_rx_), 0.5), max(float(_ry_), 0.5)
        _r_mid_ = (self.r + self.r_inner) / 2.0
        _nodes_ = set(
            self.df_node
            .with_columns(
                (self.cx + _r_mid_ * pl.col('__amr__').cos()).alias('__mx__'),
                (self.cy + _r_mid_ * pl.col('__amr__').sin()).alias('__my__'),
            )
            .filter(
                (((pl.col('__mx__') - _cx_) / _rx_).pow(2) +
                 ((pl.col('__my__') - _cy_) / _ry_).pow(2)) <= 1.0
            )
            ['__nm__'].cast(pl.String).to_list()
        )
        if not _nodes_:
            return self.df.head(0) if not remove_records else self.df
        _masks_ = []
        for _fm_, _to_ in self.relationships:
            _masks_.append(
                pl.col(_fm_).cast(pl.String).is_in(_nodes_) &
                pl.col(_to_).cast(pl.String).is_in(_nodes_)
            )
        _mask_ = _masks_[0]
        for _m_ in _masks_[1:]: _mask_ = _mask_ | _m_
        if remove_records: _mask_ = ~_mask_
        return self.df.filter(_mask_)

    def render_with(self, df, **overrides):
        return ChP(df=df, template=self, **overrides)

    #
    # renderSmallMultiples() - smallp integration
    # - SM_X: share node order (circular position of each node is identical across panels)
    # - SM_Y: share the bundled-edge routing skeleton (only meaningful when link_shape='bundled')
    # - SM_COUNT / SM_COLOR: share count / color-stat normalization ranges
    #
    def renderSmallMultiples(self, df_all, df_lu, all_key):
        _kwargs_ = {'sm_shared': self.sm_shared}
        _needs_ref_ = (self.p2s.SM_X     in self.sm_shared or
                       self.p2s.SM_Y     in self.sm_shared or
                       self.p2s.SM_COUNT in self.sm_shared or
                       self.p2s.SM_COLOR in self.sm_shared)
        if _needs_ref_:
            _ref_ = ChP(df=df_all, template=self)
            if self.p2s.SM_X in self.sm_shared:
                _kwargs_['_shared_view_x_'] = list(_ref_.order)
            if self.p2s.SM_Y in self.sm_shared and self.link_shape == 'bundled':
                _ref_._build_skeleton_()  # force skeleton build on reference
                _kwargs_['_shared_view_y_'] = _ref_._bundled_skeleton_
            if self.p2s.SM_COUNT in self.sm_shared and _ref_._count_min_ is not None:
                _kwargs_['count_range_shared'] = (_ref_._count_min_, _ref_._count_max_)
            if self.p2s.SM_COLOR in self.sm_shared and _ref_._color_stat_min_ is not None:
                _kwargs_['color_stat_range_shared'] = (_ref_._color_stat_min_, _ref_._color_stat_max_)
        return {k: ChP(df=v, template=self, **_kwargs_) for k, v in df_lu.items()}
