# Vendored from racetrack_svg_framework/rtsvg/mds_at_scale.py
# Original author: David Trimm — Apache License 2.0
# Removed: import rtsvg — replaced with optional rt_self parameter.
#   When the graph is disconnected, pass rt_self (a Polars2SVG instance) so that
#   circlePackLayout() can be used to arrange components; if rt_self is None the
#   components are placed side-by-side without packing.

import numpy as np
import networkx as nx
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix, csr_array as _csr_array
from sklearn.decomposition import PCA

#
# Implementation of Landmark MDS
#
# V. de Silva and J. Tenenbaum. Global versus local methods in nonlinear dimensionality reduction.
# In Proc. NIPS, pages 721–728, 2003.
#
class LandmarkMDSLayout(object):
    '''
    Landmark MDS graph layout (de Silva & Tenenbaum, NIPS 2003).

    Scales classical multidimensional scaling to large graphs by embedding a small
    set of landmark nodes exactly, then triangulating the rest. Satisfies the
    ``LayoutAlgorithm`` protocol: call ``.results()`` for a ``{node: (x, y)}`` dict
    to feed ``linkp``'s ``pos=``. Disconnected graphs are laid out per-component and
    arranged with circle-packing when a ``rt_self`` (Polars2SVG) instance is supplied,
    otherwise placed side-by-side.

    Parameters
    ----------
    g : networkx.Graph
    num_landmarks : int, optional
        Number of landmark nodes (defaults to a heuristic based on graph size).
    dimensions : int
        Embedding dimensionality (2 for plotting).
    landmarks, landmark_pos : optional
        Explicit landmark node set and/or their precomputed positions.
    rt_self : Polars2SVG, optional
        Enables circle-pack arrangement of disconnected components.

    Example::

        pos = LandmarkMDSLayout(g).results()
        p2s.linkp(df, [('src', 'dst')], pos)
    '''
    def __init__(self, g, num_landmarks=None, dimensions=2, landmarks=None, landmark_pos=None, rt_self=None) -> None:
        # If separate components, split up and process each separately
        if isinstance(g, nx.Graph) and nx.is_connected(g) == False:
            components = list(nx.connected_components(g))
            _pos_      = {}
            for _subgraph_nodes_ in components:
                _subgraph_        = g.subgraph(_subgraph_nodes_)
                _subgraph_layout_ = LandmarkMDSLayout(_subgraph_, num_landmarks, dimensions, landmarks, landmark_pos, rt_self)
                _pos_            |= _subgraph_layout_.results()
            if rt_self is not None:
                self.resulting_positions, _ = rt_self.circlePackLayout(g, _pos_)
            else:
                self.resulting_positions = _tileSideBySide_(g, _pos_)
            return

        # Convert graph to adjacency matrix if needed
        if isinstance(g, nx.Graph):
            n = g.number_of_nodes()
            adj_matrix = nx.to_scipy_sparse_array(g, weight='weight', format='csr')
        elif isinstance(g, (csr_matrix, _csr_array)):
            adj_matrix = g
            n = adj_matrix.shape[0]
        elif isinstance(g, np.ndarray):
            adj_matrix = csr_matrix(g)
            n = adj_matrix.shape[0]
        else:
            raise ValueError("Graph must be NetworkX graph, scipy sparse matrix, or numpy array")

        if landmarks is None and landmark_pos is None:
            if num_landmarks is None:
                num_landmarks = max(int(np.sqrt(n)), dimensions + 1)
            num_landmarks = min(num_landmarks, n)
            landmarks = []
            first_landmark = np.random.randint(0, n)
            landmarks.append(first_landmark)
            min_distances = dijkstra(adj_matrix, directed=False, indices=first_landmark)
            min_distances[np.isinf(min_distances)] = 0
            for _ in range(num_landmarks - 1):
                next_landmark = np.argmax(min_distances)
                landmarks.append(next_landmark)
                new_distances = dijkstra(adj_matrix, directed=False, indices=next_landmark)
                new_distances[np.isinf(new_distances)] = 0
                min_distances = np.minimum(min_distances, new_distances)
            landmarks = np.array(landmarks)
        else:
            if landmarks is None: landmarks = list(landmark_pos.keys())
            num_landmarks   = len(landmarks)
            _as_set_        = set(landmarks)
            _nodes_as_list_ = list(g.nodes())
            landmarks       = np.array([_nodes_as_list_.index(landmark) for landmark in _as_set_])
            if landmark_pos is not None:
                all_positions_filled = True
                for landmark in landmarks:
                    if _nodes_as_list_[landmark] not in landmark_pos:
                        all_positions_filled = False
                        break
                if not all_positions_filled: landmark_pos = None

        node_mapping = None
        if isinstance(g, nx.Graph): node_mapping = list(g.nodes())

        if landmark_pos is None:
            distances = np.zeros((n, num_landmarks))
            for i, landmark in enumerate(landmarks):
                dist = dijkstra(adj_matrix, directed=False, indices=landmark)
                distances[:, i] = dist
            max_finite_dist = np.max(distances[np.isfinite(distances)])
            distances[np.isinf(distances)] = 2 * max_finite_dist
            D_squared   = distances ** 2
            landmark_mean = D_squared.mean(axis=0)
            overall_mean  = D_squared.mean()
            B = -0.5 * (D_squared - landmark_mean - D_squared.mean(axis=1, keepdims=True) + overall_mean)
            pca  = PCA(n_components=dimensions)
            coords = pca.fit_transform(B)
        else:
            coords = np.array([landmark_pos[_nodes_as_list_[landmark]] for landmark in landmarks])

        if isinstance(g, nx.Graph):
            adj_matrix = nx.to_scipy_sparse_array(g, weight='weight', format='csr')
        elif isinstance(g, csr_matrix):
            adj_matrix = g
        else:
            adj_matrix = csr_matrix(g)

        n = adj_matrix.shape[0]
        distances = np.zeros((n, len(landmarks)))
        for i, landmark in enumerate(landmarks):
            dist = dijkstra(adj_matrix, directed=False, indices=landmark)
            distances[:, i] = dist
        max_finite_dist = np.max(distances[np.isfinite(distances)])
        distances[np.isinf(distances)] = 2 * max_finite_dist

        self.coords, self.landmarks, self.node_mapping = coords, landmarks, node_mapping

        self.resulting_positions = {}
        for i in range(len(self.coords)):
            key = self.node_mapping[i] if self.node_mapping is not None else i
            self.resulting_positions[key] = self.coords[i]

    def results(self) -> dict: return self.resulting_positions


#
# Implementation of PivotMDS
#
# U. Brandes and C. Pich. Eigensolver methods for progressive multidimensional scaling of large data.
# In Proceedings 14th Symposium on Graph Drawing (GD), pages 42–53, 2006.
#
class PivotMDSLayout(object):
    '''
    Pivot MDS graph layout (Brandes & Pich, Graph Drawing 2006).

    A progressive eigensolver-based multidimensional scaling that embeds large graphs
    from distances to a small set of pivot nodes. Satisfies the ``LayoutAlgorithm``
    protocol: call ``.results()`` for a ``{node: (x, y)}`` dict to feed ``linkp``'s
    ``pos=``. Disconnected graphs are laid out per-component and arranged with
    circle-packing when a ``rt_self`` (Polars2SVG) instance is supplied, otherwise
    placed side-by-side.

    Parameters
    ----------
    g : networkx.Graph
    num_pivots : int, optional
        Number of pivot nodes (defaults to a heuristic based on graph size).
    dimensions : int
        Embedding dimensionality (2 for plotting).
    rt_self : Polars2SVG, optional
        Enables circle-pack arrangement of disconnected components.

    Example::

        pos = PivotMDSLayout(g).results()
        p2s.linkp(df, [('src', 'dst')], pos)
    '''
    def __init__(self, g, num_pivots=None, dimensions=2, rt_self=None) -> None:
        # If separate components, split up and process each separately
        if isinstance(g, nx.Graph) and nx.is_connected(g) == False:
            components = list(nx.connected_components(g))
            _pos_      = {}
            for _subgraph_nodes_ in components:
                _subgraph_        = g.subgraph(_subgraph_nodes_)
                _subgraph_layout_ = PivotMDSLayout(_subgraph_, num_pivots, dimensions, rt_self)
                _pos_            |= _subgraph_layout_.results()
            if rt_self is not None:
                self.resulting_positions, _ = rt_self.circlePackLayout(g, _pos_)
            else:
                self.resulting_positions = _tileSideBySide_(g, _pos_)
            return

        # Convert graph to adjacency matrix if needed
        if isinstance(g, nx.Graph):
            n = g.number_of_nodes()
            node_mapping = list(g.nodes())
            adj_matrix = nx.to_scipy_sparse_array(g, weight='weight', format='csr')
        elif isinstance(g, (csr_matrix, _csr_array)):
            adj_matrix = g
            n = adj_matrix.shape[0]
            node_mapping = None
        elif isinstance(g, np.ndarray):
            adj_matrix = csr_matrix(g)
            n = adj_matrix.shape[0]
            node_mapping = None
        else:
            raise ValueError("Graph must be NetworkX graph, scipy sparse matrix, or numpy array")

        if num_pivots is None:
            num_pivots = max(int(np.sqrt(n)), dimensions + 1)
        num_pivots = min(num_pivots, n)

        pivots = []
        first_pivot = np.random.randint(0, n)
        pivots.append(first_pivot)
        min_distances = dijkstra(adj_matrix, directed=False, indices=first_pivot)
        min_distances[np.isinf(min_distances)] = 0
        for _ in range(num_pivots - 1):
            next_pivot = np.argmax(min_distances)
            pivots.append(next_pivot)
            new_distances = dijkstra(adj_matrix, directed=False, indices=next_pivot)
            new_distances[np.isinf(new_distances)] = 0
            min_distances = np.minimum(min_distances, new_distances)
        pivots = np.array(pivots)

        distances = np.zeros((n, num_pivots))
        for i, pivot in enumerate(pivots):
            dist = dijkstra(adj_matrix, directed=False, indices=pivot)
            distances[:, i] = dist
        max_finite_dist = np.max(distances[np.isfinite(distances)])
        distances[np.isinf(distances)] = 2 * max_finite_dist

        col_means = distances.mean(axis=0)
        distances_centered = distances - col_means

        if num_pivots > dimensions:
            U, S, Vt = np.linalg.svd(distances_centered, full_matrices=False)
            coords = U[:, :dimensions] * S[:dimensions]
        else:
            coords = distances_centered

        self.coords, self.pivots, self.node_mapping = coords, pivots, node_mapping

        self.resulting_positions = {}
        for i in range(len(self.coords)):
            key = self.node_mapping[i] if self.node_mapping is not None else i
            self.resulting_positions[key] = self.coords[i]

    def results(self) -> dict: return self.resulting_positions


def _tileSideBySide_(g, pos):
    """Fallback for disconnected graphs when no rt_self: place components in a row."""
    components = list(nx.connected_components(g))
    offset, result = 0.0, {}
    for comp_nodes in components:
        xs = [pos[n][0] for n in comp_nodes]
        x_min = min(xs) if xs else 0.0
        x_max = max(xs) if xs else 0.0
        for n in comp_nodes:
            result[n] = (pos[n][0] - x_min + offset, pos[n][1])
        offset += (x_max - x_min) + 0.1
    return result
