# Spectral seriation (Fiedler-vector ordering) for categorical axes.
#
# Given a category x feature matrix F (one row per category), build a
# category x category affinity S, form the graph Laplacian, and order the
# categories by the Fiedler vector -- the eigenvector of the second-smallest
# Laplacian eigenvalue.  This is the classic 1-D spectral embedding: it places
# similar categories adjacent, so block / cluster structure lines up along the
# axis.
#
# NumPy-first by design.  numpy is a core polars2svg dependency, so the common
# case (a handful to a few hundred categories) runs on numpy.linalg.eigh with no
# extra install.  scipy's sparse ARPACK solver (eigsh, the same call chordp.py
# uses) is used only opportunistically for large category counts when scipy is
# present; it is never required.  MLX is deliberately not used here -- it has no
# eigendecomposition, and the work is an n_categories x n_categories eigenproblem
# independent of row count, not a GPU-shaped dense O(N^2) kernel.

import numpy as np

# scipy is optional (the polars2svg[layouts] extra).  Only the large-n sparse
# fast path uses it; absent it, everything falls back to the dense numpy solve.
try:
    from scipy.sparse import csr_matrix as _csr_matrix
    from scipy.sparse.linalg import eigsh as _eigsh
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

# Above this many categories the dense eigh (O(n^3)) gets expensive, so try the
# sparse shift-invert solver first when scipy is available.
_DENSE_MAX = 512


def affinityFromFeatures(F, similarity='cosine'):
    '''Build a symmetric category x category affinity matrix from the feature
    matrix F (n_categories x n_features, non-negative counts / weights).

    similarity:
      'cosine'      -- L2-normalize each category's feature row, then dot
                       (categories with similar partner-distributions score high;
                       magnitude-invariant, the sensible default).
      'linear'      -- raw F @ F.T (favors high-count categories).
      'correlation' -- mean-center each row, then cosine (Pearson correlation).
    '''
    F = np.asarray(F, dtype=np.float64)
    if similarity == 'linear':
        S = F @ F.T
    elif similarity in ('cosine', 'correlation'):
        if similarity == 'correlation':
            F = F - F.mean(axis=1, keepdims=True)
        _norms_ = np.sqrt((F * F).sum(axis=1))
        _norms_[_norms_ == 0.0] = 1.0            # leave all-zero rows at 0 affinity
        Fn = F / _norms_[:, None]
        S = Fn @ Fn.T
    else:
        raise ValueError(f"affinityFromFeatures: unknown similarity {similarity!r} "
                         f"(expected 'cosine', 'linear', or 'correlation')")
    S = np.clip(S, 0.0, None)                     # negatives (from correlation) are not edges
    np.fill_diagonal(S, 0.0)                      # no self-affinity in the graph
    S = 0.5 * (S + S.T)                           # enforce exact symmetry
    return S


def connectedComponents(S):
    '''Index arrays for the connected components of the affinity graph S.

    Components come back in order of their smallest member index, so the result
    is a deterministic function of the caller's category order.
    '''
    n     = S.shape[0]
    _adj_ = S > 0.0
    _seen_, _comps_ = np.zeros(n, dtype=bool), []
    for _root_ in range(n):
        if _seen_[_root_]: continue
        _members_, _stack_ = [], [_root_]
        _seen_[_root_] = True
        while _stack_:                            # iterative DFS -- no recursion limit
            _i_ = _stack_.pop()
            _members_.append(_i_)
            _nbrs_ = np.nonzero(_adj_[_i_] & ~_seen_)[0]
            _seen_[_nbrs_] = True
            _stack_.extend(_nbrs_.tolist())
        _comps_.append(np.sort(np.array(_members_, dtype=int)))
    return _comps_


def _fiedlerVector(S, normalize=True):
    '''Return the Fiedler vector of the graph defined by affinity S.

    S must be CONNECTED.  On a connected graph the Laplacian's zero eigenvalue is
    simple, so eigenvector column 1 is the true Fiedler vector.  On a graph with c
    components the zero eigenvalue has multiplicity c and column 1 is an arbitrary
    basis vector of that null space -- which typically gives two components the
    same value and interleaves them.  spectralOrder() splits components out first
    so this only ever sees the connected case.
    '''
    n = S.shape[0]
    d = S.sum(axis=1)
    if normalize:
        # symmetric normalized Laplacian  L = I - D^-1/2 S D^-1/2
        with np.errstate(divide='ignore'):
            _dinv_ = 1.0 / np.sqrt(d)
        _dinv_[~np.isfinite(_dinv_)] = 0.0
        L = np.eye(n) - (_dinv_[:, None] * S * _dinv_[None, :])
    else:
        # combinatorial Laplacian  L = D - S
        L = np.diag(d) - S
    L = 0.5 * (L + L.T)

    # Large + scipy present: shift-invert ARPACK for the two smallest eigenvalues
    # (the exact fast path chordp.py uses).  Fall back to the dense solve on any
    # failure (degenerate / disconnected graphs where ARPACK struggles).
    if _HAS_SCIPY and n > _DENSE_MAX:
        try:
            # v0 is pinned: ARPACK defaults to a RANDOM start vector, which makes the
            # returned eigenvector (and so the axis order) vary between identical runs.
            _v0_      = np.random.default_rng(0).standard_normal(n)
            _, _vecs_ = _eigsh(_csr_matrix(L), k=2, sigma=1e-6, which='LM', tol=1e-5, v0=_v0_)
            return _anchorSign(_vecs_[:, 1])
        except Exception:  # nosec B110 - any ARPACK/solver failure is expected on degenerate or disconnected graphs; the dense eigh path below is the intended fallback
            pass
    # Dense path: eigh returns eigenvalues ascending; column 1 is the Fiedler vector
    # (column 0 is the trivial constant / D^1/2 mode with eigenvalue ~0).
    _vals_, _vecs_ = np.linalg.eigh(L)
    return _anchorSign(_vecs_[:, 1])


def _anchorSign(v):
    '''Fix the reflection ambiguity: an eigenvector and its negation are equally
    valid, and which one comes back is solver/permutation dependent.  Orienting on
    the largest-magnitude entry makes a component's order reproducible run to run.
    '''
    _i_ = int(np.argmax(np.abs(v)))
    return -v if v[_i_] < 0.0 else v


def spectralOrder(labels, F, similarity='cosine', normalize=True):
    '''Order `labels` (one per row of F) by spectral seriation.

    Returns a new list -- the labels sorted by their Fiedler-vector component.
    Disconnected graphs are seriated one component at a time and the components
    concatenated, so each component lands as one contiguous block; a single global
    Fiedler solve cannot do this, because the zero eigenvalue is then degenerate
    and the eigenvector picked out of that null space is arbitrary.  Components
    keep the caller's input order (by smallest member index).  Within a component
    the ordering is defined only up to reflection, resolved deterministically.
    Degenerate inputs (< 3 categories, or an affinity with no edges) return
    `labels` unchanged so the caller's deterministic input order stands.
    '''
    n = len(labels)
    if n < 3:
        return list(labels)
    S = affinityFromFeatures(F, similarity=similarity)
    if not np.any(S > 0.0):                       # no co-occurrence overlap anywhere
        return list(labels)

    _out_ = []
    for _comp_ in connectedComponents(S):
        if len(_comp_) < 3:                       # nothing to seriate -- keep input order
            _out_.extend(_comp_.tolist())
            continue
        _sub_     = S[np.ix_(_comp_, _comp_)]
        _fiedler_ = _fiedlerVector(_sub_, normalize=normalize)
        _out_.extend(_comp_[np.argsort(_fiedler_, kind='stable')].tolist())
    return [labels[i] for i in _out_]
