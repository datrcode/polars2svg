# Implements the force-directed origin-destination flow map layout from:
#
# Force-Directed Layout of Origin-Destination Flow Maps
# Bernhard Jenny, Daniel M. Stephen, Ian Muehlenhaus, Brooke E. Marston,
# Ritesh Sharma, Eugene Zhang & Helen Jenny
# International Journal of Geographical Information Science, 2017
# http://dx.doi.org/10.1080/13658816.2017.1307378
#
# Deviations from the paper:
# - Arrowhead rendering (section 3.3) lives in linkp; when ``arrows`` is on,
#   the move-flows-off-obstacles phase (section 3.2.3) models other flows'
#   arrowheads as circular obstacles at their arrival points.
# - Flow-against-flow forces (section 3.1.1) are evaluated at a fixed number of
#   sampled points per flow; peripherality (section 3.1.4) reuses those
#   per-point forces instead of re-sampling straight-line segments.
#
# The O(N^2) force kernels and the intersection / obstacle geometry are
# vectorized with NumPy (a core dependency).  When the optional ``mlx`` extra is
# installed and exposes a usable GPU, the two flows-against-flow / nodes-against-
# flow force kernels run on the GPU in float32; everything else stays NumPy.

import contextlib
import math
import numpy as np

# MLX is optional (polars2svg[mlx] / [mlx-cuda]).  Absent it, everything runs on
# the NumPy path; import failure must never break ODFlowLayout.
try:
    import mlx.core as mx
except ImportError:
    mx = None


# ---------------------------------------------------------------------------
# Device resolution (mirrors tfdp_layout._default_device, but self-contained so
# that importing this module never hard-requires mlx)
# ---------------------------------------------------------------------------
#
# mx.gpu is Metal on Apple silicon and CUDA on a Linux mlx[cuda*] build; the force
# kernels are backend-agnostic mlx.core and do not care which.  But mx.gpu is not
# always usable (the plain Linux mlx wheel has no GPU backend), so probe it once
# with real arithmetic rather than assume it: MLX's CUDA backend JIT-compiles
# kernels against the system CUDA headers, so a header/toolkit mismatch only
# surfaces at the first *compiled* kernel — an allocation-only probe can sail past
# it and report a healthy GPU that then explodes mid-layout.  Cached because the
# probe costs a device init plus a kernel compile.

_DEVICE_CACHE = None


def _default_device():
    """Resolve mx.gpu (Metal or CUDA) once, falling back to mx.cpu if unusable."""
    global _DEVICE_CACHE
    if _DEVICE_CACHE is None:
        try:
            with mx.stream(mx.gpu):
                _probe = mx.array([1.0, 2.0])
                mx.eval(mx.sum(_probe * _probe))
            _DEVICE_CACHE = mx.gpu
        except Exception:  # noqa: BLE001 - any backend failure means "no GPU"
            _DEVICE_CACHE = mx.cpu
    return _DEVICE_CACHE


#
# ODFlowLayout() - one quadratic Bezier control point per flow (Jenny et al. 2017)
#
class ODFlowLayout(object):
    '''
    Force-directed layout of origin-destination flows (Jenny et al., IJGIS 2017).

    Each flow is modeled as a quadratic Bezier curve whose single control point
    is positioned by an iterative equilibrium of five forces (flows-against-flow,
    nodes-against-flow, anti-torsion, spring, angular resolution) plus geometric
    constraints (per-flow and canvas rectangles), intersection reduction for
    flows sharing a node, and moving flows off unconnected nodes.

    Runtime is O(iterations x (flows x samples_per_flow)^2); intended for flow
    maps at the paper's scale (up to ~100-200 aggregated flows).  The algorithm
    is deterministic.

    The O(N^2) force kernels run on NumPy by default.  When the optional ``mlx``
    extra is installed and a GPU is available they run on the GPU in float32:
    output stays deterministic for a given machine/backend, but the fine float
    detail differs from the NumPy path (float32 vs float64) by amounts far below
    one pixel.  Intersection reduction, obstacle clearance and the per-flow
    scalar forces always run on NumPy/Python.

    Parameters
    ----------
    flows : list of (fm_x, fm_y, to_x, to_y)
        Flow endpoints in canvas (screen) coordinates.
    node_radius : float
        Radius of the node symbols; used as obstacle size when moving flows
        off unconnected nodes.
    canvas : (x0, y0, x1, y1), optional
        Rectangle control points are constrained to; defaults to the flow
        bounding box enlarged to twice its width and height (paper default).
    iterations : int
        Force iterations (paper default 100).
    samples_per_flow : int
        Points sampled along each curve for the flow/node force evaluation.

    The remaining keyword parameters are the paper's tuning constants with the
    paper's default values (section 3): force weights ``w_flows``, ``w_nodes``,
    ``w_antitorsion``, ``w_spring``, ``w_angres``; inverse-distance exponents
    ``alpha``, ``beta``; spring constants ``k_short``, ``k_long`` and peripheral
    stiffening ``c_p``; angular-resolution constants ``k_angres``, ``c_angres``;
    constraint rectangle height ``rect_pct`` (fraction of flow length); and the
    obstacle clearance ``min_obstacle_dist`` in pixels.  With ``arrows=True``,
    other flows' arrowheads (circles of ``arrow_radius`` at their arrival
    points) also count as obstacles (section 3.2.3).

    Example::

        cps = ODFlowLayout(flows).results()   # list of (cx, cy), one per flow
    '''
    def __init__(self, flows, node_radius=5.0, canvas=None, iterations=100, samples_per_flow=15,
                 w_flows=1.0, w_nodes=0.5, w_antitorsion=0.8, w_spring=1.0, w_angres=3.75,
                 alpha=4.0, beta=4.0, k_short=0.5, k_long=0.05, c_p=2.5,
                 k_angres=4.0, c_angres=4.0, rect_pct=0.5, min_obstacle_dist=4.0,
                 arrows=False, arrow_radius=0.0):
        self.flows             = [(float(a), float(b), float(c), float(d)) for a, b, c, d in flows]
        self.node_radius       = node_radius
        self.iterations        = iterations
        self.samples_per_flow  = samples_per_flow
        self.w_flows,  self.w_nodes  = w_flows,  w_nodes
        self.w_antitorsion           = w_antitorsion
        self.w_spring, self.w_angres = w_spring, w_angres
        self.alpha,    self.beta     = alpha,    beta
        self.k_short,  self.k_long   = k_short,  k_long
        self.c_p                     = c_p
        self.k_angres, self.c_angres = k_angres, c_angres
        self.rect_pct                = rect_pct
        self.min_obstacle_dist       = min_obstacle_dist
        self.arrows                  = arrows
        self.arrow_radius            = arrow_radius

        # Nodes are the unique flow endpoints; a node is "connected" to a flow
        # iff it is one of that flow's endpoints (paper section 3.1.2)
        _node_id_ = {}
        self.node_xy = []
        for _f_ in self.flows:
            for _pt_ in ((_f_[0], _f_[1]), (_f_[2], _f_[3])):
                if _pt_ not in _node_id_:
                    _node_id_[_pt_] = len(self.node_xy)
                    self.node_xy.append(_pt_)
        self.flow_nodes = [(_node_id_[(f[0], f[1])], _node_id_[(f[2], f[3])]) for f in self.flows]
        self.node_flows = [set() for _ in self.node_xy]
        for _i_, (_s_, _e_) in enumerate(self.flow_nodes):
            self.node_flows[_s_].add(_i_), self.node_flows[_e_].add(_i_)

        # Baseline lengths; zero-length flows (self loops on one pixel) are inert
        self.B      = [math.hypot(f[2] - f[0], f[3] - f[1]) for f in self.flows]
        self.active = [i for i in range(len(self.flows)) if self.B[i] > 1e-9]
        self.B_max  = max(self.B) if self.B else 1.0
        if self.B_max <= 0.0: self.B_max = 1.0

        if canvas is None and len(self.flows) > 0:
            _xs_ = [c for f in self.flows for c in (f[0], f[2])]
            _ys_ = [c for f in self.flows for c in (f[1], f[3])]
            _cw_, _ch_ = max(_xs_) - min(_xs_), max(_ys_) - min(_ys_)
            canvas = (min(_xs_) - _cw_ / 2.0, min(_ys_) - _ch_ / 2.0,
                      max(_xs_) + _cw_ / 2.0, max(_ys_) + _ch_ / 2.0)
        self.canvas = canvas

        # Control points start at the baseline midpoints (straight flows)
        self.cps = [((f[0] + f[2]) / 2.0, (f[1] + f[3]) / 2.0) for f in self.flows]

        self._pinned_ = set()  # flows moved off obstacles; immovable afterwards (3.2.3)

        # Pick the force-kernel backend: MLX GPU when installed and usable, else
        # NumPy.  On mx.cpu NumPy is faster, so only take MLX for a real GPU.
        self._use_mlx_ = (mx is not None) and (_default_device() == mx.gpu)
        self._xp_      = mx if self._use_mlx_ else np
        self._dev_     = _default_device() if self._use_mlx_ else None

        with self._stream_():
            self._buildArrays_()

        if len(self.active) > 1 or (len(self.active) == 1 and len(self.node_xy) > 2):
            self._iterate_()

    #
    # _buildArrays_() - precompute the per-run constant arrays for the force
    # kernels (backend arrays) and the intersection/obstacle geometry (NumPy)
    #
    def _buildArrays_(self):
        xp = self._xp_
        _dt_ = mx.float32 if self._use_mlx_ else np.float64

        _A_ = len(self.active)
        self._A_ = _A_
        self._act_pos_ = {f: p for p, f in enumerate(self.active)}

        # Quadratic-Bezier Bernstein coefficients at t=(k+0.5)/P for force sampling
        _P_ = self.samples_per_flow
        _t_ = (np.arange(_P_) + 0.5) / _P_
        self._bern_ = (xp.array(((1 - _t_) * (1 - _t_)).astype(np.float64), dtype=_dt_),
                       xp.array((2 * (1 - _t_) * _t_).astype(np.float64),   dtype=_dt_),
                       xp.array((_t_ * _t_).astype(np.float64),             dtype=_dt_))

        # Active-flow endpoints (backend arrays), indexed by active position
        if _A_ > 0:
            _af_ = np.array([self.flows[f] for f in self.active], dtype=np.float64)
        else:
            _af_ = np.zeros((0, 4), dtype=np.float64)
        self._afx_ = xp.array(_af_[:, 0], dtype=_dt_)
        self._afy_ = xp.array(_af_[:, 1], dtype=_dt_)
        self._atx_ = xp.array(_af_[:, 2], dtype=_dt_)
        self._aty_ = xp.array(_af_[:, 3], dtype=_dt_)

        # Per-sample-point flow id (active position); same-flow pairs get zero
        # weight in the flow force (computed per row-chunk from this vector so the
        # full N x N mask never has to be materialized)
        self._pt_flow_ = xp.array(np.repeat(np.arange(_A_), _P_).astype(np.int64))

        # Nodes and the connected (active-flow, node) mask for the node force
        _nodes_ = np.array(self.node_xy, dtype=np.float64) if self.node_xy else np.zeros((0, 2))
        self._node_xy_arr_ = xp.array(_nodes_, dtype=_dt_)
        _nn_ = len(self.node_xy)
        _cm_ = np.zeros((_A_, _nn_), dtype=bool)
        for _pos_, _f_ in enumerate(self.active):
            _s_, _e_ = self.flow_nodes[_f_]
            _cm_[_pos_, _s_] = True
            _cm_[_pos_, _e_] = True
        self._conn_mask_ = xp.array(_cm_)

        # NumPy geometry for intersections/obstacles (always float64, all flows)
        if self.flows:
            _fl_ = np.array(self.flows, dtype=np.float64)
        else:
            _fl_ = np.zeros((0, 4), dtype=np.float64)
        self._fl_fx_, self._fl_fy_ = _fl_[:, 0], _fl_[:, 1]
        self._fl_tx_, self._fl_ty_ = _fl_[:, 2], _fl_[:, 3]
        # Bernstein coefficients for the intersection test (n=20 -> 21 points)
        _ti_ = np.arange(21) / 20.0
        self._is_a_ = ((1 - _ti_) * (1 - _ti_))
        self._is_b_ = (2 * (1 - _ti_) * _ti_)
        self._is_c_ = (_ti_ * _ti_)

        # Move-off spiral (3.2.3): the candidate (theta, r) sequence depends only
        # on the constant spacing, so its (dx, dy) offsets are identical for every
        # flow - precompute them once (each flow just adds its center and cuts the
        # sequence at r <= B).  r is strictly increasing, so a searchsorted picks
        # the per-flow candidate count.
        _spc_ = self.min_obstacle_dist
        _sr_, _sdx_, _sdy_ = [], [], []
        _theta_, _r_, _tries_ = 0.0, 0.0, 0
        while _tries_ < 2000 and _spc_ > 0:
            _sr_.append(_r_)
            _sdx_.append(_r_ * math.cos(_theta_))
            _sdy_.append(_r_ * math.sin(_theta_))
            _tries_ += 1
            _theta_ += _spc_ / max(_r_, _spc_)
            _r_ = _spc_ * _theta_ / (2.0 * math.pi)
        self._spiral_r_  = np.array(_sr_)  if _sr_  else np.zeros(0)
        self._spiral_dx_ = np.array(_sdx_) if _sdx_ else np.zeros(0)
        self._spiral_dy_ = np.array(_sdy_) if _sdy_ else np.zeros(0)

    def results(self) -> list:
        '''Control points, one ``(cx, cy)`` tuple per input flow (input order).'''
        return list(self.cps)

    def _stream_(self):
        '''Pin MLX ops to the resolved GPU (Metal/CUDA); a no-op on the NumPy path.

        MLX's default device is not guaranteed to be the GPU on every build, so
        the force kernels run inside this stream (as TFDPLayout does).'''
        if self._use_mlx_:
            return mx.stream(self._dev_)
        return contextlib.nullcontext()

    #
    # _iterate_() - the main loop (pseudo-code, paper pp. 4-5)
    #
    def _iterate_(self):
        _j_ = 0
        for _i_ in range(self.iterations):
            _w_ = 1.0 - _i_ / self.iterations
            with self._stream_():
                _xs_, _ys_                   = self._points_()
                _ffx_a_, _ffy_a_, _periph_a_ = self._flowForces_(_xs_, _ys_)
                _fnx_a_, _fny_a_             = self._nodeForces_(_xs_, _ys_)
            _new_cps_ = {}
            for _pos_, _f_ in enumerate(self.active):
                if _f_ in self._pinned_: continue
                _ffx_, _ffy_          = float(_ffx_a_[_pos_]), float(_ffy_a_[_pos_])
                _fnx_, _fny_          = float(_fnx_a_[_pos_]), float(_fny_a_[_pos_])
                _atx_, _aty_          = self._antiTorsionForce_(_f_)
                _spx_, _spy_          = self._springForce_(_f_, float(_periph_a_[_pos_]))
                _arx_, _ary_          = self._angResForce_(_f_)
                _fx_ = _w_ * (self.w_flows * _ffx_ + self.w_nodes * _fnx_ +
                              self.w_antitorsion * _atx_ + self.w_spring * _spx_) + \
                       (_w_ - _w_ * _w_) * self.w_angres * _arx_
                _fy_ = _w_ * (self.w_flows * _ffy_ + self.w_nodes * _fny_ +
                              self.w_antitorsion * _aty_ + self.w_spring * _spy_) + \
                       (_w_ - _w_ * _w_) * self.w_angres * _ary_
                _cp_ = (self.cps[_f_][0] + _fx_, self.cps[_f_][1] + _fy_)
                _new_cps_[_f_] = self._constrain_(_f_, _cp_)
            for _f_, _cp_ in _new_cps_.items(): self.cps[_f_] = _cp_

            self._reduceIntersections_()

            # Moving flows off unconnected nodes (3.2.3), paced per the paper
            if _i_ > 0.1 * self.iterations and _j_ <= 0:
                _N_ = [f for f in self.active if f not in self._pinned_ and self._overlapsObstacle_(f)]
                if len(_N_) > 0:
                    _j_      = (self.iterations - _i_) / (len(_N_) + 1) / 2.0
                    _denom_  = max(1, self.iterations - _i_ - 1)
                    _n_      = math.ceil(len(_N_) / _denom_)
                    _moved_  = 0
                    for _f_ in _N_:
                        if self._moveOffObstacles_(_f_):
                            _moved_ += 1
                            if _moved_ >= _n_: break
            else:
                _j_ -= 1

    #
    # _points_() - evenly spaced sample points along every active flow, as two
    # (A, P) backend arrays (xs, ys)
    # - t at (k + 0.5)/samples so shared endpoints never yield coincident samples
    #
    def _points_(self):
        xp = self._xp_
        _dt_ = mx.float32 if self._use_mlx_ else np.float64
        _A_ = self._A_
        if _A_ == 0:
            _z_ = xp.zeros((0, self.samples_per_flow), dtype=_dt_)
            return _z_, _z_
        _cpx_ = xp.array(np.array([self.cps[f][0] for f in self.active], dtype=np.float64), dtype=_dt_)
        _cpy_ = xp.array(np.array([self.cps[f][1] for f in self.active], dtype=np.float64), dtype=_dt_)
        _a_, _b_, _c_ = self._bern_                              # each (P,)
        _xs_ = (_a_[None, :] * self._afx_[:, None] +
                _b_[None, :] * _cpx_[:, None] +
                _c_[None, :] * self._atx_[:, None])             # (A, P)
        _ys_ = (_a_[None, :] * self._afy_[:, None] +
                _b_[None, :] * _cpy_[:, None] +
                _c_[None, :] * self._aty_[:, None])
        return _xs_, _ys_

    #
    # _flowForces_() - flows-against-flow (3.1.1) + peripherality ratio (3.1.4)
    # - Shepard inverse-distance weighting of point-to-point displacement vectors
    # - returns (fx, fy, periph) NumPy arrays indexed by active position
    #
    def _flowForces_(self, _xs_, _ys_):
        _A_, _P_ = self._A_, self.samples_per_flow
        if _A_ < 2:
            _z_ = np.zeros(_A_)
            return _z_, _z_.copy(), _z_.copy()
        xp = self._xp_
        _N_ = _A_ * _P_
        _px_ = _xs_.reshape(_N_)
        _py_ = _ys_.reshape(_N_)
        _pf_ = self._pt_flow_                                    # (N,) flow id per point

        # Per-point Shepard force, computed in row-chunks so the N x N distance
        # matrix never has to be materialized whole (chunking the row axis leaves
        # each per-row reduction untouched -> identical result)
        _CH_ = 2048
        _fpx_parts_, _fpy_parts_ = [], []
        for _s_ in range(0, _N_, _CH_):
            _e_ = min(_s_ + _CH_, _N_)
            _dx_ = _px_[_s_:_e_, None] - _px_[None, :]          # (c, N)
            _dy_ = _py_[_s_:_e_, None] - _py_[None, :]
            _d_  = xp.sqrt(_dx_ * _dx_ + _dy_ * _dy_)
            _d_  = xp.maximum(_d_, 1e-4)
            _wt_ = _d_ ** (-self.alpha)
            # zero weight for pairs on the same flow (self-pairs included)
            _same_ = (_pf_[_s_:_e_, None] == _pf_[None, :])
            _wt_ = xp.where(_same_, xp.zeros_like(_wt_), _wt_)
            _swt_ = xp.sum(_wt_, axis=1)                         # (c,)
            _fpx_parts_.append(xp.sum(_dx_ * _wt_, axis=1) / _swt_)
            _fpy_parts_.append(xp.sum(_dy_ * _wt_, axis=1) / _swt_)
        _fpx_ = xp.concatenate(_fpx_parts_).reshape(_A_, _P_)
        _fpy_ = xp.concatenate(_fpy_parts_).reshape(_A_, _P_)

        _mfx_ = xp.sum(_fpx_, axis=1) / _P_                     # per-flow force = mean over P
        _mfy_ = xp.sum(_fpy_, axis=1) / _P_
        _sfx_ = xp.sum(_fpx_, axis=1)
        _sfy_ = xp.sum(_fpy_, axis=1)
        _fpm_ = xp.sqrt(_fpx_ * _fpx_ + _fpy_ * _fpy_)
        _sfm_ = xp.sum(_fpm_, axis=1)
        _num_ = xp.sqrt(_sfx_ * _sfx_ + _sfy_ * _sfy_)
        _periph_ = xp.where(_sfm_ > 0, _num_ / xp.where(_sfm_ > 0, _sfm_, xp.ones_like(_sfm_)),
                            xp.zeros_like(_sfm_))
        return self._toNP_(_mfx_), self._toNP_(_mfy_), self._toNP_(_periph_)

    #
    # _nodeForces_() - nodes-against-flow (3.1.2): each unconnected node repels
    # via the vector to the closest sampled point on the flow
    # - returns (fx, fy) NumPy arrays indexed by active position
    #
    def _nodeForces_(self, _xs_, _ys_):
        _A_ = self._A_
        _nn_ = len(self.node_xy)
        if _A_ == 0 or _nn_ == 0:
            _z_ = np.zeros(_A_)
            return _z_, _z_.copy()
        xp = self._xp_
        _nx_ = self._node_xy_arr_[:, 0]                         # (nodes,)
        _ny_ = self._node_xy_arr_[:, 1]

        # Chunk over active flows to bound the (a, P, nodes) tensor
        _ACH_ = max(1, 2048 // max(1, _nn_))
        _fx_parts_, _fy_parts_ = [], []
        for _s_ in range(0, _A_, _ACH_):
            _e_ = min(_s_ + _ACH_, _A_)
            _sx_ = _xs_[_s_:_e_]                                # (a, P)
            _sy_ = _ys_[_s_:_e_]
            _dx_ = _sx_[:, :, None] - _nx_[None, None, :]       # (a, P, nodes)
            _dy_ = _sy_[:, :, None] - _ny_[None, None, :]
            _d_  = xp.sqrt(_dx_ * _dx_ + _dy_ * _dy_)
            _d_  = xp.maximum(_d_, 1e-4)
            # closest sample per (flow, node): argmin over P; NumPy/MLX argmin
            # take the first minimum -> matches the sort_by(d).first() tie-break
            _idx_ = xp.argmin(_d_, axis=1)                      # (a, nodes)
            _ix_  = _idx_[:, None, :]
            _dxm_ = xp.take_along_axis(_dx_, _ix_, axis=1)[:, 0, :]
            _dym_ = xp.take_along_axis(_dy_, _ix_, axis=1)[:, 0, :]
            _dm_  = xp.take_along_axis(_d_,  _ix_, axis=1)[:, 0, :]
            _wt_  = _dm_ ** (-self.beta)                        # (a, nodes)
            # connected nodes exert no force -> zero weight
            _wt_  = xp.where(self._conn_mask_[_s_:_e_], xp.zeros_like(_wt_), _wt_)
            _swt_ = xp.sum(_wt_, axis=1)                        # (a,)
            _good_ = _swt_ > 0
            _den_  = xp.where(_good_, _swt_, xp.ones_like(_swt_))
            _fx_   = xp.where(_good_, xp.sum(_dxm_ * _wt_, axis=1) / _den_, xp.zeros_like(_swt_))
            _fy_   = xp.where(_good_, xp.sum(_dym_ * _wt_, axis=1) / _den_, xp.zeros_like(_swt_))
            _fx_parts_.append(_fx_)
            _fy_parts_.append(_fy_)
        _fx_all_ = xp.concatenate(_fx_parts_)
        _fy_all_ = xp.concatenate(_fy_parts_)
        return self._toNP_(_fx_all_), self._toNP_(_fy_all_)

    def _toNP_(self, _arr_):
        '''Materialize a backend array as a NumPy float64 array.'''
        if self._use_mlx_:
            mx.eval(_arr_)
            return np.array(_arr_, dtype=np.float64)
        return np.asarray(_arr_, dtype=np.float64)

    #
    # _antiTorsionForce_() - pull toward the perpendicular bisector (3.1.3);
    # length = distance to the bisector, direction along the baseline
    #
    def _antiTorsionForce_(self, f):
        _fx_, _fy_, _tx_, _ty_ = self.flows[f]
        _B_ = self.B[f]
        _ex_, _ey_ = (_tx_ - _fx_) / _B_, (_ty_ - _fy_) / _B_
        _lx_ = (self.cps[f][0] - _fx_) * _ex_ + (self.cps[f][1] - _fy_) * _ey_
        _m_  = _B_ / 2.0 - _lx_
        return (_m_ * _ex_, _m_ * _ey_)

    #
    # _springForce_() - Hooke pull toward the base point (3.1.4); spring constant
    # interpolated by flow length and stiffened for peripheral flows
    #
    def _springForce_(self, f, periph_ratio):
        _fx_, _fy_, _tx_, _ty_ = self.flows[f]
        _k_ = (self.k_long - self.k_short) * (self.B[f] / self.B_max) + self.k_short
        _k_ *= periph_ratio * self.c_p + 1.0
        _mx_, _my_ = (_fx_ + _tx_) / 2.0, (_fy_ + _ty_) / 2.0
        return (_k_ * (_mx_ - self.cps[f][0]), _k_ * (_my_ - self.cps[f][1]))

    #
    # _angResForce_() - angular resolution at shared nodes (3.1.5)
    #
    def _angResForce_(self, f):
        _cp_ = self.cps[f]

        def _end_force_(node):
            _px_, _py_ = self.node_xy[node]
            _vx_, _vy_ = _cp_[0] - _px_, _cp_[1] - _py_
            _d_ = math.hypot(_vx_, _vy_)
            if _d_ < 1e-9: return (0.0, 0.0), 0.0
            _S_ = 0.0
            for _g_ in self.node_flows[node]:
                if _g_ == f: continue
                _gx_, _gy_ = self.cps[_g_][0] - _px_, self.cps[_g_][1] - _py_
                if _gx_ == 0.0 and _gy_ == 0.0: continue
                _delta_ = math.atan2(_vx_ * _gy_ - _vy_ * _gx_, _vx_ * _gx_ + _vy_ * _gy_)
                if _delta_ != 0.0:
                    _S_ += math.copysign(1.0, _delta_) * math.exp(-self.k_angres * _delta_ * _delta_)
            # positive S: other flows are counterclockwise -> rotate f clockwise
            _mag_ = _d_ * _S_
            return (_mag_ * (_vy_ / _d_), _mag_ * (-_vx_ / _d_)), _d_

        (_sfx_, _sfy_), _ds_ = _end_force_(self.flow_nodes[f][0])
        (_efx_, _efy_), _de_ = _end_force_(self.flow_nodes[f][1])
        _rx_, _ry_ = _sfx_ + _efx_, _sfy_ + _efy_
        _norm_ = math.hypot(_rx_, _ry_)
        if _norm_ < 1e-12 or _ds_ <= 0.0 or _de_ <= 0.0: return (0.0, 0.0)
        _clamp_ = min(_ds_, _de_) / self.c_angres
        if _norm_ > _clamp_:
            _rx_, _ry_ = _rx_ * _clamp_ / _norm_, _ry_ * _clamp_ / _norm_
        return (_rx_, _ry_)

    #
    # _constrain_() - clamp a control point to the flow-aligned rectangle
    # (3.2.1) and then the canvas rectangle
    #
    def _constrain_(self, f, cp):
        _fx_, _fy_, _tx_, _ty_ = self.flows[f]
        _B_ = self.B[f]
        if _B_ < 1e-9: return (_fx_, _fy_)
        _ex_, _ey_ = (_tx_ - _fx_) / _B_, (_ty_ - _fy_) / _B_
        _rx_, _ry_ = cp[0] - _fx_, cp[1] - _fy_
        _lx_ = _rx_ * _ex_ + _ry_ * _ey_
        _ly_ = _ry_ * _ex_ - _rx_ * _ey_
        _half_ = self.rect_pct * _B_ / 2.0
        if not (0.0 <= _lx_ <= _B_ and -_half_ <= _ly_ <= _half_):
            _lx_, _ly_ = _clipTowardTarget_(_lx_, _ly_, _B_ / 2.0, 0.0, 0.0, _B_, -_half_, _half_)
        _wx_ = _fx_ + _lx_ * _ex_ - _ly_ * _ey_
        _wy_ = _fy_ + _lx_ * _ey_ + _ly_ * _ex_
        if self.canvas is not None:
            _x0_, _y0_, _x1_, _y1_ = self.canvas
            if not (_x0_ <= _wx_ <= _x1_ and _y0_ <= _wy_ <= _y1_):
                _mx_, _my_ = (_fx_ + _tx_) / 2.0, (_fy_ + _ty_) / 2.0
                _wx_, _wy_ = _clipTowardTarget_(_wx_, _wy_, _mx_, _my_, _x0_, _x1_, _y0_, _y1_)
        return (_wx_, _wy_)

    #
    # _insideConstraints_() - boolean mask over candidate control points ``cands``
    # (K, 2): True where each already satisfies both rectangle constraints of
    # _constrain_() (no reconstruction / float round-trip)
    #
    def _insideConstraints_(self, f, cands):
        _fx_, _fy_, _tx_, _ty_ = self.flows[f]
        _B_ = self.B[f]
        if _B_ < 1e-9: return np.zeros(len(cands), dtype=bool)
        _ex_, _ey_ = (_tx_ - _fx_) / _B_, (_ty_ - _fy_) / _B_
        _rx_ = cands[:, 0] - _fx_
        _ry_ = cands[:, 1] - _fy_
        _lx_ = _rx_ * _ex_ + _ry_ * _ey_
        _ly_ = _ry_ * _ex_ - _rx_ * _ey_
        _half_ = self.rect_pct * _B_ / 2.0
        _ok_ = (_lx_ >= 0.0) & (_lx_ <= _B_) & (_ly_ >= -_half_) & (_ly_ <= _half_)
        if self.canvas is not None:
            _x0_, _y0_, _x1_, _y1_ = self.canvas
            _ok_ &= (cands[:, 0] >= _x0_) & (cands[:, 0] <= _x1_) & \
                    (cands[:, 1] >= _y0_) & (cands[:, 1] <= _y1_)
        return _ok_

    #
    # _reduceIntersections_() - flows sharing a node whose curves intersect are
    # amended by the line-intersection construction of Figure 11 (3.2.2)
    # - the intersection tests are batched on the control points at entry; the
    #   sequential amend-as-you-go semantics are preserved by re-testing (only)
    #   the rare pair whose flows were amended earlier in this same pass
    #
    def _reduceIntersections_(self):
        _pairs_ = []
        for _node_ in range(len(self.node_xy)):
            _fs_ = sorted(self.node_flows[_node_])
            for _a_ in range(len(_fs_)):
                for _b_ in range(_a_ + 1, len(_fs_)):
                    _f_, _g_ = _fs_[_a_], _fs_[_b_]
                    if _f_ in self._pinned_ or _g_ in self._pinned_: continue
                    if self.B[_f_] < 1e-9 or self.B[_g_] < 1e-9: continue
                    _pairs_.append((_node_, _f_, _g_))
        if not _pairs_: return

        _batch_ = self._curvesIntersectBatch_(_pairs_)          # bool (Q,), snapshot cps
        _amended_ = set()
        for _idx_, (_node_, _f_, _g_) in enumerate(_pairs_):
            _S_ = self.node_xy[_node_]
            if _f_ in _amended_ or _g_ in _amended_:
                _hit_ = self._curvesIntersectOne_(_f_, _g_, _S_)   # re-test on current cps
            else:
                _hit_ = bool(_batch_[_idx_])
            if not _hit_: continue
            _A_  = self._otherEnd_(_f_, _node_)
            _Bp_ = self._otherEnd_(_g_, _node_)
            _M_, _N_ = self.cps[_f_], self.cps[_g_]
            _Mbar_ = _lineIntersect_(_A_, _M_, _S_, _N_)
            _Nbar_ = _lineIntersect_(_N_, _Bp_, _S_, _M_)
            if _Mbar_ is None or _Nbar_ is None: continue
            self.cps[_f_] = self._constrain_(_f_, _Mbar_)
            self.cps[_g_] = self._constrain_(_g_, _Nbar_)
            _amended_.add(_f_); _amended_.add(_g_)

    def _otherEnd_(self, f, node):
        _s_, _e_ = self.flow_nodes[f]
        return self.node_xy[_e_] if _s_ == node else self.node_xy[_s_]

    #
    # _curvesIntersectOne_() - vectorized single-pair intersection test on the
    # current control points (the amend-loop re-test path).  Exact boolean parity
    # with the scalar _curvesIntersect_, but the 20x20 segment test runs as one
    # NumPy op instead of 400 Python _segIntersect_ calls - a large win on dense
    # graphs where amendments cascade and most pairs take this path.
    #
    def _curvesIntersectOne_(self, f, g, shared_pt):
        _a_, _b_, _c_ = self._is_a_, self._is_b_, self._is_c_       # (21,) each
        _cf_, _cg_ = self.cps[f], self.cps[g]
        _AX_ = _a_ * self._fl_fx_[f] + _b_ * _cf_[0] + _c_ * self._fl_tx_[f]
        _AY_ = _a_ * self._fl_fy_[f] + _b_ * _cf_[1] + _c_ * self._fl_ty_[f]
        _BX_ = _a_ * self._fl_fx_[g] + _b_ * _cg_[0] + _c_ * self._fl_tx_[g]
        _BY_ = _a_ * self._fl_fy_[g] + _b_ * _cg_[1] + _c_ * self._fl_ty_[g]
        return bool(_segAnyBatch_(_AX_[None, :], _AY_[None, :], _BX_[None, :], _BY_[None, :],
                                  np.array([shared_pt[0]]), np.array([shared_pt[1]]))[0])

    #
    # _curvesIntersectBatch_() - vectorized _curvesIntersect_ for many (node,f,g)
    # pairs at once, on the current control points; returns a bool array over the
    # pairs.  Exact boolean parity with the scalar _curvesIntersect_.
    #
    def _curvesIntersectBatch_(self, pairs):
        _Q_ = len(pairs)
        _fa_ = np.fromiter((p[1] for p in pairs), dtype=np.int64, count=_Q_)
        _ga_ = np.fromiter((p[2] for p in pairs), dtype=np.int64, count=_Q_)
        _cpx_ = np.fromiter((c[0] for c in self.cps), dtype=np.float64, count=len(self.cps))
        _cpy_ = np.fromiter((c[1] for c in self.cps), dtype=np.float64, count=len(self.cps))
        _Sx_ = np.fromiter((self.node_xy[p[0]][0] for p in pairs), dtype=np.float64, count=_Q_)
        _Sy_ = np.fromiter((self.node_xy[p[0]][1] for p in pairs), dtype=np.float64, count=_Q_)
        _a_, _b_, _c_ = self._is_a_, self._is_b_, self._is_c_    # (21,) each

        def _samp_(idx):
            _X_ = (_a_[None, :] * self._fl_fx_[idx][:, None] +
                   _b_[None, :] * _cpx_[idx][:, None] +
                   _c_[None, :] * self._fl_tx_[idx][:, None])    # (Q, 21)
            _Y_ = (_a_[None, :] * self._fl_fy_[idx][:, None] +
                   _b_[None, :] * _cpy_[idx][:, None] +
                   _c_[None, :] * self._fl_ty_[idx][:, None])
            return _X_, _Y_

        _AX_, _AY_ = _samp_(_fa_)
        _BX_, _BY_ = _samp_(_ga_)

        _out_ = np.zeros(_Q_, dtype=bool)
        _CH_ = 1024
        for _s_ in range(0, _Q_, _CH_):
            _e_ = min(_s_ + _CH_, _Q_)
            _out_[_s_:_e_] = _segAnyBatch_(_AX_[_s_:_e_], _AY_[_s_:_e_],
                                           _BX_[_s_:_e_], _BY_[_s_:_e_],
                                           _Sx_[_s_:_e_], _Sy_[_s_:_e_])
        return _out_

    def _samples_(self, f, n=24, cp=None):
        _fx_, _fy_, _tx_, _ty_ = self.flows[f]
        _cx_, _cy_ = self.cps[f] if cp is None else cp
        _out_ = []
        for _k_ in range(n + 1):
            _t_ = _k_ / n
            _a_, _b_, _c_ = (1 - _t_) * (1 - _t_), 2 * (1 - _t_) * _t_, _t_ * _t_
            _out_.append((_a_ * _fx_ + _b_ * _cx_ + _c_ * _tx_, _a_ * _fy_ + _b_ * _cy_ + _c_ * _ty_))
        return _out_

    #
    # _sampleArr_() - n+1 curve samples of flow ``f`` at control point ``cp`` as a
    # (n+1, 2) NumPy array (vectorized _samples_)
    #
    def _sampleArr_(self, f, cp=None, n=24):
        _fx_, _fy_, _tx_, _ty_ = self.flows[f]
        _cx_, _cy_ = self.cps[f] if cp is None else cp
        _t_ = np.arange(n + 1) / n
        _a_, _b_, _c_ = (1 - _t_) * (1 - _t_), 2 * (1 - _t_) * _t_, _t_ * _t_
        _x_ = _a_ * _fx_ + _b_ * _cx_ + _c_ * _tx_
        _y_ = _a_ * _fy_ + _b_ * _cy_ + _c_ * _ty_
        return np.stack([_x_, _y_], axis=1)

    #
    # _sampleArrBatch_() - curve samples of flow ``f`` at many control points
    # ``cands`` (K, 2); returns (K, n+1, 2)
    #
    def _sampleArrBatch_(self, f, cands, n=24):
        _fx_, _fy_, _tx_, _ty_ = self.flows[f]
        _t_ = np.arange(n + 1) / n
        _a_, _b_, _c_ = (1 - _t_) * (1 - _t_), 2 * (1 - _t_) * _t_, _t_ * _t_
        _cx_ = cands[:, 0][:, None]
        _cy_ = cands[:, 1][:, None]
        _x_ = _a_[None, :] * _fx_ + _b_[None, :] * _cx_ + _c_[None, :] * _tx_   # (K, n+1)
        _y_ = _a_[None, :] * _fy_ + _b_[None, :] * _cy_ + _c_[None, :] * _ty_
        return np.stack([_x_, _y_], axis=2)

    def _curvesIntersect_(self, f, g, shared_pt, n=20):
        _pa_, _pb_ = self._samples_(f, n), self._samples_(g, n)
        _ax_ = [p[0] for p in _pa_]; _ay_ = [p[1] for p in _pa_]
        _bx_ = [p[0] for p in _pb_]; _by_ = [p[1] for p in _pb_]
        if min(_ax_) > max(_bx_) or max(_ax_) < min(_bx_) or \
           min(_ay_) > max(_by_) or max(_ay_) < min(_by_): return False
        for _i_ in range(n):
            for _j_ in range(n):
                _pt_ = _segIntersect_(_pa_[_i_], _pa_[_i_ + 1], _pb_[_j_], _pb_[_j_ + 1])
                if _pt_ is not None and math.hypot(_pt_[0] - shared_pt[0], _pt_[1] - shared_pt[1]) > 1.0:
                    return True
        return False

    #
    # _overlapsObstacle_() / _moveOffObstacles_() - section 3.2.3; obstacles are
    # the unconnected nodes plus, with arrows=True, other flows' arrowheads
    #
    def _obstacles_(self, f):
        _s_, _e_ = self.flow_nodes[f]
        return [self.node_xy[n] for n in range(len(self.node_xy)) if n != _s_ and n != _e_]

    #
    # _arrowObstacles_() - arrival points of other flows' arrowheads, from each
    # flow's current end tangent; flows arriving at one of f's own endpoints are
    # skipped so clearance near shared nodes stays satisfiable
    #
    def _arrowObstacles_(self, f):
        if not self.arrows: return []
        _own_ = set(self.flow_nodes[f])
        _out_ = []
        for _g_ in self.active:
            if _g_ == f or self.flow_nodes[_g_][1] in _own_: continue
            _gfx_, _gfy_, _gtx_, _gty_ = self.flows[_g_]
            _cx_, _cy_ = self.cps[_g_]
            _dx_, _dy_ = _gtx_ - _cx_, _gty_ - _cy_       # quadratic end tangent
            _m_ = math.hypot(_dx_, _dy_)
            if _m_ < 1e-9:
                _dx_, _dy_, _m_ = _gtx_ - _gfx_, _gty_ - _gfy_, self.B[_g_]
                if _m_ < 1e-9: continue
            _out_.append((_gtx_ - _dx_ / _m_ * self.node_radius,
                          _gty_ - _dy_ / _m_ * self.node_radius))
        return _out_

    def _clearOfObstacles_(self, f, cp):
        _pts_ = None
        _obs_ = self._obstacles_(f)
        if len(_obs_) > 0:
            _min_ = self.node_radius + self.min_obstacle_dist
            _obs_ = np.asarray(_obs_, dtype=np.float64)
            _pts_ = self._sampleArr_(f, cp=cp)                 # (S, 2)
            _dd_ = np.hypot(_pts_[:, 0][:, None] - _obs_[None, :, 0],
                            _pts_[:, 1][:, None] - _obs_[None, :, 1])
            if _dd_.min() < _min_: return False
        _arr_ = self._arrowObstacles_(f)
        if len(_arr_) > 0:
            _amin_ = self.arrow_radius + self.min_obstacle_dist
            _arr_ = np.asarray(_arr_, dtype=np.float64)
            if _pts_ is None: _pts_ = self._sampleArr_(f, cp=cp)
            _dd_ = np.hypot(_pts_[:, 0][:, None] - _arr_[None, :, 0],
                            _pts_[:, 1][:, None] - _arr_[None, :, 1])
            if _dd_.min() < _amin_: return False
        return True

    def _overlapsObstacle_(self, f):
        return not self._clearOfObstacles_(f, self.cps[f])

    def _moveOffObstacles_(self, f):
        _fx_, _fy_, _tx_, _ty_ = self.flows[f]
        _cx_, _cy_ = self.cps[f]
        _B_ = self.B[f]
        # The spiral candidate sequence is independent of the test outcomes, so
        # take the precomputed offsets (same bound as the paper's loop: candidate
        # k is included while its r <= B), then find the first candidate that is
        # inside the constraints and clear of obstacles.
        _count_ = int(np.searchsorted(self._spiral_r_, _B_, side='right'))
        if _count_ == 0: return False
        _cands_ = np.empty((_count_, 2))
        _cands_[:, 0] = _cx_ + self._spiral_dx_[:_count_]
        _cands_[:, 1] = _cy_ + self._spiral_dy_[:_count_]

        _inside_ = self._insideConstraints_(f, _cands_)
        _idxs_ = np.nonzero(_inside_)[0]
        if len(_idxs_) == 0: return False
        _incand_ = _cands_[_idxs_]

        # Restrict obstacles to those that could possibly conflict: every curve
        # sample of any inside candidate lies inside the hull of {fm, to, cp} and
        # so within the bbox of {fm, to} + all inside candidate points; an
        # obstacle farther than its clearance from that bbox can never be within
        # clearance of any sample (exact prefilter, not an approximation).
        _minO_ = self.node_radius  + self.min_obstacle_dist
        _minA_ = self.arrow_radius + self.min_obstacle_dist
        _lox_ = min(_fx_, _tx_, float(_incand_[:, 0].min()))
        _hix_ = max(_fx_, _tx_, float(_incand_[:, 0].max()))
        _loy_ = min(_fy_, _ty_, float(_incand_[:, 1].min()))
        _hiy_ = max(_fy_, _ty_, float(_incand_[:, 1].max()))
        _obs_ = _obsInBBox_(self._obstacles_(f),      _lox_, _hix_, _loy_, _hiy_, _minO_)
        _arr_ = _obsInBBox_(self._arrowObstacles_(f), _lox_, _hix_, _loy_, _hiy_, _minA_)

        # Scan inside candidates (spiral order) in geometrically growing chunks so
        # the common early success tests only a handful, while a flow that cannot
        # escape still vectorizes over the whole candidate set.
        _pos_, _chunk_ = 0, 16
        while _pos_ < len(_incand_):
            _blk_ = _incand_[_pos_:_pos_ + _chunk_]
            _samp_ = self._sampleArrBatch_(f, _blk_)                 # (k, S, 2)
            _clear_ = np.ones(len(_blk_), dtype=bool)
            if len(_obs_) > 0:
                _dd_ = np.hypot(_samp_[:, :, 0][:, :, None] - _obs_[None, None, :, 0],
                                _samp_[:, :, 1][:, :, None] - _obs_[None, None, :, 1])
                _clear_ &= _dd_.min(axis=(1, 2)) >= _minO_
            if len(_arr_) > 0:
                _dd_ = np.hypot(_samp_[:, :, 0][:, :, None] - _arr_[None, None, :, 0],
                                _samp_[:, :, 1][:, :, None] - _arr_[None, None, :, 1])
                _clear_ &= _dd_.min(axis=(1, 2)) >= _minA_
            _hit_ = np.nonzero(_clear_)[0]
            if len(_hit_) > 0:
                _sel_ = int(_idxs_[_pos_ + _hit_[0]])
                self.cps[f] = (float(_cands_[_sel_, 0]), float(_cands_[_sel_, 1]))
                self._pinned_.add(f)
                return True
            _pos_ += _chunk_
            _chunk_ = min(_chunk_ * 2, 512)
        return False


#
# _obsInBBox_() - obstacles (list of (x,y)) within ``pad`` of the axis-aligned box
# [lox,hix] x [loy,hiy], as an (O,2) NumPy array.  Obstacles outside cannot be
# within ``pad`` of any point inside the box, so dropping them is exact.
#
def _obsInBBox_(obs, lox, hix, loy, hiy, pad):
    if not obs: return np.zeros((0, 2))
    _o_ = np.asarray(obs, dtype=np.float64)
    _keep_ = (_o_[:, 0] >= lox - pad) & (_o_[:, 0] <= hix + pad) & \
             (_o_[:, 1] >= loy - pad) & (_o_[:, 1] <= hiy + pad)
    return _o_[_keep_]


#
# _segAnyBatch_() - True per pair iff any segment of curve A (samples AX,AY) meets
# any segment of curve B (BX,BY) at a point > 1px from the shared node (Sx,Sy).
# AX..BY are (Q, 21); Sx,Sy are (Q,).  Exact boolean parity with the scalar
# _segIntersect_ / _curvesIntersect_ pair loop (same arithmetic, same tie tests).
#
def _segAnyBatch_(AX, AY, BX, BY, Sx, Sy):
    # bounding-box prefilter (identical strict comparisons to the scalar path)
    _minax_, _maxax_ = AX.min(axis=1), AX.max(axis=1)
    _minay_, _maxay_ = AY.min(axis=1), AY.max(axis=1)
    _minbx_, _maxbx_ = BX.min(axis=1), BX.max(axis=1)
    _minby_, _maxby_ = BY.min(axis=1), BY.max(axis=1)
    _bbox_ = ~((_minax_ > _maxbx_) | (_maxax_ < _minbx_) |
               (_minay_ > _maxby_) | (_maxay_ < _minby_))

    _ax0_, _ay0_ = AX[:, :-1], AY[:, :-1]                        # (Q, 20)
    _d1x_, _d1y_ = AX[:, 1:] - AX[:, :-1], AY[:, 1:] - AY[:, :-1]
    _bx0_, _by0_ = BX[:, :-1], BY[:, :-1]
    _d2x_, _d2y_ = BX[:, 1:] - BX[:, :-1], BY[:, 1:] - BY[:, :-1]

    # (Q, 20i, 20j)
    _den_ = _d1x_[:, :, None] * _d2y_[:, None, :] - _d1y_[:, :, None] * _d2x_[:, None, :]
    _cx_  = _bx0_[:, None, :] - _ax0_[:, :, None]
    _cy_  = _by0_[:, None, :] - _ay0_[:, :, None]
    with np.errstate(divide='ignore', invalid='ignore'):
        _t_ = (_cx_ * _d2y_[:, None, :] - _cy_ * _d2x_[:, None, :]) / _den_
        _u_ = (_cx_ * _d1y_[:, :, None] - _cy_ * _d1x_[:, :, None]) / _den_
    _valid_ = (np.abs(_den_) >= 1e-12) & (_t_ >= 0.0) & (_t_ <= 1.0) & (_u_ >= 0.0) & (_u_ <= 1.0)
    _ix_ = _ax0_[:, :, None] + _t_ * _d1x_[:, :, None]
    _iy_ = _ay0_[:, :, None] + _t_ * _d1y_[:, :, None]
    _far_ = np.hypot(_ix_ - Sx[:, None, None], _iy_ - Sy[:, None, None]) > 1.0
    return _bbox_ & (_valid_ & _far_).any(axis=(1, 2))


#
# _clipTowardTarget_() - first point of the segment (px,py) -> (tx,ty) inside the
# axis-aligned rectangle [x0,x1] x [y0,y1] (slab clipping); target must be inside
#
def _clipTowardTarget_(px, py, tx, ty, x0, x1, y0, y1):
    _dx_, _dy_ = tx - px, ty - py
    _t0_, _t1_ = 0.0, 1.0
    for _p_, _d_, _lo_, _hi_ in ((px, _dx_, x0, x1), (py, _dy_, y0, y1)):
        if abs(_d_) < 1e-12:
            if _p_ < _lo_ or _p_ > _hi_: return (tx, ty)
        else:
            _ta_, _tb_ = (_lo_ - _p_) / _d_, (_hi_ - _p_) / _d_
            if _ta_ > _tb_: _ta_, _tb_ = _tb_, _ta_
            _t0_, _t1_ = max(_t0_, _ta_), min(_t1_, _tb_)
    if _t0_ > _t1_: return (tx, ty)
    return (px + _t0_ * _dx_, py + _t0_ * _dy_)


#
# _lineIntersect_() - intersection of the infinite lines p1-p2 and p3-p4
#
def _lineIntersect_(p1, p2, p3, p4):
    _d1x_, _d1y_ = p2[0] - p1[0], p2[1] - p1[1]
    _d2x_, _d2y_ = p4[0] - p3[0], p4[1] - p3[1]
    _den_ = _d1x_ * _d2y_ - _d1y_ * _d2x_
    if abs(_den_) < 1e-9: return None
    _t_ = ((p3[0] - p1[0]) * _d2y_ - (p3[1] - p1[1]) * _d2x_) / _den_
    return (p1[0] + _t_ * _d1x_, p1[1] + _t_ * _d1y_)


#
# _segIntersect_() - intersection point of segments a-b and c-d (None if disjoint)
#
def _segIntersect_(a, b, c, d):
    _d1x_, _d1y_ = b[0] - a[0], b[1] - a[1]
    _d2x_, _d2y_ = d[0] - c[0], d[1] - c[1]
    _den_ = _d1x_ * _d2y_ - _d1y_ * _d2x_
    if abs(_den_) < 1e-12: return None
    _t_ = ((c[0] - a[0]) * _d2y_ - (c[1] - a[1]) * _d2x_) / _den_
    _u_ = ((c[0] - a[0]) * _d1y_ - (c[1] - a[1]) * _d1x_) / _den_
    if 0.0 <= _t_ <= 1.0 and 0.0 <= _u_ <= 1.0:
        return (a[0] + _t_ * _d1x_, a[1] + _t_ * _d1y_)
    return None
