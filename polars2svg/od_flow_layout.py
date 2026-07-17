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

import math
import polars as pl


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
    maps at the paper's scale (up to ~100 aggregated flows).  The algorithm is
    deterministic.

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

        self._pinned_        = set()  # flows moved off obstacles; immovable afterwards (3.2.3)
        self._connected_df_  = pl.DataFrame({
            '__f__': [i for i in range(len(self.flows)) for _ in range(2)],
            '__n__': [n for pair in self.flow_nodes for n in pair],
        }, schema={'__f__': pl.Int32, '__n__': pl.Int32})
        self._nodes_df_ = pl.DataFrame({
            '__n__':  list(range(len(self.node_xy))),
            '__nx__': [p[0] for p in self.node_xy],
            '__ny__': [p[1] for p in self.node_xy],
        }, schema={'__n__': pl.Int32, '__nx__': pl.Float64, '__ny__': pl.Float64})

        if len(self.active) > 1 or (len(self.active) == 1 and len(self.node_xy) > 2):
            self._iterate_()

    def results(self) -> list:
        '''Control points, one ``(cx, cy)`` tuple per input flow (input order).'''
        return list(self.cps)

    #
    # _iterate_() - the main loop (pseudo-code, paper pp. 4-5)
    #
    def _iterate_(self):
        _j_ = 0
        for _i_ in range(self.iterations):
            _w_ = 1.0 - _i_ / self.iterations
            _pts_               = self._pointsDF_()
            _f_flows_, _periph_ = self._flowForces_(_pts_)
            _f_nodes_           = self._nodeForces_(_pts_)
            _new_cps_ = {}
            for _f_ in self.active:
                if _f_ in self._pinned_: continue
                _ffx_, _ffy_          = _f_flows_.get(_f_, (0.0, 0.0))
                _fnx_, _fny_          = _f_nodes_.get(_f_, (0.0, 0.0))
                _atx_, _aty_          = self._antiTorsionForce_(_f_)
                _spx_, _spy_          = self._springForce_(_f_, _periph_.get(_f_, 0.0))
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
    # _pointsDF_() - evenly spaced sample points along every active flow
    # - t at (k + 0.5)/samples so shared endpoints never yield coincident samples
    #
    def _pointsDF_(self):
        _fs_, _ps_, _xs_, _ys_ = [], [], [], []
        _P_ = self.samples_per_flow
        for _f_ in self.active:
            _fx_, _fy_, _tx_, _ty_ = self.flows[_f_]
            _cx_, _cy_             = self.cps[_f_]
            for _k_ in range(_P_):
                _t_ = (_k_ + 0.5) / _P_
                _a_, _b_, _c_ = (1 - _t_) * (1 - _t_), 2 * (1 - _t_) * _t_, _t_ * _t_
                _fs_.append(_f_), _ps_.append(_k_)
                _xs_.append(_a_ * _fx_ + _b_ * _cx_ + _c_ * _tx_)
                _ys_.append(_a_ * _fy_ + _b_ * _cy_ + _c_ * _ty_)
        return pl.DataFrame({'__f__': _fs_, '__p__': _ps_, '__px__': _xs_, '__py__': _ys_},
                            schema={'__f__': pl.Int32, '__p__': pl.Int32,
                                    '__px__': pl.Float64, '__py__': pl.Float64})

    #
    # _flowForces_() - flows-against-flow (3.1.1) + peripherality ratio (3.1.4)
    # - Shepard inverse-distance weighting of point-to-point displacement vectors
    #
    def _flowForces_(self, _pts_):
        if len(_pts_) == 0: return {}, {}
        _per_pt_ = (
            _pts_.join(_pts_, how='cross')
                 .filter(pl.col('__f__') != pl.col('__f___right'))
                 .with_columns((pl.col('__px__') - pl.col('__px___right')).alias('__dx__'),
                               (pl.col('__py__') - pl.col('__py___right')).alias('__dy__'))
                 .with_columns((pl.col('__dx__') ** 2 + pl.col('__dy__') ** 2).sqrt().alias('__d__'))
                 .with_columns(pl.when(pl.col('__d__') < 1e-4).then(pl.lit(1e-4))
                                 .otherwise(pl.col('__d__')).alias('__d__'))
                 .with_columns((1.0 / pl.col('__d__') ** self.alpha).alias('__wt__'))
                 # deterministic float summation order (group_by alone reorders rows run-to-run)
                 .sort(['__f__', '__p__', '__f___right', '__p___right'])
                 .group_by(['__f__', '__p__'], maintain_order=True)
                 .agg(((pl.col('__dx__') * pl.col('__wt__')).sum() / pl.col('__wt__').sum()).alias('__fpx__'),
                      ((pl.col('__dy__') * pl.col('__wt__')).sum() / pl.col('__wt__').sum()).alias('__fpy__'))
                 .with_columns((pl.col('__fpx__') ** 2 + pl.col('__fpy__') ** 2).sqrt().alias('__fpm__'))
        )
        if len(_per_pt_) == 0: return {}, {}
        _per_flow_ = (
            _per_pt_.group_by('__f__', maintain_order=True)
                    .agg(pl.col('__fpx__').mean().alias('__mfx__'),
                         pl.col('__fpy__').mean().alias('__mfy__'),
                         pl.col('__fpx__').sum().alias('__sfx__'),
                         pl.col('__fpy__').sum().alias('__sfy__'),
                         pl.col('__fpm__').sum().alias('__sfm__'))
        )
        _forces_, _periph_ = {}, {}
        for _row_ in _per_flow_.iter_rows(named=True):
            _forces_[_row_['__f__']] = (_row_['__mfx__'], _row_['__mfy__'])
            _sfm_ = _row_['__sfm__']
            _periph_[_row_['__f__']] = (math.hypot(_row_['__sfx__'], _row_['__sfy__']) / _sfm_) if _sfm_ > 0 else 0.0
        return _forces_, _periph_

    #
    # _nodeForces_() - nodes-against-flow (3.1.2): each unconnected node repels
    # via the vector to the closest sampled point on the flow
    #
    def _nodeForces_(self, _pts_):
        if len(_pts_) == 0 or len(self._nodes_df_) == 0: return {}
        _per_node_ = (
            _pts_.join(self._nodes_df_, how='cross')
                 .join(self._connected_df_, on=['__f__', '__n__'], how='anti')
                 .with_columns((pl.col('__px__') - pl.col('__nx__')).alias('__dx__'),
                               (pl.col('__py__') - pl.col('__ny__')).alias('__dy__'))
                 .with_columns((pl.col('__dx__') ** 2 + pl.col('__dy__') ** 2).sqrt().alias('__d__'))
                 .with_columns(pl.when(pl.col('__d__') < 1e-4).then(pl.lit(1e-4))
                                 .otherwise(pl.col('__d__')).alias('__d__'))
                 # deterministic ordering for tie-breaks and float summation
                 .sort(['__f__', '__n__', '__p__'])
                 .group_by(['__f__', '__n__'], maintain_order=True)
                 .agg(pl.col('__dx__').sort_by('__d__').first(),
                      pl.col('__dy__').sort_by('__d__').first(),
                      pl.col('__d__').min())
                 .with_columns((1.0 / pl.col('__d__') ** self.beta).alias('__wt__'))
        )
        if len(_per_node_) == 0: return {}
        _per_flow_ = (
            _per_node_.group_by('__f__', maintain_order=True)
                      .agg(((pl.col('__dx__') * pl.col('__wt__')).sum() / pl.col('__wt__').sum()).alias('__fx__'),
                           ((pl.col('__dy__') * pl.col('__wt__')).sum() / pl.col('__wt__').sum()).alias('__fy__'))
        )
        return {_r_['__f__']: (_r_['__fx__'], _r_['__fy__']) for _r_ in _per_flow_.iter_rows(named=True)}

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
    # _insideConstraints_() - True iff cp already satisfies both rectangle
    # constraints of _constrain_() (no reconstruction / float round-trip)
    #
    def _insideConstraints_(self, f, cp):
        _fx_, _fy_, _tx_, _ty_ = self.flows[f]
        _B_ = self.B[f]
        if _B_ < 1e-9: return False
        _ex_, _ey_ = (_tx_ - _fx_) / _B_, (_ty_ - _fy_) / _B_
        _rx_, _ry_ = cp[0] - _fx_, cp[1] - _fy_
        _lx_ = _rx_ * _ex_ + _ry_ * _ey_
        _ly_ = _ry_ * _ex_ - _rx_ * _ey_
        _half_ = self.rect_pct * _B_ / 2.0
        if not (0.0 <= _lx_ <= _B_ and -_half_ <= _ly_ <= _half_): return False
        if self.canvas is not None:
            _x0_, _y0_, _x1_, _y1_ = self.canvas
            if not (_x0_ <= cp[0] <= _x1_ and _y0_ <= cp[1] <= _y1_): return False
        return True

    #
    # _reduceIntersections_() - flows sharing a node whose curves intersect are
    # amended by the line-intersection construction of Figure 11 (3.2.2)
    #
    def _reduceIntersections_(self):
        for _node_ in range(len(self.node_xy)):
            _fs_ = sorted(self.node_flows[_node_])
            for _a_ in range(len(_fs_)):
                for _b_ in range(_a_ + 1, len(_fs_)):
                    _f_, _g_ = _fs_[_a_], _fs_[_b_]
                    if _f_ in self._pinned_ or _g_ in self._pinned_: continue
                    if self.B[_f_] < 1e-9 or self.B[_g_] < 1e-9: continue
                    _S_ = self.node_xy[_node_]
                    if not self._curvesIntersect_(_f_, _g_, _S_): continue
                    _A_ = self._otherEnd_(_f_, _node_)
                    _Bp_ = self._otherEnd_(_g_, _node_)
                    _M_, _N_ = self.cps[_f_], self.cps[_g_]
                    _Mbar_ = _lineIntersect_(_A_, _M_, _S_, _N_)
                    _Nbar_ = _lineIntersect_(_N_, _Bp_, _S_, _M_)
                    if _Mbar_ is None or _Nbar_ is None: continue
                    self.cps[_f_] = self._constrain_(_f_, _Mbar_)
                    self.cps[_g_] = self._constrain_(_g_, _Nbar_)

    def _otherEnd_(self, f, node):
        _s_, _e_ = self.flow_nodes[f]
        return self.node_xy[_e_] if _s_ == node else self.node_xy[_s_]

    def _samples_(self, f, n=24, cp=None):
        _fx_, _fy_, _tx_, _ty_ = self.flows[f]
        _cx_, _cy_ = self.cps[f] if cp is None else cp
        _out_ = []
        for _k_ in range(n + 1):
            _t_ = _k_ / n
            _a_, _b_, _c_ = (1 - _t_) * (1 - _t_), 2 * (1 - _t_) * _t_, _t_ * _t_
            _out_.append((_a_ * _fx_ + _b_ * _cx_ + _c_ * _tx_, _a_ * _fy_ + _b_ * _cy_ + _c_ * _ty_))
        return _out_

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
            _pts_ = self._samples_(f, cp=cp)
            for _px_, _py_ in _pts_:
                for _ox_, _oy_ in _obs_:
                    if math.hypot(_px_ - _ox_, _py_ - _oy_) < _min_: return False
        _arr_ = self._arrowObstacles_(f)
        if len(_arr_) > 0:
            _amin_ = self.arrow_radius + self.min_obstacle_dist
            if _pts_ is None: _pts_ = self._samples_(f, cp=cp)
            for _px_, _py_ in _pts_:
                for _ox_, _oy_ in _arr_:
                    if math.hypot(_px_ - _ox_, _py_ - _oy_) < _amin_: return False
        return True

    def _overlapsObstacle_(self, f):
        return not self._clearOfObstacles_(f, self.cps[f])

    def _moveOffObstacles_(self, f):
        _fx_, _fy_, _tx_, _ty_ = self.flows[f]
        _B_ = self.B[f]
        _cx_, _cy_ = self.cps[f]
        _spacing_ = self.min_obstacle_dist
        _theta_, _r_, _tries_ = 0.0, 0.0, 0
        while _r_ <= _B_ and _tries_ < 2000:
            _cand_ = (_cx_ + _r_ * math.cos(_theta_), _cy_ + _r_ * math.sin(_theta_))
            _tries_ += 1
            _theta_ += _spacing_ / max(_r_, _spacing_)
            _r_ = _spacing_ * _theta_ / (2.0 * math.pi)
            # candidates outside the constraint rectangles are not considered
            if not self._insideConstraints_(f, _cand_): continue
            if self._clearOfObstacles_(f, _cand_):
                self.cps[f] = _cand_
                self._pinned_.add(f)
                return True
        return False


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
