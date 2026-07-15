# Vendored from racetrack_svg_framework/rtsvg/convey_proximity_layout.py
# Original author: David Trimm — Apache License 2.0
# Removed: import rtsvg (unused), TypeGuard (unused), time (unused),
#          svgOfVertexAdditions() (requires external RACETrack instance)
#
# Implementation of the following:
#
# Drawing Graphs to Convey Proximity: An Incremental Arrangement Method
# J.D. Cohen
# ACM Transactions on Computer-Human Interaction, Vol. 4, No. 3, September 1997, Pages 197–229.

import polars as pl
import networkx as nx
from math import ceil
import random
import numpy as np


class ConveyProximityLayout(object):
    '''
    Proximity-preserving graph layout (Cohen, ACM ToCHI 1997, Table V).

    A multi-trial incremental arrangement that places nodes so screen distance
    reflects graph distance. Satisfies the ``LayoutAlgorithm`` protocol: call
    ``.results()`` for a ``{node: (x, y)}`` dict to feed ``linkp``'s ``pos=``.

    Parameters
    ----------
    g_connected : networkx.Graph
        A connected graph (run one instance per connected component).
    use_resistive_distances : bool
        Use resistive (effective-resistance) distances rather than shortest-path.
    k : float
        Stress-family exponent (the paper's ``S_k``): 0 = absolute stress,
        1 = semiproportional, 2 = proportional.
    iterations_min : int
        Minimum gradient-descent iterations per stage arrangement.
    iterations_multiplier : int
        Iterations per stage scale as ``iterations_multiplier * |H|``
        (floored at ``iterations_min``).
    distances : dict, optional
        Precomputed pairwise distances (else computed internally).
    cleanup : bool
        Apply the paper's cleanup (p. 225): floor the target distances up front
        and finish with one proportional-stress (k=2) iteration so intimate
        vertices are not drawn on top of one another.
    cleanup_min_ratio : float
        Target-distance floor used by ``cleanup``, as a fraction of the mean
        pairwise target distance.

    Example::

        pos = ConveyProximityLayout(g).results()
        p2s.linkp(df, [('src', 'dst')], pos)
    '''
    #
    # Table V of paper (algorithm that includes multiple trials)
    #
    def __init__(self, g_connected, use_resistive_distances=True, k=0.0, iterations_min=32, iterations_multiplier=2, distances=None, cleanup=False, cleanup_min_ratio=0.05) -> None:
        self.g_connected             = g_connected
        self.k                       = k
        self.V                       = set(self.g_connected.nodes)
        self.use_resistive_distances = use_resistive_distances
        self.iterations_min          = iterations_min
        self.iterations_multiplier   = iterations_multiplier

        self.distances = self.__getTargetDistances__(g_connected) if distances is None else distances
        if cleanup:  # paper p.225 cleanup, part 1: floor the target distances before beginning
            _all_ = [_d_ for _v_ in self.distances for _u_, _d_ in self.distances[_v_].items() if _u_ != _v_]
            _floor_ = cleanup_min_ratio * sum(_all_) / len(_all_)
            self.distances = {_v_: {_u_: (max(_d_, _floor_) if _u_ != _v_ else _d_) for _u_, _d_ in self.distances[_v_].items()} for _v_ in self.distances}

        pos            = {}
        Q              = self.__orderVertices__(self.g_connected, self.distances)
        H              = set()
        arrange_round  = 0
        stress_dfs     = []
        best_trials    = []
        vertices_added = []
        i_global       = None
        trial_global   = 0

        _to_randomize_ = set()
        vertices_added.append(set())
        for i in range(self.__numberToAddThisTime__(len(H), len(self.V))):
            v = Q.pop()
            H.add(v), _to_randomize_.add(v), vertices_added[-1].add(v)
        _best_stress_, _best_pos_, _best_trial_ = None, None, None
        for _trial_ in range(self.__numberOfTrialsThisTime__(H)):
            for _vertex_ in _to_randomize_: pos[_vertex_] = (random.random(), random.random())  # nosec B311 - non-cryptographic layout jitter
            _stress_lu_  = {'stress':[], 'i':[]}
            pos          = self.__arrangeDirect__(H, pos, _stress_lu_)
            _end_stress_ = _stress_lu_['stress'][-1]
            if _best_stress_ is None or _end_stress_ < _best_stress_: _best_stress_, _best_pos_, _best_trial_ = _end_stress_, pos, _trial_
            stress_dfs.append(pl.DataFrame(_stress_lu_).with_columns(pl.lit(_trial_).alias('trial'), pl.lit(arrange_round).alias('round'), pl.lit(trial_global).alias('trial_global'), pl.col('i').alias('i_global')))
            trial_global += 1
            i_global      = _stress_lu_['i'][-1] if i_global is None else max(_stress_lu_['i'][-1], i_global)
        arrange_round += 1
        best_trials.append(_best_trial_)
        pos = _best_pos_

        while H != self.V:
            _number_to_add_ = self.__numberToAddThisTime__(len(H), len(self.V))
            _to_randomize_  = []
            vertices_added.append(set())
            for i in range(_number_to_add_):
                v      = Q.pop()
                h1, h2 = self.__closestMembers__(H, v)  # H grows during the stage (Table V) -- earlier same-stage vertices can serve as neighbors
                _to_randomize_.append((v, h1, h2))
                H.add(v)
                vertices_added[-1].add(v)
            _best_stress_, _best_pos_, i_global_next, _best_trial_ = None, None, None, None
            for _trial_ in range(self.__numberOfTrialsThisTime__(H)):
                for _tuple_ in _to_randomize_:  # in addition order, so same-stage neighbors are placed before they are used
                    v, h1, h2 = _tuple_
                    pos[v]    = self.__neighborlyLocation__(v, h1, h2, pos)
                _stress_lu_   = {'stress':[], 'i':[]}
                pos = self.__arrangeDirect__(H, pos, _stress_lu_)
                _end_stress_  = _stress_lu_['stress'][-1]
                if _best_stress_ is None or _end_stress_ < _best_stress_: _best_stress_, _best_pos_, _best_trial_ = _end_stress_, pos, _trial_
                stress_dfs.append(pl.DataFrame(_stress_lu_).with_columns(pl.lit(_trial_).alias('trial'), pl.lit(arrange_round).alias('round'), pl.lit(trial_global).alias('trial_global'), (pl.col('i')+i_global).alias('i_global')))
                trial_global  += 1
                i_global_next  = (_stress_lu_['i'][-1] + i_global) if i_global_next is None else max(_stress_lu_['i'][-1] + i_global, i_global_next)
            arrange_round += 1
            best_trials.append(_best_trial_)
            pos, i_global = _best_pos_, i_global_next

        if cleanup:  # paper p.225 cleanup, part 2: one final proportional-stress (k=2) iteration to separate vertices that are too close
            _saved_ = (self.k, self.iterations_min, self.iterations_multiplier)
            self.k, self.iterations_min, self.iterations_multiplier = 2.0, 1, 0
            pos = self.__arrangeDirect__(self.V, pos, {'stress':[], 'i':[]})
            self.k, self.iterations_min, self.iterations_multiplier = _saved_

        self.pos, self.stress_df, self.vertices_added, self.best_trials = pos, pl.concat(stress_dfs), vertices_added, best_trials

        _lu_ = {'round':[],'trial':[], 'best_flag':[]}
        for i in range(len(best_trials)): _lu_['round'].append(i), _lu_['trial'].append(best_trials[i]), _lu_['best_flag'].append(True)
        df_bests = pl.DataFrame(_lu_)
        self.stress_df = pl.concat([self.stress_df.join(df_bests, on=['round','trial']),
                                    self.stress_df.join(df_bests, on=['round','trial'], how='anti').with_columns(pl.lit(False).alias('best_flag'))])

    def __numberOfTrialsThisTime__(self, H):
        _h_len_  = len(H)
        if _h_len_ < 1:  _h_len_ = 1
        _times_  = len(self.V) // _h_len_
        if _times_ < 1:  _times_ = 1
        if _times_ > 10: _times_ = 10
        return _times_

    def results(self) -> dict: return self.pos

    def stress(self): return self.stress_df['stress'][-1]

    def __getTargetDistances__(self, _g_):
        if self.use_resistive_distances:
            N = list(_g_.nodes())
            n = len(N)
            G = np.zeros((n, n), dtype=float)
            for i in range(n):
                i_n = N[i]
                for j in range(n):
                    j_n = N[j]
                    if j_n not in _g_[i_n]: continue
                    G[i][j] = 1.0 if 'weight' not in _g_[i_n][j_n] else _g_[i_n][j_n]['weight']
            for i in range(n):
                _sum_ = 0.0
                for j in range(n):
                    if i == j: continue
                    _sum_ += -G[i][j]
                G[i][i] = _sum_
            _inv_ = np.linalg.pinv(G)
            _dist_ = {}
            for i in range(n):
                _dist_[N[i]] = {}
                for j in range(n):
                    if i == j: continue
                    _dist_[N[i]][N[j]] = abs(_inv_[i][i] + _inv_[j][j] - 2.0 * _inv_[i][j])
            return _dist_
        return dict(nx.all_pairs_dijkstra_path_length(_g_))

    def __everyNthMember__(self, Q, n): return [Q[i] for i in range(0, len(Q), n)]
    def __disperseTheseVertices__(self, Q, increment_ratio=1):
        if len(Q) > increment_ratio + 1:
            F = self.__everyNthMember__(Q, 1+increment_ratio)
            B = [item for item in Q if item not in F]
            F = self.__disperseTheseVertices__(F)
            F.extend(B)
            return F
        return Q
    def __orderVertices__(self, _g_, _dist_):
        Q = [n for n in nx.traversal.dfs_preorder_nodes(_g_)]
        _list_ = self.__disperseTheseVertices__(Q)
        _list_.reverse()
        return _list_

    def __numberToAddThisTime__(self, _prev_, _final_, increment_ratio=1, increment_minimum=10):
        if _prev_ > 0: _inc_ = _prev_ * increment_ratio
        else:
            _inc_ = _final_
            while _inc_ > increment_minimum: _inc_ = ceil(_inc_ / (1 + increment_ratio))
        if _inc_ > _final_ - _prev_: _inc_ = _final_ - _prev_
        return _inc_

    def __closestMembers__(self, _H_, _v_):
        _h1_, _h1_d_, _h2_, _h2_d_ = None, None, None, None
        for _k_ in _H_:
            if _k_ == _v_: continue
            _d_ = self.distances[_v_][_k_]
            if   _h1_d_ is None: _h1_, _h1_d_ = _k_, _d_
            elif _h2_d_ is None: _h2_, _h2_d_ = _k_, _d_
            elif _d_ < _h1_d_ or _d_ < _h2_d_:
                if   _d_ < _h1_d_ and _d_ < _h2_d_:
                    if _h1_d_ < _h2_d_: _h2_, _h2_d_ = _k_, _d_
                    else:               _h1_, _h1_d_ = _k_, _d_
                elif _d_ < _h1_d_:      _h1_, _h1_d_ = _k_, _d_
                else:                   _h2_, _h2_d_ = _k_, _d_
        if _h2_d_ is not None and _h2_d_ < _h1_d_: _h1_, _h1_d_, _h2_, _h2_d_ = _h2_, _h2_d_, _h1_, _h1_d_  # h1 must be the closest (paper Fig. 3: j = nearest, k = next nearest)
        return _h1_, _h2_

    def __neighborlyLocation__(self, i, j, k, _pos_):
        t_ik, t_ij, t_jk = self.distances[i][k], self.distances[i][j], self.distances[j][k]
        _expr_   = (1.0 / (2 * t_jk**2)) * (t_ik**2 - t_ij**2 - t_jk**2)
        _gamma_  = min(_expr_, 0.5)
        x_j, y_j = _pos_[j]
        x_k, y_k = _pos_[k]
        e_x, e_y = (random.random()-0.5) * 0.1, (random.random()-0.5) * 0.1  # nosec B311 - non-cryptographic layout jitter
        x, y     = x_j + _gamma_ * (x_j - x_k) + e_x, y_j + _gamma_ * (y_j - y_k) + e_y
        return x, y

    def __arrangeDirect__(self, _nodes_, _pos_, _stress_lu_):
        _lu_pos_  = {'node':[], 'x':[], 'y':[], 'mu':[]}
        _lu_dist_ = {'fm':[],'to':[], 't':[]}
        for _node_ in _nodes_:
            _lu_pos_['node'].append(_node_), _lu_pos_['x'].append(_pos_[_node_][0]), _lu_pos_['y'].append(_pos_[_node_][1])
            _mu_den_ = 1.0  # the +1 keeps mu_i strictly below the paper's stability bound [2*sum(1/t^k)]^-1 and reproduces the paper's 1/(2n) example at k=0
            for _nbor_ in _nodes_:
                if _nbor_ == _node_: continue
                _t_ = self.distances[_node_][_nbor_]
                _lu_dist_['fm'].append(_node_), _lu_dist_['to'].append(_nbor_), _lu_dist_['t'].append(_t_)
                _mu_den_ += max(_t_, 0.001)**(-self.k)
            _lu_pos_['mu'].append(1.0/(2.0*_mu_den_))
        df_pos, df_dist = pl.DataFrame(_lu_pos_), pl.DataFrame(_lu_dist_)

        iterations = max(self.iterations_min, self.iterations_multiplier*len(_nodes_))
        __dx__, __dy__ = (pl.col('x') - pl.col('x_right')), (pl.col('y') - pl.col('y_right'))
        for i in range(iterations):
            df_pos = df_pos.join(df_pos, how='cross') \
                           .filter(pl.col('node') != pl.col('node_right')) \
                           .with_columns((__dx__**2 + __dy__**2).sqrt().alias('d')) \
                           .join(df_dist, left_on=['node', 'node_right'], right_on=['fm','to']) \
                           .with_columns(pl.col('t').pow(self.k).alias('t_k')) \
                           .with_columns(pl.when(pl.col('d') < 0.001).then(pl.lit(0.001)).otherwise(pl.col('d')).alias('d'),
                                         pl.when(pl.col('t') < 0.001).then(pl.lit(0.001)).otherwise(pl.col('t')).alias('t')) \
                           .with_columns((pl.col('t')**(2-self.k)).alias('__prod_1__'),
                                         ((2.0*__dx__*(1.0 - pl.col('t')/pl.col('d')))/pl.col('t_k')).alias('xadd'),
                                         ((2.0*__dy__*(1.0 - pl.col('t')/pl.col('d')))/pl.col('t_k')).alias('yadd'),
                                         (((pl.col('t') - pl.col('d'))**2)/pl.col('t_k')).alias('__prod_2__')) \
                           .group_by(['node','x','y','mu']).agg(pl.col('xadd').sum(), pl.col('yadd').sum(), pl.col('__prod_1__').sum(), pl.col('__prod_2__').sum()) \
                           .with_columns((pl.col('x') - pl.col('mu') * pl.col('xadd')).alias('x'),
                                         (pl.col('y') - pl.col('mu') * pl.col('yadd')).alias('y')) \
                           .drop(['xadd','yadd'])
            stress = (1.0 / df_pos['__prod_1__'].sum()) * df_pos['__prod_2__'].sum()
            _stress_lu_['stress'].append(stress), _stress_lu_['i'].append(i)
            _round_prec_ = 4
            if i > 8 and round(_stress_lu_['stress'][-1],_round_prec_) == round(_stress_lu_['stress'][-2],_round_prec_) and \
                         round(_stress_lu_['stress'][-2],_round_prec_) == round(_stress_lu_['stress'][-3],_round_prec_) and \
                         round(_stress_lu_['stress'][-3],_round_prec_) == round(_stress_lu_['stress'][-4],_round_prec_): break

        _updated_ = {}
        for i in range(len(df_pos)): _updated_[df_pos['node'][i]] = (df_pos['x'][i], df_pos['y'][i])
        return _updated_
