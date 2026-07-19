from __future__ import annotations

import ipaddress
import json
import random
from math import sqrt, cos, sin, pi, ceil, inf
from collections.abc import Iterable

import polars as pl

from .exceptions import InvalidSpecError

import numpy as np

# networkx/squarify back the graph-layout algorithms (treemaps, force-directed,
# neighborhood clustering, ...) but not the base render path shared by every
# component (xyp, histop, ...), so they are an optional 'layouts' extra rather
# than a core dependency. Guard the import the same way export.py guards
# svglib/reportlab: load if present, otherwise raise a clear ImportError only
# when a graph-layout method is actually called.
try:
    import networkx as nx
    import squarify
    _HAS_GRAPH_LAYOUT_DEPS_ = True
except ImportError:
    nx = None
    squarify = None
    _HAS_GRAPH_LAYOUT_DEPS_ = False

_GRAPH_LAYOUT_DEPS_MSG_ = (
    "graph-layout algorithms require the optional 'layouts' dependencies "
    "(networkx, squarify). Install them with:\n"
    "    pip install polars2svg[layouts]"
)

def _requireGraphLayoutDeps_():
    if not _HAS_GRAPH_LAYOUT_DEPS_:
        raise ImportError(_GRAPH_LAYOUT_DEPS_MSG_)

class P2SGraphMixin:
    def __init__(self):
        pass

    def __p2s_graph_mixin_init__(self):
        pass

    # -------------------------------------------------------------------------
    # Ported from rtsvg/rt_linknode_mixin.py (David Trimm, Apache 2.0)
    # -------------------------------------------------------------------------

    #
    # __graphCountAggExpr__() - Polars aggregation expression for the edge weight.
    # Mirrors the count= convention used by Histop / Timep / Chordp (__countAggExpr__):
    #   ROW_COUNTp (default) -> pl.len(); numeric field -> sum; non-numeric field or
    #   (field, SETp) -> n_unique; multi-field tuple -> struct then n_unique.
    #
    def __graphCountAggExpr__(self, df, count):
        if count == self.ROW_COUNTp:
            return pl.len().alias('__count__')
        elif isinstance(count, str):
            _is_num_ = self.numericColumn(df, count)
            self.logDtypeKeyedCount('graph', count, _is_num_)
            if _is_num_: return pl.col(count).sum()     .alias('__count__')
            else:        return pl.col(count).n_unique() .alias('__count__')
        elif isinstance(count, tuple):
            _fields_ = [_f_ for _f_ in count if isinstance(_f_, str)]
            if self.SETp in count:   return pl.col(_fields_[0]).n_unique().alias('__count__')
            elif len(_fields_) == 1: return pl.col(_fields_[0]).sum()    .alias('__count__')
            else:                    return pl.struct(_fields_).n_unique().alias('__count__')
        return pl.len().alias('__count__')

    def createNetworkXGraph(self,
                            df,
                            relationships,
                            use_digraph = False,
                            count       = None):
        _requireGraphLayoutDeps_()
        if count is None: count = self.ROW_COUNTp

        nx_g = nx.DiGraph() if use_digraph else nx.Graph()

        df = df.clone()
        new_relationships, i = [], 0
        for _edge_ in relationships:
            _fm_ = _edge_[0]
            _to_ = _edge_[1]
            if isinstance(_fm_, tuple) or isinstance(_to_, tuple):
                new_fm, new_to = _fm_, _to_
                if isinstance(_fm_, tuple):
                    new_fm = f'__fm{i}__'
                    df = self.createConcatColumn(df, _fm_, new_fm)
                if isinstance(_to_, tuple):
                    new_to = f'__to{i}__'
                    df = self.createConcatColumn(df, _to_, new_to)
                if   len(_edge_) == 2: new_relationships.append((new_fm, new_to))
                elif len(_edge_) == 3: new_relationships.append((new_fm, new_to, _edge_[2]))
                else: raise InvalidSpecError(f'createNetworkXGraph(): relationship tuples should have two or three parts "{_edge_}"')
            else:
                if   len(_edge_) == 2: new_relationships.append((_fm_, _to_))
                elif len(_edge_) == 3: new_relationships.append((_fm_, _to_, _edge_[2]))
                else: raise InvalidSpecError(f'createNetworkXGraph(): relationship tuples should have two or three parts "{_edge_}"')
            i += 1

        for rel_tuple in new_relationships:
            df_filtered = self.polarsFilterColumnsWithNaNs(df, self.flattenTuple(rel_tuple))
            counter = df_filtered.group_by(list(rel_tuple)).agg(self.__graphCountAggExpr__(df_filtered, count))
            for i in range(len(counter)):
                _row_  = counter[i]
                params = {}
                if len(rel_tuple) == 3: params[rel_tuple[2]] = _row_[rel_tuple[2]][0]
                nx_g.add_edge(_row_[rel_tuple[0]][0], _row_[rel_tuple[1]][0], weight=_row_['__count__'][0], **params)

        return nx_g

    def filterDataFrameByGraph(self, df, relationships, g):
        new_relationships = []
        for i in range(len(relationships)):
            _relationship_ = relationships[i]
            _fm_, _to_ = _relationship_[0], _relationship_[1]
            if isinstance(_relationship_[0], tuple):
                if len(_relationship_[0]) == 1: _fm_ = _relationship_[0][0]
                else:
                    _fm_ = f'__fm{i}__'
                    df = self.createConcatColumn(df, _relationship_[0], _fm_)
            if isinstance(_relationship_[1], tuple):
                if len(_relationship_[1]) == 1: _to_ = _relationship_[1][0]
                else:
                    _to_ = f'__to{i}__'
                    df = self.createConcatColumn(df, _relationship_[1], _to_)
            new_relationships.append((_fm_, _to_))

        edges = set()
        for _node_ in g:
            for _nbor_ in g.neighbors(_node_):
                edges.add((_node_, _nbor_))

        _dfs_ = []
        for _relationship_ in new_relationships:
            for k, k_df in df.group_by(_relationship_):
                if k in edges: _dfs_.append(k_df)

        if   len(_dfs_) == 0: return pl.DataFrame(schema=df.schema)
        else:                 return pl.concat(_dfs_)

    def collapseDataFrameEdgesToOneRow(self, df, relationships, selected=None):
        """Keep a single representative row for visualized edges, dropping the
        per-edge row overhead while leaving the rendered graph unchanged.

        With no selection, every edge is collapsed to one row. With a selection,
        only edges adjacent to a selected node are collapsed; all other edges
        retain all of their rows.
        """
        _orig_columns_ = df.columns
        new_relationships = []
        for i in range(len(relationships)):
            _relationship_ = relationships[i]
            _fm_, _to_ = _relationship_[0], _relationship_[1]
            if isinstance(_relationship_[0], tuple):
                if len(_relationship_[0]) == 1: _fm_ = _relationship_[0][0]
                else:
                    _fm_ = f'__fm{i}__'
                    df = self.createConcatColumn(df, _relationship_[0], _fm_)
            if isinstance(_relationship_[1], tuple):
                if len(_relationship_[1]) == 1: _to_ = _relationship_[1][0]
                else:
                    _to_ = f'__to{i}__'
                    df = self.createConcatColumn(df, _relationship_[1], _to_)
            new_relationships.append((_fm_, _to_))

        selected = selected or set()
        _idx_ = '__cde_idx__'
        df = df.with_row_index(_idx_)
        _keep_ = set()
        for _relationship_ in new_relationships:
            for k, k_df in df.group_by(_relationship_):
                _adjacent_ = (k[0] in selected) or (k[1] in selected)
                if len(selected) == 0 or _adjacent_: _keep_.add(k_df[_idx_].min())      # collapse to earliest row
                else:                                _keep_.update(k_df[_idx_].to_list()) # keep all rows

        return df.filter(pl.col(_idx_).is_in(_keep_)).select(_orig_columns_)

    # -------------------------------------------------------------------------
    # Ported from rtsvg/rt_graph_layouts_mixin.py (David Trimm, Apache 2.0)
    # -------------------------------------------------------------------------

    def treeMapLayout(self, _graph, pos, bounds_percent=0.1):
        _requireGraphLayoutDeps_()
        _graph = nx.to_undirected(_graph)
        S = [_graph.subgraph(c).copy() for c in nx.connected_components(_graph)]
        if len(S) <= 1: return pos

        my_order = []
        for _subgraph in S: my_order.append((len(_subgraph), _subgraph))
        my_order.sort(key=lambda x: x[0], reverse=True)

        nodes = []
        for tup in my_order: nodes.append(tup[0])

        normalized_sizes   = squarify.normalize_sizes(nodes, 1024, 1024)
        treemap_rectangles = squarify.squarify(normalized_sizes, 0, 0, 1024, 1024)

        new_pos = {}
        for i in range(0, len(my_order)):
            _subgraph = my_order[i][1]
            _rect     = treemap_rectangles[i]
            rx, ry, rdx, rdy = _rect['x'], _rect['y'], _rect['dx'], _rect['dy']
            if bounds_percent > 0.0 and bounds_percent < 1.0:
                xperc, yperc = rdx*bounds_percent, rdy*bounds_percent
                rx  += xperc/2
                ry  += yperc/2
                rdx -= xperc
                rdy -= yperc
            x0, y0, x1, y1 = self.positionExtents(pos, _subgraph)
            for _node in _subgraph.nodes():
                x = (pos[_node][0] - x0)/(x1-x0)
                y = (pos[_node][1] - y0)/(y1-y0)
                new_pos[_node] = [x*rdx + rx, y*rdy + ry]

        return new_pos

    def circlePackLayout(self, g, pos):
        _requireGraphLayoutDeps_()
        g_components = [g.subgraph(c) for c in nx.connected_components(g)]
        circles = []
        for _g_ in g_components:
            _pts_    = [pos[_node_] for _node_ in _g_.nodes]
            _circle_ = self.smallestEnclosingCircleApprox(_pts_)
            circles.append(_circle_)

        _packed_ = self.packCircles(circles)

        _new_pos_, _shapes_ = {}, {}
        for i in range(len(g_components)):
            _g_ = g_components[i]
            _dx_, _dy_ = circles[i][0] - _packed_[i][0], circles[i][1] - _packed_[i][1]
            for _node_ in _g_.nodes: _new_pos_[_node_] = pos[_node_][0] - _dx_, pos[_node_][1] - _dy_
            _shapes_[i] = f'<circle cx="{_packed_[i][0]}" cy="{_packed_[i][1]}" r="{_packed_[i][2]}" fill="None" stroke="#000000" stroke-width="1" />'

        return _new_pos_, _shapes_

    def rectangularLayout(self, g, nodes, pos=None, bounds=(0, 0, 1, 1)):
        x0, y0, x1, y1 = bounds
        if x0 > x1: x0, x1 = x1, x0
        if y0 > y1: y0, y1 = y1, y0
        dx, dy = float(x1-x0), float(y1-y0)
        if isinstance(nodes, list) == False: nodes = list(nodes)
        if pos is None: pos = {}
        if len(nodes) == 1:
            pos[nodes[0]] = (x0 + dx/2.0, y0 + dy/2.0)
        elif len(nodes) == 2:
            pos[nodes[0]], pos[nodes[1]] = (x0, y0+dy/2), (x1, y0+dy/2.0)
        elif len(nodes) == 3:
            pos[nodes[0]], pos[nodes[1]], pos[nodes[2]] = (x0, y0), (x0 + dx/2.0, y1), (x1, y0)
        elif len(nodes) == 4 or len(nodes) == 5:
            pos[nodes[0]], pos[nodes[1]] = (x0, y0), (x1, y0)
            pos[nodes[2]], pos[nodes[3]] = (x0, y1), (x1, y1)
            if len(nodes) == 5: pos[nodes[4]] = ((x0+x1)/2.0, (y0+y1)/2.0)
        else:
            if x0 >= x1: x1 = x0 + 1.0
            if y0 >= y1: y1 = y0 + 1.0
            dx, dy = x1 - x0, y1 - y0
            n = ceil(sqrt(len(nodes)))
            if (dx/dy) > 1.5 or (dy/dx) > 1.5:
                # Pick the grid shape whose aspect (cols/rows) best matches the
                # box aspect (dx/dy). When cols/rows ~= dx/dy, the per-axis steps
                # dx/cols and dy/rows are nearly equal, so a single uniform step
                # fills the non-square box with minimal residual margin.
                target    = dx/dy
                closest_d = inf
                for cols in range(1, len(nodes)+1):
                    rows = ceil(len(nodes)/cols)
                    d    = abs(cols/rows - target)
                    if d < closest_d:
                        closest_d, max_x_i, max_y_i = d, cols, rows
            else:
                max_x_i = max_y_i = n

            _sorter_  = []
            for node in nodes:
                _degrees_ = g.degree(node)
                _deg_     = _degrees_ if isinstance(_degrees_, int) else len(_degrees_)
                if _deg_ == 1:
                    _nbor_   = next(iter(g.neighbors(node)))
                    _weight_ = g[node][_nbor_].get('weight', 0)
                else:
                    _weight_ = 0
                _sorter_.append((_deg_, _weight_, node))
            _sorter_ = sorted(_sorter_, reverse=True)

            # Uniform spacing on both axes: pick the smaller per-axis step so the
            # grid fits within bounds, then center the grid inside the bounds.
            step   = min(dx/max_x_i, dy/max_y_i)
            n_cols = int(ceil(max_x_i))
            n_rows = int(ceil(len(nodes)/n_cols))
            ox     = x0 + (dx - (n_cols-1)*step)/2.0
            oy     = y0 + (dy - (n_rows-1)*step)/2.0
            x_i, y_i = 0, 0
            for i in range(len(nodes)):
                _x_ = ox + x_i * step
                _y_ = oy + y_i * step
                pos[_sorter_[i][2]] = (_x_, _y_)
                x_i += 1
                if x_i >= max_x_i:
                    y_i += 1
                    x_i  = 0

        return pos

    def treeMapNodeColorLayout(self,
                               g             : nx.Graph,
                               nodes         : Iterable[str],
                               node_color_lu : dict,
                               pos           : dict  = None,
                               collapse      : bool  = False,
                               bounds        : tuple = (0, 0, 1, 1)) -> dict:
        _requireGraphLayoutDeps_()
        if pos is None: pos = {}
        _default_color_ = '#4988b6'

        _color_to_nodes_ = {}
        for _node_ in nodes:
            _color_ = node_color_lu.get(_node_, _default_color_)
            if _color_ not in _color_to_nodes_: _color_to_nodes_[_color_] = set()
            _color_to_nodes_[_color_].add(_node_)

        _color_size_tuples_ = []
        for _color_ in _color_to_nodes_: _color_size_tuples_.append((len(_color_to_nodes_[_color_]), _color_))
        _color_size_tuples_.sort(reverse=True)

        _color_sizes_ = [_tuple_[0] for _tuple_ in _color_size_tuples_]

        _normalized_sizes_   = squarify.normalize_sizes(_color_sizes_, bounds[2]-bounds[0], bounds[3]-bounds[1])
        _treemap_rectangles_ = squarify.squarify(_normalized_sizes_, bounds[0], bounds[1], bounds[2]-bounds[0], bounds[3]-bounds[1])

        for i in range(len(_color_size_tuples_)):
            _color_  = _color_size_tuples_[i][1]
            _rect_   = _treemap_rectangles_[i]
            _bounds_ = (_rect_['x'], _rect_['y'], _rect_['x'] + _rect_['dx'], _rect_['y'] + _rect_['dy'])
            if collapse:
                for _node_ in _color_to_nodes_[_color_]: pos[_node_] = (_bounds_[0]+_bounds_[2])/2.0, (_bounds_[1]+_bounds_[3])/2.0
            else:
                self.rectangularLayout(g, _color_to_nodes_[_color_], pos=pos, bounds=_bounds_)

        return pos

    def ipSubnetTreeMapLayout(self,
                              g           : nx.Graph,
                              subnet_mask : int   = 24,
                              pos         : dict  = None,
                              collapse     : bool  = False,
                              cell_inset   : float = 0.0,
                              return_cells : bool  = False,
                              bounds       : tuple = (0, 0, 1, 1)) -> dict:
        """Treemap layout that groups nodes by IPv4 subnet.

        Nodes are split into two groups:

        - **IPv4 nodes** — whose ``str(node)`` parses as an IPv4 address. The
          ``subnet_mask`` (e.g. ``24`` -> ``/24``) is applied to each address
          and nodes sharing the same network are grouped together (one treemap
          cell per subnet).
        - **Other nodes** — everything that does not parse as IPv4. These are
          all treated as a single group occupying one treemap cell.

        Group cells are sized by node count via squarify; within each cell the
        nodes are placed with ``rectangularLayout`` (or collapsed to the cell
        center when ``collapse=True``).

        ``cell_inset`` is a fraction in ``[0, 1)`` that shrinks the node layout
        toward each cell's center, creating a buffer/gutter between cells. It
        does **not** change the treemap cell shapes themselves — the squarify
        partition is unchanged; only the area the nodes are placed within is
        inset. ``0.0`` fills the whole cell; ``0.2`` leaves a 10% margin on each
        side.

        When ``return_cells`` is ``True`` the method returns ``(pos, cells)``
        instead of just ``pos``. ``cells`` is a ``{label: shapely_box}`` dict
        mapping each subnet (e.g. ``'10.0.0.0/24'``, or ``'non-IPv4'`` for the
        other group) to its **full** treemap rectangle — ready to hand to
        linkp's ``background=`` parameter. The cell boxes are the un-inset
        squarify rectangles, so combining ``cell_inset`` with these cells draws
        the nodes within a visible gutter between subnet cells. When
        ``return_cells`` is ``False`` (the default) only ``pos`` is returned, so
        existing callers are unaffected.
        """
        _requireGraphLayoutDeps_()
        if pos is None: pos = {}

        # Sentinel key for the single "other" (non-IPv4) group, plus the
        # human-friendly label used for its cell when return_cells is set.
        _OTHER_       = '__non_ipv4__'
        _OTHER_LABEL_ = 'non-IPv4'

        def _toIPv4_(_node_):
            # Accept a bare address ("10.0.0.5") or one carrying a port / CIDR
            # suffix ("10.0.0.5:443", "10.0.0.5/24") by parsing the leading
            # dotted-quad token. Returns an IPv4Address or None.
            _s_ = str(_node_)
            for _candidate_ in (_s_, _s_.split(':', 1)[0], _s_.split('/', 1)[0]):
                try:
                    return ipaddress.IPv4Address(_candidate_)
                except ipaddress.AddressValueError:
                    continue
            return None

        _group_to_nodes_ = {}
        for _node_ in g.nodes():
            _ip_  = _toIPv4_(_node_)
            _key_ = str(ipaddress.ip_network(f'{_ip_}/{subnet_mask}', strict=False)) if _ip_ is not None else _OTHER_
            if _key_ not in _group_to_nodes_: _group_to_nodes_[_key_] = set()
            _group_to_nodes_[_key_].add(_node_)

        if len(_group_to_nodes_) == 0: return (pos, {}) if return_cells else pos

        if return_cells:
            from shapely.geometry import box as _box_
            cells = {}

        # Order groups by size (largest first) for a stable treemap; among
        # equal sizes sort by key so the "other" group lands deterministically.
        _group_size_tuples_ = [(len(_nodes_), _key_) for _key_, _nodes_ in _group_to_nodes_.items()]
        _group_size_tuples_.sort(key=lambda t: (t[0], t[1]), reverse=True)

        _group_sizes_        = [_tuple_[0] for _tuple_ in _group_size_tuples_]
        _normalized_sizes_   = squarify.normalize_sizes(_group_sizes_, bounds[2]-bounds[0], bounds[3]-bounds[1])
        _treemap_rectangles_ = squarify.squarify(_normalized_sizes_, bounds[0], bounds[1], bounds[2]-bounds[0], bounds[3]-bounds[1])

        for i in range(len(_group_size_tuples_)):
            _key_    = _group_size_tuples_[i][1]
            _nodes_  = _group_to_nodes_[_key_]
            _rect_   = _treemap_rectangles_[i]
            _x0_, _y0_ = _rect_['x'], _rect_['y']
            _x1_, _y1_ = _rect_['x'] + _rect_['dx'], _rect_['y'] + _rect_['dy']
            # The full (un-inset) squarify rectangle is the subnet cell.
            if return_cells:
                _label_ = _OTHER_LABEL_ if _key_ == _OTHER_ else _key_
                cells[_label_] = _box_(_x0_, _y0_, _x1_, _y1_)

            _ccx_, _ccy_ = (_x0_ + _x1_) / 2.0, (_y0_ + _y1_) / 2.0
            if collapse:
                for _node_ in _nodes_: pos[_node_] = (_ccx_, _ccy_)
                continue

            # Lay out within the full cell, then recenter the node cloud on the
            # cell center and scale it by (1 - cell_inset). rectangularLayout
            # anchors its grid at the cell's top-left and stops short of the far
            # edges, so its output is not centered; recentering on the cloud's
            # own bounding box (not the cell) makes every node count land
            # centered, and the scale leaves a (cell_inset/2) gutter per side.
            self.rectangularLayout(g, _nodes_, pos=pos, bounds=(_x0_, _y0_, _x1_, _y1_))
            _scale_ = (1.0 - cell_inset) if (cell_inset > 0.0 and cell_inset < 1.0) else 1.0
            _xs_ = [pos[_node_][0] for _node_ in _nodes_]
            _ys_ = [pos[_node_][1] for _node_ in _nodes_]
            _bcx_, _bcy_ = (min(_xs_) + max(_xs_)) / 2.0, (min(_ys_) + max(_ys_)) / 2.0
            for _node_ in _nodes_:
                _px_, _py_ = pos[_node_]
                pos[_node_] = (_ccx_ + (_px_ - _bcx_) * _scale_,
                               _ccy_ + (_py_ - _bcy_) * _scale_)

        return (pos, cells) if return_cells else pos

    def ipSubnetForceDirectedLayout(self,
                                    g              : nx.Graph,
                                    subnet_mask    : int   = 24,
                                    pos            : dict  = None,
                                    collapse       : bool  = False,
                                    cluster_frac   : float = 0.45,
                                    iterations     : int   = 50,
                                    seed           : int   = 42,
                                    return_rep_pos : bool  = False,
                                    bounds         : tuple = (0, 0, 1, 1)) -> dict:
        """Force-directed layout that groups nodes by IPv4 subnet.

        Like :meth:`ipSubnetTreeMapLayout`, nodes are bucketed by the network
        they belong to under ``subnet_mask`` (e.g. ``24`` -> ``/24``); every
        non-IPv4 node falls into a single ``'__non_ipv4__'`` group. Each subnet
        is collapsed into one **representative node**, and a *representative
        graph* is built whose edges carry the **accumulated** weight of every
        original edge crossing between the two subnets. For example, with a
        ``/24`` mask an edge ``1.2.3.4 -> 5.6.7.8`` of weight ``w`` contributes
        ``w`` to the representative edge ``1.2.3.0/24 -> 5.6.7.0/24``; multiple
        crossing edges sum together. Intra-subnet edges are dropped — they do
        not move a representative node relative to the others.

        The representative graph is placed with NetworkX's spring (force
        directed) layout, weighting the springs by the accumulated edge weight,
        and the result is fit into ``bounds``. The **original** nodes are then
        positioned: by default each subnet's members are spread around their
        representative position with ``sunflowerSeedLayout`` (radius scaled by
        ``cluster_frac`` of the nearest rep-to-rep distance so clusters stay
        visually distinct); with ``collapse=True`` every member is placed
        exactly on its representative position.

        ``iterations`` and ``seed`` are forwarded to ``spring_layout`` for
        reproducible placement. When ``return_rep_pos`` is ``True`` the method
        returns ``(pos, rep_pos)`` where ``rep_pos`` maps each subnet label
        (e.g. ``'10.0.0.0/24'``, or ``'__non_ipv4__'``) to its representative
        position; otherwise only ``pos`` (positions of the original nodes) is
        returned.
        """
        _requireGraphLayoutDeps_()
        if pos is None: pos = {}

        _OTHER_ = '__non_ipv4__'

        def _toIPv4_(_node_):
            # Accept a bare address ("10.0.0.5") or one carrying a port / CIDR
            # suffix ("10.0.0.5:443", "10.0.0.5/24") by parsing the leading
            # dotted-quad token. Returns an IPv4Address or None.
            _s_ = str(_node_)
            for _candidate_ in (_s_, _s_.split(':', 1)[0], _s_.split('/', 1)[0]):
                try:
                    return ipaddress.IPv4Address(_candidate_)
                except ipaddress.AddressValueError:
                    continue
            return None

        # Bucket every node into its subnet representative and keep the inverse
        # mapping so member nodes can be placed around their rep afterwards.
        _node_to_rep_  = {}
        _rep_to_nodes_ = {}
        for _node_ in g.nodes():
            _ip_  = _toIPv4_(_node_)
            _key_ = str(ipaddress.ip_network(f'{_ip_}/{subnet_mask}', strict=False)) if _ip_ is not None else _OTHER_
            _node_to_rep_[_node_] = _key_
            _rep_to_nodes_.setdefault(_key_, set()).add(_node_)

        if len(_rep_to_nodes_) == 0: return (pos, {}) if return_rep_pos else pos

        # Build the representative graph: one node per subnet, edge weights are
        # the sum of all original crossing-edge weights between the subnets.
        _rep_g_ = nx.Graph()
        _rep_g_.add_nodes_from(_rep_to_nodes_.keys())
        for _u_, _v_, _data_ in g.edges(data=True):
            _ru_, _rv_ = _node_to_rep_[_u_], _node_to_rep_[_v_]
            if _ru_ == _rv_: continue
            _w_ = _data_.get('weight', 1)
            if _rep_g_.has_edge(_ru_, _rv_): _rep_g_[_ru_][_rv_]['weight'] += _w_
            else:                            _rep_g_.add_edge(_ru_, _rv_, weight=_w_)

        # Force-directed placement of the representative graph, then fit the
        # result into bounds (preserving the relative layout).
        _rep_pos_ = nx.spring_layout(_rep_g_, weight='weight', iterations=iterations, seed=seed)

        _bx0_, _by0_, _bx1_, _by1_ = bounds
        _xs_ = [_p_[0] for _p_ in _rep_pos_.values()]
        _ys_ = [_p_[1] for _p_ in _rep_pos_.values()]
        _minx_, _maxx_ = min(_xs_), max(_xs_)
        _miny_, _maxy_ = min(_ys_), max(_ys_)
        _spanx_ = (_maxx_ - _minx_) or 1.0
        _spany_ = (_maxy_ - _miny_) or 1.0
        for _key_ in _rep_pos_:
            _px_, _py_ = _rep_pos_[_key_]
            _rep_pos_[_key_] = (_bx0_ + (_px_ - _minx_) / _spanx_ * (_bx1_ - _bx0_),
                                _by0_ + (_py_ - _miny_) / _spany_ * (_by1_ - _by0_))

        # Cluster radius for spreading a subnet's members around its rep: a
        # fraction of the closest rep-to-rep distance so neighbouring clusters
        # stay separate (fall back to the bounds size for a single subnet).
        if len(_rep_pos_) > 1:
            _pts_   = list(_rep_pos_.values())
            _min_d_ = inf
            for i in range(len(_pts_)):
                for j in range(i + 1, len(_pts_)):
                    _d_ = sqrt((_pts_[i][0] - _pts_[j][0]) ** 2 + (_pts_[i][1] - _pts_[j][1]) ** 2)
                    if _d_ < _min_d_: _min_d_ = _d_
            _radius_ = cluster_frac * _min_d_ / 2.0
        else:
            _radius_ = cluster_frac * min(_bx1_ - _bx0_, _by1_ - _by0_) / 2.0

        # Position the original nodes from their representative placement.
        for _key_, _nodes_ in _rep_to_nodes_.items():
            _cx_, _cy_ = _rep_pos_[_key_]
            if collapse:
                for _node_ in _nodes_: pos[_node_] = (_cx_, _cy_)
            else:
                self.sunflowerSeedLayout(g, _nodes_, pos=pos, xy=(_cx_, _cy_), r_max=_radius_)

        return (pos, _rep_pos_) if return_rep_pos else pos

    def neighborhoodLayout(self,
                           g,
                           pos              = None,
                           mode             = 'spatial',
                           return_cells     = False,
                           # 'spatial' mode (clustering over the existing layout) controls
                           spatial_method   = 'hdbscan',
                           cluster_dist     = None,
                           min_cluster_size = 5,
                           min_samples      = None,
                           cluster_selection_method = 'eom',
                           # 'graph' mode (weighted community detection) controls
                           resolution       = 1.0,
                           cluster_frac     = 0.45,
                           collapse         = False,
                           iterations       = 50,
                           seed             = 42,
                           # shared Voronoi-cell control
                           cell_pad         = 0.05,
                           bounds           = (0, 0, 1, 1)) -> dict:
        """Detect node neighborhoods and (optionally) outline them with a Voronoi background.

        Two related layouts selected by ``mode`` -- both end by tessellating the
        final node positions into one polygon per neighborhood (the Voronoi
        medial-axis of the gaps, §6 of the dev RINGROADP spec) and, when
        ``return_cells`` is set, returning that ``{neighborhood: shapely_polygon}``
        dict as the linkp ``background=`` layer (mirroring
        :meth:`ipSubnetTreeMapLayout` / :meth:`hyperTreeDonutLayout`).

        ``mode='spatial'`` (default) -- neighborhoods are read **from the existing
        layout**: nodes are clustered by their ``pos`` coordinates (where the
        layout already shows spatial grouping + separation, that *is* a
        neighborhood). Node positions are **left unchanged**; only neighborhood
        membership (and the Voronoi outline) is computed. Every node belongs to
        some neighborhood and contributes a Voronoi cell. The clustering algorithm
        is chosen by ``spatial_method``:

        - ``'hdbscan'`` (default) -- density-based clustering via
          ``sklearn.cluster.HDBSCAN``. Chosen as the default because it delineates
          neighborhoods robustly across real-world layouts where ``'single_linkage'``
          fails: single-linkage depends on a clean length gap in the layout's MST,
          which does not always exist -- on some layouts it collapses everything
          into one neighborhood or merges most nodes together, whereas HDBSCAN's
          density model keeps delineating correctly (and agrees with single-linkage
          where single-linkage works). *Noise* nodes (label ``-1``) each become
          their **own singleton neighborhood**. ``min_cluster_size`` (default 5),
          ``min_samples`` and ``cluster_selection_method`` (default ``'eom'``) are
          forwarded to ``HDBSCAN``. Defaults are tuned to avoid over-fragmentation:
          ``min_cluster_size=2`` with ``'eom'`` over-splits into many tiny clusters;
          ``5`` gives clean neighborhoods, and ``'leaf'`` selection over-fragments
          badly (keep ``'eom'``).
        - ``'single_linkage'`` -- single-linkage agglomerative clustering
          (``sklearn.cluster.AgglomerativeClustering``) cut at an absolute gap
          distance: any two nodes closer than ``cluster_dist`` layout-units belong
          to the same neighborhood (gap-based grouping that matches how the eye
          reads proximity). Isolated nodes fall out as their own singleton
          neighborhoods automatically. ``cluster_dist`` is in the same units as
          ``pos``; when ``None`` it defaults to ``2.0 ×`` the median nearest-
          neighbor distance, so it scales with the layout automatically.

        ``mode='graph'`` -- neighborhoods are derived from **graph structure**:
        weighted (``weight=`` edge attribute, i.e. whatever ``count=`` produced
        when the graph was built) community detection via Louvain
        (``resolution=``). Detection is followed by **repositioning** so the
        neighborhoods get space between them: each community collapses to a
        representative node, the representatives are placed with a weighted spring
        layout (crossing-edge weights summed, like
        :meth:`ipSubnetForceDirectedLayout`) and fit to ``bounds``, then each
        community's members are spread around their representative with
        :meth:`sunflowerSeedLayout` at a radius of ``cluster_frac`` of the nearest
        rep-to-rep distance (smaller ``cluster_frac`` -> tighter clusters -> more
        whitespace between neighborhoods). ``collapse=True`` drops every member
        onto its representative; ``iterations`` / ``seed`` are forwarded to
        ``spring_layout``.

        ``cell_pad`` pads the bounding box the Voronoi cells are clipped to (a
        fraction of the layout extent), so edge neighborhoods get a finite outline
        rather than one snapped to the outermost node. Returns ``pos`` (positions:
        unchanged for ``'spatial'``, repositioned for ``'graph'``), or
        ``(pos, cells)`` when ``return_cells`` is ``True``.
        """
        _requireGraphLayoutDeps_()
        if mode not in ('spatial', 'graph'):
            raise InvalidSpecError(f"neighborhoodLayout(): mode must be 'spatial' or 'graph', got '{mode}'")

        if mode == 'spatial':
            if pos is None:
                raise InvalidSpecError("neighborhoodLayout(mode='spatial'): an existing 'pos' is required")
            if spatial_method not in ('single_linkage', 'hdbscan'):
                raise InvalidSpecError("neighborhoodLayout(mode='spatial'): spatial_method must be "
                                f"'single_linkage' or 'hdbscan', got '{spatial_method}'")

            # Cluster nodes by their existing layout coordinates. Positions are
            # never modified -- this mode only labels neighborhoods.
            _nodes_  = [n for n in g.nodes() if n in pos]
            new_pos  = {n: (float(pos[n][0]), float(pos[n][1])) for n in _nodes_}
            _labels_ = {}
            if len(_nodes_) >= 2:
                _pts_   = np.array([[pos[n][0], pos[n][1]] for n in _nodes_], dtype=float)
                if spatial_method == 'single_linkage':
                    from sklearn.cluster import AgglomerativeClustering
                    # Gap-based grouping: merge nodes closer than cluster_dist
                    # layout-units. When cluster_dist is None, pick the cut from the
                    # data itself: cut the single-linkage tree (== the layout's MST)
                    # at its largest length gap -- the natural divide between dense
                    # within-cluster edges and sparse between-cluster edges. If no
                    # edge is more than 2x its predecessor there is no real gap, so
                    # everything is one neighborhood.
                    _thr_ = cluster_dist
                    if _thr_ is None:
                        _thr_ = self.__p2sg_singleLinkageGap__(_pts_)
                    if not np.isfinite(_thr_):
                        # No real gap to cut at -> everything is one neighborhood.
                        # sklearn rejects an infinite distance_threshold, so label
                        # directly instead of asking it to merge at distance inf.
                        _lbls_ = np.zeros(len(_nodes_), dtype=int)
                    else:
                        _clust_ = AgglomerativeClustering(n_clusters=None, linkage='single',
                                                          distance_threshold=_thr_).fit(_pts_)
                        _lbls_ = _clust_.labels_
                else:  # spatial_method == 'hdbscan'
                    from sklearn.cluster import HDBSCAN
                    _mcs_   = max(2, min(min_cluster_size, len(_nodes_)))
                    _clust_ = HDBSCAN(min_cluster_size=_mcs_, min_samples=min_samples,
                                      cluster_selection_method=cluster_selection_method,
                                      copy=False).fit(_pts_)
                    _lbls_ = _clust_.labels_

                # Each noise node (HDBSCAN label -1) becomes its own singleton
                # neighborhood; numbering continues past the real cluster labels
                # so the 'n{...}' namespace stays collision-free. (single_linkage
                # emits no -1 labels -- isolated nodes are already own clusters.)
                _next_ = max((int(l) for l in _lbls_ if l >= 0), default=-1) + 1
                for _node_, _lab_ in zip(_nodes_, _lbls_):
                    if _lab_ >= 0:
                        _labels_[_node_] = f'n{int(_lab_)}'
                    else:
                        _labels_[_node_] = f'n{_next_}'
                        _next_ += 1

            if not return_cells: return new_pos
            cells = self.__p2sg_voronoiNeighborhoodCells__(new_pos, _labels_, pad=cell_pad)
            return new_pos, cells

        # mode == 'graph' -----------------------------------------------------
        if pos is None: pos = {}
        gu = nx.to_undirected(g)

        # Weighted community detection: link connection strength == the 'weight'
        # edge attribute (whatever count= produced when the graph was built).
        _communities_ = nx.community.louvain_communities(gu, weight='weight',
                                                          resolution=resolution, seed=seed)
        _node_to_rep_, _rep_to_nodes_ = {}, {}
        for i, _comm_ in enumerate(_communities_):
            _key_ = f'c{i}'
            _rep_to_nodes_[_key_] = set(_comm_)
            for _node_ in _comm_: _node_to_rep_[_node_] = _key_

        if len(_rep_to_nodes_) == 0:
            return (pos, {}) if return_cells else pos

        # Representative graph: one node per community, edge weights are the sum
        # of all crossing-edge weights between the two communities (as in
        # ipSubnetForceDirectedLayout). Intra-community edges are dropped.
        _rep_g_ = nx.Graph()
        _rep_g_.add_nodes_from(_rep_to_nodes_.keys())
        for _u_, _v_, _data_ in gu.edges(data=True):
            _ru_, _rv_ = _node_to_rep_[_u_], _node_to_rep_[_v_]
            if _ru_ == _rv_: continue
            _w_ = _data_.get('weight', 1)
            if _rep_g_.has_edge(_ru_, _rv_): _rep_g_[_ru_][_rv_]['weight'] += _w_
            else:                            _rep_g_.add_edge(_ru_, _rv_, weight=_w_)

        _rep_pos_ = nx.spring_layout(_rep_g_, weight='weight', iterations=iterations, seed=seed)

        _bx0_, _by0_, _bx1_, _by1_ = bounds
        _xs_ = [_p_[0] for _p_ in _rep_pos_.values()]
        _ys_ = [_p_[1] for _p_ in _rep_pos_.values()]
        _minx_, _maxx_ = min(_xs_), max(_xs_)
        _miny_, _maxy_ = min(_ys_), max(_ys_)
        _spanx_ = (_maxx_ - _minx_) or 1.0
        _spany_ = (_maxy_ - _miny_) or 1.0
        for _key_ in _rep_pos_:
            _px_, _py_ = _rep_pos_[_key_]
            _rep_pos_[_key_] = (_bx0_ + (_px_ - _minx_) / _spanx_ * (_bx1_ - _bx0_),
                                _by0_ + (_py_ - _miny_) / _spany_ * (_by1_ - _by0_))

        # Cluster radius: a fraction of the closest rep-to-rep distance so
        # neighbouring communities stay separated (gap = whitespace between them).
        if len(_rep_pos_) > 1:
            _pts_   = list(_rep_pos_.values())
            _min_d_ = inf
            for i in range(len(_pts_)):
                for j in range(i + 1, len(_pts_)):
                    _d_ = sqrt((_pts_[i][0] - _pts_[j][0]) ** 2 + (_pts_[i][1] - _pts_[j][1]) ** 2)
                    if _d_ < _min_d_: _min_d_ = _d_
            _radius_ = cluster_frac * _min_d_ / 2.0
        else:
            _radius_ = cluster_frac * min(_bx1_ - _bx0_, _by1_ - _by0_) / 2.0

        _labels_ = {}
        for _key_, _nodes_ in _rep_to_nodes_.items():
            _cx_, _cy_ = _rep_pos_[_key_]
            if collapse:
                for _node_ in _nodes_: pos[_node_] = (_cx_, _cy_)
            else:
                self.sunflowerSeedLayout(g, _nodes_, pos=pos, xy=(_cx_, _cy_), r_max=_radius_)
            for _node_ in _nodes_: _labels_[_node_] = _key_

        if not return_cells: return pos
        cells = self.__p2sg_voronoiNeighborhoodCells__(pos, _labels_, pad=cell_pad)
        return pos, cells

    def sunflowerSeedLayout(self, g, nodes, pos=None, xy=None, r_max=1.0):
        if isinstance(nodes, list) == False: nodes = list(nodes)
        if xy is None: xy = (0, 0)
        n = len(nodes)

        _sorter_ = []
        for node in nodes:
            _degrees_ = g.degree(node)
            if isinstance(_degrees_, int): _sorter_.append((_degrees_,      node))
            else:                          _sorter_.append((len(_degrees_), node))
        _sorter_ = sorted(_sorter_, reverse=True)

        if pos is None: pos = {}
        r_max_formula  = np.sqrt(n)
        _golden_ratio_ = (1 + np.sqrt(5)) / 2
        for i in range(n):
            _angle_  = i * 2 * np.pi / _golden_ratio_
            _radius_ = r_max * np.sqrt(i) / r_max_formula
            pos[_sorter_[i][1]] = (xy[0] + _radius_ * np.cos(_angle_),
                                   xy[1] + _radius_ * np.sin(_angle_))
        return pos

    def linearOptimizedLayout(self, g, nodes, pos, segment=((0.0, 0.0), (1.0, 1.0))):
        if len(nodes) == 1: return {nodes[0]: ((segment[0][0]+segment[1][0])/2.0, (segment[0][1]+segment[1][1])/2.0)}
        adj_pos, as_set = {}, set(nodes)
        _externals_, _internals_ = set(), set()
        for _node_ in nodes:
            if _node_ in g:
                for _nbor_ in g.neighbors(_node_):
                    if _nbor_ not in as_set:
                        _externals_.add(_node_)
                        break
            if _node_ not in _externals_: _internals_.add(_node_)

        dx, dy = segment[1][0] - segment[0][0], segment[1][1] - segment[0][1]
        _filled_with_, _locations_ = [], []
        for i in range(len(nodes)):
            _filled_with_.append(None)
            perc = i/float(len(nodes)-1)
            _locations_.append((segment[0][0] + dx*perc, segment[0][1] + dy*perc))

        def placeNodeIntoClosestSlot(node_to_place, nodes_xy=None):
            _closest_ = 0
            if nodes_xy is not None:
                _closest_pt_ = self.closestPointOnSegment(segment, nodes_xy)[1]
                if    _closest_pt_ == segment[0]: _closest_ = 0
                elif  _closest_pt_ == segment[1]: _closest_ = len(_locations_)-1
                else:
                    _closest_d_ = sqrt((_closest_pt_[0]-_locations_[0][0])**2 + (_closest_pt_[1]-_locations_[0][1])**2)
                    for i in range(1, len(_locations_)):
                        _d_ = sqrt((_closest_pt_[0]-_locations_[i][0])**2 + (_closest_pt_[1]-_locations_[i][1])**2)
                        if _d_ < _closest_d_: _closest_, _closest_d_ = i, _d_
            if _filled_with_[_closest_] is None:
                _filled_with_[_closest_] = node_to_place
                adj_pos[node_to_place]   = _locations_[_closest_]
            else:
                for j in range(1, len(_locations_)):
                    up = _closest_+j
                    dn = _closest_-j
                    if   up < len(_locations_) and _filled_with_[up] is None:
                        _filled_with_[up]      = node_to_place
                        adj_pos[node_to_place] = _locations_[up]
                        break
                    elif dn >= 0 and _filled_with_[dn] is None:
                        _filled_with_[dn]      = node_to_place
                        adj_pos[node_to_place] = _locations_[dn]
                        break

        _sorter_ = []
        for _node_ in _externals_:
            _deg_ = g.degree(_node_)
            if _deg_ == 1:
                _nbor_   = next(iter(g.neighbors(_node_)))
                _weight_ = g[_node_][_nbor_].get('weight', 0)
            else:
                _weight_ = 0
            _sorter_.append((_deg_, _weight_, _node_))
        _sorter_ = sorted(_sorter_, reverse=True)
        for _tuple_ in _sorter_:
            _node_ = _tuple_[2]
            _x_sum_, _y_sum_, _samples_ = 0.0, 0.0, 0
            for _nbor_ in g.neighbors(_node_):
                if _nbor_ in as_set: continue
                _x_sum_   += pos[_nbor_][0]
                _y_sum_   += pos[_nbor_][1]
                _samples_ += 1
            _x_, _y_ = _x_sum_ / _samples_, _y_sum_ / _samples_
            placeNodeIntoClosestSlot(_node_, (_x_, _y_))

        _int_sorter_ = []
        for _node_ in _internals_:
            _deg_ = g.degree(_node_) if _node_ in g else 0
            if _deg_ == 1:
                _nbor_   = next(iter(g.neighbors(_node_)))
                _weight_ = g[_node_][_nbor_].get('weight', 0)
            else:
                _weight_ = 0
            _int_sorter_.append((_deg_, _weight_, _node_))
        _int_sorter_ = sorted(_int_sorter_, reverse=True)
        for _tuple_ in _int_sorter_: placeNodeIntoClosestSlot(_tuple_[2])

        return adj_pos

    def circularOptimizedLayout(self, g, nodes, pos, xy=(0.0, 0.0), r=1.0):
        adj_pos, as_set  = {}, set(nodes)

        _externals_, _internals_ = set(), set()
        for _node_ in nodes:
            if _node_ in g:
                for _nbor_ in g.neighbors(_node_):
                    if _nbor_ not in as_set:
                        _externals_.add(_node_)
                        break
            if _node_ not in _externals_: _internals_.add(_node_)

        _filled_with_, _angulars_, _angular_locations_ = [], [], []
        for i in range(len(nodes)):
            _angle_ = i * 2 * pi / len(nodes)
            _filled_with_.append(None)
            _angulars_.append(_angle_)
            _angular_locations_.append((xy[0] + r * cos(_angle_), xy[1] + r * sin(_angle_)))

        def placeNodeIntoClosestSlot(node_to_place, nodes_xy=None):
            _closest_ = 0
            if nodes_xy is None: _closest_ = random.randint(0, len(_angulars_)-1)  # nosec B311 - non-cryptographic layout slot pick
            else:
                _closest_, _closest_d_ = 0, 1e9
                for i in range(0, len(_angular_locations_)):
                    dx, dy = nodes_xy[0] - _angular_locations_[i][0], nodes_xy[1] - _angular_locations_[i][1]
                    d      = sqrt(dx*dx + dy*dy)
                    if d < _closest_d_: _closest_, _closest_d_ = i, d
            if _filled_with_[_closest_] is None:
                _filled_with_[_closest_] = node_to_place
                adj_pos[node_to_place]   = _angular_locations_[_closest_]
            else:
                for j in range(1, len(_angulars_)):
                    up =  (_closest_+j)                  % len(_angulars_)
                    dn = ((_closest_-j)+len(_angulars_)) % len(_angulars_)
                    if _filled_with_[up] is None:
                        _filled_with_[up]      = node_to_place
                        adj_pos[node_to_place] = _angular_locations_[up]
                        break
                    elif _filled_with_[dn] is None:
                        _filled_with_[dn]      = node_to_place
                        adj_pos[node_to_place] = _angular_locations_[dn]
                        break

        for _node_ in _externals_:
            _x_sum_, _y_sum_, _samples_ = 0.0, 0.0, 0
            for _nbor_ in g.neighbors(_node_):
                if _nbor_ not in _internals_ and _nbor_ not in _externals_:
                    _nbor_xy_  =  pos[_nbor_]
                    _x_sum_   += _nbor_xy_[0]
                    _y_sum_   += _nbor_xy_[1]
                    _samples_ += 1
            _x_, _y_ = _x_sum_ / _samples_, _y_sum_ / _samples_
            placeNodeIntoClosestSlot(_node_, (_x_, _y_))

        _sorter_ = []
        for _node_ in _internals_:
            _externally_connected_ = 0
            if _node_ in g:
                for _nbor_ in g.neighbors(_node_):
                    if _nbor_ in _externals_: _externally_connected_ += 1
            _sorter_.append((_externally_connected_, _node_))
        _sorter_ = sorted(_sorter_, reverse=True)
        for _tuple_ in _sorter_:
            _node_ = _tuple_[1]
            if _tuple_[0] > 0:
                _x_sum_, _y_sum_, _samples_ = 0.0, 0.0, 0
                for _nbor_ in g.neighbors(_node_):
                    if _nbor_ in _filled_with_:
                        i          =  _filled_with_.index(_nbor_)
                        _x_sum_   += _angular_locations_[i][0]
                        _y_sum_   += _angular_locations_[i][1]
                        _samples_ += 1
                _x_, _y_ = _x_sum_ / _samples_, _y_sum_ / _samples_
                placeNodeIntoClosestSlot(_node_, (_x_, _y_))
            else:
                placeNodeIntoClosestSlot(_node_)

        return adj_pos

    def hyperTreeLayout(self, _graph, roots=None, bounds_percent=0.1):
        _requireGraphLayoutDeps_()
        if roots is not None and isinstance(roots, list) == False: roots = list(roots)

        _graph = nx.to_undirected(_graph)
        S = [_graph.subgraph(c).copy() for c in nx.connected_components(_graph)]

        pos = {}
        for _subgraph in S:
            G = nx.to_undirected(nx.minimum_spanning_tree(_subgraph))

            if len(G) <= 4:
                as_list = list(G.nodes())
                if len(G) >= 1: pos[as_list[0]] = (0, 0)
                if len(G) >= 2: pos[as_list[1]] = (1, 1)
                if len(G) >= 3: pos[as_list[2]] = (1, 0)
                if len(G) >= 4: pos[as_list[3]] = (0, 1)
                continue

            my_root = None
            if roots is not None:
                for possible_root in roots:
                    if possible_root in G:
                        my_root = possible_root

            if my_root is None:
                f = G.copy()
                while len(f) > 2:
                    to_be_removed = [x for x in f.nodes() if f.degree(x) <= 1]
                    f.remove_nodes_from(to_be_removed)
                my_root = list(f)[0]

            _leaf_count = {}
            self.__p2sg_totalLeaves__(G, None, my_root, _leaf_count)
            _max_depth = self.__p2sg_treeDepth__(G, None, my_root)

            _R_ = 8.0
            # Top-down wedge assignment: each node owns a contiguous angular sector
            # [_start, _end] and is placed at its midpoint — guarantees disjoint sectors
            # across siblings, which eliminates all edge crossings in the spanning tree.
            def placeSubtree(_parent, _node, _depth, _start, _end):
                _r = _depth * _R_ / _max_depth
                _mid = (_start + _end) / 2.0
                pos[_node] = (_r * cos(_mid), _r * sin(_mid))
                children = [x for x in G[_node] if x != _parent]
                if not children:
                    return
                total = sum(max(1, _leaf_count.get(c, 0)) for c in children)
                _cur = _start
                for child in sorted(children, key=lambda x: _leaf_count.get(x, 0), reverse=True):
                    frac = max(1, _leaf_count.get(child, 0)) / total
                    child_end = _cur + frac * (_end - _start)
                    placeSubtree(_node, child, _depth + 1, _cur, child_end)
                    _cur = child_end

            placeSubtree(None, my_root, 0, 0.0, 2.0 * pi)

        if len(S) > 1: return self.treeMapLayout(_graph, pos, bounds_percent)
        else:          return pos

    def hyperTreeDonutLayout(self, _graph, roots=None, bounds_percent=0.1,
                             inner_frac=0.55, parent_gap_frac=0.12,
                             return_cells=False):
        """Donut-chart variant of :meth:`hyperTreeLayout`.

        Internal (non-leaf) nodes are placed on the inner disk by depth, exactly
        like ``hyperTreeLayout``. Leaves, however, are packed into the *outer
        ring band* (the "donut edge"), filling the angular slice that the tree
        allots to their parent across the full thickness of the ring -- so the
        leaves of one parent read as one slice of the donut.

        Each leaf-parent sits at the *center of its donut wedge* (the mid-radius,
        mid-angle of the ring band), with a small clearance (``parent_gap_frac``)
        carved out of the surrounding leaves so the parent stays distinct within
        the slice. The root stays at the center of the layout (the donut hole).

        ``inner_frac``       fraction of the outer radius where the donut ring (and
                             the leaf packing) begins; the hole below it holds the root.
        ``parent_gap_frac``  clearance carved around each leaf-parent, as a fraction of
                             the ring-band thickness, so the parent reads as distinct.

        When ``return_cells`` is ``True`` the method returns ``(pos, cells)``
        instead of just ``pos`` (mirroring :meth:`ipSubnetTreeMapLayout`).
        ``cells`` is a ``{leaf_parent_node: shapely_polygon}`` dict, one *annular
        sector* (donut-edge slice) per leaf-parent spanning that parent's angular
        sector across the ring band ``[inner_frac*R, R]`` -- ready to hand to
        linkp's ``background=`` parameter as the donut-slice background layer.
        Because the sector starts at the inner radius (not the layout center) it
        reads as a donut edge rather than a pie wedge. When ``return_cells`` is
        ``False`` (the default) only ``pos`` is returned, so existing callers are
        unaffected.
        """
        _requireGraphLayoutDeps_()
        if roots is not None and isinstance(roots, list) == False: roots = list(roots)

        _graph = nx.to_undirected(_graph)
        S = [_graph.subgraph(c).copy() for c in nx.connected_components(_graph)]

        # Per-subgraph slice specs (leaf-parent -> angular wedge), resolved into
        # shapely polygons after final positions are known. Each entry carries the
        # subgraph's node set so its polygon can ride the treeMapLayout transform.
        _cell_specs_ = []

        pos = {}
        for _subgraph in S:
            G = nx.to_undirected(nx.minimum_spanning_tree(_subgraph))

            if len(G) <= 4:
                as_list = list(G.nodes())
                if len(G) >= 1: pos[as_list[0]] = (0, 0)
                if len(G) >= 2: pos[as_list[1]] = (1, 1)
                if len(G) >= 3: pos[as_list[2]] = (1, 0)
                if len(G) >= 4: pos[as_list[3]] = (0, 1)
                continue

            my_root = None
            if roots is not None:
                for possible_root in roots:
                    if possible_root in G:
                        my_root = possible_root

            if my_root is None:
                f = G.copy()
                while len(f) > 2:
                    to_be_removed = [x for x in f.nodes() if f.degree(x) <= 1]
                    f.remove_nodes_from(to_be_removed)
                my_root = list(f)[0]

            _leaf_count = {}
            self.__p2sg_totalLeaves__(G, None, my_root, _leaf_count)
            _max_depth = self.__p2sg_treeDepth__(G, None, my_root)

            _R_        = 8.0                       # outer radius (donut edge)
            _inner_    = inner_frac * _R_          # inner radius of the donut ring
            _clear_    = parent_gap_frac * (_R_ - _inner_)  # clearance around a leaf-parent

            # Pack a group of leaves into the annular sector [r_inner, r_outer] x
            # [a_start, a_end], roughly square cells, filling the band's thickness.
            def packLeaves(_leaves, _a_start, _a_end, _r_inner, _r_outer):
                _n = len(_leaves)
                if _n == 0: return
                _dtheta = _a_end   - _a_start
                _dr     = _r_outer - _r_inner
                _r_mid  = (_r_inner + _r_outer) / 2.0
                _arc    = max(1e-6, _r_mid * _dtheta)
                _aspect = (_arc / _dr) if _dr > 1e-6 else float(_n)   # cols per row
                _rows   = max(1, int(round(sqrt(max(1.0, _n / max(_aspect, 1e-6))))))
                _cols   = int(ceil(_n / _rows))
                for _k, _leaf in enumerate(_leaves):
                    _i      = _k // _cols                       # radial row
                    _j      = _k %  _cols                       # angular column
                    _in_row = _cols if (_i + 1) * _cols <= _n else (_n - _i * _cols)
                    _r      = _r_inner + (_i + 0.5) * _dr / _rows
                    _theta  = _a_start + (_j + 0.5) * _dtheta / _in_row
                    pos[_leaf] = (_r * cos(_theta), _r * sin(_theta))

            # Place a leaf-parent at the center of its donut wedge, then pack its
            # leaves across the band and push any leaf landing within _clear_ of the
            # parent out to that clearance boundary -- leaving the parent distinct.
            def packLeavesAndParent(_node, _leaves, _a_start, _a_end, _r_inner, _r_outer):
                _r_mid = (_r_inner + _r_outer) / 2.0
                _a_mid = (_a_start + _a_end) / 2.0
                pos[_node] = (_r_mid * cos(_a_mid), _r_mid * sin(_a_mid))
                if not _leaves: return
                _px, _py = pos[_node]
                packLeaves(_leaves, _a_start, _a_end, _r_inner, _r_outer)
                for _leaf in _leaves:
                    _lx, _ly = pos[_leaf]
                    _dx, _dy = _lx - _px, _ly - _py
                    _d = sqrt(_dx * _dx + _dy * _dy)
                    if _d < _clear_:
                        if _d < 1e-9: _dx, _dy, _d = cos(_a_mid), sin(_a_mid), 1.0
                        pos[_leaf] = (_px + _dx / _d * _clear_, _py + _dy / _d * _clear_)

            # Top-down wedge assignment (as in hyperTreeLayout). Internal children
            # recurse within the inner disk; leaf children of a node are packed as
            # one group into the ring band over the wedge that remains for them.
            def placeSubtree(_parent, _node, _depth, _start, _end):
                _mid      = (_start + _end) / 2.0
                children  = [x for x in G[_node] if x != _parent]
                inner_ch  = [c for c in children if any(x != _node for x in G[c])]
                leaf_ch   = [c for c in children if not any(x != _node for x in G[c])]
                is_leaf_parent = bool(leaf_ch) and _depth > 0

                # Non-leaf-parents (root and pure internal nodes) sit on the inner
                # disk by depth; leaf-parents are positioned by packLeavesAndParent.
                if not is_leaf_parent:
                    _r = (_depth * _inner_ / _max_depth) if _max_depth > 0 else 0.0
                    pos[_node] = (_r * cos(_mid), _r * sin(_mid))

                if not children:
                    return

                total = sum(max(1, _leaf_count.get(c, 0)) for c in children)
                _cur  = _start
                for child in sorted(inner_ch, key=lambda x: _leaf_count.get(x, 0), reverse=True):
                    frac      = max(1, _leaf_count.get(child, 0)) / total
                    child_end = _cur + frac * (_end - _start)
                    placeSubtree(_node, child, _depth + 1, _cur, child_end)
                    _cur = child_end

                # Remaining wedge [_cur, _end] is the parent's donut slice for its leaves.
                if leaf_ch:
                    if is_leaf_parent: packLeavesAndParent(_node, leaf_ch, _cur, _end, _inner_, _R_)
                    else:              packLeaves(leaf_ch, _cur, _end, _inner_, _R_)
                    if return_cells:
                        _slice_wedges_.append((_node, _cur, _end, _inner_, _R_))

            _slice_wedges_ = []
            placeSubtree(None, my_root, 0, 0.0, 2.0 * pi)
            if return_cells and _slice_wedges_:
                _cell_specs_.append((set(G.nodes()), _slice_wedges_))

        if len(S) > 1:
            _pos_local_ = dict(pos)
            new_pos     = self.treeMapLayout(_graph, pos, bounds_percent)
            if not return_cells:
                return new_pos
            cells = self.__p2sg_buildDonutCells__(_cell_specs_, _pos_local_, new_pos)
            return new_pos, cells

        if not return_cells:
            return pos
        cells = self.__p2sg_buildDonutCells__(_cell_specs_, pos, None)
        return pos, cells

    def __p2sg_buildDonutCells__(self, cell_specs, pos_local, new_pos):
        """Resolve donut-slice wedge specs into shapely polygons.

        ``new_pos is None`` -> single component, polygons are built directly in
        the (origin-centered) layout coordinates. Otherwise treeMapLayout has
        repositioned each subgraph with an axis-aligned affine; the same affine
        (derived from each subgraph's local vs. final node positions) is applied
        to its polygons so the slices stay aligned with the nodes.
        """
        from shapely.geometry  import Polygon
        from shapely.affinity  import affine_transform

        def _annular_sector_(_a0, _a1, _r_in, _r_out, _steps=32):
            # Donut-edge slice: outer arc a0->a1, then inner arc a1->a0 back.
            _n_   = max(2, _steps)
            _pts_ = []
            for _k in range(_n_ + 1):
                _a_ = _a0 + (_a1 - _a0) * _k / _n_
                _pts_.append((_r_out * cos(_a_), _r_out * sin(_a_)))
            for _k in range(_n_ + 1):
                _a_ = _a1 - (_a1 - _a0) * _k / _n_
                _pts_.append((_r_in * cos(_a_), _r_in * sin(_a_)))
            return Polygon(_pts_)

        def _affine_(_nodes_):
            # Solve an axis-aligned affine (sx, sy, tx, ty) from local -> final
            # positions: final_x = sx*local_x + tx, final_y = sy*local_y + ty.
            _ns_ = [n for n in _nodes_ if n in pos_local and n in new_pos]
            _sx_ = _sy_ = None
            _tx_ = _ty_ = 0.0
            for i in range(len(_ns_)):
                for j in range(i + 1, len(_ns_)):
                    _a_, _b_ = _ns_[i], _ns_[j]
                    if _sx_ is None and abs(pos_local[_a_][0] - pos_local[_b_][0]) > 1e-9:
                        _sx_ = (new_pos[_a_][0] - new_pos[_b_][0]) / (pos_local[_a_][0] - pos_local[_b_][0])
                        _tx_ = new_pos[_a_][0] - _sx_ * pos_local[_a_][0]
                    if _sy_ is None and abs(pos_local[_a_][1] - pos_local[_b_][1]) > 1e-9:
                        _sy_ = (new_pos[_a_][1] - new_pos[_b_][1]) / (pos_local[_a_][1] - pos_local[_b_][1])
                        _ty_ = new_pos[_a_][1] - _sy_ * pos_local[_a_][1]
                if _sx_ is not None and _sy_ is not None: break
            if _sx_ is None or _sy_ is None: return None
            return [_sx_, 0.0, 0.0, _sy_, _tx_, _ty_]

        cells = {}
        for _nodes_, _wedges_ in cell_specs:
            _matrix_ = None if new_pos is None else _affine_(_nodes_)
            for _node_, _a0_, _a1_, _r_in_, _r_out_ in _wedges_:
                _poly_ = _annular_sector_(_a0_, _a1_, _r_in_, _r_out_)
                if _matrix_ is not None: _poly_ = affine_transform(_poly_, _matrix_)
                cells[_node_] = _poly_
        return cells

    # --- private helpers for neighborhoodLayout ---

    # A gap is a real neighborhood divide only if cutting there splits the points
    # into at least two *substantial* clusters. "Substantial" = at least this
    # fraction of the nodes (floored at 2 points). Robust band [0.02, 0.05].
    _P2SG_MIN_CLUSTER_FRAC = 0.02

    def __p2sg_singleLinkageGap__(self, pts):
        """Auto distance-threshold for single-linkage spatial neighborhoods.

        Single-linkage merges == the edges of the points' minimum spanning tree,
        taken in increasing length order. The natural place to cut is the largest
        *gap* between consecutive MST edge lengths: short edges live inside a
        neighborhood, the long edges bridge between them. We cut at that gap only
        if doing so actually **divides** the points -- i.e. cutting every MST edge
        longer than the gap leaves at least two clusters of size >= max(2,
        ``_P2SG_MIN_CLUSTER_FRAC`` * N). Otherwise return ``inf`` (one neighborhood).

        Why validate the cut's *outcome* rather than score the gap: real layouts are
        multi-cluster, so the bulk->bridge transition is a modest step (the VAST netflow
        graphs cut at only ~1.3-1.9x the prior edge) that sits under any fixed
        relative-step or "gap >> within-cluster scale" guard -- those guards collapse
        12 genuine neighborhoods into one, and tuning their constant runs straight into
        the single-blob case (whose largest gap scores nearly identically). The outcome
        instead separates cleanly: a true divide yields two big clusters (2nd-largest is
        ~13-50% of N here), whereas the largest gap in a smooth single blob / uniform
        cloud merely peels a stray point or two off the tail (2nd-largest ~1% of N), so
        the substantial-cluster test rejects it. That margin is ~10x, not knife-edge.
        """
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import minimum_spanning_tree, connected_components

        _n_ = len(pts)
        if _n_ < 2:
            return inf
        _D_   = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
        _mst_ = minimum_spanning_tree(_D_)                  # sparse upper-triangular tree
        _w_   = np.sort(_mst_.data[_mst_.data > 0.0])
        if _w_.size < 2:
            return inf  # 0 or 1 positive edge -> nothing to separate
        # Largest absolute jump: the bulk->bridge boundary (cutting here never
        # shatters a smooth tail, unlike picking the largest *relative* step).
        _i_   = int(np.argmax(_w_[1:] - _w_[:-1]))
        _thr_ = float((_w_[_i_] + _w_[_i_ + 1]) / 2.0)
        # Components after cutting every MST edge longer than the gap.
        _coo_  = _mst_.tocoo()
        _keep_ = _coo_.data <= _thr_
        _sub_  = csr_matrix((np.ones(int(_keep_.sum())),
                             (_coo_.row[_keep_], _coo_.col[_keep_])), shape=(_n_, _n_))
        _sizes_    = np.bincount(connected_components(_sub_, directed=False)[1])
        _min_size_ = max(2, int(np.ceil(self._P2SG_MIN_CLUSTER_FRAC * _n_)))
        if int((_sizes_ >= _min_size_).sum()) < 2:
            return inf  # cut only peels strays -> no real divide
        return _thr_

    def __p2sg_voronoiFinitePolygons2D__(self, points, radius=None):
        """Reconstruct finite Voronoi regions in 2D (standard recipe).

        ``scipy.spatial.Voronoi`` leaves boundary regions open (vertices at
        infinity). This closes every region by projecting each infinite ridge to
        a far point ``radius`` away, so each input point gets a finite polygon.
        Returns ``(regions, vertices)`` where ``regions[i]`` is the vertex-index
        list (CCW) of the cell for input ``points[i]``.
        """
        from scipy.spatial import Voronoi

        vor = Voronoi(points)
        new_regions  = []
        new_vertices = vor.vertices.tolist()
        center = vor.points.mean(axis=0)
        if radius is None: radius = np.ptp(vor.points, axis=0).max() * 2.0

        # Map each input point to the ridges (and their vertices) around it.
        all_ridges = {}
        for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
            all_ridges.setdefault(p1, []).append((p2, v1, v2))
            all_ridges.setdefault(p2, []).append((p1, v1, v2))

        for p1, region_idx in enumerate(vor.point_region):
            vertices = vor.regions[region_idx]
            if all(v >= 0 for v in vertices):
                new_regions.append(vertices)               # already finite
                continue

            new_region = [v for v in vertices if v >= 0]
            for p2, v1, v2 in all_ridges.get(p1, []):
                if v2 < 0: v1, v2 = v2, v1
                if v1 >= 0: continue                        # finite ridge -- skip
                # Project the open ridge to a far point along its outward normal.
                t = vor.points[p2] - vor.points[p1]
                t = t / np.linalg.norm(t)
                n = np.array([-t[1], t[0]])
                midpoint  = vor.points[[p1, p2]].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n
                far_point = vor.vertices[v2] + direction * radius
                new_region.append(len(new_vertices))
                new_vertices.append(far_point.tolist())

            # Order the region's vertices counter-clockwise for a valid polygon.
            vs = np.asarray([new_vertices[v] for v in new_region])
            c  = vs.mean(axis=0)
            angles     = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
            new_region = np.array(new_region)[np.argsort(angles)]
            new_regions.append(new_region.tolist())

        return new_regions, np.asarray(new_vertices)

    def __p2sg_voronoiNeighborhoodCells__(self, pos, node_labels, pad=0.05):
        """Tessellate node positions into one polygon per neighborhood.

        Each node's (clipped) Voronoi cell is assigned to its neighborhood label;
        the per-neighborhood union of those cells is the neighborhood outline --
        the Voronoi medial-axis of the gaps between neighborhoods. Cells are
        clipped to the padded layout bounding box so boundary neighborhoods stay
        finite. Falls back to a buffered convex hull per neighborhood when there
        are too few distinct points for a stable Voronoi. Returns
        ``{label: shapely_polygon}`` ready for linkp's ``background=``.
        """
        from shapely.geometry import Polygon, MultiPoint
        from shapely.geometry import box as _box_
        from shapely.ops       import unary_union

        _nodes_ = [n for n in node_labels if n in pos and node_labels[n] is not None]
        if len(_nodes_) == 0: return {}

        _pts_    = np.array([[pos[n][0], pos[n][1]] for n in _nodes_], dtype=float)
        _labels_ = [node_labels[n] for n in _nodes_]

        _minx_, _miny_ = _pts_[:, 0].min(), _pts_[:, 1].min()
        _maxx_, _maxy_ = _pts_[:, 0].max(), _pts_[:, 1].max()
        _spanx_ = (_maxx_ - _minx_) or 1.0
        _spany_ = (_maxy_ - _miny_) or 1.0
        _px_, _py_ = _spanx_ * pad, _spany_ * pad
        _bbox_ = _box_(_minx_ - _px_, _miny_ - _py_, _maxx_ + _px_, _maxy_ + _py_)

        _label_to_polys_ = {}
        _use_voronoi_ = len(np.unique(_pts_, axis=0)) >= 4
        if _use_voronoi_:
            try:
                _regions_, _vertices_ = self.__p2sg_voronoiFinitePolygons2D__(_pts_)
                for i, _region_ in enumerate(_regions_):
                    _poly_ = Polygon(_vertices_[_region_]).intersection(_bbox_)
                    if _poly_.is_empty: continue
                    _label_to_polys_.setdefault(_labels_[i], []).append(_poly_)
            except Exception:
                _use_voronoi_ = False

        if not _use_voronoi_:
            _buf_ = 0.05 * (_spanx_ + _spany_) / 2.0
            _grp_ = {}
            for i in range(len(_nodes_)): _grp_.setdefault(_labels_[i], []).append(_pts_[i])
            for _lab_, _ps_ in _grp_.items():
                _hull_ = MultiPoint([tuple(p) for p in _ps_]).buffer(_buf_).intersection(_bbox_)
                if not _hull_.is_empty: _label_to_polys_[_lab_] = [_hull_]

        cells = {}
        for _lab_, _polys_ in _label_to_polys_.items():
            _u_ = unary_union(_polys_)
            if not _u_.is_empty: cells[_lab_] = _u_
        return cells

    # --- private helpers for hyperTreeLayout ---

    def __p2sg_countSubTreeNodes__(self, _graph, _node, _ignore, _child_count):
        if _node in _child_count.keys():
            return _child_count[_node] + 1
        _sum = 0
        for x in _graph[_node]:
            if x == _ignore: continue
            _sum += self.__p2sg_countSubTreeNodes__(_graph, x, _node, _child_count)
        if _child_count is not None: _child_count[_node] = _sum
        return _sum + 1

    def __p2sg_totalLeaves__(self, _graph, _parent, _node, _leaf_count):
        children = [x for x in _graph[_node] if x != _parent]
        if not children:
            _leaf_count[_node] = 0
            return 1
        _sum = sum(self.__p2sg_totalLeaves__(_graph, _node, x, _leaf_count) for x in children)
        _leaf_count[_node] = _sum
        return _sum

    def __p2sg_treeDepth__(self, _graph, _parent, _node):
        children = [x for x in _graph[_node] if x != _parent]
        if not children:
            return 1
        return 1 + max(self.__p2sg_treeDepth__(_graph, _node, x) for x in children)

    def savePositions(self, filename, linkp):
        with open(filename, "w") as f:
            json.dump(linkp.pos, f, indent=2)

    def loadPositions(self, filename, linkp=None):
        with open(filename, "r") as f:
            pos = json.load(f)
        if linkp is not None and hasattr(linkp, 'all_nodes'):
            file_nodes  = set(pos.keys())
            linkp_nodes = linkp.all_nodes
            diff        = file_nodes.symmetric_difference(linkp_nodes)
            if diff:
                in_file_only  = file_nodes  - linkp_nodes
                in_linkp_only = linkp_nodes - file_nodes
                self.logger.warning(
                    f"{len(diff)} node(s) differ between position file and linkp "
                    f"(in file only: {len(in_file_only)}, in linkp only: {len(in_linkp_only)})")
        return pos
