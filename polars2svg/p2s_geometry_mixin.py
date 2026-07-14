import polars as pl
import math
import random
from math import sqrt

from .circle_packer import CirclePacker
from .udist_scatterplots_via_sectors_tile_opt import UDistScatterPlotsViaSectorsTileOpt
from .exceptions import DataError

__name__ = 'p2s_geometry_mixin'

class P2SGeometryMixin:
    def __init__(self):
        pass

    def __p2s_geometry_mixin_init__(self):
        pass

    #
    # heckbertNiceNumbers() - implements Heckbert (Nice Numbers) Algorithm
    # - P. Heckbert. Nice numbers for graph labels. In A. Glassner, editor, Graphics Gems, pages 61–63.
    #   Academic Press, Boston, 1990.
    #
    def heckbertNiceNumbers(self, n_ticks, _min_, _max_):
        # Handle the degenerate case (part i)
        if n_ticks < 1: n_ticks = 1
        # Handle the degenerate case (part ii)
        if abs(_max_ - _min_) < 1e-6:
            if _min_ == 0.0:
                _min_, _max_ = -1.0, 1.0
            else:
                _margin_ = abs(_min_) * 0.1
                _min_, _max_ = _min_ - _margin_, _max_ + _margin_
        # Proceed w/ Heckbert's algorithm
        I_raw   = (_max_ - _min_)/n_ticks
        E           = math.floor(math.log10(I_raw))
        Base        = I_raw / 10**E
        _base_nice_ = 1
        for _base_ in [2, 5, 10]:
            if abs(_base_ - Base) < abs(_base_nice_ - Base): _base_nice_ = _base_
        I_nice = _base_nice_ * 10**E
        return I_nice

    def heckbertFirstGridLine(self, i_nice, _min_, _max_):
        return math.floor(_min_ / i_nice) * i_nice

    # -------------------------------------------------------------------------
    # Ported from rtsvg/rt_graph_layouts_mixin.py (David Trimm, Apache 2.0)
    # -------------------------------------------------------------------------

    def positionExtents(self, pos, _graph=None):
        x0 = y0 = x1 = y1 = None
        if _graph is not None:
            from_structure = set(_graph.nodes())
            in_pos         = set(pos.keys())
            if len(in_pos & from_structure) != len(from_structure):
                raise DataError('positionExtents(): missing keys in position dictionary')
        else:
            from_structure = set(pos.keys())
        for _node in from_structure:
            x = pos[_node][0]
            y = pos[_node][1]
            x0 = x if x0 is None else min(x0, x)
            y0 = y if y0 is None else min(y0, y)
            x1 = x if x1 is None else max(x1, x)
            y1 = y if y1 is None else max(y1, y)
        if x0 == x1: x0, x1 = x0 - 0.5, x1 + 0.5
        if y0 == y1: y0, y1 = y0 - 0.5, y1 + 0.5
        return x0, y0, x1, y1

    # -------------------------------------------------------------------------
    # Ported from rtsvg/rt_geometry_mixin.py (David Trimm, Apache 2.0)
    # -------------------------------------------------------------------------

    def segmentLength(self, _segment_):
        dx, dy = _segment_[1][0] - _segment_[0][0], _segment_[1][1] - _segment_[0][1]
        return sqrt(dx*dx+dy*dy)

    def unitVector(self, _segment_):
        dx, dy = _segment_[1][0] - _segment_[0][0], _segment_[1][1] - _segment_[0][1]
        _len_  = sqrt(dx*dx+dy*dy)
        if _len_ < 0.0001: _len_ = 1.0
        return (dx/_len_, dy/_len_)

    def circlesOverlap(self, c0, c1):
        return (c0[0] - c1[0])**2 + (c0[1] - c1[1])**2 < (c0[2] + c1[2])**2

    def overlappingCirclesIntersections(self, c0, c1):
        R, r  = c0[2], c1[2]
        d  = self.segmentLength((c0, c1))
        if d == 0.0: raise DataError('overlappingCirclesIntersections(): circles have the same center')
        x  = (d**2 - r**2 + R**2)/(2.0*d)
        a  = (1.0/d) * sqrt(4.0*d**2 * R**2 - (d**2 - r**2 + R**2)**2)
        uv = self.unitVector((c0, c1))
        pp = (-uv[1], uv[0])
        return (c0[0] + uv[0]*x + pp[0] * (a/2.0), c0[1] + uv[1]*x + pp[1] * (a/2.0)), \
               (c0[0] + uv[0]*x - pp[0] * (a/2.0), c0[1] + uv[1]*x - pp[1] * (a/2.0))

    def intersectionPoint(self, line1, line2):
        def floatify(line): return (float(line[0][0]), float(line[0][1])), (float(line[1][0]), float(line[1][1]))
        line1, line2 = floatify(line1), floatify(line2)
        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])
        def det(a, b): return a[0] * b[1] - a[1] * b[0]
        div = det(xdiff, ydiff)
        if abs(div) < 0.0001 or div == 0: return None
        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return x, y

    def lineSegmentIntersectionPoint(self, line, segment):
        def floatify(line): return (float(line[0][0]), float(line[0][1])), (float(line[1][0]), float(line[1][1]))
        line, segment = floatify(line), floatify(segment)
        results = self.intersectionPoint(line, segment)
        if results is None: return None
        x, y = results
        if x >= min(segment[0][0], segment[1][0]) and x <= max(segment[0][0], segment[1][0]) and \
           y >= min(segment[0][1], segment[1][1]) and y <= max(segment[0][1], segment[1][1]): return x, y
        dx, dy = segment[1][0] - segment[0][0], segment[1][1] - segment[0][1]
        if abs(dx) >= 0.000001:
            t = (x - segment[0][0]) / dx
            if t >= 0.0 and t <= 1.0: return x, segment[0][1] + t * dy
        if abs(dy) >= 0.000001:
            t = (y - segment[0][1]) / dy
            if t >= 0.0 and t <= 1.0: return segment[0][0] + t * dx, y
        return None

    def closestPointOnSegment(self, _segment_, _pt_):
        if _segment_[0][0] == _segment_[1][0] and _segment_[0][1] == _segment_[1][1]:
            dx, dy = _pt_[0] - _segment_[0][0], _pt_[1] - _segment_[0][1]
            return sqrt(dx*dx+dy*dy), _segment_[0]
        else:
            dx, dy = _pt_[0] - _segment_[0][0], _pt_[1] - _segment_[0][1]
            d0 = dx*dx+dy*dy
            dx, dy = _pt_[0] - _segment_[1][0], _pt_[1] - _segment_[1][1]
            d1 = dx*dx+dy*dy
            dx,  dy  = _segment_[1][0] - _segment_[0][0], _segment_[1][1] - _segment_[0][1]
            pdx, pdy = -dy, dx
            _pt_line_ = (_pt_, (_pt_[0] + pdx, _pt_[1] + pdy))
            _ret_ = self.lineSegmentIntersectionPoint(_pt_line_, _segment_)
            if _ret_ is not None:
                dx, dy = _pt_[0] - _ret_[0], _pt_[1] - _ret_[1]
                d2 = dx*dx+dy*dy
                if    d2 < d0 and d2 < d1: return sqrt(d2), _ret_
                elif  d0 < d1:             return sqrt(d0), _segment_[0]
                else:                      return sqrt(d1), _segment_[1]
            else:
                if    d0 < d1:             return sqrt(d0), _segment_[0]
                else:                      return sqrt(d1), _segment_[1]

    def smallestEnclosingCircleApprox(self, points):
        _sec_ = SmallestEnclosingCircle(points)
        return (_sec_.center[0], _sec_.center[1], _sec_.radius)

    def packCircles(self, circles, into_circle=None):
        _cp_      = CirclePacker(self, circles)
        _circles_ = _cp_.packedCircles(into_circle)
        return _circles_

    # -------------------------------------------------------------------------
    # Ported from rtsvg/rt_geometry_mixin.py (David Trimm, Apache 2.0)
    # Original paper: Rave et al., "Uniform Sample Distribution in Scatterplots
    # via Sector-based Transformation", IEEE VIS 2024.
    # -------------------------------------------------------------------------

    def uniformSampleDistributionInScatterplotsViaSectorBasedTransformation(
            self, df, x_field, y_field, weight_field=None, static_field=None,
            iterations=32, vector_scalar=0.01):
        _weights_, _statics_ = None, None
        if weight_field is not None: _weights_ = df[weight_field]
        if static_field is not None: _statics_ = df[static_field]
        x0, y0, x1, y1     = df[x_field].min(), df[y_field].min(), df[x_field].max(), df[y_field].max()
        udspvsto            = UDistScatterPlotsViaSectorsTileOpt(
                                df[x_field], df[y_field],
                                weights=_weights_, static_points=_statics_,
                                vector_scalar=vector_scalar, iterations=iterations)
        x_vals, y_vals      = udspvsto.results()
        df                  = df.with_columns(pl.Series(x_field, x_vals), pl.Series(y_field, y_vals))
        x0_, y0_, x1_, y1_  = df[x_field].min(), df[y_field].min(), df[x_field].max(), df[y_field].max()
        # x1_ == x0_ (no spread in the transformed result, e.g. a single point) would otherwise divide by zero
        x_expr = pl.lit(x0).alias(x_field) if x1_ == x0_ else ((pl.col(x_field) - x0_)/(x1_ - x0_) * (x1 - x0) + x0).alias(x_field)
        y_expr = pl.lit(y0).alias(y_field) if y1_ == y0_ else ((pl.col(y_field) - y0_)/(y1_ - y0_) * (y1 - y0) + y0).alias(y_field)
        df = df.with_columns(x_expr, y_expr)
        return df


# -------------------------------------------------------------------------
# Ported from rtsvg/rt_geometry_mixin.py (David Trimm, Apache 2.0)
# Welzl's algorithm — iterative, O(n) expected.
# -------------------------------------------------------------------------

class SmallestEnclosingCircle(object):
    def __init__(self, points):
        if not points:
            self.center, self.radius = (0, 0), 0
            return
        shuffled = points.copy()
        random.shuffle(shuffled)
        circle = (shuffled[0], 0)
        for i in range(1, len(shuffled)):
            if not self._is_inside(shuffled[i], circle):
                circle = self._min_circle_with_point(shuffled[:i+1], shuffled[i])
        self.center = circle[0]
        self.radius = circle[1]

    def _min_circle_with_point(self, points, p):
        circle = (p, 0)
        for i in range(len(points)):
            if not self._is_inside(points[i], circle):
                circle = self._min_circle_with_2_points(points[:i+1], p, points[i])
        return circle

    def _min_circle_with_2_points(self, points, p1, p2):
        circle = self._circle_from_2_points(p1, p2)
        for i in range(len(points)):
            if not self._is_inside(points[i], circle):
                circle = self._circle_from_3_points(p1, p2, points[i])
        return circle

    def _circle_from_2_points(self, p1, p2):
        cx = (p1[0] + p2[0]) / 2
        cy = (p1[1] + p2[1]) / 2
        r  = sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) / 2
        return ((cx, cy), r)

    def _circle_from_3_points(self, p1, p2, p3):
        ax, ay = p1
        bx, by = p2
        cx, cy = p3
        d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(d) < 1e-10:
            return self._furthest_pair_circle([p1, p2, p3])
        ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / d
        uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / d
        r  = sqrt((ax - ux)**2 + (ay - uy)**2)
        return ((ux, uy), r)

    def _furthest_pair_circle(self, points):
        max_dist = 0
        pair = (points[0], points[1])
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                dist = sqrt((points[i][0] - points[j][0])**2 + (points[i][1] - points[j][1])**2)
                if dist > max_dist:
                    max_dist = dist
                    pair = (points[i], points[j])
        return self._circle_from_2_points(pair[0], pair[1])

    def _is_inside(self, point, circle):
        (cx, cy), r = circle
        dist = sqrt((point[0] - cx)**2 + (point[1] - cy)**2)
        return dist <= r + 1e-10
