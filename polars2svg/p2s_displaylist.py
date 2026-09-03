from typing import Any
#
# DisplayList - backend-neutral primitive recording for dual SVG / WebGPU rendering
#
# Every draw call records (svg_string, primitive_record) pairs.  The SVG string is
# supplied verbatim by the existing render code (via svg= / svg_col=), keeping the
# SVG output byte-identical to the pre-DisplayList implementation.  The primitive
# records serialize to typed GPU instance buffers via webgpu_payload().
#
# Recording is intentionally cheap (plain lists; no numpy, no glyph layout): all GPU
# work (text layout against the glyph atlas, float32 packing, base64) is deferred to
# webgpu_payload(), so the SVG-only render path pays almost nothing.
#
import base64
import math

import numpy as np
import polars as pl

# floats per instance for each primitive kind
FLOATS_PER_INSTANCE = {
    'rect':   9,    # x, y, w, h, rx, r, g, b, a
    'circle': 12,   # cx, cy, radius, stroke_w, fr, fg, fb, fa, sr, sg, sb, sa
    'line':   12,   # x0, y0, x1, y1, width, r, g, b, a, dash_on, dash_off, dash_phase
    'glyph':  16,   # ox, oy, dx, dy, w, h, cos, sin, u0, v0, u1, v1, r, g, b, a
    'tri':    6,    # per-VERTEX: x, y, r, g, b, a (separate u32 index buffer)
}

# The collapsed-node cloud icon (<use href="#cloud">) has no GPU primitive.  Both the
# svgToDisplayList() parser and LinkP's instrumented path approximate it with this
# rounded rect, centered on the icon's anchor point -- shared here so linkp and
# spreadlinesp cannot drift on what a collapsed node looks like under GPU.
CLOUD_ICON_W, CLOUD_ICON_H, CLOUD_ICON_RX = 28.0, 14.0, 6.0

# The <defs> definition of that same icon, as it is written into the SVG.  linkp emits
# one copy (#cloud) and spreadlinesp two (#cloud and #cloud_outline); those were three
# hand-maintained literals of the same 644-byte string until they were hoisted here, so
# the only differences between them -- the id, and whether the path carries a stroke --
# are the two arguments below.
#
# Source: https://www.svgrepo.com/svg/520637/cloud
# License: CC Attribution License
# COLLECTION: Xnix Circular Interface Icons
# AUTHOR: Ankush Syal
# Modified to remove the bottom path (the second one).  The d= floats are rounded to two
# decimals in this source string rather than by a pass over finished output (PLANNING.md
# S4/S5) -- it is a fixed asset, so the rounding costs nothing at runtime.
#
_CLOUD_ICON_D_ = (
    'M14.09 7C14.99 6.97 15.87 7.31 16.52 7.93C17.18 8.55 17.56 9.4 17.59 10.3'
    'C17.59 10.62 17.54 10.94 17.45 11.25C18.61 11.43 19.47 12.42 19.5 13.6'
    'C19.46 14.97 18.32 16.04 16.95 16H8.04C6.68 16.04 5.54 14.97 5.5 13.6'
    'C5.52 12.48 6.31 11.52 7.41 11.28C7.41 11.25 7.41 11.23 7.41 11.2'
    'C7.45 9.84 8.59 8.76 9.96 8.8C10.27 8.8 10.59 8.86 10.89 8.97'
    'C11.49 7.75 12.73 6.98 14.09 7Z'
)

#
# cloudIconDef() - the '<g id=...>' cloud icon definition for an SVG <defs> block
# - id     : the symbol name a matching '<use href="#...">' will reference
# - stroke : outline color, or None for the unstroked variant (spreadlinesp's #cloud_outline)
#
def cloudIconDef(id: str = 'cloud', stroke: str | None = '#000000') -> str:
    _stroke_ = f'stroke="{stroke}" ' if stroke is not None else ''
    return (f'<g id="{id}" transform="translate(-50,-25)">'
            '<svg x="0" y="0" width="100px" height="50px" viewBox="-5 -5.5 35 35"'
            ' xmlns="http://www.w3.org/2000/svg">'
            '<path fill-rule="evenodd" clip-rule="evenodd" '
            f'd="{_CLOUD_ICON_D_}" '
            f'{_stroke_}stroke-linecap="round" stroke-linejoin="round"/>'
            '</svg></g>')

_NAMED_COLORS_ = {
    'black': (0.0, 0.0, 0.0), 'white': (1.0, 1.0, 1.0), 'red':  (1.0, 0.0, 0.0),
    'green': (0.0, 0.5, 0.0), 'blue':  (0.0, 0.0, 1.0), 'gray': (0.5, 0.5, 0.5),
    'grey':  (0.5, 0.5, 0.5), 'yellow': (1.0, 1.0, 0.0),
}

#
# hexToRGBA() - parse an SVG color string into an (r, g, b, a) float tuple
# - supports '#rrggbb', '#rrggbbaa', '#rgb', a few named colors, and 'none' (alpha 0)
# - opacity multiplies the alpha channel; unparseable colors fall back to gray so a
#   GPU-side parse problem can never break the SVG render path it is embedded in
#
def hexToRGBA(color: str | None, opacity: float = 1.0) -> tuple:
    if color is None or color == 'none': return (0.0, 0.0, 0.0, 0.0)
    if color.startswith('#'):
        _h_ = color[1:]
        try:
            if   len(_h_) == 3: r, g, b, a = int(_h_[0]*2, 16), int(_h_[1]*2, 16), int(_h_[2]*2, 16), 255
            elif len(_h_) == 6: r, g, b, a = int(_h_[0:2], 16), int(_h_[2:4], 16), int(_h_[4:6], 16), 255
            elif len(_h_) == 8: r, g, b, a = int(_h_[0:2], 16), int(_h_[2:4], 16), int(_h_[4:6], 16), int(_h_[6:8], 16)
            else:               return (0.5, 0.5, 0.5, opacity)
        except ValueError:
            return (0.5, 0.5, 0.5, opacity)
        return (r/255.0, g/255.0, b/255.0, (a/255.0) * opacity)
    if color in _NAMED_COLORS_:
        r, g, b = _NAMED_COLORS_[color]
        return (r, g, b, opacity)
    return (0.5, 0.5, 0.5, opacity)

#
# triangulatePolygon() - ear-clipping triangulation of a simple polygon
# - pts is a list of (x, y) tuples (closing point optional)
# - returns a list of index triples into pts
#
def triangulatePolygon(pts: list) -> list:
    if len(pts) >= 2 and pts[0] == pts[-1]: pts = pts[:-1]
    n = len(pts)
    if n < 3: return []
    # Signed area to establish winding
    _area_ = 0.0
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i+1) % n]
        _area_ += x0*y1 - x1*y0
    _ccw_ = _area_ > 0.0
    def _cross_(o: tuple, a: tuple, b: tuple) -> float:
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    def _inside_(p: tuple, a: tuple, b: tuple, c: tuple) -> bool:
        d0, d1, d2 = _cross_(a, b, p), _cross_(b, c, p), _cross_(c, a, p)
        _has_neg_ = (d0 < 0) or (d1 < 0) or (d2 < 0)
        _has_pos_ = (d0 > 0) or (d1 > 0) or (d2 > 0)
        return not (_has_neg_ and _has_pos_)
    _idx_  = list(range(n))
    _tris_ = []
    _guard_ = 0
    while len(_idx_) > 3 and _guard_ < 10 * n:
        _guard_ += 1
        _clipped_ = False
        for k in range(len(_idx_)):
            i0, i1, i2 = _idx_[k-1], _idx_[k], _idx_[(k+1) % len(_idx_)]
            a, b, c = pts[i0], pts[i1], pts[i2]
            _convex_ = _cross_(a, b, c) > 0 if _ccw_ else _cross_(a, b, c) < 0
            if not _convex_: continue
            _ear_ = True
            for j in _idx_:
                if j in (i0, i1, i2): continue
                if _inside_(pts[j], a, b, c):
                    _ear_ = False
                    break
            if _ear_:
                _tris_.append((i0, i1, i2))
                del _idx_[k]
                _clipped_ = True
                break
        if not _clipped_: break  # degenerate polygon -- emit what we have
    if len(_idx_) == 3: _tris_.append((_idx_[0], _idx_[1], _idx_[2]))
    return _tris_


#
# flattenPathD() - flatten an SVG path 'd' string (M/L/C/Z tokens, space separated)
# into a list of (points, closed) subpaths; cubic beziers are sampled
#
def flattenPathD(d: str, samples_per_curve: int = 16) -> list:
    _tokens_ = d.replace(',', ' ').split()
    _subpaths_, _cur_ = [], []
    i = 0
    while i < len(_tokens_):
        _t_ = _tokens_[i]
        if _t_ == 'M':
            if len(_cur_) > 1: _subpaths_.append((_cur_, False))
            _cur_ = [(float(_tokens_[i+1]), float(_tokens_[i+2]))]
            i += 3
        elif _t_ == 'L':
            _cur_.append((float(_tokens_[i+1]), float(_tokens_[i+2])))
            i += 3
        elif _t_ == 'C':
            if len(_cur_) > 0:
                _p0_ = _cur_[-1]
                _p1_ = (float(_tokens_[i+1]), float(_tokens_[i+2]))
                _p2_ = (float(_tokens_[i+3]), float(_tokens_[i+4]))
                _p3_ = (float(_tokens_[i+5]), float(_tokens_[i+6]))
                for k in range(1, samples_per_curve + 1):
                    t = k / samples_per_curve
                    mt = 1.0 - t
                    _cur_.append((mt*mt*mt*_p0_[0] + 3*mt*mt*t*_p1_[0] + 3*mt*t*t*_p2_[0] + t*t*t*_p3_[0],
                                  mt*mt*mt*_p0_[1] + 3*mt*mt*t*_p1_[1] + 3*mt*t*t*_p2_[1] + t*t*t*_p3_[1]))
            i += 7
        elif _t_ == 'Z':
            if len(_cur_) > 1: _subpaths_.append((_cur_, True))
            _cur_ = []
            i += 1
        else:
            i += 1   # unknown token -- skip
    if len(_cur_) > 1: _subpaths_.append((_cur_, False))
    return _subpaths_


#
# roundedRectPoints() - closed clockwise outline of a rounded rect, as a point list
#
# The rect primitive is fill-only (its rounded corners come from the shader's SDF),
# so a component that wants a *stroked* rounded rect draws the fill with rect() and
# runs its outline through strokePolylineDL() using these points.  Four straight
# edges would cut the corners off.
#
def roundedRectPoints(x: float, y: float, w: float, h: float, rx: float, samples_per_corner: int = 6) -> list:
    r = max(0.0, min(float(rx), w / 2.0, h / 2.0))
    if r <= 0.0:
        return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    _pts_ = []
    for cx, cy, a0 in ((x + w - r, y + r,     -math.pi / 2),   # top-right
                       (x + w - r, y + h - r,  0.0),           # bottom-right
                       (x + r,     y + h - r,  math.pi / 2),   # bottom-left
                       (x + r,     y + r,      math.pi)):      # top-left
        for k in range(samples_per_corner + 1):
            a = a0 + (math.pi / 2) * k / samples_per_corner
            _pts_.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return _pts_


#
# strokePolylineDL() - stroke a flattened polyline as line primitives
# - carries the running arc length as each segment's dash_phase so a dash pattern
#   continues across vertices the way SVG runs it along the whole path
#
def strokePolylineDL(dl: Any, pts: list, color: str, width: float = 1.0, opacity: float = 1.0, dash: tuple | None = None, scissor: tuple | None = None) -> None:
    _ph_ = 0.0
    for j in range(len(pts) - 1):
        dl.line(pts[j][0], pts[j][1], pts[j+1][0], pts[j+1][1], color,
                width=width, opacity=opacity, dash=dash, dash_phase=_ph_, scissor=scissor)
        if dash is not None:
            _ph_ += math.hypot(pts[j+1][0] - pts[j][0], pts[j+1][1] - pts[j][1])


#
# dashArrayToTuple() - SVG stroke-dasharray string -> the (on, off) pair the line
# primitive carries (FLOATS_PER_INSTANCE['line'] has dash_on / dash_off slots).
# A one-value dasharray means equal on/off, matching SVG's own rule.  Shared so the
# svgToDisplayList() parser and components that compose a dasharray directly (the
# background records) cannot disagree about how a pattern string is read.
#
def dashArrayToTuple(dasharray: str | None) -> tuple | None:
    if dasharray is None or dasharray in ('', 'none'): return None
    _dv_ = [float(_x_) for _x_ in str(dasharray).replace(',', ' ').split()]
    if len(_dv_) == 0: return None
    return (_dv_[0], _dv_[1] if len(_dv_) > 1 else _dv_[0])


#
# pathToDL() - record an SVG path 'd' string as GPU primitives
#
# Curves and arcs are flattened by flattenPathD(); closed subpaths fill as triangles
# and any subpath strokes as a polyline.  Shared by svgToDisplayList() and by
# components that compose a 'd' string directly (SpreadLinesP's smoothed bin
# outlines, cross-connects and channel pills) -- the point of sharing is that the
# flattening lives in one place, so an instrumented component and the parser can
# never disagree about where a curve goes.  xform maps each flattened point into
# the target space (identity when the caller records in its own coordinates).
#
def pathToDL(dl: Any, d: str, fill: str | None = None, stroke: str | None = None, width: float = 1.0, fill_opacity: float = 1.0,
             stroke_opacity: float = 1.0, dash: tuple | None = None, scissor: Any = None, xform: Any = None) -> None:
    for _pts_, _closed_ in flattenPathD(d):
        if xform is not None: _pts_ = [xform(px, py) for px, py in _pts_]
        if fill is not None and fill != 'none' and _closed_ and len(_pts_) >= 3:
            dl.polygon(_pts_, fill, opacity=fill_opacity, scissor=scissor)
        if stroke is not None and stroke != 'none':
            strokePolylineDL(dl, _pts_ + [_pts_[0]] if _closed_ else _pts_, stroke,
                             width=width, opacity=stroke_opacity, dash=dash, scissor=scissor)


#
# cubicBezierSegmentsTable() - flatten cubic beziers into line segments, polars-side
# - df has one row per curve with endpoint/control-point columns (names passed in)
# - returns one row per segment with __bx__/__by__ -> __bx2__/__by2__ endpoints;
#   all other input columns (colors, widths, counts) are carried through
#
def cubicBezierSegmentsTable(df: pl.DataFrame, x0: str, y0: str, cx0: str, cy0: str, cx1: str, cy1: str, x1: str, y1: str, n: int = 24) -> pl.DataFrame:
    _df_ = df.with_row_index('__bz_id__')
    _t_  = pl.DataFrame({'__t__': [i / n for i in range(n + 1)]})
    _j_  = _df_.join(_t_, how='cross')
    _tc_ = pl.col('__t__')
    _mt_ = 1.0 - _tc_
    _j_  = _j_.with_columns([
        (_mt_**3 * pl.col(x0) + 3*_mt_**2*_tc_ * pl.col(cx0) + 3*_mt_*_tc_**2 * pl.col(cx1) + _tc_**3 * pl.col(x1)).alias('__bx__'),
        (_mt_**3 * pl.col(y0) + 3*_mt_**2*_tc_ * pl.col(cy0) + 3*_mt_*_tc_**2 * pl.col(cy1) + _tc_**3 * pl.col(y1)).alias('__by__'),
    ]).sort(['__bz_id__', '__t__'])
    _j_  = _j_.with_columns([
        pl.col('__bx__').shift(-1).over('__bz_id__').alias('__bx2__'),
        pl.col('__by__').shift(-1).over('__bz_id__').alias('__by2__'),
    ]).filter(pl.col('__bx2__').is_not_null())
    return _j_


#
# _rootViewBoxTransform_() - parse the document's root <svg> width/height/viewBox
# and return the (scale, tx, ty) that maps viewBox coordinates to canvas pixels.
#
# Mirrors SVG's default preserveAspectRatio="xMidYMid meet": one uniform scale
# (so circles stay circular and text stays uniform) with the scaled viewBox
# centered in the canvas.  Returns identity (1, 0, 0) when there is no viewBox.
#
def _rootViewBoxTransform_(svg_str: str) -> tuple:
    import re
    m = re.search(r'<svg\b([^>]*)>', svg_str)
    if m is None: return (1.0, 0.0, 0.0)
    attrs = dict(re.findall(r'([\w-]+)="([^"]*)"', m.group(1)))
    vb = attrs.get('viewBox')
    if vb is None: return (1.0, 0.0, 0.0)
    try:
        vx0, vy0, vw, vh = (float(v) for v in vb.replace(',', ' ').split())
        cw = float(attrs.get('width',  vw).replace('px', ''))
        ch = float(attrs.get('height', vh).replace('px', ''))
    except (ValueError, TypeError):
        return (1.0, 0.0, 0.0)
    if vw <= 0 or vh <= 0: return (1.0, 0.0, 0.0)
    s  = min(cw / vw, ch / vh)
    tx = (cw - vw * s) / 2.0 - vx0 * s
    ty = (ch - vh * s) / 2.0 - vy0 * s
    return (s, tx, ty)


#
# svgToDisplayList() - generic SVG-string -> GPU primitive fallback
#
# Parses the primitive elements this codebase generates (rect, circle, line,
# polygon, path, text, use) in document order and records them into dl.  Every
# component in the package now dual-records instead (SpreadLinesP was the last
# holdout, instrumented 2026-08-06), so this is a universal fallback for markup
# that arrives as a finished string -- keep it for that, not as a shortcut for a
# new component.  The document's root viewBox is honored (coordinates, lengths,
# and font sizes are mapped into canvas pixels), so viewBox-scaled views convert
# at the correct size.  <defs> blocks are skipped; <use href="#cloud"> (the
# linkp/spreadlines cloud icon) approximates as a rounded rect.
# Gradients/clip-paths are ignored -- a component needing a clip should record a
# per-op scissor instead.
#
def svgToDisplayList(svg_str: str, dl: Any, p2s: Any) -> None:
    from polars2svg.p2s_text_mixin import svgUnescape
    import re
    _scale_, _tx_, _ty_ = _rootViewBoxTransform_(svg_str)
    def _TX_(x: float) -> float: return x * _scale_ + _tx_
    def _TY_(y: float) -> float: return y * _scale_ + _ty_
    def _TL_(l: float) -> float: return l * _scale_
    def _TDASH_(d: tuple | None) -> tuple: return None if d is None else (d[0] * _scale_, d[1] * _scale_)
    _s_ = re.sub(r'<defs>.*?</defs>', '', svg_str, flags=re.DOTALL)
    _elem_re_ = re.compile(r'<(rect|circle|line|polygon|path|text|use)\b([^>]*?)(/>|>)', re.DOTALL)
    _attr_re_ = re.compile(r'([\w-]+)="([^"]*)"')
    pos = 0
    while True:
        m = _elem_re_.search(_s_, pos)
        if m is None: break
        tag, attr_str, close = m.group(1), m.group(2), m.group(3)
        pos = m.end()
        a = dict(_attr_re_.findall(attr_str))
        def _f_(name: str, default: float = 0.0) -> float:
            v = a.get(name)
            if v is None: return default
            try:    return float(v.replace('px', ''))
            except ValueError: return default
        _opacity_      = _f_('opacity', 1.0)
        _fill_         = a.get('fill')
        _fill_op_      = _f_('fill-opacity', 1.0) * _opacity_
        _stroke_       = a.get('stroke')
        _stroke_op_    = _f_('stroke-opacity', 1.0) * _opacity_
        _stroke_w_     = _TL_(_f_('stroke-width', 1.0))
        _dash_ = _TDASH_(dashArrayToTuple(a.get('stroke-dasharray')))

        def _strokePolyline_(pts: list) -> None:
            if _stroke_ is None: return          # callers check, but not visibly to a checker
            strokePolylineDL(dl, pts, _stroke_, width=_stroke_w_,
                             opacity=_stroke_op_, dash=_dash_)

        if tag == 'rect':
            x, y, w_, h_ = _f_('x'), _f_('y'), _f_('width'), _f_('height')
            if _fill_ is not None and _fill_ != 'none':
                dl.rect(_TX_(x), _TY_(y), _TL_(w_), _TL_(h_), _fill_, rx=_TL_(_f_('rx')), opacity=_fill_op_)
            if _stroke_ is not None and _stroke_ != 'none':
                for (lx0, ly0, lx1, ly1) in ((x, y, x+w_, y), (x, y+h_, x+w_, y+h_),
                                             (x, y, x, y+h_), (x+w_, y, x+w_, y+h_)):
                    dl.line(_TX_(lx0), _TY_(ly0), _TX_(lx1), _TY_(ly1), _stroke_,
                            width=_stroke_w_, opacity=_stroke_op_, dash=_dash_)
        elif tag == 'circle':
            _has_fill_ = _fill_ is not None and _fill_ != 'none'
            dl.circle(_TX_(_f_('cx')), _TY_(_f_('cy')), _TL_(_f_('r')),
                      _fill_ if _has_fill_ else 'none',
                      stroke=_stroke_ if (_stroke_ not in (None, 'none')) else None,
                      stroke_w=_stroke_w_,
                      opacity=_fill_op_ if _has_fill_ else _stroke_op_)
        elif tag == 'line':
            if _stroke_ is not None and _stroke_ != 'none':
                dl.line(_TX_(_f_('x1')), _TY_(_f_('y1')), _TX_(_f_('x2')), _TY_(_f_('y2')), _stroke_,
                        width=_stroke_w_, opacity=_stroke_op_, dash=_dash_)
        elif tag == 'polygon':
            _pts_ = [(_TX_(float(p.split(',')[0])), _TY_(float(p.split(',')[1])))
                     for p in a.get('points', '').split() if ',' in p]
            if len(_pts_) >= 3 and _fill_ is not None and _fill_ != 'none':
                dl.polygon(_pts_, _fill_, opacity=_fill_op_)
            if len(_pts_) >= 2 and _stroke_ is not None and _stroke_ != 'none':
                _strokePolyline_(_pts_ + [_pts_[0]])
        elif tag == 'path':
            pathToDL(dl, a.get('d', ''), fill=_fill_, stroke=_stroke_, width=_stroke_w_,
                     fill_opacity=_fill_op_, stroke_opacity=_stroke_op_, dash=_dash_,
                     xform=lambda px, py: (_TX_(px), _TY_(py)))
        elif tag == 'use':
            # cloud icon approximation: rounded rect centered on (x, y)
            dl.rect(_TX_(_f_('x') - CLOUD_ICON_W / 2.0), _TY_(_f_('y') - CLOUD_ICON_H / 2.0),
                    _TL_(CLOUD_ICON_W), _TL_(CLOUD_ICON_H),
                    _fill_ or '#ffffff', rx=_TL_(CLOUD_ICON_RX), opacity=_opacity_)
        elif tag == 'text' and close == '>':
            _end_ = _s_.find('</text>', pos)
            if _end_ < 0: continue
            _content_ = _s_[pos:_end_]
            pos = _end_ + len('</text>')
            _rot_ = None
            _tm_ = re.match(r'rotate\(([-\d.]+)', a.get('transform', ''))
            if _tm_ is not None: _rot_ = float(_tm_.group(1))
            _anchor_ = a.get('text-anchor', 'start')
            _th_     = _TL_(_f_('font-size', 12.0))
            _co_     = _fill_ if _fill_ is not None else None
            _bshift_ = 0.35 * _th_ if a.get('dominant-baseline') == 'central' else 0.0
            if '<tspan' in _content_:
                _ty_pen_ = _f_('y')
                for _sm_ in re.finditer(r'<tspan x="([^"]*)" dy="([^"]*)">([^<]*)</tspan>', _content_):
                    _ty_pen_ += float(_sm_.group(2))
                    dl.text(p2s, svgUnescape(_sm_.group(3)), _TX_(float(_sm_.group(1))), _TY_(_ty_pen_),
                            txt_h=_th_, anchor=_anchor_, color=_co_, rotation=_rot_,
                            baseline_shift=_bshift_, svg='')
            elif '<' not in _content_:
                dl.text(p2s, svgUnescape(_content_), _TX_(_f_('x')), _TY_(_f_('y')),
                        txt_h=_th_, anchor=_anchor_, color=_co_, rotation=_rot_,
                        baseline_shift=_bshift_, svg='')


class DisplayList:
    #
    # Op stream encoding (self._ops_): ordered (kind, payload, scissor) tuples
    # - scalar kinds ('rect','circle','line'): payload = plain list of floats
    # - table kinds  (same names):             payload = 2d np.float32 array
    # - 'text':                                payload = (txt, x, y, txt_h, anchor, rotation, rgba)
    #                                          -- resolved to glyph instances at payload time
    # - 'tri':                                 payload = (verts np.float32 [n,6], indices np.uint32)
    #
    def __init__(self, w: int, h: int, bg: str = '#ffffff') -> None:
        self.w, self.h = w, h
        self.bg        = bg
        self._svg_     = []   # ordered svg strings (svg() output = join)
        self._ops_     = []

    # ── scalar emitters ──────────────────────────────────────────────────
    def _record_(self, kind: str, values: list, svg: str | None, scissor: tuple | None = None) -> str:
        if svg: self._svg_.append(svg)
        self._ops_.append((kind, values, scissor))
        return svg if svg else ''

    def rect(self, x: float, y: float, w: float, h: float, fill: str, rx: float = 0.0, opacity: float = 1.0, svg: str | None = None, scissor: tuple | None = None) -> str:
        r, g, b, a = hexToRGBA(fill, opacity)
        return self._record_('rect', [x, y, w, h, rx, r, g, b, a], svg, scissor)

    # stroke_opacity defaults to opacity; pass it when SVG's fill-opacity and
    # stroke-opacity differ on the same element (the instance carries both alphas)
    def circle(self, cx: float, cy: float, r: float, fill: str, stroke: str | None = None, stroke_w: float = 0.0, opacity: float = 1.0,
               stroke_opacity: float | None = None, svg: str | None = None, scissor: Any = None) -> str:
        _so_ = opacity if stroke_opacity is None else stroke_opacity
        fr, fg, fb, fa = hexToRGBA(fill, opacity)
        sr, sg, sb, sa = hexToRGBA(stroke, _so_) if stroke is not None else (0.0, 0.0, 0.0, 0.0)
        return self._record_('circle', [cx, cy, r, stroke_w, fr, fg, fb, fa, sr, sg, sb, sa], svg, scissor)

    #
    # line() - one straight segment
    # - dash_phase is the arc length already travelled along the logical stroke this
    #   segment belongs to.  SVG runs a dash pattern continuously along a whole path, so
    #   a caller flattening a polyline or a curve must pass the running total; leaving it
    #   at 0 restarts the pattern at every vertex.  Ignored when dash is None.
    #
    def line(self, x0: float, y0: float, x1: float, y1: float, color: str, width: float = 1.0, dash: tuple | None = None, opacity: float = 1.0, dash_phase: float = 0.0,
             svg: str | None = None, scissor: tuple | None = None) -> str:
        r, g, b, a = hexToRGBA(color, opacity)
        _don_, _doff_ = (float(dash[0]), float(dash[1])) if dash is not None else (0.0, 0.0)
        return self._record_('line', [x0, y0, x1, y1, width, r, g, b, a, _don_, _doff_,
                                      float(dash_phase)], svg, scissor)

    #
    # tris() - filled triangle geometry
    # - xy is a flat list/array of vertex coordinates [x0, y0, x1, y1, ...]
    # - indices is a flat list of index triples; rgba is one color tuple or a per-vertex array
    #
    def tris(self, xy: list, indices: list, rgba: np.ndarray | tuple, svg: str | None = None, scissor: Any = None) -> str:
        _xy_  = np.asarray(xy, dtype=np.float32).reshape(-1, 2)
        _n_   = len(_xy_)
        _rgba_ = np.asarray(rgba, dtype=np.float32)
        if _rgba_.ndim == 1: _rgba_ = np.tile(_rgba_, (_n_, 1))
        _verts_ = np.hstack([_xy_, _rgba_]).astype(np.float32)
        _idx_   = np.asarray(indices, dtype=np.uint32)
        if svg: self._svg_.append(svg)
        self._ops_.append(('tri', (_verts_, _idx_), scissor))
        return svg if svg else ''

    #
    # polygon() - convenience: ear-clip a simple polygon into a tri op
    #
    def polygon(self, pts: list, fill: str, opacity: float = 1.0, svg: str | None = None, scissor: Any = None) -> str:
        _tris_ = triangulatePolygon(list(pts))
        if len(_tris_) == 0:
            if svg: self._svg_.append(svg)
            return svg if svg else ''
        _xy_  = [c for p in (pts[:-1] if len(pts) >= 2 and pts[0] == pts[-1] else pts) for c in p]
        _idx_ = [i for t in _tris_ for i in t]
        return self.tris(_xy_, _idx_, hexToRGBA(fill, opacity), svg=svg, scissor=scissor)

    #
    # text() - record a text run for GPU glyph rendering while passing the SVG string
    # through verbatim; glyph layout against the atlas is deferred to payload time
    # - p2s supplies svgText() (for the canonical string when svg= is omitted)
    # - mirrors the svgText() signature so call sites swap mechanically
    #
    def text(self, p2s: Any, txt: int | str, x: float, y: float, txt_h: float = 12, color: str | None = None, anchor: str = 'start', font: Any = None,
             font_style: Any = None, rotation: float | None = None, baseline_shift: float = 0.0, svg: str | None = None, scissor: Any = None) -> str:
        if svg is None:
            svg = p2s.svgText(txt, x, y, txt_h=txt_h, color=color, anchor=anchor,
                              font=font, font_style=font_style, rotation=rotation)
        if txt is None or str(txt) in ('', '\n', '\r', '\t'):
            return svg
        if color is None: color = p2s.colorTyped('label', 'defaultfg')
        if svg: self._svg_.append(svg)
        self._ops_.append(('text', (str(txt), float(x), float(y), float(txt_h),
                                    anchor, rotation, hexToRGBA(color), float(baseline_shift)), scissor))
        return svg

    #
    # raw() - svg-only content with no GPU primitive (defs, gradients, unsupported markup)
    #
    def raw(self, svg_str: str) -> str:
        if svg_str: self._svg_.append(svg_str)
        return svg_str

    # ── table emitters (polars DataFrames -> instance arrays) ────────────
    #
    # Column arguments may be a column name (str) or a constant (int/float).
    # rgba columns: pass (r_col, g_col, b_col) of floats in [0,1], with optional
    # opacity as a column name or constant.  svg_col='__svg__' appends that
    # column's strings to the svg stream verbatim; svg_col=None contributes
    # nothing to the SVG output (caller already appended the strings).
    #
    def _colexpr_(self, c: float | int | str) -> pl.Expr:
        if isinstance(c, str): return pl.col(c).cast(pl.Float32)
        return pl.lit(float(c), dtype=pl.Float32)

    def _df_to_op_(self, kind: str, df: pl.DataFrame, exprs: list, svg_col: str | None, scissor: Any) -> str:
        if len(df) == 0: return ''
        _arr_ = df.select([e.alias(f'__c{i}__') for i, e in enumerate(exprs)]).to_numpy().astype(np.float32)
        _svg_ = ''
        if svg_col is not None:
            _svg_ = ''.join(df[svg_col].to_list())
            self._svg_.append(_svg_)
        self._ops_.append((kind, _arr_, scissor))
        return _svg_

    def rects_table(self, df: pl.DataFrame, x: int | str, y: str, w: float | int | str, h: float | int | str, rgba: tuple, rx: float = 0.0, opacity: float = 1.0, svg_col: str | None = '__svg__', scissor: Any = None) -> str:
        _r_, _g_, _b_ = rgba
        _exprs_ = [self._colexpr_(x), self._colexpr_(y), self._colexpr_(w), self._colexpr_(h),
                   self._colexpr_(rx), self._colexpr_(_r_), self._colexpr_(_g_), self._colexpr_(_b_),
                   self._colexpr_(opacity)]
        return self._df_to_op_('rect', df, _exprs_, svg_col, scissor)

    def circles_table(self, df: pl.DataFrame, cx: str, cy: str, r: float | int | str, rgba: tuple, opacity: float | str = 1.0, stroke: tuple | None = None, stroke_w: float = 0.0,
                      svg_col: str | None = '__svg__', scissor: Any = None) -> str:
        _r_, _g_, _b_ = rgba
        if stroke is not None:
            _sr_, _sg_, _sb_, _sa_ = hexToRGBA(stroke, opacity) if isinstance(stroke, str) else stroke
        else:
            _sr_, _sg_, _sb_, _sa_ = 0.0, 0.0, 0.0, 0.0
        _exprs_ = [self._colexpr_(cx), self._colexpr_(cy), self._colexpr_(r), self._colexpr_(stroke_w),
                   self._colexpr_(_r_), self._colexpr_(_g_), self._colexpr_(_b_), self._colexpr_(opacity),
                   self._colexpr_(_sr_), self._colexpr_(_sg_), self._colexpr_(_sb_), self._colexpr_(_sa_)]
        return self._df_to_op_('circle', df, _exprs_, svg_col, scissor)

    # dash_phase: see line() -- a column name (per-segment running arc length) or a constant
    def lines_table(self, df: pl.DataFrame, x0: str, y0: str, x1: str, y1: str, rgba: tuple, width: float | int | str = 1.0, opacity: float | str = 1.0, dash: tuple | None = None,
                    dash_phase: float | str = 0.0, svg_col: Any = '__svg__', scissor: Any = None) -> str:
        _r_, _g_, _b_ = rgba
        _don_, _doff_ = (float(dash[0]), float(dash[1])) if dash is not None else (0.0, 0.0)
        _exprs_ = [self._colexpr_(x0), self._colexpr_(y0), self._colexpr_(x1), self._colexpr_(y1),
                   self._colexpr_(width), self._colexpr_(_r_), self._colexpr_(_g_), self._colexpr_(_b_),
                   self._colexpr_(opacity), self._colexpr_(_don_), self._colexpr_(_doff_),
                   self._colexpr_(dash_phase)]
        return self._df_to_op_('line', df, _exprs_, svg_col, scissor)

    # ── composition ──────────────────────────────────────────────────────
    #
    # applyTransform() - map every recorded op from world coordinates into canvas
    # pixels, in place
    #
    # For components that lay out in their own coordinate space and only learn the
    # viewBox after the whole body is rendered (SpreadLinesP sizes its viewBox from
    # the bins it just placed): record in world units, then call this once at the end.
    # (scale, tx, ty) has the same meaning as the triple svgToDisplayList() derives
    # from the root viewBox, so an instrumented component and a parsed one land in the
    # same place.  Screen-space pieces -- a legend positioned in canvas pixels -- are
    # kept in their own DisplayList and extend()ed in *after* this call.
    #
    # Points move, pure lengths (radii, widths, dash periods, font heights) scale, and
    # scissor rectangles do both.  Colors and text are untouched.
    #
    _XFORM_LAYOUT_ = {                  # kind -> ((x, y) index pairs, length indices)
        'rect':   ([(0, 1)],         [2, 3, 4]),
        'circle': ([(0, 1)],         [2, 3]),
        'line':   ([(0, 1), (2, 3)], [4, 9, 10, 11]),
    }

    def applyTransform(self, scale: float, tx: float, ty: float) -> Any:
        s, tx, ty = float(scale), float(tx), float(ty)
        _ops_ = []
        for kind, payload, scissor in self._ops_:
            if scissor is not None:
                scissor = (scissor[0] * s + tx, scissor[1] * s + ty, scissor[2] * s, scissor[3] * s)
            if kind == 'tri':
                verts, idx = payload
                verts = verts.copy()
                verts[:, 0] = verts[:, 0] * s + tx
                verts[:, 1] = verts[:, 1] * s + ty
                payload = (verts, idx)
            elif kind == 'text':
                txt, x, y, txt_h, anchor, rotation, rgba, bshift = payload
                payload = (txt, x * s + tx, y * s + ty, txt_h * s, anchor, rotation, rgba, bshift * s)
            else:
                _pts_, _lens_ = self._XFORM_LAYOUT_[kind]
                if isinstance(payload, np.ndarray):
                    payload = payload.copy()
                    for xi, yi in _pts_:
                        payload[:, xi] = payload[:, xi] * s + tx
                        payload[:, yi] = payload[:, yi] * s + ty
                    for li in _lens_: payload[:, li] *= s
                else:
                    payload = list(payload)
                    for xi, yi in _pts_:
                        payload[xi] = payload[xi] * s + tx
                        payload[yi] = payload[yi] * s + ty
                    for li in _lens_: payload[li] *= s
            _ops_.append((kind, payload, scissor))
        self._ops_ = _ops_
        return self

    #
    # extend() - splice another DisplayList's recorded primitives into this one
    # - offset translates coordinates; scissor (x, y, w, h) overrides per-op scissors
    # - svg strings are NOT copied (callers keep their own svg assembly, e.g. smallp's
    #   <g transform=...> path) unless copy_svg=True
    #
    def extend(self, other: Any, offset: tuple = (0, 0), scissor: tuple | None = None, copy_svg: bool = False) -> None:
        ox, oy = float(offset[0]), float(offset[1])
        _translate_ = (ox != 0.0 or oy != 0.0)
        for kind, payload, op_scissor in other._ops_:
            _sc_ = scissor if scissor is not None else op_scissor
            if not _translate_:
                self._ops_.append((kind, payload, _sc_))
                continue
            if kind == 'tri':
                verts, idx = payload
                verts = verts.copy()
                verts[:, 0] += ox
                verts[:, 1] += oy
                self._ops_.append((kind, (verts, idx), _sc_))
            elif kind == 'text':
                txt, x, y, txt_h, anchor, rotation, rgba, bshift = payload
                self._ops_.append((kind, (txt, x + ox, y + oy, txt_h, anchor, rotation, rgba, bshift), _sc_))
            elif isinstance(payload, np.ndarray):
                arr = payload.copy()
                arr[:, 0] += ox; arr[:, 1] += oy
                if kind == 'line':
                    arr[:, 2] += ox; arr[:, 3] += oy
                self._ops_.append((kind, arr, _sc_))
            else:
                vals = list(payload)
                vals[0] += ox; vals[1] += oy
                if kind == 'line':
                    vals[2] += ox; vals[3] += oy
                self._ops_.append((kind, vals, _sc_))
        if copy_svg: self._svg_.extend(other._svg_)

    # ── output ───────────────────────────────────────────────────────────
    def svg(self) -> str:
        return ''.join(self._svg_)

    #
    # webgpu_payload() - pack the ordered primitive stream into typed buffers + manifest
    # - consecutive ops of the same kind & scissor merge into one instanced batch
    # - text runs are laid out against the glyph atlas here (deferred from text())
    # - returns a JSON-safe dict: buffers are base64-encoded little-endian float32/uint32
    #
    def webgpu_payload(self, atlas: Any = None) -> dict:
        _chunks_   = {k: [] for k in FLOATS_PER_INSTANCE}   # per-kind list of np arrays
        _tri_v_, _tri_i_ = [], []
        _counts_   = {k: 0 for k in FLOATS_PER_INSTANCE}    # instances emitted so far per kind
        _tri_vtx_count_, _tri_idx_count_ = 0, 0
        _manifest_ = []
        _has_text_ = False

        def _scissor_list_(sc: tuple | None) -> list | None:
            if sc is None: return None
            x, y, w, h = sc
            x0, y0 = max(0, int(math.floor(x))), max(0, int(math.floor(y)))
            x1, y1 = min(self.w, int(math.ceil(x + w))), min(self.h, int(math.ceil(y + h)))
            return [x0, y0, max(0, x1 - x0), max(0, y1 - y0)]

        for kind, payload, scissor in self._ops_:
            _sc_ = _scissor_list_(scissor)
            if kind == 'tri':
                verts, idx = payload
                _tri_v_.append(verts)
                _tri_i_.append(idx.astype(np.uint32) + np.uint32(_tri_vtx_count_))
                _first_, _count_ = _tri_idx_count_, len(idx)
                _tri_vtx_count_ += len(verts)
                _tri_idx_count_ += len(idx)
            elif kind == 'text':
                _has_text_ = True
                if atlas is None: continue
                txt, x, y, txt_h, anchor, rotation, rgba, bshift = payload
                _glyphs_ = atlas.layoutText(txt, x, y, txt_h, anchor=anchor, rotation=rotation, dy=bshift)
                if len(_glyphs_) == 0: continue
                _arr_ = np.asarray([list(g) + list(rgba) for g in _glyphs_], dtype=np.float32)
                kind  = 'glyph'
                _chunks_['glyph'].append(_arr_)
                _first_, _count_ = _counts_['glyph'], len(_arr_)
                _counts_['glyph'] += len(_arr_)
            else:
                if isinstance(payload, np.ndarray): _arr_ = payload
                else:                               _arr_ = np.asarray(payload, dtype=np.float32).reshape(1, -1)
                _chunks_[kind].append(_arr_)
                _first_, _count_ = _counts_[kind], len(_arr_)
                _counts_[kind] += len(_arr_)
            # merge with the previous manifest entry when contiguous & same scissor
            if (_manifest_ and _manifest_[-1]['kind'] == kind and
                    _manifest_[-1].get('scissor') == _sc_ and
                    _manifest_[-1]['first'] + _manifest_[-1]['count'] == _first_):
                _manifest_[-1]['count'] += _count_
            else:
                _entry_ = {'kind': kind, 'first': _first_, 'count': _count_}
                if _sc_ is not None: _entry_['scissor'] = _sc_
                _manifest_.append(_entry_)

        _buffers_ = {}
        for kind in ('rect', 'circle', 'line', 'glyph'):
            if _counts_[kind] > 0:
                _all_ = np.concatenate(_chunks_[kind], axis=0).astype('<f4')
                _buffers_[kind] = base64.b64encode(_all_.tobytes()).decode()
        if _tri_idx_count_ > 0:
            _buffers_['tri_v'] = base64.b64encode(np.concatenate(_tri_v_, axis=0).astype('<f4').tobytes()).decode()
            _buffers_['tri_i'] = base64.b64encode(np.concatenate(_tri_i_, axis=0).astype('<u4').tobytes()).decode()

        _payload_ = {
            'wxh':      [int(self.w), int(self.h)],
            'bg':       self.bg,
            'buffers':  _buffers_,
            'manifest': _manifest_,
        }
        if atlas is not None and _has_text_:
            _payload_['atlas'] = atlas.payload()
        return _payload_
