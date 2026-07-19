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
    'line':   11,   # x0, y0, x1, y1, width, r, g, b, a, dash_on, dash_off
    'glyph':  16,   # ox, oy, dx, dy, w, h, cos, sin, u0, v0, u1, v1, r, g, b, a
    'tri':    6,    # per-VERTEX: x, y, r, g, b, a (separate u32 index buffer)
}

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
def hexToRGBA(color, opacity=1.0):
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
def triangulatePolygon(pts):
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
    def _cross_(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    def _inside_(p, a, b, c):
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
def flattenPathD(d, samples_per_curve=16):
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
# cubicBezierSegmentsTable() - flatten cubic beziers into line segments, polars-side
# - df has one row per curve with endpoint/control-point columns (names passed in)
# - returns one row per segment with __bx__/__by__ -> __bx2__/__by2__ endpoints;
#   all other input columns (colors, widths, counts) are carried through
#
def cubicBezierSegmentsTable(df, x0, y0, cx0, cy0, cx1, cy1, x1, y1, n=24):
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
def _rootViewBoxTransform_(svg_str):
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
# polygon, path, text, use) in document order and records them into dl.
# Used by components whose SVG assembly is too string-composed to instrument
# (SpreadLinesP) and as a universal fallback.  The document's root viewBox is
# honored (coordinates, lengths, and font sizes are mapped into canvas pixels),
# so viewBox-scaled views convert at the correct size.  <defs> blocks are
# skipped; <use href="#cloud"> (the linkp/spreadlines cloud icon) approximates
# as a rounded rect.  Gradients/clip-paths are ignored.
#
def svgToDisplayList(svg_str, dl, p2s):
    import html as _html_
    import re
    _scale_, _tx_, _ty_ = _rootViewBoxTransform_(svg_str)
    def _TX_(x): return x * _scale_ + _tx_
    def _TY_(y): return y * _scale_ + _ty_
    def _TL_(l): return l * _scale_
    def _TDASH_(d): return None if d is None else (d[0] * _scale_, d[1] * _scale_)
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
        def _f_(name, default=0.0):
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
        _dash_ = None
        if 'stroke-dasharray' in a and a['stroke-dasharray'] not in ('', 'none'):
            _dv_ = [float(x) for x in a['stroke-dasharray'].replace(',', ' ').split()]
            _dash_ = _TDASH_((_dv_[0], _dv_[1] if len(_dv_) > 1 else _dv_[0]))

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
                _closed_ = _pts_ + [_pts_[0]]
                for j in range(len(_closed_) - 1):
                    dl.line(_closed_[j][0], _closed_[j][1], _closed_[j+1][0], _closed_[j+1][1],
                            _stroke_, width=_stroke_w_, opacity=_stroke_op_, dash=_dash_)
        elif tag == 'path':
            for _pts_, _closed_ in flattenPathD(a.get('d', '')):
                _pts_ = [(_TX_(px), _TY_(py)) for px, py in _pts_]
                if _fill_ is not None and _fill_ != 'none' and _closed_ and len(_pts_) >= 3:
                    dl.polygon(_pts_, _fill_, opacity=_fill_op_)
                if _stroke_ is not None and _stroke_ != 'none':
                    _seq_ = _pts_ + [_pts_[0]] if _closed_ else _pts_
                    for j in range(len(_seq_) - 1):
                        dl.line(_seq_[j][0], _seq_[j][1], _seq_[j+1][0], _seq_[j+1][1],
                                _stroke_, width=_stroke_w_, opacity=_stroke_op_, dash=_dash_)
        elif tag == 'use':
            # cloud icon approximation: rounded rect centered on (x, y)
            dl.rect(_TX_(_f_('x') - 14), _TY_(_f_('y') - 7), _TL_(28), _TL_(14),
                    _fill_ or '#ffffff', rx=_TL_(6), opacity=_opacity_)
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
                    dl.text(p2s, _html_.unescape(_sm_.group(3)), _TX_(float(_sm_.group(1))), _TY_(_ty_pen_),
                            txt_h=_th_, anchor=_anchor_, color=_co_, rotation=_rot_,
                            baseline_shift=_bshift_, svg='')
            elif '<' not in _content_:
                dl.text(p2s, _html_.unescape(_content_), _TX_(_f_('x')), _TY_(_f_('y')),
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
    def __init__(self, w, h, bg='#ffffff'):
        self.w, self.h = w, h
        self.bg        = bg
        self._svg_     = []   # ordered svg strings (svg() output = join)
        self._ops_     = []

    # ── scalar emitters ──────────────────────────────────────────────────
    def _record_(self, kind, values, svg, scissor=None):
        if svg: self._svg_.append(svg)
        self._ops_.append((kind, values, scissor))
        return svg if svg else ''

    def rect(self, x, y, w, h, fill, rx=0.0, opacity=1.0, svg=None, scissor=None):
        r, g, b, a = hexToRGBA(fill, opacity)
        return self._record_('rect', [x, y, w, h, rx, r, g, b, a], svg, scissor)

    def circle(self, cx, cy, r, fill, stroke=None, stroke_w=0.0, opacity=1.0, svg=None, scissor=None):
        fr, fg, fb, fa = hexToRGBA(fill, opacity)
        sr, sg, sb, sa = hexToRGBA(stroke, opacity) if stroke is not None else (0.0, 0.0, 0.0, 0.0)
        return self._record_('circle', [cx, cy, r, stroke_w, fr, fg, fb, fa, sr, sg, sb, sa], svg, scissor)

    def line(self, x0, y0, x1, y1, color, width=1.0, dash=None, opacity=1.0, svg=None, scissor=None):
        r, g, b, a = hexToRGBA(color, opacity)
        _don_, _doff_ = (float(dash[0]), float(dash[1])) if dash is not None else (0.0, 0.0)
        return self._record_('line', [x0, y0, x1, y1, width, r, g, b, a, _don_, _doff_], svg, scissor)

    #
    # tris() - filled triangle geometry
    # - xy is a flat list/array of vertex coordinates [x0, y0, x1, y1, ...]
    # - indices is a flat list of index triples; rgba is one color tuple or a per-vertex array
    #
    def tris(self, xy, indices, rgba, svg=None, scissor=None):
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
    def polygon(self, pts, fill, opacity=1.0, svg=None, scissor=None):
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
    def text(self, p2s, txt, x, y, txt_h=12, color=None, anchor='start', font=None,
             font_style=None, rotation=None, baseline_shift=0.0, svg=None, scissor=None):
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
    def raw(self, svg_str):
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
    def _colexpr_(self, c):
        if isinstance(c, str): return pl.col(c).cast(pl.Float32)
        return pl.lit(float(c), dtype=pl.Float32)

    def _df_to_op_(self, kind, df, exprs, svg_col, scissor):
        if len(df) == 0: return ''
        _arr_ = df.select([e.alias(f'__c{i}__') for i, e in enumerate(exprs)]).to_numpy().astype(np.float32)
        _svg_ = ''
        if svg_col is not None:
            _svg_ = ''.join(df[svg_col].to_list())
            self._svg_.append(_svg_)
        self._ops_.append((kind, _arr_, scissor))
        return _svg_

    def rects_table(self, df, x, y, w, h, rgba, rx=0.0, opacity=1.0, svg_col='__svg__', scissor=None):
        _r_, _g_, _b_ = rgba
        _exprs_ = [self._colexpr_(x), self._colexpr_(y), self._colexpr_(w), self._colexpr_(h),
                   self._colexpr_(rx), self._colexpr_(_r_), self._colexpr_(_g_), self._colexpr_(_b_),
                   self._colexpr_(opacity)]
        return self._df_to_op_('rect', df, _exprs_, svg_col, scissor)

    def circles_table(self, df, cx, cy, r, rgba, opacity=1.0, stroke=None, stroke_w=0.0,
                      svg_col='__svg__', scissor=None):
        _r_, _g_, _b_ = rgba
        if stroke is not None:
            _sr_, _sg_, _sb_, _sa_ = hexToRGBA(stroke, opacity) if isinstance(stroke, str) else stroke
        else:
            _sr_, _sg_, _sb_, _sa_ = 0.0, 0.0, 0.0, 0.0
        _exprs_ = [self._colexpr_(cx), self._colexpr_(cy), self._colexpr_(r), self._colexpr_(stroke_w),
                   self._colexpr_(_r_), self._colexpr_(_g_), self._colexpr_(_b_), self._colexpr_(opacity),
                   self._colexpr_(_sr_), self._colexpr_(_sg_), self._colexpr_(_sb_), self._colexpr_(_sa_)]
        return self._df_to_op_('circle', df, _exprs_, svg_col, scissor)

    def lines_table(self, df, x0, y0, x1, y1, rgba, width=1.0, opacity=1.0, dash=None,
                    svg_col='__svg__', scissor=None):
        _r_, _g_, _b_ = rgba
        _don_, _doff_ = (float(dash[0]), float(dash[1])) if dash is not None else (0.0, 0.0)
        _exprs_ = [self._colexpr_(x0), self._colexpr_(y0), self._colexpr_(x1), self._colexpr_(y1),
                   self._colexpr_(width), self._colexpr_(_r_), self._colexpr_(_g_), self._colexpr_(_b_),
                   self._colexpr_(opacity), self._colexpr_(_don_), self._colexpr_(_doff_)]
        return self._df_to_op_('line', df, _exprs_, svg_col, scissor)

    # ── composition ──────────────────────────────────────────────────────
    #
    # extend() - splice another DisplayList's recorded primitives into this one
    # - offset translates coordinates; scissor (x, y, w, h) overrides per-op scissors
    # - svg strings are NOT copied (callers keep their own svg assembly, e.g. smallp's
    #   <g transform=...> path) unless copy_svg=True
    #
    def extend(self, other, offset=(0, 0), scissor=None, copy_svg=False):
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
    def svg(self):
        return ''.join(self._svg_)

    #
    # webgpu_payload() - pack the ordered primitive stream into typed buffers + manifest
    # - consecutive ops of the same kind & scissor merge into one instanced batch
    # - text runs are laid out against the glyph atlas here (deferred from text())
    # - returns a JSON-safe dict: buffers are base64-encoded little-endian float32/uint32
    #
    def webgpu_payload(self, atlas=None):
        _chunks_   = {k: [] for k in FLOATS_PER_INSTANCE}   # per-kind list of np arrays
        _tri_v_, _tri_i_ = [], []
        _counts_   = {k: 0 for k in FLOATS_PER_INSTANCE}    # instances emitted so far per kind
        _tri_vtx_count_, _tri_idx_count_ = 0, 0
        _manifest_ = []
        _has_text_ = False

        def _scissor_list_(sc):
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
