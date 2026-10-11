from pathlib import Path
#
# export - save rendered components to disk (SVG always; PNG via optional deps)
#
# Every rendered component (XYp, Histop, Timep, Piep, LinkP, ChP, Smallp,
# SpreadLinesP) exposes a standalone SVG document through _repr_svg_().  The
# ExportMixin turns that into a file:
#
#   plot.save('chart.svg')     # write the SVG document verbatim
#   plot.save('chart.png')     # rasterize to PNG (extension-dispatched)
#   plot.savePNG('chart.png')  # rasterize to PNG explicitly
#
# SVG export is dependency-free.  PNG export rasterizes with svglib + reportlab,
# which are an optional install (`pip install polars2svg[export]`); savePNG /
# svgToPNGBytes raise a clear ImportError naming that extra when they are absent.
#
from bisect import bisect_right
from math import atan2, ceil, cos, degrees, radians, sin, sqrt, pi
from typing import Any
import io
import os
import re
import warnings


#
# _fixSVGForRasterize_() - work around a svglib/tinycss2 crash
# - svglib raises AttributeError on a CSS <style> rule with no declaration block
#   (e.g. "<style> .foo </style>").  Such rules are visually inert, so blank out
#   any <style> element whose content has no '{'.
#
def _fixSVGForRasterize_(svg: str) -> str:
    def _blank_empty_style_(m: Any) -> str:
        return '<style></style>' if '{' not in m.group(1) else m.group(0)
    return re.sub(r'<style>(.*?)</style>', _blank_empty_style_, svg, flags=re.DOTALL)


#
# Text on a path, for the rasterizer (PLANNING.md C-textpath-png)
#
# svglib has no <textPath> handling at all: it draws the glyphs at (0, 0), off the
# visible canvas, so chordp's label_style='circular' and linkp's link_shape='curve'
# link labels were missing from every PNG.  _flattenTextPaths_() lays each one out the
# way an SVG renderer would -- every glyph centred on the path at its own arc length and
# rotated to the tangent there -- and replaces it with plain rotated <text>, which svglib
# draws.  The labels therefore curve in the PNG as they do in the SVG.
#
# It works on the parsed document rather than on a component, so it covers a label
# wherever it ends up: a direct save, a smallp panel, a tile().  Glyph advances come
# from the same font converter svglib renders with (Helvetica for the package's
# "'Noto Sans', sans-serif"), not from textLength()'s Noto Sans table -- the spacing has
# to match the face that is actually drawn.
#
# Scope is what an SVG path can say in M/L/H/V/C/Q/A/Z, absolute or relative, with
# startOffset, text-anchor and a single dx/dy.  Anything else (S/T, a path with its own
# transform, an href that does not resolve, mixed content around the <textPath>) is left
# in place and counted, so svgToPNGBytes() can still warn about exactly those.
#
_SVG_PATH_TOKEN_ = re.compile(r'[MmLlHhVvCcSsQqTtAaZz]|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?')
_SVG_PATH_ARITY_ = {'M': 2, 'L': 2, 'H': 1, 'V': 1, 'C': 6, 'Q': 4, 'A': 7, 'Z': 0}
_FLATTEN_STEP_PX_ = 1.0     # chord length when a curve is sampled into line segments
_FLATTEN_MAX_STEPS_ = 4096  # per segment, so a degenerate huge curve stays bounded

def _localName_(el: Any) -> str:
    return el.tag.rsplit('}', 1)[-1] if isinstance(el.tag, str) else ''

#
# _inheritedAttr_() - the value an SVG presentation attribute takes at el: its own
# attribute or style declaration, else the nearest ancestor's
#
def _inheritedAttr_(el: Any, name: str) -> str | None:
    while el is not None:
        _v_ = el.get(name)
        if _v_ is not None: return str(_v_)
        for _decl_ in (el.get('style') or '').split(';'):
            _k_, _, _val_ = _decl_.partition(':')
            if _k_.strip() == name and _val_.strip(): return _val_.strip()
        el = el.getparent()
    return None

#
# _pathSegments_() - an SVG path's d attribute as straight segments (x0, y0, x1, y1)
# - curves are sampled to about _FLATTEN_STEP_PX_ per chord; a moveto adds no length
# - None when the path uses anything outside M/L/H/V/C/Q/A/Z or is malformed
#
def _pathSegments_(d: str) -> list[tuple[float, float, float, float]] | None:
    _tok_ = _SVG_PATH_TOKEN_.findall(d)
    if not _tok_ or not _tok_[0].isalpha(): return None
    _segs_: list[tuple[float, float, float, float]] = []
    _x_ = _y_ = _sx_ = _sy_ = 0.0
    def _line_(x1: float, y1: float) -> None:
        if (x1, y1) != (_x_, _y_): _segs_.append((_x_, _y_, x1, y1))
    def _steps_(length: float) -> int:
        return max(1, min(_FLATTEN_MAX_STEPS_, ceil(length / _FLATTEN_STEP_PX_)))
    _i_, _cmd_ = 0, ''
    while _i_ < len(_tok_):
        if _tok_[_i_].isalpha():
            _cmd_ = _tok_[_i_]; _i_ += 1
            if _cmd_ in 'Zz':
                _line_(_sx_, _sy_); _x_, _y_ = _sx_, _sy_
                continue
        _up_ = _cmd_.upper()
        if _up_ not in _SVG_PATH_ARITY_ or _up_ == 'Z': return None
        _n_ = _SVG_PATH_ARITY_[_up_]
        if _i_ + _n_ > len(_tok_) or any(_t_.isalpha() for _t_ in _tok_[_i_:_i_ + _n_]): return None
        _a_ = [float(_t_) for _t_ in _tok_[_i_:_i_ + _n_]]; _i_ += _n_
        _rel_ = _cmd_.islower()
        _ox_, _oy_ = (_x_, _y_) if _rel_ else (0.0, 0.0)
        if _up_ == 'M':
            _x_, _y_ = _sx_, _sy_ = _a_[0] + _ox_, _a_[1] + _oy_
            _cmd_ = 'l' if _rel_ else 'L'          # further pairs are implicit linetos
        elif _up_ in 'LHV':
            _nx_ = _a_[0] + _ox_ if _up_ != 'V' else _x_
            _ny_ = (_a_[1] + _oy_ if _up_ == 'L' else _a_[0] + _oy_) if _up_ != 'H' else _y_
            _line_(_nx_, _ny_); _x_, _y_ = _nx_, _ny_
        elif _up_ in 'CQ':
            _pts_ = [(_x_, _y_)] + [(_a_[_k_] + _ox_, _a_[_k_ + 1] + _oy_) for _k_ in range(0, _n_, 2)]
            _poly_ = sum(sqrt((_pts_[_k_ + 1][0] - _pts_[_k_][0]) ** 2 + (_pts_[_k_ + 1][1] - _pts_[_k_][1]) ** 2)
                         for _k_ in range(len(_pts_) - 1))
            _m_ = _steps_(_poly_)
            for _k_ in range(1, _m_ + 1):
                _t_ = _k_ / _m_
                _b_: tuple[float, ...] = (
                    ((1 - _t_) ** 3, 3 * (1 - _t_) ** 2 * _t_, 3 * (1 - _t_) * _t_ ** 2, _t_ ** 3)
                    if _up_ == 'C' else ((1 - _t_) ** 2, 2 * (1 - _t_) * _t_, _t_ ** 2))
                _nx_ = sum(_w_ * _p_[0] for _w_, _p_ in zip(_b_, _pts_))
                _ny_ = sum(_w_ * _p_[1] for _w_, _p_ in zip(_b_, _pts_))
                _line_(_nx_, _ny_); _x_, _y_ = _nx_, _ny_
        else:  # 'A' -- endpoint to centre parameterization, SVG 1.1 implementation notes F.6.5
            _rx_, _ry_, _phi_ = abs(_a_[0]), abs(_a_[1]), radians(_a_[2])
            _fa_, _fs_ = _a_[3] != 0.0, _a_[4] != 0.0
            _x2_, _y2_ = _a_[5] + _ox_, _a_[6] + _oy_
            if (_x2_, _y2_) == (_x_, _y_): continue
            if _rx_ == 0.0 or _ry_ == 0.0:
                _line_(_x2_, _y2_); _x_, _y_ = _x2_, _y2_
                continue
            _c_, _s_ = cos(_phi_), sin(_phi_)
            _dx_, _dy_ = (_x_ - _x2_) / 2.0, (_y_ - _y2_) / 2.0
            _x1p_, _y1p_ = _c_ * _dx_ + _s_ * _dy_, -_s_ * _dx_ + _c_ * _dy_
            _lam_ = _x1p_ ** 2 / _rx_ ** 2 + _y1p_ ** 2 / _ry_ ** 2
            if _lam_ > 1.0: _rx_, _ry_ = _rx_ * sqrt(_lam_), _ry_ * sqrt(_lam_)
            _num_ = _rx_ ** 2 * _ry_ ** 2 - _rx_ ** 2 * _y1p_ ** 2 - _ry_ ** 2 * _x1p_ ** 2
            _den_ = _rx_ ** 2 * _y1p_ ** 2 + _ry_ ** 2 * _x1p_ ** 2
            _co_ = sqrt(max(0.0, _num_ / _den_)) * (-1.0 if _fa_ == _fs_ else 1.0)
            _cxp_, _cyp_ = _co_ * _rx_ * _y1p_ / _ry_, -_co_ * _ry_ * _x1p_ / _rx_
            _cx_ = _c_ * _cxp_ - _s_ * _cyp_ + (_x_ + _x2_) / 2.0
            _cy_ = _s_ * _cxp_ + _c_ * _cyp_ + (_y_ + _y2_) / 2.0
            _ux_, _uy_ = (_x1p_ - _cxp_) / _rx_, (_y1p_ - _cyp_) / _ry_
            _vx_, _vy_ = (-_x1p_ - _cxp_) / _rx_, (-_y1p_ - _cyp_) / _ry_
            _t1_ = atan2(_uy_, _ux_)
            _dt_ = atan2(_ux_ * _vy_ - _uy_ * _vx_, _ux_ * _vx_ + _uy_ * _vy_)
            if not _fs_ and _dt_ > 0.0: _dt_ -= 2.0 * pi
            elif _fs_ and _dt_ < 0.0:   _dt_ += 2.0 * pi
            _m_ = _steps_(abs(_dt_) * max(_rx_, _ry_))
            for _k_ in range(1, _m_ + 1):
                if _k_ == _m_:
                    _nx_, _ny_ = _x2_, _y2_
                else:
                    _th_ = _t1_ + _dt_ * _k_ / _m_
                    _nx_ = _cx_ + _rx_ * cos(_th_) * _c_ - _ry_ * sin(_th_) * _s_
                    _ny_ = _cy_ + _rx_ * cos(_th_) * _s_ + _ry_ * sin(_th_) * _c_
                _line_(_nx_, _ny_); _x_, _y_ = _nx_, _ny_
    return _segs_

#
# _lengthValue_() - a single dx/dy/startOffset-style length in user units, or None
# - a list ("1 2 3") counts by its first entry; startOffset's '%' is the caller's job
#
def _lengthValue_(v: Any, attr_conv: Any, em_base: float) -> float | None:
    if v is None: return None
    _first_ = str(v).replace(',', ' ').split()
    if not _first_: return None
    try:
        return float(attr_conv.convertLength(_first_[0], em_base=em_base))
    except (ValueError, TypeError, AttributeError):
        return None

#
# _flattenOneTextPath_() - replace one <text><textPath/></text> with a <g> of per-glyph
# rotated <text>; False (and the element untouched) when it is outside the scope above
#
def _flattenOneTextPath_(text_el: Any, ids: dict, attr_conv: Any, string_width: Any) -> bool:
    _kids_ = [_k_ for _k_ in text_el if isinstance(_k_.tag, str)]
    if len(_kids_) != 1 or _localName_(_kids_[0]) != 'textPath': return False
    _tp_ = _kids_[0]
    if (text_el.text or '').strip() or (_tp_.tail or '').strip(): return False

    _href_ = _tp_.get('href') or _tp_.get('{http://www.w3.org/1999/xlink}href') or ''
    _path_el_ = ids.get(_href_[1:]) if _href_.startswith('#') else None
    if _path_el_ is None or _localName_(_path_el_) != 'path' or _path_el_.get('transform'): return False
    _segs_ = _pathSegments_(_path_el_.get('d') or '')
    if not _segs_: return False

    # SVG's default white-space handling: strip the ends, collapse every run to one space
    _txt_ = ' '.join(''.join(_tp_.itertext()).split())
    if not _txt_: return False

    _ff_ = _inheritedAttr_(text_el, 'font-family') or 'Helvetica'
    _fw_ = _inheritedAttr_(text_el, 'font-weight') or 'normal'
    _fst_ = _inheritedAttr_(text_el, 'font-style') or 'normal'
    try:
        _font_ = attr_conv.convertFontFamily(_ff_, _fw_, _fst_) or 'Helvetica'
        _fs_ = float(attr_conv.convertLength(_inheritedAttr_(text_el, 'font-size') or '12'))
    except (ValueError, TypeError, AttributeError):
        return False

    _cum_ = [0.0]
    for _x0_, _y0_, _x1_, _y1_ in _segs_:
        _cum_.append(_cum_[-1] + sqrt((_x1_ - _x0_) ** 2 + (_y1_ - _y0_) ** 2))
    _total_ = _cum_[-1]

    _so_raw_ = (_tp_.get('startOffset') or '0').strip()
    if _so_raw_.endswith('%'):
        try:    _offset_ = float(_so_raw_[:-1]) / 100.0 * _total_
        except ValueError: return False
    else:
        _so_ = _lengthValue_(_so_raw_, attr_conv, _fs_)
        if _so_ is None: return False
        _offset_ = _so_
    _anchor_ = _inheritedAttr_(_tp_, 'text-anchor') or 'start'
    _width_ = float(string_width(_txt_, _font_, _fs_))
    _start_ = _offset_ - {'middle': _width_ / 2.0, 'end': _width_}.get(_anchor_, 0.0)
    # dx slides along the path; dy offsets perpendicular to it (the path's own +y)
    _start_ += sum(_lengthValue_(_e_.get('dx'), attr_conv, _fs_) or 0.0 for _e_ in (text_el, _tp_))
    _dy_ = sum(_lengthValue_(_e_.get('dy'), attr_conv, _fs_) or 0.0 for _e_ in (text_el, _tp_))

    # The <g> keeps everything the <text> carried that is not positional, so fill, font
    # and any transform still apply to the glyphs by inheritance.
    _positional_ = {'x', 'y', 'dx', 'dy', 'rotate', 'text-anchor', 'textLength', 'lengthAdjust'}
    _g_ = text_el.makeelement(text_el.tag.rsplit('}', 1)[0] + '}g' if text_el.tag.startswith('{') else 'g',
                              {_k_: _v_ for _k_, _v_ in text_el.attrib.items() if _k_ not in _positional_})
    def _r2_(v: float) -> str: return f'{round(v, 2):g}'
    _adv_ = 0.0
    for _ch_ in _txt_:
        _w_ = float(string_width(_ch_, _font_, _fs_))
        _s_ = _start_ + _adv_ + _w_ / 2.0     # the glyph's midpoint, which is what sits on the path
        _adv_ += _w_
        if _ch_ == ' ' or _s_ < 0.0 or _s_ > _total_: continue   # SVG drops glyphs off the path's ends
        _j_ = min(max(bisect_right(_cum_, _s_) - 1, 0), len(_segs_) - 1)
        _x0_, _y0_, _x1_, _y1_ = _segs_[_j_]
        _f_ = (_s_ - _cum_[_j_]) / max(_cum_[_j_ + 1] - _cum_[_j_], 1e-12)
        _th_ = atan2(_y1_ - _y0_, _x1_ - _x0_)
        _gx_ = _x0_ + (_x1_ - _x0_) * _f_ - sin(_th_) * _dy_
        _gy_ = _y0_ + (_y1_ - _y0_) * _f_ + cos(_th_) * _dy_
        _glyph_ = _g_.makeelement(text_el.tag, {
            'x': _r2_(_gx_), 'y': _r2_(_gy_), 'text-anchor': 'middle',
            'transform': f'rotate({_r2_(degrees(_th_))},{_r2_(_gx_)},{_r2_(_gy_)})'})
        _glyph_.text = _ch_
        _g_.append(_glyph_)
    _g_.tail = text_el.tail
    text_el.getparent().replace(text_el, _g_)
    return True

#
# _flattenTextPaths_() - flatten every <textPath> in a parsed (svglib/lxml) document in
# place; returns how many are left, i.e. still missing from the PNG
#
def _flattenTextPaths_(root: Any, attr_conv: Any, string_width: Any) -> int:
    _tps_ = [_e_ for _e_ in root.iter() if _localName_(_e_) == 'textPath']
    if not _tps_: return 0
    _ids_ = {_e_.get('id'): _e_ for _e_ in root.iter() if isinstance(_e_.tag, str) and _e_.get('id')}
    _left_ = 0
    for _tp_ in _tps_:
        _text_el_ = _tp_.getparent()
        if _text_el_ is None or _localName_(_text_el_) != 'text' or _text_el_.getparent() is None \
           or not _flattenOneTextPath_(_text_el_, _ids_, attr_conv, string_width):
            _left_ += 1
    return _left_


#
# svgToPNGBytes() - rasterize an SVG document string to PNG bytes
# - uses svglib (SVG -> reportlab drawing) + reportlab renderPM (drawing -> PNG)
# - raises a clear ImportError pointing at the [export] extra when either the
#   svglib or reportlab dependency is missing
#
# Note on the svglib<2 cap in pyproject's [export] extra: svglib 1.x carries the
# SVG's px width/height straight through as reportlab points (200px -> 200pt), so
# renderPM's default 72dpi rasterizes one SVG user unit to one PNG pixel.  svglib
# 2.x instead applies the correct CSS px->pt conversion (200px -> 150pt), which at
# 72dpi silently shrinks every export to 0.75 scale and mis-scales nested <svg>
# tiles (smallp/spreadlinesp) outright.  reportlab 5 itself is fine here — it is
# svglib 2 that changes the geometry — so the cap is on svglib, not reportlab.
#
# The io.StringIO wrapper below is LOAD-BEARING — do not "simplify" it into a
# temp-file path.  svglib resolves xlink:href/href on <image> and <use> against
# the *source path* of the document it parsed, so rasterizing from a file path
# turns any external reference in the SVG into a filesystem read whose contents
# land in the output PNG.  tile() embeds foreign SVG verbatim (PLANNING.md A3,
# SECURITY.md), so that reference can come from markup polars2svg did not
# produce.  Handing svglib a stream instead leaves its source_path a non-str and
# SvgRenderer.xlink_href_target() then refuses to resolve a path at all — true of
# every version the [export] extra admits (svglib 1.5.0 through 1.6.0), but a
# property of svglib rather than of us, which is why
# tests/test_export_save.py::TestRasterizeDoesNotReadLocalFiles pins the
# behaviour end-to-end instead of trusting this comment.
#
def svgToPNGBytes(svg: str) -> bytes:
    try:
        from svglib.svglib import SvgRenderer, load_svg_file
        from reportlab.graphics import renderPM
        from reportlab.pdfbase.pdfmetrics import stringWidth
    except ImportError as _e_:
        raise ImportError(
            "PNG export requires the optional 'export' dependencies "
            "(svglib, reportlab). Install them with:\n"
            "    pip install polars2svg[export]"
        ) from _e_
    # svg2rlg(), unrolled so the <textPath> labels can be flattened between svglib's
    # parse and its render -- same parser, same renderer, same stream as source path.
    # Typed Any because svglib's annotations say str | PathLike while its docstrings,
    # and the security property above, rely on it taking a file-like object.
    _stream_: Any = io.StringIO(_fixSVGForRasterize_(svg))
    _root_ = load_svg_file(_stream_)
    if _root_ is None:
        raise ValueError('svgToPNGBytes: the SVG document could not be parsed')
    _renderer_ = SvgRenderer(_stream_)
    _n_textpath_ = _flattenTextPaths_(_root_, _renderer_.attrConverter, stringWidth)
    if _n_textpath_:
        warnings.warn(
            f"{_n_textpath_} <textPath> label(s) will be missing from the PNG: svglib cannot "
            f"draw text that follows a path, and these use a form polars2svg cannot flatten "
            f"into straight glyphs. The SVG is correct; save it as .svg to keep them.",
            UserWarning, stacklevel=3)
    return renderPM.drawToString(_renderer_.render(_root_), fmt='PNG')


class ExportMixin:
    #
    # ExportMixin - save()/savePNG() for any component with a _repr_svg_()
    #
    # Mixed into every rendered component.  Relies only on _repr_svg_() returning
    # a standalone SVG document, so it works uniformly across the eight components.
    #

    # ---------------------------------------------------------------------
    # Host-class attributes this mixin reads off `self`.
    #
    # A mixin is never instantiated on its own -- these are provided by whatever
    # class mixes it in (every component).  Declared here so a checker
    # can follow the mixin's own methods; bare annotations, so nothing exists at
    # runtime and nothing is shadowed.
    #
    # Typed `Any` on purpose: the mixin genuinely does not know the concrete type,
    # and several hosts declare the same name with a narrower type of their own.
    # ---------------------------------------------------------------------
    _repr_svg_: Any

    #
    # save() - write this rendering to disk, dispatching on the file extension
    # - a '.png' path rasterizes (see savePNG); every other path writes the SVG
    #   document verbatim (UTF-8).  Returns the path written.
    #
    def save(self, path: str | Path) -> str:
        _path_ = os.fspath(path)
        if _path_.lower().endswith('.png'):
            return self.savePNG(_path_)
        with open(_path_, 'w', encoding='utf-8') as _f_:
            _f_.write(self._repr_svg_())
        return _path_

    #
    # savePNG() - rasterize this rendering to a PNG file
    # - requires the optional [export] extra (svglib + reportlab); raises a clear
    #   ImportError naming it when absent.  Returns the path written.
    #
    def savePNG(self, path: str | Path) -> str:
        _path_      = os.fspath(path)
        _png_bytes_ = svgToPNGBytes(self._repr_svg_())
        with open(_path_, 'wb') as _f_:
            _f_.write(_png_bytes_)
        return _path_
