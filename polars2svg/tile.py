# Ported from racetrack_svg_framework/rtsvg/rt_small_multiples_mixin.py (tile / table)
# Original author: David Trimm — Apache License 2.0
# The two methods are one component here; see the class comment for what changed.

import time
import random
import re

import polars2svg
from polars2svg.export import ExportMixin

#
# Tile - compose already-rendered SVGs into a strip or a grid
#
# The one composition primitive that does not touch a DataFrame: it takes renderings
# (any component, any SVG string, another Tile) and lays them out as a single SVG
# document.  Ported from rtsvg's tile()/table() pair, which are one component here --
# per_row is the whole layout model: None is a single row (rtsvg's tile), <n> is a grid
# (rtsvg's table), and 1 is a single column (rtsvg's tile(horz=False), which needed a
# second parameter to say the same thing).
#
# Placement is by <g transform="translate(x,y)"> around each child's own <svg> element,
# the same nesting smallp uses for its cells, rather than rtsvg's rewrite of the child's
# x=/y= attributes (which was string surgery on the first x=" it found anywhere in the
# markup).  Child sizes come from the width/height of the child's root <svg> tag, so a
# component whose rendered size differs from the wxh it was asked for (smallp) measures
# correctly without the caller knowing which attribute to read.
#
_SVG_ROOT_TAG_  = re.compile(r'<svg\b[^>]*>')
# The leading \s matters: \bwidth= would also match a stroke-width= on the root tag.
_SVG_ATTR_W_    = re.compile(r'\swidth="([^"]*)"')
_SVG_ATTR_H_    = re.compile(r'\sheight="([^"]*)"')


#
# _svgRootWxh_() - (width, height) from the root <svg> tag of an SVG document
# - the framework always emits numeric, unit-less width/height on that tag; anything
#   else (units, percentages, a missing attribute) raises rather than guessing a size
#
def _svgRootWxh_(svg, where):
    _tag_ = _SVG_ROOT_TAG_.search(svg)
    if _tag_ is None:
        raise ValueError(f'Tile: {where} does not contain an <svg> element')
    _w_, _h_ = _SVG_ATTR_W_.search(_tag_.group(0)), _SVG_ATTR_H_.search(_tag_.group(0))
    if _w_ is None or _h_ is None:
        raise ValueError(f'Tile: {where} has no width/height on its root <svg> tag -- '
                         f'tile() cannot place a rendering whose size it cannot read')
    try:
        return float(_w_.group(1)), float(_h_.group(1))
    except ValueError:
        raise ValueError(f'Tile: {where} has a non-numeric root width/height '
                         f'("{_w_.group(1)}" x "{_h_.group(1)}") -- units and percentages '
                         f'are not supported') from None


#
# _fmt_() - shortest exact string for a coordinate; whole numbers lose the '.0'
# - sizes are pixel counts and are almost always integral, so this keeps the composed
#   markup as clean as the components' own (float precision is preserved when it matters)
#
def _fmt_(v):
    return str(int(v)) if float(v).is_integer() else str(v)


class Tile(ExportMixin):

    _VALID_KWARGS = frozenset({
        'svg_list', 'per_row', 'spacer', 'wxh', 'bg_color',
    })

    def __init__(self, *args, **kwargs):
        self.t_start        = time.time()
        self.p2s            = polars2svg.Polars2SVG()
        self.timing_metrics = {}
        self.gatherMetrics(self.__parseInput__, *args, **kwargs)
        self.gatherMetrics(self.__validateInput__)
        rand_id = random.randint(0, 2**32)  # nosec B311 - non-cryptographic SVG id scoping, see SECURITY.md
        self.gatherMetrics(self.__constructGeometry__)
        self.gatherMetrics(self.__renderSVG__, rand_id)
        self.t_end     = time.time()
        self.t_overall = self.t_end - self.t_start

    def _repr_svg_(self): return self.svg

    def gatherMetrics(self, callable, *args, **kwargs):
        t0 = time.time()
        _results_ = callable(*args, **kwargs)
        t1 = time.time()
        if callable.__name__ not in self.timing_metrics: self.timing_metrics[callable.__name__] = 0.0
        self.timing_metrics[callable.__name__] += t1 - t0
        return _results_

    def __parseInput__(self, *args, **kwargs):
        _unknown_ = set(kwargs) - self._VALID_KWARGS
        if _unknown_:
            raise TypeError(f'Tile: unexpected keyword argument(s): {sorted(_unknown_)}')

        # Single source of truth for tile's parameters (name -> default).  svg_list
        # arrives positionally and is handled explicitly, so it is the spec's `extra`.
        _defaults_ = {
            'per_row':  None,
            'spacer':   0,
            'wxh':      None,
            'bg_color': None,
        }
        self.p2s.assertParamSpecMatches('Tile', self._VALID_KWARGS, _defaults_, extra=('svg_list',))

        # A lone rendering is a one-element list: tile(chart, wxh=...) is the natural
        # way to scale a single component into a fixed viewport.
        self.svg_list = None
        for _arg_ in args:
            _as_list_ = list(_arg_) if isinstance(_arg_, (list, tuple)) else [_arg_]
            if self.svg_list is None: self.svg_list = _as_list_
            else:                     raise ValueError('Tile.__parseInput__(): svg_list already set')
        if 'svg_list' in kwargs:
            if self.svg_list is not None: raise ValueError('Tile.__parseInput__(): svg_list already set')
            _kw_ = kwargs['svg_list']
            self.svg_list = list(_kw_) if isinstance(_kw_, (list, tuple)) else [_kw_]

        self.p2s.assignKwargsWithDefaults(self, _defaults_, kwargs)

    def __validateInput__(self):
        if self.svg_list is None:
            raise ValueError('Tile.__validateInput__(): svg_list must be specified')

        if self.per_row is not None:
            if isinstance(self.per_row, bool) or not isinstance(self.per_row, int):
                raise ValueError(f'Tile.__validateInput__(): per_row must be None or an int, '
                                 f'got {type(self.per_row).__name__} {self.per_row!r}')
            if self.per_row < 1:
                raise ValueError(f'Tile.__validateInput__(): per_row must be at least 1, got {self.per_row}')

        self.spacer = self.__normalizeSpacer__(self.spacer)

        if self.wxh is not None:
            self.wxh = self.p2s.normalizeWxh(self.wxh, 'Tile', allow_none=True)

        if self.bg_color is not None and not isinstance(self.bg_color, str):
            raise ValueError(f'Tile.__validateInput__(): bg_color must be a color string, '
                             f'got {type(self.bg_color).__name__} {self.bg_color!r}')

    #
    # __normalizeSpacer__() - one number is that much separation both horizontally and
    # vertically; a 2-sequence is (horizontal, vertical).  Always normalized to the pair,
    # so self.spacer reads back as what was actually used and can be handed straight to
    # another tile() call.  A layout with a single row has no vertical gaps to draw (and
    # per_row=1 no horizontal ones), so there the unused half is simply never applied.
    #
    def __normalizeSpacer__(self, spacer):
        def _check_(_v_, _side_):
            if isinstance(_v_, bool) or not isinstance(_v_, (int, float)):
                raise ValueError(f'Tile.__validateInput__(): spacer {_side_} must be a number, '
                                 f'got {type(_v_).__name__} {_v_!r}')
            if _v_ < 0:
                raise ValueError(f'Tile.__validateInput__(): spacer {_side_} must not be negative, got {_v_}')
            return _v_
        if isinstance(spacer, (tuple, list)):
            if len(spacer) != 2:
                raise ValueError(f'Tile.__validateInput__(): spacer must be a number or a 2-sequence '
                                 f'(horizontal, vertical), got {len(spacer)} elements ({spacer!r})')
            return (_check_(spacer[0], 'horizontal'), _check_(spacer[1], 'vertical'))
        return (_check_(spacer, 'horizontal'), _check_(spacer, 'vertical'))

    #
    # __constructGeometry__() - measure every tile, then place them row by row
    # - the strip and the grid are the same layout: a row of everything (per_row=None),
    #   one tile per row (per_row=1), or n per row
    # - within a row tiles are top-aligned and the row is as tall as its tallest tile;
    #   rows are left-aligned and the canvas is as wide as its widest row (rtsvg's
    #   table() built the same shape by tiling rows of tiles)
    #
    def __constructGeometry__(self):
        self.svg_strings = []
        _wh_ = []
        for _i_, _item_ in enumerate(self.svg_list):
            if isinstance(_item_, str):
                _svg_ = _item_
            else:
                _renderer_ = getattr(_item_, '_repr_svg_', None)
                _svg_      = _renderer_() if callable(_renderer_) else None
            if not isinstance(_svg_, str):
                raise ValueError(f'Tile.__constructGeometry__(): svg_list[{_i_}] is a '
                                 f'{type(_item_).__name__} -- expected an SVG string or a rendered '
                                 f'component (anything with a _repr_svg_())')
            self.svg_strings.append(_svg_)
            _wh_.append(_svgRootWxh_(_svg_, f'svg_list[{_i_}]'))

        self.xy_list = []
        if len(_wh_) == 0:
            _w_, _h_ = self.wxh if self.wxh is not None else (None, None)
            self.content_wxh = self.wxh_actual = (_w_ if _w_ is not None else 256,
                                                  _h_ if _h_ is not None else 256)
            self.scale, self.offset = 1.0, (0, 0)
            return

        _per_row_            = self.per_row if self.per_row is not None else len(_wh_)
        _x_space_, _y_space_ = self.spacer
        _y_, _w_max_         = 0.0, 0.0
        for _i0_ in range(0, len(_wh_), _per_row_):
            _row_ = _wh_[_i0_:_i0_ + _per_row_]
            _x_   = 0.0
            for _w_, _h_ in _row_:
                self.xy_list.append((_x_, _y_))
                _x_ += _w_ + _x_space_
            _w_max_ = max(_w_max_, _x_ - _x_space_)        # trailing spacer is not part of the row
            _y_    += max(_h_ for _, _h_ in _row_) + _y_space_
        self.content_wxh = (_w_max_, _y_ - _y_space_)      # ... nor is the trailing row spacer

        self.wxh_actual = self.content_wxh if self.wxh is None else self.__viewportWxh__()
        self.scale, self.offset = self.__viewportTransform__()

    #
    # __viewportWxh__() - the requested canvas size, completing a None side from the
    # content's aspect ratio (wxh=(640, None) is "640 wide, as tall as that makes it")
    #
    def __viewportWxh__(self):
        _cw_, _ch_ = self.content_wxh
        _w_,  _h_  = self.wxh
        if _w_ is None: _w_ = round(_cw_ * _h_ / _ch_) if _ch_ > 0 else _cw_
        if _h_ is None: _h_ = round(_ch_ * _w_ / _cw_) if _cw_ > 0 else _ch_
        return (_w_, _h_)

    #
    # __viewportTransform__() - scale factor and centering offset for the viewport
    # - this is what SVG's own preserveAspectRatio="xMidYMid meet" would do on a nested
    #   <svg viewBox=...>: scale uniformly by whichever side binds first (so the whole
    #   tiling stays visible and undistorted) and center the result in the canvas.
    # - it is computed here and emitted as a transform rather than left to a viewBox
    #   because svglib -- the rasterizer behind savePNG()/save('.png') -- does not
    #   honor a viewBox on a *nested* <svg>: it scaled x and y independently and
    #   clipped the overflow, so a tile that looked right in a browser exported wrong.
    #   A translate+scale transform renders identically in both.
    #
    def __viewportTransform__(self):
        _ow_, _oh_ = self.wxh_actual
        _cw_, _ch_ = self.content_wxh
        _scales_   = [_s_ for _s_ in (_ow_ / _cw_ if _cw_ > 0 else None,
                                      _oh_ / _ch_ if _ch_ > 0 else None) if _s_ is not None]
        _scale_    = min(_scales_) if _scales_ else 1.0
        return _scale_, ((_ow_ - _cw_ * _scale_) / 2, (_oh_ - _ch_ * _scale_) / 2)

    def __renderSVG__(self, rand_id):
        _ow_, _oh_ = self.wxh_actual
        if len(self.svg_strings) == 0:
            self.svg = self.p2s.placeholderSVG(_ow_, _oh_, message='no data - empty svg_list')
            return
        _bg_ = self.bg_color if self.bg_color is not None else self.p2s.colorTyped('background', 'default')
        _svg_ = [f'<svg id="tile_{rand_id}" x="0" y="0" width="{_fmt_(_ow_)}" height="{_fmt_(_oh_)}" '
                 f'font-family="{self.p2s.default_font}" xmlns="http://www.w3.org/2000/svg">',
                 f'<rect x="0" y="0" width="{_fmt_(_ow_)}" height="{_fmt_(_oh_)}" fill="{_bg_}" />']
        # A requested wxh scales the whole tiling into the canvas (see
        # __viewportTransform__); bg_color shows through whatever the aspect difference
        # leaves over.  When the request matches the natural size there is nothing to
        # scale, so the wrapper is omitted and the markup is identical to wxh=None.
        _tx_, _ty_ = self.offset
        _scaled_   = (self.scale, _tx_, _ty_) != (1.0, 0, 0)
        if _scaled_:
            _svg_.append(f'<g transform="translate({_fmt_(round(_tx_, 3))},{_fmt_(round(_ty_, 3))}) '
                         f'scale({_fmt_(round(self.scale, 6))})">')
        for _child_, (_x_, _y_) in zip(self.svg_strings, self.xy_list):
            _svg_.append(f'<g transform="translate({_fmt_(_x_)},{_fmt_(_y_)})">{_child_}</g>')
        if _scaled_: _svg_.append('</g>')
        _svg_.append('</svg>')
        self.svg = ''.join(_svg_)
