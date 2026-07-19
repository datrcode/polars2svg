import math

import polars as pl

from .exceptions      import InvalidSpecError
from .p2s_displaylist import DisplayList

# Accepted legend= positions (layer 2 of the spec); True aliases to 'right'.
LEGEND_POSITIONS = ('right', 'left', 'top', 'bottom')
# Accepted keys for the dict form (layer 3 of the spec).
LEGEND_DICT_KEYS = frozenset({'pos', 'title', 'fmt', 'max_items', 'order'})

# Layout constants shared by reserve sizing and rendering -- reserve promises the
# space that render consumes, so they must come from one place.
_PAD_             = 4     # inner padding around the legend strip content
_SWATCH_GAP_      = 3     # gap between a categorical swatch and its label
_ROW_GAP_         = 2     # vertical gap between stacked rows
_BAR_W_           = 10    # colorbar thickness (vertical orientation)
_ENTRY_GAP_       = 10    # gap between horizontal categorical entries
_MAX_LABEL_EMS_   = 8.0   # categorical label width cap, in multiples of txt_h
_CAPTURE_CAP_     = 64    # max categorical entries captured when max_items is unset

#
# LegendInfo - the metadata-capture hook: what color scale / categories a render
# actually used, captured by the component during its color pipeline and consumed
# by legendRenderDL().  Exposed on rendered components as `component.legend_info`
# (None when no legend was requested or there was nothing to legend).
#
class LegendInfo:
    def __init__(self, kind, title=None):
        self.kind       = kind    # 'colorbar' | 'categorical'
        self.title      = title   # resolved title string ('' suppresses)
        # categorical
        self.entries    = None    # ordered [(label, '#rrggbb'), ...]
        self.overflow   = 0       # count of categories beyond the captured entries
        # colorbar
        self.vmin       = None    # numeric domain minimum (post-aggregation)
        self.vmax       = None    # numeric domain maximum (post-aggregation)
        self.vmin_label = None    # formatted tick label for vmin
        self.vmax_label = None    # formatted tick label for vmax

    def __repr__(self):
        if self.kind == 'categorical':
            _n_ = 0 if self.entries is None else len(self.entries)
            return f'LegendInfo(categorical, title={self.title!r}, entries={_n_}, overflow={self.overflow})'
        return f'LegendInfo(colorbar, title={self.title!r}, vmin={self.vmin}, vmax={self.vmax})'


class P2SLegendMixin:
    def __init__(self):
        pass

    #
    # __p2s_legend_mixin_init__() - initialization via the mixin methodology
    #
    def __p2s_legend_mixin_init__(self):
        pass

    #
    # legendResolveSpec() - normalize the layered legend= value into a spec dict
    # - False/None -> None (no legend); True -> 'right'; a position string; or a
    #   dict with keys pos/title/fmt/max_items/order (layer 3)
    # - raises InvalidSpecError on anything else so a typo'd position fails fast
    #
    def legendResolveSpec(self, legend):
        if legend is None or legend is False:
            return None
        if legend is True:
            legend = 'right'
        if isinstance(legend, str):
            if legend not in LEGEND_POSITIONS:
                raise InvalidSpecError(f'legend= position must be one of {LEGEND_POSITIONS}; got {legend!r}')
            return {'pos': legend, 'title': None, 'fmt': None, 'max_items': None, 'order': 'count'}
        if isinstance(legend, dict):
            _unknown_ = set(legend) - LEGEND_DICT_KEYS
            if _unknown_:
                raise InvalidSpecError(f'legend= dict has unknown key(s) {sorted(_unknown_)}; '
                                       f'valid keys: {sorted(LEGEND_DICT_KEYS)}')
            _pos_ = legend.get('pos', 'right')
            if _pos_ is True: _pos_ = 'right'
            if _pos_ not in LEGEND_POSITIONS:
                raise InvalidSpecError(f'legend= position must be one of {LEGEND_POSITIONS}; got {_pos_!r}')
            _max_items_ = legend.get('max_items')
            if _max_items_ is not None and (isinstance(_max_items_, bool) or
                                            not isinstance(_max_items_, int) or _max_items_ < 1):
                raise InvalidSpecError(f'legend= max_items must be a positive int; got {_max_items_!r}')
            _order_ = legend.get('order', 'count')
            if _order_ is None: _order_ = 'count'
            if not (_order_ in ('count', 'label') or isinstance(_order_, (list, tuple))):
                raise InvalidSpecError(f"legend= order must be 'count', 'label', or an explicit value list; "
                                       f'got {_order_!r}')
            _fmt_ = legend.get('fmt')
            if _fmt_ is not None and not (isinstance(_fmt_, str) or callable(_fmt_)):
                raise InvalidSpecError(f'legend= fmt must be a format string or a callable; got {_fmt_!r}')
            _title_ = legend.get('title')
            if _title_ is not None and not isinstance(_title_, str):
                raise InvalidSpecError(f'legend= title must be a string; got {_title_!r}')
            return {'pos': _pos_, 'title': _title_, 'fmt': _fmt_, 'max_items': _max_items_, 'order': _order_}
        raise InvalidSpecError(f'legend= expects bool, a position string, or a dict; got {type(legend).__name__}')

    #
    # legendKind() - auto-select the legend kind from a resolved ColorTypeP mode
    # - CSETp -> categorical swatch list; every other ColorTypeP member maps a value
    #   onto the spectrum -> colorbar; None (flat color / literal hex) -> no legend
    #
    def legendKind(self, color_mode):
        if color_mode is None:            return None
        if color_mode == self.CSETp:      return 'categorical'
        if isinstance(color_mode, self.ColorTypeP): return 'colorbar'
        return None

    #
    # legendModeIsStretched() - True for the rank-equalized spectrum modes (the
    # colorbar gradient is then rank-based between the labeled min/max, not linear)
    #
    def legendModeIsStretched(self, color_mode):
        return color_mode in (self.CROW_STRETCHEDp,    self.CSET_STRETCHEDp,
                              self.CSTRETCHED_SUMp,    self.CSTRETCHED_MINp,
                              self.CSTRETCHED_MEDIANp, self.CSTRETCHED_MEANp,
                              self.CSTRETCHED_MAXp)

    #
    # legendFormatValue() - format a numeric tick / category value for display
    # - fmt: None (default humanized), a str.format spec string, or a callable
    #
    def legendFormatValue(self, value, fmt=None):
        if value is None: return ''
        if fmt is not None:
            if callable(fmt): return str(fmt(value))
            return fmt.format(value)
        if isinstance(value, bool):  return str(value)
        if isinstance(value, int):   return self.unitizeInt(value)
        if isinstance(value, float):
            if abs(value) >= 1000.0: return self.unitizeInt(value)
            return f'{value:.4g}'
        return str(value)

    #
    # legendCategoricalValueCounts() - shared capture query for categorical legends
    # - groups by the *string-cast* column (the exact keys the CSETp render pipeline
    #   colorizes, so the swatch hash matches the drawn color) while retaining a raw
    #   value per key so fmt= can format numeric categories
    # - returns [(key_str, count, raw_value), ...] (unordered; legendInfoCategorical sorts)
    # - weight: optional numeric column summed per key instead of counting rows (used
    #   by components whose df is already aggregated, e.g. histop's df_agg '__count__')
    #
    def legendCategoricalValueCounts(self, df, column, weight=None):
        _n_expr_ = pl.len() if weight is None else pl.col(weight).sum()
        _agg_ = df.group_by(pl.col(column).cast(pl.String).alias('__legend_key__')) \
                  .agg(_n_expr_.alias('__legend_n__'),
                       pl.col(column).first().alias('__legend_raw__'))
        return list(zip(_agg_['__legend_key__'].to_list(),
                        _agg_['__legend_n__'].to_list(),
                        _agg_['__legend_raw__'].to_list()))

    #
    # legendInfoCategorical() - build the categorical LegendInfo from value counts
    # - value_counts: [(key_str, count, raw_value), ...] from legendCategoricalValueCounts()
    # - order: 'count' (frequency desc, the default), 'label' (alphabetic), or an
    #   explicit value list (unlisted values follow, frequency desc)
    # - entries are capped at max_items (or a 64-entry capture cap); the remainder
    #   is carried in .overflow and rendered as '... N more'
    # - hex_lu: optional {key_str: '#rrggbb'} overriding the default string-hash
    #   colors, for components that colorize the raw (non-string-cast) values
    #
    def legendInfoCategorical(self, spec, value_counts, title, hex_lu=None):
        _order_ = spec['order']
        if isinstance(_order_, (list, tuple)):
            _pos_lu_ = {str(_v_): _i_ for _i_, _v_ in enumerate(_order_)}
            _pairs_  = sorted(value_counts, key=lambda t: (_pos_lu_.get(t[0], len(_pos_lu_)), -t[1], t[0]))
        elif _order_ == 'label':
            _pairs_  = sorted(value_counts, key=lambda t: t[0])
        else:  # 'count'
            _pairs_  = sorted(value_counts, key=lambda t: (-t[1], t[0]))
        _cap_   = spec['max_items'] if spec['max_items'] is not None else _CAPTURE_CAP_
        _shown_ = _pairs_[:_cap_]
        _hex_lu_ = hex_lu if hex_lu is not None else self.colors([_t_[0] for _t_ in _shown_])
        _info_  = LegendInfo('categorical', title)
        _info_.entries = []
        for _key_, _n_, _raw_ in _shown_:
            if spec['fmt'] is not None and isinstance(_raw_, (int, float)) and not isinstance(_raw_, bool):
                _label_ = self.legendFormatValue(_raw_, spec['fmt'])
            else:
                _label_ = self.formatMultiFieldValue(_key_)
            _info_.entries.append((_label_, _hex_lu_[_key_]))
        _info_.overflow = len(_pairs_) - len(_shown_)
        return _info_

    #
    # legendInfoColorbar() / legendInfoColorbarFinalize() - two-phase colorbar capture:
    # the shell is created when the component reserves space (domain still unknown);
    # the domain is filled in after the color aggregation has actually run
    #
    def legendInfoColorbar(self, title):
        return LegendInfo('colorbar', title)

    def legendInfoColorbarFinalize(self, info, spec, vmin, vmax):
        info.vmin, info.vmax = vmin, vmax
        info.vmin_label = self.legendFormatValue(vmin, spec['fmt'])
        info.vmax_label = self.legendFormatValue(vmax, spec['fmt'])
        return info

    #
    # legendReserve() - pixels to carve out of wxh for the legend strip
    # - returns (left, right, top, bottom); exactly one side is non-zero
    # - the plot region shrinks by this amount (wxh stays the physical output size)
    # - deterministic before any aggregation: colorbar width is sized for a worst-case
    #   unitized tick label; categorical width from the already-captured entry labels
    #   (plus the '... N more' overflow row, which carries no swatch)
    #
    def legendReserve(self, spec, info, txt_h, wxh):
        _pos_ = spec['pos']
        if _pos_ in ('top', 'bottom'):
            _lh_ = min(txt_h + 2 * _PAD_, wxh[1] // 3)
            return (0, 0, _lh_, 0) if _pos_ == 'top' else (0, 0, 0, _lh_)
        if info.kind == 'colorbar':
            _lw_ = int(math.ceil(max(self.textLength('-9.999M', txt_h), float(_BAR_W_)))) + 2 * _PAD_
        else:
            _sw_    = max(8, txt_h - 4)
            _labels_ = [_e_[0] for _e_ in (info.entries or [])]
            _lab_w_ = max((self.textLength(str(_l_), txt_h) for _l_ in _labels_), default=0.0)
            if info.overflow > 0:
                _lab_w_ = max(_lab_w_, self.textLength(f'... {info.overflow} more', txt_h) - _sw_ - _SWATCH_GAP_)
            _lab_w_ = min(_lab_w_, _MAX_LABEL_EMS_ * txt_h)
            _lw_    = _PAD_ + _sw_ + _SWATCH_GAP_ + int(math.ceil(_lab_w_)) + _PAD_
        if info.title:
            _lw_ = max(_lw_, int(math.ceil(min(self.textLength(str(info.title), txt_h),
                                               _MAX_LABEL_EMS_ * txt_h))) + 2 * _PAD_)
        _lw_ = min(_lw_, wxh[0] // 3)
        return (_lw_, 0, 0, 0) if _pos_ == 'left' else (0, _lw_, 0, 0)

    #
    # legendSpectrumColor() - Python-side sample of the same spectrum the polars
    # pipeline interpolates (colorSpectrumPolarsOperations); t in [0,1] -> '#rrggbb'
    #
    def legendSpectrumColor(self, t):
        _tuples_ = self.colorSpectrumTuples()
        t = 0.0 if t is None else min(max(float(t), 0.0), 1.0)
        j = 1
        while j < len(_tuples_) - 1 and t >= _tuples_[j][3]:
            j += 1
        _r0_, _g0_, _b0_, _m0_ = _tuples_[j - 1]
        _r1_, _g1_, _b1_, _m1_ = _tuples_[j]
        _f_ = (t - _m0_) / (_m1_ - _m0_)
        _r_ = min(max(_r0_ + (_r1_ - _r0_) * _f_, 0.0), 1.0)
        _g_ = min(max(_g0_ + (_g1_ - _g0_) * _f_, 0.0), 1.0)
        _b_ = min(max(_b0_ + (_b1_ - _b0_) * _f_, 0.0), 1.0)
        return f'#{int(_r_ * 255):02x}{int(_g_ * 255):02x}{int(_b_ * 255):02x}'

    #
    # legendRenderDL() - draw the legend into a DisplayList (SVG + WebGPU both get it)
    # - canvas_wxh: the component's full wxh (DisplayList canvas dims)
    # - region: (x, y, w, h) strip reserved by legendReserve()
    # - the gradient is drawn as stacked solid rects (backend-neutral -- no SVG-only
    #   <linearGradient>), sampled from legendSpectrumColor()
    # - scale: unit multiplier for every pixel-sized quantity (pads, swatches, text
    #   height).  1.0 draws in canvas pixels; a viewBox-scaled component
    #   (spreadlinesp) passes region in *world* units with scale = 1/viewbox_scale
    #   so the legend renders at true pixel size after the root transform
    #
    def legendRenderDL(self, canvas_wxh, region, spec, info, txt_h, scale=1.0):
        _dl_ = DisplayList(canvas_wxh[0], canvas_wxh[1])
        if info is None or region is None: return _dl_
        if spec['pos'] in ('top', 'bottom'):
            self.__legendRenderHorizontal__(_dl_, region, info, txt_h, scale)
        else:
            self.__legendRenderVertical__(_dl_, region, info, txt_h, scale)
        return _dl_

    # rect helper: DisplayList.rect() records the GPU op; the SVG string must be
    # supplied by the caller (mirrors how components feed the display list)
    def __legendRect__(self, _dl_, x, y, w, h, fill):
        _svg_ = f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" fill="{fill}" />'
        _dl_.rect(x, y, w, h, fill, svg=_svg_)

    def __legendGradientRects__(self, _dl_, x, y, w, h, vertical, k=1.0):
        _len_ = h if vertical else w
        _n_   = min(64, max(8, int(_len_ / k)))
        _seg_ = _len_ / _n_
        for _i_ in range(_n_):
            _t_ = (_i_ + 0.5) / _n_
            if vertical:  # top = vmax
                self.__legendRect__(_dl_, x, y + _i_ * _seg_, w, _seg_ + 0.5 * k, self.legendSpectrumColor(1.0 - _t_))
            else:         # left = vmin
                self.__legendRect__(_dl_, x + _i_ * _seg_, y, _seg_ + 0.5 * k, h, self.legendSpectrumColor(_t_))

    def __legendRenderVertical__(self, _dl_, region, info, txt_h, k=1.0):
        _x_, _y_, _w_, _h_ = region
        txt_h      = txt_h * k
        _pad_      = _PAD_ * k
        _rgap_     = _ROW_GAP_ * k
        _sgap_     = _SWATCH_GAP_ * k
        _fg_       = self.colorTyped('label', 'defaultfg')
        _muted_    = self.colorTyped('axis',  'label')
        _baseline_ = txt_h - 3 * k
        _cy_       = _y_ + _pad_
        _bottom_   = _y_ + _h_ - _pad_
        if info.title:
            _t_ = self.cropText(str(info.title), txt_h, _w_ - 2 * _pad_)
            _dl_.text(self, _t_, _x_ + _pad_, _cy_ + _baseline_, txt_h=txt_h, color=_fg_)
            _cy_ += txt_h + _rgap_
        if info.kind == 'colorbar':
            _vmax_ = self.cropText(str(info.vmax_label or ''), txt_h, _w_ - 2 * _pad_)
            _vmin_ = self.cropText(str(info.vmin_label or ''), txt_h, _w_ - 2 * _pad_)
            _dl_.text(self, _vmax_, _x_ + _pad_, _cy_ + _baseline_, txt_h=txt_h, color=_muted_)
            _cy_ += txt_h + _rgap_
            _bar_bot_ = _bottom_ - txt_h - _rgap_
            if _bar_bot_ - _cy_ >= 8 * k:
                self.__legendGradientRects__(_dl_, _x_ + _pad_, _cy_, _BAR_W_ * k, _bar_bot_ - _cy_, vertical=True, k=k)
            _dl_.text(self, _vmin_, _x_ + _pad_, _bottom_ - txt_h + _baseline_, txt_h=txt_h, color=_muted_)
        else:
            _rh_       = txt_h + _rgap_
            _entries_  = info.entries or []
            _overflow_ = info.overflow
            _n_fit_    = max(0, int((_bottom_ - _cy_) // _rh_))
            if len(_entries_) + (1 if _overflow_ > 0 else 0) > _n_fit_:
                _shown_    = _entries_[:max(0, _n_fit_ - 1)]
                _overflow_ = _overflow_ + (len(_entries_) - len(_shown_))
            else:
                _shown_    = _entries_
            _sw_ = max(8 * k, txt_h - 4 * k)
            for _label_, _hex_ in _shown_:
                self.__legendRect__(_dl_, _x_ + _pad_, _cy_ + (_rh_ - _sw_) / 2.0 - k, _sw_, _sw_, _hex_)
                _lab_ = self.cropText(str(_label_), txt_h, _w_ - 2 * _pad_ - _sw_ - _sgap_)
                _dl_.text(self, _lab_, _x_ + _pad_ + _sw_ + _sgap_, _cy_ + _baseline_, txt_h=txt_h, color=_fg_)
                _cy_ += _rh_
            if _overflow_ > 0 and _cy_ + _rh_ <= _bottom_ + 0.5 * k:
                _lab_ = self.cropText(f'... {_overflow_} more', txt_h, _w_ - 2 * _pad_)
                _dl_.text(self, _lab_, _x_ + _pad_, _cy_ + _baseline_, txt_h=txt_h, color=_muted_)

    def __legendRenderHorizontal__(self, _dl_, region, info, txt_h, k=1.0):
        _x_, _y_, _w_, _h_ = region
        txt_h      = txt_h * k
        _pad_      = _PAD_ * k
        _sgap_     = _SWATCH_GAP_ * k
        _egap_     = _ENTRY_GAP_ * k
        _fg_       = self.colorTyped('label', 'defaultfg')
        _muted_    = self.colorTyped('axis',  'label')
        _by_       = _y_ + (_h_ - txt_h) / 2.0 + txt_h - 3 * k   # single-row baseline
        _cursor_   = _x_ + _pad_
        _right_    = _x_ + _w_ - _pad_
        if info.title:
            _t_ = self.cropText(str(info.title), txt_h, _w_ / 4.0)
            _dl_.text(self, _t_, _cursor_, _by_, txt_h=txt_h, color=_fg_)
            _cursor_ += self.textLength(_t_, txt_h) + 2 * _pad_
        if info.kind == 'colorbar':
            _vmin_   = str(info.vmin_label or '')
            _vmax_   = str(info.vmax_label or '')
            _vmax_w_ = self.textLength(_vmax_, txt_h)
            _dl_.text(self, _vmin_, _cursor_, _by_, txt_h=txt_h, color=_muted_)
            _cursor_ += self.textLength(_vmin_, txt_h) + _pad_
            _bar_w_  = _right_ - _vmax_w_ - _pad_ - _cursor_
            _bh_     = max(6 * k, txt_h - 4 * k)
            if _bar_w_ >= 8 * k:
                self.__legendGradientRects__(_dl_, _cursor_, _y_ + (_h_ - _bh_) / 2.0, _bar_w_, _bh_, vertical=False, k=k)
            _dl_.text(self, _vmax_, _right_, _by_, txt_h=txt_h, color=_muted_, anchor='end')
        else:
            _sw_       = max(8 * k, txt_h - 4 * k)
            _sy_       = _y_ + (_h_ - _sw_) / 2.0
            _entries_  = info.entries or []
            _overflow_ = info.overflow
            _more_w_   = self.textLength(f'... {_overflow_ + len(_entries_)} more', txt_h)  # worst case
            for _i_, (_label_, _hex_) in enumerate(_entries_):
                _lab_w_  = min(self.textLength(str(_label_), txt_h), _MAX_LABEL_EMS_ * txt_h)
                _item_w_ = _sw_ + _sgap_ + _lab_w_
                _tail_w_ = _more_w_ + _egap_ if (_overflow_ > 0 or _i_ < len(_entries_) - 1) else 0.0
                if _cursor_ + _item_w_ + _tail_w_ > _right_:
                    _overflow_ += len(_entries_) - _i_
                    break
                self.__legendRect__(_dl_, _cursor_, _sy_, _sw_, _sw_, _hex_)
                _lab_ = self.cropText(str(_label_), txt_h, _lab_w_ + 1)
                _dl_.text(self, _lab_, _cursor_ + _sw_ + _sgap_, _by_, txt_h=txt_h, color=_fg_)
                _cursor_ += _item_w_ + _egap_
            if _overflow_ > 0:
                _dl_.text(self, f'... {_overflow_} more', _cursor_, _by_, txt_h=txt_h, color=_muted_)
