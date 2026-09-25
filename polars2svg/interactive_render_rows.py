#
# interactive_render_rows - settings-panel rows that change how a generic view RENDERS
#
# The five generic interactive views (xypi, histopi, timepi, chordpi, piepi) carried two
# panel rows -- selection shape and tooltip -- neither of which touches the render.  This
# module is what lets a row change a component parameter: a row set declares its rows
# (mnemonic, value list, the value that reproduces the view as built, when it is live),
# and turns the panel's current values into keyword overrides for the component's
# render_with().  The view (_InteractivePBase) owns everything else: the param the browser
# writes, re-rendering, the cache.
#
# The rule that shapes every row set here: **an untouched panel changes nothing.**  Each
# row's initial value is read off the template, and overrides() returns {} for as long
# as every row still holds it -- not an equivalent spelling of the template's settings,
# nothing -- so a view nobody has opened the panel on renders byte-for-byte as before.
# A row is also only ever the source of its own parameter: moving the aspect row does
# not re-state the distributions.
#
# Deliberately free of panel/param: the rows are plain data and the override logic is
# plain Python, so all of it is testable without the `interactive` extra.
#

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

#: Mnemonic for a value the template was built with that the standard list does not
#: carry -- linkpi's link-shape and spacing pickers use the same one.
AS_BUILT_MNEMONIC = '#'


@dataclass(frozen=True)
class RenderRow:
    '''One settings-panel row that changes the render.

    kind     the menu kind, and the key of this row's value in the view's render_settings
    items    [[mnemonic, label], ...] -- the values `space` cycles and `Enter` lists
    initial  the value that reproduces the template; always one of items
    enabled  settings -> bool; a row whose prerequisite is off is greyed and skipped
    '''
    mnemonic: str
    kind:     str
    label:    str
    items:    list[list[str]]
    initial:  str
    enabled:  Callable[[dict[str, str]], bool]


def _always_(settings: dict[str, str]) -> bool:
    return True


def _with_as_built_(items: list[list[str]], current: str) -> list[list[str]]:
    '''The standard items, plus the template's own value when the list does not have it.'''
    if current in [_label_ for _, _label_ in items]: return [list(_i_) for _i_ in items]
    return [list(_i_) for _i_ in items] + [[AS_BUILT_MNEMONIC, current]]


class RenderRowSet:
    '''A component's render rows.  Subclasses fill self.rows and implement overrides().'''

    def __init__(self) -> None:
        self.rows: list[RenderRow] = []

    def initial_settings(self) -> dict[str, str]:
        return {_r_.kind: _r_.initial for _r_ in self.rows}

    def overrides(self, settings: dict[str, str]) -> dict[str, Any]:
        raise NotImplementedError

    def _changed_(self, settings: dict[str, str], *kinds: str) -> bool:
        _init_ = self.initial_settings()
        return any(settings.get(_k_, _init_[_k_]) != _init_[_k_] for _k_ in kinds)


# ─────────────────────────────────────────────────────────────────────────────
# xyp
# ─────────────────────────────────────────────────────────────────────────────

_DIST_OFF_, _DIST_X_, _DIST_Y_, _DIST_XY_ = 'off', 'x', 'y', 'x+y'
_DIST_ITEMS_ = [['o', _DIST_OFF_], ['x', _DIST_X_], ['y', _DIST_Y_], ['b', _DIST_XY_]]

_PLACE_ITEMS_ = [['u', 'auto'], ['i', 'inside'], ['o', 'outside']]

#: Multiples of xyp's own auto bin count (a pixel-derived one: plot extent / 6, or / the
#: dot size) rather than absolute counts, so every choice is already scaled to the view.
_BIN_MULTIPLIERS_ = {'auto': None, 'auto /4': 0.25, 'auto /2': 0.5, 'auto x2': 2.0, 'auto x4': 4.0}
_BIN_ITEMS_ = [['u', 'auto'], ['1', 'auto /4'], ['2', 'auto /2'], ['3', 'auto x2'], ['4', 'auto x4']]

_ASPECT_ITEMS_ = [['n', 'none'], ['e', 'equal'], ['g', 'geo']]

#: linkpi's link-opacity list, so the two opacity rows read the same.
_OPACITY_ITEMS_ = [[str((_p_ // 10) % 10), str(_p_)] for _p_ in range(10, 101, 10)]

_COLOR_SCALE_ITEMS_ = [['m', 'magnitude'], ['s', 'stretched']]

_LEGEND_ITEMS_ = [['o', 'off'], ['r', 'right'], ['b', 'bottom'], ['t', 'top'], ['l', 'left']]

#: The default (None) is the ascending sort, so 'sorted' IS the default -- see xyp's
#: _ORDER_STRINGS_ for the rest.
_ORDER_TO_PARAM_ = {'sorted': None, 'reverse': 'reverse', 'by count': 'count', 'spectral': 'spectral'}
_ORDER_ITEMS_    = [['s', 'sorted'], ['r', 'reverse'], ['c', 'by count'], ['p', 'spectral']]


class XYpRenderRows(RenderRowSet):
    '''xypi's render rows: distributions (which axes, placement, bins), aspect, dot
    opacity, colour scale, legend, and the order of a categorical x / y axis.'''

    def __init__(self, template: Any) -> None:
        super().__init__()
        self.t   = template
        self.p2s = template.p2s
        _p_      = self.p2s
        self._placement_enums_ = {_p_.DISTRIBUTION_INSIDEp, _p_.DISTRIBUTION_OUTSIDEp}

        # ── distributions ──
        self._raw_dist_ = {'x': template.x_distributions, 'y': template.y_distributions}
        _on_ = [_a_ for _a_ in ('x', 'y') if self._raw_dist_[_a_] is not None]
        _dist_init_ = {(): _DIST_OFF_, ('x',): _DIST_X_, ('y',): _DIST_Y_, ('x', 'y'): _DIST_XY_}[tuple(_on_)]
        self._measure_ = {_a_: self.__measure__(self._raw_dist_[_a_]) for _a_ in ('x', 'y')}
        _shown_ = self._measure_['x'] if self._measure_['x'] is not None else self._measure_['y']
        self._dist_label_ = f'distributions ({self.__describe__(_shown_)})'

        _place_init_ = self.__placementAsBuilt__()
        _bins_init_  = self.__binsAsBuilt__()
        def _dist_on_(s: dict[str, str]) -> bool:
            return s.get('distributions', _dist_init_) != _DIST_OFF_

        _binnable_   = [_a_ for _a_ in ('x', 'y') if not template.axisIsPeriodicTime(_a_)]

        def _bins_live_(s: dict[str, str]) -> bool:
            _d_ = s.get('distributions', _dist_init_)
            _axes_ = {_DIST_X_: ['x'], _DIST_Y_: ['y'], _DIST_XY_: ['x', 'y']}.get(_d_, [])
            return any(_a_ in _binnable_ for _a_ in _axes_)

        # ── the rest ──
        _numeric_     = template.axisIsNumeric('x') and template.axisIsNumeric('y')
        _aspect_init_ = template.aspect if isinstance(template.aspect, str) else \
                        ('none' if template.aspect is None else str(template.aspect))
        _opacity_init_ = self.__opacityAsBuilt__()
        self._color_scale_init_ = self.__colorScaleAsBuilt__()
        _legend_init_  = self.__legendAsBuilt__()
        self._order_init_ = {_a_: self.__orderAsBuilt__(_a_) for _a_ in ('x', 'y')}

        self.rows = [
            RenderRow('d', 'distributions', self._dist_label_, [list(_i_) for _i_ in _DIST_ITEMS_],
                      _dist_init_, _always_),
            RenderRow('p', 'placement', 'placement', _with_as_built_(_PLACE_ITEMS_, _place_init_),
                      _place_init_, _dist_on_),
            RenderRow('b', 'bins', 'bins', _with_as_built_(_BIN_ITEMS_, _bins_init_),
                      _bins_init_, _bins_live_),
            RenderRow('e', 'aspect', 'aspect', _with_as_built_(_ASPECT_ITEMS_, _aspect_init_),
                      _aspect_init_, lambda s: _numeric_),
            RenderRow('o', 'opacity', 'dot opacity', _with_as_built_(_OPACITY_ITEMS_, _opacity_init_),
                      _opacity_init_, lambda s: self.__opacityIsLiteral__()),
            RenderRow('c', 'color_scale', 'color scale',
                      _with_as_built_(_COLOR_SCALE_ITEMS_, self._color_scale_init_ or 'magnitude'),
                      self._color_scale_init_ or 'magnitude',
                      lambda s: self._color_scale_init_ is not None),
            RenderRow('l', 'legend', 'legend', _with_as_built_(_LEGEND_ITEMS_, _legend_init_),
                      _legend_init_, lambda s: template.color is not None),
            RenderRow('x', 'x_order', 'x order', _with_as_built_(_ORDER_ITEMS_, self._order_init_['x']),
                      self._order_init_['x'], lambda s: template.axisIsCategorical('x')),
            RenderRow('y', 'y_order', 'y order', _with_as_built_(_ORDER_ITEMS_, self._order_init_['y']),
                      self._order_init_['y'], lambda s: template.axisIsCategorical('y')),
        ]

    # ── overrides ────────────────────────────────────────────────────────────

    def overrides(self, settings: dict[str, str]) -> dict[str, Any]:
        _init_ = self.initial_settings()
        _s_    = {**_init_, **settings}
        _out_: dict[str, Any] = {}

        if self._changed_(_s_, 'aspect'):
            _out_['aspect'] = None if _s_['aspect'] == 'none' else _s_['aspect']
        if self._changed_(_s_, 'opacity'):
            _out_['opacity'] = int(_s_['opacity']) / 100.0
        if self._changed_(_s_, 'color_scale'):
            _out_['color'] = self.__swapColorScale__(self.t.color, _s_['color_scale'])
        if self._changed_(_s_, 'legend'):
            _out_['legend'] = None if _s_['legend'] == 'off' else _s_['legend']
        for _a_ in ('x', 'y'):
            if self._changed_(_s_, f'{_a_}_order'):
                _out_[f'{_a_}_order'] = _ORDER_TO_PARAM_[_s_[f'{_a_}_order']]

        # Last, because a bin multiple is a multiple of the auto count THIS render would
        # choose -- which depends on the plot extent, which the rows above change.
        if self._changed_(_s_, 'distributions', 'placement', 'bins'):
            _out_.update(self.__distributionOverrides__(_s_, _out_))
        return _out_

    def __distributionOverrides__(self, s: dict[str, str], others: dict[str, Any]) -> dict[str, Any]:
        _axes_ = {_DIST_OFF_: [], _DIST_X_: ['x'], _DIST_Y_: ['y'], _DIST_XY_: ['x', 'y']}[s['distributions']]
        _mult_ = _BIN_MULTIPLIERS_.get(s['bins'])
        _spec_: dict[str, list | None] = {
            _a_: (self.__spec__(_a_, s['placement'], s['bins']) if _a_ in _axes_ else None)
            for _a_ in ('x', 'y')}
        if _mult_ is not None and _axes_:
            # A probe render at auto: xyp records the count it chose in *_distributions_clean.
            _probe_ = self.t.render_with(self.t.df_orig, **others,
                                         x_distributions=_spec_['x'], y_distributions=_spec_['y'])
            for _a_ in _axes_:
                if self.t.axisIsPeriodicTime(_a_): continue   # one bar per period unit, always
                _auto_ = getattr(_probe_, f'{_a_}_distributions_clean')['bins'][0]
                _spec_[_a_] = list(_spec_[_a_] or []) + [max(1, int(round(_auto_ * _mult_)))]
        return {'x_distributions': _spec_['x'], 'y_distributions': _spec_['y']}

    def __spec__(self, axis: str, placement: str, bins: str) -> list:
        '''One axis's distribution spec: what to measure, plus placement, plus bins (auto
        unless 'as built').  The measure is the axis's own from the template, else the
        other axis's, else row counts.'''
        _p_ = self.p2s
        _measure_ = self._measure_[axis] if self._measure_[axis] is not None else self._measure_['y' if axis == 'x' else 'x']
        _out_ = list(_measure_) if _measure_ is not None else [_p_.ROW_COUNTp]
        if   placement == 'inside':  _out_.append(_p_.DISTRIBUTION_INSIDEp)
        elif placement == 'outside': _out_.append(_p_.DISTRIBUTION_OUTSIDEp)
        elif placement != 'auto':    _out_.extend(_i_ for _i_ in self.__items__(self._raw_dist_[axis]) if _i_ in self._placement_enums_)
        if bins not in _BIN_MULTIPLIERS_:   # 'as built'
            _out_.extend(_i_ for _i_ in self.__items__(self._raw_dist_[axis]) if self.__isBins__(_i_))
        return _out_

    # ── reading the template ─────────────────────────────────────────────────

    def __items__(self, raw: Any) -> list:
        '''A distribution spec's top-level items.  A tuple or list at the TOP is a list of
        items; a tuple inside it is a multi-field entry and is kept whole.'''
        if raw is None:                      return []
        if isinstance(raw, (list, tuple)):   return list(raw)
        return [raw]

    def __isBins__(self, item: Any) -> bool:
        return (isinstance(item, int) and not isinstance(item, bool)) or item is self.p2s.DISTRIBUTION_AUTOBINp

    def __measure__(self, raw: Any) -> list | None:
        '''What a spec measures: everything but its placement and bins, which the panel
        owns.  An empty remainder measures rows.'''
        if raw is None: return None
        _kept_ = [_i_ for _i_ in self.__items__(raw)
                  if _i_ not in self._placement_enums_ and not self.__isBins__(_i_)]
        _names_ = [_i_ for _i_ in _kept_ if isinstance(_i_, (str, tuple)) and not isinstance(_i_, self.p2s.HexColorString)]
        if not _names_ and self.p2s.ROW_COUNTp not in _kept_: _kept_.append(self.p2s.ROW_COUNTp)
        return _kept_

    def __describe__(self, measure: list | None) -> str:
        if measure is None: return 'rows'
        _names_ = [(_i_[0] if isinstance(_i_, tuple) else _i_) for _i_ in measure
                   if isinstance(_i_, (str, tuple)) and not isinstance(_i_, self.p2s.HexColorString)]
        return ', '.join(str(_n_) for _n_ in _names_) if _names_ else 'rows'

    def __placementAsBuilt__(self) -> str:
        _set_ = [[_i_ for _i_ in self.__items__(self._raw_dist_[_a_]) if _i_ in self._placement_enums_]
                 for _a_ in ('x', 'y') if self._raw_dist_[_a_] is not None]
        if not any(_set_): return 'auto'
        return 'as built'

    def __binsAsBuilt__(self) -> str:
        for _a_ in ('x', 'y'):
            if any(isinstance(_i_, int) and not isinstance(_i_, bool) for _i_ in self.__items__(self._raw_dist_[_a_])):
                return 'as built'
        return 'auto'

    def __opacityIsLiteral__(self) -> bool:
        return self.t.opacity is None or (isinstance(self.t.opacity, (int, float)) and not isinstance(self.t.opacity, bool))

    def __opacityAsBuilt__(self) -> str:
        if self.t.opacity is None or not self.__opacityIsLiteral__(): return '100'
        return str(int(round(float(self.t.opacity) * 100)))

    def __colorScaleAsBuilt__(self) -> str | None:
        _items_ = self.__items__(self.t.color)
        for _i_ in _items_:
            _name_ = getattr(_i_, 'name', '')
            if not isinstance(_name_, str): continue
            if 'MAGNITUDE' in _name_: return 'magnitude'
            if 'STRETCHED' in _name_: return 'stretched'
        return None

    def __swapColorScale__(self, color: Any, scale: str) -> Any:
        _from_, _to_ = ('STRETCHED', 'MAGNITUDE') if scale == 'magnitude' else ('MAGNITUDE', 'STRETCHED')
        def _swap_(item: Any) -> Any:
            _name_ = getattr(item, 'name', None)
            if isinstance(_name_, str) and _from_ in _name_:
                return getattr(self.p2s, _name_.replace(_from_, _to_), item)
            return item
        if isinstance(color, tuple): return tuple(_swap_(_i_) for _i_ in color)
        if isinstance(color, list):  return [_swap_(_i_) for _i_ in color]
        return _swap_(color)

    def __legendAsBuilt__(self) -> str:
        _l_ = self.t.legend
        if _l_ is None or _l_ is False: return 'off'
        if _l_ is True:                 return 'right'
        if isinstance(_l_, str):        return _l_
        return 'as built'

    def __orderAsBuilt__(self, axis: str) -> str:
        _o_ = getattr(self.t, f'{axis}_order')
        for _label_, _param_ in _ORDER_TO_PARAM_.items():
            if not isinstance(_o_, (list, dict)) and _o_ == _param_: return _label_
        return 'as built'
