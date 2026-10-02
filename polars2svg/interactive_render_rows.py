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
# Nearly every row is a ParamRow: one parameter, and a fixed parameter value for each
# label it offers.  Those need nothing beyond their declaration -- RenderRowSet turns
# them into overrides -- and the ones more than one component carries (legend, colour
# scale, opacity, count, on/off, bar style) have a shared builder below that reads its own
# starting value.  Only a row that shares parameters needs code of its own: xyp's
# distributions, per-axis placement and per-axis bins all land in the same two, and
# histop's order row sets order= and descending= together.
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
from typing import Any, cast

#: Mnemonic for a value the template was built with that the standard list does not
#: carry -- linkpi's link-shape and spacing pickers use the same one.
AS_BUILT_MNEMONIC = '#'

#: Each render row's key, by row kind: one key per setting, the same on every view that
#: has it, so a key learned on one view works on the next.  A row takes its key from here
#: and nowhere else, and every row's kind must be listed (the render-row tests check both),
#: which also makes this the one place that shows which keys are taken.
#:
#: The mapping runs one way.  Keys only have to be unique within a view, and there are not
#: enough letters for a key to mean one thing everywhere: linkpi's panel predates this
#: table, and its g / r / t / d / p are layout shape, arrows, timing marks, community
#: detection and spacing.
RENDER_ROW_KEYS: dict[str, str] = {
    # settings most views have
    'legend':           'g',
    'color_scale':      'c',
    'node_color_scale': 'C',    # chordpi's second colour parameter: shift, for the variant
    'count':            'm',    # measure -- 'n' is node size
    'style':            't',    # 's' is the selection-shape row
    'order':            'r',
    'draw_labels':      'l',    # linkpi's labels row is 'l' as well
    # marks, on linkpi's keys wherever linkpi has the row
    'opacity':          'o',
    'link_opacity':     'o',
    'node_opacity':     'f',    # not 'e', which is xypi's aspect
    'node_size':        'n',
    'link_size':        'z',
    'link_shape':       'h',
    # settings only one view has
    'distributions':    'd',    # xypi -- and histopi's strip below is the same idea
    'distribution':     'd',
    'x_placement':      'p',    # xypi; shift for the y axis's, as chordpi's second colour scale
    'y_placement':      'P',
    'x_bins':           'b',    # xypi
    'y_bins':           'B',
    'aspect':           'e',    # xypi
    'x_order':          'x',    # xypi
    'y_order':          'y',    # xypi
    'granularity':      'q',    # timepi
    'bundle_strength':  'u',    # chordpi
    'label_style':      'v',    # chordpi
}

#: Keys no render row may take: the panel's cursor keys (a / A / j / k), and the two rows
#: every generic panel starts with -- selection shape (s) and tooltip (i).
RESERVED_ROW_KEYS = frozenset('aAjksi')


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


@dataclass(frozen=True)
class ParamRow(RenderRow):
    '''A render row that is the sole source of one component parameter.

    param    the parameter it sets
    values   {label: parameter value} for every standard item.  An as-built item has no
             entry and needs none: it can only be the row's initial value, and a row at
             its initial value sets nothing.  The values are handed to every render
             as-is, which is safe because no component mutates a parameter in place.
    '''
    param:  str
    values: dict[str, Any]


def _always_(settings: dict[str, str]) -> bool:
    return True


def _with_as_built_(items: list[list[str]], current: str) -> list[list[str]]:
    '''The standard items, plus the template's own value when the list does not have it.'''
    if current in [_label_ for _, _label_ in items]: return [list(_i_) for _i_ in items]
    return [list(_i_) for _i_ in items] + [[AS_BUILT_MNEMONIC, current]]


def _param_row_(kind: str, label: str, param: str,
                choices: list[tuple[str, str, Any]], initial: str,
                enabled: Callable[[dict[str, str]], bool] = _always_) -> ParamRow:
    '''A ParamRow from its (mnemonic, label, parameter value) choices, keyed from
    RENDER_ROW_KEYS and offering the template's own value as built when the choices do not
    include it.'''
    return ParamRow(RENDER_ROW_KEYS[kind], kind, label,
                    _with_as_built_([[_m_, _l_] for _m_, _l_, _ in choices], initial),
                    initial, enabled, param, {_l_: _v_ for _, _l_, _v_ in choices})


class RenderRowSet:
    '''A component's render rows.  Subclasses fill self.rows: a ParamRow needs nothing
    more, and a set with rows that share parameters implements _groupOverrides_().

    A set with a count row takes count_fields= (accepts_count_fields): the further fields
    the row may switch to -- see count_row().'''

    accepts_count_fields: bool = False

    def __init__(self) -> None:
        self.rows: list[RenderRow] = []

    def initial_settings(self) -> dict[str, str]:
        return {_r_.kind: _r_.initial for _r_ in self.rows}

    def overrides(self, settings: dict[str, str]) -> dict[str, Any]:
        '''The keyword overrides the panel's values call for: the parameter of each
        ParamRow that has moved off its initial value, then whatever the grouped rows set
        -- last, because a group may depend on the rest (xyp's bins probe-render at them).'''
        _s_ = {**self.initial_settings(), **settings}
        _out_: dict[str, Any] = {}
        for _r_ in self.rows:
            if isinstance(_r_, ParamRow) and self._changed_(_s_, _r_.kind):
                _out_[_r_.param] = _r_.values[_s_[_r_.kind]]
        _out_.update(self._groupOverrides_(_s_, _out_))
        return _out_

    def _groupOverrides_(self, settings: dict[str, str], others: dict[str, Any]) -> dict[str, Any]:
        '''What the rows that are not ParamRows set.  `settings` is complete, and `others`
        is what the ParamRows are already setting.  None, unless a subclass has such rows.'''
        return {}

    def _changed_(self, settings: dict[str, str], *kinds: str) -> bool:
        _init_ = self.initial_settings()
        return any(settings.get(_k_, _init_[_k_]) != _init_[_k_] for _k_ in kinds)


# ─────────────────────────────────────────────────────────────────────────────
# Shared rows -- one builder per row that more than one component carries.  Each reads
# its own starting value off the template, so a row set only says which rows it has.
# ─────────────────────────────────────────────────────────────────────────────

def _top_items_(raw: Any) -> list:
    '''A spec's top-level items.  A tuple or list at the TOP is a list of items; a tuple
    inside it is a multi-field entry and is kept whole.'''
    if raw is None:                      return []
    if isinstance(raw, (list, tuple)):   return list(raw)
    return [raw]


_LEGEND_CHOICES_ = [('o', 'off', None), ('r', 'right', 'right'), ('b', 'bottom', 'bottom'),
                    ('t', 'top', 'top'), ('l', 'left', 'left')]


#: Every shared builder takes `also`: the component's own condition for the row being
#: live, on top of the builder's.  A histop boxplot, for one, draws no legend.
Gate = Callable[[dict[str, str]], bool]


def legend_row(template: Any, also: Gate = _always_, legendable: bool | None = None) -> ParamRow:
    '''legend= as off / right / bottom / top / left.  True reads as 'right' (the side
    legendResolveSpec gives it) and a dict spec as 'as built'.  Live when the plot has a
    colour for a legend to explain -- template.color, unless the component says otherwise
    through `legendable` (chordp's legend can describe its node colour instead).'''
    _l_ = template.legend
    if   _l_ is None or _l_ is False: _init_ = 'off'
    elif _l_ is True:                 _init_ = 'right'
    elif isinstance(_l_, str):        _init_ = _l_
    else:                             _init_ = 'as built'
    _colored_ = template.color is not None if legendable is None else legendable
    return _param_row_('legend', 'legend', 'legend', _LEGEND_CHOICES_, _init_,
                       lambda s: _colored_ and also(s))


def _color_scale_of_(color: Any) -> str | None:
    '''magnitude / stretched for a colour spec holding such an enum, else None.'''
    for _i_ in _top_items_(color):
        _name_ = getattr(_i_, 'name', '')
        if not isinstance(_name_, str): continue
        if 'MAGNITUDE' in _name_: return 'magnitude'
        if 'STRETCHED' in _name_: return 'stretched'
    return None


def _swap_color_scale_(p2s: Any, color: Any, scale: str) -> Any:
    '''The colour spec with its magnitude / stretched enum moved to `scale`, keeping a
    list or tuple spec's shape.'''
    _from_, _to_ = ('STRETCHED', 'MAGNITUDE') if scale == 'magnitude' else ('MAGNITUDE', 'STRETCHED')
    def _swap_(item: Any) -> Any:
        _name_ = getattr(item, 'name', None)
        if isinstance(_name_, str) and _from_ in _name_:
            return getattr(p2s, _name_.replace(_from_, _to_), item)
        return item
    if isinstance(color, tuple): return tuple(_swap_(_i_) for _i_ in color)
    if isinstance(color, list):  return [_swap_(_i_) for _i_ in color]
    return _swap_(color)


def color_scale_row(template: Any, param: str = 'color', also: Gate = _always_) -> ParamRow:
    '''A colour parameter's scale, magnitude or stretched.  The two are enum pairs that
    differ in that one word (CROW_MAGNITUDEp / CROW_STRETCHEDp, CMAGNITUDE_SUMp /
    CSTRETCHED_SUMp, ...), so each value is the template's own spec with the enum swapped
    by name.  Live only when the parameter holds such an enum.  The row's kind is
    `<param>_scale`, so a component with two colour parameters can carry both.'''
    _color_ = getattr(template, param)
    _scale_ = _color_scale_of_(_color_)
    _choices_ = [('m', 'magnitude', _swap_color_scale_(template.p2s, _color_, 'magnitude')),
                 ('s', 'stretched', _swap_color_scale_(template.p2s, _color_, 'stretched'))]
    _kind_ = f'{param}_scale'
    return _param_row_(_kind_, _kind_.replace('_', ' '), param, _choices_,
                       _scale_ or 'magnitude', lambda s: _scale_ is not None and also(s))


#: linkpi's link-opacity list, so every opacity row reads the same.
_OPACITY_CHOICES_ = [(str((_p_ // 10) % 10), str(_p_), _p_ / 100.0) for _p_ in range(10, 101, 10)]


def opacity_row(template: Any, param: str = 'opacity', label: str = 'opacity') -> ParamRow:
    '''An opacity parameter as a percentage, 10 to 100; the row's kind is the parameter's
    name.  Greyed out when the opacity is data-driven (a column), which a fixed
    percentage would silently replace.'''
    _v_ = getattr(template, param)
    _literal_ = _v_ is None or (isinstance(_v_, (int, float)) and not isinstance(_v_, bool))
    _init_ = str(int(round(float(_v_) * 100))) if _v_ is not None and _literal_ else '100'
    return _param_row_(param, label, param, _OPACITY_CHOICES_, _init_, lambda s: _literal_)


#: linkpi's on/off values, so every on/off row reads the same.
_ON_OFF_CHOICES_ = [('f', 'off', False), ('n', 'on', True)]


def on_off_row(template: Any, param: str, label: str, also: Gate = _always_) -> ParamRow:
    '''A boolean parameter as off / on; the row's kind is the parameter's name.'''
    return _param_row_(param, label, param, _ON_OFF_CHOICES_,
                       'on' if getattr(template, param) else 'off', also)


def _describe_spec_(spec: Any) -> str:
    '''A field spec as the panel shows it: 'bytes', or 'ip (SETp)' for a tuple.'''
    if not isinstance(spec, tuple): return str(spec)
    _fields_ = [str(_i_) for _i_ in spec if isinstance(_i_, str)]
    _enums_  = [_i_.name for _i_ in spec if not isinstance(_i_, str) and hasattr(_i_, 'name')]
    return ', '.join(_fields_) + (f' ({", ".join(_enums_)})' if _enums_ else '')


#: Picker keys for count_fields= entries: digits, then letters -- never r / f, which are
#: rows and the field the view was built with.
_COUNT_FIELD_KEYS_ = '123456789' + ''.join(_c_ for _c_ in 'abcdeghijklmnopqstuvwxyz')


def count_row(template: Any, also: Gate = _always_, fields: list | None = None) -> ParamRow:
    '''count= as rows, the field the view was built with, and the fields count_fields=
    names.  Every numeric column is not offered by default, because in netflow data a
    numeric column is as often an identifier (a port) as a quantity -- and a column's type
    cannot tell bytes from a port; the caller names the ones that are quantities.  Greyed
    out when there is nothing but rows to choose.'''
    _p_ = template.p2s
    def _label_(spec: Any) -> str:
        _l_ = _describe_spec_(spec)
        return 'rows (field)' if _l_ == 'rows' else _l_
    _choices_: list[tuple[str, str, Any]] = [('r', 'rows', _p_.ROW_COUNTp)]
    _init_ = 'rows'
    if template.count is not _p_.ROW_COUNTp:
        _init_ = _label_(template.count)
        _choices_.append(('f', _init_, template.count))
    _keys_ = iter(_COUNT_FIELD_KEYS_)
    for _spec_ in (fields or []):
        if _label_(_spec_) in [_l_ for _, _l_, _ in _choices_]: continue   # already offered
        _choices_.append((next(_keys_, AS_BUILT_MNEMONIC), _label_(_spec_), _spec_))
    _live_ = len(_choices_) > 1
    return _param_row_('count', 'count', 'count', _choices_, _init_, lambda s: _live_ and also(s))


def _count_numeric_gate_(template: Any, count: ParamRow) -> Gate:
    '''Live while the count row holds a numeric field (Histop / Timep.numericCountField()):
    what a boxplot needs.  Follows whichever field the row has moved to.'''
    def _gate_(s: dict[str, str]) -> bool:
        _label_ = s.get('count', count.initial)
        if _label_ not in count.values: return template.numericCountField() is not None   # as built
        return template.numericCountField(count.values[_label_]) is not None
    return _gate_


def bar_style_row(template: Any, also: Gate = _always_) -> ParamRow:
    '''style= for the bar components (histop, timep): bar, boxplot, boxplot with a swarm.
    STACKEDBARp draws exactly what BARCHARTp does -- a categorical colour stacks either
    one -- so it is not offered, and shows only when the view was built with it.  Whether
    a boxplot is possible is the component's call, made through `also`.'''
    _p_ = template.p2s
    _choices_ = [('b', 'bar', _p_.BARCHARTp), ('x', 'boxplot', _p_.BOXPLOTp),
                 ('s', 'boxplot+swarm', _p_.BOXPLOT_W_SWARMp)]
    _init_ = next((_l_ for _, _l_, _v_ in _choices_ if template.style is _v_),
                  'stacked' if template.style is _p_.STACKEDBARp else str(template.style))
    return _param_row_('style', 'style', 'style', _choices_, _init_, also)


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

_ASPECT_CHOICES_ = [('n', 'none', None), ('e', 'equal', 'equal'), ('g', 'geo', 'geo')]

#: The default (None) is the ascending sort, so 'sorted' IS the default -- see xyp's
#: _ORDER_STRINGS_ for the rest.
_ORDER_CHOICES_ = [('s', 'sorted', None), ('r', 'reverse', 'reverse'),
                   ('c', 'by count', 'count'), ('p', 'spectral', 'spectral')]


class XYpRenderRows(RenderRowSet):
    '''xypi's render rows: distributions (which axes, and each axis's placement and bins),
    aspect, dot opacity, colour scale, legend, and the order of a categorical x / y axis.

    Placement and bins are a row per axis.  The axes are rarely alike -- time along a long
    x against a number up a short y is the common case -- so one row setting both could
    only ever suit one of them (user feedback 2026-09-27).'''

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

        def _axis_on_(axis: str) -> Callable[[dict[str, str]], bool]:
            _with_ = {'x': (_DIST_X_, _DIST_XY_), 'y': (_DIST_Y_, _DIST_XY_)}[axis]
            return lambda s: s.get('distributions', _dist_init_) in _with_

        _binnable_ = {_a_: not template.axisIsPeriodicTime(_a_) for _a_ in ('x', 'y')}

        def _bins_live_(axis: str) -> Callable[[dict[str, str]], bool]:
            _on_ = _axis_on_(axis)
            return lambda s: _binnable_[axis] and _on_(s)

        # ── the rest ──
        _numeric_     = template.axisIsNumeric('x') and template.axisIsNumeric('y')
        _aspect_init_ = template.aspect if isinstance(template.aspect, str) else \
                        ('none' if template.aspect is None else str(template.aspect))

        _keys_ = RENDER_ROW_KEYS
        self.rows = [
            RenderRow(_keys_['distributions'], 'distributions', self._dist_label_,
                      [list(_i_) for _i_ in _DIST_ITEMS_], _dist_init_, _always_),
            *[RenderRow(_keys_[f'{_a_}_placement'], f'{_a_}_placement', f'{_a_} placement',
                        [list(_i_) for _i_ in _PLACE_ITEMS_], self.__placementAsBuilt__(_a_), _axis_on_(_a_))
              for _a_ in ('x', 'y')],
            *[RenderRow(_keys_[f'{_a_}_bins'], f'{_a_}_bins', f'{_a_} bins',
                        _with_as_built_(_BIN_ITEMS_, self.__binsAsBuilt__(_a_)), self.__binsAsBuilt__(_a_),
                        _bins_live_(_a_))
              for _a_ in ('x', 'y')],
            _param_row_('aspect', 'aspect', 'aspect', _ASPECT_CHOICES_, _aspect_init_,
                        lambda s: _numeric_),
            opacity_row(template, label='dot opacity'),
            color_scale_row(template),
            legend_row(template),
            self.__orderRow__('x'),
            self.__orderRow__('y'),
        ]

    def __orderRow__(self, axis: str) -> ParamRow:
        '''A categorical axis's order.  A list or dict x_order= / y_order= is as built.'''
        _o_ = getattr(self.t, f'{axis}_order')
        _init_ = next((_l_ for _, _l_, _v_ in _ORDER_CHOICES_
                       if not isinstance(_o_, (list, dict)) and _o_ == _v_), 'as built')
        return _param_row_(f'{axis}_order', f'{axis} order', f'{axis}_order',
                           _ORDER_CHOICES_, _init_, lambda s: self.t.axisIsCategorical(axis))

    # ── overrides ────────────────────────────────────────────────────────────

    # The distributions group: which axes, and each axis's placement and bins, all land in
    # x_distributions / y_distributions.  After the ParamRows, because a bin multiple is a
    # multiple of the auto count THIS render would choose -- which depends on the plot
    # extent, which those rows change.
    def _groupOverrides_(self, settings: dict[str, str], others: dict[str, Any]) -> dict[str, Any]:
        if not self._changed_(settings, 'distributions', 'x_placement', 'y_placement', 'x_bins', 'y_bins'): return {}
        return self.__distributionOverrides__(settings, others)

    def __distributionOverrides__(self, s: dict[str, str], others: dict[str, Any]) -> dict[str, Any]:
        _axes_ = {_DIST_OFF_: [], _DIST_X_: ['x'], _DIST_Y_: ['y'], _DIST_XY_: ['x', 'y']}[s['distributions']]
        _mult_ = {_a_: _BIN_MULTIPLIERS_.get(s[f'{_a_}_bins']) for _a_ in ('x', 'y')}
        _spec_: dict[str, list | None] = {
            _a_: (self.__spec__(_a_, s[f'{_a_}_placement'], s[f'{_a_}_bins']) if _a_ in _axes_ else None)
            for _a_ in ('x', 'y')}
        # Periodic time axes draw one bar per period unit, always, so a multiple means nothing.
        _scaled_ = [_a_ for _a_ in _axes_ if _mult_[_a_] is not None and not self.t.axisIsPeriodicTime(_a_)]
        if _scaled_:
            # One probe render with every scaled axis at auto: xyp records the count it chose
            # in *_distributions_clean, and the plot extent it chose it for includes the
            # other axis's distribution as the real render will have it.
            _probe_ = self.t.render_with(self.t.df_orig, **others,
                                         x_distributions=_spec_['x'], y_distributions=_spec_['y'])
            for _a_ in _scaled_:
                _auto_ = getattr(_probe_, f'{_a_}_distributions_clean')['bins'][0]
                # Aimed with AUTOBINp rather than given as a bare count, so a whole-number axis
                # keeps whole integers per bin (PLANNING.md §5 C-xypi-bin-multiple-whole-numbers).
                _spec_[_a_] = list(_spec_[_a_] or []) + [self.p2s.DISTRIBUTION_AUTOBINp, max(1, int(round(_auto_ * cast(float, _mult_[_a_]))))]
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
        if bins not in _BIN_MULTIPLIERS_:   # 'as built'
            _out_.extend(_i_ for _i_ in _top_items_(self._raw_dist_[axis]) if self.__isBins__(_i_))
        return _out_

    # ── reading the template ─────────────────────────────────────────────────

    def __isBins__(self, item: Any) -> bool:
        return (isinstance(item, int) and not isinstance(item, bool)) or item is self.p2s.DISTRIBUTION_AUTOBINp

    def __measure__(self, raw: Any) -> list | None:
        '''What a spec measures: everything but its placement and bins, which the panel
        owns.  An empty remainder measures rows.'''
        if raw is None: return None
        _kept_ = [_i_ for _i_ in _top_items_(raw)
                  if _i_ not in self._placement_enums_ and not self.__isBins__(_i_)]
        _names_ = [_i_ for _i_ in _kept_ if isinstance(_i_, (str, tuple)) and not isinstance(_i_, self.p2s.HexColorString)]
        if not _names_ and self.p2s.ROW_COUNTp not in _kept_: _kept_.append(self.p2s.ROW_COUNTp)
        return _kept_

    def __describe__(self, measure: list | None) -> str:
        if measure is None: return 'rows'
        _names_ = [(_i_[0] if isinstance(_i_, tuple) else _i_) for _i_ in measure
                   if isinstance(_i_, (str, tuple)) and not isinstance(_i_, self.p2s.HexColorString)]
        return ', '.join(str(_n_) for _n_ in _names_) if _names_ else 'rows'

    def __placementAsBuilt__(self, axis: str) -> str:
        '''inside / outside when the axis's spec names one, else auto (xyp resolves it).'''
        _items_ = _top_items_(self._raw_dist_[axis])
        if self.p2s.DISTRIBUTION_INSIDEp  in _items_: return 'inside'
        if self.p2s.DISTRIBUTION_OUTSIDEp in _items_: return 'outside'
        return 'auto'

    def __binsAsBuilt__(self, axis: str) -> str:
        if any(isinstance(_i_, int) and not isinstance(_i_, bool) for _i_ in _top_items_(self._raw_dist_[axis])):
            return 'as built'
        return 'auto'


# ─────────────────────────────────────────────────────────────────────────────
# histop
# ─────────────────────────────────────────────────────────────────────────────

_BOXPLOTS_ = ('boxplot', 'boxplot+swarm')

#: The order row's values, each a (order=, descending=) pair.  'sorted' and 'reverse' are
#: the words xypi uses for the same two orders of the values themselves.
_HISTOP_ORDER_ITEMS_ = [['l', 'largest first'], ['s', 'smallest first'], ['a', 'sorted'], ['z', 'reverse']]


class HistopRenderRows(RenderRowSet):
    '''histopi's render rows: style, count, order, labels, the distribution strip, legend
    and colour scale.

    The boxplot styles tie the rows together, and each tie is a rule histop applies:
      - a boxplot needs a numeric count field (Histop.numericCountField()), so the style
        row is live only while the count row holds the template's numeric field, and the
        count row is greyed out while a boxplot is chosen;
      - a boxplot draws no legend, no distribution strip and no colour, so those rows
        are greyed out while one is chosen.
    Order is a group: it sets order= and descending= together, because the values'
    own order ascends and a count descends.'''

    accepts_count_fields = True

    def __init__(self, template: Any, count_fields: list | None = None) -> None:
        super().__init__()
        self.t = template
        _p_    = template.p2s

        def _boxplot_(s: dict[str, str]) -> bool:
            return s.get('style') in _BOXPLOTS_

        def _not_boxplot_(s: dict[str, str]) -> bool:
            return not _boxplot_(s)

        _count_ = count_row(template, also=_not_boxplot_, fields=count_fields)

        self._orders_ = {'largest first':  (_p_.ROW_COUNTp, True),
                         'smallest first': (_p_.ROW_COUNTp, False),
                         'sorted':         (_p_.LABELp,     False),
                         'reverse':        (_p_.LABELp,     True)}
        _order_init_ = self.__orderAsBuilt__()

        self.rows = [
            bar_style_row(template, also=_count_numeric_gate_(template, _count_)),
            _count_,
            RenderRow(RENDER_ROW_KEYS['order'], 'order', 'order',
                      _with_as_built_(_HISTOP_ORDER_ITEMS_, _order_init_), _order_init_, _always_),
            on_off_row(template, 'draw_labels', 'labels'),
            on_off_row(template, 'distribution', 'distribution',
                       also=lambda s: _not_boxplot_(s) and template.distributionStripFits()),
            legend_row(template, also=_not_boxplot_),
            color_scale_row(template, also=_not_boxplot_),
        ]

    def __orderAsBuilt__(self) -> str:
        '''The order row's value for the template's (order=, descending=): one of the four,
        else the field it sorts by -- 'by bytes', 'by bytes, ascending'.'''
        for _label_, (_order_, _desc_) in self._orders_.items():
            if self.t.order is _order_ and bool(self.t.descending) == _desc_: return _label_
        return f'by {_describe_spec_(self.t.order)}' + ('' if self.t.descending else ', ascending')

    # Only what differs from the template: moving from 'largest first' to 'smallest
    # first' changes descending= and leaves order= alone.
    def _groupOverrides_(self, settings: dict[str, str], others: dict[str, Any]) -> dict[str, Any]:
        if not self._changed_(settings, 'order'): return {}
        _order_, _desc_ = self._orders_[settings['order']]
        _out_: dict[str, Any] = {}
        if _order_ is not self.t.order: _out_['order']      = _order_
        if _desc_ != self.t.descending: _out_['descending'] = _desc_
        return _out_


# ─────────────────────────────────────────────────────────────────────────────
# piep
# ─────────────────────────────────────────────────────────────────────────────

class PiepRenderRows(RenderRowSet):
    '''piepi's render rows: style, count, slice order, labels, legend and colour scale.

    Each gate is a rule piep applies (Piep.styleDrawsLabels(), Piep.colorMode()): a
    waffle draws no slice labels; only a categorical or spectrum colour draws a legend;
    and only a spectrum has a magnitude / stretched scale -- a magnitude enum on a text
    field falls back to categorical, so swapping it would change nothing.'''

    accepts_count_fields = True

    def __init__(self, template: Any, count_fields: list | None = None) -> None:
        super().__init__()
        _p_ = template.p2s
        _styles_ = [('p', 'pie', _p_.PIEp), ('d', 'donut', _p_.DONUTp), ('w', 'waffle', _p_.WAFFLEp)]
        _style_of_ = {_l_: _v_ for _, _l_, _v_ in _styles_}
        _style_init_ = next((_l_ for _, _l_, _v_ in _styles_ if template.style is _v_), str(template.style))
        _mode_ = template.colorMode()

        def _labels_drawn_(s: dict[str, str]) -> bool:
            return template.styleDrawsLabels(_style_of_.get(s.get('style', _style_init_), template.style))

        self.rows = [
            _param_row_('style', 'style', 'style', _styles_, _style_init_),
            count_row(template, fields=count_fields),
            _param_row_('order', 'slice order', 'descending',
                        [('l', 'largest first', True), ('s', 'smallest first', False)],
                        'largest first' if template.descending else 'smallest first'),
            on_off_row(template, 'draw_labels', 'labels', also=_labels_drawn_),
            legend_row(template, also=lambda s: _mode_ in ('cset', 'spectrum')),
            color_scale_row(template, also=lambda s: _mode_ == 'spectrum'),
        ]


# ─────────────────────────────────────────────────────────────────────────────
# chordp
# ─────────────────────────────────────────────────────────────────────────────

#: linkpi's size names and keys, less 'none' for nodes (chordp has no way to hide them).
_CHORDP_LINK_SIZES_ = [('o', 'none', None), ('n', 'nil', 'nil'), ('s', 'small', 'small'),
                       ('m', 'medium', 'medium'), ('g', 'large', 'large'), ('v', 'vary', 'vary')]


class ChordpRenderRows(RenderRowSet):
    '''chordpi's render rows: link shape and bundle strength, node and link size, node and
    link opacity, labels and label style, count, legend, and colour scale -- a second one
    for node_color when that holds a magnitude / stretched mode.

    What chordp draws decides the lists and gates:
      - node size is only ever 'vary' or not: every other name draws the same arcs, so
        the row offers fixed / vary rather than five values of which four look alike;
      - count= reaches the picture only through a 'vary' size, because a re-render keeps
        the template's node order (see ChP._order_pinned_), so the count row is live only
        while a size row is on 'vary';
      - bundle strength matters only to bundled links, label style only to drawn labels;
      - the legend describes the link colour when it is data-driven, else the node colour
        (ChP.colorIsLegendable()).
    Order is left out: chordp orders only by its leaf walk or an explicit list.'''

    accepts_count_fields = True

    def __init__(self, template: Any, count_fields: list | None = None) -> None:
        super().__init__()
        _shapes_ = [('l', 'line', 'line'), ('c', 'curve', 'curve'), ('b', 'bundled', 'bundled')]
        _shape_init_ = str(template.link_shape)
        _strengths_ = [('5', '0.5', 0.5), ('8', '0.85', 0.85), ('1', '1.0', 1.0)]
        _strength_init_ = next((_l_ for _, _l_, _v_ in _strengths_ if template.bundle_strength == _v_),
                               str(template.bundle_strength))
        _link_size_init_ = next((_l_ for _, _l_, _v_ in _CHORDP_LINK_SIZES_ if template.link_size == _v_),
                                str(template.link_size))

        def _vary_(s: dict[str, str]) -> bool:
            return s.get('node_size') == 'vary' or s.get('link_size') == 'vary'

        self.rows = [
            _param_row_('link_shape', 'link shape', 'link_shape', _shapes_, _shape_init_),
            _param_row_('bundle_strength', 'bundle strength', 'bundle_strength', _strengths_, _strength_init_,
                        lambda s: s.get('link_shape') == 'bundled'),
            _param_row_('node_size', 'node size', 'node_size',
                        [('f', 'fixed', 'medium'), ('v', 'vary', 'vary')],
                        'vary' if template.node_size == 'vary' else 'fixed'),
            _param_row_('link_size', 'link size', 'link_size', _CHORDP_LINK_SIZES_, _link_size_init_),
            opacity_row(template, 'node_opacity', 'node opacity'),
            opacity_row(template, 'link_opacity', 'link opacity'),
            on_off_row(template, 'draw_labels', 'labels'),
            _param_row_('label_style', 'label style', 'label_style',
                        [('r', 'radial', 'radial'), ('c', 'circular', 'circular')], str(template.label_style),
                        lambda s: s.get('draw_labels') == 'on'),
            count_row(template, also=_vary_, fields=count_fields),
            legend_row(template, legendable=template.colorIsLegendable()),
            color_scale_row(template),
        ]
        if _color_scale_of_(template.node_color) is not None:
            self.rows.append(color_scale_row(template, param='node_color'))


# ─────────────────────────────────────────────────────────────────────────────
# timep
# ─────────────────────────────────────────────────────────────────────────────

#: A fixed key per time level, so a level keeps its key whichever levels a view offers:
#: lower case for the linear levels and upper case for the periodic ones, a letter of the
#: name where one is free.
_TIME_LEVEL_KEYS_ = {
    'LT_Yp': 'y', 'LT_Y_Qp': 'q', 'LT_Y_mp': 'm', 'LT_Y_m_dp': 'd', 'LT_Y_m_d_4Hp': '4',
    'LT_Y_m_d_Hp': 'h', 'LT_Y_m_d_H_15Mp': 'f', 'LT_Y_m_d_H_Mp': 'n', 'LT_Y_m_d_H_M_15Sp': 'e',
    'LT_Y_m_d_H_M_Sp': 's',
    'PT_Qp': 'Q', 'PT_mp': 'M', 'PT_m_dp': 'D', 'PT_m_d_Hp': 'E', 'PT_DoYp': 'Y', 'PT_DoWp': 'W',
    'PT_DoW_Hp': 'X', 'PT_DoW_H_Mp': 'Z', 'PT_dp': 'O', 'PT_d_Hp': 'P', 'PT_d_H_Mp': 'R',
    'PT_Hp': 'H', 'PT_H_Mp': 'I', 'PT_H_M_Sp': 'T', 'PT_Mp': 'N', 'PT_M_Sp': 'U', 'PT_Sp': 'S',
}


def time_level_label(template: Any, level: Any) -> str:
    '''A time level as the granularity row offers it: timep's name for it, marked as a
    point on the timeline or a position in a cycle.  'monthly' and 'month' were the only
    difference between the two month levels, and the list mixes both kinds (user feedback
    2026-09-27): monthly (timeline) runs Jan 2024, Feb 2024, ...; month (cycle) folds
    every January into one bar.'''
    _kind_ = 'cycle' if isinstance(level, template.p2s.TimePeriodicTypeP) else 'timeline'
    return f'{template.timeLevelName(level)} ({_kind_})'


class TimepRenderRows(RenderRowSet):
    '''timepi's render rows: time granularity, style, count, legend and colour scale.

    Granularity offers 'auto' -- the template's own resolution, frame by frame -- and the
    levels Timep.timeLevels() lists: the ones whose spine fits the plot on the widest frame,
    so an explicit level is safe at every stack level.  Each is marked (timeline) or
    (cycle) -- see time_level_label().  An explicit level applies to every
    level of the stack; 'auto' keeps resolving per frame.  On a periodic level the time-
    expand keys (e / u) do nothing: a cycle has no timeframe to widen.

    Style, count, legend and colour scale are histopi's, gated by the same boxplot rules
    through timep's own numericCountField(): a boxplot needs a numeric count field and
    draws no legend and no colour.'''

    accepts_count_fields = True

    def __init__(self, template: Any, count_fields: list | None = None) -> None:
        super().__init__()
        _p_ = template.p2s

        def _not_boxplot_(s: dict[str, str]) -> bool:
            return s.get('style') not in _BOXPLOTS_

        _count_ = count_row(template, also=_not_boxplot_, fields=count_fields)

        _col_    = template._time_field_
        _levels_ = template.timeLevels()
        if   isinstance(template.time, _p_.TField): _built_ = template.time.transform
        elif isinstance(template.time, tuple):      _built_ = template.time[1]
        else:                                       _built_ = None
        _choices_ = [('u', 'auto', _col_)] + [(_TIME_LEVEL_KEYS_[_e_.name], time_level_label(template, _e_),
                                                _p_.tField(_col_, _e_)) for _e_ in _levels_]
        self.rows = [
            _param_row_('granularity', 'granularity', 'time', _choices_,
                        'auto' if _built_ is None else time_level_label(template, _built_),
                        lambda s: len(_levels_) > 0),
            bar_style_row(template, also=_count_numeric_gate_(template, _count_)),
            _count_,
            legend_row(template, also=_not_boxplot_),
            color_scale_row(template, also=_not_boxplot_),
        ]
