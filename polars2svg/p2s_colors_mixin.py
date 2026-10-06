import datetime
import decimal
import hashlib
import json
from typing import Any, TypeGuard
import numpy as np
import polars as pl

from .p2s_palettes import resolvePalette, paletteNames

_HEX_DIGITS_ = frozenset('0123456789abcdefABCDEF')

#
# isHexColor() - the single, canonical hex-color detector for the whole framework.
# - Recognizes exactly the three forms the renderer (p2s_displaylist.hexToRGBA) can
#   paint: #RGB, #RRGGBB, and #RRGGBBAA (3, 6, or 8 hex digits). #RRGGBBAA carries an
#   alpha channel and is *supported* — modern SVG/CSS renderers honour it and hexToRGBA
#   parses it. Any other '#'-prefixed string (a bare '#', #RGBA/4-digit, #ggg, wrong
#   length) is deliberately NOT a color, so it stays available as a DataFrame column
#   name rather than being silently rendered as a broken / gray fill.
# - Before this was unified, xyp keyed on the HexColorString metaclass (accepted only
#   #RGB / #RRGGBB) while linkp/chordp/spreadlinesp used ad-hoc str.startswith('#')
#   (accepted anything) and the background-shape code required an exact len == 7. The
#   result: '#ff000080' was a field name to xyp but a color to linkp. Every component
#   now routes through this one function via isinstance(x, HexColorString).
# - It is typed `TypeGuard[str]` because HexColorString is a *virtual* class: its
#   metaclass __instancecheck__ delegates here, so isinstance(x, HexColorString) is a
#   predicate over strings rather than a base-class test, and a checker seeing only the
#   isinstance narrows the value to HexColorString -- which is then not a `str` and
#   cannot be returned where one is declared. Call isHexColor() directly at any site
#   whose result has to satisfy a `str` annotation.
#
def isHexColor(value: Any) -> TypeGuard[str]:
    if not isinstance(value, str) or len(value) < 4 or value[0] != '#':
        return False
    _body_ = value[1:]
    if len(_body_) not in (3, 6, 8):
        return False
    return all(_c_ in _HEX_DIGITS_ for _c_ in _body_)

#
# colorHashKey() - the bytes a value's hash colour is derived from (PLANNING.md
# C-polars2-color-hash).
# - A type tag, a NUL, then a canonical rendering of the *Python* value.  Never a
#   polars cast to String: polars' formatting is free to change between versions, and
#   version-independence is the whole point.  pl.Expr.hash() was replaced for exactly
#   that reason -- polars documents it as unstable across releases.
# - The tag keeps '10' and 10 apart, as they always were.  Within a tag the dtype does
#   not matter: every integer width is 'int' (so histop and linkp give one integer one
#   colour), and String, Categorical and Enum are all 'str'.
# - Floats go through v + 0.0, which turns -0.0 into 0.0.  Polars' unique() treats the
#   two as one value and keeps whichever it met first, so without it a colour would
#   depend on row order.  NaN renders as 'nan' whatever its sign.
# - Lists and structs are compact JSON over the same canonical forms (_jsonable_).
# - Anything else falls back to repr(), which is only as stable as that type's repr.
#
def colorHashKey(value: Any) -> bytes:
    if value is None:                                   return b'null\x00'
    if isinstance(value, str):                          return b'str\x00'       + value.encode('utf-8')
    if isinstance(value, bool):                         return b'bool\x00'      + (b'1' if value else b'0')
    if isinstance(value, int):                          return b'int\x00'       + str(value).encode('ascii')
    if isinstance(value, float):                        return b'float\x00'     + repr(value + 0.0).encode('ascii')
    if isinstance(value, datetime.datetime):            return b'datetime\x00'  + value.isoformat().encode('ascii')
    if isinstance(value, datetime.date):                return b'date\x00'      + value.isoformat().encode('ascii')
    if isinstance(value, datetime.time):                return b'time\x00'      + value.isoformat().encode('ascii')
    if isinstance(value, datetime.timedelta):           return b'timedelta\x00' + f'{value.days},{value.seconds},{value.microseconds}'.encode('ascii')
    if isinstance(value, decimal.Decimal):              return b'decimal\x00'   + str(value).encode('ascii')
    if isinstance(value, bytes | bytearray):            return b'bytes\x00'     + bytes(value)
    if isinstance(value, list | tuple | dict):          return b'json\x00'      + json.dumps(_jsonable_(value), ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    return b'repr\x00' + repr(value).encode('utf-8', errors='backslashreplace')

#
# _jsonable_() - a nested value in the canonical forms colorHashKey() uses, ready for json
# - a scalar JSON cannot hold keeps its colorHashKey() rendering, tag included, so a
#   date inside a list cannot collide with a string inside a list
#
def _jsonable_(value: Any) -> Any:
    if value is None or isinstance(value, str | bool | int): return value
    if isinstance(value, float):                              return value + 0.0
    if isinstance(value, list | tuple):                       return [_jsonable_(_v_) for _v_ in value]
    if isinstance(value, dict):                               return {str(_k_): _jsonable_(_v_) for _k_, _v_ in value.items()}
    return colorHashKey(value).decode('utf-8', errors='backslashreplace')

#
# colorHashes() - one 32-bit blake2b hash per element of s, as a UInt32 Series of the
# same length and name
# - Each element costs one hashlib call, so callers hand this distinct values only.
# - Nulls hash like any other value (to colorHashKey(None)'s digest); the result has
#   no nulls.
# - 32 bits because the hash colour reads exactly that many (H, S and V).
# - String-like columns skip colorHashKey()'s dispatch; tests hold the two paths equal.
#
def colorHashes(s: pl.Series) -> pl.Series:
    _blake2b_ = hashlib.blake2b
    if s.dtype == pl.String or isinstance(s.dtype, pl.Categorical | pl.Enum):
        _keys_ = [b'null\x00' if _v_ is None else b'str\x00' + _v_ for _v_ in s.cast(pl.String).cast(pl.Binary).to_list()]
    else:
        _keys_ = [colorHashKey(_v_) for _v_ in s.to_list()]
    _digests_ = b''.join(_blake2b_(_k_, digest_size=4).digest() for _k_ in _keys_)
    return pl.Series(s.name, np.frombuffer(_digests_, dtype='<u4'), dtype=pl.UInt32)

#
# _unitRGBToHexExpr_() - '#rrggbb' from three [0, 1] channel expressions
# - Rounds to the nearest of the 256 levels.  It used to truncate, so a channel computed
#   as 93.999... drew as 0x5d, and float noise decided which (PLANNING.md
#   C-hex-truncation).  floor(x + 0.5) rather than Expr.round(), whose tie mode is
#   polars' to choose.
#
def _unitRGBToHexExpr_(r: pl.Expr, g: pl.Expr, b: pl.Expr) -> pl.Expr:
    _hex_digits_ = '0123456789abcdef'
    _parts_: list[pl.Expr] = [pl.lit('#')]
    for _c_ in (r, g, b):
        _byte_ = (_c_ * 255 + 0.5).floor().clip(0, 255).cast(pl.UInt8)
        _parts_ += [pl.lit(_hex_digits_).str.slice(_byte_ // 16, 1), pl.lit(_hex_digits_).str.slice(_byte_ % 16, 1)]
    return pl.concat_str(_parts_)

#
# _hashToHexExpr_() - a hash colour from a 32-bit hash column
# - H from bits 16-31, S (0.1-0.9) from bits 8-15, V (0.5-0.9) from bits 0-7; the band
#   is fixed, so hash colours do not depend on the palette
# - hc must be a materialised column: every channel reads it many times over
#
def _hashToHexExpr_(hc: pl.Expr) -> pl.Expr:
    _hsv_h_     = (((hc //  2**16) & 0x00ffff)/65535.0)
    _hsv_s_     = (0.1 + 0.8 * ((hc //  2** 8) & 0x0000ff)/  255.0)
    _hsv_v_     = (0.5 + 0.4 * ((hc          ) & 0x0000ff)/  255.0)
    _conv_i_    = ((_hsv_h_*6).floor().cast(pl.Int8))
    _conv_f_    = ((_hsv_h_*6) - (_hsv_h_*6).floor())
    _conv_p_    = (_hsv_v_ * (1.0 - _hsv_s_))
    _conv_q_    = (_hsv_v_ * (1.0 - _conv_f_ * _hsv_s_))
    _conv_t_    = (_hsv_v_ * (1.0 - (1 - _conv_f_) * _hsv_s_))
    _r_ = pl.when((_conv_i_ == 0) | (_conv_i_ == 5)).then(_hsv_v_)  \
            .when(_conv_i_ == 1)                  .then(_conv_q_) \
            .when((_conv_i_ == 2) | (_conv_i_ == 3)).then(_conv_p_) \
            .otherwise(                                   _conv_t_)
    _g_ = pl.when(_conv_i_ == 0)                  .then(_conv_t_) \
            .when((_conv_i_ == 1) | (_conv_i_ == 2)).then(_hsv_v_)  \
            .when(_conv_i_ == 3)                  .then(_conv_q_) \
            .otherwise(                                   _conv_p_)
    _b_ = pl.when((_conv_i_ == 0) | (_conv_i_ == 1)).then(_conv_p_) \
            .when(_conv_i_ == 2)                  .then(_conv_t_) \
            .when((_conv_i_ == 3) | (_conv_i_ == 4)).then(_hsv_v_)  \
            .otherwise(                                   _conv_q_)
    return _unitRGBToHexExpr_(_r_, _g_, _b_)

class P2SColorsMixin:
    def __init__(self) -> None:
        pass

    # Expose the canonical detector as a callable for programmatic use / tests.
    isHexColor = staticmethod(isHexColor)

    #
    # HexColorStringMeta - metaclass for HexColorString
    # - lets isinstance(x, HexColorString) be the one hex-color check used everywhere;
    #   it delegates to the module-level isHexColor() so the rule lives in one place.
    #
    class HexColorStringMeta(type):
        def __instancecheck__(cls, instance: Any) -> bool:
            return bool(isHexColor(instance))

    class HexColorString(metaclass=HexColorStringMeta):
        pass

    #
    # __p2s_colors_mixin_init__() - initialization via the mixin methodology
    #
    def __p2s_colors_mixin_init__(self, palette: str = 'light') -> None:
        # Both are per-instance: to_color_lu memoizes base hash-derived colors (never
        # override results), color_overrides_lu holds what setColorOverrides() was given.
        # Two Polars2SVG instances share neither, so an override set on one is invisible
        # to the other -- see the Configuration section of the class docstring.
        # to_color_lu is keyed by colorHashKey(value), not by the value: Python treats
        # -1 == -1.0 and 1 == True as one dict key, but the colorizer gives each its own
        # colour, so a value-keyed memo let whichever was asked for first fix the colour
        # of the rest (PLANNING.md C-color-memo-numeric-keys).
        self.to_color_lu: dict = {}
        self.color_overrides_lu: dict = {}

        #
        # The palette supplies three pieces of state; the tables themselves live in
        # p2s_palettes.py.  resolvePalette() hands back copies, so mutating
        # color_type_lu on one instance (which tests do) cannot reach another.
        #
        # Multi-set Color Derivations -- the ('multiset', *) slots record the colour a
        # node whose rows carry several values of a CSETp field ends up with: the cset
        # path in p2s_component_color_mixin marks it -1 in the field's dtype and hash-
        # colours that.  So each slot is the hash colour of -1 as a str, an int and a
        # float (tests/test_palettes.py holds them to colorizeColumnPolarsOperations):
        #
        # p2s.color('-1')   # "#a88196"
        # p2s.color(-1)     # "#b7b846"
        # p2s.color(-1.0)   # "#ae3ba6"
        #
        self.__applyPalette__(resolvePalette(palette))

    #
    # __applyPalette__() - install resolved palette state onto the instance
    # - the single place the three palette-derived attributes are assigned, so
    #   __p2s_colors_mixin_init__() and setPalette() cannot drift apart
    #
    def __applyPalette__(self, resolved: dict) -> None:
        self.palette_name:    str               = resolved['name']
        self.color_type_lu:   dict              = resolved['color_type_lu']
        self.spectrum_palette: list             = resolved['spectrum_palette']
        self.grayscale_ramp:  tuple[float, float] = resolved['grayscale_ramp']

    #
    # setPalette() - switch this instance to a named palette
    # - palette   : a name from p2s.paletteNames() ('light' or 'dark')
    # - overrides : optional {(type, subtype): hex} layered onto that palette
    #
    # Deliberately does NOT touch two things:
    #
    # - color_overrides_lu.  A setColorOverrides() entry is an explicit instruction
    #   about one data value; the palette is a default for chart furniture.  The
    #   explicit instruction outranks it, and survives a palette switch.
    # - to_color_lu.  Hash-derived colors do not depend on the palette (the HSV band
    #   in colorizeColumnPolarsOperations is fixed), so the memo stays valid; clearing
    #   it would cost a recompute to arrive at identical values.
    #
    def setPalette(self, palette: str = 'light',
                   overrides: dict | None = None) -> None:
        if overrides:
            for _k_, _v_ in overrides.items():
                if not isinstance(_v_, self.HexColorString):
                    raise ValueError(f'setPalette(): value for {_k_!r} is not a valid hex color: {_v_!r}')
        self.__applyPalette__(resolvePalette(palette, overrides))

    #
    # getPalette() - the name of the palette currently installed
    #
    def getPalette(self) -> str: return self.palette_name

    #
    # interactivePalette() - the colors the browser-side views need, as a plain dict
    #
    # The JS has no colorTyped(): its overlays are literals inside string-concatenated
    # markup.  This is the channel -- one param.Dict per view, read as model.palette.
    # It is deliberately a SMALL, renamed subset rather than the whole table: the JS
    # should not have to know the framework's slot taxonomy, and every key here is a
    # color some overlay actually paints.
    #
    # Why these and not others: black-on-dark overlays are invisible (1.12:1 against
    # #121212), which is a broken interaction rather than an ugly one -- the drag band
    # and the status line simply vanish.  The panel chrome (tooltip box, config panel)
    # paints its own light surface and stays legible, so it is not here yet.
    #
    def interactivePalette(self) -> dict:
        return {
            'ink':       self.colorTyped('label',      'defaultfg'),   # overlay strokes, status line
            'bg':        self.colorTyped('background', 'default'),
            'hint':      self.colorTyped('overlay',    'hint'),        # keyboard-help text
            'selection': self.colorTyped('selection',  'default'),
            # The drag band names the set operation the mouseup is about to perform;
            # _resolve_set_op() in interactive_controller decides which.  Keys match its
            # return values exactly so the two cannot drift.
            'setop': {_op_: self.colorTyped('setop', _op_)
                      for _op_ in ('replace', 'add', 'subtract', 'intersect')},
        }

    # The selectable palette names, exposed for programmatic use / tests.
    paletteNames = staticmethod(paletteNames)

    #
    # color() - for any object, return a unique color
    # - overrides are resolved live (matching colorizeColumnPolarsOperations, which
    #   matches on the string cast of the value); only base colors are cached
    #
    def color(self, _obj_: str | tuple) -> str:
        if self.color_overrides_lu:
            _override_ = self.color_overrides_lu.get(_obj_ if isinstance(_obj_, str) else str(_obj_))
            if _override_ is not None: return _override_
        _key_ = colorHashKey(_obj_)
        if _key_ not in self.to_color_lu:
            if isinstance(_obj_, self.HexColorString):
                self.to_color_lu[_key_] = _obj_
            else:
                _df_ = pl.DataFrame({'to_color':[_obj_]})
                _df_ = _df_.with_columns(self.colorizeColumnPolarsOperations('to_color', apply_overrides=False).alias('__hexcolor__'))
                self.to_color_lu[_key_] = _df_['__hexcolor__'][0]
        return self.to_color_lu[_key_]

    #
    # colors() - batch version of color() ... one Polars pass per Python type among the misses
    # - returns {value: hex_color} for every distinct value in _objs_
    # - the misses are batched by type: one Series cannot hold 1 and 'a', or -1 and -1.0
    #   without casting one to the other's dtype, and a cast value takes the other
    #   type's colour.  A single type is the usual case, and one pass.
    # - the returned dict is keyed by value, so values Python counts as equal (-1 and
    #   -1.0, 1 and True) share one entry, which holds the first one's colour.  color()
    #   and the memo keep them apart.
    #
    def colors(self, _objs_: list | set) -> dict:
        _result_: dict = {}                     # a miss holds its place (None) until computed
        _to_compute_: dict[type, dict] = {}     # type -> {value: memo key}, in input order
        for _obj_ in _objs_:
            if _obj_ in _result_: continue
            if self.color_overrides_lu:
                _override_ = self.color_overrides_lu.get(_obj_ if isinstance(_obj_, str) else str(_obj_))
                if _override_ is not None:
                    _result_[_obj_] = _override_
                    continue
            _key_ = colorHashKey(_obj_)
            if   _key_ in self.to_color_lu:
                _result_[_obj_] = self.to_color_lu[_key_]
            elif isinstance(_obj_, self.HexColorString):
                self.to_color_lu[_key_] = _obj_
                _result_[_obj_]         = _obj_
            else:
                _to_compute_.setdefault(type(_obj_), {})[_obj_] = _key_
                _result_[_obj_] = None
        for _group_ in _to_compute_.values():
            _df_ = pl.DataFrame({'to_color': list(_group_)})
            _df_ = _df_.with_columns(self.colorizeColumnPolarsOperations('to_color', apply_overrides=False).alias('__hexcolor__'))
            for (_obj_, _key_), _hex_ in zip(_group_.items(), _df_['__hexcolor__'].to_list()):
                self.to_color_lu[_key_] = _hex_
                _result_[_obj_]         = _hex_
        return _result_

    #
    # colorTyped() - return a typed color (usually a chart element, not a data element)
    #
    def colorTyped(self, _type_: str, _subtype_: str) -> str: return self.color_type_lu[(_type_, _subtype_)]

    #
    # setColorOverrides() - register explicit hex colors for specific cell values
    # - overrides is a dict mapping cell value strings to hex color strings (#RRGGBB or #RGB)
    # - calling multiple times merges; later calls overwrite earlier ones for the same key
    #
    def setColorOverrides(self, overrides: dict | list) -> None:
        if not isinstance(overrides, dict):
            raise ValueError(f'setColorOverrides(): expected dict, got {type(overrides).__name__}')
        for _k_, _v_ in overrides.items():
            if not isinstance(_v_, self.HexColorString):
                raise ValueError(f'setColorOverrides(): value for {_k_!r} is not a valid hex color: {_v_!r}')
        self.color_overrides_lu.update(overrides)

    #
    # removeColorOverrides() - remove cell values from the override registry
    # - keys may be a single string or any iterable of strings
    # - silently ignores keys that are not present
    #
    def removeColorOverrides(self, keys: list | str) -> None:
        _keys_ = [keys] if isinstance(keys, str) else keys
        for _k_ in _keys_:
            self.color_overrides_lu.pop(_k_, None)

    #
    # grayscaleSpectrumPolarsOperations() - abridged grayscale ramp for distribution strips
    # - normalized_input : column of float64 values in [0.0, 1.0]
    # - {red|green|blue}_output : destination column names (all set to the same gray value)
    # - light_gray : RGB value for the minimum (0.0); defaults to the palette's ramp low end
    #
    # The ramp runs lo at 0.0 to hi at 1.0, both taken from the palette:
    #   light -> (0.8, 0.0): #cccccc, barely perceptible on white, down to black
    #   dark  -> (0.2, 1.0): #333333, barely perceptible on #121212, up to white
    #
    # Dark has to INVERT the ramp rather than merely lighten it.  Keeping the light
    # direction on a dark canvas would make a near-zero bin the brightest mark in the
    # plot, which reads as the opposite of what the data says.
    #
    # light_gray= overrides lo only, keeping its original meaning for existing callers.
    #
    def grayscaleSpectrumPolarsOperations(self, normalized_input: str, red_output: str, green_output: str, blue_output: str, light_gray: float | None = None) -> list:
        _lo_, _hi_ = self.grayscale_ramp
        if light_gray is not None: _lo_ = float(light_gray)
        _n_    = pl.col(normalized_input)
        _gray_ = (pl.lit(float(_lo_)) + (pl.lit(float(_hi_)) - pl.lit(float(_lo_))) * _n_).clip(0.0, 1.0)
        return [
            _gray_.alias(red_output),
            _gray_.alias(green_output),
            _gray_.alias(blue_output),
        ]

    #
    # colorSpectrumTuples() - return a list of rgb tuples for the color spectrum
    #
    def colorSpectrumTuples(self) -> list:
        _palette_   = self.spectrum_palette
        _list_      = []
        for i in range(len(_palette_)):
            _color_ = _palette_[i]
            _list_.append((float(int(_color_[1:3],16))/255.0, float(int(_color_[3:5],16))/255.0, float(int(_color_[5:7],16))/255.0, float(i)/(len(_palette_)-1)))
        return _list_

    #
    # colorSpectrumPolarsOperations() - return a color from the color spectrum
    # - normalized_{input} is a value between 0.0 and 1.0
    # - {red|green|blue}_output is the destination column of the calculated spectrum color
    #
    def colorSpectrumPolarsOperations(self, normalized_input: str, red_output: str, green_output: str, blue_output: str) -> list:
        # Get the tuples & their mixture index (r, g, b, mix) where all values are between 0.0 and 1.0 ... the mix should come in sorted from 0.0 to 1.0
        _tuples_ = self.colorSpectrumTuples()
        _col_    = pl.col(normalized_input)
        _ops_    = []
        for _dst_, i in [(red_output, 0), (green_output, 1), (blue_output, 2)]:
            # first segment: interpolate between tuple 0 and tuple 1
            _expr_ = pl.when(_col_ < _tuples_[1][3]) \
                       .then(     _tuples_[0][i] + (_tuples_[1][i] - _tuples_[0][i])*(_col_ - _tuples_[0][3])/(_tuples_[1][3] - _tuples_[0][3]))
            # middle segments: interpolate between tuple j-1 and tuple j
            for j in range(2, len(_tuples_)-1):
                _expr_ = _expr_.when(_col_ < _tuples_[j][3]) \
                               .then(     _tuples_[j-1][i] + (_tuples_[j][i] - _tuples_[j-1][i])*(_col_ - _tuples_[j-1][3])/(_tuples_[j][3] - _tuples_[j-1][3]))
            # final segment (everything at/above the last threshold): interpolate between the last two tuples
            j = len(_tuples_)-1
            _expr_ = _expr_.otherwise(_tuples_[j-1][i] + (_tuples_[j][i] - _tuples_[j-1][i])*(_col_ - _tuples_[j-1][3])/(_tuples_[j][3] - _tuples_[j-1][3])).alias(_dst_)
            _ops_.append(_expr_)
        return _ops_

    #
    # colorSpectrumPolarsOperations_LIMITED_TO_EXACTLY_FIVE() - original prototype for interpolating a color spectrum
    # - this version is limited to exactly 5 colors
    # - the final version is in colorSpectrumPolarsOperations()
    #
    def colorSpectrumPolarsOperations_LIMITED_TO_EXACTLY_FIVE(self, normalized_input: str, red_output: str, green_output: str, blue_output: str) -> list:
        _tuples_ = self.colorSpectrumTuples()
        return [
            pl.when(pl.col(normalized_input) < _tuples_[1][3])
              .then(     _tuples_[0][0] + (_tuples_[1][0] - _tuples_[0][0])*(pl.col(normalized_input)-_tuples_[0][3])/(_tuples_[1][3] - _tuples_[0][3]))
              .when(pl.col(normalized_input) < _tuples_[2][3])
              .then(     _tuples_[1][0] + (_tuples_[2][0] - _tuples_[1][0])*(pl.col(normalized_input)-_tuples_[1][3])/(_tuples_[2][3] - _tuples_[1][3]))
              .when(pl.col(normalized_input) < _tuples_[3][3])
              .then(     _tuples_[2][0] + (_tuples_[3][0] - _tuples_[2][0])*(pl.col(normalized_input)-_tuples_[2][3])/(_tuples_[3][3] - _tuples_[2][3]))
              .otherwise(_tuples_[3][0] + (_tuples_[4][0] - _tuples_[3][0])*(pl.col(normalized_input)-_tuples_[3][3])/(_tuples_[4][3] - _tuples_[3][3])).alias(red_output),
            pl.when(pl.col(normalized_input) < _tuples_[1][3])
              .then(     _tuples_[0][1] + (_tuples_[1][1] - _tuples_[0][1])*(pl.col(normalized_input)-_tuples_[0][3])/(_tuples_[1][3] - _tuples_[0][3]))
              .when(pl.col(normalized_input) < _tuples_[2][3])
              .then(     _tuples_[1][1] + (_tuples_[2][1] - _tuples_[1][1])*(pl.col(normalized_input)-_tuples_[1][3])/(_tuples_[2][3] - _tuples_[1][3]))
              .when(pl.col(normalized_input) < _tuples_[3][3])
              .then(     _tuples_[2][1] + (_tuples_[3][1] - _tuples_[2][1])*(pl.col(normalized_input)-_tuples_[2][3])/(_tuples_[3][3] - _tuples_[2][3]))
              .otherwise(_tuples_[3][1] + (_tuples_[4][1] - _tuples_[3][1])*(pl.col(normalized_input)-_tuples_[3][3])/(_tuples_[4][3] - _tuples_[3][3])).alias(green_output),
            pl.when(pl.col(normalized_input) < _tuples_[1][3])
              .then(     _tuples_[0][2] + (_tuples_[1][2] - _tuples_[0][2])*(pl.col(normalized_input)-_tuples_[0][3])/(_tuples_[1][3] - _tuples_[0][3]))
              .when(pl.col(normalized_input) < _tuples_[2][3])
              .then(     _tuples_[1][2] + (_tuples_[2][2] - _tuples_[1][2])*(pl.col(normalized_input)-_tuples_[1][3])/(_tuples_[2][3] - _tuples_[1][3]))
              .when(pl.col(normalized_input) < _tuples_[3][3])
              .then(     _tuples_[2][2] + (_tuples_[3][2] - _tuples_[2][2])*(pl.col(normalized_input)-_tuples_[2][3])/(_tuples_[3][3] - _tuples_[2][3]))
              .otherwise(_tuples_[3][2] + (_tuples_[4][2] - _tuples_[3][2])*(pl.col(normalized_input)-_tuples_[3][3])/(_tuples_[4][3] - _tuples_[3][3])).alias(blue_output),
        ]

    #
    # hexColorFromRGBTriplesPolarsOperations() - convert RGB triples to hex color strings
    # - red_column, green_column, blue_column are columns of type float from 0.0 to 1.0
    #
    def hexColorFromRGBTriplesPolarsOperations(self, red_column: str, green_column: str, blue_column: str) -> pl.Expr:
        return _unitRGBToHexExpr_(pl.col(red_column), pl.col(green_column), pl.col(blue_column))

    #
    # rgbFromHexPolarsOperations() - unpack a '#rrggbb' hex color string column into
    # float [0,1] r/g/b columns (the inverse of hexColorFromRGBTriplesPolarsOperations;
    # hex is the quantizer in both directions, so the round-trip is exact)
    #
    def rgbFromHexPolarsOperations(self, hex_column: str, red_output: str, green_output: str, blue_output: str) -> list:
        _c_ = pl.col(hex_column)
        return [
            (_c_.str.slice(1, 2).str.to_integer(base=16) / 255.0).alias(red_output),
            (_c_.str.slice(3, 2).str.to_integer(base=16) / 255.0).alias(green_output),
            (_c_.str.slice(5, 2).str.to_integer(base=16) / 255.0).alias(blue_output),
        ]

    #
    # colorizeColumnPolarsOperations() - colorize a column (of any type) into a hex color string column.
    # - apply_overrides=False skips color_overrides_lu (used by color()/colors(), which
    #   resolve overrides in Python and cache only the base hash colors)
    #
    # One map_batches UDF, so the hash runs once per call: distinct values -> colorHashes()
    # (blake2b, version-stable) -> HSV band -> hex, overrides, then a join back onto the
    # rows.  The colour math used to read pl.Expr.hash() directly, and eager polars has no
    # common-subexpression elimination, so every with_columns evaluated the hash 94 times
    # -- and pl.Expr.hash() recoloured every chart at polars 2.0 (PLANNING.md
    # C-polars2-color-hash).  Overrides still match on the value's polars String cast,
    # so a key means what it always meant.
    #
    def colorizeColumnPolarsOperations(self, input: Any, apply_overrides: bool = True) -> pl.Expr:
        _overrides_ = dict(self.color_overrides_lu) if apply_overrides and self.color_overrides_lu else {}
        def _colorize_(s: pl.Series) -> pl.Series:
            _distinct_ = s.unique()
            _hex_      = colorHashes(_distinct_).to_frame('h').select(_hashToHexExpr_(pl.col('h'))).to_series()
            if _overrides_:
                _hex_  = _distinct_.cast(pl.String).replace_strict(_overrides_, default=_hex_, return_dtype=pl.String)
            _lut_      = pl.DataFrame({'k': _distinct_, 'hex': _hex_})
            return s.to_frame('k').join(_lut_, on='k', how='left', nulls_equal=True, maintain_order='left').get_column('hex')
        return pl.col(input).map_batches(_colorize_, return_dtype=pl.String)

