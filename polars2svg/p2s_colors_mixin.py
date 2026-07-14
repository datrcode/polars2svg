import polars as pl

__name__ = 'p2s_colors_mixin'

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
#
def isHexColor(value):
    if not isinstance(value, str) or len(value) < 4 or value[0] != '#':
        return False
    _body_ = value[1:]
    if len(_body_) not in (3, 6, 8):
        return False
    return all(_c_ in _HEX_DIGITS_ for _c_ in _body_)

class P2SColorsMixin:
    def __init__(self):
        pass

    # Expose the canonical detector as a callable for programmatic use / tests.
    isHexColor = staticmethod(isHexColor)

    #
    # HexColorStringMeta - metaclass for HexColorString
    # - lets isinstance(x, HexColorString) be the one hex-color check used everywhere;
    #   it delegates to the module-level isHexColor() so the rule lives in one place.
    #
    class HexColorStringMeta(type):
        def __instancecheck__(cls, instance):
            return isHexColor(instance)

    class HexColorString(metaclass=HexColorStringMeta):
        pass

    #
    # __p2s_colors_mixin_init__() - initialization via the mixin methodology
    #
    def __p2s_colors_mixin_init__(self):
        # to_color_lu holds only base hash-derived colors (never override results),
        # so it can persist across re-inits without going stale when overrides change.
        if not hasattr(self, 'to_color_lu'):
            self.to_color_lu = {}
        if not hasattr(self, 'color_overrides_lu'):
            self.color_overrides_lu = {}
        self.color_type_lu = {
            ('background',    'default'):   '#ffffff',
            ('data',          'default'):   "#3939ff",
            ('axis',          'default'):   "#a0a0a0",
            ('axis',          'label'):     "#404040",
            ('axis',          'min'):       "#0015D2",
            ('axis',          'max'):       "#A10000",
            ('axis',          'inner'):     "#a0a0a0",
            ('axis',          'origin'):    "#404040",
            ('error',         'default'):   "#ff0000",
            ('label',         'defaultfg'): '#000000',
            ('label',         'inner'):     '#a0a0a0',
            ('distributions', 'default'):   '#000000',
            ('distributions', 'fill'):      '#f0f0f0',
            ('indicator',     'more_rows'): '#cc3333',
            ('selection',     'default'):   '#ff0000',
            ('multiset',      'str'):       '#7f8367', # derived from polarsOperation behavior / not used 
            ('multiset',      'int'):       '#19d084', # derived from polarsOperation behavior / not used
            ('multiset',      'float'):     '#e3e294', # derived from polarsOperation behavior / not used
        }

        #
        # Multi-set Color Derivations
        #
        # _df_ = pl.DataFrame({
        #     'val':  ['a','a','b','b','c','c'],
        #     'cat':  ['x','y','x','y','w','z'],
        #     'cat_n':[ 1,  2,  1,  2,  3,  4],
        #     'cat_f':[2.1,3.1,2.1,3.1,4.1,6.2],
        # })
        # _params_ = {'df':_df_, 'x':'val', 'y':'val', 'dot_size':10.0, 'draw_context':False, 'insets':(16,16), 'wxh':(96,96)}
        # p2s.xyp(**_params_, color='cat')                # "#7f8367"
        # p2s.xyp(**_params_, color=('cat',   p2s.CSETp)) # "#7f8367"
        # p2s.xyp(**_params_, color=('cat_n', p2s.CSETp)) # "#19d084"
        # p2s.xyp(**_params_, color=('cat_f', p2s.CSETp)) # "#e3e294"

        #
        # Color Spectrum: the 10-class "Spectral" diverging scheme from ColorBrewer
        # (colorbrewer2.org). Color specifications and designs © 2002 Cynthia Brewer,
        # Mark Harrower, and The Pennsylvania State University; used under the
        # Apache-style ColorBrewer license (attribution reproduced in NOTICE).
        #
        # M. Harrower and C. A. Brewer, "ColorBrewer.org: An Online Tool for
        # Selecting Colour Schemes for Maps," The Cartographic Journal, vol. 40,
        # no. 1, pp. 27-37, 2003, doi: 10.1179/000870403235002042.
        #
        # - for the colorSpectrumPolarsOperations() to work, there needs to be at least three colors
        self.spectrum_palette = ['#9e0142','#d53e4f','#f46d43','#fdae61','#fee08b',
                                 '#e6f598','#abdda4','#66c2a5','#3288bd','#5e4fa2']

    #
    # color() - for any object, return a unique color
    # - overrides are resolved live (matching colorizeColumnPolarsOperations, which
    #   matches on the string cast of the value); only base colors are cached
    #
    def color(self, _obj_):
        if self.color_overrides_lu:
            _override_ = self.color_overrides_lu.get(_obj_ if isinstance(_obj_, str) else str(_obj_))
            if _override_ is not None: return _override_
        if _obj_ not in self.to_color_lu.keys():
            if isinstance(_obj_, self.HexColorString):
                self.to_color_lu[_obj_] = _obj_
            else:
                _df_ = pl.DataFrame({'to_color':[_obj_]})
                _df_ = _df_.with_columns(self.colorizeColumnPolarsOperations('to_color', apply_overrides=False).alias('__hexcolor__'))
                self.to_color_lu[_obj_] = _df_['__hexcolor__'][0]
        return self.to_color_lu[_obj_]

    #
    # colors() - batch version of color() ... resolves all values in a single Polars pass
    # - returns {value: hex_color} for every distinct value in _objs_
    # - one collect for all cache misses instead of one 1-row collect per miss
    #
    def colors(self, _objs_):
        _result_, _to_compute_ = {}, []
        for _obj_ in _objs_:
            if _obj_ in _result_: continue
            if self.color_overrides_lu:
                _override_ = self.color_overrides_lu.get(_obj_ if isinstance(_obj_, str) else str(_obj_))
                if _override_ is not None:
                    _result_[_obj_] = _override_
                    continue
            if   _obj_ in self.to_color_lu:
                _result_[_obj_] = self.to_color_lu[_obj_]
            elif isinstance(_obj_, self.HexColorString):
                self.to_color_lu[_obj_] = _obj_
                _result_[_obj_]         = _obj_
            else:
                _to_compute_.append(_obj_)
        if _to_compute_:
            _df_ = pl.DataFrame({'to_color': _to_compute_})
            _df_ = _df_.with_columns(self.colorizeColumnPolarsOperations('to_color', apply_overrides=False).alias('__hexcolor__'))
            for _obj_, _hex_ in zip(_to_compute_, _df_['__hexcolor__'].to_list()):
                self.to_color_lu[_obj_] = _hex_
                _result_[_obj_]         = _hex_
        return _result_

    #
    # colorTyped() - return a typed color (usually a chart element, not a data element)
    #
    def colorTyped(self, _type_, _subtype_): return self.color_type_lu[(_type_, _subtype_)]

    #
    # setColorOverrides() - register explicit hex colors for specific cell values
    # - overrides is a dict mapping cell value strings to hex color strings (#RRGGBB or #RGB)
    # - calling multiple times merges; later calls overwrite earlier ones for the same key
    #
    def setColorOverrides(self, overrides):
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
    def removeColorOverrides(self, keys):
        if isinstance(keys, str):
            keys = (keys,)
        for _k_ in keys:
            self.color_overrides_lu.pop(_k_, None)

    #
    # grayscaleSpectrumPolarsOperations() - abridged grayscale ramp for distribution strips
    # - normalized_input : column of float64 values in [0.0, 1.0]
    # - {red|green|blue}_output : destination column names (all set to the same gray value)
    # - light_gray : RGB value for the minimum (0.0); defaults to 0.8 (#cccccc)
    # Mapping: 0.0 → light_gray (barely perceptible on white), 1.0 → 0.0 (black)
    #
    def grayscaleSpectrumPolarsOperations(self, normalized_input, red_output, green_output, blue_output, light_gray=0.8):
        _gray_ = (pl.lit(float(light_gray)) * (1.0 - pl.col(normalized_input))).clip(0.0, 1.0)
        return [
            _gray_.alias(red_output),
            _gray_.alias(green_output),
            _gray_.alias(blue_output),
        ]

    #
    # colorSpectrumTuples() - return a list of rgb tuples for the color spectrum
    #
    def colorSpectrumTuples(self):
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
    def colorSpectrumPolarsOperations(self, normalized_input, red_output, green_output, blue_output):
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
    def colorSpectrumPolarsOperations_LIMITED_TO_EXACTLY_FIVE(self, normalized_input, red_output, green_output, blue_output):
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
    def hexColorFromRGBTriplesPolarsOperations(self, red_column, green_column, blue_column):
        hex_digits = "0123456789abcdef"
        return  pl.concat_str([
                pl.lit("#"),
                pl.lit(hex_digits).str.slice((pl.col(red_column)   * 255).cast(pl.UInt8) // 16, 1),
                pl.lit(hex_digits).str.slice((pl.col(red_column)   * 255).cast(pl.UInt8) %  16, 1),
                pl.lit(hex_digits).str.slice((pl.col(green_column) * 255).cast(pl.UInt8) // 16, 1),
                pl.lit(hex_digits).str.slice((pl.col(green_column) * 255).cast(pl.UInt8) %  16, 1),
                pl.lit(hex_digits).str.slice((pl.col(blue_column)  * 255).cast(pl.UInt8) // 16, 1),
                pl.lit(hex_digits).str.slice((pl.col(blue_column)  * 255).cast(pl.UInt8) %  16, 1),
        ])

    #
    # rgbFromHexPolarsOperations() - unpack a '#rrggbb' hex color string column into
    # float [0,1] r/g/b columns (the inverse of hexColorFromRGBTriplesPolarsOperations;
    # hex is the quantizer in both directions, so the round-trip is exact)
    #
    def rgbFromHexPolarsOperations(self, hex_column, red_output, green_output, blue_output):
        _c_ = pl.col(hex_column)
        return [
            (_c_.str.slice(1, 2).str.to_integer(base=16) / 255.0).alias(red_output),
            (_c_.str.slice(3, 2).str.to_integer(base=16) / 255.0).alias(green_output),
            (_c_.str.slice(5, 2).str.to_integer(base=16) / 255.0).alias(blue_output),
        ]

    #
    # colorizeColumnPolarsOperations() - colorize a column (of any type) into a hex color string column.
    # - apply_overrides=False skips the color_overrides_lu when/then chain (used by color()/colors(),
    #   which resolve overrides in Python and cache only the base hash colors)
    #
    def colorizeColumnPolarsOperations(self, input, apply_overrides=True):
            _color_     = pl.col(input)
            _hc_        = _color_.hash()
            _hsv_h_     = (((_hc_ //  2**16) & 0x00ffff)/65535.0)
            _hsv_s_     = (0.1 + 0.8 * ((_hc_ //  2** 8) & 0x0000ff)/  255.0)
            _hsv_v_     = (0.5 + 0.4 * ((_hc_          ) & 0x0000ff)/  255.0)
            _conv_i_    = ((_hsv_h_*6).floor().cast(pl.Int8))
            _conv_f_    = ((_hsv_h_*6) - (_hsv_h_*6).floor())
            _conv_p_    = (_hsv_v_ * (1.0 - _hsv_s_))
            _conv_q_    = (_hsv_v_ * (1.0 - _conv_f_ * _hsv_s_))  
            _conv_t_    = (_hsv_v_ * (1.0 - (1 - _conv_f_) * _hsv_s_))
            _r_ = pl.when((_conv_i_ == 0) | (_conv_i_ == 5)).then(_hsv_v_)  \
                    .when((_conv_i_ == 1))                  .then(_conv_q_) \
                    .when((_conv_i_ == 2) | (_conv_i_ == 3)).then(_conv_p_) \
                    .otherwise(                                   _conv_t_)
            _g_ = pl.when((_conv_i_ == 0))                  .then(_conv_t_) \
                    .when((_conv_i_ == 1) | (_conv_i_ == 2)).then(_hsv_v_)  \
                    .when((_conv_i_ == 3))                  .then(_conv_q_) \
                    .otherwise(                                   _conv_p_)
            _b_ = pl.when((_conv_i_ == 0) | (_conv_i_ == 1)).then(_conv_p_) \
                    .when((_conv_i_ == 2))                  .then(_conv_t_) \
                    .when((_conv_i_ == 3) | (_conv_i_ == 4)).then(_hsv_v_)  \
                    .otherwise(                                   _conv_q_)
            hex_digits = "0123456789abcdef"
            _base_ = pl.concat_str([pl.lit("#"),
                                    pl.lit(hex_digits).str.slice((_r_ * 255).cast(pl.UInt8) // 16, 1),
                                    pl.lit(hex_digits).str.slice((_r_ * 255).cast(pl.UInt8) %  16, 1),
                                    pl.lit(hex_digits).str.slice((_g_ * 255).cast(pl.UInt8) // 16, 1),
                                    pl.lit(hex_digits).str.slice((_g_ * 255).cast(pl.UInt8) %  16, 1),
                                    pl.lit(hex_digits).str.slice((_b_ * 255).cast(pl.UInt8) // 16, 1),
                                    pl.lit(hex_digits).str.slice((_b_ * 255).cast(pl.UInt8) %  16, 1)])
            if not apply_overrides or not self.color_overrides_lu:
                return _base_
            _expr_ = _base_
            for _val_, _hex_ in self.color_overrides_lu.items():
                _expr_ = pl.when(pl.col(input).cast(pl.String) == _val_).then(pl.lit(_hex_)).otherwise(_expr_)
            return _expr_

