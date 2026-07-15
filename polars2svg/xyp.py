import polars as pl
import polars.selectors as cs
from   decimal import Decimal
import datetime as dt
from   datetime import timedelta, datetime, date
import typing
import random
import time

import polars2svg
from polars2svg.p2s_displaylist import DisplayList, hexToRGBA
from polars2svg.export import ExportMixin
from polars2svg.exceptions import DataError, Polars2SVGError
import xml.etree.ElementTree as ET  # nosec B405 - background= shape descriptors are trusted caller config, not untrusted data; see SECURITY.md

__name__ = 'xyp'

#
# _CalendarStep - calendar-aware month/year step for temporal grid lines.
#
# Replaces pandas' pd.offsets.DateOffset, which was the sole reason xyp imported
# pandas.  pandas is not a declared dependency (it was only present transitively
# via bokeh), so relying on it made `import polars2svg` fragile.  This helper adds
# calendar months/years to a datetime and supports the `datetime += step` idiom
# through __radd__, so the grid-line stepping loop is unchanged.  Grid anchors are
# always day-1 (see the '%Y-%m-01' / '%Y-01-01' rounding formats used below), so no
# end-of-month day clamping is required.
#
class _CalendarStep:
    def __init__(self, months=0, years=0):
        self.months = months + years * 12
    def __radd__(self, when):
        total = when.month - 1 + self.months
        return when.replace(year=when.year + total // 12, month=total % 12 + 1)

#
# XYp
#
class XYp(ExportMixin):

    _VALID_KWARGS = frozenset({
        'template', 'df', 'x', 'y',
        'color', 'dot_size', 'dot_size_supersample', 'opacity', 'line', 'line_order_by',
        'dot_size_range', 'opacity_range',
        'x_range', 'y_range', 'x_shared_label_range', 'y_shared_label_range',
        'color_magnitude_min', 'color_magnitude_max', 'color_stretched_global_values',
        'dot_size_global_min', 'dot_size_global_max',
        'x_distributions', 'y_distributions',
        'x_order', 'y_order',
        'background', 'background_label_color', 'background_opacity',
        'background_fill', 'background_stroke_w', 'background_stroke',
        'draw_context', 'insets', 'wxh', 'txt_h', 'sm_shared',
        'use_lazy_execution', 'legend',
    })

    #
    # __init__()
    #
    def __init__(self, *args, **kwargs):
        self.t_start             = time.time()
        self.p2s                 = polars2svg.Polars2SVG()
        # use_lazy_execution is a normal parameter, resolved in __parseInput__ via the
        # shared spec (so a global set_defaults('xyp', use_lazy_execution=...) is now
        # honoured — it used to be read here from raw kwargs, bypassing the merge).
        self.timing_metrics      = {}
        self.gatherMetrics(self.__parseInput__, *args, **kwargs)
        self.gatherMetrics(self.__validateInput__)
        if self.df is not None:
            # Flatten Stage
            self.gatherMetrics(self.__flattenDataFrame__)
            # Indexing Stage
            self.gatherMetrics(self.__indexXandY_join__)
            # Geometry Stage
            self.gatherMetrics(self.__constructGeometry__)
            if   isinstance(self.dot_size_orig, int): self.gatherMetrics(self.__toPixelCoordinates_int__)
            else:                                     self.gatherMetrics(self.__toPixelCoordinates_float__)
            # Calculations Stage
            self.df_x_distribution = None
            self.df_y_distribution = None
            if self.x_distributions is not None or \
               self.y_distributions is not None: self.gatherMetrics(self.__distributeElements__)
            # Render Stage
            _randid_ = random.randint(0,2**32)  # nosec B311 - non-cryptographic SVG id scoping, see SECURITY.md
            self.gatherMetrics(self.__renderBackground__)
            self.gatherMetrics(self.__renderContext__, _randid_)
            self.gatherMetrics(self.__renderDistributions__)
            self.gatherMetrics(self.__renderLines__, _randid_)
            self.df_pixels = self.gatherMetrics(self.__renderDots__, line_rendering_mode=False)
            self.gatherMetrics(self.__renderLegend__)
            self.gatherMetrics(self.__renderSVG__, _randid_)
        # trim verbose float tails from the finished SVG (idempotent; no-op on the
        # dataless placeholder) -- see Polars2SVG.roundSvgFloats
        self.svg = self.p2s.roundSvgFloats(self.svg)
        self.t_end          = time.time()
        self.t_overall      = self.t_end - self.t_start

    def _repr_svg_(self): return self.svg

    #
    # webgpu() - WebGPU payload of the same render (buffers + manifest), composed from
    # the per-phase DisplayLists recorded during the render stage; the polars compute
    # is shared with the SVG path -- only the serialization differs.  Lazy + cached.
    #
    def webgpu(self):
        if getattr(self, '_gpu_payload_', None) is not None: return self._gpu_payload_
        _dl_ = self.gpuDisplayList()
        if _dl_ is None: return None
        self._gpu_payload_ = _dl_.webgpu_payload(self.p2s.glyphAtlas())
        return self._gpu_payload_

    #
    # gpuDisplayList() - the composed backend-neutral display list (also consumed
    # by smallp when this component renders as a cell)
    #
    def gpuDisplayList(self):
        if self.df is None or getattr(self, 'svg', None) is None: return None
        if getattr(self, '_gpu_dl_', None) is not None: return self._gpu_dl_
        w, h  = self.wxh
        _bg_  = self.p2s.colorTyped('background', 'default')
        _dl_  = DisplayList(w, h, bg=_bg_)
        _dl_.rect(0, 0, w, h, _bg_)
        _dl_.extend(self._dl_background_)
        _dl_.extend(self._dl_context_)
        _dl_.extend(self._dl_lines_)
        # SVG clips only the circle-mode dot group (rect dots carry no clip-path)
        _dot_scissor_ = self._clip_rect_ if not isinstance(self.dot_size_orig, int) else None
        _dl_.extend(self.__dotsToInstances__(), scissor=_dot_scissor_)
        _dl_.extend(self._dl_distributions_)
        if getattr(self, '_dl_legend_', None) is not None: _dl_.extend(self._dl_legend_)
        self._gpu_dl_ = _dl_
        return _dl_

    #
    # __dotsToInstances__() - convert the already-computed df_pixels table into GPU
    # circle/rect instances (same columns the SVG concat_str serializes)
    #
    def __dotsToInstances__(self):
        _dl_ = DisplayList(self.wxh[0], self.wxh[1])
        if self.dot_size_orig is None or self.df_pixels is None or len(self.df_pixels) == 0:
            return _dl_
        _df_ = self.df_pixels
        _color_default_ = self.p2s.colorTyped('data', 'default')
        if '__hexcolor__' in _df_.columns:
            _df_   = _df_.with_columns(self.p2s.rgbFromHexPolarsOperations('__hexcolor__', '__r_f__', '__g_f__', '__b_f__'))
            _rgba_ = ('__r_f__', '__g_f__', '__b_f__')
        else:
            _r_, _g_, _b_, _ = hexToRGBA(_color_default_)
            _rgba_ = (_r_, _g_, _b_)
        _opacity_ = '__fill_opacity__' if '__fill_opacity__' in _df_.columns else 1.0
        if isinstance(self.dot_size_orig, int):
            # SVG rect dots take width/height from a CSS style block; bake per-instance
            _dl_.rects_table(_df_, '__xpx__', '__ypx__', self.dot_size_orig, self.dot_size_orig,
                             _rgba_, opacity=_opacity_, svg_col=None)
        else:
            if   '__radius__' in _df_.columns:           _r_col_ = '__radius__'
            elif isinstance(self.dot_size_orig, float):  _r_col_ = self.dot_size_orig
            else:                                        _r_col_ = 1.0
            _dl_.circles_table(_df_, '__xpx__', '__ypx__', _r_col_,
                               _rgba_, opacity=_opacity_, svg_col=None)
        return _dl_

    #
    # gatherMetrics()
    #
    def gatherMetrics(self, callable, *args, **kwargs):
        t0 = time.time()
        _results_ = callable(*args, **kwargs)
        t1 = time.time()
        if callable.__name__ not in self.timing_metrics: self.timing_metrics[callable.__name__] = 0.0
        self.timing_metrics[callable.__name__] += t1 - t0
        return _results_

    #
    # __parseInput__()
    #
    def __parseInput__(self, *args, **kwargs):
        _unknown_ = set(kwargs) - self._VALID_KWARGS
        if _unknown_:
            raise TypeError(f'XYp: unexpected keyword argument(s): {sorted(_unknown_)}')

        # Single source of truth for every parameter (name -> from-scratch default);
        # drives both the from-scratch assignment and the keyword-override copy below.
        # (x/y also arrive positionally; the positional loop below fills those in.)
        _defaults_ = {
            'x':                     None,
            'y':                     None,
            'color':                 None,
            'opacity':               None,
            'line':                  None,
            'line_order_by':         None,
            'dot_size':              1,          # note: converted to a list later (saved as dot_size_orig)
            'dot_size_supersample':  1,          # int >= 1; only affects integer dot_size (raster) plots
            'x_distributions':       None,
            'y_distributions':       None,
            'x_order':               None,
            'y_order':               None,
            'background':                    None,      # {key: shapely_object, ...}
            'background_label_color':        None,      # None / 'vary' / dict / '#rrggbb'
            'background_opacity':            1.0,       # None / number / dict
            'background_fill':               None,      # None / 'vary' / dict / '#rrggbb'
            'background_stroke_w':           1.0,       # None / number / dict
            'background_stroke':             'default', # None / 'default' / dict / '#rrggbb'
            'draw_context':          True,
            'insets':                (2, 2),
            'wxh':                   (256, 256),
            'txt_h':                 12,
            'x_range':               None,
            'y_range':               None,
            'x_shared_label_range':  None,
            'y_shared_label_range':  None,
            'dot_size_range':        (0.5, 4.0),
            'opacity_range':         (0.5, 1.0),
            'sm_shared':             set(),
            'color_magnitude_min':             None,
            'color_magnitude_max':             None,
            'color_stretched_global_values':   None,
            'dot_size_global_min':             None,
            'dot_size_global_max':             None,
            'use_lazy_execution':    True,
            'legend':                False,
        }
        self.p2s.assertParamSpecMatches('XYp', self._VALID_KWARGS, _defaults_)

        self.df, self.df_orig, self.x, self.y = None, None, None, None
        # Is there a template?
        self.template = None
        for i in range(len(args)):
            if isinstance(args[i], XYp): self.template = args[i]
        if 'template' in kwargs: self.template = kwargs['template']
        if self.template is not None:
            _template_copy_ = self.template
            self.p2s._clone_template_state(self, _template_copy_)
            self.template = _template_copy_
            if self.df is not None and '__p2s_index__' in self.df.columns: self.df = self.df.drop('__p2s_index__')
        else:
            self.p2s.assignScratchDefaults(self, _defaults_)
            # from-scratch builds only — a template clone is an exact snapshot and
            # must not re-apply session defaults (see Polars2SVG._apply_defaults)
            kwargs = self.p2s._apply_defaults('xyp', kwargs)

        # Handle the arguments first. Positional dispatch keys on type: the first
        # str/tuple/list positional is x, the second is y. Track which of x/y were
        # inferred positionally (vs. a keyword) so a downstream "column not found"
        # can name positional dispatch as the likely cause (see positionalDispatchHint).
        self._x_from_positional_, self._y_from_positional_ = False, False
        _is_x_, _is_y_ = True, False
        for i in range(len(args)):
            if   isinstance(args[i], pl.DataFrame) and self.df is None:
                self.df = self.df_orig = args[i]
            elif isinstance(args[i], str) or isinstance(args[i], tuple) or isinstance(args[i], list):
                # A hex color positioned as x/y is almost certainly a color= value
                # passed positionally by mistake -- x/y can never be a literal color,
                # so catch it here with a targeted message instead of deferring to a
                # confusing "dataframe does not contain column #ff0000".
                _which_ = 'x' if _is_x_ else ('y' if _is_y_ else None)
                if _which_ is not None and isinstance(args[i], self.p2s.HexColorString):
                    raise TypeError(f'XYp.__parseInput__():  positional argument @ index {i} "{args[i]}" '
                                    f'is a hex color but was dispatched to {_which_} (x/y cannot be a color); '
                                    f'pass it explicitly as color=... ')
                if   _is_x_: self.x, self._x_from_positional_, _is_x_, _is_y_ = args[i], True, False, True
                elif _is_y_: self.y, self._y_from_positional_,         _is_y_ = args[i], True,        False
                else: raise TypeError(f'XYp.__parseInput__():  Unknown argument @ index {i}: "{args[i]}"')
            elif isinstance(args[i], XYp):
                pass # handled already
            else:
                raise TypeError(f'XYp.__parseInput__():  Unknown argument @ index {i}: "{args[i]}"')

        # Then the keyword arguments next — every parameter in _defaults_ (x/y included).
        # An explicit x=/y= keyword overrides any positional, so the assignment is no
        # longer an inference -- clear the positional-origin flag in that case.
        if 'x' in kwargs: self._x_from_positional_ = False
        if 'y' in kwargs: self._y_from_positional_ = False
        self.p2s.assignKwargOverrides(self, _defaults_, kwargs)
        # df is not in the spec (it sets two attributes); catch it if passed by name.
        if 'df'              in kwargs: self.df = self.df_orig  = kwargs['df']

        # Remember the original dot size... if it's an int, it changes the pipeline
        # (initialization normalizes dot_size, so the original is saved here first)
        if self.template is None:
            self.dot_size_orig = self.dot_size
            if isinstance(self.dot_size_orig, float): self.dot_size_range = (self.dot_size_orig, self.dot_size_orig)

        # "No data" placeholder for early error visibility -- only ever seen when
        # no df is supplied (a successful render overwrites self.svg); makes a
        # dropped-df plumbing mistake visible instead of a silently blank canvas.
        # The x/y specs are echoed as notes to aid debugging.
        self.wxh = self.p2s.normalizeWxh(self.wxh, 'Xyp')
        w, h = self.wxh
        self.svg = self.p2s.placeholderSVG(w, h, notes=[f'x: {self.x}', f'y: {self.y}'])

    #
    # __cleanTuple__() -- separate into a clean tuple and a set of enums
    #
    #
    # __isEnum__() -- safe enum-membership test.  `all_enums` is a set, so a raw
    # `x in all_enums` raises `TypeError: unhashable type` for a list/dict/set
    # nested inside a spec -- and even a tuple that *contains* an unhashable item
    # passes `isinstance(x, Hashable)` yet still raises when actually hashed.
    # An unhashable item can never be an enum, so short-circuit to False rather
    # than crashing.
    #
    def __isEnum__(self, _item_):
        try:
            return _item_ in self.p2s.all_enums
        except TypeError:
            return False

    #
    # __assertHashableSpecItem__() -- reject an unhashable item nested inside a
    # spec with a descriptive error instead of an opaque
    # `TypeError: unhashable type: 'list'`.
    #
    def __assertHashableSpecItem__(self, _item_, where):
        if not isinstance(_item_, typing.Hashable):
            raise ValueError(f'XYp.{where}():  {type(_item_).__name__} {_item_!r} cannot be nested inside a '
                             f'field/enum spec (expected a field-name string, hex color, or enum)')

    def __cleanTuple__(self, _tuple_):
        _list_, _enums_ = [], set()
        for i in range(len(_tuple_)):
            self.__assertHashableSpecItem__(_tuple_[i], '__cleanTuple__')
            if self.__isEnum__(_tuple_[i]): _enums_.add(_tuple_[i])
            else: _list_.append(_tuple_[i])
        return tuple(_list_), _enums_

    #
    # __isLiteral__() -- determine if an object (list, tuple, or single value) is a literal
    # - certain fields can't be literals -- for example, 'x', 'y', 'line', 'line_order_by'
    #
    def __isLiteral__(self, _obj_, attr_name):
        if   isinstance(_obj_, list):
            for i in range(len(_obj_)):
                if self.__isLiteral__(_obj_[i], attr_name) == False: return False
            return True
        elif isinstance(_obj_, tuple) and len(_obj_) == 1: 
            return self.__isLiteral__(_obj_[0], attr_name)
        else:
            if   attr_name == 'x':             return False
            elif attr_name == 'y':             return False
            elif attr_name == 'color':         return isinstance(_obj_, self.p2s.HexColorString)
            elif attr_name == 'dot_size':      return isinstance(_obj_, int) or isinstance(_obj_, float)
            elif attr_name == 'opacity':       return isinstance(_obj_, float)
            elif attr_name == 'line':          return False
            elif attr_name == 'line_order_by': return False
            else: raise ValueError(f'XYp.__isLiteral__():  attribute name: {attr_name} is not valid')

    #
    # __toListAndExtractEnums__() -- convert to a list, determine if it's a literal, and extract any enums
    #
    def __toListAndExtractEnums__(self, v, attr_name):
        if   isinstance(v, list):
            _list_  = []
            _enums_ = set()
            for i in range(len(v)):
                _item_ = v[i]
                if    isinstance(_item_, tuple):
                    _tuple_, _enums_from_tuple_ = self.__cleanTuple__(_item_)
                    _list_.append(_tuple_)
                    _enums_ |= _enums_from_tuple_
                else:
                    self.__assertHashableSpecItem__(_item_, '__toListAndExtractEnums__')
                    if self.__isEnum__(_item_): _enums_.add(_item_)
                    else:                       _list_.append(_item_)
            return _list_, self.__isLiteral__(_list_, attr_name), _enums_
        elif isinstance(v, tuple):
            _tuple_, _enums_ = self.__cleanTuple__(v)
            return [_tuple_], self.__isLiteral__(_tuple_, attr_name), _enums_
        else:
            self.__assertHashableSpecItem__(v, '__toListAndExtractEnums__')
            if self.__isEnum__(v):
                return None, False, {v}
            return [v], self.__isLiteral__(v, attr_name), set()

    #
    # __separateAndCleanParam__() - separate into a clean tuple and a set of enums
    #
    def __separateAndCleanParam__(self, distribution_param, initial_call=True):
        _ints_, _floats_, _colors_, _fields_, _enums_ = [], [], [], [], set()
        if   isinstance(distribution_param, str): _fields_.append((distribution_param,))
        elif self.__isEnum__(distribution_param): _enums_ |= {distribution_param}
        else:
            if isinstance(distribution_param, list) == False: distribution_param = list(distribution_param)
            for i in range(len(distribution_param)):
                if   isinstance(distribution_param[i], tuple):
                    _tuple_fields_, _tuple_ints_, _tuple_floats_, _tuple_colors_, _tuple_enums_ = self.__separateAndCleanParam__(list(distribution_param[i]), False)
                    if len(_tuple_colors_) > 0: _tuple_fields_.extend(_tuple_colors_)  # add the color to the end of the tuple
                    if len(_tuple_ints_)   > 0: _ints_        .extend(_tuple_ints_)
                    if len(_tuple_floats_) > 0: _floats_      .extend(_tuple_floats_)
                    _enums_ |= _tuple_enums_
                    _fields_.append(tuple(_tuple_fields_))
                elif isinstance(distribution_param[i], int):                     _ints_    .append(distribution_param[i])
                elif isinstance(distribution_param[i], float):                   _floats_  .append(distribution_param[i])
                elif isinstance(distribution_param[i], self.p2s.HexColorString): _colors_  .append(distribution_param[i])
                elif isinstance(distribution_param[i], str):                     _fields_  .append(distribution_param[i])
                elif self.__isEnum__(distribution_param[i]):                     _enums_   .add   (distribution_param[i])
                else: raise ValueError(f'XYp.__separateAndCleanParam__():  {type(distribution_param[i]).__name__} {distribution_param[i]!r} is not valid')

        # If it's the original call, then do additional cleaning on the return values
        if initial_call:
            # Fit the fields into a specific format
            if len(_fields_) >  0:
                # Make sure they are all tuples
                _new_fields_ = []
                for i in range(len(_fields_)):
                    if isinstance(_fields_[i], tuple): _new_fields_.append(_fields_[i])
                    else:                              _new_fields_.append((_fields_[i],))
                _fields_ = _new_fields_
                # Make sure the last field is a color
                _new_fields_ = []
                for i in range(len(_fields_)):
                    if   isinstance(_fields_[i][-1], self.p2s.HexColorString): _new_fields_.append(_fields_[i])
                    elif len(_colors_) == 1:                                   _new_fields_.append(_fields_[i] + (_colors_[0],))
                    else:
                        _color_ = self.p2s.color(_fields_[i])
                        _new_fields_.append(_fields_[i] + (_color_,))
                _fields_ = _new_fields_

        return _fields_, _ints_, _floats_, _colors_, _enums_

    #
    # __distributionsParamSetDefaults__() -- set defaults for distributions / do some validation too
    #
    def __distributionsParamSetDefaults__(self, _fields_, _ints_, _floats_, _colors_, _enums_):
        if   len(_ints_)   >  1: raise ValueError('XYp.__distributionsParamSetDefaults__():  more than one bin found')
        elif len(_ints_)   == 0: _enums_ |= {self.p2s.DISTRIBUTION_AUTOBINp}
        if   len(_floats_) >  1: raise ValueError('XYp.__distributionsParamSetDefaults__():  more than one height percentage found')
        if   len(_colors_) >  1: raise ValueError('XYp.__distributionsParamSetDefaults__():  more than one distribution color found')
        if   self.p2s.DISTRIBUTION_INSIDEp in _enums_ and self.p2s.DISTRIBUTION_OUTSIDEp in _enums_: 
            raise ValueError('XYp.__distributionsParamSetDefaults__():  cannot specify both inside and outside distributions')
        _relative_to_ = _enums_ & {self.p2s.DISTRIBUTION_COLOR_MIN_TO_COLOR_MAX,
                                   self.p2s.DISTRIBUTION_ZERO_TO_COLOR_MAX,
                                   self.p2s.DISTRIBUTION_ALL_MIN_TO_ALL_MAX,
                                   self.p2s.DISTRIBUTION_ZERO_TO_ALL_MAX}
        if len(_relative_to_) >  1: raise ValueError(f'XYp.__distributionsParamSetDefaults__():  only one relative setting allowed ({_relative_to_})')
        if len(_relative_to_) == 0: _enums_ |= {self.p2s.DISTRIBUTION_ZERO_TO_COLOR_MAX}
        return _fields_, _ints_, _floats_, _colors_, _enums_

    #
    # __cleanLineParamTuple__()
    #
    def __cleanLineParamTuple__(self, _param_, widths_default, int_lists_default, colors_default, enums_default):
        def isListOfInts(data): return isinstance(data, list) and all(type(item) is int for item in data)
        _fields_, _widths_, _colors_, _int_lists_, _enums_ = [], [], [], [], set()
        for i in range(len(_param_)):
            _obj_ = _param_[i]
            if   isinstance(_obj_, self.p2s.HexColorString):         _colors_   .append(_obj_)
            elif isinstance(_obj_, str):                             _fields_   .append(_obj_)
            elif isinstance(_obj_, int) or isinstance(_obj_, float): _widths_   .append(_obj_)
            elif isinstance(_obj_, self.p2s.RenderEnumsP):           _enums_    .add   (_obj_)
            elif isListOfInts(_obj_):                                _int_lists_.append(_obj_)
            else: raise TypeError(f'XYp.__cleanLineParamTuple__() - param "{_param_}" not of type string, int/float, HexColorString, RenderEnumsP, or list of ints')

        # Ensure that there aren't too many...
        if len(_widths_)    >  1: raise ValueError(f'XYp.__cleanLineParamTuple__() - param "{_param_}" has more than one width')
        if len(_int_lists_) >  1: raise ValueError(f'XYp.__cleanLineParamTuple__() - only one svg dot array allowed ... param "{_param_}')        
        if len(_colors_)    >  1: raise ValueError(f'XYp.__cleanLineParamTuple__() - param "{_param_}" has more than one color')
        if len(_fields_)    == 0: raise ValueError(f'XYp.__cleanLineParamTuple__() - param "{_param_}" has no fields -- at least one field required for the group_by')

        # Inherit default values (if they weren't specified)
        _widths_,    _enums_ = self.__lineParamLineWidthCleaner__  (_param_, _widths_,    _enums_, widths_default,    enums_default)
        _int_lists_, _enums_ = self.__lineParamLineStyleCleaner__  (_param_, _int_lists_, _enums_, int_lists_default, enums_default)
        _colors_,    _enums_ = self.__lineParamLineColorCleaner__  (_param_, _colors_,    _enums_, colors_default,    enums_default)
        _enums_              = self.__lineParamLineOpacityCleaner__(_param_,              _enums_,                    enums_default)

        # Final form should be ['field', 'field', ..., None, []|[int,...], None|HexColor, set({enum,enum,...})]
        return tuple(_fields_ + _widths_ + _int_lists_ + _colors_ + [_enums_])

    #
    # __lineParamLineWidthCleaner__()
    #
    def __lineParamLineWidthCleaner__(self, _param_, _widths_, _enums_, _parent_widths_, _parent_enums_):
        _width_enums_ = {self.p2s.LINEWIDTH_DOTSIZE_MEAN, self.p2s.LINEWIDTH_DOTSIZE_VARIABLE, self.p2s.LINEWIDTH_DOTSIZE_SPECIFIED}
        _lw_enum_     = _enums_        & _width_enums_
        _lw_parent_   = _parent_enums_ & _width_enums_
        # Can't specify more than one enum
        if   len(_lw_enum_)  > 1: raise ValueError(f'XYp.__lineParamLineWidthCleaner__() - param "{_param_}" has more than one line width setting ({_lw_enum_})')
        # If the enum is not specified, then pull from the parent or use the default
        elif len(_lw_enum_) == 0 and len(_widths_) > 0:
            _enums_ |= {self.p2s.LINEWIDTH_DOTSIZE_SPECIFIED}
        elif len(_lw_enum_) == 0:
            if   len(_lw_parent_) == 1: 
                _enums_ |= _lw_parent_
                if self.p2s.LINEWIDTH_DOTSIZE_SPECIFIED in _lw_parent_:
                    if len(_widths_) == 0: 
                        if len(_parent_widths_) == 0: _widths_.append(0.5)  # Default line width is 0.5
                        else:                         _widths_.append(_parent_widths_[0])
            elif len(_widths_) == 0: 
                _widths_.append(0.5)  # Default line width is 0.5
                _enums_ |= {self.p2s.LINEWIDTH_DOTSIZE_SPECIFIED}
        # If the enum is specified, then make sure the value is in the _widths_ list
        elif len(_lw_enum_) == 1 and self.p2s.LINEWIDTH_DOTSIZE_SPECIFIED in _lw_enum_:
            if len(_widths_) == 0:
                if len(_parent_widths_) == 0: _widths_.append(0.5)  # Default line width is 0.5
                else:                         _widths_.append(_parent_widths_[0])
        # Make sure there's an item in the widths
        if len(_widths_) == 0: _widths_.append(None)
        # Do some checking...
        if len(_enums_ & _width_enums_) > 1: raise ValueError(f'XYp.__lineParamLineWidthCleaner__() - checker - param "{_param_}" has more than one line width setting ({_enums_ & _width_enums_})')
        if len(_widths_) != 1:               raise ValueError(f'XYp.__lineParamLineWidthCleaner__() - checker - param "{_param_}" does not have a single value for line width ({_widths_})')
        if (isinstance(_widths_[0], int) or isinstance(_widths_[0], float)) and (self.p2s.LINEWIDTH_DOTSIZE_SPECIFIED not in _enums_):
            raise ValueError(f'XYp.__lineParamLineWidthCleaner__() - checker - param "{_param_}" has a width value ({_widths_[0]}) but the correct enum is not specified ({_enums_})')
        if _widths_[0] is None and len({self.p2s.LINEWIDTH_DOTSIZE_MEAN, self.p2s.LINEWIDTH_DOTSIZE_VARIABLE} & _enums_) != 1:
            raise ValueError(f'XYp.__lineParamLineWidthCleaner__() - checker - param "{_param_}" has a width value of None but the correct enum is not specified ({_enums_})')
        return _widths_, _enums_

    #
    # __lineParamLineStyleCleaner__()
    #
    def __lineParamLineStyleCleaner__(self, _param_, _int_lists_, _enums_, _parent_int_lists_, _parent_enums_):
        _style_enums_ = {self.p2s.LINESTYLE_SOLID, self.p2s.LINESTYLE_DOTTED, self.p2s.LINESTYLE_SPECIFIED}
        _ls_enum_   = _enums_        & _style_enums_
        _ls_parent_ = _parent_enums_ & _style_enums_
        if   len(_ls_enum_)  > 1: raise ValueError(f'XYp.__lineParamLineStyleCleaner__() - param "{_param_}" has more than one line style setting ({_ls_enum_})')
        elif len(_ls_enum_) == 0 and len(_int_lists_) > 0 and _int_lists_[0] != []:
            _enums_ |= {self.p2s.LINESTYLE_SPECIFIED}
        elif len(_ls_enum_) == 0:
            if len(_ls_parent_) == 1:
                _enums_ |= _ls_parent_
                if len(_int_lists_) == 0 and len(_parent_int_lists_) > 0: _int_lists_.append(_parent_int_lists_[0])
            else:
                _enums_ |= {self.p2s.LINESTYLE_SOLID}
        elif len(_ls_enum_) == 1 and self.p2s.LINESTYLE_SPECIFIED in _ls_enum_ and len(_int_lists_) == 0:
            self.p2s.logger.warning(f'XYp.__lineParamLineStyleCleaner__() - param "{_param_}" has line style specified but no line style value - setting to LINESTYLE_SOLID')
            _enums_ |= {self.p2s.LINESTYLE_SOLID}
            _enums_ -= {self.p2s.LINESTYLE_SPECIFIED}
        # Make sure there's an item in the int_lists
        if len(_int_lists_) == 0: _int_lists_.append([])
        # Do some checking...
        if len(_enums_ & _style_enums_) > 1: raise ValueError(f'XYp.__lineParamLineStyleCleaner__() - checker - param "{_param_}" has more than one line style setting ({_enums_ & _style_enums_})')
        if len(_int_lists_) != 1:            raise ValueError(f'XYp.__lineParamLineStyleCleaner__() - checker - param "{_param_}" does not have a single value for line style ({_int_lists_})')
        if _int_lists_[0] == [] and len({self.p2s.LINESTYLE_SOLID, self.p2s.LINESTYLE_DOTTED} & _enums_) != 1:
            raise ValueError(f'XYp.__lineParamLineStyleCleaner__() - checker - param "{_param_}" has an empty line style value but the correct enum is not specified ({_enums_})')
        if _int_lists_[0] != [] and len({self.p2s.LINESTYLE_SPECIFIED} & _enums_) != 1:
            raise ValueError(f'XYp.__lineParamLineStyleCleaner__() - checker - param "{_param_}" has a non-empty line style value but the correct enum is not specified ({_enums_})')
        return _int_lists_, _enums_

    #
    # __lineParamLineColorCleaner__()
    #    
    def __lineParamLineColorCleaner__(self, _param_, _colors_, _enums_, _parent_colors_, _parent_enums_):
        _color_enums_ = {self.p2s.LINECOLOR_GROUPBY, self.p2s.LINECOLOR_FIELD, self.p2s.LINECOLOR_SPECIFIED}
        _lc_enum_   = _enums_        & _color_enums_
        _lc_parent_ = _parent_enums_ & _color_enums_
        if   len(_lc_enum_) >  1: raise ValueError(f'XYp.__lineParamLineColorCleaner__() - param "{_param_}" has more than one line color setting ({_lc_enum_})')
        elif len(_lc_enum_) == 0 and len(_colors_) > 0:
            _enums_ |= {self.p2s.LINECOLOR_SPECIFIED}
        elif len(_lc_enum_) == 0:
            if len(_lc_parent_) == 1:
                _enums_ |= _lc_parent_
                if len(_colors_) == 0 and len(_parent_colors_) > 0: _colors_.append(_parent_colors_[0])
            else:
                _enums_ |= {self.p2s.LINECOLOR_GROUPBY}
        elif len(_lc_enum_) == 1 and self.p2s.LINECOLOR_SPECIFIED in _lc_enum_ and len(_colors_) == 0:
            self.p2s.logger.warning(f'XYp.__lineParamLineColoreCleaner__() - param "{_param_}" has color specified but no color value - setting to LINECOLOR_GROUPBY')
            _enums_ |= {self.p2s.LINECOLOR_GROUPBY}
            _enums_ -= {self.p2s.LINECOLOR_SPECIFIED}
        # Make sure there's an item in the colors
        if len(_colors_) == 0: _colors_.append(None)
        # Do some checking...
        if len(_enums_ & _color_enums_) > 1: raise ValueError(f'XYp.__lineParamLineColorCleaner__() - checker - param "{_param_}" has more than one color setting ({_enums_ & _color_enums_})')
        if len(_colors_) != 1:               raise ValueError(f'XYp.__lineParamLineColorCleaner__() - checker - param "{_param_}" does not have a single value for color ({_colors_})')
        if _colors_[0] is None and len({self.p2s.LINECOLOR_GROUPBY, self.p2s.LINECOLOR_FIELD} & _enums_) != 1:
            raise ValueError(f'XYp.__lineParamLineColorCleaner__() - checker - param "{_param_}" has an empty color value but the correct enum is not specified ({_enums_})')
        if _colors_[0] is not None and len({self.p2s.LINECOLOR_SPECIFIED} & _enums_) != 1:
            raise ValueError(f'XYp.__lineParamLineColorCleaner__() - checker - param "{_param_}" has a non-empty line color value but the correct enum is not specified ({_enums_})')
        return _colors_, _enums_

    #
    # __lineParamLineOpacityCleaner__()
    #
    def __lineParamLineOpacityCleaner__(self, _param_, _enums_, _parent_enums_):
        _opacity_enums_ = {self.p2s.LINEOPACITY_FIELD_MEAN, self.p2s.LINEOPACITY_FIELD_VARIABLE,  self.p2s.LINEOPACITY_100,
                           self.p2s.LINEOPACITY_75, self.p2s.LINEOPACITY_50, self.p2s.LINEOPACITY_25, self.p2s.LINEOPACITY_10}
        _lo_enum_     = _enums_        & _opacity_enums_
        _parent_enum_ = _parent_enums_ & _opacity_enums_
        if   len(_lo_enum_)  > 1: raise ValueError(f'XYp.__lineParamLineOpacityCleaner__() - param "{_param_}" has more than one line opacity setting ({_lo_enum_})')
        elif len(_lo_enum_) == 0 and len(_parent_enum_) == 1: _enums_ |= _parent_enum_
        elif len(_lo_enum_) == 0:                             _enums_ |= {self.p2s.LINEOPACITY_100}
        # Do some checking...
        if len(_enums_ & _opacity_enums_) > 1: raise ValueError(f'XYp.__lineParamLineOpacityCleaner__() - checker - param "{_param_}" has more than one opacity setting ({_enums_ & _opacity_enums_})')
        return _enums_

    #
    # __cleanLineParam__()
    #
    def __cleanLineParam__(self, _param_):
        def isListOfInts(data): return isinstance(data, list) and all(type(item) is int for item in data)
        # Do a first pass and create a structure that can be cleaned
        _fields_, _widths_, _colors_, _int_lists_, _enums_ = [], [], [], [], set()
        if   isinstance(_param_, tuple): _fields_ = [_param_]
        elif isinstance(_param_, str):   _fields_ = [(_param_,)]
        elif isinstance(_param_, list):
            for i in range(len(_param_)):
                _obj_ = _param_[i]
                if   isinstance(_obj_, self.p2s.HexColorString):         _colors_   .append(_obj_)
                elif isinstance(_obj_, str):                             _fields_   .append((_obj_,))
                elif isinstance(_obj_, tuple):                           _fields_   .append(_obj_)
                elif isinstance(_obj_, int) or isinstance(_obj_, float): _widths_   .append(_obj_)
                elif isinstance(_obj_, self.p2s.RenderEnumsP):           _enums_    .add(_obj_)
                elif isListOfInts(_obj_):                                _int_lists_.append(_obj_)
                else: raise TypeError(f'XYp.__cleanLineParam__() - param "{_param_}" not of type string, int/float, HexColorString, RenderEnumsP, or list of ints')
        else: raise TypeError(f'XYp.__cleanLineParam__() - param "{_param_}" not of type string, int/float, HexColorString, RenderEnumsP, or list of ints')

        # Check for exceptions
        if len(_widths_)    >  1: raise ValueError(f'XYp.__cleanLineParam__() - param "{_param_}" has more than one width')        
        if len(_int_lists_) >  1: raise ValueError(f'XYp.__cleanLineParam__() - only one svg dot array allowed ... param "{_param_}')
        if len(_colors_)    >  1: raise ValueError(f'XYp.__cleanLineParam__() - param "{_param_}" has more than one color')

        # Fill in the defaults / fix up the enums
        _widths_,    _enums_ = self.__lineParamLineWidthCleaner__  (_param_, _widths_,    _enums_, [], set())
        _int_lists_, _enums_ = self.__lineParamLineStyleCleaner__  (_param_, _int_lists_, _enums_, [], set())
        _colors_,    _enums_ = self.__lineParamLineColorCleaner__  (_param_, _colors_,    _enums_, [], set())
        _enums_              = self.__lineParamLineOpacityCleaner__(_param_,              _enums_,     set())

        # Create the final list
        _final_ = []
        for _field_ in _fields_: _final_.append(self.__cleanLineParamTuple__(_field_, _widths_, _int_lists_, _colors_, _enums_))
        return _final_

    #
    # __expandToMaxLen__() -- expand a list to the max length
    #
    def __expandToMaxLen__(self, _list_, max_len):
        if   _list_ is None:         return None
        elif len(_list_) == 1:       return [_list_[0]] * max_len
        elif len(_list_) == max_len: return _list_
        else: raise ValueError(f'XYp.__expandToMaxLen__():  len(_list_) is not None, is not length one, and is not max_len ({max_len})')

    #
    # __validateDistributions__()
    # - parses x/y distribution params and resolves INSIDE/OUTSIDE placement when not explicitly set
    #
    def __validateDistributions__(self):
        if self.x_distributions is not None:
            _fields_, _ints_, _floats_, _colors_, _enums_ = self.__separateAndCleanParam__(self.x_distributions)
            _fields_, _ints_, _floats_, _colors_, _enums_ = self.__distributionsParamSetDefaults__(_fields_, _ints_, _floats_, _colors_, _enums_)
            self.x_distributions_clean = {'fields': _fields_, 'bins': _ints_, 'h_percs': _floats_, 'colors': _colors_, 'enums': _enums_}
        else: self.x_distributions_clean = None
        if self.y_distributions is not None:
            _fields_, _ints_, _floats_, _colors_, _enums_ = self.__separateAndCleanParam__(self.y_distributions)
            _fields_, _ints_, _floats_, _colors_, _enums_ = self.__distributionsParamSetDefaults__(_fields_, _ints_, _floats_, _colors_, _enums_)
            self.y_distributions_clean = {'fields': _fields_, 'bins': _ints_, 'h_percs': _floats_, 'colors': _colors_, 'enums': _enums_}
        else: self.y_distributions_clean = None

        # ... do something semi-intelligent with the distributions INSIDE/OUTSIDE settings (if the caller didn't set them)
        if   self.x_distributions_clean is not None and self.y_distributions_clean is not None:
            _x_inside_outside_set_ = len(self.x_distributions_clean['enums'] & {self.p2s.DISTRIBUTION_INSIDEp, self.p2s.DISTRIBUTION_OUTSIDEp}) > 0
            _y_inside_outside_set_ = len(self.y_distributions_clean['enums'] & {self.p2s.DISTRIBUTION_INSIDEp, self.p2s.DISTRIBUTION_OUTSIDEp}) > 0
            if   _x_inside_outside_set_ and _y_inside_outside_set_:
                pass # do nothing, they are both already set
            elif _x_inside_outside_set_: # if x is set, then set y
                if self.dot_size_orig is None:
                    if self.p2s.DISTRIBUTION_INSIDEp in self.x_distributions_clean['enums']: self.y_distributions_clean['enums'].add(self.p2s.DISTRIBUTION_OUTSIDEp)
                    else:                                                                    self.y_distributions_clean['enums'].add(self.p2s.DISTRIBUTION_INSIDEp)
                else: self.y_distributions_clean['enums'].add(self.p2s.DISTRIBUTION_OUTSIDEp)
            elif _y_inside_outside_set_: # if y is set, then set x
                if self.dot_size_orig is None:
                    if self.p2s.DISTRIBUTION_INSIDEp in self.y_distributions_clean['enums']: self.x_distributions_clean['enums'].add(self.p2s.DISTRIBUTION_OUTSIDEp)
                    else:                                                                    self.x_distributions_clean['enums'].add(self.p2s.DISTRIBUTION_INSIDEp)
                else: self.x_distributions_clean['enums'].add(self.p2s.DISTRIBUTION_OUTSIDEp)
            else:
                if self.dot_size_orig is None: # there's no dot size, put one on the inside and one on the outside
                    if     self.wxh[0] > self.wxh[1]: # the view is wider than taller ... put x inside and y outside
                        self.x_distributions_clean['enums'].add(self.p2s.DISTRIBUTION_INSIDEp)
                        self.y_distributions_clean['enums'].add(self.p2s.DISTRIBUTION_OUTSIDEp)
                    else:                             # the view is taller than wider (or equal) ... put x outside and y inside
                        self.x_distributions_clean['enums'].add(self.p2s.DISTRIBUTION_OUTSIDEp)
                        self.y_distributions_clean['enums'].add(self.p2s.DISTRIBUTION_INSIDEp)
                else:                          # there's a dot size... put them both on the outside
                    self.x_distributions_clean['enums'].add(self.p2s.DISTRIBUTION_OUTSIDEp)
                    self.y_distributions_clean['enums'].add(self.p2s.DISTRIBUTION_OUTSIDEp)
        elif self.x_distributions_clean is not None and len(self.x_distributions_clean['enums'] & {self.p2s.DISTRIBUTION_INSIDEp, self.p2s.DISTRIBUTION_OUTSIDEp}) == 0:
            if self.dot_size_orig is None: self.x_distributions_clean['enums'].add(self.p2s.DISTRIBUTION_INSIDEp)
            else:                          self.x_distributions_clean['enums'].add(self.p2s.DISTRIBUTION_OUTSIDEp)
        elif self.y_distributions_clean is not None and len(self.y_distributions_clean['enums'] & {self.p2s.DISTRIBUTION_INSIDEp, self.p2s.DISTRIBUTION_OUTSIDEp}) == 0:
            if self.dot_size_orig is None: self.y_distributions_clean['enums'].add(self.p2s.DISTRIBUTION_INSIDEp)
            else:                          self.y_distributions_clean['enums'].add(self.p2s.DISTRIBUTION_OUTSIDEp)

        # Set the distribution h percents based on the inside / outside rendering option (if not set by the caller)
        if self.x_distributions_clean is not None and len(self.x_distributions_clean['h_percs']) == 0:
            if self.p2s.DISTRIBUTION_INSIDEp in self.x_distributions_clean['enums']: self.x_distributions_clean['h_percs'].append(0.95)
            else:                                                                    self.x_distributions_clean['h_percs'].append(0.40)
        if self.y_distributions_clean is not None and len(self.y_distributions_clean['h_percs']) == 0:
            if self.p2s.DISTRIBUTION_INSIDEp in self.y_distributions_clean['enums']: self.y_distributions_clean['h_percs'].append(0.95)
            else:                                                                    self.y_distributions_clean['h_percs'].append(0.40)

    #
    # __applyTimeFieldTransforms__()
    # - scans all clean params for t-field strings and applies the required Polars column transforms to self.df
    # - must only be called when self.df is not None
    #
    def __applyTimeFieldTransforms__(self):
        _ops_, _needed_columns_ = [], set()
        for _triple_ in [(self.x_clean,              self.x_is_lits,             'x'),
                         (self.y_clean,               self.y_is_lits,             'y'),
                         (self.color_clean,           self.color_is_lits,         'color'),
                         (self.dot_size_clean,        self.dot_size_is_lits,      'dot_size'),
                         (self.opacity_clean,         self.opacity_is_lits,       'opacity'),
                         (self.line_order_by_clean,   self.line_order_by_is_lits, 'line_order_by'),
                         (self.line_clean,            False,                      'line'),
                         (self.x_distributions,       False,                      'x_distributions'),
                         (self.y_distributions,       False,                      'y_distributions')]:
            _obj_, _lits_flag_, _attr_name_ = _triple_
            if _obj_ is None or _lits_flag_: continue
            if   isinstance(_obj_, list):
                for i in range(len(_obj_)):
                    _item_ = _obj_[i]
                    if   isinstance(_item_, str): self.__addTransformIfNeeded__(_item_, _ops_, _needed_columns_)
                    elif isinstance(_item_, tuple):
                        for j in range(len(_item_)):
                            _subitem_ = _item_[j]
                            if isinstance(_subitem_, str): self.__addTransformIfNeeded__(_subitem_, _ops_, _needed_columns_)
            elif isinstance(_obj_, tuple):
                for j in range(len(_obj_)):
                    _item_ = _obj_[j]
                    if isinstance(_item_, str): self.__addTransformIfNeeded__(_item_, _ops_, _needed_columns_)
            elif isinstance(_obj_, str): self.__addTransformIfNeeded__(_obj_, _ops_, _needed_columns_)
        if len(_ops_) > 0: self.df = self.df.with_columns(_ops_)

    #
    # __validateColumnTypes__()
    # - verifies that all referenced columns exist in self.df and that dtypes are consistent within each param
    # - must only be called when self.df is not None
    #
    def __validateColumnTypes__(self):
        # Most of the column checks are here / with the exception of line, line_order_by, and the x/y distributions
        for _triple_ in [(self.x_clean,             self.x_is_lits,             'x'),
                         (self.y_clean,             self.y_is_lits,             'y'),
                         (self.color_clean,         self.color_is_lits,         'color'),
                         (self.dot_size_clean,      self.dot_size_is_lits,      'dot_size'),
                         (self.opacity_clean,       self.opacity_is_lits,       'opacity'),
                         (self.line_order_by_clean, self.line_order_by_is_lits, 'line_order_by')]:
            _list_, _lits_flag_, _attr_name_ = _triple_
            if _list_ is None or _lits_flag_: continue
            # A missing x/y column may be an argument-order mistake in positional
            # dispatch -- name that as the likely cause (empty for color/etc.).
            _hint_ = self.p2s.positionalDispatchHint(
                'XYp', _attr_name_,
                (_attr_name_ == 'x' and self._x_from_positional_) or
                (_attr_name_ == 'y' and self._y_from_positional_))
            # Verify field existence
            for i in range(len(_list_)):
                _item_ = _list_[i]
                if isinstance(_item_, tuple):
                    for j in range(len(_item_)):
                        if _item_[j] not in self.df.columns: raise TypeError(f'XYp.__validateInput__():  dataframe does not contain column {_item_[j]} (Tuple Element){_hint_}')
                elif _item_ not in self.df.columns: raise TypeError(f'XYp.__validateInput__():  dataframe does not contain column {_item_}{_hint_}')
            # Verify datatypes
            _first_ = self.__columnDataTypes__(_list_[0])
            for i in range(len(_list_)):
                if self.__columnDataTypes__(_list_[i]) != _first_:
                    raise TypeError(f'XYp.__validateInput__():  dataframe column "{_attr_name_}" has inconsistent datatypes {_list_[i]} ({self.__columnDataTypes__(_list_[i])})')

        # Lines require a different methodology
        if self.line_clean is not None:
            for i in range(len(self.line_clean)):
                _tuple_ = self.line_clean[i]
                for j in range(len(_tuple_)):
                    if isinstance(_tuple_[j], str):
                        if _tuple_[j] not in self.df.columns: raise ValueError(f'XYp.__validateInput__():  dataframe does not contain column {_tuple_[j]} (line tuple element)')
                    else: break
                if self.p2s.LINECOLOR_FIELD in _tuple_[-1]:
                    if self.color is None: raise ValueError(f'XYp.__validateInput__():  line color field "{self.p2s.LINECOLOR_FIELD}" requires a color field')
                if self.p2s.LINEOPACITY_FIELD_MEAN in _tuple_[-1] or self.p2s.LINEOPACITY_FIELD_VARIABLE in _tuple_[-1]:
                    if self.opacity is None: raise ValueError(f'XYp.__validateInput__():  line opacity field "{self.p2s.LINEOPACITY_FIELD_MEAN}" or "{self.p2s.LINEOPACITY_FIELD_VARIABLE}" requires an opacity field')
                if self.p2s.LINEWIDTH_DOTSIZE_MEAN in _tuple_[-1] or self.p2s.LINEWIDTH_DOTSIZE_VARIABLE in _tuple_[-1]:
                    if isinstance(self.dot_size_orig, int): raise ValueError(f'XYp.__validateInput__():  line width field "{self.p2s.LINEWIDTH_DOTSIZE_MEAN}" or "{self.p2s.LINEWIDTH_DOTSIZE_VARIABLE}" requires a dot_size field')

        # Distributions require a different methodology
        for _pair_ in [(self.x_distributions, self.x_distributions_clean),
                       (self.y_distributions, self.y_distributions_clean)]:
            if _pair_[0] is None: continue
            _distributions_, _distributions_clean_ = _pair_
            _fields_       = _distributions_clean_['fields']
            _first_dtypes_ = None
            for i in range(len(_fields_)):
                _tuple_dtypes_ = []
                for j in range(len(_fields_[i])-1): # the last element is a color value
                    _field_ = _fields_[i][j]
                    if _field_ not in self.df.columns: raise ValueError(f'XYp.__validateInput__():  dataframe does not contain column {_field_}')
                    _tuple_dtypes_.append(self.df.dtypes[self.df.columns.index(_field_)])
                _field_dtypes_ = tuple(_tuple_dtypes_)
                if   i               == 0:              _first_dtypes_ = _field_dtypes_
                elif _field_dtypes_  != _first_dtypes_: raise TypeError(f'XYp.__validateInput__():  dataframe column "{_fields_[i]}" has inconsistent datatypes {_field_dtypes_} (first = {_first_dtypes_})')
            # Is scalar possible?  Make sure it was set correctly (if there's a column/columns specified, then either scalar or set must be specified)
            if _first_dtypes_ is not None: # it would be none if we are counting rows
                if   len(_first_dtypes_) > 1:
                    if self.p2s.SCALARp in _first_dtypes_: raise ValueError('XYp.__validateInput__():  distributions cannot be scalar when multiple fields are specified')
                    _distributions_clean_['enums'] |= {self.p2s.SETp}
                elif len(_first_dtypes_) == 1: # only one field specified... check if it's numeric
                    _column_name_ = _fields_[0][0]
                    if self.p2s.numericColumn(self.df, _column_name_):
                        if self.p2s.SETp not in _distributions_clean_['enums']: _distributions_clean_['enums'] |= {self.p2s.SCALARp}
                    else:
                        if self.p2s.SCALARp in _distributions_clean_['enums']: raise ValueError(f'XYp.__validateInput__():  distributions cannot use scalar operations when the column "{_column_name_}" is not numeric')
                        _distributions_clean_['enums'] |= {self.p2s.SETp}

    #
    # __addTransformIfNeeded__()
    # - appends a polars time-field transform operation to _ops_ if _str_ is a t-field not yet handled
    # - assumption: string spec items are only ever column names or hex colors
    #
    def __addTransformIfNeeded__(self, _str_, _ops_, _needed_columns_):
        if _str_ in _needed_columns_: return
        if isinstance(_str_, self.p2s.TField):
            self.p2s.warnIfTFieldAliasCollides(_str_, self.df_orig, 'XYp')
        elif isinstance(_str_, self.p2s.HexColorString) or \
             _str_ in self.df.columns                   or \
             self.p2s.isTField(_str_, df=self.df_orig) == False: return
        _column_, _enum_    = self.p2s.tFieldTuple(_str_)
        _types_             = self.p2s.tFieldAccepts(_str_)
        _column_dtype_okay_ = False
        for _type_ in _types_:
            if isinstance(self.df.dtypes[self.df.columns.index(_column_)], _type_):
                _column_dtype_okay_ = True
        if _column_dtype_okay_ == False: raise ValueError(f'XYp.__validateInput__():  the column {self.df.columns[self.df.columns.index(_column_)]} is not of type {_types_} for t-field {_str_}')
        _needed_columns_.add(_str_)
        _ops_.append(self.p2s.polarsOperationForEnum(_column_, _enum_).alias(_str_))

    #
    # __columnDataTypes__()
    # - returns the dtype(s) for a column name or tuple of column names from self.df
    #
    def __columnDataTypes__(self, _obj_):
        if isinstance(_obj_, tuple):
            _as_list_ = []
            for i in range(len(_obj_)):
                if isinstance(_obj_[i], str): _as_list_.append(self.df.dtypes[self.df.columns.index(_obj_[i])])
            return tuple(_as_list_)
        else: return self.df.dtypes[self.df.columns.index(_obj_)]

    #
    # __validateInput__()
    #
    def __validateInput__(self):
        self.p2s.checkReservedColumns(self.df, 'XYp')
        if self.x is None: raise ValueError('XYp.__validateInput__():  x must be specified')
        if self.y is None: raise ValueError('XYp.__validateInput__():  y must be specified')

        # Normalize legend= eagerly so a bad spec fails fast (raises InvalidSpecError).
        self.legend_spec = self.p2s.legendResolveSpec(self.legend)

        # dot_size_supersample must be a positive integer (bool excluded -- True/False
        # is a mistake, not a factor of 1/0).  It only subdivides the integer-dot raster
        # grid; for float/field dot sizes it is inert, so warn once when set > 1.
        _ss_ = getattr(self, 'dot_size_supersample', 1)
        if not isinstance(_ss_, int) or isinstance(_ss_, bool) or _ss_ < 1:
            raise ValueError(f'XYp.__validateInput__():  dot_size_supersample must be a positive integer, got {_ss_!r}')
        if _ss_ > 1 and self.df is not None and not isinstance(self.dot_size_orig, int):
            self.p2s.logger.warning('XYp: dot_size_supersample only affects integer dot_size (raster) plots; '
                                    'it is inert for float/field dot sizes.')

        #
        # Initialize variables
        #
        self.x_clean,             self.x_is_lits,             self.x_enums             = None, False, set()
        self.y_clean,             self.y_is_lits,             self.y_enums             = None, False, set()
        self.color_clean,         self.color_is_lits,         self.color_enums         = None, False, set()
        self.dot_size_clean,      self.dot_size_is_lits,      self.dot_size_enums      = None, False, set()
        self.opacity_clean,       self.opacity_is_lits,       self.opacity_enums       = None, False, set()
        self.line_order_by_clean, self.line_order_by_is_lits, self.line_order_by_enums = None, False, set()
        self.line_clean                                                                = None

        #
        # Pull out any enums and ensure everything is a list
        # - use a different variable for the clean version
        # - this is necessary to make the templating work
        #
        if self.x             is not None: self.x_clean,             self.x_is_lits,             self.x_enums             = self.__toListAndExtractEnums__(self.x,             'x')
        if self.y             is not None: self.y_clean,             self.y_is_lits,             self.y_enums             = self.__toListAndExtractEnums__(self.y,             'y')
        if self.color         is not None: self.color_clean,         self.color_is_lits,         self.color_enums         = self.__toListAndExtractEnums__(self.color,         'color')
        if self.dot_size      is not None: self.dot_size_clean,      self.dot_size_is_lits,      self.dot_size_enums      = self.__toListAndExtractEnums__(self.dot_size,      'dot_size')
        if self.opacity       is not None: self.opacity_clean,       self.opacity_is_lits,       self.opacity_enums       = self.__toListAndExtractEnums__(self.opacity,       'opacity')
        if self.line_order_by is not None: self.line_order_by_clean, self.line_order_by_is_lits, self.line_order_by_enums = self.__toListAndExtractEnums__(self.line_order_by, 'line_order_by')
        if self.line          is not None: self.line_clean                                                                = self.__cleanLineParam__(self.line)

        #
        # Some combinations of enums aren't allowed -- check for them here
        #
        for _enums_ in [self.x_enums, self.y_enums, self.color_enums, self.dot_size_enums, self.opacity_enums, self.line_order_by_enums]:
            time_enums = _enums_ & (self.p2s.time_linear_types | self.p2s.time_periodic_types)
            if len(time_enums) > 1: raise ValueError(f'XYp.__validateInput__() - more than one time enum found in the input: {time_enums}')

        #
        # Check the small multiples options
        #
        _allowed_smallp_enums_ = {self.p2s.SM_X,      # x axis is shared across small multiples
                                  self.p2s.SM_Y,      # y axis is shared across small multiples
                                  self.p2s.SM_COUNT,  # count (dot_size) is shared across small multiples // excludes the "all" small multiple
                                  self.p2s.SM_COLOR}  # color is shared across small multiples            // excludes the "all" small multiple
        if len(self.sm_shared - _allowed_smallp_enums_) > 0: raise ValueError(f'XYp.__validateInput__() - smallp_options contains an invalid enum: {self.sm_shared - _allowed_smallp_enums_}')

        #
        # background shapes require scalar (numeric / date / datetime) axes on both x and y
        #
        if self.background is not None and self.df is not None:
            def _axis_is_scalar_(clean, enums):
                if clean is None:                    return False  # enum-only axis
                if self.p2s.SETp in enums:           return False  # explicitly categorical
                for field in clean:
                    if isinstance(field, tuple):     return False  # multi-field → struct/categorical
                    if not (self.p2s.numericColumn (self.df, field) or
                            self.p2s.dateColumn    (self.df, field) or
                            self.p2s.dateTimeColumn(self.df, field)): return False
                return True
            if not _axis_is_scalar_(self.x_clean, self.x_enums) or \
               not _axis_is_scalar_(self.y_clean, self.y_enums):
                raise ValueError('XYp.__validateInput__(): background shapes require numeric (scalar) x and y axes')

        #
        # Handle the distributions (if set)
        #
        self.__validateDistributions__()

        #
        # If there's a dataframe supplied, apply any time-field transforms and check column existence/types
        #
        if self.df is not None:
            self.__applyTimeFieldTransforms__()
            self.__validateColumnTypes__()

        #
        # The attributes need to be None, length one, or length max_len
        # ... if they are a list of length one, expand them to max_len
        # ... this first part figures out the max_len
        #
        x_len               = 0 if self.x_clean               is None else len(self.x_clean)
        y_len               = 0 if self.y_clean               is None else len(self.y_clean)
        color_len           = 0 if self.color_clean           is None else len(self.color_clean)
        dot_size_len        = 0 if self.dot_size_clean        is None else len(self.dot_size_clean)
        opacity_len         = 0 if self.opacity_clean         is None else len(self.opacity_clean)
        line_len            = 0 if self.line_clean            is None else len(self.line_clean)
        line_order_by_len   = 0 if self.line_order_by_clean   is None else len(self.line_order_by_clean)
        x_distributions_len = 0 if self.x_distributions       is None else len(self.x_distributions_clean['fields'])
        y_distributions_len = 0 if self.y_distributions       is None else len(self.y_distributions_clean['fields'])

        max_len = max(x_len, y_len, color_len, dot_size_len, opacity_len, line_len, line_order_by_len, x_distributions_len, y_distributions_len)

        self.x_clean             = self.__expandToMaxLen__(self.x_clean,             max_len)
        self.y_clean             = self.__expandToMaxLen__(self.y_clean,             max_len)
        self.color_clean         = self.__expandToMaxLen__(self.color_clean,         max_len)
        self.dot_size_clean      = self.__expandToMaxLen__(self.dot_size_clean,      max_len)
        self.opacity_clean       = self.__expandToMaxLen__(self.opacity_clean,       max_len)
        self.line_clean          = self.__expandToMaxLen__(self.line_clean,          max_len)
        self.line_order_by_clean = self.__expandToMaxLen__(self.line_order_by_clean, max_len)

        # If the field is just replicated (i.e., a single field), then we are just double/triple/x counting the distribution
        if self.x_distributions is not None and self.x_distributions_clean['fields'] is not None and len(self.x_distributions_clean['fields']) > 0:
            self.x_distributions_clean['fields'] = self.__expandToMaxLen__(self.x_distributions_clean['fields'], max_len)
        if self.y_distributions is not None and self.y_distributions_clean['fields'] is not None and len(self.y_distributions_clean['fields']) > 0:
            self.y_distributions_clean['fields'] = self.__expandToMaxLen__(self.y_distributions_clean['fields'], max_len)

    #
    # __flattenDataFrame__()
    #
    def __flattenDataFrame__(self):
        # Quads |  attribute-name  | column(s)                 | final-column-in-render | was-a-literal-specification | enums-found-in-the-specification
        _quads_ = [('x',             self.x_clean,             None,                    self.x_is_lits,               self.x_enums),
                   ('y',             self.y_clean,             None,                    self.y_is_lits,               self.y_enums),
                   ('color',         self.color_clean,         '__hexcolor__',          self.color_is_lits,           self.color_enums),
                   ('dot_size',      self.dot_size_clean,      '__radius__',            self.dot_size_is_lits,        self.dot_size_enums),
                   ('opacity',       self.opacity_clean,       '__fill_opacity__',      self.opacity_is_lits,         self.opacity_enums),
                   ('line_order_by', self.line_order_by_clean, None,                    self.line_order_by_is_lits,   self.line_order_by_enums)]

        # If the dataframe is passed into the class, flatten it (i.e., turn it into a series of dataframes with a common naming scheme)
        if self.df is not None:
            if '__p2s_index__' not in self.df.columns:
                self.df = self.df.with_row_index('__p2s_index__') # Add a row index column to reference the original dataframe
            _dfs_   = []
            for i in range(len(self.x_clean)):
                _ops_, _columns_forward_ = [], ['__p2s_index__'] # create the polars operations & the columns forward (keep the index around)
                # Base columns
                for _quad_ in _quads_:
                    _str_, _list_, _final_, _lits_flag_, _enums_ = _quad_
                    if _str_ == 'dot_size' and isinstance(self.dot_size_orig, int): continue # don't move forward with a radius for fixed integer plot
                    if _list_ is not None:
                        _obj_         = _list_[i]
                        _column_name_ = '__'+_str_+'__' # normalized column name moving forward
                        if   _lits_flag_:
                            _ops_.append(pl.lit(_obj_).alias(_final_))
                            _columns_forward_.append(_final_)
                        elif isinstance(_obj_, str):
                            _ops_.append(pl.col(_obj_).alias(_column_name_))
                            _columns_forward_.append(_column_name_)
                        elif isinstance(_obj_, tuple) and len(_obj_) == 1:
                            _ops_.append(pl.col(_obj_[0]).alias(_column_name_))
                            _columns_forward_.append(_column_name_)
                        else:
                            _column_list_ = [pl.col(_obj_[j] for j in range(len(_obj_)))]
                            _renames_     = [str(j)          for j in range(len(_obj_))]
                            _ops_.append(pl.struct(_column_list_).struct.rename_fields(_renames_).alias(_column_name_))
                            _columns_forward_.append(_column_name_)

                # Line columns
                if self.line_clean is not None:
                    _tuple_ = self.line_clean[i]
                    _cols_  = []
                    for j in range(len(_tuple_)):
                        if isinstance(_tuple_[j], str): _cols_.extend([_tuple_[j], pl.lit('|')])
                        else: break
                    _ops_.append(pl.concat_str(_cols_[:-1]).alias('__line__'))
                    _ops_.append(pl.lit(i).alias('__line_index__'))
                    _columns_forward_.extend(['__line__', '__line_index__'])

                # Distribution columns
                for _triple_ in [('__xdists__', '__xdists_color__', self.x_distributions_clean), 
                                 ('__ydists__', '__ydists_color__', self.y_distributions_clean)]:
                    _column_name_, _color_name_, _clean_ = _triple_
                    if   _clean_ is not None and len(_clean_['fields']) > 0 and len(_clean_['fields'][i]) == 2: # second part of field is a color (why it's 2 vs 1)
                        _ops_.append(pl.col(_clean_['fields'][i][0]).alias(_column_name_))
                        _ops_.append(pl.lit(_clean_['fields'][i][1]).alias(_color_name_))
                        _columns_forward_.extend([_column_name_, _color_name_])
                    elif _clean_ is not None and len(_clean_['fields']) > 0 and len(_clean_['fields'][i]) >  2: # last part is a color
                        _ops_.append(pl.struct([_clean_['fields'][i][:-1]]).alias(_column_name_))
                        _ops_.append(pl.lit(_clean_['fields'][i][-1]).alias(_color_name_))
                        _columns_forward_.extend([_column_name_, _color_name_])

                # Execute the operations & append to the flattened list
                # NaN/±inf coordinates are treated like nulls (dropped) -- they cannot be positioned
                _definitize_ = cs.float().replace({float('inf'): None, float('-inf'): None}).fill_nan(None)
                if self.use_lazy_execution: _df_ = self.df.lazy().with_columns(*_ops_).select(_columns_forward_).with_columns(_definitize_).drop_nulls().collect()
                else:                       _df_ = self.df.with_columns(*_ops_).select(_columns_forward_).with_columns(_definitize_).drop_nulls()
                _dfs_.append(_df_)
            # Produce the flattened frame
            self.df_flat = pl.concat(_dfs_)
        # Otherwise, set the df_flat (dataframe moving forward) to None
        else: self.df_flat = None

    #
    # __indexXandY_join__() - uses a sort (then a join) to index categorical values
    #
    def __indexXandY_join__(self):
        if self.df_flat is not None:

            _df_ = self.df_flat.lazy() if self.use_lazy_execution else self.df_flat

            for _quad_ in [('__x__', '__xi__', self.x_clean, self.x_enums, self.x_order),
                           ('__y__', '__yi__', self.y_clean, self.y_enums, self.y_order)]:
                _src_, _dst_, _field_, _enums_, _order_ = _quad_
                _dtype_                                 = self.df_flat.dtypes[self.df_flat.columns.index(_src_)]

                # Transform the _src_ column into the _dst_ column

                #
                # Categoricals (set-based) (or string/structs) are indexed and assigned a row index (by default)
                #
                if   self.p2s.SETp in _enums_       or \
                     isinstance(_dtype_, pl.String) or \
                     isinstance(_dtype_, pl.Struct):
                    
                    #
                    # No user specified order -- make one by sorting the values
                    #
                    if _order_ is None or len(_order_) == 0:
                        _dfi_ = _df_.select([_src_]).unique().sort(_src_).with_row_index(_dst_)
                        if self.use_lazy_execution: _dfi_ = _dfi_.lazy()
                        _df_  = _df_.join(_dfi_, on=_src_, how='left')
                    
                    #
                    # Order is specified as a list -- make that a one up
                    #
                    elif isinstance(_order_, list):
                        if isinstance(_order_[0], tuple):
                            # Create the order dataframe / the fields must match how they were constructed in the struct (see __flattenDataFrame__())
                            _fields_ = len(_order_[0])
                            _dict_   = {_dst_: [i for i in range(len(_order_))]} # one up value
                            for _field_ in range(_fields_):
                                _dict_[str(_field_)] = []
                                for _tuple_ in _order_: _dict_[str(_field_)].append(_tuple_[_field_])
                            _df_order_ = pl.DataFrame(_dict_)
                            if self.use_lazy_execution: _df_order_ = _df_order_.lazy()
                            # Join on positional struct-field columns "0", "1", ... (assumes the data carries no columns with those exact names)
                            _lefton_  = [pl.col(_src_).struct.field(str(i)) for i in range(_fields_)]
                            _righton_ = [str(i) for i in range(_fields_)]
                            _df_      = _df_.join(_df_order_, left_on=_lefton_, how='left', right_on=_righton_) \
                                            .with_columns(pl.col(_dst_).fill_null(len(_order_))) \
                                            .drop(_righton_)
                        else:
                            _df_order_ = pl.DataFrame({_src_: _order_, _dst_: range(len(_order_))})
                            if self.use_lazy_execution: _df_order_ = _df_order_.lazy()
                            _df_       = _df_.join(_df_order_, on=_src_, how='left') \
                                             .with_columns(pl.col(_dst_).fill_null(len(_order_)))

                    #
                    # Order is specified as a dictionary -- the key is the value, the value is the index
                    #
                    elif isinstance(_order_, dict):
                        _key_ = list(_order_.keys())[0]
                        if isinstance(_key_, tuple):
                            # Create the order dataframe / the fields must match how they were constructed in the struct (see __flattenDataFrame__())
                            _fields_ = len(_key_)
                            _dict_   = {_dst_:[]}
                            _max_    = max(_order_.values()) + 1 # values missing from the order dict share this one fallback slot
                            for _field_ in range(_fields_): _dict_[str(_field_)] = []
                            for k, v in _order_.items():
                                for _field_ in range(_fields_): _dict_[str(_field_)].append(k[_field_])
                                _dict_[_dst_].append(v)
                            _df_order_ = pl.DataFrame(_dict_)
                            if self.use_lazy_execution: _df_order_ = _df_order_.lazy()
                            # Join on positional struct-field columns "0", "1", ... (assumes the data carries no columns with those exact names)
                            _lefton_  = [pl.col(_src_).struct.field(str(i)) for i in range(_fields_)]
                            _righton_ = [str(i) for i in range(_fields_)]
                            _df_      = _df_.join(_df_order_, left_on=_lefton_, how='left', right_on=_righton_) \
                                            .with_columns(pl.col(_dst_).fill_null(_max_)) \
                                            .drop(_righton_)
                        else:
                            _dict_ = {_src_:[], _dst_:[]}
                            _max_  = max(_order_.values()) + 1 # values missing from the order dict share this one fallback slot
                            for k, v in _order_.items(): _dict_[_src_].append(k), _dict_[_dst_].append(v)
                            _df_order_ = pl.DataFrame(_dict_)
                            if self.use_lazy_execution: _df_order_ = _df_order_.lazy()
                            _df_       = _df_.join(_df_order_, on=_src_, how='left') \
                                             .with_columns(pl.col(_dst_).fill_null(_max_))

                    #
                    # Raise an error
                    #
                    else:
                        raise TypeError(f'XYp.__indexXandY__():  Type "{type(_order_)}" for column "{_src_}" is not supported')

                #
                # Numeric columns (if not set-based) are just copied over
                #
                elif self.p2s.numericColumn(self.df_flat, _src_):
                    _df_ = _df_.with_columns(pl.col(_src_).alias(_dst_))

                #
                # Convert datetime / date to their seconds since the epoch (as a floating point number)
                #
                elif isinstance(_dtype_, pl.Datetime) or \
                     isinstance(_dtype_, pl.Date):
                    _df_ = _df_.with_columns( (pl.col(_src_) - dt.datetime(1970, 1, 1)).dt.total_seconds(fractional=True).alias(_dst_) )
                else:
                    raise TypeError(f'XYp.__indexXandY__():  Type "{_dtype_}" for column "{_src_}" is not supported')

        # Put it back together
        self.df_flat = _df_.collect() if self.use_lazy_execution else _df_

    #
    # __constructGeometry__()
    #
    #
    # __legendPrepare__() - resolve legend kind/metadata and the strip to reserve
    # - runs at geometry time: the reserve must be known before pixel coordinates,
    #   but the colorbar domain is only filled in later (__renderDots__)
    # - Decision A: a truthy legend with nothing to legend (color=None / literal
    #   hex) silently reserves nothing
    #
    def __legendPrepare__(self):
        self.legend_info      = None
        self._legend_region_  = None
        self._legend_reserve_ = (0, 0, 0, 0)
        if self.legend_spec is None or self.df_flat is None or len(self.df_flat) == 0: return
        _mode_ = self.__determineColoringMode__()
        _kind_ = self.p2s.legendKind(_mode_)
        if _kind_ is None: return
        _spec_  = self.legend_spec
        _title_ = _spec_['title'] if _spec_['title'] is not None else self.__legendDefaultTitle__(_mode_)
        if _kind_ == 'categorical':
            _vc_ = self.p2s.legendCategoricalValueCounts(self.df_flat, '__color__')
            self.legend_info = self.p2s.legendInfoCategorical(_spec_, _vc_, _title_)
        else:
            self.legend_info = self.p2s.legendInfoColorbar(_title_)
        self._legend_color_mode_ = _mode_
        _reserve_ = self.p2s.legendReserve(_spec_, self.legend_info, self.txt_h, self.wxh)
        _l_, _r_, _t_, _b_ = _reserve_
        if self.wxh[0] - (_l_ + _r_) < 48 or self.wxh[1] - (_t_ + _b_) < 48:
            self.p2s.logger.warning(f'XYp.__legendPrepare__(): not enough space for legend (wxh = {self.wxh}); legend dropped')
            self.legend_info = None
            return
        self._legend_reserve_ = _reserve_
        _pos_ = _spec_['pos']
        if   _pos_ == 'right':  self._legend_region_ = (self.wxh[0] - _r_, 0, _r_, self.wxh[1])
        elif _pos_ == 'left':   self._legend_region_ = (0, 0, _l_, self.wxh[1])
        elif _pos_ == 'top':    self._legend_region_ = (0, 0, self.wxh[0], _t_)
        else:                   self._legend_region_ = (0, self.wxh[1] - _b_, self.wxh[0], _b_)

    #
    # __legendDefaultTitle__() - default legend title from the color spec
    #
    def __legendDefaultTitle__(self, _mode_):
        if _mode_ in (self.p2s.CROW_MAGNITUDEp, self.p2s.CROW_STRETCHEDp): return 'rows'
        if self.color_clean:
            _first_ = self.color_clean[0]
            if isinstance(_first_, str): return _first_
            if isinstance(_first_, tuple):
                _strs_ = [_ for _ in _first_ if isinstance(_, str)]
                if _strs_: return '|'.join(_strs_)
        return ''

    def __constructGeometry__(self):
        w,     h     = self.wxh
        # Legend strip (if any) comes out of wxh first -- the plot region shrinks,
        # the physical output size does not ("reserve from wxh").
        self.__legendPrepare__()
        _leg_l_, _leg_r_, _leg_t_, _leg_b_ = self._legend_reserve_
        w -= (_leg_l_ + _leg_r_)
        h -= (_leg_t_ + _leg_b_)
        x_ins, y_ins = self.insets
        if self.draw_context: w_context, h_context = self.txt_h, self.txt_h
        else:                 w_context, h_context = 0,          0

        # Reserve space for distributions (if they are outside)
        h_distributions = 0
        if self.x_distributions is not None:
            if self.p2s.DISTRIBUTION_OUTSIDEp in self.x_distributions_clean['enums']:
                h_distributions = int(self.x_distributions_clean['h_percs'][0] * (h - 2*y_ins - h_context)/2.0)
        w_distributions = 0
        if self.y_distributions is not None:
            if self.p2s.DISTRIBUTION_OUTSIDEp in self.y_distributions_clean['enums']:
                w_distributions = int(self.y_distributions_clean['h_percs'][0] * (w - 2*x_ins - w_context)/2.0)
        
        # Determine if there's enough space for distributions & context & minimums
        _adj_w_ = w - 2*x_ins - w_context - w_distributions
        _adj_h  = h - 2*y_ins - h_context - h_distributions
        if w_distributions > 0 and _adj_w_ < 64:
            self.p2s.logger.warning(f'XYp.__constructGeometry__(): Not enough space for distributions (w = {_adj_w_})')
            w_distributions = 0
        if h_distributions > 0 and _adj_h  < 64:
            self.p2s.logger.warning(f'XYp.__constructGeometry__(): Not enough space for distributions (h = {_adj_h})')
            h_distributions = 0
        # Recalculate the adjusted widths & heights
        _adj_w_ = w - 2*x_ins - w_context - w_distributions
        _adj_h  = h - 2*y_ins - h_context - h_distributions
        if   _adj_w_ < 32: x_ins, w_context = 0, 0
        elif _adj_w_ < 64:        w_context =    0
        if   _adj_h  < 32: y_ins, h_context = 0, 0
        elif _adj_h  < 64:        h_context =    0

        # These are the values used in the polars operations w/in the dataframe
        plot_size_w = w - 2*x_ins - w_context - w_distributions
        plot_size_h = h - 2*y_ins - h_context - h_distributions

        # if we are using integer dot sizes, make sure the plot size is divisible by the dot size
        if   isinstance(self.dot_size_orig, int):
            plot_size_w = plot_size_w - (plot_size_w % self.dot_size_orig)
            plot_size_h = plot_size_h - (plot_size_h % self.dot_size_orig)
            self.plot_size       = (plot_size_w, plot_size_h)
            self.plot_origin     = (_leg_l_ + x_ins + w_context, _leg_t_ + y_ins + self.plot_size[1] + h_distributions)
        else:
            self.plot_size       = (plot_size_w+1, plot_size_h+1)
            self.plot_origin     = (_leg_l_ + x_ins + w_context, _leg_t_ + y_ins + plot_size_h + h_distributions)

        # Save off the distribution information (these require adjustment in the pixel calculation for floats)
        if self.x_distributions is not None:
            if self.p2s.DISTRIBUTION_OUTSIDEp in self.x_distributions_clean['enums']:
                if h_distributions > 0: self.x_distributions_clean['h'] = h_distributions
                else:                   self.x_distributions_clean['h'] = 0
                self.x_distributions_clean['base'] = y_ins + h_distributions
            else:
                self.x_distributions_clean['h']    = self.plot_size[1] * self.x_distributions_clean['h_percs'][0]
                self.x_distributions_clean['base'] = self.plot_origin[1]
            self.x_distributions_clean['sign']   = -1
            self.x_distributions_clean['u_base'] = self.plot_origin[0]
            self.x_distributions_clean['u_dist'] = self.plot_size[0]
            self.x_distributions_clean['u_sign'] = 1
        if self.y_distributions is not None:
            if self.p2s.DISTRIBUTION_OUTSIDEp in self.y_distributions_clean['enums']:
                if w_distributions > 0: self.y_distributions_clean['h'] = w_distributions
                else:                   self.y_distributions_clean['h'] = 0
                self.y_distributions_clean['base'] = _leg_l_ + w - x_ins - w_distributions
            else:
                self.y_distributions_clean['h']    = self.plot_size[0] * self.y_distributions_clean['h_percs'][0]
                self.y_distributions_clean['base'] = self.plot_origin[0]
            self.y_distributions_clean['sign']   = 1
            self.y_distributions_clean['u_base'] = self.plot_origin[1]
            self.y_distributions_clean['u_dist'] = self.plot_size[1]
            self.y_distributions_clean['u_sign'] = -1

        self.__context_geom__ = (w_context, x_ins, h_context, y_ins) # saved to avoid recalculation (and code redundancy)

    #
    # __toPixelCoordinates_int__() - compute the x and y pixel coordinates
    #
    def __toPixelCoordinates_int__(self):
        # Pull the mins/maxes from the dataframe (or the user-specified parameter)
        if self.x_range is None: _xmin_, _xmax_ = self.df_flat['__xi__'].min(), self.df_flat['__xi__'].max()
        else:                    _xmin_, _xmax_ = self.x_range
        if self.y_range is None: _ymin_, _ymax_ = self.df_flat['__yi__'].min(), self.df_flat['__yi__'].max()
        else:                    _ymin_, _ymax_ = self.y_range
        # Handle any time conversions here
        if isinstance(_xmin_, datetime) or isinstance(_xmin_, date):
            _xmin_, _xmax_ = (_xmin_ - datetime(1970, 1, 1)).total_seconds(), (_xmax_ - datetime(1970, 1, 1)).total_seconds()
        if isinstance(_ymin_, datetime) or isinstance(_ymin_, date):
            _ymin_, _ymax_ = (_ymin_ - datetime(1970, 1, 1)).total_seconds(), (_ymax_ - datetime(1970, 1, 1)).total_seconds()
        _dx_ = 1 if _xmax_ is None or _xmin_ is None else _xmax_ - _xmin_
        _dy_ = 1 if _ymax_ is None or _ymin_ is None else _ymax_ - _ymin_
        if abs(_dx_) < 0.0001: _dx_ = 1
        if abs(_dy_) < 0.0001: _dy_ = 1
        # Filter out-of-range rows when a range is set -- mirrors __toPixelCoordinates_float__().
        # Without this the integer path would only .clip() out-of-range points, smearing them
        # onto the plot edges instead of removing them.
        if   self.x_range is not None and self.y_range is not None: self.df_flat = self.df_flat.filter((pl.col('__xi__') >= _xmin_) & (pl.col('__xi__') <= _xmax_) &
                                                                                                       (pl.col('__yi__') >= _ymin_) & (pl.col('__yi__') <= _ymax_))
        elif self.x_range is None     and self.y_range is not None: self.df_flat = self.df_flat.filter((pl.col('__yi__') >= _ymin_) & (pl.col('__yi__') <= _ymax_))
        elif self.x_range is not None and self.y_range is None:     self.df_flat = self.df_flat.filter((pl.col('__xi__') >= _xmin_) & (pl.col('__xi__') <= _xmax_))
        # Clamp normalized values to [0, 0.9999999...] before scaling, then snap to
        # the raster grid.  Without supersampling the snap step is the whole dot_size
        # (points land on integer multiples: 0, N, 2N, ...).  With dot_size_supersample=s
        # (s > 1) the step becomes dot_size/s, so points fall at fractional multiples of
        # the cell (e.g. 2x -> 0, 0.5, 1, 1.5, ... in cell units) for finer positional
        # resolution while the rects are still drawn dot_size wide.
        _ss_ = getattr(self, 'dot_size_supersample', 1) or 1
        _xnorm_ = (((pl.col('__xi__') - _xmin_)/_dx_).clip(0, 0.99999999) * self.plot_size[0])
        _ynorm_ = (((pl.col('__yi__') - _ymin_)/_dy_).clip(0, 0.99999999) * self.plot_size[1])
        if _ss_ <= 1:
            # Byte-identical to the original integer raster path (integer __xpx__/__ypx__).
            _xsnap_ = (_xnorm_.cast(pl.Int32) // self.dot_size_orig) * self.dot_size_orig
            _ysnap_ = (_ynorm_.cast(pl.Int32) // self.dot_size_orig) * self.dot_size_orig
        else:
            # Snap to the finer step, but cap the top-left corner at plot_size - dot_size
            # so a dot_size-wide rect at the far (max) edge still fits inside the plot box
            # -- the same edge invariant the ss=1 raster holds (its last full cell starts
            # at plot_size - dot_size).  Only the top (dot_size - step) sub-band collapses.
            _step_  = self.dot_size_orig / _ss_
            _xsnap_ = ((_xnorm_ / _step_).floor() * _step_).clip(0, self.plot_size[0] - self.dot_size_orig)
            _ysnap_ = ((_ynorm_ / _step_).floor() * _step_).clip(0, self.plot_size[1] - self.dot_size_orig)
        self.df_flat = self.df_flat.with_columns(
            (self.plot_origin[0] +                      _xsnap_).alias('__xpx__'),
            (self.plot_origin[1] - self.dot_size_orig - _ysnap_).alias('__ypx__')
        )
        # Compute both the forward & backward transforms (same format as __toPixelCoordinates_float__)
        self.x_transform_vars = (self.plot_origin[0], _xmin_, _dx_, self.plot_size[0])
        self.y_transform_vars = (self.plot_origin[1], _ymin_, _dy_, self.plot_size[1])

    #
    # __toPixelCoordinates_float__() - compute the x and y pixel coordinates
    #
    def __toPixelCoordinates_float__(self):
        # Pull the mins/maxes from the dataframe (or the user-specified parameter)
        if self.x_range is None: _xmin_, _xmax_ = self.df_flat['__xi__'].min(), self.df_flat['__xi__'].max()
        else:                    _xmin_, _xmax_ = self.x_range
        if self.y_range is None: _ymin_, _ymax_ = self.df_flat['__yi__'].min(), self.df_flat['__yi__'].max()
        else:                    _ymin_, _ymax_ = self.y_range
        # Handle any time conversions here
        if isinstance(_xmin_, datetime) or isinstance(_xmin_, date):
            _xmin_, _xmax_ = (_xmin_ - datetime(1970, 1, 1)).total_seconds(), (_xmax_ - datetime(1970, 1, 1)).total_seconds()
        if isinstance(_ymin_, datetime) or isinstance(_ymin_, date):
            _ymin_, _ymax_ = (_ymin_ - datetime(1970, 1, 1)).total_seconds(), (_ymax_ - datetime(1970, 1, 1)).total_seconds()
        # Ensure that the range is at least 1
        _dx_, _dy_ = _xmax_ - _xmin_, _ymax_ - _ymin_
        if abs(_dx_) < 0.0001: _dx_ = 1
        if abs(_dy_) < 0.0001: _dy_ = 1
        # Filter if one or both of the range values are set
        if   self.x_range is not None and self.y_range is not None: self.df_flat = self.df_flat.filter((pl.col('__xi__') >= _xmin_) & (pl.col('__xi__') <= _xmax_) & 
                                                                                                       (pl.col('__yi__') >= _ymin_) & (pl.col('__yi__') <= _ymax_))
        elif self.x_range is None     and self.y_range is not None: self.df_flat = self.df_flat.filter((pl.col('__yi__') >= _ymin_) & (pl.col('__yi__') <= _ymax_))
        elif self.x_range is not None and self.y_range is None:     self.df_flat = self.df_flat.filter((pl.col('__xi__') >= _xmin_) & (pl.col('__xi__') <= _xmax_))
        # Clamp normalized values to [0, 0.9999999...] before scaling
        self.df_flat = self.df_flat.with_columns(
            (self.plot_origin[0] + (((pl.col('__xi__') - _xmin_)/_dx_).clip(0, 0.99999999) * self.plot_size[0]).round().cast(pl.Int32)).alias('__xpx__'),
            (self.plot_origin[1] - (((pl.col('__yi__') - _ymin_)/_dy_).clip(0, 0.99999999) * self.plot_size[1]).round().cast(pl.Int32)).alias('__ypx__')
        )
        # Compute both the forward & backward transforms
        self.x_transform_vars = (self.plot_origin[0], _xmin_, _dx_, self.plot_size[0])
        self.y_transform_vars = (self.plot_origin[1], _ymin_, _dy_, self.plot_size[1])

    #
    # __shapelyToSVGPath__() - convert a Shapely geometry to an SVG path description string
    # - background= is the only feature that needs shapely, so the import is
    #   deferred here rather than paid by every xyp render.
    #
    def __shapelyToSVGPath__(self, shape):
        try:
            from shapely.geometry import Polygon, MultiPolygon, LineString, MultiLineString, GeometryCollection
        except ImportError as _e_:
            raise ImportError(
                "background= shapes require the optional 'layouts' dependency (shapely). "
                "Install it with:\n"
                "    pip install polars2svg[layouts]"
            ) from _e_
        if isinstance(shape, MultiPolygon):
            parts = []
            for subpoly in shape.geoms:
                parts.append(self.__shapelyToSVGPath__(subpoly))
            return ' '.join(parts)
        elif isinstance(shape, MultiLineString):
            parts = []
            for subline in shape.geoms:
                parts.append(self.__shapelyToSVGPath__(subline))
            return ' '.join(parts)
        elif isinstance(shape, LineString):
            coords = shape.coords
            path_str = f'M {coords[0][0]} {coords[0][1]}'
            for i in range(1, len(coords)):
                path_str += f' L {coords[i][0]} {coords[i][1]}'
            return path_str
        elif isinstance(shape, Polygon):
            xx, yy = shape.exterior.coords.xy
            path_str = f'M {xx[0]} {yy[0]}'
            for i in range(1, len(xx)):
                path_str += f' L {xx[i]} {yy[i]}'
            for interior in shape.interiors:
                ix, iy = interior.coords.xy
                path_str += f' M {ix[0]} {iy[0]}'
                for i in range(1, len(ix)):
                    path_str += f' L {ix[i]} {iy[i]}'
            return path_str + ' Z'
        elif isinstance(shape, GeometryCollection):
            if len(shape.geoms) > 0:
                raise DataError('XYp.__shapelyToSVGPath__() - non-empty GeometryCollection not supported')
            return None
        else:
            raise DataError(f'XYp.__shapelyToSVGPath__() - unsupported type: {type(shape)}')

    #
    # __bgMinsAndMaxes__() - update a bounding box with a new point
    #
    def __bgMinsAndMaxes__(self, x, y, x0, y0, x1, y1):
        if x0 is None:
            return x, y, x, y
        return min(x, x0), min(y, y0), max(x, x1), max(y, y1)

    #
    # __backgroundShapeRenderDetails__() - build SVG fill/stroke attribute string for a background shape
    #
    def __backgroundShapeRenderDetails__(self, name, bg_shape_opacity, bg_shape_fill, bg_shape_stroke_w, bg_shape_stroke):
        svg = ''
        # Fill
        if bg_shape_fill is not None and bg_shape_opacity is not None:
            if isinstance(bg_shape_opacity, dict):
                _opacity = bg_shape_opacity.get(name, 1.0)
            else:
                _opacity = bg_shape_opacity
            svg += f' fill-opacity="{_opacity}"'

            if   isinstance(bg_shape_fill, dict) and name in bg_shape_fill:
                _co = bg_shape_fill[name]
            elif bg_shape_fill == 'vary':
                _co = self.p2s.color(name)
            elif isinstance(bg_shape_fill, self.p2s.HexColorString):
                _co = bg_shape_fill
            else:
                _co = self.p2s.colorTyped('axis', 'inner')
            svg += f' fill="{_co}"'
        else:
            svg += ' fill-opacity="0.0"'

        # Stroke
        if bg_shape_stroke_w is not None and bg_shape_stroke is not None:
            if   bg_shape_stroke == 'vary':
                _co = self.p2s.color(name)
            elif isinstance(bg_shape_stroke, self.p2s.HexColorString):
                _co = bg_shape_stroke
            elif isinstance(bg_shape_stroke, dict) and name in bg_shape_stroke:
                _co = bg_shape_stroke[name]
            else:
                _co = self.p2s.colorTyped('axis', 'inner')

            if isinstance(bg_shape_stroke_w, dict) and name in bg_shape_stroke_w:
                _wi = bg_shape_stroke_w[name]
            else:
                _wi = bg_shape_stroke_w
            svg += f' stroke="{_co}" stroke-width="{_wi}"'
        return svg

    #
    # __backgroundShapeLabel__() - render a centred text label over the shape bounding box
    #
    def __backgroundShapeLabel__(self, name, x0, y0, x1, y1, bg_shape_label_color):
        if bg_shape_label_color is None or x0 is None:
            return ''
        if   isinstance(bg_shape_label_color, dict) and name in bg_shape_label_color:
            _co = bg_shape_label_color[name]
        elif bg_shape_label_color == 'vary':
            _co = self.p2s.color(name)
        elif isinstance(bg_shape_label_color, self.p2s.HexColorString):
            _co = bg_shape_label_color
        else:
            _co = self.p2s.colorTyped('axis', 'inner')
        _cx_ = (x0 + x1) / 2
        _cy_ = self.txt_h / 2 + (y0 + y1) / 2
        return (f'<text x="{_cx_}" y="{_cy_}" text-anchor="middle" '
                f'font-family="{self.p2s.default_font}" fill="{_co}" font-size="{self.txt_h}px">'
                f'{name}</text>')

    #
    # __transformCircleSVG__() - transform a <circle> SVG element into an <ellipse> in screen coordinates
    #
    def __transformCircleSVG__(self, name, shape_desc, bg_shape_label_color, bg_shape_opacity,
                                bg_shape_fill, bg_shape_stroke_w, bg_shape_stroke):
        _root_ = ET.fromstring(shape_desc)  # nosec B314 - trusted caller config (background= shape descriptor), not untrusted data; see SECURITY.md
        cx  = float(_root_.attrib['cx'])
        cy  = float(_root_.attrib['cy'])
        r   = float(_root_.attrib['r'])
        cx_s = self.wxToSx(cx)
        cy_s = self.wyToSy(cy)
        rx_s = abs(self.wxToSx(r + cx) - cx_s)
        ry_s = abs(self.wyToSy(r + cy) - cy_s)
        svg  = f'<ellipse cx="{cx_s}" cy="{cy_s}" rx="{rx_s}" ry="{ry_s}"'
        svg += self.__backgroundShapeRenderDetails__(name, bg_shape_opacity, bg_shape_fill,
                                                     bg_shape_stroke_w, bg_shape_stroke)
        return svg + '/>', self.__backgroundShapeLabel__(name, cx_s - rx_s, cy_s - ry_s,
                                                         cx_s + rx_s, cy_s + ry_s, bg_shape_label_color)

    #
    # __transformPathDescription__() - transform an SVG path description string into screen coordinates
    #
    def __transformPathDescription__(self, name, shape_desc, bg_shape_label_color, bg_shape_opacity,
                                      bg_shape_fill, bg_shape_stroke_w, bg_shape_stroke):
        svg = '<path d="'
        x0, y0, x1, y1 = None, None, None, None
        tokens = ' '.join(shape_desc.split()).split(' ')
        i = 0
        while i < len(tokens):
            if tokens[i] == 'M':
                _x, _y = self.wxToSx(float(tokens[i+1])), self.wyToSy(float(tokens[i+2]))
                svg += f' M {_x} {_y}'
                x0, y0, x1, y1 = self.__bgMinsAndMaxes__(_x, _y, x0, y0, x1, y1)
                i += 3
            elif tokens[i] == 'L':
                _x, _y = self.wxToSx(float(tokens[i+1])), self.wyToSy(float(tokens[i+2]))
                svg += f' L {_x} {_y}'
                x0, y0, x1, y1 = self.__bgMinsAndMaxes__(_x, _y, x0, y0, x1, y1)
                i += 3
            elif tokens[i] == 'C':
                _xcp1, _ycp1 = self.wxToSx(float(tokens[i+1])), self.wyToSy(float(tokens[i+2]))
                _xcp2, _ycp2 = self.wxToSx(float(tokens[i+3])), self.wyToSy(float(tokens[i+4]))
                _x,    _y    = self.wxToSx(float(tokens[i+5])), self.wyToSy(float(tokens[i+6]))
                svg += f' C {_xcp1} {_ycp1} {_xcp2} {_ycp2} {_x} {_y}'
                x0, y0, x1, y1 = self.__bgMinsAndMaxes__(_x,    _y,    x0, y0, x1, y1)
                x0, y0, x1, y1 = self.__bgMinsAndMaxes__(_xcp1, _ycp1, x0, y0, x1, y1)
                x0, y0, x1, y1 = self.__bgMinsAndMaxes__(_xcp2, _ycp2, x0, y0, x1, y1)
                i += 7
            elif tokens[i] == 'Z':
                svg += ' Z'
                i += 1
            else:
                raise Polars2SVGError(f'XYp.__transformPathDescription__() - unhandled path token "{tokens[i]}"')
        svg += '"'
        svg += self.__backgroundShapeRenderDetails__(name, bg_shape_opacity, bg_shape_fill,
                                                     bg_shape_stroke_w, bg_shape_stroke)
        return svg + '/>', self.__backgroundShapeLabel__(name, x0, y0, x1, y1, bg_shape_label_color)

    #
    # __transformPointsList__() - transform a list of (x, y) tuples into a screen-coordinate SVG path
    #
    def __transformPointsList__(self, name, points_list, bg_shape_label_color, bg_shape_opacity,
                                 bg_shape_fill, bg_shape_stroke_w, bg_shape_stroke):
        _x, _y = self.wxToSx(points_list[0][0]), self.wyToSy(points_list[0][1])
        svg = f'<path d="M {_x} {_y}'
        x0, y0, x1, y1 = _x, _y, _x, _y
        for i in range(1, len(points_list)):
            _x, _y = self.wxToSx(points_list[i][0]), self.wyToSy(points_list[i][1])
            svg += f' L {_x} {_y}'
            x0, y0, x1, y1 = self.__bgMinsAndMaxes__(_x, _y, x0, y0, x1, y1)
        svg += ' Z"'
        svg += self.__backgroundShapeRenderDetails__(name, bg_shape_opacity, bg_shape_fill,
                                                     bg_shape_stroke_w, bg_shape_stroke)
        return svg + '/>', self.__backgroundShapeLabel__(name, x0, y0, x1, y1, bg_shape_label_color)

    #
    # __transformBackgroundShapes__() - dispatch a background shape to the appropriate transform method
    #
    def __transformBackgroundShapes__(self, name, shape_desc, bg_shape_label_color, bg_shape_opacity,
                                       bg_shape_fill, bg_shape_stroke_w, bg_shape_stroke):
        # Convert Shapely geometries to SVG path strings. shapely is an optional
        # 'layouts' dependency: if it isn't installed, shape_desc can't actually
        # be a shapely geometry (the caller couldn't have constructed one), so
        # skip these isinstance checks rather than importing/erroring for a
        # plain string/svg background.
        try:
            from shapely.geometry import Polygon, MultiPolygon, LineString, MultiLineString, GeometryCollection
            _has_shapely_ = True
        except ImportError:
            _has_shapely_ = False

        if _has_shapely_ and isinstance(shape_desc, (Polygon, MultiPolygon)):
            shape_desc = self.__shapelyToSVGPath__(shape_desc)
        if _has_shapely_ and isinstance(shape_desc, (LineString, MultiLineString)):
            shape_desc = self.__shapelyToSVGPath__(shape_desc)
            bg_shape_fill = 'none'
        if _has_shapely_ and isinstance(shape_desc, GeometryCollection):
            if len(shape_desc.geoms) > 0:
                raise DataError('XYp.__transformBackgroundShapes__() - non-empty GeometryCollection not supported')
            return '', ''

        if isinstance(shape_desc, str):
            if shape_desc.lower().startswith('<circle'):
                return self.__transformCircleSVG__(name, shape_desc, bg_shape_label_color, bg_shape_opacity,
                                                   bg_shape_fill, bg_shape_stroke_w, bg_shape_stroke)
            else:
                return self.__transformPathDescription__(name, shape_desc, bg_shape_label_color, bg_shape_opacity,
                                                         bg_shape_fill, bg_shape_stroke_w, bg_shape_stroke)
        elif isinstance(shape_desc, list):
            return self.__transformPointsList__(name, shape_desc, bg_shape_label_color, bg_shape_opacity,
                                                bg_shape_fill, bg_shape_stroke_w, bg_shape_stroke)
        else:
            raise DataError(f'XYp.__transformBackgroundShapes__() - unsupported type "{type(shape_desc)}"')

    #
    # __renderBackground__() - render background shapes into self.svg_background
    #
    def __renderBackground__(self):
        self.svg_background  = ''
        self._dl_background_ = DisplayList(self.wxh[0], self.wxh[1])
        if self.background is None:
            return
        _shapes_, _labels_ = [], []
        for name, shape_desc in self.background.items():
            _s, _l = self.__transformBackgroundShapes__(
                name, shape_desc,
                self.background_label_color,
                self.background_opacity,
                self.background_fill,
                self.background_stroke_w,
                self.background_stroke)
            _shapes_.append(_s)
            _labels_.append(_l)
            self.__backgroundShapeToDL__(_s, self._dl_background_)
        for _l in _labels_:
            self.__backgroundLabelToDL__(_l, self._dl_background_)
        self.svg_background = ''.join(_shapes_) + ''.join(_labels_)

    #
    # __backgroundShapeToDL__() - GPU geometry for a generated background shape string
    # (ellipse or M/L/C/Z path in screen coordinates); fills are ear-clipped, strokes
    # become line segments, cubic beziers are flattened
    #
    def __backgroundShapeToDL__(self, _s_, dl):
        import re
        if not _s_: return
        def _attr_(name, default=None):
            _m_ = re.search(f'{name}="([^"]*)"', _s_)
            return _m_.group(1) if _m_ else default
        _fill_         = _attr_('fill')
        _fill_opacity_ = float(_attr_('fill-opacity', '1.0'))
        _stroke_       = _attr_('stroke')
        _stroke_w_     = float(_attr_('stroke-width', '1.0'))
        if _s_.startswith('<ellipse'):
            cx, cy = float(_attr_('cx')), float(_attr_('cy'))
            rx, ry = float(_attr_('rx')), float(_attr_('ry'))
            import math as _math_
            _pts_ = [(cx + rx*_math_.cos(2*_math_.pi*i/48), cy + ry*_math_.sin(2*_math_.pi*i/48)) for i in range(48)]
            _subpaths_ = [(_pts_, True)]
        elif _s_.startswith('<path'):
            _d_ = _attr_('d', '')
            _tokens_ = _d_.split()
            _subpaths_, _cur_, _closed_ = [], [], False
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
                        for k in range(1, 17):
                            t = k / 16.0
                            mt = 1.0 - t
                            _cur_.append((mt*mt*mt*_p0_[0] + 3*mt*mt*t*_p1_[0] + 3*mt*t*t*_p2_[0] + t*t*t*_p3_[0],
                                          mt*mt*mt*_p0_[1] + 3*mt*mt*t*_p1_[1] + 3*mt*t*t*_p2_[1] + t*t*t*_p3_[1]))
                    i += 7
                elif _t_ == 'Z':
                    if len(_cur_) > 1: _subpaths_.append((_cur_, True))
                    _cur_ = []
                    i += 1
                else:
                    i += 1  # unknown token -- skip (svg path stays authoritative)
            if len(_cur_) > 1: _subpaths_.append((_cur_, False))
        else:
            return
        for _pts_, _closed_ in _subpaths_:
            if _fill_ is not None and _fill_ != 'none' and _fill_opacity_ > 0.0 and _closed_ and len(_pts_) >= 3:
                dl.polygon(_pts_, _fill_, opacity=_fill_opacity_)
            if _stroke_ is not None and _stroke_ != 'none':
                _seq_ = _pts_ + [_pts_[0]] if _closed_ else _pts_
                for j in range(len(_seq_) - 1):
                    dl.line(_seq_[j][0], _seq_[j][1], _seq_[j+1][0], _seq_[j+1][1], _stroke_, width=_stroke_w_)

    #
    # __backgroundLabelToDL__() - GPU glyphs for a generated background label string
    #
    def __backgroundLabelToDL__(self, _l_, dl):
        import re
        if not _l_: return
        _m_ = re.search(r'<text x="([^"]*)" y="([^"]*)" text-anchor="middle"[^>]*fill="([^"]*)" font-size="([^"]*)px">([^<]*)</text>', _l_)
        if _m_ is None: return
        _x_, _y_, _co_, _th_, _txt_ = float(_m_.group(1)), float(_m_.group(2)), _m_.group(3), float(_m_.group(4)), _m_.group(5)
        dl.text(self.p2s, _txt_, _x_, _y_, txt_h=_th_, anchor='middle', color=_co_, svg='')

    #
    # world to screen conversions
    #
    def wxToSx(self, wx):
        _xorigin_, _xmin_, _dx_, _width_ = self.x_transform_vars
        return _xorigin_ + ((wx - _xmin_)/_dx_) * _width_
    def wyToSy(self, wy):
        _yorigin_, _ymin_, _dy_, _height_ = self.y_transform_vars
        return _yorigin_ - ((wy - _ymin_)/_dy_) * _height_

    #
    # screen to world conversions
    #
    def sxToWx(self, sx):
        _xorigin_, _xmin_, _dx_, _width_ = self.x_transform_vars
        return _xmin_ + ((sx - _xorigin_)/_width_) * _dx_
    def syToWy(self, sy):
        _yorigin_, _ymin_, _dy_, _height_ = self.y_transform_vars
        return _dy_ * (_yorigin_ - sy)/ _height_ + _ymin_

    #
    # __distributeElements__()
    #
    def __distributeElements__(self):
        for _pair_ in [('x', self.x_distributions_clean), ('y', self.y_distributions_clean)]:
            _axis_, _clean_     = _pair_
            _field_             = f'__{_axis_}i__'                   # original name in df_flat ... doesn't survive into the df_distributions
            _bin_field_         = f'__{_axis_}i_bin__'               # the bin (which will be separated into the _min_field_ and the _max_field_)
            _totals_field_      = f'__{_axis_}i_total__'             # the total for a particular cut
            _value_field_       = f'__{_axis_}dists__'               # field to total up (as part of the group_by)
            _color_field_       = f'__{_axis_}dists_color__'         # color for this specific bar
            _min_field_         = f'__{_axis_}dists_{_axis_}i_min__' # the first coordinate (0.0 to 1.0) of the bar
            _max_field_         = f'__{_axis_}dists_{_axis_}i_max__' # the second coordinate (0.0 to 1.0) of the bar
            _totals_min_field_  = f'__{_axis_}i_total_min__'         # the min to compare the total to
            _totals_max_field_  = f'__{_axis_}i_total_max__'         # the max to compare the total to

            if _clean_ is None: continue
            if len(self.df_flat) == 0: continue # occurs when user-specified range does not include any of the data

            # Do the binning
            if   _axis_ == 'x' and self.x_range is not None: _min_, _max_ = self.x_range
            elif _axis_ == 'y' and self.y_range is not None: _min_, _max_ = self.y_range
            else:                                            _min_, _max_ = self.df_flat[_field_].min(), self.df_flat[_field_].max()

            # A periodic time transform (month-of-year, hour-of-day, day-of-week, ...) plots
            # as discrete integers, so -- like Timep's periodic bins (one bar per period unit,
            # see Timep.__constructGeometry__) -- autobin should place exactly one bar per
            # integer value spanned by the data instead of a pixel-derived count, which would
            # otherwise alias several units into a bar or leave empty gaps between them.
            _axis_clean_        = self.x_clean if _axis_ == 'x' else self.y_clean
            _axis_periodic_     = (self.__axisIsPeriodicTime__(_axis_clean_)
                                   and _min_ is not None and _max_ is not None)

            # Determine the number of bins
            if self.p2s.DISTRIBUTION_AUTOBINp in _clean_['enums']:
                if _axis_periodic_:
                    _min_, _max_  = int(_min_), int(_max_)
                    _num_of_bins_ = _max_ - _min_ + 1        # one bar per discrete period unit
                else:
                    _dim_           = self.plot_size[0] if _axis_ == 'x' else self.plot_size[1]
                    if isinstance(self.dot_size_orig, int):
                        if    self.dot_size_orig > 3: _num_of_bins_ = _dim_//self.dot_size_orig
                        else: _num_of_bins_ = _dim_//6
                    else: _num_of_bins_   = _dim_//6
                _clean_['bins'] = [_num_of_bins_]
            else:
                _num_of_bins_ = _clean_['bins'][0]

            # Calculate bin width
            if   _max_ is None or _min_ is None:                                              bin_width = 1
            elif _axis_periodic_ and self.p2s.DISTRIBUTION_AUTOBINp in _clean_['enums']:      bin_width = 1  # integer-aligned periodic bins
            else:                                                                             bin_width = (_max_ - _min_) / _num_of_bins_

            # Then apply the bin width as an operation
            _bin_ops_ = []
            if bin_width == 0:  # all values identical — everything goes into bin 0
                _bin_ops_.append(pl.when(pl.col(_field_).is_null())
                                   .then(pl.lit(None))
                                   .otherwise(pl.lit(0))
                                   .cast(pl.Int64).alias(_bin_field_))
            else:
                _bin_ops_.append(pl.when(pl.col(_field_).is_null())
                                   .then(pl.lit(None))
                                   .otherwise(((pl.col(_field_) - _min_) / bin_width).floor().clip(0, _num_of_bins_ - 1))
                                   .cast(pl.Int64).alias(_bin_field_))
            
            # Do the group_by
            if _color_field_ in self.df_flat.columns: _gb_str_ = [_bin_field_, _color_field_]
            else:                                     _gb_str_ = [_bin_field_]

            # Do the aggregation
            _agg_ops_ = []
            if   self.p2s.ROW_COUNTp in _clean_['enums']: _agg_ops_.append(pl.len().alias(_totals_field_))
            elif self.p2s.SCALARp    in _clean_['enums']: _agg_ops_.append(pl.col(_value_field_).sum().alias(_totals_field_))
            else:                                         _agg_ops_.append(pl.col(_value_field_).unique().len().alias(_totals_field_))

            # Select only the fields we need for the execution
            _required_fields_ = [_field_]
            if self.p2s.ROW_COUNTp not in _clean_['enums']: _required_fields_.append(_value_field_)
            if _color_field_ in self.df_flat.columns: _required_fields_.append(_color_field_)

            # Execute the initial operation to perform the cut and sum the correct field / attribute
            if self.use_lazy_execution: _df_ = self.df_flat.lazy().select(_required_fields_).with_columns(_bin_ops_).group_by(_gb_str_).agg(_agg_ops_).collect()
            else:                       _df_ = self.df_flat       .select(_required_fields_).with_columns(_bin_ops_).group_by(_gb_str_).agg(_agg_ops_)

            # Create the labels ... an artifact of how this was originally done with the polars cut operation
            _labels_     = [i for i in range(_num_of_bins_)]
            _labels_min_ = [i / _num_of_bins_ for i in range(_num_of_bins_)]
            _labels_max_ = [(i + 1) / _num_of_bins_ for i in range(_num_of_bins_)]
            
            # Create an all-bins dataframe that will be used to create entries for missing bins
            _all_bins_df_ = pl.DataFrame({_bin_field_:_labels_, _min_field_:_labels_min_, _max_field_:_labels_max_})
            if _color_field_ in _df_.columns: _all_bins_df_ = _all_bins_df_.join(pl.DataFrame({_color_field_:list(set(_df_[_color_field_]))}), how='cross')

            # Fill in the missing bins
            _df_ = _all_bins_df_.join(_df_, on=_gb_str_, how='left').with_columns(pl.col(_totals_field_).fill_null(0.0))

            # Assign a default color if none exists
            if _color_field_ not in self.df_flat.columns:
                _df_ = _df_.with_columns(pl.lit(self.p2s.colorTyped('distributions', 'default')).alias(_color_field_))

            # Create the minimum column
            if   self.p2s.DISTRIBUTION_COLOR_MIN_TO_COLOR_MAX in _clean_['enums']:
                _df_ = _df_.with_columns(pl.col(_totals_field_).min().over(_color_field_).alias(_totals_min_field_),
                                         pl.col(_totals_field_).max().over(_color_field_).alias(_totals_max_field_))
            elif self.p2s.DISTRIBUTION_ZERO_TO_COLOR_MAX      in _clean_['enums']:
                _df_ = _df_.with_columns(pl.lit(0.0).alias(_totals_min_field_),
                                         pl.col(_totals_field_).max().over(_color_field_).alias(_totals_max_field_))
            elif self.p2s.DISTRIBUTION_ALL_MIN_TO_ALL_MAX     in _clean_['enums']:
                _df_ = _df_.with_columns(pl.col(_totals_field_).min().alias(_totals_min_field_),
                                         pl.col(_totals_field_).max().alias(_totals_max_field_))
            elif self.p2s.DISTRIBUTION_ZERO_TO_ALL_MAX        in _clean_['enums']:
                _df_ = _df_.with_columns(pl.lit(0.0).alias(_totals_min_field_),
                                         pl.col(_totals_field_).max().alias(_totals_max_field_))

            # Assign it to the correct dataframe
            if _axis_ == 'x': self.df_x_distribution = _df_
            else:             self.df_y_distribution = _df_

    #
    # __humanReadableMinAndMax_{type}__()
    #
    def __humanReadableMinAndMax_int__(self, _min_, _max_):
        _min_unit_ = self.p2s.unitizeInt(_min_)
        _max_unit_ = self.p2s.unitizeInt(_max_)
        if _min_unit_ == _max_unit_: return self.p2s.unitizeInt_extendUntilDifferent(_min_, _max_)
        else:                        return _min_unit_, _max_unit_

    def __humanReadableMinAndMax_float__(self, _min_, _max_):
        if abs(_min_) > 1.0 or abs(_max_) > 1.0: return self.__humanReadableMinAndMax_int__(_min_, _max_)
        return f'{_min_:.4}', f'{_max_:.4}'

    #
    # __humanReadableMinAndMax__()
    #
    def __humanReadableMinAndMax__(self, _min_, _max_, clean_axis):
        if   self.__axisIsPeriodicTime__(clean_axis):
            _tfield_ = clean_axis[0]
            if isinstance(_tfield_, tuple): _tfield_ = _tfield_[0]
            _column_, _enum_ = self.p2s.tFieldTuple(_tfield_)
            return self.p2s.timePeriodicHumanReadable(_min_, _enum_), self.p2s.timePeriodicHumanReadable(_max_, _enum_)
        elif isinstance(_min_, int)   and isinstance(_max_, int):    return self.__humanReadableMinAndMax_int__  (_min_, _max_)
        elif isinstance(_min_, float) and isinstance(_max_, float):  return self.__humanReadableMinAndMax_float__(_min_, _max_)
        return str(_min_), str(_max_)

    #
    # __axisIsPeriodicTime__() - determine if an x/y axis is periodic
    # - only works with single columns ... i.e., no tuples (or rather, only tuples with one value)
    #
    def __axisIsPeriodicTime__(self, _clean_):
        _first_ = _clean_[0]
        if isinstance(_first_, tuple):
            _strs_ = [_ for _ in _first_ if isinstance(_, str)]
            if len(_strs_) != 1: return False
            _first_ = _strs_[0]
        elif isinstance(_first_, str) == False: return False
        if self.p2s.isTField(_first_, df=self.df_orig) == False: return False
        return isinstance(self.p2s.tFieldTuple(_first_)[1], self.p2s.TimePeriodicTypeP)

    #
    # __formatLabels__()
    #
    def __formatLabels__(self, axis, cell_min, cell_max, sz, clean_axis, buffer=30):
        # Figure out what the axis is called ... there's a special case for time fields
        if isinstance(cell_min, datetime) and isinstance(cell_max, datetime):
            _timedelta_ = cell_max - cell_min
            _center_ = self.p2s.humanReadableTimeDelta(_timedelta_)
        elif self.__axisIsPeriodicTime__(clean_axis):
            _tfield_ = clean_axis[0]
            if isinstance(_tfield_, tuple): _tfield_ = _tfield_[0]
            _column_, _enum_ = self.p2s.tFieldTuple(_tfield_)
            _center_ = self.p2s.humanReadablePeriodicTimeDelta(cell_max-cell_min, _enum_)
        else:
            _columns_ = self.__getattribute__(axis)
            if   isinstance(_columns_, str):  _center_ = _columns_
            elif isinstance(_columns_, list):
                for i in range(len(_columns_)):
                    if isinstance(_columns_[i], str):
                        _center_ = _columns_[i]
                        break
                    elif isinstance(_columns_[i], tuple):
                        _strs_ = []
                        for j in range(len(_columns_[i])):
                            if isinstance(_columns_[i][j], str): _strs_.append(_columns_[i][j])
                        _center_ = '|'.join(_strs_)
                        break
            elif isinstance(_columns_, tuple):
                _strs_ = []
                for i in range(len(_columns_)):
                    if isinstance(_columns_[i], str): _strs_.append(_columns_[i])
                _center_ = '|'.join(_strs_)
            else: raise TypeError(f'XYp.__formatLabels__(): Unrecognized type for {axis} ({_columns_})')
        # Convert left and right into strings
        _left_, _right_  = self.__humanReadableMinAndMax__(cell_min, cell_max, clean_axis)
        # Get the center length -- see if it fills up most of the space... if so, then it's just the center
        c_len   = self.p2s.textLength(_center_, self.txt_h)
        if c_len + 1*buffer >= sz: 
            if c_len > sz-2*buffer: _center_ = self.p2s.cropText(_center_, self.txt_h, sz-1*buffer)
            return '', _center_, ''
        # Otherwise, determine the length of the left and right strings
        l_len       = self.p2s.textLength(_left_,   self.txt_h)
        if l_len <= 0.01: l_len = 1
        r_len       = self.p2s.textLength(_right_,  self.txt_h)
        if r_len <= 0.01: r_len = 1
        _left_over_ = sz - (c_len + 2*buffer)
        # allocate _left_over_ into l_len_mod and r_len_mod based on their ratios
        l_len_mod = _left_over_ * l_len / (l_len + r_len)
        r_len_mod = _left_over_ * r_len / (l_len + r_len)
        l_len_mod, r_len_mod
        return self.p2s.cropText(_left_,  self.txt_h, l_len_mod), _center_, self.p2s.cropText(_right_, self.txt_h, r_len_mod)

    #
    # __determineColoringMode__()
    #
    def __determineColoringMode__(self):
        # Check if any of the color enums are in self.color_enums
        # ... and then make sure the column supports that mode
        for x in self.p2s.ColorTypeP:
            if x in self.color_enums:
                if x == self.p2s.CMAGNITUDE_SUMp    or x == self.p2s.CSTRETCHED_SUMp    or \
                   x == self.p2s.CMAGNITUDE_MINp    or x == self.p2s.CSTRETCHED_MINp    or \
                   x == self.p2s.CMAGNITUDE_MEDIANp or x == self.p2s.CSTRETCHED_MEDIANp or \
                   x == self.p2s.CMAGNITUDE_MEANp   or x == self.p2s.CSTRETCHED_MEANp   or \
                   x == self.p2s.CMAGNITUDE_MAXp    or x == self.p2s.CSTRETCHED_MAXp:
                    if self.p2s.numericColumn(self.df_flat, '__color__'): 
                        return x
                    else:
                        _dtype_ = self.df_flat.dtypes[self.df_flat.columns.index('__color__')]
                        self.p2s.logger.warning(f'XYp.__determineColoringMode__(): {x} requires a numeric column to work ({_dtype_})')
                else: return x
        # No enumerations were found -- see if there's a color column in df_flat and its type
        if '__color__' in self.df_flat.columns:
            if self.p2s.numericColumn(self.df_flat, '__color__'): return self.p2s.CMAGNITUDE_SUMp
            else:                                                 return self.p2s.CSETp
        return None

    #
    # __renderContext_set__()
    #
    def __renderContext_set__(self, plot_origin, plot_wxh, x_axis=True, pixel_goal=30, dl=None):
        _dim_           = plot_wxh[0] if x_axis else plot_wxh[1]
        xo, yo          = plot_origin
        xw, yh          = plot_wxh
        _color_inner_   = self.p2s.colorTyped('axis', 'inner')
        _svg_           = []
        # Render a line in the correct orientation
        def __line__(_screen_, _label_, _color_=_color_inner_, _width_=0.4):
            if _screen_ is None: return
            # Calculate the endpoints (and ensure that they within the plot itself)
            if x_axis:  x1, y1, x2, y2 = _screen_, yo-yh,   _screen_, yo
            else:       x1, y1, x2, y2 = xo,      _screen_, xo+xw,    _screen_
            # Ensure the number doesn't have a bunch of digits
            x1, y1, x2, y2 = round(x1,1), round(y1,1), round(x2,1), round(y2,1)
            # Draw the line
            _svg_.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{_color_}" stroke-width="{_width_}" />')
            if dl is not None: dl.line(x1, y1, x2, y2, _color_, width=_width_)
            # Draw the label
            _rot_ = 90 if x_axis else None
            _txt_ = self.p2s.svgText(f'{_label_}', x1, y1, color=self.p2s.colorTyped('axis', 'inner'), txt_h=self.txt_h*0.6, rotation=_rot_)
            _svg_.append(_txt_)
            if dl is not None: dl.text(self.p2s, f'{_label_}', x1, y1, color=self.p2s.colorTyped('axis', 'inner'), txt_h=self.txt_h*0.6, rotation=_rot_, svg='')
        # Get the column names
        if    x_axis: px_col, axis_col = '__xpx__', '__x__'
        else:         px_col, axis_col = '__ypx__', '__y__'

        # Organize the data by the most rows -- this will be the priority for rendering
        _df_ = self.df_flat.group_by([px_col,axis_col]).len().sort('len', descending=True)

        # Limit the number of labels based on the pixel goal and the dimension
        _max_labels_to_render_ = _dim_ // pixel_goal
        _labels_rendered_      = 0

        # Loop through the data
        _filled_ = [False for i in range(_dim_)]
        for i in range(len(_df_)):
            screen_coordinate = _df_[px_col][i]
            label             = _df_[axis_col][i]
            # is it clear to put the label here?
            _clear_           = True
            for j in range(self.txt_h):
                k = screen_coordinate + j
                if k >= 0 and k < _dim_ and _filled_[k]: 
                    _clear_ = False
                    break
            if not _clear_: continue
            # update the clear array
            for j in range(self.txt_h):
                k = screen_coordinate + j
                if k >= 0 and k < _dim_: _filled_[k] = True
            # Render the label
            __line__(screen_coordinate, label)
            _labels_rendered_ += 1
            if _labels_rendered_ >= _max_labels_to_render_ or i > 20: break
        return _svg_

    #
    # __renderContext_numeric__()
    #
    def __renderContext_numeric__(self, _min_, _max_, plot_origin, plot_wxh, x_axis=True, pixel_goal=80, dl=None):
        if _min_ is None or _max_ is None: return [] # happens if there's no data (because user-specified axis range)
        _dim_           = plot_wxh[0] if x_axis else plot_wxh[1]
        n_ticks         = _dim_ // pixel_goal
        i_nice          = self.p2s.heckbertNiceNumbers(n_ticks, _min_, _max_)
        i_nice_decimals = abs((Decimal(str(i_nice))).as_tuple().exponent)
        xo, yo          = plot_origin
        xw, yh          = plot_wxh
        _color_origin_  = self.p2s.colorTyped('axis', 'origin')
        _color_inner_   = self.p2s.colorTyped('axis', 'inner')
        _svg_           = []
        # Render a line in the correct orientation
        def __line__(_offset_, _world_, _color_=_color_inner_, _width_=0.4):
            if _offset_ is None: return
            # Calculate the endpoints (and ensure that they within the plot itself)
            if x_axis:
                x1, y1, x2, y2 = xo+_offset_, yo-yh,       xo+_offset_, yo
                if _offset_ < 0 or _offset_ > xw: return # outside of the plot
            else:
                x1, y1, x2, y2 = xo,          yo-_offset_, xo+xw,       yo-_offset_
                if _offset_ < 0 or _offset_ > yh: return # outside of the plot
            # Ensure the number doesn't have a bunch of digits
            x1, y1, x2, y2 = round(x1,1), round(y1,1), round(x2,1), round(y2,1)
            # Draw the line
            _svg_.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{_color_}" stroke-width="{_width_}" />')
            if dl is not None: dl.line(x1, y1, x2, y2, _color_, width=_width_)
            # Draw the label
            _rot_ = 90 if x_axis else None
            _svg_.append(self.p2s.svgText(f'{_world_}', x1, y1, color=self.p2s.colorTyped('axis', 'inner'), txt_h=self.txt_h*0.6, rotation=_rot_))
            if dl is not None: dl.text(self.p2s, f'{_world_}', x1, y1, color=self.p2s.colorTyped('axis', 'inner'), txt_h=self.txt_h*0.6, rotation=_rot_, svg='')
        # Convert to screen coordinates
        def __toScreen__(_world_):
            if abs(_max_ - _min_) < 1e-6: return None
            else:                         return _dim_ * (_world_ - _min_) / (_max_ - _min_)
        # If the origin is in the range, center on that
        if _min_ < 0.0 and _max_ > 0.0:
            # Origin
            __line__(__toScreen__(0.0), 0.0, _color_origin_, 0.6)
            # Parts above the origin
            i       = i_nice
            while i < _max_:
                __line__(__toScreen__(i), round(i, i_nice_decimals))
                i += i_nice
            # Parts below the origin
            i = -i_nice
            while i > _min_:
                __line__(__toScreen__(i), round(i, i_nice_decimals))
                i -= i_nice
        # Else, find a good starting point & go from there
        else:
            i = self.p2s.heckbertFirstGridLine(i_nice, _min_, _max_)
            while i < _max_:
                __line__(__toScreen__(i), round(i, i_nice_decimals))
                i += i_nice
        
        return _svg_

    #
    # __renderContext_periodicTime__()
    #
    def __renderContext_periodicTime__(self, _clean_, _min_world_, _max_world_, plot_origin, plot_wxh, x_axis=True, pixel_goal=10, dl=None):
        _dim_          = plot_wxh[0] if x_axis else plot_wxh[1]
        xo, yo         = plot_origin
        xw, yh         = plot_wxh
        _svg_          = []
        # Convert to screen coordinates
        def __toScreen__(_world_):
            if abs(_max_world_ - _min_world_) < 1e-6: return None
            else:                                     return _dim_ * (_world_ - _min_world_) / (_max_world_ - _min_world_)
        # Distance between two world coordinates in pixels
        def __distanceBetweenLines__(_line1_, _line2_): return abs(__toScreen__(_line1_) - __toScreen__(_line2_))
        # Add a description
        def __addDescription__(_desc_, _num_, _out_of_): _svg_.append(f'<!-- xyp.__renderContext_periodicTime__(): {_desc_}|{_num_}|{_out_of_} -->')
        # Render a line in the correct orientation
        def __line__(_world_, _str_, _type_):
            if _world_ < _min_world_ or _world_ > _max_world_: return
            if   _type_ == 'major':   _color_, _width_, _txt_h_ = self.p2s.colorTyped('axis', 'origin'), 0.4, round(self.txt_h*0.8,1)
            elif _type_ == 'minor':   _color_, _width_, _txt_h_ = self.p2s.colorTyped('axis', 'inner'),  0.4, round(self.txt_h*0.6,1)
            elif _type_ == 'tick':    _color_, _width_, _txt_h_ = self.p2s.colorTyped('axis', 'inner'),  0.8, round(self.txt_h*0.5,1)
            elif _type_ == 'subtick': _color_, _width_, _txt_h_ = self.p2s.colorTyped('axis', 'inner'),  0.8, round(self.txt_h*0.4,1)
            # Determine the coordinates of the grid line
            _offset_ = __toScreen__(_world_)
            if _type_ == 'major' or _type_ == 'minor':
                if x_axis: x1, y1, x2, y2 = xo+_offset_,  yo-yh,        xo+_offset_,       yo
                else:      x1, y1, x2, y2 = xo,           yo-_offset_,  xo+xw,             yo-_offset_
            elif _type_ == 'tick':
                if x_axis: x1, y1, x2, y2 = xo+_offset_,  yo-yh,        xo+_offset_,       yo-yh + self.txt_h*0.8
                else:      x1, y1, x2, y2 = xo,           yo-_offset_,  xo+self.txt_h*0.8, yo-_offset_
            elif _type_ == 'subtick':
                if x_axis: x1, y1, x2, y2 = xo+_offset_,  yo-yh,        xo+_offset_,       yo-yh + self.txt_h*0.4
                else:      x1, y1, x2, y2 = xo,           yo-_offset_,  xo+self.txt_h*0.4, yo-_offset_
            x1, y1, x2, y2 = round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)
            # Create the grid line
            _line_svg_ = f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{_color_}" stroke-width="{_width_}" />'
            if dl is not None: dl.line(x1, y1, x2, y2, _color_, width=_width_)
            # Draw the label (if applicable)
            if x_axis: x_label, y_label, _rotation_ = x1,                  y1 + self.txt_h/4.0, 90
            else:      x_label, y_label, _rotation_ = x1 + self.txt_h/4.0, y1,                  0
            _text_svg_ = self.p2s.svgText(_str_, x_label, y_label, txt_h=_txt_h_, color=_color_, rotation=_rotation_) if _str_ is not None else ''
            if dl is not None and _str_ is not None:
                dl.text(self.p2s, _str_, x_label, y_label, txt_h=_txt_h_, color=_color_, rotation=_rotation_, svg='')
            _svg_.append(_line_svg_ + _text_svg_)
        # Get the t-field & extract the enum
        _tfield_ = _clean_[0]
        if isinstance(_tfield_, tuple): _tfield_ = _tfield_[0]
        _column_, _enum_ = self.p2s.tFieldTuple(_tfield_)
        #
        # Render
        #
        if   _enum_ == self.p2s.PT_Qp:
            if __distanceBetweenLines__(0, 1) >= pixel_goal:
                __addDescription__(_enum_, 1, 1)
                for i in range(1, 5): __line__(i, self.p2s.timePeriodicHumanReadable(i, _enum_), 'major')
        elif _enum_ == self.p2s.PT_mp:
            if __distanceBetweenLines__(0, 1) >= pixel_goal:
                __addDescription__(_enum_, 1, 1)
                for i in range(1,13): __line__(i, self.p2s.timePeriodicHumanReadable(i, _enum_), 'major')
        elif _enum_ == self.p2s.PT_m_dp:
            _month_lu_    = self.p2s.__monthLookup__()
            _max_days_lu_ = self.p2s.__maxDaysInMonthLookup__()
            if   __distanceBetweenLines__(0, 1)  >= pixel_goal:
                __addDescription__(_enum_, 1, 3)
                _day_i_       = 1
                for _month_ in range(1,13):
                    __line__(_day_i_, _month_lu_[_month_], 'major')
                    for i in range(0, _max_days_lu_[_month_]):
                        if    (i+1)     == 15: continue
                        elif ((i+1)% 5) ==  0: __line__(_day_i_+i, None, 'tick')
                        else:                  __line__(_day_i_+i, None, 'subtick')
                    __line__(_day_i_+14, '15', 'minor')
                    _day_i_ += _max_days_lu_[_month_]
            elif __distanceBetweenLines__(1, 5)  >= pixel_goal:
                __addDescription__(_enum_, 2, 3)
                _day_i_       = 1
                for _month_ in range(1,13):
                    __line__(_day_i_, _month_lu_[_month_], 'major')
                    __line__(_day_i_+ 4, None, 'tick')
                    __line__(_day_i_+ 9, None, 'tick')
                    __line__(_day_i_+14, '15', 'minor')
                    __line__(_day_i_+19, None, 'tick')
                    __line__(_day_i_+24, None, 'tick')
                    _day_i_ += _max_days_lu_[_month_]
            elif __distanceBetweenLines__(1, 15) >= pixel_goal:
                __addDescription__(_enum_, 3, 3)
                _day_i_       = 1
                for _month_ in range(1,13):
                    __line__(_day_i_, _month_lu_[_month_], 'major')
                    __line__(_day_i_+14, '15', 'minor')
                    _day_i_ += _max_days_lu_[_month_]
        #
        # Month Day Hour
        #
        elif _enum_ == self.p2s.PT_m_d_Hp:
            _month_lu_    = self.p2s.__monthLookup__()
            _max_days_lu_ = self.p2s.__maxDaysInMonthLookup__()
            if   __distanceBetweenLines__(0, 24) >= pixel_goal:
                __addDescription__(_enum_, 1, 3)
                _day_i_       = 1
                for _month_ in range(1,13):
                    __line__(_day_i_*24, _month_lu_[_month_], 'major')
                    for i in range(0, _max_days_lu_[_month_]):
                        if (i+1)%5 == 0: __line__((_day_i_+i)*24, str(i+1), 'minor')
                        else:            __line__((_day_i_+i)*24, None,     'tick')
                    _day_i_ += _max_days_lu_[_month_]
            elif __distanceBetweenLines__(0, 24*10) >= pixel_goal:
                __addDescription__(_enum_, 2, 3)
                _day_i_       = 1
                for _month_ in range(1,13):
                    __line__(_day_i_*24, _month_lu_[_month_], 'major')
                    __line__((_day_i_+14)*24, '15', 'minor')
                    _day_i_ += _max_days_lu_[_month_]
            elif __distanceBetweenLines__(0, 24*30) >= pixel_goal:
                __addDescription__(_enum_, 3, 3)
                _day_i_       = 1
                for _month_ in range(1,13):
                    __line__(_day_i_*24, _month_lu_[_month_], 'minor')
                    _day_i_ += _max_days_lu_[_month_]
        #
        # Day of Year
        #
        elif _enum_ == self.p2s.PT_DoYp:
            if   __distanceBetweenLines__(0, 5) >= pixel_goal:
                __addDescription__(_enum_, 1, 3)
                for i in range(1, 367):
                    if   i    == 1: __line__(1, str(1), 'major')
                    elif i%50 == 0: __line__(i, str(i), 'major')
                    elif i%25 == 0: __line__(i, str(i), 'tick')
                    elif i%5  == 0: __line__(i, None,   'subtick')
            elif __distanceBetweenLines__(0, 10) >= pixel_goal:
                __addDescription__(_enum_, 2, 3)
                for i in range(1,367):
                    if i == 1 or i%100 == 0: __line__(i, str(i), 'major')
                    elif i%50 == 0:          __line__(i, str(i), 'tick')
                    elif i%10 == 0:          __line__(i, None,   'subtick')
            else:
                __addDescription__(_enum_, 3, 3)
                for i in range(1, 367):
                    if   i     == 1: __line__(1, str(1), 'minor')
                    elif i%100 == 0: __line__(i, str(i), 'minor')

        #
        # Day of Week
        #
        elif _enum_ == self.p2s.PT_DoWp:
            if __distanceBetweenLines__(0, 1) >= pixel_goal:
                __addDescription__(_enum_, 1, 1)
                for i in range(1, 8):  __line__(i,       self.p2s.timePeriodicHumanReadable(i,       _enum_), 'major')
        #
        # Day of Week Hour
        #
        elif _enum_ == self.p2s.PT_DoW_Hp:
            if   __distanceBetweenLines__(0,  1) >= pixel_goal:
                __addDescription__(_enum_, 1, 3)
                for _day_ in range(1,8):
                    __line__(_day_*24,    self.p2s.timePeriodicHumanReadable(_day_*24, _enum_).split(' ')[0], 'major')
                    for _hour_ in range(1,24):
                        if   (_hour_%12) == 0: __line__(_day_*24 + _hour_, '12', 'minor')
                        elif (_hour_% 6) == 0: __line__(_day_*24 + _hour_, None, 'tick')
                        else:                  __line__(_day_*24 + _hour_, None, 'subtick')
            elif __distanceBetweenLines__(0,  6) >= pixel_goal:
                __addDescription__(_enum_, 2, 3)
                for _day_ in range(1,8):
                    __line__(_day_*24, self.p2s.timePeriodicHumanReadable(_day_*24, _enum_).split(' ')[0], 'major')
                    __line__(_day_*24 +  6, None, 'tick')
                    __line__(_day_*24 + 12, '12', 'minor')
                    __line__(_day_*24 + 18, None, 'tick')
            elif __distanceBetweenLines__(0, 12) >= pixel_goal:
                __addDescription__(_enum_, 3, 3)
                for _day_ in range(1,8):
                    __line__(_day_*24, self.p2s.timePeriodicHumanReadable(_day_*24, _enum_).split(' ')[0], 'major')
                    __line__(_day_*24 + 12, '12', 'minor')
        #
        # Day of Week Hour Minute
        #
        elif _enum_ == self.p2s.PT_DoW_H_Mp:
            if   __distanceBetweenLines__(0, 5)     >= pixel_goal: # Every 5 minutes
                __addDescription__(_enum_, 1, 3)
                for _day_i_ in range(1, 8):
                    __line__(_day_i_*24*60, self.p2s.timePeriodicHumanReadable(_day_i_*24*60, _enum_).split(' ')[0], 'major')
                    for _hour_i_ in range(0,24):
                        __line__(_day_i_*24*60 + _hour_i_*60, str(_hour_i_), 'minor')
                        for _min_i_ in range(0,60,5):
                            if   _min_i_    == 0: continue
                            elif _min_i_%15 == 0: __line__(_day_i_*24*60 + _hour_i_*60 + _min_i_, str(_min_i_), 'tick')
                            else:                 __line__(_day_i_*24*60 + _hour_i_*60 + _min_i_, None,         'subtick')
            elif __distanceBetweenLines__(0, 60)    >= pixel_goal: # Every hour
                __addDescription__(_enum_, 2, 3)
                for _day_i_ in range(1, 8):
                    __line__(_day_i_*24*60, self.p2s.timePeriodicHumanReadable(_day_i_*24*60, _enum_).split(' ')[0], 'major')
                    for _hour_i_ in range(0,24):
                        __line__(_day_i_*24*60 + _hour_i_*60, str(_hour_i_), 'minor')
            elif __distanceBetweenLines__(0, 60*6)  >= pixel_goal: # Every 6 hours
                __addDescription__(_enum_, 3, 3)
                for _day_i_ in range(1, 8):
                    __line__(_day_i_*24*60, self.p2s.timePeriodicHumanReadable(_day_i_*24*60, _enum_).split(' ')[0], 'major')
                    for _hour_i_ in range(0, 24, 6):
                        __line__(_day_i_*24*60 + _hour_i_*60, str(_hour_i_), 'minor')
        #
        # Day of Month
        #
        elif _enum_ == self.p2s.PT_dp:
            if   __distanceBetweenLines__(0,1) >= pixel_goal:
                __addDescription__(_enum_, 1, 2)
                for i in range(1, 32): __line__(i,  str(i), 'major')
            elif __distanceBetweenLines__(0,5) >= pixel_goal:
                __addDescription__(_enum_, 2, 2)
                __line__(1, '1', 'major')
                for i in range(0, 32, 5): __line__(i, str(i), 'major')
        #
        # Day of Month Hour
        #
        elif _enum_ == self.p2s.PT_d_Hp:
            if   __distanceBetweenLines__(0,1)     >= pixel_goal:
                __addDescription__(_enum_, 1, 4)
                for _day_i_ in range(1, 32): 
                    __line__(_day_i_*24, str(_day_i_), 'major')
                    for _hour_i_ in range(24):
                        if   _hour_i_    == 0: continue
                        elif _hour_i_%12 == 0: __line__(_day_i_*24 + _hour_i_, '12', 'minor')
                        elif _hour_i_% 6 == 0: __line__(_day_i_*24 + _hour_i_, None, 'tick')
                        else:                  __line__(_day_i_*24 + _hour_i_, None, 'subtick')
            elif __distanceBetweenLines__(0,12)    >= pixel_goal:
                __addDescription__(_enum_, 2, 4)
                for _day_i_ in range(1, 32): 
                    __line__(_day_i_*24, str(_day_i_), 'major')
            elif __distanceBetweenLines__(0,24)    >= pixel_goal:
                __addDescription__(_enum_, 3, 4)
                for _day_i_ in range(1, 32):
                    if _day_i_ == 1 or _day_i_%5 == 0: __line__(_day_i_*24, str(_day_i_), 'major')
                    else:                              __line__(_day_i_*24, None,         'tick')
            elif __distanceBetweenLines__(0,5*24)  >= pixel_goal:
                __addDescription__(_enum_, 4, 4)
                for _day_i_ in range(1, 32, 1):
                    if   _day_i_    == 1: __line__(_day_i_*24, str(1),       'major')
                    elif _day_i_%10 == 0: __line__(_day_i_*24, str(_day_i_), 'major')
                    elif _day_i_%5  == 0: __line__(_day_i_*24, None,         'tick')
                    else:                 __line__(_day_i_*24, None,         'subtick')
        #
        # Day of Month Hour Minute
        #
        elif _enum_ == self.p2s.PT_d_H_Mp:
            if   __distanceBetweenLines__(0, 1)     >= pixel_goal:
                __addDescription__(_enum_, 1, 3)
                for _day_i_ in range(1, 32):
                    __line__(_day_i_*24*60, str(_day_i_), 'major')
                    for _hour_i_ in range(0, 24):
                        __line__(_day_i_*24*60 + _hour_i_*60, str(_hour_i_), 'minor')
                        for _minute_i_ in range(0, 60):
                            if _minute_i_%10 == 0: __line__(_day_i_*24*60 + _hour_i_*60 + _minute_i_, str(_minute_i_), 'tick')
                            else:                  __line__(_day_i_*24*60 + _hour_i_*60 + _minute_i_, None,            'subtick')
            elif __distanceBetweenLines__(0, 24)    >= pixel_goal:
                __addDescription__(_enum_, 2, 3)
                for _day_i_ in range(1, 32): 
                    __line__(_day_i_*24*60, str(_day_i_), 'major')
                    for _hour_i_ in range(0, 24):
                        if _hour_i_%6 == 0: __line__(_day_i_*24*60 + _hour_i_*60, str(_hour_i_), 'tick')
                        else:               __line__(_day_i_*24*60 + _hour_i_*60, None,          'subtick')
            elif __distanceBetweenLines__(0, 24*60) >= pixel_goal:
                __addDescription__(_enum_, 3, 3)
                for _day_i_ in range(1, 32): 
                    __line__(_day_i_*24*60, str(_day_i_), 'major')
        #
        # Hour
        #
        elif _enum_ == self.p2s.PT_Hp:
            if   __distanceBetweenLines__(0,  0.5) >= pixel_goal:
                __addDescription__(_enum_, 1, 2)
                for i in range(0, 24): __line__(i, str(i), 'major')
            else:
                __addDescription__(_enum_, 2, 2)
                for i in range(0, 24):
                    if   i%3 == 0: __line__(i, str(i), 'major')
                    else:          __line__(i, None,   'tick')
        #
        # Hour Minute
        #
        elif _enum_ == self.p2s.PT_H_Mp:
            if   __distanceBetweenLines__(0, 5)     >= pixel_goal:
                __addDescription__(_enum_, 1, 5)
                for _hour_i_ in range(0, 24):
                    __line__(_hour_i_*60,    str(_hour_i_), 'major')
                    for _minute_i_ in range(1, 60):
                        if   _minute_i_%15 == 0: __line__(_hour_i_*60+_minute_i_, str(30),       'minor')
                        elif _minute_i_%5  == 0: __line__(_hour_i_*60+_minute_i_, None,          'tick')
                        else:                    __line__(_hour_i_*60+_minute_i_, None,          'subtick')
            elif __distanceBetweenLines__(0, 20)   >= pixel_goal:
                __addDescription__(_enum_, 2, 5)
                for _hour_i_ in range(0, 24): 
                    __line__(_hour_i_*60,    str(_hour_i_), 'major')
                    __line__(_hour_i_*60+15, None,          'tick')
                    __line__(_hour_i_*60+30, str(30),       'minor')
                    __line__(_hour_i_*60+45, None,          'tick')
            elif __distanceBetweenLines__(0,   30) >= pixel_goal:
                __addDescription__(_enum_, 3, 5)
                for _hour_i_ in range(0, 24):
                    if   _hour_i_%3 == 0: __line__(_hour_i_*60, str(_hour_i_), 'major')
                    else:                 __line__(_hour_i_*60, None,          'tick')
                    __line__(_hour_i_*60+30, None, 'subtick')
            elif __distanceBetweenLines__(0,  40)  >= pixel_goal:
                __addDescription__(_enum_, 4, 5)
                for _hour_i_ in range(0, 24):
                    if   _hour_i_%6 == 0: __line__(_hour_i_*60, str(_hour_i_), 'major')
                    elif _hour_i_%3 == 0: __line__(_hour_i_*60, None,          'tick')
                    else:                 __line__(_hour_i_*60, None,          'subtick')
            else:
                __addDescription__(_enum_, 5, 5)
                for i in range(0, 24):
                    if   i%6 == 0: __line__(60*i, str(i), 'major')
                    else:          __line__(60*i, None,   'tick')
        #
        # Hour Minute Second
        #
        elif _enum_ == self.p2s.PT_H_M_Sp:
            if   __distanceBetweenLines__(0,      5*60) >= pixel_goal:
                __addDescription__(_enum_, 1, 5)
                for _hour_i_ in range(24):
                    __line__(_hour_i_*3600, str(_hour_i_), 'major')
                    for _minute_i_ in range(1, 60):
                        if   _minute_i_%15 == 0: __line__(_hour_i_*3600+_minute_i_*60, str(_minute_i_), 'minor')
                        elif _minute_i_%5  == 0: __line__(_hour_i_*3600+_minute_i_*60, None,            'tick')
                        else:                    __line__(_hour_i_*3600+_minute_i_*60, None,            'subtick')
            elif __distanceBetweenLines__(0,     15*60) >= pixel_goal:
                __addDescription__(_enum_, 2, 5)
                for _hour_i_ in range(24):
                    __line__(_hour_i_*3600, str(_hour_i_), 'major')
                    for _minute_i_ in range(1, 60):
                        if   _minute_i_%15 == 0: __line__(_hour_i_*3600+_minute_i_*60, str(_minute_i_), 'minor')
                        elif _minute_i_%5  == 0: __line__(_hour_i_*3600+_minute_i_*60, None,            'tick')
            elif __distanceBetweenLines__(0,     30*60) >= pixel_goal:
                __addDescription__(_enum_, 3, 5)
                for _hour_i_ in range(24):
                    if   _hour_i_%3  == 0: __line__(_hour_i_*3600, str(_hour_i_), 'major')
                    else:                  __line__(_hour_i_*3600, str(_hour_i_), 'minor')
            elif __distanceBetweenLines__(0,     45*60) >= pixel_goal:
                __addDescription__(_enum_, 4, 5)
                for _hour_i_ in range(24):
                    if   _hour_i_%6  == 0: __line__(_hour_i_*3600, str(_hour_i_), 'major')
                    elif _hour_i_%3 == 0:  __line__(_hour_i_*3600, str(_hour_i_), 'minor')
                    else:                  __line__(_hour_i_*3600, None,          'tick')
            elif __distanceBetweenLines__(0,     60*60) >= pixel_goal:
                __addDescription__(_enum_, 5, 5)
                for _hour_i_ in range(24):
                    if _hour_i_%12 == 0: __line__(_hour_i_*3600, str(_hour_i_), 'major')
                    if _hour_i_%3  == 0: __line__(_hour_i_*3600, None,          'tick')
                    else:                __line__(_hour_i_*3600, None,          'subtick')
        #
        # Minutes *OR* Seconds ... it's the same
        #
        elif _enum_ == self.p2s.PT_Mp or _enum_ == self.p2s.PT_Sp:
            if   __distanceBetweenLines__(0,  0.5) >= pixel_goal:
                __addDescription__(_enum_, 1, 4)
                for i in range(0, 60):
                    if   i%15 == 0: __line__(i, str(i), 'major')
                    elif i%5  == 0: __line__(i, str(i), 'minor')
                    else:           __line__(i, str(i), 'tick')
            elif __distanceBetweenLines__(0,  1) >= pixel_goal:
                __addDescription__(_enum_, 2, 4)
                for i in range(0, 60):
                    if   i%15 == 0: __line__(i, str(i), 'major')
                    elif i%5  == 0: __line__(i, str(i), 'minor')
                    else:           __line__(i, None,   'tick')
            elif __distanceBetweenLines__(0, 2) >= pixel_goal:
                __addDescription__(_enum_, 3, 4)
                for i in range(0, 60):
                    if   i%15 == 0: __line__(i, str(i), 'major')
                    elif i%5  == 0: __line__(i, str(i), 'tick')
                    else:           __line__(i, None,   'subtick')
            elif __distanceBetweenLines__(0, 3) >= pixel_goal:
                __addDescription__(_enum_, 4, 4)
                for i in range(0, 60, 15):
                    if   i%30 == 0: __line__(i, str(i), 'major')
                    else:           __line__(i, str(i), 'tick')
        #
        # Minutes Seconds
        #
        elif _enum_ == self.p2s.PT_M_Sp:
            if   __distanceBetweenLines__(0,  1)    >= pixel_goal:
                __addDescription__(_enum_, 1, 6)
                for _min_ in range(0, 60):
                    __line__(_min_*60, str(_min_)+':00', 'major')
                    for _sec_ in range(1, 60):
                        if _sec_%15 == 0: __line__(_min_*60 + _sec_, str(_sec_), 'minor')
                        else:             __line__(_min_*60 + _sec_, None,       'tick')
            elif __distanceBetweenLines__(0,  5)    >= pixel_goal:
                __addDescription__(_enum_, 2, 6)
                for _min_ in range(0, 60):
                    __line__(_min_*60, str(_min_), 'major')
                    for _sec_ in range(0, 60, 5):
                        if _sec_      == 0: continue
                        elif _sec_%15 == 0: __line__(_min_*60 + _sec_, str(_sec_), 'minor')
                        else:               __line__(_min_*60 + _sec_, None,       'tick')
            elif __distanceBetweenLines__(0, 15)    >= pixel_goal:
                __addDescription__(_enum_, 3, 6)
                for _min_ in range(0, 60):
                    __line__(_min_*60, str(_min_), 'major')
                    for _sec_ in range(0, 60, 15):
                        if _sec_      == 0: continue
                        else:               __line__(_min_*60 + _sec_, None,       'tick')
            elif __distanceBetweenLines__(0, 30)    >= pixel_goal:
                __addDescription__(_enum_, 4, 6)
                for _min_ in range(0, 60):
                    __line__(_min_*60, str(_min_), 'major')
                    __line__(_min_*60 + 30, '30', 'tick')
            elif __distanceBetweenLines__(0, 60)    >= pixel_goal:
                __addDescription__(_enum_, 5, 6)
                for _min_ in range(0, 60):
                    if _min_%5 == 0: __line__(_min_*60, str(_min_), 'major')
                    else:            __line__(_min_*60, None,       'tick')
            elif __distanceBetweenLines__(0, 60*5)  >= pixel_goal:
                __addDescription__(_enum_, 6, 6)
                for _min_ in range(0, 60, 5):
                    if _min_%15 == 0: __line__(_min_*60, str(_min_), 'major')
                    else:             __line__(_min_*60, None,       'tick')
        #
        # Else... raise an exception
        #
        else: raise ValueError(f'XYp.__renderContext_periodicTime__():  _enum_: {_enum_} is not valid')

        #
        # Join the lines to return the rendering
        #
        return ''.join(_svg_)

    #
    # __renderContext_linearTime__()
    #
    def __renderContext_linearTime__(self, dt_start, dt_end, plot_origin, plot_wxh, x_axis=True, pixel_goal=80, dl=None):
        _dim_ = plot_wxh[0] if x_axis else plot_wxh[1]
        _sec_total_         = (dt_end - dt_start).total_seconds()
        if _sec_total_ == 0: return []
        _num_of_grid_lines_ = {}
        _to_compute_ = {('1sec',  1,               '%H:%M:%S',     '%Y-%m-%d %H:%M:%S', timedelta(seconds=1)),
                        ('5sec',  5,               '%H:%M:%S',     '%Y-%m-%d %H:%M:%S', timedelta(seconds=5)),
                        ('10sec', 10,              '%H:%M:%S',     '%Y-%m-%d %H:%M:%S', timedelta(seconds=10)),
                        ('15sec', 15,              '%H:%M:%S',     '%Y-%m-%d %H:%M:%S', timedelta(seconds=15)),
                        ('30sec', 30,              '%H:%M:%S',     '%Y-%m-%d %H:%M:%S', timedelta(seconds=30)),  
                        ('45sec', 45,              '%H:%M:%S',     '%Y-%m-%d %H:%M:%S', timedelta(seconds=45)),
                        ('1min',  60*1,            '%H:%M',        '%Y-%m-%d %H:%M:00', timedelta(minutes=1)),
                        ('5min',  60*5,            '%H:%M',        '%Y-%m-%d %H:%M:00', timedelta(minutes=5)),
                        ('15min', 60*15,           '%H:%M',        '%Y-%m-%d %H:%M:00', timedelta(minutes=15)),
                        ('30min', 60*30,           '%H:%M',        '%Y-%m-%d %H:%M:00', timedelta(minutes=30)),
                        ('45min', 60*45,           '%H:%M',        '%Y-%m-%d %H:%M:00', timedelta(minutes=45)),
                        ('1hr',   60*60*1,         '%H:%M',        '%Y-%m-%d %H:00:00', timedelta(hours=1)),
                        ('2hr',   60*60*2,         '%H:%M',        '%Y-%m-%d %H:00:00', timedelta(hours=2)),
                        ('3hr',   60*60*3,         '%m-%d %H:%M',  '%Y-%m-%d %H:00:00', timedelta(hours=3)),
                        ('4hr',   60*60*4,         '%m-%d %H:%M',  '%Y-%m-%d %H:00:00', timedelta(hours=4)),
                        ('6hr',   60*60*6,         '%m-%d %H:%M',  '%Y-%m-%d %H:00:00', timedelta(hours=6)),
                        ('8hr',   60*60*8,         '%m-%d %H:%M',  '%Y-%m-%d %H:00:00', timedelta(hours=8)),
                        ('1day',  60*60*24*1,      '%m-%d',        '%Y-%m-%d 00:00:00', timedelta(days=1)),
                        ('2day',  60*60*24*2,      '%m-%d',        '%Y-%m-%d 00:00:00', timedelta(days=2)),
                        ('3day',  60*60*24*3,      '%m-%d',        '%Y-%m-%d 00:00:00', timedelta(days=3)),
                        ('4day',  60*60*24*4,      '%m-%d',        '%Y-%m-%d 00:00:00', timedelta(days=4)),
                        ('1wk',   60*60*24*7*1,    '%m-%d',        '%Y-%m-%d 00:00:00', timedelta(days=7)),
                        ('2wk',   60*60*24*7*2,    '%m-%d',        '%Y-%m-%d 00:00:00', timedelta(days=14)),
                        ('1mo',   60*60*24*30*1,   '%Y-%m',        '%Y-%m-01 00:00:00', _CalendarStep(months=1)),
                        ('2mo',   60*60*24*30*2,   '%Y-%m',        '%Y-%m-01 00:00:00', _CalendarStep(months=2)),
                        ('3mo',   60*60*24*30*3,   '%Y-%m',        '%Y-%m-01 00:00:00', _CalendarStep(months=3)),
                        ('6mo',   60*60*24*30*6,   '%Y-%m',        '%Y-%m-01 00:00:00', _CalendarStep(months=6)),
                        ('1yr',   60*60*24*365*1,  '%Y',           '%Y-01-01 00:00:00', _CalendarStep(years=1)),
                        ('5yr',   60*60*24*365*5,  '%Y',           '%Y-01-01 00:00:00', _CalendarStep(years=5)),
                        ('10yr',  60*60*24*365*10, '%Y',           '%Y-01-01 00:00:00', _CalendarStep(years=10)),}
        # Compute how many lines would be required for the specified time intervals
        for _tuple_ in _to_compute_:
            _str_, _div_ = _tuple_[0], _tuple_[1]
            _num_of_grid_lines_[_tuple_] = _sec_total_ // _div_
        # Choose the time interval with the closest number of lines to meet the pixel goal
        _closest_tuple_, _closest_pixels_  = None, None
        for _tuple_ in _num_of_grid_lines_:
            if _num_of_grid_lines_[_tuple_] == 0: continue # avoid divide by zero
            _str_, _div_ = _tuple_[0], _tuple_[1]
            _pixels_btwn_lines_ = _dim_ // _num_of_grid_lines_[_tuple_]
            if _pixels_btwn_lines_ < self.txt_h*2: continue
            if _closest_tuple_ is None or abs(pixel_goal - _pixels_btwn_lines_) < _closest_pixels_:
                _closest_tuple_, _closest_pixels_  = _tuple_, abs(pixel_goal - _pixels_btwn_lines_)
        if _closest_tuple_ is None: return []
        # Round that value to the closest time interval
        _rounder_        = _closest_tuple_[3]
        dt_start_rounded = datetime.strptime(dt_start.strftime(_rounder_), '%Y-%m-%d %H:%M:%S')
        if isinstance(dt_start, date): dt_start = datetime.combine(dt_start, datetime.min.time())
        if isinstance(dt_end,   date): dt_end   = datetime.combine(dt_end,   datetime.max.time())
        # Iterate through the time interval & construct the svg
        _dt_             = dt_start_rounded
        _display_format_ = _closest_tuple_[2]
        _time_delta_     = _closest_tuple_[4]
        _svg_            = []
        _inner_color_    = self.p2s.colorTyped('axis',  'inner')
        _text_color_     = self.p2s.colorTyped('label', 'inner')
        while _dt_ < dt_end:
            if _dt_ > dt_start and _dt_ < dt_end:
                _str_ = _dt_.strftime(_display_format_)
                _v_   = _dim_ * ((_dt_ - dt_start).total_seconds()) / _sec_total_
                if x_axis:
                    _x1_, _y1_ = plot_origin[0] + _v_, plot_origin[1]
                    _x2_, _y2_ = plot_origin[0] + _v_, plot_origin[1] - plot_wxh[1]
                    _svg_.append(f'<line x1="{_x1_}" y1="{_y1_}" x2="{_x2_}" y2="{_y2_}" stroke="{_inner_color_}" stroke-width="0.5" />')
                    _svg_.append(self.p2s.svgText(_str_, _x1_, _y2_ + self.txt_h/2.0, txt_h=self.txt_h*0.8, color=_text_color_, rotation=90))
                    if dl is not None:
                        dl.line(_x1_, _y1_, _x2_, _y2_, _inner_color_, width=0.5)
                        dl.text(self.p2s, _str_, _x1_, _y2_ + self.txt_h/2.0, txt_h=self.txt_h*0.8, color=_text_color_, rotation=90, svg='')
                else:
                    _x1_, _y1_ = plot_origin[0],               plot_origin[1] - _v_
                    _x2_, _y2_ = plot_origin[0] + plot_wxh[0], plot_origin[1] - _v_
                    _svg_.append(f'<line x1="{_x1_}" y1="{_y1_}" x2="{_x2_}" y2="{_y2_}" stroke="{_inner_color_}" stroke-width="0.5" />')
                    _svg_.append(self.p2s.svgText(_str_, _x1_ + self.txt_h/2.0, _y2_, txt_h=self.txt_h*0.8, color=_text_color_))
                    if dl is not None:
                        dl.line(_x1_, _y1_, _x2_, _y2_, _inner_color_, width=0.5)
                        dl.text(self.p2s, _str_, _x1_ + self.txt_h/2.0, _y2_, txt_h=self.txt_h*0.8, color=_text_color_, svg='')
            _dt_ += _time_delta_
        return _svg_

    #
    # __renderContext__()
    #
    def __renderContext__(self, _randid_):
        # Render the context
        self._dl_context_ = _dl_ = DisplayList(self.wxh[0], self.wxh[1])
        self._clip_rect_  = None
        w_context, x_ins, h_context, y_ins = self.__context_geom__ # because of wxh may not be big enough, context may not be possible...
        if self.draw_context and (w_context > 0 or h_context > 0 or x_ins > 0 or y_ins > 0):
            _axis_color_       = self.p2s.colorTyped('axis', 'default')
            _axis_label_color_ = self.p2s.colorTyped('axis', 'label')
            _axis_min_color_   = self.p2s.colorTyped('axis', 'min')
            _axis_max_color_   = self.p2s.colorTyped('axis', 'max')
            xyo      = self.plot_origin[0], self.plot_origin[1]
            _w_, _h_ = self.plot_size[0],   self.plot_size[1]
            _svg_ = []
            # Label the axes
            for _quad_ in [('x', _w_, h_context, self.x_clean),
                           ('y', _h_, w_context, self.y_clean)]:
                _axis_, _dim_, _screen_dim_, _clean_ = _quad_
                if _screen_dim_ == 0: continue # meaning that a guardrail kicked in
                # Find the min value & pull the label -- reformat if necessary
                if len(self.df_flat) > 0: _min_cell_label_ = self.df_flat[f'__{_axis_}__'][self.df_flat[f'__{_axis_}i__'].arg_min()]
                else:                     _min_cell_label_ = None
                if isinstance(_min_cell_label_, dict): _min_cell_label_ = '|'.join([str(_) for _ in list(_min_cell_label_.values())])
                if   _axis_ == 'x' and self.x_range is not None and \
                     (self.p2s.numericColumn(self.df_flat, '__x__') or self.p2s.dateColumn(self.df_flat, '__x__') or self.p2s.dateTimeColumn(self.df_flat, '__x__')):
                    _min_cell_label_ = self.x_range[0]
                elif _axis_ == 'y' and self.y_range is not None and \
                     (self.p2s.numericColumn(self.df_flat, '__y__') or self.p2s.dateColumn(self.df_flat, '__y__') or self.p2s.dateTimeColumn(self.df_flat, '__y__')):
                    _min_cell_label_ = self.y_range[0]
                if   _axis_ == 'x' and self.x_shared_label_range is not None: _min_cell_label_ = self.x_shared_label_range[0]
                elif _axis_ == 'y' and self.y_shared_label_range is not None: _min_cell_label_ = self.y_shared_label_range[0]
                # Find the max value & pull the label -- reformat if necessary
                if len(self.df_flat) > 0: _max_cell_label_ = self.df_flat[f'__{_axis_}__'][self.df_flat[f'__{_axis_}i__'].arg_max()]
                else:                     _max_cell_label_ = None
                if isinstance(_max_cell_label_, dict): _max_cell_label_ = '|'.join([str(_) for _ in list(_max_cell_label_.values())])
                if   _axis_ == 'x' and self.x_range is not None and \
                     (self.p2s.numericColumn(self.df_flat, '__x__') or self.p2s.dateColumn(self.df_flat, '__x__') or self.p2s.dateTimeColumn(self.df_flat, '__x__')):
                    _max_cell_label_ = self.x_range[1]
                elif _axis_ == 'y' and self.y_range is not None and \
                     (self.p2s.numericColumn(self.df_flat, '__y__') or self.p2s.dateColumn(self.df_flat, '__y__') or self.p2s.dateTimeColumn(self.df_flat, '__y__')):
                    _max_cell_label_ = self.y_range[1]
                if   _axis_ == 'x' and self.x_shared_label_range is not None: _max_cell_label_ = self.x_shared_label_range[1]
                elif _axis_ == 'y' and self.y_shared_label_range is not None: _max_cell_label_ = self.y_shared_label_range[1]
                # Helper to further format the labels based on the width available
                _label_left_, _label_center_, _label_right_ = self.__formatLabels__(_axis_, _min_cell_label_, _max_cell_label_, _dim_, _clean_)
                l_len, _, r_len = self.p2s.textLength(_label_left_,  self.txt_h), self.p2s.textLength(_label_center_, self.txt_h), self.p2s.textLength(_label_right_,  self.txt_h)
                if   _axis_ == 'x':
                    _space_  = _w_ - (l_len + r_len)
                    _middle_ = xyo[0] + l_len + _space_//2
                    _svg_.append(_dl_.text(self.p2s, _label_left_,   xyo[0],       xyo[1] + self.txt_h, txt_h=self.txt_h, anchor='start',  color=_axis_min_color_))
                    _svg_.append(_dl_.text(self.p2s, _label_center_, _middle_,     xyo[1] + self.txt_h, txt_h=self.txt_h, anchor='middle', color=_axis_label_color_))
                    _svg_.append(_dl_.text(self.p2s, _label_right_,  xyo[0] + _w_, xyo[1] + self.txt_h, txt_h=self.txt_h, anchor='end',    color=_axis_max_color_))
                elif _axis_ == 'y':
                    _space_  = _h_ - (l_len + r_len)
                    _middle_ = xyo[1] - l_len - _space_//2
                    _svg_.append(_dl_.text(self.p2s, _label_left_,   xyo[0] - 2, xyo[1],       txt_h=self.txt_h, anchor='start',  rotation=270, color=_axis_min_color_))
                    _svg_.append(_dl_.text(self.p2s, _label_center_, xyo[0] - 2, _middle_,     txt_h=self.txt_h, anchor='middle', rotation=270, color=_axis_label_color_))
                    _svg_.append(_dl_.text(self.p2s, _label_right_,  xyo[0] - 2, xyo[1] - _h_, txt_h=self.txt_h, anchor='end',    rotation=270, color=_axis_max_color_))
                else: raise ValueError(f'XYp.__renderContext__(): Unrecognized axis {_axis_} (shouldn\'t be possible)')
            # Render the internal grid lines
            # -- helper functions
            def __min__(col):
                if col == 'x':
                    if self.x_range is not None: return self.x_range[0]
                    return self.df_flat['__x__'].min()
                if col == 'y': 
                    if self.y_range is not None: return self.y_range[0]
                    return self.df_flat['__y__'].min()
                raise ValueError(f'XYp.__renderContext__().__min__(): Unrecognized axis {col} (shouldn\'t be possible)')
            def __max__(col):
                if col == 'x': 
                    if self.x_range is not None: return self.x_range[1]
                    return self.df_flat['__x__'].max()
                if col == 'y': 
                    if self.y_range is not None: return self.y_range[1]
                    return self.df_flat['__y__'].max()
                raise ValueError(f'XYp.__renderContext__().__max__(): Unrecognized axis {col} (shouldn\'t be possible)')
            # - x axis
            if   self.p2s.dateColumn(self.df_flat, '__x__') or self.p2s.dateTimeColumn(self.df_flat, '__x__'):
                _dt_context_ = self.__renderContext_linearTime__(__min__('x'), __max__('x'), xyo, (_w_, _h_), x_axis=True, dl=_dl_)
                _svg_.extend(_dt_context_)
            elif self.__axisIsPeriodicTime__(self.x_clean):
                _periodic_context_ = self.__renderContext_periodicTime__(self.x_clean, __min__('x'), __max__('x'), xyo, (_w_, _h_), x_axis=True, dl=_dl_)
                _svg_.extend(_periodic_context_)
            elif self.p2s.numericColumn(self.df_flat, '__x__'):
                _numeric_context_ = self.__renderContext_numeric__(__min__('x'), __max__('x'), xyo, (_w_, _h_), x_axis=True, dl=_dl_)
                _svg_.extend(_numeric_context_)
            else:
                _set_context_ = self.__renderContext_set__(xyo, (_w_, _h_), x_axis=True, dl=_dl_)
                _svg_.extend(_set_context_)
            # - y axis
            if self.p2s.dateColumn(self.df_flat, '__y__') or self.p2s.dateTimeColumn(self.df_flat, '__y__'):
                _dt_context_ = self.__renderContext_linearTime__(__min__('y'), __max__('y'), xyo, (_w_, _h_), x_axis=False, dl=_dl_)
                _svg_.extend(_dt_context_)
            elif self.__axisIsPeriodicTime__(self.y_clean):
                _periodic_context_ = self.__renderContext_periodicTime__(self.y_clean, __min__('y'), __max__('y'), xyo, (_w_, _h_), x_axis=False, dl=_dl_)
                _svg_.extend(_periodic_context_)
            elif self.p2s.numericColumn(self.df_flat, '__y__'):
                _numeric_context_ = self.__renderContext_numeric__(__min__('y'), __max__('y'), xyo, (_w_, _h_), x_axis=False, dl=_dl_)
                _svg_.extend(_numeric_context_)
            else:
                _set_context_ = self.__renderContext_set__(xyo, (_w_, _h_), x_axis=False, dl=_dl_)
                _svg_.extend(_set_context_)
            # Render the plot outline (stroke-only rect -> 4 GPU line instances)
            _svg_.append(f'<rect x="{xyo[0]}" y="{xyo[1]-_h_}" width="{_w_}" height="{_h_}" stroke="{_axis_color_}" fill="none" stroke-width="0.25" />')
            _dl_.line(xyo[0],      xyo[1]-_h_, xyo[0]+_w_, xyo[1]-_h_, _axis_color_, width=0.25)
            _dl_.line(xyo[0],      xyo[1],     xyo[0]+_w_, xyo[1],     _axis_color_, width=0.25)
            _dl_.line(xyo[0],      xyo[1]-_h_, xyo[0],     xyo[1],     _axis_color_, width=0.25)
            _dl_.line(xyo[0]+_w_,  xyo[1]-_h_, xyo[0]+_w_, xyo[1],     _axis_color_, width=0.25)
            # Join all the svg objects together to finish the context
            _context_ = ''.join(_svg_)
            # Define a clip path for the plot (it keeps the larger dots from bleeding out)
            _svg_defs_ = f'''<defs>
                               <clipPath id="plotClip-{_randid_}">
                                 <rect x="{xyo[0]-1}" y="{xyo[1]-_h_-1}" width="{_w_+2}" height="{_h_+2}" />
                               </clipPath>
                             </defs>'''
            _clip_url_ = f'clip-path="url(#plotClip-{_randid_})"'
            self._clip_rect_ = (xyo[0]-1, xyo[1]-_h_-1, _w_+2, _h_+2)
        else: 
            _context_  = ''
            _svg_defs_ = ''
            _clip_url_ = ''
        
        self.svg_context = _context_
        self.svg_defs    = _svg_defs_
        self.clip_url    = _clip_url_

    #
    # __renderDistributions__()
    #
    def __renderDistributions__(self):
        self._dl_distributions_ = _dl_ = DisplayList(self.wxh[0], self.wxh[1])
        if self.x_distributions is None and self.y_distributions is None:
            self.svg_distributions = ''
            return
        _svg_ = [] # svg return object

        # for each axis...
        for _tuple_ in [('x', self.x_distributions, self.x_distributions_clean, self.df_x_distribution),
                        ('y', self.y_distributions, self.y_distributions_clean, self.df_y_distribution)]:
            _axis_, _param_, _clean_, _df_ = _tuple_
            if _param_ is None or _df_ is None: continue

            # Column-name scheme mirrors __distributeElements__() (only a subset is used here)
            _totals_field_      = f'__{_axis_}i_total__'             # the total for a particular cut
            _color_field_       = f'__{_axis_}dists_color__'         # color for this specific bar
            _min_field_         = f'__{_axis_}dists_{_axis_}i_min__' # the first coordinate (0.0 to 1.0) of the bar
            _max_field_         = f'__{_axis_}dists_{_axis_}i_max__' # the second coordinate (0.0 to 1.0) of the bar
            _totals_min_field_  = f'__{_axis_}i_total_min__'         # the min to compare the total to
            _totals_max_field_  = f'__{_axis_}i_total_max__'         # the max to compare the total to

            # "u" is the baseline of the distribution rendering ... the "v" is the height of the bar
            u_base, u_dist = _clean_['u_base'], _clean_['u_dist']
            u_sign         = _clean_['u_sign']

            # transform the distributions counts into the coordinate system
            _df_ = _df_.with_columns(pl.when(pl.col(_totals_min_field_) == pl.col(_totals_max_field_))
                                       .then(pl.lit(_clean_['base'] + _clean_['h']))
                                       .otherwise(
                                           _clean_['base'] + _clean_['sign'] * _clean_['h'] * (pl.col(_totals_field_)     - pl.col(_totals_min_field_)) / 
                                                                                              (pl.col(_totals_max_field_) - pl.col(_totals_min_field_))                                           
                                       ).alias('v'),
                                     (u_base + u_sign * u_dist * (pl.col(_min_field_)       - pl.col(_min_field_).min())/
                                                                 (pl.col(_max_field_).max() - pl.col(_min_field_).min())).alias('u0'),
                                     (u_base + u_sign * u_dist * (pl.col(_max_field_)       - pl.col(_min_field_).min())/
                                                                 (pl.col(_max_field_).max() - pl.col(_min_field_).min())).alias('u1'))
            
            # Sort by increasing u (these are the bin coordinates)
            _reversed_ = (_axis_ == 'y')
            _df_       = _df_.sort('u0', descending=_reversed_)

            # What are the conditions for drawing it in as a simple line (or set of lines?)
            # ... more than one line
            # ... the distribution inside & the dot size is shown
            # ... the size of the rectangles would be less than five pixels
            _draw_simple_lines_ = len(set(_df_[_color_field_])) > 1 or \
                                  (self.p2s.DISTRIBUTION_INSIDEp in _clean_['enums'] and self.dot_size_orig is not None) or \
                                  (u_dist//_clean_['bins'][0]) < 5
                                  
            # If it's complicated, just draw the trend line (per color)
            if _draw_simple_lines_:
                # Per color... create a path for each
                for k, k_df in _df_.group_by(_color_field_, maintain_order=True):
                    # Reverse the coordinates if it's x vs y
                    if _axis_ == 'x':
                        _concat_op_ = self.p2s.polarsConcatString('L {u0:.2f} {v:.2f} L {u1:.2f} {v:.2f}')
                        _start_pt_str_, _end_pt_str_ = f'{u_base} {_clean_["base"]}', f'{u_base + u_sign*u_dist} {_clean_["base"]}'
                    else:             
                        _concat_op_ = self.p2s.polarsConcatString('L {v:.2f} {u0:.2f} L {v:.2f} {u1:.2f}')
                        _start_pt_str_, _end_pt_str_ = f'{_clean_["base"]} {u_base} ', f'{_clean_["base"]} {u_base + u_sign*u_dist} '
                    # Create the dataframe field for the "line to" path description
                    k_df = k_df.with_columns(pl.concat_str(_concat_op_).alias('__lineto__'))
                    # Add begins and ends & format the path appropriately
                    _path_ = f'M {_start_pt_str_}' + ' '.join(k_df['__lineto__']) + f' L {_end_pt_str_}'
                    # Add the path to the return svg object
                    _svg_.append(f'<path d="{_path_}" stroke="{k[0]}" stroke-width="0.5" fill="none" />')
                    # GPU: same polyline as consecutive segments
                    _u0s_, _u1s_, _vs_ = k_df['u0'].to_list(), k_df['u1'].to_list(), k_df['v'].to_list()
                    if _axis_ == 'x':
                        _pts_  = [(u_base, _clean_['base'])]
                        for _u0_, _u1_, _v_ in zip(_u0s_, _u1s_, _vs_): _pts_.extend([(_u0_, _v_), (_u1_, _v_)])
                        _pts_.append((u_base + u_sign*u_dist, _clean_['base']))
                    else:
                        _pts_  = [(_clean_['base'], u_base)]
                        for _u0_, _u1_, _v_ in zip(_u0s_, _u1s_, _vs_): _pts_.extend([(_v_, _u0_), (_v_, _u1_)])
                        _pts_.append((_clean_['base'], u_base + u_sign*u_dist))
                    for _j_ in range(len(_pts_) - 1):
                        _dl_.line(_pts_[_j_][0], _pts_[_j_][1], _pts_[_j_+1][0], _pts_[_j_+1][1], k[0], width=0.5)
            # Else draw it more traditionally
            else:
                for k, k_df in _df_.group_by(_color_field_, maintain_order=True): # group_by mirrors the trend-line branch above (single group when uncolored)
                    _base_ = _clean_['base']
                    if _axis_ == 'x': k_df   = k_df.with_columns((pl.col('u1') - pl.col('u0')).alias('w'), (_base_       - pl.col('v')).alias('h'))
                    else:             k_df   = k_df.with_columns((pl.col('u0') - pl.col('u1')).alias('h'), (pl.col('v')  - _base_).alias('w'))
                    _common_svg_ = f'width="{{w}}" height="{{h}}" stroke-opacity="0.9" stroke="{k[0]}" stroke-width="0.5" fill-opacity="0.5" fill="{k[0]}"'
                    if _axis_ == 'x': _concat_str_ = f'<rect x="{{u0:.2f}}" y="{{v:.2f}}" {_common_svg_} />'
                    else:             _concat_str_ = f'<rect x="{_base_:.2f}" y="{{u1:.2f}}" {_common_svg_} />'
                    _concat_op_ = self.p2s.polarsConcatString(_concat_str_)
                    k_df = k_df.with_columns(pl.concat_str(_concat_op_).alias('__svg__'))
                    _svg_.extend(k_df['__svg__'])
                    # GPU: same bars as rect instances (fill only; svg keeps the stroke detail)
                    _r_, _g_, _b_, _ = hexToRGBA(k[0])
                    if _axis_ == 'x': _dl_.rects_table(k_df, 'u0', 'v', 'w', 'h', (_r_, _g_, _b_), opacity=0.5, svg_col=None)
                    else:             _dl_.rects_table(k_df, _base_, 'u1', 'w', 'h', (_r_, _g_, _b_), opacity=0.5, svg_col=None)
            if _axis_ == 'x': self.df_x_distribution = _df_
            else:             self.df_y_distribution = _df_

        self.svg_distributions = ''.join(_svg_)

    #
    # __strokeOpacityAttr__() - returns the SVG stroke-opacity attribute string for the given enums
    #
    def __strokeOpacityAttr__(self, _enums_):
        if self.p2s.LINEOPACITY_75 in _enums_: return 'stroke-opacity="0.75"'
        if self.p2s.LINEOPACITY_50 in _enums_: return 'stroke-opacity="0.50"'
        if self.p2s.LINEOPACITY_25 in _enums_: return 'stroke-opacity="0.25"'
        if self.p2s.LINEOPACITY_10 in _enums_: return 'stroke-opacity="0.10"'
        return ''

    #
    # __dashArrayAttr__() - returns the SVG stroke-dasharray attribute string for the given enums
    #
    def __dashArrayAttr__(self, _enums_, _dash_array_):
        if   self.p2s.LINESTYLE_DOTTED    in _enums_: return 'stroke-dasharray="5 5"'
        elif self.p2s.LINESTYLE_SPECIFIED in _enums_: return f'stroke-dasharray="{" ".join(str(x) for x in _dash_array_)}"'
        elif self.p2s.LINESTYLE_SOLID     in _enums_: return 'stroke-dasharray="none"'
        return ''

    #
    # __coordinateAdjustOps__() - returns Polars operations to center rectangle dots on their coordinates
    #
    def __coordinateAdjustOps__(self):
        if isinstance(self.dot_size_orig, int):
            _half_ = self.dot_size_orig / 2.0
            return [(pl.col('__xpx__') + _half_).alias('__xpx__'),
                    (pl.col('__ypx__') + _half_).alias('__ypx__')]
        return []

    #
    # __gpuDashFromEnums__() / __gpuOpacityFromEnums__() - GPU equivalents of the
    # stroke-dasharray / stroke-opacity attribute helpers (dash approximated on/off)
    #
    def __gpuDashFromEnums__(self, _enums_, _dash_array_):
        if self.p2s.LINESTYLE_DOTTED in _enums_: return (5.0, 5.0)
        if self.p2s.LINESTYLE_SPECIFIED in _enums_ and _dash_array_ is not None and len(_dash_array_) > 0:
            if len(_dash_array_) >= 2: return (float(_dash_array_[0]), float(_dash_array_[1]))
            return (float(_dash_array_[0]), float(_dash_array_[0]))
        return None

    def __gpuOpacityFromEnums__(self, _enums_):
        if self.p2s.LINEOPACITY_75 in _enums_: return 0.75
        if self.p2s.LINEOPACITY_50 in _enums_: return 0.50
        if self.p2s.LINEOPACITY_25 in _enums_: return 0.25
        if self.p2s.LINEOPACITY_10 in _enums_: return 0.10
        return 1.0

    #
    # __lineSegmentsToDL__() - emit GPU line segments for polyline dataframes
    # - consecutive (__xpx__, __ypx__) pairs within each '__line__' become one segment
    # - color: constant hex (color=) or per-row hex column (color_col=)
    #
    def __lineSegmentsToDL__(self, df, dl, color=None, color_col=None,
                             width=1.0, width_col=None, opacity=1.0, opacity_col=None,
                             dash=None, apply_adjust=True):
        if dl is None or len(df) == 0: return
        if apply_adjust: df = df.with_columns(self.__coordinateAdjustOps__())
        _seg_ = df.with_columns([pl.col('__xpx__').shift(-1).over('__line__').alias('__x2__'),
                                 pl.col('__ypx__').shift(-1).over('__line__').alias('__y2__')]) \
                  .filter(pl.col('__x2__').is_not_null() & pl.col('__xpx__').is_not_null())
        if len(_seg_) == 0: return
        if color_col is not None:
            _seg_  = _seg_.with_columns(self.p2s.rgbFromHexPolarsOperations(color_col, '__r_f__', '__g_f__', '__b_f__'))
            _rgba_ = ('__r_f__', '__g_f__', '__b_f__')
        else:
            _r_, _g_, _b_, _ = hexToRGBA(color)
            _rgba_ = (_r_, _g_, _b_)
        dl.lines_table(_seg_, '__xpx__', '__ypx__', '__x2__', '__y2__', _rgba_,
                       width=(width_col if width_col is not None else width),
                       opacity=(opacity_col if opacity_col is not None else opacity),
                       dash=dash, svg_col=None)

    #
    # __renderLines_simple__()
    # - everything is a constant
    # - columns needed in the input dataframe: '__line__', '__xpx__', '__ypx__'
    #
    def __renderLines_simple__(self, df, _enums_, _color_, _dash_array_, _width_):
        _df_lines_ = df.with_columns(
            self.__coordinateAdjustOps__()
        ).group_by('__line__', maintain_order=True).agg([
            pl.format('L {} {}', '__xpx__', '__ypx__').alias('points'),
        ]).with_columns(
            pl.col('points').list.join(' ').str.replace('L', 'M', literal=True, n=1).alias('path_d')
        ).with_columns(
            pl.format(
                '<path d="{}"/>',
                'path_d',
            ).alias('svg_path')
        )

        self.__lineSegmentsToDL__(df, self._dl_lines_, color=_color_, width=_width_,
                                  opacity=self.__gpuOpacityFromEnums__(_enums_),
                                  dash=self.__gpuDashFromEnums__(_enums_, _dash_array_))

        return [f'<g fill="none" stroke="{_color_}" stroke-width="{_width_}" {self.__strokeOpacityAttr__(_enums_)} {self.__dashArrayAttr__(_enums_, _dash_array_)}>'] + \
               _df_lines_['svg_path'].to_list() + \
               ['</g>']

    #
    # __renderLines_groupByColor__()
    # - everything is a constant except the color -- and that's the group_by element
    # - columns needed in the input dataframe: ['__line__', '__xpx__', '__ypx__']
    #
    def __renderLines_groupByColor__(self, df, _enums_, _color_, _dash_array_, _width_):
        _df_lines_ = df.with_columns(
            self.__coordinateAdjustOps__()
        ).group_by('__line__', maintain_order=True).agg([
            pl.format('L {} {}', '__xpx__', '__ypx__').alias('points'),
        ]).with_columns(
            pl.col('points').list.join(' ').str.replace('L', 'M', literal=True, n=1).alias('path_d'),
            self.p2s.colorizeColumnPolarsOperations('__line__').alias('__line_color__')
        ).with_columns(
            pl.format(
                '<path d="{}" stroke="{}"/>',
                'path_d',
                '__line_color__',
            ).alias('svg_path')
        )

        self.__lineSegmentsToDL__(
            df.with_columns(self.p2s.colorizeColumnPolarsOperations('__line__').alias('__line_color__')),
            self._dl_lines_, color_col='__line_color__', width=_width_,
            opacity=self.__gpuOpacityFromEnums__(_enums_),
            dash=self.__gpuDashFromEnums__(_enums_, _dash_array_))

        return [f'<g fill="none" stroke-width="{_width_}" {self.__strokeOpacityAttr__(_enums_)} {self.__dashArrayAttr__(_enums_, _dash_array_)}>'] + \
               _df_lines_['svg_path'].to_list() + \
               ['</g>']

    #
    # __renderLines_complex__()
    # - assumes df is pre-sorted by the line_order option
    #
    def __renderLines_complex__(self, df, _enums_, _color_, _dash_array_, _width_, _randid_):
        # Add the line width column
        if   self.p2s.LINEWIDTH_DOTSIZE_MEAN      in _enums_: df = df.with_columns(pl.col('__radius__').mean().over('__line__').alias('__line_width__'))
        elif self.p2s.LINEWIDTH_DOTSIZE_VARIABLE  in _enums_: df = df.with_columns(pl.col('__radius__').alias('__line_width__'))
        elif self.p2s.LINEWIDTH_DOTSIZE_SPECIFIED in _enums_: df = df.with_columns(pl.lit(_width_).alias('__line_width__'))
        else: raise ValueError(f'XYp.__renderLines_complex__():  self.p2s.LINEWIDTH_DOTSIZE_* is not specified ({_enums_=})')

        # Determine the line style
        _line_dash_array_ = ''
        if   self.p2s.LINESTYLE_DOTTED            in _enums_: _line_dash_array_ = 'stroke-dasharray="5 5"'
        elif self.p2s.LINESTYLE_SPECIFIED         in _enums_: _line_dash_array_ = f'stroke-dasharray="{' '.join([str(x) for x in _dash_array_])}"'
        elif self.p2s.LINESTYLE_SOLID             in _enums_: _line_dash_array_ = ''
        else: raise ValueError(f'XYp.__renderLines_complex__():  self.p2s.LINESTYLE_* is not specified ({_enums_=})')

        # Determine the line color
        if   self.p2s.LINECOLOR_GROUPBY           in _enums_: df = df.with_columns(self.p2s.colorizeColumnPolarsOperations('__line__').alias('__line_color__'))
        elif self.p2s.LINECOLOR_FIELD             in _enums_: df = df.with_columns(pl.col('__hexcolor__').alias('__line_color__'))
        elif self.p2s.LINECOLOR_SPECIFIED         in _enums_: df = df.with_columns(pl.lit(_color_).alias('__line_color__'))
        else: raise ValueError(f'XYp.__renderLines_complex__():  self.p2s.LINECOLOR_* is not specified ({_enums_=})')

        # Determine the stroke opacity
        if   self.p2s.LINEOPACITY_FIELD_MEAN      in _enums_: df = df.with_columns(pl.col('__fill_opacity__').mean().over('__line__').alias('__line_opacity__'))
        elif self.p2s.LINEOPACITY_FIELD_VARIABLE  in _enums_: df = df.with_columns(pl.col('__fill_opacity__').alias('__line_opacity__'))
        elif self.p2s.LINEOPACITY_100             in _enums_: df = df.with_columns(pl.lit(1.00).alias('__line_opacity__'))
        elif self.p2s.LINEOPACITY_75              in _enums_: df = df.with_columns(pl.lit(0.75).alias('__line_opacity__'))
        elif self.p2s.LINEOPACITY_50              in _enums_: df = df.with_columns(pl.lit(0.50).alias('__line_opacity__'))
        elif self.p2s.LINEOPACITY_25              in _enums_: df = df.with_columns(pl.lit(0.25).alias('__line_opacity__'))
        elif self.p2s.LINEOPACITY_10              in _enums_: df = df.with_columns(pl.lit(0.10).alias('__line_opacity__'))
        else: raise ValueError(f'XYp.__renderLines_complex__():  self.p2s.LINEOPACITY_* is not specified ({_enums_=})')

        # Per end point determination / i.e., do we need to draw each segment separately or not
        _per_endpoint_ = len({self.p2s.LINEWIDTH_DOTSIZE_VARIABLE, 
                              self.p2s.LINECOLOR_FIELD, 
                              self.p2s.LINEOPACITY_FIELD_VARIABLE} & _enums_) > 0
        #
        # We have to create each segment independently
        #
        if _per_endpoint_:
            # Duplicate the next line into the current line to create the line segments
            _cols_      = ['__xpx__', '__ypx__', '__line_color__', '__line_width__', '__line_opacity__']
            _shift_ops_ = []
            for _col_ in _cols_: _shift_ops_.append(pl.col(_col_).shift(-1).over('__line__').alias(f'{_col_}2'))
            # Keep the following columns -- they are the only ones we'll need for the segment creation
            _to_keep_ = _cols_ + ['__line__', '__line_index__']
            # Make sure the line is safe to use as an svg id
            _make_safe_ = pl.col('__line__').str.replace_all('[#|()_" ]','-').alias('__line_safe__')
            # Create the svg gradient id
            _create_svg_id_ = pl.format(f'lines_{_randid_}_{{}}_{{}}_{{}}', pl.col('__line_safe__'), pl.col('__line_index__'), pl.col('__uniq__')).alias('__gradient_def_id__')
            # Create the gradient definition
            _gradient_definition_ = pl.format('''<linearGradient id="{}" x1="{}" y1="{}" x2="{}" y2="{}" gradientUnits="userSpaceOnUse">
                                                 <stop offset="0%" stop-color="{}" stop-opacity="{}"/><stop offset="100%" stop-color="{}" stop-opacity="{}"/></linearGradient>''',
                                              pl.col('__gradient_def_id__'), pl.col('__xpx__'), pl.col('__ypx__'), pl.col('__xpx__2'), pl.col('__ypx__2'),
                                              pl.col('__line_color__'), pl.col('__line_opacity__'), pl.col('__line_color__2'), pl.col('__line_opacity__2'))
            # Create the line definition
            _line_definition_     = pl.format('''<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="url(#{})" stroke-width="{}"/>''',
                                              pl.col('__xpx__'), pl.col('__ypx__'), pl.col('__xpx__2'), pl.col('__ypx__2'), pl.col('__gradient_def_id__'),
                                              (pl.col('__line_width__') + pl.col('__line_width__2'))/2.0)
            # Perform the operations
            _df_lines_ = df.select(_to_keep_) \
                           .with_columns(_shift_ops_) \
                           .drop_nulls() \
                           .with_row_index('__uniq__') \
                           .with_columns(_make_safe_) \
                           .with_columns(_create_svg_id_) \
                           .with_columns(_gradient_definition_.alias('__gradient_def__'), _line_definition_.alias('svg_line'))
            # GPU: per-vertex-colored tapered quads -- matches the SVG per-segment
            # linearGradient (color/opacity/width interpolate between endpoints)
            if self._dl_lines_ is not None and len(_df_lines_) > 0:
                import numpy as _np_
                _seg_ = _df_lines_.with_columns(
                    self.p2s.rgbFromHexPolarsOperations('__line_color__',  '__r0__', '__g0__', '__b0__') +
                    self.p2s.rgbFromHexPolarsOperations('__line_color__2', '__r1__', '__g1__', '__b1__')
                ).select(['__xpx__', '__ypx__', '__xpx__2', '__ypx__2',
                          '__line_width__', '__line_width__2',
                          '__r0__', '__g0__', '__b0__', '__line_opacity__',
                          '__r1__', '__g1__', '__b1__', '__line_opacity__2']).to_numpy().astype('f8')
                _p0_ = _seg_[:, 0:2]
                _p1_ = _seg_[:, 2:4]
                _d_  = _p1_ - _p0_
                _len_ = _np_.maximum(_np_.linalg.norm(_d_, axis=1, keepdims=True), 1e-6)
                _perp_ = _np_.hstack([-_d_[:, 1:2], _d_[:, 0:1]]) / _len_
                _hw0_ = _np_.maximum(_seg_[:, 4:5], 0.5) * 0.5
                _hw1_ = _np_.maximum(_seg_[:, 5:6], 0.5) * 0.5
                _v_ = _np_.empty((len(_seg_) * 4, 2))
                _v_[0::4] = _p0_ + _perp_ * _hw0_
                _v_[1::4] = _p0_ - _perp_ * _hw0_
                _v_[2::4] = _p1_ + _perp_ * _hw1_
                _v_[3::4] = _p1_ - _perp_ * _hw1_
                _c_ = _np_.empty((len(_seg_) * 4, 4))
                _c_[0::4] = _seg_[:, 6:10]
                _c_[1::4] = _seg_[:, 6:10]
                _c_[2::4] = _seg_[:, 10:14]
                _c_[3::4] = _seg_[:, 10:14]
                _base_ = _np_.arange(len(_seg_), dtype=_np_.uint32) * 4
                _idx_ = _np_.column_stack([_base_, _base_+1, _base_+2,
                                           _base_+1, _base_+3, _base_+2]).flatten()
                self._dl_lines_.tris(_v_.flatten().tolist(), _idx_.tolist(), _c_)
            # Format them for return
            _return_ = ['<defs>']
            _return_.extend(_df_lines_['__gradient_def__'].to_list())
            _return_.append('</defs>')
            _return_.extend(_df_lines_['svg_line'].to_list())
            return _return_
        #
        # We can still use path groupings
        #
        else:
            # GPU: per-line color/opacity/width segments (dash approximated on/off)
            self.__lineSegmentsToDL__(
                df.with_columns([pl.col('__line_opacity__').mean().over('__line__').alias('__line_opacity__'),
                                 pl.col('__line_width__').mean().over('__line__').alias('__line_width__')]),
                self._dl_lines_, color_col='__line_color__',
                width_col='__line_width__', opacity_col='__line_opacity__',
                dash=self.__gpuDashFromEnums__(_enums_, _dash_array_))
            _df_lines_ = df.with_columns(
                self.__coordinateAdjustOps__()
            ).group_by('__line__', maintain_order=True).agg([
                pl.format('L {} {}', '__xpx__', '__ypx__').alias('points'),
                pl.col('__line_color__')  .unique().get(0).alias('__line_color__'),
                pl.col('__line_opacity__').mean()         .alias('__line_opacity__'),
                pl.col('__line_width__')  .mean()         .alias('__line_width__'),
            ]).with_columns(
                pl.col('points').list.join(' ').str.replace('L', 'M', literal=True, n=1).alias('path_d')
            ).with_columns(
                pl.format(
                    f'<path d="{{}}" fill="none" stroke="{{}}" stroke-opacity="{{}}" stroke-width="{{}}" {_line_dash_array_}/>',
                    'path_d',
                    '__line_color__',
                    '__line_opacity__',
                    '__line_width__'
                ).alias('svg_path')
            )
            return _df_lines_['svg_path'].to_list()

    #
    # __dispatchLineRenderer__()
    # - routes a per-line dataframe to the appropriate renderer based on enum flags
    # - _randid_ must be supplied when __renderLines_complex__ is a valid path; omit to raise on that branch
    #
    def __dispatchLineRenderer__(self, k_df, _enums_, _color_, _dash_array_, _width_, _fixed_opacities_, _randid_=None):
        if   self.p2s.LINECOLOR_SPECIFIED         in _enums_ and \
             self.p2s.LINEWIDTH_DOTSIZE_SPECIFIED in _enums_ and \
             len(_fixed_opacities_ & _enums_) == 1:
            return self.__renderLines_simple__(k_df, _enums_, _color_, _dash_array_, _width_)
        elif self.p2s.LINECOLOR_GROUPBY           in _enums_ and \
             self.p2s.LINEWIDTH_DOTSIZE_SPECIFIED in _enums_ and \
             len(_fixed_opacities_ & _enums_) == 1:
            return self.__renderLines_groupByColor__(k_df, _enums_, _color_, _dash_array_, _width_)
        elif _randid_ is not None:
            return self.__renderLines_complex__(k_df, _enums_, _color_, _dash_array_, _width_, _randid_)
        else:
            raise ValueError(f'XYp.__renderLines__: invalid state (i.e., shouldn\'t happen): {(_enums_, _color_, _dash_array_, _width_)}')

    #
    # __renderLines__()
    #
    def __renderLines__(self, _randid_):
        self.svg_lines  = ''
        self._dl_lines_ = DisplayList(self.wxh[0], self.wxh[1])
        if self.line_clean is None: return
        # Initialize the holding variable
        _svg_ = []
        # Initialize the set of fixed opacities
        _fixed_opacities_ = {self.p2s.LINEOPACITY_100, self.p2s.LINEOPACITY_75, self.p2s.LINEOPACITY_50,
                             self.p2s.LINEOPACITY_25,  self.p2s.LINEOPACITY_10}
        # All seen enums
        _all_seen_enums_ = set()
        for _tuple_ in self.line_clean: _all_seen_enums_ |= _tuple_[-1]
        # Determine if any require pixel level information
        # ... if so, then render the dots into the df_dot_level dataframe
        if len({self.p2s.LINEWIDTH_DOTSIZE_MEAN, self.p2s.LINEWIDTH_DOTSIZE_VARIABLE,
                self.p2s.LINECOLOR_FIELD,
                self.p2s.LINEOPACITY_FIELD_MEAN, self.p2s.LINEOPACITY_FIELD_VARIABLE} & _all_seen_enums_) > 0:
            # Use the dot renderer to do the work
            df_dot_level = self.__renderDots__(line_rendering_mode=True)
            # Sort according to line order
            if '__line_order_by__' in df_dot_level: df_dot_level = df_dot_level.sort(['__line__', '__line_order_by__']) # <--- the other rendering path
            else:                                   df_dot_level = df_dot_level.sort(['__line__', '__xpx__']) # <--- this varies from the other rendering path
            # For each definitional group of lines, route to the appropriate renderer
            for k, k_df in df_dot_level.group_by('__line_index__', maintain_order=True):
                _enums_, _color_, _dash_array_, _width_ = self.line_clean[k[0]][-1:-5:-1]
                _svg_.extend(self.__dispatchLineRenderer__(k_df, _enums_, _color_, _dash_array_, _width_, _fixed_opacities_, _randid_))
        #
        # Otherwise, it's a simpler rendering path -- handle both of those here
        #
        else:
            # Sort according to line order
            if '__line_order_by__' in self.df_flat.columns: df_sorted = self.df_flat.sort('__line_order_by__') # <--- the other rendering path
            else:                                           df_sorted = self.df_flat.sort('__xi__') # <--- this varies from the other rendering path
            # For each definitional group of lines, route to the appropriate renderer (no complex path available here)
            for k, k_df in df_sorted.group_by('__line_index__', maintain_order=True):
                _enums_, _color_, _dash_array_, _width_ = self.line_clean[k[0]][-1:-5:-1]
                _svg_.extend(self.__dispatchLineRenderer__(k_df, _enums_, _color_, _dash_array_, _width_, _fixed_opacities_))

        self.svg_lines = '\n'.join(_svg_)

    #
    # __buildColorOps__()
    # - populates the provided operation lists with color pipeline ops for the given color mode
    # - mutates _agg_ops_, _norm_ops_, _spectrum_ops_, _tohexcolor_ops_, _fill_nulls_, _shape_template_ in place
    #
    def __buildColorOps__(self, _color_mode_, _color_default_, _color_error_,
                          _agg_ops_, _norm_ops_, _spectrum_ops_, _tohexcolor_ops_, _fill_nulls_, _shape_template_):
        if   _color_mode_ in [self.p2s.CROW_MAGNITUDEp,      self.p2s.CROW_STRETCHEDp,
                              self.p2s.CMAGNITUDE_SUMp,      self.p2s.CSTRETCHED_SUMp,
                              self.p2s.CMAGNITUDE_MINp,      self.p2s.CSTRETCHED_MINp,
                              self.p2s.CMAGNITUDE_MEDIANp,   self.p2s.CSTRETCHED_MEDIANp,
                              self.p2s.CMAGNITUDE_MEANp,     self.p2s.CSTRETCHED_MEANp,
                              self.p2s.CMAGNITUDE_MAXp,      self.p2s.CSTRETCHED_MAXp,
                              self.p2s.CSET_MAGNITUDEp,      self.p2s.CSET_STRETCHEDp]:
            if   _color_mode_ in [self.p2s.CROW_MAGNITUDEp,    self.p2s.CROW_STRETCHEDp]:    _agg_ops_.append(pl.len()                          .alias('__color_sum__'))
            elif _color_mode_ in [self.p2s.CMAGNITUDE_SUMp,    self.p2s.CSTRETCHED_SUMp]:    _agg_ops_.append(pl.col('__color__').sum()         .alias('__color_sum__'))
            elif _color_mode_ in [self.p2s.CMAGNITUDE_MINp,    self.p2s.CSTRETCHED_MINp]:    _agg_ops_.append(pl.col('__color__').min()         .alias('__color_sum__'))
            elif _color_mode_ in [self.p2s.CMAGNITUDE_MEDIANp, self.p2s.CSTRETCHED_MEDIANp]: _agg_ops_.append(pl.col('__color__').median()      .alias('__color_sum__'))
            elif _color_mode_ in [self.p2s.CMAGNITUDE_MEANp,   self.p2s.CSTRETCHED_MEANp]:   _agg_ops_.append(pl.col('__color__').mean()        .alias('__color_sum__'))
            elif _color_mode_ in [self.p2s.CMAGNITUDE_MAXp,    self.p2s.CSTRETCHED_MAXp]:    _agg_ops_.append(pl.col('__color__').max()         .alias('__color_sum__'))
            elif _color_mode_ in [self.p2s.CSET_MAGNITUDEp,    self.p2s.CSET_STRETCHEDp]:    _agg_ops_.append(pl.col('__color__').unique().len().alias('__color_sum__'))
            # Apply the spectrum
            if   _color_mode_ in [self.p2s.CROW_MAGNITUDEp,  self.p2s.CMAGNITUDE_SUMp, self.p2s.CMAGNITUDE_MINp, self.p2s.CMAGNITUDE_MEDIANp,
                                  self.p2s.CMAGNITUDE_MEANp, self.p2s.CMAGNITUDE_MAXp, self.p2s.CSET_MAGNITUDEp]:
                if self.color_magnitude_min is not None and self.color_magnitude_max is not None:
                    _cmin_, _cmax_ = self.color_magnitude_min, self.color_magnitude_max
                    _norm_ops_.append(pl.when(pl.lit(_cmin_) == pl.lit(_cmax_))
                                        .then(pl.lit(0.0))
                                        .otherwise((pl.col('__color_sum__') - _cmin_) / (_cmax_ - _cmin_)).alias('__color_norm__'))
                else:
                    _norm_ops_.append(pl.when(pl.col('__color_sum__').min() == pl.col('__color_sum__').max())
                                        .then(pl.lit(0.0))
                                        .otherwise((pl.col('__color_sum__')       - pl.col('__color_sum__').min()) /
                                                   (pl.col('__color_sum__').max() - pl.col('__color_sum__').min())).alias('__color_norm__'))
            # Apply the stretched spectrum
            else:
                if self.color_stretched_global_values is not None:
                    _n_ = len(self.color_stretched_global_values)
                    _norms_ = [float(i) / float(_n_ - 1) for i in range(_n_)] if _n_ > 1 else [0.0]
                    _norm_ops_.append(pl.col('__color_sum__').replace_strict(
                                        old=self.color_stretched_global_values,
                                        new=_norms_,
                                        default=0.0,
                                        return_dtype=pl.Float64).alias('__color_norm__'))
                else:
                    _norm_ops_.append(pl.when(pl.col('__color_sum__').n_unique() == 1)
                                        .then(pl.lit(0.0))
                                        .otherwise((pl.col("__color_sum__").rank('dense') - 1) / (pl.col("__color_sum__").n_unique() - 1)).alias("__color_norm__"))
            # Do the final conversion parts
            _spectrum_ops_  .extend(self.p2s.colorSpectrumPolarsOperations('__color_norm__', '__r__', '__g__', '__b__'))
            _tohexcolor_ops_.append(self.p2s.hexColorFromRGBTriplesPolarsOperations('__r__', '__g__', '__b__').alias('__hexcolor__'))
            _fill_nulls_    .append(pl.col('__hexcolor__').fill_null(_color_error_))
            _shape_template_.append('fill="{__hexcolor__}"')
        elif _color_mode_ == self.p2s.CSETp:
            _agg_ops_  .append(pl.col('__color__').cast(pl.String).unique().alias('__color_set__'))
            _norm_ops_ .append(pl.when(pl.col('__color_set__').list.len() == 1)
                                 .then(pl.col('__color_set__').list.get(0))
                                 .otherwise(pl.lit(-1)).alias('__set_element__'))
            _tohexcolor_ops_.append(self.p2s.colorizeColumnPolarsOperations('__set_element__').alias('__hexcolor__'))
            _fill_nulls_    .append(pl.col('__hexcolor__').fill_null(_color_error_))
            _shape_template_.append('fill="{__hexcolor__}"')
        elif '__hexcolor__' in self.df_flat.columns:
            _agg_ops_  .append(pl.col('__hexcolor__').unique().alias('__hexcolor_set__'))
            _norm_ops_ .append(pl.when(pl.col('__hexcolor_set__').list.len() == 1)
                                 .then(pl.col('__hexcolor_set__').list.get(0))
                                 .otherwise(pl.lit(_color_default_)).alias('__hexcolor__'))
            _fill_nulls_    .append(pl.col('__hexcolor__').fill_null(_color_error_))
            _shape_template_.append('fill="{__hexcolor__}"')

    #
    # __buildSizeAndOpacityOps__()
    # - populates the provided operation lists with dot-size and opacity pipeline ops
    # - mutates _agg_ops_, _norm_ops_, _shape_template_ in place
    #
    def __buildSizeAndOpacityOps__(self, _agg_ops_, _norm_ops_, _shape_template_):
        for _tuple_ in [(self.dot_size, '__dot_size__', '__dot_size_sum__', '__radius__',       'r',            self.dot_size_range, self.dot_size_enums),
                        (self.opacity,  '__opacity__',  '__opacity_sum__',  '__fill_opacity__', 'fill-opacity', self.opacity_range,  self.opacity_enums)]:
            _var_, _flat_column_, _sum_column_, _final_column_, _svg_attribute_, _range_, _enums_ = _tuple_

            if _sum_column_ == '__dot_size_sum__' and self.dot_size_global_min is not None:
                _norm_min_ = pl.lit(float(self.dot_size_global_min))
                _norm_max_ = pl.lit(float(self.dot_size_global_max))
            else:
                _norm_min_ = pl.col(_sum_column_).min()
                _norm_max_ = pl.col(_sum_column_).max()
            _normalize_ = pl.when(_norm_min_ == _norm_max_) \
                            .then(pl.lit(_range_[0])) \
                            .otherwise(_range_[0] + (_range_[1]-_range_[0])*(pl.col(_sum_column_) - _norm_min_) /
                                                                             (_norm_max_           - _norm_min_)).alias(_final_column_)
            if self.p2s.ROW_COUNTp in _enums_:
                _agg_ops_.append(pl.len().alias(_sum_column_))
                _norm_ops_.append(_normalize_)
                _shape_template_.append(f'{_svg_attribute_}="{{{_final_column_}}}"')
            elif _flat_column_ in self.df_flat.columns:
                if self.p2s.SETp in _enums_ or \
                   self.p2s.numericColumn(self.df_flat, _flat_column_) == False: _agg_ops_.append(pl.col(_flat_column_).unique().len().alias(_sum_column_))
                else:                                                            _agg_ops_.append(pl.col(_flat_column_).sum()         .alias(_sum_column_))
                _norm_ops_.append(_normalize_)
                _shape_template_.append(f'{_svg_attribute_}="{{{_final_column_}}}"')
            elif _final_column_ in self.df_flat.columns:
                _agg_ops_.append(pl.col(_final_column_).mean())
                _shape_template_.append(f'{_svg_attribute_}="{{{_final_column_}}}"')

    #
    # __renderDots__()
    #
    def __renderDots__(self, line_rendering_mode=False):
        _color_default_                   = self.p2s.colorTyped('data',       'default')
        _color_error_                     = self.p2s.colorTyped('error',      'default')
        _color_mode_                      = self.__determineColoringMode__()

        if    isinstance(self.dot_size_orig, int): _shape_template_ = ['<rect x="{__xpx__}" y="{__ypx__}"']
        else:                                      _shape_template_ = ['<circle cx="{__xpx__}" cy="{__ypx__}"']

        # Pipeline Stages: Group By, Aggregation, Normalization, Spectrum, To Hex Color, Fill Nulls, Concat
        _agg_ops_, _norm_ops_, _spectrum_ops_, _tohexcolor_ops_, _fill_nulls_, _concat_ops_ = [], [], [], [], [], []

        self.__buildColorOps__(_color_mode_, _color_default_, _color_error_,
                               _agg_ops_, _norm_ops_, _spectrum_ops_, _tohexcolor_ops_, _fill_nulls_, _shape_template_)
        self.__buildSizeAndOpacityOps__(_agg_ops_, _norm_ops_, _shape_template_)

        if line_rendering_mode:
            _groupby_ = ['__xpx__', '__ypx__', '__line__', '__line_index__']
            if '__line_order_by__' in self.df_flat.columns: _groupby_.append('__line_order_by__')
        else:
            _groupby_ = ['__xpx__', '__ypx__']
            #
            # Finalize the string concats
            #
            _shape_template_.append('/>')
            _str_concat_op_ = self.p2s.polarsConcatString(' '.join(_shape_template_))
            _concat_ops_.append(pl.concat_str(*_str_concat_op_).alias('__svg__'))

        #
        # Perform the transformation
        #
        if self.use_lazy_execution:
            _df_pixels_ = self.df_flat.lazy().group_by(_groupby_) \
                                             .agg(_agg_ops_) \
                                             .with_columns(_norm_ops_) \
                                             .with_columns(_spectrum_ops_) \
                                             .with_columns(_tohexcolor_ops_) \
                                             .with_columns(_fill_nulls_) \
                                             .with_columns(_concat_ops_) \
                                             .collect()
        else:
            _df_pixels_ = self.df_flat       .group_by(_groupby_) \
                                             .agg(_agg_ops_) \
                                             .with_columns(_norm_ops_) \
                                             .with_columns(_spectrum_ops_) \
                                             .with_columns(_tohexcolor_ops_) \
                                             .with_columns(_fill_nulls_) \
                                             .with_columns(_concat_ops_)
        if not line_rendering_mode: self.__legendCaptureColorbarDomain__(_df_pixels_)
        return _df_pixels_

    #
    # __legendCaptureColorbarDomain__() - metadata-capture hook (colorbar half):
    # record the min/max the spectrum normalization actually used, honoring the
    # same precedence as __buildColorOps__ (explicit color_magnitude_min/max for
    # magnitude modes; color_stretched_global_values for stretched modes; else
    # the per-pixel aggregated data)
    #
    def __legendCaptureColorbarDomain__(self, _df_pixels_):
        if getattr(self, 'legend_info', None) is None or self.legend_info.kind != 'colorbar': return
        _stretched_ = self.p2s.legendModeIsStretched(self._legend_color_mode_)
        if   not _stretched_ and self.color_magnitude_min is not None and self.color_magnitude_max is not None:
            _vmin_, _vmax_ = self.color_magnitude_min, self.color_magnitude_max
        elif _stretched_ and self.color_stretched_global_values:
            _vmin_, _vmax_ = min(self.color_stretched_global_values), max(self.color_stretched_global_values)
        elif '__color_sum__' in _df_pixels_.columns and len(_df_pixels_) > 0:
            _vmin_, _vmax_ = _df_pixels_['__color_sum__'].min(), _df_pixels_['__color_sum__'].max()
        else:
            _vmin_ = _vmax_ = None
        self.p2s.legendInfoColorbarFinalize(self.legend_info, self.legend_spec, _vmin_, _vmax_)

    #
    # __renderLegend__() - draw the captured legend metadata into the reserved strip
    #
    def __renderLegend__(self):
        self.svg_legend  = ''
        self._dl_legend_ = DisplayList(self.wxh[0], self.wxh[1])
        if getattr(self, 'legend_info', None) is None or self._legend_region_ is None: return
        self._dl_legend_ = self.p2s.legendRenderDL(self.wxh, self._legend_region_, self.legend_spec,
                                                   self.legend_info, self.txt_h)
        self.svg_legend  = self._dl_legend_.svg()

    #
    # __renderSVG__()
    #
    def __renderSVG__(self, _randid_):
        self._gpu_payload_ = self._gpu_dl_ = None   # invalidate GPU state cached from a template
        w, h                 = self.wxh
        _color_default_      = self.p2s.colorTyped('data',       'default')
        _background_default_ = self.p2s.colorTyped('background', 'default')
        if self.dot_size_orig is not None:
            if isinstance(self.dot_size_orig, int):
                _svg_style_ = f'<style> .rect-group-{_randid_} rect {{ width: {self.dot_size_orig}px; height: {self.dot_size_orig}px; }} </style>'
                _svg_plot_  = f'''<g class="rect-group-{_randid_}" stroke="none" fill="{_color_default_}">''' + ''.join(self.df_pixels['__svg__']) + '''</g>'''
            else:
                if isinstance(self.dot_size_orig, float): _svg_style_ = f'<style> .circle-group-{_randid_} circle {{ r: {self.dot_size_orig}px; }} </style>'
                else:                                     _svg_style_ = f'<style> .circle-group-{_randid_} </style>'
                _svg_plot_  = f'''<g class="circle-group-{_randid_}" stroke="none" fill="{_color_default_}" {self.clip_url}>''' + ''.join(self.df_pixels['__svg__']) + '''</g>'''
        else:
            _svg_style_, _svg_plot_ = '', ''

        self.svg = f'''<svg id="xyp_{_randid_}" x="0" y="0" width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">
                        {self.svg_defs}
                        {_svg_style_}
                        <rect x="0" y="0" width="{w}" height="{h}" fill="{_background_default_}" stroke="{_background_default_}" />
                        {self.svg_background}{self.svg_context} {self.svg_lines} {_svg_plot_} {self.svg_distributions} {self.svg_legend}</svg>'''

    #
    # renderSmallMultiples() - render small multiples and return a lookup table to the rendering
    # - df_all  = the full dataframe
    # - df_lu   = lookup table to each dataframe that requires rendering
    # - all_key = the key for the full dataframe (i.e. '__all__') or None if the full dataframe is not required
    #
    # - return a lookup table w/ the original df_lu keys pointing to the rendered small multiples
    #
    def renderSmallMultiples(self, df_all, df_lu, all_key):
        _kwargs_shared_ = {}   # kwargs for ALL instances (including all_key)
        _kwargs_subset_ = {}   # kwargs for non-all instances only (SM_COUNT / SM_COLOR)

        # SM_X / SM_Y: compute global axis ranges from a reference instance built on df_all.
        if (self.p2s.SM_X in self.sm_shared or self.p2s.SM_Y in self.sm_shared):
            _ref_ = self.p2s.xyp(df=df_all, template=self)
            if self.p2s.SM_X in self.sm_shared:
                if self.p2s.dateColumn(_ref_.df_flat, '__x__') or self.p2s.dateTimeColumn(_ref_.df_flat, '__x__'):
                    _kwargs_shared_['x_range'] = (_ref_.x_range if _ref_.x_range is not None
                                                  else (_ref_.df_flat['__x__'].min(), _ref_.df_flat['__x__'].max()))
                else:
                    _xmin_ = _ref_.x_transform_vars[1]
                    _kwargs_shared_['x_range'] = (_xmin_, _xmin_ + _ref_.x_transform_vars[2])
            if self.p2s.SM_Y in self.sm_shared:
                if self.p2s.dateColumn(_ref_.df_flat, '__y__') or self.p2s.dateTimeColumn(_ref_.df_flat, '__y__'):
                    _kwargs_shared_['y_range'] = (_ref_.y_range if _ref_.y_range is not None
                                                  else (_ref_.df_flat['__y__'].min(), _ref_.df_flat['__y__'].max()))
                else:
                    _ymin_ = _ref_.y_transform_vars[1]
                    _kwargs_shared_['y_range'] = (_ymin_, _ymin_ + _ref_.y_transform_vars[2])
            # Pass global boundary labels so all sub-instances show the same axis min/max text.
            if self.p2s.SM_X in self.sm_shared and len(_ref_.df_flat) > 0:
                _xmin_lbl_ = _ref_.df_flat['__x__'][_ref_.df_flat['__xi__'].arg_min()]
                _xmax_lbl_ = _ref_.df_flat['__x__'][_ref_.df_flat['__xi__'].arg_max()]
                if isinstance(_xmin_lbl_, dict): _xmin_lbl_ = '|'.join([str(_) for _ in list(_xmin_lbl_.values())])
                if isinstance(_xmax_lbl_, dict): _xmax_lbl_ = '|'.join([str(_) for _ in list(_xmax_lbl_.values())])
                _kwargs_shared_['x_shared_label_range'] = (_xmin_lbl_, _xmax_lbl_)
            if self.p2s.SM_Y in self.sm_shared and len(_ref_.df_flat) > 0:
                _ymin_lbl_ = _ref_.df_flat['__y__'][_ref_.df_flat['__yi__'].arg_min()]
                _ymax_lbl_ = _ref_.df_flat['__y__'][_ref_.df_flat['__yi__'].arg_max()]
                if isinstance(_ymin_lbl_, dict): _ymin_lbl_ = '|'.join([str(_) for _ in list(_ymin_lbl_.values())])
                if isinstance(_ymax_lbl_, dict): _ymax_lbl_ = '|'.join([str(_) for _ in list(_ymax_lbl_.values())])
                _kwargs_shared_['y_shared_label_range'] = (_ymin_lbl_, _ymax_lbl_)

        # SM_COUNT / SM_COLOR: two-pass approach.
        # Pass 1 renders all non-all instances so we can collect the per-pixel aggregated values;
        # Pass 2 re-renders with the global normalization injected.
        _needs_shared_count_ = self.p2s.SM_COUNT in self.sm_shared
        _needs_shared_color_ = self.p2s.SM_COLOR in self.sm_shared

        if _needs_shared_count_ or _needs_shared_color_:
            _pass1_lu_ = {k: self.p2s.xyp(df=v, template=self, **_kwargs_shared_)
                          for k, v in df_lu.items() if k != all_key}

            if _needs_shared_count_ and not isinstance(self.dot_size_orig, int) \
                    and self.dot_size is not None:
                _ds_cols_ = [inst.df_pixels['__dot_size_sum__']
                             for inst in _pass1_lu_.values()
                             if '__dot_size_sum__' in inst.df_pixels.columns]
                if _ds_cols_:
                    _combined_ = pl.concat(_ds_cols_)
                    _kwargs_subset_['dot_size_global_min'] = float(_combined_.min())
                    _kwargs_subset_['dot_size_global_max'] = float(_combined_.max())

            if _needs_shared_color_:
                _cs_cols_ = [inst.df_pixels['__color_sum__']
                             for inst in _pass1_lu_.values()
                             if '__color_sum__' in inst.df_pixels.columns]
                if _cs_cols_:
                    _combined_        = pl.concat(_cs_cols_)
                    _color_mode_      = self.__determineColoringMode__()
                    _magnitude_modes_ = {self.p2s.CROW_MAGNITUDEp,   self.p2s.CSET_MAGNITUDEp,
                                         self.p2s.CMAGNITUDE_SUMp,    self.p2s.CMAGNITUDE_MINp,
                                         self.p2s.CMAGNITUDE_MEDIANp, self.p2s.CMAGNITUDE_MEANp,
                                         self.p2s.CMAGNITUDE_MAXp}
                    _stretched_modes_ = {self.p2s.CROW_STRETCHEDp,   self.p2s.CSET_STRETCHEDp,
                                         self.p2s.CSTRETCHED_SUMp,    self.p2s.CSTRETCHED_MINp,
                                         self.p2s.CSTRETCHED_MEDIANp, self.p2s.CSTRETCHED_MEANp,
                                         self.p2s.CSTRETCHED_MAXp}
                    if _color_mode_ in _magnitude_modes_:
                        _kwargs_subset_['color_magnitude_min'] = float(_combined_.min())
                        _kwargs_subset_['color_magnitude_max'] = float(_combined_.max())
                    elif _color_mode_ in _stretched_modes_:
                        _unique_sorted_ = sorted(_combined_.drop_nulls().unique().to_list())
                        _kwargs_subset_['color_stretched_global_values'] = _unique_sorted_

            _render_lu_ = {}
            for k, v in df_lu.items():
                _kw_ = dict(_kwargs_shared_)
                if k != all_key:
                    _kw_.update(_kwargs_subset_)
                _render_lu_[k] = self.p2s.xyp(df=v, template=self, **_kw_)
            return _render_lu_

        # No SM_COUNT or SM_COLOR: single pass (possibly with shared x/y ranges)
        return {k: self.p2s.xyp(df=v, template=self, **_kwargs_shared_) for k, v in df_lu.items()}

    def render_with(self, df, **overrides):
        return self.p2s.xyp(df=df, template=self, **overrides)

    #
    # methods supporting interactive operations
    #
    def filterByRectangle(self, bounding_box, remove_records=False):
        _x0_, _y0_, _x1_, _y1_ = bounding_box
        if _x0_ > _x1_: _x0_, _x1_ = _x1_, _x0_
        if _y0_ > _y1_: _y0_, _y1_ = _y1_, _y0_
        _df_filtered_ = self.df_flat.filter((pl.col('__xpx__') >= _x0_) &
                                            (pl.col('__xpx__') <= _x1_) &
                                            (pl.col('__ypx__') >= _y0_) &
                                            (pl.col('__ypx__') <= _y1_))                
        _df_filtered_ = _df_filtered_.drop(set(_df_filtered_.columns) - set(['__p2s_index__']))
        if remove_records: return self.df.join(_df_filtered_, on='__p2s_index__', how='anti').drop('__p2s_index__')
        else:              return self.df.join(_df_filtered_, on='__p2s_index__')            .drop('__p2s_index__')

    def filterByOval(self, oval, remove_records=False):
        _cx_, _cy_, _rx_, _ry_ = oval
        # A plain click arrives as a zero-radius oval: keep it covering the pixel under the cursor.
        _rx_, _ry_ = max(float(_rx_), 0.5), max(float(_ry_), 0.5)
        _df_filtered_ = self.df_flat.filter(
            (((pl.col('__xpx__') - _cx_) / _rx_).pow(2) +
             ((pl.col('__ypx__') - _cy_) / _ry_).pow(2)) <= 1.0
        )
        _df_filtered_ = _df_filtered_.drop(set(_df_filtered_.columns) - set(['__p2s_index__']))
        if remove_records: return self.df.join(_df_filtered_, on='__p2s_index__', how='anti').drop('__p2s_index__')
        else:              return self.df.join(_df_filtered_, on='__p2s_index__')            .drop('__p2s_index__')

    def filterByColorAtXY(self, xy, remove_records=False, distance_threshold=2.0):
        _x_, _y_ = xy

        # No hexcolor column means all dots share the same default color.
        if '__hexcolor__' not in self.df_pixels.columns:
            _df_all_idx_ = self.df_flat.drop(set(self.df_flat.columns) - set(['__p2s_index__']))
            if remove_records: return self.df.join(_df_all_idx_, on='__p2s_index__', how='anti').drop('__p2s_index__')
            else:              return self.df.join(_df_all_idx_, on='__p2s_index__')            .drop('__p2s_index__')

        # Find the pixel nearest to the given coordinate and read its color.
        _df_near_ = self.df_pixels.with_columns(
            ((pl.col('__xpx__') - _x_).pow(2) + (pl.col('__ypx__') - _y_).pow(2)).alias('__dist2__')
        ).sort('__dist2__').head(1)

        if _df_near_['__dist2__'][0] ** 0.5 > distance_threshold:
            return None

        _color_at_xy_ = _df_near_['__hexcolor__'][0]

        # Collect all pixel positions that share that color.
        _df_same_color_ = self.df_pixels.filter(pl.col('__hexcolor__') == _color_at_xy_).select(['__xpx__', '__ypx__'])

        # Map those pixel positions back to source records via df_flat.
        _df_filtered_ = self.df_flat.join(_df_same_color_, on=['__xpx__', '__ypx__'], how='inner')
        _df_filtered_ = _df_filtered_.drop(set(_df_filtered_.columns) - set(['__p2s_index__']))

        if remove_records: return self.df.join(_df_filtered_, on='__p2s_index__', how='anti').drop('__p2s_index__')
        else:              return self.df.join(_df_filtered_, on='__p2s_index__')            .drop('__p2s_index__')

    def recordsAt(self, xy, shape=None, threshold=2.0):
        if shape is None: shape = self.p2s.SELECT_CIRCLEp # SELECT_HORIZONTALp, SELECT_VERTICALp
        _x_, _y_ = xy
        if   shape == self.p2s.SELECT_CIRCLEp:
            # __xpx__/__ypx__ are Int32; square them in Float64 so a distant query point
            # can't overflow i32 into a negative "distance" that spuriously passes the test.
            _df_filtered_ = self.df_flat.filter(
                ((pl.col('__xpx__').cast(pl.Float64) - _x_).pow(2) +
                 (pl.col('__ypx__').cast(pl.Float64) - _y_).pow(2)) <= threshold ** 2
            )
        elif shape == self.p2s.SELECT_HORIZONTALp:
            _df_filtered_ = self.df_flat.filter(
                (pl.col('__ypx__') - _y_).abs() <= threshold
            )
        elif shape == self.p2s.SELECT_VERTICALp:
            _df_filtered_ = self.df_flat.filter(
                (pl.col('__xpx__') - _x_).abs() <= threshold
            )
        else:
            raise ValueError(f'recordsAt(): unknown shape {shape}')
        _df_filtered_ = _df_filtered_.drop(set(_df_filtered_.columns) - set(['__p2s_index__']))
        return self.df.join(_df_filtered_, on='__p2s_index__').drop('__p2s_index__')

