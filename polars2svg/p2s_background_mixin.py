import math
import xml.etree.ElementTree as ET  # nosec B405 - background= shape descriptors are trusted caller config, not untrusted data; see SECURITY.md

from polars2svg.exceptions      import DataError, InvalidSpecError, Polars2SVGError
from polars2svg.p2s_displaylist import DisplayList, pathToDL, strokePolylineDL, dashArrayToTuple


#
# INHERIT - "use the component's background_* parameter for this field".
#
# The load-bearing distinction in a background record is INHERIT ("defer") vs None
# ("explicitly off").  It cannot be None (that is the other meaning) and it cannot be a
# string ('inherit' would collide with the 'vary' / 'default' / '#rrggbb' vocabulary the
# colour fields already use), so it is a singleton of its own type.
#
class _InheritSentinel_:
    _instance_ = None
    def __new__(cls):
        if cls._instance_ is None: cls._instance_ = super().__new__(cls)
        return cls._instance_
    def __repr__(self):  return 'INHERIT'
    def __reduce__(self): return (_InheritSentinel_, ())

INHERIT = _InheritSentinel_()


#
# BackgroundShape - one background entry: a shape plus the appearance it wants.
#
# Before this existed, appearance was keyed by name in five parallel dicts beside the
# geometry (background_fill / _opacity / _stroke / _stroke_w / _label_color), which made
# absence unstateable -- a name missing from background_fill was filled with the axis
# colour rather than left unfilled -- so callers suppressed a fill with fill-opacity 0 and
# a stroke with stroke-width 0.  A record can say None.
#
# Every field defaults to INHERIT, so BackgroundShape(shape_desc) is exactly the old bare
# descriptor: that is what makes the two forms mix freely in one background= dict.
#
# Frozen after construction.  _copy_mutable_containers_() (polars2svg.py) copies dict /
# list / set containers when a template is cloned and shares everything else by reference,
# so a mutable record would leak edits from a clone back into its template and every
# sibling -- the exact class of bug that function was written to kill.
#
class BackgroundShape:
    _FIELDS_ = ('shape', 'fill', 'fill_opacity', 'stroke', 'stroke_opacity', 'stroke_width',
                'dash', 'stroke_linecap', 'stroke_linejoin', 'label', 'label_color')

    def __init__(self, shape, fill=INHERIT, fill_opacity=INHERIT, stroke=INHERIT,
                 stroke_opacity=INHERIT, stroke_width=INHERIT, dash=INHERIT,
                 stroke_linecap=INHERIT, stroke_linejoin=INHERIT,
                 label=INHERIT, label_color=INHERIT):
        # written straight into __dict__ so __setattr__ can refuse unconditionally
        _d_ = self.__dict__
        _d_['shape']          = shape
        _d_['fill']           = fill
        _d_['fill_opacity']   = fill_opacity
        _d_['stroke']         = stroke
        _d_['stroke_opacity'] = stroke_opacity
        _d_['stroke_width']   = stroke_width
        _d_['dash']           = dash
        _d_['stroke_linecap'] = stroke_linecap
        _d_['stroke_linejoin'] = stroke_linejoin
        _d_['label']          = label
        _d_['label_color']    = label_color

    def __setattr__(self, name, value):
        raise AttributeError(f'BackgroundShape is immutable (attempted to set {name!r}); '
                             f'build a new one with p2s.bgShape(...)')
    def __delattr__(self, name):
        raise AttributeError('BackgroundShape is immutable')

    def __repr__(self):
        _set_ = [f'{_f_}={getattr(self, _f_)!r}' for _f_ in self._FIELDS_[1:]
                 if getattr(self, _f_) is not INHERIT]
        return f'BackgroundShape({type(self.shape).__name__}{", " if _set_ else ""}{", ".join(_set_)})'


#
# _BackgroundStyle_ - a record with every INHERIT resolved against the component's
# background_* parameters: concrete colour strings / numbers, or None for "off".
# What both writers (SVG and display list) consume; neither reads the other's output.
#
class _BackgroundStyle_:
    __slots__ = ('fill', 'fill_opacity', 'stroke', 'stroke_opacity', 'stroke_width',
                 'dash', 'linecap', 'linejoin', 'label', 'label_color')
    def __repr__(self):
        return '_BackgroundStyle_(' + ', '.join(f'{_s_}={getattr(self, _s_)!r}' for _s_ in self.__slots__) + ')'


class P2SBackgroundMixin:
    #
    # P2SBackgroundMixin - shared background-shape transform/render helpers for
    # the coordinate-plane components (XYp, LinkP). These methods were duplicated in
    # both components, differing only in (a) the world->screen coordinate hook each
    # used and (b) the component's own name in error messages. The coordinate hook is
    # abstracted behind __bgX__/__bgY__, which each component defines to delegate to
    # its own transform (XYp: wxToSx/wyToSy; LinkP: xT/yT); the name comes from
    # _COMPONENT_NAME_.
    #
    # The component still owns __shapelyToSVGPath__ (called here via self).
    #
    # Pipeline, once per background= entry:
    #   __normalizeBackgroundEntry__  bare descriptor | dict | BackgroundShape -> record
    #   __resolveBackgroundStyle__    record + background_* parameters -> _BackgroundStyle_
    #   __transformBackgroundShapes__ world -> screen geometry, SVG for it, label for it
    #   __backgroundShapeToDL__       the SAME geometry + style -> GPU primitives
    # The last two both read the resolved style; neither parses the other's output.
    #
    def __bgX__(self, _v_):
        raise NotImplementedError   # each component overrides with its world->screen X
    def __bgY__(self, _v_):
        raise NotImplementedError   # each component overrides with its world->screen Y

    #
    # __bgMinsAndMaxes__() - update a bounding box with a new point
    #
    def __bgMinsAndMaxes__(self, x, y, x0, y0, x1, y1):
        if x0 is None:
            return x, y, x, y
        return min(x, x0), min(y, y0), max(x, x1), max(y, y1)

    #
    # __normalizeBackgroundEntry__() - every accepted background= value -> a BackgroundShape
    # - a bare shape descriptor becomes a record whose every field is INHERIT, which is
    #   what makes bare descriptors and records mix freely inside one dict
    # - a plain dict is the record form for producers that would rather not import
    #   anything (a layout returning cells); unambiguous, since a dict was never a
    #   valid shape descriptor
    #
    def __normalizeBackgroundEntry__(self, name, value):
        if isinstance(value, BackgroundShape): return value
        if isinstance(value, dict):
            _unknown_ = [_k_ for _k_ in value if _k_ not in BackgroundShape._FIELDS_]
            if _unknown_:
                raise InvalidSpecError(f'{self._COMPONENT_NAME_}.background["{name}"] - unknown record '
                                       f'field(s) {sorted(_unknown_)}; valid fields are {list(BackgroundShape._FIELDS_)}')
            if 'shape' not in value:
                raise InvalidSpecError(f'{self._COMPONENT_NAME_}.background["{name}"] - a background record '
                                       f'requires a "shape"')
            return BackgroundShape(**value)
        return BackgroundShape(value)

    #
    # __resolveBackgroundColor__() - the one colour ladder for fill, stroke and label
    # - dict keyed by name -> 'vary' -> HexColorString -> the axis-inner fallback
    #   ('default', and anything unrecognized, lands on the fallback)
    # - fill and stroke used to test these in different orders; harmless while the types
    #   stay disjoint, but the kind of divergence a single resolver removes rather than
    #   documents
    #
    def __resolveBackgroundColor__(self, value, name):
        if isinstance(value, dict) and name in value:              return value[name]
        if value == 'vary':                                        return self.p2s.color(name)
        if isinstance(value, self.p2s.HexColorString):             return value
        return self.p2s.colorTyped('axis', 'inner')

    #
    # __resolveBackgroundStyle__() - a record + the background_* parameters -> concrete style
    #
    # The background_* parameters are what INHERIT means; they keep their own (older)
    # semantics, in which background_fill=None / background_opacity=None means "no fill"
    # and background_stroke=None / background_stroke_w=None means "no stroke".  A record
    # field is uniform instead: None = off / omit the attribute.
    #
    def __resolveBackgroundStyle__(self, name, record):
        _st_ = _BackgroundStyle_()

        # ---- fill (background_fill + background_opacity) -----------------------------
        _fill_on_ = self.background_fill is not None and self.background_opacity is not None
        _st_.fill = self.__resolveBackgroundColor__(self.background_fill, name) if _fill_on_ else None
        if self.background_opacity is None:
            _st_.fill_opacity = None
        elif isinstance(self.background_opacity, dict):
            _st_.fill_opacity = self.background_opacity.get(name, 1.0)
        else:
            _st_.fill_opacity = self.background_opacity

        # ---- stroke (background_stroke + background_stroke_w) ------------------------
        _stroke_on_ = self.background_stroke is not None and self.background_stroke_w is not None
        _st_.stroke = self.__resolveBackgroundColor__(self.background_stroke, name) if _stroke_on_ else None
        if self.background_stroke_w is None:
            _st_.stroke_width = None
        elif isinstance(self.background_stroke_w, dict):
            # .get(), not the bare dict: a name missing from the dict used to fall through
            # to the dict OBJECT and emit stroke-width="{'a': 1.0, ...}"
            _st_.stroke_width = self.background_stroke_w.get(name, 1.0)
        else:
            _st_.stroke_width = self.background_stroke_w
        _st_.stroke_opacity = None   # no sidecar parameter -- records only
        _st_.dash = _st_.linecap = _st_.linejoin = None

        # ---- label (background_label_color) ------------------------------------------
        _st_.label       = name if self.background_label_color is not None else None
        _st_.label_color = (self.__resolveBackgroundColor__(self.background_label_color, name)
                            if self.background_label_color is not None else None)

        # ---- record overrides ---------------------------------------------------------
        if record.fill           is not INHERIT: _st_.fill           = None if record.fill   is None else self.__resolveBackgroundColor__(record.fill, name)
        if record.fill_opacity   is not INHERIT: _st_.fill_opacity   = record.fill_opacity
        if record.stroke         is not INHERIT: _st_.stroke         = None if record.stroke is None else self.__resolveBackgroundColor__(record.stroke, name)
        if record.stroke_opacity is not INHERIT: _st_.stroke_opacity = record.stroke_opacity
        if record.stroke_width   is not INHERIT: _st_.stroke_width   = record.stroke_width
        if record.dash           is not INHERIT: _st_.dash           = record.dash
        if record.stroke_linecap is not INHERIT: _st_.linecap        = record.stroke_linecap
        if record.stroke_linejoin is not INHERIT: _st_.linejoin      = record.stroke_linejoin
        if record.label          is not INHERIT: _st_.label          = record.label
        if record.label_color    is not INHERIT: _st_.label_color    = (None if record.label_color is None
                                                                        else self.__resolveBackgroundColor__(record.label_color, name))

        # ---- consistency --------------------------------------------------------------
        # An off fill/stroke carries no sub-attributes; a label needs both text and colour
        # (so the interactive 'b' cycle, which drives background_label_color, still governs
        # whether labels are drawn -- a record picks the TEXT, not the presence, unless it
        # also supplies its own label_color).
        if _st_.fill   is None: _st_.fill_opacity = None
        if _st_.stroke is None: _st_.stroke_opacity = _st_.stroke_width = _st_.dash = _st_.linecap = _st_.linejoin = None
        if _st_.label  is None or _st_.label_color is None: _st_.label = None
        return _st_

    #
    # __backgroundShapeRenderDetails__() - SVG paint attributes for a resolved style
    #
    # fill=None emits fill="none" rather than omitting the attribute: background shapes are
    # emitted directly under <svg> with no ancestor carrying a fill, so an attribute-less
    # path would take SVG's initial value and render solid black.  stroke=None does omit
    # its attributes -- SVG's initial stroke is already none.
    #
    def __backgroundShapeRenderDetails__(self, style):
        if style.fill is None:
            svg = ' fill="none"'
        else:
            svg = f' fill="{style.fill}"'
            if style.fill_opacity is not None: svg += f' fill-opacity="{style.fill_opacity}"'
        if style.stroke is not None:
            svg += f' stroke="{style.stroke}"'
            if style.stroke_opacity is not None: svg += f' stroke-opacity="{style.stroke_opacity}"'
            if style.stroke_width   is not None: svg += f' stroke-width="{style.stroke_width}"'
            if style.dash           is not None: svg += f' stroke-dasharray="{style.dash}"'
            if style.linecap        is not None: svg += f' stroke-linecap="{style.linecap}"'
            if style.linejoin       is not None: svg += f' stroke-linejoin="{style.linejoin}"'
        return svg

    #
    # __backgroundShapeLabel__() - centred text label over the shape bounding box
    # - returns (svg, dl_args); dl_args feeds the display list directly rather than
    #   being regexed back out of the svg
    #
    def __backgroundShapeLabel__(self, style, x0, y0, x1, y1):
        if style.label is None or x0 is None:
            return '', None
        _cx_ = (x0 + x1) / 2
        _cy_ = self.txt_h / 2 + (y0 + y1) / 2
        return ((f'<text x="{_cx_}" y="{_cy_}" text-anchor="middle" '
                 f'font-family="{self.p2s.default_font}" fill="{style.label_color}" font-size="{self.txt_h}px">'
                 f'{style.label}</text>'),
                (style.label, _cx_, _cy_, style.label_color))

    #
    # __transformCircleSVG__() - transform a <circle> SVG element into an <ellipse> in screen coordinates
    #
    def __transformCircleSVG__(self, shape_desc, style):
        _root_ = ET.fromstring(shape_desc)  # nosec B314 - trusted caller config (background= shape descriptor), not untrusted data; see SECURITY.md
        cx  = float(_root_.attrib['cx'])
        cy  = float(_root_.attrib['cy'])
        r   = float(_root_.attrib['r'])
        cx_s = self.__bgX__(cx)
        cy_s = self.__bgY__(cy)
        rx_s = abs(self.__bgX__(r + cx) - cx_s)
        ry_s = abs(self.__bgY__(r + cy) - cy_s)
        svg  = f'<ellipse cx="{cx_s}" cy="{cy_s}" rx="{rx_s}" ry="{ry_s}"'
        svg += self.__backgroundShapeRenderDetails__(style)
        _label_svg_, _label_dl_ = self.__backgroundShapeLabel__(style, cx_s - rx_s, cy_s - ry_s,
                                                                cx_s + rx_s, cy_s + ry_s)
        return svg + '/>', _label_svg_, ('ellipse', (cx_s, cy_s, rx_s, ry_s)), _label_dl_

    #
    # __transformPathDescription__() - transform an SVG path description string into screen coordinates
    #
    def __transformPathDescription__(self, shape_desc, style):
        _d_ = ''
        x0, y0, x1, y1 = None, None, None, None
        tokens = ' '.join(shape_desc.split()).split(' ')
        i = 0
        while i < len(tokens):
            if tokens[i] == 'M':
                _x, _y = self.__bgX__(float(tokens[i+1])), self.__bgY__(float(tokens[i+2]))
                _d_ += f' M {_x} {_y}'
                x0, y0, x1, y1 = self.__bgMinsAndMaxes__(_x, _y, x0, y0, x1, y1)
                i += 3
            elif tokens[i] == 'L':
                _x, _y = self.__bgX__(float(tokens[i+1])), self.__bgY__(float(tokens[i+2]))
                _d_ += f' L {_x} {_y}'
                x0, y0, x1, y1 = self.__bgMinsAndMaxes__(_x, _y, x0, y0, x1, y1)
                i += 3
            elif tokens[i] == 'C':
                _xcp1, _ycp1 = self.__bgX__(float(tokens[i+1])), self.__bgY__(float(tokens[i+2]))
                _xcp2, _ycp2 = self.__bgX__(float(tokens[i+3])), self.__bgY__(float(tokens[i+4]))
                _x,    _y    = self.__bgX__(float(tokens[i+5])), self.__bgY__(float(tokens[i+6]))
                _d_ += f' C {_xcp1} {_ycp1} {_xcp2} {_ycp2} {_x} {_y}'
                x0, y0, x1, y1 = self.__bgMinsAndMaxes__(_x,    _y,    x0, y0, x1, y1)
                x0, y0, x1, y1 = self.__bgMinsAndMaxes__(_xcp1, _ycp1, x0, y0, x1, y1)
                x0, y0, x1, y1 = self.__bgMinsAndMaxes__(_xcp2, _ycp2, x0, y0, x1, y1)
                i += 7
            elif tokens[i] == 'Z':
                _d_ += ' Z'
                i += 1
            else:
                raise Polars2SVGError(f'{self._COMPONENT_NAME_}.__transformPathDescription__() - unhandled path token "{tokens[i]}"')
        svg = f'<path d="{_d_}"' + self.__backgroundShapeRenderDetails__(style)
        _label_svg_, _label_dl_ = self.__backgroundShapeLabel__(style, x0, y0, x1, y1)
        return svg + '/>', _label_svg_, ('path', _d_), _label_dl_

    #
    # __transformPointsList__() - transform a list of (x, y) tuples into a screen-coordinate SVG path
    #
    def __transformPointsList__(self, points_list, style):
        _x, _y = self.__bgX__(points_list[0][0]), self.__bgY__(points_list[0][1])
        _d_ = f'M {_x} {_y}'
        x0, y0, x1, y1 = _x, _y, _x, _y
        for i in range(1, len(points_list)):
            _x, _y = self.__bgX__(points_list[i][0]), self.__bgY__(points_list[i][1])
            _d_ += f' L {_x} {_y}'
            x0, y0, x1, y1 = self.__bgMinsAndMaxes__(_x, _y, x0, y0, x1, y1)
        _d_ += ' Z'
        svg = f'<path d="{_d_}"' + self.__backgroundShapeRenderDetails__(style)
        _label_svg_, _label_dl_ = self.__backgroundShapeLabel__(style, x0, y0, x1, y1)
        return svg + '/>', _label_svg_, ('path', _d_), _label_dl_

    #
    # __transformBackgroundShapes__() - dispatch a background shape to the appropriate transform method
    # - returns (shape_svg, label_svg, geometry, label_dl_args); geometry is the SCREEN-space
    #   ('path', d) or ('ellipse', (cx, cy, rx, ry)) that the display-list writer consumes
    #
    def __transformBackgroundShapes__(self, name, record, style):
        shape_desc = record.shape
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
            # An open line has no interior worth painting -- SVG would close the subpath
            # implicitly in order to fill it.  Forced off unless the record asked for a
            # fill explicitly, in which case the caller outranks the coercion.
            if record.fill is INHERIT:
                style.fill, style.fill_opacity = None, None
        if _has_shapely_ and isinstance(shape_desc, GeometryCollection):
            if len(shape_desc.geoms) > 0:
                raise DataError(f'{self._COMPONENT_NAME_}.__transformBackgroundShapes__() - non-empty GeometryCollection not supported')
            return '', '', None, None

        if isinstance(shape_desc, str):
            if shape_desc.lower().startswith('<circle'):
                return self.__transformCircleSVG__(shape_desc, style)
            else:
                return self.__transformPathDescription__(shape_desc, style)
        elif isinstance(shape_desc, list):
            return self.__transformPointsList__(shape_desc, style)
        else:
            raise DataError(f'{self._COMPONENT_NAME_}.__transformBackgroundShapes__() - unsupported type "{type(shape_desc)}"')

    #
    # __renderBackground__() - render background= into self.svg_background and self._dl_background_
    #
    # Draw order is dict insertion order, all shapes then all labels -- documented contract,
    # not incidental: a producer emitting several layers depends on it to stack them.
    #
    def __renderBackground__(self):
        self.svg_background  = ''
        self._dl_background_ = DisplayList(self.wxh[0], self.wxh[1])
        if self.background is None:
            return
        _shapes_, _labels_, _label_dls_ = [], [], []
        for _name_, _value_ in self.background.items():
            _record_ = self.__normalizeBackgroundEntry__(_name_, _value_)
            _style_  = self.__resolveBackgroundStyle__(_name_, _record_)
            _s_, _l_, _geom_, _label_dl_ = self.__transformBackgroundShapes__(_name_, _record_, _style_)
            _shapes_.append(_s_)
            _labels_.append(_l_)
            _label_dls_.append(_label_dl_)
            if _geom_ is not None:
                self.__backgroundShapeToDL__(_geom_, _style_, self._dl_background_)
        for _label_dl_ in _label_dls_:
            self.__backgroundLabelToDL__(_label_dl_, self._dl_background_)
        self.svg_background = ''.join(_shapes_) + ''.join(_labels_)

    #
    # __backgroundShapeToDL__() - GPU geometry for a background shape, from the same
    # resolved style the SVG writer used
    #
    # This used to regex fill / fill-opacity / stroke / stroke-width back out of the SVG
    # string built moments earlier -- the same class of defect as the svgToDisplayList()
    # re-parse route removed from spreadlinesp (PLANNING.md §8).  Both writers now read
    # the record, so stroke-opacity and dash need no regex to reach the GPU path.
    #
    def __backgroundShapeToDL__(self, geometry, style, dl):
        if geometry is None: return
        _kind_, _payload_ = geometry
        _fo_     = 1.0 if style.fill_opacity   is None else float(style.fill_opacity)
        _so_     = 1.0 if style.stroke_opacity is None else float(style.stroke_opacity)
        _sw_     = 1.0 if style.stroke_width   is None else float(style.stroke_width)
        _fill_   = None if (style.fill is None or style.fill == 'none' or _fo_ <= 0.0) else style.fill
        _stroke_ = None if (style.stroke is None or style.stroke == 'none') else style.stroke
        _dash_   = dashArrayToTuple(style.dash)
        if _kind_ == 'ellipse':
            cx, cy, rx, ry = _payload_
            _pts_ = [(cx + rx*math.cos(2*math.pi*i/48), cy + ry*math.sin(2*math.pi*i/48)) for i in range(48)]
            if _fill_   is not None: dl.polygon(_pts_, _fill_, opacity=_fo_)
            if _stroke_ is not None: strokePolylineDL(dl, _pts_ + [_pts_[0]], _stroke_,
                                                      width=_sw_, opacity=_so_, dash=_dash_)
        else:
            pathToDL(dl, _payload_, fill=_fill_, stroke=_stroke_, width=_sw_,
                     fill_opacity=_fo_, stroke_opacity=_so_, dash=_dash_)

    #
    # __backgroundLabelToDL__() - GPU glyphs for a background label (also from the
    # resolved values, not from a regex over the <text> element just emitted)
    #
    def __backgroundLabelToDL__(self, label_dl, dl):
        if label_dl is None: return
        _txt_, _x_, _y_, _co_ = label_dl
        dl.text(self.p2s, _txt_, _x_, _y_, txt_h=self.txt_h, anchor='middle', color=_co_, svg='')
