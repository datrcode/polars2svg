import xml.etree.ElementTree as ET  # nosec B405 - background= shape descriptors are trusted caller config, not untrusted data; see SECURITY.md

from polars2svg.exceptions import DataError, Polars2SVGError


class P2SBackgroundMixin:
    #
    # P2SBackgroundMixin - shared background-shape transform/render helpers for
    # the coordinate-plane components (XYp, LinkP). These seven methods were
    # duplicated in both components, differing only in (a) the world->screen
    # coordinate hook each used and (b) the component's own name in error
    # messages. The coordinate hook is now abstracted behind __bgX__/__bgY__,
    # which each component defines to delegate to its own transform (XYp:
    # wxToSx/wyToSy; LinkP: xT/yT); the name comes from _COMPONENT_NAME_.
    #
    # The component still owns the parts that genuinely differ: __shapelyToSVGPath__
    # (called here via self) and __renderBackground__ / the GPU display-list paths.
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
        cx_s = self.__bgX__(cx)
        cy_s = self.__bgY__(cy)
        rx_s = abs(self.__bgX__(r + cx) - cx_s)
        ry_s = abs(self.__bgY__(r + cy) - cy_s)
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
                _x, _y = self.__bgX__(float(tokens[i+1])), self.__bgY__(float(tokens[i+2]))
                svg += f' M {_x} {_y}'
                x0, y0, x1, y1 = self.__bgMinsAndMaxes__(_x, _y, x0, y0, x1, y1)
                i += 3
            elif tokens[i] == 'L':
                _x, _y = self.__bgX__(float(tokens[i+1])), self.__bgY__(float(tokens[i+2]))
                svg += f' L {_x} {_y}'
                x0, y0, x1, y1 = self.__bgMinsAndMaxes__(_x, _y, x0, y0, x1, y1)
                i += 3
            elif tokens[i] == 'C':
                _xcp1, _ycp1 = self.__bgX__(float(tokens[i+1])), self.__bgY__(float(tokens[i+2]))
                _xcp2, _ycp2 = self.__bgX__(float(tokens[i+3])), self.__bgY__(float(tokens[i+4]))
                _x,    _y    = self.__bgX__(float(tokens[i+5])), self.__bgY__(float(tokens[i+6]))
                svg += f' C {_xcp1} {_ycp1} {_xcp2} {_ycp2} {_x} {_y}'
                x0, y0, x1, y1 = self.__bgMinsAndMaxes__(_x,    _y,    x0, y0, x1, y1)
                x0, y0, x1, y1 = self.__bgMinsAndMaxes__(_xcp1, _ycp1, x0, y0, x1, y1)
                x0, y0, x1, y1 = self.__bgMinsAndMaxes__(_xcp2, _ycp2, x0, y0, x1, y1)
                i += 7
            elif tokens[i] == 'Z':
                svg += ' Z'
                i += 1
            else:
                raise Polars2SVGError(f'{self._COMPONENT_NAME_}.__transformPathDescription__() - unhandled path token "{tokens[i]}"')
        svg += '"'
        svg += self.__backgroundShapeRenderDetails__(name, bg_shape_opacity, bg_shape_fill,
                                                     bg_shape_stroke_w, bg_shape_stroke)
        return svg + '/>', self.__backgroundShapeLabel__(name, x0, y0, x1, y1, bg_shape_label_color)

    #
    # __transformPointsList__() - transform a list of (x, y) tuples into a screen-coordinate SVG path
    #
    def __transformPointsList__(self, name, points_list, bg_shape_label_color, bg_shape_opacity,
                                 bg_shape_fill, bg_shape_stroke_w, bg_shape_stroke):
        _x, _y = self.__bgX__(points_list[0][0]), self.__bgY__(points_list[0][1])
        svg = f'<path d="M {_x} {_y}'
        x0, y0, x1, y1 = _x, _y, _x, _y
        for i in range(1, len(points_list)):
            _x, _y = self.__bgX__(points_list[i][0]), self.__bgY__(points_list[i][1])
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
                raise DataError(f'{self._COMPONENT_NAME_}.__transformBackgroundShapes__() - non-empty GeometryCollection not supported')
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
            raise DataError(f'{self._COMPONENT_NAME_}.__transformBackgroundShapes__() - unsupported type "{type(shape_desc)}"')

