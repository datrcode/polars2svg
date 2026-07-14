import html

__name__ = 'p2s_text_mixin'

class P2STextMixin:
    def __init__(self):
        pass

    def __p2s_text_mixin_init__(self):
        self.default_font = "'Noto Sans', sans-serif"
        self._glyph_atlas_ = None

    #
    # glyphAtlas() - lazily constructed shared GlyphAtlas (GPU text); built from the
    # same bundled TTF as textLength() so GPU text layout matches SVG text layout
    #
    def glyphAtlas(self):
        if self._glyph_atlas_ is None:
            from polars2svg.p2s_glyph_atlas import GlyphAtlas
            self._glyph_atlas_ = GlyphAtlas()
        return self._glyph_atlas_

    #
    # svgText() - Render SVG Text In A Consistent Manner
    #
    def svgText(self,
                txt,
                x,
                y,
                txt_h      = 12,
                just_xy    = False,   # for the text block widget -- that will use an SVG group to consolidate the rendering
                color      = None,
                anchor     = 'start',
                font       = None,
                font_style = None,
                rotation   = None):
        if txt == '\n' or txt == '' or txt == '\r' or txt == '\t': return ''
        if font  is None: font  = self.default_font
        if color is None: color = self.colorTyped('label','defaultfg')
        txt = str(txt)

        _html_txt = html.escape(txt)
        # Spaces are deliberately NOT replaced with &nbsp; -- doing so breaks
        # JupyterLab rendering in some configurations.

        _font_style_str_ = f' font-style="{font_style}"' if font_style is not None else ''

        if   just_xy:
            return f'<text x="{x:0.2f}" y="{y:0.2f}">{_html_txt}</text>'
        elif rotation is not None:
            return f'<text x="{x}" text-anchor="{anchor}" y="{y}" font-family="{font}"{_font_style_str_} fill="{color}" font-size="{txt_h}px"' + \
                   f' transform="rotate({rotation},{x},{y})">{_html_txt}</text>'
        else:
            return f'<text x="{x}" text-anchor="{anchor}" y="{y}" font-family="{font}"{_font_style_str_} fill="{color}" font-size="{txt_h}px">{_html_txt}</text>'

    #
    # cropText() - Based on the height of the font, shorten the string to fit into a specific width...
    # ... empirically derived values for letters / so unlikely to work exactly right if the font changes
    #
    def cropText(self, txt, txt_h, w):
        # If it fits, it ships
        if self.textLength(txt,txt_h) <= w:
            return txt

        # Otherwise... iterate until it doesn't fit
        i = 1
        while self.textLength(txt[:i],txt_h) < w:
            i += 1
            
        # Assumption is the the '...' doesn't add too much...
        if i == 0:
            i += 1
        return txt[:i-1] + '...' 

    #
    # svgAxisLabels() - Render a 3-part axis label row (left / center / right) with
    # space-adaptive visibility.  Returns a list of SVG text strings; caller does
    # _svg_.extend(result).
    #
    # prefer_center=True  (timep default): keep center when LR don't fit, drop last.
    # prefer_center=False (histop default): keep LR when center doesn't fit, drop LR last.
    #
    def svgAxisLabels(self, lbl_left, lbl_center, lbl_right,
                      x0, available_w, y, txt_h,
                      color_left=None, color_center=None, color_right=None,
                      gap=None, prefer_center=True, dl=None):
        if gap is None:           gap          = txt_h * 0.5
        if color_left   is None:  color_left   = self.colorTyped('label', 'defaultfg')
        if color_center is None:  color_center = color_left
        if color_right  is None:  color_right  = color_left
        l_w = self.textLength(lbl_left,   txt_h)
        c_w = self.textLength(lbl_center, txt_h)
        r_w = self.textLength(lbl_right,  txt_h)
        _all_fit_ = (l_w + c_w / 2 + gap <= available_w / 2 and
                     c_w / 2 + r_w + gap  <= available_w / 2)
        _c_fits_  = c_w           <= available_w
        _lr_fits_ = l_w + gap + r_w <= available_w
        def _emit_(txt, x, anchor, color):
            _s_ = self.svgText(txt, x, y, txt_h=txt_h, anchor=anchor, color=color)
            if dl is not None: dl.text(self, txt, x, y, txt_h=txt_h, anchor=anchor, color=color, svg=_s_)
            return _s_
        out = []
        if _all_fit_:
            out.append(_emit_(lbl_left,   x0,                   'start',  color_left))
            out.append(_emit_(lbl_center, x0 + available_w / 2, 'middle', color_center))
            out.append(_emit_(lbl_right,  x0 + available_w,     'end',    color_right))
        elif prefer_center:
            if _c_fits_:
                out.append(_emit_(lbl_center, x0 + available_w / 2, 'middle', color_center))
        else:
            if _lr_fits_:
                out.append(_emit_(lbl_left,  x0,               'start', color_left))
                out.append(_emit_(lbl_right, x0 + available_w, 'end',   color_right))
            elif r_w <= available_w:
                out.append(_emit_(lbl_right, x0 + available_w, 'end',   color_right))
        return out

    #
    # __SignXDivUnit__() - Separate a number into its sign, value, divisor and unit (for a unitized number)
    #
    def __SignXDivUnit__(self, x):
        _sign_ = '-' if x < 0 else ''
        x      = abs(x)
        if   x < 10e2:  _div_, _unit_ = 1,     ''
        elif x < 10e5:  _div_, _unit_ = 10e2,  'K'
        elif x < 10e8:  _div_, _unit_ = 10e5,  'M'
        elif x < 10e11: _div_, _unit_ = 10e8,  'B'
        elif x < 10e14: _div_, _unit_ = 10e11, 'T'
        else:           _div_, _unit_ = 10e14, 'Q'
        return _sign_, x, _div_, _unit_

    #
    # unitizeInt_extendUntilDifferent() - format a pair of numbers, extending decimal places until they differ
    #
    def unitizeInt_extendUntilDifferent(self, a, b):
        _xsign_, _x_, _xdiv_, _xunit_ = self.__SignXDivUnit__(a)
        _ysign_, _y_, _ydiv_, _yunit_ = self.__SignXDivUnit__(b)
        _round_ = 0
        while round(_x_/_xdiv_, _round_) == round(_y_/_ydiv_, _round_) and _round_ < 4: _round_ += 1
        return _xsign_ + str(round(_x_/_xdiv_, _round_)) + _xunit_, _ysign_ + str(round(_y_/_ydiv_, _round_)) + _yunit_

    #
    # unitizeInt() - format a number with K/M/B/T/Q suffix to a target digit count
    #
    def unitizeInt(self, a, num_of_digits=5):
        _xsign_, _x_, _xdiv_, _xunit_ = self.__SignXDivUnit__(a)
        _round_ = 0
        while len(str(round(_x_/_xdiv_, _round_))) < num_of_digits:
            if _round_ > 0 and str(round(_x_/_xdiv_, _round_ - 1)) == str(round(_x_/_xdiv_, _round_)): break
            _round_ += 1
        return _xsign_ + str(round(_x_/_xdiv_, _round_)) + _xunit_

    #
    # textLength() - calculate the expected pixel width of txt rendered at txt_h points
    # Advances come from the baked NotoSans-Regular-subset.ttf table (p2s_font_metrics.py),
    # not from Pillow: getlength() answers differently depending on whether the installed
    # Pillow was built with Raqm, which made SVG output machine-dependent.  See that
    # module's header.
    #
    # The size is still quantized to an integer, matching GlyphAtlas.layoutText() so GPU
    # text and SVG text agree on every pen position.
    #
    def textLength(self, txt, txt_h):
        if not txt or txt in ('\n', '\r', '\t', ''):
            return 0
        from polars2svg.p2s_font_metrics import textAdvance
        return textAdvance(txt, int(round(txt_h)))

