from typing import Any
import html


#
# svgEscape() / svgUnescape() - the single door between text and markup.
#
# Escaping used to be spelled five different ways across the package: the standard library
# call here, the same call inline in chordp and spreadlinesp, two hand-rolled .replace()
# chains in linkp (one each way), and nothing at all in the interactive chrome, whose
# strings are its own.  Five mechanisms is four too many -- not because any one of them was
# wrong, but because a convention nobody can point at is a convention that breaks where two
# of them meet.  It did: linkp escaped a label column and then cut the escaped string by
# character count, splitting an entity across two <tspan>s and taking the whole document
# out of well-formed XML with it.
#
# So there is one function, and it has one rule:
#
#   **Escape last.**  Everything that measures, crops, wraps, truncates or slices text works
#   on the raw string; svgEscape() is applied to what comes out, immediately before it is
#   interpolated into markup.  An entity is five characters that draw as one, so any cut
#   taken after escaping is cutting the wrong string -- it can land inside the entity (bad
#   markup) and it miscounts the width either way ('&' is not five characters wide).
#
# Both directions are here because both directions have callers: the GPU paths hand text to
# the glyph atlas, which draws characters and knows nothing about entities, and
# svgToDisplayList() parses finished markup back into a display list.  Whatever escaped the
# text should be what un-escapes it.
#
# The standard library's behaviour is the contract, quotes included.  Quotes need escaping
# in an attribute value and are merely harmless in element content, and one function that is
# safe in both places beats two that differ by where you are allowed to use them.
#
# tests/test_svg_escaping.py holds both halves: that this behaves, and -- by scanning the
# package for a sixth mechanism -- that it stays the only door.
#
def svgEscape(txt: int | str | None) -> str:
    return html.escape('' if txt is None else str(txt))


def svgUnescape(txt: str | None) -> str:
    return html.unescape('' if txt is None else str(txt))


class P2STextMixin:
    # ---------------------------------------------------------------------
    # Host-class attributes this mixin reads off `self`.
    #
    # A mixin is never instantiated on its own -- these are provided by whatever
    # class mixes it in (Polars2SVG).  Declared here so a checker
    # can follow the mixin's own methods; bare annotations, so nothing exists at
    # runtime and nothing is shadowed.
    #
    # Typed `Any` on purpose: the mixin genuinely does not know the concrete type,
    # and several hosts declare the same name with a narrower type of their own.
    # ---------------------------------------------------------------------
    colorTyped: Any

    def __init__(self) -> None:
        pass

    #
    # default_font - the face every text measurement in this package assumes.
    #
    # textLength()/textInk() answer from the bundled NotoSans-Regular-subset.ttf, and the
    # GPU glyph atlas rasterizes that same file, so every text-derived coordinate in the
    # output is a statement about Noto Sans.  The markup has to say so, or a renderer
    # substitutes a face with different metrics and the layout drifts from what was
    # measured -- silently, because a slightly-too-long label just looks like a label.
    #
    # Contract: every component's *root* <svg> element carries font-family="{default_font}",
    # so the assumption is stated once per document and CSS inheritance carries it to every
    # <text> underneath -- including the raw <text> strings linkp/spreadlinesp/xyp emit
    # without going through svgText().  A <text> that wants a different face (the interactive
    # chrome's monospace help overlays) overrides it with its own font-family attribute.
    # tests/test_font_consistency.py holds both halves of that contract.
    #
    # This is a request, not a guarantee: a machine with no Noto Sans installed falls back to
    # the generic sans and the measurement is approximate again.  Guaranteeing it would mean
    # embedding the subset as a base64 @font-face (~90KB/document) -- deliberately not done.
    #
    def __p2s_text_mixin_init__(self) -> None:
        self.default_font = "'Noto Sans', sans-serif"
        self._glyph_atlas_ = None

    #
    # svgEscape() / svgUnescape() - method routes onto the module-level door above, for the
    # components, which reach this package through their `self.p2s` handle rather than by
    # importing.  Same function, same rule: escape last.
    #
    def svgEscape(self, txt: str) -> str:
        return svgEscape(txt)

    def svgUnescape(self, txt: str) -> str:
        return svgUnescape(txt)

    #
    # glyphAtlas() - lazily constructed shared GlyphAtlas (GPU text); built from the
    # same bundled TTF as textLength() so GPU text layout matches SVG text layout
    #
    def glyphAtlas(self) -> Any:
        if self._glyph_atlas_ is None:
            from polars2svg.p2s_glyph_atlas import GlyphAtlas
            self._glyph_atlas_ = GlyphAtlas()
        return self._glyph_atlas_

    #
    # svgText() - Render SVG Text In A Consistent Manner
    #
    def svgText(self,
                txt: str,
                x: float,
                y: float,
                txt_h: float       = 12,
                just_xy: bool     = False,   # for the text block widget -- that will use an SVG group to consolidate the rendering
                color: str | None       = None,
                anchor: str      = 'start',
                font: Any        = None,
                font_style: Any  = None,
                rotation: int | None    = None) -> str:
        if txt == '\n' or txt == '' or txt == '\r' or txt == '\t': return ''
        if font  is None: font  = self.default_font
        if color is None: color = self.colorTyped('label','defaultfg')
        txt = str(txt)

        _html_txt = svgEscape(txt)
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
    def cropText(self, txt: str, txt_h: float, w: float) -> str:
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
    def svgAxisLabels(self, lbl_left: str, lbl_center: str, lbl_right: str,
                      x0: float, available_w: float, y: int, txt_h: float,
                      color_left: str | None = None, color_center: str | None = None, color_right: str | None = None,
                      gap: float | None = None, prefer_center: bool = True, dl: Any = None) -> list:
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
        def _emit_(txt: str, x: float, anchor: str, color: str) -> str:
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
    def __SignXDivUnit__(self, x: float) -> tuple:
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
    def unitizeInt_extendUntilDifferent(self, a: float, b: float) -> tuple:
        _xsign_, _x_, _xdiv_, _xunit_ = self.__SignXDivUnit__(a)
        _ysign_, _y_, _ydiv_, _yunit_ = self.__SignXDivUnit__(b)
        _round_ = 0
        while round(_x_/_xdiv_, _round_) == round(_y_/_ydiv_, _round_) and _round_ < 4: _round_ += 1
        return _xsign_ + str(round(_x_/_xdiv_, _round_)) + _xunit_, _ysign_ + str(round(_y_/_ydiv_, _round_)) + _yunit_

    #
    # unitizeInt() - format a number with K/M/B/T/Q suffix to a target digit count
    #
    def unitizeInt(self, a: float, num_of_digits: int = 5) -> str:
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
    def textLength(self, txt: str, txt_h: float) -> float:
        if not txt or txt in ('\n', '\r', '\t', ''):
            return 0
        from polars2svg.p2s_font_metrics import textAdvance
        return textAdvance(txt, int(round(txt_h)))

    #
    # textInk() - (above, below) reach of txt's ink from its own baseline, in pixels
    #
    # The vertical companion to textLength(), from the same baked table and for the same
    # reason: a caller that positions text by where its glyphs actually land -- rather than
    # by the baseline -- needs metrics that belong to the font, not to whatever face the
    # viewer's renderer happened to substitute.  'cow' rises to the x-height, 'CAT' to the
    # cap height, 'dog' hangs a descender; both returned values are non-negative distances
    # from the baseline, and a run with no ink (empty or whitespace) measures (0.0, 0.0).
    #
    # Size is quantized to an integer exactly as textLength() does, so a run's measured
    # width and its measured height are always taken at the same size.
    #
    # Guarded on None rather than falsiness: a label value of 0 is a glyph like any other,
    # and str() it before measuring so a numeric label measures its rendered form.
    #
    def textInk(self, txt: float | int | str | None, txt_h: float) -> tuple:
        if txt is None:
            return 0.0, 0.0
        from polars2svg.p2s_font_metrics import textInk
        return textInk(str(txt), int(round(txt_h)))

