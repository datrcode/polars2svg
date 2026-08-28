#
# gen_font_metrics.py - regenerate polars2svg/p2s_font_metrics.py from the bundled TTF
#
# Run this whenever fonts/NotoSans-Regular-subset.ttf changes:
#
#   uv run --with fonttools python tools/gen_font_metrics.py
#
# ...then regenerate the goldens (UPDATE_GOLDEN=1 python -m pytest tests/), because any
# advance-width change moves every text-derived coordinate in the SVG output, and any
# ink-extent change moves every label positioned off its own ink (linkp link labels).
#
# Two tables come out of the font: horizontal advances (hmtx) and vertical ink extents
# (the glyph outlines' yMin/yMax).  tests/test_font_metrics_consistency.py re-derives
# both from the TTF and fails if this file has drifted from it.
#
# fontTools is a build-time-only dependency: the emitted tables are plain ints, so the
# runtime never parses the TTF to measure text.  See p2s_font_metrics.py's header for
# why the metrics are baked in rather than measured with Pillow.
#
import math
import os

from fontTools.pens.boundsPen import BoundsPen
from fontTools.ttLib import TTFont

_HERE_     = os.path.dirname(os.path.abspath(__file__))
_FONT_     = os.path.join(_HERE_, '..', 'polars2svg', 'fonts', 'NotoSans-Regular-subset.ttf')
_OUT_      = os.path.join(_HERE_, '..', 'polars2svg', 'p2s_font_metrics.py')

_HEADER_ = '''#
# p2s_font_metrics.py - GENERATED, do not edit by hand
#
# Regenerate with: uv run --with fonttools python tools/gen_font_metrics.py
#
# Text metrics for the bundled NotoSans-Regular-subset.ttf: horizontal advance
# widths (from the font's hmtx table) and vertical ink extents (from the glyph
# outlines), both in font units.
#
# Why baked in rather than measured at runtime: textLength() used to call Pillow's
# ImageFont.getlength().  Pillow's answer depends on whether that particular Pillow
# build ships Raqm -- with it, FreeType returns fractional advances; without it,
# advances are hinted and rounded to whole pixels.  The macOS wheel has Raqm and the
# Linux one does not, so the same DataFrame rendered different SVG on different
# machines (width('A', 12) was 7.671875 on macOS and 8.0 on Linux), and every text-
# derived coordinate drifted with it.  Reading hmtx makes the metrics a property of
# the font instead of a property of the user's Pillow build.
#
# These are unhinted, unkerned advances: the font has a GPOS table, but pair kerning
# is deliberately not applied -- SVG renderers do not apply it to <text> either, so
# summing bare advances is what actually matches the rendered output.
#
# INK_EXTENTS is the vertical counterpart and exists for the same reason: a component
# that positions a label by how far its glyphs actually reach (linkp offsets a link
# label off its edge by the run's own ascent or descent) was doing it from hand-tuned
# per-character-class constants eyeballed against whatever font the browser happened to
# pick.  Reading the outlines makes that a property of the font too -- and the markup
# now names that font (every component's root <svg> carries font-family), so measurement
# and rendering finally describe the same face.
#
# Pillow is still used to rasterize glyph bitmaps for the GPU atlas (p2s_glyph_atlas.py);
# it is only text *measurement* that no longer depends on it.
#

__name__ = 'p2s_font_metrics'

'''

_BODY_ = '''

#
# textAdvance() - width in px of txt rendered at px_size, from the baked advance table
#
# Unknown codepoints fall back to the font's .notdef advance, mirroring what a renderer
# does when it has no glyph -- so an out-of-subset character still takes up space rather
# than silently collapsing to zero width.
#
def textAdvance(txt, px_size):
    if not txt: return 0.0
    _units_ = 0
    for _ch_ in txt:
        _units_ += ADVANCES.get(ord(_ch_), NOTDEF_ADVANCE)
    return _units_ * px_size / UNITS_PER_EM


#
# textInk() - (above, below) reach of txt's ink from its own baseline, in px at px_size
#
# The vertical counterpart of textAdvance(): the advance table answers "how wide is this
# string", this answers "how far do its glyphs actually rise and hang".  'cow' rises only
# to the x-height, 'CAT' to the cap height, 'dog' hangs a descender below -- so a caller
# that keeps a label clear of something (linkp's link labels) can offset by the run's own
# ink rather than by a worst-case constant.
#
# Both returned values are non-negative distances from the baseline.  Characters that lay
# down no ink contribute nothing -- the font's own blank glyphs (space) are simply absent
# from the table, and whitespace the font has no glyph for at all (tab, newline) is skipped
# explicitly so it cannot be mistaken for a missing glyph.  A run of only those measures
# (0.0, 0.0).  Anything else the font cannot draw falls back to .notdef's box, the same
# substitution textAdvance() makes.
#
def textInk(txt, px_size):
    if not txt: return 0.0, 0.0
    _above_ = _below_ = 0
    for _ch_ in txt:
        _cp_ = ord(_ch_)
        _e_  = INK_EXTENTS.get(_cp_)
        if _e_ is None:
            if _cp_ in ADVANCES or _ch_.isspace(): continue   # draws nothing
            _e_ = NOTDEF_INK
        if _e_[1] > _above_: _above_ = _e_[1]
        if _e_[0] < _below_: _below_ = _e_[0]
    return _above_ * px_size / UNITS_PER_EM, -_below_ * px_size / UNITS_PER_EM
'''


#
# glyphInk() - (yMin, yMax) of a glyph's drawn outline in font units, or None if it draws
# nothing.  Bounds are taken from the outline (curve extrema included, not just the control
# points) and rounded outward, so a rounded ink box never under-reports the ink.
#
def glyphInk(glyph_set, glyph_name):
    _pen_ = BoundsPen(glyph_set)
    glyph_set[glyph_name].draw(_pen_)
    if _pen_.bounds is None: return None
    _x0_, _y0_, _x1_, _y1_ = _pen_.bounds
    return int(math.floor(_y0_)), int(math.ceil(_y1_))


def main():
    _font_ = TTFont(_FONT_)
    _upem_ = _font_['head'].unitsPerEm
    _hmtx_ = _font_['hmtx']
    _cmap_ = _font_.getBestCmap()
    _gset_ = _font_.getGlyphSet()

    _has_notdef_ = '.notdef' in _font_.getGlyphOrder()
    _notdef_     = _hmtx_['.notdef'][0] if _has_notdef_ else _upem_ // 2
    _notdef_ink_ = (glyphInk(_gset_, '.notdef') if _has_notdef_ else None) or (0, _upem_)
    _adv_        = {_cp_: _hmtx_[_gname_][0] for _cp_, _gname_ in _cmap_.items()}
    _ink_        = {}
    for _cp_, _gname_ in _cmap_.items():
        _e_ = glyphInk(_gset_, _gname_)
        if _e_ is not None: _ink_[_cp_] = _e_

    _lines_ = [_HEADER_]
    _lines_.append(f'UNITS_PER_EM    = {_upem_}\n')
    _lines_.append(f'NOTDEF_ADVANCE  = {_notdef_}\n')
    _lines_.append(f'NOTDEF_INK      = {_notdef_ink_}\n\n')
    _lines_.append('# codepoint -> horizontal advance in font units\n')
    _lines_.append('ADVANCES = {\n')
    for _cp_ in sorted(_adv_):
        _lines_.append(f'    0x{_cp_:04x}: {_adv_[_cp_]},\n')
    _lines_.append('}\n\n')
    _lines_.append('# codepoint -> (yMin, yMax) of the glyph outline in font units, relative to the\n'
                   '# baseline.  Codepoints whose glyph has no contours (space and friends) are absent.\n')
    _lines_.append('INK_EXTENTS = {\n')
    for _cp_ in sorted(_ink_):
        _lines_.append(f'    0x{_cp_:04x}: {_ink_[_cp_]},\n')
    _lines_.append('}\n')
    _lines_.append(_BODY_)

    with open(_OUT_, 'w') as _f_:
        _f_.write(''.join(_lines_))

    print(f'wrote {_OUT_}: {len(_adv_)} glyphs ({len(_ink_)} with ink), '
          f'unitsPerEm={_upem_}, .notdef={_notdef_}/{_notdef_ink_}')


if __name__ == '__main__':
    main()
