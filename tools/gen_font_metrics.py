#
# gen_font_metrics.py - regenerate polars2svg/p2s_font_metrics.py from the bundled TTF
#
# Run this whenever fonts/NotoSans-Regular-subset.ttf changes:
#
#   uv run --with fonttools python tools/gen_font_metrics.py
#
# ...then regenerate the goldens (UPDATE_GOLDEN=1 python -m pytest tests/), because any
# advance-width change moves every text-derived coordinate in the SVG output.
#
# fontTools is a build-time-only dependency: the emitted table is plain ints, so the
# runtime never parses the TTF to measure text.  See p2s_font_metrics.py's header for
# why the advances are baked in rather than measured with Pillow.
#
import os

from fontTools.ttLib import TTFont

_HERE_     = os.path.dirname(os.path.abspath(__file__))
_FONT_     = os.path.join(_HERE_, '..', 'polars2svg', 'fonts', 'NotoSans-Regular-subset.ttf')
_OUT_      = os.path.join(_HERE_, '..', 'polars2svg', 'p2s_font_metrics.py')

_HEADER_ = '''#
# p2s_font_metrics.py - GENERATED, do not edit by hand
#
# Regenerate with: uv run --with fonttools python tools/gen_font_metrics.py
#
# Horizontal advance widths (in font units) for every glyph in the bundled
# NotoSans-Regular-subset.ttf, read straight from the font's hmtx table.
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
'''


def main():
    _font_ = TTFont(_FONT_)
    _upem_ = _font_['head'].unitsPerEm
    _hmtx_ = _font_['hmtx']
    _cmap_ = _font_.getBestCmap()

    _notdef_ = _hmtx_['.notdef'][0] if '.notdef' in _font_.getGlyphOrder() else _upem_ // 2
    _adv_    = {_cp_: _hmtx_[_gname_][0] for _cp_, _gname_ in _cmap_.items()}

    _lines_ = [_HEADER_]
    _lines_.append(f'UNITS_PER_EM    = {_upem_}\n')
    _lines_.append(f'NOTDEF_ADVANCE  = {_notdef_}\n\n')
    _lines_.append('# codepoint -> horizontal advance in font units\n')
    _lines_.append('ADVANCES = {\n')
    for _cp_ in sorted(_adv_):
        _lines_.append(f'    0x{_cp_:04x}: {_adv_[_cp_]},\n')
    _lines_.append('}\n')
    _lines_.append(_BODY_)

    with open(_OUT_, 'w') as _f_:
        _f_.write(''.join(_lines_))

    print(f'wrote {_OUT_}: {len(_adv_)} glyphs, unitsPerEm={_upem_}, .notdef={_notdef_}')


if __name__ == '__main__':
    main()
