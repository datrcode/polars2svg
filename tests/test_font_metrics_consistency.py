#
# test_font_metrics_consistency.py
#
# p2s_font_metrics.py is generated from the bundled NotoSans-Regular-subset.ttf by
# tools/gen_font_metrics.py and is the single source of text measurement for the whole
# package: ADVANCES backs textLength()/cropText() (how wide), INK_EXTENTS backs textInk()
# (how tall, and how far under the baseline).  Because it is a checked-in generated file,
# two things can silently go wrong:
#
#   - the font is replaced or re-subset and the table is not regenerated, so every measured
#     coordinate describes a font that is no longer bundled; or
#   - the table is hand-edited (its header says not to, which is not a mechanism).
#
# The drift tests below re-derive both tables from the TTF and compare.  They need fontTools
# -- a build-time-only dependency -- so they skip where it is absent; everything above them
# is the part that runs everywhere.
#
# The vertical half of this file exists because linkp's link labels used to be positioned
# from four hand-tuned per-character-class constants (0.67em ascender, 0.46em x-height...)
# calibrated by eye against whatever face WebKit picked.  Real Noto Sans ascends 0.76em, so
# every one of those numbers was wrong for the font actually being measured.  See
# tests/test_font_consistency.py for the other half of the fix -- naming the font in the
# markup, without which none of these metrics describe the rendered glyphs.
#
import os
import unittest

import polars2svg
from polars2svg import Polars2SVG
from polars2svg import p2s_font_metrics as fm

_FONT_PATH_ = os.path.join(os.path.dirname(os.path.abspath(polars2svg.__file__)),
                           'fonts', 'NotoSans-Regular-subset.ttf')


def _fontToolsAvailable_():
    try:
        import fontTools.ttLib  # noqa: F401
        import fontTools.pens.boundsPen  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Table shape
# ---------------------------------------------------------------------------

class TestTableIntegrity(unittest.TestCase):

    def test_both_tables_are_populated(self):
        self.assertGreater(len(fm.ADVANCES), 100)
        self.assertGreater(len(fm.INK_EXTENTS), 100)
        self.assertEqual(fm.UNITS_PER_EM, 1000)

    def test_ink_is_a_subset_of_the_glyphs_the_font_has(self):
        '''A codepoint with ink but no advance would be a glyph the width table forgot.'''
        self.assertEqual(set(fm.INK_EXTENTS) - set(fm.ADVANCES), set())

    def test_the_only_glyphs_without_ink_are_blanks(self):
        '''Absence from INK_EXTENTS means "draws nothing".  If a visible character ever
        lands in that gap it measures as zero-height and whatever is placed against it
        collides.  In this subset the gap is the space family plus the zero-width
        formatting controls -- all non-printing.'''
        for _cp_ in set(fm.ADVANCES) - set(fm.INK_EXTENTS):
            _ch_ = chr(_cp_)
            self.assertTrue(_ch_.isspace() or not _ch_.isprintable(),
                            f'U+{_cp_:04X} ({_ch_!r}) has an advance but no ink')

    def test_extents_are_ordered_integers(self):
        for _cp_, (_y0_, _y1_) in fm.INK_EXTENTS.items():
            self.assertIsInstance(_y0_, int, f'U+{_cp_:04X}')
            self.assertIsInstance(_y1_, int, f'U+{_cp_:04X}')
            self.assertLessEqual(_y0_, _y1_, f'U+{_cp_:04X} has yMin above yMax')

    def test_notdef_has_ink(self):
        '''Unknown codepoints measure as .notdef; a zero-height fallback would let an
        out-of-subset character silently take no vertical space.'''
        self.assertGreater(fm.NOTDEF_INK[1], 0)
        self.assertGreater(fm.NOTDEF_ADVANCE, 0)

    def test_latin_letters_all_have_ink(self):
        for _ch_ in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789':
            self.assertIn(ord(_ch_), fm.INK_EXTENTS, f'{_ch_!r} has no ink extent')


# ---------------------------------------------------------------------------
# textInk() semantics
# ---------------------------------------------------------------------------

class TestTextInk(unittest.TestCase):

    def setUp(self):
        self.p2s = Polars2SVG()

    def test_returns_non_negative_distances_from_the_baseline(self):
        _above_, _below_ = self.p2s.textInk('dog', 12)
        self.assertGreater(_above_, 0.0)
        self.assertGreater(_below_, 0.0)

    def test_ascender_outreaches_cap_which_outreaches_x_height(self):
        _asc_ = self.p2s.textInk('l', 12)[0]
        _cap_ = self.p2s.textInk('B', 12)[0]
        _xh_  = self.p2s.textInk('x', 12)[0]
        self.assertGreater(_asc_, _cap_)
        self.assertGreater(_cap_, _xh_)

    def test_a_run_measures_its_tallest_and_deepest_character(self):
        self.assertEqual(self.p2s.textInk('xlg', 12),
                         (self.p2s.textInk('l', 12)[0], self.p2s.textInk('g', 12)[1]))

    def test_scales_linearly_with_size(self):
        _a12_ = self.p2s.textInk('Ag', 12)
        _a24_ = self.p2s.textInk('Ag', 24)
        self.assertAlmostEqual(_a24_[0], 2 * _a12_[0], places=9)
        self.assertAlmostEqual(_a24_[1], 2 * _a12_[1], places=9)

    def test_size_is_quantized_like_textlength(self):
        '''Width and height of a run are always taken at the same size, so a fractional
        txt_h cannot make one of them disagree with the other.'''
        self.assertEqual(self.p2s.textInk('Ag', 12.4), self.p2s.textInk('Ag', 12))
        self.assertEqual(self.p2s.textLength('Ag', 12.4), self.p2s.textLength('Ag', 12))

    def test_blank_runs_have_no_ink(self):
        for _s_ in ('', ' ', '   ', '\t', '\n', '\r', ' \t\n '):
            self.assertEqual(self.p2s.textInk(_s_, 12), (0.0, 0.0), msg=repr(_s_))

    def test_none_and_non_strings_are_tolerated(self):
        '''Callers pass label values straight through (linkp labels can be numeric).'''
        self.assertEqual(self.p2s.textInk(None, 12), (0.0, 0.0))
        self.assertEqual(self.p2s.textInk(12.5, 12), self.p2s.textInk('12.5', 12))

    def test_a_zero_label_still_has_ink(self):
        '''A falsy-but-real label: 0 draws a glyph, so it must not measure as blank the way
        None and '' do.'''
        self.assertEqual(self.p2s.textInk(0, 12), self.p2s.textInk('0', 12))
        self.assertGreater(self.p2s.textInk(0, 12)[0], 0.0)

    def test_unknown_codepoints_fall_back_to_notdef(self):
        _expect_ = fm.NOTDEF_INK[1] * 12 / fm.UNITS_PER_EM
        self.assertAlmostEqual(self.p2s.textInk('\U0001f600', 12)[0], _expect_, places=9)

    def test_a_string_of_blanks_and_glyphs_measures_the_glyphs(self):
        self.assertEqual(self.p2s.textInk(' A\tg ', 12), self.p2s.textInk('Ag', 12))


# ---------------------------------------------------------------------------
# The generated table still describes the bundled font
# ---------------------------------------------------------------------------

@unittest.skipUnless(_fontToolsAvailable_(),
                     'fontTools is a build-time-only dependency (uv run --with fonttools)')
class TestGeneratedTableMatchesTheFont(unittest.TestCase):
    '''Re-derive both tables from fonts/NotoSans-Regular-subset.ttf exactly as
    tools/gen_font_metrics.py does.  A failure here means the font and the baked metrics
    have parted company -- regenerate with:

        uv run --with fonttools python tools/gen_font_metrics.py

    ...and then regenerate the goldens, because every text-derived coordinate moves.'''

    @classmethod
    def setUpClass(cls):
        from fontTools.ttLib import TTFont
        cls.font = TTFont(_FONT_PATH_)
        cls.cmap = cls.font.getBestCmap()
        cls.gset = cls.font.getGlyphSet()

    def _ink_(self, glyph_name):
        import math
        from fontTools.pens.boundsPen import BoundsPen
        _pen_ = BoundsPen(self.gset)
        self.gset[glyph_name].draw(_pen_)
        if _pen_.bounds is None: return None
        return int(math.floor(_pen_.bounds[1])), int(math.ceil(_pen_.bounds[3]))

    def test_font_file_is_where_the_package_expects_it(self):
        self.assertTrue(os.path.exists(_FONT_PATH_), _FONT_PATH_)

    def test_units_per_em_matches(self):
        self.assertEqual(fm.UNITS_PER_EM, self.font['head'].unitsPerEm)

    def test_advances_match_the_hmtx_table(self):
        _hmtx_ = self.font['hmtx']
        _want_ = {_cp_: _hmtx_[_g_][0] for _cp_, _g_ in self.cmap.items()}
        self.assertEqual(fm.ADVANCES, _want_,
                         'ADVANCES has drifted from the bundled font -- regenerate')

    def test_ink_extents_match_the_glyph_outlines(self):
        _want_ = {}
        for _cp_, _g_ in self.cmap.items():
            _e_ = self._ink_(_g_)
            if _e_ is not None: _want_[_cp_] = _e_
        self.assertEqual(fm.INK_EXTENTS, _want_,
                         'INK_EXTENTS has drifted from the bundled font -- regenerate')

    def test_notdef_fallbacks_match(self):
        self.assertEqual(fm.NOTDEF_ADVANCE, self.font['hmtx']['.notdef'][0])
        self.assertEqual(tuple(fm.NOTDEF_INK), self._ink_('.notdef'))

    def test_the_declared_font_family_is_the_bundled_one(self):
        '''The heart of it: default_font is what the markup asks a renderer for, and this
        TTF is what the package measures.  If those two ever name different families, every
        text-derived coordinate is measured in one face and drawn in another.'''
        _names_ = {_r_.toUnicode() for _r_ in self.font['name'].names if _r_.nameID in (1, 16)}
        _declared_ = Polars2SVG().default_font
        self.assertTrue(any(_n_ in _declared_ for _n_ in _names_),
                        f'default_font {_declared_!r} names none of the bundled family '
                        f'names {sorted(_names_)}')


if __name__ == '__main__':
    unittest.main()
