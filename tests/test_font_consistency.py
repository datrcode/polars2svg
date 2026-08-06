#
# test_font_consistency.py
#
# Every text-derived coordinate this package emits is measured in one specific font: the
# bundled NotoSans-Regular-subset.ttf, via textLength() (advances) and textInk() (vertical
# extents), both reading the baked table in p2s_font_metrics.py.  If the emitted markup
# does not *name* that font, the renderer substitutes another one and every one of those
# coordinates is wrong by however much the two faces differ:
#
#   - horizontally, cropText() truncates to fit a width computed in Noto Sans, so under a
#     wider face the label overflows the space it was cropped to fit;
#   - vertically, linkp offsets a link label off its edge by the run's own ink, so under a
#     taller face the label touches the edge (or floats away from it).
#
# Neither failure looks like a metrics bug downstream -- an overflowing label just reads as
# a label that is slightly too long -- so nothing else in the suite would notice.  The
# SVG-string goldens compare our markup to our markup, and the PNG goldens rasterize through
# svglib/reportlab, which is a third font engine again.
#
# The contract, held here in both directions:
#
#   1. every component's root <svg> carries font-family="{p2s.default_font}", and
#   2. every <text> in a rendered document resolves to that font by inheritance -- so a
#      newly added raw <text> emitter (linkp alone has twelve, none of which name a font)
#      cannot quietly opt out of the measured face.
#
import re
import unittest
import xml.etree.ElementTree as ET

import polars as pl

from polars2svg import Polars2SVG
from histop_dataframes import makeHistoDf
from timep_dataframes import makeTimeDf
from piep_dataframes import makePieDf

_SVG_NS_ = '{http://www.w3.org/2000/svg}'

_XY_DF_ = pl.DataFrame({
    'a':   [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    'b':   [4, 5, 6, 7, 9, 8, 1, 3, 5,  2,  6,  7],
    'cat': ['alpha', 'alpha', 'alpha', 'beta', 'beta', 'beta',
            'gamma', 'gamma', 'gamma', 'delta', 'delta', 'delta'],
})

_LINK_DF_ = pl.DataFrame({
    'fm':   ['a', 'b', 'c', 'a', 'd', 'b', 'c', 'a'],
    'to':   ['b', 'a', 'a', 'c', 'a', 'c', 'b', 'd'],
    'time': [1,   1,   1,   2,   2,   2,   3,   3  ],
    'dsc':  ['calls', 'answers', 'pings', 'emails', 'calls', 'pings', 'calls', 'emails'],
})
_POS_ = {'a': (0.0, 0.0), 'b': (1.0, 0.1), 'c': (0.4, 1.0), 'd': (0.9, 0.8)}


#
# _renderedComponents_() - one rendered document per component, as (name, svg)
#
# Deliberately configured to *emit text*: labels and axis context on, since a component
# with no <text> in it proves nothing here.
#
def _renderedComponents_(p2s):
    _xyp_ = p2s.xyp(_XY_DF_, 'a', 'b', color='cat', wxh=(256, 256))
    _out_ = [
        ('xyp',      _xyp_.svg),
        ('histop',   p2s.histop(makeHistoDf(n=100), 'cat', color='group', wxh=(256, 256)).svg),
        ('timep',    p2s.timep(makeTimeDf(n=100, year=(2020, 2023), month=(1, 12)), 'ts',
                               color='category', wxh=(384, 256)).svg),
        ('piep',     p2s.piep(makePieDf(n=200), 'cat', draw_labels=True, wxh=(256, 256)).svg),
        ('linkp',    p2s.linkp(_LINK_DF_, relationships=[('fm', 'to', 'dsc')], pos=_POS_,
                               wxh=(400, 300), draw_node_labels=True,
                               draw_link_labels=True).svg),
        ('chordp',   p2s.chordp(_LINK_DF_, relationships=[('fm', 'to')], draw_labels=True,
                                wxh=(300, 300)).svg),
        ('spreadlinesp', p2s.spreadlinesp(_LINK_DF_, [('fm', 'to')], ego='a', time='time',
                                          wxh=(700, 300)).svg),
        ('smallp',   p2s.smallp(_XY_DF_, 'cat', _xyp_, wxh=(384, 384)).svg),
        # legends measure every entry with textLength() to size themselves, so they belong
        # in this sweep as much as the plots do
        ('xyp+legend',    p2s.xyp(_XY_DF_, 'a', 'b', color='cat', legend='right',
                                  wxh=(320, 240)).svg),
        ('histop+legend', p2s.histop(makeHistoDf(n=100), 'cat', color='group',
                                     legend='bottom', wxh=(256, 320)).svg),
    ]
    return _out_


#
# _resolvedFont_() - the font-family a <text> element actually renders in
#
# font-family is an inherited CSS property, so an element with no font-family of its own
# takes its nearest ancestor's.  Returns None when nothing in the chain declares one --
# which is the defect this file exists to catch.
#
def _resolvedFont_(chain):
    for _el_ in reversed(chain):
        _ff_ = _el_.get('font-family')
        if _ff_ is not None: return _ff_
        _style_ = _el_.get('style')
        if _style_:
            _m_ = re.search(r'(?:^|;)\s*font-family\s*:\s*([^;]+)', _style_)
            if _m_ is not None: return _m_.group(1).strip()
    return None


#
# _textElementsWithFonts_() - [(text-content, resolved font-family)] for every <text>
#
def _textElementsWithFonts_(svg):
    _root_ = ET.fromstring(svg)
    _out_  = []

    def _walk_(el, chain):
        _chain_ = chain + [el]
        if el.tag == f'{_SVG_NS_}text':
            _out_.append((''.join(el.itertext()).strip(), _resolvedFont_(_chain_)))
        for _kid_ in el:
            _walk_(_kid_, _chain_)

    _walk_(_root_, [])
    return _out_


class _FontTestBase_(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.p2s        = Polars2SVG()
        cls.components = _renderedComponents_(cls.p2s)


# ---------------------------------------------------------------------------
# 1. The root element states the assumption
# ---------------------------------------------------------------------------

class TestRootFontFamily(_FontTestBase_):

    def test_every_component_roots_the_default_font(self):
        for _name_, _svg_ in self.components:
            with self.subTest(component=_name_):
                _root_ = ET.fromstring(_svg_)
                self.assertEqual(_root_.get('font-family'), self.p2s.default_font,
                                 f'{_name_} root <svg> does not name the measured font')

    def test_the_default_font_names_the_bundled_face(self):
        '''The measurement reads NotoSans-Regular-subset.ttf; the markup has to ask for
        Noto Sans, or the two describe different faces.'''
        self.assertIn('Noto Sans', self.p2s.default_font)

    def test_repr_svg_carries_it_too(self):
        '''_repr_svg_() is what a notebook and save() emit -- the same document, not a
        separately assembled one.'''
        _lp_ = self.p2s.linkp(_LINK_DF_, relationships=[('fm', 'to', 'dsc')], pos=_POS_,
                              wxh=(400, 300), draw_node_labels=True)
        self.assertIn(f'font-family="{self.p2s.default_font}"', _lp_._repr_svg_())

    def test_placeholder_document_carries_it(self):
        '''The dataless "no data" canvas is a rendered document like any other.'''
        _root_ = ET.fromstring(self.p2s.placeholderSVG(200, 100))
        self.assertEqual(_root_.get('font-family'), self.p2s.default_font)


# ---------------------------------------------------------------------------
# 2. Every <text> inherits it
# ---------------------------------------------------------------------------

class TestTextInheritsTheFont(_FontTestBase_):

    def test_every_text_element_resolves_to_a_font(self):
        for _name_, _svg_ in self.components:
            for _txt_, _font_ in _textElementsWithFonts_(_svg_):
                with self.subTest(component=_name_, text=_txt_[:24]):
                    self.assertIsNotNone(
                        _font_, f'{_name_}: <text>{_txt_[:24]}</text> names no font and '
                                f'inherits none -- it renders in whatever face the host '
                                f'page or viewer supplies, not the one it was measured in')

    def test_every_text_element_resolves_to_the_measured_font(self):
        '''Stronger than "some font": the resolved face has to be the one textLength() and
        textInk() measured, otherwise the coordinates around it are approximations.'''
        for _name_, _svg_ in self.components:
            for _txt_, _font_ in _textElementsWithFonts_(_svg_):
                with self.subTest(component=_name_, text=_txt_[:24]):
                    self.assertEqual(_font_, self.p2s.default_font,
                                     f'{_name_}: <text>{_txt_[:24]}</text> renders in '
                                     f'{_font_!r}')

    def test_the_components_actually_emitted_text(self):
        '''Guard on the guard: an inheritance check over zero <text> elements passes
        vacuously.  linkp is called out separately because its twelve raw emitters are the
        ones that carried no font-family at all.'''
        _counts_ = {_n_: len(_textElementsWithFonts_(_s_)) for _n_, _s_ in self.components}
        for _name_, _n_ in _counts_.items():
            with self.subTest(component=_name_):
                self.assertGreater(_n_, 0, f'{_name_} rendered no <text> to check')
        self.assertGreater(_counts_['linkp'], 4)

    def test_link_labels_and_node_labels_are_both_covered(self):
        '''The two newest raw emitters: node labels (with <tspan> children) and link labels
        (bare, and inside a <textPath> for the curve shape).'''
        for _shape_ in ('line', 'curve'):
            _lp_ = self.p2s.linkp(_LINK_DF_, relationships=[('fm', 'to', 'dsc')], pos=_POS_,
                                  wxh=(400, 300), link_shape=_shape_, draw_node_labels=True,
                                  draw_link_labels=True, label_line_width=3)
            _fonts_ = [_f_ for _, _f_ in _textElementsWithFonts_(_lp_.svg)]
            with self.subTest(link_shape=_shape_):
                self.assertGreater(len(_fonts_), 4)
                self.assertEqual(set(_fonts_), {self.p2s.default_font})

    def test_chordp_labels_name_the_measured_font_on_the_element(self):
        '''chordp's two label emitters carry their own font-family (they predate the root
        pin).  Whatever they name wins over the root, so it had better be the measured
        face -- the circular style steps glyphs along the arc by textLength() advances.'''
        for _style_ in ('radial', 'circular'):
            _cp_ = self.p2s.chordp(_LINK_DF_, relationships=[('fm', 'to')], draw_labels=True,
                                   label_style=_style_, wxh=(300, 300))
            _own_ = [_t_.get('font-family')
                     for _t_ in ET.fromstring(_cp_.svg).iter(f'{_SVG_NS_}text')]
            with self.subTest(label_style=_style_):
                self.assertTrue(_own_)
                self.assertEqual(set(_own_), {self.p2s.default_font})


# ---------------------------------------------------------------------------
# The guard's own machinery -- a check that cannot fail is not a check
# ---------------------------------------------------------------------------

class TestFontResolution(unittest.TestCase):

    def _fonts_(self, svg):
        return [_f_ for _, _f_ in _textElementsWithFonts_(svg)]

    def test_a_bare_text_under_a_bare_root_resolves_to_nothing(self):
        '''The defect, reproduced in miniature: this is what every linkp <text> looked
        like before the root was pinned, and it must come back as None.'''
        self.assertEqual(
            self._fonts_('<svg xmlns="http://www.w3.org/2000/svg"><text>hi</text></svg>'),
            [None])

    def test_the_root_font_reaches_a_nested_text(self):
        _svg_ = ('<svg xmlns="http://www.w3.org/2000/svg" font-family="A">'
                 '<g><g><text>hi</text></g></g></svg>')
        self.assertEqual(self._fonts_(_svg_), ['A'])

    def test_the_nearest_declaration_wins(self):
        _svg_ = ('<svg xmlns="http://www.w3.org/2000/svg" font-family="A">'
                 '<g font-family="B"><text>hi</text></g>'
                 '<text font-family="C">there</text>'
                 '<text style="fill:#000; font-family: D; font-size:9px">and</text></svg>')
        self.assertEqual(self._fonts_(_svg_), ['B', 'C', 'D'])


# ---------------------------------------------------------------------------
# 3. The whole point: measurement and rendering describe the same face
# ---------------------------------------------------------------------------

class TestMeasurementMatchesMarkup(_FontTestBase_):

    def test_cropped_text_is_measured_in_the_font_the_markup_names(self):
        '''cropText() decides where to truncate from Noto Sans advances.  That is only a
        statement about the rendered label if the markup asks for Noto Sans -- so the
        cropping test and the font-family test are the same test, split in two.'''
        _long_ = 'a' * 200
        _df_   = pl.DataFrame({'fm': ['a', 'b'], 'to': ['b', 'a'],
                               'dsc': [_long_, _long_]})
        _lp_   = self.p2s.linkp(_df_, relationships=[('fm', 'to', 'dsc')],
                                pos={'a': (0.0, 0.0), 'b': (1.0, 0.0)}, wxh=(400, 200),
                                draw_link_labels=True)
        _texts_ = _textElementsWithFonts_(_lp_.svg)
        self.assertTrue(_texts_)
        for _txt_, _font_ in _texts_:
            self.assertTrue(_txt_.endswith('...'), 'label should have been cropped')
            self.assertLess(len(_txt_), len(_long_))
            self.assertEqual(_font_, self.p2s.default_font)
            # ...and it fits the width it was cropped against, in that same font
            self.assertLessEqual(self.p2s.textLength(_txt_, _lp_.txt_h), 400)

    def test_gpu_atlas_and_svg_agree_on_the_font(self):
        '''The WebGPU path rasterizes the bundled TTF into its glyph atlas, so it has
        always been metric-exact.  The SVG path now names that same file's family -- the
        two rendering paths must not describe different faces.'''
        import os
        from polars2svg import p2s_glyph_atlas
        _atlas_font_ = os.path.basename(p2s_glyph_atlas.GlyphAtlas().font_path)
        self.assertIn('NotoSans', _atlas_font_)
        self.assertIn('Noto Sans', self.p2s.default_font)


if __name__ == '__main__':
    unittest.main()
