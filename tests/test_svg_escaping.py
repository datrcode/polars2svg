"""One door for XML escaping -- and the ordering rule that comes with it.

Escaping used to have five independent mechanisms: ``svgText()`` called
``html.escape``, chordp and spreadlinesp called it inline, linkp hand-rolled two
separate ``.replace()`` chains (one to escape, one to un-escape for the GPU
path), and the interactive chrome escaped nothing because its strings are
trusted.  Nothing tied them together, so the convention broke where two of them
met.

The break was in linkp's node labels: the label column was escaped *first* and
the escaped string was then cut by **character count** -- word-wrapped into
``<tspan>``s at ``label_line_width``, and truncated by the ellipsis path.  A
``&`` or ``<`` landing near a cut split its entity across two elements::

    <tspan ...>aaaaaaaaa&</tspan><tspan ...>amp;bbbbbb</tspan>

which is not well-formed XML.  Browsers tolerate it via ``innerHTML``, so it
reads as display corruption rather than script execution, but a strict consumer
-- the PNG exporter, or an ``<img>`` tag -- rejects the whole document.  The
same escape-then-measure inversion made the culling width wrong: ``&`` counted
as the five characters of ``&amp;``.

The fix is one function, ``svgEscape()``, applied at the *last* step before the
text reaches markup.  These tests lock both halves: that the door exists and is
the only one (``TestSingleEscapeDoor`` scans the source), and that every
component's labels survive XML-special characters intact.
"""
import unittest
import datetime
import re
import pathlib
import xml.etree.ElementTree as ET

import polars as pl
from polars2svg import Polars2SVG
from polars2svg.p2s_text_mixin import svgEscape, svgUnescape


# The characters html.escape() spells as entities.  '&' has to come first in any
# hand-rolled chain, which is exactly the kind of ordering trap the single door
# removes.
_SPECIALS_ = ('&', '<', '>', '"', "'")

# Short enough to survive every component's cropping, and it carries the three
# characters that actually produce entities in text content.
_PAYLOAD_ = 'A&B<C>D'

# Long enough to force linkp's wrap and ellipsis paths at label_line_width=10.
_LONG_PAYLOAD_ = 'aaaaaaaaa&bbbbbbbbbbbbbbbbbbbb'


def _text_runs(svg):
    '''Every <text> element's character content, entities resolved by the parser.

    ET.fromstring() raises on a split entity, so this doubles as the
    well-formedness check; itertext() flattens <tspan>/<textPath> children, so a
    wrapped label reads back as the single string it was before wrapping.
    '''
    _root_ = ET.fromstring(svg)
    return [''.join(_el_.itertext()) for _el_ in _root_.iter()
            if _el_.tag.endswith('text')]


class _EscapeBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p2s = Polars2SVG()

    def _runs(self, component):
        return _text_runs(component._repr_svg_())


# ─────────────────────────────────────────────────────────────────────────────
# The door itself
# ─────────────────────────────────────────────────────────────────────────────

class TestEscapeDoorContract(_EscapeBase):

    def test_escapes_every_xml_special(self):
        _entities_ = {'&': '&amp;', '<': '&lt;', '>': '&gt;',
                      '"': '&quot;', "'": '&#x27;'}
        self.assertEqual(sorted(_entities_), sorted(_SPECIALS_))
        for _ch_, _ent_ in _entities_.items():
            with self.subTest(char=_ch_):
                self.assertEqual(svgEscape(f'x{_ch_}y'), f'x{_ent_}y')

    def test_output_reads_back_as_the_original(self):
        '''The only property that matters: parsed back out, it is what went in.'''
        for _s_ in ('A&B', '<script>alert(1)</script>', 'a & b & c',
                    '"quoted"', "it's", '&amp;', '日本語 & 🎉', ''):
            with self.subTest(s=_s_):
                _el_ = ET.fromstring(f'<t>{svgEscape(_s_)}</t>')
                self.assertEqual(''.join(_el_.itertext()), _s_)

    def test_round_trips_through_unescape(self):
        for _s_ in ('A&B', '&amp;', 'a<b>c', "d\"e'f", 'plain'):
            with self.subTest(s=_s_):
                self.assertEqual(svgUnescape(svgEscape(_s_)), _s_)

    def test_coerces_non_strings(self):
        self.assertEqual(svgEscape(42), '42')
        self.assertEqual(svgEscape(None), '')
        self.assertEqual(svgUnescape(None), '')

    def test_reachable_as_a_method(self):
        '''Components hold a Polars2SVG, not this module -- both routes are the door.'''
        self.assertEqual(self.p2s.svgEscape('A&B'), svgEscape('A&B'))
        self.assertEqual(self.p2s.svgUnescape('A&amp;B'), svgUnescape('A&amp;B'))


# ─────────────────────────────────────────────────────────────────────────────
# The reported break: linkp escaped, then cut by character count
# ─────────────────────────────────────────────────────────────────────────────

class TestLinkpNodeLabelEntityIntegrity(_EscapeBase):

    def _linkp(self, label, **kwargs):
        _df_ = pl.DataFrame({'fm': [label, 'beta'], 'to': ['beta', 'beta']})
        _kw_ = dict(relationships=[('fm', 'to')], draw_node_labels=True)
        _kw_.update(kwargs)
        return self.p2s.linkp(_df_, **_kw_)

    def test_wrap_does_not_split_an_entity(self):
        '''The audit's reproduction: '&' at offset 9 with a 10-char wrap.'''
        _runs_ = self._runs(self._linkp(_LONG_PAYLOAD_, label_line_width=10,
                                        label_max_lines=-1))
        self.assertIn(_LONG_PAYLOAD_, _runs_)

    def test_ellipsis_does_not_truncate_an_entity(self):
        '''lines[-1][:line_width - 1] cut '&amp;' down to a bare '&'.'''
        _runs_ = self._runs(self._linkp('aaaaaaaa&' + 'b' * 20, label_line_width=10,
                                        label_max_lines=1, label_ellipsis=True))
        _label_ = [_r_ for _r_ in _runs_ if _r_ != 'beta']
        self.assertEqual(len(_label_), 1)
        self.assertTrue(_label_[0].endswith('…'), _label_[0])
        self.assertTrue(('aaaaaaaa&' + 'b' * 20).startswith(_label_[0][:-1]), _label_[0])

    def test_special_char_at_every_wrap_offset(self):
        '''One offset in ten lands on a cut; sweep them all rather than guess.'''
        for _ch_ in ('&', '<', '>'):
            for _k_ in range(0, 24):
                _label_ = 'a' * _k_ + _ch_ + 'b' * (23 - _k_)
                with self.subTest(char=_ch_, offset=_k_):
                    _runs_ = self._runs(self._linkp(_label_, label_line_width=10,
                                                    label_max_lines=-1))
                    self.assertIn(_label_, _runs_)

    def test_special_char_at_every_ellipsis_offset(self):
        for _ch_ in ('&', '<', '>'):
            for _k_ in range(0, 24):
                _label_ = 'a' * _k_ + _ch_ + 'b' * (23 - _k_)
                with self.subTest(char=_ch_, offset=_k_):
                    # Parses at all == the entity survived the truncation.
                    self._runs(self._linkp(_label_, label_line_width=10,
                                           label_max_lines=2, label_ellipsis=True))

    def test_wrap_boundaries_are_measured_on_raw_text(self):
        '''Wrapping is by character count, so '&' must count as one character.

        Escaped first, 'aaaa&aaaaaaaa' is 17 characters and breaks in a
        different place than the same-length 'aaaaXaaaaaaaa'.
        '''
        def _line_lengths_(label):
            _lp_ = self._linkp(label, label_line_width=10, label_max_lines=-1)
            _lp_._repr_svg_()
            return sorted([len(_l_) for _, _, _lines_ in _lp_._node_label_info_
                           for _l_ in _lines_])
        self.assertEqual(_line_lengths_('aaaa&aaaaaaaa'),
                         _line_lengths_('aaaaXaaaaaaaa'))

    def test_gpu_label_lines_are_unescaped(self):
        '''_node_label_info_ feeds the glyph atlas, which draws characters, not
        entities -- it must hold the raw label, not a re-parsed one.'''
        _lp_ = self._linkp(_PAYLOAD_, label_line_width=32)
        _lp_._repr_svg_()
        _lines_ = [_l_ for _, _, _ls_ in _lp_._node_label_info_ for _l_ in _ls_]
        self.assertIn(_PAYLOAD_, _lines_)
        for _l_ in _lines_:
            self.assertNotIn('&amp;', _l_)
            self.assertNotIn('&lt;', _l_)

    def test_gpu_and_svg_agree_on_which_labels_survive(self):
        '''Culling width was measured on the escaped string, so a label with '&'
        was treated as wider than it draws.'''
        _lp_ = self._linkp(_LONG_PAYLOAD_, label_line_width=10, label_max_lines=-1)
        _svg_ = _lp_._repr_svg_()
        self.assertEqual(len(_lp_._node_label_info_),
                         len(re.findall(r'<text', _svg_)))


class TestLinkpLinkLabelEntityIntegrity(_EscapeBase):

    def test_cropped_link_label_stays_well_formed(self):
        '''cropText() cuts by pixel width on the raw string; the escape has to
        happen after it, or the crop lands inside an entity.'''
        for _label_ in (_PAYLOAD_, _LONG_PAYLOAD_, '&' * 20, 'x<' * 15):
            with self.subTest(label=_label_):
                _df_ = pl.DataFrame({'fm': ['a', 'b'], 'to': ['b', 'a'],
                                     'l': [_label_, 'z']})
                self._runs(self.p2s.linkp(_df_, relationships=[('fm', 'to', 'l')],
                                          draw_link_labels=True, wxh=(400, 400)))


# ─────────────────────────────────────────────────────────────────────────────
# Every component that puts row data into markup
# ─────────────────────────────────────────────────────────────────────────────

class TestLabelEscapingAcrossComponents(_EscapeBase):
    '''The convention is package-wide, so the check is too.

    Each case renders a label built from untrusted row data and asserts three
    things -- that the document parses, that it actually contains text (the
    trap the old linkp special-char test fell into: it passed ``node_labels=``
    instead of ``draw_node_labels=`` and asserted nothing about zero labels),
    and that the label reads back exactly as it went in.
    '''

    def _cases(self, payload):
        _p_ = payload
        _sdf_ = pl.DataFrame({'fm': [_p_, _p_, 'C'], 'to': ['C', 'E', 'E'],
                              'time': [datetime.datetime(2024, 1, _d_)
                                       for _d_ in (1, 2, 3)]})
        _smallp_df_ = pl.DataFrame({'fm': ['a', 'b'], 'to': ['b', 'a'],
                                    'cat': [_p_, 'z']})
        _tmpl_ = self.p2s.chordp(df=_smallp_df_, relationships=[('fm', 'to')],
                                 wxh=(128, 128))
        return {
            'histop':            self.p2s.histop(
                pl.DataFrame({'cat': [_p_, 'b'], 'v': [1, 2]}), 'cat'),
            'piep':              self.p2s.piep(
                pl.DataFrame({'cat': [_p_, 'b', 'c'], 'v': [3, 2, 1]}), 'cat',
                draw_labels=True),
            'xyp':               self.p2s.xyp(
                pl.DataFrame({'x': [_p_, 'b'], 'y': [1.0, 2.0]}), 'x', 'y'),
            'chordp':            self.p2s.chordp(
                df=pl.DataFrame({'fm': [_p_, 'b'], 'to': ['b', _p_]}),
                relationships=[('fm', 'to')], draw_labels=True),
            'linkp_node_labels': self.p2s.linkp(
                pl.DataFrame({'fm': [_p_, 'b'], 'to': ['b', _p_]}),
                relationships=[('fm', 'to')], draw_node_labels=True),
            'linkp_link_labels': self.p2s.linkp(
                pl.DataFrame({'fm': ['a', 'b'], 'to': ['b', 'a'], 'l': [_p_, 'z']}),
                relationships=[('fm', 'to', 'l')], draw_link_labels=True,
                wxh=(400, 400)),
            'linkp_legend':      self.p2s.linkp(
                pl.DataFrame({'fm': ['a', 'b'], 'to': ['b', 'c'], 'g': [_p_, 'z']}),
                relationships=[('fm', 'to')], color='g', legend=True,
                wxh=(300, 300)),
            'spreadlinesp':      self.p2s.spreadlinesp(
                _sdf_, [('fm', 'to')], ego=_p_, time='time',
                anno={'2024-01-01 00:00:00.000000': _p_}),
            'smallp':            self.p2s.smallp(_smallp_df_, _tmpl_, 'cat'),
        }

    def test_special_chars_round_trip(self):
        for _name_, _component_ in self._cases(_PAYLOAD_).items():
            with self.subTest(component=_name_):
                _runs_ = self._runs(_component_)          # parses == well-formed
                self.assertTrue(_runs_, 'component rendered no text at all')
                self.assertIn(_PAYLOAD_, _runs_)

    def test_no_double_escaping(self):
        '''A second trip through the door would leave '&amp;' in the text.'''
        for _name_, _component_ in self._cases(_PAYLOAD_).items():
            with self.subTest(component=_name_):
                for _run_ in self._runs(_component_):
                    for _entity_ in ('&amp;', '&lt;', '&gt;', '&quot;', '&#x27;'):
                        self.assertNotIn(_entity_, _run_)

    def test_long_labels_stay_well_formed(self):
        '''The cropping/wrapping paths, where the entity-splitting bug lived.'''
        for _name_, _component_ in self._cases(_LONG_PAYLOAD_).items():
            with self.subTest(component=_name_):
                self._runs(_component_)

    def test_quote_bearing_labels_stay_well_formed(self):
        for _name_, _component_ in self._cases('it\'s "A&B"').items():
            with self.subTest(component=_name_):
                self.assertTrue(self._runs(_component_))


# ─────────────────────────────────────────────────────────────────────────────
# ... and only one door
# ─────────────────────────────────────────────────────────────────────────────

class TestSingleEscapeDoor(unittest.TestCase):
    '''The audit's actual finding was structural: five mechanisms, no single
    door.  Fixing the call sites without closing the other doors just resets the
    clock, so this scans the package source for a sixth one.
    '''

    _DOOR_ = 'p2s_text_mixin.py'

    # A hand-rolled chain -- '&' -> '&amp;' or the reverse -- in any form.
    _AD_HOC_ = re.compile(r"""replace(_all)?\(\s*(['"])(&(amp|lt|gt|quot);|[<>&])\2""")
    _HTML_MOD_ = re.compile(r'\bhtml\b[\w]*\s*\.\s*(un)?escape\b')

    def _sources(self):
        _pkg_ = pathlib.Path(__file__).resolve().parent.parent / 'polars2svg'
        for _p_ in sorted(_pkg_.rglob('*.py')):
            if _p_.name == self._DOOR_:
                continue
            yield _p_, _p_.read_text(encoding='utf-8')

    def test_no_direct_html_escape_calls(self):
        for _path_, _src_ in self._sources():
            with self.subTest(module=_path_.name):
                self.assertIsNone(
                    self._HTML_MOD_.search(_src_),
                    f'{_path_.name} calls html.escape/unescape directly; '
                    f'use svgEscape()/svgUnescape() from {self._DOOR_}')

    def test_no_ad_hoc_entity_replace_chains(self):
        for _path_, _src_ in self._sources():
            with self.subTest(module=_path_.name):
                _m_ = self._AD_HOC_.search(_src_)
                self.assertIsNone(
                    _m_,
                    f'{_path_.name} hand-rolls entity replacement ({_m_.group(0) if _m_ else ""}); '
                    f'use svgEscape()/svgUnescape() from {self._DOOR_}')

    def test_the_door_is_where_the_test_says_it_is(self):
        _pkg_ = pathlib.Path(__file__).resolve().parent.parent / 'polars2svg'
        _src_ = (_pkg_ / self._DOOR_).read_text(encoding='utf-8')
        self.assertIn('def svgEscape(', _src_)
        self.assertIn('def svgUnescape(', _src_)


if __name__ == '__main__':
    unittest.main()
