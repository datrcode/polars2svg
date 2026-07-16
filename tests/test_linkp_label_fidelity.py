import re
import unittest
import xml.etree.ElementTree as ET

import polars as pl

from polars2svg import Polars2SVG

SVG_NS = '{http://www.w3.org/2000/svg}'


# A deliberately hostile spread of node labels. Every entry must survive the linkp
# label pipeline (HTML-escape -> word-wrap -> <text> emit) and come back byte-for-byte
# after the SVG is parsed and its entities decoded. The float / IP / scientific-notation
# rows are the ones that regressed when roundSvgFloats() rounded digit-dot-digit runs
# anywhere in the finished SVG -- e.g. node label "1.172.32.1" was corrupted to
# "1.17.32.1". See the TODO on Polars2SVG.roundSvgFloats.
LABELS = [
    # integers
    '42', '0', '-17', '1000000',
    # floats (multiple fractional digits -- prime rounding targets)
    '3.14159', '123.456789', '-0.00042', '2.71828',
    # scientific notation
    '1.23e10', '6.022e23', '1.6e-19', '-4.5E+3',
    # IPv4 addresses
    '10.160.170.21', '1.172.32.1', '192.168.100.200', '255.255.255.0',
    # IPv6 addresses
    '2001:0db8:85a3:0000:0000:8a2e:0370:7334', 'fe80::1ff:fe23:4567:890a', '::1',
    # proper names / spaces / non-ASCII
    'Alice', 'Bob Smith', 'José García',
    # upper / lower / mixed case
    'UPPERCASE', 'lowercase', 'MixedCase', 'camelCaseWord',
    # special symbols that MUST be XML-escaped
    'a & b', 'x < y', 'p > q', '<tag>', 'A & B < C > D', '100% & <done>',
    # other special symbols that pass through verbatim
    'foo@bar.com', 'path/to/file', 'key=value', "it's", '#hashtag', 'c++',
    '$price', 'back\\slash', 'a"quoted"b',
]

# Labels that the (disabled) rounding logic provably corrupts. The negative test asserts
# these break so the guard test can never silently pass with rounding re-enabled.
KNOWN_ROUNDING_VICTIMS = [
    '3.14159', '123.456789', '-0.00042', '2.71828', '6.022e23',
    '10.160.170.21', '1.172.32.1', '192.168.100.200', '255.255.255.0',
]


# Faithful copy of the ORIGINAL Polars2SVG.roundSvgFloats body (now disabled in the
# source). Kept here so the negative test can prove that re-enabling that logic corrupts
# labels, without flipping the production source. If the real helper is ever rewritten
# to be text-content-safe, this local reproduction -- and the negative test -- can go.
_FAULTY_FLOAT_RE = re.compile(r'-?\d*\.\d+')


def _faulty_round_svg_floats(svg, digits=2):
    def _round(_m_):
        _s_ = _m_.group(0)
        _frac_ = _s_.split('.', 1)[1]
        if len(_frac_) <= digits:
            return _s_
        _out_ = f'{round(float(_s_), digits):.{digits}f}'.rstrip('0').rstrip('.')
        if _out_ in ('', '-', '-0'):
            _out_ = '0'
        return _out_
    return _FAULTY_FLOAT_RE.sub(_round, svg)


def _build_linkp(p2s, labels):
    '''Render every label as its own single (non-collapsed), drawn node.

    A chain of edges makes each label a node; an explicit grid pos keeps every node on a
    distinct screen pixel so none collapse (collapsed nodes are not labeled). A wide
    label_line_width with unlimited lines prevents word-wrapping, so each label stays a
    single <text> element whose content is exactly the (escaped) label.'''
    df = pl.DataFrame({'fm': labels, 'to': labels[1:] + labels[:1]})
    pos = {lab: [(i % 6) * 2.0, -(i // 6) * 2.0] for i, lab in enumerate(labels)}
    return p2s.linkp(df, [('fm', 'to')], pos=pos, draw_labels=True,
                     label_line_width=128, label_max_lines=-1, wxh=(1400, 1000))


def _extract_label_texts(svg):
    '''Decode every <text> element's content (ElementTree unescapes XML entities), joining
    multi-line <tspan> runs so the result is the reconstructed label string.'''
    root = ET.fromstring(svg)
    out = []
    for _t_ in root.iter(SVG_NS + 'text'):
        _tspans_ = list(_t_.iter(SVG_NS + 'tspan'))
        if _tspans_:
            out.append(''.join(_ts_.text or '' for _ts_ in _tspans_))
        else:
            out.append(_t_.text or '')
    return out


class TestLinkPLabelFidelity(unittest.TestCase):
    '''linkp node labels must render true to form for every kind of string.'''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def setUp(self):
        self.assertEqual(len(LABELS), len(set(LABELS)), 'test labels must be unique')

    # ------------------------------------------------------------------
    # Positive: every label survives the round trip exactly.
    # ------------------------------------------------------------------
    def test_all_labels_render_true_to_form(self):
        lp = _build_linkp(self.p2s, LABELS)
        # Sanity: no node collapsed, so every label is actually drawn.
        self.assertEqual(lp.df_node.filter(pl.col('__nodes__') == 1).height, len(LABELS))

        rendered = _extract_label_texts(lp.svg)
        # The only <text> elements are node labels -- exactly one per label, no more.
        self.assertEqual(len(rendered), len(LABELS),
                         'unexpected number of <text> elements in linkp SVG')
        for _lab_ in LABELS:
            self.assertIn(_lab_, rendered,
                          f'label {_lab_!r} missing or corrupted in rendered SVG')
        self.assertEqual(sorted(rendered), sorted(LABELS))

    def test_rendered_svg_is_well_formed_xml(self):
        # Escaping must keep the SVG parseable even with &, <, > in labels.
        lp = _build_linkp(self.p2s, LABELS)
        ET.fromstring(lp.svg)  # raises on malformed XML

    def test_special_symbol_labels_are_escaped_in_source(self):
        # The raw SVG string must carry entity-escaped forms, not bare &, <, > from labels.
        lp = _build_linkp(self.p2s, ['a & b', 'x < y', 'p > q'])
        self.assertIn('a &amp; b', lp.svg)
        self.assertIn('x &lt; y', lp.svg)
        self.assertIn('p &gt; q', lp.svg)

    # ------------------------------------------------------------------
    # Negative: re-applying the disabled rounding provably corrupts labels,
    # and this test's verification method detects it.
    # ------------------------------------------------------------------
    def test_faulty_rounding_would_corrupt_labels(self):
        lp = _build_linkp(self.p2s, LABELS)
        # Correct as rendered (rounding is off).
        self.assertIn('1.172.32.1', _extract_label_texts(lp.svg))

        # Apply the faulty (disabled) rounding to the finished SVG and re-extract.
        corrupted_svg = _faulty_round_svg_floats(lp.svg)
        rendered_after = _extract_label_texts(corrupted_svg)

        # The exact bug from the field report: 1.172.32.1 -> 1.17.32.1.
        self.assertNotIn('1.172.32.1', rendered_after)
        self.assertIn('1.17.32.1', rendered_after)

        # Every known-vulnerable label must fail to round-trip.
        for _victim_ in KNOWN_ROUNDING_VICTIMS:
            self.assertNotIn(_victim_, rendered_after,
                             f'faulty rounding should have corrupted {_victim_!r}')

    # ------------------------------------------------------------------
    # Guard: the production rounding helper must stay OFF (a no-op). If it is
    # ever re-enabled without a text-content-safe rewrite, this fails loudly.
    # ------------------------------------------------------------------
    def test_production_rounding_is_disabled(self):
        probe = '<text x="1.172">1.172.32.1</text> more="123.456789"'
        self.assertEqual(self.p2s.roundSvgFloats(probe), probe,
                         'roundSvgFloats() is enabled again -- it corrupts node labels; '
                         'see the TODO on Polars2SVG.roundSvgFloats')


if __name__ == '__main__':
    unittest.main()
