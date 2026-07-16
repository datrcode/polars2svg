"""Shared fixtures for string-fidelity tests across every component.

A single hostile corpus of strings (FIDELITY_LABELS) is reused by every component's
fidelity test so they all exercise the same edge cases: integers, floats, scientific
notation, IPv4/IPv6, names, mixed/upper/lower case, and XML-escape-required symbols.
These strings regressed when roundSvgFloats() rounded any digit-dot-digit run anywhere in
the finished SVG -- inside <text>/<tspan> content, not just numeric attributes -- e.g. a
node label "1.172.32.1" was corrupted to "1.17.32.1". See the TODO on
Polars2SVG.roundSvgFloats.

Any string that appears as user-supplied text in a component's output -- a linkp node
label, a linkp node_labels dict value, or a draw_context axis/field label naming a column
-- must come back byte-for-byte after the SVG is parsed and its XML entities decoded.
"""
import re
import xml.etree.ElementTree as ET

SVG_NS = '{http://www.w3.org/2000/svg}'

# The hostile corpus. Every entry is unique so set comparisons are unambiguous.
FIDELITY_LABELS = [
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

# Labels the (disabled) rounding logic provably corrupts. Negative tests assert these
# break so a guard test can never silently pass with rounding re-enabled.
KNOWN_ROUNDING_VICTIMS = [
    '3.14159', '123.456789', '-0.00042', '2.71828', '6.022e23',
    '10.160.170.21', '1.172.32.1', '192.168.100.200', '255.255.255.0',
]


# Faithful copy of the ORIGINAL Polars2SVG.roundSvgFloats body (now disabled in the
# source). Kept here so negative tests can prove that re-enabling that logic corrupts
# rendered text WITHOUT flipping the production source. If the real helper is ever
# rewritten to be text-content-safe, this reproduction -- and the negative tests -- go.
_FAULTY_FLOAT_RE = re.compile(r'-?\d*\.\d+')


def faulty_round_svg_floats(svg, digits=2):
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


def svg_text_contents(svg):
    """Every <text> element's content with XML entities decoded (ElementTree unescapes),
    joining multi-line <tspan> runs so the result is the reconstructed display string."""
    root = ET.fromstring(svg)
    out = []
    for _t_ in root.iter(SVG_NS + 'text'):
        _tspans_ = list(_t_.iter(SVG_NS + 'tspan'))
        if _tspans_:
            out.append(''.join(_ts_.text or '' for _ts_ in _tspans_))
        else:
            out.append(_t_.text or '')
    return out
