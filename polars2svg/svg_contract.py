#
# svg_contract - the Profile A ("Appliance") output contract
#
# polars2svg's supported deployment profiles are described in SECURITY.md.
# Profile A is the one where a server renders SVG from data it does not trust and
# embeds the result in a page somebody else views.  What makes that safe is not a
# promise that every label got escaped -- it is that the *output* is checkable,
# because polars2svg generates every element and attribute in a render itself,
# from numeric geometry and a fixed vocabulary.  Row data reaches the document
# only as text content and as a handful of values that are escaped on the way in.
#
# So the contract is an ALLOW-LIST rather than a deny-list.  A render conforms
# when every element and every attribute in it is one this module names, and
# every URL-bearing value points inside the same document.  A deny-list ("no
# <script>, no on* handlers, no <foreignObject>") only ever excludes the attacks
# somebody thought of; an allow-list excludes the ones nobody thought of too.
# What makes that affordable here is how small the real vocabulary is: 16
# elements and 44 attributes cover every render the test suite produces.
#
# checkOutputContract() reports every violation; assertOutputContract() raises on
# the first.  Both are public, and deliberately so -- an appliance can gate its
# own responses with them, which is what makes Profile A an enforced property
# rather than a documented hope.
#
# What this does NOT do, and must not be read as doing:
#
# - It is not a sanitizer.  It reports; it never rewrites.  A violation means the
#   render is not fit to serve, not that it has been made fit.
# - It says nothing about tile().  tile() embeds foreign SVG verbatim -- that is
#   the component (PLANNING.md A3) -- so a Tile is outside Profile A by
#   construction.  Running this over one is a meaningful thing for a caller to do
#   with markup it did not write, which is exactly why the parser refuses a
#   DOCTYPE instead of trusting the input to be ours.
# - It is not a defence against the *interactive* surface.  That is Profile B,
#   which is out of scope (SECURITY.md, PLANNING.md §13).
#
from typing import NamedTuple
import re
import xml.etree.ElementTree as ET


#
# The permitted element vocabulary.  Measured, not guessed: this is what the
# components actually emit across the golden corpus plus the configuration matrix
# in tests/test_profile_conformance.py.  Adding a name here widens the contract,
# so it needs the same scrutiny as any other security control -- in particular
# <a>, <foreignObject>, <script>, <animate>/<set> (which can retarget an
# attribute) and <image>/<use> pointing outward are all absent on purpose.
#
ALLOWED_ELEMENTS = frozenset({
    'svg', 'g', 'defs', 'style',
    'rect', 'circle', 'ellipse', 'line', 'path', 'polygon', 'polyline',
    'text', 'tspan', 'textPath',
    'clipPath', 'linearGradient', 'stop', 'use',
})

#
# The permitted attribute vocabulary (local names -- any namespace prefix is
# stripped before the check, so xlink:href is tested as 'href').  Presentational
# and geometric only: nothing here can introduce behaviour, and the two that can
# introduce a *reference* (href, clip-path) are additionally range-checked below.
#
ALLOWED_ATTRIBUTES = frozenset({
    # identity / structure.  'href' is permitted only as a same-document
    # fragment -- being in this set is necessary but not sufficient, see
    # URL_BEARING_ATTRIBUTES below.
    'id', 'class', 'xmlns', 'viewBox', 'transform', 'href',
    # geometry
    'x', 'y', 'x1', 'y1', 'x2', 'y2', 'cx', 'cy', 'r', 'rx', 'ry',
    'width', 'height', 'd', 'points', 'dx', 'dy',
    # paint
    'fill', 'fill-opacity', 'fill-rule', 'stroke', 'stroke-width',
    'stroke-opacity', 'stroke-dasharray', 'stroke-linecap', 'stroke-linejoin',
    'opacity', 'stop-color', 'stop-opacity', 'offset', 'gradientUnits',
    # text
    'font-family', 'font-size', 'font-weight', 'font-style',
    'text-anchor', 'dominant-baseline', 'startOffset', 'textLength',
    'lengthAdjust', 'writing-mode',
    # clipping
    'clip-path', 'clip-rule', 'clipPathUnits',
})

#
# Attributes whose value may name a resource.  Every one of them must resolve
# inside this document: '#fragment' or 'url(#fragment)' and nothing else.  An
# external reference is what turns a render into a request (a tracking pixel, an
# SSRF from whatever rasterizes it, a local file read -- see the comment above
# svgToPNGBytes() in export.py for the last one).
#
URL_BEARING_ATTRIBUTES = frozenset({'href', 'clip-path', 'fill', 'stroke', 'mask', 'filter'})

_LOCAL_REF_        = re.compile(r'^#[A-Za-z0-9_.:-]+$')
_URL_LOCAL_REF_    = re.compile(r'^url\(\s*#[A-Za-z0-9_.:-]+\s*\)$')
_URL_CALL_         = re.compile(r'url\s*\(', re.IGNORECASE)
_NS_PREFIX_        = re.compile(r'^\{[^}]*\}')
_DOCTYPE_OR_ENTITY_ = re.compile(r'<!\s*(DOCTYPE|ENTITY)', re.IGNORECASE)
# Anything that is a scheme we will not serve.  Checked on every attribute value,
# not only the URL-bearing ones, because a value does not need to be in a
# reference position to matter if the document is later transformed.
_DANGEROUS_SCHEME_ = re.compile(r'(?:javascript|vbscript|data)\s*:', re.IGNORECASE)
# CSS constructs that fetch or execute, checked inside <style> element text.
_CSS_DANGEROUS_    = re.compile(r'@import|expression\s*\(|javascript\s*:', re.IGNORECASE)


class Violation(NamedTuple):
    """One way a document failed the contract.

    `kind` is a stable machine-readable slug; `detail` is the sentence to show a
    human.  `element` / `attribute` locate it as well as a streaming parser can
    (there are no line numbers -- the renders are single-line by construction).
    """
    kind:      str
    detail:    str
    element:   str | None = None
    attribute: str | None = None
    value:     str | None = None


class OutputContractError(ValueError):
    """Raised by assertOutputContract() when a document violates the contract.

    Carries the full violation list on `.violations`, so a caller that wants to
    report all of them does not have to re-run the check.
    """
    def __init__(self, violations: list[Violation]) -> None:
        self.violations = violations
        _first_ = violations[0]
        super().__init__(
            f'SVG output contract violated ({len(violations)} violation(s)); '
            f'first: {_first_.detail}'
        )


#
# _localName_() - element/attribute name with any {namespace} prefix removed
# - ElementTree reports namespaced names as '{uri}local'; the contract is written
#   in terms of local names, so xlink:href and a bare href are one rule
#
def _localName_(name: str) -> str:
    return _NS_PREFIX_.sub('', name)


#
# _checkValue_() - the per-attribute-value rules
# - appends to `out` rather than returning, so one attribute can fail more than
#   one way and the caller sees both
#
def _checkValue_(tag: str, attr: str, value: str, out: list[Violation]) -> None:
    if _DANGEROUS_SCHEME_.search(value):
        # data: is refused alongside javascript:/vbscript: on purpose.  A
        # data:image/png is harmless, but data:text/html is a same-origin
        # document, and telling them apart by sniffing the media type is exactly the
        # kind of parsing that goes wrong.  Nothing polars2svg renders needs one.
        out.append(Violation(
            'dangerous-scheme',
            f'<{tag} {attr}="..."> uses a scheme that is not servable '
            f'(javascript:, vbscript: and data: are all refused)',
            tag, attr, value))
    _is_url_call_ = _URL_CALL_.search(value) is not None
    if attr in URL_BEARING_ATTRIBUTES or _is_url_call_:
        if _is_url_call_:
            # A url(...) in any attribute, whether or not that attribute is one we
            # expect to carry a reference.
            if _URL_LOCAL_REF_.match(value.strip()) is None:
                out.append(Violation(
                    'external-reference',
                    f'<{tag} {attr}="..."> references a resource outside this '
                    f'document; only url(#fragment) is permitted',
                    tag, attr, value))
        elif attr == 'href':
            if _LOCAL_REF_.match(value.strip()) is None:
                out.append(Violation(
                    'external-reference',
                    f'<{tag} href="..."> must reference a fragment in this same '
                    f'document (#id); external and relative targets are refused',
                    tag, attr, value))


#
# checkOutputContract() - every way `svg` fails the Profile A contract
# - returns [] for a conforming document; never raises for contract reasons, so a
#   caller can report all findings at once (a malformed document is itself
#   reported as a violation rather than propagating the parser's exception)
#
def checkOutputContract(svg: str) -> list[Violation]:
    _out_: list[Violation] = []

    # Refuse a DOCTYPE outright rather than parsing it carefully.  polars2svg
    # never emits one, and refusing is what keeps this safe to point at foreign
    # markup: entity expansion is the whole billion-laughs / XXE family, and not
    # parsing it at all beats configuring a parser not to expand it.
    if _DOCTYPE_OR_ENTITY_.search(svg):
        _out_.append(Violation(
            'doctype',
            'document contains a DOCTYPE or ENTITY declaration; polars2svg emits '
            'neither, and they are refused rather than expanded'))
        return _out_

    try:
        _root_ = ET.fromstring(svg)
    except ET.ParseError as _e_:
        return [Violation('not-well-formed', f'document is not well-formed XML: {_e_}')]

    for _el_ in _root_.iter():
        if not isinstance(_el_.tag, str):    # comments / PIs surface as callables
            _out_.append(Violation(
                'non-element-node',
                'document contains a comment or processing instruction'))
            continue
        _tag_ = _localName_(_el_.tag)
        if _tag_ not in ALLOWED_ELEMENTS:
            _out_.append(Violation(
                'element-not-allowed',
                f'<{_tag_}> is not in the permitted element vocabulary',
                _tag_))
        for _raw_attr_, _value_ in _el_.attrib.items():
            _attr_ = _localName_(_raw_attr_)
            # Checked ahead of the allow-list so the message names the real
            # problem: an event handler is never a missing-vocabulary entry.
            if _attr_.lower().startswith('on'):
                _out_.append(Violation(
                    'event-handler',
                    f'<{_tag_} {_attr_}="..."> is an event handler attribute',
                    _tag_, _attr_, _value_))
                continue
            if _attr_ not in ALLOWED_ATTRIBUTES:
                _out_.append(Violation(
                    'attribute-not-allowed',
                    f'<{_tag_} {_attr_}="..."> is not in the permitted attribute '
                    f'vocabulary',
                    _tag_, _attr_, _value_))
                continue
            _checkValue_(_tag_, _attr_, _value_, _out_)
        if _tag_ == 'style' and _el_.text and _CSS_DANGEROUS_.search(_el_.text):
            _out_.append(Violation(
                'dangerous-css',
                '<style> content uses @import, expression() or a javascript: URL',
                _tag_))

    return _out_


#
# assertOutputContract() - raise unless `svg` conforms
# - the gate an appliance puts in front of a response; OutputContractError
#   carries every violation on .violations, not just the one in the message
#
def assertOutputContract(svg: str) -> None:
    _violations_ = checkOutputContract(svg)
    if _violations_:
        raise OutputContractError(_violations_)
