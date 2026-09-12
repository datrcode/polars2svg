'''Profile conformance -- the evidence behind SECURITY.md's deployment profiles.

SECURITY.md names three profiles.  This file is what makes two of those names
mean something an auditor can check rather than something the project asserts:

  Profile A (Appliance)   -- every render conforms to the output contract in
                             polars2svg/svg_contract.py, including renders whose
                             row data is actively hostile.
  Profile B (Interactive) -- OUT OF SCOPE.  Not hardened, not tested here, and
                             SECURITY.md has to keep saying so.

The structure is deliberate, and each piece exists because the others can fail
silently without it:

  * the contract check is the actual property (an allow-list over the output),
  * the corpus drives hostile data through every text surface,
  * the non-vacuity check proves those renders actually carry the payload,
  * the coverage ratchet stops a new component shipping without cases, and
  * the self-test proves the checker can still fail.

Remove any one and the rest can go green while proving nothing.

One class sits slightly apart: tile() is excluded from Profile A, and
TestTileIsExcludedFromTheProfileButStillChecked pins the distinction between the
library making no *promise* about foreign SVG and the contract being *blind* to
it.  It is not blind, which is what an appliance gating on assertOutputContract()
depends on.
'''
import glob
import os
import unittest
import xml.etree.ElementTree as ET

import polars as pl

from polars2svg import Polars2SVG
from polars2svg.export import ExportMixin
from polars2svg.svg_contract import (ALLOWED_ATTRIBUTES, ALLOWED_ELEMENTS,
                                     OutputContractError, assertOutputContract,
                                     checkOutputContract)

from injection_corpus import PAYLOADS, renderMatrix

# Component classes that must own injection cases, mapped to the case-name prefix
# that covers them.  The mapping is explicit rather than derived from the class
# name because the factory names and the class names disagree (ChP -> chordp),
# and because an exclusion has to be a visible decision with a reason attached.
#
# TestCorpusCoversEveryComponent checks this dict against the real subclass set,
# so a ninth component fails here until somebody decides which side it is on.
_COMPONENT_PREFIXES_ = {
    'XYp':          'xyp',
    'Histop':       'histop',
    'Timep':        'timep',
    'Piep':         'piep',
    'LinkP':        'linkp',
    'ChP':          'chordp',
    'Smallp':       'smallp',
    'SpreadLinesP': 'spreadlinesp',
}

# Tile is outside Profile A by construction: it embeds foreign SVG verbatim,
# which is the component rather than a defect (PLANNING.md A3).  A Tile of
# conforming children conforms, but tile() makes no promise about what it is
# handed, so it gets no injection cases and is named here instead.
_OUT_OF_PROFILE_ = {'Tile'}

_GOLDEN_DIR_ = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'golden')
# Long enough to be unmistakable, plain enough to survive an id sanitizer.
_PROBE_      = 'ZqXmarkerQz'


class TestOutputContractSelfTest(unittest.TestCase):
    '''The checker must still be able to fail.

    Every other test in this file asserts that something conforms, so all of them
    pass just as happily against a checker that returns [] unconditionally.  These
    are hand-written documents that must be rejected, one per violation kind.
    '''

    def _kinds(self, svg):
        return {_v_.kind for _v_ in checkOutputContract(svg)}

    def test_script_element_rejected(self):
        self.assertIn('element-not-allowed', self._kinds(
            '<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>'))

    def test_foreign_object_rejected(self):
        self.assertIn('element-not-allowed', self._kinds(
            '<svg xmlns="http://www.w3.org/2000/svg"><foreignObject/></svg>'))

    def test_event_handler_attribute_rejected(self):
        self.assertIn('event-handler', self._kinds(
            '<svg xmlns="http://www.w3.org/2000/svg">'
            '<text onclick="alert(1)">x</text></svg>'))

    def test_unknown_attribute_rejected(self):
        self.assertIn('attribute-not-allowed', self._kinds(
            '<svg xmlns="http://www.w3.org/2000/svg"><rect data-payload="x"/></svg>'))

    def test_external_href_rejected(self):
        self.assertIn('external-reference', self._kinds(
            '<svg xmlns="http://www.w3.org/2000/svg">'
            '<use href="http://example.invalid/x.svg"/></svg>'))

    def test_external_url_reference_rejected(self):
        self.assertIn('external-reference', self._kinds(
            '<svg xmlns="http://www.w3.org/2000/svg">'
            '<rect fill="url(http://example.invalid/x)"/></svg>'))

    def test_javascript_scheme_rejected(self):
        self.assertIn('dangerous-scheme', self._kinds(
            '<svg xmlns="http://www.w3.org/2000/svg">'
            '<use href="javascript:alert(1)"/></svg>'))

    def test_doctype_rejected(self):
        self.assertIn('doctype', self._kinds(
            '<!DOCTYPE svg [<!ENTITY e SYSTEM "file:///etc/passwd">]>'
            '<svg xmlns="http://www.w3.org/2000/svg"/>'))

    def test_dangerous_css_rejected(self):
        self.assertIn('dangerous-css', self._kinds(
            '<svg xmlns="http://www.w3.org/2000/svg">'
            '<style>@import url("http://example.invalid/x.css");</style></svg>'))

    def test_external_css_url_rejected(self):
        # The style block used to be checked with a three-item deny-list, so a
        # reference that was not @import / expression() / javascript: conformed --
        # while the byte-identical value in a fill *attribute* was refused (see
        # test_external_url_reference_rejected above, which is this test's mirror).
        # These are the three shapes that gap was found with.
        for _name_, _css_ in {
            'fill':             'rect{fill:url(http://example.invalid/x#g)}',
            'font_face_src':    '@font-face{font-family:e;src:url(http://example.invalid/f.woff)}',
            'background_image': 'rect{background-image:url(http://example.invalid/b.png)}',
        }.items():
            with self.subTest(css=_name_):
                self.assertIn('external-reference', self._kinds(
                    '<svg xmlns="http://www.w3.org/2000/svg">'
                    f'<style>{_css_}</style></svg>'))

    def test_css_url_quoting_does_not_evade_the_rule(self):
        # CSS spells one reference three ways.  A rule written against the bare
        # form only -- which is what the attribute rule (_URL_LOCAL_REF_) can
        # afford to be, because an attribute value is the whole token -- is
        # evaded by adding quotes.  Protocol-relative and relative targets are
        # here for the same reason: "not obviously a URL" is not a range check.
        for _name_, _url_ in {
            'bare':              'url(http://example.invalid/x)',
            'double_quoted':     'url("http://example.invalid/x")',
            'single_quoted':     "url('http://example.invalid/x')",
            'inner_whitespace':  'url( http://example.invalid/x )',
            'protocol_relative': 'url(//example.invalid/x)',
            'relative_path':     'url(../secret.svg)',
        }.items():
            with self.subTest(url=_name_):
                self.assertIn('external-reference', self._kinds(
                    '<svg xmlns="http://www.w3.org/2000/svg">'
                    f'<style>rect{{fill:{_url_}}}</style></svg>'))

    def test_css_fetch_without_a_url_token_rejected(self):
        # The url() range check cannot see these: CSS Images 4 lets a bare string
        # stand in for a url() inside image-set(), so the reference is there and
        # the token is not.  They are deny-listed rather than range-checked, which
        # is the documented seam in an otherwise allow-listed rule -- see the
        # stylesheet non-goal in SECURITY.md.
        for _name_, _css_ in {
            'image_set':        'rect{background-image:image-set("http://example.invalid/x.png" 1x)}',
            'webkit_image_set': 'rect{background-image:-webkit-image-set("http://example.invalid/x.png" 1x)}',
        }.items():
            with self.subTest(css=_name_):
                self.assertIn('dangerous-css', self._kinds(
                    '<svg xmlns="http://www.w3.org/2000/svg">'
                    f'<style>{_css_}</style></svg>'))

    def test_local_css_url_accepted(self):
        # The positive control for the two above.  Without it they are satisfied
        # by a rule that flags every url() in every stylesheet, which would break
        # the gradient and clip-path references the components legitimately emit.
        self.assertEqual(checkOutputContract(
            '<svg xmlns="http://www.w3.org/2000/svg">'
            '<style>rect{fill:url(#g)}circle{clip-path:url("#c")}</style>'
            '<rect/></svg>'), [])

    def test_malformed_document_rejected(self):
        self.assertIn('not-well-formed', self._kinds(
            '<svg xmlns="http://www.w3.org/2000/svg"><text>unclosed</svg>'))

    def test_local_fragment_reference_accepted(self):
        # The mirror of the rejections: a same-document reference is the whole
        # point of allowing href at all, so it must NOT be flagged.
        self.assertEqual(checkOutputContract(
            '<svg xmlns="http://www.w3.org/2000/svg">'
            '<rect clip-path="url(#c)"/><use href="#c"/></svg>'), [])

    def test_assert_raises_and_carries_every_violation(self):
        with self.assertRaises(OutputContractError) as _ctx_:
            assertOutputContract('<svg xmlns="http://www.w3.org/2000/svg">'
                                 '<script/><foreignObject/></svg>')
        self.assertGreaterEqual(len(_ctx_.exception.violations), 2)


class TestGoldenCorpusConformsToProfileA(unittest.TestCase):
    '''Every stored golden render conforms.

    The goldens are the curated configuration matrix -- 72 renders covering
    component/parameter combinations no hand-written injection case would think
    to reproduce -- so running the contract over them is the cheapest broad
    coverage available, and it grows automatically as goldens are added.
    '''

    def test_every_golden_conforms(self):
        _paths_ = sorted(glob.glob(os.path.join(_GOLDEN_DIR_, '*.svg')))
        self.assertGreater(len(_paths_), 50, 'golden corpus not found or unexpectedly small')
        for _path_ in _paths_:
            with self.subTest(golden=os.path.basename(_path_)):
                with open(_path_, encoding='utf-8') as _f_:
                    _violations_ = checkOutputContract(_f_.read())
                self.assertEqual(_violations_, [], f'{os.path.basename(_path_)}: '
                                                   f'{[_v_.detail for _v_ in _violations_]}')


class TestInjectionCorpusConformsToProfileA(unittest.TestCase):
    '''Hostile row data through every text surface still yields a conforming render.

    This is Profile A's actual claim.  Note what it does NOT assert: that the
    output is byte-identical to anything, or that a particular escape sequence was
    used.  Either would break on a legitimate rendering change while saying
    nothing extra -- the property that matters is structural, and the contract is
    the structure.
    '''

    def setUp(self):
        self.p2s = Polars2SVG()

    def test_every_payload_through_every_surface(self):
        for _pname_, _payload_ in PAYLOADS.items():
            for _case_ in renderMatrix(self.p2s, _payload_):
                with self.subTest(payload=_pname_, case=_case_.name):
                    _svg_ = _case_.render()._repr_svg_()
                    _violations_ = checkOutputContract(_svg_)
                    self.assertEqual(
                        _violations_, [],
                        f'{_case_.name} with payload {_pname_!r} produced a render '
                        f'outside the Profile A contract: '
                        f'{[_v_.detail for _v_ in _violations_]}')

    def test_renders_are_well_formed_xml(self):
        # Implied by the contract, asserted separately so a failure says which of
        # the two properties broke.
        for _pname_, _payload_ in PAYLOADS.items():
            for _case_ in renderMatrix(self.p2s, _payload_):
                with self.subTest(payload=_pname_, case=_case_.name):
                    ET.fromstring(_case_.render()._repr_svg_())


class TestInjectionCorpusIsNotVacuous(unittest.TestCase):
    '''The corpus must actually put the string on the canvas.

    Without this, a component that silently dropped every label would pass
    TestInjectionCorpusConformsToProfileA perfectly.  Five of the first seventeen
    cases written were vacuous exactly this way (see injection_corpus.py's
    docstring), so this is a regression test for the corpus itself.

    Uses a benign probe rather than a payload: with a hostile string the id
    sanitizer and the label cropper both legitimately transform it, and "absorbed
    safely" would be indistinguishable from "never rendered".
    '''

    def setUp(self):
        self.p2s = Polars2SVG()

    def _reach(self, svg):
        _root_  = ET.fromstring(svg)
        _text_  = ' '.join(_t_ for _t_ in _root_.itertext() if _t_)
        _attrs_ = ' '.join(_v_ for _el_ in _root_.iter() for _v_ in _el_.attrib.values())
        # A prefix, not the whole probe: components crop labels to fit, and
        # cropping is a legitimate transformation (cropping an ESCAPED string is
        # not -- see the slice-then-escape note in spreadlinesp.py).
        _found_ = set()
        for _n_ in range(len(_PROBE_), 3, -1):
            if _PROBE_[:_n_] in _text_:  _found_.add('text');  break
        for _n_ in range(len(_PROBE_), 3, -1):
            if _PROBE_[:_n_] in _attrs_: _found_.add('attr');  break
        return _found_

    def test_every_case_reaches_where_it_claims(self):
        for _case_ in renderMatrix(self.p2s, _PROBE_):
            with self.subTest(case=_case_.name):
                _found_ = self._reach(_case_.render()._repr_svg_())
                if _case_.reach == 'absorbed':
                    self.assertEqual(
                        _found_, set(),
                        f'{_case_.name} claims reach="absorbed" ({_case_.why}) but the '
                        f'probe reached {sorted(_found_)} -- the claim is now false')
                else:
                    self.assertIn(
                        _case_.reach, _found_,
                        f'{_case_.name} claims reach={_case_.reach!r} but the probe '
                        f'reached {sorted(_found_) or "NOWHERE"} -- this case is '
                        f'vacuous and proves nothing about escaping')

    def test_absorbed_cases_explain_themselves(self):
        for _case_ in renderMatrix(self.p2s, _PROBE_):
            if _case_.reach == 'absorbed':
                with self.subTest(case=_case_.name):
                    self.assertTrue(_case_.why, f'{_case_.name}: reach="absorbed" needs a '
                                                f'why= saying what absorbs the value')


class TestTileIsExcludedFromTheProfileButStillChecked(unittest.TestCase):
    '''tile() is outside Profile A, and the contract still catches what it embeds.

    Two different statements, and conflating them is the mistake this class exists
    to prevent:

      * tile() makes no *promise*.  It embeds foreign SVG verbatim -- that is the
        component (PLANNING.md A3) -- so "a Tile conforms" is not something the
        library can guarantee, and `_OUT_OF_PROFILE_` says so.
      * The contract is not *blind* to it.  checkOutputContract() reads the
        composed document, so hostile markup embedded through svg_list is
        reported like any other violation.  An appliance that gates its responses
        on assertOutputContract() is therefore covered even when it tiles markup
        it did not write.

    The second is the one an auditor probes ("does your check see through the
    component you excluded?"), and it was true but untested until this landed.
    The first is why the answer is still "do not put untrusted input in svg_list":
    being caught at the gate is not the same as being safe to compose.
    '''

    # tile() reads width/height off each child's root <svg> to place it, so every
    # child here carries them; a child without them raises before any of this.
    _HOSTILE_CHILDREN_ = {
        'script_element':  '<script>alert(1)</script>',
        'event_handler':   '<rect width="10" height="10" onclick="alert(1)"/>',
        'external_image':  '<image href="http://example.invalid/pixel.png" '
                           'width="10" height="10"/>',
        'foreign_object':  '<foreignObject width="10" height="10"><div '
                           'xmlns="http://www.w3.org/1999/xhtml">x</div></foreignObject>',
        'javascript_href': '<use href="javascript:alert(1)"/>',
        'dangerous_css':   '<style>@import url("http://example.invalid/x.css");</style>',
        # Distinct from dangerous_css: no @import, no javascript: -- just a
        # reference out of the document, which is the shape that used to ride
        # through tile() unreported.
        'external_css_url': '<style>rect{fill:url(http://example.invalid/x#g)}</style>'
                           '<rect width="10" height="10"/>',
        'anchor_element':  '<a href="http://example.invalid/"><rect width="10" '
                           'height="10"/></a>',
    }

    def setUp(self):
        self.p2s = Polars2SVG()

    def _wrap(self, body):
        return ('<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">'
                + body + '</svg>')

    def test_tile_of_conforming_children_conforms(self):
        # The positive control.  Without it every assertion below is satisfied by a
        # contract that simply rejects every Tile, which would prove nothing about
        # whether it can see the embedded markup.
        _df_   = pl.DataFrame({'cat': ['a', 'b', 'c', 'a'], 'v': [1, 2, 3, 4]})
        _tile_ = self.p2s.tile([self.p2s.histop(_df_, 'cat', wxh=(96, 96)),
                                self.p2s.piep(_df_, 'cat', wxh=(96, 96))])
        self.assertEqual(checkOutputContract(_tile_._repr_svg_()), [])

    def test_hostile_child_markup_is_reported(self):
        for _name_, _body_ in self._HOSTILE_CHILDREN_.items():
            with self.subTest(child=_name_):
                _svg_ = self.p2s.tile([self._wrap(_body_)])._repr_svg_()
                # It really did get embedded verbatim -- otherwise this would be
                # testing that tile() strips things, which it does not do.
                self.assertIn(_body_, _svg_,
                              'tile() no longer embeds verbatim; this test and '
                              'PLANNING.md A3 both need revisiting')
                self.assertNotEqual(
                    checkOutputContract(_svg_), [],
                    f'a Tile embedding {_name_} passed the contract: the check is '
                    f'blind to svg_list content, which is the one thing an '
                    f'appliance gating on assertOutputContract() relies on')

    def test_doctype_in_a_child_is_reported(self):
        # tile() concatenates, so a child's DOCTYPE lands mid-document.  The
        # contract refuses a DOCTYPE anywhere rather than parsing it.
        _child_ = ('<!DOCTYPE svg [<!ENTITY e SYSTEM "file:///etc/passwd">]>'
                   + self._wrap('<rect width="10" height="10"/>'))
        _kinds_ = {_v_.kind for _v_
                   in checkOutputContract(self.p2s.tile([_child_])._repr_svg_())}
        self.assertIn('doctype', _kinds_)

    def test_malformed_child_is_reported(self):
        _svg_ = self.p2s.tile([self._wrap('<rect width="10" height="10">')])._repr_svg_()
        self.assertIn('not-well-formed', {_v_.kind for _v_ in checkOutputContract(_svg_)})

    def test_tile_stays_named_out_of_profile(self):
        # The promise half.  If somebody moves Tile into _COMPONENT_PREFIXES_ on
        # the strength of the tests above, they have confused "caught at the gate"
        # with "safe to compose".
        self.assertIn('Tile', _OUT_OF_PROFILE_)
        self.assertNotIn('Tile', _COMPONENT_PREFIXES_)


class TestCorpusCoversEveryComponent(unittest.TestCase):
    '''A new component cannot ship without injection coverage.

    This is the enforcement half of the work.  The escaping in the tree is
    currently complete -- 540 hostile renders produce zero contract violations --
    but nothing structural kept it that way, and "we were careful" does not
    survive a ninth component.  Adding one fails here until it is either given
    cases or explicitly placed outside Profile A.
    '''

    def test_every_export_component_is_classified(self):
        _known_ = set(_COMPONENT_PREFIXES_) | _OUT_OF_PROFILE_
        _actual_ = {_c_.__name__ for _c_ in ExportMixin.__subclasses__()}
        self.assertEqual(
            _actual_ - _known_, set(),
            'component(s) with no Profile A classification: add injection cases to '
            'tests/injection_corpus.py and a _COMPONENT_PREFIXES_ entry, or name it '
            'in _OUT_OF_PROFILE_ with the reason')
        self.assertEqual(
            _known_ - _actual_, set(),
            'classification names a component that no longer exists')

    def test_every_in_profile_component_has_cases(self):
        _p2s_    = Polars2SVG()
        _prefixes_ = {_c_.name.split('.', 1)[0] for _c_ in renderMatrix(_p2s_, _PROBE_)}
        for _cls_, _prefix_ in sorted(_COMPONENT_PREFIXES_.items()):
            with self.subTest(component=_cls_):
                self.assertIn(_prefix_, _prefixes_,
                              f'{_cls_} has no injection cases in renderMatrix()')


class TestContractVocabularyIsNotAccidentallyWidened(unittest.TestCase):
    '''The allow-lists are a security control, so a careless edit should be loud.

    Not a ratchet on the count (that would fail on a legitimate addition without
    saying why) -- a check that the constructs deliberately excluded stay
    excluded.  Adding any of these is a decision, not a typo.
    '''

    FORBIDDEN_ELEMENTS = {'script', 'foreignObject', 'a', 'animate', 'animateTransform',
                          'set', 'handler', 'iframe', 'image', 'audio', 'video'}
    FORBIDDEN_ATTRS    = {'onload', 'onclick', 'onerror', 'onmouseover', 'style',
                          'xlink:href', 'formaction'}

    def test_behaviour_bearing_elements_stay_out(self):
        self.assertEqual(ALLOWED_ELEMENTS & self.FORBIDDEN_ELEMENTS, set())

    def test_behaviour_bearing_attributes_stay_out(self):
        self.assertEqual(ALLOWED_ATTRIBUTES & self.FORBIDDEN_ATTRS, set())


class TestProfileBIsDocumentedOutOfScope(unittest.TestCase):
    '''Profile B's status is a decision, and decisions drift unless pinned.

    The interactive surface is not hardened and is not covered by anything in this
    file.  That is a supported position only while SECURITY.md says so plainly, so
    this fails if the statement is removed or softened.
    '''

    def setUp(self):
        _root_ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _path_ = os.path.join(_root_, 'SECURITY.md')
        if not os.path.exists(_path_):
            self.skipTest('SECURITY.md not present (installed from wheel)')
        with open(_path_, encoding='utf-8') as _f_:
            self.text = _f_.read()

    def test_profiles_are_named(self):
        for _name_ in ('Profile A', 'Profile B', 'Notebook'):
            self.assertIn(_name_, self.text, f'SECURITY.md no longer names {_name_}')

    def test_profile_b_declared_out_of_scope(self):
        self.assertIn('out of scope', self.text.lower(),
                      'SECURITY.md must state plainly that Profile B is out of scope')

    def test_contract_module_is_referenced(self):
        # The profile is only meaningful if a reader can find the enforcement.
        self.assertIn('svg_contract', self.text)


if __name__ == '__main__':
    unittest.main()
