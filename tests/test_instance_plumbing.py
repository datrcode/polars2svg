#
# test_instance_plumbing.py - the configuration channel between a Polars2SVG instance
# and the components it builds.
#
# Components used to reach the framework by calling Polars2SVG() themselves.  That
# worked only because the call returned the process-wide singleton: the instance a
# component resolved its defaults and color overrides against was "whichever one this
# process has", not "the one whose factory method built me".  Every factory now hands
# the component its own instance as p2s=, and the component-level clone helpers
# (render_with / renderSmallMultiples) pass self.p2s, so the channel is an explicit
# parameter rather than a lookup that happens to land on the right object.
#
# Polars2SVG() returns an independent instance, so these tests can simply build two and
# watch which one a render answers to.  (They were written while the singleton still
# stood, when telling "passed down correctly" apart from "happened to be the same object
# anyway" needed an instance that bypassed the __new__ cache.)
#
import ast
import inspect
import logging
import pathlib
import unittest

import polars as pl

import polars2svg
from polars2svg import Polars2SVG
from polars2svg.xyp import XYp


def _fresh_p2s_():
    '''An independent Polars2SVG -- named for what the tests below rely on.'''
    return Polars2SVG()


class _CountingHandler_(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []
    def emit(self, record):
        self.records.append(record.getMessage())


# kwargs each component needs for a blank build (mirrors tests/test_template_defaults.py)
_CREATE_KWARGS = {'histop': {}, 'timep': {}, 'xyp': {'x': 'x', 'y': 'y'}, 'linkp': {},
                  'chordp': {}, 'spreadlinesp': {}, 'piep': {}}


class TestFactoriesPassTheirOwnInstance(unittest.TestCase):

    def setUp(self):
        self.df = pl.DataFrame({'x': [1, 2, 3], 'y': [3, 1, 4], 'g': ['a', 'b', 'a']})

    def test_every_factory_hands_the_component_its_caller(self):
        _p2s_ = _fresh_p2s_()
        for _name_, _kwargs_ in _CREATE_KWARGS.items():
            if _name_ not in _p2s_._VALID_COMPONENTS_: continue   # chordp without scipy
            with self.subTest(component=_name_):
                self.assertIs(getattr(_p2s_, _name_)(**_kwargs_).p2s, _p2s_)

    def test_smallp_and_tile_hand_the_component_their_caller(self):
        _p2s_  = _fresh_p2s_()
        _tmpl_ = _p2s_.xyp(self.df, 'x', 'y', wxh=(100, 100))
        self.assertIs(_p2s_.smallp(self.df, _tmpl_, category_by='g').p2s, _p2s_)
        self.assertIs(_p2s_.tile([_tmpl_]).p2s, _p2s_)

    def test_direct_construction_builds_its_own_instance(self):
        # XYp(df, ...) is the documented alternative to p2s.xyp(df, ...).  With no
        # instance to inherit it gets a fresh, unconfigured one -- so a component built
        # this way does NOT see the configuration of an instance it was never given.
        _p2s_ = _fresh_p2s_()
        _p2s_.set_defaults('xyp', txt_h=17)
        _direct_ = XYp(self.df, x='x', y='y')
        self.assertIsNot(_direct_.p2s, _p2s_)
        self.assertNotEqual(_direct_.txt_h, 17)
        # ...and passing the instance explicitly is how you share it.
        self.assertEqual(XYp(self.df, x='x', y='y', p2s=_p2s_).txt_h, 17)


class TestNothingConstructsAnInstanceBehindYourBack(unittest.TestCase):
    '''Structural guard.  A Polars2SVG() built inside a render path is unconfigured --
    it has nobody's defaults and nobody's color overrides -- and nothing about the
    resulting figure says so.  That was the whole bug class the p2s= parameter removed,
    and while the singleton stood it was invisible, so it needs pinning rather than
    remembering.  The only legitimate constructions are the nine component fallbacks
    for direct construction (XYp(df, ...) with no p2s=).'''

    _FALLBACK_MODULES_ = {'xyp', 'histop', 'piep', 'timep', 'linkp', 'chordp',
                          'smallp', 'spreadlinesp', 'tile'}

    @staticmethod
    def _constructions_(path):
        '''Count Polars2SVG(...) / polars2svg.Polars2SVG(...) calls in one module.'''
        _n_ = 0
        for _node_ in ast.walk(ast.parse(path.read_text())):
            if not isinstance(_node_, ast.Call): continue
            _f_ = _node_.func
            if   isinstance(_f_, ast.Name)      and _f_.id   == 'Polars2SVG': _n_ += 1
            elif isinstance(_f_, ast.Attribute) and _f_.attr == 'Polars2SVG': _n_ += 1
        return _n_

    def test_only_the_component_fallbacks_construct_an_instance(self):
        _pkg_ = pathlib.Path(inspect.getfile(polars2svg)).parent
        _found_ = {f.stem: self._constructions_(f) for f in sorted(_pkg_.glob('*.py'))
                   if self._constructions_(f)}
        self.assertEqual(_found_, {m: 1 for m in sorted(self._FALLBACK_MODULES_)},
                         'a module constructs Polars2SVG() outside the documented '
                         'direct-construction fallback; it should take p2s= from its '
                         f'caller instead: {_found_}')


class TestConfigurationFollowsTheInstance(unittest.TestCase):

    def setUp(self):
        self.df = pl.DataFrame({'x': [1, 2, 3], 'y': [3, 1, 4], 'g': ['a', 'b', 'a']})

    def test_component_default_reaches_a_component_built_by_that_instance(self):
        _p2s_ = _fresh_p2s_()
        _p2s_.set_defaults('xyp', txt_h=17)
        self.assertEqual(_p2s_.xyp(self.df, 'x', 'y').txt_h, 17)

    def test_global_default_reaches_a_component_built_by_that_instance(self):
        _p2s_ = _fresh_p2s_()
        _p2s_.set_defaults(txt_h=18)
        self.assertEqual(_p2s_.histop(self.df, bin_by='g').txt_h, 18)

    def test_defaults_do_not_reach_another_instance(self):
        _a_, _b_ = _fresh_p2s_(), _fresh_p2s_()
        _a_.set_defaults('xyp', txt_h=17)
        self.assertEqual(_a_.xyp(self.df, 'x', 'y').txt_h, 17)
        self.assertNotEqual(_b_.xyp(self.df, 'x', 'y').txt_h, 17)

    def test_color_override_reaches_its_own_instances_render(self):
        _p2s_ = _fresh_p2s_()
        _p2s_.setColorOverrides({'a': '#abcdef'})
        self.assertIn('#abcdef', _p2s_.xyp(self.df, 'x', 'y', color='g', wxh=(256, 256)).svg)

    def test_color_override_does_not_reach_another_instances_render(self):
        _a_, _b_ = _fresh_p2s_(), _fresh_p2s_()
        _a_.setColorOverrides({'a': '#abcdef'})
        self.assertNotIn('#abcdef', _b_.xyp(self.df, 'x', 'y', color='g', wxh=(256, 256)).svg)


class TestCloneHelpersPassTheInstanceDown(unittest.TestCase):
    '''render_with() / renderSmallMultiples() construct their own class directly --
    there is no factory frame to inherit from, so they pass self.p2s explicitly.'''

    def setUp(self):
        self.df = pl.DataFrame({'x': [1, 2, 3], 'y': [3, 1, 4], 'g': ['a', 'b', 'a']})

    def test_render_with_keeps_the_components_instance(self):
        _p2s_ = _fresh_p2s_()
        # xyp routes its clone back through the factory, histop constructs Histop()
        # directly -- both have to end up on the same instance.
        self.assertIs(_p2s_.xyp(self.df, 'x', 'y').render_with(self.df).p2s, _p2s_)
        self.assertIs(_p2s_.histop(self.df, bin_by='g').render_with(self.df).p2s, _p2s_)

    def test_template_clone_keeps_the_callers_instance(self):
        # _clone_template_state copies the template's __dict__ onto the clone; p2s is
        # skipped so a clone built by B from A's template renders against B.
        _a_, _b_ = _fresh_p2s_(), _fresh_p2s_()
        _tmpl_   = _a_.xyp(self.df, 'x', 'y')
        self.assertIs(_b_.xyp(df=self.df, template=_tmpl_).p2s, _b_)

    def test_template_clone_resolves_colors_against_the_caller(self):
        _a_, _b_ = _fresh_p2s_(), _fresh_p2s_()
        _b_.setColorOverrides({'a': '#abcdef'})
        _tmpl_ = _a_.xyp(self.df, 'x', 'y', color='g', wxh=(256, 256))
        self.assertIn('#abcdef', _b_.xyp(df=self.df, template=_tmpl_).svg)


class TestWarnOnceIsPerProcessNotPerInstance(unittest.TestCase):
    '''The OnceFilter lives on the shared 'polars2svg_logger' but is reinstalled by
    every __init__; it carries the seen set across so a second instance does not
    re-open messages the first one already emitted.'''

    def test_a_second_instance_does_not_reset_the_dedupe(self):
        _p2s_    = _fresh_p2s_()
        _logger_ = logging.getLogger('polars2svg_logger')
        _handler_ = _CountingHandler_()
        _logger_.addHandler(_handler_)
        try:
            _msg_ = 'test_instance_plumbing: repeated warning __unique_c__'
            _p2s_.logger.warning(_msg_)
            _fresh_p2s_()                       # a whole new instance, not a re-init
            _p2s_.logger.warning(_msg_)
            self.assertEqual(_handler_.records.count(_msg_), 1)
        finally:
            _logger_.removeHandler(_handler_)


if __name__ == '__main__':
    unittest.main()
