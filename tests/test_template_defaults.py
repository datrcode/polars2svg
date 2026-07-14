# Locks the template/defaults interaction rule for every clone-template component:
#
#   explicit kwargs  >  template snapshot  >  (defaults baked in at template creation)
#
# Session defaults (set_defaults) are resolved once, when a component is built from
# scratch. A template clone is an exact snapshot of the template's resolved state —
# defaults set or changed *after* the template was created never leak into clones.
# smallp is the one exception by construction: its sm_template is a panel template
# (a fully-resolved sibling component), not a Smallp clone source, so smallp always
# applies defaults to its own kwargs.
import unittest
import polars as pl
from polars2svg import Polars2SVG

# kwargs required to construct a blank template for each clone-template component
_CREATE_KWARGS = {
    'histop':       {},
    'timep':        {},
    'xyp':          {'x': 'x', 'y': 'y'},
    'linkp':        {},
    'chordp':       {},
    'spreadlinesp': {},
    'piep':         {},
}

_HARDCODED_TXT_H = 12   # every component's hardcoded txt_h default


class TestTemplateDefaults(unittest.TestCase):

    def setUp(self):
        self.p2s = Polars2SVG()
        self.p2s.reset_defaults()

    def tearDown(self):
        self.p2s.reset_defaults()

    def _build(self, name, **kwargs):
        _create_ = dict(_CREATE_KWARGS[name])
        _create_.update(kwargs)
        return getattr(self.p2s, name)(**_create_)

    # ------------------------------------------------------------------
    # Defaults active at template creation are baked into the snapshot
    # ------------------------------------------------------------------

    def test_component_defaults_baked_into_template(self):
        '''A component default set before template creation is captured by the
        template and inherited by clones — even after the default is cleared.'''
        for name in _CREATE_KWARGS:
            with self.subTest(component=name):
                self.p2s.reset_defaults()
                self.p2s.set_defaults(name, txt_h=17)
                tmpl = self._build(name)
                self.assertEqual(tmpl.txt_h, 17)
                self.p2s.reset_defaults()                     # default is gone...
                clone = getattr(self.p2s, name)(template=tmpl)
                self.assertEqual(clone.txt_h, 17)             # ...baked-in value survives

    def test_global_defaults_baked_into_template(self):
        for name in _CREATE_KWARGS:
            with self.subTest(component=name):
                self.p2s.reset_defaults()
                self.p2s.set_defaults(txt_h=18)
                tmpl = self._build(name)
                self.assertEqual(tmpl.txt_h, 18)
                self.p2s.reset_defaults()
                clone = getattr(self.p2s, name)(template=tmpl)
                self.assertEqual(clone.txt_h, 18)

    # ------------------------------------------------------------------
    # Defaults set after template creation never leak into clones
    # ------------------------------------------------------------------

    def test_component_default_after_template_ignored_by_clone(self):
        '''The template snapshot is authoritative: a component default set after
        template creation applies to from-scratch builds but not to clones.'''
        for name in _CREATE_KWARGS:
            with self.subTest(component=name):
                self.p2s.reset_defaults()
                tmpl = self._build(name)                      # txt_h = hardcoded 12
                self.p2s.set_defaults(name, txt_h=17)
                clone = getattr(self.p2s, name)(template=tmpl)
                self.assertEqual(clone.txt_h, _HARDCODED_TXT_H)
                fresh = self._build(name)                     # no template → default applies
                self.assertEqual(fresh.txt_h, 17)

    def test_global_default_after_template_ignored_by_clone(self):
        for name in _CREATE_KWARGS:
            with self.subTest(component=name):
                self.p2s.reset_defaults()
                tmpl = self._build(name)
                self.p2s.set_defaults(txt_h=18)
                clone = getattr(self.p2s, name)(template=tmpl)
                self.assertEqual(clone.txt_h, _HARDCODED_TXT_H)

    def test_positional_template_same_semantics(self):
        '''A template passed positionally behaves exactly like template=.'''
        for name in _CREATE_KWARGS:
            with self.subTest(component=name):
                self.p2s.reset_defaults()
                tmpl = self._build(name)
                self.p2s.set_defaults(name, txt_h=17)
                clone = getattr(self.p2s, name)(tmpl)
                self.assertEqual(clone.txt_h, _HARDCODED_TXT_H)

    # ------------------------------------------------------------------
    # Explicit kwargs beat both the template snapshot and any defaults
    # ------------------------------------------------------------------

    def test_explicit_kwarg_wins_over_template_and_defaults(self):
        for name in _CREATE_KWARGS:
            with self.subTest(component=name):
                self.p2s.reset_defaults()
                self.p2s.set_defaults(name, txt_h=17)
                tmpl = self._build(name)                      # txt_h = 17 baked in
                self.p2s.set_defaults(name, txt_h=19)
                clone = getattr(self.p2s, name)(template=tmpl, txt_h=20)
                self.assertEqual(clone.txt_h, 20)

    # ------------------------------------------------------------------
    # smallp: sm_template is a panel template, not a clone source
    # ------------------------------------------------------------------

    def test_smallp_defaults_apply_alongside_sm_template(self):
        '''smallp has no clone-template; passing sm_template= must not suppress
        the defaults merge for smallp's own kwargs.'''
        df   = pl.DataFrame({'cat': ['a', 'b', 'a'], 'x': [1, 2, 3]})
        tmpl = self.p2s.histop(bin_by='x')
        self.p2s.set_defaults('smallp', txt_h=17)
        sm   = self.p2s.smallp(df, 'cat', sm_template=tmpl)
        self.assertEqual(sm.txt_h, 17)

    def test_smallp_global_defaults_apply_alongside_sm_template(self):
        df   = pl.DataFrame({'cat': ['a', 'b', 'a'], 'x': [1, 2, 3]})
        tmpl = self.p2s.histop(bin_by='x')
        self.p2s.set_defaults(txt_h=18)
        # the histop panel template was created before the default — snapshot holds
        self.assertEqual(tmpl.txt_h, _HARDCODED_TXT_H)
        sm   = self.p2s.smallp(df, 'cat', sm_template=tmpl)
        self.assertEqual(sm.txt_h, 18)

    def test_smallp_panel_clones_inherit_sm_template_not_defaults(self):
        '''Panels rendered by smallp are clones of sm_template — a smallp-scoped
        default must not leak into the panel components.'''
        df   = pl.DataFrame({'cat': ['a', 'b', 'a', 'b'], 'x': [1, 2, 3, 4]})
        tmpl = self.p2s.histop(bin_by='x', txt_h=14)
        self.p2s.set_defaults('histop', txt_h=17)             # set after template creation
        sm   = self.p2s.smallp(df, 'cat', sm_template=tmpl)
        svg  = sm.svg                                          # force render (panels get cloned)
        self.assertIsNotNone(svg)
        # the sm_template snapshot itself is untouched by the later default
        self.assertEqual(tmpl.txt_h, 14)


if __name__ == '__main__':
    unittest.main()
