import unittest
import polars as pl
from polars2svg import Polars2SVG


class TestGlobalConfig(unittest.TestCase):

    def setUp(self):
        self.p2s = Polars2SVG()
        self.p2s.reset_defaults()

    def tearDown(self):
        self.p2s.reset_defaults()

    # ------------------------------------------------------------------
    # set_defaults / get_defaults
    # ------------------------------------------------------------------

    def test_get_defaults_empty(self):
        d = self.p2s.get_defaults()
        self.assertEqual(d, {'_global': {}})

    def test_set_global_defaults_visible_in_get(self):
        self.p2s.set_defaults(txt_h=16, insets=(4, 4))
        d = self.p2s.get_defaults()
        self.assertEqual(d['_global']['txt_h'],   16)
        self.assertEqual(d['_global']['insets'], (4, 4))

    def test_set_component_defaults_visible_in_get(self):
        self.p2s.set_defaults('histop', txt_h=14, wxh=(200, 400))
        d = self.p2s.get_defaults()
        self.assertEqual(d['histop']['txt_h'],   14)
        self.assertEqual(d['histop']['wxh'], (200, 400))

    def test_set_defaults_invalid_component(self):
        with self.assertRaises(ValueError):
            self.p2s.set_defaults('bogus', txt_h=14)

    # ------------------------------------------------------------------
    # reset_defaults
    # ------------------------------------------------------------------

    def test_reset_all_clears_global_and_component(self):
        self.p2s.set_defaults(txt_h=16)
        self.p2s.set_defaults('histop', txt_h=14)
        self.p2s.reset_defaults()
        d = self.p2s.get_defaults()
        self.assertEqual(d, {'_global': {}})

    def test_reset_component_leaves_global(self):
        self.p2s.set_defaults(txt_h=16)
        self.p2s.set_defaults('histop', txt_h=14)
        self.p2s.reset_defaults('histop')
        d = self.p2s.get_defaults()
        self.assertNotIn('histop', d)
        self.assertEqual(d['_global']['txt_h'], 16)

    def test_reset_defaults_invalid_component(self):
        with self.assertRaises(ValueError):
            self.p2s.reset_defaults('bogus')

    def test_reset_nonexistent_component_is_safe(self):
        self.p2s.reset_defaults('timep')  # no timep defaults set — should not raise

    # ------------------------------------------------------------------
    # _apply_defaults / priority
    # ------------------------------------------------------------------

    def test_apply_defaults_global_wins_over_hardcoded(self):
        self.p2s.set_defaults(txt_h=16)
        h = self.p2s.histop()
        self.assertEqual(h.txt_h, 16)

    def test_apply_defaults_component_wins_over_global(self):
        self.p2s.set_defaults(txt_h=16)
        self.p2s.set_defaults('histop', txt_h=14)
        h = self.p2s.histop()
        self.assertEqual(h.txt_h, 14)

    def test_apply_defaults_explicit_kwarg_wins_over_component(self):
        self.p2s.set_defaults(txt_h=16)
        self.p2s.set_defaults('histop', txt_h=14)
        h = self.p2s.histop(txt_h=10)
        self.assertEqual(h.txt_h, 10)

    def test_apply_defaults_explicit_kwarg_wins_over_global(self):
        self.p2s.set_defaults(txt_h=16)
        h = self.p2s.histop(txt_h=10)
        self.assertEqual(h.txt_h, 10)

    def test_hardcoded_default_used_when_no_config(self):
        h = self.p2s.histop()
        self.assertEqual(h.txt_h, 12)

    # ------------------------------------------------------------------
    # Scoping: component defaults don't bleed to other components
    # ------------------------------------------------------------------

    def test_histop_default_does_not_affect_timep(self):
        self.p2s.set_defaults('histop', txt_h=20)
        t = self.p2s.timep()
        self.assertEqual(t.txt_h, 12)

    def test_global_default_applies_to_multiple_components(self):
        self.p2s.set_defaults(txt_h=18)
        h = self.p2s.histop()
        t = self.p2s.timep()
        self.assertEqual(h.txt_h, 18)
        self.assertEqual(t.txt_h, 18)

    def test_component_default_for_wxh(self):
        self.p2s.set_defaults('histop', wxh=(200, 400))
        h = self.p2s.histop()
        self.assertEqual(h.wxh, (200, 400))

    def test_component_default_for_wxh_does_not_affect_timep(self):
        self.p2s.set_defaults('histop', wxh=(200, 400))
        t = self.p2s.timep()
        self.assertEqual(t.wxh, (512, 256))  # timep hardcoded default

    # ------------------------------------------------------------------
    # Template unaffected by config defaults
    # ------------------------------------------------------------------

    def test_template_not_overridden_by_global_default(self):
        '''A template was created before set_defaults; config should not change its values.'''
        tmpl = self.p2s.histop()           # txt_h = 12 (hardcoded default)
        self.p2s.set_defaults(txt_h=16)
        h = self.p2s.histop(tmpl)         # uses template — should keep 12
        self.assertEqual(h.txt_h, 12)

    def test_template_explicit_kwarg_still_wins(self):
        '''Explicit kwarg overrides template even when config default is also set.'''
        tmpl = self.p2s.histop()
        self.p2s.set_defaults(txt_h=16)
        h = self.p2s.histop(tmpl, txt_h=20)
        self.assertEqual(h.txt_h, 20)

    # ------------------------------------------------------------------
    # Singleton guard: re-creating Polars2SVG() doesn't reset config
    # ------------------------------------------------------------------

    def test_singleton_reinit_does_not_reset_config(self):
        self.p2s.set_defaults(txt_h=16)
        p2s2 = Polars2SVG()               # re-invokes __init__ on same singleton
        self.assertEqual(p2s2._global_defaults.get('txt_h'), 16)

    # ------------------------------------------------------------------
    # All valid component names are accepted
    # ------------------------------------------------------------------

    def test_all_valid_components_accepted(self):
        for name in ('histop', 'timep', 'xyp', 'linkp', 'smallp', 'chordp'):
            self.p2s.set_defaults(name, txt_h=14)
        d = self.p2s.get_defaults()
        for name in ('histop', 'timep', 'xyp', 'linkp', 'smallp', 'chordp'):
            self.assertIn(name, d)

    # ------------------------------------------------------------------
    # Eager kwarg validation in set_defaults
    # ------------------------------------------------------------------

    def test_component_default_typo_raises(self):
        '''The motivating bug: a typo'd kwarg was stored, merged, and silently never read.'''
        with self.assertRaises(TypeError) as ctx:
            self.p2s.set_defaults('histop', barh=10)   # typo of bar_h
        self.assertIn('barh', str(ctx.exception))

    def test_component_default_valid_elsewhere_raises(self):
        '''bin_by is a histop/piep kwarg — setting it as an xyp default must error.'''
        with self.assertRaises(TypeError):
            self.p2s.set_defaults('xyp', bin_by='cat')

    def test_component_default_typo_stores_nothing(self):
        '''A rejected set_defaults call must not partially store the valid kwargs.'''
        with self.assertRaises(TypeError):
            self.p2s.set_defaults('histop', txt_h=14, barh=10)
        self.assertNotIn('histop', self.p2s.get_defaults())

    def test_global_default_unknown_everywhere_raises(self):
        with self.assertRaises(TypeError) as ctx:
            self.p2s.set_defaults(barh=10)
        self.assertIn('barh', str(ctx.exception))

    def test_global_default_valid_for_some_components_accepted(self):
        '''A global default only some components accept is legal; it is filtered per component.'''
        self.p2s.set_defaults(bin_by='cat')            # histop/piep only
        self.assertIn(   'bin_by', self.p2s._apply_defaults('histop', {}))
        self.assertNotIn('bin_by', self.p2s._apply_defaults('xyp',    {}))

    def test_global_partial_default_does_not_break_other_components(self):
        '''A histop-only global default must not disturb an xyp render.'''
        self.p2s.set_defaults(bin_by='cat')
        df = pl.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6], 'cat': ['a', 'b', 'a']})
        x  = self.p2s.xyp(df, x='x', y='y')
        self.assertIsNotNone(x.svg)
        h  = self.p2s.histop(df)                       # picks up the global bin_by
        self.assertEqual(h.bin_by, 'cat')

    def test_each_component_accepts_its_own_kwarg(self):
        for name, kw in (('histop',       {'bin_by':          'cat'}),
                         ('timep',        {'time':            'ts'}),
                         ('xyp',          {'dot_size':        'small'}),
                         ('linkp',        {'node_size':       'vary'}),
                         ('smallp',       {'grid_mode':       True}),
                         ('chordp',       {'bundle_strength': 0.5}),
                         ('spreadlinesp', {'max_rings':       2}),
                         ('piep',         {'donut_ratio':     0.4})):
            self.p2s.set_defaults(name, **kw)
        d = self.p2s.get_defaults()
        self.assertEqual(d['piep']['donut_ratio'], 0.4)

    def test_component_default_applies_to_render(self):
        '''A validated component default actually flows into the constructed component.'''
        self.p2s.set_defaults('histop', bar_h=10)
        h = self.p2s.histop()
        self.assertEqual(h.bar_h, 10)


if __name__ == '__main__':
    unittest.main()
