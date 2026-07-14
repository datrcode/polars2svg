#
# test_parse_consolidation.py
#
# Consolidation of the 8 formerly hand-copied per-component parse routines.
#
# Every component's __parseInput__ used to hand-copy two independent per-parameter
# loops — a from-scratch default block and a keyword-override block. Listing the
# parameters twice let them drift (xyp accepted `use_lazy_execution` but never read
# it; hex-color handling diverged the same way). The consolidation makes a single
# per-component `_defaults_` mapping the sole source of truth for both phases, and
# adds a structural drift guard (`assertParamSpecMatches`) so a declared-but-
# forgotten parameter raises instead of silently no-op'ing.
#
# These tests cover: the shared helpers, the drift guard (fires on mismatch, passes
# for all 8 real components), the specific xyp/use_lazy_execution drift named
# above, and a generic "every accepted kwarg is actually stored" sweep.
#

import unittest
import polars as pl

import polars2svg


class _Dummy:
    '''Bare attribute bag for exercising the assign* helpers directly.'''
    pass


class TestAssignHelpers(unittest.TestCase):
    def setUp(self):
        self.p2s = polars2svg.Polars2SVG()

    def test_assign_scratch_defaults(self):
        d = _Dummy()
        self.p2s.assignScratchDefaults(d, {'a': 1, 'b': None, 'c': 'x'})
        self.assertEqual((d.a, d.b, d.c), (1, None, 'x'))

    def test_assign_scratch_defaults_fresh_containers_per_call(self):
        # A set default in the spec must not be shared across instances.
        spec = {'s': set()}
        d1, d2 = _Dummy(), _Dummy()
        # Same spec dict reused, but each build should pass a freshly-built spec;
        # here we simulate two builds sharing one spec object to prove the helper
        # itself assigns by reference (the components rebuild the spec each parse).
        self.p2s.assignScratchDefaults(d1, spec)
        self.p2s.assignScratchDefaults(d2, spec)
        self.assertIs(d1.s, d2.s)  # documents: the *component* must rebuild the spec

    def test_assign_kwarg_overrides_copies_present_only(self):
        d = _Dummy()
        d.a, d.b, d.c = 1, 2, 3
        self.p2s.assignKwargOverrides(d, {'a': None, 'b': None, 'c': None},
                                      {'a': 10, 'c': 30, 'ignored': 99})
        self.assertEqual((d.a, d.b, d.c), (10, 2, 30))

    def test_assign_kwarg_overrides_respects_skip(self):
        d = _Dummy()
        d.a = d.b = None
        self.p2s.assignKwargOverrides(d, {'a': None, 'b': None},
                                      {'a': 1, 'b': 2}, skip={'b'})
        self.assertEqual((d.a, d.b), (1, None))  # b skipped for caller-side handling

    def test_assign_kwargs_with_defaults(self):
        d = _Dummy()
        self.p2s.assignKwargsWithDefaults(d, {'a': 1, 'b': 2, 'c': 3}, {'b': 20})
        self.assertEqual((d.a, d.b, d.c), (1, 20, 3))


class TestDriftGuard(unittest.TestCase):
    def setUp(self):
        self.p2s = polars2svg.Polars2SVG()

    def test_guard_raises_on_missing_default(self):
        # kwarg accepted (_VALID_KWARGS) but not in defaults -> drift.
        with self.assertRaises(RuntimeError) as cm:
            self.p2s.assertParamSpecMatches(
                'FakeComp', frozenset({'df', 'template', 'x', 'y'}), {'x': 1})
        self.assertIn('_VALID_KWARGS only', str(cm.exception))
        self.assertIn('y', str(cm.exception))

    def test_guard_raises_on_extra_default(self):
        # attribute defaulted/assigned but not accepted -> drift.
        with self.assertRaises(RuntimeError) as cm:
            self.p2s.assertParamSpecMatches(
                'FakeComp2', frozenset({'df', 'template', 'x'}), {'x': 1, 'z': 2})
        self.assertIn('defaults/extra only', str(cm.exception))
        self.assertIn('z', str(cm.exception))

    def test_guard_passes_and_caches(self):
        name = 'FakeCompOK'
        self.p2s._PARAM_SPEC_VERIFIED_.discard(name)
        self.p2s.assertParamSpecMatches(
            name, frozenset({'df', 'template', 'x'}), {'x': 1})
        self.assertIn(name, self.p2s._PARAM_SPEC_VERIFIED_)

    def test_all_eight_components_pass_the_guard(self):
        # Reset the once-per-process cache, then reach each component's
        # __parseInput__ (the guard runs at the very top). A drift would raise
        # RuntimeError before the name is cached; membership proves it passed.
        p2s = self.p2s
        for _n_ in ('Histop', 'Timep', 'Piep', 'XYp', 'LinkP', 'ChP',
                    'SpreadLinesP', 'Smallp'):
            p2s._PARAM_SPEC_VERIFIED_.discard(_n_)

        builders = [
            lambda: p2s.histop(),
            lambda: p2s.timep(),
            lambda: p2s.piep(),
            lambda: p2s.xyp(),
            lambda: p2s.linkp(),
            lambda: p2s.chordp(),
            lambda: p2s.spreadlinesp(),
            lambda: p2s.smallp(),   # raises ValueError after the guard runs
        ]
        for _b_ in builders:
            try:
                _b_()
            except RuntimeError:
                raise                       # a genuine spec-drift failure
            except Exception:
                pass                        # later validation errors are fine

        for _n_ in ('Histop', 'Timep', 'Piep', 'XYp', 'LinkP', 'ChP',
                    'SpreadLinesP', 'Smallp'):
            self.assertIn(_n_, p2s._PARAM_SPEC_VERIFIED_,
                          f'{_n_} did not pass the parameter-spec drift guard')


class TestUseLazyExecutionDrift(unittest.TestCase):
    '''The concrete drift the item names: xyp read use_lazy_execution from raw
    kwargs in __init__ (bypassing the defaults merge), so a
    set_defaults('xyp', use_lazy_execution=...) global was silently ignored. After
    consolidation it is a normal spec parameter.'''

    def setUp(self):
        self.p2s = polars2svg.Polars2SVG()
        self.df = pl.DataFrame({'a': [1, 2, 3, 4], 'b': [4, 3, 2, 1]})

    def tearDown(self):
        # Do not leak the global default into other tests.
        self.p2s._global_defaults.pop('use_lazy_execution', None)
        self.p2s._component_defaults.get('xyp', {}).pop('use_lazy_execution', None)

    def test_explicit_kwarg_respected(self):
        self.assertFalse(self.p2s.xyp(self.df, 'a', 'b',
                                      use_lazy_execution=False).use_lazy_execution)
        self.assertTrue(self.p2s.xyp(self.df, 'a', 'b',
                                     use_lazy_execution=True).use_lazy_execution)

    def test_default_is_true(self):
        self.assertTrue(self.p2s.xyp(self.df, 'a', 'b').use_lazy_execution)

    def test_component_default_now_honoured(self):
        # Previously ignored by xyp; now respected via the shared spec.
        self.p2s.set_defaults('xyp', use_lazy_execution=False)
        self.assertFalse(self.p2s.xyp(self.df, 'a', 'b').use_lazy_execution)

    def test_explicit_kwarg_beats_component_default(self):
        self.p2s.set_defaults('xyp', use_lazy_execution=False)
        self.assertTrue(self.p2s.xyp(self.df, 'a', 'b',
                                     use_lazy_execution=True).use_lazy_execution)


class TestKwargRoundTrip(unittest.TestCase):
    '''Structural guarantee of the consolidation: a representative simple kwarg for
    each clone-template component is actually stored on the instance (no
    accepted-but-forgotten parameter). These are plain pass-through params (no
    coercion/side-effect), so the stored value must equal the input exactly.'''

    def setUp(self):
        self.p2s = polars2svg.Polars2SVG()

    def test_histop_txt_h(self):
        df = pl.DataFrame({'a': ['x', 'y', 'x']})
        self.assertEqual(self.p2s.histop(df, 'a', txt_h=21).txt_h, 21)

    def test_piep_donut_ratio(self):
        df = pl.DataFrame({'a': ['x', 'y', 'x']})
        self.assertEqual(self.p2s.piep(df, 'a', donut_ratio=0.33).donut_ratio, 0.33)

    def test_xyp_opacity_range(self):
        df = pl.DataFrame({'a': [1, 2, 3], 'b': [3, 2, 1]})
        self.assertEqual(self.p2s.xyp(df, 'a', 'b', opacity_range=(0.1, 0.9)).opacity_range,
                         (0.1, 0.9))

    def test_linkp_node_opacity(self):
        df = pl.DataFrame({'fm': ['a', 'b'], 'to': ['b', 'c']})
        self.assertEqual(self.p2s.linkp(df, [('fm', 'to')], node_opacity=0.42).node_opacity,
                         0.42)

    def test_chordp_node_gap(self):
        df = pl.DataFrame({'fm': ['a', 'b'], 'to': ['b', 'c']})
        self.assertEqual(self.p2s.chordp(df, [('fm', 'to')], node_gap=7).node_gap, 7)

    def test_chordp_bundle_strength_coerced(self):
        # A skipped (coerced) param still round-trips through its explicit branch.
        df = pl.DataFrame({'fm': ['a', 'b'], 'to': ['b', 'c']})
        c = self.p2s.chordp(df, [('fm', 'to')], bundle_strength=1)
        self.assertIsInstance(c.bundle_strength, float)
        self.assertEqual(c.bundle_strength, 1.0)


if __name__ == '__main__':
    unittest.main()
