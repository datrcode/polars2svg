"""Template state isolation across all clone-template components.

template= used to copy state via self.__dict__.update(template.__dict__), which
shared every mutable attribute between template and clone:

  - the clone's fresh timing_metrics = {} (set in __init__ before __parseInput__)
    was replaced by the template's dict mid-parse, so every clone render
    accumulated its timing into the *template's* metrics;
  - t_start was clobbered with the template's, so clone.t_overall spanned the
    template's lifetime too;
  - any in-place mutation of a shared container (sm_shared, background dicts,
    pos, anno) corrupted the template and every sibling clone.

Polars2SVG._clone_template_state now copies mutable containers (dict/set/list,
recursively, leaf objects shared) and skips per-instance lifecycle attributes so
the clone keeps its own fresh timing_metrics / t_start.  These tests lock that
contract for all 7 clone-template components, plus the copier's own semantics.
"""
import unittest
import datetime
import re
import polars as pl

from polars2svg import Polars2SVG
from polars2svg.polars2svg import _copy_mutable_containers_
from svg_test_utils import normalize_svg


def _histop_df():
    # distinct per-category counts: equal-count bins tie-break nondeterministically,
    # which would break the render-equivalence test below
    return pl.DataFrame({'cat': ['a', 'b', 'a', 'c', 'b', 'a'], 'val': [1, 2, 3, 4, 5, 6]})


def _timep_df():
    base = datetime.date(2024, 1, 1)
    return pl.DataFrame({
        'ts':  [base + datetime.timedelta(days=i * 30) for i in range(6)],
        'val': list(range(6)),
    })


def _xyp_df():
    return pl.DataFrame({'x': [1.0, 2.0, 3.0], 'y': [1.0, 2.0, 3.0]})


def _edge_df():
    return pl.DataFrame({'fm': ['a', 'b', 'c'], 'to': ['b', 'c', 'a']})


def _spread_df():
    return pl.DataFrame({
        'fm':   ['a', 'a', 'b'],
        'to':   ['b', 'c', 'c'],
        'time': [datetime.datetime(2024, 1, d) for d in (1, 2, 3)],
    })


class TestTemplateStateIsolation(unittest.TestCase):
    """Every clone-template component isolates its state from the template."""

    @classmethod
    def setUpClass(cls):
        cls.p2s = Polars2SVG()

    def _template_factories(self):
        """name -> callable building a fully-rendered template (df supplied)."""
        p2s = self.p2s
        return {
            'histop':       lambda: p2s.histop(_histop_df(), bin_by='cat'),
            'timep':        lambda: p2s.timep(_timep_df(), 'ts'),
            'xyp':          lambda: p2s.xyp(_xyp_df(), 'x', 'y'),
            'piep':         lambda: p2s.piep(_histop_df(), 'cat'),
            'chordp':       lambda: p2s.chordp(df=_edge_df(), relationships=[('fm', 'to')]),
            'linkp':        lambda: p2s.linkp(_edge_df(), relationships=[('fm', 'to')],
                                              pos={'a': [0, 0], 'b': [1, 0], 'c': [0.5, 0.866]}),
            'spreadlinesp': lambda: p2s.spreadlinesp(_spread_df(), [('fm', 'to')],
                                                     ego='a', time='time'),
        }

    def _clone(self, name, template, **kwargs):
        return getattr(self.p2s, name)(template=template, **kwargs)

    def test_clone_has_its_own_timing_metrics(self):
        """Cloning must not share the timing_metrics dict nor write timing into
        the template (the original __dict__.update bug)."""
        for name, make in self._template_factories().items():
            with self.subTest(component=name):
                tmpl     = make()
                snapshot = dict(tmpl.timing_metrics)
                clone    = self._clone(name, tmpl)
                self.assertIsNot(clone.timing_metrics, tmpl.timing_metrics)
                self.assertEqual(tmpl.timing_metrics, snapshot)   # template untouched
                self.assertIn('__parseInput__', clone.timing_metrics)

    def test_clone_lifecycle_times_are_its_own(self):
        """t_start / t_overall must reflect the clone's construction, not the
        template's lifetime."""
        for name, make in self._template_factories().items():
            with self.subTest(component=name):
                tmpl  = make()
                clone = self._clone(name, tmpl)
                self.assertGreaterEqual(clone.t_start, tmpl.t_end)
                self.assertAlmostEqual(clone.t_overall, clone.t_end - clone.t_start)

    def test_no_mutable_containers_shared_with_template(self):
        """No dict / set / list attribute of the template is the same object on
        the clone (leaf objects like DataFrames are intentionally shared)."""
        for name, make in self._template_factories().items():
            with self.subTest(component=name):
                tmpl  = make()
                clone = self._clone(name, tmpl)
                for key, value in tmpl.__dict__.items():
                    if isinstance(value, (dict, set, list)) and key in clone.__dict__:
                        self.assertIsNot(clone.__dict__[key], value,
                                         f'{name}.{key} is shared between template and clone')

    def test_mutation_does_not_cross_between_template_and_clone(self):
        """In-place mutation on either side must not leak to the other."""
        for name, make in self._template_factories().items():
            with self.subTest(component=name):
                tmpl  = make()
                clone = self._clone(name, tmpl)
                tmpl.sm_shared.add('__poisoned_by_template__')
                self.assertNotIn('__poisoned_by_template__', clone.sm_shared)
                def extra_metric(): return 42
                clone.gatherMetrics(extra_metric)
                self.assertIn('extra_metric', clone.timing_metrics)
                self.assertNotIn('extra_metric', tmpl.timing_metrics)

    def test_sm_shared_kwarg_still_shared_by_reference(self):
        """Explicit sm_shared= must still install the caller's set object on the
        clone (smallp panels rely on sharing one set across siblings)."""
        tmpl   = self.p2s.histop(_histop_df(), bin_by='cat')
        shared = {self.p2s.SM_COUNT}
        clone  = self.p2s.histop(template=tmpl, sm_shared=shared)
        self.assertIs(clone.sm_shared, shared)

    def test_clone_renders_same_svg_as_template(self):
        """State copying must not change render behavior: a plain clone of a
        rendered template produces the same (normalized) SVG.  Per-render random
        ids (e.g. id="piep_1027686477") are neutralized before comparing."""
        _randid_ = re.compile(r'_\d{5,}')
        for name, make in self._template_factories().items():
            with self.subTest(component=name):
                tmpl  = make()
                clone = self._clone(name, tmpl)
                self.assertEqual(_randid_.sub('_RID', normalize_svg(clone.svg)),
                                 _randid_.sub('_RID', normalize_svg(tmpl.svg)))


class TestCopyMutableContainers(unittest.TestCase):
    """Semantics of the recursive container copier itself."""

    def test_nested_containers_copied_leaves_shared(self):
        leaf   = object()
        value  = {'a': [1, {'b': leaf}], 'c': {leaf}}
        copied = _copy_mutable_containers_(value, {})
        self.assertIsNot(copied, value)
        self.assertIsNot(copied['a'], value['a'])
        self.assertIsNot(copied['a'][1], value['a'][1])
        self.assertIsNot(copied['c'], value['c'])
        self.assertIs(copied['a'][1]['b'], leaf)      # leaves shared by reference
        self.assertIn(leaf, copied['c'])
        value['a'].append('mutated')
        self.assertEqual(len(copied['a']), 2)         # copy unaffected

    def test_aliased_containers_stay_aliased(self):
        inner  = {'k': 1}
        value  = {'first': inner, 'second': inner}
        copied = _copy_mutable_containers_(value, {})
        self.assertIsNot(copied['first'], inner)
        self.assertIs(copied['first'], copied['second'])

    def test_self_referential_container_terminates(self):
        value = {}
        value['self'] = value
        copied = _copy_mutable_containers_(value, {})
        self.assertIsNot(copied, value)
        self.assertIs(copied['self'], copied)

    def test_non_container_passthrough(self):
        df = pl.DataFrame({'x': [1, 2]})
        for leaf in (None, 42, 'text', (1, 2), frozenset({1}), df):
            self.assertIs(_copy_mutable_containers_(leaf, {}), leaf)


if __name__ == '__main__':
    unittest.main()
