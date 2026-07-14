"""gatherMetrics() mixin behavior across all render components.

The timing/metrics machinery is shared mixin code: each component only differs
in its constructor.  One parametrized loop verifies the per-component wiring
(timing_metrics populated, __parseInput__ recorded); the user-callable pass-through
and recording semantics are mixin behavior and are verified once.
"""
import unittest
import datetime
import polars as pl
from polars2svg import Polars2SVG


def _histop_df():
    return pl.DataFrame({'cat': ['a', 'b', 'a', 'c', 'b'], 'val': [1, 2, 3, 4, 5]})


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


def _smallp_df():
    return pl.DataFrame({'x': [1, 2, 3, 4], 'y': [1, 2, 3, 4], 'cat': ['a', 'a', 'b', 'b']})


class TestGatherMetricsAcrossComponents(unittest.TestCase):
    """Every component populates timing_metrics and records __parseInput__."""

    @classmethod
    def setUpClass(cls):
        cls.p2s = Polars2SVG()

    def _component_factories(self):
        p2s = self.p2s

        def _smallp():
            df       = _smallp_df()
            template = p2s.xyp(df, 'x', 'y', wxh=(128, 128))
            return p2s.smallp(df, 'cat', template, wxh=(256, 256))

        return {
            'histop':       lambda: p2s.histop(_histop_df(), bin_by='cat'),
            'timep':        lambda: p2s.timep(_timep_df(), 'ts'),
            'xyp':          lambda: p2s.xyp(_xyp_df(), 'x', 'y'),
            'chordp':       lambda: p2s.chordp(df=_edge_df(), relationships=[('fm', 'to')]),
            'linkp':        lambda: p2s.linkp(_edge_df(), relationships=[('fm', 'to')],
                                              pos={'a': [0, 0], 'b': [1, 0], 'c': [0.5, 0.866]}),
            'spreadlinesp': lambda: p2s.spreadlinesp(_spread_df(), [('fm', 'to')],
                                                     ego='a', time='time'),
            'smallp':       _smallp,
        }

    def test_timing_metrics_populated_and_parse_input_recorded(self):
        for _name_, _make_ in self._component_factories().items():
            with self.subTest(component=_name_):
                _c_ = _make_()
                self.assertIsInstance(_c_.timing_metrics, dict)
                self.assertGreater(len(_c_.timing_metrics), 0)
                self.assertIn('__parseInput__', _c_.timing_metrics)


class TestGatherMetricsMixinBehavior(unittest.TestCase):
    """Shared mixin semantics -- verified once (histop stands in for all components)."""

    @classmethod
    def setUpClass(cls):
        cls.p2s = Polars2SVG()

    def setUp(self):
        self.hp = self.p2s.histop(_histop_df(), bin_by='cat')

    def test_user_callable_returns_result(self):
        def my_func(x, y):
            return x + y
        self.assertEqual(self.hp.gatherMetrics(my_func, 3, 4), 7)

    def test_user_callable_kwargs_forwarded(self):
        def my_func(x, y=0):
            return x - y
        self.assertEqual(self.hp.gatherMetrics(my_func, 10, y=4), 6)

    def test_user_callable_recorded_in_timing_metrics(self):
        def my_func():
            return 42
        self.hp.gatherMetrics(my_func)
        self.assertIn('my_func', self.hp.timing_metrics)
        self.assertGreaterEqual(self.hp.timing_metrics['my_func'], 0.0)

    def test_user_callable_exception_propagates(self):
        def boom():
            raise RuntimeError('boom')
        with self.assertRaises(RuntimeError):
            self.hp.gatherMetrics(boom)


if __name__ == '__main__':
    unittest.main()
