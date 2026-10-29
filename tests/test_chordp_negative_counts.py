import json
import os
import subprocess
import sys
import unittest

import polars as pl

from polars2svg import Polars2SVG

#
# chordp with a negative count= (PLANNING.md §5 C-chordp-negative-count-hang).  With the
# node order derived, leafWalkFromEdges() turned each edge weight into a distance
# 1 / (w + eps), and Dijkstra never finished over a negative one: chordp() did not return.
# Edge counts are now clamped at zero right after aggregating, with one warning, as histop
# and timep do (C-negative-counts).
#
# The order-deriving checks run in a subprocess with a timeout, so a regression fails
# here instead of hanging the suite.  Everything else pins order=, which skips the walk.
#

_TIMEOUT_S_ = 180       # generous: a cold interpreter imports polars, scipy and the package

_DERIVED_ORDER_SRC_ = (
    "import json, polars as pl\n"
    "from polars2svg import Polars2SVG\n"
    "p2s = Polars2SVG()\n"
    "def order(n):\n"
    "    df = pl.DataFrame({'fm': ['a', 'b', 'c', 'd'], 'to': ['b', 'c', 'd', 'a'], 'n': n})\n"
    "    cp = p2s.chordp(df=df, relationships=[('fm', 'to')], count='n', link_size='vary')\n"
    "    return list(cp.order), len(cp.svg)\n"
    "print(json.dumps({'negative': order([-3, 5, 10, 2]), 'zero': order([0, 5, 10, 2])}))\n"
)


def _run_derived_order():
    _out_ = subprocess.run([sys.executable, '-c', _DERIVED_ORDER_SRC_], env={**os.environ},
                           capture_output=True, text=True, timeout=_TIMEOUT_S_)
    return _out_


class TestDerivedOrderTerminates(unittest.TestCase):
    def test_a_negative_count_no_longer_hangs_and_orders_as_zero(self):
        try:
            _out_ = _run_derived_order()
        except subprocess.TimeoutExpired:
            self.fail(f'chordp() with a negative count= did not return within {_TIMEOUT_S_}s '
                      '(C-chordp-negative-count-hang is back)')
        self.assertEqual(_out_.returncode, 0, _out_.stderr[-2000:])
        _got_ = json.loads(_out_.stdout.strip().splitlines()[-1])
        # rendered, and ordered exactly as the same frame with that count set to zero
        self.assertGreater(_got_['negative'][1], 0)
        self.assertEqual(_got_['negative'][0], _got_['zero'][0])


class TestNegativeEdgeCountsClamp(unittest.TestCase):
    def setUp(self):
        self.p2s = Polars2SVG()
        for _f_ in self.p2s.logger.filters:      # the logger says each message once per process
            if type(_f_).__name__ == 'OnceFilter': _f_.seen_messages.clear()

    def _cp_(self, fm, to, n, **kw):
        _df_ = pl.DataFrame({'fm': fm, 'to': to, 'n': n})
        return self.p2s.chordp(df=_df_, relationships=[('fm', 'to')], count='n',
                               order=sorted(set(fm) | set(to)), **kw)

    def test_a_negative_edge_is_drawn_as_zero_and_named_once(self):
        with self.assertLogs(self.p2s.logger, level='WARNING') as _cm_:
            _cp_ = self._cp_(['a', 'b', 'c'], ['b', 'c', 'a'], [-3, 5, 10], node_size='vary', link_size='vary')
        self.assertIn("ChP: count='n' sums below zero in 1 edge (a -> b); a negative count is drawn as zero",
                      [r.getMessage() for r in _cm_.records])
        _w_ = dict(((f, t), c) for f, t, c in _cp_.df_edge_weights.select('__fm__', '__to__', '__count__').iter_rows())
        self.assertEqual(_w_, {('a', 'b'): 0, ('b', 'c'): 5, ('c', 'a'): 10})
        self.assertNotIn('nan', _cp_.svg.lower())

    def test_the_clamp_is_on_the_summed_edge_not_each_row(self):
        # a -> b sums to -3 and is clamped; b -> c sums to 2 from a negative row and is not
        with self.assertLogs(self.p2s.logger, level='WARNING') as _cm_:
            _cp_ = self._cp_(['a', 'a', 'b', 'b', 'c'], ['b', 'b', 'c', 'c', 'a'], [5, -8, 4, -2, 6])
        self.assertIn("ChP: count='n' sums below zero in 1 edge (a -> b); a negative count is drawn as zero",
                      [r.getMessage() for r in _cm_.records])
        _w_ = dict(((f, t), c) for f, t, c in _cp_.df_edge_weights.select('__fm__', '__to__', '__count__').iter_rows())
        self.assertEqual(_w_, {('a', 'b'): 0, ('b', 'c'): 2, ('c', 'a'): 6})

    def test_no_negative_count_warns_nothing(self):
        with self.assertNoLogs(self.p2s.logger, level='WARNING'):
            _cp_ = self._cp_(['a', 'b', 'c'], ['b', 'c', 'a'], [3, 5, 10], link_size='vary')
        self.assertEqual(sorted(_cp_.df_edge_weights['__count__'].to_list()), [3, 5, 10])

    def test_every_count_negative_draws_without_nan(self):
        _cp_ = self._cp_(['a', 'b', 'c'], ['b', 'c', 'a'], [-1, -2, -3], node_size='vary', link_size='vary')
        self.assertEqual(sorted(_cp_.df_edge_weights['__count__'].to_list()), [0, 0, 0])
        self.assertNotIn('nan', _cp_.svg.lower())


if __name__ == '__main__':
    unittest.main()
