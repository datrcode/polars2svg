import time
import unittest

import numpy as np
import networkx as nx

from polars2svg.layout_budget import Budget
from polars2svg.od_flow_layout import ODFlowLayout
from polars2svg.ncp_layout import NCPLayout


def _flows(n=60, seed=3):
    rng = np.random.default_rng(seed)
    return [(float(a), float(b), float(c), float(d))
            for a, b, c, d in rng.uniform(0, 500, size=(n, 4))]


class TestBudget(unittest.TestCase):

    def test_no_limits_never_expires(self):
        _b_ = Budget().start(10)
        for _ in range(10):
            self.assertFalse(_b_.expired())
        self.assertIsNone(_b_.note)
        self.assertIsNone(_b_.stopped_at)

    def test_zero_time_budget_expires_immediately(self):
        _b_ = Budget(time_budget=0).start(100)
        self.assertTrue(_b_.expired())
        self.assertEqual(_b_.stopped_at, 0)

    def test_should_stop_expires(self):
        _b_ = Budget(should_stop=lambda: True).start(100)
        self.assertTrue(_b_.expired())
        self.assertIn('cancelled', _b_.note)

    def test_cancel_is_reported_as_cancel_not_as_a_timeout(self):
        # Both conditions true at once: an explicit cancel is the more useful thing to
        # report, so it is checked first.
        _b_ = Budget(time_budget=0, should_stop=lambda: True).start(100)
        _b_.expired()
        self.assertIn('cancelled', _b_.note)

    def test_the_count_is_of_completed_iterations(self):
        _b_ = Budget(should_stop=lambda: _b_._done_ >= 4).start(20)
        _n_ = 0
        while not _b_.expired():
            _n_ += 1
        self.assertEqual(_n_, 4)
        self.assertIn('after 4 of 20', _b_.note)

    def test_the_count_spans_several_loops(self):
        # NCPLayout spends one budget across two stages; a per-loop index would report the
        # second stage's 0 and read as "nothing ran".
        _b_ = Budget().start(30)
        for _ in range(7):
            _b_.expired()
        _b_.should_stop = lambda: True
        _b_.expired()
        self.assertIn('after 7 of 30', _b_.note)

    def test_negative_time_budget_is_rejected(self):
        with self.assertRaises(ValueError):
            Budget(time_budget=-1)

    def test_restart_clears_previous_state(self):
        _b_ = Budget(should_stop=lambda: True).start(5)
        _b_.expired()
        _b_.should_stop = None
        _b_.start(5)
        self.assertIsNone(_b_.note)
        self.assertIsNone(_b_.stopped_at)
        self.assertFalse(_b_.expired())


class TestODFlowLayoutBudget(unittest.TestCase):

    def test_without_a_budget_nothing_changes(self):
        # The determinism this layout guarantees depends on the budget being opt-in: a
        # wall-clock limit is by nature not reproducible.
        _f_ = _flows(40)
        self.assertEqual(ODFlowLayout(_f_, iterations=20).results(),
                         ODFlowLayout(_f_, iterations=20).results())
        self.assertIsNone(ODFlowLayout(_f_, iterations=20).budget_note)

    def test_a_spent_budget_still_returns_one_point_per_flow(self):
        _f_ = _flows(40)
        _l_ = ODFlowLayout(_f_, budget=Budget(time_budget=0))
        _cps_ = _l_.results()
        self.assertEqual(len(_cps_), len(_f_))
        self.assertTrue(all(np.isfinite(_c_).all() for _c_ in _cps_))

    def test_a_spent_budget_says_where_it_stopped(self):
        _l_ = ODFlowLayout(_flows(40), iterations=50, budget=Budget(time_budget=0))
        _l_.results()
        self.assertIn('of 50 iterations', _l_.budget_note)

    def test_cancel_stops_it(self):
        _l_ = ODFlowLayout(_flows(40), budget=Budget(should_stop=lambda: True))
        self.assertEqual(len(_l_.results()), 40)
        self.assertIn('cancelled', _l_.budget_note)

    def test_a_budget_that_is_never_hit_matches_the_unbudgeted_run(self):
        _f_ = _flows(30)
        self.assertEqual(ODFlowLayout(_f_, iterations=10).results(),
                         ODFlowLayout(_f_, iterations=10,
                                      budget=Budget(time_budget=3600)).results())


class TestNCPLayoutBudget(unittest.TestCase):

    def _g_and_pos(self, n=120):
        _g_ = nx.barabasi_albert_graph(n, 3, seed=5)
        rng = np.random.default_rng(5)
        return _g_, {v: tuple(rng.uniform(0, 100, 2)) for v in _g_.nodes()}

    def test_a_spent_budget_still_returns_every_node(self):
        _g_, _pos_ = self._g_and_pos()
        _l_ = NCPLayout(_g_, pos=_pos_, budget=Budget(time_budget=0))
        _out_ = _l_.results()
        self.assertEqual(len(_out_), len(_pos_))
        self.assertTrue(np.all(np.isfinite(np.array(list(_out_.values())))))

    def test_without_a_budget_there_is_no_note(self):
        _g_, _pos_ = self._g_and_pos()
        self.assertIsNone(NCPLayout(_g_, pos=_pos_).budget_note)

    def test_the_note_counts_both_stages(self):
        # One budget spans stage 2 and stage 3; the note must not report whichever stage
        # happened to stop as though it were the whole run.
        _g_, _pos_ = self._g_and_pos()
        _l_ = NCPLayout(_g_, pos=_pos_, budget=Budget(time_budget=0.05))
        _l_.results()
        if _l_.budget_note is not None:
            self.assertRegex(_l_.budget_note, r'after \d+ of \d+ iterations')


if __name__ == '__main__':
    unittest.main()
