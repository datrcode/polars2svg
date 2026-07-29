"""
Tests for the interactive timep time-axis shortcuts.

Timep.filterByTimeframe(top_df, mode) drives the same three interactions as XYp,
but only when the time axis is *linear* (non-periodic). The visible timeframe for
a binned bar chart is [t0, t1): the start of the first visible bin to the end of
the last visible bin.

  'unfilter'      ('u')       -- every base (top-of-stack) row inside the visible
                                 timeframe; a no-op at the base of the stack.
  'expand_before' (shift+'e') -- current rows + base rows in the chunk before it.
  'expand_after'  (ctrl+'e')  -- current rows + base rows in the chunk after it.
  'expand_both'   ('e')       -- current rows + both chunks.

Chunk size is x_time_expand_perc (default 0.1) of the visible span per direction.
On a periodic time axis the feature is inert (every call returns None).
"""
import asyncio
import unittest
from datetime import datetime, timedelta

import polars as pl

from polars2svg import Polars2SVG


def _make_base():
    # 48 hourly rows over two days from 2026-07-28 00:00; grp alternates a/b.
    t0 = datetime(2026, 7, 28, 0, 0, 0)
    return t0, pl.DataFrame({
        'ts':  [t0 + timedelta(hours=i) for i in range(48)],
        'val': [i % 5 for i in range(48)],
        'grp': ['a' if i % 2 == 0 else 'b' for i in range(48)],
    })


class TestTimepFilterByTimeframe(unittest.TestCase):

    def setUp(self):
        self.p2s = Polars2SVG()
        self.t0, self.base = _make_base()
        # Current view = hours 10..20 AND grp=='a' (even hours -> 6 rows), hourly bins.
        self.cur = self.base.filter(
            (pl.col('ts') >= self.t0 + timedelta(hours=10)) &
            (pl.col('ts') <= self.t0 + timedelta(hours=20)) &
            (pl.col('grp') == 'a')
        )
        self.tp = self.p2s.timep(self.cur, time=('ts', self.p2s.LT_Y_m_d_Hp), wxh=(500, 200))

    # ── timeframe detection ─────────────────────────────────────────────────

    def test_current_timeframe_is_bin_aligned(self):
        t0, t1 = self.tp.__currentTimeframe__()
        self.assertEqual(t0, self.t0 + timedelta(hours=10))   # first visible bin start
        self.assertEqual(t1, self.t0 + timedelta(hours=21))   # end of last bin (20:00 + 1h), exclusive

    # ── unfilter ────────────────────────────────────────────────────────────

    def test_unfilter_returns_base_rows_within_timeframe(self):
        out = self.tp.filterByTimeframe(self.base, 'unfilter')
        # [10:00, 21:00) = hours 10..20 = 11 base rows
        self.assertIsNotNone(out)
        self.assertEqual(len(out), 11)

    def test_unfilter_at_base_of_stack_is_noop(self):
        tp = self.p2s.timep(self.base, time=('ts', self.p2s.LT_Y_m_d_Hp), wxh=(500, 200))
        self.assertIsNone(tp.filterByTimeframe(self.base, 'unfilter'))

    # ── expand ──────────────────────────────────────────────────────────────

    def test_expand_before_adds_earlier_chunk(self):
        # 11 visible bins, round(0.1*11)=1 bin -> [09:00, 10:00) -> hour 9 = 1 new row
        out = self.tp.filterByTimeframe(self.base, 'expand_before')
        self.assertEqual(len(out), len(self.cur) + 1)
        self.assertEqual(out.filter(pl.col('ts') < self.t0 + timedelta(hours=10)).height, 1)

    def test_expand_after_adds_later_chunk(self):
        # round(0.1*11)=1 bin -> [21:00, 22:00) -> hour 21 = 1 new row
        out = self.tp.filterByTimeframe(self.base, 'expand_after')
        self.assertEqual(len(out), len(self.cur) + 1)
        self.assertEqual(out.filter(pl.col('ts') >= self.t0 + timedelta(hours=21)).height, 1)

    def test_expand_both_adds_both_chunks(self):
        out = self.tp.filterByTimeframe(self.base, 'expand_both')
        self.assertEqual(len(out), len(self.cur) + 2)   # 1 bin before + 1 bin after

    def test_expand_percent_parameter_scales_the_chunk(self):
        tp = self.p2s.timep(self.cur, time=('ts', self.p2s.LT_Y_m_d_Hp), wxh=(500, 200),
                            x_time_expand_perc=0.5)
        out = tp.filterByTimeframe(self.base, 'expand_before')
        # round(0.5*11)=6 bins -> [04:00, 10:00) -> hours 4..9 = 6 new rows
        self.assertEqual(len(out), len(self.cur) + 6)

    def test_expansion_snaps_to_whole_bins(self):
        # Daily bars, 4 events/day: at 10% a 5-day view would be a 0.5-day raw slice; snapping
        # to whole bins adds a full day (4 events) instead of a partially-filled bar.
        d0 = datetime(2026, 1, 1)
        daily = pl.DataFrame({'ts':  [d0 + timedelta(hours=6 * i) for i in range(20 * 4)],
                              'val': list(range(20 * 4))})
        cur = daily.filter((pl.col('ts') >= d0 + timedelta(days=5)) &
                           (pl.col('ts') <  d0 + timedelta(days=10)))    # 5 visible days
        tp = self.p2s.timep(cur, time=('ts', self.p2s.LT_Y_m_dp), wxh=(500, 200))
        added = tp.filterByTimeframe(daily, 'expand_after').join(cur, on='ts', how='anti')
        self.assertEqual(added.height, 4)                                # a whole day (4 events)
        self.assertEqual(added['ts'].dt.truncate('1d').n_unique(), 1)    # exactly one day bin

    def test_x_time_expand_perc_survives_template_clone(self):
        tp = self.p2s.timep(self.cur, time=('ts', self.p2s.LT_Y_m_d_Hp), wxh=(500, 200),
                            x_time_expand_perc=0.25)
        clone = self.p2s.timep(df=self.cur, template=tp)
        self.assertEqual(clone.x_time_expand_perc, 0.25)

    # ── periodic time is inert ────────────────────────────────────────────────

    def test_periodic_time_axis_returns_none(self):
        tp = self.p2s.timep(self.base, time=('ts', self.p2s.PT_Hp), wxh=(500, 200))
        self.assertTrue(tp._is_periodic_)
        self.assertIsNone(tp.__currentTimeframe__())
        self.assertIsNone(tp.filterByTimeframe(self.base, 'unfilter'))
        self.assertIsNone(tp.filterByTimeframe(self.base, 'expand_both'))


class TestTimepiTimeframeDispatch(unittest.TestCase):

    def setUp(self):
        self.p2s = Polars2SVG()
        self.t0, self.base = _make_base()
        self.cur = self.base.filter(
            (pl.col('ts') >= self.t0 + timedelta(hours=10)) &
            (pl.col('ts') <= self.t0 + timedelta(hours=20)) &
            (pl.col('grp') == 'a')
        )
        self.tpi = self.p2s.timepi(
            self.p2s.timep(self.base, time=('ts', self.p2s.LT_Y_m_d_Hp), wxh=(500, 200)))

    def _fire_key(self, key, shift=False, ctrl=False):
        self.tpi.x_mouse, self.tpi.y_mouse = 10, 10
        self.tpi.shiftkey, self.tpi.ctrlkey = shift, ctrl
        self.tpi.key_op_finished = key
        asyncio.run(self.tpi.applyKeyOp(None))

    def _sizes(self):
        s = self.tpi.mvc.stacks['default']
        return [len(d) for d in s['dfs']]

    def test_help_and_js_wiring_present(self):
        self.assertIn('unfilter rows within the visible timeframe', self.tpi._keyboard_commands_)
        kd = type(self.tpi)._scripts['myOnKeyDown']
        self.assertIn("== 'u'", kd)
        self.assertIn("== 'e'", kd)

    def test_unfilter_then_expand_push_onto_stack(self):
        asyncio.run(self.tpi.mvc.pushStack(self.tpi, self.cur))   # 6-row filtered level
        self.assertEqual(self._sizes(), [48, 6])
        self._fire_key('u')                                       # -> 11 rows
        self.assertEqual(self._sizes(), [48, 6, 11])
        self._fire_key('e')                                       # plain e -> both (+1 bin before, +1 bin after)
        self.assertEqual(self._sizes(), [48, 6, 11, 13])


if __name__ == '__main__':
    unittest.main()
