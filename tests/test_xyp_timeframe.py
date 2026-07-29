"""
Tests for the interactive xy time-axis shortcuts.

XYp.filterByTimeframe(top_df, mode) drives three keyboard interactions that only
apply when the x-axis represents linear time:

  'unfilter'      ('u')       -- every base (top-of-stack) row inside the visible
                                 timeframe; a no-op at the base of the stack.
  'expand_before' (shift+'e') -- current rows + base rows in the chunk immediately
                                 before the visible timeframe.
  'expand_after'  (ctrl+'e')  -- current rows + base rows in the chunk immediately
                                 after  the visible timeframe.

The chunk size is x_time_expand_perc (default 0.1) of the visible time span.

The controller half (InteractionController + the XYPI ReactiveHTML view) is
exercised by pushing a filtered level onto the stack and firing applyKeyOp, then
asserting the resulting stack sizes.
"""
import asyncio
import unittest
from datetime import datetime, timedelta

import polars as pl

from polars2svg import Polars2SVG


def _make_base():
    # 120 rows, one per minute, 10:00 .. 11:59; grp alternates a/b
    t0 = datetime(2026, 7, 28, 10, 0, 0)
    return t0, pl.DataFrame({
        'ts':  [t0 + timedelta(minutes=i) for i in range(120)],
        'val': [i % 7 for i in range(120)],
        'grp': ['a' if i % 2 == 0 else 'b' for i in range(120)],
    })


class TestXYpFilterByTimeframe(unittest.TestCase):

    def setUp(self):
        self.p2s = Polars2SVG()
        self.t0, self.base = _make_base()
        # Current view = window 10:30..11:30 AND grp=='a' (31 rows) -- a filtered level
        self.cur = self.base.filter(
            (pl.col('ts') >= self.t0 + timedelta(minutes=30)) &
            (pl.col('ts') <= self.t0 + timedelta(minutes=90)) &
            (pl.col('grp') == 'a')
        )
        self.xyp = self.p2s.xyp(self.cur, 'ts', 'val', wxh=(400, 300))

    # ── detection ─────────────────────────────────────────────────────────────

    def test_x_axis_is_time(self):
        self.assertTrue(self.xyp.__xAxisIsTime__())
        self.assertEqual(self.xyp.__timeColumn__(), 'ts')

    def test_current_timeframe_is_visible_span(self):
        t_min, t_max = self.xyp.__currentTimeframe__()
        self.assertEqual(t_min, self.t0 + timedelta(minutes=30))
        self.assertEqual(t_max, self.t0 + timedelta(minutes=90))

    def test_scalar_x_axis_is_not_time(self):
        xyp = self.p2s.xyp(self.base, 'val', 'val', wxh=(400, 300))
        self.assertFalse(xyp.__xAxisIsTime__())
        self.assertIsNone(xyp.filterByTimeframe(self.base, 'unfilter'))

    # ── unfilter ──────────────────────────────────────────────────────────────

    def test_unfilter_returns_base_rows_within_timeframe(self):
        out = self.xyp.filterByTimeframe(self.base, 'unfilter')
        # 10:30..11:30 inclusive = 61 rows in the base
        self.assertIsNotNone(out)
        self.assertEqual(len(out), 61)
        # every returned row lies within the visible timeframe
        self.assertEqual(
            out.filter((pl.col('ts') >= self.t0 + timedelta(minutes=30)) &
                       (pl.col('ts') <= self.t0 + timedelta(minutes=90))).height,
            61,
        )

    def test_unfilter_at_base_of_stack_is_noop(self):
        # current == base: nothing was filtered out, so there is nothing to add back
        xyp = self.p2s.xyp(self.base, 'ts', 'val', wxh=(400, 300))
        self.assertIsNone(xyp.filterByTimeframe(self.base, 'unfilter'))

    # ── expand ────────────────────────────────────────────────────────────────

    def test_expand_before_adds_earlier_chunk(self):
        out = self.xyp.filterByTimeframe(self.base, 'expand_before')
        # span = 60 min, 10% = 6 min -> [10:24, 10:30) = 6 new rows
        self.assertEqual(len(out), len(self.cur) + 6)
        # the added rows are strictly before the visible timeframe
        self.assertEqual(out.filter(pl.col('ts') < self.t0 + timedelta(minutes=30)).height, 6)

    def test_expand_after_adds_later_chunk(self):
        out = self.xyp.filterByTimeframe(self.base, 'expand_after')
        # (11:30, 11:36] = 6 new rows
        self.assertEqual(len(out), len(self.cur) + 6)
        self.assertEqual(out.filter(pl.col('ts') > self.t0 + timedelta(minutes=90)).height, 6)

    def test_expand_both_adds_earlier_and_later_chunks(self):
        out = self.xyp.filterByTimeframe(self.base, 'expand_both')
        # 6 min before + 6 min after = 12 new rows
        self.assertEqual(len(out), len(self.cur) + 12)
        self.assertEqual(out.filter(pl.col('ts') < self.t0 + timedelta(minutes=30)).height, 6)
        self.assertEqual(out.filter(pl.col('ts') > self.t0 + timedelta(minutes=90)).height, 6)

    def test_expand_percent_parameter_scales_the_chunk(self):
        xyp = self.p2s.xyp(self.cur, 'ts', 'val', wxh=(400, 300), x_time_expand_perc=0.5)
        out = xyp.filterByTimeframe(self.base, 'expand_before')
        # 50% of 60 min = 30 min -> [10:00, 10:30) = 30 new rows
        self.assertEqual(len(out), len(self.cur) + 30)

    def test_expand_before_at_left_edge_is_noop(self):
        # window starts at the very first base timestamp -> nothing earlier exists
        cur = self.base.filter(pl.col('ts') <= self.t0 + timedelta(minutes=60))
        xyp = self.p2s.xyp(cur, 'ts', 'val', wxh=(400, 300))
        self.assertIsNone(xyp.filterByTimeframe(self.base, 'expand_before'))

    def test_x_time_expand_perc_survives_template_clone(self):
        xyp = self.p2s.xyp(self.cur, 'ts', 'val', wxh=(400, 300), x_time_expand_perc=0.25)
        clone = self.p2s.xyp(df=self.cur, template=xyp)
        self.assertEqual(clone.x_time_expand_perc, 0.25)


class TestXYpiTimeframeDispatch(unittest.TestCase):

    def setUp(self):
        self.p2s = Polars2SVG()
        self.t0, self.base = _make_base()
        self.cur = self.base.filter(
            (pl.col('ts') >= self.t0 + timedelta(minutes=30)) &
            (pl.col('ts') <= self.t0 + timedelta(minutes=90)) &
            (pl.col('grp') == 'a')
        )
        self.xypi = self.p2s.xypi(self.p2s.xyp(self.base, 'ts', 'val', wxh=(400, 300)))

    def _fire_key(self, key, shift=False, ctrl=False):
        self.xypi.x_mouse, self.xypi.y_mouse = 10, 10
        self.xypi.shiftkey, self.xypi.ctrlkey = shift, ctrl
        self.xypi.key_op_finished = key
        asyncio.run(self.xypi.applyKeyOp(None))

    def _sizes(self):
        s = self.xypi.mvc.stacks['default']
        return [len(d) for d in s['dfs']]

    def test_help_text_advertises_time_keys(self):
        self.assertIn('unfilter rows within the visible timeframe', self.xypi._keyboard_commands_)
        self.assertIn('expand timeframe backward', self.xypi._keyboard_commands_)
        self.assertIn('expand timeframe forward', self.xypi._keyboard_commands_)

    def test_keydown_js_captures_u_and_e(self):
        kd = type(self.xypi)._scripts['myOnKeyDown']
        self.assertIn("== 'u'", kd)
        self.assertIn("== 'e'", kd)

    def test_unfilter_then_expand_push_onto_stack(self):
        # filter down to the 31-row level (like a drag selection would)
        asyncio.run(self.xypi.mvc.pushStack(self.xypi, self.cur))
        self.assertEqual(self._sizes(), [120, 31])

        # 'u' -> unfilter within the visible timeframe (61 rows)
        self._fire_key('u')
        self.assertEqual(self._sizes(), [120, 31, 61])

        # ctrl+'e' -> expand forward (+6)
        self._fire_key('e', ctrl=True)
        self.assertEqual(self._sizes(), [120, 31, 61, 67])

        # shift+'e' -> expand backward (+6)
        self._fire_key('e', shift=True)
        self.assertEqual(self._sizes(), [120, 31, 61, 67, 73])

    def test_plain_e_expands_both_directions(self):
        asyncio.run(self.xypi.mvc.pushStack(self.xypi, self.cur))
        self.assertEqual(self._sizes(), [120, 31])
        self._fire_key('e')          # plain 'e' -> expand both directions (+6 before, +6 after)
        self.assertEqual(self._sizes(), [120, 31, 43])


if __name__ == '__main__':
    unittest.main()
