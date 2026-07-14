"""Cross-component consistency tests.

Verify behaviour that should be uniform across Histop, Timep, and (where
applicable) XYp.  Per-component-only tests have been moved to the relevant
component test files (test_histop_basic.py, test_histop_extended.py,
test_timep_basic.py).
"""
import unittest
import polars as pl

from polars2svg import Polars2SVG
from histop_dataframes import makeHistoDf
from timep_dataframes import makeTimeDf


class TestUndocumentedParameters(unittest.TestCase):
    """Verify that previously undocumented parameters are accepted and functional."""

    def setUp(self):
        self.p2s = Polars2SVG()
        self.hdf = makeHistoDf(n=100)
        self.tdf = makeTimeDf(n=100, year=(2020, 2023), month=(1, 12))

    # ── Histop: distribution strip ───────────────────────────────────────────

    def test_histop_distribution_true_default(self):
        t = self.p2s.histop(self.hdf, 'cat')
        self.assertTrue(t.distribution)

    def test_histop_distribution_false_accepted(self):
        t = self.p2s.histop(self.hdf, 'cat', distribution=False)
        self.assertFalse(t.distribution)
        self.assertIn('<svg', t._repr_svg_())

    def test_histop_distribution_bin_w_accepted(self):
        t = self.p2s.histop(self.hdf, 'cat', distribution_bin_w=20)
        self.assertEqual(t.distribution_bin_w, 20)
        self.assertIn('<svg', t._repr_svg_())

    # ── Histop: remainder_threshold ──────────────────────────────────────────

    def test_histop_remainder_threshold_accepted(self):
        t = self.p2s.histop(self.hdf, 'cat', color='group', remainder_threshold=1.0)
        self.assertEqual(t.remainder_threshold, 1.0)
        self.assertIn('<svg', t._repr_svg_())

    def test_histop_remainder_threshold_zero_keeps_all_segments(self):
        """With threshold=0, no segment should be collapsed into (other)."""
        t = self.p2s.histop(self.hdf, 'cat', color='group', remainder_threshold=0.0)
        if t._agg_type_ == 'stacked':
            self.assertNotIn('(other)', t.df_agg[t._color_field_].to_list())

    # ── Timep: remainder_threshold ───────────────────────────────────────────

    def test_timep_remainder_threshold_accepted(self):
        t = self.p2s.timep(self.tdf, 'ts', color='category', remainder_threshold=1.0)
        self.assertEqual(t.remainder_threshold, 1.0)
        self.assertIn('<svg', t._repr_svg_())

    def test_timep_remainder_threshold_periodic(self):
        t = self.p2s.timep(self.tdf, ('ts', self.p2s.PT_mp),
                           color='category', remainder_threshold=0.0)
        self.assertIn('<svg', t._repr_svg_())

    # ── Timep: min_label_spacing ─────────────────────────────────────────────

    def test_timep_min_label_spacing_accepted(self):
        t = self.p2s.timep(self.tdf, 'ts', min_label_spacing=5)
        self.assertEqual(t.min_label_spacing, 5)
        self.assertIn('<svg', t._repr_svg_())

    # ── Timep: date_range_shared ──────────────────────────────────────────────

    def test_timep_date_range_shared_accepted(self):
        import datetime
        lo = datetime.datetime(2020, 1, 1)
        hi = datetime.datetime(2023, 12, 31)
        t  = self.p2s.timep(self.tdf, 'ts', date_range_shared=(lo, hi))
        self.assertEqual(t.date_range_shared, (lo, hi))
        self.assertIn('<svg', t._repr_svg_())

    # ── Timep: swarm_max_pts (was already documented, regression guard) ───────

    def test_timep_swarm_max_pts_accepted(self):
        t = self.p2s.timep(self.tdf, 'ts',
                           style=self.p2s.BOXPLOT_W_SWARMp, count='value',
                           swarm_max_pts=3)
        self.assertEqual(t.swarm_max_pts, 3)


if __name__ == '__main__':
    unittest.main()
