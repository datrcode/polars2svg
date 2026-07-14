import unittest
from polars2svg import Polars2SVG
from histop_dataframes import makeHistoDf
from timep_dataframes import makeTimeDf


class TestBarChartStyles(unittest.TestCase):
    """Style logic shared between histop and timep: BARCHARTp, STACKEDBARp, BOXPLOTp, BOXPLOT_W_SWARMp."""

    @classmethod
    def setUpClass(cls):
        cls.p2s = Polars2SVG()
        cls.hdf = makeHistoDf(n=200)
        cls.tdf = makeTimeDf(n=200, year=(2020, 2024), month=(1, 12))

    def _histop(self, **kw):
        return self.p2s.histop(self.hdf, 'cat', **kw)

    def _timep(self, **kw):
        return self.p2s.timep(self.tdf, 'ts', **kw)

    def _timep_periodic(self, **kw):
        return self.p2s.timep(self.tdf, ('ts', self.p2s.PT_mp), **kw)

    # ── BARCHARTp ─────────────────────────────────────────────────────────────

    def test_barchart_default(self):
        """Default style is BARCHARTp for both components."""
        self.assertEqual(self._histop().style, self.p2s.BARCHARTp)
        self._timep()
        self._timep_periodic()

    def test_barchart_explicit(self):
        self._histop(style=self.p2s.BARCHARTp)
        self._timep(style=self.p2s.BARCHARTp)
        self._timep_periodic(style=self.p2s.BARCHARTp)

    def test_barchart_agg_type_simple(self):
        self.assertEqual(self._histop(style=self.p2s.BARCHARTp)._agg_type_, 'simple')
        self.assertEqual(self._timep(style=self.p2s.BARCHARTp)._agg_type_, 'simple')

    def test_barchart_with_categorical_color_produces_stacked(self):
        """BARCHARTp + categorical color still creates stacked bars (histop-specific)."""
        self.assertEqual(self._histop(style=self.p2s.BARCHARTp, color='group')._agg_type_, 'stacked')

    # ── STACKEDBARp ───────────────────────────────────────────────────────────

    def test_stackedbar_with_categorical_color(self):
        """STACKEDBARp + categorical color → stacked agg type."""
        self.assertEqual(self._histop(style=self.p2s.STACKEDBARp, color='group')._agg_type_, 'stacked')
        self.assertEqual(self._timep(style=self.p2s.STACKEDBARp, color='category')._agg_type_, 'stacked')
        self._timep_periodic(style=self.p2s.STACKEDBARp, color='category')

    def test_stackedbar_without_color_renders_as_simple(self):
        """STACKEDBARp without categorical color falls through to simple barchart."""
        self.assertEqual(self._histop(style=self.p2s.STACKEDBARp)._agg_type_, 'simple')
        self._timep(style=self.p2s.STACKEDBARp)
        self._timep_periodic(style=self.p2s.STACKEDBARp)

    def test_stackedbar_svg_contains_rects(self):
        self.assertIn('<rect', self._histop(style=self.p2s.STACKEDBARp, color='group')._repr_svg_())

    # ── BOXPLOTp ──────────────────────────────────────────────────────────────

    def test_boxplot_with_int_count_field(self):
        self._histop(style=self.p2s.BOXPLOTp, count='value')
        self._timep(style=self.p2s.BOXPLOTp, count='value')
        self._timep_periodic(style=self.p2s.BOXPLOTp, count='value')

    def test_boxplot_with_float_count_field(self):
        self._histop(style=self.p2s.BOXPLOTp, count='score')
        self._timep(style=self.p2s.BOXPLOTp, count='numeric')
        self._timep_periodic(style=self.p2s.BOXPLOTp, count='numeric')

    def test_boxplot_agg_type_boxplot(self):
        self.assertEqual(self._histop(style=self.p2s.BOXPLOTp, count='value')._agg_type_, 'boxplot')
        self.assertEqual(self._timep(style=self.p2s.BOXPLOTp, count='value')._agg_type_, 'boxplot')

    def test_boxplot_without_numeric_count_falls_back_to_barchart(self):
        """BOXPLOTp with no numeric count logs a warning and falls back to BARCHARTp."""
        self.assertEqual(self._histop(style=self.p2s.BOXPLOTp).style, self.p2s.BARCHARTp)
        self.assertEqual(self._timep(style=self.p2s.BOXPLOTp).style, self.p2s.BARCHARTp)

    def test_boxplot_periodic_without_numeric_falls_back(self):
        self.assertEqual(self._timep_periodic(style=self.p2s.BOXPLOTp).style, self.p2s.BARCHARTp)

    def test_boxplot_svg_contains_whisker_lines(self):
        """Box-and-whisker render produces <line> elements."""
        self.assertIn('<line', self._histop(style=self.p2s.BOXPLOTp, count='value')._repr_svg_())

    def test_boxplot_df_agg_has_stats_columns(self):
        t = self._histop(style=self.p2s.BOXPLOTp, count='value')
        for col in ('__box_min__', '__box_q1__', '__box_median__', '__box_q3__', '__box_max__'):
            self.assertIn(col, t.df_agg.columns)

    # ── BOXPLOT_W_SWARMp ──────────────────────────────────────────────────────

    def test_boxplot_swarm_with_int_count(self):
        self._histop(style=self.p2s.BOXPLOT_W_SWARMp, count='value')
        self._timep(style=self.p2s.BOXPLOT_W_SWARMp, count='value')
        self._timep_periodic(style=self.p2s.BOXPLOT_W_SWARMp, count='value')

    def test_boxplot_swarm_with_float_count(self):
        self._histop(style=self.p2s.BOXPLOT_W_SWARMp, count='score')
        self._timep(style=self.p2s.BOXPLOT_W_SWARMp, count='numeric')
        self._timep_periodic(style=self.p2s.BOXPLOT_W_SWARMp, count='numeric')

    def test_boxplot_swarm_agg_type_boxplot(self):
        self.assertEqual(self._histop(style=self.p2s.BOXPLOT_W_SWARMp, count='value')._agg_type_, 'boxplot')
        self.assertEqual(self._timep(style=self.p2s.BOXPLOT_W_SWARMp, count='value')._agg_type_, 'boxplot')

    def test_boxplot_swarm_without_numeric_falls_back(self):
        self.assertEqual(self._histop(style=self.p2s.BOXPLOT_W_SWARMp).style, self.p2s.BARCHARTp)
        self.assertEqual(self._timep(style=self.p2s.BOXPLOT_W_SWARMp).style, self.p2s.BARCHARTp)

    def test_boxplot_swarm_df_swarm_populated(self):
        h = self._histop(style=self.p2s.BOXPLOT_W_SWARMp, count='value')
        t = self._timep(style=self.p2s.BOXPLOT_W_SWARMp, count='value')
        self.assertIsNotNone(h.df_swarm)
        self.assertGreater(len(h.df_swarm), 0)
        self.assertIsNotNone(t.df_swarm)
        self.assertGreater(len(t.df_swarm), 0)

    def test_boxplot_swarm_svg_contains_circles(self):
        self.assertIn('<circle', self._histop(style=self.p2s.BOXPLOT_W_SWARMp, count='value')._repr_svg_())


if __name__ == '__main__':
    unittest.main()
