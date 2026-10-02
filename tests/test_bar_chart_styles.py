import unittest
import polars as pl
from polars2svg import Polars2SVG
from histop_dataframes import makeHistoDf
from timep_dataframes import makeTimeDf, timeBins
from svg_test_utils import normalize_svg

_BOX_ = ('__box_min__', '__box_q1__', '__box_median__', '__box_q3__', '__box_max__', '__count__')


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

    # The frame's rows, each with the bin the component puts it in, as '__bin__'
    def _binned(self, t, df: pl.DataFrame | None = None) -> pl.DataFrame:
        if hasattr(t, '_bin_col_'):
            return (self.hdf if df is None else df).with_columns(pl.col(t._bin_col_).alias('__bin__'))
        return timeBins(t, self.tdf if df is None else df, 'ts')[0]

    # Each bin's box is its rows' five numbers and count, computed here by polars; a bin
    # with no rows draws no box
    def assertBoxes(self, t, field: str, df: pl.DataFrame | None = None) -> None:
        self.assertEqual(t._agg_type_, 'boxplot')
        _want_ = {b: tuple(rest) for b, *rest in self._binned(t, df).group_by('__bin__').agg(
                      pl.col(field).min().alias('a'), pl.col(field).quantile(0.25).alias('b'),
                      pl.col(field).median().alias('c'), pl.col(field).quantile(0.75).alias('d'),
                      pl.col(field).max().alias('e'), pl.len().alias('n')).iter_rows()}
        _key_ = next(c for c in t.df_agg.columns if c not in _BOX_)
        _got_ = {b: tuple(rest) for b, *rest in t.df_agg.select(_key_, *_BOX_).iter_rows() if rest[-1] > 0}
        self.assertEqual(sorted(_got_), sorted(_want_))
        for _b_ in _want_:
            for _stat_, _w_, _g_ in zip(_BOX_, _want_[_b_], _got_[_b_]):
                self.assertAlmostEqual(float(_g_), float(_w_), places=9, msg=f'bin {_b_} {_stat_}')

    # The swarm draws each bin's own values, at most swarm_max_pts of them
    def assertSwarm(self, t, field: str, df: pl.DataFrame | None = None) -> None:
        _cap_ = t.swarm_max_pts
        _rows_ = dict(self._binned(t, df).group_by('__bin__').agg(pl.col(field)).iter_rows())
        _key_  = t.df_swarm.columns[0]
        if hasattr(t, '_bin_col_') or _key_ == '__time_bin__':
            _dots_ = dict(t.df_swarm.group_by(_key_).agg(pl.col(field)).iter_rows())
        else:
            _dots_ = dict(timeBins(t, t.df_swarm, _key_)[0].group_by('__bin__').agg(pl.col(field)).iter_rows())
        self.assertEqual(sorted(_dots_), sorted(_rows_))
        for _b_, _vals_ in _rows_.items():
            self.assertEqual(len(_dots_[_b_]), min(len(_vals_), _cap_), f'bin {_b_}')
            _left_ = list(_vals_)
            for _v_ in _dots_[_b_]:
                self.assertIn(_v_, _left_, f'bin {_b_}: a swarm dot is not one of its rows')
                _left_.remove(_v_)

    # ── BARCHARTp ─────────────────────────────────────────────────────────────

    def test_barchart_default(self):
        """Default style is BARCHARTp for both components."""
        self.assertEqual(self._histop().style, self.p2s.BARCHARTp)
        self._timep()
        self._timep_periodic()

    def test_barchart_explicit(self):
        '''Spelling out the default draws what the default draws.'''
        for _make_ in (self._histop, self._timep, self._timep_periodic):
            with self.subTest(component=_make_.__name__):
                self.assertEqual(normalize_svg(_make_(style=self.p2s.BARCHARTp).svg), normalize_svg(_make_().svg))

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
        self.assertBoxes(self._histop(style=self.p2s.BOXPLOTp, count='value'), 'value')
        self.assertBoxes(self._timep(style=self.p2s.BOXPLOTp, count='value'), 'value')
        self.assertBoxes(self._timep_periodic(style=self.p2s.BOXPLOTp, count='value'), 'value')

    def test_boxplot_with_float_count_field(self):
        self.assertBoxes(self._histop(style=self.p2s.BOXPLOTp, count='score'), 'score')
        self.assertBoxes(self._timep(style=self.p2s.BOXPLOTp, count='numeric'), 'numeric')
        self.assertBoxes(self._timep_periodic(style=self.p2s.BOXPLOTp, count='numeric'), 'numeric')

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
        for _t_ in (self._histop(style=self.p2s.BOXPLOT_W_SWARMp, count='value'),
                    self._timep(style=self.p2s.BOXPLOT_W_SWARMp, count='value'),
                    self._timep_periodic(style=self.p2s.BOXPLOT_W_SWARMp, count='value')):
            self.assertBoxes(_t_, 'value')
            self.assertSwarm(_t_, 'value')

    def test_boxplot_swarm_with_float_count(self):
        for _t_, _f_ in ((self._histop(style=self.p2s.BOXPLOT_W_SWARMp, count='score'), 'score'),
                         (self._timep(style=self.p2s.BOXPLOT_W_SWARMp, count='numeric'), 'numeric'),
                         (self._timep_periodic(style=self.p2s.BOXPLOT_W_SWARMp, count='numeric'), 'numeric')):
            self.assertBoxes(_t_, _f_)
            self.assertSwarm(_t_, _f_)

    def test_the_swarm_is_capped_per_bin_on_every_path(self):
        '''2,000 rows over one year: every monthly and every per-category bin holds more
        rows than swarm_max_pts, and each draws exactly that many.  A linear timep used to
        rank its rows per raw timestamp, which capped nothing -- 183 dots in one bin
        against a cap of 50 (PLANNING.md §5 C-timep-swarm-cap).'''
        _tdf_ = makeTimeDf(n=2000, year=(2023, 2023), month=(1, 12), seed=5)
        _hdf_ = makeHistoDf(n=2000)
        for _t_, _df_ in ((self.p2s.histop(_hdf_, 'cat', style=self.p2s.BOXPLOT_W_SWARMp, count='value'), _hdf_),
                          (self.p2s.timep(_tdf_, 'ts', style=self.p2s.BOXPLOT_W_SWARMp, count='value'), _tdf_),
                          (self.p2s.timep(_tdf_, ('ts', self.p2s.PT_mp), style=self.p2s.BOXPLOT_W_SWARMp, count='value'), _tdf_)):
            with self.subTest(component=type(_t_).__name__, level=getattr(_t_, '_time_enum_', None)):
                self.assertGreater(min(self._binned(_t_, _df_).group_by('__bin__').len()['len']), _t_.swarm_max_pts,
                                   'a bin under the cap proves nothing')
                self.assertBoxes(_t_, 'value', _df_)
                self.assertSwarm(_t_, 'value', _df_)

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
