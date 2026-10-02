import unittest
import polars as pl
from polars2svg import Polars2SVG
from histop_dataframes import makeHistoDf, makeStatisticHistoDf, HistopAssertions, histopRows, orderedBins, spectrumColors
from svg_test_utils import normalize_svg


class TestHistopColor(HistopAssertions, unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()
        self.df  = makeHistoDf(n=200)

    # {bin: value} of one aggregate over the frame's bins
    def perBin(self, expr: pl.Expr) -> dict:
        return dict(self.df.group_by('cat').agg(expr.alias('m')).iter_rows())

    # {(bin, colour value): value} of one aggregate -- what a stacked bar's segments show
    def perSegment(self, color_field: str, expr: pl.Expr) -> dict:
        return {(b, v): m for b, v, m in self.df.group_by('cat', color_field).agg(expr.alias('m')).iter_rows()}

    def assertSameRender(self, a, b) -> None:
        self.assertEqual(normalize_svg(a.svg), normalize_svg(b.svg))

    # ── no color (default) ───────────────────────────────────────────────────

    def test_no_color_default(self):
        '''Default: color=None, all bars use the default data color.'''
        _h_, _rows_ = self.p2s.histop(self.df, 'cat'), self.perBin(pl.len())
        self.assertBarColors(_h_, {b: self.p2s.colorTyped('data', 'default') for b in _rows_})
        self.assertBarsShow(_h_, _rows_, orderedBins(_rows_))

    def test_no_color_explicit_none(self):
        self.assertSameRender(self.p2s.histop(self.df, 'cat', color=None), self.p2s.histop(self.df, 'cat'))

    def test_no_color_agg_type_simple(self):
        t = self.p2s.histop(self.df, 'cat')
        self.assertEqual(t._agg_type_, 'simple')

    # ── numeric color (spectrum) ─────────────────────────────────────────────

    def test_color_numeric_int_field(self):
        '''Numeric (Int32) color field → spectrum coloring on whole bar, by its sum per bin.'''
        _h_ = self.p2s.histop(self.df, 'cat', color='value')
        self.assertBarColors(_h_, spectrumColors(self.p2s, self.perBin(pl.col('value').sum())))
        self.assertBarsShow(_h_, self.perBin(pl.len()), orderedBins(self.perBin(pl.len())))

    def test_color_numeric_float_field(self):
        '''Float64 color field → spectrum coloring on whole bar, by its sum per bin.'''
        self.assertBarColors(self.p2s.histop(self.df, 'cat', color='score'),
                             spectrumColors(self.p2s, self.perBin(pl.col('score').sum())))

    def test_color_numeric_agg_type_simple(self):
        '''Numeric color stays in simple path (spectrum, not stacked).'''
        t = self.p2s.histop(self.df, 'cat', color='value')
        self.assertEqual(t._agg_type_, 'simple')

    def test_color_numeric_not_categorical(self):
        t = self.p2s.histop(self.df, 'cat', color='value')
        self.assertFalse(t._color_is_categorical_)

    def test_color_numeric_color_stat_in_df_agg(self):
        '''Numeric color → df_agg contains __color_stat__ column.'''
        t = self.p2s.histop(self.df, 'cat', color='value')
        self.assertIn('__color_stat__', t.df_agg.columns)

    def test_color_numeric_stat_range_computed(self):
        '''Numeric color → _color_stat_min_ and _color_stat_max_ are set.'''
        t = self.p2s.histop(self.df, 'cat', color='value')
        self.assertIsNotNone(t._color_stat_min_)
        self.assertIsNotNone(t._color_stat_max_)
        self.assertLessEqual(t._color_stat_min_, t._color_stat_max_)

    def test_color_numeric_cmagnitude_mean(self):
        '''(field, CMAGNITUDE_MEANp) → mean statistic, spectrum coloring.'''
        t = self.p2s.histop(self.df, 'cat', color=('value', self.p2s.CMAGNITUDE_MEANp))
        self.assertEqual(t._agg_type_, 'simple')
        self.assertIn('__color_stat__', t.df_agg.columns)

    def test_color_numeric_cmagnitude_max(self):
        t = self.p2s.histop(self.df, 'cat', color=('value', self.p2s.CMAGNITUDE_MAXp))
        self.assertEqual(t._agg_type_, 'simple')

    def test_color_numeric_statistic_mean(self):
        '''(field, MEANp) → mean statistic, spectrum coloring.'''
        t = self.p2s.histop(self.df, 'cat', color=('value', self.p2s.MEANp))
        self.assertEqual(t._agg_type_, 'simple')

    def test_color_numeric_with_numeric_count(self):
        '''Numeric color + numeric count field: the length is one sum, the colour the other.'''
        _h_, _len_ = self.p2s.histop(self.df, 'cat', color='score', count='value'), self.perBin(pl.col('value').sum())
        self.assertBarsShow(_h_, _len_, orderedBins(_len_))
        self.assertBarColors(_h_, spectrumColors(self.p2s, self.perBin(pl.col('score').sum())))

    # ── categorical color ────────────────────────────────────────────────────

    def test_color_categorical_string_field(self):
        '''String color field with different values than bin → stacked bars: each bin's
        rows split by group, each segment the group's colour, longest total first.'''
        _h_ = self.p2s.histop(self.df, 'cat', color='group')
        self.assertSegmentsShow(_h_, self.perSegment('group', pl.len()))
        self.assertEqual([b for b, _ in histopRows(_h_)], orderedBins(self.perBin(pl.len())))

    def test_color_categorical_agg_type_stacked(self):
        t = self.p2s.histop(self.df, 'cat', color='group')
        self.assertEqual(t._agg_type_, 'stacked')

    def test_color_categorical_is_categorical_flag(self):
        t = self.p2s.histop(self.df, 'cat', color='group')
        self.assertTrue(t._color_is_categorical_)

    def test_color_cset_tuple(self):
        '''(field, CSETp) → treat field as categorical even if numeric → stacked, one
        segment per distinct value.  On the statistic frame every segment is 18px or
        more, so none is pooled (see test_pooled_segments_keep_every_row).'''
        df = makeStatisticHistoDf()
        self.assertSegmentsShow(self.p2s.histop(df, 'cat', color=('value', self.p2s.CSETp)),
                                {(b, v): m for b, v, m in df.group_by('cat', 'value').len().iter_rows()})

    def test_pooled_segments_keep_every_row(self):
        '''Segments too thin to draw are pooled -- a value too thin in every bar into
        '(other)', and within a bar the thin shares join that same segment.  Whatever the
        pooling, it must not lose rows: each bar still adds up to its row count.'''
        _h_, _rows_ = self.p2s.histop(self.df, 'cat', color=('value', self.p2s.CSETp)), self.perBin(pl.len())
        self.assertLess(max(len(segs) for _, segs in histopRows(_h_)), self.df['value'].n_unique(), 'nothing was pooled')
        self.assertBarsShow(_h_, _rows_, orderedBins(_rows_))

    # Bar A: 'a' wide; 'b' 2 rows (thin here, wide in B); 'c' 4 rows (3.4px on the canvas,
    # under 3px on a plot narrowed by a legend); twenty values of one row each (thin in
    # every bar, so pooled into '(other)')
    _TWO_REMAINDERS_ = pl.DataFrame([('A', 'a')] * 280 + [('A', 'b')] * 2 + [('A', 'c')] * 4 +
                                    [('A', f'z{i}') for i in range(20)] + [('B', 'b')] * 40 + [('B', 'c')] * 2,
                                    schema=['bin', 'clr'], orient='row')

    def test_a_bar_has_one_remainder_in_the_other_colour(self):
        '''The values pooled into '(other)' and the shares too thin for their own bar used
        to be two remainders, the second in a colour of its own; they are one segment, in
        the '(other)' colour, at the end of the bar (PLANNING.md §5 C-histop-two-remainders).'''
        _other_ = self.p2s.color('(other)').lower()
        for _kw_ in (dict(), dict(legend=True)):
            with self.subTest(**_kw_):
                _rows_ = dict(histopRows(self.p2s.histop(self._TWO_REMAINDERS_, 'bin', color='clr', wxh=(256, 128), **_kw_)))
                for _lbl_, _segs_ in _rows_.items():
                    _fills_ = [f for _, _, f in _segs_]
                    self.assertEqual(len(_fills_), len(set(_fills_)), f'bar {_lbl_} draws a colour twice')
                    self.assertEqual(_fills_[-1], _other_, f'bar {_lbl_} does not end in its remainder')
                    self.assertTrue(set(_fills_) <= {self.p2s.color(v).lower() for v in ('a', 'b', 'c', '(other)')},
                                    f'bar {_lbl_} has a remainder in a colour of its own')
                # all 26 rows of A that are not 'a' (or, on the narrower plot, 'c') are the remainder
                _w_ = {f: w for _, w, f in _rows_['A']}
                _unit_ = _w_[self.p2s.color('a').lower()] / 280
                self.assertAlmostEqual(_w_[_other_], (26 if _kw_ else 22) * _unit_, delta=0.15)

    def test_colors_are_pooled_against_the_plot_not_the_canvas(self):
        '''Whether a value is drawn is decided on the plot it is drawn on.  'c' is 3.4px on
        the canvas but under 3px on a plot narrowed by a legend: it used to be listed in the
        legend and drawn nowhere.  Every value the legend names is drawn.'''
        _h_ = self.p2s.histop(self._TWO_REMAINDERS_, 'bin', color='clr', wxh=(256, 128), legend=True)
        _drawn_ = {f for _, segs in histopRows(_h_) for _, _, f in segs}
        self.assertEqual([v for v, _ in _h_.legend_info.entries], ['a', 'b', '(other)'])
        for _v_, _c_ in _h_.legend_info.entries: self.assertIn(_c_.lower(), _drawn_, f'{_v_!r} is in the legend but not drawn')
        # without a legend the plot is wide enough, and 'c' keeps its own segment
        _h_ = self.p2s.histop(self._TWO_REMAINDERS_, 'bin', color='clr', wxh=(256, 128))
        self.assertIn(self.p2s.color('c').lower(), {f for _, segs in histopRows(_h_) for _, _, f in segs})

    def test_color_cset_tuple_agg_type_stacked(self):
        t = self.p2s.histop(self.df, 'cat', color=('value', self.p2s.CSETp))
        self.assertEqual(t._agg_type_, 'stacked')

    def test_color_categorical_with_stackedbar_style(self):
        '''Categorical color + explicit STACKEDBARp → stacked bars: what the default draws.'''
        self.assertSameRender(self.p2s.histop(self.df, 'cat', color='group', style=self.p2s.STACKEDBARp),
                              self.p2s.histop(self.df, 'cat', color='group'))

    # ── color == bin field ────────────────────────────────────────────────────

    def test_color_same_as_bin_field_renders(self):
        '''color=bin_by (categorical): each bar is one segment in its own bin's colour.'''
        _h_, _rows_ = self.p2s.histop(self.df, 'cat', color='cat'), self.perBin(pl.len())
        self.assertBarColors(_h_, self.p2s.colors(sorted(_rows_)))
        self.assertBarsShow(_h_, _rows_, orderedBins(_rows_))

    def test_color_same_as_bin_agg_type_simple(self):
        '''When color==bin there is one color per bar; stays in simple path.'''
        t = self.p2s.histop(self.df, 'cat', color='cat')
        self.assertEqual(t._agg_type_, 'simple')

    def test_color_numeric_same_as_bin_uses_spectrum(self):
        '''Numeric color==bin → spectrum coloring, __color_stat__ present.'''
        t = self.p2s.histop(self.df, 'value', color='value')
        self.assertEqual(t._agg_type_, 'simple')
        self.assertIn('__color_stat__', t.df_agg.columns)

    # ── combined color + count ────────────────────────────────────────────────

    def test_color_categorical_with_numeric_count(self):
        '''Categorical color + numeric count field: each segment is its group's sum.'''
        self.assertSegmentsShow(self.p2s.histop(self.df, 'cat', color='group', count='value'),
                                self.perSegment('group', pl.col('value').sum()))


if __name__ == '__main__':
    unittest.main()
