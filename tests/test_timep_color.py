import datetime
import unittest
import polars as pl
from polars2svg import Polars2SVG
from timep_dataframes import makeTimeDf, TimepAssertions, perTimeBin, timepColumns
from histop_dataframes import spectrumColors
from svg_test_utils import normalize_svg

_BOTH_MODES_ = ('ts', ('ts', 'PT_mp'))    # the auto linear level, and the month-of-year cycle

# Column A: 'a' tall; 'b' 2 rows (thin here, tall in B); 'c' 4 rows (3.3px on the canvas,
# under 3px on a plot shortened by a legend underneath); twenty values of one row each
# (thin in every column, so pooled into '(other)')
_A_, _B_ = datetime.datetime(2020, 1, 1), datetime.datetime(2020, 1, 3)
_TWO_REMAINDERS_ = pl.DataFrame([(_A_, 'a')] * 280 + [(_A_, 'b')] * 2 + [(_A_, 'c')] * 4 +
                                [(_A_, f'z{i}') for i in range(20)] + [(_B_, 'b')] * 40 + [(_B_, 'c')] * 2,
                                schema=['ts', 'clr'], orient='row')


class TestTimepColor(TimepAssertions, unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()
        self.df  = makeTimeDf(n=200, year=(2020, 2024), month=(1, 12), seed=8)

    def modes(self) -> list:
        return [m if isinstance(m, str) else (m[0], getattr(self.p2s, m[1])) for m in _BOTH_MODES_]

    # Two spellings that mean the same thing render identically, in both modes
    def assertSameInBothModes(self, a: dict, b: dict) -> None:
        for _time_ in self.modes():
            with self.subTest(time=str(_time_)):
                self.assertEqual(normalize_svg(self.p2s.timep(self.df, _time_, **a).svg),
                                 normalize_svg(self.p2s.timep(self.df, _time_, **b).svg))

    # Numeric colour: each column is the spectrum of its bin's per-bin sum of the field,
    # and as tall as `length` says
    def assertSpectrumInBothModes(self, field: str, length: pl.Expr = pl.len(), **extra) -> None:
        for _time_ in self.modes():
            with self.subTest(time=str(_time_)):
                _t_ = self.p2s.timep(self.df, _time_, color=field, **extra)
                self.assertColumnColors(_t_, self.df, 'ts', spectrumColors(self.p2s, perTimeBin(_t_, self.df, 'ts', pl.col(field).sum())), length)

    # Categorical colour: stacked columns, each segment its value's share of `length`
    def assertStacksInBothModes(self, color, length: pl.Expr = pl.len(), **extra) -> None:
        for _time_ in self.modes():
            with self.subTest(time=str(_time_)):
                self.assertStacksShow(self.p2s.timep(self.df, _time_, color=color, **extra), self.df, 'ts', 'category', length)

    # ── no color (default) ───────────────────────────────────────────────────

    def test_no_color_default(self):
        '''Default: color=None, all bars use the default data color.'''
        for _time_ in self.modes():
            with self.subTest(time=str(_time_)):
                _t_ = self.p2s.timep(self.df, _time_)
                self.assertColumnColors(_t_, self.df, 'ts', {b: self.p2s.colorTyped('data', 'default')
                                                            for b in perTimeBin(_t_, self.df, 'ts', pl.len())})

    def test_no_color_explicit_none(self):
        self.assertSameInBothModes({'color': None}, {})

    # ── categorical color ────────────────────────────────────────────────────

    def test_color_categorical_string_field(self):
        '''String column → color treated as categorical → stacked-bar coloring.'''
        self.assertStacksInBothModes('category')

    def test_color_cset_tuple(self):
        '''On a string field CSETp spells out what the field already is.'''
        self.assertSameInBothModes({'color': ('category', self.p2s.CSETp)}, {'color': 'category'})

    def test_color_categorical_with_stackedbar_style(self):
        '''Categorical color + STACKEDBARp → stacked bars: what the default draws.'''
        self.assertSameInBothModes({'color': 'category', 'style': self.p2s.STACKEDBARp}, {'color': 'category'})

    def test_color_categorical_agg_type_is_stacked(self):
        t = self.p2s.timep(self.df, 'ts', color='category')
        self.assertEqual(t._agg_type_, 'stacked')

    def test_a_column_has_one_remainder_in_the_other_colour(self):
        '''The values pooled into '(other)' and the shares too thin for their own column
        used to be two remainders, the second in a colour of its own; they are one segment,
        in the '(other)' colour, on top (PLANNING.md §5 C-histop-two-remainders).'''
        _other_ = self.p2s.color('(other)').lower()
        _named_ = {self.p2s.color(v).lower() for v in ('a', 'b', 'c', '(other)')}
        for _time_ in ('ts', ('ts', self.p2s.PT_dp)):
            for _legend_ in (False, 'bottom'):
                with self.subTest(time=str(_time_), legend=_legend_):
                    _cols_ = timepColumns(self.p2s.timep(_TWO_REMAINDERS_, _time_, color='clr', wxh=(128, 256), legend=_legend_))[2]
                    self.assertEqual(len(_cols_), 2)
                    for _x_, _segs_ in _cols_:
                        _fills_ = [f for _, f in _segs_]
                        self.assertEqual(len(_fills_), len(set(_fills_)), f'column at {_x_} draws a colour twice')
                        self.assertEqual(_fills_[-1], _other_, f'column at {_x_} does not end in its remainder')
                        self.assertTrue(set(_fills_) <= _named_, f'column at {_x_} has a remainder in a colour of its own')

    def test_colors_are_pooled_against_the_plot_not_the_canvas(self):
        '''Whether a value is drawn is decided on the plot it is drawn on.  'c' is 3.3px on
        the canvas but under 3px on a plot shortened by a legend underneath: it used to be
        listed in the legend and drawn nowhere.  Every value the legend names is drawn.'''
        for _time_ in ('ts', ('ts', self.p2s.PT_dp)):
            with self.subTest(time=str(_time_)):
                _t_ = self.p2s.timep(_TWO_REMAINDERS_, _time_, color='clr', wxh=(128, 256), legend='bottom')
                _drawn_ = {f for _, segs in timepColumns(_t_)[2] for _, f in segs}
                self.assertEqual([v for v, _ in _t_.legend_info.entries], ['a', 'b', '(other)'])
                for _v_, _c_ in _t_.legend_info.entries: self.assertIn(_c_.lower(), _drawn_, f'{_v_!r} is in the legend but not drawn')
                # without a legend the plot is tall enough, and 'c' keeps its own segment
                _t_ = self.p2s.timep(_TWO_REMAINDERS_, _time_, color='clr', wxh=(128, 256))
                self.assertIn(self.p2s.color('c').lower(), {f for _, segs in timepColumns(_t_)[2] for _, f in segs})

    # ── numeric color (spectrum) ─────────────────────────────────────────────

    def test_color_numeric_int_field(self):
        '''Int32 color field → spectrum coloring on whole bar (not stacked), by its sum.'''
        self.assertSpectrumInBothModes('value')

    def test_color_numeric_float_field(self):
        '''Float64 color field → spectrum coloring on whole bar (not stacked), by its sum.'''
        self.assertSpectrumInBothModes('numeric')

    def test_color_numeric_agg_type_is_simple(self):
        '''Numeric color stays in simple path.'''
        t = self.p2s.timep(self.df, 'ts', color='value')
        self.assertEqual(t._agg_type_, 'simple')

    def test_color_numeric_color_stat_in_df_agg(self):
        '''Numeric color → df_agg contains __color_stat__ column.'''
        t = self.p2s.timep(self.df, 'ts', color='value')
        self.assertIn('__color_stat__', t.df_agg.columns)

    def test_color_numeric_stat_range_computed(self):
        '''Numeric color → _color_stat_min_ and _color_stat_max_ are set.'''
        t = self.p2s.timep(self.df, 'ts', color='value')
        self.assertIsNotNone(t._color_stat_min_)
        self.assertIsNotNone(t._color_stat_max_)
        self.assertLessEqual(t._color_stat_min_, t._color_stat_max_)

    def test_color_numeric_cmagnitude_mean(self):
        '''(field, CMAGNITUDE_MEANp) → mean stat, spectrum coloring.'''
        t = self.p2s.timep(self.df, 'ts', color=('value', self.p2s.CMAGNITUDE_MEANp))
        self.assertEqual(t._agg_type_, 'simple')
        self.assertIn('__color_stat__', t.df_agg.columns)

    def test_color_numeric_cmagnitude_max(self):
        t = self.p2s.timep(self.df, 'ts', color=('value', self.p2s.CMAGNITUDE_MAXp))
        self.assertEqual(t._agg_type_, 'simple')

    def test_color_numeric_statistic_mean_tuple(self):
        '''(field, MEANp) also accepted.'''
        t = self.p2s.timep(self.df, 'ts', color=('value', self.p2s.MEANp))
        self.assertEqual(t._agg_type_, 'simple')

    def test_color_cset_numeric_is_stacked(self):
        '''(numeric-field, CSETp) → treated as categorical → stacked.'''
        t = self.p2s.timep(self.df, 'ts', color=('value', self.p2s.CSETp))
        self.assertEqual(t._agg_type_, 'stacked')

    def test_color_numeric_periodic(self):
        '''Numeric color in periodic mode → simple, spectrum.'''
        t = self.p2s.timep(self.df, ('ts', self.p2s.PT_mp), color='value')
        self.assertEqual(t._agg_type_, 'simple')
        self.assertIn('__color_stat__', t.df_agg.columns)

    # ── combined color + count ────────────────────────────────────────────────

    def test_color_categorical_with_numeric_count(self):
        '''Categorical color + numeric count field: each segment is its value's sum.'''
        self.assertStacksInBothModes('category', pl.col('value').sum(), count='value')

    def test_color_numeric_with_numeric_count(self):
        '''Numeric color + numeric count field: the height is one sum, the colour the other.'''
        self.assertSpectrumInBothModes('value', pl.col('numeric').sum(), count='numeric')


if __name__ == '__main__':
    unittest.main()
