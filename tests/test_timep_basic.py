import re
import unittest
import polars as pl
from polars2svg import Polars2SVG
from timep_dataframes import makeTimeDf, makeDateDf
from svg_test_utils import assert_valid_svg, capture_log_warnings, normalize_svg


class TestTimepBasic(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    #
    # assertBarsSumToRows() -- the bars are the rows: the tallest bar spans the plot and
    # counts the number at the top of the count axis, every bar is a whole number of
    # rows on that scale, and the bars add up to every row.  Returns the bar heights.
    #
    def assertBarsSumToRows(self, t, n_rows: int) -> list:
        _svg_ = t._repr_svg_()
        _fx_, _fy_, _fw_, _fh_ = (float(v) for v in re.search(
            r'<rect x="([\d.]+)" y="([\d.]+)" width="([\d.]+)" height="([\d.]+)" stroke="[^"]*" fill="none" stroke-width="0.5" />', _svg_).groups())
        _w_, _h_ = t.wxh
        _bars_ = [float(hh) for x, y, w, hh in re.findall(
            r'<rect x="([\d.]+)" y="([\d.]+)" width="([\d.]+)" height="([\d.]+)" fill="#[0-9a-fA-F]{6}" />', _svg_)
            if (x, y, w, hh) != ('0', '0', str(_w_), str(_h_))]
        _axis_ = sorted((float(y), txt) for x, y, txt in re.findall(r'<text x="([\d.]+)"[^>]* y="([\d.]+)"[^>]*>([^<]*)</text>', _svg_)
                        if float(x) == _fx_ - 2)
        self.assertGreater(len(_bars_), 0, 'no bars drawn')
        self.assertAlmostEqual(max(_bars_), _fh_, delta=0.1, msg='the tallest bar does not span the plot')
        _top_  = int(_axis_[0][1])                                  # the count axis's top label
        _rows_ = [hh / _fh_ * _top_ for hh in _bars_]
        for _r_ in _rows_:
            self.assertAlmostEqual(_r_, round(_r_), delta=0.05, msg='a bar is not a whole number of rows')
        self.assertEqual(sum(round(_r_) for _r_ in _rows_), n_rows, 'the bars do not add up to the rows')
        return _bars_

    # Two calls that mean the same thing render identically
    def assertSameRender(self, a, b) -> None:
        self.assertEqual(normalize_svg(a._repr_svg_()), normalize_svg(b._repr_svg_()))

    def test_auto_detect_time_field(self):
        '''No time arg: timep auto-detects the first datetime column.'''
        df = makeTimeDf(n=50, year=(2020, 2025), month=(1, 12), seed=1)
        self.assertSameRender(self.p2s.timep(df), self.p2s.timep(df, 'ts'))

    def test_explicit_time_field_positional_str(self):
        df = makeTimeDf(n=50, year=(2020, 2025), month=(1, 12), seed=1)
        self.assertBarsSumToRows(self.p2s.timep(df, 'ts'), 50)

    def test_explicit_time_field_keyword(self):
        df = makeTimeDf(n=50, year=(2020, 2025), month=(1, 12), seed=1)
        self.assertSameRender(self.p2s.timep(df, time='ts'), self.p2s.timep(df, 'ts'))

    def test_df_as_keyword_arg(self):
        df = makeTimeDf(n=50, year=(2020, 2025), month=(1, 12), seed=1)
        self.assertSameRender(self.p2s.timep(df=df, time='ts'), self.p2s.timep(df, 'ts'))

    def test_date_column(self):
        '''pl.Date column (no sub-day component) is supported: it draws what the same
        midnights in a Datetime column do.'''
        df = makeDateDf(n=50, year=(2020, 2025), month=(1, 12), seed=2)
        _t_ = self.p2s.timep(df, 'dt')
        self.assertBarsSumToRows(_t_, 50)
        self.assertSameRender(_t_, self.p2s.timep(df.with_columns(pl.col('dt').cast(pl.Datetime)), 'dt'))

    def test_date_column_auto_detect(self):
        '''Auto-detection also works for pl.Date columns.'''
        df = makeDateDf(n=50, year=(2020, 2025), month=(1, 12), seed=2)
        self.assertSameRender(self.p2s.timep(df), self.p2s.timep(df, 'dt'))

    def test_single_row(self):
        df = makeTimeDf(n=1, year=2023, month=6, day=15)
        self.assertEqual(len(self.assertBarsSumToRows(self.p2s.timep(df, 'ts'), 1)), 1)

    def test_repr_svg_returns_valid_svg_string(self):
        df = makeTimeDf(n=50, year=(2022, 2024), month=(1, 12))
        t  = self.p2s.timep(df, 'ts')
        assert_valid_svg(self, t._repr_svg_())

    def test_various_wxh(self):
        df = makeTimeDf(n=100, year=(2020, 2024), month=(1, 12), seed=3)
        for w, h in [(128, 64), (256, 128), (512, 256), (1024, 512)]:
            with self.subTest(wxh=(w, h)):
                _t_ = self.p2s.timep(df, 'ts', wxh=(w, h))
                self.assertIn(f'width="{w}" height="{h}"', _t_._repr_svg_())
                self.assertBarsSumToRows(_t_, 100)

    def test_periodic_time_field_positional_tuple(self):
        # The (column, enum) tuple is the tField; one bar per month the data has
        df = makeTimeDf(n=100, year=(2020, 2024), month=(1, 12), seed=3)
        _t_ = self.p2s.timep(df, ('ts', self.p2s.PT_mp))
        self.assertSameRender(_t_, self.p2s.timep(df, self.p2s.tField('ts', self.p2s.PT_mp)))
        self.assertEqual(len(self.assertBarsSumToRows(_t_, 100)), df['ts'].dt.month().n_unique())

    def test_linear_time_field_explicit_enum(self):
        # One bar per calendar month the data has
        df = makeTimeDf(n=100, year=(2020, 2024), month=(1, 12), seed=3)
        _t_ = self.p2s.timep(df, ('ts', self.p2s.LT_Y_mp))
        self.assertSameRender(_t_, self.p2s.timep(df, self.p2s.tField('ts', self.p2s.LT_Y_mp)))
        self.assertEqual(len(self.assertBarsSumToRows(_t_, 100)), df['ts'].dt.truncate('1mo').n_unique())


class TestTimepLazyExecution(unittest.TestCase):
    """use_lazy_execution honoured in Timep.__addColumnsToDataFrame__."""

    def setUp(self):
        self.p2s = Polars2SVG()
        self.df  = makeTimeDf(n=50, year=(2020, 2023), month=(1, 12))

    def test_periodic_lazy_false(self):
        """Periodic mode adds __time_bin__ via __addColumnsToDataFrame__; eager path must work."""
        t = self.p2s.timep(self.df, ('ts', self.p2s.PT_mp), use_lazy_execution=False)
        self.assertIn('<svg', t._repr_svg_())

    def test_periodic_lazy_true(self):
        t = self.p2s.timep(self.df, ('ts', self.p2s.PT_mp), use_lazy_execution=True)
        self.assertIn('<svg', t._repr_svg_())

    def test_count_tfield_lazy_false(self):
        """count as a tField triggers __addColumnsToDataFrame__; eager path must work."""
        tfield = self.p2s.tField('ts', self.p2s.LT_Y_mp)
        t = self.p2s.timep(self.df, 'ts', count=tfield, use_lazy_execution=False)
        self.assertIn('<svg', t._repr_svg_())

    def test_count_tfield_lazy_true(self):
        tfield = self.p2s.tField('ts', self.p2s.LT_Y_mp)
        t = self.p2s.timep(self.df, 'ts', count=tfield, use_lazy_execution=True)
        self.assertIn('<svg', t._repr_svg_())

    def test_color_tfield_lazy_false(self):
        """color as a tField triggers __addColumnsToDataFrame__; eager path must work."""
        tfield = self.p2s.tField('ts', self.p2s.LT_Y_mp)
        t = self.p2s.timep(self.df, 'ts', color=tfield, use_lazy_execution=False)
        self.assertIn('<svg', t._repr_svg_())

    def test_color_tfield_lazy_true(self):
        tfield = self.p2s.tField('ts', self.p2s.LT_Y_mp)
        t = self.p2s.timep(self.df, 'ts', color=tfield, use_lazy_execution=True)
        self.assertIn('<svg', t._repr_svg_())

    def test_no_tfields_lazy_false_is_noop(self):
        """Without tFields and in linear mode, __addColumnsToDataFrame__ does nothing."""
        t = self.p2s.timep(self.df, 'ts', use_lazy_execution=False)
        self.assertIn('<svg', t._repr_svg_())


class TestTimepWxhValidation(unittest.TestCase):
    """wxh accepts any 2-sequence of numbers and coerces floats to int
    (shared Polars2SVG.normalizeWxh); see tests/test_wxh_normalization.py."""

    def setUp(self):
        self.p2s = Polars2SVG()
        self.df  = makeTimeDf(n=50, year=(2020, 2023), month=(1, 12))

    def test_wxh_float_coerced(self):
        t = self.p2s.timep(self.df, 'ts', wxh=(512.9, 256))
        self.assertEqual(t.wxh, (512, 256))

    def test_wxh_list_coerced_to_tuple(self):
        t = self.p2s.timep(self.df, 'ts', wxh=[512, 256])
        self.assertEqual(t.wxh, (512, 256))

    def test_wxh_bad_still_raises(self):
        with self.assertRaises(ValueError):
            self.p2s.timep(self.df, 'ts', wxh=(512, None))

    def test_wxh_int_int_ok(self):
        self.assertEqual(self.p2s.timep(self.df, 'ts', wxh=(512, 256)).wxh, (512, 256))


class TestTimepSmSharedWarnings(unittest.TestCase):
    """Unsupported SM_* values log a warning; supported values do not."""

    def setUp(self):
        self.p2s = Polars2SVG()
        self.df  = makeTimeDf(n=100, year=(2020, 2023), month=(1, 12))

    def test_sm_y_warns(self):
        records = capture_log_warnings(
            lambda: self.p2s.timep(self.df, 'ts', sm_shared={self.p2s.SM_Y})
        )
        self.assertTrue(any('SM_Y' in r.getMessage() or 'sm_shared' in r.getMessage()
                            for r in records))

    def test_sm_x_no_warning(self):
        """SM_X is supported by Timep — no warning expected."""
        records = capture_log_warnings(
            lambda: self.p2s.timep(self.df, 'ts', sm_shared={self.p2s.SM_X})
        )
        self.assertEqual([r for r in records if 'sm_shared' in r.getMessage()], [])

    def test_sm_count_no_warning(self):
        records = capture_log_warnings(
            lambda: self.p2s.timep(self.df, 'ts', sm_shared={self.p2s.SM_COUNT})
        )
        self.assertEqual([r for r in records if 'sm_shared' in r.getMessage()], [])

    def test_sm_color_no_warning(self):
        records = capture_log_warnings(
            lambda: self.p2s.timep(self.df, 'ts', sm_shared={self.p2s.SM_COLOR})
        )
        self.assertEqual([r for r in records if 'sm_shared' in r.getMessage()], [])


if __name__ == '__main__':
    unittest.main()
