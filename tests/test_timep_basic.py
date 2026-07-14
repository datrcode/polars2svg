import unittest
import polars as pl
from polars2svg import Polars2SVG
from timep_dataframes import makeTimeDf, makeDateDf
from svg_test_utils import assert_valid_svg, capture_log_warnings


class TestTimepBasic(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def test_auto_detect_time_field(self):
        '''No time arg: timep auto-detects the first datetime column.'''
        df = makeTimeDf(n=50, year=(2020, 2025), month=(1, 12))
        self.p2s.timep(df)

    def test_explicit_time_field_positional_str(self):
        df = makeTimeDf(n=50, year=(2020, 2025), month=(1, 12))
        self.p2s.timep(df, 'ts')

    def test_explicit_time_field_keyword(self):
        df = makeTimeDf(n=50, year=(2020, 2025), month=(1, 12))
        self.p2s.timep(df, time='ts')

    def test_df_as_keyword_arg(self):
        df = makeTimeDf(n=50, year=(2020, 2025), month=(1, 12))
        self.p2s.timep(df=df, time='ts')

    def test_date_column(self):
        '''pl.Date column (no sub-day component) is supported.'''
        df = makeDateDf(n=50, year=(2020, 2025), month=(1, 12))
        self.p2s.timep(df, 'dt')

    def test_date_column_auto_detect(self):
        '''Auto-detection also works for pl.Date columns.'''
        df = makeDateDf(n=50, year=(2020, 2025), month=(1, 12))
        self.p2s.timep(df)

    def test_single_row(self):
        df = makeTimeDf(n=1, year=2023, month=6, day=15)
        self.p2s.timep(df, 'ts')

    def test_repr_svg_returns_valid_svg_string(self):
        df = makeTimeDf(n=50, year=(2022, 2024), month=(1, 12))
        t  = self.p2s.timep(df, 'ts')
        assert_valid_svg(self, t._repr_svg_())

    def test_various_wxh(self):
        df = makeTimeDf(n=100, year=(2020, 2024), month=(1, 12))
        for w, h in [(128, 64), (256, 128), (512, 256), (1024, 512)]:
            self.p2s.timep(df, 'ts', wxh=(w, h))

    def test_periodic_time_field_positional_tuple(self):
        df = makeTimeDf(n=100, year=(2020, 2024), month=(1, 12))
        self.p2s.timep(df, ('ts', self.p2s.PT_mp))

    def test_linear_time_field_explicit_enum(self):
        df = makeTimeDf(n=100, year=(2020, 2024), month=(1, 12))
        self.p2s.timep(df, ('ts', self.p2s.LT_Y_mp))


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
        self.p2s.timep(self.df, 'ts', wxh=(512, 256))


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
