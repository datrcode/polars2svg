import unittest
import polars as pl
import datetime
import random
from polars2svg import Polars2SVG


class Testp2s(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def test_humanReadablTimeDeltas(self):
        _seconds_ = 1.0
        while _seconds_ < 1_000_000_000:
            _td_  = datetime.timedelta(seconds=_seconds_)
            _str_ = self.p2s.humanReadableTimeDelta(_td_)
            _seconds_ *= 1.1


class TestIsTemplate(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def _simple_df(self):
        return pl.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

    def _simple_df_ts(self):
        base = datetime.date(2024, 1, 1)
        return pl.DataFrame({'ts': [base + datetime.timedelta(days=i * 30) for i in range(4)]})

    def _chord_df(self):
        return pl.DataFrame({'fm': ['a', 'b'], 'to': ['b', 'a']})

    def test_histop_is_template(self):
        hp = self.p2s.histop(self._simple_df(), bin_by='a')
        self.assertTrue(self.p2s.isTemplate(hp))

    def test_timep_is_template(self):
        tp = self.p2s.timep(self._simple_df_ts(), 'ts')
        self.assertTrue(self.p2s.isTemplate(tp))

    def test_xyp_is_template(self):
        xy = self.p2s.xyp(self._simple_df(), 'a', 'b')
        self.assertTrue(self.p2s.isTemplate(xy))

    def test_linkp_is_template(self):
        df = pl.DataFrame({'src': ['x', 'y'], 'dst': ['y', 'x']})
        lp = self.p2s.linkp(df=df, relationships=[('src', 'dst')])
        self.assertTrue(self.p2s.isTemplate(lp))

    def test_chordp_is_template(self):
        cp = self.p2s.chordp(df=self._chord_df(), relationships=[('fm', 'to')])
        self.assertTrue(self.p2s.isTemplate(cp))

    def test_none_is_not_template(self):
        self.assertFalse(self.p2s.isTemplate(None))

    def test_string_is_not_template(self):
        self.assertFalse(self.p2s.isTemplate('not a template'))

    def test_int_is_not_template(self):
        self.assertFalse(self.p2s.isTemplate(42))

    def test_dict_is_not_template(self):
        self.assertFalse(self.p2s.isTemplate({}))


class TestColumnInDataFrame(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()
        self.df = pl.DataFrame({'a': [1, 2], 'b': ['x', 'y']})

    def test_existing_column_returns_true(self):
        self.assertTrue(self.p2s.columnInDataFrame('a', self.df))

    def test_missing_column_returns_false(self):
        self.assertFalse(self.p2s.columnInDataFrame('z', self.df))

    def test_tfield_resolves_base_column(self):
        df = pl.DataFrame({'ts': [datetime.date(2024, 1, 1), datetime.date(2024, 6, 1)]})
        tf = self.p2s.tField('ts', self.p2s.LT_Y_mp)
        self.assertTrue(self.p2s.columnInDataFrame(tf, df))

    def test_tfield_missing_base_column_returns_false(self):
        tf = self.p2s.tField('missing', self.p2s.LT_Y_mp)
        self.assertFalse(self.p2s.columnInDataFrame(tf, self.df))


class TestIsTField(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def test_tfield_string_is_tfield(self):
        tf = self.p2s.tField('ts', self.p2s.LT_Y_mp)
        self.assertTrue(self.p2s.isTField(tf))

    def test_plain_column_name_is_not_tfield(self):
        self.assertFalse(self.p2s.isTField('mycolumn'))

    def test_pipe_without_valid_suffix_is_not_tfield(self):
        self.assertFalse(self.p2s.isTField('col|unknownsuffix'))

    def test_empty_string_is_not_tfield(self):
        self.assertFalse(self.p2s.isTField(''))

    def test_periodic_tfield_is_tfield(self):
        tf = self.p2s.tField('ts', self.p2s.PT_Hp)
        self.assertTrue(self.p2s.isTField(tf))


class TestTFieldAccepts(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def test_linear_tfield_accepts_date_and_datetime(self):
        tf = self.p2s.tField('ts', self.p2s.LT_Y_mp)
        accepted = self.p2s.tFieldAccepts(tf)
        self.assertIn(pl.Date, accepted)
        self.assertIn(pl.Datetime, accepted)

    def test_periodic_tfield_accepts_date_and_datetime(self):
        tf = self.p2s.tField('ts', self.p2s.PT_Hp)
        accepted = self.p2s.tFieldAccepts(tf)
        self.assertIn(pl.Date, accepted)
        self.assertIn(pl.Datetime, accepted)


class TestColumnTypeChecks(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()
        self.df = pl.DataFrame({
            'i':  pl.Series([1, 2, 3], dtype=pl.Int64),
            'f':  pl.Series([1.0, 2.0, 3.0], dtype=pl.Float64),
            's':  pl.Series(['a', 'b', 'c'], dtype=pl.String),
            'dt': pl.Series([datetime.date(2024, 1, i+1) for i in range(3)], dtype=pl.Date),
            'tm': pl.Series([datetime.time(i, 0) for i in range(3)], dtype=pl.Time),
            'dttm': pl.Series(
                [datetime.datetime(2024, 1, i+1) for i in range(3)],
                dtype=pl.Datetime,
            ),
        })

    def test_numeric_column_true_for_int(self):
        self.assertTrue(self.p2s.numericColumn(self.df, 'i'))

    def test_numeric_column_true_for_float(self):
        self.assertTrue(self.p2s.numericColumn(self.df, 'f'))

    def test_numeric_column_false_for_string(self):
        self.assertFalse(self.p2s.numericColumn(self.df, 's'))

    def test_numeric_column_false_for_date(self):
        self.assertFalse(self.p2s.numericColumn(self.df, 'dt'))

    def test_date_column_true_for_date(self):
        self.assertTrue(self.p2s.dateColumn(self.df, 'dt'))

    def test_date_column_false_for_datetime(self):
        self.assertFalse(self.p2s.dateColumn(self.df, 'dttm'))

    def test_date_column_false_for_int(self):
        self.assertFalse(self.p2s.dateColumn(self.df, 'i'))

    def test_time_column_true_for_time(self):
        self.assertTrue(self.p2s.timeColumn(self.df, 'tm'))

    def test_time_column_false_for_date(self):
        self.assertFalse(self.p2s.timeColumn(self.df, 'dt'))

    def test_datetime_column_true_for_datetime(self):
        self.assertTrue(self.p2s.dateTimeColumn(self.df, 'dttm'))

    def test_datetime_column_false_for_date(self):
        self.assertFalse(self.p2s.dateTimeColumn(self.df, 'dt'))

    def test_datetime_column_false_for_int(self):
        self.assertFalse(self.p2s.dateTimeColumn(self.df, 'i'))


if __name__ == '__main__':
    unittest.main()
