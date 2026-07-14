import unittest
import polars as pl
import datetime
import random
from polars2svg import Polars2SVG, TField

class Testp2s_tfields(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def test_timeLinearConversions(self):
        for _column_name_ in ['a', 'basic', 'with|pipe', 'under_score']:
            for _enum_ in self.p2s.TimeLinearTypeP:
                _t_field_ = self.p2s.tField(_column_name_, _enum_)
                _c_, _e_  = self.p2s.tFieldTuple(_t_field_)
                assert _c_ == _column_name_
                assert _e_ == _enum_

    def test_timePeriodicConversions(self):
        for _column_name_ in ['a', 'basic', 'with|pipe', 'under_score']:
            for _enum_ in self.p2s.TimePeriodicTypeP:
                _t_field_ = self.p2s.tField(_column_name_, _enum_)
                _c_, _e_  = self.p2s.tFieldTuple(_t_field_)
                assert _c_ == _column_name_
                assert _e_ == _enum_

    def test_polarsOperationsForEnum_forDateTime(self):
        df    = pl.DataFrame({'ts':['2025-11-08 22:24:01']}).with_columns(pl.col('ts').str.to_datetime())
        _ops_ = []
        for _enum_ in self.p2s.TimeLinearTypeP:   _ops_.append(self.p2s.polarsOperationForEnum('ts', _enum_).alias(self.p2s.tField('ts', _enum_)))
        for _enum_ in self.p2s.TimePeriodicTypeP: _ops_.append(self.p2s.polarsOperationForEnum('ts', _enum_).alias(self.p2s.tField('ts', _enum_)))
        df    = df.with_columns(_ops_)
        assert df["ts|Yp"][0]           == datetime.datetime(2025,  1, 1,  0,  0, 0)
        assert df["ts|Y_Qp"][0]         == datetime.datetime(2025, 10, 1,  0,  0, 0)
        assert df["ts|Y_mp"][0]         == datetime.datetime(2025, 11, 1,  0,  0, 0)
        assert df["ts|Y_m_dp"][0]       == datetime.datetime(2025, 11, 8,  0,  0, 0)
        assert df["ts|Y_m_d_Hp"][0]     == datetime.datetime(2025, 11, 8, 22,  0, 0)
        assert df["ts|Y_m_d_H_Mp"][0]   == datetime.datetime(2025, 11, 8, 22, 24, 0)
        assert df["ts|Y_m_d_H_M_Sp"][0] == datetime.datetime(2025, 11, 8, 22, 24, 1)
        assert df["ts|Qp"][0]           == 4     # validated / self-evident
        assert df["ts|mp"][0]           == 11    # validated / self-evident
        assert df["ts|m_dp"][0]         == 313   # ... not validate / what the day would be if not a leap year
        assert df["ts|m_d_Hp"][0]       == 7534  # validated 312*24 + 22
        assert df["ts|DoYp"][0]         == 312   # validated
        assert df["ts|DoWp"][0]         == 6     # validated / 6 == Saturday
        assert df["ts|DoW_Hp"][0]      == 166    # validated / 6*24 + 22
        assert df["ts|DoW_H_Mp"][0]    == 9984   # validated / 6*24*60 + 24*60 + 22*60
        assert df["ts|dp"][0]          == 8      # validated / self-evident
        assert df["ts|d_Hp"][0]        == 214    # validated 8*24 + 22 
        assert df["ts|d_H_Mp"][0]      == 12864  # validated / 8*24*60 + 22*60 + 24
        assert df["ts|Hp"][0]          == 22     # validated / self-evident
        assert df["ts|H_Mp"][0]        == 1344   # validated / 22*60 + 24
        assert df["ts|H_M_Sp"][0]      == 80641  # validated / 22*60*60 + 24*60 + 1
        assert df["ts|Mp"][0]          == 24     # validated / self-evident
        assert df["ts|M_Sp"][0]        == 1441   # validated / 24*60 + 1
        assert df["ts|Sp"][0]          == 1      # validated / self-evident

    def test_humanReadablePeriodicTimeDelta(self):
        for _enum_ in self.p2s.TimePeriodicTypeP:
            _range_ = self.p2s.timePeriodicRange(_enum_)
            for _diff_ in range(_range_[0], _range_[1]+1):
                self.p2s.humanReadablePeriodicTimeDelta(_diff_, _enum_)

    def test_timePeriodicHumanReadable(self):
        _lu_  = {'ts':['2026-03-01 10:31:59']}
        df    = pl.DataFrame(_lu_).with_columns(pl.col('ts').str.to_datetime())
        _ops_ = []
        for _enum_ in self.p2s.TimePeriodicTypeP: 
            _tfield_ = self.p2s.tField('ts', _enum_)
            _ops_.append(self.p2s.polarsOperationForEnum('ts', _enum_).alias(_tfield_))
        df = df.with_columns(_ops_)
        _answers_ = {
            self.p2s.TimePeriodicTypeP.PT_Qp:       'q1',
            self.p2s.TimePeriodicTypeP.PT_mp:       'mar',
            self.p2s.TimePeriodicTypeP.PT_m_dp:     'mar-01',
            self.p2s.TimePeriodicTypeP.PT_m_d_Hp:   'mar-01 10h',
            self.p2s.TimePeriodicTypeP.PT_DoYp:     '60',
            self.p2s.TimePeriodicTypeP.PT_DoWp:     'sun',
            self.p2s.TimePeriodicTypeP.PT_DoW_Hp:   'sun 10h',
            self.p2s.TimePeriodicTypeP.PT_DoW_H_Mp: 'sun 10:31',
            self.p2s.TimePeriodicTypeP.PT_dp:       '01d',
            self.p2s.TimePeriodicTypeP.PT_d_Hp:     '01 10h',
            self.p2s.TimePeriodicTypeP.PT_d_H_Mp:   '01 10:31',
            self.p2s.TimePeriodicTypeP.PT_Hp:       '10h',
            self.p2s.TimePeriodicTypeP.PT_H_Mp:     '10:31m',
            self.p2s.TimePeriodicTypeP.PT_H_M_Sp:   '10:31:59',
            self.p2s.TimePeriodicTypeP.PT_Mp:       '31m',
            self.p2s.TimePeriodicTypeP.PT_M_Sp:     '31:59s',
            self.p2s.TimePeriodicTypeP.PT_Sp:       '59s',
        }
        for _k_, _v_ in _answers_.items():
            assert self.p2s.timePeriodicHumanReadable(df[self.p2s.tField('ts', _k_)][0], _k_) == _v_

    def test_timePeriodicHumanReadable_nye(self):
        _lu_      = {'ts':['2026-12-31 10:31:59',   # row 0
                           '2025-12-31 09:31:59',   # row 1
                           '2024-12-31 23:31:59',   # row 2 <-- leap year
                           '2023-12-31 06:31:59',   # row 3
                           '2022-12-31 12:31:59']}  # row 4
        df        = pl.DataFrame(_lu_).with_columns(pl.col('ts').str.to_datetime())
        _ops_     = []
        _tfields_ = []
        for _enum_ in [self.p2s.TimePeriodicTypeP.PT_m_dp, self.p2s.TimePeriodicTypeP.PT_m_d_Hp, self.p2s.TimePeriodicTypeP.PT_DoYp,]:
            _tfield_ = self.p2s.tField('ts', _enum_)
            _tfields_.append(_tfield_)
            _ops_.append(self.p2s.polarsOperationForEnum('ts', _enum_).alias(_tfield_))
        df = df.with_columns(_ops_)

        _answers_ = {}
        _answers_[(0, "ts|m_dp")]   = "dec-31"
        _answers_[(1, "ts|m_dp")]   = "dec-31"
        _answers_[(2, "ts|m_dp")]   = "dec-31"
        _answers_[(3, "ts|m_dp")]   = "dec-31"
        _answers_[(4, "ts|m_dp")]   = "dec-31"

        _answers_[(0, "ts|m_d_Hp")] = "dec-31 10h"
        _answers_[(1, "ts|m_d_Hp")] = "dec-31 9h"
        _answers_[(2, "ts|m_d_Hp")] = "dec-31 23h"
        _answers_[(3, "ts|m_d_Hp")] = "dec-31 6h"
        _answers_[(4, "ts|m_d_Hp")] = "dec-31 12h"

        _answers_[(0, "ts|DoYp")]   = "365"
        _answers_[(1, "ts|DoYp")]   = "365"
        _answers_[(2, "ts|DoYp")]   = "366" # leap year
        _answers_[(3, "ts|DoYp")]   = "365"
        _answers_[(4, "ts|DoYp")]   = "365"

        for _row_ in range(len(df)):
            for _tfield_ in _tfields_:
                _column_, _enum_ = self.p2s.tFieldTuple(_tfield_)
                _value_          = df[_tfield_][_row_]
                _str_            = self.p2s.timePeriodicHumanReadable(_value_, _enum_)
                assert _answers_[(_row_, _tfield_)] == _str_

    def test_timePeriodicRanges(self):
        _lu_ = {'ts':[], 'value':[] }
        n    = 10_000
        for i in range(n):
            _year_    = random.randint(1000, 3000)
            _month_   = random.randint(1, 12)
            _max_day_ = self.p2s.__daysInMonth__(_year_, _month_)
            _day_     = random.randint(1, _max_day_)
            _hour_    = random.randint(0, 23)
            _minute_  = random.randint(0, 59)
            _second_  = random.randint(0, 59)
            _as_str_  = f'{_year_:04}-{_month_:02}-{_day_:02} {_hour_:02}:{_minute_:02}:{_second_:02}'
            _lu_['ts'].append(_as_str_)
            _lu_['value'].append(random.random()*1000)
        df = pl.DataFrame(_lu_).with_columns(pl.col('ts').str.to_datetime())

        _tiles_ = []
        for _enum_ in self.p2s.TimePeriodicTypeP:
            _xyp_ = self.p2s.xyp(df, self.p2s.tField('ts',_enum_), 'value', dot_size=3.0, wxh=(256, 128), opacity=0.8, x_distributions=self.p2s.ROW_COUNTp)
            _min_ = _xyp_.df_flat['__x__'].min()
            _max_ = _xyp_.df_flat['__x__'].max()
            if self.p2s.periodic_ranges[_enum_][0] > _min_: raise Exception(f'test_timePeriodicRanges() - New minimum ({_min_}) seen for {_enum_}')
            if self.p2s.periodic_ranges[_enum_][1] < _max_: raise Exception(f'test_timePeriodicRanges() - New maximum ({_max_}) seen for {_enum_}')
            _xyp_ = self.p2s.xyp(df, self.p2s.tField('ts',_enum_), 'value', dot_size=3.0, wxh=(256, 128), opacity=0.8, 
                                 x_distributions=self.p2s.ROW_COUNTp, x_range=self.p2s.timePeriodicRange(_enum_))
            _tiles_.append(_xyp_)

class TestTField(unittest.TestCase):
    '''Unit tests for the typed TField replacement for the magic 'column|suffix' string.

    TField subclasses str so its value *is* the legacy alias -- these tests lock
    that equivalence in (attrs, isinstance, equality/hash, immutability) so the
    typed and legacy forms stay interchangeable everywhere a t-field is accepted.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def test_attrs(self):
        tf = self.p2s.tField('ts', self.p2s.PT_mp)
        self.assertEqual(tf.column, 'ts')
        self.assertEqual(tf.transform, self.p2s.PT_mp)
        self.assertEqual(tf.alias, 'ts|mp')
        self.assertEqual(str(tf), 'ts|mp')

    def test_is_a_str_subclass(self):
        tf = self.p2s.tField('ts', self.p2s.PT_mp)
        self.assertIsInstance(tf, str)
        self.assertIsInstance(tf, TField)

    def test_equality_and_hash_match_legacy_string(self):
        tf = self.p2s.tField('ts', self.p2s.PT_mp)
        self.assertEqual(tf, 'ts|mp')
        self.assertEqual(hash(tf), hash('ts|mp'))
        _d_ = {tf: 1}
        self.assertEqual(_d_['ts|mp'], 1)   # legacy string looks up the TField's slot
        self.assertIn('ts|mp', {tf})

    def test_immutable(self):
        tf = self.p2s.tField('ts', self.p2s.PT_mp)
        with self.assertRaises(AttributeError):
            tf.column = 'other'
        with self.assertRaises(AttributeError):
            tf.transform = self.p2s.PT_Sp

    def test_repr(self):
        tf = self.p2s.tField('ts', self.p2s.PT_mp)
        self.assertIn('ts', repr(tf))
        self.assertIn('PT_mp', repr(tf))

    def test_unknown_enum_raises(self):
        with self.assertRaises(Exception):
            self.p2s.tField('ts', 'not-an-enum')

    def test_direct_construction_matches_tField_factory(self):
        self.assertEqual(TField('ts', self.p2s.PT_mp), self.p2s.tField('ts', self.p2s.PT_mp))

    def test_isTField_truth_table(self):
        tf = self.p2s.tField('price', self.p2s.PT_mp)
        df_with_literal    = pl.DataFrame({'price|mp': [1, 2, 3]})
        df_without_literal = pl.DataFrame({'price': [1, 2, 3]})

        self.assertTrue(self.p2s.isTField(tf))                                  # TField always a t-field
        self.assertTrue(self.p2s.isTField(tf, df=df_with_literal))              # ...even if alias collides
        self.assertTrue(self.p2s.isTField('price|mp'))                          # legacy, no df -> nothing to hijack
        self.assertFalse(self.p2s.isTField('price|mp', df=df_with_literal))     # legacy, literal column wins
        self.assertTrue(self.p2s.isTField('price|mp', df=df_without_literal))   # legacy, literal absent -> t-field
        self.assertFalse(self.p2s.isTField('not_a_tfield'))
        self.assertFalse(self.p2s.isTField(123))                                # non-str, non-TField


if __name__ == '__main__':
    unittest.main()
