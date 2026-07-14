import unittest
import polars as pl
from polars2svg import Polars2SVG

from random_dataframe import randomDataFrame

class Testxyp_label(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def test_labels(self, samples=1):
        # Use representative column type pairs (int×int, int×float, datetime×int, str×int)
        # rather than enumerating all 11×11 pairs — label/grid code branches on data type,
        # not on which specific column is chosen.
        _representative_pairs_ = [('a','b'), ('a','c'), ('g','a'), ('h','c'), ('j','a'), ('j','k')]
        for _sample_ in range(samples):
            df = randomDataFrame(10)
            for _col0_, _col1_ in _representative_pairs_:
                for _sz_ in [256, 128, 64, 32, 16]:
                    _xyp_ = self.p2s.xyp(df, _col0_, _col1_, wxh=(_sz_, _sz_))

    def test_SignXDivUnit(self):
        assert ('',  200,              1,               '')  == self.p2s.__SignXDivUnit__(200)
        assert ('',  234123,           1000.0,          'K') == self.p2s.__SignXDivUnit__(234_123)
        assert ('',  1233321.2314,     1000000.0,       'M') == self.p2s.__SignXDivUnit__(1_233_321.2314)
        assert ('-', 0.123,            1,               '')  == self.p2s.__SignXDivUnit__(-0.123)
        assert ('-', 9123898141,       1000000000.0,    'B') == self.p2s.__SignXDivUnit__(-9_123_898_141)
        assert ('',  6219123898141,    1000000000000.0, 'T') == self.p2s.__SignXDivUnit__(6_219_123_898_141)
        assert ('',  6219123898141.24, 1000000000000.0, 'T') == self.p2s.__SignXDivUnit__(6_219_123_898_141.24)

    def test_unittizeInt(self):
        assert '1.231K'   == self.p2s.unitizeInt(1_231)
        assert '100.0M'   == self.p2s.unitizeInt(100_213_112)
        assert '5.121B'   == self.p2s.unitizeInt(5_121_213_112)
        assert '-124.0K'  == self.p2s.unitizeInt(-123_999)
        assert '-121.67M' == self.p2s.unitizeInt(-121_669_123, num_of_digits=6)

    def __innerGridForTimeHelper__(self, df):
        _tiles_ = []
        for w in range(96,1600,64):
            _xyp_ = self.p2s.xyp(df, 'ts', 'value', dot_size=3.0, wxh=(w,128))
            _tiles_.append(_xyp_)

    def test_innerGridForTime_hours(self):
        df  = pl.DataFrame({
            'ts':['2026-01-02 00:00:00',
                  '2026-01-02 04:00:00',
                  '2026-01-02 08:00:00',
                  '2026-01-02 12:00:00'],
            'value':[1, 2, 3, 4]
        }).with_columns(pl.col('ts').str.to_datetime())
        self.__innerGridForTimeHelper__(df)

    def test_innerGridForTime_days(self):
        df  = pl.DataFrame({
            'ts':['2026-01-02 00:00:00',
                  '2026-01-03 04:00:00',
                  '2026-01-05 08:00:00',
                  '2026-01-12 12:00:00'],
            'value':[1, 2, 3, 4]
        }).with_columns(pl.col('ts').str.to_datetime())
        self.__innerGridForTimeHelper__(df)

    def test_innerGridForTime_months(self):
        df  = pl.DataFrame({
            'ts':['2026-01-02 00:00:00',
                  '2026-02-03 04:00:00',
                  '2026-06-05 08:00:00',
                  '2026-12-12 12:00:00'],
            'value':[1, 2, 3, 4]
        }).with_columns(pl.col('ts').str.to_datetime())
        self.__innerGridForTimeHelper__(df)

    def test_innerGridForTime_years(self):
        df  = pl.DataFrame({
            'ts':['2006-01-02 00:00:00',
                  '2010-02-03 04:00:00',
                  '2021-06-05 08:00:00',
                  '2026-12-12 12:00:00'],
            'value':[1, 2, 3, 4]
        }).with_columns(pl.col('ts').str.to_datetime())
        self.__innerGridForTimeHelper__(df)

if __name__ == '__main__':
    unittest.main()
