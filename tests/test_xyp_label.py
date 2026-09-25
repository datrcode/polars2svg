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
                    self.p2s.xyp(df, _col0_, _col1_, wxh=(_sz_, _sz_))

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
        assert '100.2M'   == self.p2s.unitizeInt(100_213_112)    # was '100.0M' -- round(x, 0) is a float
        assert '5.121B'   == self.p2s.unitizeInt(5_121_213_112)
        assert '-124K'    == self.p2s.unitizeInt(-123_999)
        assert '-121.67M' == self.p2s.unitizeInt(-121_669_123, num_of_digits=6)

    def test_unitizeInt_regressions(self):
        # The old loop stopped extending when one more decimal place did not change the
        # string, so a zero first decimal truncated everything after it.
        assert '2.024K'   == self.p2s.unitizeInt(2_024)            # was '2.0K'
        assert '2.02K'    == self.p2s.unitizeInt(2_024, num_of_digits=4)
        # round(x, 0) returns a float, so whole numbers grew a '.0'.
        assert '443'      == self.p2s.unitizeInt(443)              # was '443.0'
        assert '2K'       == self.p2s.unitizeInt(2_000)            # was '2.0K'
        # No num_of_digits reached this one: 5 gave '45.61M' and 4 gave '46.0M'.
        assert '45.6M'    == self.p2s.unitizeInt(45_612_314, num_of_digits=4)
        # Small values keep their significant digits rather than collapsing to zero.
        assert '0.000123' == self.p2s.unitizeInt(0.000123, num_of_digits=4)  # was '0.0'
        assert '12.5'     == self.p2s.unitizeInt(12.5, num_of_digits=4)      # was '12.0'
        # Rounding up to 1000 moves to the next unit.
        assert '1M'       == self.p2s.unitizeInt(999_999)
        assert '0'        == self.p2s.unitizeInt(0)
        assert 'nan'      == self.p2s.unitizeInt(float('nan'))

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
