"""Regression tests: verify no exceptions are raised across xyp parameter combinations.
No assertions are made — pass = no crash."""
import unittest
import polars as pl
from polars2svg import Polars2SVG

from random_dataframe import randomDataFrame

class Testxyp_exhaustive(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def test_exhaustiveSingles(self, samples=1):
        for _sample_ in range(samples):
            for n in [1]:
                df = randomDataFrame(n, na_probability=0.0)
                for _col0_ in df.columns:
                    for _col1_ in df.columns:
                        _xyp_ = self.p2s.xyp(df, _col0_, _col1_)
                        _xyp_ = self.p2s.xyp(df, (_col0_, self.p2s.SETp),    _col1_)
                        _xyp_ = self.p2s.xyp(df, (_col0_, self.p2s.SCALARp), _col1_)
                        _xyp_ = self.p2s.xyp(df, _col0_, (_col1_, self.p2s.SETp))
                        _xyp_ = self.p2s.xyp(df, _col0_, (_col1_, self.p2s.SCALARp))
                        _xyp_ = self.p2s.xyp(df, (_col0_, self.p2s.SETp),    (_col1_, self.p2s.SETp))
                        _xyp_ = self.p2s.xyp(df, (_col0_, self.p2s.SCALARp), (_col1_, self.p2s.SETp))
                        _xyp_ = self.p2s.xyp(df, (_col0_, self.p2s.SCALARp), (_col1_, self.p2s.SCALARp))
                        _xyp_ = self.p2s.xyp(df, (_col0_, self.p2s.SETp),    (_col1_, self.p2s.SCALARp))


    def test_exhaustiveSmalls(self, samples=1):
        for _sample_ in range(samples):
            for n in [2,10]:
                df = randomDataFrame(n, na_probability=0.0)
                for _col0_ in df.columns:
                    for _col1_ in df.columns:
                        _xyp_ = self.p2s.xyp(df, _col0_, _col1_)
                        _xyp_ = self.p2s.xyp(df, (_col0_, self.p2s.SETp),    _col1_)
                        _xyp_ = self.p2s.xyp(df, (_col0_, self.p2s.SCALARp), _col1_)
                        _xyp_ = self.p2s.xyp(df, _col0_, (_col1_, self.p2s.SETp))
                        _xyp_ = self.p2s.xyp(df, _col0_, (_col1_, self.p2s.SCALARp))
                        _xyp_ = self.p2s.xyp(df, (_col0_, self.p2s.SETp),    (_col1_, self.p2s.SETp))
                        _xyp_ = self.p2s.xyp(df, (_col0_, self.p2s.SCALARp), (_col1_, self.p2s.SETp))
                        _xyp_ = self.p2s.xyp(df, (_col0_, self.p2s.SCALARp), (_col1_, self.p2s.SCALARp))
                        _xyp_ = self.p2s.xyp(df, (_col0_, self.p2s.SETp),    (_col1_, self.p2s.SCALARp))

    def test_exhaustiveMediums(self, samples=1):
        # One representative pair per type-combination: int×int, int×float, float×float,
        # datetime×int, date×float, str×int, str×str.  Full all-column enumeration is
        # covered at small sizes in test_exhaustiveSmalls; here we only need to confirm
        # that medium-sized DataFrames don't surface new code paths.
        _representative_pairs_ = [('a','b'), ('a','c'), ('c','d'), ('g','a'), ('h','c'), ('j','a'), ('j','k')]
        for _sample_ in range(samples):
            for n in [100, 500]:
                df = randomDataFrame(n)
                for _col0_, _col1_ in _representative_pairs_:
                    _xyp_ = self.p2s.xyp(df, _col0_, _col1_)
                    _xyp_ = self.p2s.xyp(df, (_col0_, self.p2s.SETp),    _col1_)
                    _xyp_ = self.p2s.xyp(df, (_col0_, self.p2s.SCALARp), _col1_)
                    _xyp_ = self.p2s.xyp(df, _col0_, (_col1_, self.p2s.SETp))
                    _xyp_ = self.p2s.xyp(df, _col0_, (_col1_, self.p2s.SCALARp))
                    _xyp_ = self.p2s.xyp(df, (_col0_, self.p2s.SETp),    (_col1_, self.p2s.SETp))
                    _xyp_ = self.p2s.xyp(df, (_col0_, self.p2s.SCALARp), (_col1_, self.p2s.SETp))
                    _xyp_ = self.p2s.xyp(df, (_col0_, self.p2s.SCALARp), (_col1_, self.p2s.SCALARp))
                    _xyp_ = self.p2s.xyp(df, (_col0_, self.p2s.SETp),    (_col1_, self.p2s.SCALARp))

    #
    # test_structVStack() - tests issue where a struct has to be vstacked
    # - this requires that the struct field names have to be the same in all frames
    #
    def test_structVStack(self):
        df = pl.DataFrame({
            'a':[1,2,3],
            'b':['a','b','c'],
            'c':[1,2,3],
            'd':['a','b','d']
        })
        _xyp_ = self.p2s.xyp(df, [('a','b'),('c','d')], 'a', dot_size=8)
        _xyp_        

if __name__ == '__main__':
    unittest.main()
