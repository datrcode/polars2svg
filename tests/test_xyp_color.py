"""Regression tests: verify no exceptions are raised across xyp color-mode combinations.
No assertions are made — pass = no crash."""
import unittest
import polars as pl
import time
from polars2svg import Polars2SVG

from random_dataframe import randomDataFrame

class Testxyp_color(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def test_colorSingles(self, samples=1):
        t = time.time()
        for _sample_ in range(samples):
            for n in [1]:
                df = randomDataFrame(n, na_probability=0.0)
                for _col0_ in ['a','c']:
                    for _col1_ in ['b','d']:
                        for _col2_ in df.columns:
                            for _color_ in [
                                _col2_,
                                (_col2_,),
                                self.p2s.CROW_MAGNITUDEp,
                                self.p2s.CROW_STRETCHEDp,
                                (_col2_, self.p2s.CMAGNITUDE_SUMp),
                                (_col2_, self.p2s.CMAGNITUDE_MINp),
                                (_col2_, self.p2s.CMAGNITUDE_MEDIANp),
                                (_col2_, self.p2s.CMAGNITUDE_MEANp),
                                (_col2_, self.p2s.CMAGNITUDE_MAXp),
                                (_col2_, self.p2s.CSTRETCHED_SUMp),
                                (_col2_, self.p2s.CSTRETCHED_MINp),
                                (_col2_, self.p2s.CSTRETCHED_MEDIANp),
                                (_col2_, self.p2s.CSTRETCHED_MEANp),
                                (_col2_, self.p2s.CSTRETCHED_MAXp),
                                (_col2_, self.p2s.CSETp),
                                (_col2_, self.p2s.CSET_MAGNITUDEp),
                                (_col2_, self.p2s.CSET_STRETCHEDp),
                            ]: _xyp_ = self.p2s.xyp(df, _col0_, _col1_, color=_color_)

    def test_colorSmalls(self, samples=1):
        t = time.time()
        for _sample_ in range(samples):
            for n in [2,4]:
                df = randomDataFrame(n, na_probability=0.1)
                for _col0_ in ['a','c']:
                    for _col1_ in ['b','d']:
                        for _col2_ in df.columns:
                            for _color_ in [
                                _col2_,
                                (_col2_,),
                                self.p2s.CROW_MAGNITUDEp,
                                self.p2s.CROW_STRETCHEDp,
                                (_col2_, self.p2s.CMAGNITUDE_SUMp),
                                (_col2_, self.p2s.CMAGNITUDE_MINp),
                                (_col2_, self.p2s.CMAGNITUDE_MEDIANp),
                                (_col2_, self.p2s.CMAGNITUDE_MEANp),
                                (_col2_, self.p2s.CMAGNITUDE_MAXp),
                                (_col2_, self.p2s.CSTRETCHED_SUMp),
                                (_col2_, self.p2s.CSTRETCHED_MINp),
                                (_col2_, self.p2s.CSTRETCHED_MEDIANp),
                                (_col2_, self.p2s.CSTRETCHED_MEANp),
                                (_col2_, self.p2s.CSTRETCHED_MAXp),
                                (_col2_, self.p2s.CSETp),
                                (_col2_, self.p2s.CSET_MAGNITUDEp),
                                (_col2_, self.p2s.CSET_STRETCHEDp),
                            ]: _xyp_ = self.p2s.xyp(df, _col0_, _col1_, color=_color_)

if __name__ == '__main__':
    unittest.main()
