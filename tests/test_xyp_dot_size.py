import unittest
import polars as pl
from polars2svg import Polars2SVG

from random_dataframe import randomDataFrame

class Testxyp_dot_size(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()
        self.df  = randomDataFrame(100, na_probability=0.01)

    def test_dot_size_is_none(self):
        xyp = self.p2s.xyp(self.df, 'a', 'b', dot_size=None)
        xyp = self.p2s.xyp(self.df, ['a','b'], ['c','d'], dot_size=None)

    def test_dot_size_is_int(self):
        xyp = self.p2s.xyp(self.df, 'a', 'b', dot_size=1)
        xyp = self.p2s.xyp(self.df, 'a', 'b', dot_size=2)
        xyp = self.p2s.xyp(self.df, 'a', 'b', dot_size=10)
        xyp = self.p2s.xyp(self.df, ['a','b'], ['c','d'], dot_size=2)

    def test_dot_size_is_float(self):
        xyp = self.p2s.xyp(self.df, 'a', 'b', dot_size=0.1)
        xyp = self.p2s.xyp(self.df, 'a', 'b', dot_size=1.0)
        xyp = self.p2s.xyp(self.df, 'a', 'b', dot_size=2.0)
        xyp = self.p2s.xyp(self.df, 'a', 'b', dot_size=10.0)
        xyp = self.p2s.xyp(self.df, ['a','b'], ['c','d'], dot_size=2.0)

    def test_dot_size_is_int_list(self):
        xyp = self.p2s.xyp(self.df, ['a','b'], ['c','d'], dot_size=[1,2])
        xyp = self.p2s.xyp(self.df, ['a','b'], ['c','d'], dot_size=[5,5])

    def test_dot_size_is_float_list(self):
        xyp = self.p2s.xyp(self.df, ['a','b'], ['c','d'], dot_size=[1.0,2.0])
        xyp = self.p2s.xyp(self.df, ['a','b'], ['c','d'], dot_size=[5.0,5.0])

    def test_dot_size_randoms(self):
        df = pl.DataFrame({'i0':[1,2,3],
                           'i1':[4,5,6],
                           'f0':[1.1,2.2,3.3],
                           'f1':[4.4,5.5,6.6],})

        self.p2s.xyp(df, 'i0', 'i1', dot_size=None)

        self.p2s.xyp(df, 'i0', 'i1', dot_size=2)
        self.p2s.xyp(df, 'i0', 'i1', dot_size=3)
        self.p2s.xyp(df, 'i0', 'i1', dot_size=[4])

        self.p2s.xyp(df, 'i0', 'i1', dot_size=2.5)
        self.p2s.xyp(df, 'i0', 'i1', dot_size=3.1)
        self.p2s.xyp(df, 'i0', 'i1', dot_size=[4.5])

        self.p2s.xyp(df, ['i0','i1'], ['f0','f1'], dot_size=4)
        self.p2s.xyp(df, ['i0','i1'], ['f0','f1'], dot_size=[5,   10])
        self.p2s.xyp(df, ['i0','i1'], ['f0','f1'], dot_size=6.0)
        self.p2s.xyp(df, ['i0','i1'], ['f0','f1'], dot_size=[3.0, 10.0])

if __name__ == '__main__':
    unittest.main()
