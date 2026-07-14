import unittest
import polars as pl
import time
from polars2svg import Polars2SVG

from random_dataframe import randomDataFrame

class Testxyp_distributions(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def test_exceptions(self, samples=1, df_size=100):
        df       = randomDataFrame(df_size)
        with self.assertRaises(ValueError):
            _xyp_ = self.p2s.xyp(df, 'a', 'b', x_distributions=('c', self.p2s.DISTRIBUTION_OUTSIDEp, self.p2s.DISTRIBUTION_INSIDEp))
        with self.assertRaises(ValueError):
            _xyp_ = self.p2s.xyp(df, 'a', 'b', y_distributions=('c', self.p2s.DISTRIBUTION_OUTSIDEp, self.p2s.DISTRIBUTION_INSIDEp))
        with self.assertRaises(ValueError):
            _xyp_ = self.p2s.xyp(df, 'a', 'b', x_distributions=['c','d','e', 10, 11])
        with self.assertRaises(ValueError):
            _xyp_ = self.p2s.xyp(df, 'a', 'b', x_distributions=['c','d','e', 0.1, 0.2])
        with self.assertRaises(ValueError):
            _xyp_ = self.p2s.xyp(df, 'a', 'b', x_distributions=['c','d','e', '#ff0000', '#00ff00'])
        with self.assertRaises(ValueError):
            _xyp_ = self.p2s.xyp(df, 'a', 'b', x_distributions=('c', '#ff0000', '#00ff00'))
        with self.assertRaises(ValueError):
            _xyp_ = self.p2s.xyp(df, 'a', 'b', x_distributions=['c',self.p2s.DISTRIBUTION_COLOR_MIN_TO_COLOR_MAX,
                                                                    self.p2s.DISTRIBUTION_ZERO_TO_COLOR_MAX])

    def test_basics(self, samples=1, df_size=200):
        df       = randomDataFrame(df_size)
        _params_ = {'df':df, 'x':'j', 'y':'k'}
        _xyp_ = self.p2s.xyp(**_params_, x_distributions='a')
        _xyp_ = self.p2s.xyp(**_params_, x_distributions=('a','b'))
        _xyp_ = self.p2s.xyp(**_params_, x_distributions=['a','b'])
        _xyp_ = self.p2s.xyp(**_params_, x_distributions=['a','b', self.p2s.SETp])

    def test_rowCounts(self, sample=1, df_size=200):
        df       = randomDataFrame(df_size)
        _params_ = {'df':df, 'x':'a', 'y':'b'}
        _xyp_ = self.p2s.xyp(**_params_, x_distributions=self.p2s.ROW_COUNTp)
        _xyp_ = self.p2s.xyp(**_params_, x_distributions=self.p2s.ROW_COUNTp, y_distributions=self.p2s.ROW_COUNTp)
        _xyp_ = self.p2s.xyp(**_params_,                                      y_distributions=self.p2s.ROW_COUNTp)

    def test_constantDistributionField(self):
        # Regression: when all values in the distribution field are identical,
        # bin_width == 0 caused division by zero -> NaN -> InvalidOperationError
        # on the strict_cast to Int64.
        df = pl.DataFrame({'x': [1.0, 2.0, 3.0], 'y': [1.0, 2.0, 3.0], 'v': [5.0, 5.0, 5.0]})
        # x_distributions on a constant column (all values == 5.0)
        _xyp_ = self.p2s.xyp(df=df, x='x', y='y', x_distributions='v')
        # y_distributions on a constant column
        _xyp_ = self.p2s.xyp(df=df, x='x', y='y', y_distributions='v')
        # both axes at once
        _xyp_ = self.p2s.xyp(df=df, x='x', y='y', x_distributions='v', y_distributions='v')

    def test_insideOutside(self, samples=1, df_size=200):
        df       = randomDataFrame(df_size)
        _params_ = {'df':df, 'x':'j', 'y':'k'}
        _tiles_  = []
        for _dot_size_ in [None, 1, 4, 10, 0.75]:
            for _x_dist_ in [None, 'a', ('a', self.p2s.DISTRIBUTION_INSIDEp), ('a', self.p2s.DISTRIBUTION_OUTSIDEp)]:
                for _y_dist_ in [None, 'b', ('b', self.p2s.DISTRIBUTION_INSIDEp), ('b', self.p2s.DISTRIBUTION_OUTSIDEp)]:
                    _xyp_ = self.p2s.xyp(x_distributions=_x_dist_, y_distributions=_y_dist_, dot_size=_dot_size_, **_params_)

if __name__ == '__main__':
    unittest.main()
