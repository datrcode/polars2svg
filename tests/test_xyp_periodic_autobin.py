import unittest
import datetime
import polars as pl
from polars2svg import Polars2SVG


#
# Periodic-time autobin for xyp distributions.
#
# A periodic time transform (month-of-year, hour-of-day, day-of-week, ...) plots as
# discrete integers, so DISTRIBUTION_AUTOBINp should place exactly one bar per integer
# value the data spans -- like Timep's periodic bins -- instead of the pixel-derived
# count, which aliased several period units into a bar or left empty gaps.
#
class TestXypPeriodicAutobin(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def _monthly_df(self, n=240):
        # ~10 years of fortnightly samples -> every month (1..12) is well populated
        rows = []
        for i in range(n):
            d = datetime.datetime(2015, 1, 1) + datetime.timedelta(days=15 * i)
            rows.append({'ts': d, 'y': (i % 40), 'v': float(i)})
        return pl.DataFrame(rows)

    def test_periodic_axis_detected(self):
        df  = self._monthly_df()
        xtf = self.p2s.tField('ts', self.p2s.PT_mp)
        xyp = self.p2s.xyp(df, x=xtf, y='y', x_distributions=[self.p2s.ROW_COUNTp, self.p2s.DISTRIBUTION_AUTOBINp])
        self.assertTrue(xyp.__axisIsPeriodicTime__(xyp.x_clean))

    def test_one_bar_per_period_unit(self):
        # months present in the data span 1..12 -> 12 discrete values -> 12 bins
        df  = self._monthly_df()
        xtf = self.p2s.tField('ts', self.p2s.PT_mp)
        xyp = self.p2s.xyp(df, x=xtf, y='y', x_distributions=[self.p2s.ROW_COUNTp, self.p2s.DISTRIBUTION_AUTOBINp])
        _pmin_, _pmax_ = self.p2s.timePeriodicRange(self.p2s.PT_mp)  # (1, 12)
        # the data covers the full periodic range, so the bin count equals the discrete span
        self.assertEqual(xyp.x_distributions_clean['bins'][0], _pmax_ - _pmin_ + 1)
        self.assertEqual(xyp.df_x_distribution.height, _pmax_ - _pmin_ + 1)

    def test_no_empty_gaps(self):
        # with one bar per unit and every month populated, no autobin bar is empty --
        # the whole point of the fix (pixel-derived binning left interleaved 0.0 gaps).
        df  = self._monthly_df()
        xtf = self.p2s.tField('ts', self.p2s.PT_mp)
        xyp = self.p2s.xyp(df, x=xtf, y='y', x_distributions=[self.p2s.ROW_COUNTp, self.p2s.DISTRIBUTION_AUTOBINp])
        _totals_ = xyp.df_x_distribution['__xi_total__'].to_list()
        self.assertTrue(all(t > 0 for t in _totals_), f'expected every periodic bin populated, got {_totals_}')

    def test_periodic_y_axis(self):
        # same behavior on the y axis (hours 0..23)
        rows = []
        for i in range(24 * 20):
            d = datetime.datetime(2020, 1, 1) + datetime.timedelta(hours=i)
            rows.append({'x': (i % 30), 'ts': d})
        df  = pl.DataFrame(rows)
        ytf = self.p2s.tField('ts', self.p2s.PT_Hp)
        xyp = self.p2s.xyp(df, x='x', y=ytf, y_distributions=[self.p2s.ROW_COUNTp, self.p2s.DISTRIBUTION_AUTOBINp])
        _pmin_, _pmax_ = self.p2s.timePeriodicRange(self.p2s.PT_Hp)  # (0, 23)
        self.assertEqual(xyp.y_distributions_clean['bins'][0], _pmax_ - _pmin_ + 1)
        self.assertEqual(xyp.df_y_distribution.height, _pmax_ - _pmin_ + 1)

    def test_explicit_bins_honored_on_periodic_axis(self):
        # an explicit bin count is NOT overridden by the periodic adjustment
        df  = self._monthly_df()
        xtf = self.p2s.tField('ts', self.p2s.PT_mp)
        xyp = self.p2s.xyp(df, x=xtf, y='y', x_distributions=[self.p2s.ROW_COUNTp, 5])
        self.assertEqual(xyp.x_distributions_clean['bins'][0], 5)
        self.assertEqual(xyp.df_x_distribution.height, 5)

    def test_non_periodic_autobin_unchanged(self):
        # a plain numeric axis keeps the pixel-derived bin count (unaffected by the fix)
        df  = self._monthly_df()
        xyp = self.p2s.xyp(df, x='v', y='y', x_distributions=[self.p2s.ROW_COUNTp, self.p2s.DISTRIBUTION_AUTOBINp])
        self.assertFalse(xyp.__axisIsPeriodicTime__(xyp.x_clean))
        # pixel-derived count is much larger than the 12 a periodic axis would pick
        self.assertGreater(xyp.x_distributions_clean['bins'][0], 12)

    def test_renders_valid_svg(self):
        df  = self._monthly_df()
        xtf = self.p2s.tField('ts', self.p2s.PT_mp)
        xyp = self.p2s.xyp(df, x=xtf, y='y', x_distributions=[self.p2s.ROW_COUNTp, self.p2s.DISTRIBUTION_AUTOBINp])
        self.assertIn('<svg', xyp.svg)
        self.assertIn('</svg>', xyp.svg)


if __name__ == '__main__':
    unittest.main()
