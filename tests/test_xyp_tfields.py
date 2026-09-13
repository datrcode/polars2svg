import unittest
import polars as pl
from polars2svg import Polars2SVG
from random_dataframe import randomDataFrame
from datetime import date
from pathlib import Path

try:
    import kagglehub
    _KAGGLEHUB_AVAILABLE = True
except ImportError:
    _KAGGLEHUB_AVAILABLE = False

class Testxyp_tfields(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def test_exhaustive(self):
        df = randomDataFrame(10)
        for _x_enum_ in self.p2s.TimeLinearTypeP:
            for _y_enum_ in self.p2s.TimePeriodicTypeP:
                for _c_enum_ in self.p2s.TimePeriodicTypeP:
                    _xyp_test_ = self.p2s.xyp(df, 
                                              self.p2s.tField('i', _x_enum_), 
                                              self.p2s.tField('i', _y_enum_), 
                                              color=(self.p2s.tField('i', _c_enum_), self.p2s.CSETp), 
                                              dot_size=3)

    @unittest.skipUnless(_KAGGLEHUB_AVAILABLE, "kagglehub not importable (broken upstream dependency)")
    def test_example_globalTemperatures(self):
        # Network-dependent, but NOT credential-dependent: this dataset is public,
        # so kagglehub.dataset_download fetches it anonymously with no Kaggle
        # credentials configured. It hits the network on a cache miss and returns
        # immediately once cached. CI runs this for real on every run — see the
        # "Run the test suite" step in .github/workflows/ci.yml.
        # The try/except below is the fallback, not the normal path: a Kaggle
        # outage, a rate limit, or a fresh clone with no network degrades this to
        # a clean skip rather than an error, so the suite still passes offline.
        try:
            path = kagglehub.dataset_download("berkeleyearth/climate-change-earth-surface-temperature-data")
        except Exception as e:
            self.skipTest(f"kagglehub dataset download unavailable (no network/credentials?): {e}")
        df_orig = pl.read_csv(Path(path, 'GlobalLandTemperaturesByCountry.csv'), try_parse_dates=True)
        df      = df_orig.filter(pl.col('dt') > date.fromisoformat('1990-01-01')).filter(pl.col('Country').is_in(['Tunisia', 'Algeria', 'Cayman Islands']))
        _xyp_   = self.p2s.xyp(df, self.p2s.tField('dt', self.p2s.PT_mp), 'AverageTemperature', color='Country', 
                              line=('Country', self.p2s.tField('dt', self.p2s.LT_Yp), self.p2s.LINECOLOR_FIELD), wxh=(1024,256))

if __name__ == '__main__':
    unittest.main()
