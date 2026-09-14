import unittest
import polars as pl
import numpy as np
from datetime import datetime, date, timedelta
import random
from polars2svg import Polars2SVG
from svg_test_utils import normalize_svg

class Testxyp_xy_ranges(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()
        rng = np.random.default_rng()
        n   = 1_000
        self.df  = pl.DataFrame({'a':rng.normal(loc=2500, scale=10, size=n),
                                 'b':rng.normal(loc=500,  scale=3,  size=n)})

    def test_withoutRanges(self):
        self.p2s.xyp(self.df, 'a', 'b', dot_size=1.25, color=self.p2s.CROW_MAGNITUDEp)
        self.p2s.xyp(self.df, 'a', 'b', dot_size=1.25, color=self.p2s.CROW_MAGNITUDEp, draw_context=False)
        self.p2s.xyp(self.df, 'a', 'b', dot_size=3,    color=self.p2s.CROW_MAGNITUDEp)
        self.p2s.xyp(self.df, 'a', 'b', dot_size=3,    color=self.p2s.CROW_MAGNITUDEp, draw_context=False)

    def test_subsetRangesFloat(self):
        self.p2s.xyp(self.df, 'a', 'b', dot_size=1.25, color=self.p2s.CROW_STRETCHEDp, x_range=(2480,2500))
        self.p2s.xyp(self.df, 'a', 'b', dot_size=1.25, color=self.p2s.CROW_STRETCHEDp,                      y_range=(500, 505))
        self.p2s.xyp(self.df, 'a', 'b', dot_size=1.25, color=self.p2s.CROW_STRETCHEDp, x_range=(2480,2500), y_range=(500, 505))

    def test_externalRangesFloat(self):
        self.p2s.xyp(self.df, 'a', 'b', dot_size=1.25, color=self.p2s.CROW_STRETCHEDp, x_range=(2_200, 2_800))
        self.p2s.xyp(self.df, 'a', 'b', dot_size=1.25, color=self.p2s.CROW_STRETCHEDp,                         y_range=(450, 600))
        self.p2s.xyp(self.df, 'a', 'b', dot_size=1.25, color=self.p2s.CROW_STRETCHEDp, x_range=(2_200, 2_800), y_range=(450, 600))

    def test_externalRangesFloat_distributions(self):
        _param_ = {'x_distributions':self.p2s.ROW_COUNTp, 'y_distributions':self.p2s.ROW_COUNTp}
        self.p2s.xyp(self.df, 'a', 'b', dot_size=1.25, color=self.p2s.CROW_STRETCHEDp, x_range=(2_200, 2_800),                     **_param_)
        self.p2s.xyp(self.df, 'a', 'b', dot_size=1.25, color=self.p2s.CROW_STRETCHEDp,                         y_range=(450, 600), **_param_)
        self.p2s.xyp(self.df, 'a', 'b', dot_size=1.25, color=self.p2s.CROW_STRETCHEDp, x_range=(2_200, 2_800), y_range=(450, 600), **_param_)

    def test_outOfRangeFloat(self):
        self.p2s.xyp(self.df, 'a', 'b', dot_size=1.25, color=self.p2s.CROW_STRETCHEDp, x_range=(12_200, 14_800))
        self.p2s.xyp(self.df, 'a', 'b', dot_size=1.25, color=self.p2s.CROW_STRETCHEDp,                           y_range=(45_000, 46_000))
        self.p2s.xyp(self.df, 'a', 'b', dot_size=1.25, color=self.p2s.CROW_STRETCHEDp, x_range=(12_200, 14_800), y_range=(45_000, 46_000))


    def test_outOfRangeFloat_distributions(self):
        _param_ = {'x_distributions':self.p2s.ROW_COUNTp, 'y_distributions':self.p2s.ROW_COUNTp}
        self.p2s.xyp(self.df, 'a', 'b', dot_size=1.25, color=self.p2s.CROW_STRETCHEDp, x_range=(12_200, 14_800),                           **_param_)
        self.p2s.xyp(self.df, 'a', 'b', dot_size=1.25, color=self.p2s.CROW_STRETCHEDp,                           y_range=(45_000, 46_000), **_param_)
        self.p2s.xyp(self.df, 'a', 'b', dot_size=1.25, color=self.p2s.CROW_STRETCHEDp, x_range=(12_200, 14_800), y_range=(45_000, 46_000), **_param_)

    def test_outOfRangeFloat_distributions2(self):
        _param_ = {'x_distributions':'a', 'y_distributions':'b'}
        self.p2s.xyp(self.df, 'a', 'b', dot_size=1.25, color=self.p2s.CROW_STRETCHEDp, x_range=(12_200, 14_800),                           **_param_)
        self.p2s.xyp(self.df, 'a', 'b', dot_size=1.25, color=self.p2s.CROW_STRETCHEDp,                           y_range=(45_000, 46_000), **_param_)
        self.p2s.xyp(self.df, 'a', 'b', dot_size=1.25, color=self.p2s.CROW_STRETCHEDp, x_range=(12_200, 14_800), y_range=(45_000, 46_000), **_param_)

    def test_datetimeRanges(self):
        _lu_ = {'dt':[], 'value':[]}
        for i in range(200):
            _year_   = random.randint(1980, 2000)
            _month_  = random.randint(1, 12)
            _day_    = random.randint(1, 28)
            _hour_   = random.randint(0, 23)
            _minute_ = random.randint(0, 59)
            _second_ = random.randint(0, 59)
            _lu_['dt'].append(f'{_year_:04}-{_month_:02}-{_day_:02} {_hour_:02}:{_minute_:02}:{_second_:02}')
            _lu_['value'].append(random.randint(0, 100))
        df            = pl.DataFrame(_lu_).with_columns(pl.col('dt').str.strptime(pl.Datetime))
        format_string = "%Y-%m-%d %H:%M:%S"
        # Date-range clipping logic is independent of render dimensions; use a single fixed wxh/dot_size.
        _wxh_, _dot_size_ = (128, 128), 2.0
        _params_ = {'df':df, 'x':'dt', 'y':'value', 'wxh':_wxh_, 'dot_size':_dot_size_}
        self.p2s.xyp(**_params_)
        _params_['x'], _params_['y'] = 'value', 'dt'
        self.p2s.xyp(**_params_)
        # range begin date is within the data date range
        _fm_, _to_ = datetime.strptime('1990-06-01 00:00:00', format_string), datetime.strptime('2020-12-01 23:59:59', format_string)
        _params_ = {'df':df, 'x':'dt', 'y':'value', 'wxh':_wxh_, 'dot_size':_dot_size_, 'x_range':(_fm_,_to_)}
        self.p2s.xyp(**_params_)
        _params_['x'], _params_['y'], _params_['x_range'], _params_['y_range'] = 'value', 'dt', None, (_fm_,_to_)
        self.p2s.xyp(**_params_)
        # range end date is within the data date range
        _fm_, _to_ = datetime.strptime('1950-06-01 00:00:00', format_string), datetime.strptime('1995-12-01 23:59:59', format_string)
        _params_ = {'df':df, 'x':'dt', 'y':'value', 'wxh':_wxh_, 'dot_size':_dot_size_, 'x_range':(_fm_,_to_)}
        self.p2s.xyp(**_params_)
        _params_['x'], _params_['y'], _params_['x_range'], _params_['y_range'] = 'value', 'dt', None, (_fm_,_to_)
        self.p2s.xyp(**_params_)
        # both dates are within the data date range
        _fm_, _to_ = datetime.strptime('1985-06-01 00:00:00', format_string), datetime.strptime('1997-12-01 23:59:59', format_string)
        _params_ = {'df':df, 'x':'dt', 'y':'value', 'wxh':_wxh_, 'dot_size':_dot_size_, 'x_range':(_fm_,_to_)}
        self.p2s.xyp(**_params_)
        _params_['x'], _params_['y'], _params_['x_range'], _params_['y_range'] = 'value', 'dt', None, (_fm_,_to_)
        self.p2s.xyp(**_params_)
        # both dates are outside the data date range
        _fm_, _to_ = datetime.strptime('1960-06-01 00:00:00', format_string), datetime.strptime('2020-12-01 23:59:59', format_string)
        _params_ = {'df':df, 'x':'dt', 'y':'value', 'wxh':_wxh_, 'dot_size':_dot_size_, 'x_range':(_fm_,_to_)}
        self.p2s.xyp(**_params_)
        _params_['x'], _params_['y'], _params_['x_range'], _params_['y_range'] = 'value', 'dt', None, (_fm_,_to_)
        self.p2s.xyp(**_params_)
        # both dates are outside the data date range (no overlap, before)
        _fm_, _to_ = datetime.strptime('1960-06-01 00:00:00', format_string), datetime.strptime('1975-12-01 23:59:59', format_string)
        _params_ = {'df':df, 'x':'dt', 'y':'value', 'wxh':_wxh_, 'dot_size':_dot_size_, 'x_range':(_fm_,_to_)}
        self.p2s.xyp(**_params_)
        _params_['x'], _params_['y'], _params_['x_range'], _params_['y_range'] = 'value', 'dt', None, (_fm_,_to_)
        self.p2s.xyp(**_params_)
        # both dates are outside the data date range (no overlap, after)
        _fm_, _to_ = datetime.strptime('2005-06-01 00:00:00', format_string), datetime.strptime('2010-12-01 23:59:59', format_string)
        _params_ = {'df':df, 'x':'dt', 'y':'value', 'wxh':_wxh_, 'dot_size':_dot_size_, 'x_range':(_fm_,_to_)}
        self.p2s.xyp(**_params_)
        _params_['x'], _params_['y'], _params_['x_range'], _params_['y_range'] = 'value', 'dt', None, (_fm_,_to_)
        self.p2s.xyp(**_params_)

    def test_dateRanges(self):
        _lu_ = {'dt':[], 'value':[]}
        for i in range(200):
            _year_   = random.randint(1980, 2000)
            _month_  = random.randint(1, 12)
            _day_    = random.randint(1, 28)
            _lu_['dt'].append(f'{_year_:04}-{_month_:02}-{_day_:02}')
            _lu_['value'].append(random.randint(0, 100))
        df            = pl.DataFrame(_lu_).with_columns(pl.col('dt').cast(pl.Date))
        format_string = "%Y-%m-%d"
        # Date-range clipping logic is independent of render dimensions; use a single fixed wxh/dot_size.
        _wxh_, _dot_size_ = (128, 128), 2.0
        _params_ = {'df':df, 'x':'dt', 'y':'value', 'wxh':_wxh_, 'dot_size':_dot_size_}
        self.p2s.xyp(**_params_)
        _params_['x'], _params_['y'] = 'value', 'dt'
        self.p2s.xyp(**_params_)
        # range begin date is within the data date range
        _fm_, _to_ = datetime.strptime('1990-06-01', format_string), datetime.strptime('2020-12-01', format_string)
        _params_ = {'df':df, 'x':'dt', 'y':'value', 'wxh':_wxh_, 'dot_size':_dot_size_, 'x_range':(_fm_,_to_)}
        self.p2s.xyp(**_params_)
        _params_['x'], _params_['y'], _params_['x_range'], _params_['y_range'] = 'value', 'dt', None, (_fm_,_to_)
        self.p2s.xyp(**_params_)
        # range end date is within the data date range
        _fm_, _to_ = datetime.strptime('1950-06-01', format_string), datetime.strptime('1995-12-01', format_string)
        _params_ = {'df':df, 'x':'dt', 'y':'value', 'wxh':_wxh_, 'dot_size':_dot_size_, 'x_range':(_fm_,_to_)}
        self.p2s.xyp(**_params_)
        _params_['x'], _params_['y'], _params_['x_range'], _params_['y_range'] = 'value', 'dt', None, (_fm_,_to_)
        self.p2s.xyp(**_params_)
        # both dates are within the data date range
        _fm_, _to_ = datetime.strptime('1985-06-01', format_string), datetime.strptime('1997-12-01', format_string)
        _params_ = {'df':df, 'x':'dt', 'y':'value', 'wxh':_wxh_, 'dot_size':_dot_size_, 'x_range':(_fm_,_to_)}
        self.p2s.xyp(**_params_)
        _params_['x'], _params_['y'], _params_['x_range'], _params_['y_range'] = 'value', 'dt', None, (_fm_,_to_)
        self.p2s.xyp(**_params_)
        # both dates are outside the data date range
        _fm_, _to_ = datetime.strptime('1960-06-01', format_string), datetime.strptime('2020-12-01', format_string)
        _params_ = {'df':df, 'x':'dt', 'y':'value', 'wxh':_wxh_, 'dot_size':_dot_size_, 'x_range':(_fm_,_to_)}
        self.p2s.xyp(**_params_)
        _params_['x'], _params_['y'], _params_['x_range'], _params_['y_range'] = 'value', 'dt', None, (_fm_,_to_)
        self.p2s.xyp(**_params_)
        # both dates are outside the data date range (no overlap, before)
        _fm_, _to_ = datetime.strptime('1960-06-01', format_string), datetime.strptime('1975-12-01', format_string)
        _params_ = {'df':df, 'x':'dt', 'y':'value', 'wxh':_wxh_, 'dot_size':_dot_size_, 'x_range':(_fm_,_to_)}
        self.p2s.xyp(**_params_)
        _params_['x'], _params_['y'], _params_['x_range'], _params_['y_range'] = 'value', 'dt', None, (_fm_,_to_)
        self.p2s.xyp(**_params_)
        # both dates are outside the data date range (no overlap, after)
        _fm_, _to_ = datetime.strptime('2005-06-01', format_string), datetime.strptime('2010-12-01', format_string)
        _params_ = {'df':df, 'x':'dt', 'y':'value', 'wxh':_wxh_, 'dot_size':_dot_size_, 'x_range':(_fm_,_to_)}
        self.p2s.xyp(**_params_)
        _params_['x'], _params_['y'], _params_['x_range'], _params_['y_range'] = 'value', 'dt', None, (_fm_,_to_)
        self.p2s.xyp(**_params_)

    def test_screenWorldTransforms(self):
        df = pl.DataFrame({'x':[1, 2, 3, 4, 5,  6],
                           'y':[5, 7, 9, 3, 4, 15]})
        for _xrange_ in [None, (-10, 10), (2,5)]:
            for _yrange_ in [None, (-10, 10), (2,8)]:
                _xyp_ = self.p2s.xyp(df, 'x', 'y', dot_size=3.0, x_range=_xrange_)
                _df_ = _xyp_.df_flat
                for i in range(_df_.shape[0]):
                    _sx_, _sy_, _wx_, _wy_ = _df_['__xpx__'][i], _df_['__ypx__'][i], _df_['__x__'][i], _df_['__y__'][i]
                    assert _sx_ == round(_xyp_.wxToSx(_wx_))
                    assert _sy_ == round(_xyp_.wyToSy(_wy_))
                    assert _wx_ == round(_xyp_.sxToWx(_sx_))
                    assert _wy_ == round(_xyp_.syToWy(_sy_))


class Testxyp_date_ranges(unittest.TestCase):
    '''x_range=/y_range= given as `date` rather than `datetime`.

    datetime is a SUBCLASS of date, so the guard that used to admit these --
    `isinstance(v, datetime) or isinstance(v, date)` -- had exactly one effect beyond
    plain datetime: it let a pure `date` through.  And a pure date reached two
    subtractions that cannot take one, `date - datetime` in __resolveRanges__ and
    `datetime - date` in __renderContext_linearTime__ (which reads self.x_range raw).
    So the only input the clause existed to accept was the only input it could not
    handle, and x_range=(date(...), date(...)) was a TypeError.

    test_dateRanges above did not catch it because it builds its bounds with
    datetime.strptime(), which returns datetimes.  Ranges are now promoted to datetime
    once in __validateInput__, so a date bound means the same instant as the equivalent
    datetime bound to every consumer.  PLANNING.md §15.
    '''
    def setUp(self):
        self.p2s = Polars2SVG()
        _n_        = 60
        self.df_dt = pl.DataFrame({'t': [datetime(2024, 1, 1) + timedelta(days=i) for i in range(_n_)],
                                   'v': [float(i % 17) for i in range(_n_)]})
        self.df_d  = self.df_dt.with_columns(pl.col('t').cast(pl.Date))

    def _svg_(self, df, **kwargs):
        return normalize_svg(self.p2s.xyp(df, x='t', y='v', wxh=(256, 256), **kwargs).svg)

    def test_date_bounds_render_at_all(self):
        for _name_, _df_ in (('Datetime column', self.df_dt), ('Date column', self.df_d)):
            with self.subTest(column=_name_):
                self.assertIn('<svg', self._svg_(_df_, x_range=(date(2024, 1, 5), date(2024, 2, 5))))

    def test_date_bounds_equal_the_same_instant_as_datetime_bounds(self):
        for _name_, _df_ in (('Datetime column', self.df_dt), ('Date column', self.df_d)):
            with self.subTest(column=_name_):
                _d_  = self._svg_(_df_, x_range=(date(2024, 1, 5),          date(2024, 2, 5)))
                _dt_ = self._svg_(_df_, x_range=(datetime(2024, 1, 5),      datetime(2024, 2, 5)))
                self.assertEqual(_d_, _dt_, 'a date bound must mean midnight of that day, '
                                            'not a differently-rendered axis')

    def test_a_range_may_mix_date_and_datetime(self):
        # The paired form chose its branch on the MIN and applied it to both, so a mixed
        # range was unreachable even once date alone worked.
        for _name_, _df_ in (('Datetime column', self.df_dt), ('Date column', self.df_d)):
            with self.subTest(column=_name_):
                _mixed_ = self._svg_(_df_, x_range=(date(2024, 1, 5), datetime(2024, 2, 5)))
                _dt_    = self._svg_(_df_, x_range=(datetime(2024, 1, 5), datetime(2024, 2, 5)))
                self.assertEqual(_mixed_, _dt_)

    def test_y_range_takes_dates_too(self):
        _df_ = self.df_dt.rename({'t': 'v2', 'v': 't'}).select(['t', 'v2'])
        _svg_ = normalize_svg(self.p2s.xyp(_df_, x='t', y='v2', wxh=(256, 256),
                                           y_range=(date(2024, 1, 5), date(2024, 2, 5))).svg)
        self.assertIn('<svg', _svg_)


if __name__ == '__main__':
    unittest.main()
