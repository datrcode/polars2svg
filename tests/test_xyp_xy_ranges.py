import unittest
import polars as pl
import numpy as np
from datetime import datetime
import random
from polars2svg import Polars2SVG

class Testxyp_xy_ranges(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()
        rng = np.random.default_rng()
        n   = 1_000
        self.df  = pl.DataFrame({'a':rng.normal(loc=2500, scale=10, size=n), 
                                 'b':rng.normal(loc=500,  scale=3,  size=n)}) 
        
    def test_withoutRanges(self):
        _xyp0_ = self.p2s.xyp(self.df, 'a', 'b', dot_size=1.25, color=self.p2s.CROW_MAGNITUDEp)
        _xyp1_ = self.p2s.xyp(self.df, 'a', 'b', dot_size=1.25, color=self.p2s.CROW_MAGNITUDEp, draw_context=False)
        _xyp0_ = self.p2s.xyp(self.df, 'a', 'b', dot_size=3,    color=self.p2s.CROW_MAGNITUDEp)
        _xyp1_ = self.p2s.xyp(self.df, 'a', 'b', dot_size=3,    color=self.p2s.CROW_MAGNITUDEp, draw_context=False)

    def test_subsetRangesFloat(self):
        _xyp0_ = self.p2s.xyp(self.df, 'a', 'b', dot_size=1.25, color=self.p2s.CROW_STRETCHEDp, x_range=(2480,2500))
        _xyp1_ = self.p2s.xyp(self.df, 'a', 'b', dot_size=1.25, color=self.p2s.CROW_STRETCHEDp,                      y_range=(500, 505))
        _xyp2_ = self.p2s.xyp(self.df, 'a', 'b', dot_size=1.25, color=self.p2s.CROW_STRETCHEDp, x_range=(2480,2500), y_range=(500, 505))

    def test_externalRangesFloat(self):
        _xyp0_ = self.p2s.xyp(self.df, 'a', 'b', dot_size=1.25, color=self.p2s.CROW_STRETCHEDp, x_range=(2_200, 2_800))
        _xyp1_ = self.p2s.xyp(self.df, 'a', 'b', dot_size=1.25, color=self.p2s.CROW_STRETCHEDp,                         y_range=(450, 600))
        _xyp2_ = self.p2s.xyp(self.df, 'a', 'b', dot_size=1.25, color=self.p2s.CROW_STRETCHEDp, x_range=(2_200, 2_800), y_range=(450, 600))

    def test_externalRangesFloat_distributions(self):
        _param_ = {'x_distributions':self.p2s.ROW_COUNTp, 'y_distributions':self.p2s.ROW_COUNTp}
        _xyp0_ = self.p2s.xyp(self.df, 'a', 'b', dot_size=1.25, color=self.p2s.CROW_STRETCHEDp, x_range=(2_200, 2_800),                     **_param_)
        _xyp1_ = self.p2s.xyp(self.df, 'a', 'b', dot_size=1.25, color=self.p2s.CROW_STRETCHEDp,                         y_range=(450, 600), **_param_)
        _xyp2_ = self.p2s.xyp(self.df, 'a', 'b', dot_size=1.25, color=self.p2s.CROW_STRETCHEDp, x_range=(2_200, 2_800), y_range=(450, 600), **_param_)

    def test_outOfRangeFloat(self):
        _xyp0_ = self.p2s.xyp(self.df, 'a', 'b', dot_size=1.25, color=self.p2s.CROW_STRETCHEDp, x_range=(12_200, 14_800))
        _xyp1_ = self.p2s.xyp(self.df, 'a', 'b', dot_size=1.25, color=self.p2s.CROW_STRETCHEDp,                           y_range=(45_000, 46_000))
        _xyp2_ = self.p2s.xyp(self.df, 'a', 'b', dot_size=1.25, color=self.p2s.CROW_STRETCHEDp, x_range=(12_200, 14_800), y_range=(45_000, 46_000))


    def test_outOfRangeFloat_distributions(self):
        _param_ = {'x_distributions':self.p2s.ROW_COUNTp, 'y_distributions':self.p2s.ROW_COUNTp}
        _xyp0_ = self.p2s.xyp(self.df, 'a', 'b', dot_size=1.25, color=self.p2s.CROW_STRETCHEDp, x_range=(12_200, 14_800),                           **_param_)
        _xyp1_ = self.p2s.xyp(self.df, 'a', 'b', dot_size=1.25, color=self.p2s.CROW_STRETCHEDp,                           y_range=(45_000, 46_000), **_param_)
        _xyp2_ = self.p2s.xyp(self.df, 'a', 'b', dot_size=1.25, color=self.p2s.CROW_STRETCHEDp, x_range=(12_200, 14_800), y_range=(45_000, 46_000), **_param_)

    def test_outOfRangeFloat_distributions2(self):
        _param_ = {'x_distributions':'a', 'y_distributions':'b'}
        _xyp3_ = self.p2s.xyp(self.df, 'a', 'b', dot_size=1.25, color=self.p2s.CROW_STRETCHEDp, x_range=(12_200, 14_800),                           **_param_)
        _xyp4_ = self.p2s.xyp(self.df, 'a', 'b', dot_size=1.25, color=self.p2s.CROW_STRETCHEDp,                           y_range=(45_000, 46_000), **_param_)
        _xyp5_ = self.p2s.xyp(self.df, 'a', 'b', dot_size=1.25, color=self.p2s.CROW_STRETCHEDp, x_range=(12_200, 14_800), y_range=(45_000, 46_000), **_param_)

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

if __name__ == '__main__':
    unittest.main()
