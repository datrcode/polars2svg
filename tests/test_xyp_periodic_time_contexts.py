import unittest
import polars as pl
import datetime
import random
from polars2svg import Polars2SVG

class Testxyp_periodic_time_contexts(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()
        self._params_ = {'y':'value', 'dot_size':2.0, 'wxh':(256, 96)}

    # Extract out the periodic time label information
    def caseComment(self, _svg_):
        _start_str_ = 'xyp.__renderContext_periodicTime__():'
        if _start_str_ not in _svg_: return None
        i0 = _svg_.index(_start_str_)
        i1 = _svg_.index('-->', i0)
        s  = _svg_[i0+len(_start_str_):i1].strip()
        return eval('self.p2s.'+s.split('|')[0]), int(s.split('|')[1]), int(s.split('|')[2])


    # Determine the coverage of the periodic time contexts
    def coverageInformation(self, _list_):
        _total_, _seen_, _order_ = 0, set(), []
        for _xyp_ in _list_: 
            _tuple_ = self.caseComment(_xyp_._repr_svg_())
            if _tuple_ is None: 
                print('caseComment() returned None')
                continue
            if    _total_ == 0:          _total_ = _tuple_[2]
            elif _total_ !=  _tuple_[2]: raise ValueError('Periodic case maxes are not uniform')
            _order_.append(_tuple_[1]), _seen_.add(_tuple_[1])
        _missing_ = []
        if len(_seen_) != _total_:
            for i in range(1, _total_+1):
                if i not in _seen_: _missing_.append(i)
        print(f'Coverage: {len(_seen_)}/{_total_} | Order: {_order_} | Missing: {_missing_}')


    # Create a dataframe with time information
    def timeDataFrame(self, year=2010, month=1, day=1, hour=0, minute=0, second=0, n=10):
        _lu_ = {'ts':[], 'value':[]}
        n    = 10
        for i in range(n):
            if   isinstance(year, int):  _year_ = year
            elif isinstance(year, list): _year_ = random.choice(year)
            else:                        _year_ = random.randint(year[0], year[1])

            if   isinstance(month, int):  _month_ = month
            elif isinstance(month, list): _month_ = random.choice(month)
            else:                         _month_ = random.randint(month[0], month[1])

            if   isinstance(day, int):  _day_ = day
            elif isinstance(day, list): _day_ = random.choice(day)
            else:
                if   _month_ in [1, 3, 5, 7, 8, 10, 12]: _day_max_ = 31
                elif _month_ in [4, 6, 9, 11]:           _day_max_ = 30
                else:                                    _day_max_ = 28
                if _day_max_ < day[1]: day = (day[0],_day_max_)
                _day_ = random.randint(day[0], day[1])

            if   isinstance(hour, int):  _hour_ = hour
            elif isinstance(hour, list): _hour_ = random.choice(hour)
            else:                        _hour_ = random.randint(hour[0], hour[1])

            if   isinstance(minute, int):  _min_ = minute
            elif isinstance(minute, list): _min_ = random.choice(minute)
            else:                          _min_ = random.randint(minute[0], minute[1])

            if   isinstance(second, int):  _sec_ = second
            elif isinstance(second, list): _sec_ = random.choice(second)
            else:                          _sec_ = random.randint(second[0], second[1])

            _lu_['ts'].append(f'{_year_:04}-{_month_:02}-{_day_:02} {_hour_:02}:{_min_:02}:{_sec_:02}')
            _lu_['value'].append(random.randint(0, 100))
        return pl.DataFrame(_lu_).with_columns(pl.col('ts').str.to_datetime())

    def test_PT_Qp(self):
        _enum_, _tiles_ = self.p2s.PT_Qp, []
        df    = self.timeDataFrame(month=(1,12))
        _tiles_.append(self.p2s.xyp(df, x=self.p2s.tField('ts', _enum_), **self._params_))
        self.coverageInformation(_tiles_)

    def test_PT_mp(self):
        _enum_, _tiles_ = self.p2s.PT_mp, []
        df    = self.timeDataFrame(year=(2010, 2025), month=(1,12))
        _tiles_.append(self.p2s.xyp(df, x=self.p2s.tField('ts', _enum_), **self._params_))
        self.coverageInformation(_tiles_)


    def test_PT_m_dp(self):
        _enum_, _tiles_ = self.p2s.PT_m_dp, []
        df    = self.timeDataFrame(year=(2010, 2025), month=7, day=(1,6), hour=(0,23))
        _tiles_.append(self.p2s.xyp(df, x=self.p2s.tField('ts', _enum_), **self._params_))
        df    = self.timeDataFrame(year=(2010, 2025), month=(3,4), day=[1,5,10,15,20,25])
        _tiles_.append(self.p2s.xyp(df, x=self.p2s.tField('ts', _enum_), **self._params_))
        df    = self.timeDataFrame(year=(2010, 2025), month=(1,12), day=[1,15])
        _tiles_.append(self.p2s.xyp(df, x=self.p2s.tField('ts', _enum_), **self._params_))
        self.coverageInformation(_tiles_)

    def test_PT_m_d_Hp(self):
        _enum_, _tiles_ = self.p2s.PT_m_d_Hp, []
        df     = self.timeDataFrame(year=(2010, 2025), month=7, day=(1,6), hour=(0,23))
        _tiles_.append(self.p2s.xyp(df, x=self.p2s.tField('ts', _enum_), **self._params_))
        df     = self.timeDataFrame(year=(2010, 2025), month=(3,4), day=[1,5,10,15,20,25])
        _tiles_.append(self.p2s.xyp(df, x=self.p2s.tField('ts', _enum_), **self._params_))
        df     = self.timeDataFrame(year=(2010, 2025), month=(1,12), day=[1,15])
        _tiles_.append(self.p2s.xyp(df, x=self.p2s.tField('ts', _enum_), **self._params_))
        self.coverageInformation(_tiles_)

    def test_PT_DoYp(self):
        _enum_, _tiles_ = self.p2s.PT_DoYp, []
        df     = self.timeDataFrame(year=(2010, 2025), month=(9,12), day=(1,35))
        _tiles_.append(self.p2s.xyp(df, x=self.p2s.tField('ts', _enum_), **self._params_))
        df     = self.timeDataFrame(year=(2010, 2025), month=(6,12), day=(1,35))
        _tiles_.append(self.p2s.xyp(df, x=self.p2s.tField('ts', _enum_), **self._params_))
        df     = self.timeDataFrame(year=(2010, 2025), month=(1,12), day=(1,35))
        _tiles_.append(self.p2s.xyp(df, x=self.p2s.tField('ts', _enum_), **self._params_))
        self.coverageInformation(_tiles_)

    def test_PT_DoWp(self):
        _enum_, _tiles_ = self.p2s.PT_DoWp, []
        df     = self.timeDataFrame(year=(2010, 2025), month=(1,12), day=(1,35))
        _tiles_.append(self.p2s.xyp(df, x=self.p2s.tField('ts', _enum_), **self._params_))
        self.coverageInformation(_tiles_)

    def test_PT_DoW_Hp(self):
        _enum_, _tiles_ = self.p2s.PT_DoW_Hp, []
        df     = self.timeDataFrame(month=(4,4), day=(1,1), hour=(0,23))
        _tiles_.append(self.p2s.xyp(df, x=self.p2s.tField('ts', _enum_), **self._params_))
        df     = self.timeDataFrame(month=(3,3), day=(1,3), hour=(0,23))
        _tiles_.append(self.p2s.xyp(df, x=self.p2s.tField('ts', _enum_), **self._params_))
        df     = self.timeDataFrame(month=(1,12), day=(1,35), hour=(0,23))
        _tiles_.append(self.p2s.xyp(df, x=self.p2s.tField('ts', _enum_), **self._params_))
        self.coverageInformation(_tiles_)

    def test_PT_DoW_H_Mp(self):
        _enum_, _tiles_ = self.p2s.PT_DoW_H_Mp, []
        df     = self.timeDataFrame(month=(9,9), day=(3,3), hour=(12,13), minute=(0,59))
        _tiles_.append(self.p2s.xyp(df, x=self.p2s.tField('ts', _enum_), **self._params_))
        df     = self.timeDataFrame(month=(11,11), day=(9,9), hour=(12,15), minute=(0,59))
        _tiles_.append(self.p2s.xyp(df, x=self.p2s.tField('ts', _enum_), **self._params_))
        df     = self.timeDataFrame(month=(1,1), day=(1,3), hour=(0,23), minute=(0,59))
        _tiles_.append(self.p2s.xyp(df, x=self.p2s.tField('ts', _enum_), **self._params_))
        self.coverageInformation(_tiles_)

    def test_PT_dp(self):
        _enum_, _tiles_ = self.p2s.PT_dp, []
        df     = self.timeDataFrame(month=(1,1), day=(1,3), hour=(0,23), minute=(0,59))
        _tiles_.append(self.p2s.xyp(df, x=self.p2s.tField('ts', _enum_), **self._params_))
        df     = self.timeDataFrame(month=(1,1), day=(1,35), hour=(0,23), minute=(0,59))
        _tiles_.append(self.p2s.xyp(df, x=self.p2s.tField('ts', _enum_), **self._params_))
        self.coverageInformation(_tiles_)

    def test_PT_d_Hp(self):
        _enum_, _tiles_  = self.p2s.PT_d_Hp, []
        df     = self.timeDataFrame(month=(8,8), day=(1,1), hour=(0,23))
        _tiles_.append(self.p2s.xyp(df, x=self.p2s.tField('ts', _enum_), **self._params_))
        df     = self.timeDataFrame(month=(9,9), day=(10,12), hour=(0,23))
        _tiles_.append(self.p2s.xyp(df, x=self.p2s.tField('ts', _enum_), **self._params_))
        df     = self.timeDataFrame(month=(7,7), day=(15,30), hour=(0,23))
        _tiles_.append(self.p2s.xyp(df, x=self.p2s.tField('ts', _enum_), **self._params_))
        df     = self.timeDataFrame(month=(1,1), day=(1,35), hour=(0,23))
        _tiles_.append(self.p2s.xyp(df, x=self.p2s.tField('ts', _enum_), **self._params_))
        self.coverageInformation(_tiles_)

    def test_PT_d_H_Mp(self):
        _enum_, _tiles_ = self.p2s.PT_d_H_Mp, []
        df     = self.timeDataFrame(month=(1,1), day=(20,20), hour=(3,3), minute=(0,12))
        _tiles_.append(self.p2s.xyp(df, x=self.p2s.tField('ts', _enum_), **self._params_))
        df     = self.timeDataFrame(month=(1,1), day=(20,20), hour=(0,3), minute=(0,59))
        _tiles_.append(self.p2s.xyp(df, x=self.p2s.tField('ts', _enum_), **self._params_))
        df     = self.timeDataFrame(month=(1,1), day=(9,10), hour=(0,23), minute=(0,59))
        _tiles_.append(self.p2s.xyp(df, x=self.p2s.tField('ts', _enum_), **self._params_))
        self.coverageInformation(_tiles_)

    def test_PT_Hp(self):
        _enum_, _tiles_ = self.p2s.PT_Hp, []
        df     = self.timeDataFrame(month=(12,12), day=(9,10), hour=(0,12))
        _tiles_.append(self.p2s.xyp(df, x=self.p2s.tField('ts', _enum_), **self._params_))
        df     = self.timeDataFrame(month=(12,12), day=(9,10), hour=(0,23))
        _tiles_.append(self.p2s.xyp(df, x=self.p2s.tField('ts', _enum_), **self._params_))
        self.coverageInformation(_tiles_)

    def test_PT_H_Mp(self):
        _enum_, _tiles_ = self.p2s.PT_H_Mp, []
        df     = self.timeDataFrame(month=(12,12), day=(1,35), hour=(10,11), minute=(0,59))
        _tiles_.append(self.p2s.xyp(df, x=self.p2s.tField('ts', _enum_), **self._params_))
        df     = self.timeDataFrame(month=(12,12), day=(1,35), hour=(10,14), minute=(0,59))
        _tiles_.append(self.p2s.xyp(df, x=self.p2s.tField('ts', _enum_), **self._params_))
        df     = self.timeDataFrame(month=(12,12), day=(9,10), hour=(6,18), minute=(0,59))
        _tiles_.append(self.p2s.xyp(df, x=self.p2s.tField('ts', _enum_), **self._params_))
        df     = self.timeDataFrame(month=(12,12), day=(9,10), hour=(0,22), minute=(0,59))
        _tiles_.append(self.p2s.xyp(df, x=self.p2s.tField('ts', _enum_), **self._params_))
        df     = self.timeDataFrame(month=(12,12), day=(9,10), hour=(0,23), minute=(0,59))
        _tiles_.append(self.p2s.xyp(df, x=self.p2s.tField('ts', _enum_), **self._params_))
        self.coverageInformation(_tiles_)

    def test_PT_H_M_Sp(self):
        _enum_, _tiles_ = self.p2s.PT_H_M_Sp, []
        df     = self.timeDataFrame(month=(12,12), day=(1,35), hour=(10,10), minute=(0,59), second=(0,59))
        _tiles_.append(self.p2s.xyp(df, x=self.p2s.tField('ts', _enum_), **self._params_))
        df     = self.timeDataFrame(month=(12,12), day=(1,35), hour=(10,14), minute=(0,59), second=(0,59))
        _tiles_.append(self.p2s.xyp(df, x=self.p2s.tField('ts', _enum_), **self._params_))
        df     = self.timeDataFrame(month=(12,12), day=(9,10), hour=(6,18), minute=(0,59), second=(0,59))
        _tiles_.append(self.p2s.xyp(df, x=self.p2s.tField('ts', _enum_), **self._params_))
        df     = self.timeDataFrame(month=(12,12), day=(9,10), hour=(0,10), minute=(0,59), second=(0,59))
        _tiles_.append(self.p2s.xyp(df, x=self.p2s.tField('ts', _enum_), **self._params_))
        df     = self.timeDataFrame(month=(12,12), day=(9,10), hour=(0,23), minute=(0,59), second=(0,59))
        _tiles_.append(self.p2s.xyp(df, x=self.p2s.tField('ts', _enum_), **self._params_))
        self.coverageInformation(_tiles_)

    def test_PT_Mp(self):
        _enum_, _tiles_ = self.p2s.PT_Mp, []
        df     = self.timeDataFrame(month=(1,12), day=(9,10), hour=(0,23), minute=(0,5), second=(0,59))
        _tiles_.append(self.p2s.xyp(df, x=self.p2s.tField('ts', _enum_), **self._params_))
        df     = self.timeDataFrame(month=(1,12), day=(9,10), hour=(0,23), minute=(0,22), second=(0,59))
        _tiles_.append(self.p2s.xyp(df, x=self.p2s.tField('ts', _enum_), **self._params_))
        df     = self.timeDataFrame(month=(1,12), day=(9,10), hour=(0,23), minute=(0,44), second=(0,59))
        _tiles_.append(self.p2s.xyp(df, x=self.p2s.tField('ts', _enum_), **self._params_))
        df     = self.timeDataFrame(month=(1,12), day=(9,10), hour=(0,23), minute=(0,59), second=(0,59))
        _tiles_.append(self.p2s.xyp(df, x=self.p2s.tField('ts', _enum_), **self._params_))
        self.coverageInformation(_tiles_)

    def test_PT_M_Sp(self):
        _enum_, _tiles_ = self.p2s.PT_M_Sp, []
        df     = self.timeDataFrame(month=(1,12), day=(9,10), hour=(0,23), minute=(10,10), second=(0,10))
        _tiles_.append(self.p2s.xyp(df, x=self.p2s.tField('ts', _enum_), **self._params_))
        df     = self.timeDataFrame(month=(1,12), day=(9,10), hour=(0,23), minute=(10,10), second=(0,59))
        _tiles_.append(self.p2s.xyp(df, x=self.p2s.tField('ts', _enum_), **self._params_))
        df     = self.timeDataFrame(month=(1,12), day=(9,10), hour=(0,23), minute=(10,15), second=(0,59))
        _tiles_.append(self.p2s.xyp(df, x=self.p2s.tField('ts', _enum_), **self._params_))
        df     = self.timeDataFrame(month=(1,12), day=(9,10), hour=(0,23), minute=(10,20), second=(0,59))
        _tiles_.append(self.p2s.xyp(df, x=self.p2s.tField('ts', _enum_), **self._params_))
        df     = self.timeDataFrame(month=(1,12), day=(9,10), hour=(0,23), minute=(10,35), second=(0,59))
        _tiles_.append(self.p2s.xyp(df, x=self.p2s.tField('ts', _enum_), **self._params_))
        df     = self.timeDataFrame(month=(1,12), day=(9,10), hour=(0,23), minute=(0,59), second=(0,59))
        _tiles_.append(self.p2s.xyp(df, x=self.p2s.tField('ts', _enum_), **self._params_))
        self.coverageInformation(_tiles_)

    def test_PT_Sp(self):
        _enum_, _tiles_ = self.p2s.PT_Sp, []
        df     = self.timeDataFrame(month=(1,12), day=(9,10), hour=(0,23), minute=(0,5), second=(0,5))
        _tiles_.append(self.p2s.xyp(df, x=self.p2s.tField('ts', _enum_), **self._params_))
        df     = self.timeDataFrame(month=(1,12), day=(9,10), hour=(0,23), minute=(0,22), second=(0,22))
        _tiles_.append(self.p2s.xyp(df, x=self.p2s.tField('ts', _enum_), **self._params_))
        df     = self.timeDataFrame(month=(1,12), day=(9,10), hour=(0,23), minute=(0,44), second=(0,44))
        _tiles_.append(self.p2s.xyp(df, x=self.p2s.tField('ts', _enum_), **self._params_))
        df     = self.timeDataFrame(month=(1,12), day=(9,10), hour=(0,23), minute=(0,59), second=(0,59))
        _tiles_.append(self.p2s.xyp(df, x=self.p2s.tField('ts', _enum_), **self._params_))
        self.coverageInformation(_tiles_)

if __name__ == '__main__':
    unittest.main()
