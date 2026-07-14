import polars as pl
import random
import string

__name__ = 'random_dataframe'

def randomDataFrame(n=None, na_probability=0.01):
    _lu_ = {'a':[],'b':[],'c':[],'d':[],'e':[],'f':[],'g':[],'h':[],'i':[],'j':[],'k':[],}
    def randomString(length):
        letters = string.ascii_lowercase + string.ascii_uppercase + string.digits
        return ''.join(random.choice(letters) for i in range(length))
    if n is None: n = random.randint(1000, 10000)
    for x in range(n):
        if random.random() > na_probability or x == (n-1): _lu_['a'].append(random.randint(0,      100))
        else:                                              _lu_['a'].append(None)
        if random.random() > na_probability or x == (n-1): _lu_['b'].append(random.randint(100, 10_000))
        else:                                              _lu_['b'].append(None)
        if random.random() > na_probability or x == (n-1): _lu_['c'].append(random.random())
        else:                                              _lu_['c'].append(None)
        if random.random() > na_probability or x == (n-1): _lu_['d'].append(-50.0 + random.random()*100.0)
        else:                                              _lu_['d'].append(None)
        if random.random() > na_probability or x == (n-1): _lu_['e'].append(-10e8  + 10e9  * random.random())
        else:                                              _lu_['e'].append(None)
        if random.random() > na_probability or x == (n-1): _lu_['f'].append(-10e-9 + 10e-8 * random.random())
        else:                                              _lu_['f'].append(None)
        _year_  = random.randint(1, 2500)
        _month_ = random.randint(1, 12)
        if   _month_ in [1, 3, 5, 7, 8, 10, 12]: _day_ = random.randint(1, 31)
        elif _month_ in [4, 6, 9, 11]:           _day_ = random.randint(1, 30)
        else:                                    _day_ = random.randint(1, 28)
        _hour_   = random.randint(0, 23)
        _min_    = random.randint(0, 59)
        _sec_    = random.randint(0, 59)
        _millis_ = random.randint(0, 999)
        if random.random() > na_probability or x == (n-1): _lu_['g'].append(f'{_year_:04}-{_month_:02}-{_day_:02} {_hour_:02}:{_min_:02}:{_sec_:02}')
        else:                                              _lu_['g'].append(None)
        if random.random() > na_probability or x == (n-1): _lu_['h'].append(f'{_year_:04}-{_month_:02}-{_day_:02}')
        else:                                              _lu_['h'].append(None)
        if random.random() > na_probability or x == (n-1): _lu_['i'].append(f'{_year_:04}-{_month_:02}-{_day_:02} {_hour_:02}:{_min_:02}:{_sec_:02}.{_millis_:03}')
        else:                                              _lu_['i'].append(None)
        if random.random() > na_probability or x == (n-1): _lu_['j'].append(randomString(4))
        else:                                              _lu_['j'].append(None)
        if random.random() > na_probability or x == (n-1): _lu_['k'].append(randomString(8))
        else:                                              _lu_['k'].append(None)
    _schema_ = {
        'a': pl.Int32, 'b': pl.Int32, 'c': pl.Float64, 'd': pl.Float64, 'e': pl.Float64, 'f': pl.Float64,
        'g': pl.Datetime, 'h': pl.Date, 'i': pl.Datetime, 'j': pl.Utf8, 'k': pl.Utf8
    }

    _df_ = pl.DataFrame(_lu_, schema=_schema_)
    return _df_

