import polars as pl

from datetime import timedelta

from .exceptions import InvalidSpecError, Polars2SVGError

class P2STimeMixin:
    def __init__(self):
        pass

    def __p2s_time_mixin_init__(self):
        pass

    #
    # polarsOperationForEnum
    # - see also timePeriodicHumanReadable()
    #
    def polarsOperationForEnum(self, column, _enum_):
        if isinstance(column, tuple):
            if len(column) == 1: column = column[0]
            else: raise InvalidSpecError(f'XYp.polarsOperationForTimeEnums(): column must be a string {column=}')
        # The PT_m_dp / PT_m_d_Hp transforms below remap every timestamp into a single
        # template year before taking the ordinal day.  2024 is used because it is a leap
        # year (the longest possible year), so Feb-29 data still maps to a valid day; as a
        # consequence the result differs slightly from PT_DoYp for non-leap-year dates.
        _lu_ = {
            self.LT_Yp:              pl.col(column).dt.truncate('1y'),   # returns a date
            self.LT_Y_Qp:            pl.col(column).dt.truncate('1q'),   # returns a date
            self.LT_Y_mp:            pl.col(column).dt.truncate('1mo'),  # returns a date
            self.LT_Y_m_dp:          pl.col(column).dt.truncate('1d'),   # returns a date
            self.LT_Y_m_d_4Hp:       pl.col(column).dt.truncate('4h'),   # returns a date
            self.LT_Y_m_d_Hp:        pl.col(column).dt.truncate('1h'),   # returns a date
            self.LT_Y_m_d_H_15Mp:    pl.col(column).dt.truncate('15m'),  # returns a date
            self.LT_Y_m_d_H_Mp:      pl.col(column).dt.truncate('1m'),   # returns a date
            self.LT_Y_m_d_H_M_15Sp:  pl.col(column).dt.truncate('15s'),  # returns a date
            self.LT_Y_m_d_H_M_Sp:    pl.col(column).dt.truncate('1s'),   # returns a date

            self.PT_m_dp:         pl.col(column).dt.replace(year=2024).dt.ordinal_day().cast(pl.Int64),                                                # template-year remap (see note above)
            self.PT_m_d_Hp:       pl.col(column).dt.replace(year=2024).dt.ordinal_day().cast(pl.Int64) * 24 + pl.col(column).dt.hour().cast(pl.Int64), # template-year remap (see note above)

            self.PT_Qp:           pl.col(column).dt.quarter()    .cast(pl.Int64),  # this and below return integers
            self.PT_mp:           pl.col(column).dt.month()      .cast(pl.Int64),
            self.PT_DoYp:         pl.col(column).dt.ordinal_day().cast(pl.Int64),
            self.PT_DoWp:         pl.col(column).dt.weekday()    .cast(pl.Int64),
            self.PT_DoW_Hp:       pl.col(column).dt.weekday()    .cast(pl.Int64) * 24      + pl.col(column).dt.hour()  .cast(pl.Int64),
            self.PT_DoW_H_Mp:     pl.col(column).dt.weekday()    .cast(pl.Int64) * 24 * 60 + pl.col(column).dt.hour()  .cast(pl.Int64) * 60 + pl.col(column).dt.minute().cast(pl.Int64),
            self.PT_dp:           pl.col(column).dt.day()        .cast(pl.Int64),
            self.PT_d_Hp:         pl.col(column).dt.day()        .cast(pl.Int64) * 24      + pl.col(column).dt.hour()  .cast(pl.Int64),
            self.PT_d_H_Mp:       pl.col(column).dt.day()        .cast(pl.Int64) * 24 * 60 + pl.col(column).dt.hour()  .cast(pl.Int64) * 60 + pl.col(column).dt.minute().cast(pl.Int64),
            self.PT_Hp:           pl.col(column).dt.hour()       .cast(pl.Int64),
            self.PT_H_Mp:         pl.col(column).dt.hour()       .cast(pl.Int64) * 60      + pl.col(column).dt.minute().cast(pl.Int64),
            self.PT_H_M_Sp:       pl.col(column).dt.hour()       .cast(pl.Int64) * 60 * 60 + pl.col(column).dt.minute().cast(pl.Int64) * 60 + pl.col(column).dt.second().cast(pl.Int64),
            self.PT_Mp:           pl.col(column).dt.minute()     .cast(pl.Int64),
            self.PT_M_Sp:         pl.col(column).dt.minute()     .cast(pl.Int64) * 60      + pl.col(column).dt.second().cast(pl.Int64),
            self.PT_Sp:           pl.col(column).dt.second()     .cast(pl.Int64),
        }
        return _lu_[_enum_]

    #
    # __monthLookup__()
    #
    def __monthLookup__(self):
        return {
            1: 'jan', 2: 'feb', 3: 'mar',  4: 'apr',  5: 'may',  6: 'jun',
            7: 'jul', 8: 'aug', 9: 'sep', 10: 'oct', 11: 'nov', 12: 'dec',
        }
    
    #
    # __daysInMonth__()
    # - the most days in a month ... i.e., in a leap year
    def __maxDaysInMonthLookup__(self):
        return {
            1: 31, 2: 29, 3: 31,  4: 30,  5: 31,  6: 30,
            7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31,
        }

    #
    # __daysInMonth__() - return the number of days in a month
    #
    def __daysInMonth__(self, year, month):
        _max_days_lu_ = self.__maxDaysInMonthLookup__()
        _max_day_     = _max_days_lu_[month]
        if month == 2:
            if   year%400 == 0: _max_day_ = 29
            elif year%100 == 0: _max_day_ = 28
            elif year%4   == 0: _max_day_ = 29
            else:               _max_day_ = 28
        return _max_day_

    #
    # __daysOfWeekLookup__()
    #
    def __daysOfWeekLookup__(self):
        return {
            1: 'mon', 2: 'tue', 3: 'wed', 4: 'thu', 5: 'fri', 6: 'sat', 7: 'sun',
        }

    #
    # humanReadablePeriodicTimeDelta()
    #
    def humanReadablePeriodicTimeDelta(self, _diff_, _enum_):
        if   _diff_ == 0: return ''
        elif _enum_ == self.PT_Qp:        # (1,     4)          # quarters ... i.e., 3 months
            return f'{_diff_}q'
        elif _enum_ == self.PT_mp:        # (1,     12)         # months
            return f'{_diff_}mo'
        elif _enum_ == self.PT_m_dp:      # (1,     366)        # days
            return f'{_diff_}d'
        elif _enum_ == self.PT_m_d_Hp:    # (24,    367*24-1)   # hours
            _days_  = _diff_//24
            _hours_ = _diff_%24
            if   _days_ == 0: return f'{_hours_}h'
            elif _days_ > 10: return f'{_days_}d'
            else:             return f'{_days_}d {_hours_}h'
        elif _enum_ == self.PT_DoYp:      # (1,     366)        # days
            return f'{_diff_}d'
        elif _enum_ == self.PT_DoWp:      # (1,     7)          # days
            return f'{_diff_}d'
        elif _enum_ == self.PT_DoW_Hp:    # (24,    24*8-1)     # hours
            _days_  = _diff_//24
            _hours_ = _diff_%24
            return f'{_days_}d {_hours_}h'
        elif _enum_ == self.PT_DoW_H_Mp:  # (24*60, 24*60*8-1)  # minutes
            return self.humanReadableTimeDelta(timedelta(seconds=_diff_*60))
        elif _enum_ == self.PT_dp:        # (1,     31)         # days
            return f'{_diff_}d'
        elif _enum_ == self.PT_d_Hp:      # (24,    24*32-1)    # hours
            _days_  = _diff_//24
            _hours_ = _diff_%24
            if   _days_ == 0: return f'{_hours_}h'
            elif _days_ > 10: return f'{_days_}d'
            else:             return f'{_days_}d {_hours_}h'
        elif _enum_ == self.PT_d_H_Mp:    # (24*60, 32*24*60-1) # minutes
            return self.humanReadableTimeDelta(timedelta(seconds=_diff_*60))
        elif _enum_ == self.PT_Hp:        # (0,     23)         # hours
            return f'{_diff_}h'
        elif _enum_ == self.PT_H_Mp:      # (0,     24*60-1)    # minutes
            _hours_   = _diff_//60
            _minutes_ = _diff_%60
            if   _hours_ == 0: return f'{_minutes_}m'
            elif _hours_ > 10: return f'{_hours_}h'
            else:              return f'{_hours_}h {_minutes_}m'
        elif _enum_ == self.PT_H_M_Sp:    # (0,     24*60*60-1) # seconds
            return self.humanReadableTimeDelta(timedelta(seconds=_diff_))
        elif _enum_ == self.PT_Mp:        # (0,     59)         # minutes
            return f'{_diff_}m'
        elif _enum_ == self.PT_M_Sp:      # (0,     60*60-1)    # seconds
            _minutes_ = _diff_//60
            _seconds_ = _diff_%60
            if   _minutes_ == 0: return f'{_seconds_}s'
            elif _minutes_ > 10: return f'{_minutes_}m'
            else:                return f'{_minutes_}m {_seconds_}s'
        elif _enum_ == self.PT_Sp:        # (0,     59)         # seconds
            return f'{_diff_}s'
        else: raise ValueError(f'polars2svg.humanReadablePeriodicTimeDelta(): unknown periodic time delta enum {_enum_}')

    #
    # humanReadableTimeDelta() - return a short human readable time delta
    #
    def humanReadableTimeDelta(self, td):
        _seconds_ = int(td.total_seconds())
        # Years
        _seconds_in_a_year_ = int(24*60*60*(365*3+366*1)/4.0)
        _years_   = _seconds_ // _seconds_in_a_year_
        _seconds_ = _seconds_ % _seconds_in_a_year_
        if _years_ > 10: return f'{_years_}y'
        # Months
        _seconds_in_a_month_ = int(24*60*60*30)
        _months_  = _seconds_ // _seconds_in_a_month_
        _seconds_ = _seconds_ % _seconds_in_a_month_
        if   _years_ > 0 and _months_ ==  0: return f'{_years_}y'
        elif _years_ > 0:                    return f'{_years_}y {_months_}mo'
        # _days_
        _days_    = _seconds_ // (24*60*60)
        _seconds_ = _seconds_ % (24*60*60)
        if   _months_ >  2 and _days_ > 27: return f'{_months_+1}mo'
        elif _months_ >  2 and _days_ <= 3: return f'{_months_}mo'
        elif _months_ >  2:                 return f'{_months_}mo {_days_}d'
        # ... need to recenter at this point ...
        _seconds_ = int(td.total_seconds())
        _days_    = _seconds_ // (24*60*60)
        _seconds_ = _seconds_ %  (24*60*60)
        # _hours_
        _hours_   = _seconds_ // (60*60)
        _seconds_ = _seconds_ %  (60*60)
        _minutes_ = _seconds_ // 60
        _seconds_ = _seconds_ %  60
        if   _days_ > 20 and _hours_ > 20: return f'{_days_+1}d'
        elif _days_ > 20 and _hours_ <  4: return f'{_days_}d'
        elif _days_ > 0:
            if   _hours_ == 0 and _minutes_ > 30: return f'{_days_}d 1h'
            elif _hours_ == 0:                    return f'{_days_}d'
            elif                  _minutes_ > 30: return f'{_days_}d {_hours_+1}h'
            else:                                 return f'{_days_}d {_hours_}h'
        # _minutes_
        if   _hours_ > 20 and _minutes_ >  50: return f'{_hours_+1}h'
        elif _hours_ > 20 and _minutes_ <  10: return f'{_hours_}h'
        elif _hours_ > 0  and _minutes_ <=  5: return f'{_hours_}h'
        elif _hours_ > 0  and _minutes_ >= 55: return f'{_hours_+1}h'
        elif _hours_ > 0:                      return f'{_hours_}h {_minutes_}m'
        # _seconds_
        if   _minutes_ > 30 and _seconds_ > 30: return f'{_minutes_+1}m'
        elif _minutes_ > 30:                    return f'{_minutes_}m'
        elif _minutes_ > 0 and  _seconds_ == 0: return f'{_minutes_}m'
        elif _minutes_ >  0:                    return f'{_minutes_}m {_seconds_}s'
        # _seconds_ or smaller
        _seconds_ = round(td.total_seconds(), 1)
        if _seconds_ > 1.0: return f'{_seconds_}s'
        return f'{round(td.total_seconds(), 2)}s'

    #
    # timePeriodicRange() - return the range for a time periodic type
    #
    def timePeriodicRange(self, _enum_): return self.periodic_ranges[_enum_]

    #
    # timePeriodicHumanReadable() - return a human readable string for a time periodic type
    # - see also polarsOperationForEnum
    #
    def timePeriodicHumanReadable(self, value, _enum_):
        _months_   = self.__monthLookup__()
        _max_days_ = self.__maxDaysInMonthLookup__() # max days (in a year with a leap year)
        _dow_      = self.__daysOfWeekLookup__()
        # Handle the degenerate cases
        if _enum_ in [self.TimePeriodicTypeP.PT_m_dp, self.TimePeriodicTypeP.PT_m_d_Hp]:
            if   _enum_ == self.TimePeriodicTypeP.PT_m_dp:
                _days_   = value
                _append_ = ''
            elif _enum_ == self.TimePeriodicTypeP.PT_m_d_Hp:
                _hours_  = value % 24
                _days_   = value // 24
                _append_ = f' {_hours_}h'
            for _month_ in range(1, 13):
                if _days_ <= _max_days_[_month_]: return f'{_months_[_month_]}-{_days_:02}{_append_}'
                else:                             _days_ -= _max_days_[_month_]
            raise Polars2SVGError(f'XYp.timePeriodicHumanReadable(): {value=}, {_enum_=} ... degenerative case exceeded months (should not happen)')
        # Handle the straightforward cases
        _lu_ = {
            self.TimePeriodicTypeP.PT_Qp:       lambda x: f'q{x}',
            self.TimePeriodicTypeP.PT_mp:       lambda x: _months_[x],
            self.TimePeriodicTypeP.PT_DoYp:     lambda x: str(x), # day of year (okay)
            self.TimePeriodicTypeP.PT_DoWp:     lambda x: _dow_[x],
            self.TimePeriodicTypeP.PT_DoW_Hp:   lambda x: _dow_[x//24]      + f' {x%24:02}h',
            self.TimePeriodicTypeP.PT_DoW_H_Mp: lambda x: _dow_[x//(24*60)] + f' {x%(24*60)//60:02}:{x%(24*60)%60:02}',
            self.TimePeriodicTypeP.PT_dp:       lambda x: f'{x:02}d',
            self.TimePeriodicTypeP.PT_d_Hp:     lambda x: f'{x//24:02} {x%24:02}h',
            self.TimePeriodicTypeP.PT_d_H_Mp:   lambda x: f'{x//(24*60):02} {x%(24*60)//60:02}:{x%(24*60)%60:02}',
            self.TimePeriodicTypeP.PT_Hp:       lambda x: f'{x:02}h',
            self.TimePeriodicTypeP.PT_H_Mp:     lambda x: f'{x//60:02}:{x%60:02}m',
            self.TimePeriodicTypeP.PT_H_M_Sp:   lambda x: f'{x//3600:02}:{x%3600//60:02}:{x%60:02}',
            self.TimePeriodicTypeP.PT_Mp:       lambda x: f'{x:02}m',
            self.TimePeriodicTypeP.PT_M_Sp:     lambda x: f'{x//60:02}:{x%60:02}s',
            self.TimePeriodicTypeP.PT_Sp:       lambda x: f'{x:02}s',
        }
        return _lu_[_enum_](value)

