import datetime as dt
import unittest

import polars as pl

from polars2svg import Polars2SVG
from timep_dataframes import makeTimeDf, makeDateDf


class TestTimepTimeFields(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    # ── LINEAR enums ─────────────────────────────────────────────────────────

    def test_linear_auto_resolve_various_spreads(self):
        '''Auto-resolution adapts to data spread at a fixed widget width.'''
        df_years  = makeTimeDf(n=50, year=(2000, 2025))
        df_months = makeTimeDf(n=50, year=2023,  month=(1, 12))
        df_days   = makeTimeDf(n=50, year=2023,  month=6,    day=(1, 30))
        df_hours  = makeTimeDf(n=50, year=2023,  month=6,    day=15,   hour=(0, 23))
        df_mins   = makeTimeDf(n=50, year=2023,  month=6,    day=15,   hour=(12, 13), minute=(0, 59))
        df_secs   = makeTimeDf(n=50, year=2023,  month=6,    day=15,   hour=12,  minute=(0, 4), second=(0, 59))
        for df in [df_years, df_months, df_days, df_hours, df_mins, df_secs]:
            self.p2s.timep(df, 'ts', wxh=(512, 128))

    def test_all_linear_enums_datetime(self):
        '''Force every TimeLinearTypeP on datetime data matched to that granularity.

        The key constraint: the datetime_range spine size must stay small, so the
        data time-span for each enum is kept proportional to the bin size.
        '''
        p = self.p2s
        # (enum, kwargs for makeTimeDf) — each combination produces ≤ ~300 spine bins
        _cases_ = [
            (p.LT_Yp,             dict(year=(2010, 2025))),
            (p.LT_Y_Qp,           dict(year=(2020, 2025), month=(1, 12))),
            (p.LT_Y_mp,           dict(year=(2022, 2024), month=(1, 12))),
            (p.LT_Y_m_dp,         dict(year=2023,         month=(1, 12),  day=(1, 28))),
            (p.LT_Y_m_d_4Hp,      dict(year=2023,         month=6,        day=(1, 28),   hour=(0, 23))),
            (p.LT_Y_m_d_Hp,       dict(year=2023,         month=6,        day=(1, 7),    hour=(0, 23))),
            (p.LT_Y_m_d_H_15Mp,   dict(year=2023,         month=6,        day=15,        hour=(0, 23),  minute=(0, 59))),
            (p.LT_Y_m_d_H_Mp,     dict(year=2023,         month=6,        day=15,        hour=(12, 13), minute=(0, 59))),
            (p.LT_Y_m_d_H_M_15Sp, dict(year=2023,         month=6,        day=15,        hour=12,       minute=(0, 4),  second=(0, 59))),
            (p.LT_Y_m_d_H_M_Sp,   dict(year=2023,         month=6,        day=15,        hour=12,       minute=(0, 4),  second=(0, 59))),
        ]
        for _enum_, _kw_ in _cases_:
            df = makeTimeDf(n=50, **_kw_)
            self.p2s.timep(df, ('ts', _enum_), wxh=(512, 128))

    def test_coarse_linear_enums_on_date_column(self):
        '''Date (pl.Date) columns support linear enums up to LT_Y_m_dp.'''
        df = makeDateDf(n=100, year=(2020, 2025), month=(1, 12))
        _coarse_ = [
            self.p2s.LT_Yp,
            self.p2s.LT_Y_Qp,
            self.p2s.LT_Y_mp,
            self.p2s.LT_Y_m_dp,
        ]
        for _enum_ in _coarse_:
            self.p2s.timep(df, ('dt', _enum_), wxh=(512, 128))

    def test_granularity_auto_coarsens_for_narrow_widget(self):
        '''Narrower widget forces coarser auto-selected granularity.'''
        # ~10 years of data → yearly=10 bins, quarterly=~40, monthly=~120, daily=~3650
        df = makeTimeDf(n=200, year=(2015, 2025), month=(1, 12), day=(1, 28))
        t_wide   = self.p2s.timep(df, 'ts', wxh=(1024, 128))
        t_narrow = self.p2s.timep(df, 'ts', wxh=(64,   128))
        _order_ = [
            self.p2s.LT_Yp, self.p2s.LT_Y_Qp, self.p2s.LT_Y_mp,
            self.p2s.LT_Y_m_dp, self.p2s.LT_Y_m_d_4Hp, self.p2s.LT_Y_m_d_Hp,
            self.p2s.LT_Y_m_d_H_15Mp, self.p2s.LT_Y_m_d_H_Mp,
            self.p2s.LT_Y_m_d_H_M_15Sp, self.p2s.LT_Y_m_d_H_M_Sp,
        ]
        wide_idx   = _order_.index(t_wide._time_enum_)
        narrow_idx = _order_.index(t_narrow._time_enum_)
        self.assertLessEqual(narrow_idx, wide_idx)

    # ── PERIODIC enums ────────────────────────────────────────────────────────

    def test_all_periodic_enums_datetime(self):
        '''All TimePeriodicTypeP enums render without exception on datetime data.'''
        df = makeTimeDf(n=100, year=(2020, 2025), month=(1, 12), day=(1, 28),
                        hour=(0, 23), minute=(0, 59), second=(0, 59))
        for _enum_ in self.p2s.TimePeriodicTypeP:
            self.p2s.timep(df, ('ts', _enum_), wxh=(512, 128))

    def test_periodic_enums_date_column_day_and_coarser(self):
        '''Periodic enums that only need date components work on pl.Date columns.

        BUG: enums that extract hour/minute/second (PT_m_d_Hp, PT_DoW_Hp,
        PT_DoW_H_Mp, PT_d_Hp, PT_d_H_Mp, PT_H_Mp, PT_H_M_Sp, PT_M_Sp,
        PT_Sp) raise InvalidOperationError because .dt.hour() is not supported
        on pl.Date.  Only the date-compatible subset is tested here.
        '''
        _date_compatible_ = [
            self.p2s.PT_Qp,    # quarter: month only
            self.p2s.PT_mp,    # month
            self.p2s.PT_m_dp,  # day of month
            self.p2s.PT_DoYp,  # ordinal day of year
            self.p2s.PT_DoWp,  # day of week
            self.p2s.PT_dp,    # day of month
        ]
        df = makeDateDf(n=100, year=(2020, 2025), month=(1, 12))
        for _enum_ in _date_compatible_:
            self.p2s.timep(df, ('dt', _enum_), wxh=(512, 128))

    def test_periodic_enum_repr_svg(self):
        df = makeTimeDf(n=100, year=(2020, 2025), month=(1, 12))
        for _enum_ in self.p2s.TimePeriodicTypeP:
            t = self.p2s.timep(df, ('ts', _enum_), wxh=(256, 96))
            self.assertIn('<svg', t._repr_svg_())


class TestTimepTimeLevels(unittest.TestCase):
    """timeLevels(): the levels an interactive view may offer.  Its job is to keep a picker
    from building a spine the plot has no room for -- PT_H_M_Sp is 86,400 bins -- and from
    offering a level the data cannot resolve or that draws a single bar."""

    def setUp(self):
        self.p2s = Polars2SVG()

    def _hourly(self, days=3):
        return pl.DataFrame({'ts': pl.datetime_range(dt.datetime(2026, 1, 1), dt.datetime(2026, 1, days, 23),
                                                     '1h', eager=True)})

    def _names(self, levels):
        return [_e_.name for _e_ in levels]

    def test_linear_levels_stop_where_the_spine_outgrows_the_plot(self):
        """72 hourly bins fit 512 px (two px a bar) but not 128, so hourly drops out there."""
        _p_ = self.p2s
        _wide_   = _p_.timep(self._hourly(), 'ts').timeLevels()
        _narrow_ = _p_.timep(self._hourly(), 'ts', wxh=(128, 256)).timeLevels()
        _linear_ = lambda levels: [_e_ for _e_ in levels if isinstance(_e_, _p_.TimeLinearTypeP)]  # noqa: E731
        self.assertEqual(_linear_(_wide_),   [_p_.LT_Y_m_dp, _p_.LT_Y_m_d_4Hp, _p_.LT_Y_m_d_Hp])
        self.assertEqual(_linear_(_narrow_), [_p_.LT_Y_m_dp, _p_.LT_Y_m_d_4Hp])

    def test_the_auto_level_is_the_finest_linear_level_offered(self):
        """Same budget and cap as the auto resolution, which settles on one of these."""
        _p_ = self.p2s
        for _df_, _wxh_ in ((self._hourly(), (512, 256)), (self._hourly(), (128, 256)),
                            (makeTimeDf(n=200, year=(2015, 2025), month=(1, 12), day=(1, 28)), (512, 256))):
            with self.subTest(wxh=_wxh_, rows=len(_df_)):
                _t_ = _p_.timep(_df_, 'ts', wxh=_wxh_)
                _linear_ = [_e_ for _e_ in _t_.timeLevels() if isinstance(_e_, _p_.TimeLinearTypeP)]
                self.assertEqual(_linear_[-1], _t_._time_enum_)

    def test_a_periodic_cycle_has_to_fit_the_plot_too(self):
        """Day of year is 366 bars: too many for 512 px, room enough at 1024.  The
        second-of-the-hour cycle never fits a default view."""
        _year_ = pl.DataFrame({'ts': pl.datetime_range(dt.datetime(2025, 1, 1), dt.datetime(2025, 12, 31, 23),
                                                       '1h', eager=True)})
        self.assertNotIn('PT_DoYp', self._names(self.p2s.timep(_year_, 'ts').timeLevels()))
        self.assertIn('PT_DoYp', self._names(self.p2s.timep(_year_, 'ts', wxh=(1024, 256)).timeLevels()))
        _secs_ = pl.DataFrame({'ts': pl.datetime_range(dt.datetime(2026, 1, 1), dt.datetime(2026, 1, 1, 1),
                                                       '1s', eager=True)})
        self.assertNotIn('PT_H_M_Sp', self._names(self.p2s.timep(_secs_, 'ts').timeLevels()))

    def test_nothing_finer_than_the_data_resolves(self):
        """A Date column has no hour -- polars cannot even compute one, which is why these
        are dropped before the select -- and midnight-only timestamps never move theirs."""
        _dates_ = pl.DataFrame({'d': pl.date_range(dt.date(2025, 1, 1), dt.date(2025, 3, 31), '1d', eager=True)})
        _midnights_ = pl.DataFrame({'ts': pl.datetime_range(dt.datetime(2025, 1, 1), dt.datetime(2025, 3, 31),
                                                            '1d', eager=True)})
        for _df_, _col_ in ((_dates_, 'd'), (_midnights_, 'ts')):
            with self.subTest(column=_col_):
                _names_ = self._names(self.p2s.timep(_df_, _col_).timeLevels())
                self.assertIn('LT_Y_m_dp', _names_)
                self.assertFalse({'LT_Y_m_d_4Hp', 'LT_Y_m_d_Hp', 'PT_Hp', 'PT_DoW_Hp'} & set(_names_))

    def test_a_level_that_draws_one_bar_is_left_out(self):
        """Three days of data are one year, one quarter and one month, whichever way the
        calendar is cut."""
        _names_ = self._names(self.p2s.timep(self._hourly(), 'ts').timeLevels())
        self.assertFalse({'LT_Yp', 'LT_Y_Qp', 'LT_Y_mp', 'PT_Qp', 'PT_mp'} & set(_names_))
        self.assertEqual(self.p2s.timep(pl.DataFrame({'ts': [dt.datetime(2026, 1, 1)] * 3}), 'ts').timeLevels(), [])

    def test_it_is_computed_on_the_widest_frame(self):
        """df_orig by default: a level offered there is safe at every stack level, where a
        narrower frame alone would allow finer ones."""
        _t_ = self.p2s.timep(self._hourly(days=10), 'ts', wxh=(256, 256))
        self.assertEqual(_t_.timeLevels(), _t_.timeLevels(_t_.df_orig))
        self.assertNotIn('LT_Y_m_d_Hp', self._names(_t_.timeLevels()))
        self.assertIn('LT_Y_m_d_Hp', self._names(_t_.timeLevels(self._hourly(days=2))))


if __name__ == '__main__':
    unittest.main()
