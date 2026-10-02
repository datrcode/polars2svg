#
# xyp's periodic-time axis (__renderContext_periodicTime__()) -- every case of every
# periodic enum, and the truth of every label it draws.
#
# The axis picks one of up to six label cases from a single number: how many pixels
# one time unit gets.  So a case is reached by the width of the time window alone,
# and each window below is fixed to land in the middle of its case's band at the
# default 256x96 (plot width 241 px).  These tests used to build unseeded random
# windows, print which cases they reached and assert nothing; 4-6 of 17 missed a
# case on any given run, a different set each time, and all 17 passed (PLANNING.md
# V11).
#
# Every label must be a true reading of the time at the pixel where it is drawn -- a
# month name on the 1st, an hour on the hour, a minute at that minute.  The window
# spans the plot frame exactly, so the frame fixes the pixel -> time map, and the two
# data rows (one at each end of the window) have to sit on the frame's edges: the dots
# and the axis agree about where the window is.  The readings are computed here from
# the calendar, not from the framework's own transform or formatting, so a label drawn
# in the right place with the wrong text, or the right text in the wrong place, fails
# either way.
#
import unittest
from datetime import date, datetime, timedelta
import math
import re

import polars as pl
from polars2svg import Polars2SVG

_MON_ = {1: 'jan', 2: 'feb', 3: 'mar', 4: 'apr', 5: 'may', 6: 'jun', 7: 'jul', 8: 'aug', 9: 'sep', 10: 'oct', 11: 'nov', 12: 'dec'}
_DOW_ = {1: 'mon', 2: 'tue', 3: 'wed', 4: 'thu', 5: 'fri', 6: 'sat', 7: 'sun'}

# A label's x is written to 0.1 px (+-0.05), and the frame it is mapped through is
# written unrounded, so a true label lies within 0.1 px of its time.
_TOL_PX_ = 0.1

def _dt_(s: str) -> datetime: return datetime.fromisoformat(s)

#
# One window per case, in case order: window i must draw case i+1 of len(windows).
# 2024-01-01 is a Monday in a leap year -- ordinal day 1 of the template year that
# PT_m_dp / PT_m_d_Hp remap every date onto, and the start of a PT_DoW_* week.
# PT_DoYp uses 2023 so that its ordinal days are not the template year's.
#
_WINDOWS_ = {
    'PT_Qp':       [('2024-02-10 00:00', '2024-11-20 00:00')],
    'PT_mp':       [('2024-01-15 00:00', '2024-12-10 00:00')],
    'PT_m_dp':     [('2024-02-20 00:00', '2024-03-08 00:00'),     # 17 days, across Feb 29
                    ('2024-03-03 00:00', '2024-05-01 00:00'),     # 59 days
                    ('2024-02-10 00:00', '2024-11-20 00:00')],    # 284 days
    'PT_m_d_Hp':   [('2024-02-22 05:00', '2024-03-05 17:00'),     # 12.5 days, across Feb 29
                    ('2024-03-03 00:00', '2024-06-01 00:00'),     # 90 days
                    ('2024-01-20 00:00', '2024-12-10 00:00')],    # 325 days
    'PT_DoYp':     [('2023-02-10 00:00', '2023-04-30 00:00'),     # 79 days
                    ('2023-02-10 00:00', '2023-08-15 00:00'),     # 186 days
                    ('2023-01-05 00:00', '2023-12-20 00:00')],    # 349 days
    'PT_DoWp':     [('2024-01-01 00:00', '2024-01-07 00:00')],
    'PT_DoW_Hp':   [('2024-01-02 14:00', '2024-01-03 06:00'),     # 16 hours, across midnight
                    ('2024-01-02 06:00', '2024-01-05 18:00'),     # 84 hours
                    ('2024-01-01 03:00', '2024-01-07 20:00')],    # 161 hours
    'PT_DoW_H_Mp': [('2024-01-02 23:20', '2024-01-03 01:00'),     # 100 minutes, across midnight
                    ('2024-01-02 15:00', '2024-01-03 09:00'),     # 18 hours, across midnight
                    ('2024-01-02 08:00', '2024-01-05 20:00')],    # 3.5 days
    'PT_dp':       [('2024-03-04 00:00', '2024-03-20 00:00'),     # 16 days
                    ('2024-03-02 00:00', '2024-03-30 00:00')],    # 28 days
    'PT_d_Hp':     [('2024-03-09 14:00', '2024-03-10 08:00'),     # 18 hours, across midnight
                    ('2024-03-10 04:00', '2024-03-17 12:00'),     # 176 hours
                    ('2024-03-05 00:00', '2024-03-24 00:00'),     # 19 days
                    ('2024-03-02 00:00', '2024-03-30 00:00')],    # 28 days
    'PT_d_H_Mp':   [('2024-03-09 23:52', '2024-03-10 00:12'),     # 20 minutes, across midnight
                    ('2024-03-09 20:00', '2024-03-10 04:00'),     # 8 hours, across midnight
                    ('2024-03-08 06:00', '2024-03-20 18:00')],    # 12.5 days
    'PT_Hp':       [('2024-01-01 06:00', '2024-01-01 16:00'),     # 10 hours
                    ('2024-01-01 01:00', '2024-01-01 22:00')],    # 21 hours
    'PT_H_Mp':     [('2024-01-01 09:10', '2024-01-01 10:50'),     # 100 minutes
                    ('2024-01-01 06:00', '2024-01-01 11:00'),     # 5 hours
                    ('2024-01-01 06:00', '2024-01-01 16:00'),     # 10 hours
                    ('2024-01-01 04:00', '2024-01-01 18:00'),     # 14 hours
                    ('2024-01-01 01:00', '2024-01-01 22:30')],    # 21.5 hours
    'PT_H_M_Sp':   [('2024-01-01 09:10', '2024-01-01 10:50'),     # 100 minutes
                    ('2024-01-01 06:00', '2024-01-01 10:00'),     # 4 hours
                    ('2024-01-01 04:00', '2024-01-01 14:00'),     # 10 hours
                    ('2024-01-01 03:00', '2024-01-01 19:00'),     # 16 hours
                    ('2024-01-01 00:30', '2024-01-01 23:45')],    # 23.25 hours
    'PT_Mp':       [('2024-01-01 10:20', '2024-01-01 10:30'),     # 10 minutes
                    ('2024-01-01 10:10', '2024-01-01 10:30'),     # 20 minutes
                    ('2024-01-01 10:10', '2024-01-01 10:50'),     # 40 minutes
                    ('2024-01-01 10:02', '2024-01-01 10:57')],    # 55 minutes
    'PT_M_Sp':     [('2024-01-01 10:05:50', '2024-01-01 10:06:10'),   # 20 seconds, across a minute
                    ('2024-01-01 10:05:10', '2024-01-01 10:06:50'),   # 100 seconds
                    ('2024-01-01 10:05:00', '2024-01-01 10:09:00'),   # 4 minutes
                    ('2024-01-01 10:05:00', '2024-01-01 10:15:00'),   # 10 minutes
                    ('2024-01-01 10:05:00', '2024-01-01 10:25:00'),   # 20 minutes
                    ('2024-01-01 10:02:00', '2024-01-01 10:58:00')],  # 56 minutes
    'PT_Sp':       [('2024-01-01 10:05:20', '2024-01-01 10:05:30'),   # 10 seconds
                    ('2024-01-01 10:05:10', '2024-01-01 10:05:30'),   # 20 seconds
                    ('2024-01-01 10:05:10', '2024-01-01 10:05:50'),   # 40 seconds
                    ('2024-01-01 10:05:02', '2024-01-01 10:05:57')],  # 55 seconds
}

#
# The world value of a timestamp on each periodic axis, from the calendar.
#
def _templateOrdinal_(t: datetime) -> int: return date(2024, t.month, t.day).timetuple().tm_yday

_WORLD_ = {
    'PT_Qp':       lambda t: (t.month - 1)//3 + 1,
    'PT_mp':       lambda t: t.month,
    'PT_m_dp':     lambda t: _templateOrdinal_(t),
    'PT_m_d_Hp':   lambda t: _templateOrdinal_(t)*24 + t.hour,
    'PT_DoYp':     lambda t: t.timetuple().tm_yday,
    'PT_DoWp':     lambda t: t.isoweekday(),
    'PT_DoW_Hp':   lambda t: t.isoweekday()*24 + t.hour,
    'PT_DoW_H_Mp': lambda t: t.isoweekday()*24*60 + t.hour*60 + t.minute,
    'PT_dp':       lambda t: t.day,
    'PT_d_Hp':     lambda t: t.day*24 + t.hour,
    'PT_d_H_Mp':   lambda t: t.day*24*60 + t.hour*60 + t.minute,
    'PT_Hp':       lambda t: t.hour,
    'PT_H_Mp':     lambda t: t.hour*60 + t.minute,
    'PT_H_M_Sp':   lambda t: t.hour*3600 + t.minute*60 + t.second,
    'PT_Mp':       lambda t: t.minute,
    'PT_M_Sp':     lambda t: t.minute*60 + t.second,
    'PT_Sp':       lambda t: t.second,
}

#
# _readings_() -- every true label for world value w, one entry per calendar unit:
# (the unit's shortest length in world units, the texts that name w in that unit).
# A unit only has a text where w is aligned to it -- a month name on the 1st at
# midnight, an hour on the hour -- and a number label is the unit's own number.
#
def _templateMonthDay_(ordinal: int) -> tuple:
    if not 1 <= ordinal <= 366: return None, None
    _d_ = date(2024, 1, 1) + timedelta(days=ordinal - 1)
    return _d_.month, _d_.day

def _named_(lu: dict, k: int, aligned: bool = True) -> set:
    return {lu[k]} if aligned and k in lu else set()

def _num_(k: int, aligned: bool = True) -> set:
    return {str(k)} if aligned else set()

def _readings_(enum_name: str, w: int) -> list:
    if enum_name == 'PT_Qp': return [(1, {f'q{w}'})]
    if enum_name == 'PT_mp': return [(1, _named_(_MON_, w))]
    if enum_name == 'PT_m_dp':
        _m_, _d_ = _templateMonthDay_(w)
        if _m_ is None: return []
        return [(29, _named_(_MON_, _m_, _d_ == 1)), (1, _num_(_d_))]
    if enum_name == 'PT_m_d_Hp':
        _day_, _h_ = divmod(w, 24)
        _m_, _d_   = _templateMonthDay_(_day_)
        if _m_ is None: return []
        return [(29*24, _named_(_MON_, _m_, _d_ == 1 and _h_ == 0)), (24, _num_(_d_, _h_ == 0)), (1, _num_(_h_))]
    if enum_name in ('PT_DoYp', 'PT_dp', 'PT_Hp', 'PT_Mp', 'PT_Sp'): return [(1, _num_(w))]
    if enum_name == 'PT_DoWp': return [(1, _named_(_DOW_, w))]
    if enum_name == 'PT_DoW_Hp':
        _dw_, _h_ = divmod(w, 24)
        return [(24, _named_(_DOW_, _dw_, _h_ == 0)), (1, _num_(_h_))]
    if enum_name == 'PT_DoW_H_Mp':
        _dw_, _r_ = divmod(w, 24*60)
        _h_, _mi_ = divmod(_r_, 60)
        return [(24*60, _named_(_DOW_, _dw_, _r_ == 0)), (60, _num_(_h_, _mi_ == 0)), (1, _num_(_mi_))]
    if enum_name == 'PT_d_Hp':
        _d_, _h_ = divmod(w, 24)
        return [(24, _num_(_d_, _h_ == 0)), (1, _num_(_h_))]
    if enum_name == 'PT_d_H_Mp':
        _d_, _r_  = divmod(w, 24*60)
        _h_, _mi_ = divmod(_r_, 60)
        return [(24*60, _num_(_d_, _r_ == 0)), (60, _num_(_h_, _mi_ == 0)), (1, _num_(_mi_))]
    if enum_name == 'PT_H_Mp':
        _h_, _mi_ = divmod(w, 60)
        return [(60, _num_(_h_, _mi_ == 0)), (1, _num_(_mi_))]
    if enum_name == 'PT_H_M_Sp':
        _h_, _r_  = divmod(w, 3600)
        _mi_, _s_ = divmod(_r_, 60)
        return [(3600, _num_(_h_, _r_ == 0)), (60, _num_(_mi_, _s_ == 0)), (1, _num_(_s_))]
    if enum_name == 'PT_M_Sp':
        _mi_, _s_ = divmod(w, 60)
        return [(60, _num_(_mi_, _s_ == 0) | ({f'{_mi_}:00'} if _s_ == 0 else set())), (1, _num_(_s_))]
    raise ValueError(f'no readings for {enum_name}')


class Testxyp_periodic_time_contexts(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def render(self, enum_name: str, t0: datetime, t1: datetime, w: int = 256) -> str:
        df = pl.DataFrame({'ts': [t0, t1], 'value': [0.0, 1.0]})
        _x_ = self.p2s.xyp(df, x=self.p2s.tField('ts', getattr(self.p2s, enum_name)), y='value', dot_size=2.0, wxh=(w, 96))
        return _x_._repr_svg_()

    # (enum name, case, out of) from the comment the axis writes, or None if it drew no labels at all
    def caseComment(self, svg: str) -> tuple | None:
        _m_ = re.search(r'<!-- xyp\.__renderContext_periodicTime__\(\): TimePeriodicTypeP\.(\w+)\|(\d+)\|(\d+) -->', svg)
        return None if _m_ is None else (_m_.group(1), int(_m_.group(2)), int(_m_.group(3)))

    # The periodic x-axis labels: the rotate(90) text after the axis's comment (the y axis's are unrotated)
    def labels(self, svg: str) -> list:
        _i_ = svg.index('xyp.__renderContext_periodicTime__()')
        return [(float(x), t) for x, t in re.findall(r'<text x="([-\d.]+)"[^>]*transform="rotate\(90,[^"]*"[^>]*>([^<]*)</text>', svg[_i_:])]

    # (left edge, width) of the plot frame -- the origin and width the axis is drawn with
    def frame(self, svg: str) -> tuple:
        _m_ = re.search(r'<rect x="([-\d.]+)" y="[-\d.]+" width="([\d.]+)" height="[\d.]+" stroke="[^"]*" fill="none" stroke-width="0.25" />', svg)
        self.assertIsNotNone(_m_, 'no plot frame')
        return float(_m_.group(1)), float(_m_.group(2))

    # x of the dot at the window's start (value 0, lowest) and at its end (value 1, highest)
    def dots(self, svg: str) -> tuple:
        _dots_ = sorted((float(cy), float(cx)) for cx, cy in re.findall(r'<circle cx="([-\d.]+)" cy="([-\d.]+)"', svg))
        self.assertEqual(len(_dots_), 2, 'expected exactly the two window dots')
        return _dots_[1][1], _dots_[0][1]

    # Labels that no reading of their pixel's time can produce
    def untruthful(self, enum_name: str, svg: str, t0: datetime, t1: datetime) -> list:
        _w0_, _w1_  = _WORLD_[enum_name](t0), _WORLD_[enum_name](t1)
        _x0_, _dim_ = self.frame(svg)
        _px_per_w_  = _dim_ / (_w1_ - _w0_)
        _tol_w_     = _TOL_PX_ / _px_per_w_
        _bad_      = []
        for _x_, _text_ in self.labels(svg):
            _w_est_ = _w0_ + (_x_ - _x0_) / _px_per_w_
            _true_  = set()
            for _w_ in range(math.ceil(_w_est_ - _tol_w_), math.floor(_w_est_ + _tol_w_) + 1):
                for _unit_, _texts_ in _readings_(enum_name, _w_):
                    # A unit narrower than the tolerance window cannot be told from its neighbours at this zoom
                    if _unit_ * _px_per_w_ >= 2 * _TOL_PX_: _true_ |= _texts_
            if _text_ not in _true_: _bad_.append((_x_, _text_, sorted(_true_)))
        return _bad_

    # Pairs of labels drawn at the same x -- one prints over the other
    def collisions(self, svg: str) -> list:
        _ls_ = sorted(self.labels(svg))
        return [(a, b) for a, b in zip(_ls_, _ls_[1:]) if b[0] - a[0] < 1.0]

    # x of every vertical gridline the periodic axis drew more than once -- a tick drawn
    # over a major line shows as a 0.8 px stub on the 0.4 px line
    def overdrawn(self, svg: str) -> list:
        _i_ = svg.index('xyp.__renderContext_periodicTime__()')
        _xs_ = [x1 for x1, x2 in re.findall(r'<line x1="([-\d.]+)" y1="[-\d.]+" x2="([-\d.]+)"', svg[_i_:]) if x1 == x2]
        return sorted({x for x in _xs_ if _xs_.count(x) > 1})

    def assertPeriodicAxisTruthful(self, enum_name: str) -> None:
        _windows_ = _WINDOWS_[enum_name]
        for _i_, (_s0_, _s1_) in enumerate(_windows_):
            with self.subTest(case=_i_ + 1, window=(_s0_, _s1_)):
                _t0_, _t1_ = _dt_(_s0_), _dt_(_s1_)
                _svg_ = self.render(enum_name, _t0_, _t1_)
                self.assertEqual(self.caseComment(_svg_), (enum_name, _i_ + 1, len(_windows_)),
                                 'the window no longer reaches this case -- if the plot width changed, re-derive the windows')
                _x0_, _dim_ = self.frame(_svg_)
                _d0_, _d1_  = self.dots(_svg_)
                self.assertAlmostEqual(_d0_, _x0_,        delta=0.5, msg='the first row is not drawn at the left edge of the axis')   # cx is written as an integer
                self.assertAlmostEqual(_d1_, _x0_ + _dim_, delta=0.5, msg='the last row is not drawn at the right edge of the axis')
                self.assertGreater(len(self.labels(_svg_)), 0, 'the case drew no labels')
                self.assertEqual(self.untruthful(enum_name, _svg_, _t0_, _t1_), [],
                                 '(x, label, what a label there could truthfully read)')
                self.assertEqual(self.collisions(_svg_), [], 'labels drawn on top of each other')
                self.assertEqual(self.overdrawn(_svg_), [], 'gridlines drawn twice at the same x')

    # One whole cycle of each enum
    _CYCLES_ = {'PT_Qp': ('2024-01-01 00:00', '2024-12-31 23:00'), 'PT_mp': ('2024-01-01 00:00', '2024-12-31 23:00'),
                'PT_m_dp': ('2024-01-01 00:00', '2024-12-31 23:00'), 'PT_m_d_Hp': ('2024-01-01 00:00', '2024-12-31 23:00'),
                'PT_DoYp': ('2023-01-01 00:00', '2023-12-31 23:00'), 'PT_DoWp': ('2024-01-01 00:00', '2024-01-07 23:00'),
                'PT_DoW_Hp': ('2024-01-01 00:00', '2024-01-07 23:00'), 'PT_DoW_H_Mp': ('2024-01-01 00:00', '2024-01-07 23:59'),
                'PT_dp': ('2024-03-01 00:00', '2024-03-31 23:00'), 'PT_d_Hp': ('2024-03-01 00:00', '2024-03-31 23:00'),
                'PT_d_H_Mp': ('2024-03-01 00:00', '2024-03-31 23:59'), 'PT_Hp': ('2024-01-01 00:00', '2024-01-01 23:00'),
                'PT_H_Mp': ('2024-01-01 00:00', '2024-01-01 23:59'), 'PT_H_M_Sp': ('2024-01-01 00:00:00', '2024-01-01 23:59:59'),
                'PT_Mp': ('2024-01-01 10:00', '2024-01-01 10:59'), 'PT_M_Sp': ('2024-01-01 10:00:00', '2024-01-01 10:59:59'),
                'PT_Sp': ('2024-01-01 10:05:00', '2024-01-01 10:05:59')}

    def test_a_narrow_plot_still_labels_every_enum(self):
        '''A whole cycle at 96 and 128 px, where no case gives its lines 10 px: every enum
        still labels its axis, every label true, none closer than 10 px to the next, no line
        twice.  Nine of the 17 used to draw bare axes at 128 px -- their case chains had no
        fallback (PLANNING.md §5 C-xyp-periodic-no-fallback).  Those that fall back say so:
        case 0.'''
        self.assertEqual(set(self._CYCLES_), set(_WINDOWS_))
        _fell_back_ = set()
        for _enum_, (_s0_, _s1_) in self._CYCLES_.items():
            for _w_ in (96, 128):
                with self.subTest(enum=_enum_, w=_w_):
                    _t0_, _t1_ = _dt_(_s0_), _dt_(_s1_)
                    _svg_ = self.render(_enum_, _t0_, _t1_, w=_w_)
                    _case_ = self.caseComment(_svg_)
                    self.assertIsNotNone(_case_, 'the axis drew no labels at all')
                    if _case_[1] == 0:
                        _fell_back_.add(_enum_)
                        # thinned at one level -- jan, apr, jul, oct, not jan, 15, apr, 15 --
                        # which draws its labels in one colour
                        _i_ = _svg_.index('xyp.__renderContext_periodicTime__()')
                        self.assertEqual(len(set(re.findall(r'<text [^>]*fill="([^"]+)"[^>]*transform="rotate\(90,', _svg_[_i_:]))), 1)
                    _labels_ = sorted(x for x, _ in self.labels(_svg_))
                    self.assertGreater(len(_labels_), 0)
                    self.assertEqual(self.untruthful(_enum_, _svg_, _t0_, _t1_), [])
                    self.assertTrue(all(b - a >= 10 - _TOL_PX_ for a, b in zip(_labels_, _labels_[1:])),
                                    f'labels closer than 10 px: {_labels_}')
                    self.assertEqual(self.overdrawn(_svg_), [])
        self.assertGreaterEqual(len(_fell_back_), 9, 'the fallback is not what these widths exercise')

    def test_windows_cover_every_enum(self):
        self.assertEqual(set(_WINDOWS_), {e.name for e in self.p2s.TimePeriodicTypeP})

    def test_PT_Qp(self):       self.assertPeriodicAxisTruthful('PT_Qp')
    def test_PT_mp(self):       self.assertPeriodicAxisTruthful('PT_mp')
    def test_PT_m_dp(self):     self.assertPeriodicAxisTruthful('PT_m_dp')
    def test_PT_m_d_Hp(self):   self.assertPeriodicAxisTruthful('PT_m_d_Hp')
    def test_PT_DoYp(self):     self.assertPeriodicAxisTruthful('PT_DoYp')
    def test_PT_DoWp(self):     self.assertPeriodicAxisTruthful('PT_DoWp')
    def test_PT_DoW_Hp(self):   self.assertPeriodicAxisTruthful('PT_DoW_Hp')
    def test_PT_DoW_H_Mp(self): self.assertPeriodicAxisTruthful('PT_DoW_H_Mp')
    def test_PT_dp(self):       self.assertPeriodicAxisTruthful('PT_dp')
    def test_PT_d_Hp(self):     self.assertPeriodicAxisTruthful('PT_d_Hp')
    def test_PT_d_H_Mp(self):   self.assertPeriodicAxisTruthful('PT_d_H_Mp')
    def test_PT_Hp(self):       self.assertPeriodicAxisTruthful('PT_Hp')
    def test_PT_H_Mp(self):     self.assertPeriodicAxisTruthful('PT_H_Mp')
    def test_PT_H_M_Sp(self):   self.assertPeriodicAxisTruthful('PT_H_M_Sp')
    def test_PT_Mp(self):       self.assertPeriodicAxisTruthful('PT_Mp')
    def test_PT_M_Sp(self):     self.assertPeriodicAxisTruthful('PT_M_Sp')
    def test_PT_Sp(self):       self.assertPeriodicAxisTruthful('PT_Sp')

if __name__ == '__main__':
    unittest.main()
