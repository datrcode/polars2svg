#
# x_range= / y_range= on xyp.  The contract (__resolveRanges__()):
#
#   - a range sets the window its axis is drawn over;
#   - rows outside the window are DROPPED, not clipped onto the plot edges;
#   - an axis without a range keeps the whole data's extent, even when the other
#     axis's range drops rows.
#
# So a range is the same picture as filtering the frame to the window yourself and
# pinning the other axis to the data's extent -- and these tests assert exactly that,
# byte for byte, alongside what each case is about: a subset range drops rows, a wider
# one places the data in its share of the axis, one that misses the data draws nothing.
# They used to render the same cases from unseeded random data and assert nothing
# (PLANNING.md V11).  Asserting them found that an axis with no range was labelled from
# the rows the OTHER axis's range kept: a y_range keeping late September to December
# relabelled a whole-year x axis sep..dec, with the dots still where jan..dec put them.
#
import unittest
import re
from datetime import datetime, date, timedelta

import numpy as np
import polars as pl
from polars2svg import Polars2SVG
from svg_test_utils import normalize_svg

_DOT_GROUP_RE_ = re.compile(r'<g class="(?:circle|rect)-group-TESTID"[^>]*>(.*?)</g>', re.S)
_FRAME_RE_     = re.compile(r'<rect x="([-\d.]+)" y="([-\d.]+)" width="([\d.]+)" height="([\d.]+)" stroke="[^"]*" fill="none" stroke-width="0.25" />')
_BAR_RE_       = re.compile(r'<rect x="([-\d.]+)" y="([-\d.]+)" width="([-\d.]+)" height="([-\d.]+)" stroke-opacity')

# A range bound or a data value as a number on its axis -- epoch seconds for time, as __xi__/__yi__ are
def _num_(v) -> float:
    if isinstance(v, datetime): return (v - datetime(1970, 1, 1)).total_seconds()
    if isinstance(v, date):     return (datetime(v.year, v.month, v.day) - datetime(1970, 1, 1)).total_seconds()
    return float(v)

# A bound in the column's own type, for filtering it by hand (the ranges here fall on midnight for Date columns)
def _asColumnType_(df: pl.DataFrame, col: str, v):
    if df.schema[col] == pl.Date and isinstance(v, datetime): return v.date()
    return v


class Testxyp_xy_ranges(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()
        rng      = np.random.default_rng(7)
        n        = 1_000
        self.df  = pl.DataFrame({'a':rng.normal(loc=2500, scale=10, size=n),
                                 'b':rng.normal(loc=500,  scale=3,  size=n)})
        self.base = {'dot_size':1.25, 'color':self.p2s.CROW_STRETCHEDp}

    def svg(self, df: pl.DataFrame, x, y, **kwargs) -> str:
        return normalize_svg(self.p2s.xyp(df, x=x, y=y, **kwargs).svg)

    # (x, y) of every dot -- circle centres, or the rects' corners at larger dot sizes
    def dots(self, svg: str) -> list:
        _m_ = _DOT_GROUP_RE_.search(svg)
        if _m_ is None: return []
        return [(float(x), float(y)) for x, y in re.findall(r'<(?:circle cx|rect x)="([-\d.]+)" (?:cy|y)="([-\d.]+)"', _m_.group(1))]

    # (left, top, width, height) of the plot frame
    def frame(self, svg: str) -> tuple:
        _m_ = _FRAME_RE_.search(svg)
        self.assertIsNotNone(_m_, 'no plot frame')
        return tuple(float(v) for v in _m_.groups())

    # The non-empty distribution bars: (x-distribution bars along the top, y-distribution bars down the right)
    def bars(self, svg: str) -> tuple:
        _fx_, _fy_, _fw_, _fh_ = self.frame(svg)
        _bars_ = [tuple(float(v) for v in b) for b in _BAR_RE_.findall(svg)]
        _bars_ = [b for b in _bars_ if b[2] > 0 and b[3] > 0]
        return ([b for b in _bars_ if b[1] + b[3] <= _fy_ + 1.5],
                [b for b in _bars_ if b[0] >= _fx_ + _fw_ - 1.5])

    #
    # assertSameAsPrefiltered() -- the render with ranges equals the render of the frame
    # filtered to the window by hand, with each axis that has no range pinned to the data's
    # extent (`pin` overrides that extent, for an axis whose units are not its column's).
    # Returns the filtered frame, so a test can say how many rows the range kept.
    #
    def assertSameAsPrefiltered(self, df: pl.DataFrame, x, y, x_range=None, y_range=None, pin: dict | None = None, **kwargs) -> pl.DataFrame:
        _xcol_ = self.p2s.tFieldTuple(x)[0] if isinstance(x, Polars2SVG.TField) else x
        _ycol_ = self.p2s.tFieldTuple(y)[0] if isinstance(y, Polars2SVG.TField) else y
        _f_    = df
        for _col_, _rng_ in ((_xcol_, x_range), (_ycol_, y_range)):
            if _rng_ is not None:
                _lo_, _hi_ = _asColumnType_(df, _col_, _rng_[0]), _asColumnType_(df, _col_, _rng_[1])
                _f_ = _f_.filter((pl.col(_col_) >= _lo_) & (pl.col(_col_) <= _hi_))
        _pin_ = {'x': (df[_xcol_].min(), df[_xcol_].max()), 'y': (df[_ycol_].min(), df[_ycol_].max())} | (pin or {})
        _ranged_ = self.svg(df, x, y, **({'x_range': x_range} if x_range else {}), **({'y_range': y_range} if y_range else {}), **kwargs)
        _byhand_ = self.svg(_f_, x, y, x_range=x_range or _pin_['x'], y_range=y_range or _pin_['y'], **kwargs)
        self.assertEqual(_ranged_, _byhand_, 'a range must draw what filtering the frame to it does, on an axis window of the data extent')
        return _f_

    # The dots occupy the share of a ranged axis that the data does: [data_lo, data_hi] within [lo, hi]
    def assertDotsSpanTheirShare(self, svg: str, axis: str, rng: tuple, data: tuple, dot_size: float) -> None:
        _fx_, _fy_, _fw_, _fh_ = self.frame(svg)
        _lo_, _hi_   = _num_(rng[0]),  _num_(rng[1])
        _dlo_, _dhi_ = _num_(data[0]), _num_(data[1])
        _ps_  = [d[0] if axis == 'x' else d[1] for d in self.dots(svg)]
        _tol_ = dot_size + 1.5                                        # dots snap to a dot-size grid
        if axis == 'x': _want_ = (_fx_ + _fw_*(_dlo_ - _lo_)/(_hi_ - _lo_),       _fx_ + _fw_*(_dhi_ - _lo_)/(_hi_ - _lo_))
        else:           _want_ = (_fy_ + _fh_ - _fh_*(_dhi_ - _lo_)/(_hi_ - _lo_), _fy_ + _fh_ - _fh_*(_dlo_ - _lo_)/(_hi_ - _lo_))
        self.assertAlmostEqual(min(_ps_), _want_[0], delta=_tol_, msg=f'{axis}: the data starts in the wrong place on its axis')
        self.assertAlmostEqual(max(_ps_), _want_[1], delta=_tol_, msg=f'{axis}: the data ends in the wrong place on its axis')

    # Each distribution's bars sit under the dots they count -- within one bin at each end
    def assertBarsUnderDots(self, svg: str, dot_size: float) -> None:
        _xbars_, _ybars_ = self.bars(svg)
        _dots_ = self.dots(svg)
        self.assertGreater(len(_xbars_), 0, 'no x distribution drawn')
        self.assertGreater(len(_ybars_), 0, 'no y distribution drawn')
        for _name_, _bars_, _lo_i_, _len_i_, _dot_i_ in (('x', _xbars_, 0, 2, 0), ('y', _ybars_, 1, 3, 1)):
            _bin_ = max(b[_len_i_] for b in _bars_)
            _tol_ = _bin_ + dot_size + 1.5
            _blo_, _bhi_ = min(b[_lo_i_] for b in _bars_), max(b[_lo_i_] + b[_len_i_] for b in _bars_)
            _dlo_, _dhi_ = min(d[_dot_i_] for d in _dots_), max(d[_dot_i_] for d in _dots_)
            self.assertAlmostEqual(_blo_, _dlo_, delta=_tol_, msg=f'the {_name_} distribution does not start under the dots')
            self.assertAlmostEqual(_bhi_, _dhi_, delta=_tol_, msg=f'the {_name_} distribution does not end under the dots')

    def extent(self, col: str) -> tuple: return self.df[col].min(), self.df[col].max()

    def test_withoutRanges(self):
        # No range is the range the data spans
        for _dot_size_ in (1.25, 3):
            for _context_ in (True, False):
                with self.subTest(dot_size=_dot_size_, draw_context=_context_):
                    _kw_ = {'dot_size':_dot_size_, 'color':self.p2s.CROW_MAGNITUDEp, 'draw_context':_context_}
                    _svg_ = self.svg(self.df, 'a', 'b', **_kw_)
                    self.assertGreater(len(self.dots(_svg_)), 0)
                    self.assertEqual(_svg_, self.svg(self.df, 'a', 'b', x_range=self.extent('a'), y_range=self.extent('b'), **_kw_))

    def test_subsetRangesFloat(self):
        for _xr_, _yr_ in (((2480, 2500), None), (None, (500, 505)), ((2480, 2500), (500, 505))):
            with self.subTest(x_range=_xr_, y_range=_yr_):
                _f_ = self.assertSameAsPrefiltered(self.df, 'a', 'b', x_range=_xr_, y_range=_yr_, **self.base)
                self.assertTrue(0 < len(_f_) < len(self.df), 'the range should keep some rows and drop others')

    def test_externalRangesFloat(self):
        for _xr_, _yr_ in (((2_200, 2_800), None), (None, (450, 600)), ((2_200, 2_800), (450, 600))):
            with self.subTest(x_range=_xr_, y_range=_yr_):
                _f_ = self.assertSameAsPrefiltered(self.df, 'a', 'b', x_range=_xr_, y_range=_yr_, **self.base)
                self.assertEqual(len(_f_), len(self.df), 'a range wider than the data drops nothing')
                _svg_ = self.svg(self.df, 'a', 'b', x_range=_xr_, y_range=_yr_, **self.base)
                if _xr_: self.assertDotsSpanTheirShare(_svg_, 'x', _xr_, self.extent('a'), 1.25)
                if _yr_: self.assertDotsSpanTheirShare(_svg_, 'y', _yr_, self.extent('b'), 1.25)

    def test_externalRangesFloat_distributions(self):
        _dist_ = {'x_distributions':self.p2s.ROW_COUNTp, 'y_distributions':self.p2s.ROW_COUNTp}
        for _xr_, _yr_ in (((2_200, 2_800), None), (None, (450, 600)), ((2_200, 2_800), (450, 600))):
            with self.subTest(x_range=_xr_, y_range=_yr_):
                self.assertSameAsPrefiltered(self.df, 'a', 'b', x_range=_xr_, y_range=_yr_, **self.base, **_dist_)
                _svg_ = self.svg(self.df, 'a', 'b', x_range=_xr_, y_range=_yr_, **self.base, **_dist_)
                if _xr_: self.assertDotsSpanTheirShare(_svg_, 'x', _xr_, self.extent('a'), 1.25)
                if _yr_: self.assertDotsSpanTheirShare(_svg_, 'y', _yr_, self.extent('b'), 1.25)
                self.assertBarsUnderDots(_svg_, 1.25)

    # A range that misses the data draws an empty window -- nothing smeared onto its edges
    def assertOutOfRange(self, **dist) -> None:
        for _xr_, _yr_ in (((12_200, 14_800), None), (None, (45_000, 46_000)), ((12_200, 14_800), (45_000, 46_000))):
            with self.subTest(x_range=_xr_, y_range=_yr_, **{k: str(v) for k, v in dist.items()}):
                _f_ = self.assertSameAsPrefiltered(self.df, 'a', 'b', x_range=_xr_, y_range=_yr_, **self.base, **dist)
                self.assertEqual(len(_f_), 0)
                _svg_ = self.svg(self.df, 'a', 'b', x_range=_xr_, y_range=_yr_, **self.base, **dist)
                self.assertEqual(self.dots(_svg_), [], 'dots drawn for rows outside the range')
                self.assertEqual(self.bars(_svg_), ([], []), 'distribution bars drawn for rows outside the range')

    def test_outOfRangeFloat(self):                 self.assertOutOfRange()
    def test_outOfRangeFloat_distributions(self):   self.assertOutOfRange(x_distributions=self.p2s.ROW_COUNTp, y_distributions=self.p2s.ROW_COUNTp)
    def test_outOfRangeFloat_distributions2(self):  self.assertOutOfRange(x_distributions='a', y_distributions='b')

    #
    # Temporal ranges, on both axes: the six ways a range can meet the data
    #
    def timeFrame(self, dates_only: bool) -> pl.DataFrame:
        _rng_  = np.random.default_rng(11)
        _t0_   = datetime(1980, 1, 1)
        _secs_ = _rng_.integers(0, int((datetime(2000, 12, 31) - _t0_).total_seconds()), 200)
        _df_   = pl.DataFrame({'dt':    [_t0_ + timedelta(seconds=int(s)) for s in _secs_],
                               'value': _rng_.integers(0, 100, 200)})
        return _df_.with_columns(pl.col('dt').dt.date()) if dates_only else _df_

    def assertTemporalRanges(self, df: pl.DataFrame, end_of_day: tuple) -> None:
        _kw_  = {'wxh':(128, 128), 'dot_size':2.0}
        _hms_ = end_of_day
        _cases_ = {   # name -> (range, what it keeps)
            'begins inside the data':   ((datetime(1990, 6, 1), datetime(2020, 12, 1, *_hms_)), 'some'),
            'ends inside the data':     ((datetime(1950, 6, 1), datetime(1995, 12, 1, *_hms_)), 'some'),
            'inside the data':          ((datetime(1985, 6, 1), datetime(1997, 12, 1, *_hms_)), 'some'),
            'wider than the data':      ((datetime(1960, 6, 1), datetime(2020, 12, 1, *_hms_)), 'all'),
            'before the data':          ((datetime(1960, 6, 1), datetime(1975, 12, 1, *_hms_)), 'none'),
            'after the data':           ((datetime(2005, 6, 1), datetime(2010, 12, 1, *_hms_)), 'none'),
        }
        _extent_ = (df['dt'].min(), df['dt'].max())
        for _axis_, (_x_, _y_) in (('x', ('dt', 'value')), ('y', ('value', 'dt'))):
            with self.subTest(axis=_axis_, case='no range'):
                _ranges_ = {f'{_axis_}_range': _extent_}
                self.assertEqual(self.svg(df, _x_, _y_, **_kw_), self.svg(df, _x_, _y_, **_ranges_, **_kw_),
                                 'no range is the range the data spans')
            for _name_, (_rng_, _keeps_) in _cases_.items():
                with self.subTest(axis=_axis_, case=_name_):
                    _ranges_ = {f'{_axis_}_range': _rng_}
                    _f_   = self.assertSameAsPrefiltered(df, _x_, _y_, **_ranges_, **_kw_)
                    _svg_ = self.svg(df, _x_, _y_, **_ranges_, **_kw_)
                    if   _keeps_ == 'none': self.assertEqual((len(_f_), self.dots(_svg_)), (0, []))
                    elif _keeps_ == 'all':
                        self.assertEqual(len(_f_), len(df))
                        self.assertDotsSpanTheirShare(_svg_, _axis_, _rng_, _extent_, 2.0)
                    else:                   self.assertTrue(0 < len(_f_) < len(df))

    def test_datetimeRanges(self):
        self.assertTemporalRanges(self.timeFrame(dates_only=False), end_of_day=(23, 59, 59))

    def test_dateRanges(self):
        _dates_ = self.timeFrame(dates_only=True)
        self.assertTemporalRanges(_dates_, end_of_day=())
        # A Date column puts its rows where the same midnights in a Datetime column do
        _midnights_ = _dates_.with_columns(pl.col('dt').cast(pl.Datetime))
        for _x_, _y_ in (('dt', 'value'), ('value', 'dt')):
            with self.subTest(date_axis=_x_ == 'dt' and 'x' or 'y'):
                _kw_ = {'wxh':(128, 128), 'dot_size':2.0}
                self.assertEqual(sorted(self.dots(self.svg(_dates_, _x_, _y_, **_kw_))), sorted(self.dots(self.svg(_midnights_, _x_, _y_, **_kw_))))

    #
    # An axis with no range keeps the whole data's extent -- its labels too -- when the
    # other axis's range drops rows.  The data here makes that visible: `value` rises with
    # time through 2024, so y_range=(150, 199) keeps only late September to December --
    # a narrower extent on every kind of x axis: dates, months of the year, categories.
    #
    def test_unrangedAxisKeepsTheDataExtent(self):
        _ts_ = [datetime(2024, 1, 1) + timedelta(hours=43*i) for i in range(200)]
        _df_ = pl.DataFrame({'dt': _ts_, 'value': list(range(200)), 'cat': [f'c{i//20:02}' for i in range(200)]})
        _kw_ = {'wxh':(256, 128), 'dot_size':2.0, 'y_range':(150, 199)}
        with self.subTest(axis='datetime'):
            self.assertSameAsPrefiltered(_df_, 'dt', 'value', **_kw_)
        with self.subTest(axis='date'):
            self.assertSameAsPrefiltered(_df_.with_columns(pl.col('dt').dt.date()), 'dt', 'value', **_kw_)
        with self.subTest(axis='periodic'):
            self.assertSameAsPrefiltered(_df_, self.p2s.tField('dt', self.p2s.PT_mp), 'value', pin={'x': (1, 12)}, **_kw_)
        with self.subTest(axis='categorical'):
            # a categorical axis takes no range to pin, so read its end labels instead: the
            # x axis's min label is anchored at its start and its max at its end (they are
            # no longer told apart by colour -- both take the axis label colour)
            _svg_ = self.svg(_df_, 'cat', 'value', **_kw_)
            _ends_ = [re.search(rf'text-anchor="{_anchor_}" y="[^"]+" [^>]*font-size="12px">([^<]*)<', _svg_).group(1)
                      for _anchor_ in ('start', 'end')]
            self.assertEqual(_ends_, ['c00', 'c09'])

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
