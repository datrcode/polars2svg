import re
import polars as pl
import random
import calendar

__name__ = 'timep_dataframes'


def makeTimeDf(n=50, year=(2020, 2025), month=(1, 12), day=(1, 28),
               hour=0, minute=0, second=0, seed=None):
    '''Create a datetime DataFrame for timep testing.

    Returns a pl.DataFrame with columns:
      ts        (pl.Datetime) – constructed from the given components
      value     (pl.Int32)    – random int 0-100
      category  (pl.Utf8)     – random choice of 'A', 'B', 'C'
      numeric   (pl.Float64)  – random float 0-10

    Each component may be:
      int        – fixed value for all rows
      (lo, hi)   – independently uniform-random for each row

    seed=None draws from the module-global `random`, as every caller always has;
    an int draws from a private generator, so the frame is the same on every run
    and nothing else's random state moves.
    '''
    rng  = random if seed is None else random.Random(seed)
    rows = {'ts': [], 'value': [], 'category': [], 'numeric': []}
    for _ in range(n):
        y  = rng.randint(*year)   if isinstance(year,   tuple) else year
        m  = rng.randint(*month)  if isinstance(month,  tuple) else month
        d  = rng.randint(*day)    if isinstance(day,    tuple) else day
        h  = rng.randint(*hour)   if isinstance(hour,   tuple) else hour
        mi = rng.randint(*minute) if isinstance(minute, tuple) else minute
        s  = rng.randint(*second) if isinstance(second, tuple) else second
        d  = min(d, calendar.monthrange(y, m)[1])   # clamp to valid month-end
        rows['ts'].append(f'{y:04}-{m:02}-{d:02} {h:02}:{mi:02}:{s:02}')
        rows['value'].append(rng.randint(0, 100))
        rows['category'].append(rng.choice(['A', 'B', 'C']))
        rows['numeric'].append(round(rng.uniform(0.0, 10.0), 3))
    return pl.DataFrame(rows).with_columns(pl.col('ts').str.to_datetime())


def makeDateDf(n=50, year=(2020, 2025), month=(1, 12), day=(1, 28), seed=None):
    '''Create a date-only DataFrame for timep testing.

    Returns a pl.DataFrame with columns:
      dt        (pl.Date)
      value     (pl.Int32)
      category  (pl.Utf8)
      numeric   (pl.Float64)

    seed= as for makeTimeDf().
    '''
    rng  = random if seed is None else random.Random(seed)
    rows = {'dt': [], 'value': [], 'category': [], 'numeric': []}
    for _ in range(n):
        y = rng.randint(*year)  if isinstance(year,  tuple) else year
        m = rng.randint(*month) if isinstance(month, tuple) else month
        d = rng.randint(*day)   if isinstance(day,   tuple) else day
        d = min(d, calendar.monthrange(y, m)[1])
        rows['dt'].append(f'{y:04}-{m:02}-{d:02}')
        rows['value'].append(rng.randint(0, 100))
        rows['category'].append(rng.choice(['A', 'B', 'C']))
        rows['numeric'].append(round(rng.uniform(0.0, 10.0), 3))
    return pl.DataFrame(rows).with_columns(pl.col('dt').str.to_date())   # not .cast(): deprecated for strings (PLANNING.md R9)


# ─────────────────────────────────────────────────────────────────────────────
# Reading a rendered timep back (PLANNING.md V11): which column is which time bin,
# and what it shows, so a test can compare it with what polars computes.
# ─────────────────────────────────────────────────────────────────────────────
# The interval each linear level truncates to -- the calendar's own definitions
_LINEAR_EVERY_ = {'LT_Yp': '1y', 'LT_Y_Qp': '1q', 'LT_Y_mp': '1mo', 'LT_Y_m_dp': '1d',
                  'LT_Y_m_d_4Hp': '4h', 'LT_Y_m_d_Hp': '1h', 'LT_Y_m_d_H_15Mp': '15m',
                  'LT_Y_m_d_H_Mp': '1m', 'LT_Y_m_d_H_M_15Sp': '15s', 'LT_Y_m_d_H_M_Sp': '1s'}
_FRAME_RE_ = re.compile(r'<rect x="([\d.]+)" y="([\d.]+)" width="([\d.]+)" height="([\d.]+)" stroke="[^"]*" fill="none" stroke-width="0.5" />')
_DATA_RE_  = re.compile(r'<rect x="([\d.]+)" y="([\d.]+)" width="([\d.]+)" height="([\d.]+)" fill="(#[0-9a-fA-F]{6})"(?: stroke="none")? />')


#
# timeBins() -- each row's bin at the level the view drew, and the spine that level
# lays across the plot: every linear interval from the first bin to the last, or the
# whole periodic cycle.  Returns (rows with '__bin__', spine as a list).
#
def timeBins(t, df: pl.DataFrame, col: str) -> tuple:
    _enum_ = t._time_enum_
    if _enum_.name in _LINEAR_EVERY_:
        _every_ = _LINEAR_EVERY_[_enum_.name]
        _rows_  = df.filter(pl.col(col).is_not_null()).with_columns(pl.col(col).dt.truncate(_every_).alias('__bin__'))
        _lo_, _hi_ = _rows_['__bin__'].min(), _rows_['__bin__'].max()
        _range_ = pl.date_range if df.schema[col] == pl.Date else pl.datetime_range
        return _rows_, _range_(_lo_, _hi_, _every_, eager=True).to_list()
    _lo_, _hi_ = t.p2s.timePeriodicRange(_enum_)
    _rows_ = df.filter(pl.col(col).is_not_null()).with_columns(t.p2s.polarsOperationForEnum(col, _enum_).alias('__bin__'))
    return _rows_, list(range(_lo_, _hi_ + 1))


# The plot (x0, y0, w, h), the frame drawn around it (None without context), and the
# drawn columns left to right as [(x, [(height, fill), ...] bottom up)].  A column is a
# stack: it starts at a rect standing on the plot's baseline and climbs through each
# segment that starts where the one below ends.  So two plain bars that share an x --
# which bins a fraction of a pixel apart do once x is written to 0.1px -- are two columns.
def timepColumns(t) -> tuple:
    _svg_   = t._repr_svg_()
    _plot_  = (float(t._plot_x0_), float(t._plot_y0_), float(t._plot_w_), float(t._plot_h_))
    _m_     = _FRAME_RE_.search(_svg_)
    _frame_ = None if _m_ is None else tuple(float(v) for v in _m_.groups())
    _w_, _h_ = t.wxh
    _base_  = _plot_[1] + _plot_[3]
    _lr_    = getattr(t, '_legend_region_', None)
    _by_x_: dict = {}
    for x, y, w, hh, fill in _DATA_RE_.findall(_svg_):
        if (x, y, w, hh) == ('0', '0', str(_w_), str(_h_)): continue       # the background
        if _lr_ is not None and _lr_[0] <= float(x) < _lr_[0] + _lr_[2] and _lr_[1] <= float(y) < _lr_[1] + _lr_[3]: continue   # a legend swatch
        _by_x_.setdefault(float(x), []).append((float(y), float(hh), fill.lower()))
    _cols_ = []
    for x in sorted(_by_x_):
        _rest_ = sorted(_by_x_[x], key=lambda r: -(r[0] + r[1]))           # lowest bottom first
        while _rest_:
            _chain_, _top_ = [], None
            for r in list(_rest_):
                if (_top_ is None and abs(r[0] + r[1] - _base_) <= 0.06) or (_top_ is not None and abs(r[0] + r[1] - _top_) <= 0.06):
                    _chain_.append((r[1], r[2])); _top_ = r[0]; _rest_.remove(r)
            if not _chain_: raise AssertionError(f'a rect at x={x} does not stand on the baseline or on another segment')
            _cols_.append((x, _chain_))
    return _plot_, _frame_, _cols_


def _rgbClose_(a: str, b: str) -> bool:
    return all(abs(int(a[i:i + 2], 16) - int(b[i:i + 2], 16)) <= 1 for i in (1, 3, 5))


class TimepAssertions:
    '''Assertions over a rendered timep, for a unittest.TestCase to mix in.'''

    # The drawn columns are the bins with a non-zero `metric`, in time order, each in its
    # bin's slot on the spine and as tall as its metric on the plot's scale (the tallest
    # spans the plot).  Returns [(bin, segments)] for further checks.
    def assertColumnsShow(self, t, df: pl.DataFrame, col: str, metric: pl.Expr) -> list:
        _rows_, _spine_ = timeBins(t, df, col)
        _want_ = {b: m for b, m in _rows_.group_by('__bin__').agg(metric.alias('m')).iter_rows() if m}
        _bins_ = sorted(_want_, key=_spine_.index)
        (_fx_, _fy_, _fw_, _fh_), _frame_, _cols_ = timepColumns(t)
        if _frame_ is not None: self.assertEqual(_frame_, (_fx_, _fy_, _fw_, _fh_), 'the frame is not drawn round the plot')
        self.assertEqual(len(_cols_), len(_bins_), 'one column per bin with data')
        _slot_, _top_ = _fw_ / len(_spine_), max(_want_.values())
        # Each column is matched to the first unmatched bin in its slot and of its height --
        # position and height together, since sub-pixel slots can share a written x
        _open_ = [(_b_, _fx_ + _spine_.index(_b_) * _slot_, _fh_ * _want_[_b_] / _top_) for _b_ in _bins_]
        _matched_ = []
        for x, _segs_ in _cols_:
            _hh_ = sum(hh for hh, _ in _segs_)
            _hit_ = next((o for o in _open_ if abs(o[1] - x) <= 0.11 and abs(o[2] - _hh_) <= 0.11), None)
            self.assertIsNotNone(_hit_, f'the column at x={x}, {_hh_:.1f} tall, is no bin in its slot at its height')
            _open_.remove(_hit_)
            _matched_.append((_hit_[0], _segs_))
        return _matched_

    # Every column is one segment in the colour `colors` gives its bin (and shows `metric`)
    def assertColumnColors(self, t, df: pl.DataFrame, col: str, colors: dict, metric: pl.Expr = pl.len()) -> None:
        for _b_, _segs_ in self.assertColumnsShow(t, df, col, metric):
            self.assertEqual(len(_segs_), 1, f'bin {_b_} should be one segment')
            self.assertTrue(_rgbClose_(_segs_[0][1], colors[_b_].lower()), f'bin {_b_} is {_segs_[0][1]}, expected {colors[_b_]}')

    # Stacked columns: each bin's segments are its colour values' shares of `metric`, each in
    # that value's colour, on the plot's scale (the tallest column's total spans the plot).
    # A share thinner than remainder_threshold is pooled with the others like it -- and with
    # the values pooled into '(other)' -- into one segment in the '(other)' colour, which must
    # hold exactly their height (PLANNING.md §5 C-histop-two-remainders).
    def assertStacksShow(self, t, df: pl.DataFrame, col: str, color_field: str, metric: pl.Expr) -> None:
        _rows_, _ = timeBins(t, df, col)
        _parts_ = {(b, v): m for b, v, m in _rows_.group_by('__bin__', color_field).agg(metric.alias('m')).iter_rows() if m}
        _totals_: dict = {}
        for (b, _), m in _parts_.items(): _totals_[b] = _totals_.get(b, 0) + m
        _fh_    = timepColumns(t)[0][3]
        _scale_ = _fh_ / max(_totals_.values())
        _colors_ = t.p2s.colors(sorted({str(v) for _, v in _parts_}))
        for _b_, _segs_ in self.assertColumnsShow(t, df, col, metric):
            _all_  = [(_colors_[str(v)].lower(), m * _scale_) for (b, v), m in _parts_.items() if b == _b_]
            _want_ = sorted((c, hh) for c, hh in _all_ if hh >= t.remainder_threshold)
            _thin_ = sum(hh for _, hh in _all_ if hh < t.remainder_threshold)
            _got_  = sorted((fill, hh) for hh, fill in _segs_)
            if _thin_ > 0:     # the pool is the drawn segment in none of the named shares' colours
                _pool_ = [g for g in _got_ if g[0] not in {c for c, _ in _want_}]
                self.assertEqual(len(_pool_), 1, f'bin {_b_}: expected one pooled remainder')
                self.assertEqual(_pool_[0][0], t.p2s.color('(other)').lower(), f'bin {_b_}: the pool is not in the (other) colour')
                self.assertAlmostEqual(_pool_[0][1], _thin_, delta=0.11, msg=f'bin {_b_}: the pool does not hold the thin shares')
                _got_.remove(_pool_[0])
            self.assertEqual([f for f, _ in _got_], [f for f, _ in _want_], f'bin {_b_} has the wrong colours')
            for (_, a), (_, b) in zip(_got_, _want_):
                self.assertAlmostEqual(a, b, delta=0.11, msg=f'a segment of bin {_b_} is the wrong height')


# {bin: value} of one aggregate over a frame, binned at the level the view drew
def perTimeBin(t, df: pl.DataFrame, col: str, expr: pl.Expr) -> dict:
    return dict(timeBins(t, df, col)[0].group_by('__bin__').agg(expr.alias('m')).iter_rows())
