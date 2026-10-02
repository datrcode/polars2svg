import re
import unittest
from datetime import datetime, timedelta
import polars as pl
from polars2svg import Polars2SVG

from random_dataframe import randomDataFrame
from svg_test_utils import XypSweepAssertions

_TEXT_  = re.compile(r'<text x="([-\d.]+)" text-anchor="(\w+)" y="([-\d.]+)"[^>]*font-size="([\d.]+)px"(?: transform="rotate\((-?\d+),[^"]*\)")?>([^<]*)</text>')
_FRAME_ = re.compile(r'<rect x="([-\d.]+)" y="([-\d.]+)" width="([\d.]+)" height="([\d.]+)" stroke="[^"]*" fill="none" stroke-width="0.25" />')

#
# _namedInstant_() -- the instant a time-axis label names.  The axis writes %Y, %Y-%m,
# %m-%d, %m-%d %H:%M, %H:%M or %H:%M:%S; the fields a label leaves out are taken from
# `near`, the time at the label's own pixel.
#
def _namedInstant_(text: str, near: datetime) -> datetime | None:
    for _pat_, _names_ in ((r'(\d{4})', 'Y'), (r'(\d{4})-(\d{2})', 'Ym'), (r'(\d{2})-(\d{2})', 'md'),
                           (r'(\d{2})-(\d{2}) (\d{2}):(\d{2})', 'mdHM'), (r'(\d{2}):(\d{2})', 'HM'), (r'(\d{2}):(\d{2}):(\d{2})', 'HMS')):
        _m_ = re.fullmatch(_pat_, text)
        if _m_ is None: continue
        _v_ = dict(zip(_names_, (int(g) for g in _m_.groups())))
        _whole_date_ = 'Y' in _v_ or 'm' in _v_              # a year or a month names its first instant
        return datetime(_v_.get('Y', near.year), _v_.get('m', 1 if 'Y' in _v_ else near.month),
                        _v_.get('d', 1 if _whole_date_ else near.day), _v_.get('H', 0), _v_.get('M', 0), _v_.get('S', 0))
    return None


class Testxyp_label(XypSweepAssertions, unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    # Every label's glyphs are on the canvas: its anchor, its measured length and its
    # rotation put the text inside wxh.  (A categorical y axis used to draw its top
    # category's label with its glyphs above the canvas -- 2px of it visible.)
    def assertLabelsOnCanvas(self, svg: str, wxh: tuple) -> None:
        _w_, _h_ = wxh
        for x, anchor, y, fs, rot, text in _TEXT_.findall(svg):
            x, y, fs = float(x), float(y), float(fs)
            _len_ = self.p2s.textLength(text, fs)
            _a_   = {'start': 0, 'middle': _len_ / 2, 'end': _len_}[anchor]
            _box_ = {'':    (x - _a_, x - _a_ + _len_, y - fs * 0.75, y),        # glyphs stand on the baseline
                     '0':   (x - _a_, x - _a_ + _len_, y - fs * 0.75, y),
                     '90':  (x, x + fs * 0.75, y - _a_, y - _a_ + _len_),        # runs down, glyphs to the right
                     '270': (x - fs * 0.75, x, y + _a_ - _len_, y + _a_)}[rot]   # runs up, glyphs to the left
            self.assertTrue(_box_[0] >= -0.5 and _box_[1] <= _w_ + 0.5 and _box_[2] >= -0.5 and _box_[3] <= _h_ + 0.5,
                            f'label {text!r} runs off the {_w_}x{_h_} canvas: {[round(v, 1) for v in _box_]}')

    # Every time label on the x axis is on the plot and names the time at its pixel, to within
    # half a pixel.  Returns how many labels there were.
    def assertTimeLabelsTruthful(self, xyp) -> int:
        _fx_, _fy_, _fw_, _fh_ = (float(v) for v in _FRAME_.search(xyp.svg).groups())
        _lo_, _hi_ = xyp.x_effective_range
        _n_ = 0
        for x, anchor, y, fs, rot, text in _TEXT_.findall(xyp.svg):
            if rot != '90': continue
            x = float(x)
            self.assertTrue(_fx_ - 0.5 <= x <= _fx_ + _fw_ + 0.5, f'time label {text!r} at x={x} is off the plot [{_fx_}, {_fx_ + _fw_}]')
            _at_    = datetime(1970, 1, 1) + timedelta(seconds=_lo_ + (x - _fx_) / _fw_ * (_hi_ - _lo_))
            _named_ = _namedInstant_(text, _at_)
            self.assertIsNotNone(_named_, f'unreadable time label {text!r}')
            self.assertLessEqual(abs(_named_ - _at_), timedelta(seconds=(_hi_ - _lo_) / _fw_ * 0.6),
                                 f'label {text!r} is drawn where the axis is {_at_}')
            _n_ += 1
        return _n_

    def test_labels(self, samples=1):
        # Use representative column type pairs (int×int, int×float, datetime×int, str×int)
        # rather than enumerating all 11×11 pairs — label/grid code branches on data type,
        # not on which specific column is chosen.  Every label stays on the canvas.
        _representative_pairs_ = [('a','b'), ('a','c'), ('g','a'), ('h','c'), ('j','a'), ('j','k')]
        for _sample_ in range(samples):
            df = randomDataFrame(10, seed=600 + _sample_)
            for _col0_, _col1_ in _representative_pairs_:
                for _sz_ in [256, 128, 64, 32, 16]:
                    with self.subTest(x=_col0_, y=_col1_, wxh=_sz_):
                        _xyp_ = self.assertCleanXyp(df, _col0_, _col1_, wxh=(_sz_, _sz_))
                        self.assertLabelsOnCanvas(_xyp_.svg, _xyp_.wxh)

    def test_SignXDivUnit(self):
        assert ('',  200,              1,               '')  == self.p2s.__SignXDivUnit__(200)
        assert ('',  234123,           1000.0,          'K') == self.p2s.__SignXDivUnit__(234_123)
        assert ('',  1233321.2314,     1000000.0,       'M') == self.p2s.__SignXDivUnit__(1_233_321.2314)
        assert ('-', 0.123,            1,               '')  == self.p2s.__SignXDivUnit__(-0.123)
        assert ('-', 9123898141,       1000000000.0,    'B') == self.p2s.__SignXDivUnit__(-9_123_898_141)
        assert ('',  6219123898141,    1000000000000.0, 'T') == self.p2s.__SignXDivUnit__(6_219_123_898_141)
        assert ('',  6219123898141.24, 1000000000000.0, 'T') == self.p2s.__SignXDivUnit__(6_219_123_898_141.24)

    def test_unittizeInt(self):
        assert '1.231K'   == self.p2s.unitizeInt(1_231)
        assert '100.2M'   == self.p2s.unitizeInt(100_213_112)    # was '100.0M' -- round(x, 0) is a float
        assert '5.121B'   == self.p2s.unitizeInt(5_121_213_112)
        assert '-124K'    == self.p2s.unitizeInt(-123_999)
        assert '-121.67M' == self.p2s.unitizeInt(-121_669_123, num_of_digits=6)

    def test_unitizeInt_regressions(self):
        # The old loop stopped extending when one more decimal place did not change the
        # string, so a zero first decimal truncated everything after it.
        assert '2.024K'   == self.p2s.unitizeInt(2_024)            # was '2.0K'
        assert '2.02K'    == self.p2s.unitizeInt(2_024, num_of_digits=4)
        # round(x, 0) returns a float, so whole numbers grew a '.0'.
        assert '443'      == self.p2s.unitizeInt(443)              # was '443.0'
        assert '2K'       == self.p2s.unitizeInt(2_000)            # was '2.0K'
        # No num_of_digits reached this one: 5 gave '45.61M' and 4 gave '46.0M'.
        assert '45.6M'    == self.p2s.unitizeInt(45_612_314, num_of_digits=4)
        # Small values keep their significant digits rather than collapsing to zero.
        assert '0.000123' == self.p2s.unitizeInt(0.000123, num_of_digits=4)  # was '0.0'
        assert '12.5'     == self.p2s.unitizeInt(12.5, num_of_digits=4)      # was '12.0'
        # Rounding up to 1000 moves to the next unit.
        assert '1M'       == self.p2s.unitizeInt(999_999)
        assert '0'        == self.p2s.unitizeInt(0)
        assert 'nan'      == self.p2s.unitizeInt(float('nan'))

    # At every width from 96 to 1536, every time gridline label is on the plot and names the
    # time at its pixel, and the labels stay on the canvas; some width draws labels.
    def assertTimeGridAcrossWidths(self, df):
        _labels_ = 0
        for w in range(96,1600,64):
            with self.subTest(w=w):
                _xyp_ = self.assertCleanXyp(df, 'ts', 'value', dot_size=3.0, wxh=(w,128))
                _labels_ += self.assertTimeLabelsTruthful(_xyp_)
                self.assertLabelsOnCanvas(_xyp_.svg, _xyp_.wxh)
        self.assertGreater(_labels_, 0, 'no time labels drawn at any width')

    def test_innerGridForTime_hours(self):
        df  = pl.DataFrame({
            'ts':['2026-01-02 00:00:00',
                  '2026-01-02 04:00:00',
                  '2026-01-02 08:00:00',
                  '2026-01-02 12:00:00'],
            'value':[1, 2, 3, 4]
        }).with_columns(pl.col('ts').str.to_datetime())
        self.assertTimeGridAcrossWidths(df)

    def test_innerGridForTime_days(self):
        df  = pl.DataFrame({
            'ts':['2026-01-02 00:00:00',
                  '2026-01-03 04:00:00',
                  '2026-01-05 08:00:00',
                  '2026-01-12 12:00:00'],
            'value':[1, 2, 3, 4]
        }).with_columns(pl.col('ts').str.to_datetime())
        self.assertTimeGridAcrossWidths(df)

    def test_innerGridForTime_months(self):
        df  = pl.DataFrame({
            'ts':['2026-01-02 00:00:00',
                  '2026-02-03 04:00:00',
                  '2026-06-05 08:00:00',
                  '2026-12-12 12:00:00'],
            'value':[1, 2, 3, 4]
        }).with_columns(pl.col('ts').str.to_datetime())
        self.assertTimeGridAcrossWidths(df)

    def test_innerGridForTime_years(self):
        df  = pl.DataFrame({
            'ts':['2006-01-02 00:00:00',
                  '2010-02-03 04:00:00',
                  '2021-06-05 08:00:00',
                  '2026-12-12 12:00:00'],
            'value':[1, 2, 3, 4]
        }).with_columns(pl.col('ts').str.to_datetime())
        self.assertTimeGridAcrossWidths(df)

    def test_innerGridForTime_offMidnight(self):
        '''A datetime axis that does not start at midnight.  The gridlines used to be placed
        as if it did: 06:00 was drawn where 12:00 is, a 09:30-11:00 axis had every label
        shifted 9.5 hours (off the plot), and the gridlines ran on to the end of the last
        day, past the plot.  datetime is a subclass of date, so a check meant for Date
        columns moved every datetime to midnight -- PLANNING.md §5 C-xyp-datetime-grid-shift.'''
        for _ts_ in (['2026-01-02 06:00:00', '2026-01-02 18:00:00'],
                     ['2026-01-02 09:30:00', '2026-01-02 11:00:00'],
                     ['2026-01-02 15:00:00', '2026-01-05 03:00:00']):
            with self.subTest(span=_ts_):
                self.assertTimeGridAcrossWidths(pl.DataFrame({'ts': _ts_, 'value': [1, 2]}).with_columns(pl.col('ts').str.to_datetime()))

if __name__ == '__main__':
    unittest.main()
