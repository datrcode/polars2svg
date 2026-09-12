"""The time-axis grid rounds a start date down to the step it draws, and for years
before 1000 it used to do that by formatting the date and parsing the result back.

That round-trip was platform-dependent in the worst way: strftime's ``%Y`` is
zero-padded on macOS and not on glibc, while strptime's ``%Y`` matches exactly four
digits on both.  ``datetime(284, 1, 1)`` therefore survived the trip on a Mac and
raised ``ValueError: time data '284-01-01 00:00:00' does not match format`` on Linux,
so whether a plot rendered depended on the machine rather than the data.

Pre-1000 dates are ordinary input -- any historical series carries them -- and the
whole suite ran green on macOS for as long as the bug existed.  It surfaced only when
an unrelated change to CI's ``-k`` filter shifted the shared RNG stream (see the note
in test_linkp_basic.py) and moved one generated dataframe onto the broken path.
"""
import unittest
from datetime import date, datetime

import polars as pl

from polars2svg import Polars2SVG
from polars2svg.xyp import _ROUNDER_FIELDS_, _truncateToStep_


class TestTruncateToStep(unittest.TestCase):
    """The rounding itself, at every granularity the grid table defines."""

    # Deliberately a 3-digit year: this is the input the old round-trip could not
    # survive on glibc, and it must now behave identically everywhere.
    ANCIENT = datetime(284, 7, 15, 13, 47, 29, 123456)

    def test_every_step_truncates_a_pre_1000_year(self):
        _expected_ = {
            '%Y-%m-%d %H:%M:%S': datetime(284, 7, 15, 13, 47, 29),
            '%Y-%m-%d %H:%M:00': datetime(284, 7, 15, 13, 47),
            '%Y-%m-%d %H:00:00': datetime(284, 7, 15, 13),
            '%Y-%m-%d 00:00:00': datetime(284, 7, 15),
            '%Y-%m-01 00:00:00': datetime(284, 7,  1),
            '%Y-01-01 00:00:00': datetime(284, 1,  1),
        }
        for _step_, _want_ in _expected_.items():
            with self.subTest(step=_step_):
                self.assertEqual(_truncateToStep_(self.ANCIENT, _step_), _want_)

    def test_the_table_and_the_test_cover_the_same_steps(self):
        # A new grid step added without a truncation would otherwise fail at render
        # time on whichever machine first drew it.
        self.assertEqual(set(_ROUNDER_FIELDS_), {
            '%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M:00', '%Y-%m-%d %H:00:00',
            '%Y-%m-%d 00:00:00', '%Y-%m-01 00:00:00', '%Y-01-01 00:00:00'})

    def test_a_date_is_accepted_and_a_datetime_comes_back(self):
        _out_ = _truncateToStep_(date(284, 7, 15), '%Y-01-01 00:00:00')
        self.assertEqual(_out_, datetime(284, 1, 1))
        self.assertIsInstance(_out_, datetime)

    def test_a_modern_year_is_unaffected(self):
        self.assertEqual(_truncateToStep_(datetime(2026, 9, 12, 8, 30, 15),
                                          '%Y-%m-01 00:00:00'), datetime(2026, 9, 1))

    def test_an_unknown_step_names_itself(self):
        with self.assertRaises(KeyError) as _ctx_:
            _truncateToStep_(self.ANCIENT, '%Y-%m-%d %H:%M:%S.%f')
        self.assertIn('_ROUNDER_FIELDS_', str(_ctx_.exception))


class TestXYpRendersPre1000Dates(unittest.TestCase):
    """The defect as a user meets it: a plot whose x-axis is a historical series.

    This is the case that failed on the Linux clean-room runner while passing on
    every macOS run, so read a green result here as the weaker half of the evidence
    -- the unit tests above are what hold on both platforms.
    """

    def setUp(self):
        self.p2s = Polars2SVG()

    def _render(self, years):
        _df_ = pl.DataFrame({'ts': [datetime(_y_, 1, 1) for _y_ in years],
                             'v':  list(range(len(years)))})
        return self.p2s.xyp(_df_, 'ts', 'v')._repr_svg_()

    def test_a_span_of_pre_1000_years_renders(self):
        # Spans centuries, so the yearly step is chosen and the start is rounded to
        # '%Y-01-01 00:00:00' -- the exact combination that raised on glibc.
        self.assertIn('<svg', self._render([284, 512, 733, 961]))

    def test_a_span_crossing_the_year_1000_renders(self):
        self.assertIn('<svg', self._render([284, 1024, 1536, 2026]))

    def test_a_single_pre_1000_day_renders(self):
        # A narrow span picks a sub-day step, so the other rounder patterns see the
        # short year too.
        _df_ = pl.DataFrame({'ts': [datetime(284, 7, 15, 8, _m_) for _m_ in (0, 17, 41)],
                             'v':  [1, 2, 3]})
        self.assertIn('<svg', self.p2s.xyp(_df_, 'ts', 'v')._repr_svg_())


if __name__ == '__main__':
    unittest.main()
