"""
Tests for zone-aware Datetime columns in xyp.

A column parsed with %z (which polars converts to UTC) or given any other zone used to
fail inside __indexXandY_join__: the epoch conversion subtracts a naive datetime, and
polars finds no supertype for datetime[us, UTC] and datetime[us].  xyp now plots such a
column as wall-clock time in its own zone (see _wallClockExpr_ in xyp.py), so it must
draw exactly what the same wall-clock values draw as a naive column -- labels included.
"""
import unittest
from datetime import datetime, timedelta, UTC

import polars as pl

from polars2svg import Polars2SVG
from svg_test_utils import normalize_svg


def _naive_():
    t0 = datetime(2024, 3, 1, 8)
    return pl.DataFrame({'t': [t0 + timedelta(minutes=10 * i) for i in range(289)]}) \
             .with_columns(v=pl.int_range(pl.len()) % 7, g=pl.int_range(pl.len()) % 3)


class TestXYpTimeZones(unittest.TestCase):

    def setUp(self):
        self.p2s   = Polars2SVG()
        self.naive = _naive_()
        self.utc   = self.naive.with_columns(pl.col('t').dt.replace_time_zone('UTC'))
        self.ny    = self.naive.with_columns(pl.col('t').dt.replace_time_zone('America/New_York'))

    def _svg_(self, df, **kw):
        return normalize_svg(self.p2s.xyp(df, 't', 'v', wxh=(300, 200), **kw)._repr_svg_())

    # The regression itself: this raised SchemaError.
    def test_utc_column_renders_as_its_wall_clock(self):
        self.assertEqual(self._svg_(self.utc), self._svg_(self.naive))

    # Wall-clock in the column's own zone -- not converted to UTC first.
    def test_other_zone_renders_in_that_zone(self):
        self.assertEqual(self._svg_(self.ny), self._svg_(self.naive))

    def test_distributions_on_a_utc_axis(self):
        self.assertEqual(self._svg_(self.utc, x_distributions=[self.p2s.ROW_COUNTp]),
                         self._svg_(self.naive, x_distributions=[self.p2s.ROW_COUNTp]))

    # An aware bound is converted into the column's zone: 15:00 UTC is 10:00 in New York
    # (EST, before the 2024-03-10 change).
    def test_aware_range_bound_converts_into_the_column_zone(self):
        _aware_ = (datetime(2024, 3, 1, 15, tzinfo=UTC), datetime(2024, 3, 2, 15, tzinfo=UTC))
        _xyp_   = self.p2s.xyp(self.ny, 't', 'v', wxh=(300, 200), x_range=_aware_)
        self.assertEqual(_xyp_.x_range, (datetime(2024, 3, 1, 10), datetime(2024, 3, 2, 10)))
        self.assertEqual(normalize_svg(_xyp_._repr_svg_()),
                         self._svg_(self.naive, x_range=(datetime(2024, 3, 1, 10), datetime(2024, 3, 2, 10))))

    # On a naive column there is no zone to convert into: the bound keeps its wall clock.
    def test_aware_range_bound_on_a_naive_column_keeps_its_wall_clock(self):
        _xyp_ = self.p2s.xyp(self.naive, 't', 'v', wxh=(300, 200),
                             x_range=(datetime(2024, 3, 1, 10, tzinfo=UTC), datetime(2024, 3, 2, 10)))
        self.assertEqual(_xyp_.x_range, (datetime(2024, 3, 1, 10), datetime(2024, 3, 2, 10)))

    # The records handed back are the user's own, zone intact.
    def test_records_at_keeps_the_zone(self):
        _xyp_ = self.p2s.xyp(self.ny, 't', 'v', wxh=(300, 200))
        _xyp_._repr_svg_()
        _recs_ = _xyp_.recordsAt((150, 100), shape=self.p2s.SELECT_VERTICALp, threshold=20)
        self.assertGreater(len(_recs_), 0)
        self.assertEqual(_recs_.schema['t'], pl.Datetime('us', 'America/New_York'))

    # The time keys compare the aware base frame against the plotted (naive) timeframe.
    def test_filter_by_timeframe_on_an_aware_column(self):
        _cur_ = self.ny.filter(pl.col('t').dt.day() == 2, pl.col('g') == 0)
        _xyp_ = self.p2s.xyp(_cur_, 't', 'v')
        _unf_ = _xyp_.filterByTimeframe(self.ny, 'unfilter')
        _exp_ = _xyp_.filterByTimeframe(self.ny, 'expand_both')
        self.assertIsNotNone(_unf_)
        self.assertIsNotNone(_exp_)
        self.assertGreater(len(_unf_), len(_cur_))
        self.assertGreater(len(_exp_), len(_cur_))
        self.assertEqual(_unf_.schema['t'], pl.Datetime('us', 'America/New_York'))
        # Same rows the naive column gives.
        _naive_cur_ = self.naive.filter(pl.col('t').dt.day() == 2, pl.col('g') == 0)
        _naive_unf_ = self.p2s.xyp(_naive_cur_, 't', 'v').filterByTimeframe(self.naive, 'unfilter')
        self.assertEqual(len(_unf_), len(_naive_unf_))

    def test_smallp_with_a_shared_utc_axis(self):
        _tmpl_ = self.p2s.xyp(self.utc, 't', 'v', wxh=(100, 60), sm_shared={self.p2s.SM_X})
        _ntmpl_ = self.p2s.xyp(self.naive, 't', 'v', wxh=(100, 60), sm_shared={self.p2s.SM_X})
        self.assertEqual(normalize_svg(self.p2s.smallp(self.utc,   _tmpl_,  'g')._repr_svg_()),
                         normalize_svg(self.p2s.smallp(self.naive, _ntmpl_, 'g')._repr_svg_()))

    # A zone-aware value in a non-axis field is carried naive too.
    def test_aware_line_order_field(self):
        _df_ = self.utc.with_columns(o=pl.col('t'))
        _nf_ = self.naive.with_columns(o=pl.col('t'))
        self.assertEqual(normalize_svg(self.p2s.xyp(_df_, 't', 'v', line=('g',), line_order_by='o', wxh=(300, 200))._repr_svg_()),
                         normalize_svg(self.p2s.xyp(_nf_, 't', 'v', line=('g',), line_order_by='o', wxh=(300, 200))._repr_svg_()))


if __name__ == '__main__':
    unittest.main()
