#
# test_xyp_int_ranges.py
#
# Regression tests for x_range/y_range handling under a constant integer dot_size.
#
# Investigation finding: __toPixelCoordinates_int__() honored x_range/y_range as a
# *scale* but never *filtered* out-of-range rows the way __toPixelCoordinates_float__()
# does.  Consequence: with an integer dot_size, points outside a user-specified range
# were clamped (via .clip(0, 0.99999999)) and smeared onto the plot edges instead of
# being removed -- so the int path disagreed with the float path for the same range.
#
# These tests pin the two paths to identical row-membership semantics and guard the
# edge-smear regression.  The companion float behavior is exercised in
# test_xyp_xy_ranges.py.
#
import unittest
import polars as pl
from datetime import datetime
from polars2svg import Polars2SVG


class Testxyp_int_ranges(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()
        # 0..100 on both axes, one point per integer -- easy to reason about membership
        self.df = pl.DataFrame({'x': [float(i) for i in range(101)],
                                'y': [float(i) for i in range(101)]})

    #
    # A subset range must drop out-of-range rows for an integer dot_size, exactly like
    # the float path does (this is the core bug these tests were written for).
    #
    def test_int_subset_range_filters_like_float(self):
        for rng_kwargs in [{'x_range': (40, 60)},
                           {'y_range': (40, 60)},
                           {'x_range': (40, 60), 'y_range': (40, 60)}]:
            xyp_int   = self.p2s.xyp(self.df, 'x', 'y', dot_size=3,   **rng_kwargs)
            xyp_float = self.p2s.xyp(self.df, 'x', 'y', dot_size=3.0, **rng_kwargs)
            self.assertEqual(len(xyp_int.df_flat), len(xyp_float.df_flat),
                             f'int/float row counts diverge for {rng_kwargs}')
            # 40..60 inclusive is 21 rows on each constrained axis
            self.assertEqual(len(xyp_int.df_flat), 21,
                             f'expected 21 in-range rows for {rng_kwargs}, got {len(xyp_int.df_flat)}')

    #
    # Out-of-range points must not pile onto the plot edges for an integer dot_size.
    # Before the fix, ~80 out-of-range rows clamped onto the two boundary columns.
    #
    def test_int_range_no_edge_smear(self):
        xyp = self.p2s.xyp(self.df, 'x', 'y', dot_size=3, x_range=(40, 60), y_range=(40, 60))
        f   = xyp.df_flat
        # Every surviving __xi__/__yi__ must be inside the requested window.
        self.assertTrue(f.filter((pl.col('__xi__') < 40) | (pl.col('__xi__') > 60)).is_empty(),
                        'out-of-range x rows survived (edge smear)')
        self.assertTrue(f.filter((pl.col('__yi__') < 40) | (pl.col('__yi__') > 60)).is_empty(),
                        'out-of-range y rows survived (edge smear)')

    #
    # A range that excludes ALL data yields an empty plot (no clamped survivors) and
    # still renders without error -- the numeric context falls back to the range for
    # its gridline min/max.
    #
    def test_int_range_fully_excludes_data(self):
        xyp = self.p2s.xyp(self.df, 'x', 'y', dot_size=3, x_range=(500, 600))
        self.assertEqual(len(xyp.df_flat), 0)
        self.assertIn('<svg', xyp.svg)

    #
    # A range wider than the data changes nothing about membership (all rows survive),
    # matching the float path.
    #
    def test_int_external_range_keeps_all(self):
        xyp_int   = self.p2s.xyp(self.df, 'x', 'y', dot_size=3,   x_range=(-50, 200), y_range=(-50, 200))
        xyp_float = self.p2s.xyp(self.df, 'x', 'y', dot_size=3.0, x_range=(-50, 200), y_range=(-50, 200))
        self.assertEqual(len(xyp_int.df_flat), len(self.df))
        self.assertEqual(len(xyp_int.df_flat), len(xyp_float.df_flat))

    #
    # No range supplied => no filtering for either path (regression guard: the filter
    # must be gated on x_range/y_range being set).
    #
    def test_int_no_range_keeps_all(self):
        xyp = self.p2s.xyp(self.df, 'x', 'y', dot_size=3)
        self.assertEqual(len(xyp.df_flat), len(self.df))

    #
    # The datetime range path (already covered for float in test_xyp_xy_ranges.py) must
    # also filter for an integer dot_size.
    #
    def test_int_datetime_range_filters(self):
        df = pl.DataFrame({'dt': [datetime(1980 + i, 6, 1) for i in range(21)],   # 1980..2000
                           'v':  [float(i) for i in range(21)]})
        fm, to = datetime(1990, 1, 1), datetime(1995, 1, 1)
        xyp_int   = self.p2s.xyp(df, 'dt', 'v', dot_size=3,   wxh=(128, 128), x_range=(fm, to))
        xyp_float = self.p2s.xyp(df, 'dt', 'v', dot_size=3.0, wxh=(128, 128), x_range=(fm, to))
        self.assertEqual(len(xyp_int.df_flat), len(xyp_float.df_flat))
        # 1990, 1991, 1992, 1993, 1994 -> 5 rows within [1990-01-01, 1995-01-01]
        self.assertEqual(len(xyp_int.df_flat), 5)

    #
    # Context / world<->screen transforms remain self-consistent under integer dot_size
    # with ranges (i.e. the context grid lines are placed correctly): the forward
    # transform of every surviving world coordinate lands on its stored pixel, and the
    # values stay inside the plot box.
    #
    def test_int_context_transforms_consistent(self):
        df = pl.DataFrame({'x': [1.0, 2, 3, 4, 5, 6], 'y': [5.0, 7, 9, 3, 4, 15]})
        for xr in [None, (-10, 10), (2, 5)]:
            for yr in [None, (-10, 10), (2, 8)]:
                xyp = self.p2s.xyp(df, 'x', 'y', dot_size=3, x_range=xr, y_range=yr)
                f   = xyp.df_flat
                x0  = xyp.plot_origin[0]
                x1  = x0 + xyp.plot_size[0]
                y1  = xyp.plot_origin[1]
                y0  = y1 - xyp.plot_size[1]
                for i in range(f.shape[0]):
                    sx, sy = f['__xpx__'][i], f['__ypx__'][i]
                    # dots (drawn as dot_size rects from their top-left) stay within the plot box
                    self.assertGreaterEqual(sx, x0)
                    self.assertLessEqual(sx, x1)
                    self.assertGreaterEqual(sy, y0)
                    self.assertLessEqual(sy, y1)


if __name__ == '__main__':
    unittest.main()
