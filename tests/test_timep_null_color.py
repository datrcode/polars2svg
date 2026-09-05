"""
Tests for a null in timep's color field.

Last of the null-handling series (histop / piep / chordp / smallp), and the only
one where nothing was selectable because nothing could be *built*: a stacked
timep whose color field held a null raised, from __computeAggregates2__,

    TypeError: '<' not supported between instances of 'NoneType' and 'str'

`sorted()` over the distinct color values compared None against the strings.  A
null is a legitimate category -- group_by gives it its own group and its rows are
counted into df_agg like any other -- so the sort keys on `(v is None, v)`, which
puts nulls last and never compares None with anything, leaving the order of the
non-null values exactly as it was.

Unlike the other four this is not a selection bug: timep bins on the time axis,
so the color field never takes part in filtering, and the select/remove partition
was always total.  It is here because the crash made a whole component
unreachable for data the other components render.
"""
import datetime
import unittest

import polars as pl
from polars2svg import Polars2SVG


_N_ = 12
_TS_ = [datetime.datetime(2024, 1, 1, _i_) for _i_ in range(_N_)]
_DF_ = pl.DataFrame({
    'ts':  _TS_,
    'cat': ['a', 'a', 'b', None, None, 'c', 'd', 'd', 'e', 'e', 'f', 'f'],
    'val': list(range(_N_)),
})
_DF_CLEAN_ = _DF_.filter(pl.col('cat').is_not_null())


class TestTimepNullColorCategory(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.p2s = Polars2SVG()
        cls.tp  = cls.p2s.timep(_DF_, time='ts', color='cat', wxh=(256, 256))

    def test_constructs(self):
        self.assertIsNotNone(self.tp.svg)

    def test_is_actually_the_stacked_path(self):
        """The crash was in the stacked branch -- a test on the simple branch
        would pass without exercising anything."""
        self.assertEqual(self.tp._agg_type_, 'stacked')

    def test_null_is_a_category(self):
        self.assertIn(None, self.tp._color_categories_)

    def test_nulls_sort_last(self):
        self.assertEqual(self.tp._color_categories_[-1], None)

    def test_non_null_order_is_unchanged(self):
        self.assertEqual([_c_ for _c_ in self.tp._color_categories_ if _c_ is not None],
                         ['a', 'b', 'c', 'd', 'e', 'f'])

    def test_null_rows_are_still_counted(self):
        _null_rows_ = self.tp.df_agg.filter(pl.col('cat').is_null())
        self.assertEqual(_null_rows_['__count__'].sum(), 2)

    def test_ordering_matches_a_null_free_frame(self):
        """The sort key must not perturb data that never had a null."""
        _clean_ = self.p2s.timep(_DF_CLEAN_, time='ts', color='cat', wxh=(256, 256))
        self.assertEqual(_clean_._color_categories_,
                         [_c_ for _c_ in self.tp._color_categories_ if _c_ is not None])

    def test_selection_still_partitions_the_frame(self):
        """timep filters on the time axis, so the color field is not involved --
        asserted so a later change to that cannot pass unnoticed."""
        _w_, _h_ = self.tp.wxh
        _keep_   = self.tp.filterByRectangle((0, 0, _w_, _h_))
        _remove_ = self.tp.filterByRectangle((0, 0, _w_, _h_), remove_records=True)
        self.assertEqual(len(_keep_) + len(_remove_), len(_DF_))


class TestTimepNullColorVariants(unittest.TestCase):

    def setUp(self):
        self.p2s = Polars2SVG()

    def test_every_color_value_null(self):
        _df_ = pl.DataFrame({'ts': _TS_, 'cat': [None] * _N_, 'val': list(range(_N_))},
                            schema={'ts': pl.Datetime, 'cat': pl.String, 'val': pl.Int64})
        _tp_ = self.p2s.timep(_df_, time='ts', color='cat', wxh=(256, 256))
        self.assertIsNotNone(_tp_.svg)
        self.assertEqual(_tp_._color_categories_, [None])

    def test_periodic_time_axis(self):
        """__computeAggregates2__ has two sorted() sites, linear and periodic."""
        _tp_ = self.p2s.timep(_DF_, time=('ts', self.p2s.TimePeriodicTypeP.PT_Hp),
                              color='cat', wxh=(256, 256))
        self.assertIsNotNone(_tp_.svg)
        self.assertIn(None, _tp_._color_categories_)

    def test_single_null_among_many(self):
        _cats_ = ['a'] * (_N_ - 1) + [None]
        _df_   = pl.DataFrame({'ts': _TS_, 'cat': _cats_, 'val': list(range(_N_))})
        _tp_   = self.p2s.timep(_df_, time='ts', color='cat', wxh=(256, 256))
        self.assertEqual(_tp_._color_categories_, ['a', None])


class TestAllNullColorRendersEverywhere(unittest.TestCase):
    """A color column that is *entirely* null reaches p2s_render_mixin's rank join,
    which built its lookup frame by inferring the dtype from the category list.  An
    all-null list infers as Null, which cannot be a join key against String or Int64:

        SchemaError: datatypes of join keys don't match -
                     `cat`: str on left does not match `cat`: null on right

    So timep and histop -- the two components that go through colorizeBar() /
    colorizeAllBars*() -- could not render such a frame at all.  The lookup now pins
    the column's dtype from the frame being joined.  Covered for both dtypes and for
    every component, because the mixin is shared and xyp/piep reach the same colors by
    a different path.
    """

    def setUp(self):
        self.p2s = Polars2SVG()

    def _frame(self, cats, dtype):
        return pl.DataFrame(
            {'ts': _TS_, 'cat': cats,
             'bin': ['p', 'p', 'q', 'q', 'r', 'r', 's', 's', 'p', 'q', 'r', 's'],
             'x': [float(_i_) for _i_ in range(_N_)],
             'y': [float(_i_) for _i_ in range(_N_)],
             'val': list(range(_N_))},
            schema={'ts': pl.Datetime, 'cat': dtype, 'bin': pl.String,
                    'x': pl.Float64, 'y': pl.Float64, 'val': pl.Int64})

    def _renderers(self):
        return {
            'timep stacked': lambda _d_: self.p2s.timep(_d_, time='ts', color='cat', wxh=(256, 256)),
            'histop stacked': lambda _d_: self.p2s.histop(_d_, bin_by='bin', color='cat',
                                                          style=self.p2s.STACKEDBARp, wxh=(256, 256)),
            'histop bar':    lambda _d_: self.p2s.histop(_d_, bin_by='bin', color='cat', wxh=(256, 256)),
            'xyp':           lambda _d_: self.p2s.xyp(_d_, x='x', y='y', color='cat', wxh=(256, 256)),
            'piep':          lambda _d_: self.p2s.piep(_d_, 'bin', color='cat', wxh=(256, 256)),
        }

    def test_all_null_color_renders(self):
        for _dtype_, _cats_ in ((pl.String, [None] * _N_), (pl.Int64, [None] * _N_)):
            for _name_, _fn_ in self._renderers().items():
                with self.subTest(dtype=_dtype_, component=_name_):
                    _obj_ = _fn_(self._frame(_cats_, _dtype_))
                    self.assertTrue(_obj_._repr_svg_().startswith('<svg'))

    def test_partly_null_color_renders(self):
        _str_ = ['a', 'a', None, None, 'b', 'b', 'c', 'c', 'a', 'b', 'c', None]
        _int_ = [1, 1, None, None, 2, 2, 3, 3, 1, 2, 3, None]
        for _dtype_, _cats_ in ((pl.String, _str_), (pl.Int64, _int_)):
            for _name_, _fn_ in self._renderers().items():
                with self.subTest(dtype=_dtype_, component=_name_):
                    _obj_ = _fn_(self._frame(_cats_, _dtype_))
                    self.assertTrue(_obj_._repr_svg_().startswith('<svg'))

    # Null-free output is deliberately not re-asserted here: the SVG carries a random
    # id, so two renders of one frame never compare equal, and the exact-string golden
    # tests already pin that output.  nulls_equal and the pinned dtype only differ from
    # the old behaviour when a null key exists, so the goldens staying green is the
    # evidence that this change is inert for data without nulls.


if __name__ == '__main__':
    unittest.main()
