"""
Tests for the "None" facet — the tile produced by rows whose category_by value
is null.

Completes the null-selection series (histop / piep / chordp).  smallp's version
was the quietest of the four: polars' group_by gives a null category its own
group, so the tile was created, laid out and labelled "None" like any other, but
`__filterForKey__` populated it with `pl.col(c) == key[0]`, and comparing to None
yields *null* for every row.  filter() drops those, so the tile rendered
permanently empty -- and polars said so on every render:

    UserWarning: Comparisons with None always result in null.
                 Consider using `.is_null()` or `.is_not_null()`.

The multi-column form did not even get that far: a key holding a null types its
struct literal as Null, and `is_in` raised
`InvalidOperationError: 'is_in' cannot check for Struct({...}) values in
List(Struct({'region': Null, ...})) data`.

The remainder tile had the same defect from the other side: it was built as
`~col.is_in(visible)`, and since `is_in` yields null for a null value and ~null
is null, rows with a null category were dropped from the remainder too.  They
belonged to no tile at all -- present in the frame, drawn nowhere.

Both now go through `__predicateForKey__`, which uses `eq_missing()`: null equals
itself, and the result is never null, so the &/|/~ built on it stay total.  The
invariant the tests assert is coverage -- every row lands in exactly one tile,
whether that is its own facet or the remainder.
"""
import unittest
import warnings

import polars as pl
from polars2svg import Polars2SVG


# 'n' x2, 's' x2, null x2 — small enough that every category gets its own tile
_DF_ = pl.DataFrame({
    'x':      [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    'y':      [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    'region': ['n', 'n', 's', 's', None, None],
    'val':    [1,   2,   3,   4,   5,    6],
})

# a wider frame whose null group is small enough to be pushed into the remainder
_DF_WIDE_ = pl.DataFrame({
    'x':   [float(i) for i in range(24)],
    'y':   [float(i) for i in range(24)],
    'cat': ['a']*6 + ['b']*5 + ['c']*4 + ['d']*3 + ['e']*2 + ['f']*1 + [None]*3,
    'val': list(range(24)),
})


def _tiles(smp):
    """The tiles that actually hold rows, excluding the '__all__' duplicate."""
    return {k: v for k, v in smp.category_to_df.items()
            if v is not None and k != '__all__'}


def _covered_indices(smp):
    _idx_ = []
    for _v_ in _tiles(smp).values():
        _idx_ += _v_['__p2s_index__'].to_list()
    return _idx_


class TestSmallpNullCategoryTile(unittest.TestCase):
    """Single-column category_by, every category visible."""

    @classmethod
    def setUpClass(cls):
        cls.p2s  = Polars2SVG()
        cls.tmpl = cls.p2s.xyp(x='x', y='y')
        cls.smp  = cls.p2s.smallp(_DF_, cls.tmpl, 'region', wxh=(400, 200))
        cls.smp._repr_svg_()

    def test_null_gets_its_own_tile(self):
        self.assertIn((None,), cls_keys := list(self.smp.category_to_df.keys()),
                      f'keys: {cls_keys}')

    def test_null_tile_holds_its_rows(self):
        _tile_ = self.smp.category_to_df[(None,)]
        self.assertEqual(len(_tile_), 2)
        self.assertEqual(_tile_['region'].null_count(), 2)

    def test_null_tile_holds_the_right_rows(self):
        self.assertEqual(sorted(self.smp.category_to_df[(None,)]['val'].to_list()), [5, 6])

    def test_other_tiles_unaffected(self):
        self.assertEqual(sorted(self.smp.category_to_df[('n',)]['val'].to_list()), [1, 2])

    def test_tile_is_labelled_none(self):
        self.assertIn('None', self.smp.svg)

    def test_every_row_lands_in_exactly_one_tile(self):
        _idx_ = _covered_indices(self.smp)
        self.assertEqual(sorted(_idx_), sorted(set(_idx_)), 'a row was placed twice')
        self.assertEqual(len(_idx_), len(_DF_))

    def test_render_emits_no_none_comparison_warning(self):
        """polars warns on `col == None`; the fix has to make that warning go away."""
        with warnings.catch_warnings(record=True) as _caught_:
            warnings.simplefilter('always')
            _smp_ = self.p2s.smallp(_DF_, self.tmpl, 'region', wxh=(400, 200))
            _smp_._repr_svg_()
        _msgs_ = [str(_w_.message) for _w_ in _caught_
                  if 'Comparisons with None' in str(_w_.message)]
        self.assertEqual(_msgs_, [])


class TestSmallpNullCategoryMultiColumn(unittest.TestCase):
    """Tuple category_by used to raise InvalidOperationError outright."""

    @classmethod
    def setUpClass(cls):
        cls.p2s  = Polars2SVG()
        cls.tmpl = cls.p2s.xyp(x='x', y='y')
        cls.df   = _DF_.with_columns(
            pl.Series('tier', ['1', '1', '2', '2', '1', None]))
        cls.smp  = cls.p2s.smallp(cls.df, cls.tmpl, ('region', 'tier'), wxh=(600, 400))
        cls.smp._repr_svg_()

    def test_constructs(self):
        self.assertIsNotNone(self.smp.svg)

    def test_key_with_one_null_part_is_populated(self):
        _key_ = (None, '1')
        if _key_ in self.smp.category_to_df:
            self.assertEqual(len(self.smp.category_to_df[_key_]), 1)
        else:   # folded into the remainder at this size — still has to hold the row
            self.assertGreater(len(self.smp.category_to_df['__remainder__']), 0)

    def test_every_row_lands_in_exactly_one_tile(self):
        _idx_ = _covered_indices(self.smp)
        self.assertEqual(sorted(_idx_), sorted(set(_idx_)))
        self.assertEqual(len(_idx_), len(self.df))

    def test_both_parts_null_is_matched_too(self):
        _df_ = _DF_.with_columns(pl.Series('tier', ['1', '1', '2', '2', None, None]))
        _smp_ = self.p2s.smallp(_df_, self.tmpl, ('region', 'tier'), wxh=(600, 400))
        _smp_._repr_svg_()
        _idx_ = _covered_indices(_smp_)
        self.assertEqual(len(_idx_), len(_df_))
        _null_tiles_ = [v for v in _tiles(_smp_).values() if v['region'].null_count() > 0]
        self.assertTrue(_null_tiles_, 'the doubly-null rows landed in no tile')


class TestSmallpNullCategoryRemainder(unittest.TestCase):
    """When the null group does not get its own tile it must reach the remainder."""

    @classmethod
    def setUpClass(cls):
        cls.p2s  = Polars2SVG()
        cls.tmpl = cls.p2s.xyp(x='x', y='y')

    def test_null_rows_reach_the_remainder(self):
        _smp_ = self.p2s.smallp(_DF_WIDE_, self.tmpl, 'cat', wxh=(600, 400))
        _smp_._repr_svg_()
        self.assertNotIn((None,), _smp_.category_to_xy,
                         'this layout is meant to push the null group into the remainder')
        _rem_ = _smp_.category_to_df['__remainder__']
        self.assertEqual(_rem_['cat'].null_count(), 3)

    def test_coverage_holds_at_every_size(self):
        """Own tile or remainder, the rows are placed exactly once either way."""
        for _wxh_ in [(600, 400), (400, 200), (300, 150), (200, 100)]:
            with self.subTest(wxh=_wxh_):
                _smp_ = self.p2s.smallp(_DF_WIDE_, self.tmpl, 'cat', wxh=_wxh_)
                _smp_._repr_svg_()
                _idx_ = _covered_indices(_smp_)
                self.assertEqual(sorted(_idx_), sorted(set(_idx_)), 'a row was placed twice')
                self.assertEqual(len(_idx_), len(_DF_WIDE_), 'a row was placed nowhere')

    def test_null_rows_are_never_lost(self):
        for _wxh_ in [(600, 400), (400, 200), (300, 150), (200, 100)]:
            with self.subTest(wxh=_wxh_):
                _smp_ = self.p2s.smallp(_DF_WIDE_, self.tmpl, 'cat', wxh=_wxh_)
                _smp_._repr_svg_()
                _nulls_ = sum(_v_['cat'].null_count() for _v_ in _tiles(_smp_).values())
                self.assertEqual(_nulls_, 3)


class TestSmallpNullCategoryTypes(unittest.TestCase):
    """The category column need not be a string."""

    def setUp(self):
        self.p2s  = Polars2SVG()
        self.tmpl = self.p2s.xyp(x='x', y='y')

    def test_numeric_category_with_nulls(self):
        _df_ = pl.DataFrame({'x': [1.0, 2.0, 3.0, 4.0], 'y': [1.0, 2.0, 3.0, 4.0],
                             'code': [1, 1, None, None], 'val': [1, 2, 3, 4]},
                            schema={'x': pl.Float64, 'y': pl.Float64,
                                    'code': pl.Int64, 'val': pl.Int64})
        _smp_ = self.p2s.smallp(_df_, self.tmpl, 'code', wxh=(400, 200))
        _smp_._repr_svg_()
        self.assertEqual(len(_smp_.category_to_df[(None,)]), 2)
        self.assertEqual(len(_covered_indices(_smp_)), len(_df_))

    def test_every_row_null(self):
        """One category, and it is the null one.  Which tile it lands in is a
        layout question -- a single category goes to the remainder at this size
        whether or not it is null (asserted against a non-null control below);
        what matters is that the rows are placed."""
        _df_ = pl.DataFrame({'x': [1.0, 2.0], 'y': [1.0, 2.0],
                             'region': [None, None], 'val': [1, 2]},
                            schema={'x': pl.Float64, 'y': pl.Float64,
                                    'region': pl.String, 'val': pl.Int64})
        _smp_ = self.p2s.smallp(_df_, self.tmpl, 'region', wxh=(400, 200))
        _smp_._repr_svg_()
        self.assertEqual(_smp_._sorted_category_keys_, [(None,)])
        self.assertEqual(len(_covered_indices(_smp_)), len(_df_))

        _control_ = self.p2s.smallp(_df_.with_columns(pl.Series('region', ['z', 'z'])),
                                    self.tmpl, 'region', wxh=(400, 200))
        _control_._repr_svg_()
        self.assertEqual(list(_smp_.category_to_df.keys()),
                         list(_control_.category_to_df.keys()),
                         'the null category should be laid out like any other')


if __name__ == '__main__':
    unittest.main()
