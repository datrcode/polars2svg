"""
Tests for the "None" slice — the wedge/cell produced by rows whose bin_by value
is null.  Companion to tests/test_histop_null_bin.py; piep had *two* defects
stacked on top of each other:

1. The join.  `__binsForBins__()` resolves a bin selection to records with a join
   on the bin column, and polars joins treat null as unequal to itself by
   default, so the "None" slice could neither be selected (inner → no rows) nor
   filtered out (anti → nothing removed).

2. The miss sentinel.  `__binAtAngleDist__()` / `__binAtWaffleXY__()` returned
   None to mean "no slice under this pixel" — the same value as the null bin
   itself — and every caller resolved the ambiguity in favour of "miss".  Fixing
   only the join would have left click and brush still dead on the None slice
   while a *drag* across it worked, because the drag path reads s['bin'] straight
   off the slice list.  They now return the _NO_BIN_ sentinel.

Both directions are covered for pie, donut and waffle, along with the case the
sentinel protects: a genuine click on empty space still selects nothing.
"""
import unittest
import polars as pl
from math import atan2, cos, sin, pi, radians, sqrt
from polars2svg import Polars2SVG


def _point_in_slice(piep, bin_value):
    """A pixel inside the wedge (pie/donut) or a cell (waffle) of `bin_value`."""
    if piep.style == piep.p2s.WAFFLEp:
        _hit_ = getattr(piep, '__binAtWaffleXY__')
        for _x_ in range(0, piep.wxh[0], 2):
            for _y_ in range(0, piep.wxh[1], 2):
                if _hit_(_x_, _y_) == bin_value and (_hit_(_x_, _y_) is None) == (bin_value is None):
                    return (_x_, _y_)
        raise AssertionError(f'no waffle cell found for bin {bin_value!r}')
    for s in piep._slices_:
        if s['bin'] == bin_value or (s['bin'] is None and bin_value is None):
            _a_ = radians((s['a0'] + s['a1']) / 2.0)
            _r_ = (piep.r_inner + piep.r) / 2.0
            return (piep.cx + _r_ * cos(_a_), piep.cy + _r_ * sin(_a_))
    raise AssertionError(f'no slice found for bin {bin_value!r}')


class _NullSliceMixin:
    """The same battery of checks for each style; STYLE is set by the subclass."""

    STYLE = None

    @classmethod
    def setUpClass(cls):
        cls.p2s = Polars2SVG()
        # Distinct counts keep the descending sort deterministic: None=5, a=3, b=2.
        cls.df = pl.DataFrame({
            'cat': [None] * 5 + ['a'] * 3 + ['b'] * 2,
            'val': list(range(10)),
        })
        cls.pp = cls.p2s.piep(cls.df, 'cat', style=getattr(cls.p2s, cls.STYLE),
                              wxh=(256, 256), draw_labels=True)

    # ── the slice exists ──────────────────────────────────────────────────────

    def test_null_is_its_own_slice(self):
        self.assertIn(None, [s['bin'] for s in self.pp._slices_])

    def test_null_slice_carries_its_rows(self):
        _null_ = [s for s in self.pp._slices_ if s['bin'] is None][0]
        self.assertEqual(_null_['count'], 5.0)

    # ── hit test distinguishes the null slice from a miss ─────────────────────

    def test_hit_test_returns_the_null_bin_not_a_miss(self):
        _x_, _y_ = _point_in_slice(self.pp, None)
        if self.pp.style == self.p2s.WAFFLEp:
            _got_ = getattr(self.pp, '__binAtWaffleXY__')(_x_, _y_)
        else:
            _dx_, _dy_ = _x_ - self.pp.cx, _y_ - self.pp.cy
            _got_ = getattr(self.pp, '__binAtAngleDist__')(
                atan2(_dy_, _dx_) * 180.0 / pi, sqrt(_dx_ * _dx_ + _dy_ * _dy_))
        self.assertIsNone(_got_, 'the null slice must report itself, not a miss')

    # ── recordsAt (brush) ─────────────────────────────────────────────────────

    def test_records_at_null_slice(self):
        result = self.pp.recordsAt(_point_in_slice(self.pp, None))
        self.assertEqual(len(result), 5)
        self.assertEqual(result['cat'].null_count(), 5)

    def test_records_at_non_null_slice_unaffected(self):
        result = self.pp.recordsAt(_point_in_slice(self.pp, 'a'))
        self.assertEqual(sorted(result['cat'].to_list()), ['a'] * 3)

    # ── click select / remove ─────────────────────────────────────────────────

    def test_click_rectangle_selects_null_slice(self):
        _x_, _y_ = _point_in_slice(self.pp, None)
        result = self.pp.filterByRectangle((_x_, _y_, _x_, _y_))
        self.assertEqual(len(result), 5)
        self.assertEqual(result['cat'].null_count(), 5)

    def test_click_rectangle_removes_null_slice(self):
        _x_, _y_ = _point_in_slice(self.pp, None)
        result = self.pp.filterByRectangle((_x_, _y_, _x_, _y_), remove_records=True)
        self.assertEqual(len(result), 5)
        self.assertEqual(result['cat'].null_count(), 0)

    def test_click_oval_selects_null_slice(self):
        _x_, _y_ = _point_in_slice(self.pp, None)
        result = self.pp.filterByOval((_x_, _y_, 0, 0))
        self.assertEqual(len(result), 5)
        self.assertEqual(result['cat'].null_count(), 5)

    def test_click_oval_removes_null_slice(self):
        _x_, _y_ = _point_in_slice(self.pp, None)
        result = self.pp.filterByOval((_x_, _y_, 0, 0), remove_records=True)
        self.assertEqual(len(result), 5)
        self.assertEqual(result['cat'].null_count(), 0)

    def test_click_select_and_remove_partition_the_frame(self):
        _x_, _y_ = _point_in_slice(self.pp, None)
        keep   = self.pp.filterByRectangle((_x_, _y_, _x_, _y_))
        remove = self.pp.filterByRectangle((_x_, _y_, _x_, _y_), remove_records=True)
        self.assertEqual(len(keep) + len(remove), len(self.df))

    def test_click_on_other_slice_keeps_null_rows(self):
        _x_, _y_ = _point_in_slice(self.pp, 'a')
        result = self.pp.filterByRectangle((_x_, _y_, _x_, _y_), remove_records=True)
        self.assertEqual(result['cat'].null_count(), 5)
        self.assertNotIn('a', result['cat'].to_list())

    # ── the sentinel: a real miss is still a miss ─────────────────────────────

    # off-canvas: outside the pie's radius and outside the waffle's grid box
    _MISS = (-50, -50)

    def test_click_on_empty_space_selects_nothing(self):
        _x_, _y_ = self._MISS
        self.assertEqual(len(self.pp.recordsAt((_x_, _y_))), 0)
        self.assertEqual(len(self.pp.filterByRectangle((_x_, _y_, _x_, _y_))), 0)
        self.assertEqual(len(self.pp.filterByOval((_x_, _y_, 0, 0))), 0)

    def test_click_on_empty_space_removes_nothing(self):
        _x_, _y_ = self._MISS
        result = self.pp.filterByRectangle((_x_, _y_, _x_, _y_), remove_records=True)
        self.assertEqual(len(result), len(self.df))

    # ── substring search ──────────────────────────────────────────────────────

    def test_substring_none_selects_null_slice(self):
        result = self.pp.filterBySubstring('none')
        self.assertEqual(len(result), 5)
        self.assertEqual(result['cat'].null_count(), 5)

    def test_substring_none_remove_drops_null_slice(self):
        result = self.pp.filterBySubstring('none', remove_bins=True)
        self.assertEqual(len(result), 5)
        self.assertEqual(result['cat'].null_count(), 0)

    # ── result hygiene ────────────────────────────────────────────────────────

    def test_null_selection_drops_p2s_index(self):
        result = self.pp.filterByRectangle(_point_in_slice(self.pp, None) * 2)
        self.assertNotIn('__p2s_index__', result.columns)

    def test_null_selection_keeps_original_columns(self):
        result = self.pp.filterByRectangle(_point_in_slice(self.pp, None) * 2)
        self.assertEqual(result.columns, self.df.columns)


class TestPiepNullSlicePie(_NullSliceMixin, unittest.TestCase):
    STYLE = 'PIEp'


class TestPiepNullSliceDonut(_NullSliceMixin, unittest.TestCase):
    STYLE = 'DONUTp'


class TestPiepNullSliceWaffle(_NullSliceMixin, unittest.TestCase):
    STYLE = 'WAFFLEp'


class TestPiepNullSliceMultiField(unittest.TestCase):
    """Tuple bin_by concatenates into '__bin__'; pl.concat_str yields null when
    any part is null, so those rows fold into one 'None' slice."""

    @classmethod
    def setUpClass(cls):
        cls.p2s = Polars2SVG()
        cls.df = pl.DataFrame({
            'cat':   ['A',  'A',  'A',  None, None, 'B'],
            'group': ['x',  'x',  'x',  'y',  'z',  None],
            'val':   [1,    2,    3,    4,    5,    6],
        })
        cls.pp = cls.p2s.piep(cls.df, ('cat', 'group'), wxh=(256, 256))

    def test_null_slice_present(self):
        self.assertIn(None, [s['bin'] for s in self.pp._slices_])

    def test_click_selects_all_rows_with_a_null_part(self):
        _x_, _y_ = _point_in_slice(self.pp, None)
        result = self.pp.filterByRectangle((_x_, _y_, _x_, _y_))
        self.assertEqual(sorted(result['val'].to_list()), [4, 5, 6])

    def test_click_removes_all_rows_with_a_null_part(self):
        _x_, _y_ = _point_in_slice(self.pp, None)
        result = self.pp.filterByRectangle((_x_, _y_, _x_, _y_), remove_records=True)
        self.assertEqual(sorted(result['val'].to_list()), [1, 2, 3])

    def test_internal_bin_column_dropped(self):
        _x_, _y_ = _point_in_slice(self.pp, None)
        result = self.pp.filterByRectangle((_x_, _y_, _x_, _y_))
        self.assertNotIn('__bin__', result.columns)


class TestPiepNullBinFoldedIntoOther(unittest.TestCase):
    """A null bin small enough to be folded into the synthetic '(other)' slice is
    still reachable through it — __expandBins__ hands the real None back."""

    @classmethod
    def setUpClass(cls):
        cls.p2s = Polars2SVG()
        cls.df = pl.DataFrame({
            'cat': ['a'] * 20 + ['b'] * 15 + ['c'] * 10 + ['d'] * 5 + [None] * 2,
            'val': list(range(52)),
        })
        # 2 null rows out of 52 sweep ~14 degrees, so min_slice_deg=30 folds them in
        cls.pp = cls.p2s.piep(cls.df, 'cat', wxh=(256, 256), min_slice_deg=30)

    def test_null_folded_into_other(self):
        self.assertIn('(other)', [s['bin'] for s in self.pp._slices_])
        self.assertIn(None, self.pp._other_members_)

    def test_selecting_other_includes_the_null_rows(self):
        _x_, _y_ = _point_in_slice(self.pp, '(other)')
        result = self.pp.filterByRectangle((_x_, _y_, _x_, _y_))
        self.assertEqual(result['cat'].null_count(), 2)

    def test_removing_other_drops_the_null_rows(self):
        _x_, _y_ = _point_in_slice(self.pp, '(other)')
        result = self.pp.filterByRectangle((_x_, _y_, _x_, _y_), remove_records=True)
        self.assertEqual(result['cat'].null_count(), 0)


if __name__ == '__main__':
    unittest.main()
