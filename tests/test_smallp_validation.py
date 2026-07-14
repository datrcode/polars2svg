import unittest
import polars as pl
import datetime
from polars2svg import Polars2SVG


class TestSmallpValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p2s = Polars2SVG()
        # Basic df with columns a, b, c, cat and a datetime column ts
        cls.df = pl.DataFrame({
            'a':   [1, 2, 3, 4, 5, 6],
            'b':   [6, 5, 4, 3, 2, 1],
            'c':   [1, 1, 2, 2, 3, 3],
            'cat': ['x', 'x', 'y', 'y', 'z', 'z'],
            'ts':  pl.Series([
                datetime.datetime(2024, 1, 1),
                datetime.datetime(2024, 2, 1),
                datetime.datetime(2024, 3, 1),
                datetime.datetime(2024, 4, 1),
                datetime.datetime(2024, 5, 1),
                datetime.datetime(2024, 6, 1),
            ]),
            'int_col': [1, 2, 3, 4, 5, 6],   # int, NOT datetime — for bad-tfield tests
        })
        # A simple xyp template
        cls.xyp = cls.p2s.xyp(df=cls.df, x='a', y='b', wxh=(64, 64))

    # ── grid_mode validation ─────────────────────────────────────────────────

    def test_grid_mode_requires_tuple(self):
        '''grid_mode=True with a plain string category_by → ValueError.'''
        with self.assertRaises(ValueError):
            self.p2s.smallp(self.df, 'cat', self.xyp, grid_mode=True, wxh=(256, 256))

    def test_grid_mode_requires_exactly_two_fields(self):
        '''grid_mode=True with a 3-tuple → ValueError.'''
        with self.assertRaises(ValueError):
            self.p2s.smallp(self.df, ('a', 'b', 'c'), self.xyp, grid_mode=True, wxh=(256, 256))

    def test_grid_mode_accepts_valid_two_tuple(self):
        '''grid_mode=True with a 2-tuple should not raise during validation.'''
        # May raise later during geometry if the grid is too small, but NOT a ValueError
        # from __validateInput__; we just confirm validation passes.
        try:
            self.p2s.smallp(self.df, ('a', 'cat'), self.xyp, grid_mode=True, wxh=(256, 256))
        except ValueError as e:
            self.fail(f'Unexpected ValueError with valid grid_mode tuple: {e}')

    # ── wxh validation ───────────────────────────────────────────────────────

    def test_wxh_both_none_raises(self):
        '''wxh=(None, None) must raise ValueError.'''
        with self.assertRaises(ValueError):
            self.p2s.smallp(self.df, 'cat', self.xyp, wxh=(None, None))

    def test_wxh_floats_coerced(self):
        '''wxh floats are now coerced to int (shared Polars2SVG.normalizeWxh);
        see tests/test_wxh_normalization.py.'''
        sp = self.p2s.smallp(self.df, 'cat', self.xyp, wxh=(100.5, 200.5))
        self.assertEqual(sp.wxh, (100, 200))

    def test_wxh_list_coerced_to_tuple(self):
        '''wxh accepts a list and returns a tuple.'''
        sp = self.p2s.smallp(self.df, 'cat', self.xyp, wxh=[100, 200])
        self.assertEqual(sp.wxh, (100, 200))

    def test_wxh_non_sequence_raises(self):
        '''wxh that is not a 2-sequence must raise ValueError.'''
        with self.assertRaises(ValueError):
            self.p2s.smallp(self.df, 'cat', self.xyp, wxh=256)

    def test_wxh_width_only_is_valid(self):
        '''wxh=(256, None) — width-only — should be accepted.'''
        try:
            self.p2s.smallp(self.df, 'cat', self.xyp, wxh=(256, None))
        except ValueError as e:
            self.fail(f'Unexpected ValueError for wxh=(256, None): {e}')

    def test_wxh_height_only_is_valid(self):
        '''wxh=(None, 256) — height-only — should be accepted.'''
        try:
            self.p2s.smallp(self.df, 'cat', self.xyp, wxh=(None, 256))
        except ValueError as e:
            self.fail(f'Unexpected ValueError for wxh=(None, 256): {e}')

    # ── tfield column type validation ────────────────────────────────────────

    def test_tfield_string_wrong_column_type_raises(self):
        '''category_by as a tfield string pointing to an int column → ValueError.

        A tfield like "int_col|Y_mp" means "apply year-monthly binning to int_col".
        Since int_col is pl.Int64 (not pl.Date/pl.Datetime), _validate_tfield_column_
        must raise ValueError.
        '''
        tfield = self.p2s.tField('int_col', self.p2s.LT_Y_mp)  # e.g. "int_col|Y_mp"
        with self.assertRaises(ValueError):
            self.p2s.smallp(self.df, tfield, self.xyp, wxh=(256, 256))

    def test_tfield_in_tuple_wrong_type_raises(self):
        '''A 2-tuple category_by where one element is a tfield on a non-datetime column → ValueError.

        Tests the elif isinstance(self.category_by, tuple) branch in __validateInput__
        (smallp.py lines 134-137) and __addColumnsToDataFrame__ tuple path (lines 147-150).
        '''
        tfield = self.p2s.tField('int_col', self.p2s.LT_Y_mp)
        with self.assertRaises(ValueError):
            self.p2s.smallp(self.df, (tfield, 'cat'), self.xyp, wxh=(256, 256))

    def test_tfield_datetime_column_is_valid(self):
        '''A tfield pointing to a real datetime column should not raise.'''
        tfield = self.p2s.tField('ts', self.p2s.LT_Y_mp)
        try:
            self.p2s.smallp(self.df, tfield, self.xyp, wxh=(256, 256))
        except ValueError as e:
            self.fail(f'Unexpected ValueError for valid datetime tfield: {e}')

    def test_tfield_in_tuple_datetime_column_is_valid(self):
        '''A tfield in a tuple pointing to a datetime column should pass validation
        and exercise the __addColumnsToDataFrame__ tuple path.'''
        tfield = self.p2s.tField('ts', self.p2s.LT_Y_mp)
        try:
            self.p2s.smallp(self.df, (tfield, 'cat'), self.xyp, wxh=(256, 256))
        except ValueError as e:
            self.fail(f'Unexpected ValueError for valid tfield-in-tuple: {e}')

    # ── missing required args ─────────────────────────────────────────────────

    def test_no_template_raises(self):
        '''Omitting a template must raise ValueError.'''
        with self.assertRaises(ValueError):
            self.p2s.smallp(self.df, 'cat', wxh=(256, 256))

    def test_no_category_by_raises(self):
        '''Omitting category_by must raise ValueError.'''
        with self.assertRaises(ValueError):
            self.p2s.smallp(self.df, self.xyp, wxh=(256, 256))


if __name__ == '__main__':
    unittest.main()
