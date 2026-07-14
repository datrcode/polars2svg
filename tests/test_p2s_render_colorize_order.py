import unittest
import polars as pl
from polars2svg import Polars2SVG


class TestColorizeOrder(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()
        # 4 red rows (qty 1,2,3,4) and 2 blue rows (qty 1,1)
        self._df_ = pl.DataFrame({
            'cat': ['a',   'b',   'a',   'b',   'c',    'd'],
            'qty': [1.0,   2.0,   3.0,   4.0,   1.0,    1.0],
            'clr': ['red', 'red', 'red', 'red', 'blue', 'blue'],
        })

    def _assert_valid_order(self, result, expected_values):
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        self.assertEqual(set(result), set(expected_values))

    # ── ROW COUNTING ────────────────────────────────────────────────────────

    def test_row_count(self):
        '''count=ROW_COUNTp, color=str → line 30-31'''
        result = self.p2s.colorizeOrder(self._df_, self.p2s.ROW_COUNTp, 'clr')
        self._assert_valid_order(result, {'red', 'blue'})
        self.assertEqual(result[0], 'red')   # 4 red rows > 2 blue rows

    # ── SCALAR COUNTING TUPLE VARIANTS ──────────────────────────────────────

    def test_scalar_tuple_1(self):
        '''count=('qty',) len-1 numeric tuple → line 37-38'''
        result = self.p2s.colorizeOrder(self._df_, ('qty',), 'clr')
        self._assert_valid_order(result, {'red', 'blue'})
        self.assertEqual(result[0], 'red')   # red total 10 > blue total 2

    def test_scalar_tuple_scalar_enum(self):
        '''count=('qty', SCALARp) len-2 numeric+enum tuple → line 39-40'''
        result = self.p2s.colorizeOrder(self._df_, ('qty', self.p2s.SCALARp), 'clr')
        self._assert_valid_order(result, {'red', 'blue'})
        self.assertEqual(result[0], 'red')

    # ── SET COUNTING ─────────────────────────────────────────────────────────

    def test_set_count_self(self):
        '''count == color (same column, self-counting) → line 44-45'''
        result = self.p2s.colorizeOrder(self._df_, 'clr', 'clr')
        self._assert_valid_order(result, {'red', 'blue'})

    def test_set_count_row_count_in_color_str_count(self):
        '''count=str, ROW_COUNTp in color tuple → line 46-47'''
        result = self.p2s.colorizeOrder(self._df_, 'cat', ('clr', self.p2s.ROW_COUNTp))
        # Color is ('clr', ROW_COUNTp): string part is 'clr', so color values are red/blue
        self._assert_valid_order(result, {'red', 'blue'})

    def test_set_count_row_count_in_color_tuple_count(self):
        '''count=tuple, ROW_COUNTp in color tuple → line 48-51'''
        result = self.p2s.colorizeOrder(self._df_, ('cat',), ('clr', self.p2s.ROW_COUNTp))
        self._assert_valid_order(result, {'red', 'blue'})

    def test_set_count_string_col(self):
        '''count=str (categorical, not same as color) → line 52-58'''
        result = self.p2s.colorizeOrder(self._df_, 'cat', 'clr')
        self._assert_valid_order(result, {'red', 'blue'})

    def test_set_count_tuple_col(self):
        '''count=tuple of strings (multi-field set counting) → line 60-72'''
        result = self.p2s.colorizeOrder(self._df_, ('cat', 'clr'), 'clr')
        self._assert_valid_order(result, {'red', 'blue'})

    # ── TUPLE COLOR (CONCATENATED) ────────────────────────────────────────────

    def test_tuple_color_concatenated_numeric_count(self):
        '''color is a tuple of strings → concatenated column → lines 20-26'''
        result = self.p2s.colorizeOrder(self._df_, 'qty', ('clr', 'cat'))
        # Color values are concatenations of the two fields joined by the internal
        # non-printable MULTI_FIELD_SEP (shown as '|' at display time), e.g.
        # "red\x1fa", "red\x1fb", "blue\x1fc", "blue\x1fd".
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        for val in result:
            self.assertIn(self.p2s.MULTI_FIELD_SEP, val)

    def test_tuple_color_row_count(self):
        '''color is a tuple, count=ROW_COUNTp → tuple color + row counting'''
        result = self.p2s.colorizeOrder(self._df_, self.p2s.ROW_COUNTp, ('clr', 'cat'))
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

    # ── CONSISTENCY: colorizeOrder → colorizeBar pipeline ────────────────────

    def test_order_drives_bar_consistently(self):
        '''colorizeOrder result can be fed into colorizeBar without error.'''
        color_order = self.p2s.colorizeOrder(self._df_, self.p2s.ROW_COUNTp, 'clr')
        svg = self.p2s.colorizeBar(self._df_, (10, 5, 16, 200), self.p2s.ROW_COUNTp, 'clr',
                                   color_order=color_order)
        self.assertIn('<rect', svg)

    def test_unknown_count_raises(self):
        '''Unrecognized count type raises an exception.'''
        with self.assertRaises(Exception):
            self.p2s.colorizeOrder(self._df_, object(), 'clr')


if __name__ == '__main__':
    unittest.main()
