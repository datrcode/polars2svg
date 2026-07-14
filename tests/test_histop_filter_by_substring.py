"""
Tests for Histop.filterBySubstring(substring, remove_bins=False).

Histop bins are the distinct values of the bin_by column (or the '__bin__'
concatenation for tuple bin_by).  filterBySubstring does a case-insensitive
substring match against str(bin_value) for every bin in self._sorted_bins_,
then returns the original rows whose bin matched (inner) or didn't match (anti).
"""
import unittest
import polars as pl
from polars2svg import Polars2SVG


class TestHistopFilterBySubstringSimple(unittest.TestCase):
    """Single string bin_by column."""

    @classmethod
    def setUpClass(cls):
        cls.p2s = Polars2SVG()
        # Bins: 'apple' (5 rows), 'banana' (3 rows), 'apricot' (2 rows)
        cls.df = pl.DataFrame({
            'fruit': ['apple'] * 5 + ['banana'] * 3 + ['apricot'] * 2,
            'val':   list(range(10)),
        })
        cls.h = cls.p2s.histop(cls.df, 'fruit')

    def test_include_exact_match(self):
        result = self.h.filterBySubstring('banana')
        self.assertEqual(sorted(result['fruit'].to_list()), ['banana'] * 3)

    def test_include_partial_match(self):
        # 'ap' matches 'apple' and 'apricot', not 'banana'
        result = self.h.filterBySubstring('ap')
        self.assertSetEqual(set(result['fruit'].to_list()), {'apple', 'apricot'})
        self.assertEqual(len(result), 7)

    def test_include_is_case_insensitive(self):
        result = self.h.filterBySubstring('APPLE')
        self.assertEqual(sorted(result['fruit'].to_list()), ['apple'] * 5)

    def test_include_no_match_returns_empty(self):
        result = self.h.filterBySubstring('zzz')
        self.assertEqual(len(result), 0)
        self.assertIn('fruit', result.columns)

    def test_include_all_match_returns_all(self):
        # 'a' appears in apple, banana, apricot
        result = self.h.filterBySubstring('a')
        self.assertEqual(len(result), 10)

    def test_remove_matching_bins(self):
        # remove_bins=True: bins containing 'ap' (apple, apricot) are excluded
        result = self.h.filterBySubstring('ap', remove_bins=True)
        self.assertSetEqual(set(result['fruit'].to_list()), {'banana'})
        self.assertEqual(len(result), 3)

    def test_remove_no_match_keeps_all(self):
        result = self.h.filterBySubstring('zzz', remove_bins=True)
        self.assertEqual(len(result), 10)

    def test_remove_all_match_returns_empty(self):
        result = self.h.filterBySubstring('a', remove_bins=True)
        self.assertEqual(len(result), 0)

    def test_result_drops_p2s_index_column(self):
        result = self.h.filterBySubstring('apple')
        self.assertNotIn('__p2s_index__', result.columns)

    def test_result_preserves_val_column(self):
        result = self.h.filterBySubstring('apple')
        self.assertIn('val', result.columns)

    def test_result_fruit_column_present(self):
        result = self.h.filterBySubstring('apple')
        self.assertIn('fruit', result.columns)


class TestHistopFilterBySubstringTupleBinBy(unittest.TestCase):
    """Tuple bin_by: bins are concatenated as 'field1|field2' in __bin__."""

    @classmethod
    def setUpClass(cls):
        cls.p2s = Polars2SVG()
        cls.df = pl.DataFrame({
            'category': ['A', 'A', 'B', 'B', 'C'],
            'group':    ['x', 'x', 'y', 'y', 'z'],
            'val':      list(range(5)),
        })
        cls.h = cls.p2s.histop(cls.df, ('category', 'group'))

    def test_include_matches_concatenated_bin(self):
        # Bin labels are like 'A|x', 'B|y', 'C|z' — 'A' appears in 'A|x'
        result = self.h.filterBySubstring('A|x')
        self.assertEqual(len(result), 2)

    def test_include_partial_on_concat(self):
        # 'y' matches 'B|y' only
        result = self.h.filterBySubstring('y')
        self.assertEqual(len(result), 2)

    def test_result_drops_bin_column(self):
        # __bin__ is an internal column that must not leak into the result
        result = self.h.filterBySubstring('A|x')
        self.assertNotIn('__bin__', result.columns)

    def test_original_columns_preserved(self):
        result = self.h.filterBySubstring('A|x')
        self.assertIn('category', result.columns)
        self.assertIn('group', result.columns)
        self.assertIn('val', result.columns)

    def test_remove_bins_tuple(self):
        # remove 'A|x' bins → 3 rows remain (B|y x2, C|z x1)
        result = self.h.filterBySubstring('A|x', remove_bins=True)
        self.assertEqual(len(result), 3)


if __name__ == '__main__':
    unittest.main()
