import unittest
import polars as pl
from polars2svg import Polars2SVG
from histop_dataframes import makeHistoDf, makeOrderedHistoDf
from svg_test_utils import assert_ordered_keys


class TestHistopOrder(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    # ── order variants — smoke tests ──────────────────────────────────────────

    def test_order_default_row_count(self):
        df = makeHistoDf(n=100)
        self.p2s.histop(df, 'cat')

    def test_order_explicit_row_count(self):
        df = makeHistoDf(n=100)
        self.p2s.histop(df, 'cat', order=self.p2s.ROW_COUNTp)

    def test_order_numeric_field_sum(self):
        df = makeHistoDf(n=100)
        self.p2s.histop(df, 'cat', order='value')

    def test_order_set_tuple(self):
        '''(field, SETp) → order by n_unique of that field per bin.'''
        df = makeHistoDf(n=100)
        self.p2s.histop(df, 'cat', order=('group', self.p2s.SETp))

    def test_order_statistic_min(self):
        df = makeHistoDf(n=100)
        self.p2s.histop(df, 'cat', order=('value', self.p2s.MINp))

    def test_order_statistic_max(self):
        df = makeHistoDf(n=100)
        self.p2s.histop(df, 'cat', order=('value', self.p2s.MAXp))

    def test_order_statistic_mean(self):
        df = makeHistoDf(n=100)
        self.p2s.histop(df, 'cat', order=('score', self.p2s.MEANp))

    def test_order_statistic_median(self):
        df = makeHistoDf(n=100)
        self.p2s.histop(df, 'cat', order=('score', self.p2s.MEDIANp))

    def test_order_statistic_std(self):
        df = makeHistoDf(n=100)
        self.p2s.histop(df, 'cat', order=('score', self.p2s.STDp))

    def test_order_statistic_sum(self):
        df = makeHistoDf(n=100)
        self.p2s.histop(df, 'cat', order=('value', self.p2s.SUMp))

    # ── descending flag ───────────────────────────────────────────────────────

    def test_descending_true_default(self):
        '''Default is descending=True (most common first).'''
        t = self.p2s.histop(makeHistoDf(n=100), 'cat')
        self.assertTrue(t.descending)

    def test_descending_false(self):
        df = makeHistoDf(n=100)
        t = self.p2s.histop(df, 'cat', descending=False)
        self.assertFalse(t.descending)

    # ── ordering correctness ──────────────────────────────────────────────────

    def test_sorted_bins_descending_row_count(self):
        '''With known counts A=5, B=3, C=1 the descending order is [A, B, C].'''
        df = makeOrderedHistoDf()
        t  = self.p2s.histop(df, 'cat')
        assert_ordered_keys(self, t._sorted_bins_, ['A', 'B', 'C'])

    def test_sorted_bins_ascending_row_count(self):
        '''Ascending reverses to [C, B, A].'''
        df = makeOrderedHistoDf()
        t  = self.p2s.histop(df, 'cat', descending=False)
        assert_ordered_keys(self, t._sorted_bins_, ['C', 'B', 'A'])

    def test_sorted_bins_descending_value_sum(self):
        '''order="value" sorts by sum(value) per bin.
        A: 10+20+30+40+50=150, B: 5+15+25=45, C: 7=7 → [A, B, C].'''
        df = makeOrderedHistoDf()
        t  = self.p2s.histop(df, 'cat', order='value')
        assert_ordered_keys(self, t._sorted_bins_, ['A', 'B', 'C'])

    def test_sorted_bins_ascending_value_sum(self):
        df = makeOrderedHistoDf()
        t  = self.p2s.histop(df, 'cat', order='value', descending=False)
        assert_ordered_keys(self, t._sorted_bins_, ['C', 'B', 'A'])

    def test_sorted_bins_all_present(self):
        '''_sorted_bins_ contains exactly the set of unique bin values.'''
        df = makeHistoDf(n=100)
        t  = self.p2s.histop(df, 'cat')
        self.assertEqual(set(t._sorted_bins_), set(df['cat'].unique().to_list()))

    def test_sorted_bins_no_duplicates(self):
        df = makeHistoDf(n=100)
        t  = self.p2s.histop(df, 'cat')
        self.assertEqual(len(t._sorted_bins_), len(set(t._sorted_bins_)))

    def test_sorted_bins_set_count_matches_bar_order(self):
        '''When count is set-based (n_unique), default ordering sorts by n_unique, not row count.

        Row-count order would be: A(10), C(6), B(3)
        n_unique  order should be: B(3),  C(2), A(1)
        '''
        import polars as pl
        # A: 10 rows, 1 unique group → n_unique=1
        # B: 3  rows, 3 unique groups → n_unique=3
        # C: 6  rows, 2 unique groups → n_unique=2
        df = pl.DataFrame({
            'cat':   ['A']*10 + ['B']*3        + ['C']*6,
            'group': ['x']*10 + ['x','y','z']  + ['x','x','x','y','y','y'],
        })
        t = self.p2s.histop(df, 'cat', count='group')
        self.assertEqual(t._sorted_bins_, ['B', 'C', 'A'])


if __name__ == '__main__':
    unittest.main()
