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
        # A: 10 rows, 1 unique group → n_unique=1
        # B: 3  rows, 3 unique groups → n_unique=3
        # C: 6  rows, 2 unique groups → n_unique=2
        df = pl.DataFrame({
            'cat':   ['A']*10 + ['B']*3        + ['C']*6,
            'group': ['x']*10 + ['x','y','z']  + ['x','x','x','y','y','y'],
        })
        t = self.p2s.histop(df, 'cat', count='group')
        self.assertEqual(t._sorted_bins_, ['B', 'C', 'A'])

    # ── order=p2s.LABELp -- the bins' own values ──────────────────────────────

    def test_label_order_sorts_numbers_as_numbers(self):
        '''Ports: 80 before 443 before 8080, which a text sort would get wrong.  The
        counts (22 is the most common) play no part.'''
        df = pl.DataFrame({'port': [443, 80, 8080, 22, 22, 22, 80]})
        t  = self.p2s.histop(df, 'port', order=self.p2s.LABELp, descending=False)
        self.assertEqual(t._sorted_bins_, [22, 80, 443, 8080])

    def test_label_order_follows_descending_like_any_key(self):
        '''descending= keeps its one meaning -- largest first, its default -- whatever the
        key, so the labels run backwards unless it is turned off.'''
        t = self.p2s.histop(makeOrderedHistoDf(), 'cat', order=self.p2s.LABELp)
        self.assertEqual(t._sorted_bins_, ['C', 'B', 'A'])
        t = self.p2s.histop(makeOrderedHistoDf(), 'cat', order=self.p2s.LABELp, descending=False)
        self.assertEqual(t._sorted_bins_, ['A', 'B', 'C'])

    def test_label_order_puts_a_missing_value_last_either_way(self):
        df = pl.DataFrame({'cat': ['b', None, 'a', 'c', None]})
        for _desc_, _want_ in ((False, ['a', 'b', 'c', None]), (True, ['c', 'b', 'a', None])):
            with self.subTest(descending=_desc_):
                t = self.p2s.histop(df, 'cat', order=self.p2s.LABELp, descending=_desc_)
                self.assertEqual(t._sorted_bins_, _want_)

    def test_label_order_holds_for_stacked_and_boxplot_bars(self):
        df = _count_versus_field_df()
        stacked = self.p2s.histop(df, 'cat', color='grp', order=self.p2s.LABELp, descending=False)
        boxplot = self.p2s.histop(df, 'cat', count='value', style=self.p2s.BOXPLOTp,
                                  order=self.p2s.LABELp, descending=False)
        self.assertEqual(stacked._agg_type_, 'stacked')
        self.assertEqual(boxplot._agg_type_, 'boxplot')
        for _t_ in (stacked, boxplot):
            self.assertEqual(_t_._sorted_bins_, ['A', 'B', 'C'])

    def test_a_string_order_still_names_a_field_even_one_called_label(self):
        '''Why LABELp is an enum: a string order= is a column to sum, and "label" is a real
        column name in netflow datasets (UNSW-NB15's attack flag).  It must keep summing.'''
        df = pl.DataFrame({'cat': ['A', 'B', 'B', 'C'], 'label': [0, 1, 0, 5]})
        t  = self.p2s.histop(df, 'cat', order='label')
        self.assertEqual(t._sorted_bins_, ['C', 'B', 'A'])

    # ── order= on stacked and boxplot bars ────────────────────────────────────
    #
    # Both paths sorted by their count whatever order= said, with no warning -- and a
    # categorical colour makes every bar chart stacked (PLANNING.md §5,
    # C-histop-order-ignored).  In _count_versus_field_df the row-count order and the
    # field order are opposite, so a sort that ignores order= cannot pass.

    def test_a_field_order_is_honoured_when_the_bars_stack(self):
        t = self.p2s.histop(_count_versus_field_df(), 'cat', color='grp', order='value')
        self.assertEqual(t._agg_type_, 'stacked')
        self.assertEqual(t._sorted_bins_, ['C', 'B', 'A'])

    def test_a_statistic_order_is_honoured_for_boxplots(self):
        t = self.p2s.histop(_count_versus_field_df(), 'cat', count='value',
                            style=self.p2s.BOXPLOT_W_SWARMp, order=('score', self.p2s.MEANp))
        self.assertEqual(t._agg_type_, 'boxplot')
        self.assertEqual(t._sorted_bins_, ['C', 'B', 'A'])

    def test_the_default_order_is_unchanged_on_both_paths(self):
        '''ROW_COUNTp is still what the bars show: a stacked bar's total, a boxplot's rows.'''
        df = _count_versus_field_df()
        self.assertEqual(self.p2s.histop(df, 'cat', color='grp')._sorted_bins_, ['A', 'B', 'C'])
        self.assertEqual(self.p2s.histop(df, 'cat', count='value', style=self.p2s.BOXPLOTp)._sorted_bins_,
                         ['A', 'B', 'C'])


def _count_versus_field_df():
    '''Row counts A=5, B=3, C=2 -- the opposite of every field's order: value sums A=5,
    B=30, C=200 and mean scores A=1, B=2, C=3.  Two colours in each bin, so a colour of
    'grp' stacks.'''
    return pl.DataFrame({
        'cat':   ['A'] * 5 + ['B'] * 3 + ['C'] * 2,
        'grp':   ['x', 'y', 'x', 'y', 'x', 'x', 'y', 'x', 'x', 'y'],
        'value': [1] * 5 + [10] * 3 + [100] * 2,
        'score': [1.0] * 5 + [2.0] * 3 + [3.0] * 2,
    })


if __name__ == '__main__':
    unittest.main()
