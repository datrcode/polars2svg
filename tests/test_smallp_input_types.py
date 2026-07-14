import unittest
import polars as pl
from polars2svg import Polars2SVG


class TestSmallpInputTypes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p2s = Polars2SVG()
        cls.df0 = pl.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        cls.df1 = pl.DataFrame({'a': [3, 4, 5], 'b': [7, 8, 9]})
        cls.df2 = pl.DataFrame({'a': [6, 7, 8], 'b': [1, 2, 3]})
        cls.df3 = pl.DataFrame({'a': [2, 3, 4], 'b': [5, 6, 7]})
        cls.df4 = pl.DataFrame({'a': [0, 1, 2], 'b': [9, 8, 7]})
        # full dfs for the main-df argument (required for self.df to be non-None)
        cls.full3 = pl.concat([cls.df0, cls.df1, cls.df2])
        cls.full5 = pl.concat([cls.df0, cls.df1, cls.df2, cls.df3, cls.df4])
        # Template built from df0; all sub-dfs share the same columns.
        cls.xyp = cls.p2s.xyp(df=cls.df0, x='a', y='b', wxh=(128, 128))

    # ── list category_by ──────────────────────────────────────────────────────

    def test_list_category_by_keys_are_string_indices(self):
        '''list category_by produces string-index keys '0','1','2' and places each df.'''
        smp = self.p2s.smallp(self.full3, [self.df0, self.df1, self.df2],
                               self.xyp, wxh=(384, 143))
        # _sorted_category_keys_ are string indices
        self.assertEqual(smp._sorted_category_keys_, ['0', '1', '2'])
        # geometry: 3 tiles, 3x1 grid → exact fit, no remainder
        self.assertEqual(smp.wxh_actual, (384, 143))
        self.assertEqual(smp.category_to_xy['0'], (0, 0))
        self.assertEqual(smp.category_to_xy['1'], (128, 0))
        self.assertEqual(smp.category_to_xy['2'], (256, 0))
        self.assertIs(smp.category_to_df['0'], self.df0)
        self.assertIs(smp.category_to_df['1'], self.df1)
        self.assertIs(smp.category_to_df['2'], self.df2)
        self.assertIsNone(smp.category_to_df['__remainder__'])
        self.assertIn('__renderSVG__', smp.timing_metrics)

    def test_list_category_by_single_entry(self):
        '''list category_by with a single df produces key '0' with no remainder.'''
        smp = self.p2s.smallp(self.df0, [self.df0], self.xyp, wxh=(128, 143))
        self.assertEqual(smp._sorted_category_keys_, ['0'])
        self.assertEqual(len(smp.category_to_xy), 1)
        self.assertEqual(smp.category_to_xy['0'], (0, 0))
        self.assertIs(smp.category_to_df['0'], self.df0)
        self.assertIsNone(smp.category_to_df['__remainder__'])

    def test_list_category_by_overflow_collated(self):
        '''list category_by with overflow collates excess dfs into __remainder__.

        5 dfs in a 2x1 grid (2 slots):
          slot 0 → df0 at (0,0)
          slot 1 (last) → claimed as __remainder__ at (128,0)
          dfs 1–4 are collated into __remainder__ (12 rows total)
        '''
        smp = self.p2s.smallp(
            self.full5,
            [self.df0, self.df1, self.df2, self.df3, self.df4],
            self.xyp, wxh=(256, 143))
        # 2 slots: key '0' + '__remainder__'
        self.assertEqual(len(smp.category_to_xy), 2)
        self.assertEqual(smp.category_to_xy['0'], (0, 0))
        self.assertIn('__remainder__', smp.category_to_xy)
        self.assertEqual(smp.category_to_xy['__remainder__'], (128, 0))
        self.assertIs(smp.category_to_df['0'], self.df0)
        # remainder = concat(df1, df2, df3, df4) = 12 rows total
        remainder = smp.category_to_df['__remainder__']
        self.assertIsNotNone(remainder)
        expected_len = len(self.df1) + len(self.df2) + len(self.df3) + len(self.df4)
        self.assertEqual(len(remainder), expected_len)

    # ── dict category_by ──────────────────────────────────────────────────────

    def test_dict_category_by_keys_preserved(self):
        '''dict category_by preserves string keys and maps each to its df.'''
        smp = self.p2s.smallp(
            self.full3,
            {'alpha': self.df0, 'beta': self.df1, 'gamma': self.df2},
            self.xyp, wxh=(384, 143))
        self.assertEqual(smp._sorted_category_keys_, ['alpha', 'beta', 'gamma'])
        self.assertEqual(smp.wxh_actual, (384, 143))
        self.assertEqual(smp.category_to_xy['alpha'], (0, 0))
        self.assertEqual(smp.category_to_xy['beta'],  (128, 0))
        self.assertEqual(smp.category_to_xy['gamma'], (256, 0))
        self.assertIs(smp.category_to_df['alpha'], self.df0)
        self.assertIs(smp.category_to_df['beta'],  self.df1)
        self.assertIs(smp.category_to_df['gamma'], self.df2)
        self.assertIsNone(smp.category_to_df['__remainder__'])

    def test_dict_category_by_overflow_collated(self):
        '''dict category_by with overflow collates excess dfs into __remainder__.'''
        smp = self.p2s.smallp(
            self.full5,
            {'a': self.df0, 'b': self.df1, 'c': self.df2, 'd': self.df3, 'e': self.df4},
            self.xyp, wxh=(256, 143))
        self.assertEqual(len(smp.category_to_xy), 2)
        self.assertIn('__remainder__', smp.category_to_xy)
        remainder = smp.category_to_df['__remainder__']
        self.assertIsNotNone(remainder)
        # first visible df is df0 (3 rows); remainder = df1+df2+df3+df4 = 12 rows
        self.assertEqual(len(remainder), 12)

    # ── positional vs kwargs equivalence ─────────────────────────────────────

    def test_list_positional_arg_and_kwargs_equivalent(self):
        '''Passing a list positionally and via category_by= kwarg are equivalent.'''
        full2 = pl.concat([self.df0, self.df1])
        smp1 = self.p2s.smallp(full2, [self.df0, self.df1], self.xyp, wxh=(256, 143))
        smp2 = self.p2s.smallp(full2, self.xyp,
                                category_by=[self.df0, self.df1], wxh=(256, 143))
        self.assertEqual(smp1._sorted_category_keys_, smp2._sorted_category_keys_)
        self.assertEqual(smp1.wxh_actual, smp2.wxh_actual)

    # ── sketch_only with dict keys ────────────────────────────────────────────

    def test_dict_category_by_sketch_only_renders_svg(self):
        '''dict category_by with sketch_only=True produces SVG without __renderSVG__.'''
        full2 = pl.concat([self.df0, self.df1])
        smp = self.p2s.smallp(
            full2,
            {'alpha': self.df0, 'beta': self.df1},
            self.xyp, wxh=(256, 143), sketch_only=True)
        self.assertIn('<svg', smp.svg)
        self.assertNotIn('__renderSVG__', smp.timing_metrics)


if __name__ == '__main__':
    unittest.main()
