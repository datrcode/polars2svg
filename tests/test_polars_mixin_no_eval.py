#
# test_polars_mixin_no_eval.py
#
# Regression test for the removal of the injectable eval() in
# P2SPolarsMixin.polarsFilterColumnsWithNaNs.
#
# The old implementation string-interpolated column names into a Python eval():
#
#     df.filter(eval('&'.join(f'(pl.col("{col}").is_not_null())' ...)))
#
# which let a hostile DataFrame column name execute arbitrary code.  The fix
# builds the filter expression directly from pl.col(...).is_not_null().  These
# tests assert (a) correct filtering results even with a column name that would
# have been an injection payload, and (b) that the source no longer contains an
# eval( call.
#
import os
import unittest

import polars as pl

from polars2svg import Polars2SVG


class TestPolarsMixinNoEval(unittest.TestCase):
    def setUp(self):
        self.p2s = Polars2SVG()

    def test_filters_nulls_normal_columns(self):
        df = pl.DataFrame({
            'a': [1, None, 3, None],
            'b': [10, 20, None, 40],
        })
        out = self.p2s.polarsFilterColumnsWithNaNs(df, ('a', 'b'))
        # Only row 0 (a=1, b=10) has both columns non-null.
        self.assertEqual(out.height, 1)
        self.assertEqual(out['a'].to_list(), [1])
        self.assertEqual(out['b'].to_list(), [10])

    def test_single_column(self):
        df = pl.DataFrame({'a': [1, None, 3]})
        out = self.p2s.polarsFilterColumnsWithNaNs(df, ('a',))
        self.assertEqual(out['a'].to_list(), [1, 3])

    def test_empty_cols_returns_unchanged(self):
        df = pl.DataFrame({'a': [1, None, 3]})
        out = self.p2s.polarsFilterColumnsWithNaNs(df, ())
        self.assertEqual(out.height, df.height)

    def test_hostile_column_name_is_a_literal_column(self):
        # A column name that, under the old eval() implementation, would have
        # broken out of the pl.col("...") string and executed injected code.
        # Under the fixed implementation it is treated as an ordinary (literal)
        # column name and filtered on for nulls like any other.
        hostile = 'x") | (pl.lit(True)'
        df = pl.DataFrame({
            hostile: [1, None, 5],
            'ok':    [7, 8, None],
        })
        out = self.p2s.polarsFilterColumnsWithNaNs(df, (hostile, 'ok'))
        # Row 0 is the only one with both the hostile column and 'ok' non-null.
        self.assertEqual(out.height, 1)
        self.assertEqual(out[hostile].to_list(), [1])
        self.assertEqual(out['ok'].to_list(), [7])

    def test_source_contains_no_eval(self):
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        src_path = os.path.join(here, 'polars2svg', 'p2s_polars_mixin.py')
        with open(src_path) as f:
            source = f.read()
        self.assertNotIn('eval(', source)


if __name__ == '__main__':
    unittest.main()
