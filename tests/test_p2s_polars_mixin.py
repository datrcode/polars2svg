import unittest
import polars as pl
from polars2svg import Polars2SVG


class TestPolarsConcatString(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def _apply(self, template, df):
        exprs = self.p2s.polarsConcatString(template)
        return df.select(pl.concat_str(exprs).alias('out'))['out'].to_list()

    def test_literal_only(self):
        # All-literal concat_str produces a single-row result independent of df length
        df = pl.DataFrame({'x': [1, 2]})
        result = self._apply('hello', df)
        self.assertEqual(result[0], 'hello')

    def test_single_column(self):
        df = pl.DataFrame({'col': ['a', 'b']})
        result = self._apply('{col}', df)
        self.assertEqual(result, ['a', 'b'])

    def test_literal_plus_column(self):
        df = pl.DataFrame({'val': ['X', 'Y']})
        result = self._apply('item={val}', df)
        self.assertEqual(result, ['item=X', 'item=Y'])

    def test_two_columns_with_separator(self):
        df = pl.DataFrame({'a': ['foo', 'bar'], 'b': ['1', '2']})
        result = self._apply('{a}:{b}', df)
        self.assertEqual(result, ['foo:1', 'bar:2'])

    def test_float_format_rounds_correctly(self):
        df = pl.DataFrame({'val': [3.14159, 2.71828]})
        result = self._apply('{val:.2f}', df)
        self.assertEqual(result, ['3.14', '2.72'])

    def test_float_format_zero_decimals(self):
        # round(0).cast(String) in Polars yields "4.0" not "4"
        df = pl.DataFrame({'val': [3.7, 2.1]})
        result = self._apply('{val:.0f}', df)
        self.assertEqual(result, ['4.0', '2.0'])

    def test_mixed_literal_column_format(self):
        df = pl.DataFrame({'name': ['Alice', 'Bob'], 'score': [9.876, 5.432]})
        result = self._apply('{name}: {score:.1f}', df)
        self.assertEqual(result, ['Alice: 9.9', 'Bob: 5.4'])

    def test_zero_padded_integer_format(self):
        df = pl.DataFrame({'col': [42, 5, 1000]})
        result = self._apply('{col:03d}', df)
        self.assertEqual(result, ['042', '005', '1000'])

    def test_no_column_references_returns_literal_list(self):
        exprs = self.p2s.polarsConcatString('static text')
        self.assertEqual(len(exprs), 1)

    def test_trailing_literal_included(self):
        df = pl.DataFrame({'x': ['A']})
        result = self._apply('{x} units', df)
        self.assertEqual(result, ['A units'])

    def test_leading_literal_included(self):
        df = pl.DataFrame({'x': ['7']})
        result = self._apply('val={x}', df)
        self.assertEqual(result, ['val=7'])


if __name__ == '__main__':
    unittest.main()
