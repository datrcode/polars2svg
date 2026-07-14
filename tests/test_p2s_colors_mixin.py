import unittest
import re
import polars as pl
from polars2svg import Polars2SVG


HEX_RE = re.compile(r'^#[0-9a-fA-F]{6}$')


class TestColorTyped(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def test_known_keys_return_valid_hex(self):
        pairs = [
            ('background', 'default'),
            ('axis',       'default'),
            ('axis',       'label'),
            ('axis',       'min'),
            ('axis',       'max'),
            ('error',      'default'),
            ('label',      'defaultfg'),
            ('label',      'inner'),
        ]
        for _type_, _subtype_ in pairs:
            with self.subTest(type=_type_, subtype=_subtype_):
                c = self.p2s.colorTyped(_type_, _subtype_)
                self.assertRegex(c, HEX_RE)

    def test_unknown_key_raises(self):
        with self.assertRaises(KeyError):
            self.p2s.colorTyped('nonexistent', 'key')


class TestColorSpectrumTuples(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def test_length_matches_palette(self):
        tuples = self.p2s.colorSpectrumTuples()
        self.assertEqual(len(tuples), len(self.p2s.spectrum_palette))

    def test_mix_values_span_zero_to_one(self):
        tuples = self.p2s.colorSpectrumTuples()
        mixes = [t[3] for t in tuples]
        self.assertAlmostEqual(mixes[0],  0.0, places=9)
        self.assertAlmostEqual(mixes[-1], 1.0, places=9)

    def test_mix_values_are_monotone(self):
        mixes = [t[3] for t in self.p2s.colorSpectrumTuples()]
        for a, b in zip(mixes, mixes[1:]):
            self.assertLess(a, b)

    def test_rgb_values_in_unit_range(self):
        for r, g, b, _ in self.p2s.colorSpectrumTuples():
            self.assertGreaterEqual(r, 0.0)
            self.assertLessEqual(r, 1.0)
            self.assertGreaterEqual(g, 0.0)
            self.assertLessEqual(g, 1.0)
            self.assertGreaterEqual(b, 0.0)
            self.assertLessEqual(b, 1.0)


class TestColorSpectrumPolarsOperations(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def _apply(self, values):
        df = pl.DataFrame({'n': values})
        ops = self.p2s.colorSpectrumPolarsOperations('n', 'r', 'g', 'b')
        return df.with_columns(ops)

    def test_output_columns_present(self):
        result = self._apply([0.0, 0.5, 1.0])
        self.assertIn('r', result.columns)
        self.assertIn('g', result.columns)
        self.assertIn('b', result.columns)

    def test_boundary_values_produce_valid_rgb(self):
        result = self._apply([0.0, 1.0])
        for row in result.iter_rows(named=True):
            for ch in ('r', 'g', 'b'):
                self.assertGreaterEqual(row[ch], 0.0)
                self.assertLessEqual(row[ch], 1.0)

    def test_hostile_column_name_is_not_evaled(self):
        # Column names are no longer interpolated into eval()'d strings, so a name
        # containing Python/polars metacharacters is treated as a plain column name
        # (no injection, no spurious rejection). It should produce valid RGB output.
        hostile = 'n") + __import__("os").system("echo pwned'
        df  = pl.DataFrame({hostile: [0.0, 0.5, 1.0]})
        ops = self.p2s.colorSpectrumPolarsOperations(hostile, 'r', 'g', 'b')
        result = df.with_columns(ops)
        for row in result.iter_rows(named=True):
            for ch in ('r', 'g', 'b'):
                self.assertGreaterEqual(row[ch], 0.0)
                self.assertLessEqual(row[ch], 1.0)

    def test_no_python_eval_in_source(self):
        import polars2svg.p2s_colors_mixin as _mod
        with open(_mod.__file__) as _f:
            self.assertNotIn('eval(', _f.read())

    def test_produces_different_colors_at_endpoints(self):
        result = self._apply([0.0, 1.0])
        row0 = result.row(0)
        row1 = result.row(1)
        self.assertNotEqual(row0, row1)


class TestColorSpectrumPolarsOperationsLimitedFive(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def test_output_columns_present(self):
        df = pl.DataFrame({'n': [0.0, 0.25, 0.5, 0.75, 1.0]})
        ops = self.p2s.colorSpectrumPolarsOperations_LIMITED_TO_EXACTLY_FIVE('n', 'r', 'g', 'b')
        result = df.with_columns(ops)
        for col in ('r', 'g', 'b'):
            self.assertIn(col, result.columns)

    def test_values_in_defined_range_produce_valid_rgb(self):
        # LIMITED_TO_EXACTLY_FIVE was designed for 5 palette entries; the palette now has 10,
        # so only n values up to the 5th tuple's mix position are safely in-range.
        tuples = self.p2s.colorSpectrumTuples()
        safe_max = tuples[4][3]
        df = pl.DataFrame({'n': [0.0, safe_max / 2.0, safe_max]})
        ops = self.p2s.colorSpectrumPolarsOperations_LIMITED_TO_EXACTLY_FIVE('n', 'r', 'g', 'b')
        result = df.with_columns(ops)
        for row in result.iter_rows(named=True):
            for ch in ('r', 'g', 'b'):
                self.assertGreaterEqual(row[ch], 0.0)
                self.assertLessEqual(row[ch], 1.0)


class TestGrayscaleSpectrumPolarsOperations(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def _apply(self, values, light_gray=0.8):
        df = pl.DataFrame({'n': values})
        ops = self.p2s.grayscaleSpectrumPolarsOperations('n', 'r', 'g', 'b', light_gray)
        return df.with_columns(ops)

    def test_output_columns_present(self):
        result = self._apply([0.0, 0.5, 1.0])
        for col in ('r', 'g', 'b'):
            self.assertIn(col, result.columns)

    def test_zero_input_yields_light_gray(self):
        result = self._apply([0.0], light_gray=0.8)
        row = result.row(0, named=True)
        self.assertAlmostEqual(row['r'], 0.8, places=5)
        self.assertAlmostEqual(row['g'], 0.8, places=5)
        self.assertAlmostEqual(row['b'], 0.8, places=5)

    def test_one_input_yields_black(self):
        result = self._apply([1.0])
        row = result.row(0, named=True)
        self.assertAlmostEqual(row['r'], 0.0, places=5)
        self.assertAlmostEqual(row['g'], 0.0, places=5)
        self.assertAlmostEqual(row['b'], 0.0, places=5)

    def test_custom_light_gray(self):
        result = self._apply([0.0], light_gray=0.5)
        row = result.row(0, named=True)
        self.assertAlmostEqual(row['r'], 0.5, places=5)

    def test_rgb_channels_are_equal(self):
        result = self._apply([0.0, 0.25, 0.5, 0.75, 1.0])
        for row in result.iter_rows(named=True):
            self.assertAlmostEqual(row['r'], row['g'], places=9)
            self.assertAlmostEqual(row['g'], row['b'], places=9)


class TestHexColorFromRGBTriplesPolarsOperations(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def _convert(self, reds, greens, blues):
        df = pl.DataFrame({'r': reds, 'g': greens, 'b': blues})
        expr = self.p2s.hexColorFromRGBTriplesPolarsOperations('r', 'g', 'b')
        return df.select(expr.alias('hex'))['hex'].to_list()

    def test_pure_red(self):
        result = self._convert([1.0], [0.0], [0.0])
        self.assertEqual(result[0], '#ff0000')

    def test_pure_green(self):
        result = self._convert([0.0], [1.0], [0.0])
        self.assertEqual(result[0], '#00ff00')

    def test_pure_blue(self):
        result = self._convert([0.0], [0.0], [1.0])
        self.assertEqual(result[0], '#0000ff')

    def test_black(self):
        result = self._convert([0.0], [0.0], [0.0])
        self.assertEqual(result[0], '#000000')

    def test_white(self):
        result = self._convert([1.0], [1.0], [1.0])
        self.assertEqual(result[0], '#ffffff')

    def test_output_is_valid_hex_format(self):
        result = self._convert([0.5], [0.3], [0.7])
        self.assertRegex(result[0], HEX_RE)

    def test_multiple_rows(self):
        result = self._convert([0.0, 1.0], [0.0, 0.0], [0.0, 0.0])
        self.assertEqual(result[0], '#000000')
        self.assertEqual(result[1], '#ff0000')


if __name__ == '__main__':
    unittest.main()
