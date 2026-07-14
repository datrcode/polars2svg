import unittest
import polars as pl
from polars2svg import Polars2SVG
from histop_dataframes import makeHistoDf


class TestColorOverrides(unittest.TestCase):
    def setUp(self):
        self.p2s = Polars2SVG()
        self.p2s.color_overrides_lu.clear()
        self.df  = makeHistoDf(n=200)

    # ── setColorOverrides validation ─────────────────────────────────────────

    def test_set_color_overrides_accepted(self):
        self.p2s.setColorOverrides({'A': '#ff0000', 'B': '#00ff00'})

    def test_set_color_overrides_invalid_hex_raises(self):
        with self.assertRaises(ValueError):
            self.p2s.setColorOverrides({'A': 'red'})

    def test_set_color_overrides_not_dict_raises(self):
        with self.assertRaises(ValueError):
            self.p2s.setColorOverrides([('A', '#ff0000')])

    def test_set_color_overrides_merges(self):
        self.p2s.setColorOverrides({'A': '#ff0000'})
        self.p2s.setColorOverrides({'B': '#00ff00'})
        self.assertEqual(self.p2s.color_overrides_lu, {'A': '#ff0000', 'B': '#00ff00'})

    def test_set_color_overrides_later_call_wins(self):
        self.p2s.setColorOverrides({'A': '#ff0000'})
        self.p2s.setColorOverrides({'A': '#0000ff'})
        self.assertEqual(self.p2s.color_overrides_lu['A'], '#0000ff')

    # ── removeColorOverrides ─────────────────────────────────────────────────

    def test_remove_color_overrides_single_string(self):
        self.p2s.setColorOverrides({'A': '#ff0000', 'B': '#00ff00'})
        self.p2s.removeColorOverrides('A')
        self.assertNotIn('A', self.p2s.color_overrides_lu)
        self.assertIn('B', self.p2s.color_overrides_lu)

    def test_remove_color_overrides_iterable(self):
        self.p2s.setColorOverrides({'A': '#ff0000', 'B': '#00ff00', 'C': '#0000ff'})
        self.p2s.removeColorOverrides(['A', 'C'])
        self.assertNotIn('A', self.p2s.color_overrides_lu)
        self.assertNotIn('C', self.p2s.color_overrides_lu)
        self.assertIn('B', self.p2s.color_overrides_lu)

    def test_remove_color_overrides_missing_key_silent(self):
        self.p2s.removeColorOverrides('nonexistent')

    # ── colorizeColumnPolarsOperations behavior ───────────────────────────────

    def test_colorize_ops_uses_override(self):
        self.p2s.setColorOverrides({'failure': '#ff0000'})
        df = pl.DataFrame({'status': ['failure', 'success', 'pending']})
        result = df.with_columns(
            self.p2s.colorizeColumnPolarsOperations('status').alias('hex')
        )
        self.assertEqual(result['hex'][0], '#ff0000')
        self.assertNotEqual(result['hex'][1], '#ff0000')
        self.assertNotEqual(result['hex'][2], '#ff0000')

    def test_colorize_ops_no_overrides_unchanged(self):
        df = pl.DataFrame({'val': ['A', 'B', 'C']})
        result_base = df.with_columns(
            self.p2s.colorizeColumnPolarsOperations('val').alias('hex')
        )
        self.p2s.setColorOverrides({'X': '#123456'})  # key not in data
        result_after = df.with_columns(
            self.p2s.colorizeColumnPolarsOperations('val').alias('hex')
        )
        self.assertEqual(result_base['hex'].to_list(), result_after['hex'].to_list())

    def test_clear_override_restores_hash_color(self):
        df = pl.DataFrame({'val': ['A']})
        base_color = df.with_columns(
            self.p2s.colorizeColumnPolarsOperations('val').alias('hex')
        )['hex'][0]
        self.p2s.setColorOverrides({'A': '#ff0000'})
        self.p2s.removeColorOverrides('A')
        restored_color = df.with_columns(
            self.p2s.colorizeColumnPolarsOperations('val').alias('hex')
        )['hex'][0]
        self.assertEqual(base_color, restored_color)

    # ── end-to-end histop ────────────────────────────────────────────────────

    def test_override_appears_in_histop_svg(self):
        self.p2s.setColorOverrides({'A': '#ff0000'})
        t = self.p2s.histop(self.df, 'cat', color='cat', wxh=(256, 256))
        self.assertIn('#ff0000', t.svg)

    def test_override_appears_in_histop_stacked_svg(self):
        self.p2s.setColorOverrides({'x': '#abcdef'})
        t = self.p2s.histop(self.df, 'cat', color='group', wxh=(256, 256))
        self.assertIn('#abcdef', t.svg)


if __name__ == '__main__':
    unittest.main()
