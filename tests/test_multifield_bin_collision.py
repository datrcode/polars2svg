'''
Multi-field bin / color values must not collide when joined into a single
internal grouping-key string.

Before the fix, multi-field bins and colors were concatenated with a printable
'|' (e.g. ``pl.concat_str(cols, separator='|')``), so two genuinely distinct
field tuples whose values happened to contain '|' collapsed into the same key:

    ('x|y', 'z')  -> 'x|y|z'
    ('x',  'y|z') -> 'x|y|z'    # same string  ->  merged into one bar/slice

The framework now joins with a non-printable separator (``Polars2SVG.MULTI_FIELD_SEP``,
ASCII Unit Separator 0x1f) that can never appear in real data, and strips it back
to a visible '|' at every text-display site via ``formatMultiFieldValue`` — so the
grouping is collision-safe while labels are unchanged and no control character
ever leaks into the (XML) SVG output.
'''
import unittest
import re
import polars as pl
from polars2svg import Polars2SVG


def _collision_df():
    '''Two distinct 2-field tuples that both join to 'x|y|z' under a '|' separator.'''
    return pl.DataFrame({
        'a': ['x|y', 'x',   'x|y', 'x'  ],
        'b': ['z',   'y|z', 'z',   'y|z'],
        'v': [1,     2,     3,     4    ],
    })


class TestMultiFieldSeparator(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    # ── the shared primitive ─────────────────────────────────────────────────

    def test_separator_is_non_printable(self):
        self.assertFalse(self.p2s.MULTI_FIELD_SEP.isprintable())
        self.assertEqual(self.p2s.MULTI_FIELD_SEP, '\x1f')

    def test_format_restores_visible_pipe(self):
        joined = 'x|y' + self.p2s.MULTI_FIELD_SEP + 'z'
        self.assertEqual(self.p2s.formatMultiFieldValue(joined), 'x|y|z')

    def test_format_single_field_is_noop(self):
        # a plain value never contains the separator
        self.assertEqual(self.p2s.formatMultiFieldValue('plain'), 'plain')
        self.assertEqual(self.p2s.formatMultiFieldValue(42), '42')


class TestHistopBinCollision(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def test_distinct_tuples_stay_distinct_bins(self):
        '''('x|y','z') and ('x','y|z') must produce two bars, not one.'''
        t = self.p2s.histop(_collision_df(), bin_by=('a', 'b'))
        self.assertEqual(len(t._sorted_bins_), 2)

    def test_no_control_char_leaks_into_svg(self):
        svg = self.p2s.histop(_collision_df(), bin_by=('a', 'b'))._repr_svg_()
        self.assertNotIn('\x1f', svg)

    def test_label_shows_visible_pipe(self):
        '''Multi-field bin labels are displayed with a visible '|' separator.'''
        df = pl.DataFrame({'a': ['A', 'A', 'B', 'B'],
                           'b': ['x', 'x', 'y', 'y'],
                           'v': [1, 2, 3, 4]})
        svg   = self.p2s.histop(df, bin_by=('a', 'b'), wxh=(400, 300))._repr_svg_()
        texts = re.findall(r'>([^<]*)<', svg)
        # the bin-value labels 'A|x' and 'B|y' both appear as drawn text
        self.assertTrue(any(x == 'A|x' for x in texts))
        self.assertTrue(any(x == 'B|y' for x in texts))

    def test_color_tuple_stays_distinct(self):
        '''Colliding color tuples must not merge into a single color segment.'''
        df = pl.DataFrame({'bin': ['b'] * 4,
                           'ca':  ['x|y', 'x',   'x|y', 'x'  ],
                           'cb':  ['z',   'y|z', 'z',   'y|z'],
                           'v':   [1, 2, 3, 4]})
        t = self.p2s.histop(df, bin_by='bin', color=('ca', 'cb'))
        self.assertEqual(t._color_field_, '__color__')
        self.assertEqual(t.df_agg['__color__'].n_unique(), 2)
        self.assertNotIn('\x1f', t._repr_svg_())


class TestPiepBinCollision(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def test_distinct_tuples_stay_distinct_slices(self):
        t = self.p2s.piep(_collision_df(), bin_by=('a', 'b'))
        self.assertEqual(len(t._slices_), 2)

    def test_no_control_char_leaks_into_svg(self):
        svg = self.p2s.piep(_collision_df(), bin_by=('a', 'b'))._repr_svg_()
        self.assertNotIn('\x1f', svg)

    def test_color_tuple_stays_distinct(self):
        '''Within one slice, two colliding color tuples read as a 2-value set (not 1).'''
        df = pl.DataFrame({'bin': ['A', 'A', 'A', 'A'],
                           'ca':  ['x|y', 'x',   'x|y', 'x'  ],
                           'cb':  ['z',   'y|z', 'z',   'y|z'],
                           'v':   [1, 2, 3, 4]})
        t = self.p2s.piep(df, bin_by='bin', color=('ca', 'cb'))
        self.assertEqual(t._color_field_, '__color__')
        self.assertEqual(t.df['__color__'].n_unique(), 2)
        self.assertEqual(t.df_agg['__cset_n__'].to_list(), [2])
        self.assertNotIn('\x1f', t._repr_svg_())


class TestTimepColorCollision(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def test_distinct_color_tuples_stay_distinct(self):
        import datetime
        n  = 8
        df = pl.DataFrame({
            'ts': [datetime.datetime(2020, 1, 1) + datetime.timedelta(days=i % 2) for i in range(n)],
            'ca': ['x|y', 'x'] * (n // 2),
            'cb': ['z', 'y|z'] * (n // 2),
            'v':  list(range(n)),
        })
        t = self.p2s.timep(df, time='ts', color=('ca', 'cb'))
        self.assertNotIn('\x1f', t._repr_svg_())
        self.assertEqual(t.df_agg['__color__'].n_unique(), 2)


if __name__ == '__main__':
    unittest.main()
