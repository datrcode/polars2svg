import unittest
import polars as pl
from polars2svg import Polars2SVG, InvalidSpecError


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

    # Red has the most rows but the fewest distinct cats (2 against blue's 3), and cat
    # 'b' is shared -- so every set-counting mode puts blue first where a row count
    # puts red first.
    _SET_DF_ = pl.DataFrame({'cat': ['a', 'a', 'a', 'a', 'a', 'b', 'c', 'd', 'b'],
                             'clr': ['red'] * 5 + ['blue'] * 3 + ['red']})

    # The order colorizeOrder promises: largest total first, ties by colour value, descending
    @staticmethod
    def ordered(totals: dict) -> list:
        return sorted(totals, key=lambda c: (totals[c], c), reverse=True)

    # {colour: number of distinct `fields` combinations it appears in}, counted by hand
    def distinctCombos(self, fields: tuple) -> dict:
        _seen_ = {tuple(r[f] for f in fields) + (r['clr'],) for r in self._SET_DF_.iter_rows(named=True)}
        return {c: sum(1 for s in _seen_ if s[-1] == c) for c in {s[-1] for s in _seen_}}

    def test_the_set_frame_separates_set_counts_from_row_counts(self):
        self.assertEqual(self.p2s.colorizeOrder(self._SET_DF_, self.p2s.ROW_COUNTp, 'clr'), ['red', 'blue'])
        self.assertEqual(self.distinctCombos(('cat',)), {'red': 2, 'blue': 3})

    def assertValidOrder(self, result, expected_values):
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        self.assertEqual(set(result), set(expected_values))

    # ── ROW COUNTING ────────────────────────────────────────────────────────

    def test_row_count(self):
        '''count=ROW_COUNTp, color=str → line 30-31'''
        result = self.p2s.colorizeOrder(self._df_, self.p2s.ROW_COUNTp, 'clr')
        self.assertValidOrder(result, {'red', 'blue'})
        self.assertEqual(result[0], 'red')   # 4 red rows > 2 blue rows

    # ── SCALAR COUNTING TUPLE VARIANTS ──────────────────────────────────────

    def test_scalar_tuple_1(self):
        '''count=('qty',) len-1 numeric tuple → line 37-38'''
        result = self.p2s.colorizeOrder(self._df_, ('qty',), 'clr')
        self.assertValidOrder(result, {'red', 'blue'})
        self.assertEqual(result[0], 'red')   # red total 10 > blue total 2

    def test_scalar_tuple_scalar_enum(self):
        '''count=('qty', SCALARp) len-2 numeric+enum tuple → line 39-40'''
        result = self.p2s.colorizeOrder(self._df_, ('qty', self.p2s.SCALARp), 'clr')
        self.assertValidOrder(result, {'red', 'blue'})
        self.assertEqual(result[0], 'red')

    # ── SET COUNTING ─────────────────────────────────────────────────────────

    def test_set_count_self(self):
        '''count == color (same column, self-counting): every colour counts once, so the
        tie falls to the colour value -- red first here, though blue has three rows to one.'''
        df = pl.DataFrame({'clr': ['blue', 'blue', 'blue', 'red']})
        result = self.p2s.colorizeOrder(df, 'clr', 'clr')
        self.assertValidOrder(result, {'red', 'blue'})
        self.assertEqual(result, ['red', 'blue'])
        self.assertEqual(self.p2s.colorizeOrder(df, self.p2s.ROW_COUNTp, 'clr'), ['blue', 'red'])

    def test_set_count_row_count_in_color_str_count(self):
        '''count=str, ROW_COUNTp in color tuple → line 46-47'''
        result = self.p2s.colorizeOrder(self._SET_DF_, 'cat', ('clr', self.p2s.ROW_COUNTp))
        # Color is ('clr', ROW_COUNTp): string part is 'clr', so color values are red/blue,
        # each counted by the distinct cats it appears with
        self.assertValidOrder(result, {'red', 'blue'})
        self.assertEqual(result, self.ordered(self.distinctCombos(('cat',))))

    def test_set_count_row_count_in_color_tuple_count(self):
        '''count=tuple, ROW_COUNTp in color tuple → line 48-51'''
        result = self.p2s.colorizeOrder(self._SET_DF_, ('cat',), ('clr', self.p2s.ROW_COUNTp))
        self.assertValidOrder(result, {'red', 'blue'})
        self.assertEqual(result, self.ordered(self.distinctCombos(('cat',))))

    def test_set_count_string_col(self):
        '''count=str (categorical, not same as color) → line 52-58'''
        # Each distinct cat is one unit, split evenly between the colours that share it.
        # Red has three cats, but shares two of them three ways; blue has two to itself.
        # So splitting puts blue first, where counting distinct cats puts red first.
        df = pl.DataFrame({'cat': ['a', 'b', 'b', 'b', 'c', 'c', 'c', 'd', 'e'],
                           'clr': ['red', 'red', 'green', 'yellow', 'red', 'green', 'yellow', 'blue', 'blue']})
        result = self.p2s.colorizeOrder(df, 'cat', 'clr')
        self.assertValidOrder(result, {'red', 'green', 'yellow', 'blue'})
        _split_, _whole_ = {}, {}
        for _cat_ in set(df['cat']):
            _clrs_ = set(df.filter(pl.col('cat') == _cat_)['clr'])
            for _c_ in _clrs_:
                _split_[_c_] = _split_.get(_c_, 0.0) + 1.0 / len(_clrs_)
                _whole_[_c_] = _whole_.get(_c_, 0) + 1
        self.assertNotEqual(self.ordered(_split_), self.ordered(_whole_))
        self.assertEqual(result, self.ordered(_split_))

    def test_set_count_tuple_col(self):
        '''count=tuple of strings (multi-field set counting) → line 60-72'''
        result = self.p2s.colorizeOrder(self._SET_DF_, ('cat', 'clr'), 'clr')
        self.assertValidOrder(result, {'red', 'blue'})
        self.assertEqual(result, self.ordered(self.distinctCombos(('cat',))))

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
        with self.assertRaises(InvalidSpecError):
            self.p2s.colorizeOrder(self._df_, object(), 'clr')


if __name__ == '__main__':
    unittest.main()
