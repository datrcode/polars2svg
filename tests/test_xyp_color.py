"""xyp across colour modes.  Every render passes XypSweepAssertions (contract, dots in the
plot, lazy == eager), and a categorical colour -- CSETp, or any string field -- paints each
dot in the colour of a value it holds, as polars writes that value.  These used to assert
nothing -- "pass = no crash" (PLANNING.md V11)."""
import re
import unittest
import polars as pl
from polars2svg import Polars2SVG

from random_dataframe import randomDataFrame
from svg_test_utils import XypSweepAssertions

class Testxyp_color(XypSweepAssertions, unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def assertColorMode(self, df, x, y, color, lazy: bool = True) -> None:
        with self.subTest(x=x, y=y, color=str(color)):
            _xyp_ = self.assertCleanXyp(df, x, y, lazy=lazy, color=color)
            # categorical: CSETp itself, or a string field named bare (or as a 1-tuple) -- a string
            # field under a magnitude, stretched or CSET_* mode is a spectrum of its counts
            _field_ = color if isinstance(color, str) else (color[0] if isinstance(color, tuple) else None)
            _modes_ = set(color[1:]) if isinstance(color, tuple) else set()
            _categorical_ = _field_ is not None and (_modes_ == {self.p2s.CSETp} or (not _modes_ and df.schema[_field_] == pl.String))
            _group_ = self._DOT_GROUP_.search(_xyp_.svg)
            if _categorical_ and _group_ is not None:
                _kept_ = df.filter(pl.col(x).is_not_null() & pl.col(y).is_not_null() & pl.col(_field_).is_not_null())
                # ...or the default colour, where a pixel holds several values
                _allowed_ = {c.lower() for c in self.p2s.colors(_kept_[_field_].cast(pl.String).to_list()).values()}
                _allowed_.add(self.p2s.colorTyped('data', 'default').lower())
                self.assertLessEqual({f.lower() for f in re.findall(r'fill="([^"]*)"', _group_.group(2))}, _allowed_,
                                     'a dot is not in the colour of any value it holds')

    def test_a_pixel_mixing_categories_takes_the_default_colour(self):
        '''Two rows on one pixel, of categories "-1" and "a": the dot is the default colour.
        It used to be the hash of "-1" -- the colour a real category named "-1" gets, so
        "mixed" and "-1" could not be told apart (PLANNING.md §5 C-xyp-cset-mixed-pixel).'''
        df = pl.DataFrame({'x': [1.0, 1.0, 5.0, 9.0], 'y': [1.0, 1.0, 5.0, 9.0], 'c': ['-1', 'a', '-1', 'a']})
        _xyp_ = self.assertCleanXyp(df, 'x', 'y', color=('c', self.p2s.CSETp))
        _group_ = self._DOT_GROUP_.search(_xyp_.svg)
        _dots_ = sorted((float(x), fill) for x, fill in
                        re.findall(r'<(?:rect x|circle cx)="([-\d.]+)"[^>]*fill="(#[0-9a-f]{6})"', _group_.group(2)))
        self.assertEqual([fill for _, fill in _dots_],
                         [self.p2s.colorTyped('data', 'default'), self.p2s.color('-1'), self.p2s.color('a')])

    def test_colorSingles(self, samples=1):
        for _sample_ in range(samples):
            for n in [1]:
                df = randomDataFrame(n, na_probability=0.0, seed=400 + n)
                for _col0_ in ['a','c']:
                    for _col1_ in ['b','d']:
                        for _col2_ in df.columns:
                            for _color_ in [
                                _col2_,
                                (_col2_,),
                                self.p2s.CROW_MAGNITUDEp,
                                self.p2s.CROW_STRETCHEDp,
                                (_col2_, self.p2s.CMAGNITUDE_SUMp),
                                (_col2_, self.p2s.CMAGNITUDE_MINp),
                                (_col2_, self.p2s.CMAGNITUDE_MEDIANp),
                                (_col2_, self.p2s.CMAGNITUDE_MEANp),
                                (_col2_, self.p2s.CMAGNITUDE_MAXp),
                                (_col2_, self.p2s.CSTRETCHED_SUMp),
                                (_col2_, self.p2s.CSTRETCHED_MINp),
                                (_col2_, self.p2s.CSTRETCHED_MEDIANp),
                                (_col2_, self.p2s.CSTRETCHED_MEANp),
                                (_col2_, self.p2s.CSTRETCHED_MAXp),
                                (_col2_, self.p2s.CSETp),
                                (_col2_, self.p2s.CSET_MAGNITUDEp),
                                (_col2_, self.p2s.CSET_STRETCHEDp),
                            ]: self.assertColorMode(df, _col0_, _col1_, _color_)

    def test_colorSmalls(self, samples=1):
        for _sample_ in range(samples):
            for n in [2,4]:
                df = randomDataFrame(n, na_probability=0.1, seed=410 + n)
                for _col0_ in ['a','c']:
                    for _col1_ in ['b','d']:
                        for _col2_ in df.columns:
                            for _color_ in [
                                _col2_,
                                (_col2_,),
                                self.p2s.CROW_MAGNITUDEp,
                                self.p2s.CROW_STRETCHEDp,
                                (_col2_, self.p2s.CMAGNITUDE_SUMp),
                                (_col2_, self.p2s.CMAGNITUDE_MINp),
                                (_col2_, self.p2s.CMAGNITUDE_MEDIANp),
                                (_col2_, self.p2s.CMAGNITUDE_MEANp),
                                (_col2_, self.p2s.CMAGNITUDE_MAXp),
                                (_col2_, self.p2s.CSTRETCHED_SUMp),
                                (_col2_, self.p2s.CSTRETCHED_MINp),
                                (_col2_, self.p2s.CSTRETCHED_MEDIANp),
                                (_col2_, self.p2s.CSTRETCHED_MEANp),
                                (_col2_, self.p2s.CSTRETCHED_MAXp),
                                (_col2_, self.p2s.CSETp),
                                (_col2_, self.p2s.CSET_MAGNITUDEp),
                                (_col2_, self.p2s.CSET_STRETCHEDp),
                            ]: self.assertColorMode(df, _col0_, _col1_, _color_, lazy=self.everyNth(3))

if __name__ == '__main__':
    unittest.main()
