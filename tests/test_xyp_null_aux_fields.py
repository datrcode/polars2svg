"""xyp draws a row whenever it has an x and a y.

A null in any other field it used -- colour, size, opacity, a distribution field, a line
field -- used to drop the whole row: its dot, its categorical slot and its share of the
axis extents (PLANNING.md §5 C-xyp-null-aux-field-drops-row).  Decided 2026-09-29: only a
null x or y drops a row.  A null colour takes the default colour, a null size or opacity
xyp's default (radius 1, opaque), a null in a distribution field adds nothing, and a row
with a null line field keeps its dot but joins no line.

Row 'q' carries the null in every auxiliary field; every other value is present.
"""
import re
import unittest

import polars as pl

from polars2svg import Polars2SVG
from svg_test_utils import XypSweepAssertions, normalize_svg


_DF_ = pl.DataFrame({'x': ['p', 'q', 'r', 's'],
                     'y': [1.0, 2.0, 3.0, 4.0],
                     'c': ['a', None, 'b', 'a'],
                     'n': [1.0, None, 3.0, 2.0]})


class TestXypNullAuxFields(XypSweepAssertions, unittest.TestCase):
    def setUp(self):
        self.p2s = Polars2SVG()

    def _xyp(self, df=_DF_, **kw):
        kw.setdefault('wxh', (256, 128) if kw.get('legend') else (128, 96))
        return self.p2s.xyp(df, 'x', 'y', **kw)

    # [(x, y, attributes)] of every dot, left to right -- slot p, q, r, s
    def dots(self, xyp) -> list:
        _group_ = self._DOT_GROUP_.search(xyp.svg)
        self.assertIsNotNone(_group_, 'no dots drawn')
        _out_ = []
        for _el_ in re.findall(r'<(?:rect|circle) [^>]*/>', _group_.group(2)):
            _xy_ = re.search(r'(?:x|cx)="([-\d.]+)" (?:y|cy)="([-\d.]+)"', _el_)
            _out_.append((float(_xy_.group(1)), float(_xy_.group(2)), _el_))
        return sorted(_out_)

    def attr(self, el: str, name: str) -> str:
        return re.search(rf' {name}="([^"]*)"', el).group(1)

    def test_a_null_auxiliary_field_keeps_the_row_its_dot_and_its_slot(self):
        _plain_ = [(x, y) for x, y, _ in self.dots(self._xyp())]
        self.assertEqual(len(_plain_), 4)
        for _kw_ in (dict(color='c'), dict(color=('c', self.p2s.CSETp)), dict(color='n'),
                     dict(color=('n', self.p2s.CSTRETCHED_SUMp)), dict(dot_size='n'), dict(opacity='n'),
                     dict(x_distributions='n'), dict(y_distributions=('c', self.p2s.SETp)), dict(line='c')):
            with self.subTest(**{k: str(v) for k, v in _kw_.items()}):
                _xyp_ = self.assertCleanXyp(_DF_, 'x', 'y', wxh=(128, 96), **_kw_)
                self.assertEqual(len(_xyp_.df_flat), 4)
                # four dots, in four slots -- a distribution strip or circles can move the
                # pixels, so the slots are counted rather than compared with the plain render
                self.assertEqual(len({round(x) for x, _, _ in self.dots(_xyp_)}), 4)

    def test_a_null_colour_takes_the_default_colour(self):
        _default_ = self.p2s.colorTyped('data', 'default')
        for _color_ in ('c', ('c', self.p2s.CSETp), 'n', ('n', self.p2s.CMAGNITUDE_MEANp), ('n', self.p2s.CSTRETCHED_SUMp)):
            with self.subTest(color=str(_color_)):
                _fills_ = [self.attr(el, 'fill') for _, _, el in self.dots(self._xyp(color=_color_))]
                self.assertEqual(_fills_[1], _default_, 'the null row')
                self.assertNotIn(_default_, _fills_[:1] + _fills_[2:], 'a row with a value')
        # the categorical ones keep their own colours
        _fills_ = [self.attr(el, 'fill') for _, _, el in self.dots(self._xyp(color='c'))]
        self.assertEqual([_fills_[0], _fills_[2], _fills_[3]], [self.p2s.color('a'), self.p2s.color('b'), self.p2s.color('a')])

    def test_a_null_colour_has_no_legend_entry(self):
        '''The legend lists the categories -- a null is none, and used to crash its sort.'''
        _xyp_ = self._xyp(color='c', legend=True)
        self.assertEqual(_xyp_.legend_info.entries, [('a', self.p2s.color('a')), ('b', self.p2s.color('b'))])

    def test_a_null_size_or_opacity_draws_at_the_default(self):
        _r_ = [float(self.attr(el, 'r')) for _, _, el in self.dots(self._xyp(dot_size='n'))]
        self.assertEqual(_r_[1], 1.0)
        # the values place the others on dot_size_range: 1 at its bottom, 3 at its top
        self.assertEqual((_r_[0], _r_[2]), (0.5, 4.0))
        _o_ = [float(self.attr(el, 'fill-opacity')) for _, _, el in self.dots(self._xyp(opacity='n'))]
        self.assertEqual(_o_[1], 1.0)
        self.assertLess(_o_[0], _o_[3])

    def test_a_null_distribution_value_adds_nothing(self):
        '''A sum: the null draws exactly what a 0 does.'''
        _zero_ = _DF_.with_columns(pl.col('n').fill_null(0.0))
        for _axis_ in ('x_distributions', 'y_distributions'):
            with self.subTest(axis=_axis_):
                self.assertEqual(normalize_svg(self._xyp(**{_axis_: 'n'}).svg),
                                 normalize_svg(self._xyp(df=_zero_, **{_axis_: 'n'}).svg))

    def test_a_null_line_field_keeps_the_dot_but_joins_no_line(self):
        _xyp_ = self._xyp(line='c')
        _dots_ = self.dots(_xyp_)
        self.assertEqual(len(_dots_), 4)
        _paths_ = [[(float(x), float(y)) for x, y in re.findall(r'[ML] ([-\d.]+) ([-\d.]+)', d)]
                   for d in re.findall(r'<path d="([^"]+)"', _xyp_.svg)]
        _joined_ = [pt for _path_ in _paths_ for pt in _path_]
        self.assertEqual(len(_paths_), 2, 'one line for a, one for b -- none for the null')
        self.assertEqual(sorted(len(p) for p in _paths_), [1, 2])
        self.assertEqual(len(_joined_), 3, 'the null row joined a line')


if __name__ == '__main__':
    unittest.main()
