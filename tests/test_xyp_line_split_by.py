"""
Tests for xyp's line_split_by= and the smallp panels that use it.

line= groups rows into lines by its field values.  A frame holding several categories --
smallp's remainder and 'all' panels -- used to chain every category's points into one line
per line= value, which zig-zags between categories and reads as a filled area.
line_split_by= splits each line by more fields WITHOUT changing its colour, and smallp
passes its category field(s) there, so a mixed panel draws one line per category in the
colours the categories' own panels use.
"""
import re
import unittest

import polars as pl

from polars2svg import Polars2SVG
from svg_test_utils import normalize_svg


def _frame_():
    # 4 categories x 2 series x 12 time steps; each (cat, series) its own level
    _rows_ = []
    for _c_, _cat_ in enumerate(['a', 'b', 'c', 'd']):
        for _s_, _series_ in enumerate(['crude', 'total']):
            for _t_ in range(12):
                _rows_.append({'cat': _cat_, 'series': _series_, 't': _t_, 'v': 10 * _c_ + 5 * _s_ + (_t_ % 3)})
    return pl.DataFrame(_rows_)


def _paths_(xyp):
    return re.findall(r'<path d="[^"]*"( stroke="([^"]+)")?', xyp.svg_lines)


def _stroke_colours_(xyp):
    return {_m_[1] for _m_ in _paths_(xyp) if _m_[1]}


class TestXYpLineSplitBy(unittest.TestCase):

    def setUp(self):
        self.p2s = Polars2SVG()
        self.df  = _frame_()

    def _xyp_(self, df=None, **kw):
        _x_ = self.p2s.xyp(self.df if df is None else df, 't', 'v', line=('series', 2.0), wxh=(200, 120), **kw)
        _x_._repr_svg_()
        return _x_

    def test_unsplit_chains_every_category(self):
        self.assertEqual(len(_paths_(self._xyp_())), 2)

    def test_split_draws_one_line_per_category_and_series(self):
        self.assertEqual(len(_paths_(self._xyp_(line_split_by='cat'))), 8)

    def test_split_keeps_each_series_colour(self):
        _plain_ = _stroke_colours_(self._xyp_())
        self.assertEqual(len(_plain_), 2)
        self.assertEqual(_stroke_colours_(self._xyp_(line_split_by='cat')), _plain_)

    def test_split_list_form(self):
        _df_ = self.df.with_columns(half=(pl.col('t') < 6))
        self.assertEqual(len(_paths_(self._xyp_(df=_df_, line_split_by=['cat', 'half']))), 16)

    # The dot-level path (a DOTSIZE_MEAN width) groups and colours through other code.
    def test_split_on_the_dot_level_path(self):
        _x_ = self.p2s.xyp(self.df, 't', 'v', dot_size='v', wxh=(200, 120),
                           line=('series', self.p2s.LINEWIDTH_DOTSIZE_MEAN), line_split_by='cat')
        _x_._repr_svg_()
        _plain_ = self.p2s.xyp(self.df, 't', 'v', dot_size='v', wxh=(200, 120),
                               line=('series', self.p2s.LINEWIDTH_DOTSIZE_MEAN))
        _plain_._repr_svg_()
        _strokes_ = lambda x: set(re.findall(r'stroke="(#[0-9a-fA-F]{6})"', x.svg_lines))  # noqa: E731
        self.assertEqual(len(re.findall(r'<path ', _x_.svg_lines)), 8)
        self.assertEqual(len(re.findall(r'<path ', _plain_.svg_lines)), 2)
        self.assertEqual(_strokes_(_x_), _strokes_(_plain_))

    # A null split field is a category of its own; a null line field still draws no line.
    def test_nulls(self):
        _df_ = self.df.with_columns(cat=pl.when(pl.col('cat') == 'a').then(None).otherwise(pl.col('cat')))
        self.assertEqual(len(_paths_(self._xyp_(df=_df_, line_split_by='cat'))), 8)
        _df_ = self.df.with_columns(series=pl.when(pl.col('cat') == 'a').then(None).otherwise(pl.col('series')))
        self.assertEqual(len(_paths_(self._xyp_(df=_df_, line_split_by='cat'))), 6)

    def test_bad_specs(self):
        with self.assertRaises(TypeError):
            self._xyp_(line_split_by=3)
        with self.assertRaisesRegex(ValueError, 'nope'):
            self._xyp_(line_split_by='nope')


class TestSmallpMixedPanelLines(unittest.TestCase):

    def setUp(self):
        self.p2s = Polars2SVG()
        self.df  = _frame_()
        self.tmpl = self.p2s.xyp(self.df.filter(pl.col('cat') == 'a'), 't', 'v',
                                 line=('series', 2.0), wxh=(100, 60), draw_context=False)

    # Room for two panels and the remainder: 'c' and 'd' land in the remainder.
    def _smallp_(self, **kw):
        _sm_ = self.p2s.smallp(self.df, self.tmpl, 'cat', wxh=(330, 90), order='v', **kw)
        _sm_._repr_svg_()
        return _sm_

    def test_remainder_draws_one_line_per_category(self):
        _sm_  = self._smallp_()
        _rem_ = _sm_._render_lu_['__remainder__']
        _n_   = _sm_.category_to_df['__remainder__'].select('cat', 'series').n_unique()
        self.assertGreater(_sm_.category_to_df['__remainder__']['cat'].n_unique(), 1)
        self.assertEqual(len(_paths_(_rem_)), _n_)

    def test_remainder_colours_match_the_category_panels(self):
        _sm_ = self._smallp_()
        _panel_ = next(_v_ for _k_, _v_ in _sm_._render_lu_.items() if _k_ != '__remainder__')
        self.assertEqual(_stroke_colours_(_sm_._render_lu_['__remainder__']), _stroke_colours_(_panel_))

    def test_all_panel_draws_one_line_per_category(self):
        _sm_ = self._smallp_(include_all=True)
        self.assertEqual(len(_paths_(_sm_._render_lu_['__all__'])), 8)

    # In a single-category panel the split changes nothing.
    def test_single_category_panel_is_unchanged(self):
        _sm_  = self._smallp_()
        _key_ = next(_k_ for _k_ in _sm_._render_lu_ if _k_ != '__remainder__')
        _direct_ = self.p2s.xyp(df=_sm_.category_to_df[_key_], template=self.tmpl)
        self.assertEqual(normalize_svg(_sm_._render_lu_[_key_]._repr_svg_()), normalize_svg(_direct_._repr_svg_()))

    # A list of frames names no column, so there is nothing to split by.
    def test_list_categories_pass_no_fields(self):
        _sm_ = self.p2s.smallp(self.df, self.tmpl, [self.df.filter(pl.col('cat') == 'a'),
                                                    self.df.filter(pl.col('cat') != 'a')], wxh=(330, 90))
        self.assertEqual(_sm_.__categoryFields__(), [])


if __name__ == '__main__':
    unittest.main()
