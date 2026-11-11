"""smallp.render_with_df(df): the same small multiples, drawn on another frame.

smallp does not take a smallp as a template -- its sm_template is the panel component --
so render_with_df() is its way to redraw a drawn view on new rows (the interactive
controller's smallpi uses it).  It rebuilds the Smallp by copying each setting across
by hand, and nothing checked that list against _VALID_KWARGS: a new smallp setting
would be dropped on every redraw without an error.  This file is that check.
"""
import unittest
import random
import polars as pl

from polars2svg import Polars2SVG
from polars2svg.smallp import Smallp
from svg_test_utils import normalize_svg


def _frame(n, seed):
    _r_ = random.Random(seed)
    return pl.DataFrame({'g':   [_r_.choice('abc') for _ in range(n)],
                         'h':   [_r_.choice('uv')  for _ in range(n)],
                         'x':   [_r_.random()      for _ in range(n)],
                         'y':   [_r_.random()      for _ in range(n)],
                         'val': [_r_.randint(1, 9) for _ in range(n)]})


# Every smallp setting at a value other than its default.  category_by and cycle_by
# exclude each other, so they need two builds.
_SPLIT_ = dict(category_by=('g', 'h'), grid_mode=True, order='val', descending=False,
               wxh=(640, None), insets=(3, 5), include_all=True, collate_remainder=False,
               use_lazy_execution=True, draw_labels=False, draw_border=False, txt_h=10)
_CYCLE_ = dict(cycle_by={'x': ['x', 'y']})
# not settings a redraw carries: the frame, the panel component (checked by identity
# below), and sketch_only, which a redraw always turns off
_NOT_CARRIED_ = {'df', 'sm_template', 'sketch_only'}


class TestSmallpRenderWithDf(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.p2s  = Polars2SVG()
        cls.df1  = _frame(60, 1)
        cls.df2  = _frame(40, 2)
        cls.tmpl = cls.p2s.xyp(x='x', y='y', wxh=(80, 60))

    def test_the_settings_here_cover_every_smallp_parameter(self):
        '''A new smallp parameter fails here until it is added to _SPLIT_ or _CYCLE_
        at a non-default value -- which is what makes the next test catch a drop.'''
        self.assertEqual(set(_SPLIT_) | set(_CYCLE_) | _NOT_CARRIED_, set(Smallp._VALID_KWARGS))
        _plain_ = self.p2s.smallp(self.df1, self.tmpl, 'g')
        for _k_, _v_ in {**_SPLIT_, **_CYCLE_}.items():
            with self.subTest(setting=_k_):
                self.assertNotEqual(_v_, getattr(_plain_, _k_), 'must differ from the default, or a drop goes unseen')

    def test_redraw_keeps_every_setting_and_draws_the_new_frame(self):
        for _mode_, _kw_ in (('split', _SPLIT_), ('cycle', _CYCLE_)):
            with self.subTest(mode=_mode_):
                _drawn_  = self.p2s.smallp(self.df1, self.tmpl, **_kw_)
                _redraw_ = _drawn_.render_with_df(self.df2)
                self.assertIs(_redraw_.df_orig, self.df2)
                self.assertIs(_redraw_.sm_template, self.tmpl)
                for _k_ in Smallp._VALID_KWARGS - _NOT_CARRIED_:
                    self.assertEqual(getattr(_redraw_, _k_), getattr(_drawn_, _k_), f'{_k_} not carried')
                _direct_ = normalize_svg(self.p2s.smallp(self.df2, self.tmpl, **_kw_).svg)
                self.assertEqual(normalize_svg(_redraw_.svg), _direct_)
                self.assertNotEqual(normalize_svg(_drawn_.svg), _direct_, 'the second frame must draw differently')

    def test_redraw_of_a_sketch_is_a_full_render(self):
        _sketch_ = self.p2s.smallp(self.df1, self.tmpl, 'g', sketch_only=True)
        _redraw_ = _sketch_.render_with_df(self.df2)
        self.assertFalse(_redraw_.sketch_only)
        self.assertEqual(normalize_svg(_redraw_.svg), normalize_svg(self.p2s.smallp(self.df2, self.tmpl, 'g').svg))


if __name__ == '__main__':
    unittest.main()
