"""A drawn component, called with a new frame, is drawn again on that frame.

Every clone-template component takes its template three ways -- positionally, as
template=, or through template.render_with(df) -- and its frame positionally or as df=.
Each spelling must give the same result.

Bug fixed: xyp accepted a positional frame only while self.df was None, a guard meant
to reject a second frame in one call.  The template clone runs first and copies the
template's frame into self.df, so xyp(df2, drawn) and xyp(df2, template=drawn) raised
"Unknown argument @ index 0".  It went unnoticed because tests/test_xyp_templates.py
builds its templates without a frame, which leaves self.df None.  timep once had the
same bug (tests/test_timep_smallp.py); every other component counts the frames passed
in the call instead, and xyp now does too.
"""
import unittest
import datetime
import polars as pl

from polars2svg import Polars2SVG
from svg_test_utils import normalize_svg


def _cat_df(cats):
    # distinct per-category counts: equal-count bins tie-break nondeterministically
    return pl.DataFrame({'cat': cats, 'val': list(range(len(cats)))})


def _time_df(days):
    base = datetime.date(2024, 1, 1)
    return pl.DataFrame({'ts': [base + datetime.timedelta(days=d) for d in days], 'val': list(range(len(days)))})


def _edge_df(fm, to):
    return pl.DataFrame({'fm': fm, 'to': to})


def _spread_df(fm, to, days):
    return pl.DataFrame({'fm': fm, 'to': to, 'time': [datetime.datetime(2024, 1, d) for d in days]})


_POS_ = {'a': [0, 0], 'b': [1, 0], 'c': [0.5, 0.866]}

# name -> (first frame, second frame, build(p2s, df) -> component drawn on df)
_CASES_ = {
    'xyp':          (pl.DataFrame({'x': [1.0, 2.0, 3.0], 'y': [1.0, 2.0, 3.0]}),
                     pl.DataFrame({'x': [5.0, 1.0, 9.0, 4.0], 'y': [2.0, 8.0, 3.0, 7.0]}),
                     lambda p2s, df: p2s.xyp(df, 'x', 'y', dot_size=5, wxh=(220, 160))),
    'histop':       (_cat_df(['a', 'b', 'a', 'c', 'b', 'a']),
                     _cat_df(['b', 'c', 'b', 'a', 'c', 'b', 'd']),
                     lambda p2s, df: p2s.histop(df, bin_by='cat', wxh=(220, 160))),
    'piep':         (_cat_df(['a', 'b', 'a', 'c', 'b', 'a']),
                     _cat_df(['b', 'c', 'b', 'a', 'c', 'b', 'd']),
                     lambda p2s, df: p2s.piep(df, 'cat', wxh=(220, 160))),
    'timep':        (_time_df([0, 30, 60, 90, 120, 150]),
                     _time_df([0, 0, 0, 45, 200]),
                     lambda p2s, df: p2s.timep(df, 'ts', wxh=(220, 160))),
    'chordp':       (_edge_df(['a', 'b', 'c'], ['b', 'c', 'a']),
                     _edge_df(['a', 'a', 'a', 'b'], ['b', 'b', 'c', 'c']),
                     lambda p2s, df: p2s.chordp(df=df, relationships=[('fm', 'to')], wxh=(220, 220))),
    'linkp':        (_edge_df(['a', 'b', 'c'], ['b', 'c', 'a']),
                     _edge_df(['a', 'a'], ['b', 'c']),
                     lambda p2s, df: p2s.linkp(df, relationships=[('fm', 'to')], pos=dict(_POS_), wxh=(220, 220))),
    'spreadlinesp': (_spread_df(['a', 'a', 'b'], ['b', 'c', 'c'], [1, 2, 3]),
                     _spread_df(['a', 'a', 'a', 'c'], ['b', 'c', 'b', 'b'], [1, 1, 3, 4]),
                     lambda p2s, df: p2s.spreadlinesp(df, [('fm', 'to')], ego='a', time='time')),
}


class TestTemplateCallForms(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.p2s = Polars2SVG()

    def test_every_call_form_draws_the_new_frame(self):
        '''component(df2, drawn), component(drawn, df2), component(df2, template=drawn)
        and component(df=df2, template=drawn) all draw what drawn.render_with(df2) draws,
        and that is not what drawn itself drew.'''
        for name, (df1, df2, build) in _CASES_.items():
            with self.subTest(component=name):
                drawn    = build(self.p2s, df1)
                factory  = getattr(self.p2s, name)
                _expect_ = normalize_svg(drawn.render_with(df2).svg)
                self.assertNotEqual(_expect_, normalize_svg(drawn.svg), 'the second frame must draw differently')
                for label, call in (('(df2, drawn)',             lambda: factory(df2, drawn)),
                                    ('(drawn, df2)',             lambda: factory(drawn, df2)),
                                    ('(df2, template=drawn)',    lambda: factory(df2, template=drawn)),
                                    ('(df=df2, template=drawn)', lambda: factory(df=df2, template=drawn))):
                    with self.subTest(form=label):
                        _clone_ = call()
                        self.assertIs(_clone_.df_orig, df2)
                        self.assertEqual(normalize_svg(_clone_.svg), _expect_)

    def test_template_without_a_new_frame_redraws_its_own(self):
        '''No frame in the call: the clone keeps the template's frame and draws the same.'''
        for name, (df1, _df2_, build) in _CASES_.items():
            with self.subTest(component=name):
                drawn = build(self.p2s, df1)
                self.assertEqual(normalize_svg(getattr(self.p2s, name)(drawn).svg), normalize_svg(drawn.svg))

    def test_two_positional_frames_still_rejected(self):
        '''The guard the xyp check existed for: one call, one frame -- template or not.'''
        for name, (df1, df2, build) in _CASES_.items():
            with self.subTest(component=name):
                factory = getattr(self.p2s, name)
                drawn   = build(self.p2s, df1)
                with self.assertRaises((TypeError, ValueError)):
                    factory(df1, df2, drawn)
        with self.assertRaisesRegex(TypeError, 'second DataFrame @ index 1'):
            self.p2s.xyp(_CASES_['xyp'][0], _CASES_['xyp'][1], 'x', 'y')


if __name__ == '__main__':
    unittest.main()
