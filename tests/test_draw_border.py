import unittest
from polars2svg import Polars2SVG
from svg_test_utils import assert_valid_svg
from random_dataframe import randomDataFrame
from timep_dataframes import makeTimeDf
from histop_dataframes import makeHistoDf
from piep_dataframes import makePieDf


class TestDrawBorderNewlyAdded(unittest.TestCase):
    '''draw_border is new on xyp/timep/histop/piep as of item 4 of
    20260714_open_todos.md; it defaults True everywhere it's accepted
    (mirroring linkp/chordp/spreadlinesp/smallp, which already had it).'''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def test_xyp_draw_border_default_true(self):
        df = randomDataFrame(50, na_probability=0.0)
        xy = self.p2s.xyp(df, 'a', 'b')
        self.assertTrue(xy.draw_border)
        assert_valid_svg(self, xy.svg)

    def test_xyp_draw_border_false(self):
        df = randomDataFrame(50, na_probability=0.0)
        xy_on  = self.p2s.xyp(df, 'a', 'b', draw_border=True)
        xy_off = self.p2s.xyp(df, 'a', 'b', draw_border=False)
        self.assertNotEqual(xy_on.svg, xy_off.svg)
        assert_valid_svg(self, xy_off.svg)

    def test_timep_draw_border_default_true(self):
        df = makeTimeDf(n=50, year=(2020, 2025), month=(1, 12))
        tp = self.p2s.timep(df, 'ts')
        self.assertTrue(tp.draw_border)
        assert_valid_svg(self, tp.svg)

    def test_timep_draw_border_false(self):
        df = makeTimeDf(n=50, year=(2020, 2025), month=(1, 12))
        tp_on  = self.p2s.timep(df, 'ts', draw_border=True)
        tp_off = self.p2s.timep(df, 'ts', draw_border=False)
        self.assertNotEqual(tp_on.svg, tp_off.svg)
        assert_valid_svg(self, tp_off.svg)

    def test_histop_draw_border_default_true(self):
        df = makeHistoDf(n=100)
        hp = self.p2s.histop(df, 'cat')
        self.assertTrue(hp.draw_border)
        assert_valid_svg(self, hp.svg)

    def test_histop_draw_border_false(self):
        df = makeHistoDf(n=100)
        hp_on  = self.p2s.histop(df, 'cat', draw_border=True)
        hp_off = self.p2s.histop(df, 'cat', draw_border=False)
        self.assertNotEqual(hp_on.svg, hp_off.svg)
        assert_valid_svg(self, hp_off.svg)

    def test_piep_draw_border_default_true(self):
        df = makePieDf(n=200)
        pp = self.p2s.piep(df, 'cat')
        self.assertTrue(pp.draw_border)
        assert_valid_svg(self, pp.svg)

    def test_piep_draw_border_false(self):
        df = makePieDf(n=200)
        pp_on  = self.p2s.piep(df, 'cat', draw_border=True)
        pp_off = self.p2s.piep(df, 'cat', draw_border=False)
        self.assertNotEqual(pp_on.svg, pp_off.svg)
        assert_valid_svg(self, pp_off.svg)


if __name__ == '__main__':
    unittest.main()
