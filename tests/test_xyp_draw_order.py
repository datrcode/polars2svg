import random
import re
import unittest
import polars as pl
from polars2svg import Polars2SVG

#
# xyp's dot draw order (PLANNING.md §5 C-xyp-draw-order-nondeterministic).  Where dots
# overlap, the order decides which one is on top, and it used to change on every render.
# - sized dots draw largest first, so a small dot is not buried under a big one
# - dots of one size, and ties, draw top-to-bottom then left-to-right
#
class TestXYpDrawOrder(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()
        _rng_ = random.Random(17)
        _n_   = 2000
        self.df = pl.DataFrame({'x':     [_rng_.random()             for _ in range(_n_)],
                                'y':     [_rng_.random()             for _ in range(_n_)],
                                'group': [_rng_.choice('abcdef')     for _ in range(_n_)],
                                'sz':    [_rng_.randint(1, 1000)     for _ in range(_n_)]})
        self.CASES = {'int':           dict(dot_size=3),
                      'float':         dict(dot_size=2.5),
                      'column':        dict(dot_size='sz'),
                      'column + cset': dict(dot_size='sz', color='group'),
                      'float + cset':  dict(dot_size=2.5, color='group'),
                      'float + sum':   dict(dot_size=2.5, color=('sz', self.p2s.CMAGNITUDE_SUMp)),
                      'lazy':          dict(dot_size='sz', color='group', use_lazy_execution=True),
                      'eager':         dict(dot_size='sz', color='group', use_lazy_execution=False),
                      # float pixel coordinates: sorted on (y, x), not the packed integer key
                      'supersample':   dict(dot_size=4, dot_size_supersample=3, color='group')}

    # the render with its random id replaced, so two renders compare as strings
    def __svg__(self, **kwargs) -> str:
        _svg_ = self.p2s.xyp(self.df, 'x', 'y', wxh=(128, 128), **kwargs).svg
        _id_  = re.search(r'id="xyp_(\d+)"', _svg_).group(1)
        return _svg_.replace(_id_, 'ID')

    def __dots__(self, svg: str) -> list[tuple[float, float, float | None]]:
        _dots_ = []
        _body_ = re.search(r'<g class="(?:rect|circle)-group-[^"]*"[^>]*>(.*?)</g>', svg, re.S).group(1)
        for _m_ in re.finditer(r'<(rect x|circle cx)="([\d.]+)" (?:y|cy)="([\d.]+)"([^>]*)/>', _body_):
            _r_ = re.search(r' r="([\d.]+)"', _m_.group(4))
            _dots_.append((float(_m_.group(2)), float(_m_.group(3)), float(_r_.group(1)) if _r_ else None))
        return _dots_

    def test_repeat_renders_are_identical(self):
        for _name_, _kwargs_ in self.CASES.items():
            with self.subTest(case=_name_):
                _first_ = self.__svg__(**_kwargs_)
                self.assertGreater(len(self.__dots__(_first_)), 500, 'too few dots to expose an unordered group_by')
                for _ in range(5):
                    self.assertEqual(self.__svg__(**_kwargs_), _first_)

    def test_sized_dots_draw_largest_first(self):
        for _name_ in ['column', 'column + cset', 'lazy', 'eager']:
            with self.subTest(case=_name_):
                _dots_ = self.__dots__(self.__svg__(**self.CASES[_name_]))
                _radii_ = [_r_ for _x_, _y_, _r_ in _dots_]
                self.assertGreater(len(set(_radii_)), 10, 'the sizes should vary')
                self.assertEqual(_radii_, sorted(_radii_, reverse=True))
                # within one size: top-to-bottom, then left-to-right
                self.assertEqual(_dots_, sorted(_dots_, key=lambda _d_: (-_d_[2], _d_[1], _d_[0])))

    def test_constant_size_dots_draw_top_to_bottom(self):
        for _name_ in ['int', 'float', 'float + cset', 'float + sum', 'supersample']:
            with self.subTest(case=_name_):
                _xy_ = [(_x_, _y_) for _x_, _y_, _ in self.__dots__(self.__svg__(**self.CASES[_name_]))]
                self.assertEqual(_xy_, sorted(_xy_, key=lambda _d_: (_d_[1], _d_[0])))

#
# filterByColorAtXY() picks the nearest dot's colour.  Two dots at the same distance are
# a tie, and the dot drawn last wins, since it is the one on top.  The sized case puts
# the small dot on the left, so "drawn last" and "rightmost" disagree there.
#
class TestFilterByColorAtXYTie(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def assertTieGoesTo(self, dot_size: float | str, winner: str) -> None:
        df = pl.DataFrame({'x': [1, 2], 'y': [1, 1], 'group': ['a', 'b'], 'sz': [1, 10]})
        for _ in range(5):
            _xyp_ = self.p2s.xyp(df, 'x', 'y', color='group', dot_size=dot_size,
                                 x_range=(0, 3), y_range=(0, 2), wxh=(200, 200))
            _px_  = _xyp_.df_pixels.sort('__xpx__')
            self.assertEqual(len(_px_), 2)
            self.assertEqual(_px_['__ypx__'][0], _px_['__ypx__'][1], 'the two dots should share a row')
            _mid_ = ((_px_['__xpx__'][0] + _px_['__xpx__'][1]) / 2, _px_['__ypx__'][0])
            _got_ = _xyp_.filterByColorAtXY(_mid_, distance_threshold=100.0)
            self.assertEqual(_got_['group'].to_list(), [winner])
            _rest_ = _xyp_.filterByColorAtXY(_mid_, remove_records=True, distance_threshold=100.0)
            self.assertEqual(_rest_['group'].to_list(), ['b' if winner == 'a' else 'a'])

    # one size: draws left to right along a row, so the right-hand dot is on top
    def test_one_size_tie_goes_to_the_dot_on_top(self):
        self.assertTieGoesTo(4.0, 'b')

    # sized: draws largest first, so the small left-hand dot is on top
    def test_sized_tie_goes_to_the_dot_on_top(self):
        self.assertTieGoesTo('sz', 'a')

if __name__ == '__main__':
    unittest.main()
