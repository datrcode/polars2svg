import re
import unittest
import polars as pl
from polars2svg import Polars2SVG

from random_dataframe import randomDataFrame

class Testxyp_dot_size(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()
        self.df  = randomDataFrame(100, na_probability=0.01, seed=41)

    #
    # assertDots() -- what each kind of dot_size= draws (PLANNING.md V11):
    #   None     no dots at all, only the axes -- the frame distributions are drawn into
    #   int n    n-pixel squares on a raster: CSS width/height n, every square on the n-pixel
    #            grid that starts at the plot's origin
    #   float r  circles of radius r
    #   list     one radius per series, circles either way
    #
    def assertDots(self, df, x, y, dot_size) -> None:
        _xyp_ = self.p2s.xyp(df, x, y, dot_size=dot_size)
        _svg_ = _xyp_.svg
        _group_ = re.search(r'<g class="(rect|circle)-group-(\d+)"[^>]*>(.*?)</g>', _svg_, re.S)
        if dot_size is None:
            self.assertIsNone(_group_, 'dot_size=None drew dots')
            self.assertIn('<text', _svg_, 'dot_size=None should still draw the axes')
            return
        self.assertIsNotNone(_group_, 'no dots drawn')
        _kind_, _id_, _body_ = _group_.groups()
        if isinstance(dot_size, int):
            self.assertEqual(_kind_, 'rect')
            self.assertIn(f'.rect-group-{_id_} rect {{ width: {dot_size}px; height: {dot_size}px; }}', _svg_)
            _x0_, _y0_ = _xyp_.plot_origin
            for _x_, _y_ in re.findall(r'<rect x="([\d.]+)" y="([\d.]+)"', _body_):
                self.assertEqual(((float(_x_) - _x0_) % dot_size, (_y0_ - dot_size - float(_y_)) % dot_size), (0, 0),
                                 f'a {dot_size}px square at ({_x_}, {_y_}) is off the raster')
        else:
            self.assertEqual(_kind_, 'circle')
            _radii_ = {float(r) for r in re.findall(r' r="([\d.]+)"', _body_)}
            _want_  = {float(dot_size)} if isinstance(dot_size, float) else {float(s) for s in dot_size}
            self.assertEqual(_radii_, _want_)

    def test_dot_size_is_none(self):
        self.assertDots(self.df, 'a', 'b', None)
        self.assertDots(self.df, ['a','b'], ['c','d'], None)

    def test_dot_size_is_int(self):
        for _n_ in (1, 2, 10):
            with self.subTest(dot_size=_n_):
                self.assertDots(self.df, 'a', 'b', _n_)
        self.assertDots(self.df, ['a','b'], ['c','d'], 2)

    def test_dot_size_is_float(self):
        for _r_ in (0.1, 1.0, 2.0, 10.0):
            with self.subTest(dot_size=_r_):
                self.assertDots(self.df, 'a', 'b', _r_)
        self.assertDots(self.df, ['a','b'], ['c','d'], 2.0)

    def test_dot_size_is_int_list(self):
        self.assertDots(self.df, ['a','b'], ['c','d'], [1,2])
        self.assertDots(self.df, ['a','b'], ['c','d'], [5,5])

    def test_dot_size_is_float_list(self):
        self.assertDots(self.df, ['a','b'], ['c','d'], [1.0,2.0])
        self.assertDots(self.df, ['a','b'], ['c','d'], [5.0,5.0])

    def test_dot_size_randoms(self):
        df = pl.DataFrame({'i0':[1,2,3],
                           'i1':[4,5,6],
                           'f0':[1.1,2.2,3.3],
                           'f1':[4.4,5.5,6.6],})
        for _x_, _y_, _dot_size_ in (('i0', 'i1', None), ('i0', 'i1', 2), ('i0', 'i1', 3), ('i0', 'i1', [4]),
                                     ('i0', 'i1', 2.5), ('i0', 'i1', 3.1), ('i0', 'i1', [4.5]),
                                     (['i0','i1'], ['f0','f1'], 4), (['i0','i1'], ['f0','f1'], [5, 10]),
                                     (['i0','i1'], ['f0','f1'], 6.0), (['i0','i1'], ['f0','f1'], [3.0, 10.0])):
            with self.subTest(x=str(_x_), dot_size=str(_dot_size_)):
                self.assertDots(df, _x_, _y_, _dot_size_)

if __name__ == '__main__':
    unittest.main()
