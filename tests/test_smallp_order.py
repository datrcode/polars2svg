import unittest
import polars as pl
from polars2svg import Polars2SVG
from svg_test_utils import assert_ordered_keys

class Testsmallp_order(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def test_numberOfRowsOrder(self):
        df = pl.DataFrame({
            'cat':['a','b','c','a','b','a','b','a','a','a','d','z','z','z','z','z','z','z','d'],
            'val':[1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        })
        _xyp_ = self.p2s.xyp(df, 'cat', 'val')
        _smp_ = self.p2s.smallp(df, 'cat', _xyp_)
        _answers_in_order = [
        (('z',),7),
        (('a',),6),
        (('b',),3),
        (('d',),2),
        (('c',),1),
        ]
        i = 0
        for k, v in _smp_.category_by_dict.items():
            assert k      == _answers_in_order[i][0]
            assert len(v) == _answers_in_order[i][1]
            i += 1

    def test_numberOfRowsOrder_ascending(self):
        df = pl.DataFrame({
            'cat':['a','b','c','a','b','a','b','a','a','a','d','z','z','z','z','z','z','z','d'],
            'val':[1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        })
        _xyp_ = self.p2s.xyp(df, 'cat', 'val')
        _smp_ = self.p2s.smallp(df, 'cat', _xyp_, descending=False)
        _answers_in_order = [
        (('c',),1),
        (('d',),2),
        (('b',),3),
        (('a',),6),
        (('z',),7),
        ]
        i = 0
        for k, v in _smp_.category_by_dict.items():
            assert k      == _answers_in_order[i][0]
            assert len(v) == _answers_in_order[i][1]
            i += 1

    def test_fieldSumOrder(self):
        # order='val' sorts by sum of val per category (default is summation)
        # sum per cat: a=3, b=20, c=5 -> descending: b(20), c(5), a(3)
        df = pl.DataFrame({
            'cat': ['a','a','a', 'b','b', 'c'],
            'val': [  1,  1,  1,  10, 10,   5]
        })
        _xyp_ = self.p2s.xyp(df, 'cat', 'val')
        _smp_ = self.p2s.smallp(df, 'cat', _xyp_, order='val')
        _expected_keys_ = [('b',), ('c',), ('a',)]
        assert_ordered_keys(self, _smp_.category_by_dict.keys(), _expected_keys_)

    def test_fieldTupleSumOrder(self):
        # order=('val', SUMp) is equivalent to order='val'
        # sum per cat: a=3, b=20, c=5 -> descending: b(20), c(5), a(3)
        df = pl.DataFrame({
            'cat': ['a','a','a', 'b','b', 'c'],
            'val': [  1,  1,  1,  10, 10,   5]
        })
        _xyp_ = self.p2s.xyp(df, 'cat', 'val')
        _smp_ = self.p2s.smallp(df, 'cat', _xyp_, order=('val', self.p2s.SUMp))
        _expected_keys_ = [('b',), ('c',), ('a',)]
        assert_ordered_keys(self, _smp_.category_by_dict.keys(), _expected_keys_)

    def test_fieldSetOrder(self):
        # order=('color', SETp) sorts by n_unique(color) per category
        # n_unique per cat: a=3 (red,green,blue), b=1 (red), c=2 (green,blue)
        # descending: a(3), c(2), b(1)
        df = pl.DataFrame({
            'cat':   ['a',   'a',     'a',    'a',   'b',   'b',   'b',   'c',     'c'],
            'color': ['red', 'red', 'green', 'blue', 'red', 'red', 'red', 'green', 'blue'],
            'val':   [  1,     2,      3,      4,      5,     6,     7,     8,       9]
        })
        _xyp_ = self.p2s.xyp(df, 'cat', 'val')
        _smp_ = self.p2s.smallp(df, 'cat', _xyp_, order=('color', self.p2s.SETp))
        _expected_keys_ = [('a',), ('c',), ('b',)]
        assert_ordered_keys(self, _smp_.category_by_dict.keys(), _expected_keys_)

    def test_fieldMinOrder(self):
        # order=('val', MINp) sorts by min(val) per category
        # min per cat: a=5, b=1, c=10 -> descending: c(10), a(5), b(1)
        df = pl.DataFrame({
            'cat': ['a','a','a', 'b','b','b', 'c', 'c', 'c'],
            'val': [  5,  6,  7,   1,  2,  3,  10,  11,  12]
        })
        _xyp_ = self.p2s.xyp(df, 'cat', 'val')
        _smp_ = self.p2s.smallp(df, 'cat', _xyp_, order=('val', self.p2s.MINp))
        _expected_keys_ = [('c',), ('a',), ('b',)]
        assert_ordered_keys(self, _smp_.category_by_dict.keys(), _expected_keys_)

    def test_fieldMedianOrder(self):
        # order=('val', MEDIANp) sorts by median(val) per category
        # median per cat: a=5, b=3, c=20 -> descending: c(20), a(5), b(3)
        df = pl.DataFrame({
            'cat': ['a','a','a', 'b','b','b', 'c', 'c', 'c'],
            'val': [  1,  5,  9,   2,  3,  4,  10,  20,  30]
        })
        _xyp_ = self.p2s.xyp(df, 'cat', 'val')
        _smp_ = self.p2s.smallp(df, 'cat', _xyp_, order=('val', self.p2s.MEDIANp))
        _expected_keys_ = [('c',), ('a',), ('b',)]
        assert_ordered_keys(self, _smp_.category_by_dict.keys(), _expected_keys_)

    def test_fieldMeanOrder(self):
        # order=('val', MEANp) sorts by mean(val) per category
        # mean per cat: a=(2+8)/2=5, b=(1+3)/2=2, c=(10+20)/2=15 -> descending: c(15), a(5), b(2)
        df = pl.DataFrame({
            'cat': ['a','a', 'b','b', 'c', 'c'],
            'val': [  2,  8,   1,  3,  10,  20]
        })
        _xyp_ = self.p2s.xyp(df, 'cat', 'val')
        _smp_ = self.p2s.smallp(df, 'cat', _xyp_, order=('val', self.p2s.MEANp))
        _expected_keys_ = [('c',), ('a',), ('b',)]
        assert_ordered_keys(self, _smp_.category_by_dict.keys(), _expected_keys_)

    def test_fieldMaxOrder(self):
        # order=('val', MAXp) sorts by max(val) per category
        # max per cat: a=7, b=15, c=12 -> descending: b(15), c(12), a(7)
        df = pl.DataFrame({
            'cat': ['a','a','a', 'b', 'b', 'b', 'c', 'c', 'c'],
            'val': [  5,  6,  7,   1,   2,  15,  10,  11,  12]
        })
        _xyp_ = self.p2s.xyp(df, 'cat', 'val')
        _smp_ = self.p2s.smallp(df, 'cat', _xyp_, order=('val', self.p2s.MAXp))
        _expected_keys_ = [('b',), ('c',), ('a',)]
        assert_ordered_keys(self, _smp_.category_by_dict.keys(), _expected_keys_)

    def test_fieldStdOrder(self):
        # order=('val', STDp) sorts by std(val) per category
        # std per cat: a=std(1,2,3,4)≈1.29, b=std(10,10,10,10)=0, c=std(1,5,9,13)≈5.16
        # descending: c(5.16), a(1.29), b(0)
        df = pl.DataFrame({
            'cat': ['a','a','a','a',  'b', 'b', 'b', 'b',  'c','c','c', 'c'],
            'val': [  1,  2,  3,  4,  10,  10,  10,  10,    1,  5,  9,  13]
        })
        _xyp_ = self.p2s.xyp(df, 'cat', 'val')
        _smp_ = self.p2s.smallp(df, 'cat', _xyp_, order=('val', self.p2s.STDp))
        _expected_keys_ = [('c',), ('a',), ('b',)]
        assert_ordered_keys(self, _smp_.category_by_dict.keys(), _expected_keys_)

    def test_multiFieldSetOrder(self):
        # order=('color','shape') sorts by count of unique (color,shape) tuples per category
        # unique (color,shape) per cat:
        #   a: (red,sq),(green,cir),(blue,cir) = 3
        #   b: (red,sq),(green,tri)            = 2
        #   c: (blue,sq)                       = 1
        # descending: a(3), b(2), c(1)
        df = pl.DataFrame({
            'cat':   ['a',   'a',      'a',     'a',    'b',   'b',      'b',    'c',    'c'],
            'color': ['red', 'red', 'green',  'blue',  'red', 'green', 'green', 'blue', 'blue'],
            'shape': [ 'sq',  'sq',   'cir',   'cir',   'sq',   'tri',   'tri',   'sq',  'sq'],
            'val':   [  1,     2,       3,       4,       5,      6,       7,       8,     9]
        })
        _xyp_ = self.p2s.xyp(df, 'cat', 'val')
        _smp_ = self.p2s.smallp(df, 'cat', _xyp_, order=('color', 'shape'))
        _expected_keys_ = [('a',), ('b',), ('c',)]
        assert_ordered_keys(self, _smp_.category_by_dict.keys(), _expected_keys_)

if __name__ == '__main__':
    unittest.main()
