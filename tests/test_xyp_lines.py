import unittest
import polars as pl
from polars2svg import Polars2SVG

class Testxyp_lines(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def test_combos(self):
        dfa = pl.DataFrame({'time': [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
                            'value':[ 2, 2, 3, 4, 4, 4, 5, 1, 1, 2], 
                            'sample':['a']*10})
        dfb = pl.DataFrame({'time': [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
                            'value':[ 8, 7, 7, 5, 6, 6, 5, 5, 4, 2], 
                            'sample':['e']*10})
        df = pl.concat([dfa, dfb])
        for _linewidth_ in {None, self.p2s.LINEWIDTH_DOTSIZE_MEAN, self.p2s.LINEWIDTH_DOTSIZE_VARIABLE, self.p2s.LINEWIDTH_DOTSIZE_SPECIFIED}:
            for _linestyle_ in {None, self.p2s.LINESTYLE_SOLID, self.p2s.LINESTYLE_DOTTED, self.p2s.LINESTYLE_SPECIFIED}:
                for _linecolor_ in {None, self.p2s.LINECOLOR_GROUPBY, self.p2s.LINECOLOR_FIELD, self.p2s.LINECOLOR_SPECIFIED}:
                    for _lineopacity_ in {None, self.p2s.LINEOPACITY_FIELD_MEAN, self.p2s.LINEOPACITY_FIELD_VARIABLE, self.p2s.LINEOPACITY_100, 
                                          self.p2s.LINEOPACITY_75, self.p2s.LINEOPACITY_50, self.p2s.LINEOPACITY_25, self.p2s.LINEOPACITY_10}:
                        # Form the tuple
                        _line_tuple_ = ['sample']
                        if   _linewidth_ == self.p2s.LINEWIDTH_DOTSIZE_SPECIFIED: _line_tuple_.append(2)
                        elif _linewidth_ is None:                                 pass
                        else:                                                     _line_tuple_.append(_linewidth_)
                        if   _linecolor_ == self.p2s.LINECOLOR_SPECIFIED:         _line_tuple_.append('#000000')
                        elif _linecolor_ is None:                                 pass
                        else:                                                     _line_tuple_.append(_linecolor_)
                        if   _linestyle_ == self.p2s.LINESTYLE_SPECIFIED:         _line_tuple_.append([5,5])
                        elif _linestyle_ is None:                                 pass
                        else:                                                     _line_tuple_.append(_linestyle_)
                        if   _lineopacity_ is not None:                           _line_tuple_.append(_lineopacity_)
                        # Create the plot
                        _xyp_ = self.p2s.xyp(df, 'time', 'value', color='value', dot_size='value', opacity='value', line=tuple(_line_tuple_), wxh=(96,96), draw_context=False)

    def test_exceptions(self):
        dfa = pl.DataFrame({'time': [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
                            'value':[ 2, 2, 3, 4, 4, 4, 5, 1, 1, 2], 
                            'sample':['a']*10})
        dfb = pl.DataFrame({'time': [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
                            'value':[ 8, 7, 7, 5, 6, 6, 5, 5, 4, 2], 
                            'sample':['e']*10})
        df = pl.concat([dfa, dfb])
        _params_ = {'df':df, 'x':'time', 'y':'value', 'wxh':(96,96), 'draw_context':False}
        with self.assertRaises(ValueError):
            _xyp_ = self.p2s.xyp(dot_size='value', opacity='value', line=('sample', self.p2s.LINECOLOR_FIELD), **_params_)
        with self.assertRaises(ValueError):
            _xyp_ = self.p2s.xyp(color='value', opacity='value', line=('sample', self.p2s.LINEWIDTH_DOTSIZE_MEAN), **_params_)
        with self.assertRaises(ValueError):
            _xyp_ = self.p2s.xyp(color='value', opacity='value', line=('sample', self.p2s.LINEWIDTH_DOTSIZE_VARIABLE), **_params_)
        with self.assertRaises(ValueError):
            _xyp_ = self.p2s.xyp(dot_size=10, color='value', opacity='value', line=('sample', self.p2s.LINEWIDTH_DOTSIZE_MEAN), **_params_)
        with self.assertRaises(ValueError):
            _xyp_ = self.p2s.xyp(dot_size=10, color='value', opacity='value', line=('sample', self.p2s.LINEWIDTH_DOTSIZE_VARIABLE), **_params_)
        with self.assertRaises(ValueError):
            _xyp_ = self.p2s.xyp(color='value', dot_size='value', line=('sample', self.p2s.LINEOPACITY_FIELD_MEAN), **_params_)
        with self.assertRaises(ValueError):
            _xyp_ = self.p2s.xyp(color='value', dot_size='value', line=('sample', self.p2s.LINEOPACITY_FIELD_VARIABLE), **_params_)

    def test_paramCleaning(self):
        _xyp_ = self.p2s.xyp('a','b')
        p2s   = self.p2s
        _clean_ = _xyp_.__cleanLineParam__([('a', 3.0, 'a1', [23,2]),('b', 'b1', 0.1, '#00ff00'), 2.0, '#ff0000', [1,2]])
        assert _clean_ == [('a', 'a1', 3.0, [23, 2], '#ff0000', {p2s.LINECOLOR_SPECIFIED, p2s.LINEOPACITY_100, p2s.LINESTYLE_SPECIFIED, p2s.LINEWIDTH_DOTSIZE_SPECIFIED}),
                           ('b', 'b1', 0.1, [ 1, 2], '#00ff00', {p2s.LINECOLOR_SPECIFIED, p2s.LINEOPACITY_100, p2s.LINESTYLE_SPECIFIED, p2s.LINEWIDTH_DOTSIZE_SPECIFIED})]

        _clean_ = _xyp_.__cleanLineParam__('a')
        assert _clean_ == [('a', 0.5, [], None, {p2s.LINECOLOR_GROUPBY, p2s.LINEOPACITY_100, p2s.LINESTYLE_SOLID, p2s.LINEWIDTH_DOTSIZE_SPECIFIED})]

        _clean_ = _xyp_.__cleanLineParam__(['a','b'])
        assert _clean_ == [('a', 0.5, [], None, {p2s.LINECOLOR_GROUPBY, p2s.LINEOPACITY_100, p2s.LINESTYLE_SOLID, p2s.LINEWIDTH_DOTSIZE_SPECIFIED}), 
                           ('b', 0.5, [], None, {p2s.LINECOLOR_GROUPBY, p2s.LINEOPACITY_100, p2s.LINESTYLE_SOLID, p2s.LINEWIDTH_DOTSIZE_SPECIFIED})]

        _clean_ = _xyp_.__cleanLineParam__([('a','b')])
        assert _clean_ == [('a', 'b', 0.5, [], None, {p2s.LINECOLOR_GROUPBY, p2s.LINEOPACITY_100, p2s.LINESTYLE_SOLID, p2s.LINEWIDTH_DOTSIZE_SPECIFIED})]

        _clean_ = _xyp_.__cleanLineParam__([('a','b'), 0.8])
        assert _clean_ == [('a', 'b', 0.8, [], None, {p2s.LINECOLOR_GROUPBY, p2s.LINEOPACITY_100, p2s.LINESTYLE_SOLID, p2s.LINEWIDTH_DOTSIZE_SPECIFIED})]

        _clean_ = _xyp_.__cleanLineParam__([('a','b'), '#ff00ff'])
        assert _clean_ == [('a', 'b', 0.5, [], '#ff00ff', {p2s.LINECOLOR_SPECIFIED, p2s.LINEOPACITY_100, p2s.LINESTYLE_SOLID, p2s.LINEWIDTH_DOTSIZE_SPECIFIED})]

        _clean_ = _xyp_.__cleanLineParam__([('a','b'), [1,2,1]])
        assert _clean_ == [('a', 'b', 0.5, [1, 2, 1], None, {p2s.LINECOLOR_GROUPBY, p2s.LINEOPACITY_100, p2s.LINESTYLE_SPECIFIED, p2s.LINEWIDTH_DOTSIZE_SPECIFIED})]

if __name__ == '__main__':
    unittest.main()
