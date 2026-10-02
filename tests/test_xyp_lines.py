import re
import unittest
import xml.etree.ElementTree as ET
import polars as pl
from polars2svg import Polars2SVG

class Testxyp_lines(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    #
    # assertLinesDrawn() -- what one line= combination draws (PLANNING.md V11).
    # A field-driven colour, width or opacity draws each group as gradient-stroked segments.
    # Otherwise each group is one path through its own dots, left to right, whose effective
    # stroke attributes -- its own, else its wrapper's -- follow the spec:
    #   width    2 when specified, 0.5 by default; DOTSIZE_MEAN: one per line, inside
    #            dot_size_range, wider for the group with the larger mean value
    #   style    no dashes when solid or unset, '5 5' when dotted or specified [5, 5]
    #   colour   '#000000' when specified, otherwise each group's own colour
    #   opacity  the fixed level when one is given, none by default; FIELD_MEAN: one per line,
    #            inside opacity_range, more opaque for the group with the larger mean value
    # A MEAN line is also placed on the drawn DOTS' scale: its width (opacity) is the mean,
    # over its points, of each point's value mapped through dot_size_range (opacity_range)
    # over the per-pixel values the dots show -- so one number draws at one size as a dot
    # or as a line.  It used to use a scale of its own (PLANNING.md §5 C-xyp-line-mean-scale).
    #
    def meanOnDotScale(self, df, group, rng) -> float:
        _pixels_ = df.group_by('time', 'value').agg(pl.col('value').sum().alias('v'))['v']
        _lo_, _hi_ = float(_pixels_.min()), float(_pixels_.max())
        _vals_ = df.filter(pl.col('sample') == group)['value'].to_list()
        return sum(rng[0] + (rng[1] - rng[0]) * (v - _lo_) / (_hi_ - _lo_) for v in _vals_) / len(_vals_)

    def assertLinesDrawn(self, xyp, w, s, c, o, df=None) -> None:
        p2s  = self.p2s
        root = ET.fromstring(xyp.svg)
        _parent_ = {ch: pa for pa in root.iter() for ch in pa}
        _tag_    = lambda e: e.tag.split('}')[-1]  # noqa: E731
        if c == p2s.LINECOLOR_FIELD or w == p2s.LINEWIDTH_DOTSIZE_VARIABLE or o == p2s.LINEOPACITY_FIELD_VARIABLE:
            _ids_  = {e.get('id') for e in root.iter() if _tag_(e) == 'linearGradient'}
            _segs_ = [e for e in root.iter() if _tag_(e) == 'line' and (e.get('stroke') or '').startswith('url(#lines_')]
            self.assertEqual(len(_segs_), 18, 'nine segments per group')
            self.assertEqual({e.get('stroke')[5:-1] for e in _segs_}, _ids_, 'every segment strokes with a defined gradient')
            return
        _dots_  = {(float(e.get('cx')), float(e.get('cy'))) for e in root.iter() if _tag_(e) == 'circle'}
        _paths_ = [e for e in root.iter() if _tag_(e) == 'path']
        self.assertEqual(len(_paths_), 2, 'one line per group')
        def _eff_(e, name):
            while e is not None:
                if e.get(name) is not None: return e.get(name)
                e = _parent_.get(e)
            return None
        _lines_ = []
        for _p_ in _paths_:
            _v_ = [(float(x), float(y)) for x, y in re.findall(r'[ML] ([\d.]+) ([\d.]+)', _p_.get('d'))]
            self.assertEqual(len(_v_), 10)
            self.assertTrue(all(pt in _dots_ for pt in _v_), 'the line leaves its dots')
            self.assertEqual([x for x, _ in _v_], sorted(x for x, _ in _v_), 'the line does not run left to right')
            _lines_.append((_v_[0][1], _p_))
        _a_, _e_ = [pth for _, pth in sorted(_lines_, key=lambda t: -t[0])]      # group a starts lower (value 2, not 8)
        _widths_ = [float(_eff_(x, 'stroke-width')) for x in (_a_, _e_)]
        if   w == p2s.LINEWIDTH_DOTSIZE_SPECIFIED: self.assertEqual(_widths_, [2.0, 2.0])
        elif w is None:                            self.assertEqual(_widths_, [0.5, 0.5])
        else:
            _lo_, _hi_ = xyp.dot_size_range
            self.assertTrue(_lo_ <= _widths_[0] < _widths_[1] <= _hi_, f'mean widths {_widths_}')
            for _g_, _w_ in zip(('a', 'e'), _widths_):
                self.assertAlmostEqual(_w_, self.meanOnDotScale(df, _g_, xyp.dot_size_range), delta=0.01, msg=f'group {_g_} width')
        _dash_ = {_eff_(x, 'stroke-dasharray') or 'none' for x in (_a_, _e_)}
        self.assertEqual(_dash_, {'none'} if s in (None, p2s.LINESTYLE_SOLID) else {'5 5'})
        _colors_ = [(_eff_(x, 'stroke') or '').lower() for x in (_a_, _e_)]
        if c == p2s.LINECOLOR_SPECIFIED: self.assertEqual(_colors_, ['#000000', '#000000'])
        else:                            self.assertEqual(_colors_, [v.lower() for v in (p2s.colors(['a'])['a'], p2s.colors(['e'])['e'])])
        _ops_ = [_eff_(x, 'stroke-opacity') for x in (_a_, _e_)]
        _fixed_ = {p2s.LINEOPACITY_75: 0.75, p2s.LINEOPACITY_50: 0.5, p2s.LINEOPACITY_25: 0.25, p2s.LINEOPACITY_10: 0.1}
        if   o in _fixed_:                     self.assertEqual([float(v) for v in _ops_], [_fixed_[o]] * 2)
        elif o in (None, p2s.LINEOPACITY_100): self.assertEqual({1.0 if v is None else float(v) for v in _ops_}, {1.0})
        else:
            _lo_, _hi_ = xyp.opacity_range
            self.assertTrue(_lo_ <= float(_ops_[0]) < float(_ops_[1]) <= _hi_, f'mean opacities {_ops_}')
            for _g_, _o_ in zip(('a', 'e'), _ops_):
                self.assertAlmostEqual(float(_o_), self.meanOnDotScale(df, _g_, xyp.opacity_range), delta=0.002, msg=f'group {_g_} opacity')

    def test_combos(self):
        dfa = pl.DataFrame({'time': [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                            'value':[ 2, 2, 3, 4, 4, 4, 5, 1, 1, 2],
                            'sample':['a']*10})
        dfb = pl.DataFrame({'time': [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                            'value':[ 8, 7, 7, 5, 6, 6, 5, 5, 4, 2],
                            'sample':['e']*10})
        df = pl.concat([dfa, dfb])
        for _linewidth_ in [None, self.p2s.LINEWIDTH_DOTSIZE_MEAN, self.p2s.LINEWIDTH_DOTSIZE_VARIABLE, self.p2s.LINEWIDTH_DOTSIZE_SPECIFIED]:
            for _linestyle_ in [None, self.p2s.LINESTYLE_SOLID, self.p2s.LINESTYLE_DOTTED, self.p2s.LINESTYLE_SPECIFIED]:
                for _linecolor_ in [None, self.p2s.LINECOLOR_GROUPBY, self.p2s.LINECOLOR_FIELD, self.p2s.LINECOLOR_SPECIFIED]:
                    for _lineopacity_ in [None, self.p2s.LINEOPACITY_FIELD_MEAN, self.p2s.LINEOPACITY_FIELD_VARIABLE, self.p2s.LINEOPACITY_100,
                                          self.p2s.LINEOPACITY_75, self.p2s.LINEOPACITY_50, self.p2s.LINEOPACITY_25, self.p2s.LINEOPACITY_10]:
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
                        with self.subTest(line=str(tuple(_line_tuple_))):
                            _xyp_ = self.p2s.xyp(df, 'time', 'value', color='value', dot_size='value', opacity='value', line=tuple(_line_tuple_), wxh=(96,96), draw_context=False)
                            self.assertLinesDrawn(_xyp_, _linewidth_, _linestyle_, _linecolor_, _lineopacity_, df)

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
            self.p2s.xyp(dot_size='value', opacity='value', line=('sample', self.p2s.LINECOLOR_FIELD), **_params_)
        with self.assertRaises(ValueError):
            self.p2s.xyp(color='value', opacity='value', line=('sample', self.p2s.LINEWIDTH_DOTSIZE_MEAN), **_params_)
        with self.assertRaises(ValueError):
            self.p2s.xyp(color='value', opacity='value', line=('sample', self.p2s.LINEWIDTH_DOTSIZE_VARIABLE), **_params_)
        with self.assertRaises(ValueError):
            self.p2s.xyp(dot_size=10, color='value', opacity='value', line=('sample', self.p2s.LINEWIDTH_DOTSIZE_MEAN), **_params_)
        with self.assertRaises(ValueError):
            self.p2s.xyp(dot_size=10, color='value', opacity='value', line=('sample', self.p2s.LINEWIDTH_DOTSIZE_VARIABLE), **_params_)
        with self.assertRaises(ValueError):
            self.p2s.xyp(color='value', dot_size='value', line=('sample', self.p2s.LINEOPACITY_FIELD_MEAN), **_params_)
        with self.assertRaises(ValueError):
            self.p2s.xyp(color='value', dot_size='value', line=('sample', self.p2s.LINEOPACITY_FIELD_VARIABLE), **_params_)

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
