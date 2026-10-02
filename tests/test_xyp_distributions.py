import re
import math
import unittest
import polars as pl
from polars2svg import Polars2SVG
from svg_test_utils import normalize_svg

from random_dataframe import randomDataFrame

class Testxyp_distributions(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def test_exceptions(self, samples=1, df_size=100):
        df       = randomDataFrame(df_size)
        with self.assertRaises(ValueError):
            self.p2s.xyp(df, 'a', 'b', x_distributions=('c', self.p2s.DISTRIBUTION_OUTSIDEp, self.p2s.DISTRIBUTION_INSIDEp))
        with self.assertRaises(ValueError):
            self.p2s.xyp(df, 'a', 'b', y_distributions=('c', self.p2s.DISTRIBUTION_OUTSIDEp, self.p2s.DISTRIBUTION_INSIDEp))
        with self.assertRaises(ValueError):
            self.p2s.xyp(df, 'a', 'b', x_distributions=['c','d','e', 10, 11])
        with self.assertRaises(ValueError):
            self.p2s.xyp(df, 'a', 'b', x_distributions=['c','d','e', 0.1, 0.2])
        with self.assertRaises(ValueError):
            self.p2s.xyp(df, 'a', 'b', x_distributions=['c','d','e', '#ff0000', '#00ff00'])
        with self.assertRaises(ValueError):
            self.p2s.xyp(df, 'a', 'b', x_distributions=('c', '#ff0000', '#00ff00'))
        with self.assertRaises(ValueError):
            self.p2s.xyp(df, 'a', 'b', x_distributions=['c',self.p2s.DISTRIBUTION_COLOR_MIN_TO_COLOR_MAX,
                                                                    self.p2s.DISTRIBUTION_ZERO_TO_COLOR_MAX])

    #
    # Reading a distribution back (PLANNING.md V11).  A distribution is drawn as bars, or --
    # when there are several fields, when it sits inside the plot among the dots, or when a
    # bar would be under 5px wide -- as a step outline.  Either way it has one magnitude per
    # bin, and for bars those magnitudes are recounted here against polars.
    #
    _BAR_   = re.compile(r'<rect x="([-\d.]+)" y="([-\d.]+)" width="([-\d.]+)" height="([-\d.]+)" stroke-opacity')
    _PATH_  = re.compile(r'<path d="([^"]*)" stroke="(#[0-9a-f]{6})" stroke-width="0.5" fill="none" />')
    _FRAME_ = re.compile(r'<rect x="([-\d.]+)" y="([-\d.]+)" width="([\d.]+)" height="([\d.]+)" stroke="[^"]*" fill="none" stroke-width="0.25" />')

    def frame(self, svg: str) -> tuple:
        return tuple(float(v) for v in self._FRAME_.search(svg).groups())

    # The bars of each outside distribution: (x bars above the plot, left to right as (height, x);
    # y bars to its right, bottom to top as (width, y))
    def outsideBars(self, svg: str) -> tuple:
        _fx_, _fy_, _fw_, _fh_ = self.frame(svg)
        _bars_ = [tuple(float(v) for v in b) for b in self._BAR_.findall(svg)]
        return ([(h, x) for x, y, w, h in sorted(_bars_) if y + h <= _fy_ + 1.5],
                [(w, y) for x, y, w, h in sorted(_bars_, key=lambda b: -b[1]) if x >= _fx_ + _fw_ - 1.5])

    #
    # assertBinsAre() -- one axis's outside bars are `metric` over the rows, binned into as
    # many equal slices of the axis window as there are bars; the largest bin is the full
    # band and the rest in proportion.  `position` places each row on the axis (its value,
    # or a categorical axis's slot).
    #
    def assertBinsAre(self, xyp, axis: str, rows: pl.DataFrame, position: pl.Expr, metric: pl.Expr) -> None:
        _x_bars_, _y_bars_ = self.outsideBars(xyp.svg)
        _mags_ = [m for m, _ in (_x_bars_ if axis == 'x' else _y_bars_)]
        self.assertGreater(len(_mags_), 0, f'no {axis} distribution bars')
        _lo_, _hi_ = xyp.x_effective_range if axis == 'x' else xyp.y_effective_range
        _n_ = len(_mags_)
        _clean_ = xyp.x_distributions_clean if axis == 'x' else xyp.y_distributions_clean
        _pos_   = rows.select(position.alias('p'))['p'].drop_nulls()
        if self.p2s.DISTRIBUTION_AUTOBINp in _clean_['enums'] and bool((_pos_ == _pos_.floor()).all()):
            # Whole-number positions under autobin: every bar holds a whole number of
            # integers, counted from the first one in the window -- the fewest per bar that
            # make the bars drawn (so none is empty for want of a value).
            _first_ = math.ceil(_lo_)
            _slots_ = max(1, math.floor(_hi_) - _first_ + 1)
            _per_   = next(_p_ for _p_ in range(1, _slots_ + 1) if math.ceil(_slots_ / _p_) == _n_)
            _bin_   = ((position - _first_) / _per_).floor()
        else:
            # the bin width divided in first, exactly as xyp does: a row on a bin edge must land
            # in the bin xyp's own arithmetic puts it in
            _bin_   = ((position - _lo_) / ((_hi_ - _lo_) / _n_)).floor()
        _want_ = dict(rows.with_columns(_bin_.clip(0, _n_ - 1).cast(pl.Int64).alias('__bin__'))
                          .group_by('__bin__').agg(metric.alias('m')).iter_rows())
        _top_ = max(_want_.values())
        for _i_, _m_ in enumerate(_mags_):
            self.assertAlmostEqual(_m_, max(_mags_) * (_want_.get(_i_, 0) or 0) / _top_, delta=0.06,
                                   msg=f'{axis} bin {_i_} is not its {_want_.get(_i_, 0)}')

    def test_basics(self, samples=1, df_size=200):
        # Categorical axes: each row sits at its value's slot in the ascending order.  The rows
        # drawn are those with an x and a y; a null in the distribution field adds nothing to
        # the sum.  It used to drop the row, its dot and its slot as well (PLANNING.md §5
        # C-xyp-null-aux-field-drops-row).
        df       = randomDataFrame(df_size, seed=71)
        _params_ = {'df':df, 'x':'j', 'y':'k'}
        _rows_   = df.filter(pl.col('j').is_not_null() & pl.col('k').is_not_null())
        self.assertGreater(_rows_['a'].null_count(), 0, 'the frame no longer tests a null distribution value')
        _slot_   = pl.col('j').rank('dense') - 1
        self.assertBinsAre(self.p2s.xyp(**_params_, x_distributions='a'), 'x', _rows_, _slot_, pl.col('a').sum())
        # several fields draw one outline each -- the tuple and the list spell the same thing
        _tuple_ = self.p2s.xyp(**_params_, x_distributions=('a','b'))
        self.assertEqual(normalize_svg(_tuple_.svg), normalize_svg(self.p2s.xyp(**_params_, x_distributions=['a','b']).svg))
        self.assertEqual(len({c for _, c in self._PATH_.findall(_tuple_.svg)}), 2, 'expected one outline per field')
        # ... and SETp counts distinct values instead of summing them: a different picture
        _set_ = self.p2s.xyp(**_params_, x_distributions=['a','b', self.p2s.SETp])
        self.assertEqual(len({c for _, c in self._PATH_.findall(_set_.svg)}), 2)
        self.assertNotEqual(normalize_svg(_set_.svg), normalize_svg(_tuple_.svg))

    def test_outlinesFollowFieldOrder(self):
        # Several fields draw one outline each, in the order the spec names them -- each one
        # over the ones before it.  The order used to come from a set of colours and from a
        # sort tied on every bin, so one call could stack them either way from one render to
        # the next (PLANNING.md §5 C-xyp-distribution-outline-order).
        df = randomDataFrame(200, seed=71)
        _a_, _b_, _c_, _d_, _e_ = (self.p2s.color((_f_,)) for _f_ in ('a', 'b', 'c', 'd', 'e'))
        self.assertEqual(len({_a_, _b_, _c_, _d_, _e_}), 5, 'the fields no longer have distinct colours')
        # the fields of one spec share a dtype: a and b are integers, c, d and e floats
        for _axis_ in ('x', 'y'):
            for _spec_, _want_ in ((['a', 'b'], [_a_, _b_]), (['b', 'a'], [_b_, _a_]), (('e', 'c', 'd'), [_e_, _c_, _d_])):
                _xyp_  = self.p2s.xyp(df, 'j', 'k', **{f'{_axis_}_distributions': _spec_})
                self.assertEqual([c for _, c in self._PATH_.findall(_xyp_.svg)], _want_, f'{_axis_}_distributions={_spec_!r}')
                _dist_ = _xyp_.df_x_distribution if _axis_ == 'x' else _xyp_.df_y_distribution
                self.assertEqual(_dist_[f'__{_axis_}dists_color__'].unique(maintain_order=True).to_list(), _want_)

    def test_distributionsIndependentOfRowOrder(self):
        # The same rows in any order draw the same distributions, and leave the same frame.
        df     = randomDataFrame(200, seed=71)
        _kw_   = {'x_distributions': ['a', 'b'], 'y_distributions': ['b', 'a']}
        _ref_  = self.p2s.xyp(df, 'j', 'k', **_kw_)
        for _seed_ in range(8):
            _shuf_ = self.p2s.xyp(df.sample(fraction=1.0, shuffle=True, seed=_seed_), 'j', 'k', **_kw_)
            self.assertEqual(_shuf_.svg_distributions, _ref_.svg_distributions, f'shuffle seed {_seed_}')
            self.assertTrue(_shuf_.df_x_distribution.equals(_ref_.df_x_distribution), f'shuffle seed {_seed_}')
            self.assertTrue(_shuf_.df_y_distribution.equals(_ref_.df_y_distribution), f'shuffle seed {_seed_}')

    def test_rowCounts(self, sample=1, df_size=200):
        df       = randomDataFrame(df_size, seed=72)
        _params_ = {'df':df, 'x':'a', 'y':'b'}
        _rows_   = df.filter(pl.col('a').is_not_null() & pl.col('b').is_not_null())
        _both_   = self.p2s.xyp(**_params_, x_distributions=self.p2s.ROW_COUNTp, y_distributions=self.p2s.ROW_COUNTp)
        for _xyp_, _axes_ in ((self.p2s.xyp(**_params_, x_distributions=self.p2s.ROW_COUNTp), 'x'), (_both_, 'xy'),
                              (self.p2s.xyp(**_params_, y_distributions=self.p2s.ROW_COUNTp), 'y')):
            for _axis_ in _axes_:
                with self.subTest(drawn=_axes_, axis=_axis_):
                    self.assertBinsAre(_xyp_, _axis_, _rows_, pl.col('a' if _axis_ == 'x' else 'b'), pl.len())
        # SETp counts distinct values -- a missing value is not one
        self.assertBinsAre(self.p2s.xyp(**_params_, x_distributions=('c', self.p2s.SETp)), 'x', _rows_, pl.col('a'),
                           pl.col('c').drop_nulls().n_unique())

    def test_constantDistributionField(self):
        # Regression: when all values in the distribution field are identical,
        # bin_width == 0 caused division by zero -> NaN -> InvalidOperationError
        # on the strict_cast to Int64.  A constant field is still summed per bin.
        df = pl.DataFrame({'x': [1.0, 2.0, 3.0], 'y': [1.0, 2.0, 3.0], 'v': [5.0, 5.0, 5.0]})
        for _kw_, _axes_ in (({'x_distributions': 'v'}, 'x'), ({'y_distributions': 'v'}, 'y'),
                             ({'x_distributions': 'v', 'y_distributions': 'v'}, 'xy')):
            _xyp_ = self.p2s.xyp(df=df, x='x', y='y', **_kw_)
            for _axis_ in _axes_:
                with self.subTest(drawn=_axes_, axis=_axis_):
                    self.assertBinsAre(_xyp_, _axis_, df, pl.col(_axis_), pl.col('v').sum())

    #
    # Where a distribution goes, and in which form (xyp's placement defaults, read off
    # __init__): an explicit INSIDE/OUTSIDE is kept.  With dots, anything unset goes outside.
    # Without dots, a lone unset one goes inside; two unset ones split, x outside and y
    # inside on a square view; one set pushes the other to the opposite side.  Inside with
    # dots, or bars under 5px wide, draw as an outline.
    #
    def resolvedPlacement(self, dot_size, x_place, y_place, has_x: bool, has_y: bool) -> tuple:
        _default_ = 'in' if dot_size is None else 'out'
        if not (has_x and has_y):                                  # a lone distribution
            return (x_place or _default_) if has_x else None, (y_place or _default_) if has_y else None
        if x_place is not None and y_place is not None: return x_place, y_place
        if x_place is None and y_place is None:
            return ('out', 'in') if dot_size is None else ('out', 'out')
        _set_ = x_place or y_place
        _other_ = ({'in': 'out', 'out': 'in'}[_set_]) if dot_size is None else 'out'
        return (x_place, _other_) if x_place else (_other_, y_place)

    def test_insideOutside(self, samples=1, df_size=200):
        df       = randomDataFrame(df_size, seed=73)
        _params_ = {'df':df, 'x':'j', 'y':'k'}
        _spec_   = {None: None, 'plain': None, 'in': self.p2s.DISTRIBUTION_INSIDEp, 'out': self.p2s.DISTRIBUTION_OUTSIDEp}
        for _dot_size_ in [None, 1, 4, 10, 0.75]:
            for _xk_ in [None, 'plain', 'in', 'out']:
                for _yk_ in [None, 'plain', 'in', 'out']:
                    with self.subTest(dot_size=_dot_size_, x=_xk_, y=_yk_):
                        _x_dist_ = None if _xk_ is None else ('a' if _spec_[_xk_] is None else ('a', _spec_[_xk_]))
                        _y_dist_ = None if _yk_ is None else ('b' if _spec_[_yk_] is None else ('b', _spec_[_yk_]))
                        _xyp_ = self.p2s.xyp(x_distributions=_x_dist_, y_distributions=_y_dist_, dot_size=_dot_size_, **_params_)
                        _xp_, _yp_ = self.resolvedPlacement(_dot_size_,
                                                            None if _xk_ in (None, 'plain') else _xk_, None if _yk_ in (None, 'plain') else _yk_,
                                                            _xk_ is not None, _yk_ is not None)
                        _fx_, _fy_, _fw_, _fh_ = self.frame(_xyp_.svg)
                        # Inside means contained by the frame -- except a bar anchored exactly where
                        # outside bars are: a y bar beginning 1px left of the frame's right edge,
                        # an x bar ending 1px below its top.  The half bar an integer axis puts at
                        # each end of an outside distribution can be small enough to fit wholly
                        # in that 1px overlap, and containment alone counted it as inside.
                        _anchored_out_ = lambda x0, y0, x1, y1: (abs(x0 - (_fx_ + _fw_ - 1)) < 0.01 or  # noqa: E731
                                                                 abs(y1 - (_fy_ + 1)) < 0.01)
                        _inside_ = lambda x0, y0, x1, y1: (x0 >= _fx_ - 0.01 and x1 <= _fx_ + _fw_ + 0.01 and  # noqa: E731
                                                           y0 >= _fy_ - 0.01 and y1 <= _fy_ + _fh_ + 0.01 and
                                                           not _anchored_out_(x0, y0, x1, y1))
                        _drawn_ = {'in': 0, 'out': 0}
                        for x, y, w, h in (tuple(float(v) for v in b) for b in self._BAR_.findall(_xyp_.svg)):
                            _drawn_['in' if _inside_(x, y, x + w, y + h) else 'out'] += 1
                        for d, _ in self._PATH_.findall(_xyp_.svg):
                            _pts_ = [(float(a), float(b)) for a, b in re.findall(r'([-\d.]+) ([-\d.]+)', d)]
                            _drawn_['in' if _inside_(min(a for a, _ in _pts_), min(b for _, b in _pts_),
                                                     max(a for a, _ in _pts_), max(b for _, b in _pts_)) else 'out'] += 1
                        _want_ = {'in': 0, 'out': 0}
                        for _dist_, _place_, _clean_ in ((_x_dist_, _xp_, _xyp_.x_distributions_clean), (_y_dist_, _yp_, _xyp_.y_distributions_clean)):
                            if _dist_ is None: continue
                            _bins_ = _clean_['bins'][0]
                            _band_ = _xyp_.plot_size[0] if _clean_ is _xyp_.x_distributions_clean else _xyp_.plot_size[1]
                            _outline_ = (_place_ == 'in' and _dot_size_ is not None) or (_band_ // _bins_) < 5
                            _want_[_place_] += 1 if _outline_ else _bins_
                        self.assertEqual(_drawn_, _want_, f'placement {(_xp_, _yp_)}')

if __name__ == '__main__':
    unittest.main()
