import unittest
import polars as pl
from polars2svg import Polars2SVG
from svg_test_utils import normalize_svg

class Testxyp_order(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def test_list(self):
        for _lazy_ in [True, False]:
            df = pl.DataFrame({'qty':[1,2,3,10,4], 'pet':['cat','dog','parakeet','goldfish','ferret']})
            _params_ = {'df':df, 'x':'pet', 'y':'qty', 'color':'pet', 'dot_size':10}
            _params_['use_lazy_execution'] = _lazy_
            self.p2s.xyp(**_params_)                                                                    # default
            self.p2s.xyp(**_params_, x_order=['goldfish', 'ferret', 'parakeet', 'dog', 'cat'])          # complete order (decreasing qtys)
            self.p2s.xyp(**_params_, x_order=['goldfish', 'ferret', 'parakeet', 'dog', 'cat', 'snake']) # complete order + extra (decreasing qtys)
            self.p2s.xyp(**_params_, x_order=['goldfish', 'ferret'])                                    # incomplete order
            self.p2s.xyp(**_params_, x_order=['goldfish', 'ferret', 'snake'])                           # incomplete order + extra

    def test_listTuple(self):
        for _lazy_ in [True, False]:
            df = pl.DataFrame({'qty':  [2,        8,         5,          15],
                               'type': ['cat',    'cat',     'dog',      'goldfish'],
                               'color':['gray',   'orange',  'spotted',  'orange']})
            _params_ = {'df':df, 'x':('type','color'), 'y':'qty', 'color':'color', 'dot_size':10}
            _params_['use_lazy_execution'] = _lazy_
            self.p2s.xyp(**_params_)
            self.p2s.xyp(**_params_, x_order=[('goldfish','orange'), ('cat','orange'), ('dog','spotted'), ('cat','gray')])                     # complete
            self.p2s.xyp(**_params_, x_order=[('goldfish','orange'), ('cat','orange'), ('dog','spotted'), ('cat','gray'), ('snake','albino')]) # complete + extra
            self.p2s.xyp(**_params_, x_order=[('goldfish','orange'), ('cat','gray')])                                                          # incomplete
            self.p2s.xyp(**_params_, x_order=[('goldfish','orange'), ('cat','gray'), ('snake', 'albino')])                                     # incomplete + extra

    def test_dict(self):
        for _lazy_ in [True, False]:
            df = pl.DataFrame({'qty':[1,2,3,10,4], 'pet':['cat','dog','parakeet','goldfish','ferret']})
            _params_ = {'df':df, 'x':'pet', 'y':'qty', 'color':'pet', 'dot_size':10}
            _params_['use_lazy_execution'] = _lazy_
            self.p2s.xyp(**_params_)                                                                                      # default
            self.p2s.xyp(**_params_, x_order={'goldfish':10, 'ferret':15, 'parakeet':20, 'dog':25, 'cat':30})             # complete order (decreasing qtys)
            self.p2s.xyp(**_params_, x_order={'goldfish':10, 'ferret':15, 'parakeet':20, 'dog':25, 'cat':30, 'snake':35}) # complete order + extra (decreasing qtys)
            self.p2s.xyp(**_params_, x_order={'goldfish':10, 'ferret':15})                                                # incomplete order
            self.p2s.xyp(**_params_, x_order={'goldfish':10, 'ferret':15, 'snake':35})                                    # incomplete order + extra

    def test_dictTuple(self):
        for _lazy_ in [True, False]:
            df = pl.DataFrame({'qty':  [2,        8,         5,          15],
                            'type': ['cat',    'cat',     'dog',      'goldfish'],
                            'color':['gray',   'orange',  'spotted',  'orange']})
            _params_ = {'df':df, 'x':('type','color'), 'y':'qty', 'color':'color', 'dot_size':10}
            _params_['use_lazy_execution'] = _lazy_
            self.p2s.xyp(**_params_)
            self.p2s.xyp(**_params_, x_order={('goldfish','orange'):5, ('cat','orange'):6, ('dog','spotted'):7, ('cat','gray'):10})                        # complete
            self.p2s.xyp(**_params_, x_order={('goldfish','orange'):5, ('cat','orange'):6, ('dog','spotted'):7, ('cat','gray'):10, ('snake','albino'):20}) # complete + extra
            self.p2s.xyp(**_params_, x_order={('goldfish','orange'):5, ('cat','gray'):10})                                                                 # incomplete
            self.p2s.xyp(**_params_, x_order={('goldfish','orange'):5, ('cat','gray'):10, ('snake', 'albino'):20})                                         # incomplete + extra

def _axis_order_(inst, axis='x'):
    """The axis values in resolved-index order (a struct value comes back as a tuple)."""
    _v_, _i_ = f'__{axis}__', f'__{axis}i__'
    _pairs_ = inst.df_flat.select([_v_, _i_]).unique().sort(_i_)[_v_].to_list()
    return [tuple(_p_.values()) if isinstance(_p_, dict) else _p_ for _p_ in _pairs_]


class Testxyp_computed_order(unittest.TestCase):
    """x_order= / y_order= 'reverse' and 'count' -- the orders the xypi settings panel
    offers besides the default sort and 'spectral'."""

    def setUp(self):
        self.p2s = Polars2SVG()
        # dog x3, cat x2, ferret x2, ant x1: count order differs from both sorts
        self.df  = pl.DataFrame({'pet': ['dog', 'cat', 'dog', 'ferret', 'ant', 'cat', 'dog', 'ferret'],
                                 'qty': [1, 2, 3, 4, 5, 6, 7, 8]})

    def _xyp_(self, **kw):
        return self.p2s.xyp(df=self.df, x='pet', y='qty', dot_size=5, **kw)

    def test_the_default_is_the_ascending_sort(self):
        self.assertEqual(_axis_order_(self._xyp_()), ['ant', 'cat', 'dog', 'ferret'])

    def test_reverse_is_the_default_sort_descending(self):
        self.assertEqual(_axis_order_(self._xyp_(x_order='reverse')), ['ferret', 'dog', 'cat', 'ant'])

    def test_count_puts_the_most_rows_first_and_breaks_ties_by_the_sort(self):
        self.assertEqual(_axis_order_(self._xyp_(x_order='count')), ['dog', 'cat', 'ferret', 'ant'])

    def test_lazy_and_eager_agree(self):
        for _order_ in ('reverse', 'count'):
            with self.subTest(order=_order_):
                self.assertEqual(_axis_order_(self._xyp_(x_order=_order_, use_lazy_execution=True)),
                                 _axis_order_(self._xyp_(x_order=_order_, use_lazy_execution=False)))

    def test_the_y_axis_takes_them_too(self):
        _xyp_ = self.p2s.xyp(df=self.df, x='qty', y='pet', dot_size=5, y_order='count')
        self.assertEqual(_axis_order_(_xyp_, 'y'), ['dog', 'cat', 'ferret', 'ant'])

    def test_a_struct_axis_takes_them_too(self):
        _df_ = pl.DataFrame({'type':  ['cat', 'cat', 'dog', 'cat'],
                             'color': ['gray', 'orange', 'spotted', 'gray'],
                             'qty':   [1, 2, 3, 4]})
        _xyp_ = self.p2s.xyp(df=_df_, x=('type', 'color'), y='qty', dot_size=5, x_order='count')
        self.assertEqual(_axis_order_(_xyp_)[0], ('cat', 'gray'))
        _xyp_ = self.p2s.xyp(df=_df_, x=('type', 'color'), y='qty', dot_size=5, x_order='reverse')
        self.assertEqual(_axis_order_(_xyp_), [('dog', 'spotted'), ('cat', 'orange'), ('cat', 'gray')])

    def test_a_numeric_axis_is_rejected(self):
        for _order_ in ('reverse', 'count'):
            with self.subTest(order=_order_), self.assertRaises(ValueError):
                self.p2s.xyp(df=self.df, x='qty', y='pet', dot_size=5, x_order=_order_)

    def test_an_unknown_string_names_the_accepted_ones(self):
        with self.assertRaisesRegex(ValueError, "'reverse', 'count', 'spectral'"):
            self._xyp_(x_order='alphabetical')

    def test_a_shared_axis_uses_the_global_count_order_in_every_tile(self):
        """Per tile, the counts differ -- ordering each tile by its own would break the
        shared axis.  renderSmallMultiples lifts the global order into every tile."""
        _df_ = self.df.with_columns(pl.Series('grp', ['g1', 'g2', 'g1', 'g2', 'g2', 'g2', 'g2', 'g1']))
        _t_  = self.p2s.xyp(df=_df_, x='pet', y='qty', dot_size=5, x_order='count', sm_shared={self.p2s.SM_X})
        _tiles_ = _t_.renderSmallMultiples(_df_, {_g_: _df_.filter(pl.col('grp') == _g_) for _g_ in ('g1', 'g2')}, None)
        _global_ = _axis_order_(self._xyp_(x_order='count'))
        for _g_, _tile_ in _tiles_.items():
            with self.subTest(tile=_g_):
                self.assertEqual([_v_ for _v_ in _global_ if _v_ in _axis_order_(_tile_)], _axis_order_(_tile_))

    def test_the_render_changes_with_the_order(self):
        """Not only the index: the axis labels are drawn in the new order."""
        self.assertNotEqual(normalize_svg(self._xyp_().svg), normalize_svg(self._xyp_(x_order='count').svg))


class Testxyp_partial_order(unittest.TestCase):
    '''Both the list and the dict form used to send every unlisted value to one shared
    fallback slot, so unlisted categories silently overplotted each other.  Default is
    now to append them; p2s.REMAINDERp merges them into a labelled bucket.'''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    _DF_ = pl.DataFrame({'x': ['a', 'b', 'c', 'd', 'e'] * 4,
                         'y': [1.0, 2.0, 3.0, 4.0, 5.0] * 4})

    def _slots(self, **kwargs):
        '''{value: index} actually assigned, after a render.'''
        xy = self.p2s.xyp(self._DF_, x='x', y='y', wxh=(400, 300), **kwargs)
        xy._repr_svg_()
        return dict(zip(xy.df_flat['__x__'].to_list(), xy.df_flat['__xi__'].to_list())), xy

    # -- default: append ----------------------------------------------------

    def test_partial_list_gives_each_value_its_own_slot(self):
        _slots_, _ = self._slots(x_order=['a', 'b'])
        self.assertEqual(_slots_, {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4})

    def test_partial_dict_continues_past_the_highest_index(self):
        _slots_, _ = self._slots(x_order={'a': 10, 'b': 20})
        self.assertEqual(_slots_['a'], 10)
        self.assertEqual(_slots_['b'], 20)
        self.assertEqual(sorted([_slots_[_v_] for _v_ in ('c', 'd', 'e')]), [21, 22, 23])

    def test_unlisted_values_never_collide(self):
        '''The regression: c, d and e all landed on one index.'''
        for _order_ in (['a', 'b'], {'a': 0, 'b': 1}):
            _slots_, _ = self._slots(x_order=_order_)
            self.assertEqual(len(set(_slots_.values())), len(_slots_),
                             msg=f'x_order={_order_} collapsed values onto a shared slot')

    def test_partial_order_on_y_axis_too(self):
        df = pl.DataFrame({'x': [1.0, 2.0, 3.0], 'y': ['p', 'q', 'r']})
        xy = self.p2s.xyp(df, x='x', y='y', y_order=['p'], wxh=(400, 300))
        xy._repr_svg_()
        self.assertEqual(len(set(xy.df_flat['__yi__'].to_list())), 3)

    def test_partial_tuple_order_appends(self):
        df = pl.DataFrame({'qty':  [2, 8, 5, 15],
                           'type': ['cat', 'cat', 'dog', 'goldfish'],
                           'color': ['gray', 'orange', 'spotted', 'orange']})
        xy = self.p2s.xyp(df, x=('type', 'color'), y='qty',
                          x_order=[('goldfish', 'orange')], wxh=(400, 300))
        xy._repr_svg_()
        self.assertEqual(len(set(xy.df_flat['__xi__'].to_list())), 4)

    # -- REMAINDERp: merge --------------------------------------------------

    def test_remainder_merges_unlisted_into_one_slot(self):
        _slots_, _ = self._slots(x_order=['a', 'b', self.p2s.REMAINDERp])
        self.assertEqual(_slots_, {'a': 0, 'b': 1, 'remainder': 2})

    def test_remainder_honours_sentinel_position(self):
        _slots_, _ = self._slots(x_order=[self.p2s.REMAINDERp, 'a', 'b'])
        self.assertEqual(_slots_, {'remainder': 0, 'a': 1, 'b': 2})

    def test_remainder_dict_form(self):
        _slots_, _ = self._slots(x_order={'a': 0, 'b': 1, self.p2s.REMAINDERp: 2})
        self.assertEqual(_slots_, {'a': 0, 'b': 1, 'remainder': 2})

    def test_remainder_rewrites_the_value_so_the_axis_label_is_honest(self):
        '''The axis label is read from __x__ at arg_max(__xi__), so a bucket that only
        remapped the index would label itself with one of the values it swallowed.'''
        _slots_, xy = self._slots(x_order=['a', 'b', self.p2s.REMAINDERp])
        self.assertEqual(xy.df_flat['__x__'][xy.df_flat['__xi__'].arg_max()], 'remainder')
        self.assertIn('remainder', xy._repr_svg_())

    def test_remainder_keeps_every_row(self):
        _, xy = self._slots(x_order=['a', self.p2s.REMAINDERp])
        self.assertEqual(len(xy.df_flat), len(self._DF_))

    def test_remainder_with_no_unlisted_values_is_a_no_op(self):
        _slots_, _ = self._slots(x_order=['a', 'b', 'c', 'd', 'e', self.p2s.REMAINDERp])
        self.assertEqual(_slots_, {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4})

    def test_remainder_alone_collapses_everything(self):
        _slots_, _ = self._slots(x_order=[self.p2s.REMAINDERp])
        self.assertEqual(_slots_, {'remainder': 0})

    def test_remainder_casts_integer_categories(self):
        df = pl.DataFrame({'x': [10, 20, 30, 40] * 3, 'y': [1.0, 2.0, 3.0, 4.0] * 3})
        xy = self.p2s.xyp(df, x=('x', self.p2s.SETp), y='y',
                          x_order=[10, 20, self.p2s.REMAINDERp], wxh=(400, 300))
        xy._repr_svg_()
        self.assertEqual(dict(zip(xy.df_flat['__x__'].to_list(), xy.df_flat['__xi__'].to_list())),
                         {'10': 0, '20': 1, 'remainder': 2})

    def test_remainder_label_collision_raises(self):
        df = pl.DataFrame({'x': ['remainder', 'b', 'c'], 'y': [1.0, 2.0, 3.0]})
        with self.assertRaises(ValueError):
            self.p2s.xyp(df, x='x', y='y', wxh=(400, 300),
                         x_order=['remainder', self.p2s.REMAINDERp])._repr_svg_()

    def test_remainder_on_tuple_order_raises_not_implemented(self):
        '''No single struct value can name the bucket, so this fails loudly.'''
        df = pl.DataFrame({'qty':  [2, 8, 5, 15],
                           'type': ['cat', 'cat', 'dog', 'goldfish'],
                           'color': ['gray', 'orange', 'spotted', 'orange']})
        with self.assertRaises(NotImplementedError):
            self.p2s.xyp(df, x=('type', 'color'), y='qty', wxh=(400, 300),
                         x_order=[('goldfish', 'orange'), self.p2s.REMAINDERp])._repr_svg_()

    # -- lazy parity --------------------------------------------------------

    def test_lazy_and_eager_agree(self):
        for _order_ in (['a', 'b'], ['a', 'b', self.p2s.REMAINDERp],
                        {'a': 0, 'b': 1}, {'a': 0, 'b': 1, self.p2s.REMAINDERp: 2}):
            _eager_, _ = self._slots(x_order=_order_, use_lazy_execution=False)
            _lazy_,  _ = self._slots(x_order=_order_, use_lazy_execution=True)
            self.assertEqual(_eager_, _lazy_, msg=f'x_order={_order_} differs under lazy execution')



class Testxyp_categorical_label_priority(unittest.TestCase):
    '''Which category labels get drawn must not depend on chance.

    __renderContext_set__ picks label positions in priority order -- most rows first --
    and stops once they no longer fit.  That order came from
    `group_by([px, axis]).len().sort('len', descending=True)`, and group_by returns
    groups arbitrarily, so on a categorical axis where the counts are EQUAL (the common
    case) every row tied on 'len' and the priority was whatever order the engine
    happened to produce.

    The visible effect: the same data rendered twice drew different axis labels -- four
    runs of a 30-category axis produced four different sets -- and lazy and eager
    execution disagreed with each other for the same reason.  Ties now break on the
    screen coordinate.

    Lazy vs eager is the invariant worth asserting: it fails deterministically without
    the fix, where run-to-run variation needs separate processes to show up.
    PLANNING.md §15.
    '''
    def setUp(self):
        self.p2s = Polars2SVG()

    def _df_(self, cats):
        _n_ = cats * 4          # equal counts -> every row ties on 'len'
        return pl.DataFrame({'c': [f'c{i % cats:02d}' for i in range(_n_)],
                             'd': [f'd{i % 3}'        for i in range(_n_)]})

    def _svg_(self, df, lazy, wxh):
        return normalize_svg(self.p2s.xyp(df, x='c', y='d', wxh=wxh,
                                          use_lazy_execution=lazy).svg)

    def test_execution_mode_does_not_change_the_render(self):
        # More categories than fit is the case that exposed it: with everything tied,
        # the arbitrary order decided which labels survived.
        for _cats_, _wxh_ in ((4, (256, 256)), (30, (256, 256)),
                              (60, (256, 256)), (30, (128, 128))):
            with self.subTest(categories=_cats_, wxh=_wxh_):
                _df_ = self._df_(_cats_)
                self.assertEqual(self._svg_(_df_, True,  _wxh_),
                                 self._svg_(_df_, False, _wxh_),
                                 'use_lazy_execution must decide how the work is '
                                 'scheduled, never what is drawn')

    def test_repeated_renders_are_identical(self):
        _df_ = self._df_(30)
        _first_ = self._svg_(_df_, False, (256, 256))
        for _i_ in range(4):
            with self.subTest(repeat=_i_):
                self.assertEqual(self._svg_(_df_, False, (256, 256)), _first_)


if __name__ == '__main__':
    unittest.main()
