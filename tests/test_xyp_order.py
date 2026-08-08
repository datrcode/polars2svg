import unittest
import polars as pl
from polars2svg import Polars2SVG

class Testxyp_order(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def test_list(self):
        for _lazy_ in [True, False]:
            df = pl.DataFrame({'qty':[1,2,3,10,4], 'pet':['cat','dog','parakeet','goldfish','ferret']})
            _params_ = {'df':df, 'x':'pet', 'y':'qty', 'color':'pet', 'dot_size':10}
            _params_['use_lazy_execution'] = _lazy_
            _xyp0_  = self.p2s.xyp(**_params_)                                                                    # default
            _xyp1_  = self.p2s.xyp(**_params_, x_order=['goldfish', 'ferret', 'parakeet', 'dog', 'cat'])          # complete order (decreasing qtys)
            _xyp2_  = self.p2s.xyp(**_params_, x_order=['goldfish', 'ferret', 'parakeet', 'dog', 'cat', 'snake']) # complete order + extra (decreasing qtys)
            _xyp3_  = self.p2s.xyp(**_params_, x_order=['goldfish', 'ferret'])                                    # incomplete order
            _xyp4_  = self.p2s.xyp(**_params_, x_order=['goldfish', 'ferret', 'snake'])                           # incomplete order + extra

    def test_listTuple(self):
        for _lazy_ in [True, False]:
            df = pl.DataFrame({'qty':  [2,        8,         5,          15],
                               'type': ['cat',    'cat',     'dog',      'goldfish'], 
                               'color':['gray',   'orange',  'spotted',  'orange']})
            _params_ = {'df':df, 'x':('type','color'), 'y':'qty', 'color':'color', 'dot_size':10}
            _params_['use_lazy_execution'] = _lazy_
            _xyp0_   = self.p2s.xyp(**_params_)
            _xyp1_   = self.p2s.xyp(**_params_, x_order=[('goldfish','orange'), ('cat','orange'), ('dog','spotted'), ('cat','gray')])                     # complete
            _xyp2_   = self.p2s.xyp(**_params_, x_order=[('goldfish','orange'), ('cat','orange'), ('dog','spotted'), ('cat','gray'), ('snake','albino')]) # complete + extra
            _xyp3_   = self.p2s.xyp(**_params_, x_order=[('goldfish','orange'), ('cat','gray')])                                                          # incomplete
            _xyp4_   = self.p2s.xyp(**_params_, x_order=[('goldfish','orange'), ('cat','gray'), ('snake', 'albino')])                                     # incomplete + extra

    def test_dict(self):
        for _lazy_ in [True, False]:
            df = pl.DataFrame({'qty':[1,2,3,10,4], 'pet':['cat','dog','parakeet','goldfish','ferret']})
            _params_ = {'df':df, 'x':'pet', 'y':'qty', 'color':'pet', 'dot_size':10}
            _params_['use_lazy_execution'] = _lazy_
            _xyp0_  = self.p2s.xyp(**_params_)                                                                                      # default
            _xyp1_  = self.p2s.xyp(**_params_, x_order={'goldfish':10, 'ferret':15, 'parakeet':20, 'dog':25, 'cat':30})             # complete order (decreasing qtys)
            _xyp2_  = self.p2s.xyp(**_params_, x_order={'goldfish':10, 'ferret':15, 'parakeet':20, 'dog':25, 'cat':30, 'snake':35}) # complete order + extra (decreasing qtys)
            _xyp3_  = self.p2s.xyp(**_params_, x_order={'goldfish':10, 'ferret':15})                                                # incomplete order
            _xyp4_  = self.p2s.xyp(**_params_, x_order={'goldfish':10, 'ferret':15, 'snake':35})                                    # incomplete order + extra

    def test_dictTuple(self):
        for _lazy_ in [True, False]:
            df = pl.DataFrame({'qty':  [2,        8,         5,          15],
                            'type': ['cat',    'cat',     'dog',      'goldfish'], 
                            'color':['gray',   'orange',  'spotted',  'orange']})
            _params_ = {'df':df, 'x':('type','color'), 'y':'qty', 'color':'color', 'dot_size':10}
            _params_['use_lazy_execution'] = _lazy_
            _xyp0_   = self.p2s.xyp(**_params_)
            _xyp1_   = self.p2s.xyp(**_params_, x_order={('goldfish','orange'):5, ('cat','orange'):6, ('dog','spotted'):7, ('cat','gray'):10})                        # complete
            _xyp2_   = self.p2s.xyp(**_params_, x_order={('goldfish','orange'):5, ('cat','orange'):6, ('dog','spotted'):7, ('cat','gray'):10, ('snake','albino'):20}) # complete + extra
            _xyp3_   = self.p2s.xyp(**_params_, x_order={('goldfish','orange'):5, ('cat','gray'):10})                                                                 # incomplete
            _xyp4_   = self.p2s.xyp(**_params_, x_order={('goldfish','orange'):5, ('cat','gray'):10, ('snake', 'albino'):20})                                         # incomplete + extra

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


if __name__ == '__main__':
    unittest.main()
