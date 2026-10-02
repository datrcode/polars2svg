import re
import unittest
import polars as pl
from polars2svg import Polars2SVG

from random_dataframe import randomDataFrame


#
# Guard against opaque `TypeError: unhashable type` when an unhashable item
# (list/dict/set) is nested inside an xyp field/enum spec.  Membership tests
# against `all_enums` (a set) used to crash instead of raising a descriptive
# error ("enum extraction raises opaque TypeError on unhashable items").
#
class Testxyp_unhashable_spec(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()
        self.df  = randomDataFrame(100, seed=51)

    def _assertNotUnhashableTypeError(self, fn):
        # The whole point: never surface a bare `TypeError: unhashable type`.
        try:
            fn()
        except TypeError as e:
            if 'unhashable' in str(e):
                self.fail(f'opaque unhashable TypeError leaked: {e}')
            raise

    # ------------------------------------------------------------------ #
    # __toListAndExtractEnums__ / __cleanTuple__ path (x/y/color/dot_size/…) #
    # ------------------------------------------------------------------ #
    def test_list_nested_in_tuple_raises_valueerror(self):
        # ('a', ['b']) -- unhashable list nested inside a tuple spec
        for attr in ['x', 'y', 'color', 'dot_size', 'opacity', 'line_order_by']:
            with self.subTest(attr=attr):
                kw = {'x': 'a', 'y': 'b'}
                kw[attr] = ('a', ['b'])
                with self.assertRaises(ValueError):
                    self._assertNotUnhashableTypeError(lambda: self.p2s.xyp(self.df, **kw))

    def test_list_element_in_list_spec_raises_valueerror(self):
        # x=[['a']] -- a bare list nested as a list element (not a tuple)
        with self.assertRaises(ValueError):
            self._assertNotUnhashableTypeError(lambda: self.p2s.xyp(self.df, x=[['a']], y='b'))

    def test_tuple_containing_list_at_top_level_raises_valueerror(self):
        # x=('a', ['b']) -- the tuple itself is a top-level value; a tuple that
        # *contains* an unhashable still hashes-explodes if not guarded.
        with self.assertRaises(ValueError):
            self._assertNotUnhashableTypeError(lambda: self.p2s.xyp(self.df, x=('a', ['b']), y='b'))

    def test_dict_and_set_nested_in_tuple_raise_valueerror(self):
        for bad in [{'k': 1}, {'a', 'b'}]:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    self._assertNotUnhashableTypeError(lambda: self.p2s.xyp(self.df, x=('a', bad), y='b'))

    # ------------------------------------------------------------------ #
    # __separateAndCleanParam__ path (x_distributions / y_distributions)   #
    # ------------------------------------------------------------------ #
    def test_list_nested_in_distributions_raises_valueerror(self):
        for attr in ['x_distributions', 'y_distributions']:
            for spec in [[('a', ['b'])], ('a', ['b']), ['a', {'k': 1}], ['a', {'s'}]]:
                with self.subTest(attr=attr, spec=spec):
                    kw = {'x': 'a', 'y': 'b', attr: spec}
                    with self.assertRaises(ValueError):
                        self._assertNotUnhashableTypeError(lambda: self.p2s.xyp(self.df, **kw))

    # ------------------------------------------------------------------ #
    # Regression: valid specs must still parse without error               #
    # ------------------------------------------------------------------ #
    def test_valid_specs_still_parse(self):
        # ... and each draws what it says
        _kept_ = self.df.filter(pl.col('b').is_not_null())      # rows with y
        # a tuple axis is one category per distinct (a, b) pair of the rows it keeps
        _xyp_ = self.p2s.xyp(self.df, x=('a', 'b'), y='b')
        self.assertEqual(_xyp_.df_flat['__xi__'].n_unique(), _kept_.select(pl.struct('a', 'b')).unique().height)
        # SETp makes a numeric column categorical: one slot per distinct value
        _xyp_ = self.p2s.xyp(self.df, x=('a', self.p2s.SETp), y='b')
        self.assertEqual(_xyp_.df_flat['__xi__'].n_unique(), _kept_.filter(pl.col('a').is_not_null())['a'].n_unique())
        # a hex colour is every dot's colour
        _dots_ = re.search(r'<g class="(?:rect|circle)-group-\d+"[^>]*>(.*?)</g>',
                           self.p2s.xyp(self.df, x='a', y='b', color='#ff0000').svg, re.S).group(1)
        self.assertEqual(set(re.findall(r'fill="([^"]*)"', _dots_)), {'#ff0000'})
        # several distribution fields draw one outline each, in two colours
        for _spec_ in (('a', 'b'), ['a', 'b', self.p2s.SETp]):
            with self.subTest(x_distributions=str(_spec_)):
                _outlines_ = re.findall(r'<path d="[^"]*" stroke="(#[0-9a-f]{6})" stroke-width="0.5" fill="none" />',
                                        self.p2s.xyp(self.df, x='a', y='b', x_distributions=_spec_).svg)
                self.assertEqual(len(set(_outlines_)), 2, f'expected two field outlines, got {_outlines_}')
        # a row-count distribution draws its bars
        _svg_ = self.p2s.xyp(self.df, x='a', y='b', y_distributions=self.p2s.ROW_COUNTp).svg
        self.assertGreater(len(re.findall(r'<rect [^>]*stroke-opacity="0.9"', _svg_)), 0)

    # ------------------------------------------------------------------ #
    # Unit tests of the helpers themselves                                 #
    # ------------------------------------------------------------------ #
    def test_isEnum_helper_is_crash_proof(self):
        # __isEnum__ has trailing '__', so it is not name-mangled.
        _xyp_    = self.p2s.xyp(self.df, x='a', y='b')
        _isEnum_ = getattr(_xyp_, '__isEnum__')
        # unhashable items -> False, never TypeError
        self.assertFalse(_isEnum_(['a']))
        self.assertFalse(_isEnum_({'k': 1}))
        self.assertFalse(_isEnum_({'a', 'b'}))
        self.assertFalse(_isEnum_(('a', ['b'])))   # tuple containing a list
        # real enums / non-enums resolve correctly
        self.assertTrue(_isEnum_(self.p2s.SETp))
        self.assertFalse(_isEnum_('not_an_enum'))

    def test_assertHashableSpecItem_raises_on_unhashable(self):
        _xyp_    = self.p2s.xyp(self.df, x='a', y='b')
        _assert_ = getattr(_xyp_, '__assertHashableSpecItem__')
        with self.assertRaises(ValueError):
            _assert_(['a'], 'test')
        # hashable items pass through silently
        _assert_('a', 'test')
        _assert_(self.p2s.SETp, 'test')


if __name__ == '__main__':
    unittest.main()
