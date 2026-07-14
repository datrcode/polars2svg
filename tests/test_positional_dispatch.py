#
# test_positional_dispatch.py
#
# Positional dispatch keys on Python type/shape: xyp routes the first str/tuple/list
# positional to x and the second to y; linkp routes a list-of-tuples to relationships
# and a coordinate dict to pos. An argument-order mistake (or a value meant for another
# parameter) is silently routed and only surfaces downstream as a bare "column not
# found". These tests verify that such failures now name positional dispatch as the
# likely cause, that keyword assignments do NOT get the hint (no dispatch ambiguity),
# and that valid positional specs still parse/render unchanged.
#
import unittest
import polars as pl
from polars2svg import Polars2SVG


def _xy_df():
    return pl.DataFrame({'a': [1, 2, 3, 4], 'b': [4, 3, 2, 1]})


def _link_df():
    return pl.DataFrame({
        'fm':     ['a', 'b', 'c'],
        'to':     ['b', 'c', 'a'],
        'weight': [1,   2,   3  ],
    })


class TestPositionalDispatchHintHelper(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def test_returns_empty_when_keyword(self):
        self.assertEqual(self.p2s.positionalDispatchHint('XYp', 'x', False), '')

    def test_returns_hint_when_positional(self):
        _h_ = self.p2s.positionalDispatchHint('XYp', 'x', True)
        self.assertIn('positional', _h_)
        self.assertIn('x', _h_)
        self.assertIn('XYp', _h_)

    def test_names_the_param_and_component(self):
        _h_ = self.p2s.positionalDispatchHint('LinkP', 'relationships', True)
        self.assertIn('relationships', _h_)
        self.assertIn('LinkP', _h_)


class TestXYpPositionalDispatch(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def test_positional_x_missing_column_hints_dispatch(self):
        with self.assertRaises(TypeError) as _cm_:
            self.p2s.xyp(_xy_df(), 'nope', 'b')
        _msg_ = str(_cm_.exception)
        self.assertIn('nope', _msg_)
        self.assertIn('positional', _msg_)
        self.assertIn('x', _msg_)

    def test_positional_y_missing_column_hints_dispatch(self):
        with self.assertRaises(TypeError) as _cm_:
            self.p2s.xyp(_xy_df(), 'a', 'nope')
        _msg_ = str(_cm_.exception)
        self.assertIn('nope', _msg_)
        self.assertIn('positional', _msg_)

    def test_keyword_x_missing_column_no_hint(self):
        with self.assertRaises(TypeError) as _cm_:
            self.p2s.xyp(_xy_df(), x='nope', y='b')
        _msg_ = str(_cm_.exception)
        self.assertIn('nope', _msg_)
        self.assertNotIn('positional', _msg_)

    def test_keyword_overriding_positional_clears_flag(self):
        # x is passed BOTH positionally ('a') and by keyword (the keyword wins); the
        # keyword form is explicit, so a missing column must NOT get the positional hint.
        with self.assertRaises(TypeError) as _cm_:
            self.p2s.xyp(_xy_df(), 'a', 'b', x='nope')
        self.assertNotIn('positional', str(_cm_.exception))

    def test_hex_color_positioned_as_x_caught_early(self):
        with self.assertRaises(TypeError) as _cm_:
            self.p2s.xyp(_xy_df(), '#ff0000', 'b')
        _msg_ = str(_cm_.exception)
        self.assertIn('#ff0000', _msg_)
        self.assertIn('color', _msg_)
        # Should name x, not defer to a confusing "does not contain column"
        self.assertNotIn('does not contain column', _msg_)

    def test_hex_color_positioned_as_y_caught_early(self):
        with self.assertRaises(TypeError) as _cm_:
            self.p2s.xyp(_xy_df(), 'a', '#f00')
        _msg_ = str(_cm_.exception)
        self.assertIn('#f00', _msg_)
        self.assertIn('color', _msg_)

    def test_valid_positional_still_renders(self):
        _xyp_ = self.p2s.xyp(_xy_df(), 'a', 'b')
        self.assertIn('<svg', _xyp_.svg)

    def test_valid_keyword_still_renders(self):
        _xyp_ = self.p2s.xyp(_xy_df(), x='a', y='b')
        self.assertIn('<svg', _xyp_.svg)

    def test_hex_color_as_keyword_color_still_valid(self):
        # The early positional hex catch must not touch the legitimate color= path.
        _xyp_ = self.p2s.xyp(_xy_df(), 'a', 'b', color='#ff0000')
        self.assertIn('<svg', _xyp_.svg)


class TestLinkPPositionalDispatch(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def test_positional_relationships_missing_field_hints_dispatch(self):
        with self.assertRaises(ValueError) as _cm_:
            self.p2s.linkp(_link_df(), [('nope', 'to')])
        _msg_ = str(_cm_.exception)
        self.assertIn('nope', _msg_)
        self.assertIn('positional', _msg_)
        self.assertIn('relationships', _msg_)

    def test_keyword_relationships_missing_field_no_hint(self):
        with self.assertRaises(ValueError) as _cm_:
            self.p2s.linkp(_link_df(), relationships=[('nope', 'to')])
        _msg_ = str(_cm_.exception)
        self.assertIn('nope', _msg_)
        self.assertNotIn('positional', _msg_)

    def test_valid_positional_relationships_still_renders(self):
        _lp_ = self.p2s.linkp(_link_df(), [('fm', 'to')])
        self.assertIn('<svg', _lp_.svg)

    def test_valid_keyword_relationships_still_renders(self):
        _lp_ = self.p2s.linkp(_link_df(), relationships=[('fm', 'to')])
        self.assertIn('<svg', _lp_.svg)


if __name__ == '__main__':
    unittest.main()
