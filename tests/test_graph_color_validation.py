"""A colour spec that names no column is an error, in all three graph components.

linkp and chordp validated node_color= but not color=, so a misspelt color= field drew
every link in the default colour and said nothing.  A misspelt field inside a tuple
spec did fail, but deep inside polars with its own ColumnNotFoundError, naming neither
the parameter nor the spec.  spreadlinesp checked node_color= not at all: a misspelt
field fell through to the by-name hash and coloured every node by its name (PLANNING.md
§5 C-graph-color-unvalidated).

Each refusal is paired with the documented forms still being accepted, so a validator
that refused everything could not pass either.
"""
import datetime
import unittest

import polars as pl

from polars2svg import Polars2SVG


_DF_  = pl.DataFrame({'fm':       ['a', 'b', 'c', 'd', 'b'],
                      'to':       ['b', 'c', 'd', 'a', 'a'],
                      'category': ['x', 'y', 'y', 'x', 'x'],
                      'cat_n':    [10, 12, 12, 10, 10]})
_REL_ = [('fm', 'to')]
_POS_ = {'a': (0.0, 0.5), 'b': (0.5, 0.0), 'c': (1.0, 0.5), 'd': (0.5, 1.0)}


class TestGraphColorValidation(unittest.TestCase):
    def setUp(self):
        self.p2s = Polars2SVG()

    def _linkp(self, **kw):
        return self.p2s.linkp(df=_DF_, relationships=_REL_, pos=_POS_, wxh=(96, 96), **kw)

    def _chordp(self, **kw):
        return self.p2s.chordp(df=_DF_, relationships=_REL_, wxh=(96, 96), **kw)

    def _spreadlinesp(self, **kw):
        _df_ = _DF_.with_columns(pl.lit(datetime.datetime(2024, 1, 1)).alias('t'))
        return self.p2s.spreadlinesp(_df_, _REL_, ego='a', time='t', wxh=(256, 128), **kw)

    # The spec is refused with a ValueError that names the parameter and what is wrong
    def assertRefused(self, build, param, spec, names):
        with self.assertRaises(ValueError) as _ctx_:
            build(**{param: spec})
        self.assertIn(f'{param}=', str(_ctx_.exception))
        self.assertIn(names, str(_ctx_.exception))

    def test_a_misspelt_color_field_is_refused(self):
        for _name_, _build_ in (('linkp', self._linkp), ('chordp', self._chordp)):
            with self.subTest(component=_name_):
                self.assertRefused(_build_, 'color', 'categroy', "'categroy'")

    def test_a_misspelt_field_inside_a_tuple_is_refused_by_name(self):
        for _name_, _build_ in (('linkp', self._linkp), ('chordp', self._chordp)):
            for _param_, _spec_ in (('color', ('categroy', self.p2s.CSETp)),
                                    ('node_color', ('cat_nn', self.p2s.CMAGNITUDE_SUMp))):
                with self.subTest(component=_name_, param=_param_):
                    self.assertRefused(_build_, _param_, _spec_, repr(_spec_[0]))

    def test_a_misspelt_spreadlines_node_color_is_refused(self):
        self.assertRefused(self._spreadlinesp, 'node_color', 'categroy', "'categroy'")
        self.assertRefused(self._spreadlinesp, 'node_color', ('category', self.p2s.CSETp), 'node_color=')

    def test_the_documented_forms_are_still_accepted(self):
        _color_ = [None, '#ff0000', 'category', 'cat_n', self.p2s.CROW_MAGNITUDEp, self.p2s.CROW_STRETCHEDp,
                   ('category', self.p2s.CSETp), ('cat_n', self.p2s.CMAGNITUDE_SUMp),
                   ('category', self.p2s.CSET_STRETCHEDp)]
        for _name_, _build_ in (('linkp', self._linkp), ('chordp', self._chordp)):
            for _spec_ in _color_:
                with self.subTest(component=_name_, color=_spec_):
                    self.assertTrue(_build_(color=_spec_)._repr_svg_().startswith('<svg'))
            for _spec_ in _color_ + [self.p2s.COLOR_BY_NODE_NAME, {'a': '#00ff00'}]:
                with self.subTest(component=_name_, node_color=_spec_):
                    self.assertTrue(_build_(node_color=_spec_)._repr_svg_().startswith('<svg'))
        for _spec_ in (None, self.p2s.COLOR_BY_NODE_NAME, '#ff0000', {'a': '#00ff00'}, 'category'):
            with self.subTest(component='spreadlinesp', node_color=_spec_):
                self.assertTrue(self._spreadlinesp(node_color=_spec_)._repr_svg_().startswith('<svg'))

    def test_chordp_keeps_its_src_and_dst_keywords(self):
        '''No column is named src or dst here, so they are chordp's keywords -- still
        accepted -- while a near miss is not.'''
        for _kw_ in ('src', 'dst'):
            with self.subTest(keyword=_kw_):
                self.assertTrue(self._chordp(color=_kw_)._repr_svg_().startswith('<svg'))
        self.assertRefused(self._chordp, 'color', 'srcx', "'srcx'")
        # linkp has no such keywords
        self.assertRefused(self._linkp, 'color', 'src', "'src'")

    def test_link_by_node_constants_are_chordp_color_only(self):
        '''p2s.COLOR_BY_SRC_NODE / COLOR_BY_DST_NODE colour a chordp link by one of its
        nodes.  Anywhere else they are refused by name -- they used to fall through to
        "unsupported type NodeColorP".'''
        for _c_ in (self.p2s.COLOR_BY_SRC_NODE, self.p2s.COLOR_BY_DST_NODE):
            with self.subTest(constant=_c_.name):
                self.assertTrue(self._chordp(color=_c_)._repr_svg_().startswith('<svg'))
                self.assertRefused(self._chordp, 'node_color', _c_, f'p2s.{_c_.name}')
                self.assertRefused(self._linkp,  'color',      _c_, f'p2s.{_c_.name}')
                self.assertRefused(self._linkp,  'node_color', _c_, f'p2s.{_c_.name}')


if __name__ == '__main__':
    unittest.main()
