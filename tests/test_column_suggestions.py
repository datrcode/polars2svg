"""
Tests for Polars2SVG.columnSuggestion() and the "column not found" messages that use it.

smallp(df, tmpl, 'country') on a frame whose column is 'country_name' said only "Unknown
argument type: <class 'str'>", and the other components' "field not found" left the
reader to go and list the columns.  Every such message now ends with the column the name
was most likely meant to be -- or, with nothing close, the columns themselves.
"""
import unittest

import polars as pl

from polars2svg import Polars2SVG


_COLS_ = ['iso3', 'country_name', 'country_code', 'period', 'tbpd', 'crude_or_total', '__p2s_index__']


class TestColumnSuggestion(unittest.TestCase):

    def setUp(self):
        self.p2s = Polars2SVG()

    def test_contained_name(self):
        self.assertEqual(self.p2s.columnSuggestion('country', _COLS_), " -- did you mean 'country_name' or 'country_code'?")
        self.assertEqual(self.p2s.columnSuggestion('iso', _COLS_), " -- did you mean 'iso3'?")

    def test_case_comes_first(self):
        self.assertTrue(self.p2s.columnSuggestion('Period', _COLS_).startswith(" -- did you mean 'period'"))

    def test_typo(self):
        self.assertEqual(self.p2s.columnSuggestion('tbdp', _COLS_), " -- did you mean 'tbpd'?")

    def test_at_most_three(self):
        _cols_ = ['sip_a', 'sip_b', 'sip_c', 'sip_d', 'sip_e']
        self.assertEqual(self.p2s.columnSuggestion('sip', _cols_).count("'"), 6)

    def test_nothing_close_lists_a_small_frame(self):
        _s_ = self.p2s.columnSuggestion('zzz', _COLS_)
        self.assertTrue(_s_.startswith(' -- the columns are '))
        self.assertIn("'country_name'", _s_)
        self.assertNotIn('__p2s_index__', _s_)

    def test_nothing_close_in_a_large_frame_says_nothing(self):
        self.assertEqual(self.p2s.columnSuggestion('zzz', [f'c{_i_}' for _i_ in range(20)]), '')

    def test_short_names_do_not_match_by_containment(self):
        # 'ip' is in no column and no two-letter column is offered for a longer name
        self.assertFalse(self.p2s.columnSuggestion('ip', ['sip_str', 'dip_str', 'x', 'y'] + [f'c{_i_}' for _i_ in range(20)]).startswith(' -- did you mean'))

    def test_internal_columns_are_never_offered(self):
        self.assertEqual(self.p2s.columnSuggestion('p2s_index', ['__p2s_index__'] + [f'c{_i_}' for _i_ in range(20)]), '')

    def test_accepts_a_frame_and_a_tfield(self):
        _df_ = pl.DataFrame({'timestamp': [1], 'v': [2]})
        self.assertEqual(self.p2s.columnSuggestion(self.p2s.tField('timestamps', self.p2s.PT_Hp), _df_),
                         " -- did you mean 'timestamp'?")

    def test_not_a_name(self):
        self.assertEqual(self.p2s.columnSuggestion(('a', 'b'), _COLS_), '')
        self.assertEqual(self.p2s.columnSuggestion('a', None), '')


class TestComponentMessages(unittest.TestCase):

    def setUp(self):
        self.p2s = Polars2SVG()
        self.df  = pl.DataFrame({'iso3': ['a', 'b'], 'country_name': ['x', 'y'], 'period': [1, 2], 'tbpd': [1.0, 2.0],
                                 'src': ['a', 'b'], 'dst': ['b', 'a'],
                                 't': pl.datetime_range(pl.datetime(2024, 1, 1), pl.datetime(2024, 1, 2), '1d', eager=True)})
        self.tmpl = self.p2s.xyp(self.df, 'period', 'tbpd')

    def test_smallp_positional_category(self):
        with self.assertRaisesRegex(ValueError, r"category_by='country' .* did you mean 'country_name'\?"):
            self.p2s.smallp(self.df, self.tmpl, 'country')
        with self.assertRaisesRegex(ValueError, r"'iso' -- did you mean 'iso3'\?"):
            self.p2s.smallp(self.df, self.tmpl, ('iso', 'country_name'))

    def test_smallp_keyword_category_and_order(self):
        with self.assertRaisesRegex(ValueError, r"category_by='countryname' .* did you mean 'country_name'\?"):
            self.p2s.smallp(self.df, self.tmpl, category_by='countryname')
        with self.assertRaisesRegex(ValueError, r"order='tbdp' .* did you mean 'tbpd'\?"):
            self.p2s.smallp(self.df, self.tmpl, 'country_name', order='tbdp')
        with self.assertRaisesRegex(ValueError, r"'tbdp' -- did you mean 'tbpd'\?"):
            self.p2s.smallp(self.df, self.tmpl, 'country_name', order=('tbdp', self.p2s.MEANp))

    # The valid forms still work.
    def test_smallp_valid_specs(self):
        for _kw_ in ({}, {'order': 'tbpd'}, {'order': ('tbpd', self.p2s.MEANp)}):
            with self.subTest(kw=_kw_):
                self.assertIn('<svg', self.p2s.smallp(self.df, self.tmpl, 'country_name', **_kw_)._repr_svg_())

    def test_each_component(self):
        _p_ = self.p2s
        for _name_, _build_, _pattern_ in [
            ('xyp y',      lambda: _p_.xyp(self.df, 'period', 'tbdp'),                 r"tbdp -- did you mean 'tbpd'\?"),
            ('xyp color',  lambda: _p_.xyp(self.df, 'period', 'tbpd', color='countr'), r"did you mean 'country_name'\?"),
            ('xyp dist',   lambda: _p_.xyp(self.df, 'period', 'tbpd', y_distributions='tbp'), r"did you mean 'tbpd'\?"),
            ('xyp spectral', lambda: _p_.xyp(self.df, 'iso3', 'country_name', y_order='spectral', spectral_by='perio'),
                                                                                         r"'perio' -- did you mean 'period'\?"),
            ('histop',     lambda: _p_.histop(self.df, 'Country_Name'),                r"did you mean 'country_name'\?"),
            ('histop count', lambda: _p_.histop(self.df, 'iso3', count='tbp'),         r"did you mean 'tbpd'\?"),
            ('piep',       lambda: _p_.piep(self.df, 'iso'),                           r"did you mean 'iso3'\?"),
            ('timep',      lambda: _p_.timep(self.df, 'time'),                         r"the columns are .*'t'"),
            ('linkp',      lambda: _p_.linkp(self.df, [('source', 'dst')]),            r"did you mean 'src'\?"),
            ('chordp',     lambda: _p_.chordp(self.df, [('src', 'dest')]),             r"did you mean 'dst'\?"),
        ]:
            with self.subTest(component=_name_):
                with self.assertRaisesRegex((ValueError, TypeError), _pattern_):
                    _build_()._repr_svg_()


if __name__ == '__main__':
    unittest.main()
