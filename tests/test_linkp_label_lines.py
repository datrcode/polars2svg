import unittest
import polars as pl
from polars2svg import Polars2SVG

from svg_test_utils import assert_svg_matches_golden


# DT's example (20261127_linkp_label_lines_design.md): a host described by DNS columns
# that name only one end of each row.
_DF_ = pl.DataFrame({
    'sip':      ['1.2.3.4', '1.2.3.4',   '1.2.3.4',     '5.6.7.8'],
    'dip':      ['8.8.8.8', '8.8.4.4',   '23.12.31.21', '8.8.8.8'],
    'sdns':     ['home',    'home',      'home',        'other_home'],
    'ddns':     ['googledns', 'googledns2', 'news_server', 'googledns'],
    'protocol': ['udp',     'udp',       'tcp',         'udp'],
    'port':     [53,        53,          80,            53],
})
_REL_ = [('sip', 'dip')]
_POS_ = {'1.2.3.4': (0.2, 0.2), '5.6.7.8': (0.8, 0.2),
         '8.8.8.8': (0.8, 0.8), '8.8.4.4': (0.5, 0.8), '23.12.31.21': (0.2, 0.8)}

# the node is (ip, port), so its name is 'ip|port'
_DF_PORTS_ = pl.DataFrame({
    'sip':   ['10.0.0.5', '10.0.0.5', '10.0.0.5'],
    'sport': [443,        443,        22],
    'dip':   ['10.0.0.9', '10.0.0.9', '10.0.0.9'],
    'dport': [80,         80,         80],
    'sdns':  ['web',      'web',      'ssh'],
    'ddns':  ['db',       'db',       'db'],
})
_REL_PORTS_ = [(('sip', 'sport'), ('dip', 'dport'))]
_POS_PORTS_ = {'10.0.0.5|443': (0.2, 0.3), '10.0.0.5|22': (0.2, 0.7), '10.0.0.9|80': (0.8, 0.5)}


def _lp(p2s, node_labels, df=_DF_, rel=_REL_, pos=_POS_, **extra):
    return p2s.linkp(df, rel, pos=pos, wxh=(280, 280), draw_node_labels=True,
                     node_labels=node_labels, **extra)


def _lines(lp):
    '''{first line: all lines} for every label the render drew.'''
    return {_l_[0]: _l_ for _, _, _l_ in lp._node_label_info_}


class TestLabelLineSources(unittest.TestCase):
    def setUp(self):
        self.p2s = Polars2SVG()

    def test_node_id_alone(self):
        self.assertEqual(sorted(_lines(_lp(self.p2s, [self.p2s.NODE_IDp]))),
                         ['1.2.3.4', '23.12.31.21', '5.6.7.8', '8.8.4.4', '8.8.8.8'])

    def test_dict_line(self):
        _by_ = _lines(_lp(self.p2s, [self.p2s.NODE_IDp, {'8.8.8.8': 'benign'}]))
        self.assertEqual(_by_['8.8.8.8'], ['8.8.8.8', 'benign'])
        self.assertEqual(_by_['8.8.4.4'], ['8.8.4.4'])      # a dict need not name every node

    def test_a_dict_only_list_labels_only_the_named_nodes(self):
        # the old dict form drops nodes it does not name; so does a list whose lines are empty there
        _lp_ = _lp(self.p2s, [{'8.8.8.8': 'benign'}])
        self.assertEqual(list(_lines(_lp_).values()), [['benign']])

    def test_pair_reads_only_rows_where_the_node_is_the_key(self):
        _by_ = _lines(_lp(self.p2s, [self.p2s.NODE_IDp, ('sip', 'sdns')]))
        self.assertEqual(_by_['1.2.3.4'], ['1.2.3.4', 'home'])
        # 8.8.8.8 is only ever a dip: the sdns of its rows describe the OTHER end, so no line
        self.assertEqual(_by_['8.8.8.8'], ['8.8.8.8'])

    def test_roles_merge_into_one_line(self):
        _by_ = _lines(_lp(self.p2s, [self.p2s.NODE_IDp, [('sip', 'sdns'), ('dip', 'ddns')]]))
        self.assertEqual(_by_['1.2.3.4'],    ['1.2.3.4', 'home'])
        self.assertEqual(_by_['8.8.8.8'],    ['8.8.8.8', 'googledns'])
        self.assertEqual(_by_['23.12.31.21'], ['23.12.31.21', 'news_server'])

    def test_most_common_value_then_a_count_of_the_rest(self):
        # 1.2.3.4 is udp|53 twice and tcp|80 once
        _by_ = _lines(_lp(self.p2s, [self.p2s.NODE_IDp, ('sip', ('protocol', 'port'))]))
        self.assertEqual(_by_['1.2.3.4'], ['1.2.3.4', 'udp|53 (+1)'])
        self.assertEqual(_by_['5.6.7.8'], ['5.6.7.8', 'udp|53'])

    def test_a_tie_goes_to_the_first_row(self):
        _df_ = pl.DataFrame({'a': ['n', 'n'], 'b': ['m', 'm'], 'v': ['zzz', 'aaa']})   # 'aaa' sorts first, 'zzz' came first
        _by_ = _lines(_lp(self.p2s, [('a', 'v')], df=_df_, rel=[('a', 'b')], pos={'n': (0.2, 0.5), 'm': (0.8, 0.5)}))
        self.assertEqual(list(_by_.values()), [['zzz (+1)']])

    def test_composite_key(self):
        _by_ = _lines(_lp(self.p2s, [self.p2s.NODE_IDp,
                                     [(('sip', 'sport'), 'sdns'), (('dip', 'dport'), 'ddns')]],
                          df=_DF_PORTS_, rel=_REL_PORTS_, pos=_POS_PORTS_))
        self.assertEqual(_by_['10.0.0.5|443'], ['10.0.0.5|443', 'web'])
        self.assertEqual(_by_['10.0.0.5|22'],  ['10.0.0.5|22', 'ssh'])      # same ip, different node
        self.assertEqual(_by_['10.0.0.9|80'],  ['10.0.0.9|80', 'db'])

    def test_composite_value_and_both(self):
        _by_ = _lines(_lp(self.p2s, [(('sip', 'sport'), ('sdns', 'dport'))],
                          df=_DF_PORTS_, rel=_REL_PORTS_, pos=_POS_PORTS_))
        self.assertEqual(sorted(_v_[0] for _v_ in _by_.values()), ['ssh|80', 'web|80'])

    def test_separate_columns_are_separate_lines(self):
        _df_ = _DF_.with_columns(pl.lit('ACME').alias('sorg'))
        _by_ = _lines(_lp(self.p2s, [('sip', 'sdns'), ('sip', 'sorg')], df=_df_))
        self.assertEqual(_by_['home'], ['home', 'ACME'])

    def test_name_prefixes_the_value(self):
        _by_ = _lines(_lp(self.p2s, [self.p2s.labelLine({'8.8.8.8': 'up'}, name='status')]))
        self.assertEqual(list(_by_.values()), [['status=up']])

    def test_empty_values_are_skipped_not_drawn_blank(self):
        _df_ = _DF_.with_columns(pl.Series('sdns', ['', None, None, 'x']))
        _by_ = _lines(_lp(self.p2s, [self.p2s.NODE_IDp, ('sip', 'sdns')], df=_df_))
        self.assertEqual(_by_['1.2.3.4'], ['1.2.3.4'])
        self.assertEqual(_by_['5.6.7.8'], ['5.6.7.8', 'x'])

    def test_integer_node_ids_match_their_key_column(self):
        _df_ = pl.DataFrame({'a': [1, 1], 'b': [2, 3], 'n': ['one', 'one']})
        _by_ = _lines(_lp(self.p2s, [('a', 'n')], df=_df_, rel=[('a', 'b')],
                          pos={1: (0.2, 0.5), 2: (0.8, 0.2), 3: (0.8, 0.8)}))
        self.assertEqual(list(_by_.values()), [['one']])


class TestLabelLineLayout(unittest.TestCase):
    def setUp(self):
        self.p2s = Polars2SVG()

    def test_label_max_lines_does_not_cap_the_number_of_elements(self):
        _by_ = _lines(_lp(self.p2s, [self.p2s.NODE_IDp, {'8.8.8.8': 'a'}, {'8.8.8.8': 'b'},
                                     {'8.8.8.8': 'c'}, {'8.8.8.8': 'd'}, {'8.8.8.8': 'e'}],
                          label_max_lines=2))
        self.assertEqual(_by_['8.8.8.8'], ['8.8.8.8', 'a', 'b', 'c', 'd', 'e'])

    def test_a_long_line_is_cut_with_an_ellipsis_not_wrapped(self):
        _by_ = _lines(_lp(self.p2s, [{'8.8.8.8': 'a very long description of a company webserver'}],
                          label_line_width=12))
        self.assertEqual(list(_by_.values()), [['a very long…']])

    def test_a_plain_dict_still_wraps_as_before(self):
        _by_ = _lines(_lp(self.p2s, {'8.8.8.8': 'a very long description'}, label_line_width=12))
        self.assertEqual(list(_by_.values()), [['a very long', 'description']])

    def test_label_lines_survive_a_template_clone_on_a_frame_without_the_column(self):
        _lp_ = _lp(self.p2s, [self.p2s.NODE_IDp, ('sip', 'sdns')])
        _clone_ = _lp_.render_with(_DF_.drop('sdns'))
        self.assertEqual(_lines(_clone_)['1.2.3.4'], ['1.2.3.4'])

    def test_label_dict_is_the_dict_lines(self):
        _d_ = {'8.8.8.8': 'benign'}
        self.assertIs(_lp(self.p2s, _d_).labelDict(), _d_)
        self.assertEqual(_lp(self.p2s, [self.p2s.NODE_IDp, _d_, ('sip', 'sdns')]).labelDict(), _d_)
        self.assertIsNone(_lp(self.p2s, [self.p2s.NODE_IDp]).labelDict())

    def test_selected_labels_overlay_carries_every_line(self):
        _lp_ = _lp(self.p2s, [self.p2s.NODE_IDp, ('sip', 'sdns')])
        _out_ = _lp_.__createSelectedLabels__({'1.2.3.4', '8.8.8.8'})
        _got_ = sorted(tuple(_l_['lines']) for _l_ in _out_['labels'])
        self.assertEqual(_got_, [('1.2.3.4', 'home'), ('8.8.8.8',)])


class TestLabelLineColorAndSize(unittest.TestCase):
    def setUp(self):
        self.p2s = Polars2SVG()

    def _fmt_(self, lp):
        '''{first line: (sizes, fills)} for each drawn label.'''
        return {_l_[0]: _f_ for (_, _, _l_), _f_ in zip(lp._node_label_info_, lp._node_label_fmt_)}

    def test_no_options_is_the_markup_it_always_was(self):
        _svg_ = _lp(self.p2s, [self.p2s.NODE_IDp, ('sip', 'sdns')]).svg
        self.assertNotIn('fill="#', ''.join(_t_ for _t_ in _svg_.split('<text')[1:]))
        self.assertNotIn('<tspan x="15" dy="12" font-size', _svg_)

    def test_hex_colours_the_whole_line_and_only_that_line(self):
        _lp_ = _lp(self.p2s, [self.p2s.NODE_IDp, self.p2s.labelLine(('sip', 'sdns'), color='#ff0000')])
        _sizes_, _fills_ = self._fmt_(_lp_)['1.2.3.4']
        self.assertEqual(_fills_, [None, '#ff0000'])
        self.assertIn('<tspan x="15" dy="12" fill="#ff0000">home</tspan>', _lp_.svg)

    def test_color_by_text_is_one_colour_per_value_on_every_node(self):
        _lp_ = _lp(self.p2s, [self.p2s.labelLine([('sip', 'sdns'), ('dip', 'ddns')], color=self.p2s.COLOR_BY_TEXT)])
        _by_ = {_l_[0]: _f_[1][0] for (_, _, _l_), _f_ in zip(_lp_._node_label_info_, _lp_._node_label_fmt_)}
        self.assertEqual(_by_['googledns'], self.p2s.color('googledns'))
        self.assertEqual(_by_['home'], self.p2s.color('home'))
        self.assertNotEqual(_by_['home'], _by_['googledns'])

    def test_color_by_text_hashes_the_value_not_the_prefix_or_the_count(self):
        _lp_ = _lp(self.p2s, [self.p2s.labelLine(('sip', ('protocol', 'port')), name='pp',
                                                 color=self.p2s.COLOR_BY_TEXT)])
        _by_ = {_l_[0]: _f_[1][0] for (_, _, _l_), _f_ in zip(_lp_._node_label_info_, _lp_._node_label_fmt_)}
        self.assertEqual(_by_['pp=udp|53 (+1)'], self.p2s.color('udp|53'))   # 1.2.3.4
        self.assertEqual(_by_['pp=udp|53'],      self.p2s.color('udp|53'))   # 5.6.7.8

    def test_a_node_id_line_agrees_with_node_color_by_name(self):
        _lp_ = _lp(self.p2s, [self.p2s.labelLine(self.p2s.NODE_IDp, color=self.p2s.COLOR_BY_TEXT)],
                   node_color=self.p2s.COLOR_BY_NODE_NAME)
        _fills_ = {_l_[0]: _f_[1][0] for (_, _, _l_), _f_ in zip(_lp_._node_label_info_, _lp_._node_label_fmt_)}
        for _n_, _hx_ in _lp_.color_nodes_final.items():
            self.assertEqual(_fills_[_n_], _hx_)

    def test_size_moves_the_baselines_by_each_lines_own_size(self):
        _lp_ = _lp(self.p2s, [self.p2s.labelLine(self.p2s.NODE_IDp, size=20), ('sip', 'sdns'),
                              self.p2s.labelLine({'1.2.3.4': 'tiny'}, size=6)])
        _svg_ = _lp_.svg
        self.assertIn('<tspan x="15" dy="0" font-size="20px">1.2.3.4</tspan>', _svg_)
        self.assertIn('<tspan x="15" dy="12">home</tspan>', _svg_)                       # inherits txt_h
        self.assertIn('<tspan x="15" dy="6" font-size="6px">tiny</tspan>', _svg_)
        _, _y0_, _ = next(_i_ for _i_ in _lp_._node_label_info_ if _i_[2][0] == '1.2.3.4')
        _plain_ = _lp(self.p2s, [self.p2s.NODE_IDp])
        _, _y0_plain_, _ = next(_i_ for _i_ in _plain_._node_label_info_ if _i_[2][0] == '1.2.3.4')
        self.assertEqual(_y0_ - _y0_plain_, 20 - 12)        # first baseline: edge + the FIRST line's size
        self.assertEqual([_s_ for _s_ in self._fmt_(_lp_)['1.2.3.4'][0]], [20.0, 12.0, 6.0])

    def test_txt_h_is_the_default_size_and_a_line_names_only_its_exception(self):
        _lp_ = _lp(self.p2s, [self.p2s.labelLine(self.p2s.NODE_IDp, size=10), ('sip', 'sdns')], txt_h=6)
        self.assertEqual(self._fmt_(_lp_)['1.2.3.4'][0], [10.0, 6.0])

    def test_culling_measures_each_line_at_its_own_size(self):
        # 'aaa' sits ~50px left of the canvas: its 12px label ends off-canvas, its 60px one reaches in
        _df_  = pl.DataFrame({'fm': ['aaa'], 'to': ['bbb']})
        _pos_ = {'aaa': (0.0, 0.5), 'bbb': (1.0, 0.5)}
        def _n_(size):
            _lp_ = self.p2s.linkp(df=_df_, relationships=[('fm', 'to')], pos=_pos_, wxh=(200, 200), insets=(0, 0),
                                  draw_node_labels=True, view_window=(0.2, 0.0, 1.0, 1.0),
                                  node_labels=[self.p2s.labelLine(self.p2s.NODE_IDp, size=size)])
            return sorted(_l_[0] for _, _, _l_ in _lp_._node_label_info_)
        self.assertEqual(_n_(12), ['bbb'])
        self.assertEqual(_n_(60), ['aaa', 'bbb'])

    def test_the_gpu_text_ops_carry_each_lines_size_and_colour(self):
        _lp_ = _lp(self.p2s, [self.p2s.labelLine(self.p2s.NODE_IDp, size=9, color='#00ff00'), ('sip', 'sdns')])
        _ops_ = [_o_[1] for _o_ in _lp_.gpuDisplayList()._ops_ if _o_[0] == 'text' and _o_[1][0] in ('1.2.3.4', 'home')]
        _by_ = {_o_[0]: _o_ for _o_ in _ops_}
        self.assertEqual(_by_['1.2.3.4'][3], 9.0)
        self.assertEqual(_by_['1.2.3.4'][6][:3], (0.0, 1.0, 0.0))
        self.assertEqual(_by_['home'][3], 12.0)
        self.assertAlmostEqual(_by_['home'][2] - _by_['1.2.3.4'][2], 12.0)    # one line of the second line's size

    def test_label_line_rejects_a_bad_color_or_size(self):
        for _bad_ in ('red', '#12', 5, True):
            with self.assertRaisesRegex(TypeError, 'color must be'):
                self.p2s.labelLine({}, color=_bad_)
        for _bad_ in (0, -3, True, 'big', float('inf')):
            with self.assertRaisesRegex(TypeError, 'size must be a positive number'):
                self.p2s.labelLine({}, size=_bad_)


_BOOK_ = ('this is a really long string [1]\n'
          'and goes on and on and on [2]\n'
          '\n'
          'until it ends here [3]')


class TestLabelLineWrap(unittest.TestCase):
    def setUp(self):
        self.p2s = Polars2SVG()
        self.df = pl.DataFrame({'text_id': [1], 'book': ['a book title']})
        self.rel = [('book', 'text_id')]
        self.pos = {'a book title': (0.2, 0.2), '1': (0.8, 0.8)}

    def _lp_(self, labels, **kw):
        return self.p2s.linkp(self.df, self.rel, pos=self.pos, wxh=(384, 384), draw_node_labels=True,
                              node_labels=labels, bounds_percent=0.5, **kw)

    def _by_(self, lp):
        return {_l_[0]: _l_ for _, _, _l_ in lp._node_label_info_}

    def test_an_unwrapped_line_is_one_row_and_a_newline_is_a_space(self):
        _by_ = self._by_(self._lp_([{1: _BOOK_}], label_line_width=200))
        _rows_ = list(_by_.values())[0]
        self.assertEqual(len(_rows_), 1)
        self.assertEqual(_rows_[0], 'this is a really long string [1] and goes on and on and on [2] until it ends here [3]')

    def test_wrap_breaks_at_the_width_and_at_every_newline(self):
        _rows_ = list(self._by_(self._lp_([p2s_line(self, {1: _BOOK_})], label_line_width=24, label_max_lines=20)).values())[0]
        self.assertEqual(_rows_, ['this is a really long', 'string [1]', 'and goes on and on and', 'on [2]',
                                  'until it ends here [3]'])      # the blank paragraph leaves no empty row

    def test_the_row_cap_is_label_max_lines_and_ends_in_an_ellipsis(self):
        _rows_ = list(self._by_(self._lp_([p2s_line(self, {1: _BOOK_})], label_line_width=24, label_max_lines=3)).values())[0]
        self.assertEqual(len(_rows_), 3)
        self.assertTrue(_rows_[-1].endswith('…'))
        _rows_ = list(self._by_(self._lp_([p2s_line(self, {1: _BOOK_})], label_line_width=24,
                                          label_max_lines=3, label_ellipsis=False)).values())[0]
        self.assertFalse(_rows_[-1].endswith('…'))

    def test_the_cap_is_per_line_not_for_the_whole_label(self):
        _rows_ = list(self._by_(self._lp_([self.p2s.NODE_IDp, p2s_line(self, {1: _BOOK_}), p2s_line(self, {1: _BOOK_})],
                                         label_line_width=24, label_max_lines=2)).values())
        _two_ = [_r_ for _r_ in _rows_ if len(_r_) > 1][0]
        self.assertEqual(len(_two_), 1 + 2 + 2)

    def test_wrapped_rows_share_the_lines_size_and_colour(self):
        _lp_ = self._lp_([self.p2s.labelLine({1: _BOOK_}, wrap=True, size=8, color='#123456')],
                         label_line_width=24, label_max_lines=3)
        _sizes_, _fills_ = _lp_._node_label_fmt_[0]
        self.assertEqual((_sizes_, _fills_), ([8.0] * 3, ['#123456'] * 3))
        self.assertIn('dy="8"', _lp_.svg)

    def test_the_count_suffix_is_wrapped_with_the_value(self):
        _df_ = pl.DataFrame({'a': ['n', 'n', 'n'], 'b': ['m', 'm', 'm'], 'v': ['one two three four', 'one two three four', 'other']})
        _lp_ = self.p2s.linkp(_df_, [('a', 'b')], pos={'n': (0.2, 0.5), 'm': (0.8, 0.5)}, wxh=(300, 300),
                              draw_node_labels=True, label_line_width=12,
                              node_labels=[self.p2s.labelLine(('a', 'v'), wrap=True)])
        self.assertEqual(self._by_(_lp_)['one two'], ['one two', 'three four', '(+1)'])

    def test_a_wrapped_dict_line_is_what_a_bare_dict_always_was(self):
        _m_ = {1: 'a very long description of a company webserver, wrapped', '1': 'x'}
        _bare_    = self._lp_({1: _m_[1]}, label_line_width=16)
        _as_line_ = self._lp_([self.p2s.labelLine({1: _m_[1]}, wrap=True)], label_line_width=16)
        self.assertEqual(_as_line_.svg, _bare_.svg)

    def test_the_selected_nodes_overlay_wraps_without_a_cap(self):
        _lp_ = self._lp_([self.p2s.NODE_IDp, self.p2s.labelLine({1: _BOOK_}, wrap=True)], label_line_width=24,
                         label_max_lines=2)
        _got_ = [_l_['lines'] for _l_ in _lp_.__createSelectedLabels__({'1'})['labels']][0]
        self.assertEqual(len(_got_), 1 + 5)

    def test_wrap_has_to_be_a_bool(self):
        for _bad_ in (1, 'yes', None):
            with self.assertRaisesRegex(TypeError, 'wrap must be True or False'):
                self.p2s.labelLine({}, wrap=_bad_)


def p2s_line(case, source):
    return case.p2s.labelLine(source, wrap=True)


class TestLabelLineValidation(unittest.TestCase):
    def setUp(self):
        self.p2s = Polars2SVG()

    def test_a_bare_column_must_name_its_node(self):
        with self.assertRaisesRegex(ValueError, r"write \(node_column, 'sdns'\)"):
            _lp(self.p2s, ['sdns'])

    def test_the_key_must_be_a_relationship_endpoint(self):
        with self.assertRaisesRegex(ValueError, 'cannot name a node'):
            _lp(self.p2s, [('protocol', 'sdns')])

    def test_a_missing_column_is_an_error_at_construction(self):
        with self.assertRaisesRegex(ValueError, r"\['nope'\] not in the DataFrame"):
            _lp(self.p2s, [('sip', 'nope')])

    def test_a_tuple_key_must_match_the_tuple_endpoint(self):
        with self.assertRaisesRegex(ValueError, 'cannot name a node'):
            _lp(self.p2s, [('sip', 'sdns')], df=_DF_PORTS_, rel=_REL_PORTS_, pos=_POS_PORTS_)

    def test_not_a_dict_or_list(self):
        with self.assertRaisesRegex(TypeError, 'dict or a list of label lines'):
            _lp(self.p2s, 5)
        with self.assertRaisesRegex(TypeError, 'dict or a list of label lines'):
            _lp(self.p2s, ('sip', 'sdns'))

    def test_an_empty_list_and_a_foreign_element(self):
        with self.assertRaisesRegex(ValueError, 'empty list'):
            _lp(self.p2s, [])
        with self.assertRaisesRegex(TypeError, 'not a label line'):
            _lp(self.p2s, [42])

    def test_two_lines_may_not_share_a_name(self):
        with self.assertRaisesRegex(ValueError, "two lines named 'x'"):
            _lp(self.p2s, [self.p2s.labelLine({}, name='x'), self.p2s.labelLine({}, name='x')])

    def test_label_line_constructor_rejects_bad_names(self):
        with self.assertRaises(TypeError): self.p2s.labelLine({}, name='')
        with self.assertRaises(TypeError): self.p2s.labelLine(self.p2s.labelLine({}))

    def test_a_bool_is_still_rejected_as_the_flag_confusion(self):
        with self.assertRaises(TypeError):
            _lp(self.p2s, True)


class TestLabelLineGolden(unittest.TestCase):
    '''First run needs UPDATE_GOLDEN=1 to write the golden files.'''
    def setUp(self):
        self.p2s = Polars2SVG()

    def test_five_lines(self):
        _lp_ = _lp(self.p2s, [self.p2s.NODE_IDp,
                              {'8.8.8.8': 'benign', '23.12.31.21': 'compromised'},
                              [('sip', 'sdns'), ('dip', 'ddns')],
                              self.p2s.labelLine({'8.8.8.8': 'up', '1.2.3.4': 'down'}, name='status')])
        assert_svg_matches_golden(_lp_.svg, 'linkp_label_lines_five')

    def test_composite_key(self):
        _lp_ = _lp(self.p2s, [self.p2s.NODE_IDp, [(('sip', 'sport'), 'sdns'), (('dip', 'dport'), 'ddns')]],
                   df=_DF_PORTS_, rel=_REL_PORTS_, pos=_POS_PORTS_)
        assert_svg_matches_golden(_lp_.svg, 'linkp_label_lines_composite_key')

    def test_colour_and_size(self):
        _lp_ = _lp(self.p2s, [self.p2s.labelLine(self.p2s.NODE_IDp, size=14, color=self.p2s.COLOR_BY_TEXT),
                              self.p2s.labelLine([('sip', 'sdns'), ('dip', 'ddns')], size=9, color=self.p2s.COLOR_BY_TEXT),
                              self.p2s.labelLine({'8.8.8.8': 'benign', '23.12.31.21': 'compromised'}, size=9,
                                                 color='#c0392b')])
        assert_svg_matches_golden(_lp_.svg, 'linkp_label_lines_colour_and_size')

    def test_composite_value(self):
        _lp_ = _lp(self.p2s, [self.p2s.NODE_IDp, ('sip', ('protocol', 'port'))])
        assert_svg_matches_golden(_lp_.svg, 'linkp_label_lines_composite_value')

    def test_wrapped_line(self):
        _df_ = pl.DataFrame({'text_id': [1], 'book': ['a book title']})
        _lp_ = self.p2s.linkp(_df_, [('book', 'text_id')], pos={'a book title': (0.5, 0.2), '1': (0.5, 0.7)},
                              wxh=(300, 300), draw_node_labels=True, label_line_width=24, label_max_lines=4,
                              node_labels=[self.p2s.NODE_IDp, self.p2s.labelLine({1: _BOOK_}, wrap=True, size=9,
                                                                                 color=self.p2s.COLOR_BY_TEXT)])
        assert_svg_matches_golden(_lp_.svg, 'linkp_label_lines_wrapped')

    def test_separate_lines_for_separate_facts(self):
        _df_ = _DF_.with_columns(pl.lit('ACME').alias('sorg'), pl.lit('US').alias('scountry'))
        _lp_ = _lp(self.p2s, [self.p2s.NODE_IDp, ('sip', 'sdns'), ('sip', 'sorg'), ('sip', 'scountry')], df=_df_)
        assert_svg_matches_golden(_lp_.svg, 'linkp_label_lines_separate_columns')


if __name__ == '__main__':
    unittest.main()
