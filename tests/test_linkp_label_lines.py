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

    def test_label_max_lines_does_not_cap_a_list(self):
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

    def test_composite_value(self):
        _lp_ = _lp(self.p2s, [self.p2s.NODE_IDp, ('sip', ('protocol', 'port'))])
        assert_svg_matches_golden(_lp_.svg, 'linkp_label_lines_composite_value')

    def test_separate_lines_for_separate_facts(self):
        _df_ = _DF_.with_columns(pl.lit('ACME').alias('sorg'), pl.lit('US').alias('scountry'))
        _lp_ = _lp(self.p2s, [self.p2s.NODE_IDp, ('sip', 'sdns'), ('sip', 'sorg'), ('sip', 'scountry')], df=_df_)
        assert_svg_matches_golden(_lp_.svg, 'linkp_label_lines_separate_columns')


if __name__ == '__main__':
    unittest.main()
