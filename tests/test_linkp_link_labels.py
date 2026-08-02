#
# test_linkp_link_labels.py
#
# draw_link_labels= draws the third element of a relationship tuple -- or, for a two-part
# tuple, the field driving the link color -- onto the edge itself (rtsvg
# rt_linknode_mixin.py:1413-1425 parity, plus two behaviors rtsvg did not have):
#
#   - a bidirectional pair is labeled twice, once per direction, on opposite sides of
#     the shared edge, so fm->to and to->fm never overprint each other
#   - link_shape='curve' runs the label along the drawn Bezier via <textPath>;
#     'flowmap' is not labeled at all
#
# The '*' collision marker is rtsvg parity and deliberate: an edge whose rows carry
# different values still gets a label, because drawing nothing there would read as an
# edge with no data.
#
import logging
import math
import re
import unittest
import xml.etree.ElementTree as ET

import polars as pl

from polars2svg import Polars2SVG


# a<->b is bidirectional (both directions present), b->c and a->c are one-way.
_DF_ = pl.DataFrame({
    'fm':  ['a',     'b',       'b',     'a'],
    'to':  ['b',     'a',       'c',     'c'],
    'dsc': ['calls', 'answers', 'pings', 'emails'],
    'grp': ['x',     'x',       'y',     'y'],
})
_POS_ = {'a': (0.0, 0.0), 'b': (1.0, 0.05), 'c': (0.45, 1.0)}
_WXH_ = (520, 420)

_SVG_NS_ = '{http://www.w3.org/2000/svg}'


def _texts(svg):
    '''Every rendered <text> as (content, attrib-dict); textPath content is inlined.'''
    _root_ = ET.fromstring(svg)
    _out_  = []
    for _t_ in _root_.iter(f'{_SVG_NS_}text'):
        _tp_ = _t_.find(f'{_SVG_NS_}textPath')
        _out_.append((''.join(_t_.itertext()), dict(_t_.attrib),
                      dict(_tp_.attrib) if _tp_ is not None else None))
    return _out_


def _label_texts(svg):
    return [t[0] for t in _texts(svg)]


def _rotation(attrib):
    _m_ = re.match(r'rotate\(([-\d.]+)', attrib.get('transform', ''))
    return None if _m_ is None else float(_m_.group(1))


class _LinkLabelTestBase_(unittest.TestCase):

    def setUp(self):
        self.p2s = Polars2SVG()

    def linkp(self, df=_DF_, rels=(('fm', 'to', 'dsc'),), **kwargs):
        _kw_ = dict(pos=_POS_, wxh=_WXH_, draw_link_labels=True)
        _kw_.update(kwargs)
        return self.p2s.linkp(df, relationships=[tuple(r) for r in rels], **_kw_)


# ---------------------------------------------------------------------------
# The label itself
# ---------------------------------------------------------------------------

class TestLinkLabelBasics(_LinkLabelTestBase_):

    def test_off_by_default(self):
        _lp_ = self.p2s.linkp(_DF_, relationships=[('fm', 'to', 'dsc')], pos=_POS_, wxh=_WXH_)
        self.assertEqual(_label_texts(_lp_.svg), [])
        self.assertEqual(_lp_._link_label_svg_, [])

    def test_third_tuple_element_is_drawn(self):
        _lp_ = self.linkp()
        self.assertEqual(sorted(_label_texts(_lp_.svg)),
                         ['answers', 'calls', 'emails', 'pings'])

    def test_independent_of_draw_node_labels(self):
        '''draw_node_labels governs node labels; draw_link_labels governs edge labels. Neither
        implies the other (rtsvg rt_linknode_mixin.py:1413 does not test draw_node_labels).'''
        _off_ = self.linkp(draw_node_labels=False)
        self.assertIn('calls', _label_texts(_off_.svg))
        _on_  = self.linkp(draw_node_labels=True)
        _txt_ = _label_texts(_on_.svg)
        self.assertIn('calls', _txt_)     # edge label
        self.assertIn('a', _txt_)         # node label

    def test_conflicting_values_collapse_to_star(self):
        '''Two rows on one edge disagreeing -> '*', never a missing label.'''
        _df_ = pl.DataFrame({'fm': ['a', 'a'], 'to': ['b', 'b'], 'dsc': ['one', 'two']})
        _lp_ = self.linkp(df=_df_)
        self.assertEqual(_label_texts(_lp_.svg), ['*'])

    def test_repeated_identical_values_do_not_collapse(self):
        _df_ = pl.DataFrame({'fm': ['a', 'a'], 'to': ['b', 'b'], 'dsc': ['one', 'one']})
        self.assertEqual(_label_texts(self.linkp(df=_df_).svg), ['one'])

    def test_null_label_values_are_skipped(self):
        _df_ = pl.DataFrame({'fm': ['a', 'b'], 'to': ['b', 'c'], 'dsc': [None, 'pings']},
                            schema={'fm': pl.String, 'to': pl.String, 'dsc': pl.String})
        self.assertEqual(_label_texts(self.linkp(df=_df_).svg), ['pings'])

    def test_label_text_is_xml_escaped(self):
        _df_ = pl.DataFrame({'fm': ['a'], 'to': ['b'], 'dsc': ['<a & b>']})
        _lp_ = self.linkp(df=_df_)
        self.assertIn('&lt;a &amp; b&gt;', _lp_.svg)
        self.assertEqual(_label_texts(_lp_.svg), ['<a & b>'])   # parses back true to form

    def test_short_edge_drops_the_label_rather_than_drawing_a_bare_ellipsis(self):
        '''An edge with no room for even one character is left unlabeled instead of
        carrying a meaningless "...".'''
        _df_ = pl.DataFrame({'fm': ['a', 'a'], 'to': ['b', 'c'],
                             'dsc': ['a rather long label', 'x']})
        _lp_ = self.p2s.linkp(_df_, relationships=[('fm', 'to', 'dsc')],
                              pos={'a': (0.0, 0.0), 'b': (0.02, 0.0), 'c': (1.0, 1.0)},
                              wxh=(200, 200), draw_link_labels=True)
        self.assertEqual(_label_texts(_lp_.svg), ['x'])   # a->b is ~4px; a->c is labeled

    def test_render_is_deterministic(self):
        '''group_by output order is not stable and the <textPath> ids are handed out in
        iteration order, so the emission is sorted -- two renders must agree.'''
        for _shape_ in ('line', 'curve'):
            _a_ = self.linkp(link_shape=_shape_).svg
            _b_ = self.linkp(link_shape=_shape_).svg
            # ids carry a per-instance random scope; compare with them normalized out
            _norm_ = lambda s: re.sub(r'p2sll\d+_', 'ID_', s)
            self.assertEqual(_norm_(_a_), _norm_(_b_), f'{_shape_} render not deterministic')


# ---------------------------------------------------------------------------
# label_only gates both channels; link_labels renames what survives
# ---------------------------------------------------------------------------

class TestLabelOnly(_LinkLabelTestBase_):
    '''label_only holds NAMES, and both channels test their own against it: a node by
    its node name, an edge by its label value (rtsvg rt_linknode_mixin.py:1419-1422).'''

    def test_restricts_edge_labels_to_the_set(self):
        _lp_ = self.linkp(label_only={'calls', 'emails'})
        self.assertEqual(sorted(_label_texts(_lp_.svg)), ['calls', 'emails'])

    def test_empty_set_labels_everything(self):
        self.assertEqual(len(_label_texts(self.linkp(label_only=set()).svg)), 4)

    def test_gates_both_channels_from_one_set(self):
        '''One set, two kinds of name: 'a' is a node, 'calls' is an edge value.'''
        _lp_ = self.linkp(draw_node_labels=True, label_only={'a', 'calls'})
        self.assertEqual(sorted(_label_texts(_lp_.svg)), ['a', 'calls'])

    def test_node_only_set_silences_edge_labels(self):
        '''The flip side of sharing the set: naming only nodes leaves no edge labeled.'''
        _lp_ = self.linkp(draw_node_labels=True, label_only={'a', 'b'})
        self.assertEqual(sorted(_label_texts(_lp_.svg)), ['a', 'b'])

    def test_star_survives_when_a_value_behind_it_was_asked_for(self):
        '''b->c collides ('pings' + 'acks'); asking for either keeps the '*' visible.'''
        _df_ = _DF_.with_columns(pl.Series('dsc', ['calls', 'answers', 'pings', 'emails']))
        _df_ = pl.concat([_df_, pl.DataFrame({'fm': ['b'], 'to': ['c'], 'dsc': ['acks'],
                                              'grp': ['y']})])
        _lp_ = self.p2s.linkp(_df_, relationships=[('fm', 'to', 'dsc')], pos=_POS_,
                              wxh=_WXH_, draw_link_labels=True, label_only={'pings'})
        self.assertEqual(_label_texts(_lp_.svg), ['*'])

    def test_star_is_dropped_when_no_value_behind_it_was_asked_for(self):
        _df_ = pl.DataFrame({'fm': ['a', 'a', 'b'], 'to': ['b', 'b', 'c'],
                             'dsc': ['one', 'two', 'emails']})
        _lp_ = self.p2s.linkp(_df_, relationships=[('fm', 'to', 'dsc')], pos=_POS_,
                              wxh=_WXH_, draw_link_labels=True, label_only={'emails'})
        self.assertEqual(_label_texts(_lp_.svg), ['emails'])

    def test_accepts_a_bare_string(self):
        self.assertEqual(_label_texts(self.linkp(label_only='calls').svg), ['calls'])


class TestLinkLabelsDict(_LinkLabelTestBase_):
    '''link_labels is node_labels for the link channel: it renames values for display,
    and an edge whose value it does not name goes unlabeled.'''

    def test_renames_the_displayed_value(self):
        _lp_ = self.linkp(link_labels={'calls': 'telephoned', 'answers': 'picked up',
                                       'pings': 'p', 'emails': 'e'})
        self.assertEqual(sorted(_label_texts(_lp_.svg)),
                         ['e', 'p', 'picked up', 'telephoned'])

    def test_unnamed_values_go_unlabeled(self):
        _lp_ = self.linkp(link_labels={'calls': 'telephoned'})
        self.assertEqual(_label_texts(_lp_.svg), ['telephoned'])

    def test_empty_dict_is_a_no_op(self):
        self.assertEqual(len(_label_texts(self.linkp(link_labels={}).svg)), 4)

    def test_applies_after_label_only(self):
        '''label_only tests the raw value; link_labels renames what survives -- so a
        display name is never what label_only matches against.'''
        _lp_ = self.linkp(label_only={'calls'}, link_labels={'calls': 'telephoned',
                                                             'emails': 'mailed'})
        self.assertEqual(_label_texts(_lp_.svg), ['telephoned'])

    def test_display_name_is_not_a_label_only_key(self):
        _lp_ = self.linkp(label_only={'telephoned'}, link_labels={'calls': 'telephoned'})
        self.assertEqual(_label_texts(_lp_.svg), [])

    def test_can_rename_the_star(self):
        '''The collision marker is a value like any other, so it can be renamed.'''
        _df_ = pl.DataFrame({'fm': ['a', 'a'], 'to': ['b', 'b'], 'dsc': ['one', 'two']})
        _lp_ = self.linkp(df=_df_, link_labels={'*': 'mixed'})
        self.assertEqual(_label_texts(_lp_.svg), ['mixed'])

    def test_display_name_is_xml_escaped(self):
        _lp_ = self.linkp(link_labels={'calls': '<a & b>'})
        self.assertIn('&lt;a &amp; b&gt;', _lp_.svg)
        self.assertEqual(_label_texts(_lp_.svg), ['<a & b>'])

    def test_display_name_reaches_the_gpu(self):
        _lp_ = self.linkp(link_labels={'calls': 'telephoned'})
        self.assertEqual([e[3] for e in _lp_._link_label_info_], ['telephoned'])

    def test_node_labels_and_link_labels_are_independent(self):
        _lp_ = self.linkp(draw_node_labels=True, node_labels={'a': 'Alice'},
                          link_labels={'calls': 'telephoned'})
        self.assertEqual(sorted(_label_texts(_lp_.svg)), ['Alice', 'telephoned'])


# ---------------------------------------------------------------------------
# Which field supplies the label, and what color it takes
# ---------------------------------------------------------------------------

class TestLinkLabelField(_LinkLabelTestBase_):

    def test_two_part_tuple_falls_back_to_the_link_color_field(self):
        '''rtsvg rt_linknode_mixin.py:1414 -- with no third element the label is the
        color_by value; here that is whatever field drives color=.'''
        _lp_ = self.linkp(rels=(('fm', 'to'),), color='grp')
        self.assertEqual(sorted(set(_label_texts(_lp_.svg))), ['x', 'y'])

    def test_label_in_link_color_when_the_fields_match(self):
        _lp_ = self.linkp(rels=(('fm', 'to'),), color='grp')
        _fills_ = {t[0]: t[1]['fill'] for t in _texts(_lp_.svg)}
        self.assertNotEqual(_fills_['x'], _fills_['y'])
        self.assertNotIn('#000000', set(_fills_.values()))

    def test_label_in_default_foreground_when_the_fields_differ(self):
        _lp_ = self.linkp(rels=(('fm', 'to', 'dsc'),), color='grp')
        _fills_ = {t[1]['fill'] for t in _texts(_lp_.svg)}
        self.assertEqual(_fills_, {self.p2s.colorTyped('label', 'defaultfg')})

    def test_third_element_wins_over_the_color_field(self):
        _lp_ = self.linkp(rels=(('fm', 'to', 'dsc'),), color='grp')
        self.assertIn('calls', _label_texts(_lp_.svg))
        self.assertNotIn('x', _label_texts(_lp_.svg))

    def test_multiple_relationships_are_each_labeled_by_their_own_field(self):
        _df_ = pl.DataFrame({'fm': ['a'], 'to': ['b'], 'x': ['c'], 'l1': ['one'], 'l2': ['two']})
        _lp_ = self.p2s.linkp(_df_, relationships=[('fm', 'to', 'l1'), ('fm', 'x', 'l2')],
                              pos={'a': (0, 0), 'b': (1, 0), 'c': (0.5, 1)},
                              wxh=_WXH_, draw_link_labels=True)
        self.assertEqual(sorted(_label_texts(_lp_.svg)), ['one', 'two'])


# ---------------------------------------------------------------------------
# Placement: two sides for a bidirectional pair, upright text
# ---------------------------------------------------------------------------

class TestLinkLabelPlacement(_LinkLabelTestBase_):

    def _horizontal(self, **kwargs):
        '''a<->b on a horizontal edge, so "above"/"below" is just smaller/larger y.'''
        _df_ = pl.DataFrame({'fm': ['a', 'b'], 'to': ['b', 'a'], 'dsc': ['calls', 'answers']})
        return self.p2s.linkp(_df_, relationships=[('fm', 'to', 'dsc')],
                              pos={'a': (0.0, 0.0), 'b': (1.0, 0.0)},
                              wxh=(400, 200), draw_link_labels=True, **kwargs)

    def test_bidirectional_pair_gets_two_labels(self):
        self.assertEqual(sorted(_label_texts(self._horizontal().svg)), ['answers', 'calls'])

    def test_bidirectional_labels_land_on_opposite_sides(self):
        _lp_ = self._horizontal()
        _ys_ = {t[0]: float(t[1]['y']) for t in _texts(_lp_.svg)}
        # the edge's own y: both endpoints sit on one horizontal line
        _edge_y_ = float(re.search(r'<line x1="[\d.]+" y1="([\d.]+)"', _lp_.svg).group(1))
        self.assertLess(_ys_['calls'], _edge_y_, 'fm<to label should sit above the edge')
        self.assertGreater(_ys_['answers'], _edge_y_, 'to<fm label should sit below the edge')

    def test_bidirectional_labels_clear_the_edge_by_the_same_margin(self):
        '''Each side is placed so its *ink* clears the edge, not its baseline: the label
        whose glyphs grow back over the edge is pushed out by its ascent, the other only
        by its descent. Both therefore end up the same distance off the edge.'''
        _lp_ = self._horizontal(txt_h=14)
        _ys_ = {t[0]: float(t[1]['y']) for t in _texts(_lp_.svg)}
        _edge_y_ = float(re.search(r'<line x1="[\d.]+" y1="([\d.]+)"', _lp_.svg).group(1))
        # 'calls' sits above and grows away, so its descent is what reaches back down
        _above_ = _edge_y_ - _ys_['calls'] - _lp_._labelInk_('calls')[1]
        # 'answers' sits below and grows back up, so its ascent is what reaches the edge
        _below_ = _ys_['answers'] - _lp_._labelInk_('answers')[0] - _edge_y_
        self.assertAlmostEqual(_above_, _below_, places=6)
        self.assertAlmostEqual(_above_, 2.5, places=6)   # rtsvg's 2 + stroke/2, stroke=1

    def test_clearance_is_the_same_whatever_the_string_is_made_of(self):
        '''Regression: with the offset anchored on a single text height, 'cow' (no
        ascender at all) sat ~3px farther off its edge than 'CAT2', and 'dog' hung its
        descender onto the edge on the other side.'''
        for _s_ in ('cow', 'dog', 'CAT2', 'cat', 'pygmy', '12.5'):
            _df_ = pl.DataFrame({'fm': ['a', 'b'], 'to': ['b', 'a'], 'dsc': [_s_, _s_]})
            _lp_ = self.p2s.linkp(_df_, relationships=[('fm', 'to', 'dsc')],
                                  pos={'a': (0.0, 0.0), 'b': (1.0, 0.0)}, wxh=(400, 200),
                                  draw_link_labels=True)
            _ys_ = sorted(float(t[1]['y']) for t in _texts(_lp_.svg))
            _edge_y_ = float(re.search(r'<line x1="[\d.]+" y1="([\d.]+)"', _lp_.svg).group(1))
            _asc_, _desc_ = _lp_._labelInk_(_s_)
            self.assertAlmostEqual(_edge_y_ - _ys_[0] - _desc_, 2.5, places=6, msg=f'above {_s_}')
            self.assertAlmostEqual(_ys_[1] - _asc_ - _edge_y_, 2.5, places=6, msg=f'below {_s_}')

    def test_one_way_edge_gets_exactly_one_label(self):
        _df_ = pl.DataFrame({'fm': ['a'], 'to': ['b'], 'dsc': ['calls']})
        self.assertEqual(_label_texts(self.linkp(df=_df_).svg), ['calls'])

    def test_side_follows_direction_not_row_order(self):
        '''The side is fixed by the node names, so swapping which direction appears
        first in the frame must not move either label.'''
        _fwd_ = pl.DataFrame({'fm': ['a', 'b'], 'to': ['b', 'a'], 'dsc': ['calls', 'answers']})
        _rev_ = _fwd_.reverse()
        _y_ = lambda df: {t[0]: t[1]['y'] for t in _texts(
            self.p2s.linkp(df, relationships=[('fm', 'to', 'dsc')],
                           pos={'a': (0.0, 0.0), 'b': (1.0, 0.0)},
                           wxh=(400, 200), draw_link_labels=True).svg)}
        self.assertEqual(_y_(_fwd_), _y_(_rev_))

    def test_text_is_never_upside_down(self):
        '''Rotations stay within +/-90 degrees, whichever way the edge runs.'''
        _rots_ = [_rotation(t[1]) for t in _texts(self.linkp().svg)]
        self.assertEqual(len(_rots_), 4)
        for _r_ in _rots_:
            self.assertIsNotNone(_r_)
            self.assertLessEqual(abs(_r_), 90.0 + 1e-6)

    def test_label_sits_on_the_edge_it_belongs_to(self):
        '''Each label's anchor is within a text height of its own edge's midpoint.'''
        _lp_ = self.linkp()
        _mids_ = {}
        for _fm_, _to_ in (('a', 'b'), ('b', 'c'), ('a', 'c')):
            _r_ = _lp_.df_node.filter(pl.col('__first__').is_in([_fm_, _to_]))
            _mids_[(_fm_, _to_)] = (_r_['__sx__'].mean(), _r_['__sy__'].mean())
        for _txt_, _attr_, _ in _texts(_lp_.svg):
            _pt_ = (float(_attr_['x']), float(_attr_['y']))
            _near_ = min(math.dist(_pt_, _m_) for _m_ in _mids_.values())
            self.assertLess(_near_, 2 * _lp_.txt_h, f'{_txt_} is not on any edge midpoint')

    def test_stroke_width_widens_the_clearance(self):
        _thin_ = self._horizontal(link_size=1)
        _fat_  = self._horizontal(link_size=9)
        _y_ = lambda lp: {t[0]: float(t[1]['y']) for t in _texts(lp.svg)}
        _edge_y_ = float(re.search(r'<line x1="[\d.]+" y1="([\d.]+)"', _thin_.svg).group(1))
        self.assertGreater(_edge_y_ - _y_(_fat_)['calls'], _edge_y_ - _y_(_thin_)['calls'])


class TestLabelInk(_LinkLabelTestBase_):
    '''_labelInk_ approximates how far a run's glyphs reach from their baseline. It is
    what makes the clearance uniform, so its ordering has to hold.'''

    def setUp(self):
        super().setUp()
        self.lp = self.linkp()

    def test_x_height_run_reaches_least(self):
        self.assertLess(self.lp._labelInk_('cow')[0], self.lp._labelInk_('cat')[0])

    def test_short_ascender_sits_between_x_height_and_ascender(self):
        _x_, _t_, _cap_ = (self.lp._labelInk_(s)[0] for s in ('cow', 'cat', 'CAT'))
        self.assertLess(_x_, _t_)
        self.assertLess(_t_, _cap_)

    def test_digits_and_punctuation_reach_the_ascender(self):
        _cap_ = self.lp._labelInk_('CAT')[0]
        for _s_ in ('12.5', '#!?', 'ünïcode-ish'):
            self.assertAlmostEqual(self.lp._labelInk_(_s_)[0], _cap_, places=9, msg=_s_)

    def test_only_descenders_reach_below(self):
        self.assertGreater(self.lp._labelInk_('dog')[1], 0.0)
        self.assertEqual(self.lp._labelInk_('cat')[1], 0.0)
        self.assertEqual(self.lp._labelInk_('CAT')[1], 0.0)

    def test_tallest_character_wins(self):
        self.assertEqual(self.lp._labelInk_('cowB')[0], self.lp._labelInk_('B')[0])

    def test_scales_with_txt_h(self):
        _big_ = self.linkp(txt_h=24)
        self.assertAlmostEqual(_big_._labelInk_('cow')[0], 2 * self.lp._labelInk_('cow')[0],
                               places=9)

    def test_empty_and_whitespace_have_no_ink(self):
        for _s_ in ('', '   ', '\t'):
            self.assertEqual(self.lp._labelInk_(_s_), (0.0, 0.0))

    def test_spaces_do_not_count_as_unclassified_characters(self):
        self.assertEqual(self.lp._labelInk_('cow cow'), self.lp._labelInk_('cow'))


# ---------------------------------------------------------------------------
# Shapes: line rotates, curve uses <textPath>, flowmap opts out
# ---------------------------------------------------------------------------

class TestLinkLabelShapes(_LinkLabelTestBase_):

    def test_line_rotates_the_text_about_its_anchor(self):
        _lp_ = self.linkp(link_shape='line')
        for _txt_, _attr_, _tp_ in _texts(_lp_.svg):
            self.assertIsNone(_tp_)
            self.assertIn(f'{_attr_["x"]},{_attr_["y"]}', _attr_['transform'])

    def test_curve_uses_a_textpath_per_label(self):
        _lp_ = self.linkp(link_shape='curve')
        _tps_ = [t[2] for t in _texts(_lp_.svg)]
        self.assertEqual(len(_tps_), 4)
        for _tp_ in _tps_:
            self.assertIsNotNone(_tp_)
            self.assertEqual(_tp_['startOffset'], '50%')

    def test_curve_textpath_hrefs_resolve_to_defs_paths(self):
        _lp_  = self.linkp(link_shape='curve')
        _ids_ = set(re.findall(r'<path id="([^"]+)"', _lp_.svg))
        _hrefs_ = [t[2]['href'].lstrip('#') for t in _texts(_lp_.svg)]
        self.assertEqual(len(_hrefs_), len(set(_hrefs_)), 'textPath ids must be unique')
        for _h_ in _hrefs_:
            self.assertIn(_h_, _ids_)
        # ...and those paths are inside <defs> so they never paint
        _defs_ = _lp_.svg[_lp_.svg.index('<defs>'):_lp_.svg.index('</defs>')]
        for _h_ in _hrefs_:
            self.assertIn(f'id="{_h_}"', _defs_)

    def test_curve_ids_are_scoped_per_instance(self):
        '''Two linkps embedded in one page must not fight over label path ids.'''
        _a_ = set(re.findall(r'<path id="([^"]+)"', self.linkp(link_shape='curve').svg))
        _b_ = set(re.findall(r'<path id="([^"]+)"', self.linkp(link_shape='curve').svg))
        self.assertEqual(_a_ & _b_, set())

    def test_curve_bidirectional_dy_opposes(self):
        '''dy runs along the path's own +y, so the two directions take opposite signs.'''
        _df_ = pl.DataFrame({'fm': ['a', 'b'], 'to': ['b', 'a'], 'dsc': ['calls', 'answers']})
        _lp_ = self.p2s.linkp(_df_, relationships=[('fm', 'to', 'dsc')],
                              pos={'a': (0.0, 0.0), 'b': (1.0, 0.0)}, wxh=(400, 200),
                              draw_link_labels=True, link_shape='curve')
        _dy_ = {t[0]: float(t[1]['dy']) for t in _texts(_lp_.svg)}
        self.assertLess(_dy_['calls'] * _dy_['answers'], 0.0)

    def test_flowmap_is_not_labeled(self):
        _lp_ = self.linkp(link_shape='flowmap')
        self.assertEqual(_label_texts(_lp_.svg), [])
        self.assertEqual(_lp_._link_label_info_, [])

    def test_no_label_defs_when_shape_is_line(self):
        self.assertNotIn('<path id=', self.linkp(link_shape='line').svg)


# ---------------------------------------------------------------------------
# WebGPU display list
# ---------------------------------------------------------------------------

class TestLinkLabelWebGPU(_LinkLabelTestBase_):

    def _glyph_count(self, lp):
        _pay_ = lp.webgpu()
        return sum(_e_['count'] for _e_ in _pay_['manifest'] if _e_['kind'] == 'glyph')

    def test_labels_reach_the_gpu_as_glyphs(self):
        '''One glyph instance per drawn character, for both labeled shapes -- the curve
        case has no text-on-path primitive and is approximated as a straight run.'''
        _chars_ = len('calls' + 'answers' + 'pings' + 'emails')
        for _shape_ in ('line', 'curve'):
            _lp_ = self.linkp(link_shape=_shape_)
            self.assertEqual(len(_lp_._link_label_info_), 4)
            self.assertEqual(self._glyph_count(_lp_), _chars_, f'shape={_shape_}')

    def test_no_glyphs_without_draw_link_labels(self):
        _lp_ = self.p2s.linkp(_DF_, relationships=[('fm', 'to', 'dsc')], pos=_POS_, wxh=_WXH_)
        self.assertEqual(self._glyph_count(_lp_), 0)

    def test_gpu_anchor_matches_the_svg_for_the_line_shape(self):
        _lp_ = self.linkp(link_shape='line')
        _svg_pts_ = sorted((round(float(t[1]['x']), 2), round(float(t[1]['y']), 2))
                           for t in _texts(_lp_.svg))
        _gpu_pts_ = sorted((round(e[0], 2), round(e[1], 2)) for e in _lp_._link_label_info_)
        self.assertEqual(_svg_pts_, _gpu_pts_)

    def test_gpu_carries_the_label_color(self):
        _lp_ = self.linkp(rels=(('fm', 'to'),), color='grp')
        _svg_fills_ = sorted(t[1]['fill'] for t in _texts(_lp_.svg))
        self.assertEqual(sorted(e[4] for e in _lp_._link_label_info_), _svg_fills_)


# ---------------------------------------------------------------------------
# Validation, warnings, and plumbing
# ---------------------------------------------------------------------------

class _WarnBase_(_LinkLabelTestBase_):

    class _Handler_(logging.Handler):
        def __init__(self):
            super().__init__()
            self.records = []
        def emit(self, record):
            self.records.append(record.getMessage())

    def setUp(self):
        super().setUp()
        self.logger  = logging.getLogger('polars2svg_logger')
        self.handler = self._Handler_()
        self.logger.addHandler(self.handler)
        for _f_ in self.logger.filters:
            if type(_f_).__name__ == 'OnceFilter':
                _f_.seen_messages.clear()

    def tearDown(self):
        self.logger.removeHandler(self.handler)

    def linkLabelWarnings(self):
        return [m for m in self.handler.records if 'draw_link_labels=' in m]


class TestLinkLabelWarnings(_WarnBase_):

    def test_flowmap_warns(self):
        self.linkp(link_shape='flowmap')
        self.assertEqual(len(self.linkLabelWarnings()), 1)
        self.assertIn('flowmap', self.linkLabelWarnings()[0])

    def test_no_link_size_warns(self):
        self.linkp(link_size=None)
        self.assertEqual(len(self.linkLabelWarnings()), 1)
        self.assertIn('link_size=None', self.linkLabelWarnings()[0])

    def test_nothing_to_label_warns(self):
        '''Two-part tuples and a color= that names no field -> no label source.'''
        self.linkp(rels=(('fm', 'to'),))
        self.assertEqual(len(self.linkLabelWarnings()), 1)
        self.assertIn('third', self.linkLabelWarnings()[0])

    def test_no_warning_when_labels_are_drawn(self):
        self.linkp()
        self.assertEqual(self.linkLabelWarnings(), [])

    def test_no_warning_when_the_feature_is_off(self):
        self.p2s.linkp(_DF_, relationships=[('fm', 'to')], pos=_POS_, wxh=_WXH_,
                       link_shape='flowmap')
        self.assertEqual(self.linkLabelWarnings(), [])


class TestLinkLabelValidation(_LinkLabelTestBase_):

    def test_missing_label_field_raises_when_labeling(self):
        with self.assertRaises(ValueError) as _ctx_:
            self.linkp(rels=(('fm', 'to', 'nope'),))
        self.assertIn('nope', str(_ctx_.exception))

    def test_missing_label_field_is_tolerated_when_not_labeling(self):
        '''The third element was accepted-and-ignored before draw_link_labels existed;
        an unlabeled render must keep working.'''
        _lp_ = self.p2s.linkp(_DF_, relationships=[('fm', 'to', 'nope')], pos=_POS_, wxh=_WXH_)
        self.assertIn('<svg', _lp_.svg)


class TestRenamedParameters(_LinkLabelTestBase_):
    '''draw_labels became draw_node_labels when the link channel gained its own pair.
    Both old spellings fail loudly rather than being silently ignored -- and the
    link_labels *name* survived the rename with a different meaning, so a stale boolean
    has to be caught explicitly instead of blowing up on len() mid-render.'''

    def test_draw_labels_raises_with_the_new_name(self):
        with self.assertRaises(TypeError) as _ctx_:
            self.p2s.linkp(_DF_, relationships=[('fm', 'to')], pos=_POS_, draw_labels=True)
        self.assertIn('draw_labels -> draw_node_labels', str(_ctx_.exception))

    def test_old_names_are_gone_from_the_allowlist(self):
        from polars2svg.linkp import LinkP
        self.assertNotIn('draw_labels', LinkP._VALID_KWARGS)
        self.assertNotIn('draw_labels', self.p2s._COMPONENT_KWARGS_['linkp'])

    def test_boolean_link_labels_points_at_draw_link_labels(self):
        with self.assertRaises(TypeError) as _ctx_:
            self.p2s.linkp(_DF_, relationships=[('fm', 'to', 'dsc')], pos=_POS_,
                           link_labels=True)
        self.assertIn('draw_link_labels', str(_ctx_.exception))

    def test_instance_has_no_stale_attributes(self):
        _lp_ = self.linkp()
        self.assertFalse(hasattr(_lp_, 'draw_labels'))

    def test_interactive_setters_follow_the_rename(self):
        _lp_ = self.linkp()
        self.assertFalse(hasattr(_lp_, 'drawLabels'))
        _lp_.drawNodeLabels(True)
        self.assertTrue(_lp_.draw_node_labels)
        _lp_.drawLinkLabels(False)
        self.assertFalse(_lp_.draw_link_labels)
        self.assertEqual(sorted(_label_texts(_lp_.renderSVG())), ['a', 'b', 'c'])


class TestLinkLabelPlumbing(_LinkLabelTestBase_):

    def test_kwarg_is_registered(self):
        self.assertIn('draw_link_labels', self.p2s._COMPONENT_KWARGS_['linkp'])
        self.assertIn('link_labels', self.p2s._COMPONENT_KWARGS_['linkp'])

    def test_template_carries_the_setting(self):
        _lp_ = self.linkp()
        _cp_ = self.p2s.linkp(_DF_, template=_lp_)
        self.assertTrue(_cp_.draw_link_labels)
        self.assertEqual(sorted(_label_texts(_cp_.svg)), sorted(_label_texts(_lp_.svg)))

    def test_set_defaults_accepts_it(self):
        self.p2s.set_defaults('linkp', draw_link_labels=True)
        try:
            _lp_ = self.p2s.linkp(_DF_, relationships=[('fm', 'to', 'dsc')], pos=_POS_, wxh=_WXH_)
            self.assertIn('calls', _label_texts(_lp_.svg))
        finally:
            self.p2s.reset_defaults()

    def test_label_columns_do_not_leak_into_df_link(self):
        _lp_ = self.linkp()
        self.assertEqual([c for c in _lp_.df_link.columns if c.startswith('__ll_')], [])

    def test_survives_a_re_render(self):
        '''The interactive path re-runs the render phases; labels must come back and
        the textPath ids must still resolve.'''
        _lp_ = self.linkp(link_shape='curve')
        _lp_.invalidateRender()
        _svg_ = _lp_.renderSVG()
        self.assertEqual(sorted(_label_texts(_svg_)), ['answers', 'calls', 'emails', 'pings'])
        _ids_ = set(re.findall(r'<path id="([^"]+)"', _svg_))
        for _t_ in _texts(_svg_):
            self.assertIn(_t_[2]['href'].lstrip('#'), _ids_)

    def test_small_multiples_panels_label_independently(self):
        _df_ = _DF_.with_columns(pl.Series('panel', ['p1', 'p1', 'p2', 'p2']))
        _lp_ = self.linkp()
        _panels_ = _lp_.renderSmallMultiples(_df_, {k: _df_.filter(pl.col('panel') == k)
                                                    for k in ('p1', 'p2')}, 'panel')
        self.assertEqual(sorted(_label_texts(_panels_['p1'].svg)), ['answers', 'calls'])
        self.assertEqual(sorted(_label_texts(_panels_['p2'].svg)), ['emails', 'pings'])


if __name__ == '__main__':
    unittest.main()
