import logging
import re
import unittest
from math import atan2, degrees
import polars as pl
from polars2svg import Polars2SVG
from svg_test_utils import capture_log_warnings, normalize_svg

# Same DF used in test_linkp_color.py so cross-component comparisons are valid
_DF_ = pl.DataFrame({
    'fm':      ['a',     'b',     'c',     'd',     'b'],
    'to':      ['b',     'c',     'd',     'a',     'a'],
    'category':['cat_x', 'cat_y', 'cat_y', 'cat_x', 'cat_x'],
    'cat_n':   [10,      12,      12,      10,      10],
    'count':   [2.0,     5.0,     10.0,    0.1,     0.5],
})
_REL_ = [('fm', 'to')]
_POS_ = {'a': (0.0, 0.5), 'b': (0.5, 0.0), 'c': (1.0, 0.5), 'd': (0.5, 1.0)}


def _cp(**extra):
    p2s = Polars2SVG()
    return p2s.chordp(df=_DF_, relationships=_REL_, **extra)


def _lp(**extra):
    p2s = Polars2SVG()
    return p2s.linkp(df=_DF_, relationships=_REL_, pos=_POS_,
                     wxh=(96, 96), link_shape='curve', insets=(16, 16), **extra)


def _node_fills_linkp(svg):
    return sorted(re.findall(r'<circle[^>]*fill="(#[0-9a-fA-F]+)"', svg))


# Aggregates need more than one row per link: six links of 1-6 rows, each row its own
# cat_n (1-21), one link (d->a) mixing two categories, and nodes that see one category
# (c, e, f) or several (a, b, d).
_PAIRS_ = [('a', 'b', ['cat_x'] * 3), ('b', 'c', ['cat_y']), ('c', 'd', ['cat_y'] * 2),
           ('d', 'a', ['cat_x'] * 3 + ['cat_z']), ('b', 'a', ['cat_x'] * 5), ('e', 'f', ['cat_w'] * 6)]
_AGG_DF_ = pl.DataFrame({'fm':       [f for f, _, cats in _PAIRS_ for _ in cats],
                         'to':       [t for _, t, cats in _PAIRS_ for _ in cats],
                         'category': [c for _, _, cats in _PAIRS_ for c in cats]}
                        ).with_columns(pl.int_range(1, pl.len() + 1).alias('cat_n'))


def _agg(**extra):
    return Polars2SVG().chordp(df=_AGG_DF_, relationships=_REL_, **extra)


# {node: value} of one aggregate over a node's rows: each row naming it, at either end
def perNode(expr: pl.Expr) -> dict:
    _rows_ = pl.concat([_AGG_DF_.select(pl.col(_end_).alias('node'), 'category', 'cat_n') for _end_ in ('fm', 'to')])
    return dict(_rows_.group_by('node').agg(expr.alias('m')).iter_rows())


# {(fm, to): value} of one aggregate over each link's rows
def perLink(expr: pl.Expr) -> dict:
    return {(f, t): m for f, t, m in _AGG_DF_.group_by('fm', 'to').agg(expr.alias('m')).iter_rows()}


# {key: hex} on the colour scale: magnitude places a value on (v - lo) / (hi - lo), with
# hi - lo padded by 0.001 as the colour mixin does; stretched places its dense rank
def spectrum(p2s, stats: dict, stretched: bool = False) -> dict:
    _keys_ = list(stats)
    _vals_ = [float(stats[k]) for k in _keys_]
    if stretched:
        _ranks_ = sorted(set(_vals_))
        _n_ = [_ranks_.index(v) / max(len(_ranks_) - 1, 1) for v in _vals_]
    else:
        _lo_, _hi_ = min(_vals_), max(_vals_)
        _n_ = [min(max((v - _lo_) / (0.001 + _hi_ - _lo_), 0.0), 1.0) for v in _vals_]
    _df_ = (pl.DataFrame({'n': _n_}).with_columns(p2s.colorSpectrumPolarsOperations('n', 'r', 'g', 'b'))
            .with_columns(p2s.hexColorFromRGBTriplesPolarsOperations('r', 'g', 'b').alias('hx')))
    return dict(zip(_keys_, _df_['hx'].to_list()))


# {node: hex} of every node arc in the SVG, each arc named by the node whose arc starts
# where it does
def chordArcs(cp) -> dict:
    _out_ = {}
    for _x_, _y_, _hex_ in re.findall(r'<path d="M ([-\d.]+) ([-\d.]+) A [^"]* Z" fill="(#[0-9a-f]{6})" stroke="\3"', cp.svg):
        _a_ = degrees(atan2(float(_y_) - cp.cy, float(_x_) - cp.cx))
        _nm_ = min(cp.node_to_arc, key=lambda nm: abs((_a_ - cp.node_to_arc[nm][0] + 180.0) % 360.0 - 180.0))
        assert _nm_ not in _out_, f'two arcs start at node {_nm_}'
        _out_[_nm_] = _hex_
    return _out_


# {(fm, to): hex} of every link curve in the SVG, each end named by the node whose arc it
# lands on (the arc whose middle is angularly nearest)
def chordLinks(cp) -> dict:
    _mid_ = {nm: (a0 + a1) / 2.0 for nm, (a0, a1) in cp.node_to_arc.items()}

    def _node_(x: float, y: float) -> str:
        _a_ = degrees(atan2(y - cp.cy, x - cp.cx))
        return min(_mid_, key=lambda nm: abs((_a_ - _mid_[nm] + 180.0) % 360.0 - 180.0))

    _out_ = {}
    for _d_, _hex_ in re.findall(r'<path d="(M [^"]+)" fill="none" stroke="(#[0-9a-f]{6})"', cp.svg):
        _v_ = [float(_t_) for _t_ in _d_.replace('M', ' ').replace('C', ' ').split()]
        _key_ = (_node_(_v_[0], _v_[1]), _node_(_v_[-2], _v_[-1]))
        assert _key_ not in _out_, f'two links drawn for {_key_}'
        _out_[_key_] = _hex_
    return _out_


class TestChordPColorModes(unittest.TestCase):
    """Each colour mode, read back off the SVG: every node arc and every link curve is
    the colour the mode gives its rows, computed here with polars on _AGG_DF_."""

    def setUp(self):
        self.p2s = Polars2SVG()
        self.data_co = self.p2s.colorTyped('data', 'default')

    def assertNodesAre(self, cp, want: dict) -> None:
        self.assertEqual(chordArcs(cp), want)
        self.assertEqual(cp.color_nodes_final, want)

    def assertLinksAre(self, cp, want: dict) -> None:
        self.assertEqual(chordLinks(cp), want)

    def assertSameRender(self, a, b) -> None:
        self.assertEqual(normalize_svg(a.svg), normalize_svg(b.svg))

    # A categorical colour: a key whose rows hold one category takes that category's
    # colour.  One that mixes several takes `mixed` -- or, when that is None, one shared
    # colour of its own, which is no category's (CSETp's multiset sentinel).
    def assertCategorical(self, got: dict, values: dict, mixed: str | None = None) -> None:
        _single_ = {k: self.p2s.color(v[0]) for k, v in values.items() if len(v) == 1}
        _mixed_  = {k for k, v in values.items() if len(v) > 1}
        self.assertTrue(_single_ and _mixed_, 'the frame needs both kinds of key')
        self.assertEqual(set(got), set(values))
        self.assertEqual({k: got[k] for k in _single_}, _single_)
        self.assertEqual({got[k] for k in _mixed_}, {mixed} if mixed else {got[next(iter(_mixed_))]})
        if mixed is None:
            self.assertNotIn(got[next(iter(_mixed_))], {self.p2s.color(c) for c in _AGG_DF_['category'].unique()} | {self.data_co})

    def test_the_frame_separates_every_node_statistic(self):
        _maps_ = {name: spectrum(self.p2s, perNode(expr)) for name, expr in
                  (('sum', pl.col('cat_n').sum()), ('min', pl.col('cat_n').min()), ('max', pl.col('cat_n').max()),
                   ('mean', pl.col('cat_n').mean()), ('median', pl.col('cat_n').median()),
                   ('rows', pl.len()), ('set', pl.col('category').n_unique()))}
        _maps_['stretched'] = spectrum(self.p2s, perNode(pl.col('cat_n').sum()), stretched=True)
        for name, m in _maps_.items():
            self.assertGreaterEqual(len(set(m.values())), 3, f'{name} barely varies')
        _seen_ = [tuple(sorted(m.items())) for m in _maps_.values()]
        self.assertEqual(len(set(_seen_)), len(_seen_), 'two statistics colour the nodes alike')

    # --- node_color modes ---

    def test_node_color_none(self):
        self.assertNodesAre(_agg(node_color=None), {n: self.data_co for n in 'abcdef'})

    def test_node_color_hex(self):
        self.assertNodesAre(_agg(node_color='#ff0000'), {n: '#ff0000' for n in 'abcdef'})

    def test_node_color_color_by_node_name(self):
        self.assertNodesAre(_agg(node_color=self.p2s.COLOR_BY_NODE_NAME), {n: self.p2s.color(n) for n in 'abcdef'})

    def test_node_color_str_field(self):
        '''A text field is CSETp: a node that saw one category takes its colour.'''
        _cp_ = _agg(node_color='category')
        self.assertCategorical(chordArcs(_cp_), perNode(pl.col('category').unique()))
        self.assertEqual(_cp_.color_nodes_final, chordArcs(_cp_))

    def test_node_color_cset_tuple(self):
        self.assertSameRender(_agg(node_color=('category', self.p2s.CSETp)), _agg(node_color='category'))

    def test_node_color_bare_field_tuple(self):
        '''('field',) hashes the field's values; a node that saw several is the data colour.'''
        self.assertCategorical(chordArcs(_agg(node_color=('category',))), perNode(pl.col('category').unique()),
                               mixed=self.data_co)

    def test_node_color_numeric_field(self):
        '''A numeric field is CMAGNITUDE_SUMp.'''
        self.assertNodesAre(_agg(node_color='cat_n'), spectrum(self.p2s, perNode(pl.col('cat_n').sum())))

    def test_node_color_magnitude_sum(self):
        self.assertNodesAre(_agg(node_color=('cat_n', self.p2s.CMAGNITUDE_SUMp)),
                            spectrum(self.p2s, perNode(pl.col('cat_n').sum())))

    def test_node_color_magnitude_min(self):
        self.assertNodesAre(_agg(node_color=('cat_n', self.p2s.CMAGNITUDE_MINp)),
                            spectrum(self.p2s, perNode(pl.col('cat_n').min())))

    def test_node_color_magnitude_max(self):
        self.assertNodesAre(_agg(node_color=('cat_n', self.p2s.CMAGNITUDE_MAXp)),
                            spectrum(self.p2s, perNode(pl.col('cat_n').max())))

    def test_node_color_magnitude_mean(self):
        self.assertNodesAre(_agg(node_color=('cat_n', self.p2s.CMAGNITUDE_MEANp)),
                            spectrum(self.p2s, perNode(pl.col('cat_n').mean())))

    def test_node_color_magnitude_median(self):
        self.assertNodesAre(_agg(node_color=('cat_n', self.p2s.CMAGNITUDE_MEDIANp)),
                            spectrum(self.p2s, perNode(pl.col('cat_n').median())))

    def test_node_color_stretched_sum(self):
        self.assertNodesAre(_agg(node_color=('cat_n', self.p2s.CSTRETCHED_SUMp)),
                            spectrum(self.p2s, perNode(pl.col('cat_n').sum()), stretched=True))

    def test_node_color_crow_magnitude(self):
        self.assertNodesAre(_agg(node_color=self.p2s.CROW_MAGNITUDEp), spectrum(self.p2s, perNode(pl.len())))

    def test_node_color_crow_stretched(self):
        self.assertNodesAre(_agg(node_color=self.p2s.CROW_STRETCHEDp),
                            spectrum(self.p2s, perNode(pl.len()), stretched=True))

    def test_node_color_cset_magnitude(self):
        self.assertNodesAre(_agg(node_color=('category', self.p2s.CSET_MAGNITUDEp)),
                            spectrum(self.p2s, perNode(pl.col('category').n_unique())))

    def test_node_color_cset_stretched(self):
        self.assertNodesAre(_agg(node_color=('category', self.p2s.CSET_STRETCHEDp)),
                            spectrum(self.p2s, perNode(pl.col('category').n_unique()), stretched=True))

    def test_a_remainder_bucket_is_coloured_by_its_members_rows(self):
        _sums_ = perNode(pl.col('cat_n').sum())
        _want_ = {'a': _sums_['a'], 'b': _sums_['b'], self.p2s.REMAINDER_LABEL: sum(_sums_[n] for n in 'cdef')}
        self.assertNodesAre(_agg(order=['a', 'b', self.p2s.REMAINDERp], node_color=('cat_n', self.p2s.CMAGNITUDE_SUMp)),
                            spectrum(self.p2s, _want_))

    def test_node_color_dict(self):
        '''Named nodes take their colour; the rest the background.'''
        _bg_ = self.p2s.colorTyped('background', 'default')
        self.assertNodesAre(_agg(node_color={'a': '#ff0000', 'b': '#00ff00'}),
                            {'a': '#ff0000', 'b': '#00ff00', **{n: _bg_ for n in 'cdef'}})

    # --- color (link) modes ---

    def test_link_color_none(self):
        self.assertLinksAre(_agg(color=None), {k: self.data_co for k in perLink(pl.len())})

    def test_link_color_hex(self):
        self.assertLinksAre(_agg(color='#00ff00'), {k: '#00ff00' for k in perLink(pl.len())})

    def test_link_color_src(self):
        '''Each link takes its source node's colour (PLANNING.md §5 C-chordp-link-color-src).'''
        _cp_ = _agg(color=self.p2s.COLOR_BY_SRC_NODE, node_color=self.p2s.COLOR_BY_NODE_NAME)
        self.assertLinksAre(_cp_, {(f, t): self.p2s.color(f) for f, t in perLink(pl.len())})

    def test_link_color_dst(self):
        _cp_ = _agg(color=self.p2s.COLOR_BY_DST_NODE, node_color=('cat_n', self.p2s.CMAGNITUDE_SUMp))
        _nodes_ = spectrum(self.p2s, perNode(pl.col('cat_n').sum()))
        self.assertLinksAre(_cp_, {(f, t): _nodes_[t] for f, t in perLink(pl.len())})

    # The deprecation warnings a render logs, the filter's memory cleared first
    def deprecations(self, fn) -> list:
        for _f_ in logging.getLogger('polars2svg_logger').filters:
            if type(_f_).__name__ == 'OnceFilter': _f_.seen_messages.clear()
        return [r.getMessage() for r in capture_log_warnings(fn) if 'deprecated' in r.getMessage()]

    def test_a_column_named_src_is_the_field(self):
        '''A frame with a 'src' column colours by that field: color='src' is then CSETp,
        and names a column, so nothing is deprecated.'''
        _df_ = _AGG_DF_.rename({'fm': 'src', 'to': 'dst'})
        _cp_ = Polars2SVG().chordp(df=_df_, relationships=[('src', 'dst')], color='src')
        self.assertEqual(self.deprecations(lambda: _cp_._repr_svg_()), [])
        self.assertLinksAre(_cp_, {(f, t): self.p2s.color(f) for f, t in perLink(pl.len())})
        self.assertEqual(set(_cp_.color_nodes_final.values()), {self.data_co})

    def test_bare_src_and_dst_are_deprecated_spellings_of_the_enums(self):
        '''Decided 2026-09-29 (PLANNING.md §5 C-chordp-link-color-src): a string always names
        a column, so link-by-node colouring moved onto p2s.COLOR_BY_SRC_NODE / _DST_NODE.
        With no such column the bare keyword still works -- the same render -- and warns
        once, naming the enum; the enum itself warns nothing.'''
        for _kw_, _enum_ in (('src', self.p2s.COLOR_BY_SRC_NODE), ('dst', self.p2s.COLOR_BY_DST_NODE)):
            with self.subTest(keyword=_kw_):
                _msgs_ = self.deprecations(lambda: self.assertSameRender(
                    _agg(color=_kw_, node_color=self.p2s.COLOR_BY_NODE_NAME),
                    _agg(color=_enum_, node_color=self.p2s.COLOR_BY_NODE_NAME)))
                self.assertEqual(_msgs_, [f"Chordp: color='{_kw_}' is deprecated; use color=p2s.{_enum_.name} instead"])
                self.assertEqual(self.deprecations(lambda: _agg(color=_enum_)._repr_svg_()), [])

    def test_link_color_str_field(self):
        '''A text field is CSETp: a link whose rows hold one category takes its colour.'''
        self.assertCategorical(chordLinks(_agg(color='category')), perLink(pl.col('category').unique()))

    def test_link_color_cset_tuple(self):
        self.assertSameRender(_agg(color=('category', self.p2s.CSETp)), _agg(color='category'))

    def test_link_color_numeric_field(self):
        self.assertLinksAre(_agg(color='cat_n'), spectrum(self.p2s, perLink(pl.col('cat_n').sum())))

    def test_link_color_crow_magnitude(self):
        self.assertLinksAre(_agg(color=self.p2s.CROW_MAGNITUDEp), spectrum(self.p2s, perLink(pl.len())))

    def test_link_color_crow_stretched(self):
        self.assertLinksAre(_agg(color=self.p2s.CROW_STRETCHEDp), spectrum(self.p2s, perLink(pl.len()), stretched=True))

    def test_link_color_magnitude_sum(self):
        self.assertSameRender(_agg(color=('cat_n', self.p2s.CMAGNITUDE_SUMp)), _agg(color='cat_n'))

    # --- the two channels' colour scales ---

    def test_a_node_scale_alone_sets_the_legend(self):
        _cp_ = _agg(node_color=('cat_n', self.p2s.CMAGNITUDE_SUMp), legend=True, wxh=(320, 256))
        _sums_ = perNode(pl.col('cat_n').sum())
        self.assertEqual((_cp_.legend_info.title, _cp_.legend_info.vmin, _cp_.legend_info.vmax),
                         ('cat_n', min(_sums_.values()), max(_sums_.values())))

    def test_a_node_scale_beside_a_link_scale_keeps_to_its_own_range(self):
        '''The legend describes the links, so its range is theirs; the nodes still spread
        over their own (PLANNING.md §5 C-linkp-two-scales).'''
        _cp_ = _agg(color=self.p2s.CROW_MAGNITUDEp, node_color=('cat_n', self.p2s.CMAGNITUDE_SUMp),
                    legend=True, wxh=(320, 256))
        _rows_ = perLink(pl.len())
        self.assertEqual((_cp_.legend_info.title, _cp_.legend_info.vmin, _cp_.legend_info.vmax),
                         ('rows', min(_rows_.values()), max(_rows_.values())))
        self.assertLinksAre(_cp_, spectrum(self.p2s, _rows_))
        self.assertNodesAre(_cp_, spectrum(self.p2s, perNode(pl.col('cat_n').sum())))


class TestChordPColorConsistency(unittest.TestCase):
    """Verify that color modes produce correct SVG output and color_nodes_final values."""

    def setUp(self):
        self.p2s = Polars2SVG()

    def test_node_color_fixed_hex_appears_in_svg(self):
        cp = _cp(node_color='#ab1234')
        self.assertIn('#ab1234', cp.svg)

    def test_node_color_by_name_distinct_colors(self):
        # Each node gets a unique hash color when COLOR_BY_NODE_NAME is used
        cp = _cp(node_color=self.p2s.COLOR_BY_NODE_NAME)
        colors = set(cp.color_nodes_final.values())
        self.assertEqual(len(colors), len(cp.color_nodes_final),
                         f'Expected distinct colors per node, got {cp.color_nodes_final}')

    def test_node_color_by_name_covers_all_nodes(self):
        cp = _cp(node_color=self.p2s.COLOR_BY_NODE_NAME)
        self.assertEqual(set(cp.color_nodes_final.keys()), {'a', 'b', 'c', 'd'})

    def test_link_color_hex_in_svg(self):
        cp = _cp(color='#aabbcc')
        self.assertIn('#aabbcc', cp.svg)

    def test_link_color_field_renders_without_error(self):
        cp = _cp(color='category')
        self.assertIn('<path', cp.svg)

    def test_invalid_node_color_raises(self):
        with self.assertRaises(ValueError):
            _cp(node_color='nonexistent_field')

    def test_color_by_node_name_raises_for_color_param(self):
        # color=p2s.COLOR_BY_NODE_NAME is invalid (only valid for node_color)
        with self.assertRaises(ValueError):
            _cp(color=self.p2s.COLOR_BY_NODE_NAME)


class TestChordPLinkPColorInterchangeable(unittest.TestCase):
    """Cross-component: same color spec → same node color results in chordp and linkp."""

    def setUp(self):
        self.p2s = Polars2SVG()

    def test_color_by_node_name_same_color_set(self):
        # Both components hash node names the same way → identical color sets
        cp = _cp(node_color=self.p2s.COLOR_BY_NODE_NAME)
        lp = _lp(node_color=self.p2s.COLOR_BY_NODE_NAME)
        chord_colors = set(cp.color_nodes_final.values())
        linkp_colors = set(_node_fills_linkp(lp.svg))
        self.assertEqual(chord_colors, linkp_colors,
                         f'chordp colors {chord_colors} != linkp colors {linkp_colors}')

    def test_color_by_node_name_node_count(self):
        cp = _cp(node_color=self.p2s.COLOR_BY_NODE_NAME)
        lp = _lp(node_color=self.p2s.COLOR_BY_NODE_NAME)
        self.assertEqual(len(cp.color_nodes_final), len(_node_fills_linkp(lp.svg)))


if __name__ == '__main__':
    unittest.main()
