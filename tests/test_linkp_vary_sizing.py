import math
import re
import unittest

import numpy as np
import polars as pl

from polars2svg import Polars2SVG

#
# linkp's 'vary' sizes, read back off what is drawn (PLANNING.md §5 C-linkp-vary-minmax and
# C-linkp-vary-scale-parity).  The tests that came before only asserted that a <line> or a
# <circle> existed, so no rule for the sizes was ever pinned.
#
# Rule C, decided 2026-10-28: proportional from zero, with node *area* tracking the count.
#   link width  = max(range[0], range[1] * count / max)
#   node radius = max(range[0], range[1] * sqrt(count / max))
# Links share one max across every relationship (drawn links only); nodes have their own.
#

_LINK_RANGE_ = (0.25, 4)
_NODE_RANGE_ = (0.3, 4)


def _link_width(count, hi, size_range=_LINK_RANGE_):
    return max(size_range[0], size_range[1] * min(max(count, 0.0) / hi, 1.0))


def _node_radius(count, hi, size_range=_NODE_RANGE_):
    return max(size_range[0], size_range[1] * math.sqrt(min(max(count, 0.0) / hi, 1.0)))


def _svg_links(svg):
    '''{(x1, y1, x2, y2): stroke-width} for every straight link ('vary' writes one on each).'''
    return {(int(a), int(b), int(c), int(d)): float(w) for a, b, c, d, w in re.findall(
        r'<line x1="(-?\d+)" y1="(-?\d+)" x2="(-?\d+)" y2="(-?\d+)"[^>]*?stroke-width="([0-9.]+)"', svg)}


def _svg_nodes(svg):
    '''{(cx, cy): r} for every circle.'''
    return {(int(x), int(y)): float(r) for x, y, r in
            re.findall(r'<circle cx="(-?\d+)" cy="(-?\d+)" r="([0-9.]+)"', svg)}


def _gpu_column(lp, kind, col):
    '''One column of every `kind` op in the WebGPU display list, e.g. line widths (4).'''
    _out_ = []
    for _kind_, _vals_, _scissor_ in lp.gpuDisplayList()._ops_:
        if _kind_ != kind: continue
        _a_ = np.asarray(_vals_, dtype=float)
        _out_.extend((_a_[None, :] if _a_.ndim == 1 else _a_)[:, col].tolist())
    return sorted(_out_)


def _drawn_link_counts(lp):
    '''{(x1, y1, x2, y2): count} for every link a relationship draws (both ends placed).'''
    _out_ = {}
    for i in range(len(lp.relationships)):
        _cols_ = [f'__rel{i}_fm_sx__', f'__rel{i}_fm_sy__', f'__rel{i}_to_sx__', f'__rel{i}_to_sy__']
        for *_xy_, _c_ in lp.df_link.drop_nulls(subset=_cols_).select(*_cols_, '__count__').iter_rows():
            _out_[tuple(int(v) for v in _xy_)] = float(_c_)
    return _out_


def _node_counts(lp):
    return {(int(x), int(y)): float(c) for x, y, c in lp.df_node.select('__sx__', '__sy__', '__count__').iter_rows()}


def _three_links(counts):
    '''Three separate links, one per count; each node touches one link, so its count is
    that link's.  The comparison sheet's first three frames.'''
    _df_ = pl.DataFrame({'fm': ['a1', 'a2', 'a3'], 'to': ['b1', 'b2', 'b3'], 'n': list(counts)})
    _pos_ = {f'a{k}': [0, 3 - k] for k in (1, 2, 3)} | {f'b{k}': [1, 3 - k] for k in (1, 2, 3)}
    return _df_, _pos_


def _hub():
    '''One hub, twelve leaves of 1,1,1,2,2,3,5,8,13,21,34,55 rows; the hub counts 146.'''
    _counts_ = [1, 1, 1, 2, 2, 3, 5, 8, 13, 21, 34, 55]
    _df_ = pl.DataFrame({'fm': ['H'] * len(_counts_), 'to': [f'L{i:02d}' for i in range(len(_counts_))], 'n': _counts_})
    _pos_ = {'H': [0.0, 0.0]} | {f'L{i:02d}': [math.cos(2 * math.pi * i / 12), math.sin(2 * math.pi * i / 12)]
                                  for i in range(12)}
    return _df_, _pos_


def _two_relationships():
    '''A draws links of 1 and 2 rows, B of 10 and 40; every row leaves the other
    relationship's endpoints null, so each table also holds an undrawn null group
    (50 rows for A, 3 for B) -- the group that used to set the SVG's scale.'''
    _pos_ = {'x1': [0, 0], 'y1': [0, 1], 'x2': [1, 0], 'y2': [1, 1],
             'p1': [5, 0], 'q1': [5, 1], 'p2': [6, 0], 'q2': [6, 1]}
    _rows_ = ([{'a': 'x1', 'b': 'y1', 'c': None, 'd': None}] * 1 +
              [{'a': 'x2', 'b': 'y2', 'c': None, 'd': None}] * 2 +
              [{'a': None, 'b': None, 'c': 'p1', 'd': 'q1'}] * 10 +
              [{'a': None, 'b': None, 'c': 'p2', 'd': 'q2'}] * 40)
    return pl.DataFrame(_rows_), [('a', 'b'), ('c', 'd')], _pos_


class _VaryBase_(unittest.TestCase):
    def setUp(self):
        self.p2s = Polars2SVG()

    def _vary_(self, df, pos, rels=(('fm', 'to'),), **kw):
        return self.p2s.linkp(df, relationships=list(rels), pos=pos, link_size='vary', node_size='vary',
                              wxh=(300, 200), draw_border=False, **kw)

    def assertSizesFollowRuleC(self, lp, link_hi, node_hi):
        '''Every drawn link and node is the rule-C size of its own count.'''
        _links_, _nodes_ = _svg_links(lp.svg), _svg_nodes(lp.svg)
        _lc_, _nc_ = _drawn_link_counts(lp), _node_counts(lp)
        self.assertEqual(set(_links_), set(_lc_))
        self.assertEqual(set(_nodes_), set(_nc_))
        for _k_, _c_ in _lc_.items():
            self.assertAlmostEqual(_links_[_k_], _link_width(_c_, link_hi), delta=0.006, msg=f'link count {_c_}')
        for _k_, _c_ in _nc_.items():
            self.assertAlmostEqual(_nodes_[_k_], _node_radius(_c_, node_hi), delta=0.051, msg=f'node count {_c_}')


class TestVaryRuleC(_VaryBase_):
    '''The comparison sheet's four frames, under the rule DT chose.'''

    def test_counts_1_4_9_keep_their_ratio(self):
        _lp_ = self._vary_(*_three_links((1, 4, 9)), count='n')
        self.assertSizesFollowRuleC(_lp_, 9.0, 9.0)
        # widths 1:4:9 and node areas 1:4:9, i.e. radii 1:2:3
        self.assertEqual(sorted(_svg_links(_lp_.svg).values()), [0.44, 1.78, 4.0])
        self.assertEqual(sorted(set(_svg_nodes(_lp_.svg).values())), [1.3, 2.7, 4.0])

    def test_nearly_equal_counts_draw_nearly_equal(self):
        _lp_ = self._vary_(*_three_links((998, 999, 1000)), count='n')
        self.assertSizesFollowRuleC(_lp_, 1000.0, 1000.0)
        _w_ = sorted(_svg_links(_lp_.svg).values())
        self.assertLess(_w_[-1] - _w_[0], 0.02)      # min-max spread these over 0.25 .. 4

    def test_equal_counts_draw_at_the_top_of_the_range(self):
        _lp_ = self._vary_(*_three_links((5, 5, 5)), count='n')
        self.assertEqual(set(_svg_links(_lp_.svg).values()), {4.0})
        self.assertEqual(set(_svg_nodes(_lp_.svg).values()), {4.0})

    def test_a_heavy_tailed_hub(self):
        _lp_ = self._vary_(*_hub(), count='n')
        self.assertSizesFollowRuleC(_lp_, 55.0, 146.0)
        # the floor holds the smallest leaves up; the 55-row leaf is r 2.5 beside the hub's 4
        self.assertEqual(min(_svg_links(_lp_.svg).values()), 0.25)
        self.assertIn(2.5, _svg_nodes(_lp_.svg).values())

    def test_nodes_are_sized_by_area(self):
        _lp_ = self._vary_(*_three_links((1, 4, 16)), count='n')
        _r_ = sorted(set(_svg_nodes(_lp_.svg).values()))
        self.assertEqual(_r_, [1.0, 2.0, 4.0])       # area 1:4:16

    def test_a_size_range_of_the_callers_own(self):
        _lp_ = self._vary_(*_three_links((1, 4, 9)), count='n', link_size_range=(1, 10), node_size_range=(2, 8))
        self.assertEqual(sorted(_svg_links(_lp_.svg).values()), [1.11, 4.44, 10.0])
        self.assertEqual(sorted(set(_svg_nodes(_lp_.svg).values())), [2.7, 5.3, 8.0])


class TestVaryEdges(_VaryBase_):
    '''Counts the rule has to say something definite about.'''

    def _clear_once_filter_(self):
        for _f_ in self.p2s.logger.filters:      # the logger says each message once per process
            if type(_f_).__name__ == 'OnceFilter': _f_.seen_messages.clear()

    def test_a_negative_count_draws_at_the_floor_and_warns(self):
        self._clear_once_filter_()
        with self.assertLogs(self.p2s.logger, level='WARNING') as _cm_:
            _lp_ = self._vary_(*_three_links((-3, 5, 10)), count='n')
        _msgs_ = [r.getMessage() for r in _cm_.records]
        self.assertIn("LinkP: count='n' is below zero for 1 link; 'vary' draws a negative count at the size range's floor", _msgs_)
        self.assertIn("LinkP: count='n' is below zero for 2 nodes; 'vary' draws a negative count at the size range's floor", _msgs_)
        self.assertEqual(sorted(_svg_links(_lp_.svg).values()), [0.25, 2.0, 4.0])
        self.assertNotIn('nan', _lp_.svg.lower())

    def test_all_zero_counts_draw_at_the_floor(self):
        _lp_ = self._vary_(*_three_links((0, 0, 0)), count='n')
        self.assertEqual(set(_svg_links(_lp_.svg).values()), {0.25})
        self.assertEqual(set(_svg_nodes(_lp_.svg).values()), {0.3})
        self.assertNotIn('nan', _lp_.svg.lower())

    def test_a_shared_range_narrower_than_the_data_caps_at_the_top(self):
        # count_range_shared's hi is the reference max; lo plays no part in rule C
        _lp_ = self._vary_(*_three_links((1, 4, 9)), count='n', count_range_shared=(3, 4))
        self.assertEqual(sorted(_svg_links(_lp_.svg).values()), [1.0, 4.0, 4.0])


class TestVarySharedAcrossPanels(_VaryBase_):
    '''SM_COUNT: every small-multiple panel takes the largest link and the largest node of
    the whole frame, each on its own scale (PLANNING.md §5 C-linkp-vary-minmax, D3).  One
    range over link and node counts together had scaled every link against node totals.'''

    def _panels_(self, shared):
        _df_, _pos_ = _hub()
        _df_ = _df_.with_columns(pl.Series('half', ['lo' if c <= 5 else 'hi' for c in _df_['n']]))
        _template_ = self._vary_(_df_, _pos_, count='n', sm_shared=shared)
        _lu_ = {k: _df_.filter(pl.col('half') == k) for k in ('lo', 'hi')}
        return _template_.renderSmallMultiples(_df_, _lu_, 'all')

    def test_panels_scale_against_the_whole_frame(self):
        _panels_ = self._panels_({self.p2s.SM_COUNT})
        # the whole frame's largest link is 55 rows and its hub 146: the 'lo' panel, whose
        # own largest link is 5 and hub 15, still draws against those
        for _k_, _lp_ in _panels_.items():
            with self.subTest(panel=_k_):
                self.assertSizesFollowRuleC(_lp_, 55.0, 146.0)
        self.assertLess(max(_svg_links(_panels_['lo'].svg).values()), 0.4)

    def test_unshared_panels_scale_against_themselves(self):
        _panels_ = self._panels_(set())
        self.assertSizesFollowRuleC(_panels_['lo'], 5.0, 15.0)
        self.assertEqual(max(_svg_links(_panels_['lo'].svg).values()), 4.0)


class TestVaryScaleParity(_VaryBase_):
    '''One scale for the whole picture, over drawn links, the same in SVG and WebGPU.'''

    def setUp(self):
        super().setUp()
        _df_, _rels_, _pos_ = _two_relationships()
        self.lp = self._vary_(_df_, _pos_, rels=_rels_)

    def test_svg_and_gpu_draw_the_same_link_widths(self):
        _svg_ = sorted(_svg_links(self.lp.svg).values())
        _gpu_ = _gpu_column(self.lp, 'line', 4)
        self.assertEqual(len(_svg_), 4)
        self.assertEqual(len(_gpu_), 4)
        for _s_, _g_ in zip(_svg_, _gpu_):
            self.assertAlmostEqual(_s_, _g_, delta=0.01)

    def test_one_scale_across_relationships_over_drawn_links_only(self):
        # the max is the drawn 40: not A's null group of 50, and not each relationship's own
        self.assertEqual(sorted(_drawn_link_counts(self.lp).values()), [1.0, 2.0, 10.0, 40.0])
        self.assertSizesFollowRuleC(self.lp, 40.0, 40.0)

    def test_svg_and_gpu_draw_the_same_node_radii(self):
        _svg_ = sorted(_svg_nodes(self.lp.svg).values())
        _gpu_ = _gpu_column(self.lp, 'circle', 2)
        self.assertEqual(len(_svg_), 8)
        self.assertEqual(len(_gpu_), 8)
        for _s_, _g_ in zip(_svg_, _gpu_):
            self.assertAlmostEqual(_s_, _g_, delta=0.06)     # the SVG rounds r to 0.1


if __name__ == '__main__':
    unittest.main()
