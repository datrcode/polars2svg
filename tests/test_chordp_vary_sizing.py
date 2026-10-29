import re
import unittest

import numpy as np
import polars as pl

from polars2svg import Polars2SVG

#
# chordp's link_size='vary', read back off what is drawn.  It follows linkp's rule C
# (PLANNING.md §5 C-linkp-vary-minmax, D4): width proportional to count from zero, the
# largest count at link_size_range[1], link_size_range[0] a floor.  It had been a copy of
# linkp's old min-max rule.
#


def _svg_link_widths(svg):
    '''Stroke widths of the chords themselves: the unfilled Bezier paths.'''
    return sorted(float(w) for w in re.findall(r'<path d="M [^"]*C [^"]*" fill="none"[^>]*?stroke-width="([0-9.]+)"', svg))


def _gpu_line_widths(cp):
    _out_ = set()
    for _kind_, _vals_, _scissor_ in cp.gpuDisplayList()._ops_:
        if _kind_ != 'line': continue
        _a_ = np.asarray(_vals_, dtype=float)
        _out_ |= {round(float(w), 2) for w in (_a_[None, :] if _a_.ndim == 1 else _a_)[:, 4]}
    return sorted(_out_)


def _three_chords(counts):
    '''a->b, b->c and c->a, of counts[0], counts[1] and counts[2] rows.'''
    _n_ = list(counts)
    return pl.DataFrame({'fm': ['a', 'b', 'c'], 'to': ['b', 'c', 'a'], 'n': _n_})


class TestChordpVaryRuleC(unittest.TestCase):
    def setUp(self):
        self.p2s = Polars2SVG()

    def _cp_(self, df, **kw):
        return self.p2s.chordp(df=df, relationships=[('fm', 'to')], count='n', link_size='vary',
                               draw_border=False, **kw)

    def test_widths_keep_their_counts_ratio(self):
        self.assertEqual(_svg_link_widths(self._cp_(_three_chords((1, 4, 9))).svg), [0.444, 1.778, 4.0])

    def test_equal_counts_draw_at_the_top_of_the_range(self):
        self.assertEqual(_svg_link_widths(self._cp_(_three_chords((5, 5, 5))).svg), [4.0, 4.0, 4.0])

    def test_svg_and_gpu_draw_the_same_widths(self):
        _cp_ = self._cp_(_three_chords((1, 4, 9)))
        self.assertEqual(sorted({round(w, 2) for w in _svg_link_widths(_cp_.svg)}), _gpu_line_widths(_cp_))

    def test_a_negative_count_draws_at_the_floor_and_warns(self):
        for _f_ in self.p2s.logger.filters:      # the logger says each message once per process
            if type(_f_).__name__ == 'OnceFilter': _f_.seen_messages.clear()
        # order= is pinned so that, should C-chordp-negative-count-hang return, the
        # subprocess guard in test_chordp_negative_counts.py fails rather than this hanging
        with self.assertLogs(self.p2s.logger, level='WARNING') as _cm_:
            _cp_ = self._cp_(_three_chords((-3, 5, 10)), order=['a', 'b', 'c'])
        self.assertIn("ChP: count='n' sums below zero in 1 edge (a -> b); a negative count is drawn as zero",
                      [r.getMessage() for r in _cm_.records])
        self.assertEqual(_svg_link_widths(_cp_.svg), [0.25, 2.0, 4.0])
        self.assertNotIn('nan', _cp_.svg.lower())

    def test_sm_count_shares_the_largest_link_not_the_node_totals(self):
        # node a touches 1 + 9 = 10 rows, but nothing chordp's count_range_shared sizes is a
        # node: the panels share the largest drawn link, 9, as their max
        _df_ = _three_chords((1, 4, 9)).with_columns(pl.Series('half', ['lo', 'lo', 'hi']))
        _template_ = self._cp_(_df_, sm_shared={self.p2s.SM_COUNT})
        _panels_ = _template_.renderSmallMultiples(_df_, {k: _df_.filter(pl.col('half') == k) for k in ('lo', 'hi')}, 'all')
        for _k_, _p_ in _panels_.items():
            with self.subTest(panel=_k_):
                self.assertEqual(_p_.count_range_shared[1], 9.0)
        # the 'lo' panel's 4-row chord is drawn against the frame's 9, not its own 4
        self.assertEqual(_svg_link_widths(_panels_['lo'].svg), [0.444, 1.778])


if __name__ == '__main__':
    unittest.main()
