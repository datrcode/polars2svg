"""Whole-output checks on every component's finished SVG.

This file used to test `Polars2SVG.roundSvgFloats()`, a regex pass that trimmed float
tails in the finished SVG string. That helper was deleted in 0.2.0 (PLANNING.md S4): a
pass over finished markup cannot tell a coordinate from a label that merely looks like
one, so it silently rewrote IP-address node labels and numeric axis ticks. Float
precision belongs where the number becomes a string -- see PLANNING.md S1, and the
`.round()`/format expressions in linkp and xyp.

What survives here is the check that was never about the trimmer: no component may leak
the multi-field separator (\x1f, used to join tuple fields) into its output -- and, since
0.2.1, that no component reuses an SVG id across two renders.
"""
import re
import unittest

import polars as pl

from polars2svg import Polars2SVG


class TestComponentOutputHygiene(unittest.TestCase):
    '''Every component's finished SVG, checked as a whole string.'''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def _renderers(self):
        p2s = self.p2s
        # xyp / histop / piep / timep
        _df_xy_ = pl.DataFrame({'x': [1.37, 2.91, 3.14, 4.6, 5.5, 6.28],
                                'y': [3.33, 1.1, 4.77, 1.9, 5.2, 9.81]})
        _df_cat_ = pl.DataFrame({'cat': ['a', 'b', 'a', 'c', 'b', 'a', 'c', 'b'],
                                 'val': [1.5, 2.7, 3.9, 4.1, 5.3, 6.6, 7.2, 8.8]})
        _df_ts_ = pl.DataFrame({'ts': ['2021-01-03', '2021-02-11', '2021-02-27',
                                       '2021-05-19', '2021-08-01', '2021-11-30']}
                               ).with_columns(pl.col('ts').str.to_datetime())
        # graph components
        _df_g_ = pl.DataFrame({'fm': ['a', 'b', 'c', 'a', 'd', 'b', 'c', 'a'],
                               'to': ['b', 'a', 'a', 'c', 'a', 'c', 'b', 'd']})
        _df_sl_ = pl.DataFrame({'fm':   ['a', 'b', 'c', 'a', 'd', 'b', 'c', 'a'],
                                'to':   ['b', 'a', 'a', 'c', 'a', 'c', 'b', 'd'],
                                'time': [1, 1, 1, 2, 2, 2, 3, 3]})
        return {
            'xyp':     lambda: p2s.xyp(_df_xy_, 'x', 'y', wxh=(200, 200)),
            'histop':  lambda: p2s.histop(_df_cat_, 'cat', count='val', wxh=(200, 200)),
            'piep':    lambda: p2s.piep(_df_cat_, 'cat', wxh=(200, 200)),
            'timep':   lambda: p2s.timep(_df_ts_, ('ts', p2s.LT_Y_mp), wxh=(256, 128)),
            'linkp':   lambda: p2s.linkp(_df_g_, [('fm', 'to')], wxh=(300, 300)),
            'chordp':  lambda: p2s.chordp(_df_g_, [('fm', 'to')], wxh=(300, 300)),
            'spreadlinesp': lambda: p2s.spreadlinesp(_df_sl_, [('fm', 'to')],
                                                     ego='a', time='time', wxh=(500, 260)),
            # A set-valued ego with only part of it highlighted -- the only render that
            # reaches spreadlinesp's cloud <defs> and its per-bin clipPath, which is
            # where the unscoped ids were.
            'spreadlinesp_ego_set': lambda: p2s.spreadlinesp(_df_sl_, [('fm', 'to')],
                                                             ego=['a', 'b'], time='time',
                                                             highlight_nodes={'a'},
                                                             wxh=(500, 260)),
            'smallp':  lambda: p2s.smallp(_df_cat_, 'cat',
                                          p2s.xyp(_df_cat_, 'val', 'val'), wxh=(300, 300)),
        }

    def test_no_control_char_leak(self):
        # The multi-field separator joins tuple fields internally; it must never reach
        # the rendered SVG.
        for _name_, _fn_ in self._renderers().items():
            with self.subTest(component=_name_):
                self.assertNotIn('\x1f', _fn_().svg)

    def test_ids_are_scoped_to_one_render(self):
        '''No id may survive from one render into the next.

        A finished SVG is normally embedded in a page beside other figures, where they
        share one DOM: `url(#x)` and `href="#x"` resolve to the *first* element with
        that id in document order, so two figures emitting the same id silently point
        the second figure's reference at the first figure's element.  Every component
        therefore mixes a per-render random integer into every id it emits.

        spreadlinesp did not, until 0.2.1: it emitted `cloud_outline` and `ccl_<bin>`
        bare.  `ccl_<bin>` was the one that broke a render rather than just the markup
        -- the clip rect is positioned from that figure's own layout, so a second
        spreadlinesp on the page had its partial-selection rings clipped by the first
        figure's window, and where the two did not overlap the ring vanished.

        linkp's `#cloud` was the last bare one.  It was safe only while the <defs> body
        was a fixed string; theming the outline to the palette made the definition
        render-dependent, so it now carries rand_id like everything else (see
        TestLinkPCloudDefs.test_the_cloud_def_is_scoped_and_follows_the_palette).  No
        renderer here emits one anyway -- a cloud needs collapsed nodes.
        '''
        for _name_, _fn_ in self._renderers().items():
            with self.subTest(component=_name_):
                _first_  = set(re.findall(r'id="([^"]+)"', _fn_().svg))
                _second_ = set(re.findall(r'id="([^"]+)"', _fn_().svg))
                self.assertEqual(sorted(_first_ & _second_), [],
                                 f'{_name_} reuses ids across renders; two of these '
                                 f'figures on one page would collide')


if __name__ == '__main__':
    unittest.main()
