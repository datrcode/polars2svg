import re
import unittest
from unittest import mock
import polars as pl
from polars2svg.piep import Piep
from polars2svg import Polars2SVG
from piep_dataframes import makePieDf
from svg_test_utils import assert_valid_svg


class TestPiepSmallp(unittest.TestCase):
    def setUp(self):
        self.p2s = Polars2SVG()
        self.df  = makePieDf(n=300)

    # ── plain small multiples ──────────────────────────────────────────────────

    def test_smallp_plain(self):
        tmpl = self.p2s.piep(self.df, 'cat', wxh=(110, 110))
        sm   = self.p2s.smallp(self.df, 'group', tmpl)
        assert_valid_svg(self, sm._repr_svg_())

    def test_smallp_renders_a_cell_per_group(self):
        tmpl = self.p2s.piep(self.df, 'cat', wxh=(110, 110))
        sm   = self.p2s.smallp(self.df, 'group', tmpl)
        # group has 3 values → at least 3 <g transform> cells
        self.assertGreaterEqual(sm._repr_svg_().count('<g transform'), 3)

    def test_render_with_produces_new_instance(self):
        tmpl = self.p2s.piep(self.df, 'cat')
        sub  = tmpl.render_with(self.df.filter(self.df['group'] == 'x'))
        self.assertIsNot(sub, tmpl)
        self.assertLessEqual(len(sub.df), len(tmpl.df))

    # ── SM_SLICE_ORDERp: identical slice order across panels ───────────────────

    def test_shared_slice_order_makes_orders_identical(self):
        tmpl = self.p2s.piep(self.df, 'cat', wxh=(110, 110),
                             sm_shared={self.p2s.SM_SLICE_ORDERp})
        renders = tmpl.renderSmallMultiples(self.df, {
            'x': self.df.filter(self.df['group'] == 'x'),
            'y': self.df.filter(self.df['group'] == 'y'),
        }, '__all__')
        order_x = [s['bin'] for s in renders['x']._slices_]
        order_y = [s['bin'] for s in renders['y']._slices_]
        # Both panels lay out the shared bins in the same relative order
        common = [b for b in order_x if b in order_y]
        self.assertEqual(common, [b for b in order_y if b in order_x])

    def test_sm_color_also_shares_order(self):
        tmpl = self.p2s.piep(self.df, 'cat', wxh=(110, 110),
                             sm_shared={self.p2s.SM_COLOR})
        sm = self.p2s.smallp(self.df, 'group', tmpl)
        assert_valid_svg(self, sm._repr_svg_())

    def _spectrumPanels_(self, sm_shared):
        df   = pl.DataFrame({'shelter': ['a', 'a', 'b', 'b', 'b', 'c'],
                             'type':    ['cat', 'dog', 'cat', 'dog', 'ferret', 'cat'],
                             'number':  [5, 6, 3, 8, 2, 6]})
        tmpl = self.p2s.piep(df, 'type', count='number',
                             color=('number', self.p2s.CMAGNITUDE_SUMp), sm_shared=sm_shared)
        return self.p2s.smallp(df, 'shelter', tmpl)._render_lu_

    def test_sm_color_shares_the_spectrum_range(self):
        '''SM_COLOR colours every panel against one range, so a value is one colour
        everywhere (PLANNING.md §5 C-piep-sm-color-range: each panel used its own).'''
        panels = self._spectrumPanels_({self.p2s.SM_COLOR})
        ranges = {(p._color_stat_min_, p._color_stat_max_) for p in panels.values()}
        self.assertEqual(ranges, {(2.0, 14.0)})

    def test_sm_count_does_not_share_the_spectrum_range(self):
        panels = self._spectrumPanels_({self.p2s.SM_COUNT})
        self.assertEqual({k[0]: (p._color_stat_min_, p._color_stat_max_) for k, p in panels.items()},
                         {'a': (5.0, 6.0), 'b': (2.0, 8.0), 'c': (6.0, 6.0)})

    # ── binCounts(): a panel's counts without a render ─────────────────────────

    def test_bin_counts_match_a_rendered_panel(self):
        for count in (self.p2s.ROW_COUNTp, 'value', ('value', self.p2s.SETp)):
            with self.subTest(count=count):
                tmpl = self.p2s.piep(self.df, 'cat', count=count)
                sub  = self.df.filter(pl.col('group') == 'x')
                self.assertEqual(tmpl.binCounts(sub), tmpl.render_with(sub)._raw_count_lu_)

    def test_bin_counts_are_raw_and_known(self):
        tmpl = self.p2s.piep(self.df, ('cat', 'group'), min_slice_deg=180.0)
        df   = pl.DataFrame({'cat': ['A', 'A', 'B', 'C'], 'group': ['x', 'x', 'x', 'y'],
                             'value': [1, 2, 3, 4], 'score': [0.0] * 4})
        sep  = self.p2s.MULTI_FIELD_SEP
        # min_slice_deg would fold B and C into (other); binCounts reports them raw
        self.assertEqual(tmpl.binCounts(df), {f'A{sep}x': 2.0, f'B{sep}x': 1.0, f'C{sep}y': 1.0})

    def test_bin_counts_do_not_render(self):
        tmpl  = self.p2s.piep(self.df, 'cat')
        probe = Piep(df=self.df, template=tmpl, p2s=self.p2s, _counts_only_=True)
        self.assertIn('__computeAggregates__', probe.timing_metrics)
        self.assertNotIn('__renderSVG__',       probe.timing_metrics)
        # and a render cloned from the probe is a full render again
        self.assertIn('__renderSVG__', probe.render_with(self.df).timing_metrics)

    # ── SM_PARTOFWHOLEp: faded base + filled share ─────────────────────────────

    def test_part_of_whole_sets_base_slices(self):
        tmpl = self.p2s.piep(self.df, 'cat', wxh=(110, 110),
                             sm_shared={self.p2s.SM_PARTOFWHOLEp})
        renders = tmpl.renderSmallMultiples(self.df, {
            'x': self.df.filter(self.df['group'] == 'x'),
        }, '__all__')
        panel = renders['x']
        self.assertIsNotNone(panel._base_slices_)
        # every slice carries the all-rows total and the panel's own count
        for s in panel._slices_:
            self.assertIn('count_all', s)
            self.assertLessEqual(s['count'], s['count_all'] + 1e-9)

    def test_part_of_whole_base_angles_identical_across_panels(self):
        '''Every part-of-whole panel shares the same base slice angles (the "all rows" chart).'''
        tmpl = self.p2s.piep(self.df, 'cat', wxh=(110, 110),
                             sm_shared={self.p2s.SM_PARTOFWHOLEp})
        renders = tmpl.renderSmallMultiples(self.df, {
            'x': self.df.filter(self.df['group'] == 'x'),
            'y': self.df.filter(self.df['group'] == 'y'),
        }, '__all__')
        angles_x = {s['bin']: (round(s['a0'], 3), round(s['a1'], 3)) for s in renders['x']._slices_}
        angles_y = {s['bin']: (round(s['a0'], 3), round(s['a1'], 3)) for s in renders['y']._slices_}
        self.assertEqual(angles_x, angles_y)

    def test_part_of_whole_renders_all_styles(self):
        for style in (self.p2s.PIEp, self.p2s.DONUTp, self.p2s.WAFFLEp):
            tmpl = self.p2s.piep(self.df, 'cat', wxh=(110, 110), style=style,
                                 sm_shared={self.p2s.SM_PARTOFWHOLEp})
            sm = self.p2s.smallp(self.df, 'group', tmpl, include_all=True)
            assert_valid_svg(self, sm._repr_svg_())

    def test_shared_other_fold_is_consistent_across_panels(self):
        '''The "all rows" reference decides which bins fold into (other); every panel
        folds the same categories and totals its own rows for them.'''
        df = pl.DataFrame({
            'cat':   ['A'] * 90 + ['B'] * 90 + ['C', 'D', 'E', 'F'] * 5,
            'group': (['x'] * 45 + ['y'] * 45) * 2 + ['x', 'y'] * 10,
        })
        tmpl = self.p2s.piep(df, 'cat', wxh=(120, 120), min_slice_deg=12.0,
                             sm_shared={self.p2s.SM_SLICE_ORDERp})
        renders = tmpl.renderSmallMultiples(df, {
            'x': df.filter(df['group'] == 'x'),
            'y': df.filter(df['group'] == 'y'),
        }, '__all__')
        # both panels expose (other), and its count equals this panel's rows in the folded cats
        ref_bins = set(b for b in self.p2s.piep(df, 'cat', min_slice_deg=12.0)._sorted_bins_)
        self.assertIn('(other)', ref_bins)
        folded = {b for b in ['A', 'B', 'C', 'D', 'E', 'F'] if b not in ref_bins}
        for key, panel in renders.items():
            lu = {s['bin']: s['count'] for s in panel._slices_}
            self.assertIn('(other)', lu)
            sub = df.filter(df['group'] == key)
            expected = sub.filter(pl.col('cat').is_in(list(folded))).height
            self.assertEqual(lu['(other)'], float(expected))

    def _filledWedges_(self, style):
        '''(panel, [(r0, r1, a0, a1), ...]) for every full-opacity wedge each part-of-whole
        panel draws, i.e. its fills, not its faded base.'''
        _calls_ = []
        _orig_  = Piep.__wedge__
        def _spy_(this, dl, r0, r1, a0, a1, fill=None, opacity=1.0, **kw):
            if opacity == 1.0: _calls_.append((this, (r0, r1, a0, a1)))
            return _orig_(this, dl, r0, r1, a0, a1, fill=fill, opacity=opacity, **kw)
        tmpl = self.p2s.piep(self.df, 'cat', wxh=(110, 110), style=style,
                             sm_shared={self.p2s.SM_PARTOFWHOLEp})
        with mock.patch.object(Piep, '__wedge__', _spy_):
            panels = self.p2s.smallp(self.df, 'group', tmpl)._render_lu_
        return {k: [w for p, w in _calls_ if p is v] for k, v in panels.items()}, panels

    def test_part_of_whole_donut_fills_by_arc(self):
        '''A donut fills its share of each slice's sweep at full ring thickness
        (DT 2026-10-11, D5), not the whole sweep at a thinner ring.'''
        wedges, panels = self._filledWedges_(self.p2s.DONUTp)
        for key, panel in panels.items():
            with self.subTest(panel=key):
                expected = [(panel.r_inner, panel.r, s['a0'], s['a0'] + (s['a1'] - s['a0']) * s['count'] / s['count_all'])
                            for s in panel._slices_ if s['count'] > 0]
                self.assertEqual(len(wedges[key]), len(expected))
                for got, want in zip(wedges[key], expected):
                    for g, w in zip(got, want): self.assertAlmostEqual(g, w, places=9)
                self.assertTrue(any(w[3] < s['a1'] - 1e-6 for w, s in zip(wedges[key], panel._slices_)))

    def test_part_of_whole_pie_fills_by_area(self):
        wedges, panels = self._filledWedges_(self.p2s.PIEp)
        for key, panel in panels.items():
            with self.subTest(panel=key):
                expected = [(0.0, panel.r * (s['count'] / s['count_all']) ** 0.5, s['a0'], s['a1'])
                            for s in panel._slices_ if s['count'] > 0]
                self.assertEqual(len(wedges[key]), len(expected))
                for got, want in zip(wedges[key], expected):
                    for g, w in zip(got, want): self.assertAlmostEqual(g, w, places=9)

    # ── SM_COUNT: one base pie, each slice its largest count (DT 2026-10-11) ───

    _SHELTERS_ = pl.DataFrame({'shelter': ['a', 'a', 'b', 'b', 'b', 'c'],
                               'type':    ['cat', 'dog', 'cat', 'dog', 'ferret', 'cat'],
                               'number':  [5, 6, 3, 8, 2, 6]})

    def _smCount_(self, include_all=False, style=None, sm=None, **kwargs):
        tmpl = self.p2s.piep(self._SHELTERS_, 'type', count='number', color='type',
                             style=style or self.p2s.PIEp,
                             sm_shared=sm if sm is not None else {self.p2s.SM_COUNT}, **kwargs)
        return self.p2s.smallp(self._SHELTERS_, 'shelter', tmpl, include_all=include_all)._render_lu_

    @staticmethod
    def _base_(panel):
        return {s['bin']: (s['count_all'], round(s['a0'], 6), round(s['a1'], 6)) for s in panel._slices_}

    def test_sm_count_base_is_each_slices_largest_count(self):
        panels = self._smCount_()
        self.assertEqual(set(panels), {('a',), ('b',), ('c',)})
        for key, panel in panels.items():
            with self.subTest(panel=key):
                self.assertEqual({b: c for b, (c, _, _) in self._base_(panel).items()},
                                 {'dog': 8.0, 'cat': 6.0, 'ferret': 2.0})
                self.assertEqual(self._base_(panel), self._base_(panels[('a',)]))
        # the base is laid out like a pie of those counts: dog 8/16 of the ring, and so on
        sweeps = {s['bin']: s['a1'] - s['a0'] for s in panels[('a',)]._slices_}
        self.assertEqual({b: round(v, 6) for b, v in sweeps.items()},
                         {'dog': 180.0, 'cat': 135.0, 'ferret': 45.0})
        # each slice's largest value fills it; every other share is below 1
        fills = {(k[0], s['bin']): s['count'] / s['count_all'] for k, p in panels.items() for s in p._slices_}
        self.assertEqual({kb for kb, f in fills.items() if f == 1.0},
                         {('b', 'dog'), ('b', 'ferret'), ('c', 'cat')})
        self.assertTrue(all(0.0 <= f <= 1.0 for f in fills.values()))

    def test_sm_count_with_include_all_is_the_all_rows_base(self):
        counted = self._smCount_(include_all=True)
        whole   = self._smCount_(include_all=True, sm={self.p2s.SM_PARTOFWHOLEp})
        self.assertIn('__all__', counted)
        for key in counted:
            with self.subTest(panel=key):
                self.assertEqual({b: c for b, (c, _, _) in self._base_(counted[key]).items()},
                                 {'cat': 14.0, 'dog': 14.0, 'ferret': 2.0})
                self.assertEqual(self._base_(counted[key]), self._base_(whole[key]))
        self.assertTrue(all(s['count'] == s['count_all'] for s in counted['__all__']._slices_))

    def test_sm_count_filled_area_is_proportional_to_count_in_every_panel(self):
        '''A pie fills by area, so area / count is one constant across every panel and slice:
        the point of a shared base.'''
        _calls_, _orig_ = [], Piep.__wedge__
        def _spy_(this, dl, r0, r1, a0, a1, fill=None, opacity=1.0, **kw):
            if opacity == 1.0: _calls_.append((this, r1, a1 - a0))
            return _orig_(this, dl, r0, r1, a0, a1, fill=fill, opacity=opacity, **kw)
        with mock.patch.object(Piep, '__wedge__', _spy_):
            panels = self._smCount_()
        ratios = []
        for panel in panels.values():
            fills = [(r, sweep) for p, r, sweep in _calls_ if p is panel]
            counts = [s['count'] for s in panel._slices_ if s['count'] > 0]
            self.assertEqual(len(fills), len(counts))
            ratios += [(sweep / 360.0 * r * r) / c for (r, sweep), c in zip(fills, counts)]
        self.assertEqual(len(ratios), 6)
        for r in ratios: self.assertAlmostEqual(r, ratios[0], places=6)

    def test_sm_count_other_base_is_the_largest_folded_total(self):
        '''The folded slices' base is the largest total one panel has for them (3), not the
        sum of each member's largest count (3 + 3).'''
        df = pl.DataFrame({'cat': ['A'] * 40 + ['B'] * 40 + ['C'] * 3 + ['D'] * 3,
                           'grp': ['x', 'y'] * 40 + ['x'] * 3 + ['y'] * 3})
        tmpl = self.p2s.piep(df, 'cat', min_slice_deg=30.0, sm_shared={self.p2s.SM_COUNT})
        panels = self.p2s.smallp(df, 'grp', tmpl)._render_lu_
        for key, panel in panels.items():
            with self.subTest(panel=key):
                lu = {s['bin']: (s['count'], s['count_all']) for s in panel._slices_}
                self.assertEqual(lu, {'A': (20.0, 20.0), 'B': (20.0, 20.0), '(other)': (3.0, 3.0)})

    def test_sm_count_with_part_of_whole_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'SM_COUNT and SM_PARTOFWHOLEp'):
            self.p2s.piep(self._SHELTERS_, 'type',
                          sm_shared={self.p2s.SM_COUNT, self.p2s.SM_PARTOFWHOLEp})

    def test_a_base_slice_a_panel_lacks_keeps_its_colour(self):
        '''Shelter c has no dogs or ferrets; their faded base wedges take the colour those
        slices have elsewhere, so the base is the same in every panel.  They took the
        "(set)" colour, which also made dog and ferret one colour.'''
        for sm in ({self.p2s.SM_COUNT}, {self.p2s.SM_PARTOFWHOLEp}):
            with self.subTest(sm_shared=sm):
                panels = self._smCount_(sm=sm)
                bases  = {k: re.findall(r'fill="(#\w+)" fill-opacity', p.svg) for k, p in panels.items()}
                self.assertEqual(len({tuple(v) for v in bases.values()}), 1, bases)
                self.assertEqual(len(set(bases[('c',)])), 3)

    # ── base_opacity: the faded base pie ───────────────────────────────────────

    def _panelSvgs_(self, style, **kwargs):
        tmpl = self.p2s.piep(self.df, 'cat', wxh=(110, 110), style=style,
                             sm_shared={self.p2s.SM_PARTOFWHOLEp}, **kwargs)
        return [p.svg for p in self.p2s.smallp(self.df, 'group', tmpl)._render_lu_.values()]

    def test_base_opacity_default_is_unchanged(self):
        for style in (self.p2s.PIEp, self.p2s.DONUTp, self.p2s.WAFFLEp):
            with self.subTest(style=style):
                for svg in self._panelSvgs_(style):
                    self.assertIn('fill-opacity="0.220"', svg)

    def test_base_opacity_sets_the_faded_base(self):
        for style in (self.p2s.PIEp, self.p2s.DONUTp, self.p2s.WAFFLEp):
            with self.subTest(style=style):
                for svg in self._panelSvgs_(style, base_opacity=0.5):
                    self.assertIn('fill-opacity="0.500"', svg)
                    self.assertNotIn('fill-opacity="0.220"', svg)

    def test_base_opacity_is_validated(self):
        for bad in (-0.1, 1.5, 'half', True, None):
            with self.subTest(base_opacity=bad), self.assertRaises(ValueError):
                self.p2s.piep(self.df, 'cat', base_opacity=bad)

    def test_smallp_webgpu_payload(self):
        tmpl = self.p2s.piep(self.df, 'cat', wxh=(110, 110),
                             sm_shared={self.p2s.SM_PARTOFWHOLEp})
        sm = self.p2s.smallp(self.df, 'group', tmpl)
        payload = sm.webgpu()
        self.assertIn('buffers', payload)
        self.assertGreater(len(payload['manifest']), 0)


if __name__ == '__main__':
    unittest.main()
