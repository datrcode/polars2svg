import unittest
import polars as pl
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

    def test_smallp_webgpu_payload(self):
        tmpl = self.p2s.piep(self.df, 'cat', wxh=(110, 110),
                             sm_shared={self.p2s.SM_PARTOFWHOLEp})
        sm = self.p2s.smallp(self.df, 'group', tmpl)
        payload = sm.webgpu()
        self.assertIn('buffers', payload)
        self.assertGreater(len(payload['manifest']), 0)


if __name__ == '__main__':
    unittest.main()
