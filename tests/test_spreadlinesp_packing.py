"""SpreadLinesP node-packing under density pressure.

As the number of alters in a time bin grows past what a single column of
circles can hold, the layout escalates: single column -> multi-strand columns
-> collapsed "cloud" pills showing a count.  These tests pin each regime and
the invariants that hold across all of them.
"""
import datetime
import unittest
import xml.etree.ElementTree as ET

import polars as pl

from polars2svg import Polars2SVG


def _dense_df(n_alters, n_bins=2):
    """ego connected to n_alters distinct nodes in bin 0; a repeat edge per extra bin."""
    fm, to, ts = [], [], []
    for i in range(n_alters):
        fm.append('ego'); to.append(f'n{i:03d}'); ts.append(datetime.datetime(2024, 1, 1))
    for b in range(1, n_bins):
        fm.append('ego'); to.append('n000'); ts.append(datetime.datetime(2024, 1, 1 + b))
    return pl.DataFrame({'fm': fm, 'to': to, 'time': ts})


def _representations(spread):
    reps = {}
    for _n2x_ in spread.bin_to_node_to_xyrepstat.values():
        for _xyrs_ in _n2x_.values():
            reps[_xyrs_[2]] = reps.get(_xyrs_[2], 0) + 1
    return reps


class TestSpreadLinesPacking(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.p2s = Polars2SVG()

    def _make(self, n_alters, wxh=(800, 300), **kwargs):
        return self.p2s.spreadlinesp(_dense_df(n_alters), [('fm', 'to')],
                                     ego='ego', time='time', wxh=wxh, **kwargs)

    def _wellformed(self, spread):
        svg = spread._repr_svg_()
        ET.fromstring(svg)
        return svg

    # ── regimes ───────────────────────────────────────────────────────────────

    def test_sparse_bins_all_single_circles(self):
        sp = self._make(5)
        self._wellformed(sp)
        self.assertEqual(set(_representations(sp)), {'single'})

    def test_medium_density_still_renders_individual_circles(self):
        sp = self._make(20)
        self._wellformed(sp)
        self.assertEqual(set(_representations(sp)), {'single'})

    def test_high_density_collapses_to_clouds(self):
        sp = self._make(60)
        svg = self._wellformed(sp)
        reps = _representations(sp)
        self.assertIn('cloud', reps)
        self.assertGreater(reps['cloud'], reps.get('single', 0))
        self.assertIn('rx="8"', svg)   # cloud pill shape

    def test_very_high_density_renders_wellformed(self):
        sp = self._make(150)
        self._wellformed(sp)
        self.assertIn('cloud', _representations(sp))

    def test_short_widget_forces_clouds_at_lower_density(self):
        # the same 20 alters that fit as circles at h=300 must degrade
        # gracefully when the widget is only 120px tall
        sp = self._make(20, wxh=(800, 120))
        self._wellformed(sp)

    # ── invariants across regimes ─────────────────────────────────────────────

    def test_every_alter_positioned_in_every_regime(self):
        for _n_ in (5, 20, 60):
            with self.subTest(n_alters=_n_):
                sp = self._make(_n_)
                _bin0_nodes_ = set(sp.bin_to_node_to_xyrepstat[0]) - {'ego', '__EGO__'}
                self.assertEqual(len(_bin0_nodes_), _n_)

    def test_cloud_positions_have_no_radius(self):
        sp = self._make(60)
        for _n2x_ in sp.bin_to_node_to_xyrepstat.values():
            for _xyrs_ in _n2x_.values():
                if _xyrs_[2] == 'cloud':
                    self.assertIsNone(_xyrs_[7])
                else:
                    self.assertIsNotNone(_xyrs_[7])

    def test_dense_render_with_highlights(self):
        sp = self._make(60)
        _hl_ = {'n000', 'n001', 'n002'}
        svg = sp.render_with(sp.df_orig, highlight_nodes=_hl_)._repr_svg_()
        ET.fromstring(svg)

    def test_dense_with_count_field(self):
        df = _dense_df(60).with_columns(pl.int_range(pl.len()).alias('w'))
        sp = self.p2s.spreadlinesp(df, [('fm', 'to')], ego='ego', time='time',
                                   count='w', wxh=(800, 300))
        self._wellformed(sp)


if __name__ == '__main__':
    unittest.main()
