"""SpreadLinesP node-packing under density pressure.

As the number of alters in a time bin grows past what a single column of
circles can hold, the layout escalates: single column -> multi-strand columns
-> collapsed "cloud" pills showing a count.  These tests pin each regime and
the invariants that hold across all of them.
"""
import datetime
import re
import unittest
import xml.etree.ElementTree as ET

import polars as pl

from polars2svg import Polars2SVG
from svg_test_utils import normalize_svg
from webgpu_test_utils import decode_buffer


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

    # The render parses as XML, and its root is an <svg> of the size asked for
    def assertWellFormed(self, spread):
        svg = spread._repr_svg_()
        _root_ = ET.fromstring(svg)
        self.assertEqual(_root_.tag.rsplit('}', 1)[-1], 'svg')
        self.assertEqual((_root_.get('width'), _root_.get('height')), tuple(str(v) for v in spread.wxh))
        return svg

    # ── regimes ───────────────────────────────────────────────────────────────

    def test_sparse_bins_all_single_circles(self):
        sp = self._make(5)
        self.assertWellFormed(sp)
        self.assertEqual(set(_representations(sp)), {'single'})

    def test_medium_density_still_renders_individual_circles(self):
        sp = self._make(20)
        self.assertWellFormed(sp)
        self.assertEqual(set(_representations(sp)), {'single'})

    def test_high_density_collapses_to_clouds(self):
        sp = self._make(60)
        svg = self.assertWellFormed(sp)
        reps = _representations(sp)
        self.assertIn('cloud', reps)
        self.assertGreater(reps['cloud'], reps.get('single', 0))
        self.assertIn('rx="8"', svg)   # cloud pill shape

    def test_very_high_density_renders_wellformed(self):
        sp = self._make(150)
        self.assertWellFormed(sp)
        self.assertIn('cloud', _representations(sp))

    def test_a_short_widget_scales_the_drawing_to_fit(self):
        '''The same 20 alters that fit as circles at h=300 degrade gracefully when the
        widget is only 120px tall: the packing is unchanged -- still circles, no clouds --
        and the viewBox, fitted to the drawing, scales it into the smaller widget.  (This
        test used to be named for forcing clouds, which the packing never does.)'''
        sp  = self._make(20, wxh=(800, 120))
        svg = self.assertWellFormed(sp)
        self.assertEqual(_representations(sp), _representations(self._make(20)))
        _vx_, _vy_, _vw_, _vh_ = (float(v) for v in ET.fromstring(svg).get('viewBox').split())
        self.assertGreater(_vh_, 120, 'nothing needed scaling')
        for cx, cy, r in re.findall(r'<circle cx="([-\d.]+)" cy="([-\d.]+)" r="([\d.]+)"', svg):
            cx, cy, r = float(cx), float(cy), float(r)
            self.assertTrue(_vx_ <= cx - r and cx + r <= _vx_ + _vw_ and _vy_ <= cy - r and cy + r <= _vy_ + _vh_,
                            f'circle ({cx}, {cy}) r={r} is outside the viewBox')

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
        '''A highlighted node drawn as its own circle is drawn emphasised -- a wider ring
        and a near-opaque fill -- and nothing else in the picture changes but its cloud's
        pill (test_a_cloud_pill_shows_its_highlighted_members).

        Bin 0 is dense: n010 sits inside its cloud.  A third, sparse bin draws n001,
        n002 and the unhighlighted n003 as circles, so the test sees both sides.'''
        _df_ = pl.concat([_dense_df(60), pl.DataFrame({'fm': ['ego'] * 3, 'to': ['n001', 'n002', 'n003'],
                                                       'time': [datetime.datetime(2024, 1, 3)] * 3})])
        sp = self.p2s.spreadlinesp(_df_, [('fm', 'to')], ego='ego', time='time', wxh=(800, 300))
        _hl_ = {'n000', 'n001', 'n002', 'n010'}
        _hsp_ = sp.render_with(sp.df_orig, highlight_nodes=_hl_)
        svg = self.assertWellFormed(_hsp_)
        _singles_ = [(n, (f'{xy[0]:.1f}', f'{xy[1]:.1f}')) for _n2x_ in _hsp_.bin_to_node_to_xyrepstat.values()
                     for n, xy in _n2x_.items() if xy[2] == 'single' and n != 'ego']
        self.assertEqual({n for n, _ in _singles_}, {'n000', 'n001', 'n002', 'n003'})
        self.assertEqual({xy[2] for _n2x_ in _hsp_.bin_to_node_to_xyrepstat.values()
                          for n, xy in _n2x_.items() if n == 'n010'}, {'cloud'})
        # every circle a node is drawn as: (ring width, fill opacity), emphasised or plain
        _drawn_ = {(cx, cy): (sw, fo) for cx, cy, sw, fo in re.findall(
                       r'<circle cx="([-\d.]+)" cy="([-\d.]+)" r="[\d.]+" stroke="[^"]+" '
                       r'stroke-width="([\d.]+)" fill="[^"]+" fill-opacity="([\d.]+)"/>', svg)}
        for _n_, _xy_ in _singles_:
            self.assertEqual(_drawn_[_xy_], ('2.50', '0.80') if _n_ in _hl_ else ('1.25', '0.25'), _n_)
        # n010's pill takes the ring (see below); undo it first, then the circles
        _undone_ = re.sub(r'(<rect [^>]*width="32" [^>]*)stroke-width="2.50"', r'\1stroke-width="1"', svg)
        _undone_ = _undone_.replace('stroke-width="2.50"', 'stroke-width="1.25"').replace('fill-opacity="0.80"', 'fill-opacity="0.25"')
        self.assertEqual(normalize_svg(_undone_), normalize_svg(sp._repr_svg_()))

    def test_a_cloud_pill_shows_its_highlighted_members(self):
        '''A highlighted node packed into a cloud used to show nothing, so a selection
        arriving from a linked view vanished in a dense bin.  Decided 2026-09-29 (PLANNING.md
        §5 C-spreadlinesp-cloud-highlight): the pill takes a highlighted circle's emphasis.
        Every member highlighted: its ring (2.50) and fill (0.80).  Some: the ring alone.
        Bin 0's pill holds n001-n059; the GPU display list must say the same as the SVG.'''
        sp = self._make(60)
        _members_ = {n for n, xy in sp.bin_to_node_to_xyrepstat[0].items() if xy[2] == 'cloud' and n != 'ego'}
        self.assertEqual(len(_members_), 59)
        for _case_, _hl_, _want_ in (('none', set(),                        ('1',    '0.25')),
                                     ('some', {'n010'},                     ('2.50', '0.25')),
                                     ('some, and a circle', {'n000', 'n010'}, ('2.50', '0.25')),
                                     ('all',  _members_,                    ('2.50', '0.80')),
                                     ('all, and a circle', _members_ | {'n000'}, ('2.50', '0.80'))):
            with self.subTest(_case_):
                _hsp_  = sp.render_with(sp.df_orig, highlight_nodes=_hl_)
                _pill_ = re.findall(r'<rect [^>]*width="32" height="16" rx="8" [^>]*fill-opacity="([\d.]+)" '
                                    r'[^>]*stroke-width="([\d.]+)"/>', _hsp_.svg)
                self.assertEqual(_pill_, [(_want_[1], _want_[0])])
                # the display list: the pill's fill alpha, and a ring 2.5x the plain one
                _pay_  = _hsp_.webgpu()
                _rects_ = decode_buffer(_pay_, 'rect')
                _pills_ = _rects_[_rects_[:, 4] > 0]
                self.assertEqual(len(_pills_), 1)
                self.assertAlmostEqual(float(_pills_[0][8]), float(_want_[1]), places=3)
                _unit_  = float(_pills_[0][4]) / 8.0          # world -> canvas scale, from rx=8
                _ring_  = any(abs(float(w) - 2.5 * _unit_) < 1e-3 for w in decode_buffer(_pay_, 'line')[:, 4])
                self.assertEqual(_ring_, _want_[0] == '2.50')

    def test_dense_with_count_field(self):
        df = _dense_df(60).with_columns(pl.int_range(pl.len()).alias('w'))
        sp = self.p2s.spreadlinesp(df, [('fm', 'to')], ego='ego', time='time',
                                   count='w', wxh=(800, 300))
        self.assertWellFormed(sp)
        # every alter is still placed, in the same regime as without a count
        self.assertEqual(set(sp.bin_to_node_to_xyrepstat[0]) - {'ego', '__EGO__'}, {f'n{i:03d}' for i in range(60)})
        self.assertEqual(_representations(sp), _representations(self._make(60)))


if __name__ == '__main__':
    unittest.main()
