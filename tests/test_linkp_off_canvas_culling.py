#
# Off-canvas culling (PLANNING.md S2).
#
# Under zoom most of a linkp render is geometry outside the viewport, and those elements
# carry the file's widest coordinate tokens -- the half of "too many digits" that lives in
# front of the decimal, where no rounder reaches.  The emitters reject an element whose
# bounding box misses the canvas entirely.
#
# What these tests pin, in the order S2 names the risks:
#   * only *wholly* outside is dropped -- a straddling element keeps its visible part;
#   * a <textPath> label and the <defs> path it hrefs are culled as a pair;
#   * culling is invisible to hit-testing, which reads __sx__/__sy__ and never the SVG.
#
import math
import re
import unittest

import polars as pl
from polars2svg import Polars2SVG


# Six nodes strung along y=0.5.  With view_window=(0.4,0.4,0.6,0.6) only c and d land on
# the canvas; a/b sit far off the left edge and e/f far off the right.
_DF_ = pl.DataFrame({
    'fm':  ['a',  'b',  'c',  'd',  'e',  'f' ],
    'to':  ['b',  'c',  'd',  'e',  'f',  'a' ],
    'dsc': ['ab', 'bc', 'cd', 'de', 'ef', 'fa'],
})
_POS_ = {'a': (0.05, 0.5), 'b': (0.15, 0.5), 'c': (0.48, 0.5),
         'd': (0.52, 0.5), 'e': (0.85, 0.5), 'f': (0.95, 0.5)}
_REL_ = [('fm', 'to', 'dsc')]
_WXH_ = (400, 400)
_ZOOM_ = (0.4, 0.4, 0.6, 0.6)

_CIRCLE_RE_ = re.compile(r'<circle cx="([-\d.]+)" cy="([-\d.]+)"')
_LINE_RE_   = re.compile(r'<line x1="([-\d.]+)" y1="([-\d.]+)" x2="([-\d.]+)" y2="([-\d.]+)"')
_TEXT_RE_   = re.compile(r'<text x="([-\d.]+)" y="([-\d.]+)"')


def _circles(svg):  return [(float(x), float(y)) for x, y in _CIRCLE_RE_.findall(svg)]
def _lines(svg):    return [tuple(float(v) for v in m) for m in _LINE_RE_.findall(svg)]


class TestLinkPOffCanvasCulling(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def _zoomed(self, **extra):
        _kw_ = dict(df=_DF_, relationships=_REL_, pos=_POS_, wxh=_WXH_, **extra)
        _kw_.setdefault('view_window', _ZOOM_)
        return self.p2s.linkp(**_kw_)

    def _screen(self, lp):
        '''node name -> (sx, sy), read from the same columns recordsAt() hit-tests.'''
        _lu_ = {}
        for _sx_, _sy_, _nms_ in lp.df_node.select('__sx__', '__sy__', '__nm__').iter_rows():
            for _nm_ in _nms_:
                _lu_[_nm_] = (float(_sx_), float(_sy_))
        return _lu_

    # ── the fixture itself ────────────────────────────────────────────────────
    def test_fixture_actually_puts_nodes_off_canvas(self):
        '''Everything below is vacuous if the zoom stopped moving nodes off the canvas.'''
        lp = self._zoomed()
        _w_, _h_ = _WXH_
        _off_ = [n for n, (x, y) in self._screen(lp).items()
                 if x < 0 or x > _w_ or y < 0 or y > _h_]
        self.assertEqual(sorted(_off_), ['a', 'b', 'e', 'f'])

    # ── nodes ─────────────────────────────────────────────────────────────────
    def test_off_canvas_nodes_are_not_emitted(self):
        lp = self._zoomed()
        _on_ = {self._screen(lp)[n] for n in ('c', 'd')}
        self.assertEqual(set(_circles(lp.svg)), _on_)

    def test_the_retained_node_table_still_holds_every_node(self):
        '''The cull filters emission, not state: df_node feeds recordsAt() and the GPU.'''
        lp = self._zoomed()
        self.assertEqual(len(lp.df_node), 6)
        self.assertEqual(len(lp.df_link), 6)

    def test_an_unzoomed_render_emits_every_node(self):
        '''Culling is a no-op when the whole graph fits -- the goldens depend on it.'''
        lp = self.p2s.linkp(df=_DF_, relationships=_REL_, pos=_POS_, wxh=_WXH_)
        self.assertEqual(len(_circles(lp.svg)), 6)

    # ── links ─────────────────────────────────────────────────────────────────
    def test_links_wholly_off_canvas_are_dropped_and_straddlers_kept(self):
        lp   = self._zoomed()
        _sc_ = self._screen(lp)
        _emitted_ = {(round(x1), round(y1), round(x2), round(y2))
                     for x1, y1, x2, y2 in _lines(lp.svg)}

        def _seg_(fm, to):
            (x1, y1), (x2, y2) = _sc_[fm], _sc_[to]
            return (round(x1), round(y1), round(x2), round(y2))

        # a->b is off the left edge, e->f off the right: neither can be seen
        for _fm_, _to_ in (('a', 'b'), ('e', 'f')):
            self.assertNotIn(_seg_(_fm_, _to_), _emitted_, f'{_fm_}->{_to_} should be culled')
        # b->c and d->e cross the boundary; c->d is wholly inside
        for _fm_, _to_ in (('b', 'c'), ('c', 'd'), ('d', 'e'), ('f', 'a')):
            self.assertIn(_seg_(_fm_, _to_), _emitted_, f'{_fm_}->{_to_} should survive')

    # A right-to-left curve bows upward by mag/10, so an edge running just below the bottom
    # edge arcs back into view while both of its endpoints stay outside.  Testing endpoints
    # alone would cull a curve that is visibly drawn.
    def _bowing_curve_(self, wy0):
        _df_  = pl.DataFrame({'fm': ['a'], 'to': ['b']})
        _pos_ = {'a': (1.0, 0.0), 'b': (0.0, 0.0)}
        return self.p2s.linkp(df=_df_, relationships=[('fm', 'to')], pos=_pos_, wxh=(200, 200),
                              link_shape='curve', insets=(0, 0),
                              view_window=(0.0, wy0, 1.0, wy0 + 0.25))

    def test_a_curve_is_bounded_by_its_control_points_not_its_endpoints(self):
        '''The cull tests the convex hull of the control polygon, which is what actually
        bounds a cubic -- so a curve that bulges on-canvas survives both endpoints being
        off it.'''
        lp = self._bowing_curve_(0.0125)          # endpoints at y=210, control points at 190
        _fm_y_ = lp.df_link['__rel0_fm_sy__'][0]
        _cp_y_ = lp.df_link['__yo00__'][0]
        self.assertGreater(_fm_y_, 200, 'fixture no longer puts the endpoints off-canvas')
        self.assertLess(_cp_y_, 200,    'fixture no longer bows the curve back into view')
        self.assertIn('<path d="M', lp.svg)

    def test_a_curve_whose_whole_control_polygon_is_off_canvas_is_culled(self):
        '''The other side of the same fixture: push it far enough down that the bulge no
        longer reaches, and nothing is emitted.'''
        lp = self._bowing_curve_(0.1)
        self.assertGreater(lp.df_link['__yo00__'][0], 200)
        self.assertNotIn('<path d="M', lp.svg)

    # ── labels ────────────────────────────────────────────────────────────────
    def test_off_canvas_node_labels_are_not_emitted(self):
        lp = self._zoomed(draw_node_labels=True)
        _xs_ = {round(float(x)) for x, _ in _TEXT_RE_.findall(lp.svg)}
        _sc_ = self._screen(lp)
        self.assertEqual(_xs_, {round(_sc_['c'][0]), round(_sc_['d'][0])})

    def test_a_label_whose_baseline_is_off_canvas_but_whose_ink_is_not_survives(self):
        '''Node labels hang txt_h below the node, so a node near the bottom edge puts its
        baseline past h while the glyphs still rise into view.  That is a straddler.'''
        _df_  = pl.DataFrame({'fm': ['aaa'], 'to': ['bbb']})
        _pos_ = {'aaa': (0.2, 0.0), 'bbb': (0.8, 1.0)}
        lp = self.p2s.linkp(df=_df_, relationships=[('fm', 'to')], pos=_pos_, wxh=(200, 200),
                            draw_node_labels=True, insets=(0, 0))
        _ys_ = [float(y) for _, y in _TEXT_RE_.findall(lp.svg)]
        self.assertTrue(any(_y_ > 200 for _y_ in _ys_),
                        'fixture no longer places a baseline below the canvas')
        self.assertIn('bbb', lp.svg)

    def test_the_gpu_label_list_is_culled_with_the_svg(self):
        '''_node_label_info_ is what the GPU path draws; if it kept what the SVG dropped the
        two renderings of the same view would disagree.'''
        lp = self._zoomed(draw_node_labels=True)
        self.assertEqual(len(lp._node_label_info_), len(_TEXT_RE_.findall(lp.svg)))

    # ── link labels and their <defs> partner ──────────────────────────────────
    def test_textpath_labels_and_their_defs_paths_are_culled_as_a_pair(self):
        lp = self._zoomed(link_shape='curve', draw_link_labels=True)
        _defs_  = lp.svg[lp.svg.index('<defs>'):lp.svg.index('</defs>')] if '<defs>' in lp.svg else ''
        _ids_   = set(re.findall(r'<path id="([^"]+)"', _defs_))
        _hrefs_ = set(re.findall(r'<textPath href="#([^"]+)"', lp.svg))
        self.assertEqual(_ids_, _hrefs_, 'orphan <defs> path or dangling href after culling')

    def test_link_labels_are_actually_culled_by_the_zoom(self):
        _wide_ = self.p2s.linkp(df=_DF_, relationships=_REL_, pos=_POS_, wxh=_WXH_,
                                link_shape='curve', draw_link_labels=True)
        _zoom_ = self._zoomed(link_shape='curve', draw_link_labels=True)
        self.assertLess(len(_zoom_._link_label_svg_), len(_wide_._link_label_svg_))
        self.assertEqual(len(_zoom_._link_label_svg_), len(_zoom_._link_label_defs_))
        self.assertEqual(len(_zoom_._link_label_svg_), len(_zoom_._link_label_info_))

    # ── collapsed nodes: the S5 <defs> gate has to follow the cull ────────────
    def test_the_cloud_defs_go_when_every_collapsed_node_is_culled(self):
        '''S5 emits the icon only when a <use> references it.  A cull that removed the
        <use> but left the flag set would put an unreferenced 808-byte block back.'''
        _df_  = pl.DataFrame({'fm': ['a', 'b'], 'to': ['b', 'c']})
        _pos_ = {'a': (0.9, 0.9), 'b': (0.9, 0.9), 'c': (0.1, 0.1)}   # a,b collapse
        lp = self.p2s.linkp(df=_df_, relationships=[('fm', 'to')], pos=_pos_, wxh=(200, 200))
        self.assertIn('<use href="#cloud"', lp.svg)
        self.assertIn('id="cloud"', lp.svg)
        lp.setViewWindow((0.0, 0.0, 0.2, 0.2))                        # a,b now far off-canvas
        _svg_ = lp.renderSVG()
        self.assertNotIn('<use href="#cloud"', _svg_)
        self.assertNotIn('id="cloud"', _svg_)

    # ── timing marks ──────────────────────────────────────────────────────────
    def test_timing_marks_on_culled_edges_are_culled(self):
        import datetime
        _base_ = datetime.datetime(2024, 1, 1)
        _df_   = _DF_.with_columns(
            pl.Series('ts', [_base_ + datetime.timedelta(hours=i) for i in range(len(_DF_))])
        )
        _kw_ = dict(df=_df_, relationships=[('fm', 'to')], pos=_POS_, wxh=_WXH_, time='ts')
        _wide_ = self.p2s.linkp(**_kw_)
        _zoom_ = self.p2s.linkp(view_window=_ZOOM_, **_kw_)
        _mark_ = lambda lp: lp.svg.count('stroke-width="1.5"')
        self.assertGreater(_mark_(_wide_), 0)
        self.assertLess(_mark_(_zoom_), _mark_(_wide_))
        # the SVG marks and their GPU mirror stay the same set
        _tm_ = _zoom_._timing_mark_dl_table_
        self.assertEqual(0 if _tm_ is None else len(_tm_), _mark_(_zoom_))

    # ── the S2 gate: culling must be invisible to hit-testing ─────────────────
    def test_records_at_still_finds_a_culled_node(self):
        '''recordsAt() hit-tests the __sx__/__sy__ columns, not the emitted SVG, so a node
        the cull removed from the string is still selectable at its screen position.  This
        is the precondition S2 asked to be confirmed before relying on the cull.'''
        lp   = self._zoomed()
        _sx_, _sy_ = self._screen(lp)['a']
        self.assertNotIn((_sx_, _sy_), _circles(lp.svg))      # 'a' was culled from the SVG
        _hits_ = lp.recordsAt((_sx_, _sy_), threshold=5.0)
        self.assertGreater(len(_hits_), 0)
        _names_ = set(_hits_['fm'].to_list()) | set(_hits_['to'].to_list())
        self.assertIn('a', _names_)

    def test_entities_at_point_still_finds_a_culled_node(self):
        lp   = self._zoomed()
        _sx_, _sy_ = self._screen(lp)['f']
        self.assertIn('f', lp.entitiesAtPoint((_sx_, _sy_)))

    def test_moving_a_culled_node_back_on_canvas_makes_it_reappear(self):
        '''The cull is per-render, so a node that comes back into view comes back with it.'''
        lp = self._zoomed()
        self.assertEqual(len(_circles(lp.svg)), 2)
        lp.setViewWindow((0.0, 0.0, 1.0, 1.0))
        self.assertEqual(len(_circles(lp.renderSVG())), 6)

    # ── the GPU display list is built from the retained tables ────────────────
    def test_the_gpu_display_list_still_builds_after_a_cull(self):
        lp = self._zoomed(draw_node_labels=True)
        self.assertIsNotNone(lp.gpuDisplayList())


class TestLinkPZoomedCoordinateArithmetic(unittest.TestCase):
    '''Found while verifying the S2 cull: a zoomed render was emitting NaN coordinates.

    Screen coordinates are Int32, so a squared delta overflows past ~46,341 pixels of
    separation -- routine once a view_window magnifies the layout.  It wrapped negative,
    sqrt() made it NaN, and the NaN rode into the Bezier control points and out into
    d="M ... C NaN NaN NaN NaN ...", which no renderer draws.
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def _far_apart(self, **extra):
        # a zoom deep enough to put >46,341px between the two nodes
        _df_  = pl.DataFrame({'fm': ['a'], 'to': ['b']})
        _pos_ = {'a': (0.0, 0.0), 'b': (1.0, 1.0)}
        return self.p2s.linkp(df=_df_, relationships=[('fm', 'to')], pos=_pos_,
                              wxh=(400, 400), view_window=(0.4995, 0.4995, 0.5005, 0.5005), **extra)

    def test_the_fixture_really_does_overflow_int32(self):
        lp = self._far_apart(link_shape='curve')
        _dx_ = abs(lp.df_link['__rel0_to_sx__'][0] - lp.df_link['__rel0_fm_sx__'][0])
        self.assertGreater(_dx_ ** 2, 2 ** 31 - 1)

    # Asserting the length is *right* rather than merely non-NaN: a wrapped Int32 is
    # negative only about half the time, so 'no NaN' passes on the broken code whenever the
    # overflow happens to wrap positive -- and then the curve is drawn to a length that is
    # simply wrong.  Both failures are the same defect.
    def _chord_len_(self, lp):
        _d_  = lp.df_link
        _dx_ = float(_d_['__rel0_to_sx__'][0] - _d_['__rel0_fm_sx__'][0])
        _dy_ = float(_d_['__rel0_to_sy__'][0] - _d_['__rel0_fm_sy__'][0])
        return math.hypot(_dx_, _dy_)

    def test_the_curve_chord_length_is_computed_correctly(self):
        lp = self._far_apart(link_shape='curve')
        self.assertAlmostEqual(float(lp.df_link['__mag0__'][0]), self._chord_len_(lp), places=3)

    def test_the_arrowhead_direction_is_computed_correctly(self):
        lp = self._far_apart(link_shape='line', link_arrows=True)
        self.assertAlmostEqual(float(lp.df_link['__arr0_mag__'][0]), self._chord_len_(lp), places=3)

    def test_no_nan_reaches_the_svg(self):
        for _kw_ in (dict(link_shape='curve'), dict(link_shape='line', link_arrows=True)):
            with self.subTest(**_kw_):
                self.assertNotIn('NaN', self._far_apart(**_kw_).svg)


if __name__ == '__main__':
    unittest.main()
