#
# Cross-representation tests for the Phase 2-5 WebGPU conversions:
# Timep, LinkP, ChP, SpreadLinesP, Smallp, plus the shared geometry helpers
# (cubicBezierSegmentsTable, flattenPathD, svgToDisplayList).
#
import datetime
import random
import unittest

import polars as pl

import polars2svg
from polars2svg.p2s_displaylist import (DisplayList, cubicBezierSegmentsTable,
                                        flattenPathD, svgToDisplayList,
                                        _rootViewBoxTransform_)
from webgpu_test_utils import (decode_buffer, hex_to_rgb01, manifest_count,
                               parse_svg_fill_rects)

_P2S_ = polars2svg.Polars2SVG()

random.seed(20260612)
_NODES_ = [f'n{i}' for i in range(15)]
_DF_ = pl.DataFrame({
    'ts':  [datetime.datetime(2026, 1, 1 + i % 28, i % 24) for i in range(300)],
    'cat': [random.choice('abcd') for _ in range(300)],
    'fm':  [random.choice(_NODES_) for _ in range(300)],
    'to':  [random.choice(_NODES_) for _ in range(300)],
    'val': [random.random() for _ in range(300)],
})
_POS_ = {n: (random.random(), random.random()) for n in _NODES_}


class TestGeometryHelpers(unittest.TestCase):
    def test_bezier_segments_endpoints_and_count(self):
        df = pl.DataFrame({'x0': [0.0], 'y0': [0.0], 'cx0': [10.0], 'cy0': [0.0],
                           'cx1': [20.0], 'cy1': [10.0], 'x1': [30.0], 'y1': [10.0]})
        seg = cubicBezierSegmentsTable(df, 'x0', 'y0', 'cx0', 'cy0', 'cx1', 'cy1', 'x1', 'y1', n=24)
        self.assertEqual(len(seg), 24)
        self.assertAlmostEqual(float(seg['__bx__'][0]),  0.0, places=6)
        self.assertAlmostEqual(float(seg['__by__'][0]),  0.0, places=6)
        self.assertAlmostEqual(float(seg['__bx2__'][-1]), 30.0, places=6)
        self.assertAlmostEqual(float(seg['__by2__'][-1]), 10.0, places=6)

    def test_bezier_segments_carry_columns(self):
        df = pl.DataFrame({'x0': [0.0, 5.0], 'y0': [0.0, 5.0], 'cx0': [1.0, 6.0], 'cy0': [0.0, 5.0],
                           'cx1': [2.0, 7.0], 'cy1': [1.0, 6.0], 'x1': [3.0, 8.0], 'y1': [1.0, 6.0],
                           'hex': ['#ff0000', '#00ff00']})
        seg = cubicBezierSegmentsTable(df, 'x0', 'y0', 'cx0', 'cy0', 'cx1', 'cy1', 'x1', 'y1', n=8)
        self.assertEqual(len(seg), 16)
        self.assertEqual(seg.filter(pl.col('hex') == '#ff0000').height, 8)

    def test_flatten_path_d(self):
        subpaths = flattenPathD('M 0 0 L 10 0 L 10 10 Z M 20 20 L 30 20')
        self.assertEqual(len(subpaths), 2)
        self.assertTrue(subpaths[0][1])    # closed
        self.assertFalse(subpaths[1][1])   # open
        self.assertEqual(subpaths[0][0][0], (0.0, 0.0))

    def test_flatten_path_d_samples_cubics(self):
        subpaths = flattenPathD('M 0 0 C 10 0 20 10 30 10', samples_per_curve=8)
        pts, closed = subpaths[0]
        self.assertEqual(len(pts), 9)
        self.assertEqual(pts[-1], (30.0, 10.0))


class TestSvgToDisplayList(unittest.TestCase):
    def _payload_(self, svg):
        dl = DisplayList(200, 200)
        svgToDisplayList(svg, dl, _P2S_)
        return dl.webgpu_payload(_P2S_.glyphAtlas())

    def test_rect_circle_line(self):
        p = self._payload_('<rect x="1" y="2" width="3" height="4" fill="#ff0000" />'
                           '<circle cx="5" cy="6" r="7" fill="#00ff00" />'
                           '<line x1="0" y1="0" x2="9" y2="9" stroke="#0000ff" stroke-width="2" />')
        self.assertEqual(manifest_count(p, 'rect'), 1)
        self.assertEqual(manifest_count(p, 'circle'), 1)
        self.assertEqual(manifest_count(p, 'line'), 1)
        rect = decode_buffer(p, 'rect')[0]
        self.assertEqual((float(rect[0]), float(rect[1]), float(rect[2]), float(rect[3])), (1.0, 2.0, 3.0, 4.0))

    def test_stroked_rect_becomes_lines(self):
        p = self._payload_('<rect x="0" y="0" width="10" height="10" fill="none" stroke="#000000" stroke-width="1" />')
        self.assertEqual(manifest_count(p, 'rect'), 0)
        self.assertEqual(manifest_count(p, 'line'), 4)

    def test_filled_path_becomes_tris(self):
        p = self._payload_('<path d="M 0 0 L 10 0 L 10 10 L 0 10 Z" fill="#808080" />')
        self.assertGreater(manifest_count(p, 'tri'), 0)

    def test_stroked_path_becomes_segments(self):
        p = self._payload_('<path d="M 0 0 L 10 0 L 10 10" stroke="#000000" fill="none" />')
        self.assertEqual(manifest_count(p, 'line'), 2)

    def test_text_with_rotation(self):
        p = self._payload_('<text x="10" text-anchor="middle" y="20" font-family="x" fill="#102030" '
                           'font-size="12px" transform="rotate(90,10,20)">Hi</text>')
        glyphs = decode_buffer(p, 'glyph')
        self.assertEqual(len(glyphs), 2)
        self.assertEqual((float(glyphs[0][0]), float(glyphs[0][1])), (10.0, 20.0))
        self.assertAlmostEqual(float(glyphs[0][6]), 0.0, places=6)   # cos(90)
        self.assertAlmostEqual(float(glyphs[0][7]), 1.0, places=6)   # sin(90)

    def test_defs_skipped(self):
        p = self._payload_('<defs><rect x="0" y="0" width="5" height="5" fill="#ff0000" /></defs>')
        self.assertEqual(p['manifest'], [])

    def test_viewbox_transform_identity_without_viewbox(self):
        self.assertEqual(_rootViewBoxTransform_('<svg width="100" height="50"></svg>'), (1.0, 0.0, 0.0))

    def test_viewbox_transform_meet_scaling(self):
        # 200x100 content into a 100x100 canvas -> scale 0.5, vertically centered
        s, tx, ty = _rootViewBoxTransform_('<svg width="100" height="100" viewBox="0 0 200 100"></svg>')
        self.assertAlmostEqual(s, 0.5)
        self.assertAlmostEqual(tx, 0.0)
        self.assertAlmostEqual(ty, 25.0)   # (100 - 100*0.5)/2

    def test_viewbox_applied_to_primitives(self):
        # a rect at viewBox x=100 maps to canvas x=50 under 0.5 scale
        dl = DisplayList(100, 100)
        svgToDisplayList('<svg width="100" height="100" viewBox="0 0 200 100">'
                         '<rect x="100" y="0" width="40" height="20" fill="#ff0000" /></svg>', dl, _P2S_)
        rect = decode_buffer(dl.webgpu_payload(_P2S_.glyphAtlas()), 'rect')[0]
        self.assertAlmostEqual(float(rect[0]), 50.0, places=4)   # 100 * 0.5
        self.assertAlmostEqual(float(rect[1]), 25.0, places=4)   # 0 * 0.5 + 25 letterbox
        self.assertAlmostEqual(float(rect[2]), 20.0, places=4)   # 40 * 0.5 (width scales)

    def test_spreadlines_primitives_fit_canvas(self):
        sl = _P2S_.spreadlinesp(_DF_, [('fm', 'to')], ego=_NODES_[0], time='ts')
        w, h = sl.wxh
        payload = sl.webgpu()
        for kind, stride, pairs in (('rect', 9, [(0, 1)]), ('circle', 12, [(0, 1)]),
                                    ('line', 11, [(0, 1), (2, 3)]), ('glyph', 16, [(0, 1)])):
            for row in decode_buffer(payload, kind):
                for (xi, yi) in pairs:
                    self.assertGreaterEqual(float(row[xi]), -2.0)
                    self.assertLessEqual(float(row[xi]), w + 2.0)
                    self.assertGreaterEqual(float(row[yi]), -2.0)
                    self.assertLessEqual(float(row[yi]), h + 2.0)


class TestTimepWebGPU(unittest.TestCase):
    def test_simple_rect_parity(self):
        tp = _P2S_.timep(_DF_, 'ts')
        payload = tp.webgpu()
        svg_rects = parse_svg_fill_rects(tp.svg)
        gpu_rects = decode_buffer(payload, 'rect')
        self.assertEqual(len(svg_rects), len(gpu_rects))
        for (sx, sy, sw, sh, sfill), row in zip(svg_rects, gpu_rects):
            self.assertAlmostEqual(sx, float(row[0]), delta=0.06)
            self.assertAlmostEqual(sy, float(row[1]), delta=0.06)
            if sfill.startswith('#'):
                self.assertAlmostEqual(hex_to_rgb01(sfill)[0], float(row[5]), places=5)

    def test_stacked_and_boxplot_produce_payloads(self):
        for kwargs in ({'style': _P2S_.STACKEDBARp, 'color': 'cat'},
                       {'style': _P2S_.BOXPLOTp,    'count': 'val'}):
            tp = _P2S_.timep(_DF_, 'ts', **kwargs)
            payload = tp.webgpu()
            self.assertGreater(sum(m['count'] for m in payload['manifest']), 1)

    def test_context_glyphs_present(self):
        tp = _P2S_.timep(_DF_, 'ts')
        self.assertGreater(manifest_count(tp.webgpu(), 'glyph'), 0)


class TestLinkPWebGPU(unittest.TestCase):
    def test_line_links_match_df_link(self):
        lp = _P2S_.linkp(df=_DF_, relationships=[('fm', 'to')], pos=_POS_)
        payload = lp.webgpu()
        # context-free render: lines = links + 4 border lines
        self.assertEqual(manifest_count(payload, 'line'), len(lp.df_link) + 4)
        self.assertEqual(manifest_count(payload, 'circle'), len(lp.df_node))

    def test_curve_links_flatten_24_segments_each(self):
        lp = _P2S_.linkp(df=_DF_, relationships=[('fm', 'to')], pos=_POS_, link_shape='curve')
        payload = lp.webgpu()
        self.assertEqual(manifest_count(payload, 'line'), len(lp.df_link) * 24 + 4)

    def test_flowmap_links_flatten_24_segments_each(self):
        lp = _P2S_.linkp(df=_DF_, relationships=[('fm', 'to')], pos=_POS_, link_shape='flowmap')
        payload = lp.webgpu()
        self.assertEqual(manifest_count(payload, 'line'), len(lp.df_link) * 24 + 4)

    def test_link_arrows_emit_one_tri_per_link(self):
        import polars as pl
        lp = _P2S_.linkp(df=_DF_, relationships=[('fm', 'to')], pos=_POS_, link_arrows=True)
        # 3 vertices per arrowhead triangle; links collapsed to a single screen
        # pixel have no direction and draw no arrow
        _sub_ = lp.df_link.drop_nulls(subset=['__rel0_fm_sx__', '__rel0_to_sx__'])
        _n_   = len(_sub_.filter((pl.col('__rel0_fm_sx__') != pl.col('__rel0_to_sx__')) |
                                 (pl.col('__rel0_fm_sy__') != pl.col('__rel0_to_sy__'))))
        self.assertGreater(_n_, 0)
        self.assertEqual(manifest_count(lp.webgpu(), 'tri'), _n_ * 3)

    def test_no_tris_without_link_arrows(self):
        lp = _P2S_.linkp(df=_DF_, relationships=[('fm', 'to')], pos=_POS_)
        self.assertEqual(manifest_count(lp.webgpu(), 'tri'), 0)

    def test_node_colors_match(self):
        lp = _P2S_.linkp(df=_DF_, relationships=[('fm', 'to')], pos=_POS_)
        circles = decode_buffer(lp.webgpu(), 'circle')
        hexes   = lp.df_node['__nc_hex__'].to_list()
        for hx, row in zip(hexes, circles):
            r, g, b = hex_to_rgb01(hx)
            self.assertAlmostEqual(r, float(row[4]), places=5)

    def test_labels_emit_glyphs(self):
        lp = _P2S_.linkp(df=_DF_, relationships=[('fm', 'to')], pos=_POS_, draw_labels=True)
        self.assertGreater(manifest_count(lp.webgpu(), 'glyph'), 0)


class TestChPWebGPU(unittest.TestCase):
    def test_curve_payload_has_segments_arrows_sectors(self):
        ch = _P2S_.chordp(df=_DF_, relationships=[('fm', 'to')], wxh=(192, 192))
        payload = ch.webgpu()
        # 24 segments per link + 4 border lines
        self.assertEqual(manifest_count(payload, 'line'), len(ch.df_link) * 24 + 4)
        self.assertGreater(manifest_count(payload, 'tri'), 0)   # arrows + node sectors

    def test_labels_radial_and_circular(self):
        for style in ('radial', 'circular'):
            ch = _P2S_.chordp(df=_DF_, relationships=[('fm', 'to')], wxh=(192, 192),
                              draw_labels=True, label_style=style)
            self.assertGreater(manifest_count(ch.webgpu(), 'glyph'), 0, style)

    def test_bundled_links_parse_to_primitives(self):
        ch = _P2S_.chordp(df=_DF_, relationships=[('fm', 'to')], wxh=(192, 192), link_shape='bundled')
        payload = ch.webgpu()
        self.assertGreater(manifest_count(payload, 'line'), 0)


class TestSpreadLinesPWebGPU(unittest.TestCase):
    def test_payload_covers_svg_primitives(self):
        sl = _P2S_.spreadlinesp(_DF_, [('fm', 'to')], ego=_NODES_[0], time='ts')
        payload = sl.webgpu()
        total = sum(m['count'] for m in payload['manifest'])
        self.assertGreater(total, 10)
        # circle count parity with the SVG's circle elements
        import re
        svg_circles = len(re.findall(r'<circle ', sl.svg))
        self.assertEqual(manifest_count(payload, 'circle'), svg_circles)

    def test_display_list_cached(self):
        sl = _P2S_.spreadlinesp(_DF_, [('fm', 'to')], ego=_NODES_[0], time='ts')
        self.assertIs(sl.webgpu(), sl.webgpu())


class TestSmallpWebGPU(unittest.TestCase):
    def _smallp_(self):
        tmpl = _P2S_.histop(_DF_, 'cat', wxh=(96, 96))
        return _P2S_.smallp(_DF_, 'fm', tmpl, wxh=(420, 280))

    def test_cells_compose_with_offsets(self):
        sm = self._smallp_()
        payload = sm.webgpu()
        # every cell contributes its own background rect, translated to the tile
        rects = decode_buffer(payload, 'rect')
        cell_origins = {(float(x), float(y)) for x, y in sm.category_to_xy.values()}
        rect_origins = {(float(r[0]), float(r[1])) for r in rects}
        for origin in cell_origins:
            self.assertIn(origin, rect_origins, f'cell background missing at {origin}')

    def test_labels_and_borders(self):
        sm = self._smallp_()
        payload = sm.webgpu()
        self.assertGreater(manifest_count(payload, 'glyph'), 0)   # cell labels (+ cell text)

    def test_smallpi_gpu_view(self):
        from polars2svg.interactive_controller import smallpi
        v = smallpi(self._smallp_(), use_webgpu=True)
        self.assertIsNotNone(v.gpu_payload)
        self.assertIn('gpucanvas', type(v)._template)


class TestXYpGradientLines(unittest.TestCase):
    def test_per_endpoint_lines_emit_per_vertex_tris(self):
        df = pl.DataFrame({'x': sorted(random.random() for _ in range(50)),
                           'y': [random.random() for _ in range(50)],
                           'sample': [random.choice('ab') for _ in range(50)],
                           'value': [random.random() for _ in range(50)]})
        xy = _P2S_.xyp(df, x='x', y='y', color='value', dot_size='value', opacity='value',
                       line=('sample', _P2S_.LINECOLOR_FIELD))
        payload = xy.webgpu()
        # one quad (6 indices) per gradient segment
        n_segments = xy.svg.count('<linearGradient')
        self.assertEqual(manifest_count(payload, 'tri'), n_segments * 6)


_POS_ = {n: (random.random(), random.random()) for n in _NODES_}


class TestLinkpiView(unittest.TestCase):
    def _linkp_(self):
        return _P2S_.linkp(df=_DF_, relationships=[('fm', 'to')], pos=_POS_)

    def test_svg_mode_has_no_gpu_params(self):
        from polars2svg.interactive_controller import linkpi
        v = linkpi(self._linkp_())
        self.assertNotIn('gpu_payload', type(v).param)
        self.assertNotIn('gpucanvas', type(v)._template)

    def test_gpu_mode_payload_and_canvas(self):
        from polars2svg.interactive_controller import linkpi
        v = linkpi(self._linkp_(), use_webgpu=True)
        self.assertIsInstance(v.gpu_payload, dict)
        self.assertIn('gpucanvas', type(v)._template)
        self.assertEqual(v.mod_inner, '')                       # plot is on the canvas
        self.assertIn('__P2S_GPU__', type(v)._scripts['render'])
        self.assertIn('gpu_payload', type(v)._scripts)

    def test_refresh_regenerates_payload_after_relayout(self):
        from polars2svg.interactive_controller import linkpi
        v = linkpi(self._linkp_(), use_webgpu=True)
        before = v.gpu_payload
        # mutate node positions like a layout op, mark dirty, refresh
        _ln_ = v.dfs_layout[v.df_level]
        for k in list(_ln_.pos)[:3]:
            _ln_.pos[k] = (_ln_.pos[k][0] + 0.1, _ln_.pos[k][1] - 0.1)
        _ln_.invalidateRender()
        v.__refreshView__()
        self.assertIsInstance(v.gpu_payload, dict)
        self.assertIsNot(v.gpu_payload, before)                 # fresh payload, not the cached one

    def test_use_webgpu_never_leaks_to_reactivehtml(self):
        from polars2svg.interactive_controller import linkpi
        linkpi(self._linkp_(), use_webgpu=True)                 # constructs without param error


class TestSpreadlinepiView(unittest.TestCase):
    def _spread_(self):
        return _P2S_.spreadlinesp(_DF_, [('fm', 'to')], ego=_NODES_[0], time='ts')

    def test_svg_mode_has_no_gpu_params(self):
        from polars2svg.spreadlinepi import spreadlinepi
        v = spreadlinepi(self._spread_())
        self.assertNotIn('gpu_payload', type(v).param)
        self.assertNotIn('gpucanvas', type(v)._template)

    def test_gpu_mode_payload_and_canvas(self):
        from polars2svg.spreadlinepi import spreadlinepi
        v = spreadlinepi(self._spread_(), use_webgpu=True)
        self.assertIsInstance(v.gpu_payload, dict)
        self.assertIn('gpucanvas', type(v)._template)
        self.assertEqual(v.mod_inner, '')
        self.assertIn('__P2S_GPU__', type(v)._scripts['render'])
        self.assertIn('gpu_payload', type(v)._scripts)

    def test_selection_regenerates_payload(self):
        from polars2svg.spreadlinepi import spreadlinepi
        v = spreadlinepi(self._spread_(), use_webgpu=True)
        before = v.gpu_payload
        v.selected_entities = {_NODES_[1], _NODES_[2]}
        v._apply_render_()
        self.assertIsInstance(v.gpu_payload, dict)
        self.assertIsNot(v.gpu_payload, before)


class TestPanelizeAllComponents(unittest.TestCase):
    def test_every_component_supports_use_webgpu(self):
        from polars2svg.interactive_controller import panelize
        tmpl = _P2S_.histop(_DF_, 'cat', wxh=(96, 96))
        plots = [
            _P2S_.xyp(_DF_, x='val', y='val', dot_size=2.0),
            _P2S_.histop(_DF_, 'cat'),
            _P2S_.timep(_DF_, 'ts'),
            _P2S_.chordp(df=_DF_, relationships=[('fm', 'to')], wxh=(192, 192)),
            _P2S_.smallp(_DF_, 'fm', tmpl, wxh=(420, 280)),
            # LinkP + SpreadLinesP: regression guard for the latent panelize crash
            _P2S_.linkp(df=_DF_, relationships=[('fm', 'to')], pos=_POS_),
            _P2S_.spreadlinesp(_DF_, [('fm', 'to')], ego=_NODES_[0], time='ts'),
        ]
        panel = panelize([plots], use_webgpu=True)
        views = []
        def _walk_(c):
            for ch in c:
                if hasattr(ch, 'objects'): _walk_(ch)
                else: views.append(ch)
        _walk_(panel)
        self.assertEqual(len(views), len(plots))
        for v in views:
            self.assertIsNotNone(getattr(v, 'gpu_payload', None), type(v).__name__)


if __name__ == '__main__':
    unittest.main()
