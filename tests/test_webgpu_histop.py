#
# Cross-representation parity tests for Histop: the WebGPU payload buffers must
# describe the same geometry/colors as the SVG string of the same instance,
# without needing a browser.
#
import unittest

import polars as pl

import polars2svg
from webgpu_test_utils import (decode_buffer, hex_to_rgb01, manifest_count,
                               parse_svg_fill_rects)

_P2S_ = polars2svg.Polars2SVG()

_DF_ = pl.DataFrame({
    'cat': ['alpha', 'beta', 'gamma', 'alpha', 'beta', 'alpha', 'delta', 'gamma'],
    'sub': ['x', 'y', 'x', 'y', 'x', 'x', 'y', 'y'],
    'val': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
})


class TestHistopWebGPUParity(unittest.TestCase):
    def test_simple_rect_parity(self):
        h = _P2S_.histop(_DF_, 'cat')
        payload = h.webgpu()
        svg_rects = parse_svg_fill_rects(h.svg)
        gpu_rects = decode_buffer(payload, 'rect')
        self.assertEqual(len(svg_rects), len(gpu_rects))
        for (sx, sy, sw, sh, sfill), row in zip(svg_rects, gpu_rects):
            self.assertAlmostEqual(sx, float(row[0]), delta=0.06)   # svg uses :.1f
            self.assertAlmostEqual(sy, float(row[1]), delta=0.06)
            self.assertAlmostEqual(sw, float(row[2]), delta=0.06)
            self.assertAlmostEqual(sh, float(row[3]), delta=0.06)
            if sfill.startswith('#'):
                r, g, b = hex_to_rgb01(sfill)
                self.assertAlmostEqual(r, float(row[5]), places=5)
                self.assertAlmostEqual(g, float(row[6]), places=5)
                self.assertAlmostEqual(b, float(row[7]), places=5)

    def test_stacked_rect_parity(self):
        h = _P2S_.histop(_DF_, 'cat', color='sub', style=_P2S_.STACKEDBARp)
        payload = h.webgpu()
        svg_rects = parse_svg_fill_rects(h.svg)
        gpu_rects = decode_buffer(payload, 'rect')
        self.assertEqual(len(svg_rects), len(gpu_rects))
        for (sx, sy, sw, sh, sfill), row in zip(svg_rects, gpu_rects):
            self.assertAlmostEqual(sx, float(row[0]), delta=0.06)
            self.assertAlmostEqual(sy, float(row[1]), delta=0.06)
            if sfill.startswith('#'):
                r, g, b = hex_to_rgb01(sfill)
                self.assertAlmostEqual(r, float(row[5]), places=5)

    def test_context_text_renders_glyphs(self):
        h = _P2S_.histop(_DF_, 'cat')
        payload = h.webgpu()
        self.assertIn('atlas', payload)
        glyphs = decode_buffer(payload, 'glyph')
        self.assertGreater(len(glyphs), 0)
        w, hgt = h.wxh
        for g in glyphs:
            self.assertTrue(-1 <= g[0] <= w + 1,   f'glyph origin x {g[0]} outside canvas')
            self.assertTrue(-1 <= g[1] <= hgt + 1, f'glyph origin y {g[1]} outside canvas')

    def test_axis_border_becomes_lines(self):
        h = _P2S_.histop(_DF_, 'cat')
        payload = h.webgpu()
        # 4 grid lines + 4 border lines minimum when context is drawn
        self.assertGreaterEqual(manifest_count(payload, 'line'), 8)

    def test_paint_order_background_first(self):
        h = _P2S_.histop(_DF_, 'cat')
        manifest = h.webgpu()['manifest']
        self.assertEqual(manifest[0]['kind'], 'rect')
        self.assertEqual(manifest[0]['first'], 0)
        bg = decode_buffer(h.webgpu(), 'rect')[0]
        self.assertEqual((float(bg[0]), float(bg[1])), (0.0, 0.0))
        self.assertEqual((float(bg[2]), float(bg[3])), (float(h.wxh[0]), float(h.wxh[1])))

    def test_no_context_means_no_glyphs(self):
        h = _P2S_.histop(_DF_, 'cat', draw_context=False)
        payload = h.webgpu()
        self.assertEqual(manifest_count(payload, 'glyph'), 0)
        self.assertNotIn('atlas', payload)

    def test_template_rerender_payload_tracks_new_df(self):
        h1 = _P2S_.histop(_DF_, 'cat')
        c1 = manifest_count(h1.webgpu(), 'rect')
        h2 = h1.render_with(_DF_.filter(pl.col('cat') == 'alpha'))
        c2 = manifest_count(h2.webgpu(), 'rect')
        self.assertLess(c2, c1)

    def test_boxplot_runs_and_produces_primitives(self):
        h = _P2S_.histop(_DF_, 'cat', count='val', style=_P2S_.BOXPLOTp)
        payload = h.webgpu()
        self.assertGreater(manifest_count(payload, 'line'), 0)


if __name__ == '__main__':
    unittest.main()
