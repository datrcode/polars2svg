#
# Cross-representation parity tests for XYp: the WebGPU payload buffers must
# describe the same dots/lines/context as the SVG string of the same instance,
# without needing a browser.
#
import random
import unittest

import polars as pl

import polars2svg
from webgpu_test_utils import (decode_buffer, hex_to_rgb01, manifest_count,
                               parse_svg_circles)

_P2S_ = polars2svg.Polars2SVG()

random.seed(20260612)
_DF_ = pl.DataFrame({
    'x':   [random.random() * 100 for _ in range(400)],
    'y':   [random.random() * 100 for _ in range(400)],
    'cat': [random.choice(['a', 'b', 'c']) for _ in range(400)],
    'val': [random.random() for _ in range(400)],
})


class TestXYpWebGPUDots(unittest.TestCase):
    def test_circle_dot_parity(self):
        xy = _P2S_.xyp(_DF_, x='x', y='y', dot_size=2.0)
        payload = xy.webgpu()
        svg_circles = parse_svg_circles(xy.svg)
        gpu_circles = decode_buffer(payload, 'circle')
        self.assertEqual(len(svg_circles), len(gpu_circles))
        self.assertEqual(len(gpu_circles), len(xy.df_pixels))
        for (cx, cy, _fill), row in zip(svg_circles, gpu_circles):
            self.assertAlmostEqual(cx, float(row[0]), delta=0.06)
            self.assertAlmostEqual(cy, float(row[1]), delta=0.06)
            self.assertAlmostEqual(float(row[2]), 2.0, places=4)   # constant radius baked per-instance

    def test_circle_dot_crow_colors_match(self):
        xy = _P2S_.xyp(_DF_, x='x', y='y', dot_size=2.0, color=_P2S_.CROW_MAGNITUDEp)
        payload = xy.webgpu()
        gpu_circles = decode_buffer(payload, 'circle')
        hexes = xy.df_pixels['__hexcolor__'].to_list()
        self.assertEqual(len(hexes), len(gpu_circles))
        for hx, row in zip(hexes, gpu_circles):
            r, g, b = hex_to_rgb01(hx)
            self.assertAlmostEqual(r, float(row[4]), places=5)
            self.assertAlmostEqual(g, float(row[5]), places=5)
            self.assertAlmostEqual(b, float(row[6]), places=5)

    def test_rect_dot_mode_bakes_css_size(self):
        xy = _P2S_.xyp(_DF_, x='x', y='y', dot_size=3)
        payload = xy.webgpu()
        # last rect batch = the dots (background rect batch comes first)
        dots = decode_buffer(payload, 'rect')[1:]   # skip background
        self.assertEqual(len(dots), len(xy.df_pixels))
        for row in dots:
            self.assertEqual(float(row[2]), 3.0)
            self.assertEqual(float(row[3]), 3.0)

    def test_circle_dots_are_scissored_to_plot_clip(self):
        xy = _P2S_.xyp(_DF_, x='x', y='y', dot_size=2.0)
        manifest = xy.webgpu()['manifest']
        circle_entries = [m for m in manifest if m['kind'] == 'circle']
        self.assertEqual(len(circle_entries), 1)
        self.assertIn('scissor', circle_entries[0])
        cx0, cy0, cw, ch = xy._clip_rect_
        self.assertEqual(circle_entries[0]['scissor'][2], int(-(-cw // 1)))

    def test_rect_dots_are_not_scissored(self):
        xy = _P2S_.xyp(_DF_, x='x', y='y', dot_size=3)
        manifest = xy.webgpu()['manifest']
        dot_entries = [m for m in manifest if m['kind'] == 'rect' and m['first'] > 0]
        self.assertTrue(all('scissor' not in m for m in dot_entries))

    def test_payload_is_cached_until_rerender(self):
        xy = _P2S_.xyp(_DF_, x='x', y='y', dot_size=2.0)
        self.assertIs(xy.webgpu(), xy.webgpu())


class TestXYpWebGPUContextAndLines(unittest.TestCase):
    def test_context_emits_grid_lines_and_glyphs(self):
        xy = _P2S_.xyp(_DF_, x='x', y='y', dot_size=2.0)
        payload = xy.webgpu()
        self.assertGreaterEqual(manifest_count(payload, 'line'), 4)   # plot outline alone is 4
        self.assertGreater(manifest_count(payload, 'glyph'), 0)
        self.assertIn('atlas', payload)

    def test_no_context_no_glyphs(self):
        xy = _P2S_.xyp(_DF_, x='x', y='y', dot_size=2.0, draw_context=False)
        payload = xy.webgpu()
        self.assertEqual(manifest_count(payload, 'glyph'), 0)

    def test_lines_emit_segments(self):
        df = _DF_.sort('x')
        xy = _P2S_.xyp(df, x='x', y='y', line='cat')
        payload = xy.webgpu()
        # each of 3 lines with n points contributes n-1 segments
        per_line = df.group_by('cat').len()['len'].to_list()
        expected_segments = sum(n - 1 for n in per_line)
        # context grid/outline lines are also 'line' instances; segments are the rest
        self.assertGreaterEqual(manifest_count(payload, 'line'), expected_segments)

    def test_distribution_strip_emits_primitives(self):
        xy = _P2S_.xyp(_DF_, x='x', y='y', dot_size=2.0, x_distributions=_P2S_.ROW_COUNTp)
        payload = xy.webgpu()
        # distribution renders rects (bar mode) and/or lines (trend mode) after the dots
        kinds_after_dots = [m['kind'] for m in payload['manifest']]
        self.assertTrue(kinds_after_dots[-1] in ('rect', 'line'))


class TestXYpWebGPUBackground(unittest.TestCase):
    def test_polygon_background_produces_tris(self):
        bg = {'zone': [(10, 10), (90, 10), (90, 90), (10, 90)]}
        xy = _P2S_.xyp(_DF_, x='x', y='y', dot_size=2.0,
                       background=bg, background_fill='vary', background_label_color='vary')
        payload = xy.webgpu()
        self.assertIn('tri_v', payload['buffers'])
        self.assertGreater(manifest_count(payload, 'tri'), 0)


class TestWebgpuHTMLHelper(unittest.TestCase):
    def test_standalone_html_contains_canvas_and_runtime(self):
        xy = _P2S_.xyp(_DF_, x='x', y='y', dot_size=2.0)
        html = _P2S_.webgpuHTML(xy)
        self.assertIn('<canvas', html)
        self.assertIn('__P2S_GPU__', html)
        self.assertIn('"manifest"', html)

    def test_unsupported_component_raises(self):
        class _NotAComponent_: pass
        with self.assertRaises(ValueError):
            _P2S_.webgpuHTML(_NotAComponent_())


if __name__ == '__main__':
    unittest.main()
