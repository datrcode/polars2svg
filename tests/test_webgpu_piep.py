#
# Cross-representation parity tests for Piep: the WebGPU payload buffers must
# describe the same geometry/colors as the SVG string of the same instance,
# without needing a browser.
#
import unittest
import re

import polars as pl

import polars2svg
from webgpu_test_utils import decode_buffer, manifest_count, parse_svg_fill_rects

_P2S_ = polars2svg.Polars2SVG()

_DF_ = pl.DataFrame({
    'cat': ['alpha', 'beta', 'gamma', 'alpha', 'beta', 'alpha', 'delta', 'gamma'],
    'sub': ['x', 'y', 'x', 'y', 'x', 'x', 'y', 'y'],
    'val': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
})


class TestPiepWebGPUParity(unittest.TestCase):
    def _svg_path_count(self, svg):
        return len(re.findall(r'<path\b', svg))

    def test_filled_pie_wedges_become_triangles(self):
        '''A filled (color=) pie records filled polygons → tri buffer must be populated.'''
        t = _P2S_.piep(_DF_, 'cat', color='cat')
        payload = t.webgpu()
        self.assertIn('tri_v', payload['buffers'])
        self.assertIn('tri_i', payload['buffers'])
        self.assertGreater(self._svg_path_count(t.svg), 0)

    def test_none_mode_wedges_fill_and_stroke(self):
        '''Default color=None fills AND strokes slices → GPU triangles and lines.'''
        t = _P2S_.piep(_DF_, 'cat')
        payload = t.webgpu()
        self.assertIn('tri_v', payload['buffers'])
        self.assertGreater(manifest_count(payload, 'tri'), 0)
        self.assertGreater(manifest_count(payload, 'line'), 0)

    def test_filled_donut_wedges_become_triangles(self):
        t = _P2S_.piep(_DF_, 'cat', color='cat', style=_P2S_.DONUTp)
        payload = t.webgpu()
        self.assertIn('tri_v', payload['buffers'])

    def test_waffle_rect_parity(self):
        '''Waffle cells are rects: the GPU rect count must match the SVG fill rects.'''
        t = _P2S_.piep(_DF_, 'cat', style=_P2S_.WAFFLEp, waffle_n=8)
        payload = t.webgpu()
        svg_rects = parse_svg_fill_rects(t.svg)
        gpu_rects = decode_buffer(payload, 'rect')
        self.assertEqual(len(svg_rects), len(gpu_rects))
        for (sx, sy, sw, sh, sfill), row in zip(svg_rects, gpu_rects):
            self.assertAlmostEqual(sx, float(row[0]), delta=0.02)
            self.assertAlmostEqual(sy, float(row[1]), delta=0.02)
            self.assertAlmostEqual(sw, float(row[2]), delta=0.02)

    def test_paint_order_background_first(self):
        t = _P2S_.piep(_DF_, 'cat')
        manifest = t.webgpu()['manifest']
        self.assertEqual(manifest[0]['kind'], 'rect')
        bg = decode_buffer(t.webgpu(), 'rect')[0]
        self.assertEqual((float(bg[0]), float(bg[1])), (0.0, 0.0))
        self.assertEqual((float(bg[2]), float(bg[3])), (float(t.wxh[0]), float(t.wxh[1])))

    def test_context_text_renders_glyphs(self):
        t = _P2S_.piep(_DF_, 'cat')
        payload = t.webgpu()
        self.assertIn('atlas', payload)
        self.assertGreater(len(decode_buffer(payload, 'glyph')), 0)

    def test_no_context_means_no_glyphs(self):
        t = _P2S_.piep(_DF_, 'cat', draw_context=False)
        payload = t.webgpu()
        self.assertEqual(manifest_count(payload, 'glyph'), 0)

    def test_template_rerender_payload_tracks_new_df(self):
        t1 = _P2S_.piep(_DF_, 'cat', color='cat')
        c1 = manifest_count(t1.webgpu(), 'tri')
        t2 = t1.render_with(_DF_.filter(pl.col('cat') == 'alpha'))
        c2 = manifest_count(t2.webgpu(), 'tri')
        # a single slice needs no more triangles than the full four-slice pie
        self.assertLessEqual(c2, c1)

    def test_part_of_whole_gpu_renders(self):
        tmpl = _P2S_.piep(_DF_, 'cat', color='cat', sm_shared={_P2S_.SM_PARTOFWHOLEp})
        renders = tmpl.renderSmallMultiples(_DF_, {'x': _DF_.filter(pl.col('sub') == 'x')}, '__all__')
        payload = renders['x'].webgpu()
        self.assertIn('tri_v', payload['buffers'])


if __name__ == '__main__':
    unittest.main()
