import unittest

import numpy as np
import polars as pl

import polars2svg
from polars2svg.p2s_displaylist import (DisplayList, FLOATS_PER_INSTANCE,
                                        hexToRGBA, triangulatePolygon)
from webgpu_test_utils import decode_buffer


class TestHexToRGBA(unittest.TestCase):
    def test_rrggbb(self):
        self.assertEqual(hexToRGBA('#ff0000'), (1.0, 0.0, 0.0, 1.0))
        self.assertEqual(hexToRGBA('#000000'), (0.0, 0.0, 0.0, 1.0))

    def test_short_and_alpha_forms(self):
        self.assertEqual(hexToRGBA('#f00'), (1.0, 0.0, 0.0, 1.0))
        r, g, b, a = hexToRGBA('#ff000080')
        self.assertAlmostEqual(a, 128/255.0)

    def test_none_and_opacity(self):
        self.assertEqual(hexToRGBA('none'), (0.0, 0.0, 0.0, 0.0))
        self.assertEqual(hexToRGBA(None),   (0.0, 0.0, 0.0, 0.0))
        self.assertEqual(hexToRGBA('#ffffff', opacity=0.5)[3], 0.5)

    def test_unparseable_falls_back_to_gray(self):
        self.assertEqual(hexToRGBA('rebeccapurple'), (0.5, 0.5, 0.5, 1.0))
        self.assertEqual(hexToRGBA('#zzzzzz'),       (0.5, 0.5, 0.5, 1.0))


class TestTriangulatePolygon(unittest.TestCase):
    def test_triangle(self):
        self.assertEqual(len(triangulatePolygon([(0, 0), (1, 0), (0, 1)])), 1)

    def test_square_is_two_triangles(self):
        tris = triangulatePolygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        self.assertEqual(len(tris), 2)

    def test_closing_point_ignored(self):
        tris = triangulatePolygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
        self.assertEqual(len(tris), 2)

    def test_concave_polygon(self):
        # L-shape: 6 vertices -> 4 triangles
        pts = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]
        tris = triangulatePolygon(pts)
        self.assertEqual(len(tris), 4)
        # Total triangulated area must equal the polygon area (3.0)
        area = 0.0
        for i0, i1, i2 in tris:
            (x0, y0), (x1, y1), (x2, y2) = pts[i0], pts[i1], pts[i2]
            area += abs((x1-x0)*(y2-y0) - (x2-x0)*(y1-y0)) / 2.0
        self.assertAlmostEqual(area, 3.0)


class TestDisplayListSVGPassthrough(unittest.TestCase):
    def test_svg_strings_pass_through_byte_identical(self):
        dl = DisplayList(100, 100)
        s1 = '<rect x="1.0" y="2.0" width="3.0" height="4.0" fill="#ff0000" />'
        s2 = '<line x1="0" y1="0" x2="9" y2="9" stroke="#00ff00" stroke-width="2" />'
        r1 = dl.rect(1, 2, 3, 4, '#ff0000', svg=s1)
        r2 = dl.line(0, 0, 9, 9, '#00ff00', width=2, svg=s2)
        self.assertEqual(r1, s1)
        self.assertEqual(r2, s2)
        self.assertEqual(dl.svg(), s1 + s2)

    def test_empty_svg_records_gpu_only(self):
        dl = DisplayList(100, 100)
        dl.rect(1, 2, 3, 4, '#ff0000', svg='')
        self.assertEqual(dl.svg(), '')
        payload = dl.webgpu_payload()
        self.assertEqual(payload['manifest'], [{'kind': 'rect', 'first': 0, 'count': 1}])

    def test_raw_is_svg_only(self):
        dl = DisplayList(10, 10)
        dl.raw('<defs></defs>')
        self.assertEqual(dl.svg(), '<defs></defs>')
        self.assertEqual(dl.webgpu_payload()['manifest'], [])


class TestDisplayListPayload(unittest.TestCase):
    def test_buffer_strides(self):
        dl = DisplayList(100, 100)
        dl.rect(0, 0, 1, 1, '#102030')
        dl.circle(5, 5, 2, '#405060')
        dl.line(0, 0, 1, 1, '#708090')
        payload = dl.webgpu_payload()
        self.assertEqual(decode_buffer(payload, 'rect').shape,   (1, FLOATS_PER_INSTANCE['rect']))
        self.assertEqual(decode_buffer(payload, 'circle').shape, (1, FLOATS_PER_INSTANCE['circle']))
        self.assertEqual(decode_buffer(payload, 'line').shape,   (1, FLOATS_PER_INSTANCE['line']))

    def test_values_round_trip(self):
        dl = DisplayList(100, 100)
        dl.rect(1.5, 2.5, 3.5, 4.5, '#ff8000', rx=2.0, opacity=0.5)
        row = decode_buffer(dl.webgpu_payload(), 'rect')[0]
        self.assertAlmostEqual(row[0], 1.5)
        self.assertAlmostEqual(row[1], 2.5)
        self.assertAlmostEqual(row[2], 3.5)
        self.assertAlmostEqual(row[3], 4.5)
        self.assertAlmostEqual(row[4], 2.0)
        self.assertAlmostEqual(float(row[5]), 1.0,        places=6)
        self.assertAlmostEqual(float(row[6]), 128/255.0,  places=6)
        self.assertAlmostEqual(float(row[7]), 0.0,        places=6)
        self.assertAlmostEqual(float(row[8]), 0.5,        places=6)

    def test_consecutive_same_kind_ops_merge_into_one_batch(self):
        dl = DisplayList(100, 100)
        for i in range(5): dl.rect(i, 0, 1, 1, '#000000')
        payload = dl.webgpu_payload()
        self.assertEqual(payload['manifest'], [{'kind': 'rect', 'first': 0, 'count': 5}])

    def test_interleaved_kinds_preserve_paint_order(self):
        dl = DisplayList(100, 100)
        dl.rect(0, 0, 1, 1, '#000000')
        dl.line(0, 0, 1, 1, '#000000')
        dl.rect(1, 0, 1, 1, '#000000')
        manifest = dl.webgpu_payload()['manifest']
        self.assertEqual([m['kind'] for m in manifest], ['rect', 'line', 'rect'])
        self.assertEqual(manifest[0]['first'], 0)
        self.assertEqual(manifest[2]['first'], 1)   # second rect batch starts at instance 1

    def test_scissor_splits_batches_and_clamps(self):
        dl = DisplayList(100, 100)
        dl.rect(0, 0, 1, 1, '#000000')
        dl.rect(1, 0, 1, 1, '#000000', scissor=(-5.5, 10.2, 200, 20.7))
        manifest = dl.webgpu_payload()['manifest']
        self.assertEqual(len(manifest), 2)
        self.assertNotIn('scissor', manifest[0])
        # clamped to canvas bounds, floor/ceil to integers
        self.assertEqual(manifest[1]['scissor'], [0, 10, 100, 21])

    def test_tri_indices_offset_across_ops(self):
        dl = DisplayList(100, 100)
        dl.tris([0, 0, 1, 0, 0, 1], [0, 1, 2], (1, 0, 0, 1))
        dl.tris([5, 5, 6, 5, 5, 6], [0, 1, 2], (0, 1, 0, 1))
        payload = dl.webgpu_payload()
        idx = np.frombuffer(__import__('base64').b64decode(payload['buffers']['tri_i']), dtype='<u4')
        self.assertEqual(list(idx), [0, 1, 2, 3, 4, 5])
        manifest = [m for m in payload['manifest'] if m['kind'] == 'tri']
        self.assertEqual(manifest, [{'kind': 'tri', 'first': 0, 'count': 6}])


class TestDisplayListTables(unittest.TestCase):
    def test_rects_table_svg_col_and_buffer(self):
        df = pl.DataFrame({'x': [1.0, 2.0], 'y': [3.0, 4.0], 'h': [5.0, 6.0],
                           'r': [0.5, 0.25], 'g': [0.0, 0.0], 'b': [1.0, 1.0],
                           '__svg__': ['<rect a/>', '<rect b/>']})
        dl = DisplayList(50, 50)
        out = dl.rects_table(df, 'x', 'y', 7.0, 'h', ('r', 'g', 'b'))
        self.assertEqual(out, '<rect a/><rect b/>')
        self.assertEqual(dl.svg(), '<rect a/><rect b/>')
        arr = decode_buffer(dl.webgpu_payload(), 'rect')
        self.assertEqual(arr.shape[0], 2)
        self.assertAlmostEqual(float(arr[0][0]), 1.0)
        self.assertAlmostEqual(float(arr[1][3]), 6.0)   # h column
        self.assertAlmostEqual(float(arr[0][2]), 7.0)   # constant width
        self.assertAlmostEqual(float(arr[1][5]), 0.25)  # r color column

    def test_empty_table_is_noop(self):
        df = pl.DataFrame({'x': [], 'y': [], '__svg__': []},
                          schema={'x': pl.Float64, 'y': pl.Float64, '__svg__': pl.String})
        dl = DisplayList(50, 50)
        dl.circles_table(df, 'x', 'y', 1.0, (0.0, 0.0, 0.0))
        self.assertEqual(dl.webgpu_payload()['manifest'], [])


class TestDisplayListExtend(unittest.TestCase):
    def test_extend_offsets_scalar_and_line_coords(self):
        src = DisplayList(50, 50)
        src.rect(1, 2, 3, 4, '#000000')
        src.line(0, 0, 10, 10, '#000000')
        dst = DisplayList(100, 100)
        dst.extend(src, offset=(20, 30))
        payload = dst.webgpu_payload()
        rect = decode_buffer(payload, 'rect')[0]
        line = decode_buffer(payload, 'line')[0]
        self.assertEqual((float(rect[0]), float(rect[1])), (21.0, 32.0))
        self.assertEqual((float(line[0]), float(line[1]), float(line[2]), float(line[3])),
                         (20.0, 30.0, 30.0, 40.0))

    def test_extend_offsets_text_origin(self):
        p2s = polars2svg.Polars2SVG()
        src = DisplayList(50, 50)
        src.text(p2s, 'A', 5, 6, txt_h=12, svg='')
        dst = DisplayList(100, 100)
        dst.extend(src, offset=(10, 20))
        glyphs = decode_buffer(dst.webgpu_payload(p2s.glyphAtlas()), 'glyph')
        self.assertEqual(glyphs.shape[0], 1)
        self.assertEqual((float(glyphs[0][0]), float(glyphs[0][1])), (15.0, 26.0))

    def test_extend_scissor_override(self):
        src = DisplayList(50, 50)
        src.rect(1, 2, 3, 4, '#000000')
        dst = DisplayList(100, 100)
        dst.extend(src, scissor=(0, 0, 10, 10))
        manifest = dst.webgpu_payload()['manifest']
        self.assertEqual(manifest[0]['scissor'], [0, 0, 10, 10])

    def test_extend_does_not_copy_svg_by_default(self):
        src = DisplayList(50, 50)
        src.rect(1, 2, 3, 4, '#000000', svg='<rect/>')
        dst = DisplayList(100, 100)
        dst.extend(src)
        self.assertEqual(dst.svg(), '')


class TestDisplayListText(unittest.TestCase):
    def test_text_svg_matches_svgText(self):
        p2s = polars2svg.Polars2SVG()
        dl  = DisplayList(100, 100)
        out = dl.text(p2s, 'hello', 10, 20, txt_h=12, anchor='middle', color='#123456')
        self.assertEqual(out, p2s.svgText('hello', 10, 20, txt_h=12, anchor='middle', color='#123456'))
        self.assertEqual(dl.svg(), out)

    def test_payload_includes_atlas_only_with_text(self):
        p2s = polars2svg.Polars2SVG()
        dl  = DisplayList(100, 100)
        dl.rect(0, 0, 1, 1, '#000000')
        self.assertNotIn('atlas', dl.webgpu_payload(p2s.glyphAtlas()))
        dl.text(p2s, 'x', 5, 5, svg='')
        payload = dl.webgpu_payload(p2s.glyphAtlas())
        self.assertIn('atlas', payload)
        self.assertIn('png_b64', payload['atlas'])


if __name__ == '__main__':
    unittest.main()
