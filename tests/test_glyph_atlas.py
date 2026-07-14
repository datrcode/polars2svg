import math
import unittest

import polars2svg
from polars2svg.p2s_glyph_atlas import GlyphAtlas


class TestGlyphAtlasLayout(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p2s   = polars2svg.Polars2SVG()
        cls.atlas = GlyphAtlas()

    #
    # The critical invariant: GPU glyph runs must occupy exactly the horizontal extent
    # that textLength() reports, because every SVG layout decision (cropText,
    # svgAxisLabels, axis label fitting) is made through textLength().
    #
    def test_run_extent_matches_textLength(self):
        for txt in ['hello world', 'Rows', '+12 more', '0.25', 'category|field']:
            for txt_h in [9.6, 12, 16]:
                glyphs = self.atlas.layoutText(txt, 0, 0, txt_h)
                size   = int(round(txt_h))
                font   = self.atlas._sizeFont_(size)
                # the final glyph's pen position + advance == full string advance
                total  = self.p2s.textLength(txt, txt_h)
                # last glyph dx must be < total; pen of a hypothetical next glyph == total
                self.assertGreater(len(glyphs), 0)
                self.assertAlmostEqual(float(font.getlength(txt)), float(total), places=4)
                for g in glyphs:
                    self.assertLessEqual(g[2], total + 1.0)   # dx within the run

    def test_pen_positions_use_prefix_advances(self):
        txt, txt_h = 'AVAVA', 12
        glyphs = self.atlas.layoutText(txt, 0, 0, txt_h)
        font   = self.atlas._sizeFont_(12)
        scale  = 12 / 48.0
        for i, g in enumerate(glyphs):
            expected_pen = float(font.getlength(txt[:i]))
            bearing      = self.atlas._glyphs_[txt[i]]['bx'] * scale
            self.assertAlmostEqual(g[2], expected_pen + bearing, places=4)

    def test_anchor_offsets(self):
        txt, txt_h = 'middle', 12
        total = self.p2s.textLength(txt, txt_h)
        g_start  = self.atlas.layoutText(txt, 0, 0, txt_h, anchor='start')
        g_middle = self.atlas.layoutText(txt, 0, 0, txt_h, anchor='middle')
        g_end    = self.atlas.layoutText(txt, 0, 0, txt_h, anchor='end')
        self.assertAlmostEqual(g_middle[0][2], g_start[0][2] - total / 2.0, places=4)
        self.assertAlmostEqual(g_end[0][2],    g_start[0][2] - total,       places=4)

    def test_rotation_matrix_values(self):
        g0  = self.atlas.layoutText('R', 10, 20, 12)[0]
        self.assertEqual((g0[6], g0[7]), (1.0, 0.0))
        g90 = self.atlas.layoutText('R', 10, 20, 12, rotation=90)[0]
        self.assertAlmostEqual(g90[6], 0.0, places=9)
        self.assertAlmostEqual(g90[7], 1.0, places=9)
        g270 = self.atlas.layoutText('R', 10, 20, 12, rotation=270)[0]
        self.assertAlmostEqual(g270[6], 0.0,  places=9)
        self.assertAlmostEqual(g270[7], -1.0, places=9)
        # origin is the anchor point, unrotated
        self.assertEqual((g90[0], g90[1]), (10.0, 20.0))

    def test_whitespace_advances_but_emits_no_quad(self):
        glyphs = self.atlas.layoutText('a b', 0, 0, 12)
        self.assertEqual(len(glyphs), 2)
        font = self.atlas._sizeFont_(12)
        # second glyph's pen reflects the space advance
        self.assertGreater(glyphs[1][2], float(font.getlength('a')))

    def test_uvs_within_unit_square(self):
        for g in self.atlas.layoutText('xyz', 0, 0, 12):
            u0, v0, u1, v1 = g[8], g[9], g[10], g[11]
            self.assertTrue(0.0 <= u0 < u1 <= 1.0)
            self.assertTrue(0.0 <= v0 < v1 <= 1.0)


class TestGlyphAtlasLifecycle(unittest.TestCase):
    def test_lazy_charset_extension_bumps_version(self):
        atlas = GlyphAtlas()
        v0 = atlas.version
        atlas.layoutText('abc', 0, 0, 12)        # ASCII -- already in the initial charset
        self.assertEqual(atlas.version, v0)
        glyphs = atlas.layoutText('héllo', 0, 0, 12)
        self.assertEqual(atlas.version, v0 + 1)  # é triggered a rebuild
        self.assertEqual(len(glyphs), 5)

    def test_existing_uvs_still_valid_after_rebuild(self):
        atlas = GlyphAtlas()
        before = atlas.layoutText('abc', 0, 0, 12)
        atlas.layoutText('é', 0, 0, 12)          # force rebuild
        after  = atlas.layoutText('abc', 0, 0, 12)
        # geometry (positions/sizes) identical; uvs may move but stay valid
        for b, a in zip(before, after):
            self.assertEqual(b[:8], a[:8])

    def test_payload_shape(self):
        atlas = GlyphAtlas()
        p = atlas.payload()
        self.assertIn('png_b64', p)
        self.assertGreater(p['w'], 0)
        self.assertGreater(p['h'], 0)
        self.assertEqual(p['version'], atlas.version)

    def test_deterministic_layout(self):
        a1, a2 = GlyphAtlas(), GlyphAtlas()
        self.assertEqual(a1.layoutText('determinism', 3, 4, 12),
                         a2.layoutText('determinism', 3, 4, 12))


if __name__ == '__main__':
    unittest.main()
