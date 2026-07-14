import unittest
from polars2svg import Polars2SVG
from histop_dataframes import makeHistoDf
from timep_dataframes import makeTimeDf


class TestBarChartTemplate(unittest.TestCase):
    """Template inheritance for histop and timep: a rendered object is passed as
    a positional arg or via template= to inherit all settings, with optional overrides."""

    @classmethod
    def setUpClass(cls):
        cls.p2s = Polars2SVG()
        cls.hdf = makeHistoDf(n=100)
        cls.tdf = makeTimeDf(n=100, year=(2021, 2024), month=(1, 12))

    def _histop(self, *args, **kw):
        return self.p2s.histop(*args, **kw) if args else self.p2s.histop(self.hdf, 'cat', **kw)

    def _timep(self, *args, **kw):
        return self.p2s.timep(*args, **kw) if args else self.p2s.timep(self.tdf, 'ts', **kw)

    # ── positional / keyword inheritance ──────────────────────────────────────

    def test_template_positional_inherits_settings(self):
        """Template passed positionally; new render inherits wxh and style."""
        h_orig = self._histop(wxh=(256, 512), style=self.p2s.BARCHARTp)
        h_copy = self.p2s.histop(h_orig)
        self.assertEqual(h_copy.wxh,   (256, 512))
        self.assertEqual(h_copy.style, self.p2s.BARCHARTp)

        t_orig = self._timep(wxh=(512, 256), style=self.p2s.BARCHARTp)
        t_copy = self.p2s.timep(t_orig)
        self.assertEqual(t_copy.wxh,   (512, 256))
        self.assertEqual(t_copy.style, self.p2s.BARCHARTp)

    def test_template_keyword_inherits_settings(self):
        """Template passed via template= kwarg."""
        h_orig = self._histop(wxh=(300, 400))
        self.assertEqual(self.p2s.histop(template=h_orig).wxh, (300, 400))

        t_orig = self._timep(wxh=(256, 128))
        self.assertEqual(self.p2s.timep(template=t_orig).wxh, (256, 128))

    # ── single-field overrides ────────────────────────────────────────────────

    def test_template_override_wxh(self):
        h_new = self.p2s.histop(self._histop(wxh=(256, 512)), wxh=(128, 256))
        self.assertEqual(h_new.wxh, (128, 256))

        t_new = self.p2s.timep(self._timep(wxh=(512, 256)), wxh=(128, 64))
        self.assertEqual(t_new.wxh, (128, 64))

    def test_template_override_style(self):
        h_new = self.p2s.histop(self._histop(style=self.p2s.BARCHARTp),
                                style=self.p2s.STACKEDBARp, color='group')
        self.assertEqual(h_new.style, self.p2s.STACKEDBARp)

        t_new = self.p2s.timep(self._timep(style=self.p2s.BARCHARTp),
                               style=self.p2s.BOXPLOTp, count='value')
        self.assertEqual(t_new.style, self.p2s.BOXPLOTp)

    def test_template_override_count_range(self):
        h_new = self.p2s.histop(self._histop(), count_range=(0, 999))
        self.assertEqual(h_new._count_min_, 0)
        self.assertEqual(h_new._count_max_, 999)

        t_new = self.p2s.timep(self._timep(), count_range=(0, 999))
        self.assertEqual(t_new._count_min_, 0)
        self.assertEqual(t_new._count_max_, 999)

    def test_template_override_draw_context(self):
        h_new = self.p2s.histop(self._histop(draw_context=True), draw_context=False)
        self.assertFalse(h_new.draw_context)

        t_new = self.p2s.timep(self._timep(draw_context=True), draw_context=False)
        self.assertFalse(t_new.draw_context)

    def test_template_override_descending(self):
        """descending is a histop-specific parameter."""
        h_new = self.p2s.histop(self._histop(descending=True), descending=False)
        self.assertFalse(h_new.descending)

    # ── SVG validity ──────────────────────────────────────────────────────────

    def test_template_produces_valid_svg(self):
        h_new = self.p2s.histop(self._histop(), wxh=(128, 256))
        self.assertIn('<svg',   h_new._repr_svg_())
        self.assertIn('</svg>', h_new._repr_svg_())

        t_new = self.p2s.timep(self._timep(), wxh=(256, 128))
        self.assertIn('<svg',   t_new._repr_svg_())
        self.assertIn('</svg>', t_new._repr_svg_())

    # ── multiple overrides ────────────────────────────────────────────────────

    def test_template_multiple_overrides(self):
        """Multiple kwargs can be overridden in one call; others are inherited."""
        h_orig = self._histop(wxh=(256, 512), style=self.p2s.BARCHARTp, draw_context=True)
        h_new  = self.p2s.histop(h_orig, wxh=(128, 256), draw_context=False)
        self.assertEqual(h_new.wxh, (128, 256))
        self.assertFalse(h_new.draw_context)
        self.assertEqual(h_new.style, self.p2s.BARCHARTp)

        t_orig = self._timep(wxh=(512, 256), style=self.p2s.BARCHARTp, draw_context=True)
        t_new  = self.p2s.timep(t_orig, wxh=(128, 64), draw_context=False)
        self.assertEqual(t_new.wxh, (128, 64))
        self.assertFalse(t_new.draw_context)
        self.assertEqual(t_new.style, self.p2s.BARCHARTp)

    # ── template chaining ─────────────────────────────────────────────────────

    def test_template_chain(self):
        """Template of a template still renders correctly."""
        h1 = self._histop(wxh=(256, 512))
        h2 = self.p2s.histop(h1, wxh=(200, 400))
        h3 = self.p2s.histop(h2, wxh=(100, 200))
        self.assertEqual(h3.wxh, (100, 200))
        self.assertIn('<svg', h3._repr_svg_())

        t1 = self._timep(wxh=(512, 256))
        t2 = self.p2s.timep(t1, wxh=(256, 128))
        t3 = self.p2s.timep(t2, wxh=(128, 64))
        self.assertEqual(t3.wxh, (128, 64))
        self.assertIn('<svg', t3._repr_svg_())

    # ── histop-specific ───────────────────────────────────────────────────────

    def test_template_with_new_df(self):
        """A new df can be passed alongside the template, overriding the template df."""
        df2   = makeHistoDf(n=50, seed=7)
        t_new = self.p2s.histop(df2, self._histop(wxh=(256, 512)))
        self.assertEqual(len(t_new.df_orig), 50)

    def test_template_inherits_bin_by(self):
        t_copy = self.p2s.histop(self._histop())
        self.assertEqual(t_copy.bin_by, 'cat')


if __name__ == '__main__':
    unittest.main()
