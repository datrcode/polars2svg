import unittest
import polars as pl
from polars2svg import Polars2SVG
from histop_dataframes import makeHistoDf


def _makeSmallpDf(n_per_panel=50):
    '''DataFrame with a "panel" column suitable for splitting into small multiples.'''
    rows = {'cat': [], 'group': [], 'value': [], 'score': [], 'panel': []}
    import random
    rng = random.Random(0)
    for panel in ['P1', 'P2', 'P3']:
        for _ in range(n_per_panel):
            rows['cat'].append(rng.choice(['A', 'B', 'C']))
            rows['group'].append(rng.choice(['x', 'y']))
            rows['value'].append(rng.randint(1, 100))
            rows['score'].append(round(rng.uniform(0.0, 10.0), 3))
            rows['panel'].append(panel)
    return pl.DataFrame(rows)


class TestHistopSmalp(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    # ── basic smallp + histop integration ────────────────────────────────────

    def test_smallp_with_histop_template_renders(self):
        '''smallp(df, histop_template, category_by) produces a valid SVG.'''
        df       = _makeSmallpDf()
        template = self.p2s.histop(df, 'cat')
        result   = self.p2s.smallp(df, template, 'panel')
        svg      = result._repr_svg_()
        self.assertIn('<svg',  svg)
        self.assertIn('</svg>', svg)

    def test_smallp_histop_three_panels_placed(self):
        '''Three panel values → three tiles placed in the small-multiple grid.'''
        df       = _makeSmallpDf()
        template = self.p2s.histop(df, 'cat')
        result   = self.p2s.smallp(df, template, 'panel')
        self.assertEqual(len(result.category_to_xy), 3)

    def test_smallp_histop_default_barchart(self):
        df       = _makeSmallpDf()
        template = self.p2s.histop(df, 'cat')
        self.p2s.smallp(df, template, 'panel')

    def test_smallp_histop_categorical_color(self):
        df       = _makeSmallpDf()
        template = self.p2s.histop(df, 'cat', color='group')
        self.p2s.smallp(df, template, 'panel')

    # ── various styles ────────────────────────────────────────────────────────

    def test_smallp_histop_stackedbar_style(self):
        df       = _makeSmallpDf()
        template = self.p2s.histop(df, 'cat', color='group', style=self.p2s.STACKEDBARp)
        self.p2s.smallp(df, template, 'panel')

    def test_smallp_histop_boxplot_style(self):
        df       = _makeSmallpDf()
        template = self.p2s.histop(df, 'cat', count='value', style=self.p2s.BOXPLOTp)
        self.p2s.smallp(df, template, 'panel')

    def test_smallp_histop_boxplot_swarm_style(self):
        df       = _makeSmallpDf()
        template = self.p2s.histop(df, 'cat', count='value', style=self.p2s.BOXPLOT_W_SWARMp)
        self.p2s.smallp(df, template, 'panel')

    # ── include_all ───────────────────────────────────────────────────────────

    def test_smallp_histop_include_all(self):
        df       = _makeSmallpDf()
        template = self.p2s.histop(df, 'cat')
        result   = self.p2s.smallp(df, template, 'panel', include_all=True)
        self.assertIn('__all__', result.category_to_df)
        self.assertIn('<svg', result._repr_svg_())

    # ── draw_context / wxh ────────────────────────────────────────────────────

    def test_smallp_histop_no_draw_context(self):
        df       = _makeSmallpDf()
        template = self.p2s.histop(df, 'cat', draw_context=False)
        self.p2s.smallp(df, template, 'panel')

    def test_smallp_histop_custom_wxh(self):
        df       = _makeSmallpDf()
        template = self.p2s.histop(df, 'cat', wxh=(200, 400))
        self.p2s.smallp(df, template, 'panel', wxh=(1024, None))


if __name__ == '__main__':
    unittest.main()
