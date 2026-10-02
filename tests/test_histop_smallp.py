import re
import unittest
import polars as pl
from polars2svg import Polars2SVG
from histop_dataframes import HistopAssertions, orderedBins
from svg_test_utils import SmallpAssertions, normalize_svg


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


class TestHistopSmalp(HistopAssertions, SmallpAssertions, unittest.TestCase):
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
        # Each panel is its own rows' histogram, labelled with its panel's name
        df       = _makeSmallpDf()
        template = self.p2s.histop(df, 'cat')
        result   = self.p2s.smallp(df, template, 'panel')
        for _k_, _panel_ in self.assertPanelsAreTemplateOn(result, df, 'panel', template).items():
            _rows_ = dict(df.filter(pl.col('panel') == _k_).group_by('cat').len().iter_rows())
            self.assertBarsShow(_panel_, _rows_, orderedBins(_rows_))
        self.assertEqual(re.findall(r'>(P\d)</text>', result._repr_svg_()), ['P1', 'P2', 'P3'])

    def test_smallp_histop_categorical_color(self):
        df       = _makeSmallpDf()
        template = self.p2s.histop(df, 'cat', color='group')
        result   = self.p2s.smallp(df, template, 'panel')
        for _k_, _panel_ in self.assertPanelsAreTemplateOn(result, df, 'panel', template).items():
            self.assertSegmentsShow(_panel_, {(b, g): n for b, g, n in
                                    df.filter(pl.col('panel') == _k_).group_by('cat', 'group').len().iter_rows()})

    # ── various styles ────────────────────────────────────────────────────────

    def test_smallp_histop_stackedbar_style(self):
        # A categorical colour already stacks: STACKEDBARp spells out the default
        df       = _makeSmallpDf()
        _explicit_ = self.p2s.smallp(df, self.p2s.histop(df, 'cat', color='group', style=self.p2s.STACKEDBARp), 'panel')
        _default_  = self.p2s.smallp(df, self.p2s.histop(df, 'cat', color='group'), 'panel')
        self.assertEqual(normalize_svg(_explicit_._repr_svg_()), normalize_svg(_default_._repr_svg_()))

    def test_smallp_histop_boxplot_style(self):
        df       = _makeSmallpDf()
        template = self.p2s.histop(df, 'cat', count='value', style=self.p2s.BOXPLOTp)
        self.assertPanelsAreTemplateOn(self.p2s.smallp(df, template, 'panel'), df, 'panel', template)

    def test_smallp_histop_boxplot_swarm_style(self):
        df       = _makeSmallpDf()
        template = self.p2s.histop(df, 'cat', count='value', style=self.p2s.BOXPLOT_W_SWARMp)
        self.assertPanelsAreTemplateOn(self.p2s.smallp(df, template, 'panel'), df, 'panel', template)

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
        for _panel_ in self.assertPanelsAreTemplateOn(self.p2s.smallp(df, template, 'panel'), df, 'panel', template).values():
            self.assertNotIn('<line', _panel_._repr_svg_())

    def test_smallp_histop_custom_wxh(self):
        # The composite is 1024 wide and the 200-wide tiles sit on a 200px grid inside it
        df       = _makeSmallpDf()
        template = self.p2s.histop(df, 'cat', wxh=(200, 400))
        result   = self.p2s.smallp(df, template, 'panel', wxh=(1024, None))
        self.assertEqual(result.wxh[0], 1024)
        self.assertPanelsAreTemplateOn(result, df, 'panel', template)
        for _k_, (_x_, _y_) in result.category_to_xy.items():
            self.assertTrue(_x_ % 200 == 0 and _x_ + 200 <= 1024, f'panel {_k_} at x={_x_}')


if __name__ == '__main__':
    unittest.main()
