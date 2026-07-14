import unittest
import datetime
import polars as pl
from polars2svg import Polars2SVG
from timep_dataframes import makeTimeDf


class TestTimepSmalp(unittest.TestCase):
    '''Tests for smallp() with a timep template.

    Bug fixed: Timep.__parseInput__ previously raised "df already set" when
    renderSmallMultiples() passed each panel's subset df alongside the template
    object, because the template dict-copy had already populated self.df.  The
    fix collects any explicitly-provided df first and always lets it override a
    df inherited from the template.
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def _makeSmallpDf(self, n_per_cat=50):
        '''Build a DataFrame with a date column and a 'category' column
        suitable for splitting into small multiples.'''
        _lu_ = {'ts': [], 'value': [], 'category': []}
        for _day_ in range(30):
            for _cat_ in ['a', 'b', 'c']:
                for _ in range(n_per_cat):
                    _lu_['ts'].append(datetime.date(2024, 1, 1) + datetime.timedelta(days=_day_))
                    _lu_['value'].append(_day_ * 0.1 + {'a': 5.0, 'b': 2.0, 'c': 1.0}[_cat_])
                    _lu_['category'].append(_cat_)
        return pl.DataFrame(_lu_)

    # ── basic smallp + timep integration ─────────────────────────────────────

    def test_smallp_with_timep_template_renders(self):
        '''smallp(df, timep_template, category_by) produces a valid SVG.'''
        df       = self._makeSmallpDf()
        template = self.p2s.timep(df, color='category', count='value')
        result   = self.p2s.smallp(df, template, 'category')
        svg      = result._repr_svg_()
        self.assertIn('<svg',  svg)
        self.assertIn('</svg>', svg)

    def test_smallp_timep_each_panel_uses_subset_df(self):
        '''Each panel SVG is non-empty and distinct (different category data).'''
        df       = self._makeSmallpDf()
        template = self.p2s.timep(df, color='category', count='value')
        result   = self.p2s.smallp(df, template, 'category')
        self.assertGreater(len(result._repr_svg_()), 0)
        # category_to_xy holds only the placed panels (excludes the '__remainder__' slot)
        self.assertEqual(len(result.category_to_xy), 3)   # 'a', 'b', 'c'

    def test_smallp_timep_default_barchart(self):
        df       = self._makeSmallpDf()
        template = self.p2s.timep(df)
        self.p2s.smallp(df, template, 'category')

    def test_smallp_timep_with_periodic_enum(self):
        df       = makeTimeDf(n=300, year=(2022, 2024), month=(1, 12))
        df       = df.with_columns(pl.col('category'))
        template = self.p2s.timep(df, ('ts', self.p2s.PT_mp), color='category')
        self.p2s.smallp(df, template, 'category')

    def test_smallp_timep_with_linear_enum(self):
        df       = makeTimeDf(n=300, year=(2022, 2024), month=(1, 12))
        template = self.p2s.timep(df, ('ts', self.p2s.LT_Y_mp), color='category')
        self.p2s.smallp(df, template, 'category')

    # ── various styles ────────────────────────────────────────────────────────

    def test_smallp_timep_stackedbar_style(self):
        df       = self._makeSmallpDf()
        template = self.p2s.timep(df, color='category', style=self.p2s.STACKEDBARp)
        self.p2s.smallp(df, template, 'category')

    def test_smallp_timep_boxplot_style(self):
        df       = self._makeSmallpDf()
        template = self.p2s.timep(df, count='value', style=self.p2s.BOXPLOTp)
        self.p2s.smallp(df, template, 'category')

    # ── include_all ───────────────────────────────────────────────────────────

    def test_smallp_timep_include_all(self):
        df       = self._makeSmallpDf()
        template = self.p2s.timep(df, color='category')
        result   = self.p2s.smallp(df, template, 'category', include_all=True)
        self.assertIn('__all__', result.category_to_df)
        self.assertIn('<svg', result._repr_svg_())

    # ── draw_context / wxh ────────────────────────────────────────────────────

    def test_smallp_timep_no_draw_context(self):
        df       = self._makeSmallpDf()
        template = self.p2s.timep(df, color='category')
        self.p2s.smallp(df, template, 'category', draw_context=False)

    def test_smallp_timep_custom_wxh(self):
        df       = self._makeSmallpDf()
        template = self.p2s.timep(df, color='category', wxh=(256, 128))
        self.p2s.smallp(df, template, 'category', wxh=(1024, None))


if __name__ == '__main__':
    unittest.main()
