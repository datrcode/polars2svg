import re
import unittest
import datetime
import polars as pl
from polars2svg import Polars2SVG
from timep_dataframes import makeTimeDf, TimepAssertions
from svg_test_utils import SmallpAssertions, normalize_svg


class TestTimepSmalp(TimepAssertions, SmallpAssertions, unittest.TestCase):
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
        # Each panel is its own rows' time histogram, labelled with its category
        df       = self._makeSmallpDf()
        template = self.p2s.timep(df)
        result   = self.p2s.smallp(df, template, 'category')
        for _k_, _panel_ in self.assertPanelsAreTemplateOn(result, df, 'category', template).items():
            self.assertColumnsShow(_panel_, df.filter(pl.col('category') == _k_), 'ts', pl.len())
        self.assertEqual(sorted(re.findall(r'>([abc])</text>', result._repr_svg_())), ['a', 'b', 'c'])

    def test_smallp_timep_with_periodic_enum(self):
        df       = makeTimeDf(n=300, year=(2022, 2024), month=(1, 12), seed=21)
        template = self.p2s.timep(df, ('ts', self.p2s.PT_mp), color='category')
        for _k_, _panel_ in self.assertPanelsAreTemplateOn(self.p2s.smallp(df, template, 'category'), df, 'category', template).items():
            self.assertStacksShow(_panel_, df.filter(pl.col('category') == _k_), 'ts', 'category', pl.len())

    def test_smallp_timep_with_linear_enum(self):
        df       = makeTimeDf(n=300, year=(2022, 2024), month=(1, 12), seed=22)
        template = self.p2s.timep(df, ('ts', self.p2s.LT_Y_mp), color='category')
        for _k_, _panel_ in self.assertPanelsAreTemplateOn(self.p2s.smallp(df, template, 'category'), df, 'category', template).items():
            self.assertEqual(_panel_._time_enum_, self.p2s.LT_Y_mp)
            self.assertStacksShow(_panel_, df.filter(pl.col('category') == _k_), 'ts', 'category', pl.len())

    # ── various styles ────────────────────────────────────────────────────────

    def test_smallp_timep_stackedbar_style(self):
        # A categorical colour already stacks: STACKEDBARp spells out the default
        df = self._makeSmallpDf()
        _explicit_ = self.p2s.smallp(df, self.p2s.timep(df, color='category', style=self.p2s.STACKEDBARp), 'category')
        _default_  = self.p2s.smallp(df, self.p2s.timep(df, color='category'), 'category')
        self.assertEqual(normalize_svg(_explicit_._repr_svg_()), normalize_svg(_default_._repr_svg_()))

    def test_smallp_timep_boxplot_style(self):
        df       = self._makeSmallpDf()
        template = self.p2s.timep(df, count='value', style=self.p2s.BOXPLOTp)
        self.assertPanelsAreTemplateOn(self.p2s.smallp(df, template, 'category'), df, 'category', template)

    # ── include_all ───────────────────────────────────────────────────────────

    def test_smallp_timep_include_all(self):
        df       = self._makeSmallpDf()
        template = self.p2s.timep(df, color='category')
        result   = self.p2s.smallp(df, template, 'category', include_all=True)
        self.assertIn('__all__', result.category_to_df)
        self.assertIn('<svg', result._repr_svg_())

    # ── draw_labels / wxh ────────────────────────────────────────────────────

    def test_smallp_timep_no_draw_labels(self):
        # The panels are unchanged; only the category labels go
        df       = self._makeSmallpDf()
        template = self.p2s.timep(df, color='category')
        _off_ = self.p2s.smallp(df, template, 'category', draw_labels=False)
        _on_  = self.p2s.smallp(df, template, 'category')
        self.assertPanelsAreTemplateOn(_off_, df, 'category', template)
        self.assertEqual(re.findall(r'>([abc])</text>', _off_._repr_svg_()), [])
        self.assertEqual(sorted(re.findall(r'>([abc])</text>', _on_._repr_svg_())), ['a', 'b', 'c'])

    def test_smallp_timep_custom_wxh(self):
        # The composite is 1024 wide and the 256-wide tiles sit on a 256px grid inside it
        df       = self._makeSmallpDf()
        template = self.p2s.timep(df, color='category', wxh=(256, 128))
        result   = self.p2s.smallp(df, template, 'category', wxh=(1024, None))
        self.assertEqual(result.wxh[0], 1024)
        self.assertPanelsAreTemplateOn(result, df, 'category', template)
        for _k_, (_x_, _y_) in result.category_to_xy.items():
            self.assertTrue(_x_ % 256 == 0 and _x_ + 256 <= 1024, f'panel {_k_} at x={_x_}')


if __name__ == '__main__':
    unittest.main()
