import unittest
import polars as pl
from polars2svg import Polars2SVG

from svg_test_utils import assert_svg_matches_golden, assert_image_matches_golden


class TestSmallpBasic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p2s = Polars2SVG()
        df0 = pl.DataFrame({'a':[1,2,3],'b':[4,5,6],'cat':['a','a','a']})
        df1 = pl.DataFrame({'a':[3,4,5],'b':[7,9,8],'cat':['b','b','b']})
        df2 = pl.DataFrame({'a':[6,7,8],'b':[1,3,5],'cat':['c','c','c']})
        df3 = pl.DataFrame({'a':[2,3,4],'b':[1,3,4],'cat':['d','d','d']})
        df4 = pl.DataFrame({'a':[1,2,3,4,5,6,7,8],'b':[8,8,7,7,6,6,5,5],
                            'cat':['e','e','e','e','e','e','e','e']})
        cls.df = pl.concat([df0, df1, df2, df3, df4])
        cls.xyp = cls.p2s.xyp(df=cls.df, x='a', y='b', color='cat',
                               line=('cat', 1.5), dot_size=2.0, wxh=(128, 128))

    # Cell: p2s.smallp(df, 'cat', _xyp_, include_all=True, wxh=(384,384))
    # tmpl: 128w x 143h_adj → tiles_across=3, tiles_down=2 → 6 spaces
    # tiles_needed = 5 cats + 1 all = 6 → fits exactly, no remainder
    def test_include_all_fits_exactly(self):
        smp = self.p2s.smallp(self.df, 'cat', self.xyp, include_all=True, wxh=(384, 384))
        assert smp.wxh_actual == (384, 384)
        assert len(smp.category_to_xy) == 6
        assert '__all__' in smp.category_to_xy
        assert '__remainder__' not in smp.category_to_xy
        assert_svg_matches_golden(smp.svg, 'smallp_include_all_fits_exactly')
        assert_image_matches_golden(smp.svg, 'smallp_include_all_fits_exactly')

    # Cell: p2s.smallp(df, 'cat', _xyp_, wxh=(384,384))
    # 6 spaces for 5 categories → all fit, no all, no remainder
    def test_basic_all_fit(self):
        smp = self.p2s.smallp(self.df, 'cat', self.xyp, wxh=(384, 384))
        assert smp.wxh_actual == (384, 384)
        assert len(smp.category_to_xy) == 5
        assert '__all__' not in smp.category_to_xy
        assert '__remainder__' not in smp.category_to_xy
        assert_svg_matches_golden(smp.svg, 'smallp_basic_all_fit')
        assert_image_matches_golden(smp.svg, 'smallp_basic_all_fit')

    # Cell: p2s.smallp(df, 'cat', _xyp_, wxh=(190,190))
    # tiles_across=1, tiles_down=1 → 1 space for 5 categories
    # the single slot becomes '__remainder__' (all 5 categories collated)
    def test_small_wxh_single_slot_all_remainder(self):
        smp = self.p2s.smallp(self.df, 'cat', self.xyp, wxh=(190, 190))
        assert len(smp.category_to_xy) == 1
        assert '__remainder__' in smp.category_to_xy
        assert smp.category_to_df['__remainder__'] is not None
        assert len(smp.category_to_df['__remainder__']) == len(self.df)
        assert_svg_matches_golden(smp.svg, 'smallp_small_wxh_single_slot_all_remainder')
        assert_image_matches_golden(smp.svg, 'smallp_small_wxh_single_slot_all_remainder')

    # Cell: p2s.smallp(df, 'cat', _xyp_, wxh=(256,190))
    # tiles_across=2, tiles_down=1 → 2 spaces for 5 categories
    # first slot → top category, second slot → '__remainder__'
    def test_2x1_grid_with_remainder(self):
        smp = self.p2s.smallp(self.df, 'cat', self.xyp, wxh=(256, 190))
        assert len(smp.category_to_xy) == 2
        assert '__remainder__' in smp.category_to_xy
        assert '__all__' not in smp.category_to_xy
        assert_svg_matches_golden(smp.svg, 'smallp_2x1_grid_with_remainder')
        assert_image_matches_golden(smp.svg, 'smallp_2x1_grid_with_remainder')

    # Cell: p2s.smallp(df, 'cat', _xyp_, wxh=(256,2*143))
    # tiles_across=2, tiles_down=2 → 4 spaces for 5 categories
    # slots: 3 categories + '__remainder__'
    def test_2x2_grid_with_remainder(self):
        smp = self.p2s.smallp(self.df, 'cat', self.xyp, wxh=(256, 2*143))
        assert len(smp.category_to_xy) == 4
        assert '__remainder__' in smp.category_to_xy
        assert '__all__' not in smp.category_to_xy
        assert_svg_matches_golden(smp.svg, 'smallp_2x2_grid_with_remainder')
        assert_image_matches_golden(smp.svg, 'smallp_2x2_grid_with_remainder')

    # Cell: p2s.smallp(df, 'cat', _xyp_, wxh=(256,2*143), include_all=True)
    # 4 spaces for 6 tiles (5 cats + all)
    # slots: '__all__' + 2 categories + '__remainder__'
    def test_2x2_grid_include_all_with_remainder(self):
        smp = self.p2s.smallp(self.df, 'cat', self.xyp, wxh=(256, 2*143), include_all=True)
        assert len(smp.category_to_xy) == 4
        assert '__all__' in smp.category_to_xy
        assert '__remainder__' in smp.category_to_xy
        assert_svg_matches_golden(smp.svg, 'smallp_2x2_grid_include_all_with_remainder')
        assert_image_matches_golden(smp.svg, 'smallp_2x2_grid_include_all_with_remainder')

    # Cell: p2s.smallp(df, 'cat', _xyp_, wxh=(256,2*143), include_all=True, collate_remainder=False)
    # 4 spaces for 6 tiles, collate_remainder=False
    # the last-slot shortcut is disabled → all 4 spaces filled with '__all__' + 3 categories
    # the 2 overflow categories are silently dropped
    def test_2x2_grid_include_all_no_collate_remainder(self):
        smp = self.p2s.smallp(self.df, 'cat', self.xyp, wxh=(256, 2*143),
                              include_all=True, collate_remainder=False)
        assert len(smp.category_to_xy) == 4
        assert '__all__' in smp.category_to_xy
        assert '__remainder__' not in smp.category_to_xy
        assert_svg_matches_golden(smp.svg, 'smallp_2x2_grid_include_all_no_collate_remainder')
        assert_image_matches_golden(smp.svg, 'smallp_2x2_grid_include_all_no_collate_remainder')


if __name__ == '__main__':
    unittest.main()
