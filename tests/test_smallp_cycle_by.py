import unittest
import polars as pl
from polars2svg import Polars2SVG


class TestSmallpCycleBy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p2s = Polars2SVG()
        cls.df = pl.DataFrame({
            'a':   [1, 2, 3, 4, 5, 6],
            'b':   [6, 5, 4, 3, 2, 1],
            'cat': ['x', 'x', 'y', 'y', 'z', 'z'],
            'val': [1, 2, 3, 4, 5, 6],
        })
        cls.xyp = cls.p2s.xyp(df=cls.df, x='a', y='b', wxh=(128, 128))

    # ── key structure ─────────────────────────────────────────────────────────

    def test_cycle_by_single_param_generates_keys(self):
        '''cycle_by={'color': [...]} produces tuple keys and correct override map.'''
        smp = self.p2s.smallp(self.df, self.xyp, wxh=(256, 143),
                              cycle_by={'color': ['cat', 'val']})
        # keys are 1-tuples of the cycled values
        self.assertEqual(smp._sorted_category_keys_, [('cat',), ('val',)])
        self.assertEqual(smp._cycle_override_,
                         {('cat',): {'color': 'cat'}, ('val',): {'color': 'val'}})
        # geometry: 2 needed, 2x1 grid → exact fit
        self.assertEqual(smp.wxh_actual, (256, 143))
        self.assertEqual(len(smp.category_to_xy), 2)
        self.assertEqual(smp.category_to_xy[('cat',)], (0, 0))
        self.assertEqual(smp.category_to_xy[('val',)], (128, 0))
        # each panel holds the full df
        self.assertEqual(len(smp.category_to_df[('cat',)]), len(self.df))
        self.assertEqual(len(smp.category_to_df[('val',)]), len(self.df))

    def test_cycle_by_two_params_lockstep(self):
        '''cycle_by with two params produces 2-tuple keys zipped lockstep.'''
        smp = self.p2s.smallp(self.df, self.xyp, wxh=(384, 143),
                              cycle_by={'x': ['a', 'b', 'a'], 'y': ['b', 'a', 'a']})
        # keys are 2-tuples: [(a,b), (b,a), (a,a)]
        self.assertEqual(smp._sorted_category_keys_, [('a', 'b'), ('b', 'a'), ('a', 'a')])
        self.assertEqual(smp._cycle_override_[('a', 'b')], {'x': 'a', 'y': 'b'})
        self.assertEqual(smp._cycle_override_[('b', 'a')], {'x': 'b', 'y': 'a'})
        self.assertEqual(smp._cycle_override_[('a', 'a')], {'x': 'a', 'y': 'a'})
        # geometry: 3 needed, 3x1 grid
        self.assertEqual(len(smp.category_to_xy), 3)
        self.assertEqual(smp.category_to_xy[('a', 'b')], (0, 0))
        self.assertEqual(smp.category_to_xy[('b', 'a')], (128, 0))
        self.assertEqual(smp.category_to_xy[('a', 'a')], (256, 0))

    # ── rendering ─────────────────────────────────────────────────────────────

    def test_cycle_by_renders_svg_via_render_with(self):
        '''cycle_by mode calls render_with and produces a valid SVG.'''
        smp = self.p2s.smallp(self.df, self.xyp, wxh=(256, 143),
                              cycle_by={'color': ['cat', 'val']})
        self.assertIn('__renderSVG__', smp.timing_metrics)
        self.assertIn('<svg', smp.svg)
        # one <g transform=> per panel
        self.assertEqual(smp.svg.count('<g transform='), 2)

    def test_cycle_by_timing_metrics_populated(self):
        '''All pipeline stages record timing metrics for cycle_by.'''
        smp = self.p2s.smallp(self.df, self.xyp, wxh=(256, 143),
                              cycle_by={'color': ['cat', 'val']})
        for stage in ('__parseInput__', '__validateInput__',
                      '__computeOrderingStats__', '__constructGeometry__', '__renderSVG__'):
            self.assertIn(stage, smp.timing_metrics)

    # ── validation errors ─────────────────────────────────────────────────────

    def test_cycle_by_non_dict_raises(self):
        '''cycle_by that is not a dict raises ValueError.'''
        with self.assertRaises(ValueError):
            self.p2s.smallp(self.df, self.xyp, wxh=(256, 143),
                            cycle_by=['cat', 'val'])

    def test_cycle_by_empty_dict_raises(self):
        '''cycle_by={} (empty dict) raises ValueError.'''
        with self.assertRaises(ValueError):
            self.p2s.smallp(self.df, self.xyp, wxh=(256, 143), cycle_by={})

    def test_cycle_by_unequal_lengths_raises(self):
        '''cycle_by with value lists of different lengths raises ValueError.'''
        with self.assertRaises(ValueError):
            self.p2s.smallp(self.df, self.xyp, wxh=(256, 143),
                            cycle_by={'x': ['a', 'b', 'a'], 'y': ['b', 'a']})

    def test_cycle_by_and_category_by_mutually_exclusive_raises(self):
        '''Specifying both category_by and cycle_by raises ValueError.'''
        with self.assertRaises(ValueError):
            self.p2s.smallp(self.df, 'cat', self.xyp, wxh=(256, 143),
                            cycle_by={'color': ['cat', 'val']})

    def test_neither_cycle_by_nor_category_by_raises(self):
        '''Omitting both category_by and cycle_by raises ValueError.'''
        with self.assertRaises(ValueError):
            self.p2s.smallp(self.df, self.xyp, wxh=(256, 143))


if __name__ == '__main__':
    unittest.main()
