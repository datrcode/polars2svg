import datetime
import unittest
import polars as pl
from polars2svg import Polars2SVG


class TestSmallpGeometry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p2s = Polars2SVG()

        # 6 categories, 2 rows each (tmpl_w=128, tmpl_h=128)
        cls.df = pl.DataFrame({
            'a':   [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            'b':   [1, 1, 2, 2, 3, 3, 4, 4, 5,  5,  6,  6],
            'cat': ['a', 'a', 'b', 'b', 'c', 'c',
                    'd', 'd', 'e', 'e', 'f', 'f'],
        })
        cls.xyp = cls.p2s.xyp(df=cls.df, x='a', y='b', wxh=(128, 128))

        # 5 categories for leftover-row/column tests
        cls.df5 = pl.DataFrame({
            'a':   [1, 2, 3, 4, 5],
            'b':   [1, 2, 3, 4, 5],
            'cat': ['a', 'b', 'c', 'd', 'e'],
        })
        cls.xyp5 = cls.p2s.xyp(df=cls.df5, x='a', y='b', wxh=(128, 128))

        # datetime df for use_lazy_execution tests (6 distinct months)
        cls.df_ts = pl.DataFrame({
            'a':  [1, 2, 3, 4, 5, 6],
            'b':  [6, 5, 4, 3, 2, 1],
            'ts': pl.Series([datetime.datetime(2024, m, 1) for m in range(1, 7)]),
        })
        cls.xyp_ts = cls.p2s.xyp(df=cls.df_ts, x='a', y='b', wxh=(128, 128))

    # ── sketch_only ───────────────────────────────────────────────────────────

    def test_sketch_only_skips_render_svg(self):
        '''sketch_only=True builds geometry but never calls __renderSVG__.

        6 cats, wxh=(384,286): tiles_across=3, tiles_down=2 → exact fit.
        '''
        smp = self.p2s.smallp(self.df, 'cat', self.xyp,
                               wxh=(384, 286), sketch_only=True)
        self.assertNotIn('__renderSVG__', smp.timing_metrics)
        self.assertIn('__constructGeometry__', smp.timing_metrics)
        self.assertEqual(smp.wxh_actual, (384, 286))
        self.assertEqual(len(smp.category_to_xy), 6)
        self.assertIn('<svg', smp.svg)
        self.assertIn('<rect', smp.svg)

    def test_sketch_only_svg_contains_tile_rects_and_remainder_label(self):
        '''sketch_only SVG has tile outlines, df.len text, and "Remainder" label.

        6 cats, wxh=(256,286): tiles_across=2, tiles_down=2 → 4 slots → 3 visible + remainder.
        draw_context=True (default) adds a centered label below each tile.
        '''
        smp = self.p2s.smallp(self.df, 'cat', self.xyp,
                               wxh=(256, 286), sketch_only=True)
        self.assertNotIn('__renderSVG__', smp.timing_metrics)
        self.assertEqual(len(smp.category_to_xy), 4)
        self.assertIn('Remainder', smp.svg)
        self.assertIn('df.len = 2', smp.svg)

    def test_sketch_only_draw_context_false(self):
        '''sketch_only=True with draw_context=False uses tmpl_h_adj=128.

        draw_context=False → tmpl_h_adj=128 → tiles_down=256//128=2 → 4 slots for 6 cats.
        The label-below-tile branch (line 346) is skipped.
        '''
        smp = self.p2s.smallp(self.df, 'cat', self.xyp,
                               wxh=(256, 256), sketch_only=True, draw_context=False)
        self.assertNotIn('__renderSVG__', smp.timing_metrics)
        self.assertEqual(smp.wxh_actual, (256, 256))
        # 4 slots → 3 visible + remainder
        self.assertEqual(len(smp.category_to_xy), 4)
        self.assertIn('<svg', smp.svg)

    # ── draw_context=False ────────────────────────────────────────────────────

    def test_draw_context_false_changes_tile_height_in_geometry(self):
        '''draw_context=False uses tmpl_h_adj=tmpl_h=128 vs. 143, changing grid layout.

        wxh=(256,256):
          draw_context=True : tmpl_h_adj=143 → tiles_down=1 → 2 slots → 1 visible + remainder
          draw_context=False: tmpl_h_adj=128 → tiles_down=2 → 4 slots → 3 visible + remainder
        '''
        smp_ctx  = self.p2s.smallp(self.df, 'cat', self.xyp,
                                    wxh=(256, 256), draw_context=True)
        smp_nctx = self.p2s.smallp(self.df, 'cat', self.xyp,
                                    wxh=(256, 256), draw_context=False)
        # draw_context=True: 256//143=1 row → 2 slots
        self.assertEqual(len(smp_ctx.category_to_xy), 2)
        # draw_context=False: 256//128=2 rows → 4 slots
        self.assertEqual(len(smp_nctx.category_to_xy), 4)

    # ── insets ────────────────────────────────────────────────────────────────

    def test_insets_shift_tile_positions(self):
        '''Insets create x_ins outer margin on all sides and x_ins gaps between tiles.

        insets=(10,10), wxh=(406,163):
          tiles_across = (406-10)//(128+10) = 396//138 = 2
          tiles_down   = (163-10)//(143+10) = 153//153 = 1  → 2 slots
          6 cats → 1 visible at (10,10) + __remainder__ at (148,10)
          gap between borders = 10 = x_ins; outer margins = 10 = x_ins
        '''
        smp = self.p2s.smallp(self.df, 'cat', self.xyp,
                               wxh=(406, 163), insets=(10, 10), sketch_only=True)
        self.assertEqual(smp.wxh_actual, (406, 163))
        self.assertEqual(len(smp.category_to_xy), 2)
        xy_values = list(smp.category_to_xy.values())
        self.assertIn((10, 10), xy_values)
        self.assertEqual(smp.category_to_xy['__remainder__'], (148, 10))

    # ── wxh auto-height (w, None) ─────────────────────────────────────────────

    def test_wxh_auto_height_exact_fit(self):
        '''wxh=(256, None) computes height from tiles_down when tiles divide evenly.

        6 cats, tiles_across=2 → tiles_down=6//2=3, leftover=0 → h=3*143=429.
        All 6 cats fit, no remainder.
        '''
        smp = self.p2s.smallp(self.df, 'cat', self.xyp, wxh=(256, None))
        self.assertEqual(smp.wxh_actual, (256, 429))
        self.assertEqual(len(smp.category_to_xy), 6)
        self.assertNotIn('__remainder__', smp.category_to_xy)
        # row 2 tiles should have y_offset = 2*143 = 286
        y_offsets = {xy[1] for xy in smp.category_to_xy.values()}
        self.assertIn(286, y_offsets)

    def test_wxh_auto_height_leftover_adds_row(self):
        '''wxh=(384, None) with 5 cats: leftover forces an extra row.

        5 cats, tiles_across=3 → tiles_down=5//3=1, leftover=2 → tiles_down=2 → h=286.
        5 cats in 3x2=6 slots → all fit, no remainder.
        '''
        smp = self.p2s.smallp(self.df5, 'cat', self.xyp5, wxh=(384, None))
        self.assertEqual(smp.wxh_actual, (384, 286))
        self.assertEqual(len(smp.category_to_xy), 5)
        self.assertNotIn('__remainder__', smp.category_to_xy)

    # ── wxh auto-width (None, h) ──────────────────────────────────────────────

    def test_wxh_auto_width_exact_fit(self):
        '''wxh=(None, 286) computes width from tiles_across when tiles divide evenly.

        6 cats, tiles_down=2 → tiles_across=6//2=3, leftover=0 → w=3*128=384.
        All 6 cats fit, no remainder.
        '''
        smp = self.p2s.smallp(self.df, 'cat', self.xyp, wxh=(None, 286))
        self.assertEqual(smp.wxh_actual, (384, 286))
        self.assertEqual(len(smp.category_to_xy), 6)
        self.assertNotIn('__remainder__', smp.category_to_xy)

    def test_wxh_auto_width_leftover_adds_column(self):
        '''wxh=(None, 286) with 5 cats: leftover forces an extra column.

        5 cats, tiles_down=2 → tiles_across=5//2=2, leftover=1 → tiles_across=3 → w=384.
        5 cats in 3x2=6 slots → all fit, no remainder.
        '''
        smp = self.p2s.smallp(self.df5, 'cat', self.xyp5, wxh=(None, 286))
        self.assertEqual(smp.wxh_actual, (384, 286))
        self.assertEqual(len(smp.category_to_xy), 5)
        self.assertNotIn('__remainder__', smp.category_to_xy)

    # ── use_lazy_execution ────────────────────────────────────────────────────

    def test_use_lazy_execution_true_with_tfield(self):
        '''use_lazy_execution=True routes through df.lazy().with_columns().collect().

        A tfield category_by is required so _ops_ is non-empty and the lazy path executes.
        '''
        tfield = self.p2s.tField('ts', self.p2s.LT_Y_mp)
        smp = self.p2s.smallp(self.df_ts, tfield, self.xyp_ts,
                               wxh=(256, None), use_lazy_execution=True)
        self.assertIn('__addColumnsToDataFrame__', smp.timing_metrics)
        # tfield column added to df
        self.assertIn(tfield, smp.df.columns)
        self.assertIn('<svg', smp.svg)
        self.assertGreater(smp._num_categories_, 0)

    def test_use_lazy_execution_matches_eager(self):
        '''use_lazy_execution=True and False produce identical categories and layout.'''
        tfield = self.p2s.tField('ts', self.p2s.LT_Y_mp)
        smp_lazy  = self.p2s.smallp(self.df_ts, tfield, self.xyp_ts,
                                     wxh=(256, None), use_lazy_execution=True)
        smp_eager = self.p2s.smallp(self.df_ts, tfield, self.xyp_ts,
                                     wxh=(256, None), use_lazy_execution=False)
        self.assertEqual(smp_lazy.wxh_actual, smp_eager.wxh_actual)
        self.assertEqual(smp_lazy._sorted_category_keys_, smp_eager._sorted_category_keys_)
        self.assertEqual(smp_lazy._num_categories_, smp_eager._num_categories_)


if __name__ == '__main__':
    unittest.main()
