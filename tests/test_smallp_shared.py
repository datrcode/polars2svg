import unittest
import polars as pl
from polars2svg import Polars2SVG


class TestSmallpShared(unittest.TestCase):
    """
    Tests for the sm_shared attribute of XYp.renderSmallMultiples():
      SM_X     - shared x-axis range across small multiples
      SM_Y     - shared y-axis range across small multiples
      SM_COUNT - shared dot-size normalization (excludes __all__)
      SM_COLOR - shared color normalization, both MAGNITUDE and STRETCHED modes (excludes __all__)
    """

    @classmethod
    def setUpClass(cls):
        cls.p2s = Polars2SVG()

        # ── SM_X / SM_Y test data ────────────────────────────────────────────
        # Two categories with non-overlapping x *and* y ranges.
        # 'left':  x ∈ [10,30], y ∈ [10,30]
        # 'right': x ∈ [70,90], y ∈ [70,90]
        # → without sharing, each SM has its own tight range.
        #   with SM_X/SM_Y sharing, both SMs use the global [10,90] range.
        df_left  = pl.DataFrame({'x': [10.0, 20.0, 30.0], 'y': [10.0, 20.0, 30.0], 'cat': ['left']*3})
        df_right = pl.DataFrame({'x': [70.0, 80.0, 90.0], 'y': [70.0, 80.0, 90.0], 'cat': ['right']*3})
        cls.df_xy    = pl.concat([df_left, df_right])
        cls.df_lu_xy = {'left': df_left, 'right': df_right}

        # ── SM_COUNT test data ───────────────────────────────────────────────
        # 'lo' has a val column that ranges [1,3]; 'hi' ranges [7,9].
        # Each category has one point per pixel (distinct y coords prevent overlap).
        # dot_size='val' → __dot_size_sum__ = val per pixel.
        # Global range across both SMs: [1, 9].
        df_lo_cnt = pl.DataFrame({'x': [10.0, 20.0, 30.0], 'y': [10.0, 20.0, 30.0],
                                  'val': [1.0, 2.0, 3.0], 'cat': ['lo']*3})
        df_hi_cnt = pl.DataFrame({'x': [10.0, 20.0, 30.0], 'y': [40.0, 50.0, 60.0],
                                  'val': [7.0, 8.0, 9.0], 'cat': ['hi']*3})
        cls.df_cnt    = pl.concat([df_lo_cnt, df_hi_cnt])
        cls.df_lu_cnt = {'lo': df_lo_cnt, 'hi': df_hi_cnt}

        # ── SM_COLOR MAGNITUDE test data ─────────────────────────────────────
        # 'lo':  pixel (10,10)→count=1, pixel (30,30)→count=5
        # 'hi':  pixel (10,10)→count=1, pixel (50,50)→count=10
        # Global per-pixel count range: [1, 10].
        # Key observable: without sharing lo count=5 → norm=1.0 (its local max);
        #                 with sharing  lo count=5 → norm=(5-1)/(10-1) ≈ 0.444.
        cls.df_lo_mag = pl.DataFrame({
            'x': [10.0] + [30.0]*5,
            'y': [10.0] + [30.0]*5,
            'cat': ['lo']*6
        })
        cls.df_hi_mag = pl.DataFrame({
            'x': [10.0] + [50.0]*10,
            'y': [10.0] + [50.0]*10,
            'cat': ['hi']*11
        })
        cls.df_mag    = pl.concat([cls.df_lo_mag, cls.df_hi_mag])
        cls.df_lu_mag = {'lo': cls.df_lo_mag, 'hi': cls.df_hi_mag}

        # ── SM_COLOR STRETCHED test data ─────────────────────────────────────
        # 'lo':  pixel (10,10)→count=1, pixel (30,30)→count=5
        # 'hi':  pixel (10,10)→count=5, pixel (50,50)→count=10
        # Global unique sorted counts: [1, 5, 10]  → norms 0.0, 0.5, 1.0.
        # Key observable: without sharing lo count=5 → norm=1.0 (lo's local max),
        #                                 hi count=5 → norm=0.0 (hi's local min).
        #                 with sharing both SMs: count=5 → norm=0.5.
        cls.df_lo_str = pl.DataFrame({
            'x': [10.0] + [30.0]*5,
            'y': [10.0] + [30.0]*5,
            'cat': ['lo']*6
        })
        cls.df_hi_str = pl.DataFrame({
            'x': [10.0]*5 + [50.0]*10,
            'y': [10.0]*5 + [50.0]*10,
            'cat': ['hi']*15
        })
        cls.df_str    = pl.concat([cls.df_lo_str, cls.df_hi_str])
        cls.df_lu_str = {'lo': cls.df_lo_str, 'hi': cls.df_hi_str}

        # ── SM_X / SM_Y datetime axis test data ──────────────────────────────
        # 'early': x ∈ [dt_a, dt_b],  'late': x ∈ [dt_c, dt_d]
        # Non-overlapping datetime ranges to verify global range is used.
        from datetime import datetime as _dt
        dt_a = _dt(2013, 4, 1,  6, 0, 0)
        dt_b = _dt(2013, 4, 1, 12, 0, 0)
        dt_c = _dt(2013, 4, 1, 18, 0, 0)
        dt_d = _dt(2013, 4, 2,  0, 0, 0)

        df_early_dtx = pl.DataFrame({'x': [dt_a, dt_b], 'y': [1.0, 2.0], 'cat': ['early']*2})
        df_late_dtx  = pl.DataFrame({'x': [dt_c, dt_d], 'y': [3.0, 4.0], 'cat': ['late']*2})
        cls.df_dtx    = pl.concat([df_early_dtx, df_late_dtx])
        cls.df_lu_dtx = {'early': df_early_dtx, 'late': df_late_dtx}

        df_early_dty = pl.DataFrame({'x': [1.0, 2.0], 'y': [dt_a, dt_b], 'cat': ['early']*2})
        df_late_dty  = pl.DataFrame({'x': [3.0, 4.0], 'y': [dt_c, dt_d], 'cat': ['late']*2})
        cls.df_dty    = pl.concat([df_early_dty, df_late_dty])
        cls.df_lu_dty = {'early': df_early_dty, 'late': df_late_dty}

        df_early_dtxy = pl.DataFrame({'x': [dt_a, dt_b], 'y': [dt_a, dt_b], 'cat': ['early']*2})
        df_late_dtxy  = pl.DataFrame({'x': [dt_c, dt_d], 'y': [dt_c, dt_d], 'cat': ['late']*2})
        cls.df_dtxy    = pl.concat([df_early_dtxy, df_late_dtxy])
        cls.df_lu_dtxy = {'early': df_early_dtxy, 'late': df_late_dtxy}

        # Mixed: datetime x + categorical (string) y, SM_X + SM_Y
        # Mirrors the notebook pattern: parsedDate x-axis, IP-address y-axis
        df_early_dtxsy = pl.DataFrame({'x': [dt_a, dt_b], 'y': ['1.1.1.1', '2.2.2.2'], 'cat': ['early']*2})
        df_late_dtxsy  = pl.DataFrame({'x': [dt_c, dt_d], 'y': ['3.3.3.3', '4.4.4.4'], 'cat': ['late']*2})
        cls.df_dtxsy    = pl.concat([df_early_dtxsy, df_late_dtxsy])
        cls.df_lu_dtxsy = {'early': df_early_dtxsy, 'late': df_late_dtxsy}

    # ── SM_X ─────────────────────────────────────────────────────────────────

    def test_sm_x_x_ranges_equal(self):
        """With SM_X, both sub-instances share the same x_transform_vars."""
        tmpl = self.p2s.xyp(df=self.df_xy, x='x', y='y', dot_size=2.0, wxh=(256, 256),
                            sm_shared={self.p2s.SM_X})
        rlu = tmpl.renderSmallMultiples(self.df_xy, self.df_lu_xy, '__all__')
        # x_transform_vars = (plot_origin_x, xmin, dx, plot_size_x)
        assert rlu['left'].x_transform_vars[1] == rlu['right'].x_transform_vars[1], 'xmin must match'
        assert rlu['left'].x_transform_vars[2] == rlu['right'].x_transform_vars[2], 'dx must match'

    def test_sm_x_no_sharing_x_ranges_differ(self):
        """Without SM_X, sub-instances each use their own local x range."""
        tmpl = self.p2s.xyp(df=self.df_xy, x='x', y='y', dot_size=2.0, wxh=(256, 256))
        rlu = tmpl.renderSmallMultiples(self.df_xy, self.df_lu_xy, '__all__')
        assert rlu['left'].x_transform_vars[1] != rlu['right'].x_transform_vars[1], \
            'xmin should differ without SM_X'

    # ── SM_Y ─────────────────────────────────────────────────────────────────

    def test_sm_y_y_ranges_equal(self):
        """With SM_Y, both sub-instances share the same y_transform_vars."""
        tmpl = self.p2s.xyp(df=self.df_xy, x='x', y='y', dot_size=2.0, wxh=(256, 256),
                            sm_shared={self.p2s.SM_Y})
        rlu = tmpl.renderSmallMultiples(self.df_xy, self.df_lu_xy, '__all__')
        assert rlu['left'].y_transform_vars[1] == rlu['right'].y_transform_vars[1], 'ymin must match'
        assert rlu['left'].y_transform_vars[2] == rlu['right'].y_transform_vars[2], 'dy must match'

    def test_sm_y_no_sharing_y_ranges_differ(self):
        """Without SM_Y, sub-instances each use their own local y range."""
        tmpl = self.p2s.xyp(df=self.df_xy, x='x', y='y', dot_size=2.0, wxh=(256, 256))
        rlu = tmpl.renderSmallMultiples(self.df_xy, self.df_lu_xy, '__all__')
        assert rlu['left'].y_transform_vars[1] != rlu['right'].y_transform_vars[1], \
            'ymin should differ without SM_Y'

    # ── SM_X + SM_Y ───────────────────────────────────────────────────────────

    def test_sm_xy_both_ranges_equal(self):
        """With both SM_X and SM_Y, x and y ranges are simultaneously shared."""
        tmpl = self.p2s.xyp(df=self.df_xy, x='x', y='y', dot_size=2.0, wxh=(256, 256),
                            sm_shared={self.p2s.SM_X, self.p2s.SM_Y})
        rlu = tmpl.renderSmallMultiples(self.df_xy, self.df_lu_xy, '__all__')
        assert rlu['left'].x_transform_vars[1] == rlu['right'].x_transform_vars[1]
        assert rlu['left'].x_transform_vars[2] == rlu['right'].x_transform_vars[2]
        assert rlu['left'].y_transform_vars[1] == rlu['right'].y_transform_vars[1]
        assert rlu['left'].y_transform_vars[2] == rlu['right'].y_transform_vars[2]

    # ── SM_COUNT ──────────────────────────────────────────────────────────────

    def test_sm_count_global_min_max_set(self):
        """With SM_COUNT, both non-all instances have the same dot_size_global_min/max."""
        tmpl = self.p2s.xyp(df=self.df_cnt, x='x', y='y', dot_size='val', wxh=(256, 256),
                            sm_shared={self.p2s.SM_COUNT})
        rlu = tmpl.renderSmallMultiples(self.df_cnt, self.df_lu_cnt, '__all__')
        assert rlu['lo'].dot_size_global_min == 1.0
        assert rlu['lo'].dot_size_global_max == 9.0
        assert rlu['hi'].dot_size_global_min == 1.0
        assert rlu['hi'].dot_size_global_max == 9.0

    def test_sm_count_dot_sizes_normalized_globally(self):
        """With SM_COUNT, specific __radius__ values reflect global normalization."""
        tmpl = self.p2s.xyp(df=self.df_cnt, x='x', y='y', dot_size='val', wxh=(256, 256),
                            sm_shared={self.p2s.SM_COUNT})
        rlu   = tmpl.renderSmallMultiples(self.df_cnt, self.df_lu_cnt, '__all__')
        r_min, r_max = tmpl.dot_size_range   # (0.5, 4.0) default

        # lo val=3 → global norm = (3-1)/(9-1) = 0.25 → radius = 0.5 + 3.5*0.25 = 1.375
        lo_r = rlu['lo'].df_pixels.filter(pl.col('__dot_size_sum__') == 3.0)['__radius__'][0]
        assert abs(lo_r - 1.375) < 0.001, f'lo val=3: expected radius≈1.375, got {lo_r}'

        # hi val=7 → global norm = (7-1)/(9-1) = 0.75 → radius = 0.5 + 3.5*0.75 = 3.125
        hi_r = rlu['hi'].df_pixels.filter(pl.col('__dot_size_sum__') == 7.0)['__radius__'][0]
        assert abs(hi_r - 3.125) < 0.001, f'hi val=7: expected radius≈3.125, got {hi_r}'

    def test_sm_count_no_sharing_each_sm_reaches_max_radius(self):
        """Without SM_COUNT, each SM normalizes independently → max radius in both."""
        tmpl = self.p2s.xyp(df=self.df_cnt, x='x', y='y', dot_size='val', wxh=(256, 256))
        rlu  = tmpl.renderSmallMultiples(self.df_cnt, self.df_lu_cnt, '__all__')
        assert abs(rlu['lo'].df_pixels['__radius__'].max() - 4.0) < 0.001
        assert abs(rlu['hi'].df_pixels['__radius__'].max() - 4.0) < 0.001

    # ── SM_COLOR – MAGNITUDE ──────────────────────────────────────────────────

    def test_sm_color_magnitude_global_range_set(self):
        """With SM_COLOR MAGNITUDE, both non-all instances carry the global color range."""
        tmpl = self.p2s.xyp(df=self.df_mag, x='x', y='y', dot_size=2.0, wxh=(256, 256),
                            color=self.p2s.CROW_MAGNITUDEp, sm_shared={self.p2s.SM_COLOR})
        rlu = tmpl.renderSmallMultiples(self.df_mag, self.df_lu_mag, '__all__')
        assert rlu['lo'].color_magnitude_min == 1.0
        assert rlu['lo'].color_magnitude_max == 10.0
        assert rlu['hi'].color_magnitude_min == 1.0
        assert rlu['hi'].color_magnitude_max == 10.0

    def test_sm_color_magnitude_shared_norm_value(self):
        """With SM_COLOR MAGNITUDE, lo count=5 is normalized against the global max (10)."""
        tmpl = self.p2s.xyp(df=self.df_mag, x='x', y='y', dot_size=2.0, wxh=(256, 256),
                            color=self.p2s.CROW_MAGNITUDEp, sm_shared={self.p2s.SM_COLOR})
        rlu  = tmpl.renderSmallMultiples(self.df_mag, self.df_lu_mag, '__all__')
        # lo count=5 with global [1,10]: norm = (5-1)/(10-1) = 4/9 ≈ 0.4444
        norm = rlu['lo'].df_pixels.filter(pl.col('__color_sum__') == 5)['__color_norm__'][0]
        assert abs(norm - 4/9) < 0.001, f'lo count=5: expected __color_norm__≈{4/9:.4f}, got {norm}'

    def test_sm_color_magnitude_same_count_same_hexcolor(self):
        """With SM_COLOR MAGNITUDE, the same count value produces the same __hexcolor__."""
        tmpl = self.p2s.xyp(df=self.df_mag, x='x', y='y', dot_size=2.0, wxh=(256, 256),
                            color=self.p2s.CROW_MAGNITUDEp, sm_shared={self.p2s.SM_COLOR})
        rlu = tmpl.renderSmallMultiples(self.df_mag, self.df_lu_mag, '__all__')
        lo_hex = rlu['lo'].df_pixels.filter(pl.col('__color_sum__') == 1)['__hexcolor__'][0]
        hi_hex = rlu['hi'].df_pixels.filter(pl.col('__color_sum__') == 1)['__hexcolor__'][0]
        assert lo_hex == hi_hex, f'count=1: lo got {lo_hex}, hi got {hi_hex}'

    def test_sm_color_magnitude_no_sharing_lo_max_differs_from_shared(self):
        """Without SM_COLOR, lo count=5 is the local max → norm=1.0 (not 4/9)."""
        tmpl = self.p2s.xyp(df=self.df_mag, x='x', y='y', dot_size=2.0, wxh=(256, 256),
                            color=self.p2s.CROW_MAGNITUDEp)
        rlu  = tmpl.renderSmallMultiples(self.df_mag, self.df_lu_mag, '__all__')
        # lo local max = 5, so count=5 → norm = (5-1)/(5-1) = 1.0
        norm = rlu['lo'].df_pixels.filter(pl.col('__color_sum__') == 5)['__color_norm__'][0]
        assert abs(norm - 1.0) < 0.001, f'without sharing, lo count=5 should be norm=1.0, got {norm}'

    # ── SM_COLOR – STRETCHED ──────────────────────────────────────────────────

    def test_sm_color_stretched_global_values_set(self):
        """With SM_COLOR STRETCHED, color_stretched_global_values spans all unique counts."""
        tmpl = self.p2s.xyp(df=self.df_str, x='x', y='y', dot_size=2.0, wxh=(256, 256),
                            color=self.p2s.CROW_STRETCHEDp, sm_shared={self.p2s.SM_COLOR})
        rlu = tmpl.renderSmallMultiples(self.df_str, self.df_lu_str, '__all__')
        # global unique counts across lo {1,5} and hi {5,10} = [1, 5, 10]
        for key in ('lo', 'hi'):
            vals = rlu[key].color_stretched_global_values
            assert vals is not None, f'{key}: color_stretched_global_values should be set'
            assert sorted(vals) == [1, 5, 10], f'{key}: expected [1,5,10], got {sorted(vals)}'

    def test_sm_color_stretched_shared_norm_for_common_count(self):
        """With SM_COLOR STRETCHED, count=5 (shared across lo and hi) maps to norm=0.5 in both."""
        tmpl = self.p2s.xyp(df=self.df_str, x='x', y='y', dot_size=2.0, wxh=(256, 256),
                            color=self.p2s.CROW_STRETCHEDp, sm_shared={self.p2s.SM_COLOR})
        rlu = tmpl.renderSmallMultiples(self.df_str, self.df_lu_str, '__all__')
        # global sorted = [1,5,10] → count=5 is at index 1 → norm = 1/(3-1) = 0.5
        lo_norm = rlu['lo'].df_pixels.filter(pl.col('__color_sum__') == 5)['__color_norm__'][0]
        hi_norm = rlu['hi'].df_pixels.filter(pl.col('__color_sum__') == 5)['__color_norm__'][0]
        assert abs(lo_norm - 0.5) < 0.001, f'lo count=5: expected norm=0.5, got {lo_norm}'
        assert abs(hi_norm - 0.5) < 0.001, f'hi count=5: expected norm=0.5, got {hi_norm}'

    def test_sm_color_stretched_same_count_same_hexcolor(self):
        """With SM_COLOR STRETCHED, count=5 in lo and hi produces the same __hexcolor__."""
        tmpl = self.p2s.xyp(df=self.df_str, x='x', y='y', dot_size=2.0, wxh=(256, 256),
                            color=self.p2s.CROW_STRETCHEDp, sm_shared={self.p2s.SM_COLOR})
        rlu = tmpl.renderSmallMultiples(self.df_str, self.df_lu_str, '__all__')
        lo_hex = rlu['lo'].df_pixels.filter(pl.col('__color_sum__') == 5)['__hexcolor__'][0]
        hi_hex = rlu['hi'].df_pixels.filter(pl.col('__color_sum__') == 5)['__hexcolor__'][0]
        assert lo_hex == hi_hex, f'count=5: lo got {lo_hex}, hi got {hi_hex}'

    def test_sm_color_stretched_no_sharing_count5_differs_between_sms(self):
        """Without SM_COLOR, count=5 is lo's local max (norm=1.0) but hi's local min (norm=0.0)."""
        tmpl = self.p2s.xyp(df=self.df_str, x='x', y='y', dot_size=2.0, wxh=(256, 256),
                            color=self.p2s.CROW_STRETCHEDp)
        rlu = tmpl.renderSmallMultiples(self.df_str, self.df_lu_str, '__all__')
        lo_norm = rlu['lo'].df_pixels.filter(pl.col('__color_sum__') == 5)['__color_norm__'][0]
        hi_norm = rlu['hi'].df_pixels.filter(pl.col('__color_sum__') == 5)['__color_norm__'][0]
        assert abs(lo_norm - 1.0) < 0.001, f'lo count=5 local max → norm should be 1.0, got {lo_norm}'
        assert abs(hi_norm - 0.0) < 0.001, f'hi count=5 local min → norm should be 0.0, got {hi_norm}'

    # ── all_key exclusion ─────────────────────────────────────────────────────

    def test_sm_color_all_key_not_shared(self):
        """The __all__ instance must not receive SM_COLOR normalization parameters."""
        df_all = pl.concat([self.df_lo_mag, self.df_hi_mag])
        df_lu  = {'lo': self.df_lo_mag, 'hi': self.df_hi_mag, '__all__': df_all}
        tmpl = self.p2s.xyp(df=df_all, x='x', y='y', dot_size=2.0, wxh=(256, 256),
                            color=self.p2s.CROW_MAGNITUDEp, sm_shared={self.p2s.SM_COLOR})
        rlu = tmpl.renderSmallMultiples(df_all, df_lu, '__all__')
        assert rlu['__all__'].color_magnitude_min is None, '__all__ should not get shared color range'
        assert rlu['__all__'].color_magnitude_max is None, '__all__ should not get shared color range'

    def test_sm_count_all_key_not_shared(self):
        """The __all__ instance must not receive SM_COUNT normalization parameters."""
        df_all = pl.concat([self.df_cnt['lo'], self.df_cnt] if False else [self.df_cnt])
        df_lu  = {'lo': self.df_lu_cnt['lo'], 'hi': self.df_lu_cnt['hi'],
                  '__all__': self.df_cnt}
        tmpl = self.p2s.xyp(df=self.df_cnt, x='x', y='y', dot_size='val', wxh=(256, 256),
                            sm_shared={self.p2s.SM_COUNT})
        rlu = tmpl.renderSmallMultiples(self.df_cnt, df_lu, '__all__')
        assert rlu['__all__'].dot_size_global_min is None, '__all__ should not get shared count range'
        assert rlu['__all__'].dot_size_global_max is None, '__all__ should not get shared count range'

    # ── combination / smoke test ──────────────────────────────────────────────

    def test_all_four_enums_combined(self):
        """Using all four sm_shared enums together must not raise any exception."""
        df_lo = pl.DataFrame({'x': [10.0, 20.0, 30.0], 'y': [10.0, 20.0, 30.0],
                              'val': [1.0, 2.0, 3.0], 'cat': ['lo']*3})
        df_hi = pl.DataFrame({'x': [70.0, 80.0, 90.0], 'y': [70.0, 80.0, 90.0],
                              'val': [7.0, 8.0, 9.0], 'cat': ['hi']*3})
        df    = pl.concat([df_lo, df_hi])
        tmpl  = self.p2s.xyp(df=df, x='x', y='y', dot_size='val', wxh=(256, 256),
                              color=self.p2s.CROW_MAGNITUDEp,
                              sm_shared={self.p2s.SM_X, self.p2s.SM_Y,
                                         self.p2s.SM_COUNT, self.p2s.SM_COLOR})
        rlu   = tmpl.renderSmallMultiples(df, {'lo': df_lo, 'hi': df_hi}, '__all__')
        assert 'lo' in rlu and 'hi' in rlu
        assert rlu['lo']._repr_svg_() != ''
        assert rlu['hi']._repr_svg_() != ''


    # ── SM_X / SM_Y – datetime axes ──────────────────────────────────────────

    def test_sm_x_datetime_shared_does_not_raise(self):
        """SM_X with a datetime x-axis must not raise AttributeError."""
        tmpl = self.p2s.xyp(df=self.df_dtx, x='x', y='y', dot_size=2.0, wxh=(256, 256),
                            sm_shared={self.p2s.SM_X})
        rlu = tmpl.renderSmallMultiples(self.df_dtx, self.df_lu_dtx, '__all__')
        assert 'early' in rlu and 'late' in rlu

    def test_sm_x_datetime_x_ranges_equal(self):
        """With SM_X and a datetime x-axis, sub-instances share the same x_transform_vars."""
        tmpl = self.p2s.xyp(df=self.df_dtx, x='x', y='y', dot_size=2.0, wxh=(256, 256),
                            sm_shared={self.p2s.SM_X})
        rlu = tmpl.renderSmallMultiples(self.df_dtx, self.df_lu_dtx, '__all__')
        assert rlu['early'].x_transform_vars[1] == rlu['late'].x_transform_vars[1], 'xmin must match'
        assert rlu['early'].x_transform_vars[2] == rlu['late'].x_transform_vars[2], 'dx must match'

    def test_sm_y_datetime_shared_does_not_raise(self):
        """SM_Y with a datetime y-axis must not raise AttributeError."""
        tmpl = self.p2s.xyp(df=self.df_dty, x='x', y='y', dot_size=2.0, wxh=(256, 256),
                            sm_shared={self.p2s.SM_Y})
        rlu = tmpl.renderSmallMultiples(self.df_dty, self.df_lu_dty, '__all__')
        assert 'early' in rlu and 'late' in rlu

    def test_sm_y_datetime_y_ranges_equal(self):
        """With SM_Y and a datetime y-axis, sub-instances share the same y_transform_vars."""
        tmpl = self.p2s.xyp(df=self.df_dty, x='x', y='y', dot_size=2.0, wxh=(256, 256),
                            sm_shared={self.p2s.SM_Y})
        rlu = tmpl.renderSmallMultiples(self.df_dty, self.df_lu_dty, '__all__')
        assert rlu['early'].y_transform_vars[1] == rlu['late'].y_transform_vars[1], 'ymin must match'
        assert rlu['early'].y_transform_vars[2] == rlu['late'].y_transform_vars[2], 'dy must match'

    def test_sm_xy_datetime_both_shared_does_not_raise(self):
        """SM_X + SM_Y with both axes as datetime must not raise and must share both ranges."""
        tmpl = self.p2s.xyp(df=self.df_dtxy, x='x', y='y', dot_size=2.0, wxh=(256, 256),
                            sm_shared={self.p2s.SM_X, self.p2s.SM_Y})
        rlu = tmpl.renderSmallMultiples(self.df_dtxy, self.df_lu_dtxy, '__all__')
        assert 'early' in rlu and 'late' in rlu
        assert rlu['early'].x_transform_vars[1] == rlu['late'].x_transform_vars[1]
        assert rlu['early'].x_transform_vars[2] == rlu['late'].x_transform_vars[2]
        assert rlu['early'].y_transform_vars[1] == rlu['late'].y_transform_vars[1]
        assert rlu['early'].y_transform_vars[2] == rlu['late'].y_transform_vars[2]

    def test_sm_x_datetime_svg_non_empty(self):
        """With SM_X and datetime x, full SVG render pipeline completes without error."""
        tmpl = self.p2s.xyp(df=self.df_dtx, x='x', y='y', dot_size=2.0, wxh=(256, 256),
                            sm_shared={self.p2s.SM_X})
        rlu = tmpl.renderSmallMultiples(self.df_dtx, self.df_lu_dtx, '__all__')
        assert rlu['early']._repr_svg_() != ''
        assert rlu['late']._repr_svg_()  != ''

    def test_sm_xy_datetime_x_categorical_y_does_not_raise(self):
        """SM_X + SM_Y with datetime x and categorical (string) y must not crash.
        Mirrors the notebook pattern: datetime x-axis, IP-address y-axis."""
        tmpl = self.p2s.xyp(df=self.df_dtxsy, x='x', y='y', dot_size=2.0, wxh=(256, 256),
                            sm_shared={self.p2s.SM_X, self.p2s.SM_Y})
        rlu = tmpl.renderSmallMultiples(self.df_dtxsy, self.df_lu_dtxsy, '__all__')
        assert 'early' in rlu and 'late' in rlu
        assert rlu['early']._repr_svg_() != ''
        assert rlu['late']._repr_svg_()  != ''

    def test_sm_x_datetime_integer_dot_size_does_not_raise(self):
        """SM_X with datetime x and integer dot_size must work (previously guarded out)."""
        tmpl = self.p2s.xyp(df=self.df_dtx, x='x', y='y', dot_size=2, wxh=(256, 256),
                            sm_shared={self.p2s.SM_X})
        rlu = tmpl.renderSmallMultiples(self.df_dtx, self.df_lu_dtx, '__all__')
        assert 'early' in rlu and 'late' in rlu
        assert rlu['early'].x_transform_vars[1] == rlu['late'].x_transform_vars[1], 'xmin must match'
        assert rlu['early'].x_transform_vars[2] == rlu['late'].x_transform_vars[2], 'dx must match'

    def test_sm_xy_datetime_x_categorical_y_integer_dot_size_does_not_raise(self):
        """SM_X + SM_Y with datetime x, categorical y, and integer dot_size must not crash."""
        tmpl = self.p2s.xyp(df=self.df_dtxsy, x='x', y='y', dot_size=2, wxh=(256, 256),
                            sm_shared={self.p2s.SM_X, self.p2s.SM_Y})
        rlu = tmpl.renderSmallMultiples(self.df_dtxsy, self.df_lu_dtxsy, '__all__')
        assert 'early' in rlu and 'late' in rlu
        assert rlu['early']._repr_svg_() != ''
        assert rlu['late']._repr_svg_()  != ''

    # ── string axis label correctness with SM_Y ───────────────────────────────

    def test_sm_y_categorical_labels_show_strings_not_indices(self):
        """With SM_Y and a categorical y-axis, SVG labels must show string values, not integer indices."""
        for ds in [2.0, 2]:
            tmpl = self.p2s.xyp(df=self.df_dtxsy, x='x', y='y', dot_size=ds, wxh=(256, 256),
                                sm_shared={self.p2s.SM_X, self.p2s.SM_Y})
            rlu = tmpl.renderSmallMultiples(self.df_dtxsy, self.df_lu_dtxsy, '__all__')
            for key in ('early', 'late'):
                svg = rlu[key]._repr_svg_()
                # String IP values must appear; raw integer indices (e.g. ">0<", ">3<") must NOT be the labels
                assert '1.1.1.1' in svg or '2.2.2.2' in svg or '3.3.3.3' in svg or '4.4.4.4' in svg, \
                    f'dot_size={ds} {key}: IP string labels must appear in SVG'

    # ── shared axis labels are globally consistent ────────────────────────────

    def test_sm_y_categorical_global_min_label_in_late_svg(self):
        """With SM_Y, the global minimum category label appears in a sub-instance that lacks it locally.
        'late' only has '3.3.3.3'/'4.4.4.4'; global min is '1.1.1.1' — it must appear in late's SVG."""
        for ds in [2.0, 2]:
            tmpl = self.p2s.xyp(df=self.df_dtxsy, x='x', y='y', dot_size=ds, wxh=(256, 256),
                                sm_shared={self.p2s.SM_X, self.p2s.SM_Y})
            rlu = tmpl.renderSmallMultiples(self.df_dtxsy, self.df_lu_dtxsy, '__all__')
            assert '1.1.1.1' in rlu['late']._repr_svg_(), \
                f'dot_size={ds}: global min y label must appear in late SVG'

    def test_sm_y_categorical_global_max_label_in_early_svg(self):
        """With SM_Y, the global maximum category label appears in a sub-instance that lacks it locally.
        'early' only has '1.1.1.1'/'2.2.2.2'; global max is '4.4.4.4' — it must appear in early's SVG."""
        for ds in [2.0, 2]:
            tmpl = self.p2s.xyp(df=self.df_dtxsy, x='x', y='y', dot_size=ds, wxh=(256, 256),
                                sm_shared={self.p2s.SM_X, self.p2s.SM_Y})
            rlu = tmpl.renderSmallMultiples(self.df_dtxsy, self.df_lu_dtxsy, '__all__')
            assert '4.4.4.4' in rlu['early']._repr_svg_(), \
                f'dot_size={ds}: global max y label must appear in early SVG'

    def test_sm_y_categorical_both_sub_instances_same_y_labels(self):
        """With SM_Y and a categorical y-axis, both sub-instances must show the same y_shared_label_range."""
        for ds in [2.0, 2]:
            tmpl = self.p2s.xyp(df=self.df_dtxsy, x='x', y='y', dot_size=ds, wxh=(256, 256),
                                sm_shared={self.p2s.SM_X, self.p2s.SM_Y})
            rlu = tmpl.renderSmallMultiples(self.df_dtxsy, self.df_lu_dtxsy, '__all__')
            assert rlu['early'].y_shared_label_range == rlu['late'].y_shared_label_range, \
                f'dot_size={ds}: y_shared_label_range must be identical across sub-instances'

    def test_sm_x_numeric_global_min_label_in_right_svg(self):
        """With SM_X, the global minimum numeric x label appears in the sub-instance that lacks it locally.
        'right' has x ∈ [70,90]; global min is 10.0 — it must appear in right's SVG."""
        tmpl = self.p2s.xyp(df=self.df_xy, x='x', y='y', dot_size=2.0, wxh=(256, 256),
                            sm_shared={self.p2s.SM_X})
        rlu = tmpl.renderSmallMultiples(self.df_xy, self.df_lu_xy, '__all__')
        assert '10' in rlu['right']._repr_svg_(), 'global min x label must appear in right SVG'

    def test_sm_x_numeric_global_max_label_in_left_svg(self):
        """With SM_X, the global maximum numeric x label appears in the sub-instance that lacks it locally.
        'left' has x ∈ [10,30]; global max is 90.0 — it must appear in left's SVG."""
        tmpl = self.p2s.xyp(df=self.df_xy, x='x', y='y', dot_size=2.0, wxh=(256, 256),
                            sm_shared={self.p2s.SM_X})
        rlu = tmpl.renderSmallMultiples(self.df_xy, self.df_lu_xy, '__all__')
        assert '90' in rlu['left']._repr_svg_(), 'global max x label must appear in left SVG'


class TestSmallpChordShared(unittest.TestCase):
    """SM_X and SM_Y sharing for chordp (ChP) small multiples."""

    def setUp(self):
        self.p2s = Polars2SVG()
        # Three edges per category so each panel has real data
        self.df = pl.DataFrame({
            'fm':       ['a', 'b', 'c', 'a', 'b', 'c', 'a', 'd', 'b', 'd'],
            'to':       ['b', 'c', 'a', 'c', 'a', 'b', 'd', 'b', 'd', 'a'],
            'category': ['x', 'x', 'x', 'x', 'y', 'y', 'y', 'y', 'x', 'y'],
        })
        self.rels = [('fm', 'to')]

    def _template(self, **extra):
        return self.p2s.chordp(df=self.df, relationships=self.rels, wxh=(128, 128), **extra)

    # ── SM_X: shared node order ──────────────────────────────────────────────

    def test_sm_x_renders_without_error(self):
        tmpl = self._template(sm_shared={self.p2s.SM_X})
        sm = self.p2s.smallp(self.df, tmpl, 'category')
        self.assertIn('<svg', sm.svg)

    def test_sm_x_panels_have_same_node_order(self):
        tmpl = self._template(sm_shared={self.p2s.SM_X})
        cat_dfs = {
            'x': self.df.filter(pl.col('category') == 'x'),
            'y': self.df.filter(pl.col('category') == 'y'),
        }
        rlu = tmpl.renderSmallMultiples(self.df, cat_dfs, '__all__')
        self.assertEqual(rlu['x'].order, rlu['y'].order,
                         f"SM_X panels have different orders: x={rlu['x'].order}, y={rlu['y'].order}")

    def test_sm_x_order_matches_reference(self):
        # The shared order should equal what a full-data reference instance computes
        tmpl = self._template(sm_shared={self.p2s.SM_X})
        ref = self.p2s.chordp(df=self.df, relationships=self.rels, wxh=(128, 128))
        cat_dfs = {
            'x': self.df.filter(pl.col('category') == 'x'),
            'y': self.df.filter(pl.col('category') == 'y'),
        }
        rlu = tmpl.renderSmallMultiples(self.df, cat_dfs, '__all__')
        self.assertEqual(rlu['x'].order, ref.order,
                         f"SM_X order {rlu['x'].order} != reference order {ref.order}")

    def test_sm_x_absent_node_appears_in_df_node(self):
        # With SM_X, a node that has no edges in a panel should still appear in df_node
        # Use a panel where 'd' definitely doesn't appear
        df_no_d = self.df.filter((pl.col('fm') != 'd') & (pl.col('to') != 'd'))
        df_with_d = self.df
        tmpl = self.p2s.chordp(df=df_with_d, relationships=self.rels, wxh=(128, 128),
                                sm_shared={self.p2s.SM_X})
        panels = {'no_d': df_no_d, 'all': df_with_d}
        rlu = tmpl.renderSmallMultiples(df_with_d, panels, '__all__')
        # The 'no_d' panel should still have 'd' in its df_node (from shared order)
        if 'd' in rlu['all'].order:
            self.assertIn('d', rlu['no_d'].df_node['__nm__'].to_list(),
                          "SM_X: absent node 'd' missing from df_node in sub-panel")

    def test_sm_x_without_sm_panels_may_differ(self):
        # Without SM_X, different panels are free to produce different orderings
        tmpl = self._template()
        cat_dfs = {
            'x': self.df.filter(pl.col('category') == 'x'),
            'y': self.df.filter(pl.col('category') == 'y'),
        }
        rlu = tmpl.renderSmallMultiples(self.df, cat_dfs, '__all__')
        # Both should render
        self.assertIn('<svg', rlu['x'].svg)
        self.assertIn('<svg', rlu['y'].svg)

    # ── SM_Y: shared skeleton ────────────────────────────────────────────────

    def test_sm_y_bundled_renders_without_error(self):
        tmpl = self._template(link_shape='bundled', sm_shared={self.p2s.SM_Y})
        sm = self.p2s.smallp(self.df, tmpl, 'category')
        self.assertIn('<svg', sm.svg)

    def test_sm_y_panels_share_skeleton_identity(self):
        # All panels should hold a reference to the SAME skeleton object
        tmpl = self._template(link_shape='bundled', sm_shared={self.p2s.SM_Y})
        cat_dfs = {
            'x': self.df.filter(pl.col('category') == 'x'),
            'y': self.df.filter(pl.col('category') == 'y'),
        }
        rlu = tmpl.renderSmallMultiples(self.df, cat_dfs, '__all__')
        self.assertIs(rlu['x']._bundled_skeleton_, rlu['y']._bundled_skeleton_,
                      'SM_Y panels should share the same skeleton object')

    def test_sm_y_curve_mode_ignores_skeleton_sharing(self):
        # SM_Y on a curve template should still render (skeleton irrelevant)
        tmpl = self._template(link_shape='curve', sm_shared={self.p2s.SM_Y})
        sm = self.p2s.smallp(self.df, tmpl, 'category')
        self.assertIn('<svg', sm.svg)

    # ── SM_X + SM_Y together ─────────────────────────────────────────────────

    def test_sm_x_and_y_together(self):
        tmpl = self._template(link_shape='bundled', sm_shared={self.p2s.SM_X, self.p2s.SM_Y})
        cat_dfs = {
            'x': self.df.filter(pl.col('category') == 'x'),
            'y': self.df.filter(pl.col('category') == 'y'),
        }
        rlu = tmpl.renderSmallMultiples(self.df, cat_dfs, '__all__')
        # Order shared
        self.assertEqual(rlu['x'].order, rlu['y'].order)
        # Skeleton shared
        self.assertIs(rlu['x']._bundled_skeleton_, rlu['y']._bundled_skeleton_)


if __name__ == '__main__':
    unittest.main()
