"""aspect= on XYp -- locking the ratio between the two axis scales.

aspect is defined as world-units-per-pixel on x relative to y (equivalently,
pixels-per-y-unit / pixels-per-x-unit), matching matplotlib's set_aspect().  It
is applied by *widening* the over-magnified axis about its center, so nothing
that was visible without it becomes invisible with it.
"""
import math
import unittest

import polars as pl

from polars2svg import Polars2SVG


def _scale_ratio_(_xyp_):
    """world units per pixel on x, relative to y -- what aspect= pins."""
    _w_, _h_ = _xyp_.plot_size
    _xlo_, _xhi_ = _xyp_.x_effective_range
    _ylo_, _yhi_ = _xyp_.y_effective_range
    return ((_xhi_ - _xlo_)/_w_) / ((_yhi_ - _ylo_)/_h_)


class Testxyp_aspect(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()
        # A square 10x10 degree extent centered on 45N -- the corners pin the extent
        self.df = pl.DataFrame({'lon': [0.0, 10.0,  0.0, 10.0, 5.0],
                                'lat': [40.0, 40.0, 50.0, 50.0, 45.0]})

    #
    # The default is unchanged: each axis independently fills its own pixel extent
    #
    def test_default_is_unchanged(self):
        _xyp_ = self.p2s.xyp(self.df, 'lon', 'lat', wxh=(512, 256), dot_size=3.0)
        self.assertIsNone(_xyp_.aspect)
        self.assertEqual(_xyp_.x_effective_range, (0.0, 10.0))
        self.assertEqual(_xyp_.y_effective_range, (40.0, 50.0))

    #
    # aspect='equal' / a numeric ratio / 'geo' all pin the scale ratio
    #
    def test_equal_pins_the_ratio(self):
        for _wxh_ in [(512, 256), (256, 512), (300, 301), (256, 256)]:
            with self.subTest(wxh=_wxh_):
                _xyp_ = self.p2s.xyp(self.df, 'lon', 'lat', wxh=_wxh_, dot_size=3.0, aspect='equal')
                self.assertAlmostEqual(_scale_ratio_(_xyp_), 1.0, places=9)

    def test_numeric_ratio_pins_the_ratio(self):
        for _a_ in [0.25, 1.0, 2.0, 7.5]:
            with self.subTest(aspect=_a_):
                _xyp_ = self.p2s.xyp(self.df, 'lon', 'lat', wxh=(512, 256), dot_size=3.0, aspect=_a_)
                self.assertAlmostEqual(_scale_ratio_(_xyp_), _a_, places=9)

    def test_geo_uses_the_cosine_of_the_center_latitude(self):
        _xyp_ = self.p2s.xyp(self.df, 'lon', 'lat', wxh=(512, 256), dot_size=3.0, aspect='geo')
        self.assertAlmostEqual(_scale_ratio_(_xyp_), 1.0/math.cos(math.radians(45.0)), places=9)

    def test_geo_at_the_equator_matches_equal(self):
        _df_  = pl.DataFrame({'lon': [-5.0, 5.0], 'lat': [-5.0, 5.0]})
        _geo_ = self.p2s.xyp(_df_, 'lon', 'lat', wxh=(512, 256), dot_size=3.0, aspect='geo')
        _eq_  = self.p2s.xyp(_df_, 'lon', 'lat', wxh=(512, 256), dot_size=3.0, aspect='equal')
        self.assertAlmostEqual(_geo_.x_effective_range[0], _eq_.x_effective_range[0], places=9)
        self.assertAlmostEqual(_geo_.x_effective_range[1], _eq_.x_effective_range[1], places=9)

    #
    # Widen-only, and about the center: the data window never shrinks and never shifts
    #
    def test_widens_the_short_axis_only_and_stays_centered(self):
        # wide canvas, square data -> x widens, y untouched
        _wide_ = self.p2s.xyp(self.df, 'lon', 'lat', wxh=(512, 256), dot_size=3.0, aspect='equal')
        self.assertEqual(_wide_.y_effective_range, (40.0, 50.0))
        self.assertLess   (_wide_.x_effective_range[0], 0.0)
        self.assertGreater(_wide_.x_effective_range[1], 10.0)
        self.assertAlmostEqual(sum(_wide_.x_effective_range)/2.0, 5.0, places=9)
        # tall canvas, square data -> y widens, x untouched
        _tall_ = self.p2s.xyp(self.df, 'lon', 'lat', wxh=(256, 512), dot_size=3.0, aspect='equal')
        self.assertEqual(_tall_.x_effective_range, (0.0, 10.0))
        self.assertLess   (_tall_.y_effective_range[0], 40.0)
        self.assertGreater(_tall_.y_effective_range[1], 50.0)
        self.assertAlmostEqual(sum(_tall_.y_effective_range)/2.0, 45.0, places=9)

    def test_no_rows_are_dropped(self):
        for _a_ in ['equal', 'geo', 0.5, 4.0]:
            with self.subTest(aspect=_a_):
                _xyp_ = self.p2s.xyp(self.df, 'lon', 'lat', wxh=(512, 256), dot_size=3.0, aspect=_a_)
                self.assertEqual(len(_xyp_.df_flat), len(self.df))

    def test_an_already_matching_ratio_is_left_alone(self):
        # 10x10 degrees of data in a square plot is already equal-aspect
        _base_ = self.p2s.xyp(self.df, 'lon', 'lat', wxh=(256, 256), dot_size=3.0)
        _asp_  = self.p2s.xyp(self.df, 'lon', 'lat', wxh=(256, 256), dot_size=3.0, aspect='equal')
        self.assertAlmostEqual(_asp_.x_effective_range[0], _base_.x_effective_range[0], places=9)
        self.assertAlmostEqual(_asp_.x_effective_range[1], _base_.x_effective_range[1], places=9)
        self.assertAlmostEqual(_asp_.y_effective_range[0], _base_.y_effective_range[0], places=9)
        self.assertAlmostEqual(_asp_.y_effective_range[1], _base_.y_effective_range[1], places=9)

    #
    # The whole point: a square in data units renders square on screen
    #
    def test_a_data_unit_square_renders_square(self):
        _xyp_ = self.p2s.xyp(self.df, 'lon', 'lat', wxh=(512, 256), dot_size=3.0, aspect='equal')
        _x0_, _y0_ = _xyp_.wxToSx(0.0),  _xyp_.wyToSy(40.0)
        _x1_, _y1_ = _xyp_.wxToSx(10.0), _xyp_.wyToSy(50.0)
        self.assertAlmostEqual(abs(_x1_ - _x0_), abs(_y1_ - _y0_), places=6)

    #
    # The transform, the grid lines and the axis labels agree on the window
    #
    def test_transform_matches_the_effective_range(self):
        _xyp_ = self.p2s.xyp(self.df, 'lon', 'lat', wxh=(512, 256), dot_size=3.0, aspect='equal')
        _xlo_, _xhi_ = _xyp_.x_effective_range
        _ylo_, _yhi_ = _xyp_.y_effective_range
        self.assertAlmostEqual(_xyp_.wxToSx(_xlo_), _xyp_.plot_origin[0],                     places=6)
        self.assertAlmostEqual(_xyp_.wxToSx(_xhi_), _xyp_.plot_origin[0] + _xyp_.plot_size[0], places=6)
        self.assertAlmostEqual(_xyp_.wyToSy(_ylo_), _xyp_.plot_origin[1],                     places=6)
        self.assertAlmostEqual(_xyp_.wyToSy(_yhi_), _xyp_.plot_origin[1] - _xyp_.plot_size[1], places=6)
        # and the round trip
        self.assertAlmostEqual(_xyp_.sxToWx(_xyp_.wxToSx(3.25)), 3.25, places=6)
        self.assertAlmostEqual(_xyp_.syToWy(_xyp_.wyToSy(47.5)), 47.5, places=6)

    def test_axis_labels_and_grid_lines_report_the_widened_window(self):
        import re
        _base_ = self.p2s.xyp(self.df, 'lon', 'lat', wxh=(512, 256), dot_size=3.0)
        _asp_  = self.p2s.xyp(self.df, 'lon', 'lat', wxh=(512, 256), dot_size=3.0, aspect='equal')
        def _labels_(_xyp_): return re.findall(r'<text[^>]*>([^<]*)</text>', _xyp_.svg)
        # without aspect the x axis is labelled at the data extent
        self.assertIn('0.0',  _labels_(_base_))
        self.assertIn('10.0', _labels_(_base_))
        # with it, both the end labels and the grid lines follow the widened window
        _asp_labels_ = _labels_(_asp_)
        self.assertTrue(any(_l_.startswith('-5.31') for _l_ in _asp_labels_),
                        f'widened x min missing from {_asp_labels_}')
        self.assertTrue(any(_l_.startswith('15.31') for _l_ in _asp_labels_),
                        f'widened x max missing from {_asp_labels_}')
        # grid lines outside the data extent only exist because the window widened
        self.assertIn('-4', _asp_labels_)
        self.assertIn('14', _asp_labels_)
        # the y axis was not widened, so its labels are untouched
        for _l_ in ('40.0', '50.0'): self.assertIn(_l_, _asp_labels_)

    #
    # Interaction with x_range / y_range: the requested window is a floor, never a ceiling
    #
    def test_explicit_ranges_are_widened_not_narrowed(self):
        _xyp_ = self.p2s.xyp(self.df, 'lon', 'lat', wxh=(512, 256), dot_size=3.0,
                             x_range=(0.0, 10.0), y_range=(40.0, 50.0), aspect='equal')
        self.assertAlmostEqual(_scale_ratio_(_xyp_), 1.0, places=9)
        self.assertLessEqual   (_xyp_.x_effective_range[0], 0.0)
        self.assertGreaterEqual(_xyp_.x_effective_range[1], 10.0)
        self.assertEqual(_xyp_.y_effective_range, (40.0, 50.0))

    def test_the_row_filter_uses_the_widened_window(self):
        # x_range clips to the left half; equal aspect on a wide canvas widens it back
        # out, and the rows that widening brought into view are kept rather than left
        # as an empty band inside the frame.
        _df_  = pl.DataFrame({'lon': [0.0, 2.0, 4.0, 6.0, 8.0, 10.0],
                              'lat': [40.0, 42.0, 44.0, 46.0, 48.0, 50.0]})
        _clip_ = self.p2s.xyp(_df_, 'lon', 'lat', wxh=(512, 256), dot_size=3.0, x_range=(0.0, 4.0))
        _asp_  = self.p2s.xyp(_df_, 'lon', 'lat', wxh=(512, 256), dot_size=3.0, x_range=(0.0, 4.0), aspect='equal')
        self.assertEqual(len(_clip_.df_flat), 3)
        self.assertGreater(len(_asp_.df_flat), len(_clip_.df_flat))
        for _row_ in _asp_.df_flat.iter_rows(named=True):
            self.assertGreaterEqual(_row_['__xi__'], _asp_.x_effective_range[0])
            self.assertLessEqual   (_row_['__xi__'], _asp_.x_effective_range[1])

    #
    # Composition with the rest of the component
    #
    def test_renders_with_the_usual_options(self):
        for _kw_ in [{'dot_size': 3},                                    # integer raster path
                     {'dot_size': 3, 'dot_size_supersample': 2},
                     {'dot_size': 3.0, 'draw_context': False},
                     {'dot_size': 3.0, 'x_distributions': self.p2s.ROW_COUNTp,
                                       'y_distributions': self.p2s.ROW_COUNTp},
                     {'dot_size': 3.0, 'color': self.p2s.CROW_MAGNITUDEp},
                     {'dot_size': 3.0, 'line': 'lon'}]:
            with self.subTest(kw=sorted(_kw_)):
                _xyp_ = self.p2s.xyp(self.df, 'lon', 'lat', wxh=(512, 256), aspect='equal', **_kw_)
                self.assertIn('<svg', _xyp_.svg)
                self.assertAlmostEqual(_scale_ratio_(_xyp_), 1.0, places=9)

    def test_integer_dot_size_pins_the_ratio_on_the_snapped_plot_size(self):
        # the integer path trims plot_size down to a multiple of dot_size, and the
        # ratio has to be pinned against that trimmed size, not the raw wxh
        _xyp_ = self.p2s.xyp(self.df, 'lon', 'lat', wxh=(500, 250), dot_size=7, aspect='equal')
        self.assertEqual(_xyp_.plot_size[0] % 7, 0)
        self.assertAlmostEqual(_scale_ratio_(_xyp_), 1.0, places=9)

    def test_background_shapes_track_the_widened_window(self):
        # the geospatial workflow: an outline behind the points.  Both go through
        # x_transform_vars, so the shape has to land on the points it encloses.
        from shapely.geometry import Polygon
        _box_ = Polygon([(0.0, 40.0), (10.0, 40.0), (10.0, 50.0), (0.0, 50.0)])
        _xyp_ = self.p2s.xyp(self.df, 'lon', 'lat', wxh=(512, 256), dot_size=3.0,
                             aspect='equal', background={'box': _box_})
        _x0_, _y0_ = _xyp_.wxToSx(0.0),  _xyp_.wyToSy(40.0)
        _x1_, _y1_ = _xyp_.wxToSx(10.0), _xyp_.wyToSy(50.0)
        # the outline is square on screen, and does not span the full plot width
        self.assertAlmostEqual(abs(_x1_ - _x0_), abs(_y1_ - _y0_), places=6)
        self.assertLess(abs(_x1_ - _x0_), _xyp_.plot_size[0])
        # every dot falls inside it
        for _row_ in _xyp_.df_pixels.iter_rows(named=True):
            self.assertGreaterEqual(_row_['__xpx__'], _x0_ - 1)
            self.assertLessEqual   (_row_['__xpx__'], _x1_ + 1)

    def test_distributions_stay_under_the_dots(self):
        # distribution bars are emitted as 0..1 fractions and stretched across the full
        # plot extent, so they have to be binned over the widened window -- otherwise
        # they spread across the whole axis while the dots sit in the middle
        _base_ = self.p2s.xyp(self.df, 'lon', 'lat', wxh=(512, 256), dot_size=3.0,
                              x_distributions=self.p2s.ROW_COUNTp)
        _asp_  = self.p2s.xyp(self.df, 'lon', 'lat', wxh=(512, 256), dot_size=3.0, aspect='equal',
                              x_distributions=self.p2s.ROW_COUNTp)
        def _occupied_(_xyp_):
            _d_  = _xyp_.df_x_distribution.filter(pl.col('__xi_total__') > 0)
            return float(_d_['__xdists_xi_min__'].min()), float(_d_['__xdists_xi_max__'].max())
        # without aspect the data fills the axis
        _lo_, _hi_ = _occupied_(_base_)
        self.assertAlmostEqual(_lo_, 0.0, places=6)
        self.assertAlmostEqual(_hi_, 1.0, places=6)
        # with it, the bars occupy exactly the fraction of the window the data spans
        _lo_, _hi_   = _occupied_(_asp_)
        _wlo_, _whi_ = _asp_.x_effective_range
        self.assertAlmostEqual(_lo_, (0.0  - _wlo_)/(_whi_ - _wlo_), delta=1.5/len(_asp_.df_x_distribution))
        self.assertAlmostEqual(_hi_, (10.0 - _wlo_)/(_whi_ - _wlo_), delta=1.5/len(_asp_.df_x_distribution))
        # no rows were lost to the rebinning
        self.assertEqual(_asp_.df_x_distribution['__xi_total__'].sum(), float(len(self.df)))

    def test_the_ratio_is_pinned_against_the_plot_box_not_wxh(self):
        # outside distributions eat into the plot box, and the widening has to follow
        # the box that is left rather than the requested wxh
        _plain_ = self.p2s.xyp(self.df, 'lon', 'lat', wxh=(512, 256), dot_size=3.0, aspect='equal')
        _dist_  = self.p2s.xyp(self.df, 'lon', 'lat', wxh=(512, 256), dot_size=3.0, aspect='equal',
                               x_distributions=self.p2s.ROW_COUNTp)
        self.assertLess(_dist_.plot_size[1], _plain_.plot_size[1])   # the box did shrink
        self.assertAlmostEqual(_scale_ratio_(_dist_), 1.0, places=9)
        # a shorter box needs a wider x window to hold the ratio
        self.assertLess(_dist_.x_effective_range[0], _plain_.x_effective_range[0])

    def test_interactive_helpers_use_the_widened_window(self):
        _xyp_ = self.p2s.xyp(self.df, 'lon', 'lat', wxh=(512, 256), dot_size=6.0, aspect='equal')
        # a screen rectangle covering the whole plot selects every row
        _all_ = _xyp_.filterByRectangle((_xyp_.plot_origin[0], _xyp_.plot_origin[1] - _xyp_.plot_size[1],
                                         _xyp_.plot_origin[0] + _xyp_.plot_size[0], _xyp_.plot_origin[1]))
        self.assertEqual(len(_all_), len(self.df))
        # the left half of the *widened* window holds no data at all
        _empty_ = _xyp_.filterByRectangle((_xyp_.plot_origin[0], _xyp_.plot_origin[1] - _xyp_.plot_size[1],
                                           _xyp_.wxToSx(-1.0),   _xyp_.plot_origin[1]))
        self.assertEqual(len(_empty_), 0)
        # and a point picked at a known world coordinate hits the row that is there
        self.assertGreater(len(_xyp_.recordsAt((_xyp_.wxToSx(5.0), _xyp_.wyToSy(45.0)))), 0)

    def test_template_carries_aspect(self):
        _xyp_ = self.p2s.xyp(self.df, 'lon', 'lat', wxh=(512, 256), dot_size=3.0, aspect='geo')
        _tmpl_ = self.p2s.xyp(_xyp_, df=self.df)
        self.assertEqual(_tmpl_.aspect, 'geo')
        self.assertAlmostEqual(_scale_ratio_(_tmpl_), _scale_ratio_(_xyp_), places=9)

    #
    # Validation
    #
    def test_small_multiples_share_the_widened_window(self):
        _df_ = pl.DataFrame({'lon': [0.0, 10.0,  0.0, 10.0, 5.0, 3.0],
                             'lat': [40.0, 40.0, 50.0, 50.0, 45.0, 42.0],
                             'g':   ['a', 'a', 'b', 'b', 'a', 'b']})
        _tmpl_ = self.p2s.xyp(_df_, 'lon', 'lat', wxh=(160, 80), dot_size=3.0, aspect='equal',
                              sm_shared={self.p2s.SM_X, self.p2s.SM_Y})
        _sm_    = self.p2s.smallp(_df_, _tmpl_, 'g')
        _tiles_ = [_v_ for _v_ in _sm_._render_lu_.values() if _v_.df is not None]
        self.assertGreater(len(_tiles_), 1)
        for _t_ in _tiles_:
            # every tile keeps the ratio and the same window as the first
            self.assertAlmostEqual(_scale_ratio_(_t_), 1.0, places=9)
            self.assertAlmostEqual(_t_.x_effective_range[0], _tiles_[0].x_effective_range[0], places=9)
            self.assertAlmostEqual(_t_.x_effective_range[1], _tiles_[0].x_effective_range[1], places=9)
            # and the shared end labels name that window, not the data extent
            self.assertAlmostEqual(_t_.x_shared_label_range[0], _t_.x_effective_range[0], places=6)
            self.assertAlmostEqual(_t_.x_shared_label_range[1], _t_.x_effective_range[1], places=6)
        self.assertLess(_tiles_[0].x_effective_range[0], 0.0)   # it really did widen

    def test_bad_aspect_values_raise(self):
        for _bad_ in ['equalish', 'EQUAL', 0.0, -1.0, True, [1], float('nan'), float('inf')]:
            with self.subTest(aspect=_bad_):
                with self.assertRaises(ValueError):
                    self.p2s.xyp(self.df, 'lon', 'lat', dot_size=3.0, aspect=_bad_)

    def test_non_numeric_axes_raise(self):
        _df_ = pl.DataFrame({'cat': ['a', 'b', 'c'], 'val': [1.0, 2.0, 3.0],
                             'ts':  [1, 2, 3]}).with_columns(
                                 pl.col('ts').cast(pl.Datetime('us')))
        with self.assertRaises(ValueError):
            self.p2s.xyp(_df_, 'cat', 'val', dot_size=3.0, aspect='equal')
        with self.assertRaises(ValueError):
            self.p2s.xyp(_df_, 'val', 'cat', dot_size=3.0, aspect='equal')
        with self.assertRaises(ValueError):
            self.p2s.xyp(_df_, 'ts',  'val', dot_size=3.0, aspect='equal')

    def test_unknown_kwarg_still_rejected(self):
        with self.assertRaises(TypeError):
            self.p2s.xyp(self.df, 'lon', 'lat', dot_size=3.0, aspct='equal')


if __name__ == '__main__':
    unittest.main()
