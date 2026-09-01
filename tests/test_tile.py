"""tile() — composing finished renderings into a strip or a grid.

Geometry is asserted through the public attributes (.content_wxh / .wxh_actual /
.xy_list / .scale) plus the markup those produce, since tile() has no data model of
its own: everything it does is measure children and place them.
"""
import re
import unittest

import polars as pl
from polars2svg import Polars2SVG

from svg_test_utils import assert_valid_svg, assert_timing_metrics_populated


def _svg_(w, h):
    '''A minimal stand-in tile of a known size.'''
    return f'<svg x="0" y="0" width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg"></svg>'


def _root_wxh_(svg):
    _tag_ = re.match(r'<svg[^>]*>', svg).group(0)
    return (float(re.search(r'\bwidth="([^"]*)"',  _tag_).group(1)),
            float(re.search(r'\bheight="([^"]*)"', _tag_).group(1)))


class TestTileStrip(unittest.TestCase):
    '''The single-strip layouts: per_row=None is a row, per_row=1 is a column.'''

    @classmethod
    def setUpClass(cls):
        cls.p2s = Polars2SVG()

    def test_horizontal_strip_is_sum_by_max(self):
        t = self.p2s.tile([_svg_(100, 80), _svg_(120, 60), _svg_(90, 90)])
        self.assertEqual(t.content_wxh, (310, 90))
        self.assertEqual(t.wxh_actual,  (310, 90))
        self.assertEqual(t.xy_list, [(0, 0), (100, 0), (220, 0)])

    def test_column_is_max_by_sum(self):
        t = self.p2s.tile([_svg_(100, 80), _svg_(120, 60), _svg_(90, 90)], per_row=1)
        self.assertEqual(t.content_wxh, (120, 230))
        self.assertEqual(t.xy_list, [(0, 0), (0, 80), (0, 140)])

    def test_root_svg_matches_content_size(self):
        t = self.p2s.tile([_svg_(100, 80), _svg_(120, 60)])
        assert_valid_svg(self, t.svg)
        self.assertEqual(_root_wxh_(t.svg), t.wxh_actual)

    def test_every_child_is_placed(self):
        t = self.p2s.tile([_svg_(10, 10), _svg_(20, 20), _svg_(30, 30)])
        self.assertEqual(t.svg.count('<g transform="translate('), 3)
        for _w_ in (10, 20, 30):
            self.assertIn(f'width="{_w_}"', t.svg)

    def test_repr_svg_is_the_svg_attribute(self):
        t = self.p2s.tile([_svg_(10, 10)])
        self.assertEqual(t._repr_svg_(), t.svg)

    def test_timing_metrics_populated(self):
        assert_timing_metrics_populated(self, self.p2s.tile([_svg_(10, 10)]))


class TestTileGrid(unittest.TestCase):
    '''per_row=<n> — rtsvg's table(): row-major grid, last row may be short.'''

    @classmethod
    def setUpClass(cls):
        cls.p2s = Polars2SVG()

    def test_grid_rows_are_ragged_and_row_sized(self):
        # row 0: 100x80 + 120x60  -> 220 wide, 80 tall
        # row 1:  90x90 + 110x40  -> 200 wide, 90 tall
        # row 2:  60x50           ->  60 wide, 50 tall
        t = self.p2s.tile([_svg_(100, 80), _svg_(120, 60),
                           _svg_(90, 90),  _svg_(110, 40),
                           _svg_(60, 50)], per_row=2)
        self.assertEqual(t.content_wxh, (220, 220))
        self.assertEqual(t.xy_list, [(0, 0), (100, 0),
                                     (0, 80), (90, 80),
                                     (0, 170)])

    def test_per_row_larger_than_the_list_is_one_row(self):
        _tiles_ = [_svg_(100, 80), _svg_(120, 60)]
        self.assertEqual(self.p2s.tile(_tiles_, per_row=10).xy_list,
                         self.p2s.tile(_tiles_).xy_list)

    def test_single_rendering_needs_no_list(self):
        t = self.p2s.tile(_svg_(100, 80))
        self.assertEqual(t.content_wxh, (100, 80))
        self.assertEqual(t.xy_list, [(0, 0)])


class TestTileSpacer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.p2s = Polars2SVG()

    def test_scalar_spacer_applies_to_both_directions(self):
        t = self.p2s.tile([_svg_(100, 80), _svg_(120, 60),
                           _svg_(90, 90)], per_row=2, spacer=10)
        self.assertEqual(t.spacer, (10, 10))
        self.assertEqual(t.content_wxh, (230, 180))          # 100+10+120 ; 80+10+90
        self.assertEqual(t.xy_list, [(0, 0), (110, 0), (0, 90)])

    def test_tuple_spacer_is_horizontal_then_vertical(self):
        t = self.p2s.tile([_svg_(100, 80), _svg_(120, 60),
                           _svg_(90, 90)], per_row=2, spacer=(4, 20))
        self.assertEqual(t.spacer, (4, 20))
        self.assertEqual(t.content_wxh, (224, 190))          # 100+4+120 ; 80+20+90
        self.assertEqual(t.xy_list, [(0, 0), (104, 0), (0, 100)])

    def test_a_single_strip_uses_only_the_spacer_that_applies(self):
        _tiles_ = [_svg_(100, 80), _svg_(120, 60)]
        _row_   = self.p2s.tile(_tiles_, spacer=(6, 30))
        _col_   = self.p2s.tile(_tiles_, spacer=(6, 30), per_row=1)
        self.assertEqual(_row_.content_wxh, (226, 80))       # only the horizontal gap
        self.assertEqual(_col_.content_wxh, (120, 170))      # only the vertical gap

    def test_spacer_does_not_trail_the_last_tile(self):
        t = self.p2s.tile([_svg_(100, 80)], spacer=25)
        self.assertEqual(t.content_wxh, (100, 80))

    def test_list_spacer_is_accepted_like_a_tuple(self):
        self.assertEqual(self.p2s.tile([_svg_(10, 10)], spacer=[3, 4]).spacer, (3, 4))


class TestTileViewport(unittest.TestCase):
    '''wxh= scales the whole tiling into a fixed canvas.'''

    @classmethod
    def setUpClass(cls):
        cls.p2s = Polars2SVG()
        cls.tiles = [_svg_(100, 80), _svg_(120, 60), _svg_(90, 90)]   # 310 x 90 natural

    def test_wxh_sets_the_canvas_and_keeps_the_content_size(self):
        t = self.p2s.tile(self.tiles, wxh=(620, 400))
        self.assertEqual(t.content_wxh, (310, 90))
        self.assertEqual(t.wxh_actual,  (620, 400))
        self.assertEqual(_root_wxh_(t.svg), (620, 400))

    def test_scale_is_uniform_and_fits_the_binding_side(self):
        # 620/310 = 2.0 wide but 400/90 = 4.44 tall -> width binds, content is centered
        t = self.p2s.tile(self.tiles, wxh=(620, 400))
        self.assertEqual(t.scale, 2.0)
        self.assertEqual(t.offset, (0.0, 110.0))
        self.assertIn('translate(0,110) scale(2)', t.svg)

    def test_height_can_bind_instead(self):
        t = self.p2s.tile(self.tiles, wxh=(620, 45))
        self.assertEqual(t.scale, 0.5)
        self.assertEqual(t.offset, (232.5, 0.0))

    def test_wxh_with_one_side_none_follows_the_aspect_ratio(self):
        t = self.p2s.tile(self.tiles, wxh=(620, None))
        self.assertEqual(t.wxh_actual, (620, 180))
        self.assertEqual(t.scale, 2.0)
        t2 = self.p2s.tile(self.tiles, wxh=(None, 45))
        self.assertEqual(t2.wxh_actual, (155, 45))

    def test_wxh_matching_the_content_emits_no_transform(self):
        _natural_ = self.p2s.tile(self.tiles)
        _asked_   = self.p2s.tile(self.tiles, wxh=(310, 90))
        self.assertEqual(_asked_.scale, 1.0)
        self.assertEqual(_natural_.svg.count('<g transform='), _asked_.svg.count('<g transform='))

    def test_children_are_untouched_by_the_viewport(self):
        # Scaling is a transform on the composition, not a rewrite of the tiles.
        t = self.p2s.tile(self.tiles, wxh=(620, 400))
        for _child_ in self.tiles:
            self.assertIn(_child_, t.svg)


class TestTileBackground(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.p2s = Polars2SVG()

    def test_default_background_is_the_framework_background(self):
        t = self.p2s.tile([_svg_(10, 10)])
        self.assertIn(f'fill="{self.p2s.colorTyped("background", "default")}"', t.svg)

    def test_bg_color_fills_the_canvas(self):
        t = self.p2s.tile([_svg_(10, 10)], bg_color='#123456')
        self.assertIn('fill="#123456"', t.svg)

    def test_bg_color_must_be_a_string(self):
        with self.assertRaises(ValueError):
            self.p2s.tile([_svg_(10, 10)], bg_color=(1, 2, 3))


class TestTileInputs(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.p2s = Polars2SVG()
        cls.df  = pl.DataFrame({'x': [1, 2, 3, 4], 'y': [3, 1, 4, 2],
                                'c': ['a', 'b', 'a', 'c']})

    def test_components_strings_and_tiles_mix(self):
        _xyp_  = self.p2s.xyp(self.df, 'x', 'y', wxh=(100, 100))
        _inner_ = self.p2s.tile([_svg_(20, 20), _svg_(20, 20)])
        t = self.p2s.tile([_xyp_, _svg_(50, 100), _inner_])
        self.assertEqual(t.content_wxh, (190, 100))

    def test_measures_the_rendered_size_not_the_requested_one(self):
        # smallp auto-sizes the side it was given as None; the tiling must use what
        # was actually rendered, not the wxh the component was asked for.
        _sm_ = self.p2s.smallp(self.df, 'c', self.p2s.xyp(self.df, 'x', 'y', wxh=(64, 64)),
                               wxh=(200, None))
        self.assertIsNone(_sm_.wxh[1])
        t = self.p2s.tile([_sm_])
        self.assertEqual(t.content_wxh, tuple(float(v) for v in _sm_.wxh_actual))

    def test_empty_list_renders_a_placeholder(self):
        t = self.p2s.tile([])
        assert_valid_svg(self, t.svg)
        self.assertIn('no data', t.svg)
        self.assertEqual(t.wxh_actual, (256, 256))

    def test_empty_list_honors_wxh(self):
        self.assertEqual(self.p2s.tile([], wxh=(300, 200)).wxh_actual, (300, 200))

    def test_non_rendering_item_names_the_index_and_type(self):
        with self.assertRaises(ValueError) as _cm_:
            self.p2s.tile([_svg_(10, 10), self.df])
        self.assertIn('svg_list[1]',  str(_cm_.exception))
        self.assertIn('DataFrame',    str(_cm_.exception))

    def test_child_without_a_readable_size_raises(self):
        with self.assertRaises(ValueError):
            self.p2s.tile(['<svg xmlns="http://www.w3.org/2000/svg"></svg>'])

    def test_child_with_units_raises(self):
        with self.assertRaises(ValueError):
            self.p2s.tile(['<svg width="10px" height="10px"></svg>'])

    def test_svg_list_as_a_keyword(self):
        self.assertEqual(self.p2s.tile(svg_list=[_svg_(10, 10)]).content_wxh, (10, 10))

    def test_svg_list_twice_raises(self):
        with self.assertRaises(ValueError):
            self.p2s.tile([_svg_(10, 10)], svg_list=[_svg_(10, 10)])


class TestTileValidation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.p2s = Polars2SVG()

    def test_horz_is_not_a_parameter(self):
        # rtsvg's horz= is per_row= here: horz=True is the default, horz=False is
        # per_row=1.  It is rejected rather than quietly ignored.
        with self.assertRaises(TypeError):
            self.p2s.tile([_svg_(10, 10)], horz=False)

    def test_per_row_must_be_a_positive_int(self):
        for _bad_ in (0, -1, 2.5, True, 'two'):
            with self.subTest(per_row=_bad_), self.assertRaises(ValueError):
                self.p2s.tile([_svg_(10, 10)], per_row=_bad_)

    def test_spacer_must_be_a_number_or_a_pair(self):
        for _bad_ in ('4', (1, 2, 3), (1,), (1, 'x'), -3, (2, -1), True):
            with self.subTest(spacer=_bad_), self.assertRaises(ValueError):
                self.p2s.tile([_svg_(10, 10)], spacer=_bad_)

    def test_wxh_is_validated_like_every_other_component(self):
        for _bad_ in ((1, 2, 3), 'big', (None, None), (200, 'tall')):
            with self.subTest(wxh=_bad_), self.assertRaises(ValueError):
                self.p2s.tile([_svg_(10, 10)], wxh=_bad_)

    def test_unknown_kwarg_raises_type_error(self):
        with self.assertRaises(TypeError):
            self.p2s.tile([_svg_(10, 10)], spacing=4)


if __name__ == '__main__':
    unittest.main()
