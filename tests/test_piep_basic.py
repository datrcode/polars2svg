import unittest
import re
import polars as pl
from polars2svg import Polars2SVG
from piep_dataframes import makePieDf, makeKnownPieDf
from svg_test_utils import assert_valid_svg, assert_timing_metrics_populated, capture_log_warnings


class TestPiepBasic(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def setUp(self):
        self.df = makePieDf(n=200)

    # ── bin_by specification ──────────────────────────────────────────────────

    def test_bin_by_positional_string(self):
        self.p2s.piep(self.df, 'cat')

    def test_bin_by_keyword(self):
        self.p2s.piep(self.df, bin_by='cat')

    def test_df_as_keyword_arg(self):
        self.p2s.piep(df=self.df, bin_by='cat')

    def test_bin_by_tuple_two_fields(self):
        self.p2s.piep(self.df, ('cat', 'group'))

    # ── styles ────────────────────────────────────────────────────────────────

    def test_pie_default_style(self):
        t = self.p2s.piep(self.df, 'cat')
        self.assertEqual(t.style, self.p2s.PIEp)
        self.assertEqual(t.r_inner, 0.0)

    def test_donut_style_has_inner_radius(self):
        t = self.p2s.piep(self.df, 'cat', style=self.p2s.DONUTp)
        self.assertGreater(t.r_inner, 0.0)
        self.assertLess(t.r_inner, t.r)

    def test_waffle_style_renders_rects(self):
        # draw_border=False isolates the cell count from the (default-on) outer border rect.
        t = self.p2s.piep(self.df, 'cat', style=self.p2s.WAFFLEp, waffle_n=10, draw_border=False)
        # 100 cells + background rect
        n_rects = len(re.findall(r'<rect\b', t._repr_svg_()))
        self.assertEqual(n_rects, 101)

    def test_invalid_style_raises(self):
        with self.assertRaises(ValueError):
            self.p2s.piep(self.df, 'cat', style=self.p2s.BARCHARTp)

    # ── SVG output ────────────────────────────────────────────────────────────

    def test_repr_svg_returns_valid_svg_string(self):
        assert_valid_svg(self, self.p2s.piep(self.df, 'cat')._repr_svg_())

    def test_pie_svg_contains_paths(self):
        '''A pie chart draws each slice as an SVG <path>.'''
        self.assertIn('<path', self.p2s.piep(self.df, 'cat')._repr_svg_())

    def test_svg_dimensions_match_wxh(self):
        svg = self.p2s.piep(self.df, 'cat', wxh=(300, 300))._repr_svg_()
        self.assertIn('width="300"',  svg)
        self.assertIn('height="300"', svg)

    # ── slices / ordering ─────────────────────────────────────────────────────

    def test_slices_populated(self):
        t = self.p2s.piep(self.df, 'cat')
        self.assertGreater(len(t._slices_), 0)

    def test_slice_fractions_sum_to_one(self):
        t = self.p2s.piep(self.df, 'cat')
        self.assertAlmostEqual(sum(s['frac'] for s in t._slices_), 1.0, places=5)

    def test_slices_cover_full_circle(self):
        t = self.p2s.piep(self.df, 'cat')
        span = t._slices_[-1]['a1'] - t._slices_[0]['a0']
        self.assertAlmostEqual(span, 360.0, places=3)

    def test_default_order_is_count_descending(self):
        '''Known counts A=5,B=3,C=2 → slice order A,B,C.'''
        t = self.p2s.piep(makeKnownPieDf(), 'cat')
        self.assertEqual([s['bin'] for s in t._slices_], ['A', 'B', 'C'])

    def test_ascending_order(self):
        t = self.p2s.piep(makeKnownPieDf(), 'cat', descending=False)
        self.assertEqual([s['bin'] for s in t._slices_], ['C', 'B', 'A'])

    def test_known_fraction(self):
        '''A=5 of 10 rows → 50%.'''
        t = self.p2s.piep(makeKnownPieDf(), 'cat')
        a = next(s for s in t._slices_ if s['bin'] == 'A')
        self.assertAlmostEqual(a['frac'], 0.5, places=5)

    # ── count aggregation ─────────────────────────────────────────────────────

    def test_count_numeric_field_sums(self):
        t = self.p2s.piep(self.df, 'cat', count='value')
        self.assertGreater(len(t._slices_), 0)

    def test_count_set_field(self):
        self.p2s.piep(self.df, 'cat', count=('group', self.p2s.SETp))

    def test_count_field_not_found_raises(self):
        with self.assertRaises(ValueError):
            self.p2s.piep(self.df, 'cat', count='nonexistent')

    # ── color (mirrors xyp) ────────────────────────────────────────────────────

    def _path_fills(self, t):
        return re.findall(r'<path[^>]*fill="(#[0-9a-fA-F]+)"', t._repr_svg_())

    def _path_strokes(self, t):
        return re.findall(r'<path[^>]*stroke="(#[0-9a-fA-F]+)"', t._repr_svg_())

    def _rgb(self, h):
        return (int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16))

    def test_color_none_fills_each_slice_with_a_shade(self):
        '''Default color=None: each slice is filled with a data-color shade; the
        per-slice stroke is inherited from the delineation <g>, not on the path.'''
        df = pl.DataFrame({'cat': ['A'] * 5 + ['B'] * 3 + ['C'] * 2})
        svg = self.p2s.piep(df, 'cat')._repr_svg_()
        paths = re.findall(r'<path[^>]*/>', svg)
        self.assertEqual(len(paths), 3)
        for p in paths:
            self.assertIsNotNone(re.search(r'fill="(#[0-9a-fA-F]+)"', p))
            self.assertNotIn('stroke=', p)     # inherited from the wrapping group

    def test_slices_wrapped_in_bg_delineation_group(self):
        '''All slices (any color mode) sit inside a <g> that strokes them with the
        background color at 0.1 width for a thin separation.'''
        bg = self.p2s.colorTyped('background', 'default')
        for color in (None, 'cat'):
            df = pl.DataFrame({'cat': ['A'] * 5 + ['B'] * 3 + ['C'] * 2})
            svg = self.p2s.piep(df, 'cat', color=color)._repr_svg_()
            self.assertIn(f'<g stroke="{bg}" stroke-width="0.1">', svg)
            self.assertIn('</g>', svg)

    def test_color_none_uses_five_shades_of_data_color(self):
        '''Slices are filled in barely-distinct shades of the data color.'''
        df = pl.DataFrame({'cat': [c for c in 'ABCDEFGH']})
        t = self.p2s.piep(df, 'cat', wxh=(200, 200))
        fills = set(self._path_fills(t))
        self.assertLessEqual(len(fills), 5)
        self.assertGreater(len(fills), 1)
        # every shade is close to the data color (subtle variation)
        dr = self._rgb(self.p2s.colorTyped('data', 'default'))
        for s in fills:
            self.assertLessEqual(max(abs(a - b) for a, b in zip(self._rgb(s), dr)), 30)

    def test_color_none_no_adjacent_slices_same_shade(self):
        '''No two slices adjacent on the ring (including first↔last) share a shade.'''
        df = pl.DataFrame({'cat': [c for c in 'ABCDEFGHIJ']})
        s = self._path_fills(self.p2s.piep(df, 'cat', wxh=(200, 200)))
        self.assertEqual(len(s), 10)
        for i in range(len(s)):
            self.assertNotEqual(s[i], s[(i + 1) % len(s)])

    def test_color_none_shade_assignment_is_deterministic(self):
        '''Same data + settings → identical shade sequence across re-renders.'''
        df = pl.DataFrame({'cat': [c for c in 'ABCDEFGH']})
        a = self._path_fills(self.p2s.piep(df, 'cat', wxh=(200, 200)))
        b = self._path_fills(self.p2s.piep(df, 'cat', wxh=(200, 200)))
        self.assertEqual(a, b)
        self.assertTrue(len(set(a)) > 1)

    def test_color_none_single_slice_is_filled_circle(self):
        df = pl.DataFrame({'cat': ['A', 'A']})
        svg = self.p2s.piep(df, 'cat')._repr_svg_()
        self.assertRegex(svg, r'<circle[^>]*fill="#[0-9a-fA-F]{6}"')

    def test_color_none_donut_slices_filled(self):
        df = pl.DataFrame({'cat': ['A'] * 5 + ['B'] * 3})
        t = self.p2s.piep(df, 'cat', style=self.p2s.DONUTp)
        self.assertGreater(len(self._path_fills(t)), 0)

    def test_color_by_bin_field_is_categorical(self):
        '''color=<bin field> colors each slice by its own category.'''
        df = pl.DataFrame({'cat': ['A'] * 5 + ['B'] * 3 + ['C'] * 2})
        t = self.p2s.piep(df, 'cat', color='cat')
        self.assertEqual(len(set(self._path_fills(t))), 3)
        # matches the shared categorical hash used everywhere else
        self.assertEqual(self._path_fills(t)[0],
                         self.p2s.colors(['A'])['A'])

    def test_color_numeric_field_spectrum(self):
        t = self.p2s.piep(self.df, 'cat', color='value')
        self.assertTrue(t._color_is_spectrum_)
        self.assertIsNotNone(t._color_stat_min_)

    def test_color_categorical_other_field_cset(self):
        '''A categorical field != bin: single-value slices take that value's color,
        mixed slices share one set color (xyp CSETp).'''
        df = pl.DataFrame({
            'cat':    ['A', 'A', 'B', 'B', 'C', 'C'],
            'region': ['x', 'y', 'z', 'z', 'w', 'w'],   # A mixed; B,C single
        })
        t = self.p2s.piep(df, 'cat', color='region')
        lu = t._colorset_lu_
        self.assertIsNone(lu['A'])       # mixed → set color
        self.assertEqual(lu['B'], 'z')
        self.assertEqual(lu['C'], 'w')

    def test_color_crow_magnitude(self):
        t = self.p2s.piep(self.df, 'cat', color=self.p2s.CROW_MAGNITUDEp)
        self.assertTrue(t._color_is_crow_)
        self.assertTrue(t._color_is_spectrum_)

    def test_color_fixed_hex(self):
        t = self.p2s.piep(self.df, 'cat', color='#123456')
        self.assertEqual(set(self._path_fills(t)), {'#123456'})

    def test_color_hex_list_cycled(self):
        df = pl.DataFrame({'cat': ['A'] * 5 + ['B'] * 3 + ['C'] * 2})
        t = self.p2s.piep(df, 'cat', color=['#ff0000', '#00ff00', '#0000ff'])
        self.assertEqual(self._path_fills(t), ['#ff0000', '#00ff00', '#0000ff'])

    def test_color_enum_statistic_spectrum(self):
        for enum in (self.p2s.CMAGNITUDE_MAXp, self.p2s.CSTRETCHED_SUMp):
            t = self.p2s.piep(self.df, 'cat', color=('value', enum))
            self.assertTrue(t._color_is_spectrum_)

    def test_color_cset_magnitude_spectrum(self):
        t = self.p2s.piep(self.df, 'cat', color=('group', self.p2s.CSET_MAGNITUDEp))
        self.assertTrue(t._color_is_spectrum_)

    def test_color_magnitude_on_categorical_warns_and_falls_back(self):
        recs = capture_log_warnings(
            lambda: self.p2s.piep(self.df, 'cat', color=('group', self.p2s.CMAGNITUDE_SUMp)))
        self.assertTrue(any('numeric' in r.getMessage() for r in recs))

    def test_color_field_not_found_raises(self):
        with self.assertRaises(ValueError):
            self.p2s.piep(self.df, 'cat', color='nonexistent')

    def test_color_bare_magnitude_enum_requires_field_raises(self):
        with self.assertRaises(ValueError):
            self.p2s.piep(self.df, 'cat', color=self.p2s.CMAGNITUDE_SUMp)

    # ── edge cases ────────────────────────────────────────────────────────────

    def test_single_slice(self):
        df = pl.DataFrame({'cat': ['A', 'A', 'A'], 'value': [1, 2, 3]})
        t = self.p2s.piep(df, 'cat')
        self.assertEqual(len(t._slices_), 1)
        assert_valid_svg(self, t._repr_svg_())

    def test_single_slice_donut(self):
        df = pl.DataFrame({'cat': ['A', 'A'], 'value': [1, 2]})
        assert_valid_svg(self, self.p2s.piep(df, 'cat', style=self.p2s.DONUTp)._repr_svg_())

    def test_single_row(self):
        self.p2s.piep(self.df.head(1), 'cat')

    def test_empty_df_returns_blank_svg(self):
        t = self.p2s.piep(self.df.clear(), 'cat')
        self.assertIn('<svg', t._repr_svg_())

    def test_min_slice_deg_collapses_tiny_slices(self):
        # B,C,D each 1/100 of rows → 3.6° each; threshold 5° folds them into (other).
        df = pl.DataFrame({'cat': ['A'] * 97 + ['B', 'C', 'D']})
        t = self.p2s.piep(df, 'cat', min_slice_deg=5.0)
        bins = [s['bin'] for s in t._slices_]
        self.assertIn('(other)', bins)
        self.assertNotIn('B', bins)
        other = next(s for s in t._slices_ if s['bin'] == '(other)')
        self.assertEqual(other['count'], 3.0)   # B+C+D combined

    def test_min_slice_deg_default_keeps_discernible_slices(self):
        # Four equal slices of 90° each are all well above the 3° default.
        df = pl.DataFrame({'cat': ['A', 'B', 'C', 'D'] * 10})
        t = self.p2s.piep(df, 'cat')
        self.assertNotIn('(other)', [s['bin'] for s in t._slices_])

    def test_min_slice_deg_zero_disables_collapse(self):
        df = pl.DataFrame({'cat': ['A'] * 97 + ['B', 'C', 'D']})
        t = self.p2s.piep(df, 'cat', min_slice_deg=0.0)
        self.assertNotIn('(other)', [s['bin'] for s in t._slices_])

    # ── render options ────────────────────────────────────────────────────────

    def test_various_wxh(self):
        for w, h in [(80, 80), (160, 160), (300, 240)]:
            self.p2s.piep(self.df, 'cat', wxh=(w, h))

    def test_draw_context_false_no_text(self):
        t = self.p2s.piep(self.df, 'cat', draw_context=False, draw_labels=False)
        self.assertNotIn('<text', t._repr_svg_())

    # ── labels (draw_labels) ───────────────────────────────────────────────────

    def _texts(self, t):
        return re.findall(r'<text[^>]*>([^<]*)</text>', t._repr_svg_())

    def test_no_percentage_labels(self):
        '''Percentage labels were removed; no "%" should appear.'''
        t = self.p2s.piep(self.df, 'cat', wxh=(200, 200), draw_labels=True)
        self.assertNotIn('%', t._repr_svg_())

    def test_draw_labels_false_omits_category_labels(self):
        '''Default draw_labels=False: only the bin-field title, no category names.'''
        t = self.p2s.piep(self.df, 'cat', wxh=(200, 200))
        self.assertEqual(self._texts(t), ['cat'])

    def test_draw_labels_inside_slices(self):
        df = pl.DataFrame({'cat': ['Apple'] * 50 + ['Banana'] * 50})
        t = self.p2s.piep(df, 'cat', wxh=(220, 220), draw_labels=True)
        texts = self._texts(t)
        self.assertIn('Apple', texts)
        self.assertIn('Banana', texts)

    def test_draw_labels_outside_with_leader_lines(self):
        '''Thin slices + horizontal margin → outside labels connected by leader lines.'''
        df = pl.DataFrame({'cat': ['Big'] * 60 + [f'item_{i}' for i in range(10)]})
        t = self.p2s.piep(df, 'cat', wxh=(420, 200), draw_labels=True, min_slice_deg=0.0)
        svg = t._repr_svg_()
        self.assertIn('<line', svg)                       # leader lines drawn
        self.assertIn('item_0', self._texts(t))           # a thin slice labeled outside

    def test_draw_labels_tight_square_suppresses_outside(self):
        '''No margin around the pie → thin slices get no outside label (and no leaders).'''
        df = pl.DataFrame({'cat': ['Big'] * 60 + [f'item_{i}' for i in range(10)]})
        t = self.p2s.piep(df, 'cat', wxh=(150, 150), draw_labels=True, min_slice_deg=0.0,
                          draw_context=False)
        self.assertNotIn('<line', t._repr_svg_())
        self.assertNotIn('item_0', self._texts(t))

    def test_draw_labels_independent_of_draw_context(self):
        '''draw_labels draws category labels even when draw_context (title) is off.'''
        df = pl.DataFrame({'cat': ['Apple'] * 50 + ['Banana'] * 50})
        t = self.p2s.piep(df, 'cat', wxh=(220, 220), draw_labels=True, draw_context=False)
        texts = self._texts(t)
        self.assertIn('Apple', texts)
        self.assertNotIn('cat', texts)                    # title suppressed

    def test_draw_labels_crops_long_names(self):
        '''A long category name in a modest slice is cropped with an ellipsis.'''
        df = pl.DataFrame({'cat': ['ExtraordinarilyLongCategoryName'] * 50 + ['B'] * 50})
        t = self.p2s.piep(df, 'cat', wxh=(200, 200), draw_labels=True)
        self.assertTrue(any('...' in x for x in self._texts(t)))

    def test_draw_labels_outside_labels_stay_in_canvas(self):
        df = pl.DataFrame({'cat': ['Big'] * 60 + [f'i{i}' for i in range(14)]})
        w, h = 420, 220
        t = self.p2s.piep(df, 'cat', wxh=(w, h), draw_labels=True, min_slice_deg=0.0)
        for m in re.finditer(r'<text[^>]*\by="([-\d.]+)"', t._repr_svg_()):
            y = float(m.group(1))
            self.assertGreaterEqual(y, 0.0)
            self.assertLessEqual(y, h)

    def test_lazy_and_eager_both_render(self):
        self.assertIn('<svg', self.p2s.piep(self.df, 'cat', use_lazy_execution=True)._repr_svg_())
        self.assertIn('<svg', self.p2s.piep(self.df, 'cat', use_lazy_execution=False)._repr_svg_())

    def test_start_angle(self):
        t = self.p2s.piep(makeKnownPieDf(), 'cat', start_angle=0.0)
        self.assertAlmostEqual(t._slices_[0]['a0'], 0.0, places=5)

    # ── geometry sanity ───────────────────────────────────────────────────────

    def test_center_within_bounds(self):
        w, h = 200, 200
        t = self.p2s.piep(self.df, 'cat', wxh=(w, h))
        self.assertTrue(0 < t.cx < w and 0 < t.cy < h and t.r > 0)

    def test_wxh_float_coerced(self):
        # wxh now accepts any 2-sequence of numbers and coerces floats to int
        # (shared Polars2SVG.normalizeWxh); see tests/test_wxh_normalization.py
        t = self.p2s.piep(self.df, 'cat', wxh=(160.9, 160))
        self.assertEqual(t.wxh, (160, 160))

    def test_wxh_bad_still_raises(self):
        with self.assertRaises(ValueError):
            self.p2s.piep(self.df, 'cat', wxh=(160, 'x'))

    def test_unsupported_sm_warns(self):
        records = capture_log_warnings(
            lambda: self.p2s.piep(self.df, 'cat', sm_shared={self.p2s.SM_X})
        )
        self.assertTrue(any('sm_shared' in r.getMessage() for r in records))

    def test_timing_metrics_populated(self):
        t = self.p2s.piep(self.df, 'cat')
        assert_timing_metrics_populated(self, t,
            ('__parseInput__', '__validateInput__', '__renderSVG__'))


class TestPiepInteractivity(unittest.TestCase):
    def setUp(self):
        self.p2s = Polars2SVG()
        self.df  = makeKnownPieDf()

    def test_records_at_center_slice(self):
        '''A point inside the largest slice (A, top) returns A's rows.'''
        t = self.p2s.piep(self.df, 'cat', wxh=(200, 200), start_angle=-90.0)
        # A starts at top (-90°) sweeping clockwise; sample just right of 12 o'clock
        import math
        a = math.radians(-90 + 10)
        xy = (t.cx + t.r * 0.5 * math.cos(a), t.cy + t.r * 0.5 * math.sin(a))
        r = t.recordsAt(xy)
        self.assertTrue((r['cat'] == 'A').all())

    def test_records_at_outside_returns_empty(self):
        t = self.p2s.piep(self.df, 'cat', wxh=(200, 200))
        r = t.recordsAt((0, 0))
        self.assertEqual(len(r), 0)

    def test_records_at_donut_hole_returns_empty(self):
        t = self.p2s.piep(self.df, 'cat', wxh=(200, 200), style=self.p2s.DONUTp)
        r = t.recordsAt((t.cx, t.cy))   # dead center → inside the hole
        self.assertEqual(len(r), 0)

    def test_records_at_waffle(self):
        t = self.p2s.piep(self.df, 'cat', wxh=(200, 200), style=self.p2s.WAFFLEp)
        r = t.recordsAt((t.cx, t.cy))
        self.assertGreater(len(r), 0)

    def _point_in_slice(self, t, bin_value):
        '''A pixel at the mid-angle / mid-radius of the given slice's wedge.'''
        import math
        s = next(s for s in t._slices_ if s['bin'] == bin_value)
        a = math.radians((s['a0'] + s['a1']) / 2.0)
        rr = (t.r_inner + t.r) / 2.0 if t.r_inner > 0 else t.r * 0.5
        return (t.cx + rr * math.cos(a), t.cy + rr * math.sin(a))

    def test_click_selects_wedge(self):
        '''A click is a zero-area rectangle; clicking inside slice A selects A.'''
        t = self.p2s.piep(self.df, 'cat', wxh=(200, 200))
        px, py = self._point_in_slice(t, 'A')
        r = t.filterByRectangle((px, py, px, py))
        self.assertGreater(len(r), 0)
        self.assertTrue((r['cat'] == 'A').all())

    def test_click_selects_wedge_donut(self):
        t = self.p2s.piep(self.df, 'cat', wxh=(200, 200), style=self.p2s.DONUTp)
        px, py = self._point_in_slice(t, 'B')
        r = t.filterByRectangle((px, py, px, py))
        self.assertTrue((r['cat'] == 'B').all())

    def test_click_selects_cell_waffle(self):
        t = self.p2s.piep(self.df, 'cat', wxh=(200, 200), style=self.p2s.WAFFLEp)
        r = t.filterByRectangle((t.cx, t.cy, t.cx, t.cy))
        self.assertGreater(len(r), 0)

    def test_click_outside_selects_nothing(self):
        t = self.p2s.piep(self.df, 'cat', wxh=(200, 200))
        r = t.filterByRectangle((1, 1, 1, 1))   # corner, outside the pie
        self.assertEqual(len(r), 0)

    def test_filter_by_rectangle_selects_slices(self):
        t = self.p2s.piep(self.df, 'cat', wxh=(200, 200))
        r = t.filterByRectangle((0, 0, t.wxh[0], t.wxh[1]))   # whole canvas
        self.assertEqual(len(r), len(self.df))

    def test_filter_by_rectangle_remove(self):
        t = self.p2s.piep(self.df, 'cat', wxh=(200, 200))
        r = t.filterByRectangle((0, 0, t.wxh[0], t.wxh[1]), remove_records=True)
        self.assertEqual(len(r), 0)

    def test_filter_by_substring(self):
        t = self.p2s.piep(self.df, 'cat', wxh=(200, 200))
        r = t.filterBySubstring('A')
        self.assertTrue((r['cat'] == 'A').all())
        self.assertEqual(len(r), 5)

    def test_records_at_rejects_wrong_shape(self):
        t = self.p2s.piep(self.df, 'cat')
        with self.assertRaises(ValueError):
            t.recordsAt((t.cx, t.cy), shape=self.p2s.SELECT_HORIZONTALp)

    # ── "(other)" wedge is selectable ─────────────────────────────────────────

    def _other_df(self):
        # A dominates; C,D,E each 1/100 → 3.6°, folded into (other) at min_slice_deg=5.
        return pl.DataFrame({'cat': ['A'] * 97 + ['C', 'D', 'E']})

    def _pie_with_other(self):
        t = self.p2s.piep(self._other_df(), 'cat', wxh=(200, 200), min_slice_deg=5.0)
        assert '(other)' in [s['bin'] for s in t._slices_]
        return t

    def test_click_other_wedge_returns_folded_rows(self):
        t = self._pie_with_other()
        import math
        s = next(s for s in t._slices_ if s['bin'] == '(other)')
        a = math.radians((s['a0'] + s['a1']) / 2.0)
        xy = (t.cx + t.r * 0.5 * math.cos(a), t.cy + t.r * 0.5 * math.sin(a))
        r = t.filterByRectangle((xy[0], xy[1], xy[0], xy[1]))
        self.assertEqual(sorted(set(r['cat'].to_list())), ['C', 'D', 'E'])
        self.assertEqual(len(r), 3)

    def test_records_at_other_wedge(self):
        t = self._pie_with_other()
        import math
        s = next(s for s in t._slices_ if s['bin'] == '(other)')
        a = math.radians((s['a0'] + s['a1']) / 2.0)
        xy = (t.cx + t.r * 0.5 * math.cos(a), t.cy + t.r * 0.5 * math.sin(a))
        r = t.recordsAt(xy)
        self.assertEqual(sorted(set(r['cat'].to_list())), ['C', 'D', 'E'])

    def test_other_wedge_waffle_selectable(self):
        df = pl.DataFrame({'cat': ['A'] * 97 + ['C', 'D', 'E']})
        t = self.p2s.piep(df, 'cat', wxh=(200, 200), style=self.p2s.WAFFLEp, min_slice_deg=5.0)
        # (other) occupies a few cells; some grid click must land in those cells and
        # return the folded members C/D/E (and never A, which owns its own cells).
        hits = set()
        for gx in range(0, 200, 3):
            for gy in range(0, 200, 3):
                r = t.filterByRectangle((gx, gy, gx, gy))
                cats = set(r['cat'].to_list())
                if cats and cats != {'A'}:
                    hits |= (cats - {'A'})
        self.assertEqual(hits, {'C', 'D', 'E'})

    def test_filter_by_substring_finds_folded_member(self):
        '''A category folded into (other) is still findable by its own name.'''
        t = self._pie_with_other()
        r = t.filterBySubstring('D')
        self.assertEqual(sorted(set(r['cat'].to_list())), ['D'])

    def test_filter_by_substring_other_label_selects_all_members(self):
        t = self._pie_with_other()
        r = t.filterBySubstring('other')
        self.assertEqual(sorted(set(r['cat'].to_list())), ['C', 'D', 'E'])


if __name__ == '__main__':
    unittest.main()
