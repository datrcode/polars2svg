import unittest
import polars as pl

from polars2svg import Polars2SVG
from polars2svg.p2s_palettes import PALETTES, resolvePalette, paletteNames
from polars2svg.p2s_colors_mixin import isHexColor


#
# The palette tables, the API that selects one, and -- the part that earns its keep --
# an end-to-end check that no component still paints a light literal under a dark
# palette.  colorTyped() is a bare dict subscript with ~111 call sites, so the failure
# mode this file is guarding is not a crash but a render that silently stays light in
# one corner.
#


class TestPaletteTables(unittest.TestCase):
    '''The static data. Checked once here rather than on every construction.'''

    def test_every_palette_defines_the_same_slots(self):
        # Parity is load-bearing: colorTyped() raises KeyError at RENDER time, deep in
        # whichever component asks first, so a slot missing from one palette is a crash
        # in a code path the author of the change was probably not looking at.
        _reference_ = set(PALETTES['light']['color_type_lu'])
        for _name_, _entry_ in PALETTES.items():
            with self.subTest(palette=_name_):
                self.assertEqual(set(_entry_['color_type_lu']), _reference_)

    def test_every_value_is_a_six_digit_hex(self):
        # Six digits specifically: TestColorTyped pins ^#[0-9a-fA-F]{6}$, and an
        # 8-digit #RRGGBBAA -- which isHexColor() accepts -- would pass that check's
        # detector but fail its regex.
        for _name_, _entry_ in PALETTES.items():
            for _slot_, _hex_ in _entry_['color_type_lu'].items():
                with self.subTest(palette=_name_, slot=_slot_):
                    self.assertTrue(isHexColor(_hex_), f'{_hex_!r} is not a hex color')
                    self.assertRegex(_hex_, r'^#[0-9a-fA-F]{6}$')

    def test_the_two_palettes_actually_differ(self):
        # Guards against a copy-paste that leaves 'dark' as a second light table --
        # every assertion below would still pass, vacuously.
        _light_, _dark_ = PALETTES['light']['color_type_lu'], PALETTES['dark']['color_type_lu']
        self.assertNotEqual(_light_[('background', 'default')], _dark_[('background', 'default')])
        _same_ = {_k_ for _k_ in _light_ if _light_[_k_] == _dark_[_k_]}
        # The multiset entries record hash-colorizer output, which is palette
        # independent, so they are identical on purpose. Nothing else should be.
        self.assertEqual(_same_, {('multiset', 'str'), ('multiset', 'int'), ('multiset', 'float')})

    def test_the_dark_background_is_actually_dark(self):
        self.assertLess(_luminance_(PALETTES['dark']['color_type_lu'][('background', 'default')]), 0.1)
        self.assertGreater(_luminance_(PALETTES['light']['color_type_lu'][('background', 'default')]), 0.9)

    def test_ink_contrasts_with_its_own_background(self):
        # The one property a palette cannot be wrong about. 4.5:1 is WCAG AA for text.
        for _name_, _entry_ in PALETTES.items():
            _lu_ = _entry_['color_type_lu']
            for _slot_ in (('label', 'defaultfg'), ('axis', 'label'), ('data', 'default')):
                with self.subTest(palette=_name_, slot=_slot_):
                    self.assertGreaterEqual(
                        _contrast_(_lu_[_slot_], _lu_[('background', 'default')]), 4.5)

    def test_the_grayscale_ramp_runs_away_from_the_background(self):
        # 0.0 sits near the canvas and 1.0 far from it. On light that means starting
        # pale and ending black; on dark it must INVERT, not merely lighten, or a
        # near-zero bin becomes the brightest mark in the plot.
        for _name_, _entry_ in PALETTES.items():
            _lo_, _hi_ = _entry_['grayscale_ramp']
            _bg_ = _luminance_(_entry_['color_type_lu'][('background', 'default')])
            with self.subTest(palette=_name_):
                self.assertLess(abs(_lo_ - _bg_), abs(_hi_ - _bg_),
                                'the ramp runs toward the background, not away from it')


class TestResolvePalette(unittest.TestCase):

    def test_returns_copies_not_the_module_tables(self):
        # color_type_lu is mutated in place by callers (test_linkp_basic did exactly
        # that for years). Handing out the module table would let one instance's edit
        # reach every instance built afterwards.
        _a_, _b_ = resolvePalette('light'), resolvePalette('light')
        self.assertIsNot(_a_['color_type_lu'], _b_['color_type_lu'])
        self.assertIsNot(_a_['color_type_lu'], PALETTES['light']['color_type_lu'])
        _a_['color_type_lu'][('background', 'default')] = '#123456'
        self.assertEqual(PALETTES['light']['color_type_lu'][('background', 'default')], '#ffffff')

    def test_unknown_palette_names_the_valid_ones(self):
        with self.assertRaises(ValueError) as _cm_:
            resolvePalette('solarized')
        self.assertIn('solarized', str(_cm_.exception))
        for _n_ in paletteNames(): self.assertIn(_n_, str(_cm_.exception))

    def test_a_non_string_palette_is_a_type_error(self):
        with self.assertRaises(TypeError): resolvePalette({'background': '#fff'})

    def test_overrides_layer_onto_the_named_base(self):
        _r_ = resolvePalette('dark', {('background', 'default'): '#000000'})
        self.assertEqual(_r_['color_type_lu'][('background', 'default')], '#000000')
        # everything else still comes from dark
        self.assertEqual(_r_['color_type_lu'][('axis', 'default')],
                         PALETTES['dark']['color_type_lu'][('axis', 'default')])

    def test_an_override_naming_no_slot_is_refused(self):
        # A typo'd slot would otherwise be silently inert -- the caller would believe
        # they had themed something.
        with self.assertRaises(ValueError) as _cm_:
            resolvePalette('light', {('axis', 'defalut'): '#ff0000'})
        self.assertIn('defalut', str(_cm_.exception))


class TestPaletteSelection(unittest.TestCase):

    def test_the_default_is_light(self):
        self.assertEqual(Polars2SVG().getPalette(), 'light')

    def test_the_constructor_selects_a_palette(self):
        self.assertEqual(Polars2SVG(palette='dark').getPalette(), 'dark')
        self.assertEqual(Polars2SVG(palette='dark').colorTyped('background', 'default'),
                         PALETTES['dark']['color_type_lu'][('background', 'default')])

    def test_set_palette_round_trips(self):
        _p2s_ = Polars2SVG()
        _light_bg_ = _p2s_.colorTyped('background', 'default')
        _p2s_.setPalette('dark')
        self.assertEqual(_p2s_.getPalette(), 'dark')
        _p2s_.setPalette('light')
        self.assertEqual(_p2s_.colorTyped('background', 'default'), _light_bg_)

    def test_set_palette_moves_the_grayscale_ramp_too(self):
        _p2s_ = Polars2SVG()
        self.assertEqual(_p2s_.grayscale_ramp, PALETTES['light']['grayscale_ramp'])
        _p2s_.setPalette('dark')
        self.assertEqual(_p2s_.grayscale_ramp, PALETTES['dark']['grayscale_ramp'])

    def test_set_palette_rejects_a_bad_override_value(self):
        with self.assertRaises(ValueError):
            Polars2SVG().setPalette('dark', {('axis', 'default'): 'not-a-color'})

    def test_instances_do_not_share_palette_state(self):
        # The contract the whole per-instance design rests on (see test_instance_init).
        _light_, _dark_ = Polars2SVG(), Polars2SVG(palette='dark')
        self.assertIsNot(_light_.color_type_lu, _dark_.color_type_lu)
        self.assertNotEqual(_light_.colorTyped('background', 'default'),
                            _dark_.colorTyped('background', 'default'))
        _dark_.setPalette('light')
        self.assertEqual(Polars2SVG(palette='dark').colorTyped('background', 'default'),
                         PALETTES['dark']['color_type_lu'][('background', 'default')])

    def test_color_overrides_survive_a_palette_switch(self):
        # An override is an explicit instruction about one data value; the palette is a
        # default for chart furniture. The explicit instruction outranks it.
        _p2s_ = Polars2SVG()
        _p2s_.setColorOverrides({'a': '#123456'})
        _p2s_.setPalette('dark')
        self.assertEqual(_p2s_.color('a'), '#123456')


class TestNoLightLiteralsSurviveADarkRender(unittest.TestCase):
    '''The straggler check.

    Every component is rendered under the dark palette and searched for colors that
    belong only to the light table. A hit means some call site still holds a hardcoded
    literal instead of going through colorTyped() -- which is a render that stays
    partly light, not a crash, and so would otherwise ship unnoticed.
    '''

    def _renderers(self, p2s):
        _df_xy_  = pl.DataFrame({'x': [1.37, 2.91, 3.14, 4.6, 5.5, 6.28],
                                 'y': [3.33, 1.1, 4.77, 1.9, 5.2, 9.81]})
        _df_cat_ = pl.DataFrame({'cat': ['a', 'b', 'a', 'c', 'b', 'a', 'c', 'b'],
                                 'val': [1.5, 2.7, 3.9, 4.1, 5.3, 6.6, 7.2, 8.8]})
        _df_ts_  = pl.DataFrame({'ts': ['2021-01-03', '2021-02-11', '2021-02-27',
                                        '2021-05-19', '2021-08-01', '2021-11-30']}
                                ).with_columns(pl.col('ts').str.to_datetime())
        _df_g_   = pl.DataFrame({'fm': ['a', 'b', 'c', 'a', 'd', 'b', 'c', 'a'],
                                 'to': ['b', 'a', 'a', 'c', 'a', 'c', 'b', 'd']})
        _df_sl_  = pl.DataFrame({'fm':   ['a', 'b', 'c', 'a', 'd', 'b', 'c', 'a'],
                                 'to':   ['b', 'a', 'a', 'c', 'a', 'c', 'b', 'd'],
                                 'time': [1, 1, 1, 2, 2, 2, 3, 3]})
        # 'b' and 'c' collapse onto one pixel, which is the only way to reach the
        # cloud <defs> -- the specific literal DP4 was about.
        _collapsed_ = {'a': [0.0, 0.0], 'b': [1.0, 0.0], 'c': [1.0, 0.0], 'd': [0.5, 1.0]}
        return {
            'xyp':          lambda: p2s.xyp(_df_xy_, 'x', 'y', wxh=(200, 200)),
            'histop':       lambda: p2s.histop(_df_cat_, 'cat', count='val', wxh=(200, 200)),
            'piep':         lambda: p2s.piep(_df_cat_, 'cat', wxh=(200, 200)),
            'timep':        lambda: p2s.timep(_df_ts_, ('ts', p2s.LT_Y_mp), wxh=(256, 128)),
            'linkp':        lambda: p2s.linkp(_df_g_, [('fm', 'to')], wxh=(300, 300)),
            'linkp_cloud':  lambda: p2s.linkp(_df_g_, [('fm', 'to')], pos=_collapsed_, wxh=(300, 300)),
            'chordp':       lambda: p2s.chordp(_df_g_, [('fm', 'to')], wxh=(300, 300)),
            'spreadlinesp': lambda: p2s.spreadlinesp(_df_sl_, [('fm', 'to')], ego='a',
                                                     time='time', wxh=(500, 260)),
            'smallp':       lambda: p2s.smallp(_df_cat_, 'cat',
                                               p2s.xyp(_df_cat_, 'val', 'val'), wxh=(300, 300)),
            # tile() composes finished SVG rather than drawing marks, and its bg_color
            # falls back to the palette when unset -- so it is the one component whose
            # canvas can go light while every mark on it is correct.
            'tile':         lambda: p2s.tile([p2s.xyp(_df_xy_, 'x', 'y', wxh=(100, 100)).svg,
                                              p2s.xyp(_df_xy_, 'x', 'y', wxh=(100, 100)).svg],
                                             per_row=2),
        }

    def test_no_light_only_color_appears_in_a_dark_render(self):
        _light_ = PALETTES['light']['color_type_lu']
        _dark_  = PALETTES['dark']['color_type_lu']
        # Only colors the two tables disagree on. A value both palettes share (the
        # multiset sentinels) proves nothing either way.
        _light_only_ = {_v_.lower(): _k_ for _k_, _v_ in _light_.items()
                        if _v_.lower() != _dark_[_k_].lower()}
        # A data color can legitimately land on a hex that happens to be a light slot
        # value, and that is not a stale literal.  The dark grayscale ramp ends at
        # 1.0 -> #ffffff, which collides with light's background; the spectrum is
        # shared by both palettes.  Subtract what dark can rightfully emit.
        for _legit_ in _darkEmittableColors_(): _light_only_.pop(_legit_, None)
        _p2s_ = Polars2SVG(palette='dark')
        for _name_, _fn_ in self._renderers(_p2s_).items():
            _svg_ = _fn_().svg.lower()
            _found_ = sorted({_hex_ for _hex_ in _light_only_ if _hex_ in _svg_})
            with self.subTest(component=_name_):
                self.assertEqual(_found_, [], f'{_name_} still paints light-palette '
                                              f'color(s) under the dark palette: '
                                              f'{[(h, _light_only_[h]) for h in _found_]}')

    def test_the_dark_background_reaches_every_component(self):
        # The complement of the check above: proving light is absent is only half of
        # it, since a component that emitted no background at all would also pass.
        _p2s_  = Polars2SVG(palette='dark')
        _dark_bg_ = _p2s_.colorTyped('background', 'default').lower()
        for _name_, _fn_ in self._renderers(_p2s_).items():
            with self.subTest(component=_name_):
                self.assertIn(_dark_bg_, _fn_().svg.lower())

    def test_the_light_render_is_unchanged_by_the_palette_machinery(self):
        # The regression that matters most: 72 goldens assert this too, but this says
        # it in one place and names the palette as the suspect when it breaks.
        _p2s_ = Polars2SVG(palette='light')
        _dark_only_ = {_v_.lower() for _k_, _v_ in PALETTES['dark']['color_type_lu'].items()
                       if _v_.lower() != PALETTES['light']['color_type_lu'][_k_].lower()}
        for _name_, _fn_ in self._renderers(_p2s_).items():
            _svg_ = _fn_().svg.lower()
            with self.subTest(component=_name_):
                self.assertEqual(sorted({_h_ for _h_ in _dark_only_ if _h_ in _svg_}), [])


#
# WCAG relative luminance / contrast ratio.  Local to this file: the framework has no
# luminance helper, and these exist to check the palette tables rather than to be used
# by the renderer.
#
def _luminance_(hex_color: str) -> float:
    _h_ = hex_color.lstrip('#')
    _c_ = [int(_h_[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
    _f_ = [(_v_ / 12.92 if _v_ <= 0.03928 else ((_v_ + 0.055) / 1.055) ** 2.4) for _v_ in _c_]
    return 0.2126 * _f_[0] + 0.7152 * _f_[1] + 0.0722 * _f_[2]


def _contrast_(a: str, b: str) -> float:
    _la_, _lb_ = sorted([_luminance_(a), _luminance_(b)], reverse=True)
    return (_la_ + 0.05) / (_lb_ + 0.05)


#
# _darkEmittableColors_() - hexes a correct dark render may contain that are NOT chrome
# stragglers: the grayscale ramp's two endpoints (dark runs up to #ffffff, which is
# also light's background) and the shared Spectral spectrum.  Listing them keeps the
# straggler scan strict about everything else rather than blanket-excusing a color.
#
def _darkEmittableColors_() -> set:
    _lo_, _hi_ = PALETTES['dark']['grayscale_ramp']
    _ramp_ = {'#{0:02x}{0:02x}{0:02x}'.format(int(round(_v_ * 255))) for _v_ in (_lo_, _hi_)}
    return _ramp_ | {_c_.lower() for _c_ in PALETTES['dark']['spectrum_palette']}


if __name__ == '__main__':
    unittest.main()
