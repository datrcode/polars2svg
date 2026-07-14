"""Unified hex-color detection across components.

Previously, xyp keyed hex-color detection
on the ``HexColorString`` metaclass (accepted only ``#RGB`` / ``#RRGGBB``), while
linkp/chordp/spreadlinesp used ad-hoc ``str.startswith('#')`` (accepted anything)
and the background-shape code required an exact ``len == 7``. So ``'#ff000080'``
(an ``#RRGGBBAA`` alpha color) was a *field name* to xyp but a color to linkp.

These tests pin the single canonical detector (``Polars2SVG.isHexColor`` /
``isinstance(x, HexColorString)``) and prove every component now agrees:
  * ``#RGB``, ``#RRGGBB``, ``#RRGGBBAA`` are colors (the three forms the renderer
    ``hexToRGBA`` can actually paint);
  * ``#RGBA``, a bare ``#``, wrong lengths and non-hex chars are NOT colors;
  * an ``#RRGGBBAA`` alpha color renders literally into every component's SVG;
  * a ``#``-prefixed non-hex string is no longer silently accepted as a broken
    fixed color by linkp/chordp (it raises instead of rendering garbage).
"""
import re
import unittest
import polars as pl
from shapely.geometry import Polygon
from polars2svg import Polars2SVG
from polars2svg.p2s_displaylist import hexToRGBA


# ---- inputs shared across components --------------------------------------

def _xy_df():
    return pl.DataFrame({'x': [1.0, 2.0, 3.0], 'y': [1.0, 2.0, 3.0], 'cat': ['a', 'b', 'c']})

def _graph_df():
    return pl.DataFrame({'fm': ['a', 'b', 'c'], 'to': ['b', 'c', 'a']})

_POS_ = {'a': (10.0, 10.0), 'b': (50.0, 50.0), 'c': (90.0, 10.0)}

def _spread_df():
    return pl.DataFrame({
        'fm':   ['a', 'b', 'c', 'a', 'd', 'b'],
        'to':   ['b', 'a', 'a', 'c', 'a', 'c'],
        'time': [1,   1,   1,   2,   2,   3  ],
        'w':    [3,   1,   2,   4,   1,   2  ],
    })


# The canonical truth table. Second element = should the detector call it a color?
_TRUTH = [
    ('#abc',        True),    # #RGB
    ('#ABC',        True),    # #RGB uppercase
    ('#aabbcc',     True),    # #RRGGBB
    ('#AABBCC',     True),    # #RRGGBB uppercase
    ('#aabbccdd',   True),    # #RRGGBBAA (alpha) — the previously-divergent form
    ('#FF000080',   True),    # #RRGGBBAA uppercase
    ('#RGB',        False),   # not hex digits
    ('#abcd',       False),   # 4-digit #RGBA is NOT supported (renderer can't parse it)
    ('#ab',         False),   # too short
    ('#aabbc',      False),   # 5 digits
    ('#aabbccd',    False),   # 7 digits
    ('#aabbccdde',  False),   # 9 digits
    ('#',           False),   # bare hash
    ('#gggggg',     False),   # non-hex chars
    ('red',         False),   # named color (not a hex literal)
    ('category',    False),   # field name
    ('',            False),   # empty
]


class TestCanonicalDetector(unittest.TestCase):
    def setUp(self):
        self.p2s = Polars2SVG()

    def test_truth_table_isHexColor(self):
        for _s_, _expected_ in _TRUTH:
            with self.subTest(s=_s_):
                self.assertEqual(self.p2s.isHexColor(_s_), _expected_)

    def test_isinstance_agrees_with_isHexColor(self):
        # isinstance(x, HexColorString) must delegate to the same rule as isHexColor().
        for _s_, _expected_ in _TRUTH:
            with self.subTest(s=_s_):
                self.assertEqual(isinstance(_s_, self.p2s.HexColorString), _expected_)

    def test_non_string_never_matches(self):
        for _obj_ in [None, 123, 4.5, ('a', 'b'), ['#aabbcc'], {'#aabbcc'}, object()]:
            with self.subTest(obj=_obj_):
                self.assertFalse(isinstance(_obj_, self.p2s.HexColorString))
                self.assertFalse(self.p2s.isHexColor(_obj_))

    def test_detector_matches_renderer_capability(self):
        # The detector is a *hex* detector aligned with the render path: a '#'-prefixed
        # string the detector rejects must also be unpaintable by hexToRGBA (which falls
        # back to opaque gray). Named colors like 'red' are paintable but are not hex
        # literals, so they are excluded here.
        for _s_ in [s for s, _ in _TRUTH if isinstance(s, str) and s.startswith('#')]:
            with self.subTest(s=_s_):
                _r_, _g_, _b_, _a_ = hexToRGBA(_s_)
                if not self.p2s.isHexColor(_s_):
                    self.assertEqual((_r_, _g_, _b_), (0.5, 0.5, 0.5),
                                     msg=f'{_s_!r} rejected by detector but hexToRGBA produced a color')


class TestCrossComponentConsistency(unittest.TestCase):
    """The same color string must be treated the same way by every component."""

    def setUp(self):
        self.p2s = Polars2SVG()

    def _xyp_svg(self, color):
        return self.p2s.xyp(_xy_df(), 'x', 'y', color=color, dot_size=10.0, wxh=(96, 96)).svg

    def _linkp_svg(self, color):
        return self.p2s.linkp(_graph_df(), relationships=[('fm', 'to')], pos=_POS_,
                              node_color=color, wxh=(128, 128)).svg

    def _chordp_svg(self, color):
        return self.p2s.chordp(df=_graph_df(), relationships=[('fm', 'to')],
                               node_color=color, wxh=(128, 128)).svg

    def _spreadlinesp_svg(self, color):
        return self.p2s.spreadlinesp(_spread_df(), [('fm', 'to')], ego='a', time='time',
                                     node_color=color, wxh=(256, 256)).svg

    def test_alpha_hex_renders_in_every_component(self):
        # #RRGGBBAA was previously a field name to xyp and a color to linkp. Now it is
        # a color everywhere and lands verbatim in the SVG.
        _color_ = '#11223344'
        for _name_, _fn_ in [('xyp', self._xyp_svg), ('linkp', self._linkp_svg),
                             ('chordp', self._chordp_svg), ('spreadlinesp', self._spreadlinesp_svg)]:
            with self.subTest(component=_name_):
                self.assertIn(_color_, _fn_(_color_))

    def test_short_hex_renders_in_every_component(self):
        _color_ = '#abc'
        for _name_, _fn_ in [('xyp', self._xyp_svg), ('linkp', self._linkp_svg),
                             ('chordp', self._chordp_svg), ('spreadlinesp', self._spreadlinesp_svg)]:
            with self.subTest(component=_name_):
                self.assertIn(_color_, _fn_(_color_))

    def test_full_hex_renders_in_every_component(self):
        _color_ = '#ff8800'
        for _name_, _fn_ in [('xyp', self._xyp_svg), ('linkp', self._linkp_svg),
                             ('chordp', self._chordp_svg), ('spreadlinesp', self._spreadlinesp_svg)]:
            with self.subTest(component=_name_):
                self.assertIn(_color_, _fn_(_color_))


class TestStricterValidation(unittest.TestCase):
    """A '#'-prefixed non-hex string must not be silently accepted as a color."""

    def setUp(self):
        self.p2s = Polars2SVG()

    def test_linkp_rejects_malformed_hex(self):
        # Previously '#notacolor' passed startswith('#') and was rendered as a broken
        # fixed color; now it is neither a valid hex nor a column, so validation raises.
        with self.assertRaises(ValueError):
            self.p2s.linkp(_graph_df(), relationships=[('fm', 'to')], pos=_POS_,
                          node_color='#notacolor', wxh=(128, 128)).svg

    def test_chordp_rejects_malformed_hex(self):
        with self.assertRaises(ValueError):
            self.p2s.chordp(df=_graph_df(), relationships=[('fm', 'to')],
                            node_color='#notacolor', wxh=(128, 128)).svg

    def test_valid_hex_still_accepted(self):
        # sanity: the stricter path does not reject genuine colors
        self.assertIn('#aabbcc', self.p2s.linkp(_graph_df(), relationships=[('fm', 'to')],
                                                pos=_POS_, node_color='#aabbcc', wxh=(128, 128)).svg)


class TestBackgroundShapeHex(unittest.TestCase):
    """Background-shape fill previously required an exact len == 7 (only #RRGGBB).
    It now uses the unified detector, so #RGB and #RRGGBBAA work too."""

    def setUp(self):
        self.p2s = Polars2SVG()

    def _xyp_bg_svg(self, fill):
        _df_ = pl.DataFrame({'x': [1, 2, 3, 4, 5], 'y': [2, 4, 1, 3, 5]})
        return self.p2s.xyp(df=_df_, x='x', y='y',
                            background={'box': Polygon([(1, 1), (4, 1), (4, 4), (1, 4)])},
                            background_fill=fill, background_opacity=0.5).svg

    def test_background_accepts_alpha_and_short_hex(self):
        # #123 (len 4) and #12345678 (len 9) were both rejected by the old len==7 gate
        # and silently fell back to the default axis-inner color; now they render.
        for _fill_ in ['#123', '#12345678']:
            with self.subTest(fill=_fill_):
                self.assertIn(f'fill="{_fill_}"', self._xyp_bg_svg(_fill_))

    def test_background_full_hex_still_works(self):
        self.assertIn('fill="#ff0000"', self._xyp_bg_svg('#ff0000'))


if __name__ == '__main__':
    unittest.main()
