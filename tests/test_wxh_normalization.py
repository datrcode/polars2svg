"""Uniform wxh (canvas-size) validation / normalization.

Historically wxh handling was inconsistent across components:

  - histop / piep / timep required a *tuple of exactly two ints* -- a list
    [128, 256] or floats (128.0, 256.0) raised.
  - chordp / linkp / spreadlinesp / xyp validated nothing at all -- a bad wxh
    only surfaced as a cryptic unpack error deep inside a render method.
  - smallp had its own None-tolerant check (one dimension may be None for
    auto-sizing).

Now every component funnels wxh through the single shared helper
Polars2SVG.normalizeWxh() in __parseInput__: it accepts any 2-element sequence
of numbers (tuple OR list), coerces each element to int, and returns a
canonical (w, h) tuple. Bad input raises a clear ValueError naming the
component. smallp opts into allow_none=True so exactly one dimension may stay
None.
"""
import datetime
import unittest

import polars as pl
from polars2svg import Polars2SVG


# ─────────────────────────────────────────────────────────────────────────────
# Per-component builders: (callable(wxh) -> component, reads self.wxh back)
# Every component is given valid data so the render path is exercised too.
# ─────────────────────────────────────────────────────────────────────────────

_BAR_DF = pl.DataFrame({
    'cat':  ['a', 'b', 'a', 'c', 'b', 'a'],
    'v':    [1, 2, 3, 4, 5, 6],
    'when': [datetime.date(2026, 1, d) for d in range(1, 7)],
})

_GRAPH_DF = pl.DataFrame({
    'fm':   ['a', 'b', 'c', 'a', 'd', 'b', 'e', 'c', 'd', 'e'],
    'to':   ['b', 'c', 'a', 'c', 'a', 'a', 'b', 'd', 'e', 'a'],
    'time': [1,   1,   1,   2,   2,   3,   3,   1,   2,   3  ],
    'w':    [30,  1,   2,   40,  1,   2,   50,  7,   9,   11 ],
})
_RELS = [('fm', 'to')]
_POS  = {'a': (0, 0), 'b': (1, 0), 'c': (1, 1), 'd': (0, 1), 'e': (0.5, 0.5)}


def _builders(p2s):
    """name -> callable(wxh) that constructs a fully-rendered component."""
    return {
        'histop':       lambda wxh: p2s.histop(_BAR_DF, 'cat', wxh=wxh),
        'timep':        lambda wxh: p2s.timep(_BAR_DF, time='when', count='v', wxh=wxh),
        'piep':         lambda wxh: p2s.piep(_BAR_DF, 'cat', wxh=wxh),
        'xyp':          lambda wxh: p2s.xyp(_BAR_DF, x='v', y='v', wxh=wxh),
        'linkp':        lambda wxh: p2s.linkp(_GRAPH_DF, relationships=_RELS, pos=_POS, wxh=wxh),
        'chordp':       lambda wxh: p2s.chordp(_GRAPH_DF, relationships=_RELS, wxh=wxh),
        'spreadlinesp': lambda wxh: p2s.spreadlinesp(_GRAPH_DF, _RELS, ego='a', time='time', wxh=wxh),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests of the shared helper
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalizeWxhHelper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p2s = Polars2SVG()

    def test_tuple_of_ints_passthrough(self):
        self.assertEqual(self.p2s.normalizeWxh((128, 256), 'Test'), (128, 256))

    def test_list_is_accepted_and_returns_tuple(self):
        out = self.p2s.normalizeWxh([128, 256], 'Test')
        self.assertEqual(out, (128, 256))
        self.assertIsInstance(out, tuple)

    def test_floats_are_coerced_to_int(self):
        self.assertEqual(self.p2s.normalizeWxh((128.9, 256.1), 'Test'), (128, 256))

    def test_mixed_number_types(self):
        self.assertEqual(self.p2s.normalizeWxh([128, 256.5], 'Test'), (128, 256))

    def test_wrong_length_raises(self):
        for bad in [(128,), (1, 2, 3), (), [1, 2, 3]]:
            with self.assertRaises(ValueError, msg=repr(bad)):
                self.p2s.normalizeWxh(bad, 'Test')

    def test_non_sequence_raises(self):
        for bad in [128, None, 3.5, {'w': 1, 'h': 2}]:
            with self.assertRaises(ValueError, msg=repr(bad)):
                self.p2s.normalizeWxh(bad, 'Test')

    def test_string_is_rejected_even_though_len_2(self):
        # a 2-char string has len 2 but is not a size
        with self.assertRaises(ValueError):
            self.p2s.normalizeWxh('12', 'Test')

    def test_non_number_element_raises(self):
        for bad in [('a', 256), (128, 'b'), (128, None), (None, 256)]:
            with self.assertRaises(ValueError, msg=repr(bad)):
                self.p2s.normalizeWxh(bad, 'Test')

    def test_bool_element_rejected(self):
        # bool is an int subclass, but wxh=(True, 256) is a mistake, not 1px
        with self.assertRaises(ValueError):
            self.p2s.normalizeWxh((True, 256), 'Test')
        with self.assertRaises(ValueError):
            self.p2s.normalizeWxh((128, False), 'Test')

    def test_error_message_names_component_and_dimension(self):
        with self.assertRaises(ValueError) as ctx:
            self.p2s.normalizeWxh((128, 'oops'), 'Histop')
        msg = str(ctx.exception)
        self.assertIn('Histop', msg)
        self.assertIn('height', msg)

    # allow_none (smallp path) ------------------------------------------------

    def test_allow_none_one_none_ok(self):
        self.assertEqual(self.p2s.normalizeWxh((1280, None), 'Smallp', allow_none=True),
                         (1280, None))
        self.assertEqual(self.p2s.normalizeWxh((None, 400), 'Smallp', allow_none=True),
                         (None, 400))

    def test_allow_none_still_coerces_the_number(self):
        self.assertEqual(self.p2s.normalizeWxh([1280.7, None], 'Smallp', allow_none=True),
                         (1280, None))

    def test_allow_none_both_none_raises(self):
        with self.assertRaises(ValueError):
            self.p2s.normalizeWxh((None, None), 'Smallp', allow_none=True)

    def test_allow_none_both_ints_ok(self):
        self.assertEqual(self.p2s.normalizeWxh((400, 300), 'Smallp', allow_none=True),
                         (400, 300))


# ─────────────────────────────────────────────────────────────────────────────
# Every component: list / float wxh now works and is canonicalized
# ─────────────────────────────────────────────────────────────────────────────

class TestComponentsAcceptListAndFloat(unittest.TestCase):
    def setUp(self):
        self.p2s = Polars2SVG()

    def test_list_wxh_accepted_everywhere(self):
        for name, build in _builders(self.p2s).items():
            with self.subTest(component=name):
                obj = build([200, 180])
                self.assertEqual(obj.wxh, (200, 180))
                self.assertIsInstance(obj.wxh, tuple)
                self.assertNotIn('no data', obj._repr_svg_())

    def test_float_wxh_coerced_everywhere(self):
        for name, build in _builders(self.p2s).items():
            with self.subTest(component=name):
                obj = build((200.6, 180.4))
                self.assertEqual(obj.wxh, (200, 180))

    def test_list_and_tuple_wxh_produce_same_canvas(self):
        # coercion makes [200, 180] and (200, 180) equivalent; assert both the
        # normalized wxh and the emitted canvas dimensions match. (SVG bodies are
        # not compared: some components -- e.g. xyp supersampling -- carry
        # per-build jitter unrelated to wxh.)
        for name, build in _builders(self.p2s).items():
            with self.subTest(component=name):
                a, b = build([200, 180]), build((200, 180))
                self.assertEqual(a.wxh, b.wxh)
                self.assertIn('width="200"', a.svg)
                self.assertIn('height="180"', a.svg)
                self.assertIn('width="200"', b.svg)
                self.assertIn('height="180"', b.svg)


# ─────────────────────────────────────────────────────────────────────────────
# Every component: bad wxh raises a clear, component-named ValueError
# ─────────────────────────────────────────────────────────────────────────────

class TestComponentsRejectBadWxh(unittest.TestCase):
    def setUp(self):
        self.p2s = Polars2SVG()

    def test_wrong_length_raises_everywhere(self):
        for name, build in _builders(self.p2s).items():
            with self.subTest(component=name):
                with self.assertRaises(ValueError):
                    build((128, 256, 512))

    def test_non_number_element_raises_everywhere(self):
        for name, build in _builders(self.p2s).items():
            with self.subTest(component=name):
                with self.assertRaises(ValueError):
                    build(('wide', 256))

    def test_none_dimension_raises_for_non_smallp(self):
        # only smallp opts into allow_none
        for name, build in _builders(self.p2s).items():
            with self.subTest(component=name):
                with self.assertRaises(ValueError):
                    build((128, None))

    def test_bad_wxh_error_names_component(self):
        with self.assertRaises(ValueError) as ctx:
            self.p2s.linkp(_GRAPH_DF, relationships=_RELS, pos=_POS, wxh=(1, 2, 3))
        self.assertIn('LinkP', str(ctx.exception))


# ─────────────────────────────────────────────────────────────────────────────
# smallp: allow_none behavior end-to-end
# ─────────────────────────────────────────────────────────────────────────────

class TestSmallpWxh(unittest.TestCase):
    def setUp(self):
        self.p2s = Polars2SVG()

    def _df(self):
        return pl.DataFrame({'cat': ['a', 'b', 'a', 'c'], 'v': [1, 2, 3, 4]})

    def test_smallp_list_wxh_accepted(self):
        tmpl = self.p2s.histop(bin_by='cat')
        sp = self.p2s.smallp(self._df(), sm_template=tmpl, category_by='cat', wxh=[400, 300])
        self.assertEqual(sp.wxh, (400, 300))
        self.assertNotIn('no data', sp._repr_svg_())

    def test_smallp_none_height_autosizes(self):
        tmpl = self.p2s.histop(bin_by='cat')
        sp = self.p2s.smallp(self._df(), sm_template=tmpl, category_by='cat', wxh=(400, None))
        self.assertEqual(sp.wxh, (400, None))
        sp._repr_svg_()                       # resolves wxh_actual without raising
        self.assertIsInstance(sp.wxh_actual[0], int)
        self.assertIsInstance(sp.wxh_actual[1], int)

    def test_smallp_float_width_coerced(self):
        tmpl = self.p2s.histop(bin_by='cat')
        sp = self.p2s.smallp(self._df(), sm_template=tmpl, category_by='cat', wxh=(400.9, None))
        self.assertEqual(sp.wxh, (400, None))

    def test_smallp_both_none_raises(self):
        tmpl = self.p2s.histop(bin_by='cat')
        with self.assertRaises(ValueError):
            self.p2s.smallp(self._df(), sm_template=tmpl, category_by='cat', wxh=(None, None))

    def test_smallp_bad_wxh_raises(self):
        tmpl = self.p2s.histop(bin_by='cat')
        with self.assertRaises(ValueError):
            self.p2s.smallp(self._df(), sm_template=tmpl, category_by='cat', wxh=(400, 'x'))


# ─────────────────────────────────────────────────────────────────────────────
# Dataless templates: normalization runs in __parseInput__, so it applies even
# without a DataFrame (list/float default still canonicalized; bad wxh raises).
# ─────────────────────────────────────────────────────────────────────────────

class TestDatalessTemplateWxh(unittest.TestCase):
    def setUp(self):
        self.p2s = Polars2SVG()

    def test_dataless_list_wxh_normalized(self):
        tmpl = self.p2s.histop(bin_by='cat', wxh=[300, 200])
        self.assertEqual(tmpl.wxh, (300, 200))

    def test_dataless_bad_wxh_raises(self):
        with self.assertRaises(ValueError):
            self.p2s.histop(bin_by='cat', wxh=(1, 2, 3))

    def test_clone_inherits_normalized_wxh(self):
        tmpl  = self.p2s.histop(bin_by='cat', wxh=[300.5, 200.5])
        clone = self.p2s.histop(_BAR_DF, template=tmpl)
        self.assertEqual(clone.wxh, (300, 200))


if __name__ == '__main__':
    unittest.main()
