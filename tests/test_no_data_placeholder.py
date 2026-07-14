"""Visible "no data" placeholder when no DataFrame is supplied.

A component built without a df (a legitimate pattern for template construction,
but also what happens when a plumbing mistake drops the df on the way in) used
to render a *silently blank* background rect -- an empty chart indistinguishable
from a real render that happened to draw nothing. Now every component paints a
centered "no data - no DataFrame supplied" message via
Polars2SVG.placeholderSVG(), so the mistake is visible instead of silent.

Key invariants locked here:
  * dataless build of every component contains the "no data" message text,
  * a real render (df supplied) overwrites the placeholder -- the message never
    leaks into a chart that actually has data,
  * dataless construction still *succeeds* (no raise, no warning) so template
    building is unaffected,
  * smallp -- which only assigned self.svg inside __renderSVG__ -- no longer
    raises AttributeError on repr when built dataless,
  * placeholderSVG() is well-formed even with None dimensions.
"""
import datetime
import logging
import unittest

import polars as pl
from polars2svg import Polars2SVG


# Every clone-template component + the kwargs needed to build it dataless.
_DATALESS_KWARGS = {
    'histop':       {},
    'timep':        {},
    'piep':         {},
    'xyp':          {'x': 'x', 'y': 'y'},
    'linkp':        {},
    'chordp':       {},
    'spreadlinesp': {},
}

_MESSAGE = 'no data'


class TestNoDataPlaceholder(unittest.TestCase):

    def setUp(self):
        self.p2s = Polars2SVG()
        self.p2s.reset_defaults()

    def tearDown(self):
        self.p2s.reset_defaults()

    # ------------------------------------------------------------------
    # placeholderSVG() helper units
    # ------------------------------------------------------------------

    def test_helper_contains_message_and_rect(self):
        svg = self.p2s.placeholderSVG(200, 100)
        self.assertIn(_MESSAGE, svg)
        self.assertIn('<rect', svg)
        self.assertIn('<text', svg)
        self.assertTrue(svg.startswith('<svg'))
        self.assertTrue(svg.rstrip().endswith('</svg>'))

    def test_helper_none_dims_fall_back(self):
        # auto-sizing may not have resolved yet; None must not crash / produce
        # a malformed width/height
        svg = self.p2s.placeholderSVG(None, None)
        self.assertIn('width="256"', svg)
        self.assertIn('height="256"', svg)
        self.assertIn(_MESSAGE, svg)

    def test_helper_empty_message_omits_text(self):
        svg = self.p2s.placeholderSVG(200, 100, message='')
        self.assertNotIn('<text', svg)
        self.assertIn('<rect', svg)

    def test_helper_notes_rendered(self):
        svg = self.p2s.placeholderSVG(200, 100, notes=['x: foo', 'y: bar'])
        self.assertIn('x: foo', svg)
        self.assertIn('y: bar', svg)

    def test_helper_no_control_chars_leak(self):
        # nothing non-printable should ever reach the XML
        svg = self.p2s.placeholderSVG(200, 100)
        self.assertFalse(any(ord(c) < 0x20 and c not in '\n\r\t' for c in svg))

    # ------------------------------------------------------------------
    # Dataless build shows the message (every component)
    # ------------------------------------------------------------------

    def test_dataless_shows_message(self):
        for name, kw in _DATALESS_KWARGS.items():
            with self.subTest(component=name):
                obj = getattr(self.p2s, name)(**kw)
                svg = obj._repr_svg_()
                self.assertIn(_MESSAGE, svg,
                              f'{name} dataless placeholder missing "no data" message')
                self.assertIn('<text', svg)

    def test_dataless_build_does_not_raise_or_warn(self):
        # template building is legitimate: no exception, no warning. (The logger's
        # OnceFilter dedupes by message, so a capturing handler is used rather than
        # assertLogs, whose sentinel would be filtered out across subtests.)
        captured = []

        class _Capture(logging.Handler):
            def emit(self, record):
                captured.append(record)

        handler = _Capture(level=logging.WARNING)
        logger = logging.getLogger('polars2svg_logger')
        logger.addHandler(handler)
        try:
            for name, kw in _DATALESS_KWARGS.items():
                with self.subTest(component=name):
                    before = len(captured)
                    getattr(self.p2s, name)(**kw)   # must not raise
                    self.assertEqual(captured[before:], [],
                                     f'{name} dataless build emitted a warning')
        finally:
            logger.removeHandler(handler)

    def test_xyp_placeholder_echoes_xy(self):
        svg = self.p2s.xyp(x='amount', y='when')._repr_svg_()
        self.assertIn(_MESSAGE, svg)
        self.assertIn('x: amount', svg)
        self.assertIn('y: when', svg)

    # ------------------------------------------------------------------
    # Real renders overwrite the placeholder -- no leak
    # ------------------------------------------------------------------

    def test_real_render_has_no_message(self):
        df = pl.DataFrame({
            'cat':  ['a', 'b', 'a', 'c', 'b', 'a'],
            'v':    [1, 2, 3, 4, 5, 6],
            'when': [datetime.date(2026, 1, d) for d in range(1, 7)],
        })
        cases = {
            'histop': self.p2s.histop(df, 'cat'),
            'piep':   self.p2s.piep(df, 'cat'),
            'timep':  self.p2s.timep(df, time='when', count='v'),
            'xyp':    self.p2s.xyp(df, x='v', y='v'),
        }
        for name, obj in cases.items():
            with self.subTest(component=name):
                self.assertNotIn(_MESSAGE, obj._repr_svg_(),
                                 f'{name} leaked the "no data" message into a real render')

    def test_clone_with_data_has_no_message(self):
        df = pl.DataFrame({'cat': ['a', 'b', 'a'], 'v': [1, 2, 3]})
        tmpl = self.p2s.histop(bin_by='cat')
        self.assertIn(_MESSAGE, tmpl._repr_svg_())          # template itself: no data
        clone = self.p2s.histop(df, template=tmpl)          # clone with data
        self.assertNotIn(_MESSAGE, clone._repr_svg_())

    # ------------------------------------------------------------------
    # smallp: no more AttributeError on dataless repr
    # ------------------------------------------------------------------

    def test_smallp_dataless_repr_does_not_raise(self):
        tmpl = self.p2s.histop(bin_by='cat')
        sp = self.p2s.smallp(sm_template=tmpl, category_by='cat', wxh=(400, 300))
        self.assertIsNone(sp.df)
        # previously raised AttributeError: 'Smallp' object has no attribute 'svg'
        svg = sp._repr_svg_()
        self.assertIn(_MESSAGE, svg)

    def test_smallp_real_render_has_no_message(self):
        df = pl.DataFrame({
            'cat': ['a', 'b', 'a', 'b'],
            'v':   [1, 2, 3, 4],
        })
        tmpl = self.p2s.histop(bin_by='v')
        sp = self.p2s.smallp(df, sm_template=tmpl, category_by='cat', wxh=(400, 300))
        self.assertNotIn(_MESSAGE, sp._repr_svg_())


if __name__ == '__main__':
    unittest.main()
