import unittest
import polars as pl
from polars2svg import Polars2SVG
from svg_test_utils import assert_valid_svg


def _make_df():
    return pl.DataFrame({
        'fm':   ['a', 'b', 'c', 'a', 'd', 'b'],
        'to':   ['b', 'a', 'a', 'c', 'a', 'c'],
        'time': [1,   1,   1,   2,   2,   3  ],
        'w':    [3,   1,   2,   4,   1,   2  ],
    })

def _rels():
    return [('fm', 'to')]


class TestSpreadLinesPDrawContextIsTimestampRow(unittest.TestCase):
    '''draw_context (not draw_labels) now controls the timestamp row along the
    bottom of each bin -- rewired as part of item 4 of 20260714_open_todos.md.'''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def test_draw_context_true_shows_timestamp_text(self):
        sp = self.p2s.spreadlinesp(_make_df(), _rels(), ego='a', time='time', draw_context=True)
        assert_valid_svg(self, sp.svg)
        self.assertIn('<text', sp.svg)

    def test_draw_context_false_hides_timestamp_text(self):
        sp = self.p2s.spreadlinesp(_make_df(), _rels(), ego='a', time='time', draw_context=False)
        assert_valid_svg(self, sp.svg)
        self.assertNotIn('<text', sp.svg)

    def test_draw_context_defaults_true(self):
        sp = self.p2s.spreadlinesp(_make_df(), _rels(), ego='a', time='time')
        self.assertTrue(sp.draw_context)
        self.assertIn('<text', sp.svg)


class TestSpreadLinesPDrawLabelsNotImplemented(unittest.TestCase):
    '''draw_labels defaults False and is accepted for API consistency with the rest
    of the framework, but the packed-circle layout has no per-node label placement,
    so setting it True raises NotImplementedError rather than being silently inert.'''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def test_draw_labels_defaults_false(self):
        sp = self.p2s.spreadlinesp(_make_df(), _rels(), ego='a', time='time')
        self.assertFalse(sp.draw_labels)

    def test_draw_labels_true_raises_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.p2s.spreadlinesp(_make_df(), _rels(), ego='a', time='time', draw_labels=True)

    def test_draw_labels_false_explicit_still_renders(self):
        sp = self.p2s.spreadlinesp(_make_df(), _rels(), ego='a', time='time', draw_labels=False)
        assert_valid_svg(self, sp.svg)


class TestSpreadLinesPDrawBorder(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def test_draw_border_defaults_true(self):
        sp = self.p2s.spreadlinesp(_make_df(), _rels(), ego='a', time='time')
        self.assertTrue(sp.draw_border)

    def test_draw_border_false(self):
        sp = self.p2s.spreadlinesp(_make_df(), _rels(), ego='a', time='time', draw_border=False)
        assert_valid_svg(self, sp.svg)


if __name__ == '__main__':
    unittest.main()
