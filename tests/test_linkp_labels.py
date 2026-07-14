import unittest
import polars as pl
from polars2svg import Polars2SVG
from polars2svg.linkp import LinkP


def _make_df():
    return pl.DataFrame({
        'fm': ['a', 'b', 'c', 'a'],
        'to': ['b', 'c', 'a', 'c'],
    })

def _make_pos():
    return {'a': [0.0, 0.0], 'b': [1.0, 0.0], 'c': [0.5, 0.866]}

def _rels():
    return [('fm', 'to')]

def _lp(**kwargs):
    p2s = Polars2SVG()
    return p2s.linkp(_make_df(), relationships=_rels(), pos=_make_pos(),
                     draw_labels=True, **kwargs)

def _bare_instance():
    """Return a LinkP with just enough state for _wrap_label_ tests."""
    obj = LinkP.__new__(LinkP)
    obj.p2s = Polars2SVG()
    return obj


class TestWrapLabel(unittest.TestCase):
    """Unit tests for the _wrap_label_() helper."""

    def setUp(self):
        self.lp = _bare_instance()

    def _wrap(self, text, line_width=32, max_lines=4, ellipsis=True):
        return self.lp._wrap_label_(text, line_width, max_lines, ellipsis)

    def test_short_label_stays_single_line(self):
        lines = self._wrap('Alice')
        self.assertEqual(lines, ['Alice'])

    def test_wrap_at_word_boundary(self):
        # "hello world" fits on one 32-char line; "hello" + "world" should stay together
        lines = self._wrap('hello world', line_width=32)
        self.assertEqual(lines, ['hello world'])

    def test_wrap_splits_at_space(self):
        # Force a wrap: each word is short but combined they exceed line_width
        lines = self._wrap('one two three four five six', line_width=12)
        # No single line should exceed 12 chars
        for line in lines:
            self.assertLessEqual(len(line), 12, msg=f'Line too long: {line!r}')
        self.assertGreater(len(lines), 1)

    def test_hard_break_of_long_word(self):
        word = 'A' * 40
        lines = self._wrap(word, line_width=32)
        self.assertEqual(lines[0], 'A' * 32)
        self.assertEqual(lines[1], 'A' * 8)

    def test_truncation_to_max_lines(self):
        text = ' '.join(['word'] * 20)
        lines = self._wrap(text, line_width=10, max_lines=3)
        self.assertEqual(len(lines), 3)

    def test_ellipsis_appended_on_truncation(self):
        text = ' '.join(['word'] * 20)
        lines = self._wrap(text, line_width=10, max_lines=2, ellipsis=True)
        self.assertTrue(lines[-1].endswith('…'))

    def test_no_ellipsis_when_disabled(self):
        text = ' '.join(['word'] * 20)
        lines = self._wrap(text, line_width=10, max_lines=2, ellipsis=False)
        self.assertFalse(lines[-1].endswith('…'))

    def test_no_ellipsis_when_not_truncated(self):
        lines = self._wrap('short', line_width=32, max_lines=4, ellipsis=True)
        self.assertFalse(lines[-1].endswith('…'))

    def test_unlimited_lines(self):
        text = ' '.join(str(i) for i in range(50))
        lines = self._wrap(text, line_width=5, max_lines=-1)
        self.assertGreater(len(lines), 4)
        for line in lines:
            self.assertLessEqual(len(line), 5)

    def test_max_lines_one(self):
        lines = self._wrap('word1 word2 word3 word4', line_width=32, max_lines=1)
        self.assertEqual(len(lines), 1)

    def test_ellipsis_truncates_line_to_width(self):
        # Last line that is already at full width should still fit ellipsis within line_width
        long_word = 'B' * 32
        lines = self._wrap(long_word + ' extra', line_width=32, max_lines=1, ellipsis=True)
        self.assertEqual(len(lines), 1)
        self.assertLessEqual(len(lines[0]), 32)
        self.assertTrue(lines[0].endswith('…'))


class TestMultilineLabelSVG(unittest.TestCase):
    """End-to-end tests: multiline labels produce correct SVG structure."""

    def setUp(self):
        self.p2s = Polars2SVG()

    def test_short_labels_produce_no_tspan(self):
        lp = _lp(node_labels={'a': 'Alice', 'b': 'Bob', 'c': 'Carol'})
        self.assertIn('<text', lp.svg)
        self.assertNotIn('<tspan', lp.svg)

    def test_long_label_produces_tspan(self):
        long = 'This label is definitely longer than thirty-two characters and should wrap'
        lp = _lp(node_labels={'a': long}, label_line_width=32, label_max_lines=4)
        self.assertIn('<tspan', lp.svg)

    def test_tspan_count_matches_lines(self):
        # 'word ' * 8 at line_width=10 → multiple lines, max 3
        text = 'word ' * 8
        lp = _lp(node_labels={'a': text.strip()}, label_line_width=10, label_max_lines=3)
        import re
        tspan_count = len(re.findall(r'<tspan', lp.svg))
        self.assertLessEqual(tspan_count, 3)
        self.assertGreater(tspan_count, 0)

    def test_ellipsis_appears_in_svg(self):
        text = ' '.join(['longword'] * 10)
        lp = _lp(node_labels={'a': text}, label_line_width=20, label_max_lines=2, label_ellipsis=True)
        self.assertIn('…', lp.svg)

    def test_no_ellipsis_in_svg_when_disabled(self):
        text = ' '.join(['longword'] * 10)
        lp = _lp(node_labels={'a': text}, label_line_width=20, label_max_lines=2, label_ellipsis=False)
        self.assertNotIn('…', lp.svg)

    def test_max_lines_one_produces_no_tspan(self):
        text = ' '.join(['word'] * 10)
        lp = _lp(node_labels={'a': text}, label_line_width=10, label_max_lines=1)
        self.assertNotIn('<tspan', lp.svg)
        self.assertIn('<text', lp.svg)

    def test_unlimited_lines(self):
        text = ' '.join(['w'] * 20)
        lp = _lp(node_labels={'a': text}, label_line_width=4, label_max_lines=-1)
        import re
        tspan_count = len(re.findall(r'<tspan', lp.svg))
        self.assertGreater(tspan_count, 4)

    def test_xml_special_chars_escaped_in_multiline(self):
        text = 'Price < 100 & discount > 10 percent off for everyone here'
        lp = _lp(node_labels={'a': text}, label_line_width=20, label_max_lines=4)
        self.assertNotIn(' < ',   lp.svg)
        self.assertNotIn(' & ',   lp.svg)
        self.assertNotIn(' > ',   lp.svg)
        self.assertIn('&lt;',    lp.svg)
        self.assertIn('&amp;',   lp.svg)
        self.assertIn('&gt;',    lp.svg)

    def test_multiline_with_node_size_vary(self):
        # Exercises the __sz__ column path (vary uses a Polars column, not a literal)
        text = 'This is a fairly long label that will need wrapping across lines'
        lp = _lp(node_labels={'a': text}, label_line_width=20, label_max_lines=4,
                 node_size='vary')
        self.assertIn('<tspan', lp.svg)

    def test_tspan_carries_x_and_dy_attributes(self):
        text = 'word ' * 6
        lp = _lp(node_labels={'a': text.strip()}, label_line_width=10, label_max_lines=4)
        import re
        tspans = re.findall(r'<tspan[^>]*>', lp.svg)
        for ts in tspans:
            self.assertIn('x=',  ts, msg=f'Missing x= in {ts!r}')
            self.assertIn('dy=', ts, msg=f'Missing dy= in {ts!r}')

    def test_first_tspan_dy_is_zero(self):
        text = 'word ' * 6
        lp = _lp(node_labels={'a': text.strip()}, label_line_width=10, label_max_lines=4)
        import re
        first_tspan = re.search(r'<tspan[^>]*>', lp.svg)
        self.assertIsNotNone(first_tspan)
        self.assertIn('dy="0"', first_tspan.group())

    def test_label_only_still_works_with_multiline(self):
        long = 'This label is long enough to trigger wrapping across lines'
        lp = _lp(
            node_labels={'a': long, 'b': 'Bob', 'c': 'Carol'},
            label_only={'a'},
            label_line_width=20, label_max_lines=4,
        )
        self.assertIn('<tspan', lp.svg)
        self.assertNotIn('Bob',   lp.svg)
        self.assertNotIn('Carol', lp.svg)


if __name__ == '__main__':
    unittest.main()
