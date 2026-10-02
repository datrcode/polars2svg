"""
Tests for the 'h' keyboard help overlay's text: that its columns line up, and that every
space in it survives into the SVG.

User report (histopi, piepi): "the help menu is not fully aligned".  Two causes.  SVG drops
a <text>'s leading spaces, so every ' .. |' row rendered one character left of the
'a . |' rows -- linkpi's builder replaced spaces with no-break spaces and the five generic
views' did not.  And the sub-rows were padded by hand: 'in the panel ...' was one
character wider than the rows under it, and xyp's 'shift+e . |' / 'ctrl+e . |' widened
the key column.  One builder now serves every view, and the sub-rows are padded by
_helpSubRow_().

(That the help is no longer clipped to the view is a browser test:
tests/interaction/test_panel_overhang.py.)
"""
import html
import re
import unittest
import xml.etree.ElementTree as ET

try:
    import panel  # noqa: F401 -- the import is the availability probe
    PANEL_AVAILABLE = True
except ImportError:
    PANEL_AVAILABLE = False

if PANEL_AVAILABLE:
    from polars2svg.interactive_controller import CHORDPI, HISTOPI, LINKPI, PIEPI, TIMEPI, XYPI

#: A sub-row: ' .. | <label> ..... | text' -- a label (or none) dot-padded to a column.
_SUB_ROW_ = re.compile(r'^ \.\. \| (?:\S[^|]*? )?\.+ \| ')


@unittest.skipUnless(PANEL_AVAILABLE, 'panel not installed')
class TestKeyboardHelp(unittest.TestCase):

    def _views_(self):
        return [XYPI, HISTOPI, TIMEPI, CHORDPI, PIEPI, LINKPI]

    def _lines_(self, cls):
        return cls._keyboard_commands_.strip('\n').split('\n')

    def test_the_key_column_lines_up(self):
        for _cls_ in self._views_():
            with self.subTest(view=_cls_.__name__):
                for _ln_ in self._lines_(_cls_):
                    if '|' in _ln_ and not _ln_.startswith('in any picker menu'):
                        self.assertEqual(_ln_.index('|'), 4, _ln_)

    def test_the_sub_row_labels_line_up(self):
        for _cls_ in self._views_():
            with self.subTest(view=_cls_.__name__):
                _cols_ = {_m_.end() for _m_ in map(_SUB_ROW_.match, self._lines_(_cls_)) if _m_}
                self.assertGreater(len([_l_ for _l_ in self._lines_(_cls_) if _SUB_ROW_.match(_l_)]), 1)
                self.assertEqual(len(_cols_), 1, f'sub-row columns differ: {sorted(_cols_)}')

    # Every line reaches the SVG character for character, leading spaces included.
    def test_every_space_survives_into_the_svg(self):
        for _cls_ in self._views_():
            with self.subTest(view=_cls_.__name__):
                _svg_  = _cls_.param['kbd_help_svg'].default
                _root_ = ET.fromstring(f'<svg xmlns="http://www.w3.org/2000/svg">{_svg_}</svg>')
                _texts_ = [(_t_.text or '') for _t_ in _root_.iter('{http://www.w3.org/2000/svg}text')]
                self.assertTrue(all(' ' not in _t_ for _t_ in _texts_), 'a plain space would be collapsed')
                self.assertEqual([_t_.replace(' ', ' ') for _t_ in _texts_], self._lines_(_cls_))

    # linkpi's help says '&intersect'; it has to be escaped to be well-formed markup.
    def test_the_text_is_escaped(self):
        _svg_ = LINKPI.param['kbd_help_svg'].default
        self.assertIn(html.escape('&intersect'), _svg_)
        ET.fromstring(f'<svg xmlns="http://www.w3.org/2000/svg">{_svg_}</svg>')

    def test_the_box_fits_the_longest_line(self):
        for _cls_ in self._views_():
            with self.subTest(view=_cls_.__name__):
                _svg_ = _cls_.param['kbd_help_svg'].default
                _w_   = float(re.search(r'<rect x="0" y="0" width="([0-9.]+)"', _svg_).group(1))
                self.assertGreaterEqual(_w_, max(len(_l_) for _l_ in self._lines_(_cls_)) * 7)

    # The time keys are sub-rows now, not wider key cells; the text tests elsewhere read.
    def test_xypi_time_keys(self):
        _cmds_ = XYPI._keyboard_commands_
        self.assertIn(' .. | shift-e ....... | (time x-axis) expand timeframe backward', _cmds_)
        self.assertIn(' .. | ctrl-e ........ | (time x-axis) expand timeframe forward', _cmds_)


# stack control builds its help apart from the six views, and its 'ctrl+shift+c . |' key
# cell was wider than 'h . |' (PLANNING.md §5 C-stack-help-key-column).
@unittest.skipUnless(PANEL_AVAILABLE, 'panel not installed')
class TestStackControlHelp(unittest.TestCase):

    def _lines_(self):
        from polars2svg.stack_control import _STACK_KEYBOARD_COMMANDS_
        return _STACK_KEYBOARD_COMMANDS_.split('\n')

    def _svg_(self):
        from polars2svg.stack_control import STACKCONTROLI
        return STACKCONTROLI.param['kbd_help_svg'].default

    def test_the_key_column_lines_up(self):
        self.assertEqual([_ln_.index('|') for _ln_ in self._lines_()], [4, 4, 4])
        self.assertRegex(self._lines_()[2], _SUB_ROW_)
        self.assertTrue(self._lines_()[2].endswith('| current -> base'))

    def test_every_space_survives_into_the_svg(self):
        _root_  = ET.fromstring(f'<svg xmlns="http://www.w3.org/2000/svg">{self._svg_()}</svg>')
        _texts_ = [(_t_.text or '') for _t_ in _root_.iter('{http://www.w3.org/2000/svg}text')]
        self.assertTrue(all(' ' not in _t_ for _t_ in _texts_), 'a plain space would be collapsed')
        self.assertEqual([_t_.replace('\u00a0', ' ') for _t_ in _texts_], self._lines_())

    def test_the_box_fits_the_longest_line(self):
        _w_ = float(re.search(r'<rect x="0" y="0" width="([0-9.]+)"', self._svg_()).group(1))
        self.assertGreaterEqual(_w_, max(len(_l_) for _l_ in self._lines_()) * 7)


if __name__ == '__main__':
    unittest.main()
