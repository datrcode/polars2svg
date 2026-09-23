#
# test_dark_palette.py - the palette channel, end to end in a browser
#
# The Python side is covered by tests/test_palettes.py and the static renders by
# tests/test_dark_palette_golden.py. Neither can see the interactive overlays: those are
# built in JS from model.palette via p2sInk(), so the only thing that proves the two ends
# agree is a real render in a real browser.
#
# What this is guarding is not a style preference. The overlays used to hardcode
# '#000000', which is 1.12:1 against the dark palette's #121212 -- the drag band and the
# status line did not look wrong, they were invisible, and a user on a dark palette had
# no feedback for which set operation a drag was about to perform.
#
import unittest

import pytest

from polars2svg.p2s_palettes import PALETTES

_DARK_ = PALETTES['dark']['color_type_lu']
_LIGHT_ = PALETTES['light']['color_type_lu']

pytestmark = pytest.mark.interaction


def test_the_drag_band_uses_the_dark_setop_colour(dark_page):
    """The band names the pending set operation, in the palette the figure was built
    with -- not the light literal the JS used to carry."""
    dark_page.mouse_down(20, 20)
    dark_page.mouse_move_to(200, 160)
    try:
        _stroke_ = dark_page.drag_rect()['stroke']
        assert _stroke_ == dark_page.setopColor('replace', 'dark')
        # ...and specifically NOT the light value, which is what a missing channel
        # would silently fall back to (p2sInk's fallback table is the light palette).
        assert _stroke_ != dark_page.setopColor('replace', 'light')
    finally:
        dark_page.mouse_up()


@pytest.mark.parametrize('shift,ctrl,op', [(True,  False, 'subtract'),
                                           (False, True,  'add'),
                                           (True,  True,  'intersect')])
def test_every_modifier_resolves_through_the_palette(dark_page, shift, ctrl, op):
    # Focus first: the band's colour comes from model.shiftkey/ctrlkey, which only
    # myOnKeyDown sets -- the same reason test_mouse_gestures hovers before holding.
    dark_page.hover(200, 150)
    with dark_page.holding(shift=shift, ctrl=ctrl):
        dark_page.mouse_down(20, 20)
        dark_page.mouse_move_to(200, 160)
        try:
            assert dark_page.drag_rect()['stroke'] == dark_page.setopColor(op, 'dark')
        finally:
            dark_page.mouse_up()


def test_the_status_line_is_not_black_on_a_dark_canvas(dark_page):
    """#infostr is the one overlay that is pure text, so an unthemed fill makes it
    vanish rather than merely dim."""
    _fill_ = dark_page.el('infostr').get_attribute('fill')
    assert _fill_ == _DARK_[('label', 'defaultfg')]
    assert _fill_ != _LIGHT_[('label', 'defaultfg')]


def test_the_light_page_is_unaffected(quad_page):
    """The same assertion against the default palette, so a regression in the channel
    cannot be mistaken for 'dark is broken' when in fact both ends moved."""
    quad_page.mouse_down(20, 20)
    quad_page.mouse_move_to(200, 160)
    try:
        assert quad_page.drag_rect()['stroke'] == quad_page.setopColor('replace', 'light')
    finally:
        quad_page.mouse_up()


if __name__ == '__main__':
    unittest.main()
