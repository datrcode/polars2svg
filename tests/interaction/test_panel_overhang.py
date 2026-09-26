"""The settings panel and its pickers overhang a view too small to hold them.

Both overlays are drawn inside the view's root <svg> and sized to their text, not to the
view.  At every component's DEFAULT size the panel is wider than histopi (128), piepi
(160) and linkpi (256), and xypi's render rows made it wider than xypi (256) too -- the
value column was cut off, and a picker opened from the panel (drawn at panel_w + 16)
landed wholly outside the view.  Every other fixture is 400x300, which is why nothing
caught it.

Two halves, and a lone view only shows the first: the root clipped the overhang, and in
a grid the NEXT view painted over it.  So these run on ``default_size_grid``, where each
panel has a neighbour to overhang (see the fixture for the layout).

The fix, in ``js/fragments/p2s_config_panel.js``: while an overlay is open and the view
has focus, the root's clip becomes the view plus the two overlay boxes and the view is
raised above its siblings.  Each half of that has a test below that fails without it.
"""
import unittest

import pytest


_KINDS_ = ['histopi', 'xypi', 'timepi', 'piepi', 'chordpi', 'linkpi']

#: The first row's picker, which is where Enter on a freshly opened panel lands.
_FIRST_ROW_PICKER_ = {'linkpi': 'link arrows:'}


def _open_panel(ip):
    ip.settle()
    ip.hover(10, 10)
    ip.press('a')
    ip.expect_panel_open()


@pytest.mark.parametrize('kind', _KINDS_)
def test_the_panel_and_its_picker_are_whole_at_the_default_size(default_size_grid, kind):
    """Five points of each overlay -- corners and centre -- are the overlay itself, not
    a clipped-away hole and not the neighbouring view."""
    _ip_ = default_size_grid[kind]
    _open_panel(_ip_)
    assert _ip_.overlay_is_on_top('configpanel') == [True] * 5
    _ip_.press('Enter')
    _ip_.expect_menu_open(_FIRST_ROW_PICKER_.get(kind, 'selection shape:'))
    assert _ip_.overlay_is_on_top('pickermenu') == [True] * 5


def test_a_view_that_loses_focus_stops_covering_its_neighbour(default_size_grid):
    """The overhang belongs to the view being worked in.  Moving onto the neighbour
    clips histopi's still-open panel back to histopi's own box (so the neighbour is not
    covered while it is the one in use), and coming back restores it."""
    _hp_, _xy_ = default_size_grid['histopi'], default_size_grid['xypi']
    _open_panel(_hp_)
    assert _hp_.overlay_is_on_top('configpanel') == [True] * 5

    _xy_.hover(200, 100)
    _hp_.expect_panel_open()                        # left open, not closed
    # The panel spans x 7.5..294.5 and histopi is 128 wide: the left-hand points are
    # still histopi's own, the right-hand points and the centre are xypi's now.
    assert _hp_.overlay_is_on_top('configpanel') == [True, False, False, True, False]

    _hp_.hover(10, 10)
    assert _hp_.overlay_is_on_top('configpanel') == [True] * 5


def test_the_overhang_does_not_uncover_what_the_view_parks_off_screen(default_size_grid):
    """Why the clip is the view plus the overlays rather than a bare overflow:visible.

    The idle drag band is parked at (-10,-10) and the hidden keyboard help at
    translate(-1000).  chordpi's parked band lands over xypi, one row up; with the panel
    open it has to stay clipped rather than draw a small box over the neighbour."""
    _cp_ = default_size_grid['chordpi']
    _open_panel(_cp_)
    _hidden_ = _cp_.root.evaluate("""(rootEl) => {
        const d = rootEl.querySelector('[id="drag"]');
        const b = d.getBoundingClientRect();
        const x = b.left + b.width / 2, y = b.top + b.height / 2;
        if (x < 0 || y < 0) { return 'off-page'; }          // the check would be vacuous
        d.setAttribute('pointer-events', 'all');
        try {
            let hit = document.elementFromPoint(x, y);
            while (hit && hit.shadowRoot) {
                const inner = hit.shadowRoot.elementFromPoint(x, y);
                if (!inner || inner === hit) { break; }
                hit = inner;
            }
            return hit !== d;
        } finally {
            d.removeAttribute('pointer-events');
        }
    }""")
    assert _hidden_ is True, f'the parked drag band is visible (or the check is vacuous: {_hidden_!r})'


if __name__ == '__main__':
    unittest.main()
