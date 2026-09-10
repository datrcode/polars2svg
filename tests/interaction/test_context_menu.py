"""The context-menu guard -- PLANNING.md **U1**.

On macOS ctrl+click *is* the secondary click, so holding ctrl to add to a rectangular
selection makes the browser raise its popup mid-drag: the drag is interrupted and the
selection lost.  The report notes a previous fix that "didn't hold", and the plan named
this as one of the things only a browser can reach -- *"a Playwright test turns it into
a red test and says immediately whether the next fix holds."*

Running that test showed the fix had gone in on the generic ``_interactivep``
components and been missed on LINKPI, which is the component with the most dragging in
it. Both are now guarded, and both are asserted here so a future template edit that
drops one is a red test rather than a bug report months later.

Note the shape of the check: ``defaultPrevented``, read from a listener added after the
component's own. Whether the popup *visually* appears is not observable from inside the
page, and it is not the contract anyway -- the contract is that the component claims the
event.
"""
import unittest

import pytest


def test_linkp_suppresses_the_context_menu(linkpi_page):
    """The one that was missing. Without it a ctrl-drag on a link plot pops the menu."""
    assert linkpi_page.context_menu_prevented(), (
        'LINKPI let the browser context menu through -- U1 is back, and a ctrl-drag '
        'selection will be interrupted by the popup')


@pytest.mark.parametrize('fixture', ['xypi_page', 'histopi_page', 'timepi_page'])
def test_the_generic_components_suppress_the_context_menu(request, fixture):
    """Where the original fix landed. Guarded so a template refactor cannot quietly
    take it away again -- which is exactly how LINKPI came to be missing it."""
    _ip_ = request.getfixturevalue(fixture)
    assert _ip_.context_menu_prevented()


def test_a_ctrl_drag_survives_the_secondary_click(quad_page):
    """The gesture the guard exists for, end to end.

    A ctrl-drag is *supposed* to add to the selection. On macOS the same press is a
    secondary click, so without the guard the popup interrupts it. Driving the real
    gesture -- ctrl held across mousedown, move and mouseup -- and asserting the
    selection actually grew is the closest a page can get to "the popup did not eat
    the drag".
    """
    _c_ = {_r_['__first__']: (int(_r_['__sx__']), int(_r_['__sy__']))
           for _r_ in quad_page.plot.df_node.iter_rows(named=True)}
    _mid_y_ = (_c_['nw'][1] + _c_['sw'][1]) // 2

    quad_page.drag(2, 2, 398, _mid_y_)          # the northern row
    quad_page.expect_selected(2)

    with quad_page.holding(ctrl=True):
        quad_page.drag(2, 2, 398, 298)          # ctrl-drag everything: add
    quad_page.expect_selected(4)


def test_the_guard_does_not_swallow_the_ordinary_left_drag(quad_page):
    """A blanket contextmenu guard is cheap; a blanket *mouse* guard would not be.

    Asserts the plain rubber band still works, so the fix cannot be "prevent
    everything".
    """
    quad_page.drag(2, 2, 398, 298)
    quad_page.expect_selected(4)


if __name__ == '__main__':
    unittest.main()
