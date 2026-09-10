"""Phase 1 of PLANNING.md 2.1: prove the browser harness end-to-end.

These tests are the harness's own acceptance criteria, not coverage of the 68
``LINKPI`` bindings (that is phase 2).  Each one pins down a link in the chain that
no Python test can reach:

  render -> focus -> keydown -> preventDefault -> websocket -> Python -> re-render

If any of these go red, the fault is in the harness or in the interaction layer's
plumbing, and every phase-2 case built on top of it is untrustworthy.
"""
import unittest

import pytest
from playwright.sync_api import expect


# ── the chain, one link at a time ─────────────────────────────────────────────

def test_component_renders_into_the_browser(linkpi_page):
    """The ReactiveHTML model reached the DOM at all.

    Failing here means the websocket never connected -- which is what happens if the
    page is opened on 127.0.0.1 rather than localhost (see interaction_harness).
    """
    expect(linkpi_page.root).to_be_visible()
    assert linkpi_page.within('mod', 'circle').count() == 5


def test_hovering_gives_the_svg_keyboard_focus(linkpi_page):
    """Focus is the silent killer: on the wrong element every binding is a no-op.

    This is the exact signature of the ``z`` bug, and it is invisible to the Python
    tests by construction -- they call the handler directly.
    """
    linkpi_page.hover(200, 150)
    linkpi_page.expect_focused()


def test_keydown_reaches_the_interaction_root(linkpi_page):
    """The JS handler is actually wired to the element the template says it is."""
    linkpi_page.hover(200, 150)
    linkpi_page.clear_keydowns()
    linkpi_page.press('z')
    assert linkpi_page.last_keydown()['key'] == 'z'


# ── the validation case: PLANNING.md 2.1 phase 1 step 5 ───────────────────────

def test_z_selects_every_node_of_the_same_colour(linkpi_page):
    """Press ``z`` over a node and the same-coloured nodes draw as selected.

    Nodes 1/2/3 share one colour and 4/5 the other, on a graph with **integer** ids
    -- the case where ``color_nodes_final`` (keyed by the stringified name) and
    ``pos`` (keyed by the original id) have to agree.  ``expect_selected`` asserts
    the rendered selection, not the controller's opinion of it, so the
    handler-fires-and-does-nothing failure shows up as a plain mismatch.
    """
    linkpi_page.hover_node(1)
    linkpi_page.press('z')
    linkpi_page.expect_selected(3)


def test_z_over_the_other_colour_group_selects_that_group(linkpi_page):
    linkpi_page.hover_node(4)
    linkpi_page.press('z')
    linkpi_page.expect_selected(2)


def test_z_over_empty_space_clears_the_selection(linkpi_page):
    """A second ``z`` fires because applyKeyOp resets ``key_op_finished`` to ''.

    Without that reset the param watcher -- which fires on *change* -- would never
    see the repeat, and this test would hang on the old selection.
    """
    linkpi_page.hover_node(1)
    linkpi_page.press('z')
    linkpi_page.expect_selected(3)

    linkpi_page.hover(2, 2)          # canvas corner: no node there
    linkpi_page.press('z')
    linkpi_page.expect_selected(0)


# ── preventDefault: PLANNING.md 2.1 item 1, first execution ever ──────────────

def test_ctrl_a_suppresses_select_all_and_still_opens_the_menu(linkpi_page):
    """Both halves of the guard, which is what makes this browser-only.

    ``ctrl-a`` is one of the nine ``preventDefault`` sites and its browser default
    (select-all) would otherwise fight the component.  ``defaultPrevented``, read
    from a bubble-phase listener above the handler, is the browser's own answer to
    "was the default suppressed?"; the picker menu appearing is the proof that
    suppressing it did not also cost us the handler.
    """
    linkpi_page.hover(200, 150)
    linkpi_page.clear_keydowns()
    linkpi_page.press('a', ctrl=True)

    _ev_ = linkpi_page.last_keydown()
    assert _ev_['ctrlKey'] is True
    assert _ev_['defaultPrevented'] is True, 'ctrl-a did not call preventDefault()'

    # ...and the handler still ran.  The menu arms a 2.5s auto-commit timer, so this
    # assertion is deliberately on an auto-retrying expect rather than a sleep.
    linkpi_page.expect_menu_open('timing mark spacing')


def test_plain_a_does_not_suppress_the_browser_default(linkpi_page):
    """The guard is scoped to the modifier, not applied to the bare key.

    Without this, a test asserting ``defaultPrevented`` on ctrl-a would still pass if
    the handler blanket-suppressed every keystroke.
    """
    linkpi_page.hover(200, 150)
    linkpi_page.clear_keydowns()
    linkpi_page.press('a')
    assert linkpi_page.last_keydown()['defaultPrevented'] is False


# ── the cheapest useful slice: keys that must change the render ───────────────

@pytest.mark.parametrize('key', ['d', 'a'])
def test_representative_keys_change_the_rendered_svg(linkpi_page, key):
    """d = louvain community colours, a = link arrows.

    Asserting only "something changed" is weak on purpose -- it is the assertion that
    would have caught the ``z`` bug class (handler runs, returns cleanly, changes
    nothing) at the layer where it happens.
    """
    linkpi_page.hover(200, 150)
    _before_ = linkpi_page.mod_html()
    linkpi_page.press(key)
    _after_ = linkpi_page.wait_for_mod_change(_before_)
    assert _after_ != _before_


def test_b_cycles_the_background_state_without_redrawing(linkpi_page):
    """``b`` is the case where "assert the SVG changed" would be *wrong*.

    The cycle advances no-background -> background -> background + labels, but with no
    layout background computed yet ``__applyBackgroundState__`` has nothing to draw,
    so ``#mod`` is legitimately untouched.  The state still moved, and ``#infostr``
    is where it shows -- which is the argument for choosing the assertion surface per
    operation rather than reaching for a screenshot diff.
    """
    linkpi_page.hover(200, 150)
    linkpi_page.press('b')
    linkpi_page.expect_info_contains('| background')

    linkpi_page.press('b')
    linkpi_page.expect_info_contains('| background + labels')

    linkpi_page.press('b')
    linkpi_page.expect_info_contains('| no background')


if __name__ == '__main__':
    unittest.main()
