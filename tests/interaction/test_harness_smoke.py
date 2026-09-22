"""Phase 1 of PLANNING.md 2.1: prove the browser harness end-to-end.

These tests are the harness's own acceptance criteria, not coverage of the 68
``LINKPI`` bindings (that is phase 2).  Each one pins down a link in the chain that
no Python test can reach:

  render -> focus -> keydown -> preventDefault -> websocket -> Python -> re-render

If any of these go red, the fault is in the harness or in the interaction layer's
plumbing, and every phase-2 case built on top of it is untrustworthy.
"""
import time
import unittest

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

def test_ctrl_s_suppresses_save_page_and_still_reaches_the_handler(linkpi_page):
    """Both halves of the guard, which is what makes this browser-only.

    ``ctrl-s`` is a ``preventDefault`` site and its browser default (Save Page As) would
    otherwise fight the component.  ``defaultPrevented``, read from a bubble-phase
    listener above the handler, is the browser's own answer to "was the default
    suppressed?"; the sticky-label set growing is the proof that suppressing it did not
    also cost us the handler.

    This used to be ctrl-a, whose default is select-all and whose handler opened the
    timing-mark spacing picker.  The configuration panel absorbed that binding, and
    ctrl-a is deliberately unbound now -- so select-all goes back to the browser and
    there is no guard left to test there.  test_prevent_default.py asserts that
    release; this needs a chord that still has both halves.
    """
    # A node, not empty canvas: ctrl-s ADDS the selection to the sticky set, so with
    # nothing selected the handler would run and correctly change nothing -- which is
    # indistinguishable from the handler never running, and is the whole bug class the
    # second half of this assertion exists to catch.
    linkpi_page.hover_node(1)
    linkpi_page.press('z')
    linkpi_page.expect_selected(3)                # node 1's colour group on this fixture
    linkpi_page.hover_node(1)
    linkpi_page.clear_keydowns()
    linkpi_page.press('s', ctrl=True)

    _ev_ = linkpi_page.last_keydown()
    assert _ev_['ctrlKey'] is True
    assert _ev_['defaultPrevented'] is True, 'ctrl-s did not call preventDefault()'

    # ...and the handler still ran.
    _view_ = linkpi_page.app.view()
    _deadline_ = time.monotonic() + 10.0
    while time.monotonic() < _deadline_:
        if _view_.sticky_labels:
            return
        time.sleep(0.05)
    raise AssertionError('ctrl-s was suppressed but never reached the handler')


def test_plain_a_does_not_suppress_the_browser_default(linkpi_page):
    """The guard is scoped to the modifier, not applied to the bare key.

    Without this, a test asserting ``defaultPrevented`` on ctrl-a would still pass if
    the handler blanket-suppressed every keystroke.
    """
    linkpi_page.hover(200, 150)
    linkpi_page.clear_keydowns()
    linkpi_page.press('c')
    assert linkpi_page.last_keydown()['defaultPrevented'] is False


# ── the cheapest useful slice: keys that must change the render ───────────────

def test_a_representative_key_changes_the_rendered_svg(linkpi_page):
    """d = louvain community colours.

    Asserting only "something changed" is weak on purpose -- it is the assertion that
    would have caught the ``z`` bug class (handler runs, returns cleanly, changes
    nothing) at the layer where it happens.
    """
    linkpi_page.hover(200, 150)
    _before_ = linkpi_page.mod_html()
    linkpi_page.press('d')
    _after_ = linkpi_page.wait_for_mod_change(_before_)
    assert _after_ != _before_


def test_a_configuration_panel_row_changes_the_rendered_svg(linkpi_page):
    """The other half of the pair above, which used to be the 'a' key (link arrows).

    'a' is the panel key now and the arrows are its first row, so the same assertion
    lives here -- and it is worth keeping at this layer rather than folding into
    test_config_panel.py, because it is the one that says the whole chain *draws*:
    keystroke -> JS -> debounce -> websocket -> Python -> re-render -> new #mod.
    """
    linkpi_page.hover(200, 150)
    _before_ = linkpi_page.mod_html()
    linkpi_page.press('a')                        # opens on the 'arrows' row
    linkpi_page.expect_panel_open()
    linkpi_page.press(' ')                        # off -> on
    _after_ = linkpi_page.wait_for_mod_change(_before_)
    assert _after_ != _before_


def test_the_background_row_moves_without_redrawing(linkpi_page):
    """The case where "assert the SVG changed" would be *wrong*.

    'b' used to cycle this; it is the panel's 'background' row now.  The row advances
    off -> on -> on + labels, but with no layout background computed yet
    ``__applyBackgroundState__`` has nothing to draw, so ``#mod`` is legitimately
    untouched -- which is the argument for choosing the assertion surface per operation
    rather than reaching for a screenshot diff.  With nothing to draw the row is also
    gated off, so this drives it through the param the row writes rather than through
    the cursor, which would (correctly) refuse to land on it.
    """
    linkpi_page.hover(200, 150)
    _view_ = linkpi_page.app.view()
    for _state_ in ('on', 'on + labels', 'off'):
        _view_.background_state_choice = _state_
        linkpi_page.wait_until_idle()
    assert _view_.background_state == 0


def test_the_panel_is_where_the_background_state_reads(linkpi_page):
    """It left info_str when it became a panel row -- so the panel must show it."""
    linkpi_page.hover(200, 150)
    linkpi_page.press('a')
    linkpi_page.expect_panel_open()
    assert 'background' in linkpi_page.panel_values(), linkpi_page.panel_text()
    assert linkpi_page.panel_values()['background'] == 'off'


if __name__ == '__main__':
    unittest.main()
