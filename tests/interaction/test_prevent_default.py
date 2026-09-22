"""The nine ``preventDefault`` guards in the LINKPI keydown handler.

PLANNING.md 2.1 phase 2 item 1 -- highest value, and previously zero coverage. Each
guard has to prove **two** things, and only a browser can express either:

* the browser default was suppressed (ctrl-s would otherwise open Save Page As,
  ctrl-a select-all, ctrl-c clobber the component's own clipboard write, ctrl-e
  focus the search bar), and
* suppressing it did not also cost us the handler.

``defaultPrevented``, read from a listener on the interaction root, is the browser's
own answer to the first.  The second is an ordinary DOM assertion, one per binding,
because each guard protects a different operation.

The negative controls matter as much as the positives: a handler that blanket-called
``preventDefault()`` on every keystroke would satisfy all nine positive assertions
while breaking ordinary typing everywhere else in the page.
"""
import time
import unittest

import pytest


#: (event.key, modifiers, human label).  Every `preventDefault` site inside
#: myOnKeyDown's binding chain -- `grep -n preventDefault polars2svg/js/p2s_linkpi.js`.
#:
#: Four went with the configuration panel (ctrl-a, ctrl-l, ctrl-o, ctrl-p opened the
#: spacing / size / opacity / node-size pickers).  The keys they guarded are not bound
#: at all now, so there is nothing to suppress and select-all, the address bar, Open
#: File and Print go back to the browser -- which is the correct outcome and is asserted
#: below rather than left implicit.  ctrl-shift-s survives as a guard although it no
#: longer cycles the label mode: it still reaches the sticky-label handler.
CTRL_GUARDS = [
    ('c', dict(ctrl=True),             'ctrl-c  copy selection (vs native copy)'),
    ('C', dict(ctrl=True, shift=True), 'ctrl-shift-c  copy labels'),
    ('e', dict(ctrl=True),             'ctrl-e  expand reversed (vs search-bar focus)'),
    ('s', dict(ctrl=True),             'ctrl-s  sticky labels (vs Save Page As)'),
    ('S', dict(ctrl=True, shift=True), 'ctrl-shift-s  subtract from sticky labels'),
]

#: The same keys without ctrl.  None of them may suppress anything.
BARE_KEYS = ['c', 'C', 'e', 's', 'S']

#: The four the panel absorbed.  They must reach neither a handler nor a guard.
RELEASED_CHORDS = [
    ('a', 'ctrl-a  select-all'),
    ('l', 'ctrl-l  address bar'),
    ('o', 'ctrl-o  Open File'),
    ('p', 'ctrl-p  Print'),
]


def _select_one_colour_group(ip, node):
    """Give the component a selection via the already-proven ``z`` binding."""
    ip.hover_node(node)
    ip.press('z')


# ── half one: the guard fires ────────────────────────────────────────────────

@pytest.mark.parametrize('key,mods,label', CTRL_GUARDS, ids=[c[2].split()[0] for c in CTRL_GUARDS])
def test_ctrl_binding_suppresses_the_browser_default(linkpi_page, key, mods, label):
    linkpi_page.hover(200, 150)
    linkpi_page.clear_keydowns()
    linkpi_page.press(key, **mods)

    _ev_ = linkpi_page.last_keydown()
    assert _ev_['ctrlKey'] is True, f'{label}: ctrl did not reach the handler'
    assert _ev_['defaultPrevented'] is True, f'{label}: preventDefault() was not called'


@pytest.mark.parametrize('key', BARE_KEYS)
def test_bare_key_does_not_suppress_the_browser_default(linkpi_page, key):
    """The guards are scoped to the modifier rather than applied to every keystroke."""
    linkpi_page.hover(200, 150)
    linkpi_page.clear_keydowns()
    linkpi_page.press(key)

    _ev_ = linkpi_page.last_keydown()
    assert _ev_['ctrlKey'] is False
    assert _ev_['defaultPrevented'] is False, (
        f'bare {key!r} suppressed the browser default -- the guard is not modifier-scoped')


# ── half two: the handler still ran ──────────────────────────────────────────

@pytest.mark.parametrize('key,label', RELEASED_CHORDS, ids=[c[1].split()[0] for c in RELEASED_CHORDS])
def test_an_absorbed_chord_is_given_back_to_the_browser(linkpi_page, key, label):
    """The panel freed these; nothing may still be claiming them.

    Not cosmetic.  A leftover `preventDefault` with no handler behind it is the worst of
    both worlds -- the browser's own shortcut stops working and nothing replaces it --
    and it is invisible from inside the app, because the only symptom is a browser
    feature that quietly does not happen.
    """
    linkpi_page.hover(200, 150)
    linkpi_page.clear_keydowns()
    linkpi_page.press(key, ctrl=True)
    assert linkpi_page.last_keydown()['defaultPrevented'] is False, (
        f'{label}: still suppressed, but nothing is bound to it any more')
    assert not linkpi_page.menu_is_open(), f'{label}: still opens a picker'
    assert not linkpi_page.panel_is_open(), f'{label}: still opens the panel'


def _await_clipboard(sentinel, timeout_s=10.0):
    """Wait for the server thread's pyperclip write to land, bounded.

    A Python-side wait, not a DOM one -- the clipboard is not in the page, so there is
    no locator to hand to expect().  Bounded polling, same idea, different surface.
    """
    import pyperclip
    _deadline_ = time.monotonic() + timeout_s
    while time.monotonic() < _deadline_:
        _now_ = pyperclip.paste()
        if _now_ != sentinel:
            return _now_
        time.sleep(0.05)
    raise AssertionError(f'clipboard still held the sentinel after {timeout_s}s '
                         f'-- ctrl-c did not reach _copyToClipboard_')


def _clipboard_or_skip():
    """Prime the clipboard with a sentinel, skipping where the host has none.

    The overlay would have been the in-page way to observe this, but it never renders
    (see test_operation_feedback_overlay_never_displays), so the assertion goes to the
    real end of the chain instead: the system clipboard the component actually writes.
    """
    pyperclip = pytest.importorskip('pyperclip')
    _sentinel_ = '__p2s_clipboard_sentinel__'
    try:
        pyperclip.copy(_sentinel_)
        if pyperclip.paste() != _sentinel_:
            pytest.skip('clipboard readback does not work on this host')
    except Exception as _e_:
        pytest.skip(f'no clipboard mechanism on this host ({_e_})')
    return _sentinel_


def test_ctrl_c_copies_the_selected_node_ids(linkpi_page):
    """End to end: keystroke in the browser, text on the system clipboard.

    ``holding`` rather than ``press(ctrl=True)``: the Python branch reads
    ``self.ctrlkey``, and a released modifier races the handler (see U3 and
    test_keyup_clears_the_modifier_before_python_reads_it).
    """
    _sentinel_ = _clipboard_or_skip()
    _select_one_colour_group(linkpi_page, 1)
    linkpi_page.expect_selected(3)

    linkpi_page.hover_node(1)
    with linkpi_page.holding(ctrl=True):
        linkpi_page.press('c')
        _text_ = _await_clipboard(_sentinel_)

    assert sorted(_text_.split('\n')) == ['1', '2', '3'], (
        f'ctrl-c copied {_text_!r}; expected the three selected node ids')


def test_ctrl_shift_c_copies_the_node_labels_instead_of_the_ids(labelled_page):
    """The distinction between the two clipboard bindings, which is the whole point
    of having both: ctrl-c writes ids, ctrl-shift-c writes the display labels."""
    _sentinel_ = _clipboard_or_skip()
    _select_one_colour_group(labelled_page, 1)
    labelled_page.expect_selected(3)

    labelled_page.hover_node(1)
    with labelled_page.holding(ctrl=True, shift=True):
        labelled_page.press('C')
        _text_ = _await_clipboard(_sentinel_)

    assert sorted(_text_.split('\n')) == ['alpha', 'bravo', 'charlie'], (
        f'ctrl-shift-c copied {_text_!r}; expected the node labels, not the ids')


def test_ctrl_e_still_expands_along_reversed_edges(chain_page):
    """On the chain 1->2->3->4, ctrl-e grows a selection *against* edge direction.

    Selecting node 4 and expanding backwards must pick up 3 and nothing else.  The
    chain fixture exists for this: on a cycle the answer would be the whole graph
    whether the binding worked or not.
    """
    chain_page.hover_node(4)
    chain_page.press('z')
    chain_page.expect_selected(1)

    chain_page.hover_node(4)
    with chain_page.holding(ctrl=True):
        chain_page.press('e')
        chain_page.expect_selected(2)


def test_the_labels_row_cycles_the_label_mode(linkpi_page):
    """ctrl-shift-s used to do this; the mode is the panel's 'labels' row now.

    This graph has no link labels, so the cycle is the three-entry one and
    'no labels' advances to 'node labels'.
    """
    linkpi_page.hover(200, 150)
    linkpi_page.press('a')
    linkpi_page.expect_panel_open()
    linkpi_page.press('l')                        # the 'labels' row
    linkpi_page.press(' ')
    linkpi_page.expect_panel_value('labels', 'node labels')


def test_ctrl_s_still_adds_the_selection_to_sticky_labels(linkpi_page):
    """Sticky labels only *draw* in a label mode that shows them, so get there first.

    Two `space` presses on the panel's 'labels' row walk the no-link cycle 'no labels'
    -> 'node labels' -> 'sticky labels'; with the mode showing them and the set still
    empty, ctrl-s is then the only thing that can put text on the canvas.  ctrl-shift-s
    used to be what walked this; the row absorbed it.
    """
    linkpi_page.hover(200, 150)
    linkpi_page.press('a')
    linkpi_page.expect_panel_open()
    linkpi_page.press('l')
    linkpi_page.press(' ')
    linkpi_page.expect_panel_value('labels', 'node labels')
    linkpi_page.press(' ')
    linkpi_page.expect_panel_value('labels', 'sticky labels')
    linkpi_page.press('Escape')
    linkpi_page.expect_panel_closed()

    _select_one_colour_group(linkpi_page, 1)
    linkpi_page.expect_selected(3)

    _before_ = linkpi_page.mod_html()
    linkpi_page.hover_node(1)
    with linkpi_page.holding(ctrl=True):
        linkpi_page.press('s')
        _after_ = linkpi_page.wait_for_mod_change(_before_)
    assert _after_ != _before_


# ── the race the above used to have to work around (PLANNING.md U3) ─────────

def test_a_tapped_ctrl_c_copies_just_like_a_held_one(linkpi_page):
    """Tapping ctrl-c must do the same thing as holding it. It did not until U3.

    ``myOnKeyUp`` writes ``data.ctrlkey = event.ctrlKey`` unconditionally, so releasing
    Control set it back to false -- and ``applyKeyOp`` runs asynchronously behind a
    lock, so it frequently read the *post-keyup* value and took the unmodified branch:
    ctrl-c zoomed the view instead of copying, silently doing the wrong operation.

    The fix is a second pair of params, ``op_ctrlkey`` / ``op_shiftkey``, written only
    by the keydown, which ``applyKeyOp`` adopts before dispatching. This test presses
    the modifier the way a fast user does -- down and up, back to back -- and asserts
    the copy still happens.
    """
    _sentinel_ = _clipboard_or_skip()
    _select_one_colour_group(linkpi_page, 1)
    linkpi_page.expect_selected(3)

    linkpi_page.hover_node(1)
    linkpi_page.press('c', ctrl=True)           # tapped, not held
    _text_ = _await_clipboard(_sentinel_)

    assert sorted(_text_.split('\n')) == ['1', '2', '3'], (
        f'a tapped ctrl-c copied {_text_!r}; the modifier was lost between the keyup '
        f'and the handler again')


def test_the_live_modifier_state_still_follows_the_keyboard(linkpi_page):
    """The snapshot is *additional*, not a replacement.

    ``ctrlkey`` / ``shiftkey`` stay live because the mouse handlers read them at their
    own moment -- ``myOnMouseUp`` decides a drag's set-operation from them. Pinning
    that they still clear on release keeps the fix from quietly freezing the modifier
    state that the drag path depends on.
    """
    _view_ = linkpi_page.app.view()
    linkpi_page.hover(200, 150)
    with linkpi_page.holding(ctrl=True):
        linkpi_page.press('c')
        _await_true(lambda: _view_.ctrlkey is True, 'ctrlkey never went live')

    _await_true(lambda: _view_.ctrlkey is False,
                'ctrlkey stayed set after the modifier was released')


def _await_true(predicate, message, budget_s=10.0):
    _deadline_ = time.monotonic() + budget_s
    while time.monotonic() < _deadline_:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError(message)


def test_the_confirm_gate_prompt_reaches_the_info_line(linkpi_page):
    """The one message that survived the overlay's removal, and the reason it had to.

    ``setAnimation()`` used to announce sixteen things -- stack pushes, community
    counts, clipboard results -- through an overlay that never rendered (U4).  The
    overlay is gone; fifteen of those were status chatter that nobody could see anyway.

    The sixteenth is a *prompt*: an operation over its size threshold refuses and asks
    to be repeated to confirm. All three callers return without refreshing, so before
    this the refusal produced nothing at all on screen and the gesture was
    undiscoverable. The gate now puts its note on the info line, which demonstrably
    renders -- this suite asserts on ``#infostr`` throughout.

    Driven through the *browser* rather than by calling the gate: the point is that the
    prompt survives the whole round trip and lands somewhere a user would see it.
    """
    _view_ = linkpi_page.app.view()
    _lower_confirm_threshold(_view_, _view_.layout_operation)

    linkpi_page.hover(200, 150)
    linkpi_page.press('w')                      # apply the layout operation
    linkpi_page.expect_info_contains('repeat to run')


def _lower_confirm_threshold(view, op, limit=1):
    """Make one layout operation ask about a graph this small.

    Same shape as the helper in tests/test_interactive_controller.py: replace the
    registry entry's treatment rather than reaching into the gate, so the test exercises
    the real threshold path.
    """
    from polars2svg.interactive_treatments import RegistryEntry, Treatment
    _handler_ = view._layout_registry[op].handler
    view._layout_registry[op] = RegistryEntry(_handler_, Treatment(confirm_above=limit))


if __name__ == '__main__':
    unittest.main()
