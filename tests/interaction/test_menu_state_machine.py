"""The picker-menu state machine -- PLANNING.md 2.1 phase 2 item 3.

Six entry points, five exit paths, and until now zero tests, because the whole thing
lives in JavaScript: ``menu_open``, ``menu_kind``, ``menu_index``, ``menu_items`` and
``menu_timer`` are five of the 29 ``state.*`` variables that never cross into Python.
Nothing reaches the server until ``menuCommit`` writes a choice back, so a menu that
opened on the wrong item, cycled the wrong way, or committed when it should not have
was previously unobservable by any means except opening a browser and looking.

The observables here are the two the menu actually has:

* ``#pickermenu`` -- drawn entirely by ``menuRender``.  Its highlight rect encodes
  ``state.menu_index`` in its ``y`` (see ``InteractivePage.menu_index``), which is how
  a JS-only variable becomes assertable.
* ``#infostr`` -- carries ``layout_mode`` and ``layout_operation``, so a *commit* is
  visible where a mere selection is not.  That difference is the point of several of
  these tests: the guard added after 'l' then '3' started a force layout in two
  keystrokes is precisely a rule about when selecting may become committing.
"""
import time
import unittest

import pytest
from playwright.sync_api import expect


#: (key, modifiers, menu header).  Every way into the menu system.
ENTRY_POINTS = [
    ('A', {},                  'timing mark spacing'),
    ('a', {'ctrl': True},      'timing mark spacing'),
    ('B', {},                  'background:'),
    ('G', {},                  'layout mode:'),
    ('W', {},                  'layout operation:'),
    ('l', {},                  'link shape:'),
    ('L', {},                  'link size:'),
    ('l', {'ctrl': True},      'link size:'),
    ('O', {},                  'link opacity:'),
    ('o', {'ctrl': True},      'link opacity:'),
    ('P', {},                  'node size:'),
    ('p', {'ctrl': True},      'node size:'),
]


def _open(ip, key, **mods):
    """Settle the page, then open a menu.

    ``settle()`` used to be mandatory: a one-off rebuild landed ~2.05s after load and
    destroyed every JS-only variable, the menu's included, so a menu opened inside that
    window vanished mid-test.  That was PLANNING.md U5 and it is fixed -- nothing
    rebuilds unprompted any more, and settle() now returns immediately on an idle page.
    It is kept because a genuine ``mod_inner`` redraw can still rebuild the subtree, so
    this stays the right thing to do before asserting on browser-only state.  See
    test_a_menu_opened_immediately_after_load_survives.
    """
    ip.settle()
    ip.hover(200, 150)
    ip.press(key, **mods)


# ── entry ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('key,mods,header', ENTRY_POINTS,
                         ids=[f'{"ctrl-" if m.get("ctrl") else ""}{k}-{h.split(":")[0]}'
                              for k, m, h in ENTRY_POINTS])
def test_entry_point_opens_the_right_menu(linkpi_page, key, mods, header):
    _open(linkpi_page, key, **mods)
    linkpi_page.expect_menu_open(header)


def test_menu_opens_on_the_current_value_not_the_first_item(linkpi_page):
    """menuOpen scans for the current value rather than starting at zero.

    Link opacity defaults to 100%, which is the *last* of the ten rows -- so an
    off-by-default implementation would be caught here and nowhere else.
    """
    _open(linkpi_page, 'O')
    linkpi_page.expect_menu_open('link opacity:')
    linkpi_page.expect_menu_index(9)


# ── cycling ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('key', ['ArrowDown', 'j'])
def test_cycles_forward(linkpi_page, key):
    _open(linkpi_page, 'G')                       # layout mode, opens on 'grid' (index 0)
    linkpi_page.expect_menu_index(0)
    linkpi_page.press(key)
    linkpi_page.expect_menu_index(1)


@pytest.mark.parametrize('key', ['ArrowUp', 'k'])
def test_cycles_backward_and_wraps(linkpi_page, key):
    """From index 0 a backward step wraps to the last of the seven layout modes."""
    _open(linkpi_page, 'G')
    linkpi_page.expect_menu_index(0)
    linkpi_page.press(key)
    linkpi_page.expect_menu_index(6)


def test_repeating_the_open_key_cycles_forward(linkpi_page):
    """shift-G again steps down -- the 'press it repeatedly' idiom."""
    _open(linkpi_page, 'G')
    linkpi_page.expect_menu_index(0)
    linkpi_page.press('G')
    linkpi_page.expect_menu_index(1)
    linkpi_page.press('G')
    linkpi_page.expect_menu_index(2)


@pytest.mark.parametrize('key,header', [('G', 'layout mode:'), ('W', 'layout operation:')])
def test_ctrl_with_the_open_key_is_inert(linkpi_page, key, header):
    """ctrl-shift-G / ctrl-shift-W used to reverse-cycle, and were deleted (U10).

    ctrl-shift-W is a reserved chrome-level accelerator off macOS -- a real XTEST press
    **closes the browser window**, and preventDefault() cannot reclaim what the page is
    never shown.  ctrl-shift-G worked, and went with it so the pure-reverse chords are
    gone as a class; ArrowUp / k reverse every menu, which is what the generic
    components have always done.

    Asserts *inert*, not "reverses" and not "does something else": the forward branch
    keeps its `!event.ctrlKey` guard precisely so a lingering habit does nothing rather
    than cycling the wrong way.  Note this cannot fail the way the real defect
    manifests -- Playwright drives keys over CDP, below browser chrome, so the window
    never closes here no matter what the binding says.  Only the source-level guard in
    test_interactive_controller.py catches a reserved chord being re-added.
    """
    _open(linkpi_page, key)
    linkpi_page.expect_menu_open(header)
    linkpi_page.press(key)
    linkpi_page.expect_menu_index(1)

    with linkpi_page.holding(ctrl=True):
        linkpi_page.press(key)

    linkpi_page.expect_menu_open(header)          # still open ...
    linkpi_page.expect_menu_index(1)              # ... and did not move


# ── committing ───────────────────────────────────────────────────────────────

def test_mnemonic_commits_an_unguarded_item_immediately(linkpi_page):
    """One keystroke selects *and* commits, for items cheap enough to allow it."""
    _open(linkpi_page, 'G')
    linkpi_page.press('c')                        # mnemonic for 'circle'
    linkpi_page.expect_menu_closed()
    linkpi_page.expect_info_contains('| circle |')


def test_enter_commits_the_highlighted_item(linkpi_page):
    _open(linkpi_page, 'G')
    linkpi_page.press('ArrowDown')                # grid -> circle
    linkpi_page.expect_menu_index(1)
    linkpi_page.press('Enter')
    linkpi_page.expect_menu_closed()
    linkpi_page.expect_info_contains('| circle |')


def test_escape_closes_without_committing(linkpi_page):
    """Commit something first, so 'unchanged' is distinguishable from 'never set'."""
    _open(linkpi_page, 'G')
    linkpi_page.press('c')
    linkpi_page.expect_info_contains('| circle |')

    _open(linkpi_page, 'G')
    linkpi_page.press('ArrowDown')
    linkpi_page.press('Escape')
    linkpi_page.expect_menu_closed()
    linkpi_page.expect_info_contains('| circle |')


def test_mouse_leaving_the_component_commits(linkpi_page):
    """myOnMouseOut calls menuCommit -- walking away accepts the highlighted item."""
    _open(linkpi_page, 'G')
    linkpi_page.press('ArrowDown')
    linkpi_page.expect_menu_index(1)

    _box_ = linkpi_page.root.bounding_box()       # leave the component for real:
    linkpi_page.page.mouse.move(                  # (5, 5) is *inside* a 400x300 plot
        _box_['x'] + _box_['width'] + 80, _box_['y'] + _box_['height'] + 80)
    linkpi_page.expect_menu_closed()
    linkpi_page.expect_info_contains('| circle |')


def test_inactivity_timeout_commits_an_unguarded_item(linkpi_page):
    """menuArmTimer's 2.5s timeout commits, so a menu never sits open forever.

    One of the few places a real wait is correct: the behaviour under test *is* the
    passage of time, and there is no earlier signal to key off.
    """
    _open(linkpi_page, 'G')
    linkpi_page.press('ArrowDown')
    linkpi_page.expect_menu_index(1)

    time.sleep(3.0)                               # > the 2.5s menu timer
    linkpi_page.expect_menu_closed()
    linkpi_page.expect_info_contains('| circle |')


# ── the guard: expensive items may not be committed by accident ──────────────

def test_guarded_mnemonic_selects_but_does_not_commit(linkpi_page):
    """'l' then '3' used to start the force layout in two keystrokes.

    flowmap is annotated as expensive and therefore guarded: its mnemonic may move the
    highlight, but only an explicit Enter may commit it.
    """
    _open(linkpi_page, 'l')                       # link shape: line / curve / flowmap
    linkpi_page.expect_menu_open('link shape:')
    linkpi_page.press('3')                        # mnemonic for the guarded 'flowmap'

    linkpi_page.expect_menu_index(2)
    assert linkpi_page.menu_is_open(), (
        'a guarded mnemonic committed on its own -- the two-keystroke force-layout '
        'hazard is back')


def test_guarded_item_times_out_closed_rather_than_committed(linkpi_page):
    """Walking away from a guarded item must not start it (menuArmTimer's branch)."""
    _open(linkpi_page, 'l')
    linkpi_page.press('3')
    linkpi_page.expect_menu_index(2)

    time.sleep(3.0)
    linkpi_page.expect_menu_closed()
    assert 'flowmap' not in linkpi_page.info_text(), (
        'the timeout committed a guarded item')


# ── modality ─────────────────────────────────────────────────────────────────

def test_the_open_menu_swallows_ordinary_bindings(linkpi_page):
    """While open, the menu returns before the dispatch chain -- it is modal.

    'z' would otherwise select a colour group; with a menu up it must do nothing at
    all, or a mnemonic keystroke would fire an unrelated operation behind the overlay.
    """
    _open(linkpi_page, 'G')
    linkpi_page.expect_menu_open('layout mode:')

    linkpi_page.press('z')
    linkpi_page.expect_menu_closed()              # 'z' is not a mode mnemonic... see below
    assert '0 Selected' in linkpi_page.info_text() or 'Selected' not in linkpi_page.info_text(), (
        "'z' reached the selection handler while the picker menu was open")


def test_the_open_menu_suppresses_browser_defaults(linkpi_page):
    """The menu branch calls preventDefault() for every key, not just its own.

    Its keystrokes are single characters that would otherwise reach the page, so the
    blanket guard here is correct -- the opposite of the modifier-scoped guards in
    test_prevent_default.py.
    """
    _open(linkpi_page, 'G')
    linkpi_page.expect_menu_open('layout mode:')
    linkpi_page.clear_keydowns()
    linkpi_page.press('ArrowDown')
    assert linkpi_page.last_keydown()['defaultPrevented'] is True


# ── the load-time rebuild that used to make all of the above need settle() ───

def test_a_menu_opened_immediately_after_load_survives(linkpi_page):
    """No unprompted rebuild on an idle page (PLANNING.md U5, inverted now it is fixed).

    This case used to assert the opposite: a rebuild landed ~2.05s after load with no
    interaction of any kind and took every JS-only variable with it, so opening a picker
    in the first couple of seconds and spending a moment reading the list meant it
    vanished without applying anything -- no error, no message, the keystroke simply had
    no effect.

    **This pins the idle half only, and that half was closed by U4** (the animation
    overlay's ``animation_inner`` was the bound param behind it).  It is deliberately
    kept as a cheap standing guard, but do not mistake it for the U5 regression test:
    restoring ``info_str``'s content binding leaves this one passing, because an idle
    page never writes info_str.  ``test_a_menu_survives_an_info_str_write`` is the sharp
    detector -- verified by putting the binding back, where that one fails and this one
    does not.

    Deliberately does **not** call ``_open()``: the point is that no settling is needed.
    Asserts on the root element's identity rather than on the menu alone, because the
    menu has a second, legitimate way to close -- ``menuArmTimer`` auto-commits after
    2500ms -- and the old version of this test ended up passing on that timer instead of
    on the behaviour it was named for.
    """
    linkpi_page.hover(200, 150)
    linkpi_page.press('G')
    linkpi_page.expect_menu_open('layout mode:')

    linkpi_page.root.evaluate("(rootEl) => { rootEl.__u5_mark__ = true; }")
    _opened_at_ = time.monotonic()

    # Two separate guarantees on two different clocks, because one window cannot serve
    # both.  The rebuild was at ~2.05s of *page age*, so the root has to be watched past
    # that; the menu's own auto-commit is at 2500ms from *menu open*, so it can only be
    # asked to still be open well inside that.  Watching the root to 3s of page age and
    # sampling the menu at 1.5s after opening keeps clear of both edges, which matters
    # on a loaded machine (see PLANNING.md on verifying timing under load).
    _menu_held_ = None
    while (time.monotonic() - linkpi_page.created_at) < 3.0:
        assert linkpi_page.root.evaluate("(rootEl) => rootEl.__u5_mark__ === true"), (
            f'the interaction root was replaced at page age '
            f'{time.monotonic() - linkpi_page.created_at:.2f}s with no interaction -- '
            f'an unprompted rebuild is back (U5)')
        if _menu_held_ is None and (time.monotonic() - _opened_at_) >= 1.5:
            _menu_held_ = linkpi_page.menu_is_open()
        time.sleep(0.05)

    assert _menu_held_ is not False, (
        'the menu opened right after load and was gone 1.5s later, even though the root '
        'was never rebuilt')


def test_a_menu_survives_an_info_str_write(linkpi_page):
    """The specific write that used to rebuild the page, driven from Python.

    ``info_str`` is rewritten after almost every operation, so while it was a child
    this was the single largest source of destroyed menus.  Goes through the view
    object rather than the browser because that is the half a keystroke cannot fake:
    the point is a *server*-side param write arriving while a menu is open.
    """
    _open(linkpi_page, 'W')
    linkpi_page.expect_menu_open('layout operation:')
    linkpi_page.expect_menu_index(0)
    linkpi_page.press('ArrowDown')
    linkpi_page.expect_menu_index(1)

    linkpi_page.app.view().info_str = '0 Selected | no labels | grid | poke | no background'

    time.sleep(1.0)
    assert linkpi_page.menu_is_open(), 'a Python-side info_str write destroyed the open menu'
    linkpi_page.expect_menu_index(1)              # and did not reset the highlight


def test_the_info_line_shows_the_real_status_at_load(linkpi_page):
    """While ``info_str`` was a child it never reached the browser at all.

    ``_init_params`` drops children from ``data``, so the line rendered its param
    default -- the placeholder ``" | | grid"`` -- and stayed there until the first
    operation pushed an update.  The status line was wrong on every freshly loaded
    plot, which is the same defect as U5 seen from the other side.
    """
    expect(linkpi_page.el('infostr')).to_contain_text('0 Selected',
                                                      timeout=linkpi_page.timeout_ms)
    assert linkpi_page.info_text().strip() != '| | grid'


if __name__ == '__main__':
    unittest.main()
