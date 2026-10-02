"""The LINKPI configuration panel -- 20260921_config_panel_design.md.

**These tests had to be written rather than inherited, and that is the point.** The
parity goldens are structurally blind to a JS-only overlay: the digest is taken after a
gesture settles, so ``#selbox`` and ``#drag_rect`` went uncovered for as long as they
existed because they only exist mid-gesture. A configuration panel is that shape --
``panel_open``, ``panel_row`` and ``panel_pending`` never cross into Python, and the only
thing that ever does is a committed value, arriving ~300ms later with nothing to say
which of nine rows produced it. The drawn overlay is therefore the observable, exactly as
``#pickermenu`` is for the picker menus, and ``InteractivePage.panel_row`` /
``panel_values`` are how it is read.

What is checked here is the behaviour the design argued about and nothing else -- the
value sets themselves belong to the pickers and are covered in
``tests/test_interactive_controller.py``:

* the cursor, including that it **skips a gated-off row** (section 6: get this wrong and
  ``space`` silently does nothing and the panel looks broken);
* ``space`` / ``shift-space``, and that the commit is **debounced** (CP7) rather than
  issued on every keystroke -- the objection that would have broken the feature, because
  cycling link shape passes *through* ``flowmap`` and its force layout on the way to
  ``off``;
* modality (CP5) and that ``space`` is reclaimed from the browser's scroll;
* that the panel has **no walk-away auto-commit** of its own (CP6) -- that one is in
  ``test_menu_state_machine.py``, beside the picker timer it is contrasted with.

``Enter`` re-entering the existing picker (CP3) is also in ``test_menu_state_machine.py``,
with the other entry points into those menus.
"""
import time
import unittest

import pytest

from interaction_harness import assert_eventually


#: Row label -> mnemonic, in the panel's own top-to-bottom order.  Duplicated from
#: _CONFIG_PANEL_ROWS_ on purpose: a test that imported the table could not catch a
#: mnemonic collision or a reordering, because it would agree with whatever it found.
ROWS = [
    ('arrows',       'r'),
    ('timing marks', 't'),
    ('spacing',      'p'),
    ('labels',       'l'),
    ('link shape',   'h'),
    ('link size',    'z'),
    ('link opacity', 'o'),
    ('node size',    'n'),
    ('tooltip',      'i'),
    ('layout shape',        'g'),
    ('layout operation',    'w'),
    ('background producer', 'f'),
    ('community detection', 'd'),
    ('background',   'b'),
]

#: The rows ``timing_page`` has live.  'background' is gated off on **every** fixture
#: here, and not for want of a better one: the row decides whether a layout-produced
#: background is *drawn*, and only running a layout (or a producer, with 'b') creates
#: one.  A fixture that pre-seeded it would be testing a state the user cannot
#: reach by opening the panel, so the row's own coverage is the gating tests below plus
#: test_harness_smoke.py's state walk.
ENABLED_ROWS = [_r_ for _r_ in ROWS if _r_[0] != 'background']


def _open(ip):
    """Settle the page, take focus, open the panel."""
    ip.settle()
    ip.hover(200, 150)
    ip.press('a')
    ip.assert_panel_open()


# ── the frame ────────────────────────────────────────────────────────────────

def test_a_opens_the_panel_and_escape_closes_it(linkpi_page):
    assert not linkpi_page.panel_is_open(), 'the panel must not be open at mount'
    _open(linkpi_page)
    linkpi_page.press('Escape')
    linkpi_page.assert_panel_closed()


def test_every_row_is_drawn_with_a_value(timing_page):
    """The success criterion from the design: glance at it and know how the view is drawn.

    Driven against the timing fixture so every row but 'background' is live; a blank value column
    would mean the row is showing a param the panel cannot read.
    """
    _open(timing_page)
    _values_ = timing_page.panel_values()
    assert [_l_ for _l_, _ in ROWS] == list(_values_), timing_page.panel_text()
    for _label_, _ in ROWS:
        assert _values_[_label_] != '', f'row {_label_!r} drew no value'


def test_the_rows_show_what_the_view_is_actually_doing(timing_page):
    """Not placeholders: this fixture is built with time= and defaults elsewhere."""
    _open(timing_page)
    _values_ = timing_page.panel_values()
    assert _values_['arrows']       == 'off'
    assert _values_['timing marks'] == 'on'      # built with time='ts'
    assert _values_['link shape']   == 'line'
    assert _values_['link opacity'] == '100'
    assert _values_['labels']       == 'no labels'


# ── the row cursor ───────────────────────────────────────────────────────────

def test_the_panel_key_advances_the_cursor(timing_page):
    _open(timing_page)
    timing_page.assert_panel_row(0)
    timing_page.press('a')
    timing_page.assert_panel_row(1)
    timing_page.press('a')
    timing_page.assert_panel_row(2)


def test_shift_panel_key_retreats_the_cursor(timing_page):
    """CP4 -- shift reverses, never ctrl.  Same idiom as shift-space on a value.

    It wraps to the last *enabled* row, which is 'community detection' and not
    'background':
    skipping a gated row is the cursor's job in both directions, and a backwards wrap is
    the one place a one-directional implementation would show.
    """
    _open(timing_page)
    timing_page.assert_panel_row(0)
    timing_page.press('A')
    timing_page.assert_panel_row(len(ROWS) - 2)   # wraps past the gated 'background'
    timing_page.press('A')
    timing_page.assert_panel_row(len(ROWS) - 3)


@pytest.mark.parametrize('label,mnemonic', ENABLED_ROWS,
                         ids=[r[0].replace(' ', '-') for r in ENABLED_ROWS])
def test_a_mnemonic_jumps_straight_to_its_row(timing_page, label, mnemonic):
    _open(timing_page)
    timing_page.press(mnemonic)
    timing_page.assert_panel_row([_l_ for _l_, _ in ROWS].index(label))


def test_the_mnemonics_are_distinct_and_do_not_include_the_panel_key(timing_page):
    """Guard the table above: a/A and j/k move the cursor, so none can also be a row --
    the panel tests them before it scans for a row mnemonic."""
    _ms_ = [_m_ for _, _m_ in ROWS]
    assert len(set(_ms_)) == len(_ms_), f'duplicate row mnemonic in {_ms_}'
    assert not set(_ms_) & set('aAjk')


# ── dependency gating ────────────────────────────────────────────────────────

def test_an_unavailable_row_is_greyed(linkpi_page):
    """This graph has no date column, so timing marks cannot be turned on at all."""
    _open(linkpi_page)
    assert linkpi_page.panel_row_is_disabled('timing marks')
    assert linkpi_page.panel_row_is_disabled('spacing')
    assert linkpi_page.panel_row_is_disabled('background')   # no layout background yet
    assert not linkpi_page.panel_row_is_disabled('arrows')


def test_the_cursor_skips_a_disabled_row(linkpi_page):
    """Greying alone is not enough.

    Landing on a row where `space` does nothing is the failure the design names by
    name: the panel looks broken rather than the setting looking unavailable.  Rows 1
    and 2 are the gated pair here, so one advance from row 0 must reach row 3.
    """
    _open(linkpi_page)
    linkpi_page.assert_panel_row(0)               # arrows
    linkpi_page.press('a')
    linkpi_page.assert_panel_row(3)               # labels -- 1 and 2 skipped


def test_a_disabled_row_ignores_its_mnemonic(linkpi_page):
    _open(linkpi_page)
    linkpi_page.assert_panel_row(0)
    linkpi_page.press('t')                        # 'timing marks', gated off here
    linkpi_page.assert_panel_row(0)


def test_gating_follows_the_view_rather_than_the_graph(timing_page):
    """'spacing' depends on the marks being ON, which is state the user changes.

    So the gate has to be recomputed on every refresh, not decided once at construction.
    """
    _open(timing_page)
    assert not timing_page.panel_row_is_disabled('spacing')
    timing_page.press('t')                        # the 'timing marks' row
    timing_page.press(' ')                        # -> off
    timing_page.assert_panel_value('timing marks', 'off')
    # Polled, not asserted: the value shows locally at once, while the gate comes back
    # from Python on the refresh that the debounced commit provokes.
    timing_page.assert_panel_row_disabled('spacing')
    timing_page.press(' ')                        # -> back on
    timing_page.assert_panel_row_disabled('spacing', False)


# ── cycling ──────────────────────────────────────────────────────────────────

def test_space_cycles_the_selected_row_forward(timing_page):
    _open(timing_page)
    timing_page.press('h')                        # link shape: line / curve / flowmap
    timing_page.press(' ')
    timing_page.assert_panel_value('link shape', 'curve')


def test_shift_space_cycles_backward(timing_page):
    """The whole reason the panel shows the current value only (design section 5): once
    reversing costs one keypress, a permanent inline preview on every row is paying full
    price for a solved problem."""
    _open(timing_page)
    timing_page.press('r')                        # arrows: off / on
    timing_page.press(' ')
    timing_page.assert_panel_value('arrows', 'on')
    timing_page.press(' ', shift=True)
    timing_page.assert_panel_value('arrows', 'off')


def test_a_rows_key_pressed_again_cycles_it(timing_page):
    """User feedback 2026-09-27: the key that jumps to a row should, pressed again,
    move the row to its next value -- the same as `space`."""
    _open(timing_page)
    _before_ = timing_page.panel_values()['node size']
    timing_page.press('n')                        # jump: node size, value unchanged
    timing_page.assert_panel_row([_l_ for _l_, _ in ROWS].index('node size'))
    assert timing_page.panel_values()['node size'] == _before_
    timing_page.press('n')                        # again: cycles
    assert_eventually(lambda: timing_page.panel_values()['node size'] != _before_,
                      "pressing the row's key a second time did not cycle it")


def test_the_first_press_on_the_row_the_panel_opened_on_only_jumps(timing_page):
    """The panel opens on 'arrows'.  Its key, pressed once, must not flip it: the user
    has not asked that row for anything yet.  Twice is a cycle."""
    _open(timing_page)
    timing_page.assert_panel_row(0)
    timing_page.press('r')
    timing_page.assert_panel_value('arrows', 'off')
    timing_page.press('r')
    timing_page.assert_panel_value('arrows', 'on')


def test_a_rows_key_pressed_again_cycles_it_on_a_generic_view(histopi_page):
    """The generic views share the panel code with linkpi but have their own entry
    module and state object -- the tooltip row is on every one of them."""
    _open(histopi_page)
    _before_ = histopi_page.panel_values()['tooltip']
    histopi_page.press('i')
    assert histopi_page.panel_values()['tooltip'] == _before_
    histopi_page.press('i')
    assert_eventually(lambda: histopi_page.panel_values()['tooltip'] != _before_,
                      "pressing the tooltip row's key a second time did not cycle it")


def test_xypi_sets_each_axis_placement_and_bins_on_its_own(xypi_page):
    """User feedback 2026-09-27: time along a long x and a number up a short y need their
    own bins, and their own placement.  The y rows take the shifted keys (P / B)."""
    _open(xypi_page)
    for _ in range(4):                            # jump, then off -> x -> y -> x+y
        xypi_page.press('d')
    xypi_page.assert_panel_value('distributions (rows)', 'x+y')
    xypi_page.assert_panel_row_disabled('y placement', False)     # the gate comes back from Python
    xypi_page.press('P')
    xypi_page.press('P')
    xypi_page.assert_panel_value('y placement', 'inside')
    xypi_page.press('B')
    xypi_page.press('B')
    xypi_page.assert_panel_value('y bins', 'auto /4')
    assert xypi_page.panel_values()['x placement'] == 'auto'
    assert xypi_page.panel_values()['x bins'] == 'auto'
    _view_ = xypi_page.app.view()
    _inside_ = _view_.template.p2s.DISTRIBUTION_INSIDEp
    assert_eventually(lambda: _inside_ in (_view_._overrides_.get('y_distributions') or [])
                              and _inside_ not in (_view_._overrides_.get('x_distributions') or []),
                      "y's placement did not reach the render, or reached x too")


def test_moving_the_cursor_away_and_back_resets_the_key(timing_page):
    _open(timing_page)
    timing_page.press('r')                        # jump to arrows (already there)
    timing_page.press('a')                        # cursor away ...
    timing_page.press('A')                        # ... and back, by the cursor keys
    timing_page.assert_panel_row(0)
    timing_page.press('r')                        # a first press again: jump only
    timing_page.assert_panel_value('arrows', 'off')
    timing_page.press('r')
    timing_page.assert_panel_value('arrows', 'on')


def test_space_between_presses_keeps_the_row_selected_by_its_key(timing_page):
    _open(timing_page)
    timing_page.press('r')
    timing_page.press(' ')
    timing_page.assert_panel_value('arrows', 'on')
    timing_page.press('r')                        # still the key's row: cycles
    timing_page.assert_panel_value('arrows', 'off')


def test_a_cycled_value_reaches_python(timing_page):
    """The row is not a label: the LinkP has to actually change."""
    _open(timing_page)
    timing_page.press('r')
    timing_page.press(' ')
    timing_page.assert_panel_value('arrows', 'on')
    _view_ = timing_page.app.view()
    assert_eventually(lambda: all(_lp_.link_arrows for _lp_ in _view_.dfs_layout),
                      'the arrows row committed on screen but not onto the LinkP')


def test_the_two_timing_rows_are_independent(timing_page):
    """Splitting the old 'a' cycle in two is the point: that one flipped exactly one of
    arrows / marks per step, so a given combination cost up to three presses."""
    _open(timing_page)
    timing_page.press('r'); timing_page.press(' ')
    timing_page.assert_panel_value('arrows', 'on')
    timing_page.assert_panel_value('timing marks', 'on')      # untouched
    timing_page.press('t'); timing_page.press(' ')
    timing_page.assert_panel_value('timing marks', 'off')
    timing_page.assert_panel_value('arrows', 'on')            # still untouched


# ── CP7: the commit is debounced ─────────────────────────────────────────────

def test_cycling_past_a_value_does_not_commit_it(timing_page):
    """The objection that would have broken the feature.

    The pickers navigate without rendering and commit once on Enter; the panel renders
    live.  Cycling link shape line -> curve -> flowmap -> line would render **flowmap on
    the way past** -- a force layout whose cost grows faster than linearly (linkp.py's
    own warning), which on a netflow-scale graph hangs the view to reach the value you
    actually wanted.

    Three fast presses land back on 'line', and the assertion is that Python was never
    told about the two in between: link_shape_choice must never have read 'flowmap'.
    Watched from the server side because an intermediate value is precisely what the
    on-screen row is *supposed* to show.
    """
    _open(timing_page)
    _view_ = timing_page.app.view()
    _seen_ = []
    _view_.param.watch(lambda e: _seen_.append(e.new), 'link_shape_choice')

    timing_page.press('h')
    for _ in range(3):                            # line -> curve -> flowmap -> line
        timing_page.page.keyboard.press(' ')      # no idle wait between them
    timing_page.assert_panel_value('link shape', 'line')

    time.sleep(1.0)                               # well past the 300ms debounce
    assert 'flowmap' not in _seen_, (
        f'the debounce let an intermediate value through: {_seen_} -- cycling to "off" '
        f'would run the flowmap force layout on the way past')


def test_the_debounced_commit_does_eventually_land(timing_page):
    """The other half: debounced is not discarded."""
    _open(timing_page)
    _view_ = timing_page.app.view()
    timing_page.press('h')
    timing_page.press(' ')
    assert_eventually(lambda: _view_.link_shape_choice == 'curve',
                      lambda: f'link_shape_choice is still {_view_.link_shape_choice!r} '
                              f'-- the debounced commit never fired')


def test_escape_flushes_rather_than_discards(timing_page):
    """Closing inside the 300ms window is a normal thing to do.

    Dropping the choice there would read as the panel ignoring a keystroke.  This is not
    the walk-away commit CP6 rejects -- it takes an explicit esc.
    """
    _open(timing_page)
    _view_ = timing_page.app.view()
    timing_page.press('r')
    timing_page.page.keyboard.press(' ')
    timing_page.page.keyboard.press('Escape')     # inside the debounce window
    timing_page.assert_panel_closed()
    assert_eventually(lambda: _view_.link_arrows_choice == 'on',
                      'esc discarded a value that had already been cycled on screen')


# ── CP5: modality ────────────────────────────────────────────────────────────

def test_the_open_panel_swallows_ordinary_bindings(linkpi_page):
    """Modal for v1: the panel block returns before the dispatch chain, as the picker's
    does.  Non-modal is more useful but re-opens the keyspace conflict the panel exists
    to close."""
    _open(linkpi_page)
    linkpi_page.press('z')                        # would otherwise select a colour group
    assert linkpi_page.panel_is_open(), "'z' closed the panel"
    assert '0 Selected' in linkpi_page.info_text() or 'Selected' not in linkpi_page.info_text(), (
        "'z' reached the selection handler while the panel was open")


def test_the_open_panel_suppresses_browser_defaults(linkpi_page):
    """``space`` scrolls the page by default and is genuinely reclaimable -- unlike
    ctrl-t or ctrl-shift-w, which the page is never shown."""
    _open(linkpi_page)
    linkpi_page.clear_keydowns()
    linkpi_page.press(' ')
    assert linkpi_page.last_keydown()['defaultPrevented'] is True, (
        'space was not reclaimed -- it will scroll the page out from under the panel')


def test_the_panel_survives_a_render(timing_page):
    """It is meant to be left open while you work, so a redraw must not destroy it.

    U5 is the precedent: while ``info_str`` was a ReactiveHTML *child*, every write to it
    rebuilt the subtree and took every JS-only variable with it -- an open picker lost
    its index, the search buffer emptied, ``brush_state`` reset.  The panel is the same
    kind of state and it is up for far longer.
    """
    _open(timing_page)
    timing_page.press('r')
    timing_page.press(' ')                        # a real re-render: arrows redraw
    timing_page.assert_panel_value('arrows', 'on')
    timing_page.assert_panel_row(ROWS.index(('arrows', 'r')))
    assert timing_page.panel_is_open()


if __name__ == '__main__':
    unittest.main()
