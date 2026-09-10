"""The generic ``_interactivep`` components -- PLANNING.md 2.1 phase 2 item 7.

``xyp`` / ``timep`` / ``histop`` / ``chordp`` / ``piep`` share one keydown handler
built by ``_interactivep()``, specialised per kind by ``_INTERACTIVEP_CONFIG_``:
whether ``z``, the time keys and ``/`` search exist at all, and which brush shapes
``R`` cycles through.  That specialisation is the interesting part -- the same
keystroke is a different feature depending on which component has focus, and only a
browser can tell you which one you got.

Their interaction root is ``svgparent<kind>``, which is why ``InteractivePage`` takes
a ``root_id``.  Note these components do not *select*: a drag filters the dataframe
and pushes the stack, so the observable is the redrawn plot rather than a count.

This file also covers the brush's reason for existing -- the cross-component fan-out
-- which needs two components on a page and so could not be tested with the brush
itself in ``test_brush.py``.
"""
import unittest

import pytest
from playwright.sync_api import expect


def _brush_shape(ip):
    """Which primitive the brush cursor drew: circle, vertical or horizontal line."""
    _html_ = ip.el('brushindicator').inner_html() or ''
    if '<circle' in _html_:
        return 'circle'
    if '<line' in _html_:
        _el_ = ip.page.locator(f'[id="brushindicator{ip.suffix}"] line')
        _x1_, _x2_ = _el_.get_attribute('x1'), _el_.get_attribute('x2')
        return 'vertical' if _x1_ == _x2_ else 'horizontal'
    return 'none'


# ── bindings every generic component has ─────────────────────────────────────

@pytest.mark.parametrize('fixture', ['xypi_page', 'histopi_page', 'timepi_page'])
def test_h_toggles_the_help_overlay(request, fixture):
    _ip_ = request.getfixturevalue(fixture)
    assert _ip_.el('keyboardhelp').get_attribute('transform') == 'translate(-1000 0)'

    _ip_.hover(200, 150)
    _ip_.press('h')
    expect(_ip_.el('keyboardhelp')).to_have_attribute(
        'transform', 'translate(5 0)', timeout=_ip_.timeout_ms)


@pytest.mark.parametrize('fixture', ['xypi_page', 'histopi_page', 'timepi_page'])
def test_r_turns_the_brush_on_and_off(request, fixture):
    _ip_ = request.getfixturevalue(fixture)
    _ip_.settle()
    _ip_.hover(200, 150)

    _ip_.press('r')
    expect(_ip_.page.locator(f'[id="brushindicator{_ip_.suffix}"] circle')).to_have_count(
        1, timeout=_ip_.timeout_ms)

    _ip_.press('r')
    assert _brush_shape(_ip_) == 'none'


@pytest.mark.parametrize('fixture', ['xypi_page', 'histopi_page', 'timepi_page'])
def test_F_opens_the_selection_shape_picker(request, fixture):
    """Every generic component can switch its rubber band between rectangle and oval.

    LINKPI has no such menu -- this is one of the places the two handlers genuinely
    differ, rather than one being a subset of the other.
    """
    _ip_ = request.getfixturevalue(fixture)
    _ip_.settle()
    _ip_.hover(200, 150)
    _ip_.press('F')
    _ip_.expect_menu_open('rectangle')
    assert 'oval' in _ip_.menu_text()


# ── the per-kind brush sequences ─────────────────────────────────────────────

def test_the_xy_brush_cycles_through_every_shape(xypi_page):
    """xy is 2-D, so its sequence [0,1,2,3,4,5,6] reaches circles, verticals *and*
    horizontals."""
    xypi_page.settle()
    xypi_page.hover(200, 150)

    _seen_ = []
    for _ in range(6):
        xypi_page.press('R')
        _seen_.append(_brush_shape(xypi_page))
    assert _seen_ == ['circle', 'circle', 'vertical', 'vertical', 'horizontal', 'horizontal']


def test_the_time_brush_reaches_vertical_bands_only(timepi_page):
    """timep's x-axis is time, so a vertical band is a time window -- its sequence
    [0,1,2,3,4] stops there and never offers a horizontal one."""
    timepi_page.settle()
    timepi_page.hover(200, 150)

    _seen_ = []
    for _ in range(4):
        timepi_page.press('R')
        _seen_.append(_brush_shape(timepi_page))
    assert _seen_ == ['circle', 'circle', 'vertical', 'vertical']
    assert 'horizontal' not in _seen_


def test_the_histogram_brush_reaches_horizontal_bands_only(histopi_page):
    """histop's bars are horizontal, so its sequence [0,1,2,5,6] offers the horizontal
    band and skips the vertical one -- the mirror image of timep."""
    histopi_page.settle()
    histopi_page.hover(200, 150)

    _seen_ = []
    for _ in range(4):
        histopi_page.press('R')
        _seen_.append(_brush_shape(histopi_page))
    assert _seen_ == ['circle', 'circle', 'horizontal', 'horizontal']
    assert 'vertical' not in _seen_


def test_the_brush_wraps_within_its_own_sequence(timepi_page):
    """Past the end it returns to the first non-zero state, never to 'off'."""
    timepi_page.settle()
    timepi_page.hover(200, 150)
    for _ in range(5):
        timepi_page.press('R')
    assert _brush_shape(timepi_page) == 'circle'


# ── gated bindings: present on one component, absent on another ──────────────

def test_z_filters_by_colour_on_xy(xypi_page):
    """has_z_key is True only for xy; the binding filters the dataframe and pushes.

    Aimed at a mark that is actually drawn: over blank canvas the binding correctly
    does nothing, which would look identical to it not being wired up.
    """
    _x_, _y_ = xypi_page.first_mark_xy()
    xypi_page.hover(_x_, _y_)
    _before_ = xypi_page.mod_html()
    xypi_page.press('z')
    assert xypi_page.wait_for_mod_change(_before_) != _before_


def test_z_does_nothing_on_a_component_without_it(histopi_page):
    """has_z_key is False for histop, so the branch is not even emitted into the JS.

    Asserted as "the plot did not change", which is what a user would see -- and it is
    the reason a source-substring test is not enough: the branch's *absence* is as much
    a behaviour as its presence.
    """
    histopi_page.settle()
    histopi_page.hover(200, 150)
    _before_ = histopi_page.mod_html()
    histopi_page.press('z')
    histopi_page.wait_until_idle()
    assert histopi_page.mod_html() == _before_


def test_the_search_prompt_exists_where_search_is_enabled(histopi_page):
    """has_search adds #searchtext to the template."""
    assert histopi_page.el('searchtext').count() == 1


def test_the_search_prompt_is_absent_where_search_is_not(xypi_page):
    """xy has no search, so the element is not emitted at all.

    Two tests rather than one taking both fixtures: each component fixture navigates
    the shared `page`, so asking for two of them in one test leaves the first one's
    server serving a page nobody is looking at, and every assertion about it is really
    about the second.
    """
    assert xypi_page.el('searchtext').count() == 0


def test_slash_opens_search_on_the_histogram(histopi_page):
    histopi_page.settle()
    histopi_page.hover(200, 150)
    histopi_page.press('/')
    assert histopi_page.el('searchtext').text_content() == '/ ▋'


# ── the rubber band filters rather than selects ──────────────────────────────

def test_a_drag_filters_the_data_and_redraws(xypi_page):
    """Generic components push a filtered dataframe rather than marking a selection."""
    _before_ = xypi_page.mod_html()
    xypi_page.drag(20, 20, 200, 160)
    assert xypi_page.wait_for_mod_change(_before_) != _before_


def test_the_oval_shape_draws_an_ellipse_instead_of_a_rectangle(xypi_page):
    """Committing 'oval' in the F picker changes which band element the drag draws.

    #dragoval starts hidden; the rectangle is what a fresh component uses.
    """
    xypi_page.settle()
    xypi_page.hover(200, 150)
    xypi_page.press('F')
    xypi_page.expect_menu_open('rectangle')
    xypi_page.press('o')                       # mnemonic for 'oval'
    xypi_page.expect_menu_closed()

    xypi_page.mouse_down(200, 150)
    xypi_page.mouse_move_to(260, 200)
    _oval_ = xypi_page.el('dragoval')
    assert _oval_.get_attribute('display') != 'none', 'the oval band stayed hidden'
    assert (_oval_.get_attribute('rx'), _oval_.get_attribute('ry')) == ('60', '50')
    xypi_page.mouse_up()


# ── the brush's reason for existing: cross-component fan-out ─────────────────
#
# These assert at the MVC boundary as well as on the peer's SVG.  Until U7 was fixed
# only the former was possible: the peer *did* redraw but did not reliably stay
# redrawn, because brush results were applied in completion order rather than issue
# order and a stale "nothing here" clear routinely landed after a fresh update.  Each
# op now takes a ticket and only the newest may broadcast, so the peer's rendered state
# is stable enough to assert directly.


def _arm_brush(link, xy):
    """Turn the brush on and wait for its first result to reach the peer.

    Enabling the brush fires an op at wherever the pointer happens to be -- bare canvas
    here, so a clear -- and waiting for the peer to settle at its full, unbrushed count
    is the signal that op is done.  Since U7 a straggler can no longer overwrite a later
    result, but starting from a known state still makes the assertions below read
    unambiguously.

    Returns the peer's unbrushed mark count.
    """
    _full_ = xy.marks().count()
    link.hover(200, 150)
    link.press('r')
    xy.expect_marks(_full_)
    return _full_


def _spy_on_brush(view):
    """Record every brushUpdate / brushClear the controller broadcasts."""
    _calls_ = []
    _orig_update_, _orig_clear_ = view.mvc.brushUpdate, view.mvc.brushClear

    async def _update_(caller, df):
        _calls_.append(('update', len(df)))
        return await _orig_update_(caller, df)

    async def _clear_(caller):
        _calls_.append(('clear', 0))
        return await _orig_clear_(caller)

    view.mvc.brushUpdate, view.mvc.brushClear = _update_, _clear_
    return _calls_


def _await_call(calls, kind, budget_s=15.0):
    import time
    _deadline_ = time.monotonic() + budget_s
    while time.monotonic() < _deadline_:
        _hit_ = [_c_ for _c_ in calls if _c_[0] == kind]
        if _hit_:
            return _hit_[0]
        time.sleep(0.05)
    raise AssertionError(f'no {kind!r} broadcast within {budget_s}s; saw {calls}')


def test_brushing_broadcasts_the_records_under_the_pointer(linked_pair):
    """Moving with the brush on sends the records beneath it to every linked view.

    Node 'a' sits on two of the five rows, so the broadcast must carry exactly two --
    'some records' would pass on a brush that grabbed the whole dataframe.
    """
    _link_, _xy_ = linked_pair
    _link_.settle()
    _link_.hover(200, 150)
    _link_.press('r')

    _calls_ = _spy_on_brush(_link_.app.view())
    _link_.hover(*_link_.node_screen_xy('a'))
    assert _await_call(_calls_, 'update') == ('update', 2)


def test_leaving_the_component_broadcasts_a_clear(linked_pair):
    """brush_leave_done -> brushClear, so peers return to the unbrushed dataframe."""
    _link_, _xy_ = linked_pair
    _link_.settle()
    _link_.hover(200, 150)
    _link_.press('r')
    _link_.hover(*_link_.node_screen_xy('a'))

    _calls_ = _spy_on_brush(_link_.app.view())
    _box_ = _link_.root.bounding_box()
    _link_.page.mouse.move(_box_['x'] + _box_['width'] + 100,
                           _box_['y'] + _box_['height'] + 100)
    _await_call(_calls_, 'clear')


def test_nothing_is_broadcast_while_the_brush_is_off(linked_pair):
    """Without this, "a broadcast happened" could just mean the component broadcasts on
    any mouse movement, and the two tests above would prove nothing about the brush."""
    import time
    _link_, _xy_ = linked_pair
    _link_.settle()

    _calls_ = _spy_on_brush(_link_.app.view())
    _sx_, _sy_ = _link_.node_screen_xy('a')
    _link_.hover(_sx_, _sy_)
    _link_.hover(_sx_ + 20, _sy_ + 20)
    _link_.wait_until_idle()
    time.sleep(1.0)

    # Only 'update' is asserted absent.  A 'clear' can legitimately arrive without any
    # brushing: the render script resets data.brush_changed to 0 on a rebuild, which is
    # a change like any other, so the watcher fires and -- with brush_state 0 -- clears.
    # Harmless, and not what this test is about; broadcasting *records* is.
    _updates_ = [_c_ for _c_ in _calls_ if _c_[0] == 'update']
    assert _updates_ == [], f'the brush broadcast records while switched off: {_calls_}'


def test_a_stale_brush_result_cannot_overwrite_a_newer_one(linked_pair):
    """The U7 property: the newest pointer position wins, whatever finishes first.

    Switching the brush on fires an op at wherever the pointer already is -- bare
    canvas here, so its result is a clear -- and `_doBrushAt` runs outside the
    controller lock.  Moving onto a node immediately afterwards puts a second op in
    flight, and before the fix the slower first one landed last and wiped the update.
    Measured then: `update(2 records)` followed by `clear`, from a single move.

    Asserted on the *last* broadcast rather than by counting: both ops legitimately
    run, and which finishes first is a race.  What must not happen is the older one
    having the final word.
    """
    _link_, _xy_ = linked_pair
    _link_.settle()
    _link_.hover(200, 150)

    _calls_ = _spy_on_brush(_link_.app.view())
    _link_.press('r')                              # op 1: bare canvas -> a clear
    _link_.hover(*_link_.node_screen_xy('a'))      # op 2: on a node   -> an update
    _await_call(_calls_, 'update')

    import time
    time.sleep(1.0)                                # give any straggler time to land
    assert _calls_[-1][0] == 'update', (
        f'a stale brush result had the last word: {_calls_}')


def test_the_peer_stays_brushed_while_the_pointer_rests_on_a_node(linked_pair):
    """The DOM half, which only became assertable once U7 was fixed.

    Before, a peer assertion had to win the same race the component was losing, and
    three attempts at one failed between a third and two thirds of runs.
    """
    _link_, _xy_ = linked_pair
    _link_.settle()
    _full_ = _arm_brush(_link_, _xy_)

    _link_.hover(*_link_.node_screen_xy('a'))
    _xy_.expect_marks_to_change_from(_full_)

    import time
    time.sleep(1.0)                                # a stale clear would land by now
    assert _xy_.marks().count() != _full_, (
        'the peer reverted to the unbrushed view while the pointer sat on a node')


if __name__ == '__main__':
    unittest.main()
