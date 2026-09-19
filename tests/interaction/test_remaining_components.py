"""SMALLPI, SLPI and the stack control -- the last three interaction surfaces.

Each builds its own ReactiveHTML class with its own keydown handler, sharing nothing
with LINKPI or the `_interactivep` family, so none of them were reached by anything
already in this directory.  Between them they add three more `preventDefault` sites
(all on the stack control) to the nine LINKPI ones covered in `test_prevent_default.py`.

Their roots are `svgparentsmallpi`, `svgparentslpi` and `svgstackcontrol` -- which is
what `InteractivePage(root_id=...)` exists for.

The stack control is the odd one out and worth reading twice: it is an
*interactive-only* leaf with no static twin, it acts on **another component's** stack
rather than on itself, and its `c` / ctrl-shift-`c` bindings mutate that shared stack
in place.  So its tests need both halves of the pair and assert on the controller's
stack, which is where the effect actually is.
"""
import unittest

import pytest
from playwright.sync_api import expect


# ── SMALLPI ──────────────────────────────────────────────────────────────────

def test_smallpi_renders_its_tiles(smallpi_page):
    """The component reached the browser at all -- one tile per category."""
    expect(smallpi_page.root).to_be_visible()
    assert len(smallpi_page.mod_html()) > 0


def test_smallpi_selection_box_tracks_the_drag_and_clears_on_release(smallpi_page):
    """`#selbox` is the rubber band SMALLPI draws while a selection drag is in flight.

    It is browser-only state -- nothing about it crosses into Python -- and it had **no
    coverage at all** before the JSComponent port (PLANNING.md W1) rewrote the code that
    draws it.  The parity goldens cannot reach it either: they digest the DOM *after* a
    gesture completes, and `myOnMouseUp` has hidden the box again by then.  A mutant that
    never showed the box passed both the goldens and the whole suite.

    Asserted mid-drag, with the button still down, which is the only moment it exists.
    """
    _ip_ = smallpi_page
    _box_ = _ip_.el('selbox')

    assert _box_.get_attribute('display') == 'none', 'the band is visible before any drag'

    _ip_.mouse_down(80, 80)
    _ip_.mouse_move_to(200, 170)
    try:
        assert _box_.get_attribute('display') == 'block', 'the band never appeared during the drag'
        # Geometry is min/abs of the two corners, so it is orientation-independent.
        assert int(float(_box_.get_attribute('x'))) == 80
        assert int(float(_box_.get_attribute('y'))) == 80
        assert int(float(_box_.get_attribute('width'))) == 120
        assert int(float(_box_.get_attribute('height'))) == 90
    finally:
        _ip_.mouse_up()

    _wait_for(lambda: _box_.get_attribute('display') == 'none',
              'the band outlived the drag that drew it')


def test_smallpi_selection_box_is_drawn_for_an_upward_drag(smallpi_page):
    """The min/abs geometry, exercised in the direction that would break a naive
    implementation writing (x0, y0, x1-x0, y1-y0) straight through."""
    _ip_ = smallpi_page
    _box_ = _ip_.el('selbox')

    _ip_.mouse_down(220, 190)
    _ip_.mouse_move_to(100, 90)
    try:
        assert _box_.get_attribute('display') == 'block'
        assert int(float(_box_.get_attribute('x'))) == 100
        assert int(float(_box_.get_attribute('y'))) == 90
        assert int(float(_box_.get_attribute('width'))) == 120
        assert int(float(_box_.get_attribute('height'))) == 100
    finally:
        _ip_.mouse_up()


def test_smallpi_r_toggles_the_brush_flag(smallpi_page):
    """SMALLPI's brush is a plain boolean, not the radius cycle LINKPI has.

    It has no cursor to draw, so the flag on the Python side is the only observable --
    and it is a `data.*` param the JS writes, which is exactly the crossing this suite
    exists to exercise.
    """
    _view_ = smallpi_page.app.view()
    assert _view_.brush_on is False

    smallpi_page.hover(100, 100)
    smallpi_page.press('r')
    _wait_for(lambda: _view_.brush_on is True, 'r did not switch the brush on')

    smallpi_page.hover(100, 100)
    smallpi_page.press('r')
    _wait_for(lambda: _view_.brush_on is False, 'r did not switch the brush off')


def test_smallpi_only_announces_the_brush_going_off(smallpi_page):
    """Switching the brush *off* also sends 'brush_off' so peers can be un-brushed;
    switching it on sends nothing, because there is nothing to broadcast until the
    pointer moves.  That asymmetry is the behaviour worth pinning."""
    _view_ = smallpi_page.app.view()
    _seen_ = _record_param(_view_, 'key_op_finished')

    smallpi_page.hover(100, 100)
    smallpi_page.press('r')                      # on
    _wait_for(lambda: _view_.brush_on is True, 'the brush never came on')
    assert 'brush_off' not in _seen_, 'switching the brush on announced brush_off'

    smallpi_page.hover(100, 100)
    smallpi_page.press('r')                      # off
    _wait_for(lambda: 'brush_off' in _seen_,
              f'switching the brush off announced nothing (saw {_seen_})')


@pytest.mark.parametrize('key', ['q', 'Q'])
def test_smallpi_q_reaches_python_as_itself(smallpi_page, key):
    """'q' subtracts the current frame from the top of the stack; 'Q' is its shifted
    twin, and the handler tells them apart by shiftKey as well as by character.

    Watched rather than read after the fact: ``applyKeyOp`` clears
    ``key_op_finished`` as soon as it has consumed it, so sampling the param later
    only ever sees the empty string -- which is how the first draft of this test
    managed to assert nothing at all.
    """
    _seen_ = _record_param(smallpi_page.app.view(), 'key_op_finished')
    smallpi_page.hover(100, 100)
    smallpi_page.press(key)
    _wait_for(lambda: key in _seen_,
              f'{key!r} never arrived as key_op_finished (saw {_seen_})')


# ── SLPI (spreadlinesp) ──────────────────────────────────────────────────────

def test_slpi_renders(slpi_page):
    expect(slpi_page.root).to_be_visible()
    assert len(slpi_page.mod_html()) > 0


@pytest.mark.parametrize('key', ['x', 'X', 'c'])
def test_slpi_bindings_reach_python_as_themselves(slpi_page, key):
    """SLPI has exactly three bindings and its handler is a single line of JS.

    Thin on purpose: the point is that all three cross the websocket carrying the
    right character -- lower and upper `x` are different operations, so a handler that
    normalised case would be caught here and nowhere else.
    """
    _seen_ = _record_param(slpi_page.app.view(), 'key_op_finished')
    slpi_page.hover(300, 150)
    slpi_page.press(key)
    _wait_for(lambda: key in _seen_,
              f'{key!r} never arrived as key_op_finished (saw {_seen_})')


def _slpi_band(ip):
    """`#drag_rect`'s geometry and stroke.

    Not `InteractivePage.drag_rect()` -- that helper reads `#drag`, which is LINKPI's
    band.  SLPI draws its own into a differently-named element, which is why nothing
    in this suite had ever looked at it.
    """
    _el_ = ip.el('drag_rect')
    return {_a_: (_el_.get_attribute(_a_) or '') for _a_ in
            ('x', 'y', 'width', 'height', 'stroke')}


def test_slpi_drag_rect_tracks_the_drag_and_clears_on_release(slpi_page):
    """SLPI's rubber band had no coverage before the JSComponent port (PLANNING.md W1)
    rewrote the code that draws it -- the same gap `#selbox` had on SMALLPI.

    The parity goldens cannot reach it: they digest the DOM after a gesture, and
    `myOnMouseUp` has already collapsed the band to 0x0 by then.
    """
    _ip_ = slpi_page
    _ip_.mouse_down(150, 90)
    _ip_.mouse_move_to(330, 210)
    try:
        _band_ = _slpi_band(_ip_)
        assert (_band_['x'], _band_['y']) == ('150', '90')
        assert (_band_['width'], _band_['height']) == ('180', '120')
    finally:
        _ip_.mouse_up()

    _wait_for(lambda: _slpi_band(_ip_)['width'] == '0',
              'the band outlived the drag that drew it')


def test_slpi_drag_rect_normalises_a_backwards_drag(slpi_page):
    """min/abs geometry, in the direction a naive (x0, y0, x1-x0, y1-y0) breaks on."""
    _ip_ = slpi_page
    _ip_.mouse_down(330, 210)
    _ip_.mouse_move_to(150, 90)
    try:
        _band_ = _slpi_band(_ip_)
        assert (_band_['x'], _band_['y']) == ('150', '90')
        assert (_band_['width'], _band_['height']) == ('180', '120')
    finally:
        _ip_.mouse_up()


@pytest.mark.parametrize('shift,ctrl,stroke', [(False, False, '#000000'),
                                               (True,  False, '#ff0000'),
                                               (False, True,  '#00ff00'),
                                               (True,  True,  '#0000ff')])
def test_slpi_drag_rect_stroke_encodes_the_set_operation(slpi_page, shift, ctrl, stroke):
    """The colour is the only feedback for which set-operation the drag will perform.

    **No focus dance here, unlike LINKPI.** LINKPI's band colours come from
    `data.ctrlkey`, which only its `myOnKeyDown` sets, so a modifier pressed while focus
    is elsewhere never reaches the band (see `test_mouse_gestures.py`, and PLANNING.md U8
    for the two colours that were dead code because of a typo in that path). SLPI reads
    `event.shiftKey` / `event.ctrlKey` straight off the mouse event in `myOnMouseDown`
    and `myOnMouseMove`, so the colour is correct without focus ever being taken. That
    difference is behaviour the port had to preserve, and it is asserted rather than
    assumed by deliberately *not* hovering first.
    """
    _ip_ = slpi_page
    with _ip_.holding(shift=shift, ctrl=ctrl):
        _ip_.mouse_down(150, 90)
        _ip_.mouse_move_to(330, 210)
        try:
            assert _slpi_band(_ip_)['stroke'] == stroke
        finally:
            _ip_.mouse_up()


# ── the stack control ────────────────────────────────────────────────────────

def test_stack_control_renders_beside_its_component(stack_control_page):
    _xy_, _ct_ = stack_control_page
    expect(_xy_.root).to_be_visible()
    expect(_ct_.root).to_be_visible()


def test_stack_control_h_toggles_the_help_overlay(stack_control_page):
    """A third `display` state machine, independent of LINKPI's keyboardhelp_x."""
    _xy_, _ct_ = stack_control_page
    _ct_.hover(80, 170)
    _ct_.press('h')

    _sc_ = _stack_control_view(_ct_)
    _wait_for(lambda: _sc_.help_display == 'inline', 'h did not open the help overlay')

    _ct_.hover(80, 170)
    _ct_.press('h')
    _wait_for(lambda: _sc_.help_display == 'none', 'h did not close the help overlay')


@pytest.mark.parametrize('key,ctrl,shift', [('h', False, False),
                                            ('c', False, False),
                                            ('C', True,  True)])
def test_stack_control_bindings_suppress_the_browser_default(stack_control_page,
                                                             key, ctrl, shift):
    """Three more `preventDefault` sites, none of them covered before.

    ctrl-shift-c in particular would otherwise be the browser's own shortcut; the
    other two are guarded so the key cannot also scroll or type into the page behind.
    """
    _xy_, _ct_ = stack_control_page
    _ct_.hover(80, 170)
    _ct_.clear_keydowns()
    if ctrl or shift:
        with _ct_.holding(ctrl=ctrl, shift=shift):
            _ct_.press(key)
    else:
        _ct_.press(key)

    _ev_ = _ct_.last_keydown()
    assert _ev_['defaultPrevented'] is True, (
        f'the stack control let {key!r} through to the browser')


def test_stack_control_c_collapses_the_stack(stack_control_page):
    """'c' keeps the base frame and the current one, dropping everything between.

    Asserted on the shared stack rather than on the control's own render, because that
    is where the operation lands -- the control is a view of someone else's state.
    """
    _xy_, _ct_ = stack_control_page
    _mvc_ = _ct_.app.container.mvc
    _stack_ = _mvc_.stacks['default']
    _stack_['dfs'] = list(_stack_['dfs']) + [_stack_['dfs'][0], _stack_['dfs'][0]]
    _stack_['index'] = len(_stack_['dfs']) - 1
    assert len(_stack_['dfs']) >= 3

    _ct_.hover(80, 170)
    _ct_.press('c')
    _wait_for(lambda: len(_mvc_.stacks['default']['dfs']) <= 2,
              'c did not collapse the stack')


# ── helpers ──────────────────────────────────────────────────────────────────

def _stack_control_view(ct_page):
    """The STACKCONTROLI instance out of the mvc's view registry."""
    for _v_ in ct_page.app.container.mvc.view_refs.values():
        if type(_v_).__name__ == 'STACKCONTROLI':
            return _v_
    raise LookupError('no STACKCONTROLI registered with the controller')


def _record_param(view, name):
    """Every value a param takes, in order.

    The trigger params are cleared by their own handlers the moment they are consumed,
    so anything that samples them afterwards sees only the reset value.  A watcher
    catches the transient one, which is the thing the browser actually sent.
    """
    _seen_ = []
    view.param.watch(lambda _ev_: _seen_.append(_ev_.new), name)
    return _seen_


def _wait_for(predicate, message, budget_s=15.0):
    """Bounded poll on Python-side state.

    These components put their results on params and shared stacks rather than into
    the SVG, so there is no locator to hand to expect(); this is the same bounded-poll
    idea against a different surface.
    """
    import time
    _deadline_ = time.monotonic() + budget_s
    while time.monotonic() < _deadline_:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError(message)


if __name__ == '__main__':
    unittest.main()
