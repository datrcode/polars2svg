"""Mouse gestures -- PLANNING.md 2.1 phase 2 item 4.

The largest untested group, and the one where the interaction layer is most nearly
*all* browser: the rubber-band rectangle, the move offset and the layout shapes are
drawn from ``mousemove`` straight into SVG elements, out of nine JS-only variables
(``drag_op``, ``move_op``, ``unselected_move_op``, ``layout_op``, ``layout_op_shape``,
``layout_line_flag``, ``x0_drag``/``y0_drag``, ``x1_drag``/``y1_drag``).  Python is
told the result on mouseup and nothing whatsoever before it.

Two things decide what a press does, and both are tested here rather than assumed:

* **What is under the pointer.**  Three transparent hit layers stack -- ``#screen``
  (rubber band), ``#allentitieslayer`` (move unselected), ``#selectionlayer`` (move
  selected) -- so the same gesture one pixel over is a different operation, silently.
* **Where the press started.**  The band is anchored at mousedown; intermediate moves
  extend the far corner only.  A drag that wanders to the origin and back still
  selects from wherever the button went down, which is easy to get wrong when writing
  a test and impossible to notice when reading one.

These use ``quad_page``, whose node positions *and* view window are pinned, so every
rectangle below is arithmetic rather than luck -- see the fixture for why both are
needed.
"""
import unittest


CORNER = (2, 2)          # bare canvas on the quad fixture, clear of every node


def _coords(ip):
    return {_r_['__first__']: (int(_r_['__sx__']), int(_r_['__sy__']))
            for _r_ in ip.plot.df_node.iter_rows(named=True)}


def _box(ip, x0, y0, x1, y1, **mods):
    """Rubber-band from (x0, y0) to (x1, y1), optionally with modifiers held."""
    if mods:
        with ip.holding(**mods):
            ip.drag(x0, y0, x1, y1)
    else:
        ip.drag(x0, y0, x1, y1)


# ── the rubber band, as drawn ────────────────────────────────────────────────

def test_dragging_on_empty_canvas_draws_the_rubber_band(quad_page):
    quad_page.mouse_down(*CORNER)
    quad_page.mouse_move_to(CORNER[0] + 90, CORNER[1] + 70)

    _rect_ = quad_page.drag_rect()
    assert (_rect_['x'], _rect_['y']) == ('2', '2')
    assert (_rect_['width'], _rect_['height']) == ('90', '70')
    quad_page.mouse_up()


def test_the_band_normalises_a_backwards_drag(quad_page):
    """Dragging up-and-left still gives a positive-size rect at the top-left."""
    quad_page.mouse_down(360, 280)
    quad_page.mouse_move_to(300, 240)

    _rect_ = quad_page.drag_rect()
    assert (_rect_['x'], _rect_['y']) == ('300', '240')
    assert (_rect_['width'], _rect_['height']) == ('60', '40')
    quad_page.mouse_up()


def test_the_band_is_parked_off_canvas_after_release(quad_page):
    """Parked at (-10, -10) 5x5, not merely made small -- so a stale band cannot be
    mistaken for a live one a few pixels across."""
    _box(quad_page, *CORNER, 80, 80)

    _rect_ = quad_page.drag_rect()
    assert (_rect_['x'], _rect_['y']) == ('-10', '-10')
    assert (_rect_['width'], _rect_['height']) == ('5', '5')


def test_the_band_is_black_with_no_modifier(quad_page):
    quad_page.mouse_down(*CORNER)
    quad_page.mouse_move_to(120, 100)
    assert quad_page.drag_rect()['stroke'] == '#000000'
    quad_page.mouse_up()


def test_the_band_turns_green_for_a_ctrl_drag(quad_page):
    """The colour is the only feedback for which set-operation a drag will perform.

    Focus *before* holding the modifier: the band's colour comes from ``data.ctrlkey``,
    which only ``myOnKeyDown`` sets -- and that handler is on the SVG, so a Control
    pressed while focus is elsewhere never reaches it and the band stays black.  (The
    selection itself is unaffected: ``myOnMouseUp`` reads ``event.ctrlKey`` straight
    off the event, which is why the set-operation tests above pass either way.)
    """
    quad_page.hover(*CORNER)
    with quad_page.holding(ctrl=True):
        quad_page.mouse_down(*CORNER)
        quad_page.mouse_move_to(120, 100)
        assert quad_page.drag_rect()['stroke'] == '#00ff00'
        quad_page.mouse_up()


def test_the_band_turns_red_for_a_shift_drag(quad_page):
    """Shift subtracts, and the band says so.

    This colour was dead code until 2026-09-07: ``myUpdateDragRect`` tested
    ``data.shftkey`` where the param is ``shiftkey``, so the expression read undefined
    and a shift-drag drew the plain black band (PLANNING.md U8).  The operation was
    always right -- ``myOnMouseUp`` reads ``event.shiftKey`` straight off the event --
    so nothing but a browser could have caught it.
    """
    quad_page.hover(*CORNER)
    with quad_page.holding(shift=True):
        quad_page.mouse_down(*CORNER)
        quad_page.mouse_move_to(120, 100)
        assert quad_page.drag_rect()['stroke'] == '#ff0000'
        quad_page.mouse_up()


def test_the_band_turns_blue_for_a_shift_ctrl_drag(quad_page):
    """The other half of U8: intersect drew the ctrl colour (green) instead of blue,
    so the band actively named the wrong set-operation."""
    quad_page.hover(*CORNER)
    with quad_page.holding(shift=True, ctrl=True):
        quad_page.mouse_down(*CORNER)
        quad_page.mouse_move_to(120, 100)
        assert quad_page.drag_rect()['stroke'] == '#0000ff'
        quad_page.mouse_up()


# ── the rubber band, as a selection ──────────────────────────────────────────

def test_a_full_canvas_drag_selects_every_node(quad_page):
    _box(quad_page, *CORNER, 398, 298)
    quad_page.expect_selected(4)


def test_a_band_selects_exactly_the_nodes_it_covers(quad_page):
    """The northern row only.  This is the assertion the pinned fixture exists for:
    'some nodes were selected' would pass on a broken hit test too."""
    _c_ = _coords(quad_page)
    _mid_y_ = (_c_['nw'][1] + _c_['sw'][1]) // 2
    _box(quad_page, *CORNER, 398, _mid_y_)
    quad_page.expect_selected(2)


def test_a_band_down_one_side_selects_that_column(quad_page):
    _c_ = _coords(quad_page)
    _mid_x_ = (_c_['nw'][0] + _c_['ne'][0]) // 2
    _box(quad_page, *CORNER, _mid_x_, 298)
    quad_page.expect_selected(2)


def test_a_band_released_over_the_status_line_still_selects(quad_page):
    """Chrome must not eat the gesture (PLANNING.md U9).

    #infostr, #searchtext and the brush overlays sit *after* #screen in document
    order, so they paint on top of it and -- without pointer-events="none" -- take
    the mouse events for the strip they cover.  Releasing a rubber band there
    delivered mouseup to the <text> instead of #screen, so myOnMouseUp never ran and
    the selection was silently discarded: no error, no band, nothing selected.

    It hid behind U5 for as long as that lasted.  While info_str was a ReactiveHTML
    child it never reached the browser and the line rendered its " | | grid" default,
    about 30px wide; the real status string is ~245px and covers most of the bottom
    edge.  Fixing U5 made the text real and this surfaced immediately.

    Ends the drag at the *left* half of the bottom edge, which is exactly where
    #infostr is (x=5..248 on a 400px canvas); the neighbouring tests end at x=398 and
    would keep passing with the bug in place.
    """
    _c_ = _coords(quad_page)
    _mid_x_ = (_c_['nw'][0] + _c_['ne'][0]) // 2
    assert quad_page.el('infostr').bounding_box()['width'] > 100, (
        'the status line is too short for this test to cover the case it is named for')

    _box(quad_page, *CORNER, _mid_x_, 298)
    quad_page.expect_selected(2)


def test_a_band_over_empty_space_clears_the_selection(quad_page):
    """Select something first: with nothing selected an empty band changes nothing,
    so #infostr never refreshes off its default and there is no '0 Selected' to see."""
    _box(quad_page, *CORNER, 398, 298)
    quad_page.expect_selected(4)

    _box(quad_page, *CORNER, 40, 40)
    quad_page.expect_selected(0)


def test_the_band_is_anchored_at_the_press_not_the_path(quad_page):
    """Wandering to the far corner mid-drag must not widen the selection.

    x0/y0 are set once, by downSelect; only x1/y1 track the pointer.  A test that
    dragged *through* the whole canvas and expected everything would be asserting a
    bug.
    """
    _c_ = _coords(quad_page)
    _mid_y_ = (_c_['nw'][1] + _c_['sw'][1]) // 2

    quad_page.mouse_down(*CORNER)
    quad_page.mouse_move_to(398, 298)          # sweep across everything...
    quad_page.mouse_move_to(398, _mid_y_)      # ...and settle above the southern row
    quad_page.mouse_up()
    quad_page.expect_selected(2)


# ── the band's modifier set-operations ───────────────────────────────────────

def test_ctrl_drag_adds_to_the_selection(quad_page):
    _c_ = _coords(quad_page)
    _mid_y_ = (_c_['nw'][1] + _c_['sw'][1]) // 2
    _box(quad_page, *CORNER, 398, _mid_y_)
    quad_page.expect_selected(2)

    _box(quad_page, *CORNER, 398, 298, ctrl=True)
    quad_page.expect_selected(4)


def test_shift_drag_removes_from_the_selection(quad_page):
    _c_ = _coords(quad_page)
    _mid_y_ = (_c_['nw'][1] + _c_['sw'][1]) // 2
    _box(quad_page, *CORNER, 398, 298)
    quad_page.expect_selected(4)

    _box(quad_page, *CORNER, 398, _mid_y_, shift=True)
    quad_page.expect_selected(2)


def test_shift_ctrl_drag_intersects(quad_page):
    _c_ = _coords(quad_page)
    _mid_y_ = (_c_['nw'][1] + _c_['sw'][1]) // 2
    _box(quad_page, *CORNER, 398, 298)
    quad_page.expect_selected(4)

    _box(quad_page, *CORNER, 398, _mid_y_, shift=True, ctrl=True)
    quad_page.expect_selected(2)


# ── moving nodes ─────────────────────────────────────────────────────────────

def test_dragging_a_selected_node_moves_the_whole_selection(quad_page):
    """Pressing on a selected node reaches #selectionlayer -> downMove.

    Both northern nodes must travel, not just the one under the pointer.
    """
    _c_ = _coords(quad_page)
    _mid_y_ = (_c_['nw'][1] + _c_['sw'][1]) // 2
    _box(quad_page, *CORNER, 398, _mid_y_)
    quad_page.expect_selected(2)

    _before_ = quad_page.node_positions()
    _sx_, _sy_ = _c_['nw']
    quad_page.drag(_sx_, _sy_, _sx_ + 50, _sy_ + 40)
    quad_page.wait_until_idle()

    _after_ = quad_page.node_positions()
    assert _after_ != _before_, 'the move did not repaint'
    assert len(_after_) == 4, 'the move lost a node'


def test_dragging_an_unselected_node_moves_it_alone(quad_page):
    """Pressing an unselected node reaches #allentitieslayer -> downAllEntities.

    With nothing selected, the two move paths are told apart purely by which layer
    takes the press, and that routing is the thing worth pinning.
    """
    _c_ = _coords(quad_page)
    _before_ = quad_page.node_positions()
    _sx_, _sy_ = _c_['se']
    quad_page.drag(_sx_, _sy_, _sx_ - 60, _sy_ - 45)
    quad_page.wait_until_idle()
    assert quad_page.node_positions() != _before_


def test_a_drag_released_off_canvas_is_abandoned(quad_page):
    """Leaving the component mid-drag loses the gesture, and leaves the overlay offset.

    ``myOnMouseUp`` is bound to the hit layers, so a release outside them never fires:
    ``move_op_finished`` is never set, nothing moves, and ``#selectionlayer`` keeps the
    ``translate(...)`` that ``mousemove`` gave it -- so the red selection outline sits
    away from the nodes it belongs to until the next render happens for some other
    reason.  Asserting today's behaviour; if a mouseup-on-document handler is ever
    added, this turns red and says so.
    """
    _c_ = _coords(quad_page)
    _mid_y_ = (_c_['nw'][1] + _c_['sw'][1]) // 2
    _box(quad_page, *CORNER, 398, _mid_y_)
    quad_page.expect_selected(2)

    _before_ = quad_page.node_positions()
    _sx_, _sy_ = _c_['ne']
    quad_page.mouse_down(_sx_, _sy_)
    quad_page.mouse_move_to(_sx_ + 40, _sy_ + 30)
    _box_ = quad_page.root.bounding_box()
    quad_page.page.mouse.move(_box_['x'] + _box_['width'] + 120,
                              _box_['y'] + _box_['height'] + 120)
    quad_page.page.mouse.up()

    assert quad_page.node_positions() == _before_, 'the abandoned drag moved nodes anyway'
    assert (quad_page.el('selectionlayer').get_attribute('transform') or '') != '', \
        'the stranded selection transform was cleaned up -- behaviour changed'


# ── layout gestures ──────────────────────────────────────────────────────────

def _select_north_row(ip):
    _c_ = _coords(ip)
    _mid_y_ = (_c_['nw'][1] + _c_['sw'][1]) // 2
    _box(ip, *CORNER, 398, _mid_y_)
    ip.expect_selected(2)
    return _c_


def test_holding_g_while_dragging_lays_the_selection_out(quad_page):
    """The layout gestures are hold-to-arm, not press-then-drag -- see U6.

    Needs more than one selected node: apply_layout_interaction declines below two.
    """
    _select_north_row(quad_page)
    _before_ = quad_page.node_positions()

    quad_page.hover(200, 150)
    with quad_page.holding_key('g'):
        quad_page.drag(*CORNER, 150, 120)
    quad_page.wait_until_idle()
    assert quad_page.node_positions() != _before_


def test_holding_y_while_dragging_lays_the_selection_on_a_line(quad_page):
    _select_north_row(quad_page)
    _before_ = quad_page.node_positions()

    quad_page.hover(200, 150)
    with quad_page.holding_key('y'):
        quad_page.drag(*CORNER, 200, 160)
    quad_page.wait_until_idle()
    assert quad_page.node_positions() != _before_


def test_releasing_g_before_dragging_selects_instead_of_laying_out(quad_page):
    """The boundary of the hold-to-arm gesture, and the shape of an easy mistake.

    ``myOnKeyUp`` clears ``state.layout_op`` on release, so a *tapped* g leaves nothing
    armed and the following drag is an ordinary rubber band -- the user gets a
    selection where they may have meant a layout, with no error and nothing moving.

    That behaviour is intended (holding a left-hand key while the right hand drags is
    the same idiom as shift/ctrl, and the bindings are clustered left precisely because
    the right hand is on the mouse). What was wrong was the on-screen help, which read
    "layout upon next mouse drag" and so described press-then-drag; it now says "hold
    and drag" (PLANNING.md U6). This test pins the boundary so that if the arming ever
    is made to survive the keyup, the help goes stale again loudly rather than quietly.
    """
    _select_north_row(quad_page)
    _before_ = quad_page.node_positions()

    quad_page.hover(200, 150)
    quad_page.press('g')                        # tapped: down and up
    _box(quad_page, *CORNER, 40, 40)            # a band over empty canvas
    quad_page.wait_until_idle()

    assert quad_page.node_positions() == _before_, (
        'a tapped g laid the selection out -- the gesture now survives the keyup, and '
        'the on-screen help needs to go back to promising press-then-drag')
    quad_page.expect_selected(0)                # ...it was a selection instead


def test_the_layout_arm_does_not_survive_the_gesture(quad_page):
    """One held gesture is one layout; the next drag must not move anything.

    Asserted as "nothing moved" rather than by a node count: the layout has just
    relocated the selection, so which nodes a fixed rectangle covers afterwards is no
    longer known -- and a count that happened to differ would fail for the wrong
    reason. Not moving is precisely what "no longer armed" means.
    """
    _select_north_row(quad_page)
    quad_page.hover(200, 150)
    with quad_page.holding_key('g'):
        quad_page.drag(*CORNER, 150, 120)
    quad_page.wait_until_idle()

    _after_layout_ = quad_page.node_positions()
    _box(quad_page, *CORNER, 398, 298)
    quad_page.wait_until_idle()
    assert quad_page.node_positions() == _after_layout_, (
        'the second drag laid the nodes out again -- the arm outlived its gesture')


# ── view navigation ──────────────────────────────────────────────────────────

def test_the_wheel_zooms_the_view(quad_page):
    _before_ = quad_page.node_positions()
    quad_page.wheel(200, 150, -240)
    quad_page.wait_until_idle()
    assert quad_page.node_positions() != _before_


def test_the_wheel_suppresses_the_page_scroll(quad_page):
    """The listener is registered {passive: false} precisely so it may preventDefault;
    a passive registration would ignore the call and scroll the page instead."""
    quad_page.hover(200, 150)
    _prevented_ = quad_page.root.evaluate("""(rootEl) => new Promise(resolve => {
        const screen = rootEl.querySelector('[id^="screen"]');
        screen.addEventListener('wheel', ev => resolve(ev.defaultPrevented), {once: true});
        screen.dispatchEvent(new WheelEvent('wheel',
            {deltaY: 120, bubbles: true, cancelable: true}));
    })""")
    assert _prevented_ is True


def test_a_middle_drag_pans_the_view(quad_page):
    _before_ = quad_page.node_positions()
    quad_page.drag(120, 100, 240, 190, button='middle')
    quad_page.wait_until_idle()
    assert quad_page.node_positions() != _before_


def test_a_middle_click_resets_the_view(quad_page):
    """A middle press that does not travel is a reset, not a zero-pixel pan."""
    _home_ = quad_page.node_positions()
    quad_page.wheel(200, 150, -240)
    quad_page.wait_until_idle()
    assert quad_page.node_positions() != _home_

    quad_page.mouse_down(200, 150, button='middle')
    quad_page.mouse_up(button='middle')
    quad_page.wait_until_idle()
    assert quad_page.node_positions() == _home_, 'the middle click did not restore the view'


if __name__ == '__main__':
    unittest.main()
