"""F1 -- the tooltip overlay and its configuration-panel row, in a real browser.

**These tests had to be written rather than inherited.** W1's phases 2-4 established the
lesson and the config panel repeated it: a JS-only overlay is structurally invisible to
the parity goldens, because it exists only mid-gesture and the digest is taken after.
``#selbox`` and ``#drag_rect`` went uncovered for exactly that reason.

A tooltip is that shape twice over. It exists only while the pointer rests, *and* the
mode that produces it is panel state that never crosses into Python until it is
committed. So the drawn overlay is the observable, as ``#pickermenu`` is for the pickers.

The Python half -- hit test, text content, the icon render, the cache, the U7 ticket --
is in ``tests/test_tooltip.py``, where it needs no browser. What is here is what only a
browser can say: that the dwell fires, that the box is drawn and placed, that the mode
cycles from the panel, and that it clears when it should.
"""
import time
import unittest

import pytest


def _set_mode(ip, mode):
    """Cycle the panel's tooltip row to `mode` and close the panel.

    Through the panel and not by writing the param, because the row IS the interface --
    a param write would test a path the user has no way to take.

    The hover is not incidental: keys go to the focused root, and focus arrives through
    `myOnMouseOver`.  Blank canvas rather than a mark, so that arming the mode cannot
    itself be what puts a tooltip on screen.
    """
    ip.hover(*ip.blank_canvas_xy())
    ip.press('a')
    ip.expect_panel_open()
    ip.press('i')                       # the tooltip row's mnemonic, wherever it sits
    _deadline_ = time.monotonic() + 10.0
    while time.monotonic() < _deadline_:
        if ip.panel_values().get('tooltip') == mode:
            ip.press('Escape')
            ip.expect_panel_closed()
            return
        ip.press(' ')
        time.sleep(0.05)
    raise AssertionError(f'the tooltip row never reached {mode!r} '
                         f'(shows {ip.panel_values().get("tooltip")!r})')


# ── the row is reachable, on every component ────────────────────────────────

def test_the_panel_opens_on_a_generic_component(xypi_page):
    """The five generic views had no configuration panel before F1 -- 'a' was unbound.

    This is the whole reason the panel moved out of p2s_linkpi.js and into a shared
    fragment, so it is worth asserting rather than assuming.
    """
    xypi_page.hover(*xypi_page.blank_canvas_xy())   # focus
    xypi_page.press('a')
    xypi_page.expect_panel_open()
    assert 'tooltip' in xypi_page.panel_values()


@pytest.mark.parametrize('fixture_name', ['xypi_page', 'histopi_page', 'timepi_page'])
def test_every_generic_kind_gets_the_row(request, fixture_name):
    _ip_ = request.getfixturevalue(fixture_name)
    _ip_.hover(*_ip_.blank_canvas_xy())
    _ip_.press('a')
    _ip_.expect_panel_open()
    assert 'tooltip' in _ip_.panel_values(), _ip_.panel_text()


def test_the_row_starts_off(xypi_page):
    xypi_page.hover(*xypi_page.blank_canvas_xy())
    xypi_page.press('a')
    xypi_page.expect_panel_value('tooltip', 'off')


def test_without_an_icon_the_row_cycles_two_states(xypi_page):
    """off -> text -> off.  The third state is not offered by a view that has no icon,
    which is the row's value LIST varying rather than the row being gated."""
    xypi_page.hover(*xypi_page.blank_canvas_xy())
    xypi_page.press('a')
    xypi_page.press('i')                   # the tooltip row's mnemonic
    xypi_page.expect_panel_value('tooltip', 'off')
    xypi_page.press(' ')
    xypi_page.expect_panel_value('tooltip', 'text')
    xypi_page.press(' ')
    xypi_page.expect_panel_value('tooltip', 'off')


def test_with_an_icon_the_row_cycles_three_states(tooltip_page):
    tooltip_page.hover(*tooltip_page.blank_canvas_xy())
    tooltip_page.press('a')
    tooltip_page.press('i')                   # the tooltip row's mnemonic
    tooltip_page.expect_panel_value('tooltip', 'off')
    tooltip_page.press(' ')
    tooltip_page.expect_panel_value('tooltip', 'text')
    tooltip_page.press(' ')
    tooltip_page.expect_panel_value('tooltip', 'icon')
    tooltip_page.press(' ')
    tooltip_page.expect_panel_value('tooltip', 'off')


def test_enter_opens_the_rows_own_picker(tooltip_page):
    """CP3 -- a row and its picker are one value list, not two that drift."""
    tooltip_page.hover(*tooltip_page.blank_canvas_xy())
    tooltip_page.press('a')
    tooltip_page.press('i')
    tooltip_page.press('Enter')
    tooltip_page.expect_menu_open('tooltip:')
    _txt_ = tooltip_page.menu_text()
    for _state_ in ('off', 'text', 'icon'):
        assert _state_ in _txt_, _txt_


# ── off is really off ───────────────────────────────────────────────────────

def test_hovering_with_the_row_off_draws_nothing(xypi_page):
    """Default-off is a decision (PLANNING.md section 7): no round trips for anyone who
    has not asked for them, and no collision with the mouseover every browser test fires
    at mount.  A tooltip appearing here would mean the default leaked."""
    _x_, _y_ = xypi_page.first_mark_xy()
    xypi_page.hover(_x_, _y_)
    xypi_page.expect_no_tooltip()


# ── text mode ───────────────────────────────────────────────────────────────

def test_resting_on_a_mark_draws_a_tooltip(xypi_page):
    _set_mode(xypi_page, 'text')
    _x_, _y_ = xypi_page.first_mark_xy()
    assert xypi_page.hover_and_dwell(_x_, _y_), 'no tooltip appeared over a drawn mark'
    assert 'record' in xypi_page.tooltip_text(), xypi_page.tooltip_text()


def test_it_names_the_encoded_fields(xypi_page):
    _set_mode(xypi_page, 'text')
    _x_, _y_ = xypi_page.first_mark_xy()
    assert xypi_page.hover_and_dwell(_x_, _y_)
    _txt_ = xypi_page.tooltip_text()
    for _f_ in ('x', 'y', 'cat'):
        assert _f_ in _txt_, _txt_


def test_it_is_drawn_as_separate_lines(xypi_page):
    """One tspan per line, the same backdrop-and-tspan shape renderSelectedLabels uses."""
    _set_mode(xypi_page, 'text')
    _x_, _y_ = xypi_page.first_mark_xy()
    assert xypi_page.hover_and_dwell(_x_, _y_)
    assert len(xypi_page.tooltip_lines()) >= 2, xypi_page.tooltip_lines()


def test_moving_to_empty_canvas_clears_it(xypi_page):
    """A tooltip left behind over blank canvas describes a mark that is not there."""
    _set_mode(xypi_page, 'text')
    _x_, _y_ = xypi_page.first_mark_xy()
    assert xypi_page.hover_and_dwell(_x_, _y_)
    xypi_page.hover(*xypi_page.blank_canvas_xy())
    _deadline_ = time.monotonic() + 8.0
    while time.monotonic() < _deadline_:
        if not xypi_page.tooltip_is_showing():
            return
        time.sleep(0.05)
    raise AssertionError(f'the tooltip survived a move to blank canvas: '
                         f'{xypi_page.tooltip_text()!r}')


def test_turning_the_row_off_removes_the_drawing(xypi_page):
    """Not just "no new ones": the one on screen has to go with the mode."""
    _set_mode(xypi_page, 'text')
    _x_, _y_ = xypi_page.first_mark_xy()
    assert xypi_page.hover_and_dwell(_x_, _y_)
    _set_mode(xypi_page, 'off')
    assert not xypi_page.tooltip_is_showing(), xypi_page.tooltip_text()


def test_a_drag_suppresses_it(xypi_page):
    """A tooltip is a pure READ; nothing is being read while the pointer is pulling a
    rubber band, and a box following the cursor would cover the band."""
    _set_mode(xypi_page, 'text')
    _x_, _y_ = xypi_page.first_mark_xy()
    xypi_page.mouse_down(_x_, _y_)
    try:
        xypi_page.mouse_move_to(_x_ + 40, _y_ + 40)
        xypi_page.expect_no_tooltip()
    finally:
        xypi_page.mouse_up()


# ── icon mode ───────────────────────────────────────────────────────────────

def test_icon_mode_embeds_a_rendered_component(tooltip_page):
    """The point of the feature: a whole component, re-rendered against the hit, nested
    inside the view's own SVG -- the same thing stack_controli does per stack frame."""
    _set_mode(tooltip_page, 'icon')
    _x_, _y_ = tooltip_page.first_mark_xy()
    assert tooltip_page.hover_and_dwell(_x_, _y_), 'no tooltip appeared in icon mode'
    assert tooltip_page.tooltip_icon_svg_count() >= 1, (
        'icon mode drew no nested <svg>')


def test_text_mode_embeds_no_component(tooltip_page):
    """The contrast that makes the assertion above mean something."""
    _set_mode(tooltip_page, 'text')
    _x_, _y_ = tooltip_page.first_mark_xy()
    assert tooltip_page.hover_and_dwell(_x_, _y_)
    assert tooltip_page.tooltip_icon_svg_count() == 0


def test_the_icon_box_is_sized_from_the_icons_own_wxh(tooltip_page):
    """No new parameter: the box is the icon's own wxh (48x48 here) plus padding."""
    _set_mode(tooltip_page, 'icon')
    _x_, _y_ = tooltip_page.first_mark_xy()
    assert tooltip_page.hover_and_dwell(_x_, _y_)
    _box_ = tooltip_page.tooltip_box()
    assert _box_['w'] >= 48, _box_
    assert _box_['h'] > 48, _box_


# ── placement ───────────────────────────────────────────────────────────────

def test_the_box_stays_on_canvas_at_the_far_corner(xypi_page):
    """Flipped at the edge rather than clamped: a box pinned to the right edge sits ON
    the mark it describes, which is the one thing it must not cover."""
    _set_mode(xypi_page, 'text')
    _marks_ = xypi_page.root.locator(
        f'[id="mod{xypi_page.suffix}"] circle, [id="mod{xypi_page.suffix}"] rect'
    ).evaluate_all("""els => els.map(e => {
        const b = e.getBBox();
        return {cx: b.x + b.width/2, cy: b.y + b.height/2,
                area: Math.max(b.width,1) * Math.max(b.height,1)};
    })""")
    _small_ = [_m_ for _m_ in _marks_ if _m_['area'] < 200]
    _far_   = max(_small_, key=lambda _m_: _m_['cx'] + _m_['cy'])
    assert xypi_page.hover_and_dwell(_far_['cx'], _far_['cy']), 'no tooltip at the far mark'
    _box_ = xypi_page.tooltip_box()
    _w_, _h_ = xypi_page.plot.wxh
    assert _box_['x'] >= -1 and _box_['y'] >= -1, _box_
    assert _box_['x'] + _box_['w'] <= _w_ + 1, (_box_, _w_)
    assert _box_['y'] + _box_['h'] <= _h_ + 1, (_box_, _h_)


# ── it does not disturb anything else ───────────────────────────────────────

def test_a_hover_does_not_change_the_selection(xypi_page):
    """A pure read.  Resting the pointer somewhere must not be a destructive act."""
    _set_mode(xypi_page, 'text')
    _before_ = xypi_page.mod_html()
    _x_, _y_ = xypi_page.first_mark_xy()
    assert xypi_page.hover_and_dwell(_x_, _y_)
    assert xypi_page.mod_html() == _before_, 'the plot re-rendered on a hover'


def test_the_tooltip_layer_does_not_swallow_the_mouse(xypi_page):
    """U9 -- chrome along an edge silently ate gestures for as long as it went
    unnoticed.  A box that follows the pointer is the worst possible offender."""
    _set_mode(xypi_page, 'text')
    _x_, _y_ = xypi_page.first_mark_xy()
    assert xypi_page.hover_and_dwell(_x_, _y_)
    assert xypi_page.el('tooltip').get_attribute('pointer-events') == 'none'


if __name__ == '__main__':
    unittest.main()
