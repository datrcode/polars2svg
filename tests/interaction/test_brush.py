"""The radius brush -- PLANNING.md 2.1 phase 2 item 6.

Six of the 29 JS-only variables are the brush: ``brush_state`` (it is also a synced
param, but the *cursor* built from it is not), ``brush_defs``, ``brush_names``, and
``last_brush_x``/``last_brush_y``, which throttle re-brushing to ~3px of travel.  The
cursor and its mode label are drawn by ``updateBrushCursor`` straight into
``#brushindicator`` and ``#brushmodelabel`` and never cross into Python.

The Python half (``_doBrushAt``) already has unit coverage.  What had none is
everything the user actually sees: the toggle, the radius cycle, the circle tracking
the pointer, and the indicator clearing when the pointer leaves.

``settle()`` first throughout.  The cursor is browser-only state, and the load-time
rebuild (U5) used to erase it; that is fixed, so settle() now returns immediately on
an idle page, but a genuine ``mod_inner`` redraw still rebuilds and settling first is
still the right thing to do before asserting on browser-only state.
"""
import unittest

import pytest

#: state.brush_defs = [null, ['circle', 5], ['circle', 15]]
RADII = {1: '5', 2: '15'}
NAMES = {1: 'circ r=5', 2: 'circ r=15'}


def _brush_circle(ip):
    return ip.page.locator(f'[id="brushindicator{ip.suffix}"] circle')


# ── the toggle ───────────────────────────────────────────────────────────────

def test_r_turns_the_brush_on(brush_ready_page):
    brush_ready_page.press('r')
    expect_r(brush_ready_page, RADII[1])
    assert NAMES[1] in brush_ready_page.el('brushmodelabel').text_content()


def test_r_again_turns_the_brush_off(brush_ready_page):
    brush_ready_page.press('r')
    expect_r(brush_ready_page, RADII[1])

    brush_ready_page.press('r')
    assert _brush_circle(brush_ready_page).count() == 0, 'the brush cursor outlived the toggle'
    assert brush_ready_page.el('brushmodelabel').text_content() == ''


# ── the radius cycle ─────────────────────────────────────────────────────────

def test_shift_r_starts_the_brush_at_the_small_radius(brush_ready_page):
    brush_ready_page.press('R')
    expect_r(brush_ready_page, RADII[1])


def test_shift_r_cycles_to_the_large_radius(brush_ready_page):
    brush_ready_page.press('R')
    expect_r(brush_ready_page, RADII[1])

    brush_ready_page.press('R')
    expect_r(brush_ready_page, RADII[2])
    assert NAMES[2] in brush_ready_page.el('brushmodelabel').text_content()


def test_the_radius_cycle_wraps_and_never_returns_to_off(brush_ready_page):
    """The cycle is [1, 2] -- shift-R must not be a third way to switch the brush off."""
    for _ in range(3):
        brush_ready_page.press('R')
    expect_r(brush_ready_page, RADII[1])          # 0 -> 1 -> 2 -> 1
    assert _brush_circle(brush_ready_page).count() == 1


# ── the cursor follows the pointer ───────────────────────────────────────────

def test_the_cursor_tracks_the_pointer(brush_ready_page):
    brush_ready_page.press('r')
    expect_r(brush_ready_page, RADII[1])

    brush_ready_page.hover(120, 90)
    _c_ = _brush_circle(brush_ready_page)
    assert (_c_.get_attribute('cx'), _c_.get_attribute('cy')) == ('120', '90')

    brush_ready_page.hover(260, 200)
    assert (_c_.get_attribute('cx'), _c_.get_attribute('cy')) == ('260', '200')


def test_leaving_the_component_clears_the_cursor(brush_ready_page):
    """myOnMouseOut wipes the indicator, so the brush circle cannot be left stranded."""
    brush_ready_page.press('r')
    expect_r(brush_ready_page, RADII[1])

    _box_ = brush_ready_page.root.bounding_box()
    brush_ready_page.page.mouse.move(_box_['x'] + _box_['width'] + 80,
                                     _box_['y'] + _box_['height'] + 80)
    assert _brush_circle(brush_ready_page).count() == 0


def test_brushing_selects_nothing_by_itself(brush_ready_page):
    """The brush broadcasts to peers; it must not hijack the local selection."""
    brush_ready_page.press('r')
    expect_r(brush_ready_page, RADII[1])
    brush_ready_page.hover_node(1)

    assert '0 Selected' in brush_ready_page.info_text() or \
           'Selected' not in brush_ready_page.info_text(), \
           'brushing changed the local selection'


def expect_r(ip, radius):
    from playwright.sync_api import expect
    expect(_brush_circle(ip)).to_have_attribute('r', radius, timeout=ip.timeout_ms)


if __name__ == '__main__':
    unittest.main()
