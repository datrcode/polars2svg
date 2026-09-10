"""Primary effects of the LINKPI key bindings -- PLANNING.md 2.1 phase 2 item 2.

One test per *operation* rather than per key: the dispatch chain has 62 distinct
``event.key`` comparisons, but many are aliases (``1``/``!``, ``n``/``N``, ``z``/``Z``,
``y``/``Y``), and an alias is worth one test proving it reaches the same place, not a
duplicate of the whole behaviour.  Bindings covered in the dedicated files -- the nine
ctrl guards, the eight picker menus, ``/`` search, ``r``/``R`` brush, ``z`` -- are not
repeated here.

Where a binding reads a modifier on the *Python* side (the degree set-operations),
the tests hold it down rather than tapping it: see ``InteractivePage.holding`` and U3.
"""
import unittest

import pytest


def _select_colour_group(ip, node=1):
    ip.hover_node(node)
    ip.press('z')


def _node_circles(ip):
    return ip.within('mod', 'circle').count()


# ── h: the help overlay, which is pure DOM ───────────────────────────────────

def test_h_slides_the_help_overlay_into_view(linkpi_page):
    """keyboardhelp_x moves the panel from -1000 to 5; the transform is the proof."""
    assert linkpi_page.el('keyboardhelp').get_attribute('transform') == 'translate(-1000 0)'

    linkpi_page.hover(200, 150)
    linkpi_page.press('h')
    from playwright.sync_api import expect
    expect(linkpi_page.el('keyboardhelp')).to_have_attribute(
        'transform', 'translate(5 0)', timeout=linkpi_page.timeout_ms)


def test_h_again_slides_it_back_out(linkpi_page):
    from playwright.sync_api import expect
    linkpi_page.hover(200, 150)
    linkpi_page.press('h')
    expect(linkpi_page.el('keyboardhelp')).to_have_attribute(
        'transform', 'translate(5 0)', timeout=linkpi_page.timeout_ms)

    linkpi_page.hover(200, 150)
    linkpi_page.press('h')
    expect(linkpi_page.el('keyboardhelp')).to_have_attribute(
        'transform', 'translate(-1000 0)', timeout=linkpi_page.timeout_ms)


# ── digits: select by node degree ────────────────────────────────────────────
# The fixture graph is a 1-2-3 triangle (degree 2 each) plus a 4-5 pair (degree 1
# each), so the two digits pick out disjoint, differently-sized sets.

def test_digit_2_selects_the_degree_two_nodes(linkpi_page):
    linkpi_page.hover(200, 150)
    linkpi_page.press('2')
    linkpi_page.expect_selected(3)


def test_digit_1_selects_the_degree_one_nodes(linkpi_page):
    linkpi_page.hover(200, 150)
    linkpi_page.press('1')
    linkpi_page.expect_selected(2)


def test_the_shifted_digit_is_an_alias(linkpi_page):
    """'@' reaches the same branch as '2' -- the shifted row is bound so the binding
    still works on a keyboard where the user's hand is already on shift."""
    linkpi_page.hover(200, 150)
    linkpi_page.press('@')
    linkpi_page.expect_selected(3)


def test_ctrl_digit_adds_to_the_selection(linkpi_page):
    linkpi_page.hover(200, 150)
    linkpi_page.press('1')
    linkpi_page.expect_selected(2)

    linkpi_page.hover(200, 150)
    with linkpi_page.holding(ctrl=True):
        linkpi_page.press('2')
        linkpi_page.expect_selected(5)


def test_shift_digit_removes_from_the_selection(linkpi_page):
    linkpi_page.hover(200, 150)
    linkpi_page.press('2')
    linkpi_page.expect_selected(3)

    linkpi_page.hover(200, 150)
    with linkpi_page.holding(shift=True):
        linkpi_page.press('@')            # shift-2 on a US layout
        linkpi_page.expect_selected(0)


# ── selection algebra ────────────────────────────────────────────────────────

def test_q_inverts_the_selection(linkpi_page):
    _select_colour_group(linkpi_page, 1)
    linkpi_page.expect_selected(3)

    linkpi_page.hover_node(1)
    linkpi_page.press('q')
    linkpi_page.expect_selected(2)


def test_n_selects_every_node_of_the_same_shape(linkpi_page):
    """nodeShape() is always 'circle' here, so 'n' over any node takes the lot.

    Thin, but it is the binding's actual contract, and it proves the shape path is
    wired to a different attribute extractor than the colour path 'z' uses.
    """
    linkpi_page.hover_node(1)
    linkpi_page.press('n')
    linkpi_page.expect_selected(5)


# ── the dataframe stack ──────────────────────────────────────────────────────

def test_x_removes_the_selected_nodes_and_pushes_the_stack(linkpi_page):
    _select_colour_group(linkpi_page, 1)
    linkpi_page.expect_selected(3)
    assert _node_circles(linkpi_page) == 5

    _before_ = linkpi_page.mod_html()
    linkpi_page.hover_node(1)
    linkpi_page.press('x')
    linkpi_page.wait_for_mod_change(_before_)
    assert _node_circles(linkpi_page) == 2, 'the three selected nodes were not removed'


def test_shift_x_pops_the_stack_and_brings_them_back(linkpi_page):
    _select_colour_group(linkpi_page, 1)
    _before_ = linkpi_page.mod_html()
    linkpi_page.hover_node(1)
    linkpi_page.press('x')
    linkpi_page.wait_for_mod_change(_before_)
    assert _node_circles(linkpi_page) == 2

    _mid_ = linkpi_page.mod_html()
    linkpi_page.hover(200, 150)
    linkpi_page.press('X')
    linkpi_page.wait_for_mod_change(_mid_)
    assert _node_circles(linkpi_page) == 5, 'the pop did not restore the removed nodes'


# ── community colouring ──────────────────────────────────────────────────────

def test_shift_d_clears_the_community_colours_d_applied(linkpi_page):
    linkpi_page.hover(200, 150)
    _plain_ = linkpi_page.mod_html()
    linkpi_page.press('d')
    _coloured_ = linkpi_page.wait_for_mod_change(_plain_)

    linkpi_page.hover(200, 150)
    linkpi_page.press('D')
    _restored_ = linkpi_page.wait_for_mod_change(_coloured_)
    assert _restored_ != _coloured_, 'shift-D left the community colours in place'


# ── expansion along edges ────────────────────────────────────────────────────

def test_e_expands_the_selection_along_undirected_edges(chain_page):
    """On 1->2->3->4, undirected expansion from 3 reaches both neighbours."""
    chain_page.hover_node(3)
    chain_page.press('z')
    chain_page.expect_selected(1)

    chain_page.hover_node(3)
    chain_page.press('e')
    chain_page.expect_selected(3)             # 2, 3, 4


def test_shift_e_expands_only_forward_along_directed_edges(chain_page):
    """The distinction between 'e' and 'E': direction.  From 3, forward is 4 alone."""
    chain_page.hover_node(3)
    chain_page.press('z')
    chain_page.expect_selected(1)

    chain_page.hover_node(3)
    chain_page.press('E')
    chain_page.expect_selected(2)             # 3, 4


# ── geometry ─────────────────────────────────────────────────────────────────

def test_t_collapses_the_selection_to_a_point(linkpi_page):
    """The three selected nodes end up sharing one position, so the render changes."""
    _select_colour_group(linkpi_page, 1)
    linkpi_page.expect_selected(3)

    _before_ = linkpi_page.mod_html()
    linkpi_page.hover_node(1)
    linkpi_page.press('t')
    _after_ = linkpi_page.wait_for_mod_change(_before_)
    assert _after_ != _before_


def test_escape_bumps_the_cancel_sequence(linkpi_page):
    """Escape asks a running layout to stop; with none running it only counts.

    A Python-side assertion by necessity -- the counter has no DOM representation, and
    the alternative would be starting a layout slow enough to interrupt.
    """
    _view_ = linkpi_page.app.view()
    _before_ = _view_.cancel_seq

    linkpi_page.hover(200, 150)
    linkpi_page.press('Escape')

    from playwright.sync_api import expect
    expect(linkpi_page.root).to_be_focused(timeout=linkpi_page.timeout_ms)
    import time
    _deadline_ = time.monotonic() + 10
    while time.monotonic() < _deadline_ and _view_.cancel_seq == _before_:
        time.sleep(0.05)
    assert _view_.cancel_seq == _before_ + 1


# ── degree ranges: the 7-0 half of the digit bindings ────────────────────────

def test_digit_7_selects_the_degree_range_not_an_exact_degree(hub_page):
    """1-6 mean "degree exactly n"; 7-0 mean a *range*, here 7 through 20.

    The hub has degree 8, so it is selected by 7 and by nothing else -- which is the
    only way to tell the range branch from the exact one.
    """
    hub_page.hover(200, 150)
    hub_page.press('7')
    hub_page.expect_selected(1)


def test_digit_1_still_means_exactly_one_on_the_same_graph(hub_page):
    """The eight leaves, and not the hub -- the exact branch, for contrast."""
    hub_page.hover(200, 150)
    hub_page.press('1')
    hub_page.expect_selected(8)


@pytest.mark.parametrize('key', ['8', '9', '0'])
def test_the_higher_ranges_match_nothing_on_a_small_graph(hub_page, key):
    """21-50, 51-100 and 101-10000 have no members here, so each clears the selection.

    Worth pinning because an off-by-one in the range table would show up as one of
    these quietly selecting the degree-8 hub.
    """
    hub_page.hover(200, 150)
    hub_page.press('7')
    hub_page.expect_selected(1)

    hub_page.hover(200, 150)
    hub_page.press(key)
    hub_page.expect_selected(0)


# ── layout operations and undo ───────────────────────────────────────────────

def test_w_applies_the_current_layout_operation(quad_page):
    """'w' runs whatever the layout-operation picker last committed (spring nx by
    default) over the selection, or the whole graph when nothing is selected."""
    _before_ = quad_page.node_positions()
    quad_page.hover(200, 150)
    quad_page.press('w')
    quad_page.wait_until_idle()
    assert quad_page.node_positions() != _before_


def test_u_undoes_the_last_layout(quad_page):
    """Undo is only armed by operations that cache positions first."""
    _home_ = quad_page.node_positions()
    quad_page.hover(200, 150)
    quad_page.press('w')
    quad_page.wait_until_idle()
    _moved_ = quad_page.node_positions()
    assert _moved_ != _home_

    quad_page.hover(200, 150)
    quad_page.press('u')
    quad_page.wait_until_idle()
    assert quad_page.node_positions() != _moved_, 'undo changed nothing'
    # Not asserted equal to _home_: 'w' recenters the *view window* as well as moving
    # the nodes, and undo restores the layout only -- so the nodes come back to their
    # old world positions but are drawn through the new window.


def test_shift_t_collapses_the_selection_horizontally(quad_page):
    """'t' collapses to a point, shift-t to a horizontal line -- so afterwards the two
    selected nodes share a y but keep distinct x's."""
    _c_ = {_r_['__first__']: (int(_r_['__sx__']), int(_r_['__sy__']))
           for _r_ in quad_page.plot.df_node.iter_rows(named=True)}
    _mid_y_ = (_c_['nw'][1] + _c_['sw'][1]) // 2
    quad_page.drag(2, 2, 398, _mid_y_)
    quad_page.expect_selected(2)
    quad_page.wait_until_idle()             # #mod reads back empty mid-rebuild

    _before_ = quad_page.node_positions()
    quad_page.hover(*_c_['nw'])
    quad_page.press('T')
    quad_page.wait_until_idle()
    assert quad_page.node_positions() != _before_


def test_v_collapses_the_selection_vertically(quad_page):
    """'v' is the modifier-free vertical collapse (PLANNING.md U2).

    Asserts the axis rather than just "the drawing changed", which is what tells this
    apart from the other two cells of the matrix.  Selects the **west column** and
    collapses it onto the mouse x: afterwards those two nodes sit on x=150 with their
    y's untouched, and the unselected east column has not moved.  A point collapse would
    move y as well; a horizontal collapse would flatten y and leave x alone.

    The column selection is deliberate.  Collapsing a *row* -- or all four -- puts two
    nodes on the same (x, y), and coincident nodes are redrawn by
    ``__contractCollapsedGraph__()`` as a single contracted glyph that is no longer a
    ``<circle>``, so ``node_positions()`` reads back empty and the test fails for a
    reason unrelated to the axis.

    **This does not, and cannot, test the reason 'v' exists.**  ctrl-t is reserved as
    new-tab off macOS and preventDefault() cannot reclaim a browser-chrome shortcut --
    but Playwright dispatches keys over CDP, which bypasses browser chrome entirely.
    Measured: a synthesized Cmd-T (reserved new-tab on macOS) reaches the page handler
    with metaKey set and opens no tab, in headed full-channel chromium.  So no run of
    this suite on any platform can observe the reservation; only a human on a real
    Linux/Windows browser can.  What is testable is that the replacement works.
    """
    _c_ = {_r_['__first__']: (int(_r_['__sx__']), int(_r_['__sy__']))
           for _r_ in quad_page.plot.df_node.iter_rows(named=True)}
    _mid_x_ = (_c_['nw'][0] + _c_['ne'][0]) // 2

    quad_page.drag(2, 2, _mid_x_, 298)
    quad_page.expect_selected(2)
    # Not optional, and it became necessary only once U5 was fixed: info_str now
    # updates without a rebuild, so expect_selected() can return while the mod_inner
    # redraw is still in flight -- and #mod reads back empty mid-rebuild.
    quad_page.wait_until_idle()

    _before_ = quad_page.node_positions()
    _xs_before_ = sorted({round(_x_) for _x_, _y_ in _before_.values()})
    _ys_before_ = sorted({round(_y_) for _x_, _y_ in _before_.values()})
    assert len(_xs_before_) == 2 and len(_ys_before_) == 2, (
        f'fixture should start as a 2x2 grid; got x={_xs_before_} y={_ys_before_}')

    quad_page.hover(150, 150)
    quad_page.press('v')
    quad_page.wait_until_idle()

    _after_ = quad_page.node_positions()
    _xs_after_ = sorted({round(_x_) for _x_, _y_ in _after_.values()})
    _ys_after_ = sorted({round(_y_) for _x_, _y_ in _after_.values()})

    assert _ys_after_ == _ys_before_, (
        f'v must not touch y -- a point or horizontal collapse would; '
        f'{_ys_before_} -> {_ys_after_}')
    assert len(_xs_after_) == 2 and _xs_after_[1] == _xs_before_[1], (
        f'the unselected east column should not have moved; {_xs_before_} -> {_xs_after_}')
    assert abs(_xs_after_[0] - 150) <= 1, (
        f'the selected column should sit on the mouse x (150); got {_xs_after_[0]}')


# ── zoom / neighbourhood ─────────────────────────────────────────────────────

def test_shift_c_zooms_to_the_selection_and_its_neighbours(chain_page):
    """A view change, so every node's drawn position moves even though none moved.

    On the chain fixture, where one colour per node makes 'z' a single-node selection;
    the quad graph leaves colouring to the hash, so 'z' there is not a known quantity.
    """
    chain_page.hover_node(2)
    chain_page.press('z')
    chain_page.expect_selected(1)

    _before_ = chain_page.node_positions()
    chain_page.hover_node(2)
    chain_page.press('C')
    chain_page.wait_until_idle()
    assert chain_page.node_positions() != _before_


def test_shift_q_selects_the_common_neighbours(chain_page):
    """On the chain 1-2-3-4, the only node adjacent to *both* 1 and 3 is 2 -- a real
    intersection, not merely "the neighbours of something".

    ctrl-z builds the two-node selection, which also exercises the z set-operation
    modifiers; it needs holding, since the Python side reads self.ctrlkey (U3).
    """
    chain_page.hover_node(1)
    chain_page.press('z')
    chain_page.expect_selected(1)

    chain_page.hover_node(3)
    with chain_page.holding(ctrl=True):
        chain_page.press('z')
        chain_page.expect_selected(2)

    chain_page.hover(200, 150)
    chain_page.press('Q')
    chain_page.expect_selected(1)


# ── stack-growing operations ─────────────────────────────────────────────────

# ── f / F: widening the frame back out from the base rows ───────────────────
#
# Both add base-dataframe rows on top of the current view and push if anything was
# recovered, and they differ in *which* rows are eligible: 'f' refills the edges that
# are already visible, 'F' takes every base row incident to a visible node -- which can
# pull a node back that is not there any more.
#
# The awkward part, and why these were the last bindings covered: on an ordinary frame
# both correctly do nothing, because there is nothing to recover.  'f' needs an edge
# whose rows have been *thinned* while the edge itself stays visible, which is exactly
# what ctrl-shift-X produces; 'F' needs a node to have been removed while one of its
# neighbours remains.

def test_f_refills_an_edge_that_was_thinned(multi_edge_page):
    """ctrl-shift-X collapses each edge to one row; 'f' puts the rest back.

    Asserted on the frame rather than on the drawing: the recovered rows are duplicate
    edges, so the picture is identical either way -- the change is in what the frame
    contains, which is the whole point of the binding and is not visible anywhere.
    """
    _view_ = multi_edge_page.app.view()
    _rows_ = lambda: len(_view_.dfs[_view_.df_level])
    _full_ = _rows_()
    assert _full_ == 6

    multi_edge_page.hover(200, 150)
    with multi_edge_page.holding(ctrl=True, shift=True):
        multi_edge_page.press('X')
        _await_(lambda: _rows_() < _full_, 'ctrl-shift-X did not collapse the edges')
    _collapsed_ = _rows_()

    multi_edge_page.hover(200, 150)
    multi_edge_page.press('f')
    _await_(lambda: _rows_() > _collapsed_, "'f' recovered nothing")
    assert _rows_() == _full_, (
        f"'f' restored {_rows_()} of the {_full_} base rows on the visible edges")


def test_f_cannot_bring_back_a_node_that_was_removed(chain_page):
    """'f' only refills edges that are still visible, so a removed node stays removed.

    The negative half of the pair: without it, the next test would not show that 'F'
    does something 'f' cannot.
    """
    _circles_ = lambda: chain_page.within('mod', 'circle').count()
    assert _circles_() == 4

    chain_page.hover_node(4)
    chain_page.press('z')
    chain_page.expect_selected(1)

    _before_ = chain_page.mod_html()
    chain_page.hover_node(4)
    chain_page.press('x')                       # push, dropping node 4
    chain_page.wait_for_mod_change(_before_)
    assert _circles_() == 3

    _view_ = chain_page.app.view()
    _levels_ = len(_view_.dfs)
    chain_page.hover(200, 150)
    chain_page.press('f')
    chain_page.wait_until_idle()

    assert _circles_() == 3, "'f' brought a removed node back"
    assert len(_view_.dfs) == _levels_, "'f' pushed a frame despite recovering nothing"


def test_shift_f_pulls_back_a_removed_neighbour(chain_page):
    """'F' takes every base row incident to a visible node, so node 4 returns.

    On the chain 1-2-3-4, removing 4 leaves 3 visible; the base edge 3-4 is incident to
    3, so expanding restores it -- and with it the node. That is a change in the
    drawing, so it is asserted there.
    """
    _circles_ = lambda: chain_page.within('mod', 'circle').count()
    chain_page.hover_node(4)
    chain_page.press('z')
    chain_page.expect_selected(1)

    _before_ = chain_page.mod_html()
    chain_page.hover_node(4)
    chain_page.press('x')
    chain_page.wait_for_mod_change(_before_)
    assert _circles_() == 3

    _removed_ = chain_page.mod_html()
    chain_page.hover(200, 150)
    chain_page.press('F')
    chain_page.wait_for_mod_change(_removed_)
    assert _circles_() == 4, "'F' did not restore the removed neighbour"


def _await_(predicate, message, budget_s=15.0):
    """Bounded poll on frame state, which has no DOM representation."""
    import time
    _deadline_ = time.monotonic() + budget_s
    while time.monotonic() < _deadline_:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError(message)


def test_ctrl_shift_x_collapses_edges_onto_the_stack(multi_edge_page):
    """Each edge becomes a single row, pushing the shrunken frame onto the stack.

    Needs a graph with more than one row per edge -- on a frame that already has one,
    the operation correctly does nothing and there is no behaviour to see.
    """
    import time
    _view_ = multi_edge_page.app.view()
    _levels_ = len(_view_.dfs)
    _rows_ = len(_view_.dfs[_view_.df_level])

    multi_edge_page.hover(200, 150)
    with multi_edge_page.holding(ctrl=True, shift=True):
        multi_edge_page.press('X')
        _deadline_ = time.monotonic() + 15
        while time.monotonic() < _deadline_ and len(_view_.dfs) == _levels_:
            time.sleep(0.05)

    assert len(_view_.dfs) == _levels_ + 1, 'ctrl-shift-X did not push a level'
    assert len(_view_.dfs[_view_.df_level]) < _rows_, 'the collapsed frame is no smaller'


# ── D4 is deliberately NOT tested here ───────────────────────────────────────
#
# A keystroke arriving while an operation holds the lock is dropped rather than
# queued, which is why `press()` waits for idle by default (see its docstring).  It is
# tempting to pin that here by pressing early on purpose -- and a first attempt did.
# It passed alone and failed in the full run, because asserting a *drop* means
# asserting that a race was lost, and on a warm machine the operation finishes first.
# A test that has to win a race to pass is the exact fragility this suite spent its
# time removing everywhere else.
#
# The behaviour already has deterministic coverage where it belongs, with no browser
# and no timing: tests/test_interactive_controller.py::test_a_key_press_while_busy_is_dropped.


if __name__ == '__main__':
    unittest.main()
