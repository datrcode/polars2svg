"""The selected-node label overlay, in a real browser.

The feature's contract has two halves and only a browser can check both at once:

* Selecting nodes draws their **complete** labels, wrapped and centred under each
  node, in ``#selectedlabels``.
* It does so **outside the graph drawing** -- ``#mod`` must be byte-identical
  across the selection, which is what rules out the ``label_only`` re-render that
  sticky labels use.  Note this is not the same claim as "no re-render happened";
  see ``test_selecting_does_not_change_the_graph_drawing`` for where that one lives.
* The labels are translated with the selection outline during a node move, so they
  do not lag a drag by a server round-trip.

Waiting is on the labels themselves, never on ``#infostr``: ``__refreshView__``
writes ``info_str`` first and ``selection_labels`` last, so a test that waits on the
count and then reads the overlay is racing the two params (the hazard documented on
``InteractivePage.expect_selected``).
"""
import unittest

import polars as pl
import pytest
from playwright.sync_api import expect

from polars2svg import Polars2SVG

from interaction_harness import InteractivePage


# Long enough to wrap at the default label_line_width (32), and long enough that the
# rendered node label would have been cropped -- so "complete" is a real claim here.
_LONG_ = ('gamma node with a deliberately long and thoroughly '
          'descriptive name that has to wrap')
_LABELS_ = {'nw': 'northwest', 'ne': 'northeast', 'sw': 'southwest', 'se': _LONG_}


@pytest.fixture
def labelled_quad_linkp():
    """The pinned 2x2 grid, with one node carrying a label far too long for a line.

    Positions and view window are pinned for the reason quad_linkp pins them: a
    random layout puts nodes in the corners, where a drag runs off the canvas.
    """
    _p2s_ = Polars2SVG()
    _df_  = pl.DataFrame({'fm': ['nw', 'ne', 'se', 'sw'],
                          'to': ['ne', 'se', 'sw', 'nw']})
    _lp_  = _p2s_.linkp(_df_, relationships=[('fm', 'to')], wxh=(400, 300),
                        view_window=(0.0, 0.0, 1.0, 1.0),
                        pos={'nw': (0.25, 0.75), 'ne': (0.75, 0.75),
                             'sw': (0.25, 0.25), 'se': (0.75, 0.25)},
                        node_labels=_LABELS_)
    _lp_._repr_svg_()
    return _lp_


@pytest.fixture
def labels_page(page, served, labelled_quad_linkp):
    _app_ = served([[labelled_quad_linkp]])
    page.goto(_app_.url, wait_until='load')
    _ip_ = InteractivePage(page, labelled_quad_linkp)
    _ip_.app = _app_
    return _ip_


CORNER = (2, 2)          # bare canvas, clear of every node on this fixture


def _select_all(ip):
    ip.drag(*CORNER, 398, 298)


def _label_texts(ip):
    """One string per label block, lines joined with a space."""
    _out_ = []
    for _t_ in ip.within('selectedlabels', 'text').all():
        _spans_ = _t_.locator('tspan').all()
        if _spans_: _out_.append(' '.join((_s_.text_content() or '') for _s_ in _spans_))
        else:       _out_.append(_t_.text_content() or '')
    return _out_


def _expect_labels(ip, n):
    expect(ip.within('selectedlabels', 'text')).to_have_count(n, timeout=ip.timeout_ms)


# ── the overlay appears, and says the whole thing ────────────────────────────

def test_selecting_every_node_labels_every_node(labels_page):
    _select_all(labels_page)
    _expect_labels(labels_page, 4)


def test_labels_are_the_complete_text(labels_page):
    _select_all(labels_page)
    _expect_labels(labels_page, 4)
    assert sorted(_label_texts(labels_page)) == sorted(_LABELS_.values())


def test_a_long_label_is_not_cropped(labels_page):
    _select_all(labels_page)
    _expect_labels(labels_page, 4)
    _texts_ = _label_texts(labels_page)
    assert _LONG_ in _texts_
    assert not any('…' in _t_ for _t_ in _texts_)


def test_a_long_label_wraps_onto_several_lines(labels_page):
    _select_all(labels_page)
    _expect_labels(labels_page, 4)
    _long_el_ = labels_page.within('selectedlabels', 'text').filter(
        has_text='deliberately').first
    assert _long_el_.locator('tspan').count() > 1


def test_wrapped_lines_are_centred_on_one_x(labels_page):
    """Every line shares the text element's x, and the anchor is middle -- which is
    what puts the block centred under the node rather than left-aligned from it."""
    _select_all(labels_page)
    _expect_labels(labels_page, 4)
    _long_el_ = labels_page.within('selectedlabels', 'text').filter(
        has_text='deliberately').first
    assert _long_el_.get_attribute('text-anchor') == 'middle'
    _xs_ = {_s_.get_attribute('x') for _s_ in _long_el_.locator('tspan').all()}
    assert len(_xs_) == 1
    assert _xs_ == {_long_el_.get_attribute('x')}


def test_a_label_sits_below_its_node(labels_page):
    _coords_ = {_r_['__first__']: (int(_r_['__sx__']), int(_r_['__sy__']))
                for _r_ in labels_page.plot.df_node.iter_rows(named=True)}
    _select_all(labels_page)
    _expect_labels(labels_page, 4)
    _el_ = labels_page.within('selectedlabels', 'text').filter(has_text='northwest').first
    _nx_, _ny_ = _coords_['nw']
    assert abs(float(_el_.get_attribute('x')) - _nx_) <= 1
    assert float(_el_.get_attribute('y')) > _ny_


# ── the constraint: the graph drawing is untouched ───────────────────────────

def test_selecting_does_not_change_the_graph_drawing(labels_page):
    """#mod is the whole link-node drawing, and a selection must not alter it.

    This rejects the obvious alternative implementation -- re-rendering with
    ``label_only`` set, the way sticky labels work -- because that one puts the
    labels *inside* #mod.  It is NOT a check that no re-render happened: renderSVG()
    early-returns the cached string unless something invalidated it, so an
    unnecessary re-render of an unchanged graph is invisible from here.  That
    sharper claim is unit-tested in test_interactive_controller.py
    (TestLINKPISelectionLabels.test_selecting_does_not_call_rendersvg).
    """
    labels_page.settle()
    _before_ = labels_page.mod_html()
    _select_all(labels_page)
    _expect_labels(labels_page, 4)
    assert labels_page.mod_html() == _before_


def test_the_labels_are_drawn_outside_the_graph(labels_page):
    """Same claim from the other side: the label text is in the overlay, not #mod."""
    _select_all(labels_page)
    _expect_labels(labels_page, 4)
    assert 'northwest' not in labels_page.mod_html()


# ── lifecycle ────────────────────────────────────────────────────────────────

def test_deselecting_clears_the_overlay(labels_page):
    _select_all(labels_page)
    _expect_labels(labels_page, 4)
    labels_page.drag(*CORNER, CORNER[0] + 4, CORNER[1] + 4)   # empty band -> no selection
    _expect_labels(labels_page, 0)


def test_a_narrower_band_labels_only_what_it_covers(labels_page):
    _coords_ = {_r_['__first__']: (int(_r_['__sx__']), int(_r_['__sy__']))
                for _r_ in labels_page.plot.df_node.iter_rows(named=True)}
    _x_, _y_ = _coords_['nw']
    labels_page.drag(_x_ - 20, _y_ - 20, _x_ + 20, _y_ + 20)
    _expect_labels(labels_page, 1)
    assert _label_texts(labels_page) == ['northwest']


def test_over_the_cap_nothing_is_labeled(labels_page):
    labels_page.app.view().max_selection_labels = 2
    _select_all(labels_page)
    labels_page.expect_info_contains('labels capped')
    _expect_labels(labels_page, 0)


def test_labels_survive_a_rerender(labels_page):
    """The render script re-runs on every refresh; the overlay must be redrawn."""
    _select_all(labels_page)
    _expect_labels(labels_page, 4)
    labels_page.press('h')           # toggles the help overlay -> a refresh
    _expect_labels(labels_page, 4)


# ── the labels follow a drag of the selection ────────────────────────────────

def test_labels_track_a_drag_of_the_selection(labels_page):
    """Mid-drag the selection outline is translated in JS; the labels must be too,
    or they sit at the old positions until the mouseup round-trips to Python."""
    _coords_ = {_r_['__first__']: (int(_r_['__sx__']), int(_r_['__sy__']))
                for _r_ in labels_page.plot.df_node.iter_rows(named=True)}
    _select_all(labels_page)
    _expect_labels(labels_page, 4)
    assert labels_page.el('selectedlabels').get_attribute('transform') in (None, '')

    _x_, _y_ = _coords_['nw']
    labels_page.mouse_down(_x_, _y_)                 # press ON a selected node -> move op
    labels_page.mouse_move_to(_x_ + 30, _y_ + 20)
    _tr_labels_ = labels_page.el('selectedlabels').get_attribute('transform')
    _tr_sel_    = labels_page.el('selectionlayer').get_attribute('transform')
    labels_page.mouse_up()

    assert _tr_labels_ == _tr_sel_
    assert 'translate(30,20)' in (_tr_labels_ or '').replace(' ', '')


def test_the_translate_is_cleared_when_new_geometry_arrives(labels_page):
    """After the move commits, positions are absolute again."""
    _coords_ = {_r_['__first__']: (int(_r_['__sx__']), int(_r_['__sy__']))
                for _r_ in labels_page.plot.df_node.iter_rows(named=True)}
    _select_all(labels_page)
    _expect_labels(labels_page, 4)
    _x_, _y_ = _coords_['nw']
    labels_page.drag(_x_, _y_, _x_ + 30, _y_ + 20)
    labels_page.wait_until_idle()
    _expect_labels(labels_page, 4)
    assert labels_page.el('selectedlabels').get_attribute('transform') in (None, '')


# ── the overlay must not become a hit target ─────────────────────────────────

def test_the_overlay_does_not_intercept_the_mouse(labels_page):
    """Labels sit above the graph; if they took pointer events, a band started on a
    label would silently do nothing and node picking would break under them."""
    _select_all(labels_page)
    _expect_labels(labels_page, 4)
    assert labels_page.el('selectedlabels').get_attribute('pointer-events') == 'none'


if __name__ == '__main__':
    unittest.main()
