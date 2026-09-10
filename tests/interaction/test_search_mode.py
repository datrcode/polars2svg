"""The ``/`` search mode -- PLANNING.md 2.1 phase 2 item 5.

``search_mode`` and ``search_buffer`` are two more of the 29 JS-only variables, and
the on-screen ``/ ...▋`` echo is drawn by the keydown handler directly into
``#searchtext``.  None of it exists in Python until Enter ships the finished string,
so everything before that keystroke -- entering the mode, accumulating characters,
backspacing, cancelling -- was unobservable to the Python suite by construction.

The mode is also the component's only *modal* text entry: while it is on, every
ordinary binding must stop working, or typing a node name would fire a dozen
unrelated operations. That is asserted here too.

``settle()`` before each case: search state is browser-only.  The load-time rebuild
(U5) used to wipe it mid-test; that is fixed, and settle() now returns immediately on
an idle page, but it still guards against a ``mod_inner`` redraw landing mid-test.
"""
import unittest

import pytest

CURSOR = '▋'          # the '▋' the handler appends to the echo


def _search(ip, text):
    """Enter search mode and type, without committing."""
    ip.settle()
    ip.hover(200, 150)
    ip.press('/')
    for _ch_ in text:
        ip.press(_ch_)


# ── entering, echoing, editing ───────────────────────────────────────────────

def test_slash_enters_search_mode_and_shows_the_prompt(search_page):
    search_page.settle()
    search_page.hover(200, 150)
    search_page.press('/')
    assert search_page.el('searchtext').text_content() == f'/ {CURSOR}'


def test_typing_accumulates_into_the_echo(search_page):
    _search(search_page, 'alp')
    assert search_page.el('searchtext').text_content() == f'/ alp{CURSOR}'


def test_backspace_removes_the_last_character(search_page):
    _search(search_page, 'alp')
    search_page.press('Backspace')
    assert search_page.el('searchtext').text_content() == f'/ al{CURSOR}'


def test_backspace_on_an_empty_buffer_is_harmless(search_page):
    """slice(0, -1) on '' stays '' -- no underflow, no stuck cursor."""
    _search(search_page, '')
    search_page.press('Backspace')
    assert search_page.el('searchtext').text_content() == f'/ {CURSOR}'


def test_escape_cancels_without_selecting(search_page):
    _search(search_page, 'alp')
    search_page.press('Escape')
    assert search_page.el('searchtext').text_content() == ''
    search_page.expect_selected(0)          # nothing selected, nothing drawn


def test_enter_on_an_empty_buffer_leaves_the_mode_without_searching(search_page):
    _search(search_page, '')
    search_page.press('Enter')
    assert search_page.el('searchtext').text_content() == ''
    search_page.expect_selected(0)


# ── committing, and the set operations ───────────────────────────────────────

def test_enter_commits_a_substring_and_selects_the_matches(search_page):
    """'alp' matches alpha and alpine, and only those."""
    _search(search_page, 'alp')
    search_page.press('Enter')
    search_page.expect_selected(2)
    assert search_page.el('searchtext').text_content() == ''


def test_a_second_search_replaces_the_selection_by_default(search_page):
    _search(search_page, 'alp')
    search_page.press('Enter')
    search_page.expect_selected(2)

    _search(search_page, 'beta')
    search_page.press('Enter')
    search_page.expect_selected(1)


def test_plus_prefix_adds_to_the_selection(search_page):
    _search(search_page, 'alp')
    search_page.press('Enter')
    search_page.expect_selected(2)

    _search(search_page, '+beta')
    search_page.press('Enter')
    search_page.expect_selected(3)


def test_minus_prefix_subtracts_from_the_selection(search_page):
    _search(search_page, 'alp')
    search_page.press('Enter')
    search_page.expect_selected(2)

    _search(search_page, '-alpha')
    search_page.press('Enter')
    search_page.expect_selected(1)


def test_ampersand_prefix_intersects_with_the_selection(search_page):
    """alpha+alpine intersected with everything starting 'alpi' leaves alpine."""
    _search(search_page, 'alp')
    search_page.press('Enter')
    search_page.expect_selected(2)

    _search(search_page, '&alpi')
    search_page.press('Enter')
    search_page.expect_selected(1)


def test_a_slash_delimited_pattern_is_treated_as_a_regex(search_page):
    """'/^a/' anchors, so it matches alpha and alpine but not gamma.

    A substring 'a' would match all four nodes, which is what makes this
    distinguishable from the regex path actually running.
    """
    _search(search_page, '/^a/')
    search_page.press('Enter')
    search_page.expect_selected(2)


def test_matching_is_case_insensitive(search_page):
    _search(search_page, 'ALP')
    search_page.press('Enter')
    search_page.expect_selected(2)


def test_a_search_matching_nothing_clears_the_selection(search_page):
    _search(search_page, 'alp')
    search_page.press('Enter')
    search_page.expect_selected(2)

    _search(search_page, 'zzzz')
    search_page.press('Enter')
    search_page.expect_selected(0)


# ── modality ─────────────────────────────────────────────────────────────────

# NOTE on the oracle used below and in the two cancellation tests above.  These
# assert "no selection ran" with expect_selected(0) -- the count in #infostr *and*
# an empty #selectionlayer.  They used to assert `'Selected' not in info_text()`,
# which passed for the wrong reason: until PLANNING.md U5 was fixed the info line
# was pinned to its param default (" | | grid") for the whole life of the page, so
# the substring was absent no matter what the search keys did.  That assertion
# would have held even if 'z' had selected every node.

def test_ordinary_bindings_are_inert_while_typing(search_page):
    """'z' is a selection binding; in search mode it must be just a character.

    Without the early return, typing a node name would fire community detection,
    stack pushes and layout operations on the way past.
    """
    _search(search_page, 'z')
    assert search_page.el('searchtext').text_content() == f'/ z{CURSOR}'
    search_page.expect_selected(0)          # 'z' reached the selection handler if this fails


def test_digits_type_rather_than_selecting_by_degree(search_page):
    """The digit bindings select by node degree; in search mode they are text."""
    _search(search_page, 'a1')
    assert search_page.el('searchtext').text_content() == f'/ a1{CURSOR}'
    search_page.expect_selected(0)


if __name__ == '__main__':
    unittest.main()
