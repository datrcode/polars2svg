"""The stack control's shelf, trashcans and set-operation clicks, in a browser
(PLANNING.md §7 F18).

The unit half (tests/test_stack_control.py, tests/test_stack_shelf.py) sets the click
params by hand.  What only a browser can show is that a real click sends its x and its
modifiers, so these drive the mouse and assert on the shared stack.  Click coordinates
come from the widget's own hit maps, which are what Python tests the click against.
"""
import unittest

import pytest
from playwright.sync_api import expect

from interaction_harness import assert_eventually


def _stack(sc):
    return sc.mvc.stacks['default']


def _seed(sc, n=3, index=1):
    """Base plus n-1 shrinking frames, frame `index` displayed."""
    _s_    = _stack(sc)
    _base_ = _s_['dfs'][0]
    _s_['dfs']   = [_base_] + [_base_.head(len(_base_) - 4 * _i_) for _i_ in range(1, n)]
    _s_['index'] = index
    sc.refreshShelf()


def _frameXY(sc, idx, x=20):
    _y0_, _y1_, _ = next(_r_ for _r_ in sc._frame_map_ if _r_[2] == idx)
    return x, (_y0_ + _y1_) // 2


def _rowXY(sc, name, x=20):
    _y0_, _y1_, _ = next(_r_ for _r_ in sc._shelf_map_ if _r_[2] == name)
    return x, (_y0_ + _y1_) // 2


def _click(ct, x, y, ctrl=False, shift=False):
    ct.hover(x, y)
    if ctrl or shift:
        with ct.holding(ctrl=ctrl, shift=shift):
            ct.page.mouse.click(*ct._page_xy(x, y))
    else:
        ct.page.mouse.click(*ct._page_xy(x, y))


def test_a_plain_click_on_a_shelf_row_pushes_it_on_top(stack_shelf_page):
    _xy_, _ct_, _sc_ = stack_shelf_page
    _seed(_sc_)
    _n_ = _sc_.mvc.pinFrame('scan 1', _stack(_sc_)['dfs'][0].tail(5))
    _click(_ct_, *_rowXY(_sc_, 'scan 1'))
    assert_eventually(lambda: _stack(_sc_)['dfs'][-1] is _n_,
                      lambda: f'the shelf entry was not pushed: {len(_stack(_sc_)["dfs"])} frames')
    assert len(_stack(_sc_)['dfs']) == 4, 'the push cut the frame above the displayed one'


@pytest.mark.parametrize('ctrl,shift,op,label', [(True,  False, 'union',     '∪ scan'),
                                                 (False, True,  'subtract',  '− scan'),
                                                 (True,  True,  'intersect', '∩ scan')])
def test_the_modifiers_reach_python_as_the_set_operation(stack_shelf_page, ctrl, shift, op, label):
    """ctrl adds, shift removes, ctrl-shift intersects -- linkpi's [[p(cs)^2]]."""
    _xy_, _ct_, _sc_ = stack_shelf_page
    _seed(_sc_)                                        # displayed: the first 16 rows
    _sc_.mvc.pinFrame('scan', _stack(_sc_)['dfs'][0].slice(12, 8))   # rows 12-19
    _click(_ct_, *_rowXY(_sc_, 'scan'), ctrl=ctrl, shift=shift)
    assert_eventually(lambda: _sc_.mvc.frameLabel('default', _stack(_sc_)['dfs'][-1]) == label,
                      lambda: f'{op}: top frame is labelled '
                              f'{_sc_.mvc.frameLabel("default", _stack(_sc_)["dfs"][-1])!r}, status {_sc_._status_!r}')
    _expect_ = {'union': 20, 'subtract': 12, 'intersect': 4}[op]
    assert len(_stack(_sc_)['dfs'][-1]) == _expect_


def test_a_stack_trashcan_deletes_that_frame(stack_shelf_page):
    _xy_, _ct_, _sc_ = stack_shelf_page
    _seed(_sc_, n=3, index=1)
    _gone_ = _stack(_sc_)['dfs'][2]
    _click(_ct_, *_frameXY(_sc_, 2, x=_sc_._tc_x0_ + 4))
    assert_eventually(lambda: all(_d_ is not _gone_ for _d_ in _stack(_sc_)['dfs']),
                      'the trashcan did not delete the frame')
    assert _stack(_sc_)['index'] == 1


def test_a_shelf_trashcan_needs_a_second_click(stack_shelf_page):
    _xy_, _ct_, _sc_ = stack_shelf_page
    _sc_.mvc.pinFrame('keep', _stack(_sc_)['dfs'][0].head(3))
    _x_, _y_ = _rowXY(_sc_, 'keep', x=_sc_._tc_x0_ + 4)
    _click(_ct_, _x_, _y_)
    assert_eventually(lambda: _sc_._armed_ is not None, 'the first click did not arm the trashcan')
    assert list(_sc_.mvc.shelf()) == ['keep']
    _click(_ct_, _x_, _y_)
    assert_eventually(lambda: _sc_.mvc.shelf() == {}, 'the second click did not delete the entry')


def test_a_refusal_shows_on_the_status_row(stack_shelf_page):
    """Q6: a ctrl-click on the displayed frame changes nothing -- visibly."""
    _xy_, _ct_, _sc_ = stack_shelf_page
    _seed(_sc_)
    _click(_ct_, *_frameXY(_sc_, 1), ctrl=True)
    assert_eventually(lambda: _sc_._status_ == 'no change', lambda: f'status {_sc_._status_!r}')
    expect(_ct_.page.locator('#svgstackcontrol #mod')).to_contain_text('no change')
    assert len(_stack(_sc_)['dfs']) == 3


def test_a_cut_off_name_is_shown_whole_on_hover(stack_shelf_page):
    _xy_, _ct_, _sc_ = stack_shelf_page
    _long_ = 'a shelf entry with a name far too long for its row'
    _sc_.mvc.pinFrame(_long_, _stack(_sc_)['dfs'][0].head(3))
    _tip_ = _ct_.page.locator('#svgstackcontrol #stacktip')
    expect(_tip_).to_have_attribute('display', 'none')
    _ct_.hover(*_rowXY(_sc_, _long_))
    expect(_tip_).to_have_attribute('display', 'inline')
    assert _tip_.text_content() == _long_


# ── the naming dialog (`s`, diagram 5 / Q11) ─────────────────────────────────

def _type(ct, text):
    for _ch_ in text:
        ct.press('Space' if _ch_ == ' ' else _ch_)


def _dialog(ct):
    return ct.page.locator('#svgstackcontrol #shelfname')


def test_s_names_and_shelves_the_displayed_frame(stack_shelf_page):
    _xy_, _ct_, _sc_ = stack_shelf_page
    _seed(_sc_)
    _shown_ = _stack(_sc_)['dfs'][1]
    _ct_.hover(20, 380)
    _ct_.press('s')
    expect(_dialog(_ct_)).to_have_attribute('display', 'inline')
    expect(_dialog(_ct_)).to_contain_text('frame 1')                 # the pre-fill
    _type(_ct_, 'scan 2')                                            # replaces it
    expect(_dialog(_ct_)).to_contain_text('scan 2')
    _ct_.press('Enter')
    assert_eventually(lambda: _sc_.mvc.shelf().get('scan 2') is _shown_,
                      lambda: f'not shelved: shelf {list(_sc_.mvc.shelf())}, status {_sc_._status_!r}')
    expect(_dialog(_ct_)).to_have_attribute('display', 'none')
    assert _sc_.mvc.frameLabel('default', _shown_) == 'scan 2'


def test_the_dialog_opens_on_the_first_free_row(stack_shelf_page):
    """Not over an existing entry: the new one is drawn where it was named (DT,
    2026-10-27)."""
    _xy_, _ct_, _sc_ = stack_shelf_page
    _sc_.mvc.pinFrame('already here', _stack(_sc_)['dfs'][0].head(3))
    _y0_, _y1_, _ = _sc_._shelf_map_[0]
    _ct_.hover(20, 380)
    _ct_.press('s')
    _rect_ = _ct_.page.locator('#svgstackcontrol #shelfname rect')
    expect(_dialog(_ct_)).to_have_attribute('display', 'inline')
    assert float(_rect_.get_attribute('y')) == _y1_, 'the dialog covers the existing entry'
    _ct_.press('Enter')
    assert_eventually(lambda: [_r_[2] for _r_ in _sc_._shelf_map_] == ['already here', 'frame 1'],
                      lambda: f'shelf drawn as {[_r_[2] for _r_ in _sc_._shelf_map_]}')
    assert _sc_._shelf_map_[0][:2] == (_y0_, _y1_), 'the existing entry moved'
    assert _sc_._shelf_map_[1][0] == _y1_


def test_typed_keys_are_not_shortcuts(stack_shelf_page):
    """A 'c' typed into the name must not also collapse the stack."""
    _xy_, _ct_, _sc_ = stack_shelf_page
    _seed(_sc_, n=4, index=2)
    _ct_.hover(20, 380)
    _ct_.press('s')
    expect(_dialog(_ct_)).to_have_attribute('display', 'inline')
    _type(_ct_, 'cch')
    _ct_.press('Backspace')
    _ct_.press('Enter')
    assert_eventually(lambda: 'cc' in _sc_.mvc.shelf(),
                      lambda: f'shelf {list(_sc_.mvc.shelf())}, status {_sc_._status_!r}')
    assert len(_stack(_sc_)['dfs']) == 4, 'a typed c collapsed the stack'
    assert _sc_.help_display == 'none', 'a typed h opened the help'


def test_escape_cancels(stack_shelf_page):
    _xy_, _ct_, _sc_ = stack_shelf_page
    _ct_.hover(20, 380)
    _ct_.press('s')
    expect(_dialog(_ct_)).to_have_attribute('display', 'inline')
    _ct_.press('Escape')
    expect(_dialog(_ct_)).to_have_attribute('display', 'none')
    _ct_.press('Enter')
    _ct_.wait_until_idle()
    assert _sc_.mvc.shelf() == {}


def test_s_on_a_frame_that_cannot_be_shelved_says_why(stack_shelf_page):
    _xy_, _ct_, _sc_ = stack_shelf_page
    _sc_.mvc.pinFrame('already')                                     # the displayed frame
    _ct_.hover(20, 380)
    _ct_.press('s')
    assert_eventually(lambda: _sc_._status_ == 'already shelved as "already"',
                      lambda: f'status {_sc_._status_!r}')
    expect(_dialog(_ct_)).to_have_attribute('display', 'none')


if __name__ == '__main__':
    unittest.main()
