"""F3 -- the selection, read back from Python, driven by real gestures (PLANNING.md §7 F3).

``tests/test_selection_api.py`` drives the controller directly; this file is its browser
half.  These make the gesture in
the browser and check that what Python reads -- ``layout.selectedDataFrame()`` and what an
``onSelection()`` callback is handed -- is what the gesture selected: compared with the
plot's own filter for xypi, and with the pinned quad layout's arithmetic for linkpi.

The callbacks run on the server's thread, so every check waits with assert_eventually.
"""
import unittest

from interaction_harness import assert_eventually


def _input_cols_(df, frame):
    """``frame`` in ``df``'s columns: what selectedDataFrame() promises to return."""
    return frame.select([_c_ for _c_ in df.columns if _c_ in frame.columns])


def _same_rows_(a, b):
    return a.columns == b.columns and a.sort(a.columns).equals(b.sort(b.columns))


def test_an_xypi_drag_is_read_back_as_the_rows_it_filtered(xypi_page):
    _layout_ = xypi_page.app.container
    _view_   = xypi_page.app.view()
    _base_   = _layout_.mvc.stacks['default']['dfs'][0]
    _got_    = []
    _layout_.onSelection(_got_.append)

    _box_  = (20, 20, 200, 160)
    _want_ = _input_cols_(_base_, _view_._plot_.filterByRectangle(_box_))
    assert 0 < _want_.height < _base_.height
    xypi_page.drag(*_box_)

    assert_eventually(lambda: len(_got_) == 1, lambda: f'{len(_got_)} callbacks after one drag')
    assert _same_rows_(_got_[0], _want_), f'called back with {_got_[0].height} rows, the band holds {_want_.height}'
    assert _same_rows_(_layout_.selectedDataFrame(), _want_)
    assert _same_rows_(_view_.selectedDataFrame(), _want_)


def test_a_drag_over_nothing_pops_back_to_the_whole_frame(xypi_page):
    _layout_ = xypi_page.app.container
    _base_   = _layout_.mvc.stacks['default']['dfs'][0]
    _got_    = []
    _layout_.onSelection(lambda df: _got_.append(df.height))

    xypi_page.drag(20, 20, 200, 160)
    assert_eventually(lambda: len(_got_) == 1, 'the drag did not call back')
    _x_, _y_ = xypi_page.blank_canvas_xy()
    xypi_page.drag(_x_ - 3, _y_ - 3, _x_ + 3, _y_ + 3)            # an empty band pops
    assert_eventually(lambda: len(_got_) == 2, lambda: f'{len(_got_)} callbacks after the pop')
    assert _got_[1] == _base_.height
    assert _layout_.selectedDataFrame().equals(_base_)


def test_a_linkpi_band_is_read_back_as_the_rows_at_its_nodes(quad_page):
    """The northern row, nw and ne: every link touching either -- nw->ne, ne->se and
    sw->nw -- and not se->sw, which touches neither."""
    _layout_ = quad_page.app.container
    _got_    = []
    _layout_.onSelection(_got_.append)

    _c_ = {_r_['__first__']: (int(_r_['__sx__']), int(_r_['__sy__']))
           for _r_ in quad_page.plot.df_node.iter_rows(named=True)}
    _mid_y_ = (_c_['nw'][1] + _c_['sw'][1]) // 2
    quad_page.drag(2, 2, 398, _mid_y_)
    quad_page.assert_selection(['nw', 'ne'])

    def _rows_(df):
        return sorted(zip(df['fm'].to_list(), df['to'].to_list()))
    _want_ = [('ne', 'se'), ('nw', 'ne'), ('sw', 'nw')]
    assert_eventually(lambda: len(_got_) >= 1, 'the band did not call back')
    assert _rows_(_got_[-1]) == _want_, _rows_(_got_[-1])
    assert _rows_(_layout_.selectedDataFrame()) == _want_


if __name__ == '__main__':
    unittest.main()
