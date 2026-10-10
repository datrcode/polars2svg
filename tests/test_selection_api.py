#
# F3 -- the selection, read from Python (PLANNING.md §7 F3).
#
# InteractionController.selectedDataFrame() is the frame the views are showing (the current
# stack level; the whole input frame before any drill-down), in the input frame's columns.
# onSelection(callback) is told after every change to it.  These drive the controller's
# async stack methods directly, as test_interactive_controller.py does; the browser half
# is tests/interaction/test_selection_readback.py.
#
import asyncio
import logging
import unittest

import polars as pl

from polars2svg import Polars2SVG
from polars2svg.interactive_controller import InteractionController


class _View_:
    """The least a stack peer needs: display()."""
    def __init__(self):
        self.displayed = []

    async def display(self, df, dfs, index):
        self.displayed.append(df)


def _df_():
    return pl.DataFrame({'a': [1, 2, 3, 4], 'b': ['w', 'x', 'y', 'z']})


class TestSelectedDataFrame(unittest.TestCase):

    def setUp(self):
        self.df  = _df_()
        self.mvc = InteractionController()
        self.mvc.addStack('default', self.df)
        self.v   = _View_()
        self.mvc.link(self.v, [], on='stack', stack='default')

    def test_before_any_drill_down_it_is_the_whole_frame(self):
        # DT, 2026-10-09: the current level, always -- never None.
        self.assertTrue(self.mvc.selectedDataFrame().equals(self.df))

    def test_it_follows_the_stack_down_and_back_up(self):
        df2 = self.df.filter(pl.col('a') > 2)
        asyncio.run(self.mvc.pushStack(self.v, df2))
        self.assertTrue(self.mvc.selectedDataFrame().equals(df2))
        asyncio.run(self.mvc.popStack(self.v))
        self.assertTrue(self.mvc.selectedDataFrame().equals(self.df))
        asyncio.run(self.mvc.setStackIndex(self.v, 1))
        self.assertTrue(self.mvc.selectedDataFrame().equals(df2))

    def test_internal_columns_are_stripped_and_input_order_kept(self):
        # chordp and histop push frames carrying __p2s_index__; the user never handed it in.
        _pushed_ = self.df.with_row_index('__p2s_index__').select(['b', '__p2s_index__', 'a']).head(2)
        asyncio.run(self.mvc.pushStack(self.v, _pushed_))
        _sel_ = self.mvc.selectedDataFrame()
        self.assertEqual(_sel_.columns, ['a', 'b'])
        self.assertEqual(_sel_['a'].to_list(), [1, 2])

    def test_a_replaced_base_frame_is_the_selection(self):
        df_new = pl.DataFrame({'a': [9], 'b': ['q']})
        asyncio.run(self.mvc.replaceStack('default', df_new))
        self.assertTrue(self.mvc.selectedDataFrame().equals(df_new))

    def test_several_stacks_must_be_named(self):
        self.mvc.addStack('other', self.df.head(1))
        with self.assertRaisesRegex(ValueError, "2 stacks, so name one: \\['default', 'other'\\]"):
            self.mvc.selectedDataFrame()
        self.assertEqual(self.mvc.selectedDataFrame('other').height, 1)
        with self.assertRaisesRegex(ValueError, "no stack named 'nope'"):
            self.mvc.selectedDataFrame('nope')


class TestOnSelection(unittest.TestCase):

    def setUp(self):
        self.df  = _df_()
        self.df2 = self.df.filter(pl.col('a') > 2)
        self.mvc = InteractionController()
        self.mvc.addStack('default', self.df)
        self.v   = _View_()
        self.mvc.link(self.v, [], on='stack', stack='default')
        self.got = []
        self.mvc.onSelection(lambda df: self.got.append(df.height))

    def test_every_stack_move_calls_back_with_the_new_selection(self):
        asyncio.run(self.mvc.pushStack(self.v, self.df2))         # 2 rows
        asyncio.run(self.mvc.popStack(self.v))                    # back to 4
        asyncio.run(self.mvc.setStackIndex(self.v, 1))            # 2 again
        asyncio.run(self.mvc.replaceStack(self.v, self.df.head(3)))
        self.assertEqual(self.got, [2, 4, 2, 3])

    def test_nothing_that_leaves_the_selection_alone_calls_back(self):
        asyncio.run(self.mvc.popStack(self.v))                    # already at the base
        asyncio.run(self.mvc.setStackIndex(self.v, 0))            # already there
        asyncio.run(self.mvc.setStackIndex(self.v, 5))            # out of range
        peer = _View_()
        self.mvc.link(peer, [], on='stack', stack='default')
        asyncio.run(self.mvc.brushUpdate(self.v, self.df2))       # a brush is a preview
        asyncio.run(self.mvc.brushClear(self.v))
        self.assertEqual(self.got, [])
        self.assertEqual(len(peer.displayed), 2)                  # the brush did reach the peer

    def test_a_refused_push_does_not_call_back(self):
        self.mvc.max_stack_depth = 1
        with self.assertLogs('polars2svg_logger', logging.WARNING):
            self.assertFalse(asyncio.run(self.mvc.pushStack(self.v, self.df2)))
        self.assertEqual(self.got, [])

    def test_a_coroutine_callback_is_awaited(self):
        _seen_ = []
        async def _cb_(df):
            await asyncio.sleep(0)
            _seen_.append(df['a'].to_list())
        self.mvc.onSelection(_cb_)
        asyncio.run(self.mvc.pushStack(self.v, self.df2))
        self.assertEqual(_seen_, [[3, 4]])

    def test_a_raising_callback_is_logged_and_the_rest_still_run(self):
        def _bad_(df):
            raise RuntimeError('boom')
        _after_ = []
        mvc = InteractionController()
        mvc.addStack('default', self.df)
        mvc.link(self.v, [], on='stack', stack='default')
        mvc.onSelection(_bad_)
        mvc.onSelection(lambda df: _after_.append(df.height))
        with self.assertLogs('polars2svg_logger', logging.ERROR) as _log_:
            self.assertTrue(asyncio.run(mvc.pushStack(self.v, self.df2)))
        self.assertIn('boom', '\n'.join(_log_.output))
        self.assertEqual(_after_, [2])
        self.assertEqual(mvc.stacks['default']['index'], 1)

    def test_it_returns_the_callback_and_refuses_a_non_callable(self):
        def _cb_(df): pass
        self.assertIs(self.mvc.onSelection(_cb_), _cb_)
        with self.assertRaisesRegex(TypeError, 'expected a callable, got str'):
            self.mvc.onSelection('not a function')


class TestSelectionFromARealView(unittest.TestCase):
    """The same through a real view: a chordpi rubber band pushes chordp's own records,
    which carry __p2s_index__."""

    def test_a_chordpi_drag_selects_input_columns_only(self):
        p2s = Polars2SVG()
        df  = pl.DataFrame({'fm': ['a', 'a', 'b', 'c'], 'to': ['b', 'c', 'c', 'a'], 'n': [1, 2, 3, 4]})
        view = p2s.chordpi(p2s.chordp(df=df, relationships=[('fm', 'to')], wxh=(256, 256)))
        _got_ = []
        view.mvc.onSelection(_got_.append)
        _recs_ = view._plot_.filterByRectangle((128, 0, 256, 256), False)   # the right half
        self.assertIn('__p2s_index__', _recs_.columns)            # what the view pushes
        self.assertTrue(0 < _recs_.height < df.height)
        asyncio.run(view.mvc.pushStack(view, _recs_))
        _sel_ = view.mvc.selectedDataFrame()
        self.assertEqual(_sel_.columns, ['fm', 'to', 'n'])
        self.assertEqual(_sel_.height, _recs_.height)
        self.assertEqual(len(_got_), 1)
        self.assertTrue(_got_[0].equals(_sel_))



async def _settle_():
    """Let a selection scheduled with call_soon (setSelectedEntitiesAndNotifyOthers) run."""
    for _ in range(5):
        await asyncio.sleep(0)


class TestLinkpiNodeSelection(unittest.TestCase):
    """A linkpi node selection narrows the selection to the rows touching a selected node:
    either end, in the frame on screen (DT, 2026-10-09)."""

    def setUp(self):
        self.p2s = Polars2SVG()
        self.df  = pl.DataFrame({'s': ['a', 'a', 'b', 'c', 'd'],
                                 'd': ['b', 'c', 'c', 'd', 'a'],
                                 'n': [1, 2, 3, 4, 5]})

    def _view_(self, df, rels):
        return self.p2s.linkpi(self.p2s.linkp(df, rels, wxh=(256, 256)))

    def test_no_node_selected_is_the_whole_frame(self):
        v = self._view_(self.df, [('s', 'd')])
        self.assertTrue(v.mvc.selectedDataFrame().equals(self.df))

    def test_selected_nodes_give_the_rows_at_either_end(self):
        v = self._view_(self.df, [('s', 'd')])
        v.selected_entities = {'c'}
        self.assertEqual(v.mvc.selectedDataFrame()['n'].to_list(), [2, 3, 4])   # a-c, b-c, c-d
        v.selected_entities = {'a', 'd'}
        self.assertEqual(v.mvc.selectedDataFrame()['n'].to_list(), [1, 2, 4, 5])

    def test_integer_endpoints(self):
        df = pl.DataFrame({'s': [1, 1, 2, 3], 'd': [2, 3, 3, 1], 'n': [1, 2, 3, 4]})
        v  = self._view_(df, [('s', 'd')])
        v.selected_entities = {2}
        self.assertEqual(v.mvc.selectedDataFrame()['n'].to_list(), [1, 3])

    def test_a_tuple_endpoint_matches_on_its_joined_name(self):
        df = pl.DataFrame({'h': ['a', 'a', 'b'], 'p': [80, 443, 80], 'd': ['x', 'y', 'x']})
        v  = self._view_(df, [(('h', 'p'), 'd')])
        self.assertIn('a|443', set(v.graphs[0].nodes()))
        v.selected_entities = {'a|443'}
        _sel_ = v.mvc.selectedDataFrame()
        self.assertEqual(_sel_.columns, ['h', 'p', 'd'])          # the joined column stays internal
        self.assertEqual(_sel_.rows(), [('a', 443, 'y')])

    def test_several_relationships_are_each_matched(self):
        df = self.df.with_columns(t=pl.Series(['q', 'q', 'r', 'r', 'q']), u=pl.Series(['z', 'z', 'z', 'z', 'b']))
        v  = self._view_(df, [('s', 'd'), ('t', 'u')])
        v.selected_entities = {'b'}
        self.assertEqual(v.mvc.selectedDataFrame()['n'].to_list(), [1, 3, 5])   # s/d: 1 and 3; t/u: 5

    def test_it_narrows_the_frame_on_screen_not_the_input(self):
        v = self._view_(self.df, [('s', 'd')])
        asyncio.run(v.mvc.pushStack(v, self.df.filter(pl.col('n') >= 3)))
        v.selected_entities = {'c'}
        self.assertEqual(v.mvc.selectedDataFrame()['n'].to_list(), [3, 4])

    def test_selecting_calls_back_through_the_real_path(self):
        v    = self._view_(self.df, [('s', 'd')])
        _got_ = []
        v.mvc.onSelection(lambda df: _got_.append(df['n'].to_list()))
        async def _go_():
            v.selectEntities({'c'})                       # schedules mvc.selectionUpdate()
            await _settle_()
            v.selectEntities(set())
            await _settle_()
        asyncio.run(_go_())
        self.assertEqual(_got_, [[2, 3, 4], [1, 2, 3, 4, 5]])

    def test_two_linked_linkpis_share_the_selection_and_call_back_once(self):
        lp     = self.p2s.linkp(self.df, [('s', 'd')], wxh=(256, 256))
        layout = self.p2s.panelize([[lp, lp]])
        v1, v2 = [_v_ for _v_ in layout.mvc.view_refs.values() if hasattr(_v_, '_selectionRowMask_')]
        _got_  = []
        layout.mvc.onSelection(lambda df: _got_.append(df['n'].to_list()))
        async def _go_():
            v1.selectEntities({'d'})
            await _settle_()
        asyncio.run(_go_())
        self.assertEqual(v2.selected_entities, {'d'})
        self.assertEqual(_got_, [[4, 5]])
        self.assertEqual(layout.mvc.selectedDataFrame()['n'].to_list(), [4, 5])



class TestOnEveryViewAndTheLayout(unittest.TestCase):
    """selectedDataFrame() / onSelection() on panelize()'s container and on every view."""

    def setUp(self):
        self.p2s = Polars2SVG()
        self.df  = pl.DataFrame({'x': [1.0, 2.0, 3.0, 4.0], 'y': [1.0, 2.0, 3.0, 4.0],
                                 'c': ['a', 'b', 'a', 'b']})

    def test_every_view_class_carries_the_api(self):
        # Every JSComponent the interactive modules define is a view or a view's base, so a
        # view added later without the mixin fails here rather than silently lacking it.
        import inspect as _inspect_
        from panel.custom import JSComponent
        from polars2svg import interactive_controller, spreadlinepi, stack_control
        from polars2svg.interactive_controller import SelectionAPIMixin
        _views_ = {_c_ for _m_ in (interactive_controller, spreadlinepi, stack_control)
                   for _, _c_ in _inspect_.getmembers(_m_, _inspect_.isclass)
                   if issubclass(_c_, JSComponent) and _c_.__module__ == _m_.__name__}
        self.assertGreaterEqual(len(_views_), 14)       # 5 generic + 5 GPU, smallpi, linkpi, slpi, stack control
        for _c_ in _views_:
            self.assertTrue(issubclass(_c_, SelectionAPIMixin), _c_.__name__)
            self.assertTrue(callable(getattr(_c_, 'selectedDataFrame')), _c_.__name__)
            self.assertTrue(callable(getattr(_c_, 'onSelection')), _c_.__name__)

    def test_the_layout_reads_and_hears_its_stack(self):
        layout = self.p2s.panelize([[self.p2s.xyp(self.df, 'x', 'y', wxh=(128, 128)),
                                     self.p2s.histop(self.df, 'c', wxh=(128, 128))]])
        self.assertTrue(layout.selectedDataFrame().equals(self.df))
        _got_ = []
        _cb_  = _got_.append
        self.assertIs(layout.onSelection(_cb_), _cb_)
        xv = next(_v_ for _v_ in layout.mvc.view_refs.values() if type(_v_).__name__ == 'XYPI')
        asyncio.run(layout.mvc.pushStack(xv, self.df.filter(pl.col('c') == 'a')))
        self.assertEqual(layout.selectedDataFrame()['x'].to_list(), [1.0, 3.0])
        self.assertEqual(len(_got_), 1)
        self.assertTrue(_got_[0].equals(layout.selectedDataFrame()))
        for _v_ in layout.mvc.view_refs.values():       # every view agrees with the layout
            self.assertTrue(_v_.selectedDataFrame().equals(layout.selectedDataFrame()), type(_v_).__name__)

    def test_a_standalone_view(self):
        v = self.p2s.xypi(self.p2s.xyp(self.df, 'x', 'y', wxh=(128, 128)))
        _got_ = []
        v.onSelection(lambda df: _got_.append(df.height))
        asyncio.run(v.mvc.pushStack(v, self.df.head(1)))
        self.assertEqual(v.selectedDataFrame().height, 1)
        self.assertEqual(_got_, [1])

    def test_a_view_hears_only_its_own_stack(self):
        mvc = InteractionController()
        mvc.addStack('a', self.df)
        mvc.addStack('b', self.df)
        va, vb = _View_(), _View_()
        mvc.link(va, [], on='stack', stack='a')
        mvc.link(vb, [], on='stack', stack='b')
        _a_, _all_ = [], []
        mvc.onSelection(lambda df: _a_.append(df.height), stack='a')
        mvc.onSelection(lambda df: _all_.append(df.height))
        asyncio.run(mvc.pushStack(vb, self.df.head(1)))
        asyncio.run(mvc.pushStack(va, self.df.head(2)))
        self.assertEqual(_a_, [2])
        self.assertEqual(_all_, [1, 2])
        with self.assertRaisesRegex(ValueError, "no stack named 'c'"):
            mvc.onSelection(print, stack='c')

    def test_a_view_on_no_stack_says_so(self):
        sc = self.p2s.stack_controli(self.p2s.xyp(self.df, 'x', 'y', wxh=(128, 128)))
        self.assertIsNone(sc.mvc)
        with self.assertRaisesRegex(RuntimeError, 'STACKCONTROLI: this view is not on a stack'):
            sc.selectedDataFrame()
        with self.assertRaisesRegex(RuntimeError, 'not on a stack'):
            sc.onSelection(print)


if __name__ == '__main__':
    unittest.main()
