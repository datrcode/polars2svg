#
# F18 -- the stack's shelf of named frames and the labels stack frames carry (PLANNING.md
# §7 F18; design 20261026_stack_control_interactions.md, cited by Q-number).  These drive
# the controller directly; the widget's half is tests/test_stack_control.py.
#
import asyncio
import logging
import unittest

import polars as pl

from polars2svg import Polars2SVG
from polars2svg.interactive_controller import InteractionController
from polars2svg.stack_shelf import (_MAX_SHELF_, _MAX_SHELF_NAME_, baseColumns, cleanShelfName,
                                    missingColumns, setOperation)


class _View_:
    """A stack peer that records display() calls and shelf refreshes."""
    def __init__(self):
        self.displayed = []
        self.refreshes = 0

    async def display(self, df, dfs, index):
        self.displayed.append(df)

    def refreshShelf(self):
        self.refreshes += 1


def _df_():
    return pl.DataFrame({'a': [1, 2, 3, 4], 'b': ['w', 'x', 'y', 'z']})


class _ShelfBase_(unittest.TestCase):
    def setUp(self):
        self.df  = _df_()
        self.mvc = InteractionController()
        self.mvc.addStack('default', self.df)
        self.v   = _View_()
        self.mvc.link(self.v, [], on='stack', stack='default')

    def _push(self, df):
        asyncio.run(self.mvc.pushStack(self.v, df))
        return df


class TestColumnHelpers(unittest.TestCase):

    def test_internal_columns_are_not_base_columns(self):
        _df_ = pl.DataFrame({'a': [1], '__p2s_index__': [0], 'b': [2]})
        self.assertEqual(baseColumns(_df_), ['a', 'b'])

    def test_missing_columns_names_what_the_frame_lacks(self):
        _base_ = pl.DataFrame({'a': [1], 'b': [2], 'c': [3]})
        self.assertEqual(missingColumns(_base_, pl.DataFrame({'b': [2], 'x': [0]})), ['a', 'c'])
        self.assertEqual(missingColumns(_base_, _base_.with_columns(z=pl.lit(1))), [])


class TestPinFrame(_ShelfBase_):

    def test_default_shelves_the_displayed_frame_and_labels_it(self):
        # Diagram 5: the stack row of the shelved frame now shows the shelf name.
        _f1_ = self._push(self.df.head(2))
        self.assertIs(self.mvc.pinFrame('Named 3'), _f1_)
        self.assertIs(self.mvc.shelf()['Named 3'], _f1_)
        self.assertEqual(self.mvc.frameLabel('default', _f1_), 'Named 3')
        self.assertIsNone(self.mvc.frameLabel('default', self.df))

    def test_a_frame_from_python_need_not_be_on_the_stack(self):
        _other_ = self.df.tail(1)
        self.mvc.pinFrame('scan 1', _other_)
        self.assertIs(self.mvc.shelf()['scan 1'], _other_)
        self.assertIsNone(self.mvc.frameLabel('default', _other_))

    def test_entries_keep_the_order_shelved(self):
        # A new entry takes the first free row, so no existing entry moves (DT, 2026-10-27).
        for _i_ in range(3):
            self.mvc.pinFrame(f'n{_i_}', self.df.head(_i_ + 1))
        self.assertEqual(list(self.mvc.shelf()), ['n0', 'n1', 'n2'])

    def test_a_full_shelf_refuses(self):
        for _i_ in range(_MAX_SHELF_):
            self.mvc.pinFrame(f'n{_i_}', self.df.head(_i_ + 1))
        with self.assertRaisesRegex(ValueError, r'shelf is full \(4\)'):
            self.mvc.pinFrame('one more', self.df.tail(1))
        self.assertEqual(len(self.mvc.shelf()), _MAX_SHELF_)

    def test_max_shelf_zero_turns_the_shelf_off(self):
        self.mvc.max_shelf = 0
        with self.assertRaisesRegex(ValueError, 'shelf is off'):
            self.mvc.pinFrame('x')

    def test_a_taken_name_is_refused(self):
        # Q12: names are what Python looks frames up by; replacing would lose a frame.
        self.mvc.pinFrame('scan', self.df.head(1))
        with self.assertRaisesRegex(ValueError, '"scan" is already on the shelf'):
            self.mvc.pinFrame('scan', self.df.head(2))

    def test_the_same_frame_twice_is_refused_by_identity(self):
        # Q8: same object, not equal contents.
        _f_ = self.df.head(2)
        self.mvc.pinFrame('first', _f_)
        with self.assertRaisesRegex(ValueError, 'already shelved as "first"'):
            self.mvc.pinFrame('second', _f_)
        self.mvc.pinFrame('equal but not the same', self.df.head(2))
        self.assertEqual(len(self.mvc.shelf()), 2)

    def test_a_frame_missing_base_columns_is_refused(self):
        # Q15: every view on the stack must be able to draw it.
        with self.assertRaisesRegex(ValueError, 'missing column: b'):
            self.mvc.pinFrame('x', pl.DataFrame({'a': [1]}))

    def test_internal_columns_do_not_count_against_a_frame(self):
        _f_ = self.df.with_row_index('__p2s_index__').head(2)
        self.assertIs(self.mvc.pinFrame('with index', _f_), _f_)

    def test_bad_arguments(self):
        for _bad_ in ('', '   ', None, 3):
            with self.assertRaises(ValueError, msg=repr(_bad_)):
                self.mvc.pinFrame(_bad_)
        with self.assertRaisesRegex(TypeError, 'polars DataFrame'):
            self.mvc.pinFrame('x', {'a': [1]})
        self.assertEqual(self.mvc.shelf(), {})

    def test_a_bad_stack_name_says_who_asked(self):
        with self.assertRaisesRegex(ValueError, r"pinFrame\(\): no stack named 'nope'"):
            self.mvc.pinFrame('x', stack='nope')
        self.mvc.addStack('other', self.df)
        with self.assertRaisesRegex(ValueError, r'shelf\(\): 2 stacks, so name one'):
            self.mvc.shelf()

    def test_the_widget_is_told_to_redraw(self):
        self.mvc.pinFrame('x')
        self.mvc.unpinFrame('x')
        self.assertEqual(self.v.refreshes, 2)


class TestOneShelfPerStack(_ShelfBase_):

    def test_stacks_do_not_share_a_shelf(self):
        # Q17.
        self.mvc.addStack('dns', pl.DataFrame({'domain': ['a.com'], 'ip': ['1.2.3.4']}))
        self.mvc.pinFrame('flows', stack='default')
        self.mvc.pinFrame('names', stack='dns')
        self.assertEqual(list(self.mvc.shelf('default')), ['flows'])
        self.assertEqual(list(self.mvc.shelf('dns')), ['names'])

    def test_the_cap_applies_to_each_stack(self):
        self.mvc.addStack('other', self.df)
        for _i_ in range(_MAX_SHELF_):
            self.mvc.pinFrame(f'n{_i_}', self.df.head(_i_ + 1), stack='default')
        self.mvc.pinFrame('fits', stack='other')
        self.assertEqual(len(self.mvc.shelf('other')), 1)


class TestShelfSurvivesTheStack(_ShelfBase_):

    def test_replace_stack_leaves_the_shelf_alone(self):
        self.mvc.pinFrame('kept', self.df.head(1))
        asyncio.run(self.mvc.replaceStack('default', self.df.tail(2)))
        self.assertEqual(list(self.mvc.shelf()), ['kept'])

    def test_a_push_that_cuts_leaves_the_shelf_alone(self):
        # The shelf is the safety net for a component push, which still cuts.
        _f1_ = self._push(self.df.head(3))
        self.mvc.pinFrame('safe')
        asyncio.run(self.mvc.popStack(self.v))
        self._push(self.df.head(1))
        self.assertFalse(any(_d_ is _f1_ for _d_ in self.mvc.stacks['default']['dfs']))
        self.assertIs(self.mvc.shelf()['safe'], _f1_)

    def test_an_entry_whose_columns_no_longer_match_is_refused_on_use(self):
        # Q18 (Claude, for DT to confirm): checked again when the entry is used.
        _f_ = self.df.head(1)
        self.mvc.pinFrame('old', _f_)
        self.assertIsNone(self.mvc.entryRefusal('default', _f_))
        asyncio.run(self.mvc.replaceStack('default', pl.DataFrame({'q': [1]})))
        self.assertEqual(self.mvc.entryRefusal('default', _f_), 'columns no longer match: q')


class TestUnpinFrame(_ShelfBase_):

    def test_returns_the_frame_and_removes_it(self):
        _f_ = self.df.head(2)
        self.mvc.pinFrame('x', _f_)
        self.assertIs(self.mvc.unpinFrame('x'), _f_)
        self.assertEqual(self.mvc.shelf(), {})

    def test_an_unknown_name_raises(self):
        with self.assertRaisesRegex(ValueError, "no shelf entry named 'x'"):
            self.mvc.unpinFrame('x')

    def test_the_stack_row_keeps_the_name(self):
        # Diagram 2 / Q1: deleting the shelf entry leaves the stack frame's label.
        _f1_ = self._push(self.df.head(2))
        self.mvc.pinFrame('Named 1')
        self.mvc.unpinFrame('Named 1')
        self.assertEqual(self.mvc.frameLabel('default', _f1_), 'Named 1')


class TestFrameLabels(_ShelfBase_):

    def test_a_label_is_identity_guarded(self):
        _f1_ = self._push(self.df.head(2))
        self.mvc.pinFrame('n')
        self.assertIsNone(self.mvc.frameLabel('default', self.df.head(2)))   # equal, not it
        self.assertEqual(self.mvc.frameLabel('default', _f1_), 'n')

    def test_shelving_again_relabels(self):
        _f1_ = self._push(self.df.head(2))
        self.mvc.pinFrame('first')
        self.mvc.unpinFrame('first')
        self.mvc.pinFrame('second')
        self.assertEqual(self.mvc.frameLabel('default', _f1_), 'second')

    def test_a_label_follows_its_frame_through_rebase(self):
        # Q16: rebase (ctrl-shift-c) is replaceStack(current); the label goes with it.
        _f1_ = self._push(self.df.head(2))
        self.mvc.pinFrame('n')
        asyncio.run(self.mvc.replaceStack('default', _f1_))
        self.assertEqual(self.mvc.frameLabel('default', _f1_), 'n')

    def test_a_label_is_dropped_with_its_frame(self):
        _f1_ = self._push(self.df.head(2))
        self.mvc.pinFrame('n')
        asyncio.run(self.mvc.replaceStack('default', self.df))
        self.mvc._pruneFrameLabels_('default')
        self.assertEqual(self.mvc.frame_labels['default'], {})
        self.assertIsNone(self.mvc.frameLabel('default', _f1_))


class TestOnShelve(_ShelfBase_):

    def test_told_name_and_frame(self):
        _got_ = []
        self.mvc.onShelve(lambda name, df: _got_.append((name, df)))
        _f_ = self.mvc.pinFrame('x')
        self.assertEqual(_got_, [('x', _f_)])

    def test_returns_the_callback_so_it_decorates(self):
        def _cb_(name, df): pass
        self.assertIs(self.mvc.onShelve(_cb_), _cb_)

    def test_a_coroutine_callback_runs_without_a_loop(self):
        _got_ = []
        async def _cb_(name, df): _got_.append(name)
        self.mvc.onShelve(_cb_)
        self.mvc.pinFrame('x')
        self.assertEqual(_got_, ['x'])

    def test_a_coroutine_callback_is_scheduled_on_a_running_loop(self):
        _got_ = []
        async def _cb_(name, df): _got_.append(name)
        self.mvc.onShelve(_cb_)
        async def _cell_():
            self.mvc.pinFrame('x')
            self.assertEqual(_got_, [])
            await asyncio.sleep(0)
        asyncio.run(_cell_())
        self.assertEqual(_got_, ['x'])

    def test_stack_limits_it(self):
        self.mvc.addStack('other', self.df)
        _got_ = []
        self.mvc.onShelve(lambda name, df: _got_.append(name), stack='other')
        self.mvc.pinFrame('here', stack='default')
        self.mvc.pinFrame('there', stack='other')
        self.assertEqual(_got_, ['there'])

    def test_one_that_raises_is_logged_and_the_shelf_kept(self):
        def _bad_(name, df): raise RuntimeError('boom')
        self.mvc.onShelve(_bad_)
        with self.assertLogs('polars2svg_logger', level=logging.ERROR) as _log_:
            self.mvc.pinFrame('x')
        self.assertIn('onShelve(): callback', _log_.output[0])
        self.assertEqual(list(self.mvc.shelf()), ['x'])

    def test_not_callable(self):
        with self.assertRaisesRegex(TypeError, 'expected a callable'):
            self.mvc.onShelve('nope')


class TestOnTheLayout(unittest.TestCase):

    def test_panelize_container_carries_the_shelf_api(self):
        p2s = Polars2SVG()
        df  = pl.DataFrame({'x': [1.0, 2.0, 3.0], 'y': [1.0, 2.0, 3.0]})
        layout = p2s.panelize([[p2s.xyp(df, 'x', 'y', wxh=(128, 128))]])
        _f_ = layout.pinFrame('all')
        self.assertTrue(_f_.equals(df))
        self.assertEqual(list(layout.shelf()), ['all'])
        self.assertIs(layout.unpinFrame('all'), _f_)
        _cb_ = layout.onShelve(lambda name, df: None)
        self.assertTrue(callable(_cb_))


# ── step 3: deletion, set operations, pushes on top ──────────────────────────────

def _run(co):
    return asyncio.run(co)


class TestSetOperation(unittest.TestCase):
    """Q5: built from A, so A's duplicate rows survive; matched on the base's columns."""

    A = pl.DataFrame({'k': [1, 1, 2, 3], 'v': ['a', 'a', 'b', 'c']})
    B = pl.DataFrame({'k': [2, 3, 4, 4], 'v': ['b', 'c', 'd', 'd']})
    C = ['k', 'v']

    def test_union_is_a_then_what_b_adds(self):
        _r_ = setOperation('union', self.A, self.B, self.C)
        self.assertEqual(_r_['k'].to_list(), [1, 1, 2, 3, 4, 4])

    def test_intersect_keeps_a_duplicates(self):
        _a_ = self.A.vstack(pl.DataFrame({'k': [2], 'v': ['b']}))
        self.assertEqual(setOperation('intersect', _a_, self.B, self.C)['k'].to_list(), [2, 3, 2])

    def test_subtract_is_an_anti_join(self):
        self.assertEqual(setOperation('subtract', self.A, self.B, self.C)['k'].to_list(), [1, 1])

    def test_nulls_are_equal(self):
        _a_ = pl.DataFrame({'k': [None, 1], 'v': ['x', 'y']})
        _b_ = pl.DataFrame({'k': [None], 'v': ['x']}, schema={'k': pl.Int64, 'v': pl.String})
        self.assertEqual(len(setOperation('intersect', _a_, _b_, self.C)), 1)

    def test_internal_columns_are_ignored_and_dropped(self):
        _a_ = self.A.with_row_index('__p2s_index__')
        _b_ = self.B.with_row_index('__p2s_index__')
        for _op_ in ('union', 'intersect', 'subtract'):
            _r_ = setOperation(_op_, _a_, _b_, self.C)
            self.assertNotIn('__p2s_index__', _r_.columns, _op_)
        self.assertEqual(len(setOperation('intersect', _a_, _b_, self.C)), 2)

    def test_unknown_op(self):
        with self.assertRaisesRegex(ValueError, 'unknown op'):
            setOperation('xor', self.A, self.B, self.C)


class TestCleanShelfName(unittest.TestCase):

    def test_cleans_what_the_browser_sends(self):
        self.assertEqual(cleanShelfName('  scan\t1\n '), 'scan 1')
        self.assertEqual(cleanShelfName('a\x00b\u200bc'), 'abc')
        self.assertEqual(len(cleanShelfName('x' * 500)), _MAX_SHELF_NAME_)
        for _bad_ in ('', '   ', '\x07', None, 7):
            self.assertIsNone(cleanShelfName(_bad_), repr(_bad_))


class _OpsBase_(_ShelfBase_):
    """Diagram 1: Base, Filter 1 (displayed), Filter 2; Named 1 and Named 2 shelved."""

    def setUp(self):
        super().setUp()
        self.df   = pl.DataFrame({'a': list(range(10)), 'b': [str(_i_) for _i_ in range(10)]})
        self.mvc  = InteractionController()
        self.mvc.addStack('default', self.df)
        self.v    = _View_()
        self.mvc.link(self.v, [], on='stack', stack='default')
        self.base = self.df
        self.f1   = self._push(self.df.filter(pl.col('a') < 6))
        self.f2   = self._push(self.f1.filter(pl.col('a') < 3))
        _run(self.mvc.popStack(self.v))
        self.n1   = self.mvc.pinFrame('Named 1', self.df.filter(pl.col('a') >= 4))
        self.n2   = self.mvc.pinFrame('Named 2', self.df.filter(pl.col('a') == 9))
        self.v.displayed.clear()
        self.sel  = []
        self.mvc.onSelection(self.sel.append)

    @property
    def s(self):
        return self.mvc.stacks['default']

    def _labels(self):
        return [self.mvc.frameLabel('default', _d_) for _d_ in self.s['dfs']]


class TestPushFromTheShelf(_OpsBase_):

    def test_plain_click_pushes_on_top_without_cutting(self):
        # Diagram 2.
        self.assertIsNone(_run(self.mvc.useShelfEntry('default', 'Named 1')))
        self.assertEqual(len(self.s['dfs']), 4)
        self.assertIs(self.s['dfs'][2], self.f2)
        self.assertIs(self.s['dfs'][3], self.n1)
        self.assertEqual(self.s['index'], 3)
        self.assertEqual(self._labels(), [None, None, None, 'Named 1'])
        self.assertIs(self.v.displayed[-1], self.n1)
        self.assertEqual(len(self.sel), 1)

    def test_an_entry_already_on_the_stack_is_jumped_to(self):
        # Q9: anywhere on the stack; nothing on it changes.
        _run(self.mvc.useShelfEntry('default', 'Named 1'))
        _run(self.mvc.jumpTo('default', 0))
        self.assertIsNone(_run(self.mvc.useShelfEntry('default', 'Named 1')))
        self.assertEqual(len(self.s['dfs']), 4)
        self.assertEqual(self.s['index'], 3)

    def test_the_stack_limit_refuses(self):
        self.mvc.max_stack_depth = 3
        self.assertEqual(_run(self.mvc.useShelfEntry('default', 'Named 1')),
                         'stack is at its 3-frame limit')
        self.assertEqual(len(self.s['dfs']), 3)

    def test_an_incompatible_entry_refuses(self):
        # Q18.
        _run(self.mvc.replaceStack('default', pl.DataFrame({'q': [1]})))
        self.assertEqual(_run(self.mvc.useShelfEntry('default', 'Named 1')),
                         'columns no longer match: q')

    def test_an_unknown_entry(self):
        self.assertEqual(_run(self.mvc.useShelfEntry('default', 'nope')), 'no such shelf entry')


class TestDeleteFrame(_OpsBase_):

    def test_the_displayed_frame_cannot_be_deleted(self):
        self.assertEqual(_run(self.mvc.deleteFrame('default', 1)), 'the displayed frame cannot be deleted')
        self.assertEqual(len(self.s['dfs']), 3)

    def test_deleting_the_base_makes_the_next_frame_the_base(self):
        # Diagram 3.
        self.assertIsNone(_run(self.mvc.deleteFrame('default', 0)))
        self.assertEqual(self.s['dfs'], [self.f1, self.f2])
        self.assertEqual(self.s['index'], 0)
        self.assertIs(self.mvc.stackTopDataFrame(self.v), self.f1)
        self.assertIs(self.v.displayed[-1], self.f1)
        self.assertEqual(self.sel, [])          # the displayed frame did not change

    def test_deleting_from_the_middle_keeps_the_frames_above(self):
        # Diagram 6: Base, F1, F2, F3, F4 (displayed); delete F2.
        _run(self.mvc.jumpTo('default', 2))
        _f3_ = self._push(self.f2.head(2))
        _f4_ = self._push(_f3_.head(1))
        self.assertIsNone(_run(self.mvc.deleteFrame('default', 2)))
        self.assertEqual(self.s['dfs'], [self.base, self.f1, _f3_, _f4_])
        self.assertEqual(self.s['index'], 3)

    def test_deleting_above_the_displayed_frame_leaves_the_index(self):
        self.assertIsNone(_run(self.mvc.deleteFrame('default', 2)))
        self.assertEqual(self.s['dfs'], [self.base, self.f1])
        self.assertEqual(self.s['index'], 1)

    def test_a_deleted_frame_takes_its_label_with_it(self):
        _run(self.mvc.useShelfEntry('default', 'Named 1'))
        _run(self.mvc.jumpTo('default', 1))
        _run(self.mvc.deleteFrame('default', 3))
        self.assertNotIn(id(self.n1), self.mvc.frame_labels['default'])

    def test_out_of_range(self):
        self.assertEqual(_run(self.mvc.deleteFrame('default', 9)), 'no such frame')


class TestCombine(_OpsBase_):

    def test_intersect_with_a_shelf_entry(self):
        # Diagram 4: Filter 1 (a < 6) with Named 1 (a >= 4), labelled with the operand.
        self.assertIsNone(_run(self.mvc.useShelfEntry('default', 'Named 1', 'intersect')))
        self.assertEqual(self.s['dfs'][-1]['a'].to_list(), [4, 5])
        self.assertIs(self.s['dfs'][2], self.f2)
        self.assertEqual(self._labels()[-1], '\u2229 Named 1')
        self.assertEqual(self.s['index'], 3)

    def test_subtract_a_stack_frame_above_keeps_the_operand(self):
        # Diagram 7: Filter 1 - Filter 2; Filter 2 is not cut, and an unnamed operand
        # leaves the operator alone.
        self.assertIsNone(_run(self.mvc.useStackFrame('default', 2, 'subtract')))
        self.assertEqual(self.s['dfs'][-1]['a'].to_list(), [3, 4, 5])
        self.assertIs(self.s['dfs'][2], self.f2)
        self.assertEqual(self._labels()[-1], '\u2212')

    def test_union_two_scans_from_the_shelf(self):
        # Q13: push scan 1, then ctrl-click scan 2.
        _run(self.mvc.useShelfEntry('default', 'Named 2'))
        self.assertIsNone(_run(self.mvc.useShelfEntry('default', 'Named 1', 'union')))
        self.assertEqual(sorted(self.s['dfs'][-1]['a'].to_list()), [4, 5, 6, 7, 8, 9])
        self.assertEqual(self._labels()[-1], '\u222a Named 1')

    def test_the_operand_label_is_used_verbatim(self):
        _run(self.mvc.useShelfEntry('default', 'Named 1', 'intersect'))      # "∩ Named 1"
        _run(self.mvc.jumpTo('default', 0))
        _run(self.mvc.useStackFrame('default', 3, 'subtract'))
        self.assertEqual(self._labels()[-1], '\u2212 \u2229 Named 1')

    def test_no_change_and_empty_change_nothing(self):
        # Diagram 8.
        _n_ = len(self.s['dfs'])
        self.assertEqual(_run(self.mvc.useStackFrame('default', 1, 'union')), 'no change')
        self.assertEqual(_run(self.mvc.useStackFrame('default', 1, 'subtract')), 'no change')
        self.assertEqual(_run(self.mvc.useStackFrame('default', 0, 'intersect')), 'no change')   # F1 ⊂ base
        self.assertEqual(_run(self.mvc.useShelfEntry('default', 'Named 2', 'intersect')), 'empty result')
        self.assertEqual(len(self.s['dfs']), _n_)
        self.assertEqual(self.sel, [])

    def test_a_result_equal_to_another_frame_is_a_copy(self):
        # Q7: Filter 1 ∪ Base = Base, pushed as a new frame.
        self.assertIsNone(_run(self.mvc.useStackFrame('default', 0, 'union')))
        self.assertTrue(self.s['dfs'][-1].equals(self.base))
        self.assertIsNot(self.s['dfs'][-1], self.base)

    def test_mismatched_types_refuse_rather_than_raise(self):
        _odd_ = pl.DataFrame({'a': ['x'], 'b': ['0']})
        self.mvc.pinFrame('odd', _odd_)
        with self.assertLogs('polars2svg_logger', level=logging.WARNING):
            self.assertEqual(_run(self.mvc.useShelfEntry('default', 'odd', 'union')),
                             'cannot union these frames')

    def test_a_plain_click_on_a_stack_frame_jumps_and_notifies(self):
        self.assertIsNone(_run(self.mvc.useStackFrame('default', 2)))
        self.assertEqual(self.s['index'], 2)
        self.assertEqual(len(self.sel), 1)
        self.assertIsNone(_run(self.mvc.useStackFrame('default', 2)))
        self.assertEqual(len(self.sel), 1)       # no move, no notification


class TestShelveAndDeleteFromTheWidget(_OpsBase_):

    def test_shelve_the_displayed_frame(self):
        # Diagram 5.
        _got_ = []
        self.mvc.onShelve(lambda name, df: _got_.append(name))
        self.assertIsNone(_run(self.mvc.shelveDisplayed('default', ' Named 3 ')))
        self.assertEqual(list(self.mvc.shelf()), ['Named 1', 'Named 2', 'Named 3'])
        self.assertEqual(self._labels(), [None, 'Named 3', None])
        self.assertEqual(_got_, ['Named 3'])

    def test_refusals_are_reasons(self):
        self.assertEqual(_run(self.mvc.shelveDisplayed('default', '  ')), 'a shelf name cannot be empty')
        self.assertEqual(_run(self.mvc.shelveDisplayed('default', 'Named 1')),
                         '"Named 1" is already on the shelf')
        _run(self.mvc.shelveDisplayed('default', 'ok'))
        self.assertEqual(_run(self.mvc.shelveDisplayed('default', 'again')), 'already shelved as "ok"')

    def test_delete_a_shelf_entry(self):
        _run(self.mvc.useShelfEntry('default', 'Named 1'))
        self.assertIsNone(_run(self.mvc.deleteShelfEntry('default', 'Named 1')))
        self.assertEqual(list(self.mvc.shelf()), ['Named 2'])
        self.assertEqual(self._labels()[-1], 'Named 1')       # diagram 2: the stack keeps it
        self.assertEqual(_run(self.mvc.deleteShelfEntry('default', 'Named 1')), 'no such shelf entry')


class TestLinkpiKeepsHeldLevels(unittest.TestCase):
    """A push on top while a lower frame is displayed must not re-render the levels in
    between: linkpi would lose their arranged layouts."""

    def test_levels_between_are_moved_through_not_re_pushed(self):
        p2s = Polars2SVG()
        df  = pl.DataFrame({'src': ['a', 'b', 'c', 'd'], 'dst': ['b', 'c', 'd', 'a']})
        layout = p2s.panelize([[p2s.linkpi(p2s.linkp(df, [('src', 'dst')], wxh=(256, 256)))]])
        mvc = layout.mvc
        w   = next(_v_ for _v_ in mvc.view_refs.values() if type(_v_).__name__.startswith('LINKPI'))
        _run(mvc.pushStack(w, df.head(3)))
        _run(mvc.pushStack(w, df.head(2)))
        _l2_ = w.dfs_layout[2]
        _run(mvc.jumpTo('default', 1))
        _run(mvc.pushOnTop('default', df.tail(2), 'n'))
        self.assertEqual(w.df_level, 3)
        self.assertEqual(len(w.dfs), 4)
        self.assertIs(w.dfs_layout[2], _l2_)
        self.assertIs(w.dfs[3], mvc.stacks['default']['dfs'][3])


if __name__ == '__main__':
    unittest.main()
