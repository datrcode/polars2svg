"""stack_controli widget: frame layout (fit + all three overflow cases),
click-to-select routing through the InteractionController, display() cache
management, and sketchHtml().

Placement follows the "Stack Control Requirements" worked out in
polars2svg_prototyping/stack_control_shrinkage.ipynb: the current index always
renders; base (index 0) and top (index n-1) are tried next, with top only
attempted once base has succeeded; then the cluster grows outward from the
current index one frame at a time. At most two "... N stack frame(s) ..."
skip labels are ever needed. See TestStackControlPlacement for the exact
worked traces (see polars2svg.stack_control._placement).

Overflow geometry notes (wxh=(160, 256), insets=(2, 2), hgap=4, txt_h=10):
  slot   = component_h + 4
  avail  = 256 - 2*2 - component_h - headerHeight()
  A stack of n frames fits iff (n-1)*slot <= avail; larger stacks trigger the
  outward-fill placement with skip labels ("... N stack frames ...").

The indicator header (MLX/CUDA availability) eats into the same vertical budget,
so the boundary tests below size their component from headerHeight() instead of a
literal -- adding a third indicator row shifts them automatically.
"""
import asyncio
import json
import re
import unittest

import polars as pl
from view_js_utils import component_markup

from polars2svg import Polars2SVG
from polars2svg.interactive_controller import InteractionController
from polars2svg.stack_control import _CONFIRM_S_, headerHeight, shelfHeight, _placement

_HGAP_  = 4
_TXT_H_ = 10
_ELL_H_ = 2 * _HGAP_ + _TXT_H_     # height of one "... N stack frames ..." label
_STACK_H_ = 256                    # wxh[1] default


def _make_df(n=12):
    return pl.DataFrame({'x': [float(i) for i in range(n)],
                         'y': [float(i % 5) for i in range(n)]})


class _PeerView:
    """Minimal stack peer that records display() calls."""
    def __init__(self):
        self.display_calls = []

    async def display(self, df, dfs, index):
        self.display_calls.append((df, index))


class _StackControlBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p2s = Polars2SVG()

    def _component(self, h=60):
        return self.p2s.xyp(_make_df(), 'x', 'y', wxh=(120, h))

    # Frame budget, with slot = component_h + hgap:  avail = 256 - slot - headerHeight()
    def _budget(self):
        return _STACK_H_ - headerHeight(_TXT_H_)

    def _component_h_fitting(self, n):
        """Tallest component where a stack of n frames still fits without overflow:
        (n-1)*slot <= avail  <=>  n*slot <= 256 - header."""
        slot = self._budget() // n
        return slot - _HGAP_

    def _component_h_overflow(self):
        """A component small enough that a 10-frame stack overflows while still
        leaving room for a multi-frame cluster around the selection."""
        return self._budget() // 7 - _HGAP_

    def _skips(self, sc):
        """[(count, noun)] for each '... N stack frame(s) ...' label, in render order."""
        return [(int(m.group(1)), m.group(2))
                for m in re.finditer(r'\.\.\. (\d+) stack (frames?) \.\.\.', sc.mod_inner)]

    def _assertSkipsAccountForAll(self, sc, n):
        """Every unrendered frame is covered by exactly one skip label, correctly pluralized."""
        _rendered_ = {fm[2] for fm in sc._frame_map_}
        _skips_    = self._skips(sc)
        self.assertEqual(sum(c for c, _ in _skips_), n - len(_rendered_))
        for _count_, _noun_ in _skips_:
            self.assertEqual(_noun_, 'frame' if _count_ == 1 else 'frames')

    def _component_h_one_skip(self, n):
        """Tallest component where a stack of n overflows with exactly one frame
        skipped: (n-2)*slot + ELL_H <= avail < (n-1)*slot."""
        slot = (self._budget() - _ELL_H_) // (n - 1)
        self.assertGreater(n * slot, self._budget(),
                           f'geometry no longer admits a single-skip case for n={n:d}')
        return slot - _HGAP_

    def _stack_of(self, n):
        """A stack of n distinct dataframes (base is largest)."""
        df = _make_df()
        return [df.head(len(df) - i) for i in range(n)]

    def _make_sc(self, dfs=None, index=0, component=None, **kwargs):
        # The frame-stack geometry below is calibrated without a shelf; the shelf has its
        # own tests (TestStackControlShelf), which pass max_shelf explicitly.
        kwargs.setdefault('max_shelf', 0)
        component = component or self._component()
        if dfs is None:
            return self.p2s.stack_controli(component, **kwargs)
        mvc = InteractionController()
        mvc.addStack('default', dfs[0])
        mvc.stacks['default'] = {'dfs': list(dfs), 'index': index}
        sc = self.p2s.stack_controli(component, mvc=mvc, **kwargs)
        return sc, mvc


class TestStackControlConstruction(_StackControlBase):

    def test_initial_render_single_base_frame(self):
        sc = self._make_sc()
        self.assertIn('<svg', sc.mod_inner)
        self.assertEqual(len(sc._frame_map_), 1)
        self.assertEqual(sc._frame_map_[0][2], 0)

    def test_initial_render_shows_row_count(self):
        sc = self._make_sc()
        self.assertIn('Rows', sc.mod_inner)

    def test_sketchHtml_returns_current_frame(self):
        sc = self._make_sc()
        self.assertEqual(sc.sketchHtml(), sc.mod_inner)

    def test_construction_with_populated_stack_renders_all_frames(self):
        sc, _ = self._make_sc(dfs=self._stack_of(3), index=1)
        self.assertEqual(len(sc._frame_map_), 3)
        self.assertEqual(sorted(fm[2] for fm in sc._frame_map_), [0, 1, 2])

    def test_base_and_top_sublabels_present(self):
        sc, _ = self._make_sc(dfs=self._stack_of(3), index=0)
        self.assertIn('base', sc.mod_inner)
        self.assertIn('top',  sc.mod_inner)

    def test_middle_frame_gets_position_of_n_label(self):
        sc, _ = self._make_sc(dfs=self._stack_of(3), index=1)
        self.assertIn('2 of 3', sc.mod_inner)


class TestStackControlOverflowLayouts(_StackControlBase):
    """Ten frames cannot fit; the widget must render base + top + a cluster
    around the selected index with skip labels for the rest."""

    def _overflow_sc(self, index):
        return self._make_sc(dfs=self._stack_of(10), index=index,
                             component=self._component(h=self._component_h_overflow()))

    def test_selected_near_base_packs_from_base(self):
        sc, _ = self._overflow_sc(index=1)
        rendered = sorted(fm[2] for fm in sc._frame_map_)
        self.assertIn(0, rendered)
        self.assertIn(9, rendered)
        self.assertIn(1, rendered)
        self.assertLess(len(rendered), 10)
        self.assertEqual(len(self._skips(sc)), 1)   # single label, near the top
        self._assertSkipsAccountForAll(sc, 10)

    def test_selected_near_top_packs_from_top(self):
        sc, _ = self._overflow_sc(index=8)
        rendered = sorted(fm[2] for fm in sc._frame_map_)
        self.assertIn(0, rendered)
        self.assertIn(9, rendered)
        self.assertIn(8, rendered)
        self.assertEqual(len(self._skips(sc)), 1)   # single label, near the base
        self._assertSkipsAccountForAll(sc, 10)

    def test_selected_in_middle_gets_two_skip_labels(self):
        sc, _ = self._overflow_sc(index=5)
        rendered = sorted(fm[2] for fm in sc._frame_map_)
        self.assertIn(5, rendered)
        self.assertEqual(len(self._skips(sc)), 2)   # labels above and below the cluster
        self._assertSkipsAccountForAll(sc, 10)

    def test_singular_skip_label_uses_frame_not_frames(self):
        # size the component so six frames overflow with exactly one skipped
        sc, _ = self._make_sc(dfs=self._stack_of(6), index=1,
                              component=self._component(h=self._component_h_one_skip(6)))
        self.assertIn('... 1 stack frame ...', sc.mod_inner)
        self.assertNotIn('1 stack frames', sc.mod_inner)

    def test_frame_map_entries_do_not_overlap(self):
        sc, _ = self._overflow_sc(index=5)
        spans = sorted((fm[0], fm[1]) for fm in sc._frame_map_)
        for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
            self.assertLessEqual(a1, b0)


class TestStackControlClickRouting(_StackControlBase):

    def _click(self, sc, y):
        sc.click_y = int(y)
        asyncio.run(sc.applyClickOp(None))

    def test_click_on_frame_updates_stack_index(self):
        dfs = self._stack_of(3)
        sc, mvc = self._make_sc(dfs=dfs, index=0)
        _target_ = next(fm for fm in sc._frame_map_ if fm[2] == 2)
        self._click(sc, (_target_[0] + _target_[1]) // 2)
        self.assertEqual(mvc.stacks['default']['index'], 2)

    def test_click_propagates_display_to_stack_peers(self):
        dfs = self._stack_of(3)
        sc, mvc = self._make_sc(dfs=dfs, index=0)
        peer = _PeerView()
        mvc.view_stack[id(peer)] = 'default'
        mvc.view_refs[id(peer)]  = peer
        _target_ = next(fm for fm in sc._frame_map_ if fm[2] == 1)
        self._click(sc, (_target_[0] + _target_[1]) // 2)
        self.assertEqual(len(peer.display_calls), 1)
        self.assertIs(peer.display_calls[0][0], dfs[1])
        self.assertEqual(peer.display_calls[0][1], 1)

    def test_click_outside_frames_is_ignored(self):
        dfs = self._stack_of(2)
        sc, mvc = self._make_sc(dfs=dfs, index=1)
        _max_bot_ = max(fm[1] for fm in sc._frame_map_)
        _gap_y_   = next((int(fm0[1]) + 1 for fm0, fm1 in
                          zip(sorted(sc._frame_map_), sorted(sc._frame_map_)[1:])
                          if fm1[0] - fm0[1] > 2), int(_max_bot_) + 1000)
        self._click(sc, _gap_y_)
        self.assertEqual(mvc.stacks['default']['index'], 1)

    def test_click_without_mvc_is_noop(self):
        '''A click on a frame, with no controller to tell: nothing to change, and nothing
        changes -- not the frames, not the drawing.'''
        sc = self._make_sc()
        self.assertIsNone(sc.mvc)
        _frames_, _drawn_ = list(sc._frame_map_), sc.mod_inner
        _y_ = (sc._frame_map_[0][0] + sc._frame_map_[0][1]) // 2
        self._click(sc, _y_)
        self.assertEqual((sc._frame_map_, sc.mod_inner), (_frames_, _drawn_))


class TestStackControlKeyOps(_StackControlBase):
    """Keyboard shortcuts: 'c' collapses to base+current, ctrl+shift+c rebases
    the visible dataframe as the new base, and 'h' toggles the help overlay."""

    def _key(self, sc, op):
        sc.key_op_finished = op
        asyncio.run(sc.applyKeyOp(None))

    def test_collapse_reduces_to_base_and_current(self):
        dfs = self._stack_of(4)
        sc, mvc = self._make_sc(dfs=dfs, index=2)
        self._key(sc, 'collapse')
        s = mvc.stacks['default']
        self.assertEqual(len(s['dfs']), 2)
        self.assertIs(s['dfs'][0], dfs[0])   # base unchanged
        self.assertIs(s['dfs'][1], dfs[2])   # previously-visible frame on top
        self.assertEqual(s['index'], 1)

    def test_collapse_when_viewing_base_leaves_single_frame(self):
        dfs = self._stack_of(4)
        sc, mvc = self._make_sc(dfs=dfs, index=0)
        self._key(sc, 'collapse')
        s = mvc.stacks['default']
        self.assertEqual(len(s['dfs']), 1)   # no duplicated base frame
        self.assertIs(s['dfs'][0], dfs[0])
        self.assertEqual(s['index'], 0)

    def test_collapse_propagates_to_peers(self):
        dfs = self._stack_of(4)
        sc, mvc = self._make_sc(dfs=dfs, index=2)
        peer = _PeerView()
        mvc.view_stack[id(peer)] = 'default'
        mvc.view_refs[id(peer)]  = peer
        self._key(sc, 'collapse')
        self.assertEqual(len(peer.display_calls), 1)
        self.assertIs(peer.display_calls[0][0], dfs[2])
        self.assertEqual(peer.display_calls[0][1], 1)

    def test_rebase_sets_current_as_new_base(self):
        dfs = self._stack_of(4)
        sc, mvc = self._make_sc(dfs=dfs, index=2)
        self._key(sc, 'rebase')
        s = mvc.stacks['default']
        self.assertEqual(len(s['dfs']), 1)
        self.assertIs(s['dfs'][0], dfs[2])   # visible frame is now the sole base
        self.assertEqual(s['index'], 0)

    def test_rebase_propagates_to_peers(self):
        dfs = self._stack_of(4)
        sc, mvc = self._make_sc(dfs=dfs, index=2)
        peer = _PeerView()
        mvc.view_stack[id(peer)] = 'default'
        mvc.view_refs[id(peer)]  = peer
        self._key(sc, 'rebase')
        self.assertEqual(len(peer.display_calls), 1)
        self.assertIs(peer.display_calls[0][0], dfs[2])
        self.assertEqual(peer.display_calls[0][1], 0)

    def test_collapse_does_not_serve_stale_tile_on_id_collision(self):
        # Simulate a reused id(): the kept frame's cache slot holds another frame's
        # render. The identity guard must re-render rather than serve the stale tile.
        dfs = self._stack_of(4)
        sc, _ = self._make_sc(dfs=dfs, index=2)
        sc._svg_cache_[id(dfs[2])] = (_make_df(), '<svg id="STALE_TILE"></svg>')
        self._key(sc, 'collapse')
        self.assertNotIn('STALE_TILE', sc.mod_inner)
        self.assertIs(sc._svg_cache_[id(dfs[2])][0], dfs[2])   # slot rebound to the kept df

    def test_rebase_does_not_serve_stale_tile_on_id_collision(self):
        dfs = self._stack_of(4)
        sc, _ = self._make_sc(dfs=dfs, index=2)
        sc._svg_cache_[id(dfs[2])] = (_make_df(), '<svg id="STALE_TILE"></svg>')
        self._key(sc, 'rebase')
        self.assertNotIn('STALE_TILE', sc.mod_inner)

    def test_key_op_finished_is_reset_after_apply(self):
        dfs = self._stack_of(3)
        sc, _ = self._make_sc(dfs=dfs, index=1)
        self._key(sc, 'collapse')
        self.assertEqual(sc.key_op_finished, '')

    def test_unknown_key_op_is_ignored(self):
        dfs = self._stack_of(3)
        sc, mvc = self._make_sc(dfs=dfs, index=1)
        self._key(sc, 'nonsense')
        self.assertEqual(len(mvc.stacks['default']['dfs']), 3)   # untouched

    def test_key_op_without_mvc_is_noop(self):
        '''The key is consumed -- reset, so an identical next press re-fires -- and nothing
        else changes.'''
        sc = self._make_sc()
        self.assertIsNone(sc.mvc)
        _frames_, _drawn_ = list(sc._frame_map_), sc.mod_inner
        self._key(sc, 'collapse')
        self.assertEqual(sc.key_op_finished, '')
        self.assertEqual((sc._frame_map_, sc.mod_inner), (_frames_, _drawn_))

    def test_help_overlay_is_hidden_by_default_and_driven_by_display(self):
        """The help chrome ships, the widget builds a node for it, and it is hidden by
        `display` rather than parked off-screen.

        The overlay is wider than this (narrow) widget and the root <svg> is
        overflow:visible, so an off-screen copy would spill past the left edge into the
        neighbouring component instead of being clipped away -- which is why
        `translate(-1000 ...)`, the idiom LINKPI uses for its own help, is wrong here.

        The markup itself used to be concatenated into `_template`; since the
        JSComponent port (PLANNING.md W1) it travels as the `kbd_help_svg` param, because
        an ESM module has no markup to concatenate it into.  Asserted in the two places
        it now lives rather than pretending it is still one string.
        """
        sc = self._make_sc()
        _markup_ = component_markup(sc)
        self.assertEqual(sc.help_display, 'none')          # not rendered until toggled
        # the chrome itself -- its spaces are no-break spaces, which SVG keeps
        self.assertIn('toggle help', sc.kbd_help_svg.replace('\u00a0', ' '))
        self.assertIn('keyboardhelp', _markup_)            # ...and the node it goes into
        self.assertIn('kbd_help_svg', _markup_)            # ...wired to the param
        self.assertIn('help_display', _markup_)            # ...shown/hidden by this one
        self.assertNotIn('translate(-1000', _markup_)
        self.assertNotIn('translate(-1000', sc.kbd_help_svg)


class TestStackControlDisplay(_StackControlBase):

    def test_display_rerenders_frame_map(self):
        dfs = self._stack_of(3)
        sc, _ = self._make_sc(dfs=dfs[:1], index=0)
        asyncio.run(sc.display(dfs[2], dfs, 2))
        self.assertEqual(len(sc._frame_map_), 3)

    def test_display_ignores_mismatched_df(self):
        dfs = self._stack_of(3)
        sc, _ = self._make_sc(dfs=dfs, index=0)
        _before_ = sc.mod_inner
        asyncio.run(sc.display(dfs[0], dfs, 2))   # df is not dfs[2] → ignored
        self.assertEqual(sc.mod_inner, _before_)

    def test_display_prunes_cache_of_dropped_frames(self):
        dfs = self._stack_of(4)
        # all four frames must render (and so be cached) -- size the component to fit
        sc, _ = self._make_sc(dfs=dfs, index=0,
                              component=self._component(h=self._component_h_fitting(4)))
        # cache holds every stack frame (plus the component's df_orig from construction)
        self.assertTrue({id(d) for d in dfs} <= set(sc._svg_cache_.keys()))
        shrunk = dfs[:2]
        asyncio.run(sc.display(shrunk[1], shrunk, 1))
        self.assertEqual(set(sc._svg_cache_.keys()), {id(d) for d in shrunk})

    def test_display_reuses_cached_tiles(self):
        dfs = self._stack_of(2)
        sc, _ = self._make_sc(dfs=dfs, index=0)
        _tile_ = sc._svg_cache_[id(dfs[1])]
        asyncio.run(sc.display(dfs[0], dfs, 0))
        self.assertIs(sc._svg_cache_[id(dfs[1])], _tile_)

    def test_display_identity_guard_rerenders_on_id_collision(self):
        # A cache slot holding another object's render (as a reused id() would)
        # must not be served: display re-renders and rebinds the slot to the real df.
        dfs = self._stack_of(2)
        sc, _ = self._make_sc(dfs=dfs, index=1)
        sc._svg_cache_[id(dfs[1])] = (_make_df(), '<svg id="STALE_TILE"></svg>')
        asyncio.run(sc.display(dfs[1], dfs, 1))
        self.assertNotIn('STALE_TILE', sc.mod_inner)
        self.assertIs(sc._svg_cache_[id(dfs[1])][0], dfs[1])


class TestStackControlPlacement(unittest.TestCase):
    """_placement() traced against the exact geometry of
    polars2svg_prototyping/stack_control_shrinkage.ipynb (wxh=(128,128),
    component wxh=(32,32)) -- the notebook that found the drift this rewrite
    fixes. n=2 fits fully; n=3/n=4 are tight enough to exercise the base/top
    chain and the outward fill.
    """
    HC, GAP, TXT_H = 32, 4, 10
    H, INSET_Y = 128, 2

    @classmethod
    def setUpClass(cls):
        header_h  = headerHeight(cls.TXT_H)
        cls.avail = (cls.H - 2 * cls.INSET_Y - header_h) - cls.HC
        cls.slot  = cls.HC + cls.GAP
        cls.ell_h = 2 * cls.GAP + cls.TXT_H

    def _place(self, n, index):
        return _placement(n, index, self.avail, self.slot, self.ell_h)

    def test_two_frames_always_fully_contiguous(self):
        for index in (0, 1):
            lo, hi, base_on, top_on, gap_below, gap_above = self._place(2, index)
            self.assertEqual((lo, hi), (0, 1))
            self.assertTrue(base_on and top_on)
            self.assertEqual((gap_below, gap_above), (0, 0))

    def test_rendered_and_skipped_frames_account_for_all_of_n(self):
        for n in range(1, 12):
            for index in range(n):
                lo, hi, base_on, top_on, gap_below, gap_above = self._place(n, index)
                self.assertTrue(lo <= index <= hi)   # the cluster always seeds from index
                rendered = (hi - lo + 1) + (1 if base_on and lo > 0 else 0) \
                                         + (1 if top_on and hi < n - 1 else 0)
                self.assertEqual(rendered + gap_below + gap_above, n)

    def test_at_most_two_skip_labels(self):
        for n in range(1, 15):
            for index in range(n):
                *_, gap_below, gap_above = self._place(n, index)
                self.assertLessEqual((gap_below > 0) + (gap_above > 0), 2)

    def test_middle_index_with_room_only_for_base_and_current(self):
        # n=3, index=1: base fits (merges, no gap), top doesn't -- "Base Index <missing>"
        lo, hi, base_on, top_on, gap_below, gap_above = self._place(3, 1)
        self.assertEqual((lo, hi, base_on, top_on), (0, 1, True, False))
        self.assertEqual((gap_below, gap_above), (0, 1))

    def test_top_never_rendered_without_base(self):
        # Rule 5 ("after rendering the current index and the base dataframe")
        # never fires once rule 4 has failed.
        for n in range(2, 15):
            for index in range(1, n - 1):
                lo, hi, base_on, top_on, _, _ = self._place(n, index)
                if top_on and hi == n - 1 and lo > 0:
                    self.assertTrue(base_on)

    def test_severely_starved_budget_still_renders_current(self):
        # Only just enough room for the mandatory icon + two skip labels.
        lo, hi, base_on, top_on, gap_below, gap_above = _placement(
            20, 10, avail=2 * self.ell_h, slot=self.slot, ell_h=self.ell_h)
        self.assertEqual((lo, hi), (10, 10))
        self.assertFalse(base_on or top_on)
        self.assertEqual(gap_below, 10)
        self.assertEqual(gap_above, 9)


class TestStackControlMinimumSize(_StackControlBase):

    def test_raises_when_too_short_for_header_icon_and_two_labels(self):
        component = self.p2s.xyp(_make_df(), 'x', 'y', wxh=(120, 32))
        with self.assertRaises(ValueError):
            self.p2s.stack_controli(component, wxh=(160, 40))

    def test_does_not_raise_at_the_minimum(self):
        component = self.p2s.xyp(_make_df(), 'x', 'y', wxh=(120, 32))
        ell_h    = _ELL_H_
        for _rows_ in (0, 4):
            min_h = 2 * 2 + headerHeight(_TXT_H_) + shelfHeight(_TXT_H_, _rows_, _HGAP_) + 32 + 2 * ell_h
            # the minimum builds, at the size asked for; one pixel less does not
            self.assertEqual(self.p2s.stack_controli(component, wxh=(160, min_h), max_shelf=_rows_).svg_h,
                             min_h)
            with self.assertRaisesRegex(ValueError, 'smaller max_shelf'):
                self.p2s.stack_controli(component, wxh=(160, min_h - 1), max_shelf=_rows_)

    def test_max_shelf_must_be_a_count(self):
        component = self.p2s.xyp(_make_df(), 'x', 'y', wxh=(120, 32))
        for _bad_ in (-1, 2.0, True, '4'):
            with self.assertRaisesRegex(ValueError, 'max_shelf must be an int', msg=repr(_bad_)):
                self.p2s.stack_controli(component, max_shelf=_bad_)


class TestStackControlAlwaysRendersCurrent(_StackControlBase):
    """The shrinkage notebook's exact reproduction: wxh=(128,128), component
    wxh=(32,32). The old algorithm could drop the selected frame entirely
    when squeezed; the new one guarantees it always shows.
    """
    def _sc(self, n, index):
        component = self.p2s.xyp(_make_df(), 'x', 'y', wxh=(32, 32))
        return self._make_sc(dfs=self._stack_of(n), index=index,
                             component=component, wxh=(128, 128))

    def test_selected_frame_always_in_frame_map(self):
        for n in (2, 3, 4):
            for index in range(n):
                sc, _ = self._sc(n, index)
                rendered = {fm[2] for fm in sc._frame_map_}
                self.assertIn(index, rendered, f'n={n} index={index}')

    def test_skip_label_text_is_not_clipped_off_canvas(self):
        sc, _ = self._sc(4, 2)
        for m in re.finditer(r'<text x="([-\d.]+)"[^>]*font-style="italic"', sc.mod_inner):
            self.assertGreaterEqual(float(m.group(1)), 0)


class TestStackControlShelf(_StackControlBase):
    """F18 (PLANNING.md §7): the shelf area, stack-row labels, trashcans, the status row,
    and clicks that carry x and modifiers.  The operations themselves are tested on the
    controller in test_stack_shelf.py; these test the widget's routing and drawing."""

    _WXH_ = (160, 320)

    def _sc(self, n=3, index=1, max_shelf=4):
        dfs = self._stack_of(n)
        # A thumbnail-sized component, as in real use, so the labels have room.
        _icon_  = self.p2s.xyp(_make_df(), 'x', 'y', wxh=(48, 40))
        sc, mvc = self._make_sc(dfs=dfs, index=index, component=_icon_,
                                wxh=self._WXH_, max_shelf=max_shelf)
        return sc, mvc, dfs

    def _click(self, sc, x, y, mods=''):
        sc.click_x, sc.click_y, sc.click_mods = int(x), int(y), mods
        asyncio.run(sc.applyClickOp(None))

    def _rowY(self, sc, name):
        _y0_, _y1_, _ = next(_r_ for _r_ in sc._shelf_map_ if _r_[2] == name)
        return (_y0_ + _y1_) // 2

    def _frameY(self, sc, idx):
        _y0_, _y1_, _ = next(_r_ for _r_ in sc._frame_map_ if _r_[2] == idx)
        return (_y0_ + _y1_) // 2

    def _trashcans(self, sc):
        return sc.mod_inner.count('<path d="M0.5 2 H8.5')

    def test_the_shelf_is_drawn_in_the_order_shelved(self):
        # A new entry fills the first free row; the ones already there do not move.
        sc, mvc, dfs = self._sc()
        mvc.pinFrame('old', dfs[2])
        _old_row_ = sc._shelf_map_[0]
        mvc.pinFrame('new', dfs[0].head(5))
        self.assertEqual([_r_[2] for _r_ in sc._shelf_map_], ['old', 'new'])
        self.assertEqual(sc._shelf_map_[0], _old_row_)
        self.assertLess(sc.mod_inner.index('>old<'), sc.mod_inner.index('>new<'))
        self.assertIn('>5 Rows<', sc.mod_inner)
        # the shelf sits above every frame
        self.assertLess(max(_r_[1] for _r_ in sc._shelf_map_), min(_r_[0] for _r_ in sc._frame_map_))

    def test_no_shelf_draws_no_shelf(self):
        sc, mvc, dfs = self._sc(max_shelf=0)
        self.assertEqual(mvc.max_shelf, 0)
        self.assertEqual(sc._shelf_map_, [])
        self.assertNotIn('p2s-shelf-rule', sc.mod_inner)

    def test_only_the_displayed_frame_has_no_trashcan(self):
        sc, mvc, dfs = self._sc(n=3, index=1)
        mvc.pinFrame('n', dfs[0].head(2))
        self.assertEqual(self._trashcans(sc), 1 + 2)       # one shelf row, two frames

    def test_a_plain_click_on_a_shelf_row_pushes_it_labelled(self):
        sc, mvc, dfs = self._sc()
        _n_ = mvc.pinFrame('Named 1', dfs[0].head(3))
        self._click(sc, 10, self._rowY(sc, 'Named 1'))
        self.assertIs(mvc.stacks['default']['dfs'][-1], _n_)
        self.assertEqual(len(mvc.stacks['default']['dfs']), 4)      # nothing cut
        self.assertIn('>Named 1<', sc.mod_inner.split('p2s-shelf-rule')[1])  # on the stack row too

    def test_modifiers_pick_the_set_operation(self):
        sc, mvc, dfs = self._sc(n=2, index=1)                       # displayed: 11 rows
        mvc.pinFrame('n', dfs[0].tail(3))                           # rows 9-11 of the base
        self._click(sc, 10, self._rowY(sc, 'n'), 'cs')
        _top_ = mvc.stacks['default']['dfs'][-1]
        self.assertEqual(len(_top_), 2)
        self.assertEqual(mvc.frameLabel('default', _top_), '\u2229 n')
        self.assertIn('\u2229 n', sc.mod_inner)

    def test_a_stack_trashcan_deletes_at_once_and_ignores_modifiers(self):
        sc, mvc, dfs = self._sc(n=3, index=1)
        self._click(sc, sc._tc_x0_ + 3, self._frameY(sc, 0), 's')
        self.assertEqual(mvc.stacks['default']['dfs'], [dfs[1], dfs[2]])
        self.assertEqual(mvc.stacks['default']['index'], 0)

    def test_the_displayed_frame_trashcan_strip_is_just_a_click(self):
        sc, mvc, dfs = self._sc(n=3, index=1)
        self._click(sc, sc._tc_x0_ + 3, self._frameY(sc, 1))
        self.assertEqual(len(mvc.stacks['default']['dfs']), 3)
        self.assertEqual(sc._status_, '')

    def test_a_shelf_trashcan_asks_twice(self):
        # Q14: the first click arms it (red), the second within _CONFIRM_S_ deletes.
        sc, mvc, dfs = self._sc()
        mvc.pinFrame('keep me', dfs[2])
        _x_ = sc._tc_x0_ + 3
        self._click(sc, _x_, self._rowY(sc, 'keep me'))
        self.assertEqual(list(mvc.shelf()), ['keep me'])
        self.assertIn('click again to delete', sc.mod_inner)
        self.assertIn('stroke="#cc3333"', sc.mod_inner)
        self._click(sc, _x_, self._rowY(sc, 'keep me'))
        self.assertEqual(mvc.shelf(), {})
        self.assertNotIn('click again', sc.mod_inner)

    def test_an_armed_trashcan_times_out(self):
        sc, mvc, dfs = self._sc()
        mvc.pinFrame('n', dfs[2])
        _x_, _y_ = sc._tc_x0_ + 3, self._rowY(sc, 'n')
        self._click(sc, _x_, _y_)
        sc._armed_ = (sc._armed_[0], sc._armed_[1] - _CONFIRM_S_ - 1)   # the time passed
        self._click(sc, _x_, _y_)
        self.assertEqual(list(mvc.shelf()), ['n'])                      # re-armed, not deleted
        sc._disarm_(sc._armed_)                                         # the timer firing
        self.assertIsNone(sc._armed_)
        self.assertNotIn('click again', sc.mod_inner)
        self.assertNotIn('stroke="#cc3333"', sc.mod_inner)

    def test_another_click_disarms(self):
        sc, mvc, dfs = self._sc()
        mvc.pinFrame('n', dfs[2])
        self._click(sc, sc._tc_x0_ + 3, self._rowY(sc, 'n'))
        self._click(sc, 10, self._frameY(sc, 0))
        self.assertIsNone(sc._armed_)

    def test_a_refusal_is_shown_and_the_next_action_clears_it(self):
        # Q6 / diagram 8: ctrl-click the displayed frame changes nothing, visibly.
        sc, mvc, dfs = self._sc(n=3, index=1)
        self._click(sc, 10, self._frameY(sc, 1), 'c')
        self.assertIn('>no change<', sc.mod_inner)
        self.assertEqual(len(mvc.stacks['default']['dfs']), 3)
        self._click(sc, 10, self._frameY(sc, 0))
        self.assertNotIn('>no change<', sc.mod_inner)
        self.assertEqual(mvc.stacks['default']['index'], 0)

    def test_an_incompatible_entry_is_grayed_and_refuses(self):
        # Q18 (Claude, for DT to confirm).
        sc, mvc, dfs = self._sc()
        mvc.pinFrame('old', dfs[2])
        asyncio.run(mvc.replaceStack('default', dfs[0].with_columns(z=pl.lit(1))))
        self.assertRegex(sc.mod_inner, r'fill="#cccccc" font-size="8px">old<')
        self._click(sc, 10, self._rowY(sc, 'old'))
        self.assertEqual(mvc.entryRefusal('default', dfs[2]), 'columns no longer match: z')
        self.assertIn('columns no longer', sc.mod_inner)

    def test_a_long_name_is_cut_and_shown_whole_on_hover(self):
        sc, mvc, dfs = self._sc()
        _long_ = 'a rather long name for a shelf entry, far too wide'
        mvc.pinFrame(_long_, dfs[2])
        self.assertNotIn(_long_, sc.mod_inner)
        self.assertIn('\u2026', sc.mod_inner)
        # Shelving a frame on the stack labels its row too, so both are cut.
        _regions_ = json.loads(sc.hover_regions)
        self.assertEqual([_r_[4] for _r_ in _regions_], [_long_, _long_])
        _y_ = self._rowY(sc, _long_)
        self.assertTrue(any(_r_[1] <= _y_ < _r_[3] for _r_ in _regions_))

    def test_more_entries_than_slots(self):
        sc, mvc, dfs = self._sc(max_shelf=2)
        mvc.max_shelf = 3                                     # a controller bigger than this widget
        for _i_ in range(3):
            mvc.pinFrame(f'n{_i_}', dfs[0].head(_i_ + 1))
        self.assertEqual([_r_[2] for _r_ in sc._shelf_map_], ['n0'])
        self.assertIn('... 2 more on the shelf ...', sc.mod_inner)

    def test_a_label_replaces_the_position_line(self):
        # Q2: line 1 stays the row count; line 2 is the label, else the position.
        sc, mvc, dfs = self._sc(n=3, index=1)
        self.assertIn('>2 of 3<', sc.mod_inner)
        mvc.pinFrame('mine')
        _stack_part_ = sc.mod_inner.split('p2s-shelf-rule')[1]
        self.assertIn('>mine<', _stack_part_)
        self.assertNotIn('>2 of 3<', _stack_part_)

    def test_the_constructor_hands_its_size_to_the_controller(self):
        sc, mvc, dfs = self._sc(max_shelf=2)
        self.assertEqual(mvc.max_shelf, 2)

    def test_panelize_hands_the_size_on(self):
        _comp_ = self._component(h=40)
        sc     = self.p2s.stack_controli(_comp_, wxh=self._WXH_, max_shelf=2)
        layout = self.p2s.panelize([[self.p2s.xyp(_make_df(), 'x', 'y', wxh=(120, 120)), sc]])
        self.assertEqual(layout.mvc.max_shelf, 2)
        layout.pinFrame('from python')
        self.assertIn('>from python<', sc.mod_inner)


class TestStackControlNamingDialog(_StackControlBase):
    """F18 step 5: `s` shelves the displayed frame under a typed name (diagram 5, Q11).
    The browser half of the dialog is tests/interaction/test_stack_shelf_browser.py."""

    def _sc(self, max_shelf=4):
        dfs  = self._stack_of(3)
        _icon_ = self.p2s.xyp(_make_df(), 'x', 'y', wxh=(48, 40))
        sc, mvc = self._make_sc(dfs=dfs, index=1, component=_icon_, wxh=(160, 320), max_shelf=max_shelf)
        return sc, mvc, dfs

    def _key(self, sc, op, name=None):
        if name is not None:
            sc.shelf_name = name
        sc.key_op_finished = op
        asyncio.run(sc.applyKeyOp(None))

    def test_shelve_puts_the_displayed_frame_on_the_shelf(self):
        sc, mvc, dfs = self._sc()
        self._key(sc, 'shelve', '  scan\t1 ')
        self.assertIs(mvc.shelf()['scan 1'], dfs[1])
        self.assertEqual(mvc.frameLabel('default', dfs[1]), 'scan 1')
        self.assertIn('>scan 1<', sc.mod_inner)
        self.assertEqual(sc.key_op_finished, '')

    def test_a_refused_name_says_why(self):
        sc, mvc, dfs = self._sc()
        mvc.pinFrame('taken', dfs[2])
        self._key(sc, 'shelve', 'taken')
        self.assertEqual(sc._status_, '"taken" is already on the shelf')
        self.assertIn('already on the shelf', sc.mod_inner)
        self._key(sc, 'shelve', '\x07  ')
        self.assertEqual(sc._status_, 'a shelf name cannot be empty')

    def test_shelve_blocked_follows_the_state(self):
        sc, mvc, dfs = self._sc()
        self.assertEqual(sc.shelve_blocked, '')
        mvc.pinFrame('mine')                                   # the displayed frame
        self.assertEqual(sc.shelve_blocked, 'already shelved as "mine"')
        asyncio.run(mvc.jumpTo('default', 0))
        self.assertEqual(sc.shelve_blocked, '')
        for _i_ in range(3):
            mvc.pinFrame(f'n{_i_}', dfs[0].head(_i_ + 1))
        self.assertEqual(sc.shelve_blocked, 'shelf is full (4)')

    def test_shelve_blocked_shows_the_reason(self):
        sc, mvc, dfs = self._sc()
        mvc.pinFrame('mine')
        self._key(sc, 'shelve_blocked')
        self.assertIn('already shelved as', sc.mod_inner)

    def test_no_shelf_blocks_and_has_no_dialog(self):
        sc, mvc, dfs = self._sc(max_shelf=0)
        self.assertEqual(sc.shelve_blocked, 'shelf is off')
        self.assertEqual(sc.shelf_dialog_box, '')

    def test_the_dialog_box_is_the_first_free_row(self):
        # Where the new entry will be drawn, so it covers no existing name (DT, 2026-10-27).
        sc, mvc, dfs = self._sc()
        _x_, _y_, _w_, _h_ = json.loads(sc.shelf_dialog_box)
        mvc.pinFrame('first', dfs[2])
        self.assertEqual((_y_, _y_ + _h_), sc._shelf_map_[0][:2])     # empty shelf: row 0
        _x_, _y_, _w_, _h_ = json.loads(sc.shelf_dialog_box)
        self.assertEqual(_y_, sc._shelf_map_[0][1])                    # now: just below it
        self.assertGreater(_w_, 100)
        self._key(sc, 'shelve', 'second')
        self.assertEqual(sc._shelf_map_[1][:2], (_y_, _y_ + _h_))     # drawn where it was named

    def test_the_default_name_skips_taken_ones(self):
        sc, mvc, dfs = self._sc()
        self.assertEqual(sc.shelf_default_name, 'frame 1')
        mvc.pinFrame('frame 1', dfs[2])
        mvc.pinFrame('frame 2', dfs[0].head(2))
        self.assertEqual(sc.shelf_default_name, 'frame 3')

    def test_onShelve_hears_the_widget(self):
        sc, mvc, dfs = self._sc()
        _got_ = []
        mvc.onShelve(lambda name, df: _got_.append((name, df)))
        self._key(sc, 'shelve', 'heard')
        self.assertEqual(_got_, [('heard', dfs[1])])

    def test_panelize_lets_s_work_straight_away(self):
        # Built with no controller, the widget cannot shelve; panelize() redraws it.
        sc = self.p2s.stack_controli(self.p2s.xyp(_make_df(), 'x', 'y', wxh=(48, 40)), wxh=(160, 320))
        self.assertEqual(sc.shelve_blocked, 'not on a stack yet')
        self.p2s.panelize([[self.p2s.xyp(_make_df(), 'x', 'y', wxh=(120, 120)), sc]])
        self.assertEqual(sc.shelve_blocked, '')


if __name__ == '__main__':
    unittest.main()
