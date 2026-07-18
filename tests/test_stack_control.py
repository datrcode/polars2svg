"""stack_controli widget: frame layout (fit + all three overflow cases),
click-to-select routing through the InteractionController, display() cache
management, and sketchHtml().

Overflow geometry notes (wxh=(160, 256), insets=(2, 2), hgap=4, txt_h=10):
  slot   = component_h + 4
  avail  = 256 - 2*2 - component_h - headerHeight()
  A stack of n frames fits iff (n-1)*slot <= avail; larger stacks trigger the
  spiral-fill placement with skip labels ("... N stack frames ...").

The indicator header (MLX/CUDA availability) eats into the same vertical budget,
so the boundary tests below size their component from headerHeight() instead of a
literal -- adding a third indicator row shifts them automatically.
"""
import asyncio
import re
import unittest

import polars as pl

from polars2svg import Polars2SVG
from polars2svg.interactive_controller import InteractionController
from polars2svg.stack_control import headerHeight

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
                           'geometry no longer admits a single-skip case for n=%d' % n)
        return slot - _HGAP_

    def _stack_of(self, n):
        """A stack of n distinct dataframes (base is largest)."""
        df = _make_df()
        return [df.head(len(df) - i) for i in range(n)]

    def _make_sc(self, dfs=None, index=0, component=None, **kwargs):
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
        self.assertIn('(Base)', sc.mod_inner)
        self.assertIn('Top',    sc.mod_inner)


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
        sc = self._make_sc()
        _y_ = (sc._frame_map_[0][0] + sc._frame_map_[0][1]) // 2
        self._click(sc, _y_)   # must not raise


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


if __name__ == '__main__':
    unittest.main()
