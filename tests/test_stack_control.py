"""stack_controli widget: frame layout (fit + all three overflow cases),
click-to-select routing through the InteractionController, display() cache
management, and sketchHtml().

Overflow geometry notes (wxh=(160, 256), insets=(2, 2), hgap=4, txt_h=10):
  slot   = component_h + 4
  avail  = 256 - 2*2 - component_h
  A stack of n frames fits iff (n-1)*slot <= avail; larger stacks trigger the
  spiral-fill placement with skip labels ("... N stack frames ...").
"""
import asyncio
import unittest

import polars as pl

from polars2svg import Polars2SVG
from polars2svg.interactive_controller import InteractionController


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
        return self._make_sc(dfs=self._stack_of(10), index=index)

    def test_selected_near_base_packs_from_base(self):
        sc, _ = self._overflow_sc(index=1)
        rendered = sorted(fm[2] for fm in sc._frame_map_)
        self.assertIn(0, rendered)
        self.assertIn(9, rendered)
        self.assertIn(1, rendered)
        self.assertLess(len(rendered), 10)
        self.assertIn('... 7 stack frames ...', sc.mod_inner)

    def test_selected_near_top_packs_from_top(self):
        sc, _ = self._overflow_sc(index=8)
        rendered = sorted(fm[2] for fm in sc._frame_map_)
        self.assertIn(0, rendered)
        self.assertIn(9, rendered)
        self.assertIn(8, rendered)
        self.assertIn('... 7 stack frames ...', sc.mod_inner)

    def test_selected_in_middle_gets_two_skip_labels(self):
        sc, _ = self._overflow_sc(index=5)
        rendered = sorted(fm[2] for fm in sc._frame_map_)
        self.assertIn(5, rendered)
        self.assertIn('... 3 stack frames ...', sc.mod_inner)
        self.assertIn('... 4 stack frames ...', sc.mod_inner)

    def test_singular_skip_label_uses_frame_not_frames(self):
        # component h=40 → slot=44: six frames overflow with exactly one skipped
        sc, _ = self._make_sc(dfs=self._stack_of(6), index=1,
                              component=self._component(h=40))
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
        sc, _ = self._make_sc(dfs=dfs, index=0)
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
