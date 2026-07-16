import unittest

import polars as pl
from polars2svg import Polars2SVG
from polars2svg.chordp import ChP


_DF_ = pl.DataFrame({
    'fm': ['a', 'b', 'c', 'd', 'b'],
    'to': ['b', 'c', 'd', 'a', 'a'],
})
_REL_ = [('fm', 'to')]


def _chordp_params(**extra):
    return dict(df=_DF_, relationships=_REL_, wxh=(96, 96), **extra)


class TestChordPInertParamsRemoved(unittest.TestCase):
    '''node_shape and draw_context were accepted-but-inert in chordp (stored, never
    read by __renderSVG__) -- the same class of bug already fixed in linkp
    (see test_linkp_inert_params.py). They are removed from chordp entirely so
    passing them raises instead of being ignored. See item 4 of 20260714_open_todos.md.'''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def tearDown(self):
        self.p2s.reset_defaults()

    # ── the two params are gone from the allowlist ─────────────────────────────
    def test_node_shape_not_in_valid_kwargs(self):
        self.assertNotIn('node_shape', ChP._VALID_KWARGS)

    def test_draw_context_not_in_valid_kwargs(self):
        self.assertNotIn('draw_context', ChP._VALID_KWARGS)

    def test_registry_reflects_removal(self):
        '''The eager-validation registry is derived from _VALID_KWARGS, so it must
        agree that chordp no longer accepts either param.'''
        self.assertNotIn('node_shape', self.p2s._COMPONENT_KWARGS_['chordp'])
        self.assertNotIn('draw_context', self.p2s._COMPONENT_KWARGS_['chordp'])

    # ── passing them at call time raises (kinder than ignoring) ────────────────
    def test_node_shape_kwarg_raises(self):
        with self.assertRaises(TypeError) as ctx:
            self.p2s.chordp(**_chordp_params(node_shape={'a': 'square'}))
        self.assertIn('node_shape', str(ctx.exception))

    def test_draw_context_kwarg_raises(self):
        with self.assertRaises(TypeError) as ctx:
            self.p2s.chordp(**_chordp_params(draw_context=False))
        self.assertIn('draw_context', str(ctx.exception))

    # ── set_defaults eager validation also rejects them for chordp ─────────────
    def test_set_defaults_node_shape_raises(self):
        with self.assertRaises(TypeError):
            self.p2s.set_defaults('chordp', node_shape={'a': 'square'})

    def test_set_defaults_draw_context_raises(self):
        with self.assertRaises(TypeError):
            self.p2s.set_defaults('chordp', draw_context=False)

    # ── removal doesn't leave stale attributes on the instance ─────────────────
    def test_instance_has_no_inert_attrs(self):
        cp = self.p2s.chordp(**_chordp_params())
        self.assertFalse(hasattr(cp, 'node_shape'))
        self.assertFalse(hasattr(cp, 'draw_context'))

    # ── a normal chordp still renders (regression) ──────────────────────────────
    def test_normal_chordp_still_renders(self):
        cp = self.p2s.chordp(**_chordp_params())
        self.assertIn('<svg', cp.svg)

    # ── a *global* draw_context default is still valid (other components use it)
    #    and must not leak into / break chordp ──────────────────────────────────
    def test_global_draw_context_default_does_not_break_chordp(self):
        self.p2s.set_defaults(draw_context=False)
        cp = self.p2s.chordp(**_chordp_params())
        self.assertIn('<svg', cp.svg)
        self.assertFalse(hasattr(cp, 'draw_context'))


if __name__ == '__main__':
    unittest.main()
