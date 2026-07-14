import unittest

import polars as pl
from polars2svg import Polars2SVG
from polars2svg.linkp import LinkP


_DF_ = pl.DataFrame({
    'fm': ['a', 'b', 'c', 'd', 'b'],
    'to': ['b', 'c', 'd', 'a', 'a'],
})
_REL_ = [('fm', 'to')]
_POS_ = {'a': (0.0, 0.5), 'b': (0.5, 0.0), 'c': (1.0, 0.5), 'd': (0.5, 1.0)}


def _linkp_params(**extra):
    return dict(df=_DF_, relationships=_REL_, pos=_POS_, wxh=(96, 96), **extra)


class TestLinkPInertParamsRemoved(unittest.TestCase):
    '''node_shape and draw_context were accepted-but-inert in linkp (stored, never
    read by __renderSVG__).  They are now removed from linkp entirely so passing
    them raises instead of being ignored.'''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def tearDown(self):
        self.p2s.reset_defaults()

    # ── the two params are gone from the allowlist ─────────────────────────────
    def test_node_shape_not_in_valid_kwargs(self):
        self.assertNotIn('node_shape', LinkP._VALID_KWARGS)

    def test_draw_context_not_in_valid_kwargs(self):
        self.assertNotIn('draw_context', LinkP._VALID_KWARGS)

    def test_registry_reflects_removal(self):
        '''The eager-validation registry is derived from _VALID_KWARGS, so it must
        agree that linkp no longer accepts either param.'''
        self.assertNotIn('node_shape', self.p2s._COMPONENT_KWARGS_['linkp'])
        self.assertNotIn('draw_context', self.p2s._COMPONENT_KWARGS_['linkp'])

    # ── passing them at call time raises (kinder than ignoring) ────────────────
    def test_node_shape_kwarg_raises(self):
        with self.assertRaises(TypeError) as ctx:
            self.p2s.linkp(**_linkp_params(node_shape={'a': 'square'}))
        self.assertIn('node_shape', str(ctx.exception))

    def test_draw_context_kwarg_raises(self):
        with self.assertRaises(TypeError) as ctx:
            self.p2s.linkp(**_linkp_params(draw_context=False))
        self.assertIn('draw_context', str(ctx.exception))

    # ── set_defaults eager validation also rejects them for linkp ──────────────
    def test_set_defaults_node_shape_raises(self):
        with self.assertRaises(TypeError):
            self.p2s.set_defaults('linkp', node_shape={'a': 'square'})

    def test_set_defaults_draw_context_raises(self):
        with self.assertRaises(TypeError):
            self.p2s.set_defaults('linkp', draw_context=False)

    # ── removal doesn't leave stale attributes on the instance ─────────────────
    def test_instance_has_no_inert_attrs(self):
        lp = self.p2s.linkp(**_linkp_params())
        self.assertFalse(hasattr(lp, 'node_shape'))
        self.assertFalse(hasattr(lp, 'draw_context'))

    # ── a normal linkp still renders (regression) ──────────────────────────────
    def test_normal_linkp_still_renders(self):
        lp = self.p2s.linkp(**_linkp_params())
        self.assertIn('<svg', lp.svg)
        self.assertIn('<circle', lp.svg)

    # ── a *global* draw_context default is still valid (other components use it)
    #    and must not leak into / break linkp ───────────────────────────────────
    def test_global_draw_context_default_does_not_break_linkp(self):
        # draw_context is a real param on histop/timep/etc., so the global default
        # is accepted; _apply_defaults filters it out for linkp's allowlist.
        self.p2s.set_defaults(draw_context=False)
        lp = self.p2s.linkp(**_linkp_params())
        self.assertIn('<svg', lp.svg)
        self.assertFalse(hasattr(lp, 'draw_context'))


if __name__ == '__main__':
    unittest.main()
