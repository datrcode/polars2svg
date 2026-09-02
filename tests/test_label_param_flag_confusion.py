"""A display-name dict is not an on/off flag.

`draw_node_labels=` turns node labels on; `node_labels=` supplies the
`{name: display_str}` map.  The names differ by a prefix, so passing the flag's
value to the map is an easy mistake -- and it used to be a silent one.  With the
draw flag off, `node_labels=True` was simply stored and never looked at; with it
on, the render fell over deep inside with `object of type 'bool' has no len()`,
which names neither the parameter nor the fix.

That mistake is not hypothetical.  `test_edge_case_inputs.py`'s linkp
special-character test passed `node_labels=True` where it meant
`draw_node_labels=`, rendered zero `<text>` elements, and asserted nothing about
labels for as long as it existed -- the escaping bug it was meant to be watching
(tests/test_svg_escaping.py) went unnoticed underneath it.

linkp already caught this for the *link* channel: `link_labels=True` has raised a
clear TypeError at parse time since edge labels landed.  The guard was simply
never applied to the parameters beside it.  It now lives in one place --
`Polars2SVG.rejectBoolParam()` -- and covers both label maps and the shared
`label_only=` filter on both components that have them.

spreadlinesp also declares `node_labels=`, but it has a different defect (the
parameter is never read at all, with or without a bool) and is deliberately left
alone here.
"""
import unittest

import polars as pl
from polars2svg import Polars2SVG


_DF_  = pl.DataFrame({'fm': ['a', 'b', 'c', 'd', 'b'],
                      'to': ['b', 'c', 'd', 'a', 'a']})
_REL_ = [('fm', 'to')]
_POS_ = {'a': (0.0, 0.5), 'b': (0.5, 0.0), 'c': (1.0, 0.5), 'd': (0.5, 1.0)}


class _FlagConfusionBase(unittest.TestCase):
    def setUp(self):
        self.p2s = Polars2SVG()

    def tearDown(self):
        self.p2s.reset_defaults()

    def _linkp(self, **extra):
        return self.p2s.linkp(df=_DF_, relationships=_REL_, pos=_POS_,
                              wxh=(96, 96), **extra)

    def _chordp(self, **extra):
        return self.p2s.chordp(df=_DF_, relationships=_REL_, wxh=(96, 96), **extra)

    # (label, callable-taking-**kwargs, param, draw-flag or None)
    def _cases(self):
        return [
            ('linkp',  self._linkp,  'node_labels', 'draw_node_labels'),
            ('linkp',  self._linkp,  'link_labels', 'draw_link_labels'),
            ('linkp',  self._linkp,  'label_only',  None),
            ('chordp', self._chordp, 'node_labels', 'draw_labels'),
            ('chordp', self._chordp, 'label_only',  None),
        ]


class TestBoolRejectedForLabelParams(_FlagConfusionBase):

    def test_true_raises(self):
        '''It is the constructor that raises -- parse-time, like every other
        parameter complaint.  The old failure needed a render to surface it, and
        surfaced from the middle of one.'''
        for _comp_, _fn_, _param_, _flag_ in self._cases():
            with self.subTest(component=_comp_, param=_param_):
                with self.assertRaises(TypeError) as _cm_:
                    _fn_(**{_param_: True})
                self.assertIn(f'{_param_}=', str(_cm_.exception))

    def test_false_raises_too(self):
        '''False is the same confusion -- "don't label" belongs on the flag.'''
        for _comp_, _fn_, _param_, _flag_ in self._cases():
            with self.subTest(component=_comp_, param=_param_):
                with self.assertRaises(TypeError):
                    _fn_(**{_param_: False})

    def test_raises_with_the_draw_flag_on_as_well(self):
        '''The old failure only surfaced here, and surfaced as a len()/items()
        AttributeError from somewhere in the middle of the render.'''
        for _comp_, _fn_, _param_, _flag_ in self._cases():
            if _flag_ is None: continue
            with self.subTest(component=_comp_, param=_param_):
                with self.assertRaises(TypeError) as _cm_:
                    _fn_(**{_param_: True, _flag_: True})
                self.assertIn(f'{_param_}=', str(_cm_.exception))

    def test_message_names_the_flag_to_use_instead(self):
        for _comp_, _fn_, _param_, _flag_ in self._cases():
            if _flag_ is None: continue
            with self.subTest(component=_comp_, param=_param_):
                with self.assertRaises(TypeError) as _cm_:
                    _fn_(**{_param_: True})
                self.assertIn(f'{_flag_}=', str(_cm_.exception))


class TestLegitimateValuesStillWork(_FlagConfusionBase):
    '''The guard is bool-only.  Everything these parameters actually accept has
    to keep working, or the fix is worse than the bug.'''

    def test_linkp_node_labels_dict(self):
        _svg_ = self._linkp(node_labels={'a': 'Alpha'},
                            draw_node_labels=True)._repr_svg_()
        self.assertIn('Alpha', _svg_)

    def test_linkp_link_labels_dict(self):
        self._linkp(link_labels={'a': 'Alpha'}, draw_link_labels=True,
                    color='fm')._repr_svg_()

    def test_chordp_node_labels_dict(self):
        _svg_ = self._chordp(node_labels={'a': 'Alpha'}, draw_labels=True)._repr_svg_()
        self.assertIn('Alpha', _svg_)

    def test_none_is_still_the_default(self):
        for _comp_, _fn_, _param_, _flag_ in self._cases():
            if _param_ == 'label_only': continue
            with self.subTest(component=_comp_, param=_param_):
                _fn_(**{_param_: None})._repr_svg_()

    def test_label_only_accepts_its_documented_shapes(self):
        for _value_ in ({'a', 'b'}, ['a', 'b'], 'a', set()):
            with self.subTest(value=_value_):
                self._linkp(label_only=_value_, draw_node_labels=True)._repr_svg_()
                self._chordp(label_only=_value_, draw_labels=True)._repr_svg_()

    def test_label_only_still_filters(self):
        _svg_ = self._linkp(label_only={'a'}, draw_node_labels=True)._repr_svg_()
        self.assertIn('>a<', _svg_)
        self.assertNotIn('>b<', _svg_)


class TestOneGuardNotFive(unittest.TestCase):
    '''The point of routing these through one helper: the next label map added to
    a component gets the guard by calling it, not by remembering to re-derive it.
    '''

    def test_helper_exists_on_the_framework(self):
        self.assertTrue(callable(getattr(Polars2SVG(), 'rejectBoolParam', None)))

    def test_helper_passes_through_non_bools(self):
        _p2s_ = Polars2SVG()
        for _v_ in (None, {}, {'a': 'b'}, set(), ['a'], 'a', 0, 1):
            with self.subTest(value=_v_):
                _p2s_.rejectBoolParam(_v_, 'X', 'p', 'a dict')

    def test_helper_rejects_bools(self):
        _p2s_ = Polars2SVG()
        for _v_ in (True, False):
            with self.subTest(value=_v_):
                with self.assertRaises(TypeError):
                    _p2s_.rejectBoolParam(_v_, 'X', 'p', 'a dict')

    def test_components_do_not_hand_roll_the_check(self):
        '''linkp's original inline guard is gone, not merely joined by others.'''
        import pathlib
        _pkg_ = pathlib.Path(__file__).resolve().parent.parent / 'polars2svg'
        for _name_ in ('linkp.py', 'chordp.py'):
            with self.subTest(module=_name_):
                _src_ = (_pkg_ / _name_).read_text(encoding='utf-8')
                self.assertNotIn('isinstance(self.link_labels, bool)', _src_)
                self.assertNotIn('isinstance(self.node_labels, bool)', _src_)


if __name__ == '__main__':
    unittest.main()
