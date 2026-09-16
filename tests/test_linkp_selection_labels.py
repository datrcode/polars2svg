"""Tests for the selected-node label overlay (LinkP.__createSelectedLabels__).

The overlay is the interactive counterpart of the node labels __renderNodes__ emits,
and the two differ on purpose: the rendered label stops at label_max_lines and
ellipsizes, the overlay shows the COMPLETE name wrapped over as many lines as it
takes.  Most of what is asserted here is that difference.

The controller half of the feature -- that a selection change updates the overlay
WITHOUT re-rendering the graph -- lives in test_interactive_controller.py, and the
browser half in tests/interaction/test_selection_labels.py.
"""
import unittest
import polars as pl
from polars2svg import Polars2SVG


_DF_  = pl.DataFrame({'fm': ['a', 'b', 'c', 'a'], 'to': ['b', 'c', 'a', 'c']})
_REL_ = [('fm', 'to')]
_POS_ = {'a': (0.0, 0.0), 'b': (1.0, 0.0), 'c': (0.5, 0.866)}

# 'c' shares 'b's world position, so the two land on one screen pixel and draw as the
# single cloud glyph -- the collapsed case the overlay has to summarize.
_POS_COLLAPSED_ = {'a': (0.0, 0.0), 'b': (1.0, 0.0), 'c': (1.0, 0.0)}

# Long enough to exceed label_max_lines (4) at the default label_line_width (32),
# so that a regression back to the render's own crop settings fails the completeness
# tests below rather than slipping under them.
_LONG_ = ('alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu '
          'nu xi omicron pi rho sigma tau upsilon phi chi psi omega '
          'digamma koppa sampi stigma heta sho san')


def _lp(pos=None, **extra):
    p2s = Polars2SVG()
    _lp_ = p2s.linkp(_DF_, relationships=_REL_, pos=pos or _POS_,
                     wxh=(400, 400), insets=(16, 16), **extra)
    _lp_.renderSVG()          # df_node is built by the render, not by __init__
    return _lp_


def _lp_long(**extra):
    """A graph whose node 'a' carries a label far too long for one line."""
    p2s  = Polars2SVG()
    _df_ = pl.DataFrame({'fm': [_LONG_, 'b'], 'to': ['b', _LONG_]})
    _lp_ = p2s.linkp(_df_, relationships=_REL_,
                     pos={_LONG_: (0.0, 0.0), 'b': (1.0, 1.0)},
                     wxh=(400, 400), insets=(16, 16), **extra)
    _lp_.renderSVG()
    return _lp_


def _lines_for(payload, first_word):
    for _e_ in payload['labels']:
        if _e_['lines'][0].startswith(first_word):
            return _e_['lines']
    return None


class TestSelectionLabelsEmptyCases(unittest.TestCase):
    """Nothing selected, nothing resolvable, or too much selected -- all draw nothing."""

    def test_no_selection_returns_no_labels(self):
        self.assertEqual(_lp().__createSelectedLabels__(my_selection=set()), {'labels': []})

    def test_none_selection_returns_no_labels(self):
        self.assertEqual(_lp().__createSelectedLabels__(my_selection=None), {'labels': []})

    def test_unknown_nodes_return_no_labels(self):
        self.assertEqual(_lp().__createSelectedLabels__(my_selection={'nope', 'also-nope'}),
                         {'labels': []})

    def test_selection_over_the_cap_labels_nothing(self):
        """The cap is all-or-nothing, as it was in the java original (at 100).

        A lasso around half the graph is not a request to read anything, so the
        answer is no labels at all rather than an arbitrary 32 of them.
        """
        _payload_ = _lp().__createSelectedLabels__(my_selection={'a', 'b', 'c'}, max_nodes=2)
        self.assertEqual(_payload_, {'labels': []})

    def test_selection_at_the_cap_still_labels(self):
        _payload_ = _lp().__createSelectedLabels__(my_selection={'a', 'b', 'c'}, max_nodes=3)
        self.assertEqual(len(_payload_['labels']), 3)

    def test_default_cap_is_32(self):
        import inspect
        _sig_ = inspect.signature(_lp().__createSelectedLabels__)
        self.assertEqual(_sig_.parameters['max_nodes'].default, 32)


class TestSelectionLabelsPayload(unittest.TestCase):
    """Shape of the dict the view carries to the browser."""

    def setUp(self):
        self.lp      = _lp()
        self.payload = self.lp.__createSelectedLabels__(my_selection={'a', 'b', 'c'})

    def test_carries_text_height_and_colors(self):
        self.assertEqual(self.payload['h'],  float(self.lp.txt_h))
        self.assertEqual(self.payload['fg'], self.lp.p2s.colorTyped('label', 'defaultfg'))
        self.assertEqual(self.payload['bg'], self.lp.p2s.colorTyped('background', 'default'))

    def test_every_entry_has_the_keys_the_script_reads(self):
        for _e_ in self.payload['labels']:
            self.assertEqual(set(_e_), {'x', 'y', 'w', 'lines'})

    def test_payload_is_json_serializable(self):
        """It crosses a WebSocket as a param.Dict, so nothing in it may be a numpy scalar."""
        import json
        json.dumps(self.payload)
        for _e_ in self.payload['labels']:
            self.assertIsInstance(_e_['x'], float)
            self.assertIsInstance(_e_['y'], float)
            self.assertIsInstance(_e_['w'], float)

    def test_no_markup_is_built_python_side(self):
        """The browser builds the elements; this side must never emit a tag."""
        import json
        self.assertNotIn('<', json.dumps(self.payload))

    def test_order_is_stable(self):
        _again_ = self.lp.__createSelectedLabels__(my_selection={'c', 'a', 'b'})
        self.assertEqual(self.payload, _again_)


class TestSelectionLabelsGeometry(unittest.TestCase):
    """Positions, and the measured width the backdrop rect is sized from."""

    def setUp(self):
        self.lp = _lp()

    def _entry(self, node):
        _payload_ = self.lp.__createSelectedLabels__(my_selection={node})
        self.assertEqual(len(_payload_['labels']), 1)
        return _payload_['labels'][0]

    def test_label_sits_under_its_own_node(self):
        _row_ = (self.lp.df_node.explode('__nm__')
                     .filter(pl.col('__nm__') == 'a')
                     .select('__sx__', '__sy__').row(0))
        _e_ = self._entry('a')
        self.assertAlmostEqual(_e_['x'], round(float(_row_[0]), 2))
        self.assertGreater(_e_['y'], _row_[1], 'the label belongs BELOW the node')

    def test_width_is_the_widest_line(self):
        _e_ = self._entry('a')
        self.assertAlmostEqual(
            _e_['w'],
            round(max(self.lp.p2s.textLength(_l_, self.lp.txt_h) for _l_ in _e_['lines']), 2))

    def test_one_label_per_screen_coordinate(self):
        """Two entities on one pixel draw one cloud, so they get one label, not two."""
        _lp_ = _lp(pos=_POS_COLLAPSED_)
        _payload_ = _lp_.__createSelectedLabels__(my_selection={'b', 'c'})
        self.assertEqual(len(_payload_['labels']), 1)

    def test_collapsed_coordinate_is_summarized(self):
        _lp_ = _lp(pos=_POS_COLLAPSED_)
        _payload_ = _lp_.__createSelectedLabels__(my_selection={'b', 'c'})
        self.assertEqual(_payload_['labels'][0]['lines'], ['2 nodes'])

    def test_off_canvas_labels_are_culled(self):
        """Same cull the rendered labels get -- a label wholly off the canvas is dropped."""
        _lp_ = _lp(view_window=(-10.0, -10.0, -5.0, -5.0))
        self.assertEqual(_lp_.__createSelectedLabels__(my_selection={'a', 'b', 'c'}),
                         {'labels': []})


class TestSelectionLabelsAreComplete(unittest.TestCase):
    """The point of the feature: the whole label, never a cropped one."""

    def test_long_label_wraps_to_several_lines(self):
        _lines_ = _lines_for(_lp_long().__createSelectedLabels__(my_selection={_LONG_}), 'alpha')
        self.assertGreater(len(_lines_), 1)

    def test_every_word_survives_the_wrap(self):
        _lines_ = _lines_for(_lp_long().__createSelectedLabels__(my_selection={_LONG_}), 'alpha')
        self.assertEqual(' '.join(_lines_), _LONG_)

    def test_no_ellipsis(self):
        _lines_ = _lines_for(_lp_long().__createSelectedLabels__(my_selection={_LONG_}), 'alpha')
        self.assertNotIn('…', ''.join(_lines_))

    def test_label_max_lines_does_not_truncate_the_overlay(self):
        """label_max_lines governs the RENDER; the overlay ignores it by design."""
        _lp_ = _lp_long(label_max_lines=1)
        _lines_ = _lines_for(_lp_.__createSelectedLabels__(my_selection={_LONG_}), 'alpha')
        self.assertGreater(len(_lines_), 1)
        self.assertEqual(' '.join(_lines_), _LONG_)

    def test_the_rendered_label_really_is_cropped(self):
        """Guard the comparison: if the render stopped cropping, the tests above are empty."""
        _lp_ = _lp_long(draw_node_labels=True, label_max_lines=1)
        _lp_.renderSVG()
        self.assertIn('…', _lp_.svg)

    def test_wrap_respects_label_line_width(self):
        _lp_ = _lp_long(label_line_width=16)
        _lines_ = _lines_for(_lp_.__createSelectedLabels__(my_selection={_LONG_}), 'alpha')
        for _l_ in _lines_:
            self.assertLessEqual(len(_l_), 16)


class TestSelectionLabelsNaming(unittest.TestCase):
    """How a node's display name is resolved."""

    def test_node_labels_renames_the_label(self):
        _lp_ = _lp(node_labels={'a': 'Alice'})
        _payload_ = _lp_.__createSelectedLabels__(my_selection={'a'})
        self.assertEqual(_payload_['labels'][0]['lines'], ['Alice'])

    def test_a_node_the_map_does_not_name_keeps_its_own_name(self):
        """Deliberate divergence from __renderNodes__, which drops such a node.

        The overlay answers "what did I just select"; a selected node with no label
        answers nothing, so an unnamed node falls back to its own name here.
        """
        _lp_ = _lp(node_labels={'a': 'Alice'})
        _payload_ = _lp_.__createSelectedLabels__(my_selection={'b'})
        self.assertEqual(_payload_['labels'][0]['lines'], ['b'])

    def test_null_node_shows_its_display_form(self):
        _p2s_ = Polars2SVG()
        _df_  = pl.DataFrame({'fm': ['a', 'b'], 'to': [None, 'a']})
        _lp_  = _p2s_.linkp(_df_, relationships=_REL_, null_nodes=True,
                            wxh=(400, 400), insets=(16, 16))
        _lp_.renderSVG()
        _null_ = [_n_ for _n_ in _lp_.pos if _p2s_.isNullNode(_n_)]
        self.assertTrue(_null_, 'fixture should have produced a null node')
        _payload_ = _lp_.__createSelectedLabels__(my_selection=set(_null_))
        _texts_   = [' '.join(_e_['lines']) for _e_ in _payload_['labels']]
        self.assertTrue(_texts_, 'the null node should be labeled')
        for _t_ in _texts_:
            self.assertNotIn(_p2s_.NULL_NODE_PREFIX, _t_)
            self.assertEqual(_t_, _p2s_.nullNodeDisplay(_p2s_.NULL_NODE_PREFIX))


class TestSelectionLabelsIndependentOfRenderSettings(unittest.TestCase):
    """The overlay is a view artifact: the render's own label switches must not gate it."""

    def test_works_with_node_labels_off(self):
        _lp_ = _lp(draw_node_labels=False)
        self.assertEqual(len(_lp_.__createSelectedLabels__(my_selection={'a'})['labels']), 1)

    def test_works_with_label_only_set_elsewhere(self):
        _lp_ = _lp(draw_node_labels=True, label_only={'b'})
        self.assertEqual(len(_lp_.__createSelectedLabels__(my_selection={'a'})['labels']), 1)

    def test_does_not_mutate_the_render(self):
        _lp_ = _lp()
        _before_ = _lp_.svg
        _lp_.__createSelectedLabels__(my_selection={'a', 'b', 'c'})
        self.assertEqual(_lp_.svg, _before_)

    def test_node_size_vary_offsets_by_each_nodes_own_radius(self):
        """'vary' carries a per-node __sz__ column; the offset must read it."""
        _lp_ = _lp(node_size='vary')
        _payload_ = _lp_.__createSelectedLabels__(my_selection={'a', 'b', 'c'})
        self.assertEqual(len(_payload_['labels']), 3)

    def test_node_size_none_still_places_labels(self):
        _lp_ = _lp(node_size=None)
        self.assertEqual(len(_lp_.__createSelectedLabels__(my_selection={'a'})['labels']), 1)


if __name__ == '__main__':
    unittest.main()
