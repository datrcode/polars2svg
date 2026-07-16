import unittest
import xml.etree.ElementTree as ET

import polars as pl

from polars2svg import Polars2SVG
from label_fidelity_data import (
    FIDELITY_LABELS, KNOWN_ROUNDING_VICTIMS,
    faulty_round_svg_floats, svg_text_contents,
)


def _build_linkp(p2s, labels, node_labels=None):
    '''Render every label as its own single (non-collapsed), drawn node.

    A chain of edges makes each label a node; an explicit grid pos keeps every node on a
    distinct screen pixel so none collapse (collapsed nodes are not labeled). A wide
    label_line_width with unlimited lines prevents word-wrapping, so each label stays a
    single <text> element whose content is exactly the (escaped) label.

    When node_labels is given, node names are opaque ids and the dict maps each to its
    display string -- so the labels under test are the dict *values*.'''
    df = pl.DataFrame({'fm': labels, 'to': labels[1:] + labels[:1]})
    pos = {lab: [(i % 6) * 2.0, -(i // 6) * 2.0] for i, lab in enumerate(labels)}
    return p2s.linkp(df, [('fm', 'to')], pos=pos, draw_labels=True, node_labels=node_labels,
                     label_line_width=128, label_max_lines=-1, wxh=(1400, 1000))


class TestLinkPLabelFidelity(unittest.TestCase):
    '''linkp node labels must render true to form for every kind of string, whether the
    label comes from the node name itself or from the node_labels display dict.'''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def setUp(self):
        self.assertEqual(len(FIDELITY_LABELS), len(set(FIDELITY_LABELS)),
                         'fidelity labels must be unique')

    # ------------------------------------------------------------------
    # Positive: labels from the node NAME survive the round trip exactly.
    # ------------------------------------------------------------------
    def test_node_name_labels_render_true_to_form(self):
        lp = _build_linkp(self.p2s, FIDELITY_LABELS)
        # Sanity: no node collapsed, so every label is actually drawn.
        self.assertEqual(lp.df_node.filter(pl.col('__nodes__') == 1).height, len(FIDELITY_LABELS))

        rendered = svg_text_contents(lp.svg)
        # The only <text> elements are node labels -- exactly one per label, no more.
        self.assertEqual(len(rendered), len(FIDELITY_LABELS),
                         'unexpected number of <text> elements in linkp SVG')
        for _lab_ in FIDELITY_LABELS:
            self.assertIn(_lab_, rendered,
                          f'label {_lab_!r} missing or corrupted in rendered SVG')
        self.assertEqual(sorted(rendered), sorted(FIDELITY_LABELS))

    # ------------------------------------------------------------------
    # Positive: labels from the node_labels DICT survive the round trip exactly.
    # ------------------------------------------------------------------
    def test_node_labels_dict_render_true_to_form(self):
        _nodes_ = [f'n{i}' for i in range(len(FIDELITY_LABELS))]
        _node_labels_ = {_n_: FIDELITY_LABELS[_i_] for _i_, _n_ in enumerate(_nodes_)}
        lp = _build_linkp(self.p2s, _nodes_, node_labels=_node_labels_)

        rendered = svg_text_contents(lp.svg)
        # Node names (n0, n1, ...) must NOT appear -- only their mapped display strings.
        for _n_ in _nodes_:
            self.assertNotIn(_n_, rendered)
        self.assertEqual(len(rendered), len(FIDELITY_LABELS))
        for _disp_ in FIDELITY_LABELS:
            self.assertIn(_disp_, rendered,
                          f'node_labels display {_disp_!r} missing or corrupted in SVG')
        self.assertEqual(sorted(rendered), sorted(FIDELITY_LABELS))

    def test_rendered_svg_is_well_formed_xml(self):
        # Escaping must keep the SVG parseable even with &, <, > in labels.
        lp = _build_linkp(self.p2s, FIDELITY_LABELS)
        ET.fromstring(lp.svg)  # raises on malformed XML

    def test_special_symbol_labels_are_escaped_in_source(self):
        # The raw SVG string must carry entity-escaped forms, not bare &, <, > from labels.
        lp = _build_linkp(self.p2s, ['a & b', 'x < y', 'p > q'])
        self.assertIn('a &amp; b', lp.svg)
        self.assertIn('x &lt; y', lp.svg)
        self.assertIn('p &gt; q', lp.svg)

    # ------------------------------------------------------------------
    # Negative: re-applying the disabled rounding provably corrupts labels
    # from both the node name and the node_labels dict.
    # ------------------------------------------------------------------
    def test_faulty_rounding_would_corrupt_node_name_labels(self):
        lp = _build_linkp(self.p2s, FIDELITY_LABELS)
        self.assertIn('1.172.32.1', svg_text_contents(lp.svg))  # correct as rendered

        rendered_after = svg_text_contents(faulty_round_svg_floats(lp.svg))
        # The exact bug from the field report: 1.172.32.1 -> 1.17.32.1.
        self.assertNotIn('1.172.32.1', rendered_after)
        self.assertIn('1.17.32.1', rendered_after)
        for _victim_ in KNOWN_ROUNDING_VICTIMS:
            self.assertNotIn(_victim_, rendered_after,
                             f'faulty rounding should have corrupted {_victim_!r}')

    def test_faulty_rounding_would_corrupt_node_labels_dict(self):
        _nodes_ = [f'n{i}' for i in range(len(FIDELITY_LABELS))]
        _node_labels_ = {_n_: FIDELITY_LABELS[_i_] for _i_, _n_ in enumerate(_nodes_)}
        lp = _build_linkp(self.p2s, _nodes_, node_labels=_node_labels_)

        rendered_after = svg_text_contents(faulty_round_svg_floats(lp.svg))
        for _victim_ in KNOWN_ROUNDING_VICTIMS:
            self.assertNotIn(_victim_, rendered_after,
                             f'faulty rounding should have corrupted dict display {_victim_!r}')

    # ------------------------------------------------------------------
    # Guard: the production rounding helper must stay OFF (a no-op). If it is
    # ever re-enabled without a text-content-safe rewrite, this fails loudly.
    # ------------------------------------------------------------------
    def test_production_rounding_is_disabled(self):
        probe = '<text x="1.172">1.172.32.1</text> more="123.456789"'
        self.assertEqual(self.p2s.roundSvgFloats(probe), probe,
                         'roundSvgFloats() is enabled again -- it corrupts labels; '
                         'see the TODO on Polars2SVG.roundSvgFloats')


if __name__ == '__main__':
    unittest.main()
