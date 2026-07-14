import unittest
from math import cos, sin

import polars as pl
from polars2svg import Polars2SVG

_DF_ = pl.DataFrame({
    'fm':     ['a', 'b', 'c', 'a', 'b', 'd'],
    'to':     ['b', 'c', 'a', 'c', 'a', 'a'],
    'weight': [1,   3,   2,   1,   4,   2  ],
})
_RELS_ = [('fm', 'to')]


def _ch(**extra):
    p2s = Polars2SVG()
    return p2s.chordp(df=_DF_, relationships=_RELS_, **extra)


def _arc_midpoint(ch, node):
    """Return (x, y) on the ring at the arc midpoint angle of the given node."""
    row = ch.df_node.filter(pl.col('__nm__').cast(pl.String) == str(node))
    amr = row['__amr__'][0]
    r_mid = (ch.r + ch.r_inner) / 2.0
    return (ch.cx + r_mid * cos(amr), ch.cy + r_mid * sin(amr))


def _small_bbox(x, y, size=5.0):
    return (x - size, y - size, x + size, y + size)


class TestChordPRecordsAt(unittest.TestCase):

    def setUp(self):
        self.ch = _ch()

    # --- basic hit ---

    def test_records_at_node_a_arc_returns_rows_involving_a(self):
        xy = _arc_midpoint(self.ch, 'a')
        result = self.ch.recordsAt(xy)
        fms = set(result['fm'].to_list())
        tos = set(result['to'].to_list())
        self.assertTrue('a' in fms or 'a' in tos,
                        'Expected rows involving node a')

    def test_records_at_node_b_arc_returns_rows_involving_b(self):
        xy = _arc_midpoint(self.ch, 'b')
        result = self.ch.recordsAt(xy)
        fms = set(result['fm'].to_list())
        tos = set(result['to'].to_list())
        self.assertTrue('b' in fms or 'b' in tos,
                        'Expected rows involving node b')

    def test_records_at_returns_dataframe(self):
        xy = _arc_midpoint(self.ch, 'a')
        result = self.ch.recordsAt(xy)
        self.assertIsInstance(result, pl.DataFrame)

    def test_records_at_correct_columns(self):
        xy = _arc_midpoint(self.ch, 'a')
        result = self.ch.recordsAt(xy)
        self.assertIn('fm', result.columns)
        self.assertIn('to', result.columns)

    # --- outside the ring ---

    def test_records_at_center_returns_empty(self):
        # Center of circle is far inside the ring → empty result
        xy = (self.ch.cx, self.ch.cy)
        result = self.ch.recordsAt(xy)
        self.assertEqual(len(result), 0)

    def test_records_at_far_outside_returns_empty(self):
        # Far outside the ring
        xy = (self.ch.cx + self.ch.r * 3, self.ch.cy)
        result = self.ch.recordsAt(xy)
        self.assertEqual(len(result), 0)

    # --- default shape ---

    def test_records_at_default_shape_matches_circle(self):
        p2s = Polars2SVG()
        xy = _arc_midpoint(self.ch, 'a')
        result_default = self.ch.recordsAt(xy)
        result_circle  = self.ch.recordsAt(xy, shape=p2s.SELECT_CIRCLEp)
        self.assertTrue(result_default.equals(result_circle))

    # --- unsupported shapes raise ValueError ---

    def test_records_at_horizontal_shape_raises(self):
        p2s = Polars2SVG()
        xy = _arc_midpoint(self.ch, 'a')
        with self.assertRaises(ValueError):
            self.ch.recordsAt(xy, shape=p2s.SELECT_HORIZONTALp)

    def test_records_at_vertical_shape_raises(self):
        p2s = Polars2SVG()
        xy = _arc_midpoint(self.ch, 'a')
        with self.assertRaises(ValueError):
            self.ch.recordsAt(xy, shape=p2s.SELECT_VERTICALp)


class TestChordPFilterByRectangle(unittest.TestCase):

    def setUp(self):
        self.ch = _ch()

    # --- full bbox selects all rows ---

    def test_filter_full_bbox_returns_all_rows(self):
        w, h = self.ch.wxh
        result = self.ch.filterByRectangle((0, 0, w, h))
        self.assertEqual(len(result), len(_DF_))

    def test_filter_full_bbox_remove_records_returns_empty(self):
        w, h = self.ch.wxh
        result = self.ch.filterByRectangle((0, 0, w, h), remove_records=True)
        self.assertEqual(len(result), 0)

    # --- empty bbox returns empty ---

    def test_filter_empty_bbox_returns_empty(self):
        result = self.ch.filterByRectangle((-100, -100, -50, -50))
        self.assertEqual(len(result), 0)

    def test_filter_empty_bbox_remove_records_returns_all(self):
        result = self.ch.filterByRectangle((-100, -100, -50, -50), remove_records=True)
        self.assertEqual(len(result), len(_DF_))

    # --- returns a DataFrame ---

    def test_filter_returns_dataframe(self):
        w, h = self.ch.wxh
        result = self.ch.filterByRectangle((0, 0, w, h))
        self.assertIsInstance(result, pl.DataFrame)

    def test_filter_preserves_columns(self):
        w, h = self.ch.wxh
        result = self.ch.filterByRectangle((0, 0, w, h))
        for col in _DF_.columns:
            self.assertIn(col, result.columns)

    # --- induced-subgraph semantics: only edges within selected nodes are returned ---

    def test_filter_single_node_returns_empty_no_self_loops(self):
        # Selecting only one node: no edge can have BOTH endpoints in a 1-element set
        mx, my = _arc_midpoint(self.ch, 'a')
        result = self.ch.filterByRectangle(_small_bbox(mx, my, size=3.0))
        # Result may be empty or have rows, but all returned rows must have fm AND to in {'a'}
        for row in result.iter_rows(named=True):
            self.assertIn(str(row['fm']), {'a'})
            self.assertIn(str(row['to']), {'a'})

    def test_filter_two_nodes_a_and_b_returns_only_ab_edges(self):
        # Construct a bbox enclosing both node 'a' and node 'b' arc midpoints
        ax, ay = _arc_midpoint(self.ch, 'a')
        bx, by = _arc_midpoint(self.ch, 'b')
        x0, x1 = min(ax, bx) - 5, max(ax, bx) + 5
        y0, y1 = min(ay, by) - 5, max(ay, by) + 5
        result = self.ch.filterByRectangle((x0, y0, x1, y1))
        # All returned rows must have fm AND to within the selected set
        selected = {'a', 'b'}
        for row in result.iter_rows(named=True):
            self.assertIn(str(row['fm']), selected,
                          f"fm={row['fm']} should be in {selected}")
            self.assertIn(str(row['to']), selected,
                          f"to={row['to']} should be in {selected}")

    # --- bbox normalization (x0>x1 or y0>y1 should still work) ---

    def test_filter_inverted_bbox_normalized(self):
        w, h = self.ch.wxh
        result_normal   = self.ch.filterByRectangle((0, 0, w, h))
        result_inverted = self.ch.filterByRectangle((w, h, 0, 0))
        self.assertEqual(len(result_normal), len(result_inverted))


class TestChordPControllerRegistration(unittest.TestCase):
    """Smoke test: ChP is registered in _PLOT_TYPE_TO_WRAPPER_."""

    def test_chordpi_in_wrapper_dict(self):
        from polars2svg.interactive_controller import _PLOT_TYPE_TO_WRAPPER_
        self.assertIn('ChP', _PLOT_TYPE_TO_WRAPPER_)

    def test_chordpi_callable(self):
        from polars2svg.interactive_controller import _PLOT_TYPE_TO_WRAPPER_
        self.assertTrue(callable(_PLOT_TYPE_TO_WRAPPER_['ChP']))

    def test_chordpi_config_entry_exists(self):
        from polars2svg.interactive_controller import _INTERACTIVEP_CONFIG_
        self.assertIn('chordpi', _INTERACTIVEP_CONFIG_)

    def test_chordpi_config_has_required_keys(self):
        from polars2svg.interactive_controller import _INTERACTIVEP_CONFIG_
        cfg = _INTERACTIVEP_CONFIG_['chordpi']
        for key in ('class_name', 'svg_parent_id', 'render_fn',
                    'fallback_shape', 'brush_seq', 'has_z_key'):
            self.assertIn(key, cfg)

    def test_chordpi_render_fn_is_chordp(self):
        from polars2svg.interactive_controller import _INTERACTIVEP_CONFIG_
        self.assertEqual(_INTERACTIVEP_CONFIG_['chordpi']['render_fn'], 'chordp')

    def test_chordpi_fallback_shape_is_circle(self):
        from polars2svg.interactive_controller import _INTERACTIVEP_CONFIG_
        self.assertEqual(_INTERACTIVEP_CONFIG_['chordpi']['fallback_shape'], 'SELECT_CIRCLEp')


if __name__ == '__main__':
    unittest.main()
