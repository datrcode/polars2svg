"""Adversarial input combinations across components.

Each test feeds a degenerate or hostile input shape (empty frames, single rows,
NaN/inf coordinates, nulls, XML-special and unicode characters, self-loops,
zero/negative counts) and asserts the component renders *well-formed* SVG --
parsed with an XML parser, not just substring-checked -- or degrades gracefully.

Regression anchors for previously-fixed bugs:
  - chordp draw_labels emitted raw node names -> malformed XML when a name
    contained '&' or '<' (fixed with html.escape).
  - chordp on an empty/single-node edge frame crashed in scipy linkage
    (leafWalkFromEdges now short-circuits when n <= 1).
  - timep on an empty frame crashed building the datetime spine from null
    min/max (now produces an empty spine -> blank chart).
  - xyp crashed casting NaN/±inf coordinates (now dropped like nulls).
"""
import unittest
import datetime
import xml.etree.ElementTree as ET

import polars as pl
from polars2svg import Polars2SVG


def _assert_wellformed_svg(tc, svg):
    tc.assertIsInstance(svg, str)
    root = ET.fromstring(svg)  # raises ParseError on malformed markup
    tc.assertTrue(root.tag.endswith('svg'))


class _EdgeCaseBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p2s = Polars2SVG()

    def _wellformed(self, component):
        svg = component._repr_svg_()
        _assert_wellformed_svg(self, svg)
        return svg


# ─────────────────────────────────────────────────────────────────────────────
# Empty DataFrames: every component must render a blank chart, not crash
# ─────────────────────────────────────────────────────────────────────────────

class TestEmptyDataFrames(_EdgeCaseBase):

    def test_histop_empty(self):
        df = pl.DataFrame({'cat': [], 'v': []}, schema={'cat': pl.String, 'v': pl.Int64})
        self._wellformed(self.p2s.histop(df, 'cat'))

    def test_xyp_empty(self):
        df = pl.DataFrame({'x': [], 'y': []}, schema={'x': pl.Float64, 'y': pl.Float64})
        self._wellformed(self.p2s.xyp(df, 'x', 'y'))

    def test_timep_empty_datetime(self):
        '''Regression: empty df crashed building the datetime spine (null start).'''
        df = pl.DataFrame({'ts': [], 'v': []}, schema={'ts': pl.Datetime, 'v': pl.Int64})
        self._wellformed(self.p2s.timep(df, 'ts'))

    def test_timep_empty_date(self):
        df = pl.DataFrame({'ts': [], 'v': []}, schema={'ts': pl.Date, 'v': pl.Int64})
        self._wellformed(self.p2s.timep(df, 'ts'))

    def test_timep_empty_periodic(self):
        df = pl.DataFrame({'ts': [], 'v': []}, schema={'ts': pl.Datetime, 'v': pl.Int64})
        self._wellformed(self.p2s.timep(df, ('ts', self.p2s.PT_DoWp)))

    def test_timep_empty_with_count_and_color(self):
        df = pl.DataFrame({'ts': [], 'v': []}, schema={'ts': pl.Datetime, 'v': pl.Int64})
        self._wellformed(self.p2s.timep(df, 'ts', count='v', color='v'))

    def test_chordp_empty(self):
        '''Regression: empty edge frame crashed in scipy linkage.'''
        df = pl.DataFrame({'fm': [], 'to': []}, schema={'fm': pl.String, 'to': pl.String})
        self._wellformed(self.p2s.chordp(df=df, relationships=[('fm', 'to')]))

    def test_linkp_empty(self):
        df = pl.DataFrame({'fm': [], 'to': []}, schema={'fm': pl.String, 'to': pl.String})
        self._wellformed(self.p2s.linkp(df, relationships=[('fm', 'to')], pos={}))

    def test_spreadlinesp_empty(self):
        df = pl.DataFrame({'fm': [], 'to': [], 'time': []},
                          schema={'fm': pl.String, 'to': pl.String, 'time': pl.Datetime})
        self._wellformed(self.p2s.spreadlinesp(df, [('fm', 'to')], ego='a', time='time'))


# ─────────────────────────────────────────────────────────────────────────────
# Single-row / single-node degenerate inputs
# ─────────────────────────────────────────────────────────────────────────────

class TestSingleRowInputs(_EdgeCaseBase):

    def test_histop_single_row(self):
        self._wellformed(self.p2s.histop(pl.DataFrame({'cat': ['a'], 'v': [1]}), 'cat'))

    def test_timep_single_timestamp_zero_range(self):
        df = pl.DataFrame({'ts': [datetime.datetime(2024, 1, 1)] * 3, 'v': [1, 2, 3]})
        self._wellformed(self.p2s.timep(df, 'ts'))

    def test_xyp_single_point(self):
        self._wellformed(self.p2s.xyp(pl.DataFrame({'x': [1.0], 'y': [2.0]}), 'x', 'y'))

    def test_xyp_zero_range_all_identical(self):
        df = pl.DataFrame({'x': [5.0, 5.0, 5.0], 'y': [7.0, 7.0, 7.0]})
        self._wellformed(self.p2s.xyp(df, 'x', 'y'))

    def test_chordp_single_self_loop_node(self):
        '''Regression: a lone node crashed in scipy linkage (1x1 distance matrix).'''
        self._wellformed(self.p2s.chordp(df=pl.DataFrame({'fm': ['a'], 'to': ['a']}),
                                         relationships=[('fm', 'to')]))

    def test_linkp_single_edge(self):
        self._wellformed(self.p2s.linkp(pl.DataFrame({'fm': ['a'], 'to': ['b']}),
                                        relationships=[('fm', 'to')],
                                        pos={'a': [0, 0], 'b': [1, 1]}))

    def test_spreadlinesp_single_row(self):
        df = pl.DataFrame({'fm': ['a'], 'to': ['b'], 'time': [datetime.datetime(2024, 1, 1)]})
        self._wellformed(self.p2s.spreadlinesp(df, [('fm', 'to')], ego='a', time='time'))


# ─────────────────────────────────────────────────────────────────────────────
# Non-finite and null values
# ─────────────────────────────────────────────────────────────────────────────

class TestNonFiniteAndNullValues(_EdgeCaseBase):

    def test_xyp_nan_coordinates_dropped(self):
        '''Regression: NaN in x/y crashed with InvalidOperationError; treated as null now.'''
        df = pl.DataFrame({'x': [1.0, float('nan'), 3.0], 'y': [1.0, 2.0, 3.0]})
        self._wellformed(self.p2s.xyp(df, 'x', 'y'))

    def test_xyp_inf_coordinates_dropped(self):
        df = pl.DataFrame({'x': [1.0, 2.0, 3.0], 'y': [1.0, float('inf'), float('-inf')]})
        self._wellformed(self.p2s.xyp(df, 'x', 'y'))

    def test_xyp_all_nan(self):
        df = pl.DataFrame({'x': [float('nan')], 'y': [float('nan')]})
        self._wellformed(self.p2s.xyp(df, 'x', 'y'))

    def test_xyp_nan_dropped_same_as_null(self):
        '''A NaN row and a null row must yield identical geometry (both dropped).
        SVG strings differ by per-render random ids, so compare shape geometry.'''
        def _rect_geometry_(svg):
            root = ET.fromstring(svg)
            return sorted(tuple(el.get(a) for a in ('x', 'y', 'width', 'height'))
                          for el in root.iter() if el.tag.endswith('rect'))
        df_nan  = pl.DataFrame({'x': [1.0, float('nan'), 3.0], 'y': [1.0, 2.0, 3.0]})
        df_null = pl.DataFrame({'x': [1.0, None,         3.0], 'y': [1.0, 2.0, 3.0]})
        svg_nan  = self.p2s.xyp(df_nan,  'x', 'y')._repr_svg_()
        svg_null = self.p2s.xyp(df_null, 'x', 'y')._repr_svg_()
        self.assertEqual(_rect_geometry_(svg_nan), _rect_geometry_(svg_null))

    def test_histop_null_bin_values(self):
        df = pl.DataFrame({'cat': ['a', None, 'b', 'a'], 'v': [1, 2, 3, None]})
        self._wellformed(self.p2s.histop(df, 'cat'))

    def test_histop_null_count_field(self):
        df = pl.DataFrame({'cat': ['a', None, 'b', 'a'], 'v': [1, 2, 3, None]})
        self._wellformed(self.p2s.histop(df, 'cat', count='v'))

    def test_timep_null_timestamps(self):
        df = pl.DataFrame({'ts': [datetime.datetime(2024, 1, 1), None,
                                  datetime.datetime(2024, 1, 3)], 'v': [1, 2, 3]})
        self._wellformed(self.p2s.timep(df, 'ts'))


# ─────────────────────────────────────────────────────────────────────────────
# Zero / negative count magnitudes
# ─────────────────────────────────────────────────────────────────────────────

class TestDegenerateCountValues(_EdgeCaseBase):

    def test_histop_all_zero_counts(self):
        df = pl.DataFrame({'cat': ['a', 'b'], 'v': [0, 0]})
        self._wellformed(self.p2s.histop(df, 'cat', count='v'))

    def test_histop_negative_counts(self):
        df = pl.DataFrame({'cat': ['a', 'b', 'c'], 'v': [-5, 3, -2]})
        self._wellformed(self.p2s.histop(df, 'cat', count='v'))

    def test_timep_all_zero_counts(self):
        df = pl.DataFrame({'ts': [datetime.datetime(2024, 1, 1),
                                  datetime.datetime(2024, 1, 2)], 'v': [0, 0]})
        self._wellformed(self.p2s.timep(df, 'ts', count='v'))

    def test_timep_negative_counts(self):
        df = pl.DataFrame({'ts': [datetime.datetime(2024, 1, 1),
                                  datetime.datetime(2024, 1, 2)], 'v': [-3, -7]})
        self._wellformed(self.p2s.timep(df, 'ts', count='v'))


# ─────────────────────────────────────────────────────────────────────────────
# XML-special and unicode characters in data-driven labels
# ─────────────────────────────────────────────────────────────────────────────

class TestSpecialCharacterLabels(_EdgeCaseBase):

    _EDGES_ = pl.DataFrame({'fm': ['A&B', 'C<D', 'E"F'], 'to': ['C<D', 'E"F', 'A&B']})

    def test_chordp_labels_radial_escaped(self):
        '''Regression: raw &/< in node names produced malformed XML.'''
        svg = self._wellformed(self.p2s.chordp(df=self._EDGES_, relationships=[('fm', 'to')],
                                               draw_labels=True, label_style='radial'))
        self.assertIn('A&amp;B', svg)
        self.assertIn('C&lt;D',  svg)

    def test_chordp_labels_circular_escaped(self):
        svg = self._wellformed(self.p2s.chordp(df=self._EDGES_, relationships=[('fm', 'to')],
                                               draw_labels=True, label_style='circular'))
        self.assertIn('A&amp;B', svg)

    def test_chordp_node_labels_map_escaped(self):
        svg = self._wellformed(self.p2s.chordp(df=self._EDGES_, relationships=[('fm', 'to')],
                                               draw_labels=True,
                                               node_labels={'A&B': 'x<&>y'}))
        self.assertNotIn('>x<&>y<', svg)

    def test_histop_special_char_bins(self):
        df = pl.DataFrame({'cat': ['a&b', 'c<d', '"q"', 'a&b'], 'v': [1, 2, 3, 4]})
        self._wellformed(self.p2s.histop(df, 'cat'))

    def test_histop_tuple_bin_special_chars(self):
        df = pl.DataFrame({'a': ['x&', 'y<'], 'b': ['1', '2'], 'v': [1, 2]})
        self._wellformed(self.p2s.histop(df, ('a', 'b')))

    def test_xyp_categorical_axis_special_chars(self):
        df = pl.DataFrame({'x': ['a&b', 'c<d', 'e'], 'y': [1.0, 2.0, 3.0]})
        self._wellformed(self.p2s.xyp(df, 'x', 'y'))

    def test_linkp_labels_special_chars(self):
        pos = {'A&B': [0, 0], 'C<D': [1, 1]}
        df  = pl.DataFrame({'fm': ['A&B', 'C<D'], 'to': ['C<D', 'A&B']})
        self._wellformed(self.p2s.linkp(df, relationships=[('fm', 'to')],
                                        pos=pos, node_labels=True))

    def test_spreadlinesp_special_char_nodes(self):
        df = pl.DataFrame({'fm': ['A&B', 'A&B', 'C<D'], 'to': ['C<D', 'E>F', 'E>F'],
                           'time': [datetime.datetime(2024, 1, d) for d in (1, 2, 3)]})
        self._wellformed(self.p2s.spreadlinesp(df, [('fm', 'to')], ego='A&B', time='time'))

    def test_unicode_labels_across_components(self):
        _hist_df_ = pl.DataFrame({'cat': ['日本語', 'emoji 🎉', 'ümlaut'], 'v': [1, 2, 3]})
        self._wellformed(self.p2s.histop(_hist_df_, 'cat'))
        _edge_df_ = pl.DataFrame({'fm': ['日本', 'α&β'], 'to': ['α&β', '日本']})
        self._wellformed(self.p2s.chordp(df=_edge_df_, relationships=[('fm', 'to')],
                                         draw_labels=True))

    # ── SECURITY.md threat-model regression: a raw script payload in row data
    #    must never survive unescaped into any labeled component's SVG output.
    _SCRIPT_PAYLOAD_ = '<script>alert(1)</script>'

    def _assert_no_raw_script(self, component):
        svg = self._wellformed(component)
        self.assertNotIn('<script>alert', svg)

    def test_script_payload_histop_bin(self):
        df = pl.DataFrame({'cat': [self._SCRIPT_PAYLOAD_, 'b'], 'v': [1, 2]})
        self._assert_no_raw_script(self.p2s.histop(df, 'cat'))

    def test_script_payload_piep_slice_label(self):
        df = pl.DataFrame({'cat': [self._SCRIPT_PAYLOAD_, 'b', 'c'], 'v': [3, 2, 1]})
        self._assert_no_raw_script(self.p2s.piep(df, 'cat', draw_labels=True))

    def test_script_payload_xyp_categorical_axis(self):
        df = pl.DataFrame({'x': [self._SCRIPT_PAYLOAD_, 'b'], 'y': [1.0, 2.0]})
        self._assert_no_raw_script(self.p2s.xyp(df, 'x', 'y'))

    def test_script_payload_chordp_label(self):
        df = pl.DataFrame({'fm': [self._SCRIPT_PAYLOAD_, 'b'], 'to': ['b', self._SCRIPT_PAYLOAD_]})
        self._assert_no_raw_script(self.p2s.chordp(df=df, relationships=[('fm', 'to')],
                                                    draw_labels=True))

    def test_script_payload_linkp_label(self):
        pos = {self._SCRIPT_PAYLOAD_: [0, 0], 'b': [1, 1]}
        df  = pl.DataFrame({'fm': [self._SCRIPT_PAYLOAD_, 'b'], 'to': ['b', self._SCRIPT_PAYLOAD_]})
        self._assert_no_raw_script(self.p2s.linkp(df, relationships=[('fm', 'to')],
                                                   pos=pos, node_labels=True))

    # spreadlinesp built <text> for timestamp/annotation labels by raw f-string
    # interpolation, bypassing svgText()/html.escape -- so an XML-special
    # character in a timestamp column value or an anno= label survived unescaped
    # and produced malformed SVG (SECURITY.md lists timestamp labels as
    # untrusted). These lock the escaping on both paths.

    def test_spreadlinesp_special_char_timestamp(self):
        # draw_context=True renders per-bin timestamp labels; feed XML-special
        # chars short enough to survive the _ts_label_len_ truncation.
        df = pl.DataFrame({'fm': ['a', 'a', 'b', 'b'], 'to': ['b', 'c', 'c', 'a'],
                           'time': ['<a>&"1', '<a>&"1', 'z>y<', 'z>y<']})
        svg = self._wellformed(self.p2s.spreadlinesp(df, [('fm', 'to')], ego='a',
                                                     time='time', draw_context=True))
        # No unescaped structural markup from the timestamp data survives.
        self.assertNotIn('<a>', svg)

    def test_spreadlinesp_special_char_anno(self):
        df = pl.DataFrame({'fm': ['a', 'a', 'b', 'b'], 'to': ['b', 'c', 'c', 'a'],
                           'time': ['2020', '2020', '2021', '2021']})
        svg = self._wellformed(self.p2s.spreadlinesp(df, [('fm', 'to')], ego='a',
                                                     time='time',
                                                     anno={'2021': self._SCRIPT_PAYLOAD_}))
        self.assertNotIn('<script>alert', svg)


# ─────────────────────────────────────────────────────────────────────────────
# Graph-shape oddities: self-loops, parallel edges, nodes missing from pos/ego
# ─────────────────────────────────────────────────────────────────────────────

class TestGraphShapeOddities(_EdgeCaseBase):

    def test_chordp_self_loops(self):
        df = pl.DataFrame({'fm': ['a', 'b', 'a'], 'to': ['a', 'b', 'b']})
        self._wellformed(self.p2s.chordp(df=df, relationships=[('fm', 'to')]))

    def test_chordp_parallel_edges(self):
        df = pl.DataFrame({'fm': ['a', 'a', 'a', 'b'], 'to': ['b', 'b', 'b', 'a']})
        self._wellformed(self.p2s.chordp(df=df, relationships=[('fm', 'to')]))

    def test_linkp_self_loops(self):
        df = pl.DataFrame({'fm': ['a', 'b', 'a'], 'to': ['a', 'b', 'b']})
        self._wellformed(self.p2s.linkp(df, relationships=[('fm', 'to')],
                                        pos={'a': [0, 0], 'b': [1, 1]}))

    def test_linkp_node_missing_from_pos(self):
        df = pl.DataFrame({'fm': ['a', 'b'], 'to': ['b', 'c']})
        self._wellformed(self.p2s.linkp(df, relationships=[('fm', 'to')],
                                        pos={'a': [0, 0], 'b': [1, 1]}))

    def test_spreadlinesp_ego_not_in_data(self):
        df = pl.DataFrame({'fm': ['a'], 'to': ['b'], 'time': [datetime.datetime(2024, 1, 1)]})
        self._wellformed(self.p2s.spreadlinesp(df, [('fm', 'to')], ego='zzz', time='time'))


if __name__ == '__main__':
    unittest.main()
