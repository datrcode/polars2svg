"""Adversarial input combinations across components.

Each test feeds a degenerate or hostile input shape (empty frames, single rows,
NaN/inf coordinates, nulls, XML-special and unicode characters, self-loops,
zero/negative counts) and asserts two things:

  1. The render is clean (assertRendersCleanly): it parses as XML, it passes
     the output contract (checkOutputContract, SECURITY.md Profile A), and the
     canvas is the size the component was asked for.
  2. It shows what the input means.  An empty frame draws no data and prints no
     Python `None`; a single row draws one mark; a NaN or null row draws exactly
     what deleting that row does; a label reads back as the text it was given.
     Wherever two inputs should mean the same thing, the test asserts that the
     two renders are identical.

These tests used to check only that the SVG parsed, so a chart that drew the
wrong thing -- or nothing -- passed (PLANNING.md V11).  Asserting what they show
found that a null timestamp switched timep to hourly bins for midnight-only data
(n_unique() counted the null as a distinct hour), and that an empty xyp printed
'None' at both ends of both axes.

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
import logging
import re
import xml.etree.ElementTree as ET

import polars as pl
from polars2svg import Polars2SVG, checkOutputContract
from svg_test_utils import capture_log_warnings, normalize_svg


# The text of every <text> element, as a reader sees it (entities decoded)
def _texts_(svg: str) -> list:
    return [''.join(_el_.itertext()) for _el_ in ET.fromstring(svg).iter() if _el_.tag.endswith('text')]


def _tag_(el: ET.Element) -> str: return el.tag.split('}')[-1]


#
# _dataShapes_() -- the marks a render draws for its data: every circle, path and
# polygon; every filled rect except the full-canvas background; and the dots inside
# xyp's dot group, whose fill lives on the group rather than on each dot.
#
def _dataShapes_(svg: str, wxh: tuple) -> list:
    _root_, _out_ = ET.fromstring(svg), []
    _bg_ = ('0', '0', str(wxh[0]), str(wxh[1]))
    for _el_ in _root_.iter():
        _t_ = _tag_(_el_)
        if _t_ == 'g' and re.search(r'(circle|rect)-group-', _el_.get('class', '')):
            _out_.extend(_tag_(_c_) for _c_ in _el_)
        elif _t_ in ('circle', 'ellipse', 'path', 'polygon', 'polyline'):
            _out_.append(_t_)
        elif _t_ == 'rect' and _el_.get('fill') not in (None, 'none') and \
             (_el_.get('x'), _el_.get('y'), _el_.get('width'), _el_.get('height')) != _bg_:
            _out_.append(_t_)
    return _out_


class _EdgeCaseBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p2s = Polars2SVG()

    # Parses, passes the output contract, and is the size it was asked to be
    def assertRendersCleanly(self, component) -> str:
        svg  = component._repr_svg_()
        self.assertIsInstance(svg, str)
        root = ET.fromstring(svg)  # raises ParseError on malformed markup
        self.assertTrue(root.tag.endswith('svg'))
        self.assertEqual(checkOutputContract(svg), [], 'the render breaks the output contract')
        self.assertEqual((root.get('width'), root.get('height')), tuple(str(v) for v in component.wxh),
                         'the canvas is not the size the component was asked for')
        return svg

    # A clean render that draws nothing for its data and prints no Python None
    def assertBlank(self, component) -> str:
        svg = self.assertRendersCleanly(component)
        self.assertEqual(_dataShapes_(svg, component.wxh), [], 'drew marks for data that is not there')
        self.assertNotIn('None', _texts_(svg), "printed Python's None on the chart")
        return svg

    # Two inputs that mean the same thing render identically (and cleanly)
    def assertSameRender(self, a, b, msg: str) -> None:
        self.assertEqual(normalize_svg(self.assertRendersCleanly(a)), normalize_svg(self.assertRendersCleanly(b)), msg)

    # Each label is on the chart as a reader sees it -- whole, or cropped to a prefix and '...'
    def assertShowsText(self, svg: str, *labels: str) -> None:
        _shown_ = _texts_(svg)
        for _label_ in labels:
            _hit_ = any(_t_ == _label_ or (_t_.endswith('...') and len(_t_) > 3 and _label_.startswith(_t_[:-3]))
                        for _t_ in _shown_)
            self.assertTrue(_hit_, f'{_label_!r} is not on the chart; its text is {_shown_}')


# ─────────────────────────────────────────────────────────────────────────────
# Empty DataFrames: every component must render a blank chart, not crash
# ─────────────────────────────────────────────────────────────────────────────

class TestEmptyDataFrames(_EdgeCaseBase):

    def test_histop_empty(self):
        df = pl.DataFrame({'cat': [], 'v': []}, schema={'cat': pl.String, 'v': pl.Int64})
        self.assertBlank(self.p2s.histop(df, 'cat'))

    def test_xyp_empty(self):
        '''An empty axis has no extent: it used to print 'None' at both ends of both axes.'''
        df = pl.DataFrame({'x': [], 'y': []}, schema={'x': pl.Float64, 'y': pl.Float64})
        self.assertBlank(self.p2s.xyp(df, 'x', 'y'))

    def test_timep_empty_datetime(self):
        '''Regression: empty df crashed building the datetime spine (null start).'''
        df = pl.DataFrame({'ts': [], 'v': []}, schema={'ts': pl.Datetime, 'v': pl.Int64})
        self.assertBlank(self.p2s.timep(df, 'ts'))

    def test_timep_empty_date(self):
        df = pl.DataFrame({'ts': [], 'v': []}, schema={'ts': pl.Date, 'v': pl.Int64})
        self.assertBlank(self.p2s.timep(df, 'ts'))

    def test_timep_empty_periodic(self):
        df = pl.DataFrame({'ts': [], 'v': []}, schema={'ts': pl.Datetime, 'v': pl.Int64})
        self.assertBlank(self.p2s.timep(df, ('ts', self.p2s.PT_DoWp)))

    def test_timep_empty_with_count_and_color(self):
        df = pl.DataFrame({'ts': [], 'v': []}, schema={'ts': pl.Datetime, 'v': pl.Int64})
        self.assertBlank(self.p2s.timep(df, 'ts', count='v', color='v'))

    def test_chordp_empty(self):
        '''Regression: empty edge frame crashed in scipy linkage.'''
        df = pl.DataFrame({'fm': [], 'to': []}, schema={'fm': pl.String, 'to': pl.String})
        self.assertBlank(self.p2s.chordp(df=df, relationships=[('fm', 'to')]))

    def test_linkp_empty(self):
        df = pl.DataFrame({'fm': [], 'to': []}, schema={'fm': pl.String, 'to': pl.String})
        self.assertBlank(self.p2s.linkp(df, relationships=[('fm', 'to')], pos={}))

    def test_spreadlinesp_empty(self):
        df = pl.DataFrame({'fm': [], 'to': [], 'time': []},
                          schema={'fm': pl.String, 'to': pl.String, 'time': pl.Datetime})
        self.assertBlank(self.p2s.spreadlinesp(df, [('fm', 'to')], ego='a', time='time'))


# ─────────────────────────────────────────────────────────────────────────────
# No relationships: the graph components name what is missing
# ─────────────────────────────────────────────────────────────────────────────

class TestNoRelationships(_EdgeCaseBase):
    '''Leaving relationships= out used to fail with "'NoneType' object is not iterable", from
    the loop that expands tuple endpoints.  An empty list already raised the named error;
    leaving it out now means the same (PLANNING.md §5 C-chordp-no-relationships).'''

    def assertNamesMissingRelationships(self, name: str, **kwargs) -> None:
        df = pl.DataFrame({'fm': ['a', 'b'], 'to': ['b', 'c'],
                           'time': [datetime.datetime(2026, 1, 1), datetime.datetime(2026, 1, 2)]})
        for _given_ in ({}, {'relationships': []}):
            with self.subTest(component=name, given=_given_):
                with self.assertRaisesRegex(ValueError, 'relationships must be specified'):
                    getattr(self.p2s, name)(df=df, **_given_, **kwargs)

    def test_chordp(self):
        self.assertNamesMissingRelationships('chordp')

    def test_linkp(self):
        self.assertNamesMissingRelationships('linkp')

    def test_spreadlinesp(self):
        self.assertNamesMissingRelationships('spreadlinesp', ego='a', time='time')


# ─────────────────────────────────────────────────────────────────────────────
# Single-row / single-node degenerate inputs
# ─────────────────────────────────────────────────────────────────────────────

class TestSingleRowInputs(_EdgeCaseBase):

    def test_histop_single_row(self):
        _h_ = self.p2s.histop(pl.DataFrame({'cat': ['a'], 'v': [1]}), 'cat')
        svg = self.assertRendersCleanly(_h_)
        self.assertGreater(len(_dataShapes_(svg, _h_.wxh)), 0, 'no bar drawn')
        self.assertShowsText(svg, 'a')

    def test_timep_single_timestamp_zero_range(self):
        # Three rows at one instant are one bar of three
        _t_ = self.p2s.timep(pl.DataFrame({'ts': [datetime.datetime(2024, 1, 1)] * 3, 'v': [1, 2, 3]}), 'ts')
        svg = self.assertRendersCleanly(_t_)
        self.assertEqual(len(_dataShapes_(svg, _t_.wxh)), 1, 'expected exactly one bar')
        self.assertShowsText(svg, '3', '2024-01-01')

    def test_xyp_single_point(self):
        _x_ = self.p2s.xyp(pl.DataFrame({'x': [1.0], 'y': [2.0]}), 'x', 'y')
        self.assertEqual(len(_dataShapes_(self.assertRendersCleanly(_x_), _x_.wxh)), 1, 'expected exactly one dot')

    def test_xyp_zero_range_all_identical(self):
        # Three identical rows are one dot, drawn where a single row would be
        df = pl.DataFrame({'x': [5.0, 5.0, 5.0], 'y': [7.0, 7.0, 7.0]})
        self.assertSameRender(self.p2s.xyp(df, 'x', 'y'), self.p2s.xyp(df.head(1), 'x', 'y'),
                              'identical rows must draw what one of them draws')

    def test_chordp_single_self_loop_node(self):
        '''Regression: a lone node crashed in scipy linkage (1x1 distance matrix).'''
        _c_ = self.p2s.chordp(df=pl.DataFrame({'fm': ['a'], 'to': ['a']}), relationships=[('fm', 'to')])
        self.assertEqual(_dataShapes_(self.assertRendersCleanly(_c_), _c_.wxh), ['path'], 'expected one arc')

    def test_linkp_single_edge(self):
        _l_ = self.p2s.linkp(pl.DataFrame({'fm': ['a'], 'to': ['b']}), relationships=[('fm', 'to')],
                             pos={'a': [0, 0], 'b': [1, 1]})
        root = ET.fromstring(self.assertRendersCleanly(_l_))
        self.assertEqual((sum(_tag_(e) == 'circle' for e in root.iter()), sum(_tag_(e) == 'line' for e in root.iter())),
                         (2, 1), 'expected two nodes and one link')

    def test_spreadlinesp_single_row(self):
        _s_ = self.p2s.spreadlinesp(pl.DataFrame({'fm': ['a'], 'to': ['b'], 'time': [datetime.datetime(2024, 1, 1)]}),
                                    [('fm', 'to')], ego='a', time='time')
        svg = self.assertRendersCleanly(_s_)
        self.assertGreater(len(_dataShapes_(svg, _s_.wxh)), 0, 'nothing drawn')
        self.assertShowsText(svg, '2024-01-01')


# ─────────────────────────────────────────────────────────────────────────────
# Non-finite and null values: a row that cannot be placed draws what deleting it does
# ─────────────────────────────────────────────────────────────────────────────

class TestNonFiniteAndNullValues(_EdgeCaseBase):

    def test_xyp_nan_coordinates_dropped(self):
        '''Regression: NaN in x/y crashed with InvalidOperationError; treated as null now.'''
        df = pl.DataFrame({'x': [1.0, float('nan'), 3.0], 'y': [1.0, 2.0, 3.0]})
        self.assertSameRender(self.p2s.xyp(df, 'x', 'y'), self.p2s.xyp(df.filter(pl.col('x').is_not_nan()), 'x', 'y'),
                              'a NaN row must draw what deleting it does')

    def test_xyp_inf_coordinates_dropped(self):
        df = pl.DataFrame({'x': [1.0, 2.0, 3.0], 'y': [1.0, float('inf'), float('-inf')]})
        self.assertSameRender(self.p2s.xyp(df, 'x', 'y'), self.p2s.xyp(df.filter(pl.col('y').is_finite()), 'x', 'y'),
                              'a +-inf row must draw what deleting it does')

    def test_xyp_all_nan(self):
        df = pl.DataFrame({'x': [float('nan')], 'y': [float('nan')]})
        _x_ = self.p2s.xyp(df, 'x', 'y')
        self.assertBlank(_x_)
        self.assertSameRender(_x_, self.p2s.xyp(df.clear(), 'x', 'y'), 'an all-NaN frame must draw what an empty one does')

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
        _h_ = self.p2s.histop(df, 'cat')
        svg = self.assertRendersCleanly(_h_)
        self.assertGreater(len(_dataShapes_(svg, _h_.wxh)), 0, 'no bars drawn')
        self.assertShowsText(svg, 'a', 'b')

    def test_histop_null_count_field(self):
        # A null count counts as zero
        df = pl.DataFrame({'cat': ['a', None, 'b', 'a'], 'v': [1, 2, 3, None]})
        self.assertSameRender(self.p2s.histop(df, 'cat', count='v'),
                              self.p2s.histop(df.with_columns(pl.col('v').fill_null(0)), 'cat', count='v'),
                              'a null count must draw what a count of 0 does')

    def test_timep_null_timestamps(self):
        '''A null timestamp draws -- and offers -- what deleting its row does.  It used to
        make midnight-only data look sub-daily: n_unique() counted the null as a second
        hour, so the axis went hourly and timeLevels() offered a yearly level of "two bars".'''
        df = pl.DataFrame({'ts': [datetime.datetime(2024, 1, 1), None,
                                  datetime.datetime(2024, 1, 3)], 'v': [1, 2, 3]})
        _with_, _without_ = self.p2s.timep(df, 'ts'), self.p2s.timep(df.drop_nulls('ts'), 'ts')
        self.assertSameRender(_with_, _without_, 'a null timestamp must draw what deleting its row does')
        self.assertEqual(_with_.timeLevels(), _without_.timeLevels(), 'a null timestamp changed the levels on offer')


# ─────────────────────────────────────────────────────────────────────────────
# Zero / negative count magnitudes
# ─────────────────────────────────────────────────────────────────────────────

class TestDegenerateCountValues(_EdgeCaseBase):

    def test_histop_all_zero_counts(self):
        _h_ = self.p2s.histop(pl.DataFrame({'cat': ['a', 'b'], 'v': [0, 0]}), 'cat', count='v')
        svg = self.assertRendersCleanly(_h_)
        self.assertEqual(_dataShapes_(svg, _h_.wxh), [], 'a zero count drew a bar')
        self.assertShowsText(svg, 'a', 'b')

    # A negative count= is drawn as zero, and says so once, naming the component, the
    # count field and the bins (PLANNING.md §5 C-negative-counts, decided 2026-09-29)
    def negativeWarnings(self, fn) -> list:
        for _f_ in logging.getLogger('polars2svg_logger').filters:
            if type(_f_).__name__ == 'OnceFilter': _f_.seen_messages.clear()
        return [r.getMessage() for r in capture_log_warnings(fn) if 'below zero' in r.getMessage()]

    def test_histop_negative_counts(self):
        df   = pl.DataFrame({'cat': ['a', 'b', 'c'], 'v': [-5, 3, -2]})
        _msgs_ = self.negativeWarnings(lambda: self.assertSameRender(
            self.p2s.histop(df, 'cat', count='v'),
            self.p2s.histop(df.with_columns(pl.col('v').clip(lower_bound=0)), 'cat', count='v'),
            'a negative count is not drawn as zero'))
        self.assertEqual(_msgs_, ["Histop: count='v' sums below zero in 2 bins (a, c); a negative count is drawn as zero"])
        # once: the same case again says nothing more
        _again_ = [r for r in capture_log_warnings(lambda: self.p2s.histop(df, 'cat', count='v')._repr_svg_())
                   if 'below zero' in r.getMessage()]
        self.assertEqual(_again_, [])

    def test_negative_segments_of_a_stacked_bar_are_drawn_as_zero(self):
        '''Each segment is clamped on its own.  A negative segment used to be drawn at its
        size (histop) or knock the stack off its baseline (timep), running the bar off its
        scale; a bar of nothing but negatives wrote NaN into the SVG.'''
        _ts_ = [datetime.datetime(2024, 1, d) for d in (1, 2, 3)]
        df   = pl.DataFrame({'ts': _ts_ * 2, 'cat': ['a', 'b', 'c'] * 2, 'g': ['x'] * 3 + ['y'] * 3,
                             'v': [10, 7, -2, -4, 1, -1]})
        _zeroed_ = df.with_columns(pl.col('v').clip(lower_bound=0))
        for _name_, _render_ in (('histop', lambda d: self.p2s.histop(d, 'cat', count='v', color='g')),
                                 ('timep',  lambda d: self.p2s.timep(d, 'ts', count='v', color='g')),
                                 ('timep periodic', lambda d: self.p2s.timep(d, ('ts', self.p2s.PT_dp), count='v', color='g'))):
            with self.subTest(_name_):
                _msgs_ = self.negativeWarnings(lambda: self.assertSameRender(_render_(df), _render_(_zeroed_),
                                                                             'a negative segment is not drawn as zero'))
                self.assertEqual(len(_msgs_), 1, _msgs_)
                self.assertIn('2 bins', _msgs_[0])
                _svg_ = _render_(df).svg
                self.assertNotIn('NaN', _svg_)
                self.assertNotRegex(_svg_, r'<rect [^>]*(width|height)="0(\.0)?"', 'drew a segment of nothing')

    def test_timep_all_zero_counts(self):
        _t_ = self.p2s.timep(pl.DataFrame({'ts': [datetime.datetime(2024, 1, 1),
                                                  datetime.datetime(2024, 1, 2)], 'v': [0, 0]}), 'ts', count='v')
        self.assertEqual(_dataShapes_(self.assertRendersCleanly(_t_), _t_.wxh), [], 'a zero count drew a bar')

    def test_timep_negative_counts(self):
        df = pl.DataFrame({'ts': [datetime.datetime(2024, 1, 1),
                                  datetime.datetime(2024, 1, 2)], 'v': [-3, -7]})
        _msgs_ = self.negativeWarnings(lambda: self.assertBlank(self.p2s.timep(df, 'ts', count='v')))
        self.assertEqual(_msgs_, ["Timep: count='v' sums below zero in 2 bins (2024-01-01 00:00:00, 2024-01-02 00:00:00); "
                                  "a negative count is drawn as zero"])

    def test_counts_at_or_above_zero_do_not_warn(self):
        df = pl.DataFrame({'cat': ['a', 'b'], 'ts': [datetime.datetime(2024, 1, 1), datetime.datetime(2024, 1, 2)], 'v': [0, 3]})
        self.assertEqual(self.negativeWarnings(lambda: (self.p2s.histop(df, 'cat', count='v')._repr_svg_(),
                                                         self.p2s.timep(df, 'ts', count='v')._repr_svg_())), [])


# ─────────────────────────────────────────────────────────────────────────────
# XML-special and unicode characters in data-driven labels
# ─────────────────────────────────────────────────────────────────────────────

class TestSpecialCharacterLabels(_EdgeCaseBase):

    _EDGES_ = pl.DataFrame({'fm': ['A&B', 'C<D', 'E"F'], 'to': ['C<D', 'E"F', 'A&B']})

    def test_chordp_labels_radial_escaped(self):
        '''Regression: raw &/< in node names produced malformed XML.'''
        svg = self.assertRendersCleanly(self.p2s.chordp(df=self._EDGES_, relationships=[('fm', 'to')],
                                                        draw_labels=True, label_style='radial'))
        self.assertIn('A&amp;B', svg)
        self.assertIn('C&lt;D',  svg)

    def test_chordp_labels_circular_escaped(self):
        svg = self.assertRendersCleanly(self.p2s.chordp(df=self._EDGES_, relationships=[('fm', 'to')],
                                                        draw_labels=True, label_style='circular'))
        self.assertIn('A&amp;B', svg)

    def test_chordp_node_labels_map_escaped(self):
        svg = self.assertRendersCleanly(self.p2s.chordp(df=self._EDGES_, relationships=[('fm', 'to')],
                                                        draw_labels=True,
                                                        node_labels={'A&B': 'x<&>y'}))
        self.assertNotIn('>x<&>y<', svg)

    def test_histop_special_char_bins(self):
        df = pl.DataFrame({'cat': ['a&b', 'c<d', '"q"', 'a&b'], 'v': [1, 2, 3, 4]})
        self.assertShowsText(self.assertRendersCleanly(self.p2s.histop(df, 'cat')), 'a&b', 'c<d', '"q"')

    def test_histop_tuple_bin_special_chars(self):
        df = pl.DataFrame({'a': ['x&', 'y<'], 'b': ['1', '2'], 'v': [1, 2]})
        self.assertShowsText(self.assertRendersCleanly(self.p2s.histop(df, ('a', 'b'))), 'x&|1', 'y<|2')

    def test_xyp_gradient_line_id_special_chars(self):
        '''Regression: the per-segment linearGradient id was derived from the
        (untrusted) line-by column via a blocklist that missed < > & ", so a
        line group name with those chars broke out of id="..."/url(#...) and
        produced malformed SVG. The id sanitizer is now an allowlist.'''
        _a_ = pl.DataFrame({'t': list(range(6)), 'v': [2, 3, 4, 3, 5, 1],
                            'grp': ['a<b&"#(']*6})
        _b_ = pl.DataFrame({'t': list(range(6)), 'v': [8, 7, 5, 6, 5, 2],
                            'grp': ['c>d']*6})
        df  = pl.concat([_a_, _b_])
        # LINEOPACITY_FIELD_VARIABLE forces the per-endpoint gradient path.
        svg = self.assertRendersCleanly(self.p2s.xyp(
            df, 't', 'v', color='v', dot_size='v', opacity='v',
            line=('grp', self.p2s.LINEOPACITY_FIELD_VARIABLE), draw_context=False))
        _ids_ = re.findall(r'id="(lines_[^"]*)"', svg)
        self.assertTrue(_ids_)                       # gradient path actually ran
        for _id_ in _ids_:
            self.assertNotRegex(_id_, r'[^A-Za-z0-9_-]')   # only safe id chars

    def test_xyp_categorical_axis_special_chars(self):
        # The axis ends are labelled with the first and last categories, as written
        df = pl.DataFrame({'x': ['a&b', 'c<d', 'e'], 'y': [1.0, 2.0, 3.0]})
        self.assertShowsText(self.assertRendersCleanly(self.p2s.xyp(df, 'x', 'y')), 'a&b', 'e')

    def test_linkp_labels_special_chars(self):
        '''node_labels= is the {name: display} map; the flag that draws them is
        draw_node_labels=.  Passed the map's name, this rendered zero <text>
        elements and asserted nothing about labels for as long as it existed --
        hence the count assertion below.  See tests/test_svg_escaping.py for the
        entity-splitting bug it was meant to be watching.'''
        pos = {'A&B': [0, 0], 'C<D': [1, 1]}
        df  = pl.DataFrame({'fm': ['A&B', 'C<D'], 'to': ['C<D', 'A&B']})
        svg = self.assertRendersCleanly(self.p2s.linkp(df, relationships=[('fm', 'to')],
                                                       pos=pos, draw_node_labels=True))
        self.assertEqual(sorted(_texts_(svg)), ['A&B', 'C<D'])

    def test_spreadlinesp_special_char_nodes(self):
        # Node names are not drawn here and colours are hashed from them, so there is
        # no plain-name render to compare with: the claim is a clean render with data.
        df = pl.DataFrame({'fm': ['A&B', 'A&B', 'C<D'], 'to': ['C<D', 'E>F', 'E>F'],
                           'time': [datetime.datetime(2024, 1, d) for d in (1, 2, 3)]})
        _s_ = self.p2s.spreadlinesp(df, [('fm', 'to')], ego='A&B', time='time')
        self.assertGreater(len(_dataShapes_(self.assertRendersCleanly(_s_), _s_.wxh)), 0, 'nothing drawn')

    def test_unicode_labels_across_components(self):
        _hist_df_ = pl.DataFrame({'cat': ['日本語', 'emoji 🎉', 'ümlaut'], 'v': [1, 2, 3]})
        self.assertShowsText(self.assertRendersCleanly(self.p2s.histop(_hist_df_, 'cat')), '日本語', 'emoji 🎉', 'ümlaut')
        _edge_df_ = pl.DataFrame({'fm': ['日本', 'α&β'], 'to': ['α&β', '日本']})
        self.assertShowsText(self.assertRendersCleanly(self.p2s.chordp(df=_edge_df_, relationships=[('fm', 'to')],
                                                                       draw_labels=True)), '日本', 'α&β')

    # ── SECURITY.md threat-model regression: a raw script payload in row data
    #    must never survive unescaped into any labeled component's SVG output.
    _SCRIPT_PAYLOAD_ = '<script>alert(1)</script>'

    # The payload is on the canvas as TEXT -- so the test reached a label, and a
    # render that dropped the label cannot pass -- and nowhere as markup.
    def assertPayloadIsText(self, component) -> None:
        svg = self.assertRendersCleanly(component)
        self.assertShowsText(svg, self._SCRIPT_PAYLOAD_)
        self.assertNotIn('<script>alert', svg)
        self.assertFalse(any(_tag_(e) == 'script' for e in ET.fromstring(svg).iter()), 'a <script> element')

    def test_script_payload_histop_bin(self):
        df = pl.DataFrame({'cat': [self._SCRIPT_PAYLOAD_, 'b'], 'v': [1, 2]})
        self.assertPayloadIsText(self.p2s.histop(df, 'cat'))

    def test_script_payload_piep_slice_label(self):
        df = pl.DataFrame({'cat': [self._SCRIPT_PAYLOAD_, 'b', 'c'], 'v': [3, 2, 1]})
        self.assertPayloadIsText(self.p2s.piep(df, 'cat', draw_labels=True))

    def test_script_payload_xyp_categorical_axis(self):
        df = pl.DataFrame({'x': [self._SCRIPT_PAYLOAD_, 'b'], 'y': [1.0, 2.0]})
        self.assertPayloadIsText(self.p2s.xyp(df, 'x', 'y'))

    def test_script_payload_chordp_label(self):
        df = pl.DataFrame({'fm': [self._SCRIPT_PAYLOAD_, 'b'], 'to': ['b', self._SCRIPT_PAYLOAD_]})
        self.assertPayloadIsText(self.p2s.chordp(df=df, relationships=[('fm', 'to')], draw_labels=True))

    def test_script_payload_linkp_label(self):
        # node_labels= is the display dict, not the on/off switch: without
        # draw_node_labels= no label was drawn and this asserted nothing.
        pos = {self._SCRIPT_PAYLOAD_: [0, 0], 'b': [1, 1]}
        df  = pl.DataFrame({'fm': [self._SCRIPT_PAYLOAD_, 'b'], 'to': ['b', self._SCRIPT_PAYLOAD_]})
        self.assertPayloadIsText(self.p2s.linkp(df, relationships=[('fm', 'to')], pos=pos, draw_node_labels=True))

    def test_script_payload_linkp_link_label(self):
        pos = {'a': [0, 0], 'b': [1, 1]}
        df  = pl.DataFrame({'fm': ['a'], 'to': ['b'], 'dsc': [self._SCRIPT_PAYLOAD_]})
        for _shape_ in ('line', 'curve'):
            with self.subTest(link_shape=_shape_):
                self.assertPayloadIsText(self.p2s.linkp(df, relationships=[('fm', 'to', 'dsc')], pos=pos,
                                                        draw_link_labels=True, link_shape=_shape_, wxh=(400, 400)))

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
        svg = self.assertRendersCleanly(self.p2s.spreadlinesp(df, [('fm', 'to')], ego='a',
                                                              time='time', draw_context=True))
        # No unescaped structural markup from the timestamp data survives.
        self.assertNotIn('<a>', svg)

    def test_spreadlinesp_special_char_anno(self):
        df = pl.DataFrame({'fm': ['a', 'a', 'b', 'b'], 'to': ['b', 'c', 'c', 'a'],
                           'time': ['2020', '2020', '2021', '2021']})
        svg = self.assertRendersCleanly(self.p2s.spreadlinesp(df, [('fm', 'to')], ego='a',
                                                              time='time',
                                                              anno={'2021': self._SCRIPT_PAYLOAD_}))
        self.assertNotIn('<script>alert', svg)


# ─────────────────────────────────────────────────────────────────────────────
# Graph-shape oddities: self-loops, parallel edges, nodes missing from pos/ego
# ─────────────────────────────────────────────────────────────────────────────

class TestGraphShapeOddities(_EdgeCaseBase):

    def test_chordp_self_loops(self):
        # chordp leaves a self-loop out: the render is the graph without them
        df = pl.DataFrame({'fm': ['a', 'b', 'a'], 'to': ['a', 'b', 'b']})
        self.assertSameRender(self.p2s.chordp(df=df, relationships=[('fm', 'to')]),
                              self.p2s.chordp(df=df.filter(pl.col('fm') != pl.col('to')), relationships=[('fm', 'to')]),
                              'self-loops changed the chord diagram')

    def test_chordp_parallel_edges(self):
        # Parallel edges are one edge with their multiplicity as its count
        df = pl.DataFrame({'fm': ['a', 'a', 'a', 'b'], 'to': ['b', 'b', 'b', 'a']})
        _agg_ = df.group_by(['fm', 'to'], maintain_order=True).len('n')
        self.assertSameRender(self.p2s.chordp(df=df, relationships=[('fm', 'to')]),
                              self.p2s.chordp(df=_agg_, relationships=[('fm', 'to')], count='n'),
                              'parallel edges must draw what one edge with count=multiplicity does')

    def test_linkp_self_loops(self):
        _l_ = self.p2s.linkp(pl.DataFrame({'fm': ['a', 'b', 'a'], 'to': ['a', 'b', 'b']}),
                             relationships=[('fm', 'to')], pos={'a': [0, 0], 'b': [1, 1]})
        root = ET.fromstring(self.assertRendersCleanly(_l_))
        self.assertEqual(sum(_tag_(e) == 'circle' for e in root.iter()), 2, 'expected the two nodes')
        _links_ = [e for e in root.iter() if _tag_(e) == 'line' and (e.get('x1'), e.get('y1')) != (e.get('x2'), e.get('y2'))]
        self.assertEqual(len(_links_), 1, 'expected the one a->b link')

    def test_linkp_node_missing_from_pos(self):
        # A node pos= does not place is placed, not dropped
        _l_ = self.p2s.linkp(pl.DataFrame({'fm': ['a', 'b'], 'to': ['b', 'c']}),
                             relationships=[('fm', 'to')], pos={'a': [0, 0], 'b': [1, 1]})
        root = ET.fromstring(self.assertRendersCleanly(_l_))
        self.assertEqual((sum(_tag_(e) == 'circle' for e in root.iter()), sum(_tag_(e) == 'line' for e in root.iter())),
                         (3, 2), 'expected three nodes and two links')
        self.assertIn('c', _l_.pos)

    def test_spreadlinesp_ego_not_in_data(self):
        # An ego that is not in the data draws what an empty frame does
        df = pl.DataFrame({'fm': ['a'], 'to': ['b'], 'time': [datetime.datetime(2024, 1, 1)]})
        _s_ = self.p2s.spreadlinesp(df, [('fm', 'to')], ego='zzz', time='time')
        self.assertBlank(_s_)
        self.assertSameRender(_s_, self.p2s.spreadlinesp(df.clear(), [('fm', 'to')], ego='zzz', time='time'),
                              'an ego that is not in the data must draw what an empty frame does')


if __name__ == '__main__':
    unittest.main()
