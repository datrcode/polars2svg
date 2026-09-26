"""Settings-panel rows that change a generic view's RENDER (interactive_render_rows.py).

Three parts.  The row logic is plain Python and runs without the `interactive` extra: what
rows a template gets, what they start on, when they are live, and which overrides a set
of values produces.  The view half (xypi) needs panel: a row change re-renders, and a
refused one snaps back.  And every row set -- whatever the component -- owes its view the
same contract, checked once over one registry of templates (_CASES_ below).

The rule under test throughout: **an untouched panel changes nothing** -- overrides() is
{} until a row moves off the value read from the template.
"""
import asyncio
import datetime
import logging
import unittest
from contextlib import contextmanager

import polars as pl

from polars2svg import Polars2SVG
from polars2svg.interactive_render_rows import (AS_BUILT_MNEMONIC, RENDER_ROW_KEYS,
                                                RESERVED_ROW_KEYS, ChordpRenderRows,
                                                HistopRenderRows, PiepRenderRows, TimepRenderRows,
                                                XYpRenderRows)
from svg_test_utils import normalize_svg

try:
    import panel  # noqa: F401
    _PANEL_AVAILABLE_ = True
except ImportError:
    _PANEL_AVAILABLE_ = False


def _df():
    return pl.DataFrame({
        'a':     [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        'b':     [3.0, 4.0, 5.0, 1.0, 2.0, 9.0, 6.0, 7.0],
        'pet':   ['cat', 'dog', 'dog', 'ant', 'cat', 'dog', 'ant', 'dog'],
        'bytes': [10, 20, 30, 40, 50, 60, 70, 80],
    })


def _settle():
    """Let the param watcher's coroutine run."""
    async def _go(): await asyncio.sleep(0.05)
    asyncio.run(_go())


class _RowsCase(unittest.TestCase):
    def setUp(self):
        self.p2s = Polars2SVG()
        self.df  = _df()

    def _rows(self, **kw):
        _kw_ = dict(dot_size=3.0)
        _kw_.update(kw)
        _t_ = self.p2s.xyp(self.df, _kw_.pop('x', 'a'), _kw_.pop('y', 'b'), **_kw_)
        return _t_, XYpRenderRows(_t_)

    def _row(self, rows, kind):
        return next(_r_ for _r_ in rows.rows if _r_.kind == kind)

    def _render(self, template, rows, **settings):
        _s_ = dict(rows.initial_settings(), **settings)
        return template.render_with(self.df, **rows.overrides(_s_))


# ── the rows a template gets ─────────────────────────────────────────────────

class TestTheRows(_RowsCase):
    def test_what_the_distributions_measure_is_in_the_row_label(self):
        self.assertEqual(self._row(self._rows()[1], 'distributions').label, 'distributions (rows)')
        _, _rows_ = self._rows(x_distributions='bytes')
        self.assertEqual(self._row(_rows_, 'distributions').label, 'distributions (bytes)')

    def test_a_value_the_list_does_not_carry_is_offered_as_built(self):
        _, _rows_ = self._rows(opacity=0.37)
        self.assertEqual(self._row(_rows_, 'opacity').initial, '37')
        self.assertIn([AS_BUILT_MNEMONIC, '37'], self._row(_rows_, 'opacity').items)
        _, _rows_ = self._rows(x='pet', x_order=['dog', 'cat', 'ant'])
        self.assertEqual(self._row(_rows_, 'x_order').initial, 'as built')


class TestGating(_RowsCase):
    def _enabled(self, rows, **settings):
        _s_ = dict(rows.initial_settings(), **settings)
        return {_r_.kind: _r_.enabled(_s_) for _r_ in rows.rows}

    def test_placement_and_bins_follow_the_distributions_row(self):
        _, _rows_ = self._rows()
        self.assertFalse(self._enabled(_rows_)['placement'])
        self.assertFalse(self._enabled(_rows_)['bins'])
        self.assertTrue(self._enabled(_rows_, distributions='x')['placement'])
        self.assertTrue(self._enabled(_rows_, distributions='x')['bins'])

    def test_aspect_needs_two_numeric_axes(self):
        self.assertTrue(self._enabled(self._rows()[1])['aspect'])
        self.assertFalse(self._enabled(self._rows(x='pet')[1])['aspect'])

    def test_an_order_needs_a_categorical_axis(self):
        _e_ = self._enabled(self._rows(x='pet')[1])
        self.assertTrue(_e_['x_order'])
        self.assertFalse(_e_['y_order'])

    def test_the_legend_needs_a_color(self):
        self.assertFalse(self._enabled(self._rows()[1])['legend'])
        self.assertTrue(self._enabled(self._rows(color='pet')[1])['legend'])

    def test_color_scale_needs_a_magnitude_or_stretched_color(self):
        self.assertFalse(self._enabled(self._rows(color='pet')[1])['color_scale'])
        self.assertTrue(self._enabled(self._rows(color=self.p2s.CROW_STRETCHEDp)[1])['color_scale'])

    def test_opacity_is_off_when_it_is_data_driven(self):
        self.assertFalse(self._enabled(self._rows(opacity='bytes')[1])['opacity'])

    def test_bins_are_off_on_a_periodic_time_axis(self):
        _df_ = pl.DataFrame({'ts': pl.datetime_range(pl.datetime(2026, 1, 1), pl.datetime(2026, 1, 2), '1h', eager=True)})
        _df_ = _df_.with_columns(pl.lit(1.0).alias('v'))
        _t_  = self.p2s.xyp(_df_, self.p2s.tField('ts', self.p2s.PT_DoWp), 'v', dot_size=3.0)
        _rows_ = XYpRenderRows(_t_)
        self.assertFalse(self._enabled(_rows_, distributions='x')['bins'])
        self.assertTrue(self._enabled(_rows_, distributions='x+y')['bins'], 'y is still binnable')


# ── the overrides ────────────────────────────────────────────────────────────

class TestOverrides(_RowsCase):
    def test_moving_a_row_back_overrides_nothing_again(self):
        _, _rows_ = self._rows()
        _s_ = dict(_rows_.initial_settings(), aspect='equal')
        self.assertEqual(_rows_.overrides(_s_), {'aspect': 'equal'})
        _s_['aspect'] = 'none'
        self.assertEqual(_rows_.overrides(_s_), {})

    def test_a_row_only_sets_its_own_parameter(self):
        _, _rows_ = self._rows(x='pet', color='pet')
        _s_ = dict(_rows_.initial_settings(), legend='bottom', x_order='by count', opacity='50')
        self.assertEqual(_rows_.overrides(_s_), {'legend': 'bottom', 'x_order': 'count', 'opacity': 0.5})

    def test_distributions_default_to_row_counts(self):
        _t_, _rows_ = self._rows()
        _xy_ = self._render(_t_, _rows_, distributions='x+y')
        self.assertIsNotNone(_xy_.df_x_distribution)
        self.assertIsNotNone(_xy_.df_y_distribution)

    def test_a_new_axis_measures_what_the_template_measured(self):
        _, _rows_ = self._rows(x_distributions='bytes')
        _ov_ = _rows_.overrides(dict(_rows_.initial_settings(), distributions='x+y'))
        self.assertEqual(_ov_['y_distributions'], ['bytes'])

    def test_off_removes_both_axes(self):
        _, _rows_ = self._rows(x_distributions='bytes')
        _ov_ = _rows_.overrides(dict(_rows_.initial_settings(), distributions='off'))
        self.assertEqual(_ov_, {'x_distributions': None, 'y_distributions': None})

    def test_placement_is_applied_to_every_distributed_axis(self):
        _t_, _rows_ = self._rows()
        _xy_ = self._render(_t_, _rows_, distributions='x+y', placement='outside')
        for _clean_ in (_xy_.x_distributions_clean, _xy_.y_distributions_clean):
            self.assertIn(self.p2s.DISTRIBUTION_OUTSIDEp, _clean_['enums'])

    def test_as_built_placement_and_bins_survive_a_change_elsewhere_in_the_group(self):
        _, _rows_ = self._rows(x_distributions=('bytes', self.p2s.DISTRIBUTION_OUTSIDEp, 16))
        _ov_ = _rows_.overrides(dict(_rows_.initial_settings(), distributions='x+y'))
        self.assertEqual(_ov_['x_distributions'], ['bytes', self.p2s.DISTRIBUTION_OUTSIDEp, 16])

    def test_a_bin_multiple_is_a_multiple_of_the_auto_count(self):
        _t_, _rows_ = self._rows()
        _auto_ = self._render(_t_, _rows_, distributions='x').x_distributions_clean['bins'][0]
        for _label_, _m_ in (('auto x2', 2.0), ('auto /2', 0.5), ('auto /4', 0.25)):
            with self.subTest(bins=_label_):
                _xy_ = self._render(_t_, _rows_, distributions='x', bins=_label_)
                self.assertEqual(_xy_.x_distributions_clean['bins'][0], max(1, round(_auto_ * _m_)))

    def test_the_color_scale_swaps_magnitude_and_stretched(self):
        _, _rows_ = self._rows(color=self.p2s.CROW_STRETCHEDp)
        self.assertEqual(_rows_.overrides(dict(_rows_.initial_settings(), color_scale='magnitude')),
                         {'color': self.p2s.CROW_MAGNITUDEp})


# ── the view ─────────────────────────────────────────────────────────────────

@unittest.skipUnless(_PANEL_AVAILABLE_, 'panel not installed')
class TestXYPIRenderRows(_RowsCase):
    def _view(self, **kw):
        _t_, _ = self._rows(**kw)
        return _t_, self.p2s.xypi(_t_)

    def test_the_panel_carries_the_rows_after_the_base_two(self):
        _, _v_ = self._view(x='pet')
        _kinds_ = [_r_[1] for _r_ in _v_.config_panel_rows]
        self.assertEqual(_kinds_[:2], ['select_shape', 'tooltip'])
        self.assertEqual(_kinds_[2:], ['distributions', 'placement', 'bins', 'aspect', 'opacity',
                                       'color_scale', 'legend', 'x_order', 'y_order'])

    def test_a_row_change_re_renders(self):
        _, _v_ = self._view(x='pet')
        _before_ = _v_.mod_inner
        _v_.render_settings = dict(_v_.render_settings, x_order='by count')
        _settle()
        self.assertEqual(_v_._overrides_, {'x_order': 'count'})
        self.assertNotEqual(_v_.mod_inner, _before_)

    def test_the_gating_follows_the_settings(self):
        _, _v_ = self._view()
        _on_ = lambda: {_r_[1]: _r_[3] for _r_ in _v_.config_panel_rows}  # noqa: E731
        self.assertFalse(_on_()['placement'])
        _v_.render_settings = dict(_v_.render_settings, distributions='x')
        _settle()
        self.assertTrue(_on_()['placement'])

    def test_a_refused_setting_snaps_back_and_leaves_the_view(self):
        _, _v_ = self._view(x='pet')          # a categorical x: aspect= raises
        _before_ = (_v_.mod_inner, dict(_v_.render_settings))
        _v_.template.p2s.logger.disabled = True
        try:
            _v_.render_settings = dict(_v_.render_settings, aspect='equal')
            _settle()
        finally:
            _v_.template.p2s.logger.disabled = False
        self.assertEqual((_v_.mod_inner, dict(_v_.render_settings)), _before_)
        self.assertEqual(_v_._overrides_, {})

    def test_every_stack_level_renders_with_the_overrides(self):
        _, _v_ = self._view(x='pet')
        _v_.render_settings = dict(_v_.render_settings, x_order='reverse')
        _settle()
        _sub_ = self.df.filter(pl.col('pet') != 'ant')
        asyncio.run(_v_.display(_sub_, [self.df, _sub_], 1))
        self.assertEqual(_v_._plot_.x_order, 'reverse')

    def test_every_generic_view_has_render_rows(self):
        from polars2svg.interactive_controller import _INTERACTIVEP_CLASSES_
        self.assertEqual({_k_: _c_._render_rows_cls_ for _k_, _c_ in _INTERACTIVEP_CLASSES_.items()},
                         {'xypi': XYpRenderRows, 'histopi': HistopRenderRows, 'timepi': TimepRenderRows,
                          'chordpi': ChordpRenderRows, 'piepi': PiepRenderRows})


@unittest.skipUnless(_PANEL_AVAILABLE_, 'panel not installed')
class TestTheBrowserHalf(unittest.TestCase):
    def test_a_render_row_writes_the_render_settings_dict(self):
        from view_js_utils import component_js, component_script
        _p2s_ = Polars2SVG()
        _v_   = _p2s_.xypi(_p2s_.xyp(_df(), 'pet', 'b'))
        _set_ = component_script(_v_, 'menuSetValue')
        self.assertIn('model.render_settings = _s_;', _set_)
        self.assertIn("model.on(_pp_, function() { syncRenderHeaders(); panelRender(); })", component_js(_v_))

    def test_the_panel_fragment_reads_through_getValue(self):
        from view_js_utils import component_js
        _p2s_ = Polars2SVG()
        _js_  = component_js(_p2s_.xypi(_p2s_.xyp(_df(), 'pet', 'b')))
        self.assertIn('var _current_ = getValue(state.menu_kind);', _js_)


# ── histop ───────────────────────────────────────────────────────────────────

class TestHistopRows(unittest.TestCase):
    """histopi's rows.  What the generic suite below cannot see: which rows are live
    depends on the others, through the boxplot styles, and the order row is a group."""

    def setUp(self):
        self.p2s = Polars2SVG()
        self.df  = _df().with_columns(pl.Series('grp', ['x', 'y', 'x', 'y', 'y', 'x', 'x', 'y']))

    def _rows(self, **kw):
        return HistopRenderRows(self.p2s.histop(self.df, 'pet', **kw))

    def _live(self, rows, **settings):
        _s_ = dict(rows.initial_settings(), **settings)
        return {_r_.kind: _r_.enabled(_s_) for _r_ in rows.rows}

    def test_the_panel_rows_in_order(self):
        _rows_ = self._rows()
        self.assertEqual([(_r_.mnemonic, _r_.kind) for _r_ in _rows_.rows],
                         [('t', 'style'), ('m', 'count'), ('r', 'order'), ('l', 'draw_labels'),
                          ('d', 'distribution'), ('g', 'legend'), ('c', 'color_scale')])

    def test_boxplots_are_offered_only_on_a_numeric_count(self):
        """histop draws a boxplot of a numeric count field, and silently draws bars for
        anything else -- rows, or a text column counted as distinct values."""
        self.assertFalse(self._live(self._rows())['style'])
        self.assertFalse(self._live(self._rows(count='grp'))['style'])
        self.assertTrue(self._live(self._rows(count='bytes'))['style'])

    def test_style_and_count_hold_each_other(self):
        _rows_ = self._rows(count='bytes')
        self.assertFalse(self._live(_rows_, style='boxplot')['count'], 'a boxplot needs its field')
        self.assertFalse(self._live(_rows_, count='rows')['style'], 'no field, no boxplot')

    def test_a_boxplot_greys_out_what_it_does_not_draw(self):
        _live_ = self._live(self._rows(count='bytes', color='grp', legend=True), style='boxplot+swarm')
        self.assertEqual({_k_ for _k_, _on_ in _live_.items() if not _on_},
                         {'count', 'distribution', 'legend', 'color_scale'})

    def test_stacked_is_offered_only_as_built(self):
        """STACKEDBARp draws what BARCHARTp draws, so offering both would be a row whose
        second value changes nothing."""
        _style_ = lambda rows: next(_r_ for _r_ in rows.rows if _r_.kind == 'style')  # noqa: E731
        self.assertNotIn('stacked', [_l_ for _, _l_ in _style_(self._rows()).items])
        _built_ = _style_(self._rows(style=self.p2s.STACKEDBARp))
        self.assertEqual(_built_.initial, 'stacked')
        self.assertIn([AS_BUILT_MNEMONIC, 'stacked'], _built_.items)

    def test_the_count_row_offers_rows_and_the_views_own_field(self):
        _count_ = next(_r_ for _r_ in self._rows(count='bytes').rows if _r_.kind == 'count')
        self.assertEqual(_count_.items, [['r', 'rows'], ['f', 'bytes']])
        self.assertEqual(_count_.values, {'rows': self.p2s.ROW_COUNTp, 'bytes': 'bytes'})

    def test_the_order_row_sets_only_the_parameters_that_change(self):
        _rows_ = self._rows()                       # ROW_COUNTp, descending=True
        _ov_ = lambda **s: _rows_.overrides(dict(_rows_.initial_settings(), **s))  # noqa: E731
        self.assertEqual(_ov_(order='largest first'), {})
        self.assertEqual(_ov_(order='smallest first'), {'descending': False})
        self.assertEqual(_ov_(order='sorted'), {'order': self.p2s.LABELp, 'descending': False})
        self.assertEqual(_ov_(order='reverse'), {'order': self.p2s.LABELp})

    def test_an_order_the_row_does_not_carry_is_as_built(self):
        _order_ = next(_r_ for _r_ in self._rows(order='bytes', descending=False).rows if _r_.kind == 'order')
        self.assertEqual(_order_.initial, 'by bytes, ascending')
        self.assertIn([AS_BUILT_MNEMONIC, 'by bytes, ascending'], _order_.items)

    def test_the_strip_row_follows_histops_own_room_rule(self):
        """distributionStripFits() is histop's rule: below 48 px there is no strip."""
        _small_ = self.p2s.histop(self.df, 'pet', wxh=(40, 40))
        self.assertFalse(_small_.distributionStripFits())
        self.assertFalse(self._live(HistopRenderRows(_small_))['distribution'])
        self.assertTrue(self.p2s.histop(self.df, 'pet').distributionStripFits())

    def test_numeric_count_field_takes_a_candidate_count(self):
        _t_ = self.p2s.histop(self.df, 'pet')
        self.assertIsNone(_t_.numericCountField())
        self.assertEqual(_t_.numericCountField('bytes'), 'bytes')
        self.assertEqual(_t_.numericCountField(('grp', 'bytes')), 'bytes')
        self.assertIsNone(_t_.numericCountField('grp'))


# ── piep ─────────────────────────────────────────────────────────────────────

class TestPiepRows(unittest.TestCase):
    """piepi's rows, and the piep rules behind their gates."""

    def setUp(self):
        self.p2s = Polars2SVG()
        self._logger_was_ = self.p2s.logger.disabled
        self.p2s.logger.disabled = True     # the text-field magnitude case warns, by design
        self.df  = _df().with_columns(pl.Series('grp', ['x', 'y', 'x', 'y', 'y', 'x', 'x', 'y']))

    def tearDown(self):
        self.p2s.logger.disabled = self._logger_was_

    def _rows(self, **kw):
        return PiepRenderRows(self.p2s.piep(self.df, 'pet', **kw))

    def _live(self, rows, **settings):
        _s_ = dict(rows.initial_settings(), **settings)
        return {_r_.kind: _r_.enabled(_s_) for _r_ in rows.rows}

    def test_the_panel_rows_in_order(self):
        self.assertEqual([(_r_.mnemonic, _r_.kind, _r_.label) for _r_ in self._rows().rows],
                         [('t', 'style', 'style'), ('m', 'count', 'count'), ('r', 'order', 'slice order'),
                          ('l', 'draw_labels', 'labels'), ('g', 'legend', 'legend'),
                          ('c', 'color_scale', 'color scale')])

    def test_a_waffle_greys_out_the_labels(self):
        _rows_ = self._rows()
        self.assertTrue(self._live(_rows_, style='donut')['draw_labels'])
        self.assertFalse(self._live(_rows_, style='waffle')['draw_labels'])

    def test_a_waffle_really_draws_no_labels(self):
        """styleDrawsLabels() states the renderer's rule, so pin it against the renderer:
        draw_labels= changes a donut and leaves a waffle byte-identical."""
        _svg_ = lambda style, on: normalize_svg(  # noqa: E731
            self.p2s.piep(self.df, 'pet', style=style, draw_labels=on)._repr_svg_())
        self.assertEqual(_svg_(self.p2s.WAFFLEp, True), _svg_(self.p2s.WAFFLEp, False))
        self.assertNotEqual(_svg_(self.p2s.DONUTp, True), _svg_(self.p2s.DONUTp, False))
        _t_ = self.p2s.piep(self.df, 'pet')
        self.assertFalse(_t_.styleDrawsLabels(self.p2s.WAFFLEp))
        self.assertTrue(_t_.styleDrawsLabels())

    def test_the_legend_needs_a_colour_that_draws_one(self):
        """Only a categorical or spectrum colour has a legend: not none, and not one fixed
        colour or a list of them."""
        for _color_, _live_ in ((None, False), ('#aa3300', False), (['#aa3300', '#0033aa'], False),
                                ('grp', True), (self.p2s.CROW_MAGNITUDEp, True)):
            with self.subTest(color=_color_):
                self.assertEqual(self._live(self._rows(color=_color_))['legend'], _live_)

    def test_the_colour_scale_needs_a_spectrum(self):
        """A magnitude enum on a text field is categorical to piep, so it has no scale."""
        self.assertTrue(self._live(self._rows(color=('bytes', self.p2s.CMAGNITUDE_SUMp)))['color_scale'])
        _fallback_ = self._rows(color=('grp', self.p2s.CMAGNITUDE_SUMp))
        self.assertFalse(self._live(_fallback_)['color_scale'])
        self.assertTrue(self._live(_fallback_)['legend'], 'it is categorical, so it has a legend')

    def test_the_slice_order_is_descending(self):
        _rows_ = self._rows()
        self.assertEqual(_rows_.overrides(dict(_rows_.initial_settings(), order='smallest first')),
                         {'descending': False})
        _back_ = self._rows(descending=False)
        self.assertEqual(_back_.overrides(dict(_back_.initial_settings(), order='largest first')),
                         {'descending': True})

    def test_style_is_pie_donut_or_waffle(self):
        _style_ = next(_r_ for _r_ in self._rows(style=self.p2s.DONUTp).rows if _r_.kind == 'style')
        self.assertEqual(_style_.items, [['p', 'pie'], ['d', 'donut'], ['w', 'waffle']])
        self.assertEqual(_style_.initial, 'donut')


# ── chordp ───────────────────────────────────────────────────────────────────

class TestChordpRows(unittest.TestCase):
    """chordpi's rows: which values chordp draws differently, and the gates between rows."""

    _POS_ = {'a': (0.0, 0.0), 'b': (1.0, 0.0), 'c': (1.0, 1.0), 'd': (0.0, 1.0), 'e': (0.5, 0.5)}

    def setUp(self):
        self.p2s = Polars2SVG()
        self.df  = pl.DataFrame({'fm': list('aabbccddee'), 'to': list('bcdeaebcad'),
                                 'w': [5, 1, 9, 2, 7, 3, 8, 4, 6, 10], 'grp': list('xyxyxyxyxy')})

    def _t(self, **kw):
        return self.p2s.chordp(self.df, [('fm', 'to')], **kw)

    def _live(self, rows, **settings):
        _s_ = dict(rows.initial_settings(), **settings)
        return {_r_.kind: _r_.enabled(_s_) for _r_ in rows.rows}

    def _row(self, rows, kind):
        return next(_r_ for _r_ in rows.rows if _r_.kind == kind)

    def test_the_panel_rows_in_order(self):
        self.assertEqual([(_r_.mnemonic, _r_.kind) for _r_ in ChordpRenderRows(self._t()).rows],
                         [('h', 'link_shape'), ('u', 'bundle_strength'), ('n', 'node_size'),
                          ('z', 'link_size'), ('f', 'node_opacity'), ('o', 'link_opacity'),
                          ('l', 'draw_labels'), ('v', 'label_style'), ('m', 'count'),
                          ('g', 'legend'), ('c', 'color_scale')])

    def test_node_size_is_fixed_or_vary_because_that_is_all_chordp_draws(self):
        """Every node_size but 'vary' draws the same arcs -- so the row offers two values,
        not linkpi's six."""
        _svg_ = lambda size: normalize_svg(self._t(count='w', node_size=size)._repr_svg_())  # noqa: E731
        self.assertEqual({_svg_(_s_) for _s_ in ('nil', 'small', 'medium', 'large')}, {_svg_('medium')})
        self.assertNotEqual(_svg_('vary'), _svg_('medium'))
        self.assertEqual(self._row(ChordpRenderRows(self._t()), 'node_size').items, [['f', 'fixed'], ['v', 'vary']])

    def test_link_size_offers_linkpis_names_and_none_hides_the_links(self):
        _row_ = self._row(ChordpRenderRows(self._t()), 'link_size')
        self.assertEqual([_l_ for _, _l_ in _row_.items], ['none', 'nil', 'small', 'medium', 'large', 'vary'])
        self.assertIsNone(_row_.values['none'])

    def test_count_is_live_only_while_a_size_varies(self):
        """A re-render keeps the node order, so count= only ever reaches a 'vary' size."""
        _rows_ = ChordpRenderRows(self._t(count='w'))
        self.assertFalse(self._live(_rows_)['count'])
        self.assertTrue(self._live(_rows_, node_size='vary')['count'])
        self.assertTrue(self._live(_rows_, link_size='vary')['count'])
        self.assertFalse(self._live(ChordpRenderRows(self._t()), node_size='vary')['count'], 'rows: nothing to switch')

    def test_bundle_strength_and_label_style_follow_their_rows(self):
        _rows_ = ChordpRenderRows(self._t())
        self.assertFalse(self._live(_rows_)['bundle_strength'])
        self.assertTrue(self._live(_rows_, link_shape='bundled')['bundle_strength'])
        self.assertFalse(self._live(_rows_)['label_style'])
        self.assertTrue(self._live(_rows_, draw_labels='on')['label_style'])

    def test_the_legend_can_describe_the_node_colour(self):
        """chordp's legend explains the link colour when that is data-driven, else the node
        colour -- so a view coloured only by node has a legend to show."""
        for _kw_, _live_ in ((dict(), False), (dict(color='#aa3300', node_color='#0033aa'), False),
                             (dict(node_color=self.p2s.COLOR_BY_NODE_NAME), True), (dict(color='grp'), True)):
            with self.subTest(template=_kw_):
                self.assertEqual(self._t(**_kw_).colorIsLegendable(), _live_)
                self.assertEqual(self._live(ChordpRenderRows(self._t(**_kw_)))['legend'], _live_)

    def test_a_node_colour_scale_row_only_when_node_color_has_one(self):
        self.assertNotIn('node_color_scale', [_r_.kind for _r_ in ChordpRenderRows(self._t()).rows])
        _rows_ = ChordpRenderRows(self._t(node_color=('w', self.p2s.CMAGNITUDE_SUMp)))
        _row_  = self._row(_rows_, 'node_color_scale')
        self.assertEqual((_row_.mnemonic, _row_.param), ('C', 'node_color'))
        self.assertEqual(_rows_.overrides(dict(_rows_.initial_settings(), node_color_scale='stretched')),
                         {'node_color': ('w', self.p2s.CSTRETCHED_SUMp)})

    def test_pos_survives_an_override(self):
        """chordpi re-renders through render_with(), whose template carries pos=; a row
        change must not lose the positions the order came from."""
        _t_    = self._t(pos=self._POS_)
        _rows_ = ChordpRenderRows(_t_)
        _re_   = _t_.render_with(_t_.df_orig, **_rows_.overrides(dict(_rows_.initial_settings(), link_shape='line')))
        self.assertEqual(_re_.pos, self._POS_)
        self.assertEqual(_re_.order, _t_.order)


# ── timep ────────────────────────────────────────────────────────────────────

class TestTimepRows(unittest.TestCase):
    """timepi's rows.  The granularity row is the one with something to say: what it
    offers is exactly Timep.timeLevels(), and 'auto' is the template resolving per frame."""

    def setUp(self):
        self.p2s = Polars2SVG()
        self.df  = pl.DataFrame({'ts': pl.datetime_range(pl.datetime(2026, 1, 1), pl.datetime(2026, 1, 3, 23),
                                                         '1h', eager=True)})

    def _granularity(self, rows):
        return rows.rows[0]

    def test_the_panel_rows_in_order(self):
        self.assertEqual([(_r_.mnemonic, _r_.kind) for _r_ in TimepRenderRows(self.p2s.timep(self.df, 'ts')).rows],
                         [('q', 'granularity'), ('t', 'style'), ('m', 'count'), ('g', 'legend'),
                          ('c', 'color_scale')])

    def test_granularity_offers_auto_and_exactly_the_levels_timep_lists(self):
        _t_   = self.p2s.timep(self.df, 'ts')
        _row_ = self._granularity(TimepRenderRows(_t_))
        self.assertEqual([_l_ for _, _l_ in _row_.items], ['auto'] + [_t_.timeLevelName(_e_) for _e_ in _t_.timeLevels()])
        self.assertEqual(_row_.initial, 'auto')

    def test_auto_sets_nothing_and_a_level_sets_a_t_field(self):
        _rows_ = TimepRenderRows(self.p2s.timep(self.df, 'ts'))
        _ov_ = lambda label: _rows_.overrides(dict(_rows_.initial_settings(), granularity=label))  # noqa: E731
        self.assertEqual(_ov_('auto'), {})
        self.assertEqual(_ov_('day of week'), {'time': self.p2s.tField('ts', self.p2s.PT_DoWp)})

    def test_a_view_built_on_a_level_starts_there_and_auto_hands_back_to_timep(self):
        _rows_ = TimepRenderRows(self.p2s.timep(self.df, self.p2s.tField('ts', self.p2s.PT_Hp)))
        self.assertEqual(self._granularity(_rows_).initial, 'hour')
        self.assertEqual(_rows_.overrides(dict(_rows_.initial_settings(), granularity='auto')), {'time': 'ts'})

    def test_a_level_the_list_does_not_carry_is_offered_as_built(self):
        """Hourly data resolves no finer than the hour, so 'by minute' is not a level the
        row offers -- but a view built on it shows it, as built."""
        _row_ = self._granularity(TimepRenderRows(self.p2s.timep(self.df, self.p2s.tField('ts', self.p2s.LT_Y_m_d_H_Mp))))
        self.assertEqual(_row_.initial, 'by minute')
        self.assertIn([AS_BUILT_MNEMONIC, 'by minute'], _row_.items)

    def test_the_row_is_greyed_when_there_is_nothing_but_auto(self):
        _rows_ = TimepRenderRows(self.p2s.timep(pl.DataFrame({'ts': [datetime.datetime(2026, 1, 1)] * 3}), 'ts'))
        self.assertFalse(self._granularity(_rows_).enabled(_rows_.initial_settings()))

    def test_a_level_keeps_its_key_whatever_else_is_offered(self):
        """Keys are per level, not per position: 'hour' is H on every timepi."""
        _keys_ = lambda t: {_l_: _m_ for _m_, _l_ in self._granularity(TimepRenderRows(t)).items}  # noqa: E731
        _narrow_ = _keys_(self.p2s.timep(self.df, 'ts', wxh=(128, 256)))
        _wide_   = _keys_(self.p2s.timep(self.df, 'ts'))
        self.assertEqual(_narrow_['hour'], 'H')
        self.assertEqual(_wide_['hour'], 'H')
        self.assertNotIn('hourly', _narrow_)                   # 72 bars do not fit 128 px
        self.assertEqual(_wide_['hourly'], 'h')

    def test_the_axis_and_the_picker_name_a_level_alike(self):
        """One table: the label under the axis is what the picker called the level."""
        _t_ = self.p2s.timep(self.df, self.p2s.tField('ts', self.p2s.PT_m_dp))
        self.assertEqual(_t_.timeLevelName(self.p2s.PT_m_dp), 'month / day')
        self.assertIn('>month / day<', _t_._repr_svg_())


@unittest.skipUnless(_PANEL_AVAILABLE_, 'panel not installed')
class TestTimePIRenderRows(unittest.TestCase):
    def setUp(self):
        self.p2s = Polars2SVG()
        self.df  = pl.DataFrame({'ts': pl.datetime_range(pl.datetime(2026, 1, 1), pl.datetime(2026, 1, 3, 23),
                                                         '1h', eager=True)}).with_columns(bytes=pl.int_range(pl.len()))

    def _view_on_(self, label):
        _v_ = self.p2s.timepi(self.p2s.timep(self.df, 'ts', count='bytes'))
        _v_.render_settings = dict(_v_.render_settings, granularity=label)
        _settle()
        return _v_

    def test_an_explicit_level_applies_at_every_stack_level(self):
        _v_   = self._view_on_('day of week')
        _sub_ = self.df.filter(pl.col('bytes') < 30)
        asyncio.run(_v_.display(_sub_, [self.df, _sub_], 1))
        self.assertEqual(_v_._plot_.time, self.p2s.tField('ts', self.p2s.PT_DoWp))

    def test_the_tooltip_keeps_the_time_field_on_a_level(self):
        """The tooltip names the columns the marks encode; a t-field's own value
        ('ts|DoWp') is no column, and used to drop the time field out of it."""
        _v_ = self._view_on_('hour')
        self.assertIn('ts', _v_._tooltipFields_(self.df))
        self.assertIn('bytes', _v_._tooltipFields_(self.df))


# ── every row set ────────────────────────────────────────────────────────────
#
# What every row set owes its view, whatever the component, checked over one registry of
# templates.  A component joins by adding cases to _CASES_; nothing below is per component.

def _suite_df():
    """One frame for every case: numeric, two categoricals, a count-like field, time, and
    an edge list."""
    return _df().with_columns(
        pl.datetime_range(pl.datetime(2026, 1, 1), pl.datetime(2026, 1, 1, 7), '1h', eager=True).alias('ts'),
        pl.Series('grp', ['x', 'y', 'x', 'y', 'y', 'x', 'x', 'y']),
        pl.Series('fm',  ['a', 'a', 'b', 'b', 'c', 'c', 'd', 'e']),
        pl.Series('to',  ['b', 'c', 'c', 'd', 'd', 'e', 'e', 'a']))


_EDGES_ = [('fm', 'to')]
_POS_   = {'a': (0.0, 0.0), 'b': (1.0, 0.0), 'c': (1.0, 1.0), 'd': (0.0, 1.0), 'e': (0.5, 0.5)}


def _xy_(p2s, df, x='a', y='b', **kw):
    return p2s.xyp(df, x, y, dot_size=3.0, **kw)


#: (name, build(p2s, df) -> template, row set class, the view that carries it).  The xyp
#: cases are every template the xyp-only versions of these checks used, plus the two axis
#: kinds those left out: a categorical y, and a periodic time x.
_CASES_ = [
    ('xyp: numeric axes', lambda p, d: _xy_(p, d), XYpRenderRows, 'xypi'),
    ('xyp: categorical x', lambda p, d: _xy_(p, d, x='pet'), XYpRenderRows, 'xypi'),
    ('xyp: categorical y, numeric colour', lambda p, d: _xy_(p, d, y='pet', color='bytes'), XYpRenderRows, 'xypi'),
    ('xyp: colour and legend', lambda p, d: _xy_(p, d, x='pet', color='pet', legend=True), XYpRenderRows, 'xypi'),
    ('xyp: distribution as built', lambda p, d: _xy_(p, d, x_distributions=('bytes', p.DISTRIBUTION_OUTSIDEp, 16),
                                                     opacity=0.4), XYpRenderRows, 'xypi'),
    ('xyp: stretched colour, equal aspect', lambda p, d: _xy_(p, d, color=p.CROW_STRETCHEDp, aspect='equal'),
     XYpRenderRows, 'xypi'),
    ('xyp: opacity off the list', lambda p, d: _xy_(p, d, opacity=0.37), XYpRenderRows, 'xypi'),
    ('xyp: numeric aspect', lambda p, d: _xy_(p, d, aspect=1.5), XYpRenderRows, 'xypi'),
    ('xyp: periodic time x', lambda p, d: _xy_(p, d, x=p.tField('ts', p.PT_DoWp)), XYpRenderRows, 'xypi'),
    # histop: every route through the boxplot gates, a stacked and a spectrum colour, an
    # order the row does not carry, and numeric bins
    ('histop: rows', lambda p, d: p.histop(d, 'pet'), HistopRenderRows, 'histopi'),
    ('histop: numeric count', lambda p, d: p.histop(d, 'pet', count='bytes'), HistopRenderRows, 'histopi'),
    ('histop: text count', lambda p, d: p.histop(d, 'pet', count='grp'), HistopRenderRows, 'histopi'),
    ('histop: boxplot', lambda p, d: p.histop(d, 'pet', count='bytes', style=p.BOXPLOT_W_SWARMp),
     HistopRenderRows, 'histopi'),
    ('histop: stacked colour and legend', lambda p, d: p.histop(d, 'pet', count='bytes', color='grp', legend=True),
     HistopRenderRows, 'histopi'),
    ('histop: stretched colour', lambda p, d: p.histop(d, 'pet', color=p.CROW_STRETCHEDp), HistopRenderRows, 'histopi'),
    ('histop: order by a field', lambda p, d: p.histop(d, 'pet', order='bytes', descending=False),
     HistopRenderRows, 'histopi'),
    ('histop: stacked style, labels off', lambda p, d: p.histop(d, 'pet', style=p.STACKEDBARp, draw_labels=False),
     HistopRenderRows, 'histopi'),
    ('histop: numeric bins by label', lambda p, d: p.histop(d, 'bytes', order=p.LABELp, descending=False),
     HistopRenderRows, 'histopi'),
    # piep: each style, each colour mode the legend and scale gates tell apart, and a
    # magnitude enum on a text field (which piep turns categorical, with a warning)
    ('piep: rows', lambda p, d: p.piep(d, 'pet'), PiepRenderRows, 'piepi'),
    ('piep: donut, labels, a count, smallest first',
     lambda p, d: p.piep(d, 'pet', style=p.DONUTp, draw_labels=True, count='bytes', descending=False),
     PiepRenderRows, 'piepi'),
    ('piep: waffle, categorical colour and legend',
     lambda p, d: p.piep(d, 'pet', style=p.WAFFLEp, color='grp', legend=True), PiepRenderRows, 'piepi'),
    ('piep: stretched colour', lambda p, d: p.piep(d, 'pet', color=p.CROW_STRETCHEDp, legend='bottom'),
     PiepRenderRows, 'piepi'),
    ('piep: magnitude of a field', lambda p, d: p.piep(d, 'pet', color=('bytes', p.CMAGNITUDE_SUMp)),
     PiepRenderRows, 'piepi'),
    ('piep: magnitude of a text field', lambda p, d: p.piep(d, 'pet', color=('grp', p.CMAGNITUDE_SUMp)),
     PiepRenderRows, 'piepi'),
    ('piep: one fixed colour', lambda p, d: p.piep(d, 'pet', color='#aa3300'), PiepRenderRows, 'piepi'),
    # chordp: every gate's both sides, both colour channels, values off the lists, and a
    # pinned pos= (whose count= warning is part of the view as built)
    ('chordp: defaults', lambda p, d: p.chordp(d, _EDGES_), ChordpRenderRows, 'chordpi'),
    ('chordp: count with vary sizes', lambda p, d: p.chordp(d, _EDGES_, count='bytes', node_size='vary',
                                                            link_size='vary'), ChordpRenderRows, 'chordpi'),
    ('chordp: bundled, circular labels, legend of node names',
     lambda p, d: p.chordp(d, _EDGES_, link_shape='bundled', draw_labels=True, label_style='circular',
                           node_color=p.COLOR_BY_NODE_NAME, legend='bottom'), ChordpRenderRows, 'chordpi'),
    ('chordp: categorical link colour and legend', lambda p, d: p.chordp(d, _EDGES_, color='grp', legend=True),
     ChordpRenderRows, 'chordpi'),
    ('chordp: both colour scales', lambda p, d: p.chordp(d, _EDGES_, color=p.CROW_STRETCHEDp,
                                                         node_color=('bytes', p.CMAGNITUDE_SUMp)),
     ChordpRenderRows, 'chordpi'),
    ('chordp: values off the lists', lambda p, d: p.chordp(d, _EDGES_, link_shape='bundled', link_size=2.0,
                                                           bundle_strength=0.6, node_opacity=0.45),
     ChordpRenderRows, 'chordpi'),
    ('chordp: pinned pos with a count', lambda p, d: p.chordp(d, _EDGES_, pos=_POS_, count='bytes'),
     ChordpRenderRows, 'chordpi'),
    # timep: auto and explicit levels (one the list carries, one it cannot), the boxplot
    # gates, and both colour kinds
    ('timep: auto', lambda p, d: p.timep(d, 'ts'), TimepRenderRows, 'timepi'),
    ('timep: an explicit periodic level', lambda p, d: p.timep(d, p.tField('ts', p.PT_Hp)),
     TimepRenderRows, 'timepi'),
    ('timep: a level finer than the data', lambda p, d: p.timep(d, p.tField('ts', p.LT_Y_m_d_H_Mp)),
     TimepRenderRows, 'timepi'),
    ('timep: numeric count, boxplot', lambda p, d: p.timep(d, 'ts', count='bytes', style=p.BOXPLOTp),
     TimepRenderRows, 'timepi'),
    ('timep: categorical colour and legend', lambda p, d: p.timep(d, 'ts', count='bytes', color='grp', legend=True),
     TimepRenderRows, 'timepi'),
    ('timep: stretched colour', lambda p, d: p.timep(d, 'ts', color=p.CROW_STRETCHEDp), TimepRenderRows, 'timepi'),
]


class _Logged_(logging.Handler):
    def __init__(self):
        super().__init__(logging.WARNING)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


@contextmanager
def _warnings_(p2s):
    """The messages p2s logs at WARNING or above inside the block.  Three things could make
    that vacuous, and each is undone for the duration: a test that disabled the logger,
    one that raised its level, and the logger's warn-once filter -- which drops any
    message already logged ONCE IN THE PROCESS, by any test, so a repeat of it would pass
    unseen."""
    _logged_, _logger_ = _Logged_(), p2s.logger
    _disabled_, _level_ = _logger_.disabled, _logger_.level
    _logger_.disabled = False
    if _logger_.getEffectiveLevel() > logging.WARNING: _logger_.setLevel(logging.WARNING)
    for _f_ in _logger_.filters:
        if type(_f_).__name__ == 'OnceFilter': _f_.seen_messages.clear()
    _logger_.addHandler(_logged_)
    try:
        yield _logged_.messages
    finally:
        _logger_.removeHandler(_logged_)
        _logger_.disabled = _disabled_
        _logger_.setLevel(_level_)


def _sampled_(items):
    """A row's values as one half of a PAIR: all of a short list, and a long one's two
    ends and middle, plus anything as built.  The last value survives on purpose -- it is
    where the lists keep the value that opens other rows ('vary', 'bundled', 'x+y')."""
    if len(items) <= 3: return items
    _std_ = [_i_ for _i_ in items if _i_[0] != AS_BUILT_MNEMONIC]
    return [_std_[0], _std_[len(_std_) // 2], _std_[-1]] + [_i_ for _i_ in items if _i_[0] == AS_BUILT_MNEMONIC]


def _reachable_(rows):
    """Every setting the panel can reach in one move from the template, with every value
    of every live row -- and in two moves, each made on a row live at the time (which is
    how the panel gates them), over sampled values.  Every value renders; pairs are for
    what rows do together, and sampling both halves is what keeps a view with a dozen
    rows (chordpi, two ten-step opacities among them) to a few hundred renders."""
    _init_ = rows.initial_settings()
    _out_ = {}
    for _r1_ in rows.rows:
        if not _r1_.enabled(_init_): continue
        for _, _v1_ in _r1_.items:
            _s1_ = dict(_init_, **{_r1_.kind: _v1_})
            _out_[tuple(sorted(_s1_.items()))] = _s1_
        for _, _v1_ in _sampled_(_r1_.items):
            _s1_ = dict(_init_, **{_r1_.kind: _v1_})
            for _r2_ in rows.rows:
                if _r2_ is _r1_ or not _r2_.enabled(_s1_): continue
                for _, _v2_ in _sampled_(_r2_.items):
                    _s2_ = dict(_s1_, **{_r2_.kind: _v2_})
                    _out_[tuple(sorted(_s2_.items()))] = _s2_
    return list(_out_.values())


class TestEveryRowSet(unittest.TestCase):
    def setUp(self):
        self.p2s = Polars2SVG()
        self.df  = _suite_df()

    def _each_(self):
        for _name_, _build_, _cls_, _ in _CASES_:
            _t_ = _build_(self.p2s, self.df)
            yield _name_, _t_, _cls_(_t_)

    def test_an_untouched_panel_overrides_nothing(self):
        for _name_, _, _rows_ in self._each_():
            with self.subTest(case=_name_):
                self.assertEqual(_rows_.overrides(_rows_.initial_settings()), {})

    def test_every_initial_value_is_one_of_its_items(self):
        for _name_, _, _rows_ in self._each_():
            for _r_ in _rows_.rows:
                with self.subTest(case=_name_, row=_r_.kind):
                    self.assertIn(_r_.initial, [_label_ for _, _label_ in _r_.items])

    def test_row_keys_come_from_the_table_and_are_distinct(self):
        """A row's key is its kind's RENDER_ROW_KEYS entry, so every kind has to be listed.
        No two rows on a view share a key, and none takes a panel cursor key (a / A / j /
        k) or a base row's (s / i)."""
        for _name_, _, _rows_ in self._each_():
            _keys_ = [_r_.mnemonic for _r_ in _rows_.rows]
            with self.subTest(case=_name_):
                self.assertEqual(len(_keys_), len(set(_keys_)))
                self.assertFalse(set(_keys_) & RESERVED_ROW_KEYS)
            for _r_ in _rows_.rows:
                with self.subTest(case=_name_, row=_r_.kind):
                    self.assertEqual(_r_.mnemonic, RENDER_ROW_KEYS.get(_r_.kind))

    def test_value_keys_and_labels_are_distinct_and_leave_the_picker_keys_alone(self):
        for _name_, _, _rows_ in self._each_():
            for _r_ in _rows_.rows:
                with self.subTest(case=_name_, row=_r_.kind):
                    _keys_   = [_m_ for _m_, _ in _r_.items]
                    _labels_ = [_l_ for _, _l_ in _r_.items]
                    self.assertEqual(len(_keys_), len(set(_keys_)))
                    self.assertEqual(len(_labels_), len(set(_labels_)))
                    self.assertFalse(set(_keys_) & set('jk'))

    def test_every_reachable_setting_renders_without_a_new_warning(self):
        """Everything within two moves of the template -- change a live row, then any row
        live after it -- renders, and logs no warning the view as built does not.  A row
        that renders something other than what it says (a boxplot that silently fell back
        to bars) is a failure too, since the component warns when it does that.

        Two moves, because rows interact: one move from the template never reaches a row
        that only goes live once another has moved (xyp's placement and bins wait on its
        distributions), nor two values that fail only together.  The first run of this
        found xypi offering aspect= on a periodic time axis, where xyp crashed."""
        for _name_, _t_, _rows_ in self._each_():
            _init_ = _rows_.initial_settings()
            with _warnings_(self.p2s) as _logged_:
                _t_.render_with(_t_.df_orig)._repr_svg_()
            _as_built_ = set(_logged_)
            for _s_ in _reachable_(_rows_):
                _moved_ = {_k_: _v_ for _k_, _v_ in _s_.items() if _v_ != _init_[_k_]}
                with self.subTest(case=_name_, moved=_moved_):
                    with _warnings_(self.p2s) as _logged_:
                        _t_.render_with(_t_.df_orig, **_rows_.overrides(_s_))._repr_svg_()
                    self.assertEqual(set(_logged_) - _as_built_, set())

    def test_the_warning_capture_is_not_vacuous(self):
        """The check above passes by seeing nothing, so prove it can see something: xyp
        warns when aspect='geo' meets a y range that cannot be latitude.  Twice, with the
        logger disabled beforehand -- the second time is the one the warn-once filter
        would swallow."""
        _df_ = pl.DataFrame({'a': [1.0, 2.0], 'b': [500.0, 900.0]})
        _was_ = self.p2s.logger.disabled
        self.p2s.logger.disabled = True                 # the capture must switch it back on
        try:
            for _time_ in ('first', 'second'):
                with _warnings_(self.p2s) as _logged_:
                    self.p2s.xyp(_df_, 'a', 'b', dot_size=3.0, aspect='geo')._repr_svg_()
                with self.subTest(time=_time_):
                    self.assertTrue(any("aspect='geo'" in _m_ for _m_ in _logged_), _logged_)
        finally:
            self.p2s.logger.disabled = _was_


@unittest.skipUnless(_PANEL_AVAILABLE_, 'panel not installed')
class TestEveryRowSetOnItsView(unittest.TestCase):
    def test_the_view_carries_the_rows_and_an_untouched_one_renders_as_built(self):
        _p2s_, _df_ = Polars2SVG(), _suite_df()
        for _name_, _build_, _cls_, _view_ in _CASES_:
            _t_ = _build_(_p2s_, _df_)
            _v_ = getattr(_p2s_, _view_)(_t_)
            with self.subTest(case=_name_):
                self.assertIs(type(_v_)._render_rows_cls_, _cls_)
                self.assertEqual([_r_[:3] for _r_ in _v_.config_panel_rows[2:]],
                                 [[_r_.mnemonic, _r_.kind, _r_.label] for _r_ in _cls_(_t_).rows])
                self.assertEqual(_v_._overrides_, {})
                self.assertEqual(normalize_svg(_v_.mod_inner), normalize_svg(_t_._repr_svg_()))


if __name__ == '__main__':
    unittest.main()
