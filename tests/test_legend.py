#
# test_legend.py
#
# legend= / colorbar support.  Covers:
#   - the layered legend= spec (True / position string / dict) and its validation
#   - kind auto-selection from the resolved color mode (categorical vs colorbar)
#   - Decision A: a truthy legend with nothing to legend silently omits
#   - Decision B: "reserve from wxh" -- the output stays wxh, the plot shrinks
#   - the metadata-capture hook (component.legend_info): categorical entries match
#     the rendered swatch colors; colorbar domain honors the same precedence as
#     the render normalization (explicit min/max, stretched globals, data)
#   - set_defaults(legend=...) global + per-component; template flow
#   - WebGPU parity: the legend is recorded into the DisplayList, not just SVG
#
import logging
import unittest

import polars as pl

import polars2svg
from polars2svg import Polars2SVG, InvalidSpecError, LegendInfo

_DF_ = pl.DataFrame({
    'x':   [1, 2, 3, 4, 5, 6, 7, 8] * 10,
    'y':   [2, 4, 1, 8, 5, 7, 3, 6] * 10,
    'cat': ['a', 'b', 'c', 'a', 'b', 'c', 'd', 'e'] * 10,
    'val': [1.0, 2.5, 3.2, 0.5, 4.4, 2.2, 1.1, 9.9] * 10,
})


class TestLegendSpecResolution(unittest.TestCase):
    def setUp(self):
        self.p2s = Polars2SVG()

    def test_false_and_none_resolve_to_no_legend(self):
        self.assertIsNone(self.p2s.legendResolveSpec(False))
        self.assertIsNone(self.p2s.legendResolveSpec(None))

    def test_true_aliases_to_right(self):
        _spec_ = self.p2s.legendResolveSpec(True)
        self.assertEqual(_spec_['pos'], 'right')

    def test_position_strings(self):
        for _pos_ in ('right', 'left', 'top', 'bottom'):
            self.assertEqual(self.p2s.legendResolveSpec(_pos_)['pos'], _pos_)

    def test_dict_form_defaults(self):
        _spec_ = self.p2s.legendResolveSpec({'title': 'T'})
        self.assertEqual(_spec_['pos'], 'right')
        self.assertEqual(_spec_['title'], 'T')
        self.assertIsNone(_spec_['max_items'])
        self.assertEqual(_spec_['order'], 'count')

    def test_invalid_specs_raise(self):
        for _bad_ in ('middle', {'poz': 'right'}, {'pos': 'up'}, {'max_items': 0},
                      {'max_items': True}, {'order': 'size'}, {'fmt': 12}, 42, 3.5):
            with self.assertRaises(InvalidSpecError, msg=f'legend={_bad_!r} should raise'):
                self.p2s.legendResolveSpec(_bad_)

    def test_component_raises_on_bad_spec(self):
        with self.assertRaises(InvalidSpecError):
            self.p2s.xyp(_DF_, 'x', 'y', color='cat', legend='center')


class TestLegendKindSelection(unittest.TestCase):
    def setUp(self):
        self.p2s = Polars2SVG()

    def test_kind_mapping(self):
        self.assertEqual(self.p2s.legendKind(self.p2s.CSETp),           'categorical')
        self.assertEqual(self.p2s.legendKind(self.p2s.CROW_MAGNITUDEp), 'colorbar')
        self.assertEqual(self.p2s.legendKind(self.p2s.CMAGNITUDE_SUMp), 'colorbar')
        self.assertEqual(self.p2s.legendKind(self.p2s.CSTRETCHED_MAXp), 'colorbar')
        self.assertIsNone(self.p2s.legendKind(None))

    def test_spectrum_endpoints_match_palette(self):
        self.assertEqual(self.p2s.legendSpectrumColor(0.0), self.p2s.spectrum_palette[0])
        self.assertEqual(self.p2s.legendSpectrumColor(1.0), self.p2s.spectrum_palette[-1])


class TestXypLegendCategorical(unittest.TestCase):
    def setUp(self):
        self.p2s = Polars2SVG()

    def test_default_is_no_legend(self):
        _r_ = self.p2s.xyp(_DF_, 'x', 'y', color='cat')
        self.assertIsNone(_r_.legend_info)
        self.assertEqual(_r_.svg_legend, '')

    def test_categorical_capture_and_swatch_colors(self):
        _r_ = self.p2s.xyp(_DF_, 'x', 'y', color='cat', legend=True)
        self.assertIsInstance(_r_.legend_info, LegendInfo)
        self.assertEqual(_r_.legend_info.kind, 'categorical')
        self.assertEqual(_r_.legend_info.title, 'cat')
        _labels_ = [_e_[0] for _e_ in _r_.legend_info.entries]
        self.assertEqual(sorted(_labels_), ['a', 'b', 'c', 'd', 'e'])
        # every swatch hex must be the same color the CSETp pipeline hashes for
        # that (string-cast) category value
        for _label_, _hex_ in _r_.legend_info.entries:
            self.assertEqual(_hex_, self.p2s.color(_label_))
            self.assertIn(f'fill="{_hex_}"', _r_.svg_legend)

    def test_order_count_is_frequency_descending(self):
        _df_ = pl.DataFrame({'x': [1, 2, 3, 4, 5, 6], 'y': [1, 2, 3, 4, 5, 6],
                             'cat': ['rare', 'mid', 'mid', 'common', 'common', 'common']})
        _r_  = self.p2s.xyp(_df_, 'x', 'y', color='cat', legend=True)
        self.assertEqual([_e_[0] for _e_ in _r_.legend_info.entries], ['common', 'mid', 'rare'])

    def test_order_label_is_alphabetic(self):
        _df_ = pl.DataFrame({'x': [1, 2, 3], 'y': [1, 2, 3], 'cat': ['b', 'c', 'a']})
        _r_  = self.p2s.xyp(_df_, 'x', 'y', color='cat', legend={'order': 'label'})
        self.assertEqual([_e_[0] for _e_ in _r_.legend_info.entries], ['a', 'b', 'c'])

    def test_order_explicit_list(self):
        _r_ = self.p2s.xyp(_DF_, 'x', 'y', color='cat', legend={'order': ['c', 'a']})
        _labels_ = [_e_[0] for _e_ in _r_.legend_info.entries]
        self.assertEqual(_labels_[:2], ['c', 'a'])

    def test_max_items_and_overflow(self):
        _r_ = self.p2s.xyp(_DF_, 'x', 'y', color='cat', legend={'max_items': 3})
        self.assertEqual(len(_r_.legend_info.entries), 3)
        self.assertEqual(_r_.legend_info.overflow, 2)
        self.assertIn('... 2 more', _r_.svg_legend)

    def test_color_overrides_respected(self):
        self.p2s.setColorOverrides({'a': '#123456'})
        try:
            _r_ = self.p2s.xyp(_DF_, 'x', 'y', color='cat', legend=True)
            _lu_ = dict(_r_.legend_info.entries)
            self.assertEqual(_lu_['a'], '#123456')
        finally:
            self.p2s.removeColorOverrides('a')

    def test_title_override(self):
        _r_ = self.p2s.xyp(_DF_, 'x', 'y', color='cat', legend={'title': 'Category'})
        self.assertEqual(_r_.legend_info.title, 'Category')
        self.assertIn('Category', _r_.svg_legend)


class TestXypLegendColorbar(unittest.TestCase):
    def setUp(self):
        self.p2s = Polars2SVG()

    def test_numeric_field_gets_colorbar(self):
        _r_ = self.p2s.xyp(_DF_, 'x', 'y', color='val', legend=True)
        self.assertEqual(_r_.legend_info.kind, 'colorbar')
        self.assertEqual(_r_.legend_info.title, 'val')
        # domain = min/max of the per-pixel aggregated sums (what the spectrum saw)
        self.assertEqual(_r_.legend_info.vmin, _r_.df_pixels['__color_sum__'].min())
        self.assertEqual(_r_.legend_info.vmax, _r_.df_pixels['__color_sum__'].max())
        # gradient endpoints from the shared spectrum palette appear as rects
        self.assertIn('<rect', _r_.svg_legend)

    def test_explicit_magnitude_range_wins(self):
        _r_ = self.p2s.xyp(_DF_, 'x', 'y', color='val', legend=True,
                           color_magnitude_min=0.0, color_magnitude_max=100.0)
        self.assertEqual(_r_.legend_info.vmin, 0.0)
        self.assertEqual(_r_.legend_info.vmax, 100.0)

    def test_stretched_globals_win(self):
        _r_ = self.p2s.xyp(_DF_, 'x', 'y', color=('val', self.p2s.CSTRETCHED_SUMp),
                           legend=True, color_stretched_global_values=[1.0, 5.0, 250.0])
        self.assertEqual(_r_.legend_info.vmin, 1.0)
        self.assertEqual(_r_.legend_info.vmax, 250.0)

    def test_crow_titles_rows(self):
        _r_ = self.p2s.xyp(_DF_, 'x', 'y', color=self.p2s.CROW_MAGNITUDEp, legend=True)
        self.assertEqual(_r_.legend_info.kind, 'colorbar')
        self.assertEqual(_r_.legend_info.title, 'rows')

    def test_fmt_applies_to_tick_labels(self):
        _r_ = self.p2s.xyp(_DF_, 'x', 'y', color='val', legend={'fmt': '{:.1f}!'})
        self.assertTrue(_r_.legend_info.vmin_label.endswith('!'))
        self.assertIn(_r_.legend_info.vmax_label, _r_.svg_legend)


class TestXypLegendSilentOmit(unittest.TestCase):
    '''Decision A: legend=True with nothing to legend renders nothing, no warning/error.'''
    def setUp(self):
        self.p2s = Polars2SVG()

    def test_no_color_omits(self):
        _r_ = self.p2s.xyp(_DF_, 'x', 'y', legend=True)
        self.assertIsNone(_r_.legend_info)
        self.assertEqual(_r_.svg_legend, '')
        self.assertEqual(_r_._legend_reserve_, (0, 0, 0, 0))

    def test_literal_hex_color_omits(self):
        _r_ = self.p2s.xyp(_DF_, 'x', 'y', color='#ff0000', legend=True)
        self.assertIsNone(_r_.legend_info)
        self.assertEqual(_r_.svg_legend, '')


class TestXypLegendReserveFromWxh(unittest.TestCase):
    '''Decision B: wxh is the physical output size; the plot region shrinks.'''
    def setUp(self):
        self.p2s = Polars2SVG()

    def test_wxh_and_svg_dims_unchanged(self):
        _r_ = self.p2s.xyp(_DF_, 'x', 'y', color='cat', legend=True, wxh=(256, 256))
        self.assertEqual(_r_.wxh, (256, 256))
        self.assertIn('width="256"', _r_.svg)
        self.assertIn('height="256"', _r_.svg)

    def test_plot_shrinks_on_the_reserved_side(self):
        _base_  = self.p2s.xyp(_DF_, 'x', 'y', color='cat')
        for _pos_, _idx_ in [('right', 0), ('left', 0), ('top', 1), ('bottom', 1)]:
            _r_ = self.p2s.xyp(_DF_, 'x', 'y', color='cat', legend=_pos_)
            self.assertLess(_r_.plot_size[_idx_], _base_.plot_size[_idx_],
                            f'legend={_pos_!r} should shrink plot_size[{_idx_}]')
        # left/top also shift the plot origin into the remaining space
        _left_ = self.p2s.xyp(_DF_, 'x', 'y', color='cat', legend='left')
        self.assertGreater(_left_.plot_origin[0], _base_.plot_origin[0])

    def test_dots_stay_out_of_the_legend_strip(self):
        _r_ = self.p2s.xyp(_DF_, 'x', 'y', color='cat', legend='right')
        _strip_x_ = _r_._legend_region_[0]
        self.assertLessEqual(_r_.df_pixels['__xpx__'].max(), _strip_x_)

    def test_raster_int_dot_size_path(self):
        _r_ = self.p2s.xyp(_DF_, 'x', 'y', color='val', dot_size=4, legend='right')
        self.assertEqual(_r_.legend_info.kind, 'colorbar')
        self.assertLessEqual(_r_.df_pixels['__xpx__'].max(), _r_._legend_region_[0])

    def test_too_small_canvas_drops_legend_with_warning(self):
        with self.assertLogs('polars2svg_logger', level='WARNING') as _cm_:
            _r_ = self.p2s.xyp(_DF_, 'x', 'y', color='cat', legend=True, wxh=(60, 60))
        self.assertIsNone(_r_.legend_info)
        self.assertTrue(any('not enough space for legend' in _m_ for _m_ in _cm_.output))


class TestXypLegendDefaultsAndTemplates(unittest.TestCase):
    def setUp(self):
        self.p2s = Polars2SVG()
        self.p2s.reset_defaults()

    def tearDown(self):
        self.p2s.reset_defaults()

    def test_global_set_defaults(self):
        self.p2s.set_defaults(legend='left')
        _r_ = self.p2s.xyp(_DF_, 'x', 'y', color='cat')
        self.assertIsNotNone(_r_.legend_info)
        self.assertEqual(_r_.legend_spec['pos'], 'left')

    def test_component_set_defaults(self):
        self.p2s.set_defaults('xyp', legend=True)
        _r_ = self.p2s.xyp(_DF_, 'x', 'y', color='cat')
        self.assertIsNotNone(_r_.legend_info)

    def test_explicit_false_overrides_default(self):
        self.p2s.set_defaults(legend=True)
        _r_ = self.p2s.xyp(_DF_, 'x', 'y', color='cat', legend=False)
        self.assertIsNone(_r_.legend_info)

    def test_template_clone_keeps_legend(self):
        _template_ = self.p2s.xyp(_DF_, 'x', 'y', color='cat', legend='right')
        _clone_    = self.p2s.xyp(df=_DF_, template=_template_)
        self.assertIsNotNone(_clone_.legend_info)
        self.assertEqual(_clone_.legend_spec['pos'], 'right')

    def test_template_clone_can_disable_legend(self):
        _template_ = self.p2s.xyp(_DF_, 'x', 'y', color='cat', legend='right')
        _clone_    = self.p2s.xyp(df=_DF_, template=_template_, legend=False)
        self.assertIsNone(_clone_.legend_info)


_GRAPH_DF_ = pl.DataFrame({
    'fm':   ['a', 'b', 'c', 'a', 'd', 'b', 'e', 'c', 'd', 'e'],
    'to':   ['b', 'c', 'a', 'c', 'a', 'a', 'b', 'd', 'e', 'a'],
    'kind': ['x', 'y', 'x', 'z', 'y', 'x', 'z', 'y', 'x', 'z'],
    'w':    [30, 1, 2, 40, 1, 2, 50, 7, 9, 11],
})
_POS_ = {'a': (0, 0), 'b': (1, 0), 'c': (1, 1), 'd': (0, 1), 'e': (0.5, 0.5)}


class TestLegendFanOut(unittest.TestCase):
    '''Every rendered component accepts legend=; kind auto-selection and the
    swatch-color contract hold across them.'''
    def setUp(self):
        self.p2s = Polars2SVG()
        self.bar_df = pl.DataFrame({
            'bin': ['alpha', 'beta', 'gamma', 'delta'] * 10,
            'grp': ['x', 'x', 'y', 'z'] * 10,
            'val': [5.0, 2.0, 8.0, 1.0] * 10,
        })

    def _assertCategoricalMatches(self, _r_):
        self.assertEqual(_r_.legend_info.kind, 'categorical')
        for _label_, _hex_ in _r_.legend_info.entries:
            self.assertEqual(_hex_, self.p2s.color(_label_))

    def test_histop_stacked_categorical(self):
        _r_ = self.p2s.histop(self.bar_df, 'bin', color='grp', legend='bottom', wxh=(192, 256))
        self._assertCategoricalMatches(_r_)
        self.assertEqual(_r_.wxh, (192, 256))
        # entries ordered by aggregated count desc: x(20) > y(10) = z(10)
        self.assertEqual(_r_.legend_info.entries[0][0], 'x')

    def test_histop_colorbar_matches_color_stat_range(self):
        _r_ = self.p2s.histop(self.bar_df, 'bin', color='val', legend='right', wxh=(256, 256))
        self.assertEqual(_r_.legend_info.kind, 'colorbar')
        self.assertEqual(_r_.legend_info.vmin, _r_._color_stat_min_)
        self.assertEqual(_r_.legend_info.vmax, _r_._color_stat_max_)

    def test_histop_boxplot_ignores_color_and_legend(self):
        _r_ = self.p2s.histop(self.bar_df, 'bin', count='val', style=self.p2s.BOXPLOTp,
                              color='grp', legend=True)
        self.assertIsNone(_r_.legend_info)

    def test_timep_categorical_and_colorbar(self):
        from datetime import datetime
        _df_ = pl.DataFrame({'ts':  [datetime(2024, _m_, 1) for _m_ in range(1, 13)] * 2,
                             'grp': ['x', 'y', 'z'] * 8,
                             'val': [float(_i_ % 7) + 1 for _i_ in range(24)]})
        _r1_ = self.p2s.timep(_df_, 'ts', color='grp', legend='right')
        self._assertCategoricalMatches(_r1_)
        _r2_ = self.p2s.timep(_df_, 'ts', color='val', legend='bottom')
        self.assertEqual(_r2_.legend_info.kind, 'colorbar')
        self.assertEqual(_r2_.legend_info.vmin, _r2_._color_stat_min_)

    def test_piep_cset_and_spectrum(self):
        _r1_ = self.p2s.piep(self.bar_df, 'bin', color='bin', legend='right', wxh=(224, 160))
        self._assertCategoricalMatches(_r1_)
        _r2_ = self.p2s.piep(self.bar_df, 'bin', color='val', legend='bottom', wxh=(160, 200))
        self.assertEqual(_r2_.legend_info.kind, 'colorbar')
        self.assertEqual(_r2_.legend_info.vmin, _r2_._color_stat_min_)

    def test_piep_fixed_and_hexlist_omit(self):
        for _c_ in ('#ff0000', ['#ff0000', '#00ff00'], None):
            _r_ = self.p2s.piep(self.bar_df, 'bin', color=_c_, legend=True)
            self.assertIsNone(_r_.legend_info, f'color={_c_!r} should not legend')

    def test_linkp_link_channel_categorical(self):
        _r_ = self.p2s.linkp(_GRAPH_DF_, [('fm', 'to')], pos=_POS_, color='kind',
                             legend='right', wxh=(288, 224))
        self._assertCategoricalMatches(_r_)
        self.assertEqual(_r_.legend_info.title, 'kind')

    def test_linkp_node_channel_fallback(self):
        _r_ = self.p2s.linkp(_GRAPH_DF_, [('fm', 'to')], pos=_POS_,
                             node_color=self.p2s.COLOR_BY_NODE_NAME, legend='right', wxh=(288, 224))
        self.assertEqual(_r_.legend_info.kind, 'categorical')
        self.assertEqual(_r_.legend_info.title, 'node')

    def test_linkp_colorbar_domain(self):
        _r_ = self.p2s.linkp(_GRAPH_DF_, [('fm', 'to')], pos=_POS_, color='w',
                             legend='bottom', wxh=(256, 256))
        self.assertEqual(_r_.legend_info.kind, 'colorbar')
        self.assertIsNotNone(_r_.legend_info.vmin)
        self.assertLessEqual(_r_.legend_info.vmin, _r_.legend_info.vmax)

    def test_linkp_nodes_shift_out_of_right_strip(self):
        _r_ = self.p2s.linkp(_GRAPH_DF_, [('fm', 'to')], pos=_POS_, color='kind',
                             legend='right', wxh=(288, 224))
        _strip_x_ = _r_._legend_region_[0]
        for _i_ in range(len(_r_.relationships)):
            for _side_ in ('fm', 'to'):
                _col_ = f'__rel{_i_}_{_side_}_sx__'
                self.assertLessEqual(_r_.df[_col_].max(), _strip_x_)

    def test_chordp_categorical_and_colorbar(self):
        try:
            from polars2svg.chordp import ChP  # noqa: F401 - needs the layouts extra
        except ImportError:
            self.skipTest('chordp requires the layouts extra (scipy)')
        _r1_ = self.p2s.chordp(_GRAPH_DF_, [('fm', 'to')], color='kind', legend='right', wxh=(288, 224))
        self._assertCategoricalMatches(_r1_)
        _r2_ = self.p2s.chordp(_GRAPH_DF_, [('fm', 'to')], color='w', legend='bottom', wxh=(256, 288))
        self.assertEqual(_r2_.legend_info.kind, 'colorbar')
        self.assertIsNotNone(_r2_.legend_info.vmin)

    def test_spreadlinesp_field_and_name(self):
        from datetime import datetime
        _df_ = pl.DataFrame({'fm':  ['ego', 'ego', 'a', 'b', 'ego', 'c'],
                             'to':  ['a', 'b', 'c', 'd', 'c', 'e'],
                             'grp': ['x', 'y', 'x', 'z', 'y', 'x'],
                             'ts':  [datetime(2024, 1, 1), datetime(2024, 1, 1),
                                     datetime(2024, 2, 1), datetime(2024, 2, 1),
                                     datetime(2024, 3, 1), datetime(2024, 3, 1)]})
        _r1_ = self.p2s.spreadlinesp(_df_, relationships=[('fm', 'to')], ego='ego', time='ts',
                                     node_color='grp', legend='right', wxh=(800, 400))
        self._assertCategoricalMatches(_r1_)
        self.assertIsNotNone(_r1_._legend_region_)
        _r2_ = self.p2s.spreadlinesp(_df_, relationships=[('fm', 'to')], ego='ego', time='ts',
                                     legend='bottom', wxh=(800, 400))
        self.assertEqual(_r2_.legend_info.title, 'node')
        _r3_ = self.p2s.spreadlinesp(_df_, relationships=[('fm', 'to')], ego='ego', time='ts',
                                     node_color='#ff0000', legend=True)
        self.assertIsNone(_r3_.legend_info)

    def test_smallp_panels_inherit_template_legend(self):
        _df_ = pl.DataFrame({'x': [1, 2, 3, 4] * 10, 'y': [2, 4, 1, 8] * 10,
                             'cat': ['a', 'b', 'c', 'a'] * 10,
                             'panel': ['P1', 'P2'] * 20})
        _tmpl_ = self.p2s.xyp(_df_, 'x', 'y', color='cat', legend=True, wxh=(192, 160))
        _sp_   = self.p2s.smallp(_df_, 'panel', sm_template=_tmpl_, wxh=(640, 240))
        self.assertIn('cat', _sp_._repr_svg_())

    def test_global_default_applies_across_components(self):
        self.p2s.reset_defaults()
        try:
            self.p2s.set_defaults(legend='right')
            _rh_ = self.p2s.histop(self.bar_df, 'bin', color='grp', wxh=(192, 256))
            _rp_ = self.p2s.piep(self.bar_df, 'bin', color='bin', wxh=(224, 160))
            self.assertIsNotNone(_rh_.legend_info)
            self.assertIsNotNone(_rp_.legend_info)
        finally:
            self.p2s.reset_defaults()


class TestXypLegendWebGPUParity(unittest.TestCase):
    def setUp(self):
        self.p2s = Polars2SVG()

    def test_legend_recorded_into_display_list(self):
        _with_    = self.p2s.xyp(_DF_, 'x', 'y', color='cat', legend=True)
        _without_ = self.p2s.xyp(_DF_, 'x', 'y', color='cat')
        _n_with_    = len(_with_.gpuDisplayList()._ops_)
        _n_without_ = len(_without_.gpuDisplayList()._ops_)
        self.assertGreater(_n_with_, _n_without_,
                           'legend must add DisplayList ops (WebGPU parity), not just SVG text')
        _payload_ = _with_.webgpu()
        self.assertIn('rect',  _payload_['buffers'])   # swatches
        self.assertIn('glyph', _payload_['buffers'])   # labels


if __name__ == '__main__':
    unittest.main()
