"""Settings-panel rows that change a generic view's RENDER (interactive_render_rows.py).

Two halves.  The row logic is plain Python and runs without the `interactive` extra: what
rows a template gets, what they start on, when they are live, and which overrides a set
of values produces.  The view half (xypi) needs panel: an untouched view renders what was
built, a row change re-renders, and a refused one snaps back.

The rule under test throughout: **an untouched panel changes nothing** -- overrides() is
{} until a row moves off the value read from the template.
"""
import asyncio
import unittest

import polars as pl

from polars2svg import Polars2SVG
from polars2svg.interactive_render_rows import XYpRenderRows, AS_BUILT_MNEMONIC
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
    def test_every_initial_value_is_one_of_its_items(self):
        for _kw_ in ({}, dict(x='pet'), dict(x_distributions=('bytes', self.p2s.DISTRIBUTION_OUTSIDEp, 16)),
                     dict(color=self.p2s.CROW_STRETCHEDp), dict(opacity=0.37), dict(aspect=1.5)):
            _, _rows_ = self._rows(**_kw_)
            for _r_ in _rows_.rows:
                with self.subTest(template=_kw_, row=_r_.kind):
                    self.assertIn(_r_.initial, [_label_ for _, _label_ in _r_.items])

    def test_row_mnemonics_leave_the_panel_keys_and_the_base_rows_alone(self):
        """a/A/j/k move the panel cursor; s and i are the selection-shape and tooltip rows."""
        _, _rows_ = self._rows()
        _ms_ = [_r_.mnemonic for _r_ in _rows_.rows]
        self.assertEqual(len(_ms_), len(set(_ms_)))
        self.assertFalse(set(_ms_) & set('aAjksi'))

    def test_item_mnemonics_are_distinct_and_leave_the_picker_keys_alone(self):
        _, _rows_ = self._rows()
        for _r_ in _rows_.rows:
            with self.subTest(row=_r_.kind):
                _ms_ = [_m_ for _m_, _ in _r_.items]
                self.assertEqual(len(_ms_), len(set(_ms_)))
                self.assertFalse(set(_ms_) & set('jk'))

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
    def test_an_untouched_panel_overrides_nothing(self):
        for _kw_ in ({}, dict(x='pet', color='pet', legend=True),
                     dict(x_distributions=('bytes', self.p2s.DISTRIBUTION_OUTSIDEp, 16), opacity=0.4),
                     dict(color=self.p2s.CROW_STRETCHEDp, aspect='equal')):
            _, _rows_ = self._rows(**_kw_)
            with self.subTest(template=_kw_):
                self.assertEqual(_rows_.overrides(_rows_.initial_settings()), {})

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

    def test_every_value_of_every_live_row_renders(self):
        """Nothing the panel offers on a live row may make the constructor raise."""
        for _kw_ in ({}, dict(x='pet', color='pet'), dict(color=self.p2s.CROW_STRETCHEDp)):
            _t_, _rows_ = self._rows(**_kw_)
            _init_ = _rows_.initial_settings()
            for _r_ in _rows_.rows:
                for _, _label_ in _r_.items:
                    _s_ = dict(_init_, **{_r_.kind: _label_})
                    if _r_.kind in ('placement', 'bins'): _s_['distributions'] = 'x+y'
                    if not _r_.enabled(_s_): continue
                    with self.subTest(template=_kw_, row=_r_.kind, value=_label_):
                        _t_.render_with(self.df, **_rows_.overrides(_s_))


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

    def test_an_untouched_view_renders_what_was_built(self):
        _t_, _v_ = self._view(x='pet', color='pet', legend=True)
        self.assertEqual(_v_._overrides_, {})
        self.assertEqual(normalize_svg(_v_.mod_inner), normalize_svg(_t_._repr_svg_()))

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

    def test_the_other_generic_views_have_no_render_rows_yet(self):
        _v_ = self.p2s.histopi(self.p2s.histop(self.df, bin_by='pet'))
        self.assertEqual([_r_[1] for _r_ in _v_.config_panel_rows], ['select_shape', 'tooltip'])
        self.assertEqual(_v_.render_settings, {})


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


if __name__ == '__main__':
    unittest.main()
