"""F1 -- per-element hover tooltips (PLANNING.md section 7).

**These tests had to be written rather than inherited, and that is the point.** W1's
phases 2-4 established the lesson and the config panel repeated it: a JS-only overlay is
structurally invisible to the parity goldens, because it exists only mid-gesture and the
digest is taken after. A tooltip is that shape, and ``icon=`` makes it worse -- *a mutant
that rendered the wrong subset into the icon would pass every one of the inherited browser
tests*, because none of them hovers and reads.

Split by what it takes to observe:

* here -- the Python half, driven by writing the params the browser writes. That reaches
  the hit test, the text content, the icon render, the cache and the U7 staleness ticket,
  none of which need a browser.
* ``tests/interaction/test_tooltip_browser.py`` -- the drawn overlay and the panel row,
  which do.
"""
import asyncio
import re
import unittest

import polars as pl

from polars2svg import Polars2SVG
from polars2svg import interactive_controller as ic


def _df():
    return pl.DataFrame({
        'x':   [1.0, 2.0, 3.0, 4.0, 5.0],
        'y':   [2.0, 4.0, 1.0, 3.0, 5.0],
        'cat': ['a', 'b', 'a', 'b', 'a'],
        'sip': ['10.0.0.1', '10.0.0.2', '10.0.0.3', '10.0.0.4', '10.0.0.5'],
    })


def _strip_svg_id(svg):
    """Drop the per-render id so two renders of the same data compare equal.

    Every component stamps a fresh `id="histop_<n>"` on its root, so the raw strings
    differ on every call and a comparison of them is always true-negative.
    """
    return re.sub(r'id="[^"]*"', 'id=""', svg)


def _hover(view, xy, seq, budget_s=5.0):
    """Drive one hover exactly as the browser's dwell timer does, and wait for the answer.

    The browser writes tooltip_x/tooltip_y and then bumps tooltip_seq; the watcher is
    async, so the answer is the payload carrying that seq back.
    """
    async def _go():
        view.tooltip_x, view.tooltip_y = int(xy[0]), int(xy[1])
        view.tooltip_seq = seq
        _deadline_ = asyncio.get_event_loop().time() + budget_s
        while asyncio.get_event_loop().time() < _deadline_:
            if view.tooltip_payload.get('seq') == seq:
                return view.tooltip_payload
            await asyncio.sleep(0.01)
        return view.tooltip_payload
    return asyncio.run(_go())


class _TooltipTestCase(unittest.TestCase):
    def setUp(self):
        self.p2s  = Polars2SVG()
        self.df   = _df()
        self.xyp  = self.p2s.xyp(self.df, x='x', y='y', color='cat', wxh=(256, 256))
        self.icon = self.p2s.histop(self.df, bin_by='cat', wxh=(48, 48))

    def _mark_xy(self):
        """A pixel that really has a mark under it, read off the rendered frame."""
        _flat_ = self.xyp.df_flat
        return int(_flat_['__xpx__'][0]), int(_flat_['__ypx__'][0])

    def _empty_xy(self):
        """A pixel with nothing under it -- far from every mark."""
        _flat_ = self.xyp.df_flat
        _taken_ = {(int(_a_), int(_b_)) for _a_, _b_ in zip(_flat_['__xpx__'], _flat_['__ypx__'])}
        for _x_ in range(0, 256, 3):
            for _y_ in range(0, 256, 3):
                if all((_x_ - _a_) ** 2 + (_y_ - _b_) ** 2 > 400 for _a_, _b_ in _taken_):
                    return _x_, _y_
        raise AssertionError('no empty pixel on this fixture')


# ── the panel row, which is how the feature is reached at all ───────────────

class TestTheTooltipRow(_TooltipTestCase):
    def test_every_generic_view_has_the_row(self):
        """The row is on all five kinds, not just the one F1 was demonstrated on."""
        for _kind_, _plot_ in (('xypi',    self.xyp),
                               ('histopi', self.p2s.histop(self.df, bin_by='cat')),
                               ('timepi',  self.p2s.timep(self.df.with_columns(
                                   pl.Series('ts', [f'2024-01-0{_i_ + 1}' for _i_ in range(5)])
                                     .str.to_date()), time='ts')),
                               ('chordpi', self.p2s.chordp(self.df, relationships=[('cat', 'sip')])),
                               ('piepi',   self.p2s.piep(self.df, bin_by='cat'))):
            with self.subTest(kind=_kind_):
                _v_ = getattr(self.p2s, _kind_)(_plot_)
                _kinds_ = [_r_[1] for _r_ in _v_.config_panel_rows]
                self.assertIn('tooltip', _kinds_, f'{_kind_} has no tooltip row')

    def test_linkpi_has_the_row_too(self):
        _lp_ = self.p2s.linkp(self.df, relationships=[('cat', 'sip')])
        _v_  = self.p2s.linkpi(_lp_)
        self.assertIn('tooltip', [_r_[1] for _r_ in _v_.config_panel_rows])

    def test_the_row_is_never_gated(self):
        """Unlike 'timing marks' or 'background', there is no state in which the row is
        inert: every hit-testable component can always show text.  It is the row's third
        VALUE that comes and goes, which is a menu_items question."""
        for _v_ in (self.p2s.xypi(self.xyp), self.p2s.xypi(self.xyp, icon=self.icon)):
            _row_ = [_r_ for _r_ in _v_.config_panel_rows if _r_[1] == 'tooltip'][0]
            self.assertTrue(_row_[3], 'the tooltip row is gated off')

    def test_without_an_icon_the_cycle_is_two_valued(self):
        _v_ = self.p2s.xypi(self.xyp)
        self.assertEqual([_i_[1] for _i_ in _v_.menu_items['tooltip']], ['off', 'text'])

    def test_with_an_icon_the_cycle_gains_the_third_state(self):
        _v_ = self.p2s.xypi(self.xyp, icon=self.icon)
        self.assertEqual([_i_[1] for _i_ in _v_.menu_items['tooltip']],
                         ['off', 'text', 'icon'])

    def test_linkpi_menu_items_gain_the_icon_state_too(self):
        """LINKPI seeds menu_items in its super().__init__ call, before __initTooltip__
        has seen the icon; __syncConfigPanel__ is what puts the third state in.  Easy to
        get wrong and invisible until someone tries to cycle to it."""
        _lp_ = self.p2s.linkp(self.df, relationships=[('cat', 'sip')])
        self.assertEqual([_i_[1] for _i_ in self.p2s.linkpi(_lp_).menu_items['tooltip']],
                         ['off', 'text'])
        self.assertEqual([_i_[1] for _i_ in
                          self.p2s.linkpi(_lp_, icon=self.icon).menu_items['tooltip']],
                         ['off', 'text', 'icon'])

    def test_the_default_is_off(self):
        """Default-off is a decision, not an oversight (PLANNING.md section 7): no
        behaviour change on upgrade, no hover round trips for anyone who has not asked
        for them, and no collision with the mouseover every browser test fires at mount."""
        self.assertEqual(self.p2s.xypi(self.xyp).tooltip, 'off')

    def test_the_row_mnemonics_are_distinct(self):
        _v_ = self.p2s.xypi(self.xyp, icon=self.icon)
        _ms_ = [_m_ for _m_, _ in _v_.menu_items['tooltip']]
        self.assertEqual(len(set(_ms_)), len(_ms_), f'duplicate value mnemonic in {_ms_}')


# ── the round trip ──────────────────────────────────────────────────────────

class TestTheHoverRoundTrip(_TooltipTestCase):
    def test_off_answers_nothing_at_all(self):
        """Not 'answers empty' -- answers *nothing*.  The whole point of default-off is
        that a view nobody asked for a tooltip on does no work per hover."""
        _v_ = self.p2s.xypi(self.xyp)
        _pay_ = _hover(_v_, self._mark_xy(), 1, budget_s=0.5)
        self.assertEqual(_pay_, {})

    def test_text_mode_names_the_records_under_the_pointer(self):
        _v_ = self.p2s.xypi(self.xyp)
        _v_.tooltip = 'text'
        _pay_ = _hover(_v_, self._mark_xy(), 1)
        self.assertFalse(_pay_.get('empty'), _pay_)
        self.assertTrue(_pay_['lines'][0].endswith('record')
                        or _pay_['lines'][0].endswith('records'), _pay_['lines'])

    def test_a_miss_is_reported_as_empty_rather_than_left_stale(self):
        """The pointer moving off a mark has to clear the box, so a miss is an answer
        and not a silence -- a silence would leave the previous tooltip on screen."""
        _v_ = self.p2s.xypi(self.xyp)
        _v_.tooltip = 'text'
        _hover(_v_, self._mark_xy(), 1)
        _pay_ = _hover(_v_, self._empty_xy(), 2)
        self.assertEqual(_pay_, {'seq': 2, 'empty': True})

    def test_the_payload_carries_the_hover_point(self):
        """The browser places the box relative to the pointer, and the pointer may have
        moved since; the payload says which point it is the answer for."""
        _v_ = self.p2s.xypi(self.xyp)
        _v_.tooltip = 'text'
        _x_, _y_ = self._mark_xy()
        _pay_ = _hover(_v_, (_x_, _y_), 1)
        self.assertEqual((_pay_['x'], _pay_['y']), (_x_, _y_))

    def test_a_hover_does_not_touch_the_stack_or_the_selection(self):
        """A tooltip is a PURE READ.  Nothing here may reach the InteractionController --
        a hover that pushed a stack frame or broadcast to peers would make resting the
        pointer somewhere a destructive act."""
        _v_ = self.p2s.xypi(self.xyp)
        _v_.tooltip = 'text'
        _stack_ = self.p2s.Polars2SVG if False else _v_.mvc.stacks['default']
        _before_ = (len(_stack_['dfs']), _stack_['index'])
        _hover(_v_, self._mark_xy(), 1)
        self.assertEqual((len(_stack_['dfs']), _stack_['index']), _before_)


# ── text content ────────────────────────────────────────────────────────────

class TestTextContent(_TooltipTestCase):
    def test_it_names_the_fields_the_component_encodes(self):
        """Not the whole row: the netflow frames are wide, and the fields the plot is
        MADE of are the ones that are never wrong about relevance."""
        _v_ = self.p2s.xypi(self.xyp)
        _v_.tooltip = 'text'
        _lines_ = _hover(_v_, self._mark_xy(), 1)['lines']
        _body_  = ' '.join(_lines_[1:])
        for _f_ in ('x', 'y', 'cat'):
            self.assertIn(_f_, _body_, _lines_)
        self.assertNotIn('sip', _body_, 'an unencoded column leaked into the default')

    def test_tooltip_fields_overrides_the_default(self):
        _v_ = self.p2s.xypi(self.xyp, tooltip_fields=['sip'])
        _v_.tooltip = 'text'
        _lines_ = _hover(_v_, self._mark_xy(), 1)['lines']
        self.assertIn('sip', ' '.join(_lines_[1:]))
        self.assertNotIn('cat', ' '.join(_lines_[1:]))

    def test_a_field_with_many_values_is_summarised_rather_than_listed(self):
        _recs_ = _df()
        _lines_ = ic._tooltipTextLines_(_recs_, ['sip'])
        self.assertIn('5 distinct', ' '.join(_lines_))

    def test_a_single_value_is_named_outright(self):
        _lines_ = ic._tooltipTextLines_(_df().head(1), ['sip'])
        self.assertIn('10.0.0.1', ' '.join(_lines_))

    def test_the_record_count_is_singular_when_there_is_one(self):
        self.assertEqual(ic._tooltipTextLines_(_df().head(1), [])[0], '1 record')
        self.assertEqual(ic._tooltipTextLines_(_df(), [])[0], '5 records')

    def test_the_line_count_is_capped(self):
        """A tooltip that grows with the frame stops being readable and starts covering
        the plot -- which is the one thing it must not do."""
        _wide_ = pl.DataFrame({f'c{_i_}': [_i_] for _i_ in range(40)})
        _lines_ = ic._tooltipTextLines_(_wide_, [f'c{_i_}' for _i_ in range(40)])
        self.assertLessEqual(len(_lines_), ic._TOOLTIP_MAX_LINES_ + 2)
        self.assertEqual(_lines_[-1], '...')


# ── magnitudes: abbreviated only where the number IS a magnitude ────────────

class TestMagnitudeText(_TooltipTestCase):
    def test_the_record_count_is_abbreviated(self):
        _recs_ = pl.DataFrame({'v': list(range(2_024))})
        self.assertEqual(ic._tooltipTextLines_(_recs_, [])[0], '2.02K records')

    def test_a_small_record_count_is_left_alone(self):
        self.assertEqual(ic._tooltipTextLines_(_df(), [])[0], '5 records')

    def test_a_distinct_count_is_abbreviated_whatever_the_field(self):
        _recs_ = pl.DataFrame({'sip': [f'10.0.{_i_ // 256}.{_i_ % 256}' for _i_ in range(1_500)]})
        self.assertIn('1.5K distinct', ' '.join(ic._tooltipTextLines_(_recs_, ['sip'])))

    def test_a_measure_field_is_abbreviated(self):
        _recs_ = pl.DataFrame({'bytes': [45_612_314]})
        _lines_ = ic._tooltipTextLines_(_recs_, ['bytes'], measures={'bytes'})
        self.assertIn('45.6M', _lines_[1])

    def test_an_identifier_field_is_not(self):
        """The reason abbreviation is opt-in per encoding: port 8080 is not '8.08K'."""
        _recs_ = pl.DataFrame({'dport': [8080], 'bytes': [8080]})
        _lines_ = ic._tooltipTextLines_(_recs_, ['dport', 'bytes'], measures={'bytes'})
        self.assertTrue(_lines_[1].endswith('8080'), _lines_)
        self.assertTrue(_lines_[2].endswith('8.08K'), _lines_)

    def test_a_non_numeric_measure_is_printed_as_is(self):
        _lines_ = ic._tooltipTextLines_(_df().head(1), ['sip'], measures={'sip'})
        self.assertIn('10.0.0.1', _lines_[1])

    def test_histopi_treats_count_as_the_measure_and_bin_by_as_not(self):
        _df_ = pl.DataFrame({'dport': [8080, 8080, 443], 'bytes': [2_000_000, 500_000, 7]})
        _v_  = self.p2s.histopi(self.p2s.histop(_df_, bin_by='dport', count='bytes', wxh=(128, 128)))
        _recs_ = _df_.filter(pl.col('dport') == 8080)
        self.assertEqual(_v_._tooltipMeasures_(_recs_), {'bytes'})
        _lines_ = ic._tooltipTextLines_(_recs_, _v_._tooltipFields_(_recs_), _v_._tooltipMeasures_(_recs_))
        _body_  = ' '.join(_lines_)
        self.assertIn('8080', _body_)
        self.assertIn('2M, 500K', _body_)

    def test_xypi_measures_dot_size_only(self):
        _df_ = pl.DataFrame({'x': [1.0, 2.0], 'y': [1.0, 2.0], 'bytes': [1_000, 2_000_000]})
        _v_  = self.p2s.xypi(self.p2s.xyp(_df_, x='x', y='y', dot_size='bytes', wxh=(128, 128)))
        self.assertEqual(_v_._tooltipMeasures_(_df_), {'bytes'})

    def test_linkpi_measures_count(self):
        _df_ = pl.DataFrame({'fm': ['a', 'b'], 'to': ['b', 'c'], 'bytes': [10, 20]})
        _v_  = self.p2s.linkpi(self.p2s.linkp(_df_, relationships=[('fm', 'to')], count='bytes'))
        self.assertEqual(_v_._tooltipMeasures_(_df_), {'bytes'})


# ── icon mode ───────────────────────────────────────────────────────────────

class TestIconMode(_TooltipTestCase):
    def test_the_icon_is_rendered_against_the_records_under_the_pointer(self):
        """The assertion a mutant would fail: the icon has to be re-rendered against the
        HIT, not against the view's own dataframe.  One mark is one row, so the icon's
        histogram has one bar -- the full frame would give it two."""
        _v_ = self.p2s.xypi(self.xyp, icon=self.icon)
        _v_.tooltip = 'icon'
        _pay_ = _hover(_v_, self._mark_xy(), 1)
        self.assertTrue(_pay_['svg'], 'icon mode produced no svg')
        _x_, _y_ = self._mark_xy()
        _recs_ = self.xyp.recordsAt((_x_, _y_), shape=self.p2s.SELECT_CIRCLEp,
                                    threshold=ic._TOOLTIP_THRESHOLD_)
        self.assertEqual(_strip_svg_id(_pay_['svg']),
                         _strip_svg_id(self.icon.render_with(_recs_)._repr_svg_()))
        self.assertNotEqual(_strip_svg_id(_pay_['svg']),
                            _strip_svg_id(self.icon.render_with(self.df)._repr_svg_()),
                            'the icon rendered the whole frame, not the hit')

    def test_the_box_is_sized_from_the_icons_own_wxh(self):
        """No new parameter: the tooltip box is the icon's own wxh, read off the
        component the user passed -- the same thing stack_controli does."""
        _v_ = self.p2s.xypi(self.xyp, icon=self.icon)
        _v_.tooltip = 'icon'
        _pay_ = _hover(_v_, self._mark_xy(), 1)
        self.assertEqual(_pay_['icon_h'], self.icon.wxh[1])
        self.assertGreaterEqual(_pay_['w'], self.icon.wxh[0])
        self.assertGreater(_pay_['h'], self.icon.wxh[1])

    def test_icon_mode_without_an_icon_falls_back_to_text(self):
        """The mode can only be reached through the panel row, which does not offer it
        without an icon -- but a stale browser write must not produce an empty box."""
        _v_ = self.p2s.xypi(self.xyp)
        _v_.tooltip = 'icon'
        _pay_ = _hover(_v_, self._mark_xy(), 1)
        self.assertEqual(_pay_['svg'], '')
        self.assertTrue(_pay_['lines'])

    def test_a_live_view_is_rejected_as_an_icon(self):
        """An icon is a RENDERED COMPONENT, not a live view: the tooltip embeds SVG.
        Worth a check because p2s.xypi(...) is the obvious thing to try."""
        with self.assertRaises(ValueError) as _ctx_:
            self.p2s.xypi(self.xyp, icon=self.p2s.xypi(self.xyp))
        self.assertIn('render_with', str(_ctx_.exception))

    def test_a_zero_sized_icon_is_rejected(self):
        _bad_ = self.p2s.histop(self.df, bin_by='cat')
        _bad_.wxh = (0, 0)
        with self.assertRaises(ValueError):
            self.p2s.xypi(self.xyp, icon=_bad_)


# ── the cache, and the U7 staleness ticket ──────────────────────────────────

class TestCacheAndStaleness(_TooltipTestCase):
    def test_resting_on_one_mark_renders_the_icon_once(self):
        """The dwell timer keeps firing while the pointer rests.  Without a cache each
        one is a full component render."""
        _v_ = self.p2s.xypi(self.xyp, icon=self.icon)
        _v_.tooltip = 'icon'
        _calls_ = []
        _real_  = self.icon.render_with
        self.icon.render_with = lambda _d_, **_kw_: (_calls_.append(1), _real_(_d_, **_kw_))[1]
        _xy_ = self._mark_xy()
        for _s_ in (1, 2, 3):
            _hover(_v_, _xy_, _s_)
        self.assertEqual(len(_calls_), 1, f'rendered {len(_calls_)} times for one mark')

    def test_moving_to_a_different_mark_invalidates_it(self):
        _v_ = self.p2s.xypi(self.xyp)
        _v_.tooltip = 'text'
        _flat_ = self.xyp.df_flat
        _a_ = (int(_flat_['__xpx__'][0]), int(_flat_['__ypx__'][0]))
        _b_ = (int(_flat_['__xpx__'][1]), int(_flat_['__ypx__'][1]))
        _pa_ = _hover(_v_, _a_, 1)['lines']
        _pb_ = _hover(_v_, _b_, 2)['lines']
        self.assertNotEqual(_pa_, _pb_, 'the cache served one mark for another')

    def test_the_cache_key_is_the_hit_and_not_the_dataframe_identity(self):
        """stack_controli's id(df) key does not transfer: every hover produces a FRESH
        dataframe, so id() reuse is the normal case here rather than a hazard to guard.
        Equal hits must therefore key equal even though they are different objects."""
        _a_, _b_ = _df(), _df()
        self.assertIsNot(_a_, _b_)
        self.assertEqual(ic._tooltipHitKey_(_a_), ic._tooltipHitKey_(_b_))
        self.assertNotEqual(ic._tooltipHitKey_(_a_), ic._tooltipHitKey_(_a_.head(3)))

    def test_the_key_is_order_independent(self):
        """The same rows in a different order are the same hit."""
        _a_ = _df()
        self.assertEqual(ic._tooltipHitKey_(_a_), ic._tooltipHitKey_(_a_.reverse()))

    def test_a_superseded_hover_does_not_overwrite_a_fresh_one(self):
        """U7, verbatim.  The brush already carries this ticket because a result landing
        after the pointer moved on wiped a fresh one; a tooltip resolving slowly is the
        identical defect, so the ticket is here from the start.

        The ticket is the watcher's own ``event.new`` and not a re-read of
        ``self.tooltip_seq``.  Re-reading hands every op queued behind the lock the same
        latest value, so the stale one cannot tell that it is stale -- it would still do
        the work, and whether the right payload won would come down to which coroutine
        finished last.
        """
        _v_ = self.p2s.xypi(self.xyp)
        _v_.tooltip = 'text'
        _slow_xy_, _fast_xy_ = self._mark_xy(), self._empty_xy()

        async def _go():
            # Issue the slow hover, then supersede it before it can answer.
            _v_.tooltip_x, _v_.tooltip_y = _slow_xy_
            _v_.tooltip_seq = 1
            _v_.tooltip_x, _v_.tooltip_y = _fast_xy_
            _v_.tooltip_seq = 2
            await asyncio.sleep(1.0)
            return _v_.tooltip_payload
        _pay_ = asyncio.run(_go())
        self.assertEqual(_pay_.get('seq'), 2,
                         f'a superseded hover won the race: {_pay_}')

    def test_a_hit_test_that_raises_does_not_take_the_view_down(self):
        """A tooltip is the least important thing on screen; every other gesture still
        has to work after one fails."""
        _v_ = self.p2s.xypi(self.xyp)
        _v_.tooltip = 'text'

        def _boom_(_xy_):
            raise RuntimeError('hit test exploded')
        _v_._tooltipRecordsAt_ = _boom_
        _pay_ = _hover(_v_, self._mark_xy(), 1, budget_s=0.5)
        self.assertEqual(_pay_, {})


if __name__ == '__main__':
    unittest.main()
