import datetime
import re
import unittest
import polars as pl
from polars2svg import Polars2SVG
from svg_test_utils import assert_timing_metrics_populated, assert_valid_svg


def _make_df():
    return pl.DataFrame({
        'fm':   ['a', 'b', 'c', 'a', 'd', 'b'],
        'to':   ['b', 'a', 'a', 'c', 'a', 'c'],
        'time': [1,   1,   1,   2,   2,   3  ],
        'w':    [3,   1,   2,   4,   1,   2  ],
    })

def _rels():
    return [('fm', 'to')]


class TestSpreadLinesPBasic(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def test_basic_render(self):
        sp = self.p2s.spreadlinesp(_make_df(), _rels(), ego='a', time='time')
        assert_valid_svg(self, sp.svg)
        self.assertIn('<circle', sp.svg)

    def test_timing_metrics_populated(self):
        sp = self.p2s.spreadlinesp(_make_df(), _rels(), ego='a', time='time')
        assert_timing_metrics_populated(self, sp, ('__parseInput__', '__calculateLayout__', '__renderSVG__'))

    def test_wxh_respected(self):
        sp = self.p2s.spreadlinesp(_make_df(), _rels(), ego='a', time='time', wxh=(600, 300))
        self.assertEqual(sp.wxh, (600, 300))

    def test_count_field(self):
        sp = self.p2s.spreadlinesp(_make_df(), _rels(), ego='a', time='time', count='w')
        assert_valid_svg(self, sp.svg)

    def test_node_color_hex(self):
        sp = self.p2s.spreadlinesp(_make_df(), _rels(), ego='a', time='time', node_color='#ff0000')
        self.assertIn('#ff0000', sp.svg)

    def test_node_color_by_node_name(self):
        sp = self.p2s.spreadlinesp(_make_df(), _rels(), ego='a', time='time',
                                   node_color=self.p2s.COLOR_BY_NODE_NAME)
        assert_valid_svg(self, sp.svg)

    def test_node_color_dict(self):
        sp = self.p2s.spreadlinesp(_make_df(), _rels(), ego='a', time='time',
                                   node_color={'b': '#00ff00', 'c': '#0000ff'})
        self.assertIn('#00ff00', sp.svg)

    def test_set_ego(self):
        sp = self.p2s.spreadlinesp(_make_df(), _rels(), ego={'a', 'b'}, time='time')
        assert_valid_svg(self, sp.svg)

    def test_repr_svg(self):
        sp = self.p2s.spreadlinesp(_make_df(), _rels(), ego='a', time='time')
        self.assertEqual(sp._repr_svg_(), sp.svg)

    def test_no_df_constructor_does_not_crash(self):
        tmpl = self.p2s.spreadlinesp(_rels(), ego='a', time='time')
        self.assertIsNone(tmpl.df)

    def test_render_with_applies_template(self):
        tmpl = self.p2s.spreadlinesp(_rels(), ego='a', time='time')
        sp   = tmpl.render_with(_make_df())
        assert_valid_svg(self, sp.svg)

    def test_render_with_overrides_ego(self):
        tmpl = self.p2s.spreadlinesp(_rels(), ego='a', time='time')
        sp   = tmpl.render_with(_make_df(), ego='b')
        assert_valid_svg(self, sp.svg)

    def test_unknown_kwarg_raises(self):
        with self.assertRaises(TypeError):
            self.p2s.spreadlinesp(_make_df(), _rels(), ego='a', time='time',
                                  _nonexistent_param_=True)

    def test_anno_does_not_crash(self):
        sp = self.p2s.spreadlinesp(_make_df(), _rels(), ego='a', time='time', anno={1: 'Event A'})
        assert_valid_svg(self, sp.svg)

    def test_highlight_nodes(self):
        sp = self.p2s.spreadlinesp(_make_df(), _rels(), ego='a', time='time',
                                   highlight_nodes={'b'})
        assert_valid_svg(self, sp.svg)

    def test_max_rings_1(self):
        sp = self.p2s.spreadlinesp(_make_df(), _rels(), ego='a', time='time', max_rings=1)
        assert_valid_svg(self, sp.svg)

    def test_max_rings_2(self):
        sp = self.p2s.spreadlinesp(_make_df(), _rels(), ego='a', time='time', max_rings=2)
        assert_valid_svg(self, sp.svg)


class TestSpreadLinesPNullTimestamps(unittest.TestCase):
    """A row with no timestamp has no bin to be drawn in.

    It used to crash the layout -- the bin labels were sorted with None among them -- and
    had it got that far, the null would have been counted as one more hour, minute and
    second, so midnight-only data looked sub-daily (PLANNING.md §5
    C-spreadlinesp-null-granularity, the bug timep had).  Now the picture, and the level
    chosen, are exactly those the row's deletion gives."""

    def setUp(self):
        import datetime
        self.p2s = Polars2SVG()
        _days_ = [datetime.datetime(2024, 1, d) for d in (1, 1, 2, 2, 3, 3, 4)]     # all at midnight
        self.df = pl.DataFrame({'fm': ['e'] * 7, 'to': list('abcdefg'), 't': _days_})
        self.df_null = pl.concat([self.df, pl.DataFrame({'fm': ['e'], 'to': ['h'], 't': [None]}, schema=self.df.schema)])

    def _sp(self, df):
        return self.p2s.spreadlinesp(df, [('fm', 'to')], ego='e', time='t', wxh=(600, 200))

    def test_a_null_timestamp_draws_what_deleting_its_row_does(self):
        from svg_test_utils import normalize_svg
        _with_, _without_ = self._sp(self.df_null), self._sp(self.df)
        self.assertEqual(normalize_svg(_with_._repr_svg_()), normalize_svg(_without_._repr_svg_()))
        self.assertEqual(_with_.ts_list, _without_.ts_list)

    def test_a_null_timestamp_does_not_make_the_data_look_sub_daily(self):
        self.assertEqual(self._sp(self.df)._ts_enum_, self.p2s.LT_Y_m_dp)
        self.assertEqual(self._sp(self.df_null)._ts_enum_, self.p2s.LT_Y_m_dp)


class TestSpreadLinesPIdScoping(unittest.TestCase):
    '''Two spreadlinesp figures on one page must not collide on SVG ids.

    Every id this component emits lives in the cloud/selection render, which needs a
    set-valued ego: `ego_is_set` puts the focal node behind the cloud glyph, and a
    partial highlight (some of the ego set selected, not all) draws the selection ring
    through a per-bin clipPath rather than as a plain <use>.

    Until 0.2.1 those ids were emitted bare -- `cloud`, `cloud_outline` and `ccl_<bin>`.
    The first two are a fixed <defs> string, so a duplicate resolved to an identical
    definition and only made the markup invalid.  `ccl_<bin>` was a real render bug: the
    clip rect is placed from that figure's own layout, so with two figures in one DOM
    the second figure's rings were clipped by the first figure's window -- and since the
    windows are only 20px tall, two figures of different heights clip to disjoint bands
    and the second figure's rings disappear completely.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def _selection(self, wxh, highlight):
        return self.p2s.spreadlinesp(_make_df(), _rels(), ego=['a', 'b'], time='time',
                                     highlight_nodes=highlight, wxh=wxh)

    # Part of the ego set selected -> the ring is drawn as a <path> through a per-bin
    # clipPath.  All of it selected -> the ring is a <use> of the outline cloud instead.
    # The two branches emit different ids, so both are exercised below.
    def _partial_selection(self, wxh): return self._selection(wxh, {'a'})
    def _full_selection(self, wxh):    return self._selection(wxh, {'a', 'b'})

    def _both(self, wxh): return (self._partial_selection(wxh), self._full_selection(wxh))

    @staticmethod
    def _ids(svg):  return re.findall(r'id="([^"]+)"', svg)

    @staticmethod
    def _refs(svg): return re.findall(r'href="#([^"]+)"|url\(#([^)]+)\)', svg)

    def test_both_selection_branches_are_actually_reached(self):
        # Everything below is vacuous if these renders stop emitting the ids at issue.
        _partial_ = self._partial_selection((700, 300)).svg
        self.assertRegex(_partial_, r'<clipPath id="ccl_\d+_\d+"')
        self.assertRegex(_partial_, r'clip-path="url\(#ccl_\d+_\d+\)"')
        self.assertRegex(_partial_, r'<use href="#cloud_\d+"')
        self.assertRegex(self._full_selection((700, 300)).svg,
                         r'<use href="#cloud_outline_\d+"')

    def test_no_bare_ids_survive(self):
        # The exact strings the audit found: `cloud_outline` and `ccl_<bin>` with no
        # random prefix between the name and the bin index.
        for _name_, _sp_ in zip(('partial', 'full'), self._both((700, 300))):
            with self.subTest(selection=_name_):
                self.assertNotRegex(_sp_.svg, r'id="cloud"')
                self.assertNotRegex(_sp_.svg, r'id="cloud_outline"')
                self.assertNotRegex(_sp_.svg, r'id="ccl_\d+"')      # ccl_<bin>, unprefixed
                self.assertNotRegex(_sp_.svg, r'href="#cloud"')
                self.assertNotRegex(_sp_.svg, r'href="#cloud_outline"')

    def test_two_figures_share_no_ids(self):
        for _name_, _fn_ in (('partial', self._partial_selection),
                             ('full',    self._full_selection)):
            with self.subTest(selection=_name_):
                _a_ = self._ids(_fn_((700, 300)).svg)
                _b_ = self._ids(_fn_((500, 200)).svg)
                self.assertNotEqual(_a_, [], 'no ids emitted -- the render changed')
                self.assertEqual(sorted(set(_a_) & set(_b_)), [])

    def test_two_figures_in_one_document_keep_unique_ids(self):
        # The page case, stated directly: concatenate the figures and every id in the
        # combined document must still be unique.  Mixed branches and mixed sizes,
        # because that is what collided -- clip rects placed from each figure's layout.
        _combined_ = ''.join(sp.svg for sp in self._both((700, 300)) + self._both((500, 200)))
        _ids_   = self._ids(_combined_)
        _dupes_ = sorted({i for i in _ids_ if _ids_.count(i) > 1})
        self.assertEqual(_dupes_, [], f'duplicate ids across figures: {_dupes_}')

    def test_every_reference_resolves_within_its_own_figure(self):
        # A scoped id is only worth anything if the reference moved with it.
        for _wxh_ in ((700, 300), (500, 200)):
            for _name_, _sp_ in zip(('partial', 'full'), self._both(_wxh_)):
                with self.subTest(wxh=_wxh_, selection=_name_):
                    _known_ = set(self._ids(_sp_.svg))
                    _used_  = {r for pair in self._refs(_sp_.svg) for r in pair if r}
                    self.assertNotEqual(_used_, set(), 'no references emitted')
                    self.assertEqual(sorted(_used_ - _known_), [],
                                     'reference to an id this figure does not define')


def _make_day_df(dtype=None):
    # The ego 'a' talks on each of three days, at hours that differ from the bin start,
    # so a value only matches its bin if it is truncated the way the bins are.
    _df_ = pl.DataFrame({
        'fm':   ['a', 'b', 'a', 'c', 'a', 'd'],
        'to':   ['b', 'a', 'c', 'a', 'd', 'a'],
        'time': [datetime.datetime(2024, 1, _d_, _h_)
                 for _d_, _h_ in ((1, 9), (1, 17), (2, 9), (2, 17), (3, 9), (3, 17))],
    })
    return _df_ if dtype is None else _df_.with_columns(pl.col('time').cast(dtype))


class TestSpreadLinesPTimeOrder(unittest.TestCase):
    """time_order= values that are not strings name the bin they fall in.

    They used to be str()'d, which never matches a polars bin label for a date or
    datetime, so every bin went undrawn (PLANNING.md §5 C-spreadlinesp-anno-key).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def _sp(self, df, **kwargs):
        return self.p2s.spreadlinesp(df, _rels(), ego='a',
                                     time=('time', self.p2s.LT_Y_m_dp), **kwargs)

    def _drawn(self, sp):
        return [sp.bin_to_timestamps[_b_] for _b_ in sorted(sp.bin_to_timestamps)]

    def test_datetimes_give_the_default_bins(self):
        _default_ = self._sp(_make_day_df())
        _ordered_ = self._sp(_make_day_df(), time_order=[datetime.datetime(2024, 1, _d_)
                                                         for _d_ in (1, 2, 3)])
        self.assertEqual(_ordered_.ts_list, _default_.ts_list)
        self.assertEqual(self._drawn(_ordered_), self._drawn(_default_))
        self.assertEqual(len(self._drawn(_ordered_)), 3)

    def test_a_datetime_inside_a_bin_names_that_bin_in_the_given_order(self):
        _sp_ = self._sp(_make_day_df(), time_order=[datetime.datetime(2024, 1, 3, 13),
                                                    datetime.datetime(2024, 1, 1, 2)])
        self.assertEqual(self._drawn(_sp_), ['2024-01-03 00:00:00.000000',
                                             '2024-01-01 00:00:00.000000'])

    def test_values_in_one_bin_collapse_to_one_entry(self):
        _sp_ = self._sp(_make_day_df(), time_order=[datetime.datetime(2024, 1, 1, 1),
                                                    datetime.datetime(2024, 1, 1, 5),
                                                    datetime.datetime(2024, 1, 2)])
        self.assertEqual(_sp_.ts_list, ['2024-01-01 00:00:00.000000',
                                        '2024-01-02 00:00:00.000000'])
        self.assertEqual(len(self._drawn(_sp_)), 2)

    def test_dates_on_a_date_column(self):
        _sp_ = self._sp(_make_day_df(pl.Date), time_order=[datetime.date(2024, 1, 2),
                                                           datetime.date(2024, 1, 1)])
        self.assertEqual(self._drawn(_sp_), ['2024-01-02', '2024-01-01'])

    def test_datetime_on_a_date_column(self):
        _sp_ = self._sp(_make_day_df(pl.Date), time_order=[datetime.datetime(2024, 1, 2, 5)])
        self.assertEqual(self._drawn(_sp_), ['2024-01-02'])

    def test_time_units_other_than_microseconds(self):
        for _unit_, _label_ in (('ms', '2024-01-02 00:00:00.000'),
                                ('ns', '2024-01-02 00:00:00.000000000')):
            with self.subTest(unit=_unit_):
                _sp_ = self._sp(_make_day_df(pl.Datetime(_unit_)),
                                time_order=[datetime.datetime(2024, 1, 2, 12)])
                self.assertEqual(self._drawn(_sp_), [_label_])

    def test_naive_datetime_on_a_tz_aware_column_is_local_time(self):
        # Cast straight to the column's type, midnight would be read as UTC and land in
        # New York's previous day -- a bin that does not exist here, so nothing drawn.
        _df_ = _make_day_df(pl.Datetime('us', 'America/New_York'))
        _sp_ = self._sp(_df_, time_order=[datetime.datetime(2024, 1, 1)])
        self.assertEqual(self._drawn(_sp_), ['2024-01-01 00:00:00.000000-05:00'])

    def test_aware_datetime_is_converted_to_the_column_zone(self):
        # 03:00 UTC on the 2nd is 22:00 on the 1st in New York.
        _df_ = _make_day_df(pl.Datetime('us', 'America/New_York'))
        _sp_ = self._sp(_df_, time_order=[datetime.datetime(2024, 1, 2, 3,
                                                            tzinfo=datetime.UTC)])
        self.assertEqual(self._drawn(_sp_), ['2024-01-01 00:00:00.000000-05:00'])

    def test_labels_are_still_taken_as_given(self):
        _sp_ = self._sp(_make_day_df(), time_order=['2024-01-02 00:00:00.000000'])
        self.assertEqual(self._drawn(_sp_), ['2024-01-02 00:00:00.000000'])

    def test_ints_on_an_int_column(self):
        _sp_ = self.p2s.spreadlinesp(_make_df(), _rels(), ego='a', time='time', time_order=[2, 1])
        self.assertEqual(_sp_.ts_list, ['2', '1'])
        self.assertEqual(self._drawn(_sp_), ['2', '1'])

    def test_ints_on_a_string_column_keep_their_str(self):
        _df_ = _make_df().with_columns(pl.col('time').cast(pl.String))
        _sp_ = self.p2s.spreadlinesp(_df_, _rels(), ego='a', time='time', time_order=[2, 1])
        self.assertEqual(self._drawn(_sp_), ['2', '1'])

    def test_a_float_is_not_truncated_into_an_int_bin(self):
        # Cast without strict, 2.5 would become 2 and draw bin 2.
        _sp_ = self.p2s.spreadlinesp(_make_df(), _rels(), ego='a', time='time',
                                     time_order=[2.5, 1])
        self.assertEqual(_sp_.ts_list, ['2.5', '1'])
        self.assertEqual(self._drawn(_sp_), ['1'])


if __name__ == '__main__':
    unittest.main()
