'''histop takes a t-field as bin_by (PLANNING.md §5 C-histop-tfield).

A t-field bins a timestamp column by a time level -- day of week, hour, month -- the way
xyp, timep and smallp already accept.  histop keeps ranking by count (D3); its bars are
labelled human-readably (D4): 'mon', not 1, alone or inside a tuple bin.
'''
import asyncio
import datetime as dt
import logging
import re
import unittest

import polars as pl

from polars2svg import Polars2SVG

try:
    import panel  # noqa: F401
    _PANEL_AVAILABLE_ = True
except ImportError:
    _PANEL_AVAILABLE_ = False

_DAYS_ = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']


# 2026-01-05 is a Monday.  Day i of that week gets 7-i rows (mon=7 ... sun=1), so the
# count order is fixed with no ties; 'cat' alternates so tuple bins exist.
def _frame_() -> pl.DataFrame:
    _rows_ = [(dt.datetime(2026, 1, 5 + _i_, (5 * _k_) % 24), 'ab'[_k_ % 2])
              for _i_ in range(7) for _k_ in range(7 - _i_)]
    return pl.DataFrame({'ts': [_r_[0] for _r_ in _rows_], 'cat': [_r_[1] for _r_ in _rows_]})


def _bar_labels_(svg: str, n: int) -> list[str]:
    '''The first n <text> elements: histop draws its bar labels before the axis text.'''
    return re.findall(r'<text[^>]*>([^<]*)</text>', svg)[:n]


class TestHistopTFieldBin(unittest.TestCase):
    def setUp(self) -> None:
        self.p2s = Polars2SVG()
        self.df  = _frame_()

    def test_day_of_week_bars_rank_by_count_and_read_as_days(self):
        _hp_ = self.p2s.histop(self.df, self.p2s.tField('ts', self.p2s.PT_DoWp), wxh=(160, 200))
        self.assertEqual(_bar_labels_(_hp_.svg, 7), _DAYS_)          # mon=7 ... sun=1: count order

    def test_label_order_is_time_order(self):
        _df_ = self.df.filter(pl.col('ts').dt.weekday() != 1)          # drop monday so count order != time order
        _hp_ = self.p2s.histop(_df_, self.p2s.tField('ts', self.p2s.PT_DoWp), order=self.p2s.LABELp,
                               descending=False, wxh=(160, 200))
        self.assertEqual(_bar_labels_(_hp_.svg, 6), _DAYS_[1:])

    def test_hour_and_month_labels(self):
        _hours_ = self.p2s.histop(self.df, self.p2s.tField('ts', self.p2s.PT_Hp), order=self.p2s.LABELp,
                                  descending=False, wxh=(160, 400))
        _want_  = sorted({f'{_h_:02}h' for _h_ in self.df['ts'].dt.hour().to_list()})
        self.assertEqual(_bar_labels_(_hours_.svg, len(_want_)), _want_)
        _months_ = self.p2s.histop(self.df, self.p2s.tField('ts', self.p2s.LT_Y_mp), wxh=(160, 100))
        self.assertEqual(_bar_labels_(_months_.svg, 1), ['2026-01'])

    def test_bar_lengths_are_the_day_counts(self):
        _hp_  = self.p2s.histop(self.df, self.p2s.tField('ts', self.p2s.PT_DoWp), wxh=(160, 200))
        _agg_ = dict(zip(_hp_.df_agg[_hp_._bin_col_].to_list(), _hp_.df_agg['__count__'].to_list()))
        self.assertEqual(_agg_, {_d_: 7 - (_d_ - 1) for _d_ in range(1, 8)})

    def test_legacy_string_spelling_draws_the_same_bars(self):
        _a_ = self.p2s.histop(self.df, self.p2s.tField('ts', self.p2s.PT_DoWp), wxh=(160, 200))
        _b_ = self.p2s.histop(self.df, 'ts|DoWp', wxh=(160, 200))
        self.assertEqual(_bar_labels_(_a_.svg, 7), _bar_labels_(_b_.svg, 7))

    def test_a_date_column_is_accepted(self):
        _df_ = self.df.with_columns(pl.col('ts').dt.date())
        _hp_ = self.p2s.histop(_df_, self.p2s.tField('ts', self.p2s.PT_DoWp), wxh=(160, 200))
        self.assertEqual(_bar_labels_(_hp_.svg, 7), _DAYS_)

    def test_tuple_bin_reads_the_day_not_its_number(self):
        _hp_     = self.p2s.histop(self.df, ('cat', self.p2s.tField('ts', self.p2s.PT_DoWp)), wxh=(160, 400))
        _labels_ = set(_bar_labels_(_hp_.svg, _hp_.df_agg.height))
        _want_   = {f'{_c_}|{_DAYS_[_r_["ts"].weekday()]}' for _r_ in self.df.iter_rows(named=True) for _c_ in [_r_['cat']]}
        self.assertEqual(_labels_, _want_)

    def test_null_timestamp_is_its_own_bar(self):
        _df_ = pl.concat([self.df, pl.DataFrame({'ts': [None], 'cat': ['a']}, schema=self.df.schema)])
        _hp_ = self.p2s.histop(_df_, self.p2s.tField('ts', self.p2s.PT_DoWp), wxh=(160, 220))
        self.assertIn('(null)', _bar_labels_(_hp_.svg, 8))

    def test_wrong_dtype_and_missing_column_are_named(self):
        with self.assertRaisesRegex(ValueError, 'needs a Date or Datetime column'):
            self.p2s.histop(self.df.with_columns(pl.col('ts').cast(pl.String)), self.p2s.tField('ts', self.p2s.PT_DoWp)).svg
        with self.assertRaisesRegex(ValueError, 'column "tz" not found'):
            self.p2s.histop(self.df, self.p2s.tField('tz', self.p2s.PT_DoWp)).svg


class TestHistopTFieldCountAndColor(unittest.TestCase):
    '''A t-field is a time bin, never a magnitude: count= counts its distinct values (D2)
    and color= colours by it categorically (D1), with a readable legend (D4).  Periodic
    values are Int64, so the dtype rule used to sum them and draw a colorbar.'''
    def setUp(self) -> None:
        self.p2s = Polars2SVG()
        self.df  = _frame_()
        self.dow = self.p2s.tField('ts', self.p2s.PT_DoWp)
        self.hr  = self.p2s.tField('ts', self.p2s.PT_Hp)

    def _counts_(self, hp) -> dict:
        return dict(zip(hp.df_agg[hp._bin_col_].to_list(), hp.df_agg['__count__'].to_list()))

    def test_count_by_a_tfield_counts_distinct_values(self):
        _want_ = dict(self.df.group_by('cat').agg(pl.col('ts').dt.hour().n_unique()).iter_rows())
        for _count_ in (self.hr, (self.hr,)):
            with self.subTest(count=_count_):
                self.assertEqual(self._counts_(self.p2s.histop(self.df, 'cat', count=_count_)), _want_)

    def test_scalarp_still_sums_a_tfield(self):
        _want_ = dict(self.df.group_by('cat').agg(pl.col('ts').dt.hour().sum()).iter_rows())
        _hp_   = self.p2s.histop(self.df, 'cat', count=(self.hr, self.p2s.SCALARp))
        self.assertEqual(self._counts_(_hp_), _want_)

    def test_a_tfield_count_is_no_boxplot_field_unless_scalar(self):
        _hp_ = self.p2s.histop(self.df, 'cat', count=self.hr)
        self.assertIsNone(_hp_.numericCountField())
        self.assertEqual(_hp_.numericCountField((self.hr, self.p2s.SCALARp)), self.hr)

    def test_color_by_a_tfield_is_categorical_with_day_names(self):
        for _color_ in (self.dow, (self.dow,)):
            with self.subTest(color=_color_):
                _hp_ = self.p2s.histop(self.df, 'cat', color=_color_, legend=True, wxh=(300, 200))
                self.assertEqual(_hp_.legend_info.kind, 'categorical')
                self.assertEqual({_e_[0] for _e_ in _hp_.legend_info.entries}, set(_DAYS_))
                # the swatches and the stacked segments are one colour per day
                _fills_ = set(re.findall(r'fill="(#[0-9a-f]{6})"', _hp_.svg))
                self.assertTrue({_e_[1] for _e_ in _hp_.legend_info.entries} <= _fills_)

    def test_a_linear_tfield_legend_reads_as_dates(self):
        _hp_ = self.p2s.histop(self.df, 'cat', color=self.p2s.tField('ts', self.p2s.LT_Y_m_dp), legend=True, wxh=(300, 220))
        self.assertEqual({_e_[0] for _e_ in _hp_.legend_info.entries},
                         {f'2026-01-{_d_:02}' for _d_ in range(5, 12)})

    def test_an_explicit_magnitude_enum_still_wins(self):
        _hp_ = self.p2s.histop(self.df, 'cat', color=(self.dow, self.p2s.CSTRETCHED_SUMp), legend=True, wxh=(300, 200))
        self.assertEqual(_hp_.legend_info.kind, 'colorbar')

    def test_color_and_bin_by_the_same_tfield(self):
        _hp_ = self.p2s.histop(self.df, self.dow, color=self.dow, legend=True, wxh=(300, 220))
        self.assertEqual(_bar_labels_(_hp_.svg, 7), _DAYS_)
        self.assertEqual({_e_[0] for _e_ in _hp_.legend_info.entries}, set(_DAYS_))


class TestHistopTFieldRecords(unittest.TestCase):
    '''Records go back with the caller's columns, not the derived t-field ones.'''
    def setUp(self) -> None:
        self.p2s = Polars2SVG()
        self.df  = _frame_()

    def test_search_matches_the_label_and_returns_the_input_schema(self):
        _hp_  = self.p2s.histop(self.df, self.p2s.tField('ts', self.p2s.PT_DoWp))
        _got_ = _hp_.filterBySubstring('mon')
        self.assertEqual(_got_.height, self.df.filter(pl.col('ts').dt.weekday() == 1).height)
        self.assertEqual(_got_.columns, self.df.columns)

    def test_count_and_color_tfields_no_longer_leak_their_columns(self):
        for _kw_ in ({'count': (self.p2s.tField('ts', self.p2s.PT_Hp), self.p2s.SETp)},
                     {'color': (self.p2s.tField('ts', self.p2s.PT_DoWp), self.p2s.CSETp)}):
            with self.subTest(kw=list(_kw_)):
                _got_ = self.p2s.histop(self.df, 'cat', **_kw_).filterBySubstring('a')
                self.assertEqual(_got_.columns, self.df.columns)
                self.assertGreater(_got_.height, 0)

    def test_records_feed_a_second_histop_without_a_shadow_warning(self):
        _tf_   = self.p2s.tField('ts', self.p2s.PT_DoWp)
        _recs_ = self.p2s.histop(self.df, _tf_).filterBySubstring('mon')
        with self.assertNoLogs(self.p2s.logger, level='WARNING'):
            _hp_ = self.p2s.histop(_recs_, _tf_)
            self.assertEqual(_bar_labels_(_hp_.svg, 1), ['mon'])



@unittest.skipUnless(_PANEL_AVAILABLE_, 'panel not installed')
class TestHistopiTFieldBin(unittest.TestCase):
    '''histopi binned by a t-field, driven the way the browser drives it: a drag writes
    the drag params and the view pushes the selected records as a new stack frame; a
    hover writes the tooltip params.  The frame's records must carry the caller's
    columns only, or the drill-down render reads its own derived column as a real one.'''
    def setUp(self) -> None:
        self.p2s = Polars2SVG()
        self.df  = _frame_()
        self.dow = self.p2s.tField('ts', self.p2s.PT_DoWp)
        self.view = self.p2s.histopi(self.p2s.histop(self.df, self.dow, color=self.p2s.tField('ts', self.p2s.PT_Hp),
                                                     wxh=(256, 256)))

    def _drag_top_bars_(self, n: int) -> None:
        _plot_ = self.view._plot_
        async def _go_() -> None:
            self.view.drag_x0, self.view.drag_y0 = 5, int(_plot_._plot_y0_ + 1)
            self.view.drag_x1, self.view.drag_y1 = 250, int(_plot_._plot_y0_ + n * _plot_._slot_h_ - 1)
            self.view.shiftkey = False
            self.view.drag_op_finished = True
            for _ in range(500):
                await asyncio.sleep(0.01)
                if len(self.view.mvc.stacks['default']['dfs']) > 1: return
        asyncio.run(_go_())

    def test_a_drag_drills_down_to_those_days_with_the_input_columns(self):
        with self.assertNoLogs(self.p2s.logger, level=logging.WARNING):
            self._drag_top_bars_(2)                                    # mon (7 rows), tue (6)
        _frame_df_ = self.view.mvc.stacks['default']['dfs'][-1]
        self.assertEqual(_frame_df_.columns, self.df.columns)
        self.assertEqual(_frame_df_.height, 13)
        self.assertEqual(_bar_labels_(self.view._plot_.svg, 2), ['mon', 'tue'])

    def test_the_tooltip_names_the_bar_s_records(self):
        self.view.tooltip = 'text'
        async def _go_() -> dict:
            self.view.tooltip_x, self.view.tooltip_y = 60, int(self.view._plot_._plot_y0_ + 3)
            self.view.tooltip_probe = False
            self.view.tooltip_seq = 1
            for _ in range(500):
                await asyncio.sleep(0.01)
                if self.view.tooltip_payload.get('seq') == 1: break
            return self.view.tooltip_payload
        self.assertEqual(asyncio.run(_go_())['lines'][0], '7 records')     # monday's bar


if __name__ == '__main__':
    unittest.main()
