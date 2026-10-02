"""
Tests for the space between histop's bars and its distribution strip, and for the hit
tests agreeing with the render about which bars were drawn.

The bars were culled against the strip's top edge itself, so the last bar could end on
it, and the "more rows" indicator drawn one v_gap below the last bar landed on the strip
(user report: "the bottom distribution blocks overlap with the chart a little bit ...
when the legend is at the bottom or at the top").  Separately, the three hit tests carried
their own copy of the culling, measured against the whole canvas height instead of the
space above a bottom legend, so a drag over the bars selected bins that were never drawn.
"""
import re
import unittest

import polars as pl

from polars2svg import Polars2SVG

_GAP_ = 3


def _frame_():
    return pl.DataFrame({'k': [f'k{_i_:02d}' for _i_ in range(40) for _ in range(_i_ + 1)]}) \
             .with_columns(v=pl.int_range(pl.len()))


class TestHistopStripGap(unittest.TestCase):

    def setUp(self):
        self.p2s = Polars2SVG()
        self.df  = _frame_()
        self.more_color = self.p2s.colorTyped('indicator', 'more_rows')

    def _cases_(self):
        for _h_ in range(150, 260, 3):
            for _legend_ in (None, 'bottom', 'top'):
                for _v_gap_ in (0, 2):
                    _histop_ = self.p2s.histop(self.df, 'k', color=('v', self.p2s.CMAGNITUDE_MEANp), legend=_legend_,
                                               distribution=True, wxh=(256, _h_), v_gap=_v_gap_)
                    _histop_._repr_svg_()
                    yield (_h_, _legend_, _v_gap_), _histop_

    def _drawn_(self, histop):
        _m_ = re.search(r'\+(\d+)( more)?<', histop.svg)
        return len(histop._sorted_bins_) - (int(_m_.group(1)) if _m_ else 0)

    def test_bars_and_indicator_clear_the_strip(self):
        _checked_ = 0
        for _case_, _h_ in self._cases_():
            if _h_._dist_h_ == 0: continue
            _checked_ += 1
            with self.subTest(case=_case_):
                _drawn_  = self._drawn_(_h_)
                _y_v_    = _h_.v_gap // 2 if _h_.v_gap > 0 else 0
                _bottom_ = _h_._plot_y0_ + (_drawn_ - 1) * _h_._slot_h_ + _y_v_ + _h_.bar_h
                self.assertLessEqual(_bottom_, _h_._dist_strip_y0_ - _GAP_)
                _ind_ = re.search(r'<rect x="[^"]+" y="([^"]+)" width="[^"]+" height="2" fill="' + self.more_color, _h_.svg)
                if _drawn_ < len(_h_._sorted_bins_):
                    self.assertIsNotNone(_ind_)
                    self.assertLessEqual(float(_ind_.group(1)) + 2, _h_._dist_strip_y0_ - _GAP_)
        self.assertGreater(_checked_, 100)

    # A drag over the bars -- stopping above the strip -- selects exactly the drawn bins.
    def test_hit_tests_agree_with_the_render(self):
        for _case_, _h_ in self._cases_():
            with self.subTest(case=_case_):
                _drawn_  = self._drawn_(_h_)
                _y1_     = _h_._dist_strip_y0_ - 1 if _h_._dist_h_ > 0 else _h_.wxh[1]
                self.assertEqual(_h_.filterByRectangle((0, 0, _h_.wxh[0], _y1_))['k'].n_unique(), _drawn_)
                self.assertEqual(_h_.filterByOval((_h_.wxh[0] / 2, 0, _h_.wxh[0], _y1_))['k'].n_unique(), _drawn_)
                # the row just past the last drawn bar has no records
                _y_past_ = _h_._plot_y0_ + _drawn_ * _h_._slot_h_ + 1
                if _y_past_ < _h_._avail_y1_ - _h_._dist_h_:
                    self.assertEqual(len(_h_.recordsAt((10, _y_past_))), 0)


if __name__ == '__main__':
    unittest.main()
