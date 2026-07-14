import re
import unittest
import polars as pl
import datetime
import random
from polars2svg import Polars2SVG

class Testp2s_render(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

        self._df_ = pl.DataFrame({
            'cat':['a',   'b',   'a',   'b',   'c',    'd'],
            'qty':[1.0,   2.0,   3.0,   4.0,   1.0,    1.0],
            'clr':['red', 'red', 'red', 'red', 'blue', 'blue'],
        })

        self._count_options_ = [self.p2s.ROW_COUNTp,
                                'qty',
                                ('qty',),
                                ('qty', self.p2s.SCALARp),
                                ('qty', self.p2s.SETp),
                                'cat',
                                'clr',
                                ('cat','clr'),
                                ('cat','clr', self.p2s.SETp),
                                ('cat','clr','qty')]

        self._color_options_ = [None,
                                'clr',
                                'cat',
                                ('clr', 'cat'),
                                ('clr', self.p2s.ROW_COUNTp),
                                ('clr', 'cat', self.p2s.ROW_COUNTp),
                                'qty']

    # Reference implementation: intentionally simple/slow, used to check the
    # framework's vectorized bar computations against.
    def barHeight(self, count, h_chart, count_max):
        _strs_ = []
        if isinstance(count, tuple): _strs_ = [_str_ for _str_ in count if isinstance(_str_, str)]
        if   count == self.p2s.ROW_COUNTp:                                                                        _count_bar_ = len(self._df_)
        elif count == 'qty':                                                                                      _count_bar_ = self._df_['qty'].sum()
        elif isinstance(count, tuple) and len(count) == 1 and count[0] == 'qty':                                  _count_bar_ = self._df_['qty'].sum()
        elif isinstance(count, tuple) and len(count) == 2 and count[0] == 'qty' and count[1] == self.p2s.SCALARp: _count_bar_ = self._df_['qty'].sum()
        elif count == ('qty', self.p2s.SETp):                                                                     _count_bar_ = self._df_['qty'].n_unique()
        elif isinstance(count, str):                                                                              _count_bar_ = self._df_[count].n_unique()
        else:                                                                                                     _count_bar_ = self._df_[_strs_].n_unique()
        return h_chart * (_count_bar_ / count_max)


    def test_basic_bar(self):
        _svg_ = ['<svg x="0" y="0" width="780" height="1280">',
                '<rect x="0" y="0" width="780" height="1280" fill="white"/>',]
        _x_         = 5    # base of the bar
        _y_         = 1    # base of the bar
        _h_chart_   = 502  # height of the chart
        _count_max_ = 24   # for the dataframe, maximum count
        _w_bar_     = 16   # width of the bar
        # Combinations of count and color
        for _color_ in self._color_options_:
            for _count_ in self._count_options_:
                _h_bar_ = self.barHeight(_count_, _h_chart_, _count_max_)
                _svg_.append(self.p2s.colorizeBar(self._df_, (_x_, _y_, _w_bar_, _h_bar_), _count_, _color_))
                _svg_.append(f'<rect x="{_x_}" y="{_y_}" width="{_h_bar_}" height="{_w_bar_}" fill="none" stroke="black" stroke-width="0.5" />')
                _svg_.append(self.p2s.svgText(f'count = "{_count_} | color = "{_color_}"', _x_ + 256, _y_ + 11, txt_h=11))
                _y_ += (_w_bar_+2)
        _svg_.append('</svg>')

    def test_vertical_bar(self):
        _svg_ = ['<svg x="0" y="0" width="1280" height="780">',
                '<rect x="0" y="0" width="1280" height="780" fill="white"/>',]
        _x_         = 1       # base of the bar
        _y_         = 780 - 5 # base of the bar
        _h_chart_   = 502     # height of the chart
        _count_max_ = 24      # for the dataframe, maximum count
        _w_bar_     = 12      # width of the bar
        # Combinations of count and color
        for _color_ in self._color_options_:
            for _count_ in self._count_options_:
                _h_bar_ = self.barHeight(_count_, _h_chart_, _count_max_)
                _svg_.append(self.p2s.colorizeBar(self._df_, (_x_, _y_, _w_bar_, _h_bar_), _count_, _color_, orientation='vertical'))
                _svg_.append(f'<rect x="{_x_}" y="{_y_-_h_bar_}" width="{_w_bar_}" height="{_h_bar_}" fill="none" stroke="black" stroke-width="0.5" />')
                _svg_.append(self.p2s.svgText(f'count = "{_count_} | color = "{_color_}"', _x_ + _w_bar_ - 2, _y_ - 256, txt_h=11, rotation=270))
                _x_ += (_w_bar_+4)
        _svg_.append('</svg>')

    def test_ordering(self):
        _lu_ = {'i':[], 'num':[], 'x':[]}
        for i in range(32):
            _num_, _x_ = abs(i-16.0)*0.12, 'red'
            _lu_['i'].append(i), _lu_['num'].append(_num_), _lu_['x'].append(_x_)
            _num_, _x_ = abs(i-16.0)*0.2, 'blue'
            _lu_['i'].append(i), _lu_['num'].append(_num_), _lu_['x'].append(_x_)
            _num_, _x_ = (1 + i%4)*0.3, 'green'
            _lu_['i'].append(i), _lu_['num'].append(_num_), _lu_['x'].append(_x_)
        df = pl.DataFrame(_lu_)
        _svg_ = ['<svg x="0" y="0" width="600" height="400">',
                '<rect x="0" y="0" width="600" height="400" fill="white"/>',]
        _x_           = 1       # base of the bar
        _y_           = 400 - 5 # base of the bar
        _h_chart_     = 384     # height of the chart
        _count_max_   = df.group_by('i').agg(pl.col('num').sum())['num'].max()
        _w_bar_       = 12      # width of the bar
        _color_order_ = self.p2s.colorizeOrder(df, 'num', 'x')
        for i in range(33):
            _df_    = df.filter(pl.col('i') == i)
            _h_bar_ = _h_chart_ * _df_['num'].sum() / _count_max_
            _svg_.append(self.p2s.colorizeBar(_df_, (_x_, _y_, _w_bar_, _h_bar_), 'num', 'x', orientation='vertical', color_order=_color_order_))
            _x_ += (_w_bar_+2)
        _svg_.append('</svg>')


    # ── colorizeAllBarsVertical vs per-bar colorizeBar ─────────────────────

    def _parse_rects(self, svg):
        """Parse <rect .../> elements; return list of attr dicts with floats rounded to 2dp, sorted by (x, y, fill)."""
        rects = []
        for m in re.finditer(r'<rect\s([^/]*)/>', svg):
            attrs = {}
            for a in re.finditer(r'(\w+)="([^"]*)"', m.group(1)):
                k, v = a.group(1), a.group(2)
                try:    attrs[k] = round(float(v), 2)
                except: attrs[k] = v
            rects.append(attrs)
        return sorted(rects, key=lambda r: (r.get('x', 0), r.get('y', 0), r.get('fill', '')))

    def _segments_by_bar(self, svg):
        """Group rects by x position; return dict mapping x → sorted list of (height, fill).
        Used when stacking order may differ but segment content should be identical."""
        from collections import defaultdict
        by_x = defaultdict(list)
        for r in self._parse_rects(svg):
            by_x[r.get('x', 0)].append((r.get('height', 0), r.get('fill', '')))
        return {x: sorted(segs) for x, segs in by_x.items()}

    def _loop_colorize(self, df_agg, bin_col, sorted_bins,
                       plot_x0, bar_w_raw, bar_w, plot_y1, plot_h,
                       count_min, count_max, color_col,
                       color_order=None, remainder_threshold=3.0):
        """Reference: call colorizeBar once per bin."""
        span = max(float(count_max) - float(count_min), 1.0)
        def count_to_bar_h(total):
            return max(0.0, plot_h * (float(total) - float(count_min)) / span)
        parts = []
        for i, bk in enumerate(sorted_bins):
            df_bin = df_agg.filter(pl.col(bin_col) == bk)
            bh = count_to_bar_h(float(df_bin['__count__'].sum()))
            if bh > 0:
                parts.append(self.p2s.colorizeBar(
                    df_bin, (plot_x0 + i * bar_w_raw, plot_y1, bar_w, bh),
                    '__count__', color_col,
                    color_order=color_order, orientation='vertical',
                    remainder_threshold=remainder_threshold
                ))
        return ''.join(parts)

    def _batch_colorize(self, df_agg, bin_col, sorted_bins,
                        plot_x0, bar_w_raw, bar_w, plot_y1, plot_h,
                        count_min, count_max, color_col,
                        color_order=None, remainder_threshold=3.0):
        """New batch path: colorizeAllBarsVertical."""
        xs        = [float(plot_x0 + i * bar_w_raw) for i in range(len(sorted_bins))]
        x_lookup  = pl.DataFrame({bin_col: sorted_bins, '__x__': xs})
        return self.p2s.colorizeAllBarsVertical(
            df_agg, bin_col, x_lookup, plot_y1, bar_w,
            plot_h, count_min, count_max, color_col,
            color_order=color_order, remainder_threshold=remainder_threshold
        )

    def test_all_bars_vertical_matches_per_bar(self):
        """colorizeAllBarsVertical must produce identical rects to per-bar colorizeBar calls."""
        df_agg = pl.DataFrame({
            'bin':       [1, 1, 1, 2, 2, 2, 3, 3, 3],
            'color':     ['red', 'blue', 'green'] * 3,
            '__count__': [10,    5,      3,      8, 12, 2,  6,  9,  7],
        })
        sorted_bins = [1, 2, 3]
        count_max   = df_agg.group_by('bin').agg(pl.col('__count__').sum())['__count__'].max()
        color_order = self.p2s.colorizeOrder(df_agg, '__count__', 'color')
        svg_loop  = self._loop_colorize( df_agg, 'bin', sorted_bins, 10, 20, 18, 500, 400, 0, count_max, 'color', color_order=color_order)
        svg_batch = self._batch_colorize(df_agg, 'bin', sorted_bins, 10, 20, 18, 500, 400, 0, count_max, 'color', color_order=color_order)
        self.assertEqual(self._parse_rects(svg_loop), self._parse_rects(svg_batch))

    def test_all_bars_vertical_no_color_order(self):
        """Without color_order the two methods sort segments differently (per-bar local sort vs
        global rank), so y-positions may differ.  The segment heights and fills per bar must
        still be identical — only the stacking order is allowed to differ."""
        df_agg = pl.DataFrame({
            'bin':       [1, 1, 1, 2, 2, 2, 3, 3, 3],
            'color':     ['red', 'blue', 'green'] * 3,
            '__count__': [10,    5,      3,      8, 12, 2,  6,  9,  7],
        })
        sorted_bins = [1, 2, 3]
        count_max   = df_agg.group_by('bin').agg(pl.col('__count__').sum())['__count__'].max()
        svg_loop  = self._loop_colorize( df_agg, 'bin', sorted_bins, 10, 20, 18, 500, 400, 0, count_max, 'color')
        svg_batch = self._batch_colorize(df_agg, 'bin', sorted_bins, 10, 20, 18, 500, 400, 0, count_max, 'color')
        self.assertEqual(self._segments_by_bar(svg_loop), self._segments_by_bar(svg_batch))

    def test_all_bars_vertical_missing_colors_in_some_bins(self):
        """Bins that lack certain color categories must still match the per-bar approach."""
        df_agg = pl.DataFrame({
            'bin':       [1, 1, 1, 2, 2, 3],
            'color':     ['red', 'blue', 'green', 'red', 'blue', 'red'],
            '__count__': [10,    5,      3,       8,     12,     6],
        })
        sorted_bins = [1, 2, 3]
        count_max   = df_agg.group_by('bin').agg(pl.col('__count__').sum())['__count__'].max()
        color_order = self.p2s.colorizeOrder(df_agg, '__count__', 'color')
        svg_loop  = self._loop_colorize( df_agg, 'bin', sorted_bins, 5, 22, 20, 400, 350, 0, count_max, 'color', color_order=color_order)
        svg_batch = self._batch_colorize(df_agg, 'bin', sorted_bins, 5, 22, 20, 400, 350, 0, count_max, 'color', color_order=color_order)
        self.assertEqual(self._parse_rects(svg_loop), self._parse_rects(svg_batch))

    def test_all_bars_vertical_remainder_aggregation(self):
        """Below-threshold segments must be aggregated identically per bin in both methods.
        With plot_h=400 and count_max=20, a segment with count=0.1 has
        h_in_px = (0.1/20.0)*400 = 2.0 px, which is below the default threshold of 3.0."""
        df_agg = pl.DataFrame({
            'bin':       [1, 1, 1, 2, 2, 2],
            'color':     ['red', 'blue', 'tiny', 'red', 'blue', 'tiny'],
            '__count__': [9.9,  10.0,   0.1,    7.9,  11.9,   0.1],
        })
        sorted_bins = [1, 2]
        count_max   = 20.0
        color_order = self.p2s.colorizeOrder(df_agg, '__count__', 'color')
        svg_loop  = self._loop_colorize( df_agg, 'bin', sorted_bins, 10, 20, 18, 500, 400, 0, count_max, 'color', color_order=color_order, remainder_threshold=3.0)
        svg_batch = self._batch_colorize(df_agg, 'bin', sorted_bins, 10, 20, 18, 500, 400, 0, count_max, 'color', color_order=color_order, remainder_threshold=3.0)
        self.assertEqual(self._parse_rects(svg_loop), self._parse_rects(svg_batch))


if __name__ == '__main__':
    unittest.main()
