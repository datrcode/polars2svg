import re
import polars as pl
import random

__name__ = 'histop_dataframes'


def makeHistoDf(n=100, seed=42):
    '''Create a DataFrame suitable for histop testing.

    Returns a pl.DataFrame with columns:
      cat     (pl.Utf8)    – random choice of 'A', 'B', 'C'
      group   (pl.Utf8)    – random choice of 'x', 'y'
      value   (pl.Int32)   – random int 1-100
      score   (pl.Float64) – random float 0-10
    '''
    rng = random.Random(seed)
    rows = {'cat': [], 'group': [], 'value': [], 'score': []}
    for _ in range(n):
        rows['cat'].append(rng.choice(['A', 'B', 'C']))
        rows['group'].append(rng.choice(['x', 'y']))
        rows['value'].append(rng.randint(1, 100))
        rows['score'].append(round(rng.uniform(0.0, 10.0), 3))
    return pl.DataFrame(rows)


def makeOrderedHistoDf():
    '''DataFrame with known per-bin row counts: A=5, B=3, C=1.

    Useful for asserting _sorted_bins_ ordering.
    '''
    return pl.DataFrame({
        'cat':   ['A'] * 5 + ['B'] * 3 + ['C'] * 1,
        'group': ['x'] * 5 + ['y'] * 3 + ['x'] * 1,
        'value': [10, 20, 30, 40, 50, 5, 15, 25, 7],
        'score': [1.0, 2.0, 3.0, 4.0, 5.0, 0.5, 1.5, 2.5, 0.7],
    })


#
# makeStatisticHistoDf() -- four bins on which every order key ranks the bins differently,
# with no ties inside any key: row count D A C B, sum D A B C, min B A D C, max B D A C,
# mean D B A C, median D B C A, std B D C A, distinct groups A D C B.  So an order=
# that fell back to any other key -- the row count above all -- cannot pass.  `score`
# is `value` as a float.  test_histop_order.py's
# test_the_statistic_frame_separates_every_order checks the property rather than
# trusting this comment.
#
def makeStatisticHistoDf() -> pl.DataFrame:
    _rows_ = {'A': ([15, 7, 6, 9],            ['p', 'q', 'r', 't']),
              'B': ([7, 20],                  ['r', 'r']),
              'C': ([13, 3, 9],               ['t', 'q', 't']),
              'D': ([15, 15, 18, 12, 19, 5],  ['q', 'r', 't', 'r', 'r', 'q'])}
    return pl.DataFrame({'cat':   [b for b, (v, _) in _rows_.items() for _ in v],
                         'value': [x for v, _ in _rows_.values() for x in v],
                         'group': [g for _, gs in _rows_.values() for g in gs]}
                        ).with_columns(pl.col('value').cast(pl.Float64).alias('score'))

# ─────────────────────────────────────────────────────────────────────────────
# Reading a rendered histop back (PLANNING.md V11): what the bars say, so a test
# can compare it with what polars computes from the same frame.
# ─────────────────────────────────────────────────────────────────────────────
_BAR_RECT_RE_ = re.compile(r'<rect x="([\d.]+)" y="([\d.]+)" width="([\d.]+)" height="([\d.]+)" fill="(#[0-9a-fA-F]{6})" stroke="none" />')


def _binLabels_(h, svg: str) -> list:
    return [t for _, t in sorted((float(y), t) for y, t in
            re.findall(rf'<text x="[\d.]+" text-anchor="start" y="([\d.]+)"[^>]*font-size="{h.txt_h}px">([^<]*)</text>', svg))]


#
# histopBars() -- the drawn bars, top to bottom, as (label, x, y, width): every
# stroke-less filled rect one bar_h tall, paired with the bin labels by row.
#
def histopBars(h) -> list:
    _svg_  = h._repr_svg_()
    _bars_ = sorted((float(y), float(x), float(w)) for x, y, w, hh in
                    re.findall(r'<rect x="([\d.]+)" y="([\d.]+)" width="([\d.]+)" height="([\d.]+)" fill="#[0-9a-fA-F]{6}" stroke="none" />', _svg_)
                    if float(hh) == float(h.bar_h))
    _labels_ = _binLabels_(h, _svg_)
    if len(_labels_) != len(_bars_): _labels_ = [None] * len(_bars_)
    return [(_lbl_, x, y, w) for _lbl_, (y, x, w) in zip(_labels_, _bars_)]


#
# histopRows() -- the drawn rows, top to bottom, as (label, [(x, width, fill), ...]):
# one segment for a simple bar, one per colour for a stacked one, left to right.
#
def histopRows(h) -> list:
    _svg_  = h._repr_svg_()
    _rows_: dict = {}
    for x, y, w, hh, fill in _BAR_RECT_RE_.findall(_svg_):
        if float(hh) == float(h.bar_h): _rows_.setdefault(float(y), []).append((float(x), float(w), fill.lower()))
    _labels_ = _binLabels_(h, _svg_)
    _ys_     = sorted(_rows_)
    if len(_labels_) != len(_ys_): _labels_ = [None] * len(_ys_)
    return [(_lbl_, sorted(_rows_[y])) for _lbl_, y in zip(_labels_, _ys_)]


# Bins ordered as histop orders them: by the metric, largest first by default, ties A to Z
def orderedBins(metric: dict, descending: bool = True) -> list:
    return sorted(metric, key=lambda k: ((-metric[k] if descending else metric[k]), k))


def _rgb_(hx: str) -> tuple: return tuple(int(hx[i:i + 2], 16) for i in (1, 3, 5))


class HistopAssertions:
    '''Assertions over a rendered histop, for a unittest.TestCase to mix in.'''

    # The rows are the bins in `order`, each as long as its value in `lengths` on the plot's
    # scale: the largest spans the plot (the axis starts at 0).
    def assertBarsShow(self, h, lengths: dict, order: list) -> None:
        _rows_ = histopRows(h)
        lengths, order = {str(k): v for k, v in lengths.items()}, [str(k) for k in order]   # bars are read by their label text
        self.assertEqual([_lbl_ for _lbl_, _ in _rows_], order, 'the bars are not in the expected order')
        _top_ = max(lengths.values())
        for _lbl_, _segs_ in _rows_:
            # segment edges are written to 0.1 px, so a length is good to two half-steps
            self.assertAlmostEqual(sum(w for _, w, _ in _segs_), h._plot_w_ * lengths[_lbl_] / _top_, delta=0.11,
                                   msg=f'bar {_lbl_!r} is not as long as its {lengths[_lbl_]}')

    # Every bar is one segment, filled with the colour `colors` gives its bin (to within a
    # channel step, so a spectrum colour may round either way)
    def assertBarColors(self, h, colors: dict) -> None:
        colors = {str(k): v for k, v in colors.items()}
        for _lbl_, _segs_ in histopRows(h):
            self.assertEqual(len(_segs_), 1, f'bar {_lbl_!r} should be one segment')
            _got_, _want_ = _rgb_(_segs_[0][2]), _rgb_(colors[_lbl_].lower())
            self.assertTrue(all(abs(a - b) <= 1 for a, b in zip(_got_, _want_)),
                            f'bar {_lbl_!r} is {_segs_[0][2]}, expected {colors[_lbl_]}')

    # Stacked bars: each bin's segments are its colour values' shares -- one segment per
    # value, filled with that value's colour, as long as its amount on the plot's scale
    # (the longest bar's total spans the plot).  `parts` maps (bin, colour value) -> amount.
    def assertSegmentsShow(self, h, parts: dict) -> None:
        _totals_: dict = {}
        for (_b_, _), _amt_ in parts.items(): _totals_[_b_] = _totals_.get(_b_, 0) + _amt_
        _scale_  = h._plot_w_ / max(_totals_.values())
        _colors_ = h.p2s.colors(sorted({str(v) for _, v in parts}))
        for _lbl_, _segs_ in histopRows(h):
            _want_ = sorted((_colors_[str(v)].lower(), _amt_ * _scale_) for (b, v), _amt_ in parts.items() if str(b) == _lbl_ and _amt_ > 0)
            _got_  = sorted((fill, w) for _, w, fill in _segs_)
            self.assertEqual([f for f, _ in _got_], [f for f, _ in _want_], f'bar {_lbl_!r} has the wrong colours')
            for (_, a), (_, b) in zip(_got_, _want_):
                self.assertAlmostEqual(a, b, delta=0.11, msg=f'a segment of bar {_lbl_!r} is the wrong length')


# The spectrum colour histop gives each bin: its statistic scaled between the smallest
# and largest over the bins, through the framework's colour ramp.  The ramp is taken as
# given; what a test checks with this is which statistic and which scaling each bar got.
def spectrumColors(p2s, stats: dict) -> dict:
    _lo_, _hi_ = min(stats.values()), max(stats.values())
    _span_     = max(_hi_ - _lo_, 1e-9)
    _keys_     = list(stats)
    _df_ = (pl.DataFrame({'n': [min(max((stats[k] - _lo_) / _span_, 0.0), 1.0) for k in _keys_]})
            .with_columns(p2s.colorSpectrumPolarsOperations('n', 'r', 'g', 'b'))
            .with_columns(p2s.hexColorFromRGBTriplesPolarsOperations('r', 'g', 'b').alias('hx')))
    return dict(zip(_keys_, _df_['hx'].to_list()))
