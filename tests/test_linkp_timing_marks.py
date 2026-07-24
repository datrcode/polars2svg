import datetime
import math
import re
import unittest

import polars as pl
from polars2svg import Polars2SVG


# A <line ... stroke-width="1.5" /> is a timing mark (links carry stroke-width on the
# enclosing <g>, never on the element, so 1.5 is unique to marks).
_MARK_RE_ = re.compile(
    r'<line x1="([-\d.]+)" y1="([-\d.]+)" x2="([-\d.]+)" y2="([-\d.]+)" '
    r'stroke="(#[0-9a-fA-F]+)" stroke-width="1.5" />'
)


def _marks(svg):
    return [(float(a), float(b), float(c), float(d), e) for a, b, c, d, e in _MARK_RE_.findall(svg)]


def _bidir_df():
    # a<->b and c<->d, three distinct timestamps per directed edge
    base = datetime.datetime(2024, 1, 1, 8, 0, 0)
    rows = []
    for k, (fm, to) in enumerate([('a', 'b'), ('b', 'a'), ('c', 'd'), ('d', 'c')]):
        for j in range(3):
            rows.append({'fm': fm, 'to': to, 'ts': base + datetime.timedelta(hours=6 * k + j)})
    return pl.DataFrame(rows)


_POS_ = {'a': (0.0, 0.0), 'b': (1.0, 0.3), 'c': (0.2, 1.0), 'd': (1.0, 1.0)}
_REL_ = [('fm', 'to')]


class TestLinkPTimingMarks(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def tearDown(self):
        self.p2s.reset_defaults()

    # ── enable / disable ───────────────────────────────────────────────────────
    def test_no_marks_without_time(self):
        lp = self.p2s.linkp(_bidir_df(), relationships=_REL_, pos=_POS_)
        self.assertEqual(len(_marks(lp.svg)), 0)

    def test_time_none_svg_unchanged(self):
        df = _bidir_df()
        a = self.p2s.linkp(df, relationships=_REL_, pos=_POS_).svg
        b = self.p2s.linkp(df, relationships=_REL_, pos=_POS_, time=None).svg
        self.assertEqual(a, b)

    # ── all three shapes render one mark per (directed edge, distinct time) ─────
    def test_marks_render_each_shape(self):
        df = _bidir_df()
        for shape in ('line', 'curve', 'flowmap'):
            lp = self.p2s.linkp(df, relationships=_REL_, pos=_POS_, link_shape=shape, time='ts')
            # 4 directed edges * 3 distinct timestamps = 12 unique marks
            self.assertEqual(len(_marks(lp.svg)), 12, f'shape={shape}')

    def test_duplicate_events_collapse(self):
        # two identical rows on the same edge/time -> a single mark
        base = datetime.datetime(2024, 1, 1)
        df = pl.DataFrame([{'fm': 'a', 'to': 'b', 'ts': base}] * 3)
        lp = self.p2s.linkp(df, relationships=_REL_, pos=_POS_, time='ts')
        self.assertEqual(len(_marks(lp.svg)), 1)

    # ── direction: opposite directions land on opposite sides ──────────────────
    def test_bidirectional_opposite_sides(self):
        # identical timestamp on a->b and b->a -> shared base point, point-reflected tips
        T = datetime.datetime(2024, 1, 1, 8, 0, 0)
        df = pl.DataFrame([{'fm': 'a', 'to': 'b', 'ts': T}, {'fm': 'b', 'to': 'a', 'ts': T}])
        lp = self.p2s.linkp(df, relationships=_REL_, pos={'a': (0.0, 0.0), 'b': (1.0, 0.3)},
                            link_shape='line', time='ts', timing_marks_length=5.0)
        ms = _marks(lp.svg)
        self.assertEqual(len(ms), 2)
        (x1a, y1a, x2a, y2a, ca), (x1b, y1b, x2b, y2b, cb) = ms
        self.assertAlmostEqual(x1a, x1b, delta=0.02)          # same base P
        self.assertAlmostEqual(y1a, y1b, delta=0.02)
        self.assertAlmostEqual((x2a + x2b) / 2, x1a, delta=0.05)  # tips reflected through P
        self.assertAlmostEqual((y2a + y2b) / 2, y1a, delta=0.05)
        self.assertEqual(ca, cb)                              # same time -> same color

    # ── length is specified in pixels and scales the tick ──────────────────────
    def test_length_scales_in_pixels(self):
        T = datetime.datetime(2024, 1, 1)
        df = pl.DataFrame([{'fm': 'a', 'to': 'b', 'ts': T}])
        pos = {'a': (0.0, 0.0), 'b': (1.0, 0.3)}

        def tick_len(L):
            lp = self.p2s.linkp(df, relationships=_REL_, pos=pos, link_shape='line',
                                time='ts', timing_marks_length=L)
            x1, y1, x2, y2, _ = _marks(lp.svg)[0]
            return math.hypot(x2 - x1, y2 - y1)

        # drawn length = L * sqrt(1 + 1/4) (perp reach L, tangential lean L/2)
        self.assertAlmostEqual(tick_len(3.0), 3.0 * math.sqrt(1.25), delta=0.05)
        self.assertAlmostEqual(tick_len(6.0), 6.0 * math.sqrt(1.25), delta=0.05)

    # ── color + position both encode the timestamp ─────────────────────────────
    def test_color_and_position_grade_with_time(self):
        base = datetime.datetime(2024, 1, 1)
        df = pl.DataFrame([{'fm': 'a', 'to': 'b', 'ts': base + datetime.timedelta(hours=6 * j)}
                           for j in range(5)])
        lp = self.p2s.linkp(df, relationships=_REL_, pos={'a': (0.0, 0.0), 'b': (1.0, 0.0)},
                            link_shape='line', time='ts', wxh=(200, 200))
        ms = sorted(_marks(lp.svg), key=lambda m: m[0])   # sort by base x (position along edge)
        self.assertEqual(len(ms), 5)
        xs = [m[0] for m in ms]
        cs = [m[4] for m in ms]
        self.assertTrue(all(xs[i] < xs[i + 1] for i in range(4)))   # position advances with time
        self.assertEqual(len(set(cs)), 5)                            # distinct spectrum colors

    # ── decimation: never draw more marks than the edge has pixels ─────────────
    def test_millisecond_events_collapse_on_days_scale(self):
        # three events on one edge: two 1ms apart, one a full day later. Normalized over
        # the day-long span, the 1ms pair shares a pixel bin and collapses to one mark.
        base = datetime.datetime(2024, 1, 1)
        df = pl.DataFrame({'fm': ['a', 'a', 'a'], 'to': ['b', 'b', 'b'],
                           'ts': [base, base + datetime.timedelta(milliseconds=1),
                                  base + datetime.timedelta(days=1)]})
        lp = self.p2s.linkp(df, relationships=_REL_, pos={'a': (0.0, 0.0), 'b': (1.0, 0.0)},
                            time='ts', wxh=(200, 200))
        self.assertEqual(len(_marks(lp.svg)), 2)

    def test_decimation_caps_and_scales_with_pixels(self):
        base = datetime.datetime(2024, 1, 1)
        n = 5000
        df = pl.DataFrame({'fm': ['a'] * n, 'to': ['b'] * n,
                           'ts': [base + datetime.timedelta(seconds=17 * k) for k in range(n)]})
        pos = {'a': (0.0, 0.0), 'b': (1.0, 0.0)}
        small = self.p2s.linkp(df, relationships=_REL_, pos=pos, time='ts', wxh=(128, 128))
        large = self.p2s.linkp(df, relationships=_REL_, pos=pos, time='ts', wxh=(512, 512))
        _ns_, _nl_ = len(_marks(small.svg)), len(_marks(large.svg))
        self.assertLess(_ns_, n)         # far fewer marks than events
        self.assertLess(_ns_, 128)       # at most ~1 per pixel of usable span (< canvas width)
        self.assertGreater(_nl_, _ns_)   # resolution scales with edge pixel length

    # ── time= mirrors timep's field forms ──────────────────────────────────────
    def test_linear_enum(self):
        lp = self.p2s.linkp(_bidir_df(), relationships=_REL_, pos=_POS_,
                            time=('ts', self.p2s.LT_Y_m_d_Hp))
        self.assertGreater(len(_marks(lp.svg)), 0)

    def test_periodic_enum(self):
        lp = self.p2s.linkp(_bidir_df(), relationships=_REL_, pos=_POS_,
                            time=('ts', self.p2s.PT_Hp))
        self.assertGreater(len(_marks(lp.svg)), 0)

    def test_tfield(self):
        lp = self.p2s.linkp(_bidir_df(), relationships=_REL_, pos=_POS_,
                            time=self.p2s.tField('ts', self.p2s.PT_Hp))
        self.assertGreater(len(_marks(lp.svg)), 0)

    def test_date_column(self):
        df = pl.DataFrame({'fm': ['a', 'b'], 'to': ['b', 'a'],
                           'd': [datetime.date(2024, 1, 1), datetime.date(2024, 3, 1)]})
        lp = self.p2s.linkp(df, relationships=_REL_, pos=_POS_, time='d')
        self.assertEqual(len(_marks(lp.svg)), 2)

    # ── validation ─────────────────────────────────────────────────────────────
    def test_missing_field_raises(self):
        with self.assertRaises(ValueError):
            self.p2s.linkp(_bidir_df(), relationships=_REL_, pos=_POS_, time='nope')

    def test_non_date_field_raises(self):
        with self.assertRaises(ValueError):
            self.p2s.linkp(_bidir_df(), relationships=_REL_, pos=_POS_, time='fm')

    def test_bad_type_raises(self):
        with self.assertRaises(ValueError):
            self.p2s.linkp(_bidir_df(), relationships=_REL_, pos=_POS_, time=42)

    # ── marks sit above the edges and below the nodes in draw order ────────────
    def test_marks_between_links_and_nodes(self):
        df = _bidir_df()
        lp = self.p2s.linkp(df, relationships=_REL_, pos=_POS_, time='ts')
        svg = lp.svg
        first_mark = svg.index('stroke-width="1.5"')
        first_node = svg.index('<circle')
        self.assertLess(first_mark, first_node)


if __name__ == '__main__':
    unittest.main()
