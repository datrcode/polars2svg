import re
import unittest
from datetime import datetime, timedelta
from xml.etree import ElementTree as ET
import polars as pl
from polars2svg import Polars2SVG


# DataFrame with 3 bins: bin 'a' has 3 rows, 'b' has 2 rows, 'c' has 1 row.
# val is intentionally uncorrelated with row count so that count='val' would
# produce different colors than count=ROW_COUNTp if crow color used __count__.
_DF_ = pl.DataFrame({
    'bin': ['a', 'a', 'a', 'b', 'b', 'c'],
    'val': [1.0, 1.0, 1.0, 5.0, 5.0, 10.0],
})

_TS_BASE_ = datetime(2024, 1, 1)
_DF_TIME_ = pl.DataFrame({
    'ts':  [_TS_BASE_ + timedelta(hours=i) for i in [0, 0, 0, 1, 1, 2]],
    'val': [1.0, 1.0, 1.0, 5.0, 5.0, 10.0],
})

_DF_LINK_ = pl.DataFrame({
    'fm':  ['a', 'a', 'a', 'b', 'b', 'c'],
    'to':  ['b', 'b', 'b', 'c', 'c', 'a'],
    'val': [1.0, 1.0, 1.0, 5.0, 5.0, 10.0],
})
_REL_  = [('fm', 'to')]
_POS_  = {'a': (0.0, 0.0), 'b': (1.0, 0.0), 'c': (0.5, 1.0)}


def _bar_fills(svg_str):
    root = ET.fromstring(svg_str)
    fills = []
    for rect in root.iter('{http://www.w3.org/2000/svg}rect'):
        fill    = rect.get('fill', '')
        x, y    = rect.get('x', ''), rect.get('y', '')
        fill_op = rect.get('fill-opacity')
        if fill in ('none', ''): continue
        if x == '0' and y == '0': continue
        if fill_op is not None:   continue
        fills.append(fill)
    return sorted(fills)


def _node_fills(svg):
    return sorted(re.findall(r'<circle[^>]*fill="(#[0-9a-fA-F]+)"', svg))


class TestCrowColorIndependence(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    # ── histop ──────────────────────────────────────────────────────────────

    def test_histop_crow_magnitude_independent_of_count(self):
        hp_row = self.p2s.histop(_DF_, 'bin', color=self.p2s.CROW_MAGNITUDEp,
                                  draw_context=False, distribution=False)
        hp_val = self.p2s.histop(_DF_, 'bin', color=self.p2s.CROW_MAGNITUDEp,
                                  count='val',
                                  draw_context=False, distribution=False)
        self.assertEqual(_bar_fills(hp_row.svg), _bar_fills(hp_val.svg))

    def test_histop_crow_stretched_independent_of_count(self):
        hp_row = self.p2s.histop(_DF_, 'bin', color=self.p2s.CROW_STRETCHEDp,
                                  draw_context=False, distribution=False)
        hp_val = self.p2s.histop(_DF_, 'bin', color=self.p2s.CROW_STRETCHEDp,
                                  count='val',
                                  draw_context=False, distribution=False)
        self.assertEqual(_bar_fills(hp_row.svg), _bar_fills(hp_val.svg))

    # ── timep ────────────────────────────────────────────────────────────────

    def test_timep_crow_magnitude_independent_of_count(self):
        tp_row = self.p2s.timep(_DF_TIME_, 'ts', color=self.p2s.CROW_MAGNITUDEp,
                                 draw_context=False)
        tp_val = self.p2s.timep(_DF_TIME_, 'ts', color=self.p2s.CROW_MAGNITUDEp,
                                 count='val',
                                 draw_context=False)
        self.assertEqual(_bar_fills(tp_row.svg), _bar_fills(tp_val.svg))

    def test_timep_crow_stretched_independent_of_count(self):
        tp_row = self.p2s.timep(_DF_TIME_, 'ts', color=self.p2s.CROW_STRETCHEDp,
                                 draw_context=False)
        tp_val = self.p2s.timep(_DF_TIME_, 'ts', color=self.p2s.CROW_STRETCHEDp,
                                 count='val',
                                 draw_context=False)
        self.assertEqual(_bar_fills(tp_row.svg), _bar_fills(tp_val.svg))

    # ── linkp ────────────────────────────────────────────────────────────────

    def test_linkp_crow_magnitude_independent_of_count(self):
        lp_row = self.p2s.linkp(_DF_LINK_, _REL_, _POS_,
                                 node_color=self.p2s.CROW_MAGNITUDEp)
        lp_val = self.p2s.linkp(_DF_LINK_, _REL_, _POS_,
                                 node_color=self.p2s.CROW_MAGNITUDEp, count='val')
        self.assertEqual(_node_fills(lp_row.svg), _node_fills(lp_val.svg))

    def test_linkp_crow_stretched_independent_of_count(self):
        lp_row = self.p2s.linkp(_DF_LINK_, _REL_, _POS_,
                                 node_color=self.p2s.CROW_STRETCHEDp)
        lp_val = self.p2s.linkp(_DF_LINK_, _REL_, _POS_,
                                 node_color=self.p2s.CROW_STRETCHEDp, count='val')
        self.assertEqual(_node_fills(lp_row.svg), _node_fills(lp_val.svg))


if __name__ == '__main__':
    unittest.main()
