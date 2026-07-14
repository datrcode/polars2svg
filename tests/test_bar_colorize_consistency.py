"""Cross-component bar-colorizer consistency: histop vs timep.

Both components render the same two-bin DataFrame.  For every combination of
count × color the bar rectangles extracted from each SVG must have identical
dimensions and fill colors:
  - histop  → horizontal bars  → compare rect ``width``
  - timep   → vertical bars    → compare rect ``height``

Rects are matched by sorting on fill color, which works for both simple
(one rect per bin) and stacked (multiple rects per bin) bar modes.
"""
import random
import unittest
from xml.etree import ElementTree as ET

import polars as pl
from polars2svg import Polars2SVG


# ── deterministic two-bin DataFrame ─────────────────────────────────────────
# Two timestamps → two bins in both histop (bin_by='ts') and timep (auto-ts).
# Fields mirror the notebook: uniform ints/floats, bimodal variants, and two
# categorical columns of different cardinality.

def _make_df(n=1_000, seed=42):
    rng = random.Random(seed)

    def _cat(length):
        chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        return ''.join(rng.choice(chars) for _ in range(length))

    rows = {
        'ts':         [],
        'int':        [],
        'float':      [],
        'int_diff':   [],
        'float_diff': [],
        'cat_small':  [],
        'cat_large':  [],
    }
    for _ in range(n):
        ts = '2025-05-10 13:42:21' if rng.random() < 0.1 else '2025-05-10 12:42:21'
        rows['ts'].append(ts)
        rows['int'].append(rng.randint(0, 100))
        rows['float'].append(1_000.0 * rng.random())
        rows['int_diff'].append(rng.randint(1_000, 10_000) if rng.random() < 0.1 else rng.randint(0, 100))
        rows['float_diff'].append(1_000.0 + 10_000.0 * rng.random() if rng.random() < 0.1 else rng.random())
        rows['cat_small'].append(_cat(1))
        rows['cat_large'].append(_cat(2))

    return pl.DataFrame(rows).with_columns(
        pl.col('ts').str.strptime(pl.Datetime, '%Y-%m-%d %H:%M:%S')
    )


_DF_ = _make_df()

_COUNTS_ = [
    None,
    'ROW_COUNT',                    # → p2s.ROW_COUNTp
    'int',
    'float',
    'int_diff',
    'float_diff',
    'cat_small',
    'cat_large',
    ('int',       'SET'),           # → ('int',       p2s.SETp)  n_unique of int values
    ('cat_small', 'SET'),           # → ('cat_small', p2s.SETp)  n_unique of categories
    ('int', 'float'),               # multi-field struct n_unique
]

_COLORS_ = [
    None,
    # ── plain field (spectrum if numeric, categorical if string) ────────────
    'int',
    'float',
    'int_diff',
    'float_diff',
    'cat_small',
    'cat_large',
    'CROW_MAGNITUDEp',
    'CROW_STRETCHEDp',
    # ── CSETp: treat field as categorical set ───────────────────────────────
    ('int',       'CSET'),          # → ('int',       p2s.CSETp)
    ('float',     'CSET'),          # → ('float',     p2s.CSETp)
    ('int_diff',  'CSET'),          # → ('int_diff',  p2s.CSETp)
    ('float_diff','CSET'),          # → ('float_diff',p2s.CSETp)
    # -- Other Color Enums ---------------------------------------------------
    ('int',       'CSET_MAGNITUDEp'),
    ('int',       'CSET_STRETCHEDp'),
    ('int',       'CMAGNITUDE_SUMp'),
    ('int',       'CMAGNITUDE_MINp'),
    ('int',       'CMAGNITUDE_MEDIANp'),
    ('int',       'CMAGNITUDE_MEANp'),
    ('int',       'CMAGNITUDE_MAXp'),
    ('int',       'CSTRETCHED_SUMp'),
    ('int',       'CSTRETCHED_MINp'),
    ('int',       'CSTRETCHED_MEDIANp'),
    ('int',       'CSTRETCHED_MEANp'),
    ('int',       'CSTRETCHED_MAXp'),
    ('cat_large', 'CSET_MAGNITUDEp'),
    ('cat_large', 'CSET_STRETCHEDp'),
    ('float',     'CMAGNITUDE_SUMp'),
    ('float',     'CMAGNITUDE_MINp'),
    ('float',     'CMAGNITUDE_MEDIANp'),
    ('float',     'CMAGNITUDE_MEANp'),
    ('float',     'CMAGNITUDE_MAXp'),
    ('float',     'CSTRETCHED_SUMp'),
    ('float',     'CSTRETCHED_MINp'),
    ('float',     'CSTRETCHED_MEDIANp'),
    ('float',     'CSTRETCHED_MEANp'),
    ('float',     'CSTRETCHED_MAXp'),
    # ── numeric stat ops: override the default sum aggregation ───────────────
    ('int',   'MIN'),               # → ('int',   p2s.MINp)
    ('int',   'MAX'),               # → ('int',   p2s.MAXp)
    ('int',   'MEDIAN'),            # → ('int',   p2s.MEDIANp)
    ('int',   'MEAN'),              # → ('int',   p2s.MEANp)
    ('int',   'STD'),               # → ('int',   p2s.STDp)
    ('float', 'MIN'),
    ('float', 'MAX'),
    ('float', 'MEAN'),
    # ── multi-field color concatenation ─────────────────────────────────────
    ('cat_small', 'cat_large'),     # two string fields joined → categorical stacking
]


# ── SVG helpers ──────────────────────────────────────────────────────────────

def _get_bar_rects(svg_str, dim_attr):
    """Return a sorted list of (dimension, fill) for data bar rects.

    Excludes:
      - background rect  (x=0, y=0)
      - axis border rect (fill="none")
      - boxplot semi-transparent rects (fill-opacity attribute present)

    dim_attr: ``'width'`` for histop, ``'height'`` for timep.
    """
    root = ET.fromstring(svg_str)
    bars = []
    for rect in root.iter('{http://www.w3.org/2000/svg}rect'):
        fill    = rect.get('fill', '')
        x, y    = rect.get('x', ''), rect.get('y', '')
        fill_op = rect.get('fill-opacity')
        if fill in ('none', ''):   continue   # axis border / empty
        if x == '0' and y == '0': continue   # background
        if fill_op is not None:    continue   # boxplot semi-transparent
        dim = float(rect.get(dim_attr, 0))
        if dim > 0:
            bars.append((round(dim, 1), fill))
    return sorted(bars, key=lambda b: b[1])


# ── test class ───────────────────────────────────────────────────────────────

class TestBarColorizeConsistency(unittest.TestCase):
    """histop bar widths must equal timep bar heights for every count×color combo."""

    @classmethod
    def setUpClass(cls):
        cls.p2s = Polars2SVG()

    _SENTINEL_MAP = {
        'ROW_COUNT':          lambda p: p.ROW_COUNTp,
        'SET':                lambda p: p.SETp,
        'CSET':               lambda p: p.CSETp,
        'CSET_MAGNITUDEp':   lambda p: p.CSET_MAGNITUDEp,
        'CSET_STRETCHEDp':   lambda p: p.CSET_STRETCHEDp,
        'CROW_MAGNITUDEp':   lambda p: p.CROW_MAGNITUDEp,
        'CROW_STRETCHEDp':   lambda p: p.CROW_STRETCHEDp,
        'CMAGNITUDE_SUMp':   lambda p: p.CMAGNITUDE_SUMp,
        'CMAGNITUDE_MINp':   lambda p: p.CMAGNITUDE_MINp,
        'CMAGNITUDE_MEDIANp':lambda p: p.CMAGNITUDE_MEDIANp,
        'CMAGNITUDE_MEANp':  lambda p: p.CMAGNITUDE_MEANp,
        'CMAGNITUDE_MAXp':   lambda p: p.CMAGNITUDE_MAXp,
        'CSTRETCHED_SUMp':   lambda p: p.CSTRETCHED_SUMp,
        'CSTRETCHED_MINp':   lambda p: p.CSTRETCHED_MINp,
        'CSTRETCHED_MEDIANp':lambda p: p.CSTRETCHED_MEDIANp,
        'CSTRETCHED_MEANp':  lambda p: p.CSTRETCHED_MEANp,
        'CSTRETCHED_MAXp':   lambda p: p.CSTRETCHED_MAXp,
        'MIN':                lambda p: p.MINp,
        'MAX':                lambda p: p.MAXp,
        'MEDIAN':             lambda p: p.MEDIANp,
        'MEAN':               lambda p: p.MEANp,
        'STD':                lambda p: p.STDp,
    }

    def _resolve(self, value):
        """Replace sentinel strings / tuples-with-sentinels with real p2s enum values."""
        if isinstance(value, str) and value in self._SENTINEL_MAP:
            return self._SENTINEL_MAP[value](self.p2s)
        if isinstance(value, tuple):
            resolved = tuple(
                self._SENTINEL_MAP[v](self.p2s) if isinstance(v, str) and v in self._SENTINEL_MAP else v
                for v in value
            )
            return resolved
        return value

    def test_bar_dimensions_and_colors_match(self):
        """All count×color combinations: histop widths == timep heights, fills identical."""
        params_shared = {'wxh': (256, 256), 'insets': (2, 2), 'draw_context': False}
        failures = []

        for _count_raw_ in _COUNTS_:
            for _color_raw_ in _COLORS_:
                count = self._resolve(_count_raw_)
                color = self._resolve(_color_raw_)

                params = dict(params_shared, color=color, count=count)
                h = self.p2s.histop(_DF_, 'ts', **params, distribution=False)
                t = self.p2s.timep(_DF_,        **params)

                h_bars = _get_bar_rects(h.svg, 'width')
                t_bars = _get_bar_rects(t.svg, 'height')

                label = f'count={_count_raw_!r} color={_color_raw_!r}'

                if len(h_bars) != len(t_bars):
                    failures.append(
                        f'{label}: bar count mismatch  histop={len(h_bars)}  timep={len(t_bars)}'
                    )
                    continue

                for i, (h_bar, t_bar) in enumerate(zip(h_bars, t_bars)):
                    h_dim, h_fill = h_bar
                    t_dim, t_fill = t_bar
                    if h_fill != t_fill:
                        failures.append(
                            f'{label}  bar[{i}]: fill mismatch  histop={h_fill!r}  timep={t_fill!r}'
                        )
                    if h_dim != t_dim:
                        failures.append(
                            f'{label}  bar[{i}]: dim mismatch  histop_w={h_dim}  timep_h={t_dim}'
                        )

        if failures:
            self.fail(f'{len(failures)} failure(s):\n' + '\n'.join(f'  {f}' for f in failures))

    def test_cset_single_value_uniform_fill(self):
        """CSETp on a single-value integer field: every component uses the same fill color.

        Regression for a dtype-hash mismatch where the stacked bar path (histop/timep)
        hashed the raw integer column value while histop's simple path and xyp hashed
        the string representation — same logical value, two visually different colors.

        src/dst are kept to ≤10 unique values so all bins fit in the chart without
        triggering the overflow indicator rect, which _get_bar_rects would otherwise
        pick up as a spurious color.
        """
        rng = random.Random(7)
        n = 500
        df = pl.DataFrame({
            'src':  [rng.randint(1, 8)  for _ in range(n)],
            'dst':  [rng.randint(1, 6)  for _ in range(n)],
            'port': [25] * n,                                # single integer color value
            'ts':   ['2025-01-01 10:00:00' if rng.random() < 0.4
                     else '2025-01-01 11:00:00' for _ in range(n)],
        }).with_columns(pl.col('ts').str.strptime(pl.Datetime, '%Y-%m-%d %H:%M:%S'))

        color = ('port', self.p2s.CSETp)
        params_shared = {'color': color, 'wxh': (256, 128), 'draw_context': False}

        def _hbar_fills(svg_str):
            return {fill for _, fill in _get_bar_rects(svg_str, 'width')}

        def _vbar_fills(svg_str):
            return {fill for _, fill in _get_bar_rects(svg_str, 'height')}

        def _dot_fills(svg_str):
            # xyp renders dots as <circle> or <rect> depending on density/field type.
            root = ET.fromstring(svg_str)
            fills = set()
            for el in root.iter():
                tag = el.tag.split('}')[-1] if '}' in el.tag else el.tag
                if tag not in ('circle', 'rect'):
                    continue
                fill = el.get('fill', '')
                if not fill or fill in ('none', '#ffffff'):
                    continue
                if el.get('x', '') == '0' and el.get('y', '') == '0':
                    continue  # background
                if el.get('fill-opacity') is not None:
                    continue  # semi-transparent (boxplot / swarm)
                fills.add(fill)
            return fills

        # histop stacked path: color_field ('port') != bin_col ('src'/'dst')
        # histop simple path:  color_field ('port') == bin_col ('port')
        components = {
            'histop(src,stacked)': _hbar_fills(self.p2s.histop(df, 'src',  **params_shared, distribution=False).svg),
            'histop(dst,stacked)': _hbar_fills(self.p2s.histop(df, 'dst',  **params_shared, distribution=False).svg),
            'histop(port,simple)': _hbar_fills(self.p2s.histop(df, 'port', **params_shared, distribution=False).svg),
            'timep':               _vbar_fills(self.p2s.timep(df,           **params_shared).svg),
            'xyp':                 _dot_fills(self.p2s.xyp(df, 'src', 'dst',
                                                           color=color, wxh=(256, 128)).svg),
        }

        errors = []
        for label, fills in components.items():
            if len(fills) != 1:
                errors.append(f'{label}: expected 1 fill, got {fills!r}')

        # All five components must agree on the same single color.
        unique = {next(iter(f)) for f in components.values() if len(f) == 1}
        if len(unique) > 1:
            summary = ', '.join(f'{k}={next(iter(v))!r}'
                                for k, v in components.items() if len(v) == 1)
            errors.append(f'cross-component fill mismatch: {summary}')

        if errors:
            self.fail('\n'.join(errors))


if __name__ == '__main__':
    unittest.main()
