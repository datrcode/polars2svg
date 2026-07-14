"""Reserved-column collision checks.

The framework aliases dunder-style working columns ('__count__', '__bin__',
'__row_count__', ...) into DataFrames during aggregation and rendering. A user
column with one of those names used to silently produce wrong aggregates; now
every component's __validateInput__ calls Polars2SVG.checkReservedColumns()
and raises ValueError on any '__name__'-pattern column.

A small persisted subset is tolerated on input because the framework writes
those columns into self.df in place and they legitimately re-enter components
when a DataFrame round-trips (smallp panel dfs, interactive drill-down stack
pushes, plot.df reuse):

  exact:   __p2s_index__, __bin__, __color__, __time_bin__, __lc_cat__
  pattern: __rel{i}_{fm|to}_{w|s}{x|y}__, __fm{i}__, __to{i}__

Each persisted column is deterministically overwritten (aliased with_columns)
or framework-managed before it is read, so tolerating them cannot corrupt
aggregates the way an arbitrary reserved-name collision can.
"""
import unittest
import datetime

import polars as pl
from polars2svg import Polars2SVG


def _base_df(**extra_cols):
    n = 6
    data = {
        'cat': ['a', 'b', 'c', 'a', 'b', 'c'],
        'x':   [1,   2,   3,   4,   5,   6  ],
        'y':   [6,   5,   4,   3,   2,   1  ],
        'ts':  [datetime.date(2024, m, 1) for m in range(1, n + 1)],
        'fm':  ['a', 'b', 'c', 'a', 'b', 'd'],
        'to':  ['b', 'c', 'a', 'c', 'a', 'a'],
        'w':   [1,   3,   2,   1,   4,   2  ],
    }
    data.update(extra_cols)
    return pl.DataFrame(data)


_RELS_ = [('fm', 'to')]
_POS_  = {'a': [0, 0], 'b': [1, 0], 'c': [0.5, 0.866], 'd': [0.5, 0.3]}


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests of the shared helper
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckReservedColumnsHelper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p2s = Polars2SVG()

    def test_raises_on_reserved_names(self):
        for name in ['__count__', '__row_count__', '__bin_key__', '__order_metric__',
                     '__color_stat__', '__totally_made_up__']:
            df = _base_df(**{name: [0] * 6})
            with self.assertRaises(ValueError, msg=name):
                self.p2s.checkReservedColumns(df, 'Test')

    def test_error_message_names_column_and_component(self):
        df = _base_df(__count__=[0] * 6)
        with self.assertRaises(ValueError) as ctx:
            self.p2s.checkReservedColumns(df, 'Histop')
        self.assertIn('__count__', str(ctx.exception))
        self.assertIn('Histop',    str(ctx.exception))

    def test_allows_persisted_exact_names(self):
        for name in ['__p2s_index__', '__bin__', '__color__', '__time_bin__', '__lc_cat__']:
            df = _base_df(**{name: [0] * 6})
            self.p2s.checkReservedColumns(df, 'Test')  # must not raise

    def test_allows_persisted_pattern_names(self):
        for name in ['__rel0_fm_wx__', '__rel12_to_sy__', '__fm0__', '__to3__']:
            df = _base_df(**{name: [0] * 6})
            self.p2s.checkReservedColumns(df, 'Test')  # must not raise

    def test_allows_non_reserved_names(self):
        # Only the full '__name__' pattern is reserved
        df = _base_df(**{'_x_': [0] * 6, '__leading': [0] * 6, 'trailing__': [0] * 6})
        self.p2s.checkReservedColumns(df, 'Test')  # must not raise

    def test_none_df_is_a_noop(self):
        self.p2s.checkReservedColumns(None, 'Test')  # must not raise

    def test_multiple_collisions_all_reported(self):
        df = _base_df(__count__=[0] * 6, __foo__=[0] * 6)
        with self.assertRaises(ValueError) as ctx:
            self.p2s.checkReservedColumns(df, 'Test')
        self.assertIn('__count__', str(ctx.exception))
        self.assertIn('__foo__',   str(ctx.exception))


# ─────────────────────────────────────────────────────────────────────────────
# Every component raises on a reserved-name collision
# ─────────────────────────────────────────────────────────────────────────────

class TestComponentsRaiseOnCollision(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p2s = Polars2SVG()
        cls.df  = _base_df(__count__=[9] * 6)

    def test_histop(self):
        with self.assertRaises(ValueError) as ctx:
            self.p2s.histop(self.df, 'cat')
        self.assertIn('__count__', str(ctx.exception))

    def test_timep(self):
        with self.assertRaises(ValueError):
            self.p2s.timep(self.df, 'ts')

    def test_xyp(self):
        with self.assertRaises(ValueError):
            self.p2s.xyp(self.df, 'x', 'y')

    def test_piep(self):
        with self.assertRaises(ValueError):
            self.p2s.piep(self.df, 'cat')

    def test_linkp(self):
        with self.assertRaises(ValueError):
            self.p2s.linkp(self.df, relationships=_RELS_, pos=_POS_)

    def test_chordp(self):
        with self.assertRaises(ValueError):
            self.p2s.chordp(df=self.df, relationships=_RELS_)

    def test_spreadlinesp(self):
        df = self.df.with_columns(pl.col('x').alias('time'))
        with self.assertRaises(ValueError):
            self.p2s.spreadlinesp(df, _RELS_, ego='a', time='time')

    def test_smallp(self):
        tmpl = self.p2s.xyp(x='x', y='y', wxh=(128, 128))
        with self.assertRaises(ValueError):
            self.p2s.smallp(self.df, 'cat', tmpl, wxh=(384, 384))


# ─────────────────────────────────────────────────────────────────────────────
# Clean DataFrames still render (no false positives)
# ─────────────────────────────────────────────────────────────────────────────

class TestNoFalsePositives(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p2s = Polars2SVG()
        cls.df  = _base_df()

    def test_histop_clean(self):
        h = self.p2s.histop(self.df, 'cat')
        self.assertIn('<svg', h.svg)

    def test_xyp_clean(self):
        x = self.p2s.xyp(self.df, 'x', 'y')
        self.assertIn('<svg', x._repr_svg_())

    def test_smallp_clean(self):
        tmpl = self.p2s.xyp(x='x', y='y', wxh=(128, 128))
        s = self.p2s.smallp(self.df, 'cat', tmpl, wxh=(384, 384))
        self.assertIn('<svg', s.svg)


# ─────────────────────────────────────────────────────────────────────────────
# Framework round-trips must keep working: a rendered plot's df (which carries
# persisted framework columns) can be fed back into a fresh component, exactly
# as the interactive drill-down stack and smallp panel cloning do.
# ─────────────────────────────────────────────────────────────────────────────

class TestRoundTripTolerance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p2s = Polars2SVG()
        cls.df  = _base_df()

    def test_histop_roundtrip_tuple_bin_and_color(self):
        # tuple bin_by adds '__bin__'; tuple color adds '__color__'; every
        # component adds '__p2s_index__'
        h1 = self.p2s.histop(self.df, ('cat', 'fm'), color=('cat', 'to'))
        self.assertIn('__p2s_index__', h1.df.columns)
        self.assertIn('__bin__',       h1.df.columns)
        h2 = self.p2s.histop(h1.df, ('cat', 'fm'), color=('cat', 'to'))  # must not raise
        self.assertIn('<svg', h2.svg)

    def test_timep_roundtrip_periodic(self):
        # periodic mode adds '__time_bin__'
        t1 = self.p2s.timep(self.df, ('ts', self.p2s.PT_mp))
        self.assertIn('__time_bin__', t1.df.columns)
        t2 = self.p2s.timep(t1.df, ('ts', self.p2s.PT_mp))  # must not raise
        self.assertIn('<svg', t2._repr_svg_())

    def test_linkp_roundtrip_categorical_color(self):
        # categorical color adds '__lc_cat__'; layout adds '__rel{i}_*__' columns
        l1 = self.p2s.linkp(self.df, relationships=_RELS_, pos=_POS_, color='cat')
        self.assertIn('__lc_cat__',     l1.df.columns)
        self.assertIn('__rel0_fm_wx__', l1.df.columns)
        l2 = self.p2s.linkp(l1.df, relationships=_RELS_, pos=_POS_, color='cat')  # must not raise
        self.assertIn('<svg', l2.svg)

    def test_linkp_roundtrip_tuple_endpoints(self):
        # tuple relationship endpoints add '__fm{i}__'/'__to{i}__' concat columns
        l1 = self.p2s.linkp(self.df, relationships=[(('fm', 'cat'), 'to')], pos={})
        self.assertIn('__fm0__', l1.df.columns)
        l2 = self.p2s.linkp(l1.df, relationships=[(('fm', 'cat'), 'to')], pos={})  # must not raise
        self.assertIn('<svg', l2.svg)

    def test_smallp_panel_dfs_reenter_template(self):
        # smallp adds '__p2s_index__' then clones its template per panel slice
        tmpl = self.p2s.xyp(x='x', y='y', wxh=(128, 128))
        s = self.p2s.smallp(self.df, 'cat', tmpl, wxh=(384, 384))
        self.assertIn('<svg', s.svg)


if __name__ == '__main__':
    unittest.main()
