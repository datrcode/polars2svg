#
# test_tfield_typed.py
#
# Covers the typed TField replacement for the magic 'column|suffix' t-field string
# (the legacy form could hijack real columns whose names contain '|').
# polars2svg.polars2svg.py:
#   - TField (nested class on Polars2SVG) is a frozen str subclass; p2s.tField()
#     now returns one instead of a plain string.
#   - Legacy 'column|suffix' strings still work, but only when the literal string
#     is NOT itself a real column in the supplied df (isTField(column, df=...)) --
#     a real column named e.g. 'price|mp' is no longer hijacked into a transform
#     of a (nonexistent) 'price' column.
#   - Each accepted legacy string emits a one-time (per-process) deprecation
#     warning; TField never warns; an explicit TField whose alias collides with
#     a real column warns that the column is shadowed (but still transforms).
#
# The hijack tests below construct DataFrames that contain ONLY the literal
# 'price|mp'-style column and deliberately omit the base 'price' column -- so
# the *old* (pre-fix) hijacking behavior would crash looking up a base column
# that doesn't exist, while the fixed behavior treats the literal column as a
# plain field and succeeds.
#
import logging
import unittest

import polars as pl

from polars2svg import Polars2SVG, TField
from svg_test_utils import normalize_svg


class _CountingHandler_(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []
    def emit(self, record):
        self.records.append(record.getMessage())


class _WarningTestBase_(unittest.TestCase):
    '''Installs a counting handler and resets the OnceFilter's seen-message set
    so every test observes warn-once behavior from a clean slate.'''

    def setUp(self):
        self.p2s     = Polars2SVG()
        self.logger  = logging.getLogger('polars2svg_logger')
        self.handler = _CountingHandler_()
        self.logger.addHandler(self.handler)
        for _f_ in self.logger.filters:
            if type(_f_).__name__ == 'OnceFilter':
                _f_.seen_messages.clear()

    def tearDown(self):
        self.logger.removeHandler(self.handler)

    def deprecationWarnings(self):
        return [m for m in self.handler.records if 'is deprecated' in m]

    def collisionWarnings(self):
        return [m for m in self.handler.records if 'is shadowed by' in m]


# ---------------------------------------------------------------------------
# Hijack fixed: a real column literally named 'field|suffix' is used as-is,
# never hijacked into a transform of a (missing) base column.
# ---------------------------------------------------------------------------

class TestHijackFixed(_WarningTestBase_):

    def _df(self):
        return pl.DataFrame({
            'price|mp': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],   # literal column, no base 'price' column
            'cat':      ['A', 'A', 'A', 'B', 'B', 'B'],
            'b':        [10, 20, 30, 40, 50, 60],
        })

    def test_xyp_x_literal_column_not_hijacked(self):
        df = self._df()
        _xyp_ = self.p2s.xyp(df, x='price|mp', y='b', dot_size=3)
        self.assertFalse(_xyp_.__axisIsPeriodicTime__(_xyp_.x_clean))
        self.assertEqual(self.deprecationWarnings(), [])

    def test_histop_count_literal_column_not_hijacked(self):
        df = self._df()
        h = self.p2s.histop(df, bin_by='cat', count='price|mp')
        _expected_ = df.group_by('cat').agg(pl.col('price|mp').sum().alias('__count__'))
        _expected_lu_ = dict(zip(_expected_['cat'], _expected_['__count__']))
        _actual_lu_   = dict(zip(h.df_agg[h._bin_col_], h.df_agg['__count__']))
        self.assertEqual(_actual_lu_, _expected_lu_)
        self.assertEqual(self.deprecationWarnings(), [])

    def test_timep_count_literal_column_not_hijacked(self):
        df = self._df().with_columns(pl.datetime(2024, 1, (pl.arange(0, 6) % 6) + 1).alias('ts'))
        t = self.p2s.timep(df, 'ts', count='price|mp')
        self.assertIn('<svg', t.svg)
        self.assertEqual(self.deprecationWarnings(), [])

    def test_piep_count_literal_column_not_hijacked(self):
        df = self._df()
        p = self.p2s.piep(df, bin_by='cat', count='price|mp')
        self.assertIn('<svg', p.svg)
        self.assertEqual(self.deprecationWarnings(), [])

    def test_smallp_category_by_literal_column_not_hijacked(self):
        df   = self._df()
        xtpl = self.p2s.xyp(x='b', y='b', wxh=(48, 48))
        s = self.p2s.smallp(df, 'price|mp', xtpl, wxh=(256, 256))
        self.assertEqual(self.deprecationWarnings(), [])


# ---------------------------------------------------------------------------
# Legacy string deprecation: warns once per process, TField never warns.
# ---------------------------------------------------------------------------

class TestDeprecationWarningOnce(_WarningTestBase_):

    def test_legacy_string_warns_exactly_once(self):
        df = pl.DataFrame({
            'ts_dep1': [f'2024-0{m}-01' for m in range(1, 7)],
            'cat':     ['A', 'A', 'A', 'B', 'B', 'B'],
        }).with_columns(pl.col('ts_dep1').str.to_datetime())
        self.p2s.histop(df, bin_by='cat', color='ts_dep1|mp')
        self.p2s.histop(df, bin_by='cat', color='ts_dep1|mp')   # second call -> deduped
        _warnings_ = self.deprecationWarnings()
        self.assertEqual(len(_warnings_), 1)
        self.assertIn('ts_dep1', _warnings_[0])
        self.assertIn('p2s.tField', _warnings_[0])

    def test_tfield_never_warns(self):
        df = pl.DataFrame({
            'ts_dep2': [f'2024-0{m}-01' for m in range(1, 7)],
            'cat':     ['A', 'A', 'A', 'B', 'B', 'B'],
        }).with_columns(pl.col('ts_dep2').str.to_datetime())
        self.p2s.histop(df, bin_by='cat', color=self.p2s.tField('ts_dep2', self.p2s.PT_mp))
        self.assertEqual(self.deprecationWarnings(), [])


class TestDeprecationWarningAcrossComponents(_WarningTestBase_):
    '''The legacy-string deprecation warning isn't histop-specific -- every
    consumer of a t-field string (xyp x=/y=/color=, timep count=/color=,
    piep count=/color=, smallp category_by=) must fire it exactly once per
    distinct column, and never fire it for the equivalent p2s.tField() call.'''

    def _timeDf(self, ts_col, extra=None):
        _cols_ = {
            ts_col: [f'2024-0{m}-01' for m in range(1, 7)],
            'cat':  ['A', 'A', 'A', 'B', 'B', 'B'],
            'b':    [10, 20, 30, 40, 50, 60],
        }
        if extra: _cols_.update(extra)
        return pl.DataFrame(_cols_).with_columns(pl.col(ts_col).str.to_datetime())

    def test_xyp_x_legacy_string_warns_once(self):
        df = self._timeDf('ts_xyp_dep')
        self.p2s.xyp(df, 'ts_xyp_dep|mp', 'b', dot_size=3)
        self.p2s.xyp(df, 'ts_xyp_dep|mp', 'b', dot_size=3)   # second call -> deduped
        _warnings_ = self.deprecationWarnings()
        self.assertEqual(len(_warnings_), 1)
        self.assertIn('ts_xyp_dep', _warnings_[0])

    def test_xyp_x_tfield_never_warns(self):
        df = self._timeDf('ts_xyp_ok')
        self.p2s.xyp(df, self.p2s.tField('ts_xyp_ok', self.p2s.PT_mp), 'b', dot_size=3)
        self.assertEqual(self.deprecationWarnings(), [])

    def test_timep_count_legacy_string_warns_once(self):
        df = self._timeDf('ts_timep_dep')
        self.p2s.timep(df, 'ts_timep_dep', count='ts_timep_dep|mp')
        self.p2s.timep(df, 'ts_timep_dep', count='ts_timep_dep|mp')   # second call -> deduped
        _warnings_ = self.deprecationWarnings()
        self.assertEqual(len(_warnings_), 1)
        self.assertIn('ts_timep_dep', _warnings_[0])

    def test_timep_count_tfield_never_warns(self):
        df = self._timeDf('ts_timep_ok')
        self.p2s.timep(df, 'ts_timep_ok', count=self.p2s.tField('ts_timep_ok', self.p2s.PT_mp))
        self.assertEqual(self.deprecationWarnings(), [])

    def test_piep_count_legacy_string_warns_once(self):
        df = self._timeDf('ts_piep_dep')
        self.p2s.piep(df, bin_by='cat', count='ts_piep_dep|mp')
        self.p2s.piep(df, bin_by='cat', count='ts_piep_dep|mp')   # second call -> deduped
        _warnings_ = self.deprecationWarnings()
        self.assertEqual(len(_warnings_), 1)
        self.assertIn('ts_piep_dep', _warnings_[0])

    def test_piep_count_tfield_never_warns(self):
        df = self._timeDf('ts_piep_ok')
        self.p2s.piep(df, bin_by='cat', count=self.p2s.tField('ts_piep_ok', self.p2s.PT_mp))
        self.assertEqual(self.deprecationWarnings(), [])

    def test_smallp_category_by_legacy_string_warns_once(self):
        df   = self._timeDf('ts_smallp_dep')
        xtpl = self.p2s.xyp(x='b', y='b', wxh=(48, 48))
        self.p2s.smallp(df, 'ts_smallp_dep|mp', xtpl, wxh=(256, 256))
        self.p2s.smallp(df, 'ts_smallp_dep|mp', xtpl, wxh=(256, 256))   # second call -> deduped
        _warnings_ = self.deprecationWarnings()
        self.assertEqual(len(_warnings_), 1)
        self.assertIn('ts_smallp_dep', _warnings_[0])

    def test_smallp_category_by_tfield_never_warns(self):
        df   = self._timeDf('ts_smallp_ok')
        xtpl = self.p2s.xyp(x='b', y='b', wxh=(48, 48))
        self.p2s.smallp(df, self.p2s.tField('ts_smallp_ok', self.p2s.PT_mp), xtpl, wxh=(256, 256))
        self.assertEqual(self.deprecationWarnings(), [])


# ---------------------------------------------------------------------------
# Mixed TField + legacy string within the same tuple spec.
# ---------------------------------------------------------------------------

class TestMixedTuples(unittest.TestCase):
    def setUp(self):
        self.p2s = Polars2SVG()

    def test_histop_count_tuple_mixes_tfield_and_plain_field(self):
        df = pl.DataFrame({
            'ts_mix': [f'2024-0{m}-01' for m in range(1, 7)],
            'cat':    ['A', 'A', 'A', 'B', 'B', 'B'],
            'g':      ['x', 'y', 'x', 'y', 'x', 'y'],
        }).with_columns(pl.col('ts_mix').str.to_datetime())
        _tfield_ = self.p2s.tField('ts_mix', self.p2s.PT_mp)
        h = self.p2s.histop(df, bin_by='cat', count=(_tfield_, 'g'))
        self.assertIn('<svg', h.svg)
        self.assertIn(str(_tfield_), h.df.columns)


# ---------------------------------------------------------------------------
# TField vs. legacy string produce byte-identical SVG (no collision case).
# ---------------------------------------------------------------------------

class TestByteIdenticalEquivalence(unittest.TestCase):
    def setUp(self):
        self.p2s = Polars2SVG()
        self.df = pl.DataFrame({
            'ts_eq': [f'2024-0{m}-01' for m in range(1, 7)],
            'cat':   ['A', 'A', 'A', 'B', 'B', 'B'],
        }).with_columns(pl.col('ts_eq').str.to_datetime())

    def test_histop_color_tfield_equals_legacy_string(self):
        # Compares the aggregated dataframe (not the rendered SVG): histop's
        # color-spectrum rank assignment depends on polars group_by() row order,
        # which isn't guaranteed stable across separate calls even with identical
        # data/parameters -- a pre-existing, unrelated nondeterminism. Comparing
        # the aggregate keeps this test about the TField/legacy parsing
        # equivalence, not that incidental instability.
        a = self.p2s.histop(self.df, bin_by='cat', color=self.p2s.tField('ts_eq', self.p2s.PT_mp))
        b = self.p2s.histop(self.df, bin_by='cat', color='ts_eq|mp')
        self.assertEqual(a.df_agg.sort(a._bin_col_).to_dict(as_series=False),
                         b.df_agg.sort(b._bin_col_).to_dict(as_series=False))

    def test_timep_time_tfield_equals_legacy_tuple(self):
        a = self.p2s.timep(self.df, self.p2s.tField('ts_eq', self.p2s.PT_mp))
        b = self.p2s.timep(self.df, ('ts_eq', self.p2s.PT_mp))
        self.assertEqual(normalize_svg(a.svg), normalize_svg(b.svg))

    def test_xyp_x_tfield_equals_legacy_string(self):
        df = self.df.with_columns(pl.Series('b', [1, 2, 3, 4, 5, 6]))
        a = self.p2s.xyp(df, self.p2s.tField('ts_eq', self.p2s.PT_mp), 'b', dot_size=3)
        b = self.p2s.xyp(df, 'ts_eq|mp', 'b', dot_size=3)
        self.assertEqual(normalize_svg(a.svg), normalize_svg(b.svg))


# ---------------------------------------------------------------------------
# Dataless templates: the hijack guard re-evaluates once a df is supplied,
# since __validateInput__ re-runs on every clone.
# ---------------------------------------------------------------------------

class TestTemplates(_WarningTestBase_):

    def test_tfield_template_clone_renders_without_warning(self):
        tmpl  = self.p2s.xyp(x=self.p2s.tField('ts_tpl1', self.p2s.PT_mp), y='b')
        df    = pl.DataFrame({
            'ts_tpl1': [f'2024-0{m}-01' for m in range(1, 7)],
            'b':       [1, 2, 3, 4, 5, 6],
        }).with_columns(pl.col('ts_tpl1').str.to_datetime())
        clone = self.p2s.xyp(template=tmpl, df=df)
        self.assertTrue(clone.__axisIsPeriodicTime__(clone.x_clean))
        self.assertEqual(self.deprecationWarnings(), [])

    def test_legacy_string_template_clone_warns(self):
        tmpl  = self.p2s.xyp(x='ts_tpl2|mp', y='b')
        df    = pl.DataFrame({
            'ts_tpl2': [f'2024-0{m}-01' for m in range(1, 7)],
            'b':       [1, 2, 3, 4, 5, 6],
        }).with_columns(pl.col('ts_tpl2').str.to_datetime())
        clone = self.p2s.xyp(template=tmpl, df=df)
        self.assertTrue(clone.__axisIsPeriodicTime__(clone.x_clean))
        self.assertEqual(len(self.deprecationWarnings()), 1)

    def test_legacy_string_guard_reevaluates_when_literal_column_exists(self):
        '''Same legacy-string template, but the clone's df happens to contain a
        real column with that literal name -- the guard must re-evaluate against
        the *clone's* df and treat it as a plain column, not a t-field.'''
        tmpl  = self.p2s.xyp(x='price_tpl|mp', y='b')
        df    = pl.DataFrame({
            'price_tpl|mp': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],   # literal column, no base 'price_tpl'
            'b':            [1, 2, 3, 4, 5, 6],
        })
        clone = self.p2s.xyp(template=tmpl, df=df)
        self.assertFalse(clone.__axisIsPeriodicTime__(clone.x_clean))
        self.assertEqual(self.deprecationWarnings(), [])


# ---------------------------------------------------------------------------
# Alias collision: an explicit TField's derived alias matches a real column.
# ---------------------------------------------------------------------------

class TestAliasCollision(_WarningTestBase_):

    def test_explicit_tfield_alias_collides_with_real_column(self):
        df = pl.DataFrame({
            'price3':      [f'2024-0{m}-01' for m in range(1, 7)],
            'price3|mp':   [100, 200, 300, 400, 500, 600],   # shadowed by the derived column
            'cat':         ['A', 'A', 'A', 'B', 'B', 'B'],
        }).with_columns(pl.col('price3').str.to_datetime())
        h = self.p2s.histop(df, bin_by='cat', color=self.p2s.tField('price3', self.p2s.PT_mp))
        self.assertIn('<svg', h.svg)
        self.assertEqual(len(self.collisionWarnings()), 1)


if __name__ == '__main__':
    unittest.main()
