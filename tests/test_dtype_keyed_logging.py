#
# test_dtype_keyed_logging.py
#
# Dtype-keyed semantics diagnostics.  Several parameters infer their meaning
# from the *dtype* of a bare field spec:
#
#   count='field'  -> sum()      if the column is numeric, n_unique() otherwise
#   color='field'  -> magnitude  if the column is numeric, categorical otherwise
#
# That inference is silent, so an upstream schema change (string IDs becoming
# integer IDs) flips the interpretation with no signal.  The framework now emits
# a one-time INFO log (per distinct message, via the logger OnceFilter) naming
# the field, the interpretation picked, and the enum that would pin intent:
#
#   Polars2SVG.logDtypeKeyedCount(component, field, is_numeric)
#   Polars2SVG.logDtypeKeyedColor(component, field, is_numeric)
#
# INFO is off by default (so normal use stays quiet); these tests raise the
# logger to INFO and capture the records.  Explicit enums ((field, SETp) /
# (field, SCALARp) for count; (field, CSETp) / CMAGNITUDE_* for color) take the
# enum path and must NOT emit the dtype-keyed log.
#
import logging
import sys
import unittest

import polars as pl

sys.path.insert(0, 'tests')

from polars2svg import Polars2SVG
from timep_dataframes import makeTimeDf


# ── shared fixtures ─────────────────────────────────────────────────────────

_GRAPH_DF_ = pl.DataFrame({
    'fm':   ['a', 'b', 'c', 'a', 'd', 'b'],
    'to':   ['b', 'a', 'a', 'c', 'a', 'c'],
    'time': [1,   1,   1,   2,   2,   3  ],
    'w':    [3,   1,   2,   4,   1,   2  ],   # numeric edge weight
})
_RELS_ = [('fm', 'to')]
_POS_  = {'a': (0, 0), 'b': (1, 0), 'c': (1, 1), 'd': (0, 1)}

_BAR_DF_ = pl.DataFrame({
    'cat':   ['A', 'B', 'A', 'C', 'B', 'A'],   # non-numeric bin field
    'group': ['x', 'y', 'x', 'y', 'x', 'y'],   # non-numeric
    'value': [10,  20,  30,  40,  50,  60 ],   # numeric
})

_NUMERIC_FRAGMENT_     = 'is numeric ->'
_NON_NUMERIC_FRAGMENT_ = 'is non-numeric ->'


class _CapturingHandler_(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []
    def emit(self, record):
        self.records.append(record.getMessage())


class _LogTestBase_(unittest.TestCase):
    """Raise the polars2svg logger to INFO, capture records, and clear the
    OnceFilter's seen-message set so each test starts from a clean slate
    (test-order independence).  The prior level is restored on teardown."""

    def setUp(self):
        self.p2s      = Polars2SVG()
        self.logger   = logging.getLogger('polars2svg_logger')
        self._prior_level_ = self.logger.level
        self.logger.setLevel(logging.INFO)
        self.handler  = _CapturingHandler_()
        self.logger.addHandler(self.handler)
        self._clearOnceFilter_()

    def tearDown(self):
        self.logger.removeHandler(self.handler)
        self.logger.setLevel(self._prior_level_)

    def _clearOnceFilter_(self):
        for _f_ in self.logger.filters:
            if type(_f_).__name__ == 'OnceFilter':
                _f_.seen_messages.clear()

    # -- assertions ----------------------------------------------------------

    def countLogs(self):
        return [m for m in self.handler.records
                if 'count=' in m and (_NUMERIC_FRAGMENT_ in m or _NON_NUMERIC_FRAGMENT_ in m)]

    def colorLogs(self):
        return [m for m in self.handler.records
                if 'color=' in m and (_NUMERIC_FRAGMENT_ in m or _NON_NUMERIC_FRAGMENT_ in m)]

    def assertLoggedNumericCount(self, component, field):
        _msgs_ = [m for m in self.countLogs() if m.startswith(f'{component}:') and repr(field) in m]
        self.assertEqual(len(_msgs_), 1, f'expected one numeric count log for {field!r}, got {self.countLogs()}')
        self.assertIn(_NUMERIC_FRAGMENT_, _msgs_[0])
        self.assertIn('SETp', _msgs_[0])   # names the override enum

    def assertLoggedNonNumericCount(self, component, field):
        _msgs_ = [m for m in self.countLogs() if m.startswith(f'{component}:') and repr(field) in m]
        self.assertEqual(len(_msgs_), 1, f'expected one non-numeric count log for {field!r}, got {self.countLogs()}')
        self.assertIn(_NON_NUMERIC_FRAGMENT_, _msgs_[0])
        self.assertIn('SCALARp', _msgs_[0])   # names the override enum

    def assertLoggedNumericColor(self, component, field):
        _msgs_ = [m for m in self.colorLogs() if m.startswith(f'{component}:') and repr(field) in m]
        self.assertEqual(len(_msgs_), 1, f'expected one numeric color log for {field!r}, got {self.colorLogs()}')
        self.assertIn(_NUMERIC_FRAGMENT_, _msgs_[0])
        self.assertIn('CSETp', _msgs_[0])

    def assertLoggedNonNumericColor(self, component, field):
        _msgs_ = [m for m in self.colorLogs() if m.startswith(f'{component}:') and repr(field) in m]
        self.assertEqual(len(_msgs_), 1, f'expected one non-numeric color log for {field!r}, got {self.colorLogs()}')
        self.assertIn(_NON_NUMERIC_FRAGMENT_, _msgs_[0])

    def assertNoCountLog(self):
        self.assertEqual(self.countLogs(), [])

    def assertNoColorLog(self):
        self.assertEqual(self.colorLogs(), [])


# ── helper unit tests ───────────────────────────────────────────────────────

class TestHelperUnits(_LogTestBase_):

    def test_count_numeric_message(self):
        self.p2s.logDtypeKeyedCount('Foo', 'amt', True)
        self.assertEqual(len(self.countLogs()), 1)
        _m_ = self.countLogs()[0]
        self.assertTrue(_m_.startswith('Foo:'))
        self.assertIn("count='amt'", _m_)
        self.assertIn('sum()', _m_)
        self.assertIn('SETp', _m_)

    def test_count_non_numeric_message(self):
        self.p2s.logDtypeKeyedCount('Foo', 'name', False)
        _m_ = self.countLogs()[0]
        self.assertIn('n_unique()', _m_)
        self.assertIn('SCALARp', _m_)

    def test_color_numeric_message(self):
        self.p2s.logDtypeKeyedColor('Foo', 'amt', True)
        _m_ = self.colorLogs()[0]
        self.assertIn('magnitude spectrum', _m_)
        self.assertIn('CSETp', _m_)

    def test_color_non_numeric_message(self):
        self.p2s.logDtypeKeyedColor('Foo', 'name', False)
        _m_ = self.colorLogs()[0]
        self.assertIn('categorical', _m_)
        self.assertIn('CMAGNITUDE_SUMp', _m_)

    def test_once_semantics_same_message(self):
        self.p2s.logDtypeKeyedCount('Foo', 'amt', True)
        self.p2s.logDtypeKeyedCount('Foo', 'amt', True)
        self.assertEqual(len(self.countLogs()), 1)   # deduped by OnceFilter

    def test_distinct_fields_log_separately(self):
        self.p2s.logDtypeKeyedCount('Foo', 'amt', True)
        self.p2s.logDtypeKeyedCount('Foo', 'qty', True)
        self.assertEqual(len(self.countLogs()), 2)

    def test_flip_interpretation_logs_both(self):
        # Same field, both interpretations (a schema flip) -> two distinct logs.
        self.p2s.logDtypeKeyedCount('Foo', 'id', False)
        self.p2s.logDtypeKeyedCount('Foo', 'id', True)
        self.assertEqual(len(self.countLogs()), 2)

    def test_info_off_by_default_is_quiet(self):
        # Drop back to WARNING: the INFO logs must not reach the handler at all.
        self.logger.setLevel(logging.WARNING)
        self.p2s.logDtypeKeyedCount('Foo', 'amt', True)
        self.p2s.logDtypeKeyedColor('Foo', 'amt', True)
        self.assertEqual(self.countLogs(), [])
        self.assertEqual(self.colorLogs(), [])


# ── histop (count + color) ──────────────────────────────────────────────────

class TestHistop(_LogTestBase_):

    def test_count_numeric(self):
        self.p2s.histop(_BAR_DF_, 'cat', count='value')
        self.assertLoggedNumericCount('Histop', 'value')

    def test_count_non_numeric(self):
        self.p2s.histop(_BAR_DF_, 'cat', count='group')
        self.assertLoggedNonNumericCount('Histop', 'group')

    def test_count_default_row_count_no_log(self):
        self.p2s.histop(_BAR_DF_, 'cat')
        self.assertNoCountLog()

    def test_count_explicit_setp_no_log(self):
        self.p2s.histop(_BAR_DF_, 'cat', count=('value', self.p2s.SETp))
        self.assertNoCountLog()

    def test_count_multi_field_tuple_no_log(self):
        self.p2s.histop(_BAR_DF_, 'cat', count=('group', 'value'))
        self.assertNoCountLog()

    def test_color_numeric(self):
        self.p2s.histop(_BAR_DF_, 'cat', color='value')
        self.assertLoggedNumericColor('Histop', 'value')

    def test_color_non_numeric(self):
        self.p2s.histop(_BAR_DF_, 'cat', color='group')
        self.assertLoggedNonNumericColor('Histop', 'group')

    def test_color_explicit_cset_no_log(self):
        self.p2s.histop(_BAR_DF_, 'cat', color=('value', self.p2s.CSETp))
        self.assertNoColorLog()

    def test_color_explicit_magnitude_no_dtype_log(self):
        self.p2s.histop(_BAR_DF_, 'cat', color=('value', self.p2s.CMAGNITUDE_SUMp))
        self.assertNoColorLog()

    def test_once_across_two_renders(self):
        self.p2s.histop(_BAR_DF_, 'cat', count='value')
        self.p2s.histop(_BAR_DF_, 'cat', count='value')
        self.assertEqual(len(self.countLogs()), 1)


# ── timep (count + color) ───────────────────────────────────────────────────

class TestTimep(_LogTestBase_):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tdf = makeTimeDf(n=40)

    def test_count_numeric(self):
        self.p2s.timep(self.tdf, 'ts', count='value')
        self.assertLoggedNumericCount('Timep', 'value')

    def test_count_non_numeric(self):
        self.p2s.timep(self.tdf, 'ts', count='category')
        self.assertLoggedNonNumericCount('Timep', 'category')

    def test_count_default_no_log(self):
        self.p2s.timep(self.tdf, 'ts')
        self.assertNoCountLog()

    def test_color_numeric(self):
        self.p2s.timep(self.tdf, 'ts', color='value')
        self.assertLoggedNumericColor('Timep', 'value')

    def test_color_non_numeric(self):
        self.p2s.timep(self.tdf, 'ts', color='category')
        self.assertLoggedNonNumericColor('Timep', 'category')

    def test_color_explicit_cset_no_log(self):
        self.p2s.timep(self.tdf, 'ts', color=('value', self.p2s.CSETp))
        self.assertNoColorLog()


# ── piep (count + color) ────────────────────────────────────────────────────

class TestPiep(_LogTestBase_):

    def test_count_numeric(self):
        self.p2s.piep(_BAR_DF_, 'cat', count='value')
        self.assertLoggedNumericCount('Piep', 'value')

    def test_count_non_numeric(self):
        self.p2s.piep(_BAR_DF_, 'cat', count='group')
        self.assertLoggedNonNumericCount('Piep', 'group')

    def test_color_numeric(self):
        self.p2s.piep(_BAR_DF_, 'cat', color='value')
        self.assertLoggedNumericColor('Piep', 'value')

    def test_color_non_numeric(self):
        self.p2s.piep(_BAR_DF_, 'cat', color='group')
        self.assertLoggedNonNumericColor('Piep', 'group')

    def test_color_explicit_cset_no_log(self):
        self.p2s.piep(_BAR_DF_, 'cat', color=('value', self.p2s.CSETp))
        self.assertNoColorLog()

    def test_color_multi_field_no_log(self):
        # multi-field color takes the categorical set path without a dtype probe
        self.p2s.piep(_BAR_DF_, 'cat', color=('cat', 'group'))
        self.assertNoColorLog()


# ── linkp (count + color) ───────────────────────────────────────────────────

class TestLinkP(_LogTestBase_):

    def test_count_numeric(self):
        self.p2s.linkp(_GRAPH_DF_, relationships=_RELS_, pos=_POS_, count='w', node_size='vary')
        self.assertLoggedNumericCount('LinkP', 'w')

    def test_count_default_no_log(self):
        self.p2s.linkp(_GRAPH_DF_, relationships=_RELS_, pos=_POS_)
        self.assertNoCountLog()

    def test_color_numeric(self):
        self.p2s.linkp(_GRAPH_DF_, relationships=_RELS_, pos=_POS_, color='w')
        self.assertLoggedNumericColor('LinkP', 'w')

    def test_color_non_numeric(self):
        self.p2s.linkp(_GRAPH_DF_, relationships=_RELS_, pos=_POS_, color='fm')
        self.assertLoggedNonNumericColor('LinkP', 'fm')

    def test_color_explicit_cset_no_log(self):
        self.p2s.linkp(_GRAPH_DF_, relationships=_RELS_, pos=_POS_, color=('w', self.p2s.CSETp))
        self.assertNoColorLog()


# ── chordp (count + color) ──────────────────────────────────────────────────

class TestChordp(_LogTestBase_):

    def test_count_numeric(self):
        self.p2s.chordp(_GRAPH_DF_, _RELS_, count='w', node_size='vary')
        self.assertLoggedNumericCount('Chordp', 'w')

    def test_count_default_no_log(self):
        self.p2s.chordp(_GRAPH_DF_, _RELS_)
        self.assertNoCountLog()

    def test_color_numeric(self):
        self.p2s.chordp(_GRAPH_DF_, _RELS_, color='w')
        self.assertLoggedNumericColor('Chordp', 'w')

    def test_color_non_numeric(self):
        self.p2s.chordp(_GRAPH_DF_, _RELS_, color='fm')
        self.assertLoggedNonNumericColor('Chordp', 'fm')


# ── spreadlinesp (count only) ───────────────────────────────────────────────

class TestSpreadLinesP(_LogTestBase_):

    def test_count_numeric(self):
        self.p2s.spreadlinesp(_GRAPH_DF_, _RELS_, ego='a', time='time', count='w')
        self.assertLoggedNumericCount('SpreadLinesP', 'w')

    def test_count_default_no_log(self):
        self.p2s.spreadlinesp(_GRAPH_DF_, _RELS_, ego='a', time='time')
        self.assertNoCountLog()


if __name__ == '__main__':
    unittest.main()
