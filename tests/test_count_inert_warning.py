#
# test_count_inert_warning.py
#
# Warn when count= is inert: linkp and chordp accept count= but, depending on
# other parameters, may never consume it.  Each component emits a one-time
# logger.warning when count= is explicitly set but nothing reads it:
#
#   - linkp:  count= is only consumed by node_size='vary' / link_size='vary'.
#             CROW_* colors use raw row count and the fallback layout is random,
#             so neither suppresses the warning.
#   - chordp: same 'vary' rule, PLUS the derived node order (leafWalkFromEdges)
#             consumes count as edge weights -- so the warning only fires when
#             the order is pinned via order= or pos=.
#   - spreadlinesp: count= (non-default) always drives the within-bin circle
#             sort order, so it never warns.
#
# Each warning test also carries a ground-truth companion asserting the actual
# render behavior (inert -> identical SVG; consumed -> different SVG), so the
# warning conditions can't drift from reality silently.
#
import logging
import unittest

import polars as pl

from polars2svg import Polars2SVG

_INERT_MSG_FRAGMENT_ = 'count= is set but has no visible effect'

_DF_ = pl.DataFrame({
    'fm':   ['a', 'b', 'c', 'a', 'd', 'b', 'e', 'c', 'd', 'e'],
    'to':   ['b', 'c', 'a', 'c', 'a', 'a', 'b', 'd', 'e', 'a'],
    'time': [1,   1,   1,   2,   2,   3,   3,   1,   2,   3  ],
    'w':    [30,  1,   2,   40,  1,   2,   50,  7,   9,   11 ],
})
_RELS_  = [('fm', 'to')]
_POS_   = {'a': (0, 0), 'b': (1, 0), 'c': (1, 1), 'd': (0, 1), 'e': (0.5, 0.5)}
_ORDER_ = ['a', 'b', 'c', 'd', 'e']


class _CountingHandler_(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []
    def emit(self, record):
        self.records.append(record.getMessage())


class _WarningTestBase_(unittest.TestCase):
    """Installs a counting handler and resets the OnceFilter's seen-message set
    so every test observes warn-once behavior from a clean slate (test order
    independence)."""

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

    def inertWarnings(self):
        return [m for m in self.handler.records if _INERT_MSG_FRAGMENT_ in m]

    def assertWarned(self, component_prefix):
        _msgs_ = self.inertWarnings()
        self.assertEqual(len(_msgs_), 1, f'expected exactly one inert-count warning, got {_msgs_}')
        self.assertTrue(_msgs_[0].startswith(component_prefix),
                        f'warning does not start with {component_prefix!r}: {_msgs_[0]}')

    def assertNotWarned(self):
        self.assertEqual(self.inertWarnings(), [])


# ---------------------------------------------------------------------------
# LinkP
# ---------------------------------------------------------------------------

class TestLinkPCountInertWarning(_WarningTestBase_):

    def test_warns_at_default_sizes(self):
        self.p2s.linkp(_DF_, relationships=_RELS_, pos=_POS_, count='w')
        self.assertWarned('LinkP')

    def test_no_warning_without_count(self):
        self.p2s.linkp(_DF_, relationships=_RELS_, pos=_POS_)
        self.assertNotWarned()

    def test_no_warning_with_explicit_row_count_default(self):
        self.p2s.linkp(_DF_, relationships=_RELS_, pos=_POS_, count=self.p2s.ROW_COUNTp)
        self.assertNotWarned()

    def test_no_warning_with_node_size_vary(self):
        self.p2s.linkp(_DF_, relationships=_RELS_, pos=_POS_, count='w', node_size='vary')
        self.assertNotWarned()

    def test_no_warning_with_link_size_vary(self):
        self.p2s.linkp(_DF_, relationships=_RELS_, pos=_POS_, count='w', link_size='vary')
        self.assertNotWarned()

    def test_warns_with_crow_color(self):
        # CROW_* colors by raw row count (not count=), so count= is still inert
        self.p2s.linkp(_DF_, relationships=_RELS_, pos=_POS_, count='w',
                       color=self.p2s.CROW_MAGNITUDEp)
        self.assertWarned('LinkP')

    def test_warns_with_tuple_count(self):
        self.p2s.linkp(_DF_, relationships=_RELS_, pos=_POS_, count=('fm', self.p2s.SETp))
        self.assertWarned('LinkP')

    def test_warning_fires_once_across_instances(self):
        self.p2s.linkp(_DF_, relationships=_RELS_, pos=_POS_, count='w')
        self.p2s.linkp(_DF_, relationships=_RELS_, pos=_POS_, count='w')
        self.assertWarned('LinkP')   # exactly one despite two renders

    def test_no_warning_for_dataless_template(self):
        # A template holds parameters only; nothing renders, so nothing warns.
        self.p2s.linkp(relationships=_RELS_, pos=_POS_, count='w')
        self.assertNotWarned()

    def test_template_clone_with_baked_count_warns(self):
        _tmpl_ = self.p2s.linkp(relationships=_RELS_, pos=_POS_, count='w')
        self.assertNotWarned()
        self.p2s.linkp(_DF_, template=_tmpl_)
        self.assertWarned('LinkP')

    # --- ground truth: the warning matches actual render behavior ---

    def test_count_is_actually_inert_at_default_sizes(self):
        _a_ = self.p2s.linkp(_DF_, relationships=_RELS_, pos=_POS_)
        _b_ = self.p2s.linkp(_DF_, relationships=_RELS_, pos=_POS_, count='w')
        self.assertEqual(_a_.svg, _b_.svg)

    def test_count_is_actually_inert_with_crow_color(self):
        _a_ = self.p2s.linkp(_DF_, relationships=_RELS_, pos=_POS_, color=self.p2s.CROW_MAGNITUDEp)
        _b_ = self.p2s.linkp(_DF_, relationships=_RELS_, pos=_POS_, color=self.p2s.CROW_MAGNITUDEp, count='w')
        self.assertEqual(_a_.svg, _b_.svg)

    def test_count_is_actually_consumed_by_node_size_vary(self):
        _a_ = self.p2s.linkp(_DF_, relationships=_RELS_, pos=_POS_, node_size='vary')
        _b_ = self.p2s.linkp(_DF_, relationships=_RELS_, pos=_POS_, node_size='vary', count='w')
        self.assertNotEqual(_a_.svg, _b_.svg)

    def test_count_is_actually_consumed_by_link_size_vary(self):
        _a_ = self.p2s.linkp(_DF_, relationships=_RELS_, pos=_POS_, link_size='vary')
        _b_ = self.p2s.linkp(_DF_, relationships=_RELS_, pos=_POS_, link_size='vary', count='w')
        self.assertNotEqual(_a_.svg, _b_.svg)


# ---------------------------------------------------------------------------
# ChordP
# ---------------------------------------------------------------------------

class TestChordPCountInertWarning(_WarningTestBase_):

    def test_no_warning_at_defaults_order_is_derived(self):
        # With no order=/pos=, count feeds the edge-weight clustering that
        # derives the node order -- count is consumed, so no warning.
        self.p2s.chordp(_DF_, relationships=_RELS_, count='w')
        self.assertNotWarned()

    def test_warns_with_explicit_order(self):
        self.p2s.chordp(_DF_, relationships=_RELS_, order=_ORDER_, count='w')
        self.assertWarned('ChP')

    def test_warns_with_pos(self):
        self.p2s.chordp(_DF_, relationships=_RELS_, pos=_POS_, count='w')
        self.assertWarned('ChP')

    def test_no_warning_without_count(self):
        self.p2s.chordp(_DF_, relationships=_RELS_, order=_ORDER_)
        self.assertNotWarned()

    def test_no_warning_with_explicit_row_count_default(self):
        self.p2s.chordp(_DF_, relationships=_RELS_, order=_ORDER_, count=self.p2s.ROW_COUNTp)
        self.assertNotWarned()

    def test_no_warning_with_node_size_vary(self):
        self.p2s.chordp(_DF_, relationships=_RELS_, order=_ORDER_, count='w', node_size='vary')
        self.assertNotWarned()

    def test_no_warning_with_link_size_vary(self):
        self.p2s.chordp(_DF_, relationships=_RELS_, order=_ORDER_, count='w', link_size='vary')
        self.assertNotWarned()

    def test_no_warning_for_dataless_template(self):
        self.p2s.chordp(relationships=_RELS_, order=_ORDER_, count='w')
        self.assertNotWarned()

    # --- ground truth: the warning matches actual render behavior ---

    def test_count_actually_changes_derived_order_at_defaults(self):
        _a_ = self.p2s.chordp(_DF_, relationships=_RELS_)
        _b_ = self.p2s.chordp(_DF_, relationships=_RELS_, count='w')
        self.assertNotEqual(_a_.order, _b_.order)

    def test_count_is_actually_inert_with_explicit_order(self):
        _a_ = self.p2s.chordp(_DF_, relationships=_RELS_, order=_ORDER_)
        _b_ = self.p2s.chordp(_DF_, relationships=_RELS_, order=_ORDER_, count='w')
        self.assertEqual(_a_.svg, _b_.svg)

    def test_count_is_actually_inert_with_pos(self):
        _a_ = self.p2s.chordp(_DF_, relationships=_RELS_, pos=_POS_)
        _b_ = self.p2s.chordp(_DF_, relationships=_RELS_, pos=_POS_, count='w')
        self.assertEqual(_a_.svg, _b_.svg)

    def test_count_is_actually_consumed_by_node_size_vary(self):
        _a_ = self.p2s.chordp(_DF_, relationships=_RELS_, order=_ORDER_, node_size='vary')
        _b_ = self.p2s.chordp(_DF_, relationships=_RELS_, order=_ORDER_, node_size='vary', count='w')
        self.assertNotEqual(_a_.svg, _b_.svg)

    def test_count_is_actually_consumed_by_link_size_vary(self):
        _a_ = self.p2s.chordp(_DF_, relationships=_RELS_, order=_ORDER_, link_size='vary')
        _b_ = self.p2s.chordp(_DF_, relationships=_RELS_, order=_ORDER_, link_size='vary', count='w')
        self.assertNotEqual(_a_.svg, _b_.svg)


# ---------------------------------------------------------------------------
# SpreadLinesP -- count= is always consumed (within-bin sort), never warns
# ---------------------------------------------------------------------------

class TestSpreadLinesPCountNotInert(_WarningTestBase_):

    def _many_alter_df_(self):
        # Enough alters per bin that weight-based sorting visibly reorders them;
        # weights ascend for one bin and descend for the other.
        _alters_ = [f'n{i:02d}' for i in range(12)]
        _fm_, _to_, _t_, _w_ = [], [], [], []
        for i, _alt_ in enumerate(_alters_):
            _fm_.append(_alt_);  _to_.append('ego'); _t_.append(1); _w_.append(100 - i * 7)
            _fm_.append('ego');  _to_.append(_alt_); _t_.append(2); _w_.append(1 + i * 9)
        return pl.DataFrame({'fm': _fm_, 'to': _to_, 'time': _t_, 'w': _w_})

    def test_no_warning_with_count_field(self):
        self.p2s.spreadlinesp(_DF_, _RELS_, ego='a', time='time', count='w')
        self.assertNotWarned()

    def test_count_actually_changes_within_bin_order(self):
        _df_ = self._many_alter_df_()
        _a_  = self.p2s.spreadlinesp(_df_, _RELS_, ego='ego', time='time')
        _b_  = self.p2s.spreadlinesp(_df_, _RELS_, ego='ego', time='time', count='w')
        self.assertNotEqual(_a_.svg, _b_.svg)


if __name__ == '__main__':
    unittest.main()
