#
# test_singleton_init.py - Polars2SVG is a singleton, but Python calls __init__ on
# every Polars2SVG() call.  These tests pin the once-only guard: repeated
# instantiation (including implicit instantiation inside every component
# constructor) must not re-run the init -- most visibly, it must not stack a new
# OnceFilter onto the shared 'polars2svg_logger' on each call.
#
import logging
import unittest

import polars as pl

from polars2svg import Polars2SVG


def _once_filter_count_(logger):
    return sum(1 for f in logger.filters if type(f).__name__ == 'OnceFilter')


class _CountingHandler_(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []
    def emit(self, record):
        self.records.append(record.getMessage())


class TestSingletonInit(unittest.TestCase):

    def setUp(self):
        self.p2s    = Polars2SVG()
        self.logger = logging.getLogger('polars2svg_logger')

    # ------------------------------------------------------------------
    # Filter accumulation
    # ------------------------------------------------------------------

    def test_exactly_one_once_filter(self):
        self.assertEqual(_once_filter_count_(self.logger), 1)

    def test_repeated_instantiation_does_not_add_filters(self):
        before = len(self.logger.filters)
        for _ in range(200): Polars2SVG()
        self.assertEqual(len(self.logger.filters), before)

    def test_component_construction_does_not_add_filters(self):
        df     = pl.DataFrame({'cat': ['a', 'b', 'a'], 'val': [1, 2, 3]})
        before = len(self.logger.filters)
        for _ in range(10):
            self.p2s.histop(df, bin_by='cat')
            self.p2s.xyp(df, x='val', y='val')
        self.assertEqual(len(self.logger.filters), before)

    # ------------------------------------------------------------------
    # Singleton identity & init-once semantics
    # ------------------------------------------------------------------

    def test_singleton_identity(self):
        self.assertIs(Polars2SVG(), self.p2s)

    def test_state_persists_across_reinstantiation(self):
        self.p2s.reset_defaults()
        try:
            self.p2s.set_defaults(txt_h=17)
            self.p2s.to_color_lu['__test_key__'] = '#123456'
            again = Polars2SVG()
            self.assertEqual(again.get_defaults()['_global']['txt_h'], 17)
            self.assertEqual(again.to_color_lu['__test_key__'], '#123456')
        finally:
            self.p2s.to_color_lu.pop('__test_key__', None)
            self.p2s.reset_defaults()

    def test_reinstantiation_preserves_logger_object(self):
        logger_before = self.p2s.logger
        Polars2SVG()
        self.assertIs(self.p2s.logger, logger_before)

    # ------------------------------------------------------------------
    # "Warn once" behavior
    # ------------------------------------------------------------------

    def test_warn_once_is_actually_once(self):
        handler = _CountingHandler_()
        self.logger.addHandler(handler)
        try:
            msg = 'test_singleton_init: repeated warning __unique_a__'
            self.p2s.logger.warning(msg)
            self.p2s.logger.warning(msg)
            Polars2SVG()                    # re-instantiation must not reset "seen"
            self.p2s.logger.warning(msg)
            self.assertEqual(handler.records.count(msg), 1)
        finally:
            self.logger.removeHandler(handler)

    def test_warn_once_still_passes_new_messages(self):
        handler = _CountingHandler_()
        self.logger.addHandler(handler)
        try:
            msg = 'test_singleton_init: fresh warning __unique_b__'
            self.p2s.logger.warning(msg)
            self.assertEqual(handler.records.count(msg), 1)
        finally:
            self.logger.removeHandler(handler)

    # ------------------------------------------------------------------
    # Singleton reset (test / reload scenario): the shared module-level logger
    # must not accumulate filters across fresh singletons either
    # ------------------------------------------------------------------

    def test_singleton_reset_does_not_leak_filters(self):
        original = Polars2SVG._instance_
        try:
            Polars2SVG._instance_ = None
            fresh = Polars2SVG()
            self.assertIsNot(fresh, original)
            self.assertEqual(_once_filter_count_(self.logger), 1)
        finally:
            Polars2SVG._instance_ = original


if __name__ == '__main__':
    unittest.main()
