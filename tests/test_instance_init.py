#
# test_instance_init.py - per-instance construction semantics for Polars2SVG.
#
# Polars2SVG() used to return a process-wide singleton: __new__ cached one instance,
# Python called __init__ on it again for every call, and the class carried a guard so
# the repeat runs did nothing (most visibly, so they did not stack another OnceFilter
# onto the shared 'polars2svg_logger').  Every call now builds a genuinely separate
# instance, and these tests pin what that has to mean:
#
#   - instances are independent: defaults and color state set on one are invisible to
#     the next, which is what makes an instance-per-session safe;
#   - the logger is NOT independent -- logging.getLogger() hands every instance the
#     same module-global object -- so filter hygiene there is now load-bearing rather
#     than an optimization, and "warn once" has to stay once per process.
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


class TestInstanceInit(unittest.TestCase):

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
    # Instances are independent
    # ------------------------------------------------------------------

    def test_each_call_returns_a_new_instance(self):
        self.assertIsNot(Polars2SVG(), self.p2s)

    def test_state_does_not_leak_into_a_new_instance(self):
        self.p2s.set_defaults(txt_h=17)
        self.p2s.setColorOverrides({'__test_value__': '#123456'})
        self.p2s.to_color_lu['__test_key__'] = '#123456'
        _other_ = Polars2SVG()
        self.assertEqual(_other_.get_defaults()['_global'], {})
        self.assertEqual(_other_.color_overrides_lu, {})
        self.assertNotIn('__test_key__', _other_.to_color_lu)

    def test_state_is_not_shared_by_reference(self):
        # The mutable stores used to be created once and reused, so two instances
        # referenced one dict -- clearing overrides on either cleared both.
        _other_ = Polars2SVG()
        for _attr_ in ('_global_defaults', '_component_defaults', 'to_color_lu',
                       'color_overrides_lu', 'color_type_lu'):
            with self.subTest(attribute=_attr_):
                self.assertIsNot(getattr(self.p2s, _attr_), getattr(_other_, _attr_))

    def test_every_instance_shares_the_logger_object(self):
        self.assertIs(Polars2SVG().logger, self.p2s.logger)

    # ------------------------------------------------------------------
    # "Warn once" behavior -- once per process, across instances
    # ------------------------------------------------------------------

    def test_warn_once_is_actually_once(self):
        handler = _CountingHandler_()
        self.logger.addHandler(handler)
        try:
            msg = 'test_instance_init: repeated warning __unique_a__'
            self.p2s.logger.warning(msg)
            self.p2s.logger.warning(msg)
            Polars2SVG()                    # a new instance must not reset "seen"
            self.p2s.logger.warning(msg)
            self.assertEqual(handler.records.count(msg), 1)
        finally:
            self.logger.removeHandler(handler)

    def test_warn_once_still_passes_new_messages(self):
        handler = _CountingHandler_()
        self.logger.addHandler(handler)
        try:
            msg = 'test_instance_init: fresh warning __unique_b__'
            self.p2s.logger.warning(msg)
            self.assertEqual(handler.records.count(msg), 1)
        finally:
            self.logger.removeHandler(handler)

    def test_many_instances_leave_exactly_one_filter(self):
        for _ in range(5): Polars2SVG()
        self.assertEqual(_once_filter_count_(self.logger), 1)


if __name__ == '__main__':
    unittest.main()
