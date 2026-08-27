import logging
import unittest
import networkx as nx

from polars2svg.polars_force_directed_layout import PolarsForceDirectedLayout
from polars2svg.convey_proximity_layout      import ConveyProximityLayout


def _g():
    return nx.path_graph(6)


class TestDeprecatedLayoutWarnings(unittest.TestCase):
    """Both classes are deprecated in 0.2.0 and removed in 0.3.0 (see CHANGELOG).

    The warning fires on instantiation rather than at import: __init__.py imports both
    modules eagerly, so a module-level warning would reach every `import polars2svg`
    including users who never touch the classes.
    """

    def setUp(self):
        # The shared logger carries a OnceFilter (installed by Polars2SVG.__init__, which
        # conftest.py triggers at import), so each distinct message is emitted once per
        # process.  Clear its memo so these tests do not depend on execution order --
        # without this, whichever test instantiates first consumes the warning and every
        # later assertLogs sees nothing.
        for _f_ in logging.getLogger('polars2svg_logger').filters:
            if hasattr(_f_, 'seen_messages'):
                _f_.seen_messages.clear()

    def test_force_directed_warns_on_instantiation(self):
        with self.assertLogs('polars2svg_logger', level='WARNING') as _cm_:
            PolarsForceDirectedLayout(_g(), iterations=2)
        _msg_ = '\n'.join(_cm_.output)
        self.assertIn('PolarsForceDirectedLayout is deprecated', _msg_)
        self.assertIn('0.3.0', _msg_)
        self.assertIn('spring_layout', _msg_)

    def test_convey_proximity_warns_on_instantiation(self):
        with self.assertLogs('polars2svg_logger', level='WARNING') as _cm_:
            ConveyProximityLayout(_g(), iterations_min=2)
        _msg_ = '\n'.join(_cm_.output)
        self.assertIn('ConveyProximityLayout is deprecated', _msg_)
        self.assertIn('0.3.0', _msg_)

    def test_force_directed_warning_names_the_distance_caveat(self):
        # spring_layout is Fruchterman-Reingold, not Cohen's stress majorization, so the
        # replacement is not distance-preserving.  The warning has to say so or it sends
        # people to a layout that quietly does something else.
        with self.assertLogs('polars2svg_logger', level='WARNING') as _cm_:
            PolarsForceDirectedLayout(_g(), iterations=2)
        self.assertIn('LandmarkMDSLayout', '\n'.join(_cm_.output))

    def test_convey_proximity_warning_names_the_resistive_distance_loss(self):
        # Resistive (effective-resistance) distances have no replacement anywhere in the
        # framework; that is the one real capability the removal costs.
        with self.assertLogs('polars2svg_logger', level='WARNING') as _cm_:
            ConveyProximityLayout(_g(), iterations_min=2)
        self.assertIn('Resistive', '\n'.join(_cm_.output))

    def test_both_still_import_and_run_this_release(self):
        # Deprecated, not removed: 0.2.0 keeps them working so callers get one release of
        # warning rather than an ImportError on upgrade.
        self.assertEqual(len(PolarsForceDirectedLayout(_g(), iterations=2).results()), 6)
        self.assertEqual(len(ConveyProximityLayout(_g(), iterations_min=2).results()),  6)


if __name__ == '__main__':
    unittest.main()
