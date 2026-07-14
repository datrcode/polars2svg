#
# test_optional_dependency_extras.py
#
# Optional-dependency extras keep the core install slim.
# Core (polars, numpy, pyarrow, pillow, platformdirs) must import and render
# xyp/histop/etc. with none of the `layouts` extra (networkx, scikit-learn,
# scipy, shapely, squarify) or `interactive` extra (panel, jupyter_bokeh,
# param) installed; features that genuinely need them must raise a clear
# ImportError naming the extra, not a bare ModuleNotFoundError or a crash at
# `import polars2svg` time.
#
# The dev venv has every extra installed, so this test simulates their
# absence by blocking the relevant module names via a sys.meta_path finder
# and purging any already-cached copies from sys.modules, then re-importing
# polars2svg fresh. sys.modules is fully snapshotted/restored in
# tearDown so no other test (many of which import polars2svg at module load
# time and rely on linkp/chordp/interactive features) is affected.
#
import importlib
import sys
import unittest

import polars as pl

# Warm up polars2svg (and its core deps -- numpy, pyarrow, ...) once at module
# load time. numpy's C extension cannot be re-initialized once loaded ("cannot
# load module more than once per process"), so if the very first setUp() ran
# before numpy was ever imported, its sys.modules snapshot wouldn't include
# 'numpy'; tearDown() would then delete it, and the next test's fresh
# `import polars2svg` would try to genuinely re-import numpy and crash. Doing
# a real import here guarantees every snapshot below already has it.
import polars2svg as _p2s_warmup_  # noqa: F401

_LAYOUTS_EXTRA_MODULES = ('networkx', 'sklearn', 'scipy', 'shapely', 'squarify')
_INTERACTIVE_EXTRA_MODULES = ('panel', 'param', 'jupyter_bokeh', 'bokeh')
_BLOCKED = _LAYOUTS_EXTRA_MODULES + _INTERACTIVE_EXTRA_MODULES


class _BlockImportsFinder:
    # A sys.meta_path finder that turns "import <blocked module>" into an
    # ImportError, simulating an environment where it isn't installed.
    def __init__(self, blocked):
        self._blocked = tuple(blocked)

    def find_spec(self, fullname, path, target=None):
        if fullname in self._blocked or any(fullname.startswith(b + '.') for b in self._blocked):
            raise ImportError(f"'{fullname}' blocked for optional-dependency test")
        return None


class TestOptionalDependencyExtras(unittest.TestCase):

    def setUp(self):
        self._orig_modules = dict(sys.modules)
        for _name_ in list(sys.modules):
            if _name_ == 'polars2svg' or _name_.startswith('polars2svg.') \
               or _name_ in _BLOCKED or any(_name_.startswith(b + '.') for b in _BLOCKED):
                del sys.modules[_name_]
        self._finder = _BlockImportsFinder(_BLOCKED)
        sys.meta_path.insert(0, self._finder)
        self.p2s_module = importlib.import_module('polars2svg')

    def tearDown(self):
        sys.meta_path.remove(self._finder)
        sys.modules.clear()
        sys.modules.update(self._orig_modules)

    def test_import_succeeds_without_layouts_or_interactive(self):
        # Sanity check that the blocking actually took effect.
        with self.assertRaises(ImportError):
            importlib.import_module('networkx')
        with self.assertRaises(ImportError):
            importlib.import_module('panel')
        # ...yet the package itself imported fine in setUp.
        self.assertTrue(hasattr(self.p2s_module, 'Polars2SVG'))

    def test_layout_algorithm_classes_absent_without_layouts_extra(self):
        # __init__.py guards these the same way it already guarded TFDPLayout;
        # they simply shouldn't exist rather than raising at import time.
        for _name_ in ('PolarsForceDirectedLayout', 'ConveyProximityLayout',
                       'LandmarkMDSLayout', 'PivotMDSLayout', 'TFDPLayout'):
            self.assertFalse(hasattr(self.p2s_module, _name_),
                              f'{_name_} should be absent without the layouts extra')

    def test_xyp_renders_without_layouts_or_interactive(self):
        p2s = self.p2s_module.Polars2SVG()
        df = pl.DataFrame({'x': [1, 2, 3, 4], 'y': [3, 1, 4, 1], 'g': ['a', 'b', 'a', 'b']})
        chart = p2s.xyp(df, 'x', 'y', color='g')
        self.assertIn('<svg', chart.svg)

    def test_histop_renders_without_layouts_or_interactive(self):
        p2s = self.p2s_module.Polars2SVG()
        df = pl.DataFrame({'g': ['a', 'b', 'a', 'c', 'b', 'a']})
        chart = p2s.histop(df, 'g')
        self.assertIn('<svg', chart.svg)

    def test_linkp_basic_render_without_layouts(self):
        # No background=/pos= supplied, so linkp never touches shapely or
        # the networkx-backed graph-layout mixin -- only its own random
        # fallback positions.
        p2s = self.p2s_module.Polars2SVG()
        df = pl.DataFrame({'src': ['a', 'b', 'c'], 'dst': ['b', 'c', 'a']})
        chart = p2s.linkp(df, [('src', 'dst')])
        self.assertIn('<svg', chart.svg)

    def test_chordp_raises_clear_error_without_layouts_extra(self):
        p2s = self.p2s_module.Polars2SVG()
        df = pl.DataFrame({'src': ['a', 'b', 'c'], 'dst': ['b', 'c', 'a']})
        with self.assertRaises(ImportError) as _ctx_:
            p2s.chordp(df, [('src', 'dst')])
        self.assertIn('layouts', str(_ctx_.exception))

    def test_graph_mixin_layout_method_raises_clear_error_without_layouts_extra(self):
        p2s = self.p2s_module.Polars2SVG()
        df = pl.DataFrame({'src': ['a', 'b', 'c'], 'dst': ['b', 'c', 'a']})
        with self.assertRaises(ImportError) as _ctx_:
            p2s.createNetworkXGraph(df, [('src', 'dst')])
        self.assertIn('layouts', str(_ctx_.exception))

    def test_panelize_raises_clear_error_without_interactive_extra(self):
        p2s = self.p2s_module.Polars2SVG()
        df = pl.DataFrame({'x': [1, 2, 3], 'y': [3, 1, 4]})
        chart = p2s.xyp(df, 'x', 'y')
        with self.assertRaises(ImportError) as _ctx_:
            p2s.xypi(chart)
        self.assertIn('interactive', str(_ctx_.exception))

    def test_isTemplate_does_not_require_chordp(self):
        # isTemplate()'s isinstance check against ChP must not itself require
        # the layouts extra just to evaluate.
        p2s = self.p2s_module.Polars2SVG()
        df = pl.DataFrame({'x': [1, 2, 3], 'y': [3, 1, 4]})
        chart = p2s.xyp(df, 'x', 'y')
        self.assertTrue(p2s.isTemplate(chart))
        self.assertFalse(p2s.isTemplate('not a component'))


if __name__ == '__main__':
    unittest.main()
