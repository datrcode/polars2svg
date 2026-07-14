import unittest

import importlib.util
import inspect
from pathlib import Path

import polars2svg
from polars2svg import (
    Polars2SVG,
    PolarsForceDirectedLayout,
    ConveyProximityLayout,
    LandmarkMDSLayout,
    PivotMDSLayout,
)


#
# The package is typed: it ships a py.typed marker so
# downstream type checkers read the hints, and the public surface (constructor,
# component factory methods, tField, panelize, exported layout classes) carries
# return annotations. These tests guard the marker's presence and that the
# annotations are not accidentally stripped.
#
class TestPyTypedMarker(unittest.TestCase):

    def test_py_typed_ships_beside_package(self):
        # py.typed must live inside the installed package directory (that is what
        # PEP 561 tools look for and what hatchling bundles into the wheel).
        pkg_dir = Path(inspect.getfile(polars2svg)).parent
        self.assertTrue((pkg_dir / 'py.typed').is_file(),
                        f'py.typed marker missing from {pkg_dir}')

    def test_py_typed_locatable_via_importlib(self):
        # Same check via the loader, independent of how the package was installed.
        spec = importlib.util.find_spec('polars2svg')
        self.assertIsNotNone(spec and spec.origin)
        pkg_dir = Path(spec.origin).parent
        self.assertTrue((pkg_dir / 'py.typed').is_file())


class TestPublicSurfaceAnnotations(unittest.TestCase):

    FACTORY_RETURNS = {
        'xyp':          'XYp',
        'histop':       'Histop',
        'timep':        'Timep',
        'linkp':        'LinkP',
        'chordp':       'ChP',
        'piep':         'Piep',
        'spreadlinesp': 'SpreadLinesP',
        'smallp':       'Smallp',
    }

    def test_factory_methods_have_return_annotation(self):
        for name, expected in self.FACTORY_RETURNS.items():
            with self.subTest(method=name):
                ann = inspect.signature(getattr(Polars2SVG, name)).return_annotation
                self.assertIsNot(ann, inspect.Signature.empty,
                                 f'{name}() is missing a return annotation')
                # Annotation is the class object itself (evaluated at def time).
                self.assertEqual(getattr(ann, '__name__', ann), expected)

    def test_init_annotated_returns_none(self):
        ann = inspect.signature(Polars2SVG.__init__).return_annotation
        self.assertIs(ann, None)

    def test_tfield_annotated(self):
        sig = inspect.signature(Polars2SVG.tField)
        self.assertIsNot(sig.return_annotation, inspect.Signature.empty)
        self.assertIsNot(sig.parameters['column'].annotation, inspect.Parameter.empty)

    def test_panelize_annotated(self):
        sig = inspect.signature(Polars2SVG.panelize)
        self.assertIsNot(sig.return_annotation, inspect.Signature.empty)
        self.assertIsNot(sig.parameters['stack'].annotation, inspect.Parameter.empty)

    def test_exported_layouts_results_return_dict(self):
        for cls in (PolarsForceDirectedLayout, ConveyProximityLayout,
                    LandmarkMDSLayout, PivotMDSLayout):
            with self.subTest(cls=cls.__name__):
                ann = inspect.signature(cls.results).return_annotation
                self.assertIs(ann, dict,
                              f'{cls.__name__}.results() should be annotated -> dict')


if __name__ == '__main__':
    unittest.main()
