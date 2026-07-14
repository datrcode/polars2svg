import io
import os
import sys
import tempfile
import unittest
from unittest import mock

import polars as pl

from polars2svg import Polars2SVG
from polars2svg.export import ExportMixin, svgToPNGBytes

# The eight rendered component classes that should all gain save()/savePNG().
from polars2svg.xyp import XYp
from polars2svg.histop import Histop
from polars2svg.timep import Timep
from polars2svg.piep import Piep
from polars2svg.linkp import LinkP
from polars2svg.chordp import ChP
from polars2svg.smallp import Smallp
from polars2svg.spreadlinesp import SpreadLinesP


_PNG_MAGIC_ = b'\x89PNG\r\n\x1a\n'


def _svglib_available():
    try:
        import svglib.svglib  # noqa: F401
        import reportlab.graphics  # noqa: F401
        return True
    except ImportError:
        return False


class TestExportInheritance(unittest.TestCase):
    '''Every rendered component must inherit the export API.'''

    def test_all_components_are_exportable(self):
        for cls in (XYp, Histop, Timep, Piep, LinkP, ChP, Smallp, SpreadLinesP):
            self.assertTrue(issubclass(cls, ExportMixin), f'{cls.__name__} lacks ExportMixin')
            self.assertTrue(hasattr(cls, 'save'))
            self.assertTrue(hasattr(cls, 'savePNG'))


class TestSaveSVG(unittest.TestCase):
    '''save() with a non-.png path writes the SVG document verbatim.'''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def _plot(self):
        _df_ = pl.DataFrame({'cat': ['a', 'b', 'c', 'a', 'b', 'a']})
        return self.p2s.histop(_df_, 'cat', wxh=(128, 128))

    def test_save_svg_roundtrips(self):
        plot = self._plot()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'chart.svg')
            returned = plot.save(path)
            self.assertEqual(returned, path)
            with open(path, encoding='utf-8') as f:
                written = f.read()
        self.assertIn('<svg', written)
        self.assertEqual(written, plot._repr_svg_())

    def test_save_default_extension_is_svg(self):
        # A path with no recognized extension is treated as SVG (no rasterize).
        plot = self._plot()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'chart.out')
            plot.save(path)
            with open(path, encoding='utf-8') as f:
                self.assertIn('<svg', f.read())

    def test_save_accepts_pathlike(self):
        import pathlib
        plot = self._plot()
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / 'chart.svg'
            plot.save(path)
            self.assertTrue(path.exists())


@unittest.skipUnless(_svglib_available(), 'svglib/reportlab not installed (optional [export] extra)')
class TestSavePNG(unittest.TestCase):
    '''savePNG() / save(*.png) rasterize to a real PNG file.'''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def _plot(self):
        _df_ = pl.DataFrame({'cat': ['a', 'b', 'c', 'a', 'b', 'a']})
        return self.p2s.histop(_df_, 'cat', wxh=(128, 128))

    def test_savePNG_writes_png(self):
        plot = self._plot()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'chart.png')
            returned = plot.savePNG(path)
            self.assertEqual(returned, path)
            with open(path, 'rb') as f:
                self.assertEqual(f.read(len(_PNG_MAGIC_)), _PNG_MAGIC_)

    def test_save_png_extension_dispatches_to_png(self):
        # save('*.png') must rasterize, not write raw SVG text.
        plot = self._plot()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'chart.PNG')  # case-insensitive
            plot.save(path)
            with open(path, 'rb') as f:
                self.assertEqual(f.read(len(_PNG_MAGIC_)), _PNG_MAGIC_)

    def test_svgToPNGBytes_returns_png(self):
        plot = self._plot()
        data = svgToPNGBytes(plot._repr_svg_())
        self.assertTrue(data.startswith(_PNG_MAGIC_))


class TestSavePNGMissingDeps(unittest.TestCase):
    '''Without the [export] extra, PNG export must fail with a clear message.'''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def test_clear_import_error(self):
        _df_ = pl.DataFrame({'cat': ['a', 'b', 'c']})
        plot = self.p2s.histop(_df_, 'cat', wxh=(64, 64))
        # Setting the module to None in sys.modules makes `import svglib.svglib`
        # raise ImportError, simulating the extra not being installed.
        with mock.patch.dict(sys.modules, {'svglib': None, 'svglib.svglib': None}):
            with self.assertRaises(ImportError) as ctx:
                plot.savePNG(os.path.join(tempfile.gettempdir(), 'nope.png'))
        self.assertIn('polars2svg[export]', str(ctx.exception))


if __name__ == '__main__':
    unittest.main()
