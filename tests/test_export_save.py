import io
import os
import sys
import tempfile
import unittest
from unittest import mock

import polars as pl

from polars2svg import Polars2SVG
from polars2svg.export import ExportMixin, svgToPNGBytes

# The eight rendered component classes that should all gain save()/savePNG(),
# plus Tile (which composes finished renderings rather than a DataFrame).
from polars2svg.xyp import XYp
from polars2svg.histop import Histop
from polars2svg.timep import Timep
from polars2svg.piep import Piep
from polars2svg.linkp import LinkP
from polars2svg.chordp import ChP
from polars2svg.smallp import Smallp
from polars2svg.spreadlinesp import SpreadLinesP
from polars2svg.tile import Tile


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
        for cls in (XYp, Histop, Timep, Piep, LinkP, ChP, Smallp, SpreadLinesP, Tile):
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


@unittest.skipUnless(_svglib_available(), 'svglib/reportlab not installed (optional [export] extra)')
class TestRasterizeDoesNotReadLocalFiles(unittest.TestCase):
    '''PNG export must not turn an external reference in the SVG into a file read.

    tile() embeds foreign SVG **verbatim** -- that is the component, not an
    oversight (PLANNING.md A3, SECURITY.md) -- so a document arriving at
    svgToPNGBytes() can carry an <image>/<use> href that polars2svg never wrote.
    svglib resolves such an href against the *source path* of the document it
    parsed, so rasterizing from a file path would read that file off disk and
    composite its contents into the output PNG.  export.py hands svglib an
    io.StringIO instead, which leaves SvgRenderer.source_path a non-str and makes
    xlink_href_target() decline to resolve a path at all.

    These tests pin the observable end of that -- the referenced file's pixels
    never reach the PNG -- so they fail either way it could break: if
    svgToPNGBytes() is refactored to rasterize from a path, or if a future svglib
    inside the [export] extra's range (>=1.5,<2) drops the source_path check.
    See the block comment above svgToPNGBytes() in polars2svg/export.py.
    '''

    # The secret file is solid red and nothing polars2svg draws here is, so "a red
    # pixel appears in the output" is exactly "the file was read".
    _SECRET_RGB_ = (255, 0, 0)

    def setUp(self):
        self.p2s = Polars2SVG()
        self._tmp_ = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp_.cleanup)
        self.dir = self._tmp_.name
        from PIL import Image
        self.secret_png = os.path.join(self.dir, 'secret.png')
        Image.new('RGB', (40, 40), self._SECRET_RGB_).save(self.secret_png)
        # An external SVG for the <use>/external-reference vector.
        self.secret_svg = os.path.join(self.dir, 'secret.svg')
        with open(self.secret_svg, 'w', encoding='utf-8') as f:
            f.write('<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">'
                    '<g id="g1"><rect x="0" y="0" width="100" height="100" fill="red"/></g>'
                    '</svg>')

    def _tiled(self, body):
        '''Compose `body` as a foreign child SVG through tile(), as a caller would.'''
        _foreign_ = ('<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg" '
                     'xmlns:xlink="http://www.w3.org/1999/xlink">' + body + '</svg>')
        return self.p2s.tile([_foreign_])._repr_svg_()

    def _secret_pixels(self, png_bytes):
        '''Count pixels matching the secret file's color in a rendered PNG.'''
        from PIL import Image
        _im_ = Image.open(io.BytesIO(png_bytes)).convert('RGB')
        _counts_ = _im_.getcolors(maxcolors=1 << 24) or []
        return sum(_n_ for _n_, _px_ in _counts_ if _px_ == self._SECRET_RGB_)

    def test_positive_control_embedded_image_is_composited(self):
        # Without this, every assertion below could pass vacuously -- e.g. if
        # svglib silently ignored <image> altogether, or the fixture were not red.
        # A data: URI is the one image source svglib resolves with no filesystem
        # access, so this proves the detection works without relying on the leak.
        import base64
        with open(self.secret_png, 'rb') as f:
            _b64_ = base64.b64encode(f.read()).decode('ascii')
        _svg_ = self._tiled(f'<image x="0" y="0" width="100" height="100" '
                            f'xlink:href="data:image/png;base64,{_b64_}"/>')
        self.assertGreater(self._secret_pixels(svgToPNGBytes(_svg_)), 100,
                           'positive control failed: svglib did not composite an '
                           'embedded <image>, so the leak checks below prove nothing')

    def test_external_references_are_not_resolved(self):
        _rel_png_ = os.path.basename(self.secret_png)
        _cases_ = {
            # <image>, every spelling of "read this file off disk"
            'image relative':      f'<image x="0" y="0" width="100" height="100" xlink:href="{_rel_png_}"/>',
            'image absolute':      f'<image x="0" y="0" width="100" height="100" xlink:href="{self.secret_png}"/>',
            'image traversal':     f'<image x="0" y="0" width="100" height="100" xlink:href="../../../..{self.secret_png}"/>',
            'image bare href':     f'<image x="0" y="0" width="100" height="100" href="{_rel_png_}"/>',
            'image file uri':      f'<image x="0" y="0" width="100" height="100" xlink:href="file://{self.secret_png}"/>',
            # <use> pulling in an external document, with and without a fragment
            'use external frag':   f'<use xlink:href="{self.secret_svg}#g1" x="0" y="0"/>',
            'use external whole':  f'<use xlink:href="{self.secret_svg}" x="0" y="0"/>',
        }
        # The relative spellings resolve against the rasterized document's own
        # directory, so run from the directory holding the secret: that is the
        # worst case for us and the one a naive temp-file refactor would produce.
        _cwd_ = os.getcwd()
        os.chdir(self.dir)
        try:
            for _name_, _body_ in _cases_.items():
                with self.subTest(vector=_name_):
                    _png_ = svgToPNGBytes(self._tiled(_body_))
                    self.assertTrue(_png_.startswith(_PNG_MAGIC_))
                    self.assertEqual(
                        self._secret_pixels(_png_), 0,
                        f'{_name_}: contents of a local file reached the exported '
                        f'PNG -- the rasterizer resolved an external reference in '
                        f'foreign SVG as a filesystem path',
                    )
        finally:
            os.chdir(_cwd_)

    def test_rasterizer_is_not_given_a_filesystem_path(self):
        # The behavioural checks above are the real guarantee, but they only fail
        # once BOTH halves break.  This one names the invariant directly, so a
        # refactor to svg2rlg(<path>) fails here with the reason attached even on
        # an svglib that still happens to refuse the resolution.
        # svgToPNGBytes() imports svg2rlg at call time, so patching the module
        # attribute is enough to see what it hands over.
        import svglib.svglib
        _real_, _seen_ = svglib.svglib.svg2rlg, []

        def _spy_(source, *args, **kwargs):
            _seen_.append(source)
            return _real_(source, *args, **kwargs)

        with mock.patch.object(svglib.svglib, 'svg2rlg', _spy_):
            svgToPNGBytes(self.p2s.tile([
                '<svg width="10" height="10" xmlns="http://www.w3.org/2000/svg"/>'
            ])._repr_svg_())

        self.assertEqual(len(_seen_), 1)
        self.assertNotIsInstance(
            _seen_[0], (str, bytes, os.PathLike),
            'svgToPNGBytes() must hand svglib a stream, not a path: svglib resolves '
            'xlink:href on <image>/<use> against the source document\'s path, so a '
            'path here makes foreign SVG passed to tile() a local file read. '
            'See the comment above svgToPNGBytes() in polars2svg/export.py.',
        )


if __name__ == '__main__':
    unittest.main()
