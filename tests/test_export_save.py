import io
import os
import sys
import tempfile
import unittest
from unittest import mock

import polars as pl

from polars2svg import Polars2SVG
from polars2svg.export import ExportMixin, svgToPNGBytes, _fetchDataURLsOnly_
from polars2svg.p2s_font_metrics import textAdvance

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


def _cairosvg_available():
    # cairocffi raises OSError, not ImportError, when the cairo library itself is missing
    try:
        import cairosvg.surface  # noqa: F401
        return True
    except (ImportError, OSError):
        return False


_SKIP_NO_CAIROSVG_ = 'cairosvg not installed, or cairo not loadable (optional [export] extra)'


def _rgba_(png_bytes):
    from PIL import Image
    return Image.open(io.BytesIO(png_bytes)).convert('RGBA')


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


@unittest.skipUnless(_cairosvg_available(), _SKIP_NO_CAIROSVG_)
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


@unittest.skipUnless(_cairosvg_available(), _SKIP_NO_CAIROSVG_)
class TestTextPathPNG(unittest.TestCase):
    '''Text that follows a path reaches the PNG where the SVG puts it.  svglib drew no
    <textPath> at all, so it used to be flattened into rotated glyphs first (PLANNING.md
    C-textpath-png); cairosvg lays it out itself, and these hold end to end.'''

    _df_ = pl.DataFrame({'fm': list('abcabd'), 'to': list('bcdcda'), 'lbl': list('uvwxyz')})
    _pos_ = {'a': (0.1, 0.1), 'b': (0.9, 0.1), 'c': (0.9, 0.9), 'd': (0.1, 0.9)}

    def _png_(self, svg):
        '''(RGB image, every warning raised while rasterizing svg)'''
        import warnings
        from PIL import Image
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            data = svgToPNGBytes(svg)
        self.assertTrue(data.startswith(_PNG_MAGIC_))
        return Image.open(io.BytesIO(data)).convert('RGB'), [str(x.message) for x in w]

    def assertInkNear(self, with_labels, without_labels, x, y, r):
        '''Some pixel within r of (x, y) differs between the two renders, i.e. a label
        was drawn there.  Everything else in the two renders is the same geometry.'''
        _w_, _h_ = with_labels.size
        _a_, _b_ = with_labels.load(), without_labels.load()
        _hits_ = sum(1 for _px_ in range(max(0, int(x - r)), min(_w_, int(x + r) + 1))
                       for _py_ in range(max(0, int(y - r)), min(_h_, int(y + r) + 1))
                       if (_px_ - x) ** 2 + (_py_ - y) ** 2 <= r * r and _a_[_px_, _py_] != _b_[_px_, _py_])
        self.assertGreater(_hits_, 0, f'no label ink within {r}px of ({x:.1f}, {y:.1f})')

    def test_curve_link_labels_are_drawn_where_the_svg_puts_them(self):
        p2s = Polars2SVG()
        def _lp_(labels):
            return p2s.linkp(df=self._df_, relationships=[('fm', 'to', 'lbl')], link_shape='curve',
                             pos=self._pos_, draw_link_labels=labels, draw_node_labels=False)
        lp = _lp_(True)
        self.assertEqual(lp.svg.count('<textPath'), 6)
        png, msgs = self._png_(lp._repr_svg_())
        self.assertEqual(msgs, [])
        bare, _ = self._png_(_lp_(False)._repr_svg_())
        # _link_label_info_ is each label's centre (the GPU path's straight-run anchor);
        # the curve is symmetric, so the glyph run's middle lands there too
        self.assertEqual(len(lp._link_label_info_), 6)
        for _cx_, _cy_, _deg_, _txt_, _co_ in lp._link_label_info_:
            self.assertInkNear(png, bare, _cx_, _cy_, lp.txt_h)

    def test_circular_chord_labels_are_drawn_on_their_arcs(self):
        from math import cos, sin, radians
        p2s = Polars2SVG()
        def _cp_(labels):
            return p2s.chordp(df=self._df_, relationships=[('fm', 'to')], draw_labels=labels,
                              label_style='circular')
        cp = _cp_(True)
        png, msgs = self._png_(cp._repr_svg_())
        self.assertEqual(msgs, [])
        bare, _ = self._png_(_cp_(False)._repr_svg_())
        _rows_ = cp.df_node.select(['__a0__', '__a1__', '__r__']).iter_rows()
        _n_ = 0
        for _a0_, _a1_, _r_ in _rows_:
            _mid_, _ro_ = radians((_a0_ + _a1_) / 2.0), _r_ + cp.txt_offset + 2
            self.assertInkNear(png, bare, cp.cx + _ro_ * cos(_mid_), cp.cy + _ro_ * sin(_mid_), cp.txt_h)
            _n_ += 1
        self.assertEqual(_n_, 4)

    def test_labels_inside_a_tile_are_drawn(self):
        # tile() nests each rendering as its own <svg>; the labels must survive that too
        p2s = Polars2SVG()
        def _tile_(labels):
            lp = p2s.linkp(df=self._df_, relationships=[('fm', 'to', 'lbl')], link_shape='curve',
                           pos=self._pos_, draw_link_labels=labels, draw_node_labels=False)
            cp = p2s.chordp(df=self._df_, relationships=[('fm', 'to')], draw_labels=labels,
                            label_style='circular')
            return p2s.tile([lp, cp])._repr_svg_()
        self.assertEqual(_tile_(True).count('<textPath'), 10)
        png, msgs = self._png_(_tile_(True))
        bare, _ = self._png_(_tile_(False))
        self.assertEqual(msgs, [])
        _a_, _b_ = png.load(), bare.load()
        _w_, _h_ = png.size
        _diff_ = sum(1 for _x_ in range(_w_) for _y_ in range(_h_) if _a_[_x_, _y_] != _b_[_x_, _y_])
        self.assertGreater(_diff_, 100, 'the tiled labels drew no ink')


@unittest.skipUnless(_cairosvg_available(), _SKIP_NO_CAIROSVG_)
class TestPNGFidelity(unittest.TestCase):
    '''The PNG draws what the SVG says.  Each of these failed under svglib: PLANNING.md
    C-png-opacity, C-png-text-scale and C-png-aspect-ratio, and the face the layout is
    measured in (the cairosvg switch, D2).'''

    @staticmethod
    def _doc_(w, h, body, extra=''):
        return f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" {extra}>{body}</svg>'

    @staticmethod
    def _ink_box_(img, threshold=128):
        '''(x0, y0, x1, y1) of the dark pixels, inclusive.  Composited onto white first:
        a grey conversion drops alpha, which would count a transparent margin as ink.'''
        import numpy as np
        from PIL import Image
        _flat_ = Image.alpha_composite(Image.new('RGBA', img.size, (255, 255, 255, 255)), img.convert('RGBA'))
        _a_ = np.asarray(_flat_.convert('L'))
        _ys_, _xs_ = np.where(_a_ < threshold)
        return int(_xs_.min()), int(_ys_.min()), int(_xs_.max()), int(_ys_.max())

    def test_translucent_fills_are_blended(self):
        _svg_ = self._doc_(40, 40, '<rect width="40" height="40" fill="#ffffff"/>'
                                   '<rect width="20" height="40" fill="#000000" fill-opacity="0.22"/>'
                                   '<rect x="20" width="20" height="40" fill="#000000" opacity="0.5"/>')
        _im_ = _rgba_(svgToPNGBytes(_svg_))
        # 255 * (1 - 0.22) = 198.9 and 255 * 0.5 = 127.5; svglib drew both solid black
        self.assertAlmostEqual(_im_.getpixel((10, 20))[0], 199, delta=1)
        self.assertAlmostEqual(_im_.getpixel((30, 20))[0], 128, delta=1)

    def test_text_is_drawn_at_its_px_size_in_the_layout_face(self):
        _txt_ = 'Source to destination 0123'
        _svg_ = self._doc_(420, 60, '<rect width="420" height="60" fill="#ffffff"/>'
                                    f'<text x="10" y="40" font-size="24px">{_txt_}</text>',
                           extra=f'font-family="{Polars2SVG().default_font}"')
        _x0_, _y0_, _x1_, _y1_ = self._ink_box_(_rgba_(svgToPNGBytes(_svg_)))
        # The run's ink is its advance less two side bearings: within 3% of what the layout
        # measured.  svglib drew 212 px for 301.5 (0.75x); Helvetica would be 283, Verdana 327.
        _adv_ = textAdvance(_txt_, 24)
        self.assertAlmostEqual(_x1_ - _x0_ + 1, _adv_, delta=0.03 * _adv_)

    def test_a_glyph_outside_the_bundled_subset_still_draws(self):
        # The registered face is a subset with no Cyrillic; a glyph it lacks must fall
        # back to a system font, not vanish.  Cyrillic because every host the suite runs
        # on has a face for it (macOS's own, DejaVu in CI's containers); CJK would need a
        # font a CI image cannot reasonably carry.
        _svg_ = self._doc_(200, 60, '<rect width="200" height="60" fill="#ffffff"/>'
                                    '<text x="10" y="40" font-size="24px">жжж</text>',
                           extra=f'font-family="{Polars2SVG().default_font}"')
        _x0_, _y0_, _x1_, _y1_ = self._ink_box_(_rgba_(svgToPNGBytes(_svg_)))
        self.assertGreater(_x1_ - _x0_, 20)

    def test_a_viewbox_of_another_aspect_is_scaled_uniformly(self):
        # SVG's default preserveAspectRatio="xMidYMid meet": the circle stays round and is
        # centred, where svglib scaled x and y apart (every spreadlinesp node an ellipse)
        _svg_ = self._doc_(200, 100, '<circle cx="50" cy="50" r="40" fill="#000000"/>',
                           extra='viewBox="0 0 100 100"')
        _x0_, _y0_, _x1_, _y1_ = self._ink_box_(_rgba_(svgToPNGBytes(_svg_)))
        self.assertAlmostEqual(_x1_ - _x0_, _y1_ - _y0_, delta=1)
        self.assertAlmostEqual((_x0_ + _x1_) / 2.0, 100.0, delta=1)

    def test_an_unpainted_area_stays_transparent(self):
        _svg_ = self._doc_(200, 100, '<rect width="100" height="100" fill="#ffffff"/>',
                           extra='viewBox="0 0 100 100"')
        _im_ = _rgba_(svgToPNGBytes(_svg_))
        self.assertEqual(_im_.getpixel((10, 50))[3], 0)       # a letterbox margin
        self.assertEqual(_im_.getpixel((100, 50)), (255, 255, 255, 255))

    def test_one_svg_pixel_is_one_png_pixel(self):
        _im_ = _rgba_(svgToPNGBytes(self._doc_(123, 45, '<rect width="123" height="45" fill="#ffffff"/>')))
        self.assertEqual(_im_.size, (123, 45))


class TestSavePNGMissingDeps(unittest.TestCase):
    '''Without the [export] extra, or without the cairo library it loads, PNG export must
    fail with a clear message naming what to install.'''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def _plot_(self):
        return self.p2s.histop(pl.DataFrame({'cat': ['a', 'b', 'c']}), 'cat', wxh=(64, 64))

    def test_clear_import_error(self):
        plot = self._plot_()
        # Setting the module to None in sys.modules makes `import cairosvg.surface`
        # raise ImportError, simulating the extra not being installed.
        with mock.patch.dict(sys.modules, {'cairosvg': None, 'cairosvg.surface': None}):
            with self.assertRaises(ImportError) as ctx:
                plot.savePNG(os.path.join(tempfile.gettempdir(), 'nope.png'))
        self.assertIn('polars2svg[export]', str(ctx.exception))

    def test_missing_cairo_library_names_it(self):
        # cairocffi raises OSError at import when it cannot load libcairo
        import builtins
        _real_import_ = builtins.__import__
        def _import_(name, *args, **kwargs):
            if name.startswith('cairosvg'):
                raise OSError('no library called "cairo-2" was found')
            return _real_import_(name, *args, **kwargs)
        plot = self._plot_()
        _without_ = {_k_: _v_ for _k_, _v_ in sys.modules.items() if not _k_.startswith('cairosvg')}
        with mock.patch.dict(sys.modules, _without_, clear=True), mock.patch('builtins.__import__', _import_):
            with self.assertRaises(ImportError) as ctx:
                plot.savePNG(os.path.join(tempfile.gettempdir(), 'nope.png'))
        self.assertIn('cairo graphics library', str(ctx.exception))
        self.assertIn('brew install cairo', str(ctx.exception))


@unittest.skipUnless(_cairosvg_available(), _SKIP_NO_CAIROSVG_)
class TestRasterizeDoesNotReadLocalFiles(unittest.TestCase):
    '''PNG export must not turn an external reference in the SVG into a file read.

    tile() embeds foreign SVG **verbatim** -- that is the component, not an
    oversight (PLANNING.md A3, SECURITY.md) -- so a document arriving at
    svgToPNGBytes() can carry an <image>/<use> href that polars2svg never wrote.
    cairosvg hands every such reference to a url_fetcher, a relative one resolved
    against the working directory; export.py passes _fetchDataURLsOnly_(), which
    serves data: URLs and returns an empty document for anything else.

    These tests pin the observable end of that -- the referenced file's pixels
    never reach the PNG -- so they fail however it could break: a refactor that
    drops the fetcher, passes unsafe=True or hands cairosvg a url= base, or a
    cairosvg release that stops routing a reference through the fetcher.
    See the comments above svgToPNGBytes() in polars2svg/export.py.
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
        # Without this, every assertion below could pass vacuously -- e.g. if the
        # rasterizer silently ignored <image> altogether, or the fixture were not red.
        # A data: URI is the one image source the fetcher serves, so this proves the
        # detection works without relying on the leak.
        import base64
        with open(self.secret_png, 'rb') as f:
            _b64_ = base64.b64encode(f.read()).decode('ascii')
        _svg_ = self._tiled(f'<image x="0" y="0" width="100" height="100" '
                            f'xlink:href="data:image/png;base64,{_b64_}"/>')
        self.assertGreater(self._secret_pixels(svgToPNGBytes(_svg_)), 100,
                           'positive control failed: the rasterizer did not composite an '
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
            # the fetcher refuses before any connection, so this costs no network time
            'image http':          '<image x="0" y="0" width="100" height="100" xlink:href="http://127.0.0.1:9/x.png"/>',
            # <use> pulling in an external document, with and without a fragment
            'use external frag':   f'<use xlink:href="{self.secret_svg}#g1" x="0" y="0"/>',
            'use external whole':  f'<use xlink:href="{self.secret_svg}" x="0" y="0"/>',
        }
        # The relative spellings resolve against the working directory, so run from the
        # directory holding the secret: the worst case for us.
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
                        f'foreign SVG',
                    )
        finally:
            os.chdir(_cwd_)

    def test_fetcher_serves_data_urls_and_nothing_else(self):
        import base64
        with open(self.secret_png, 'rb') as f:
            _raw_ = f.read()
        _data_ = 'data:image/png;base64,' + base64.b64encode(_raw_).decode('ascii')
        self.assertEqual(_fetchDataURLsOnly_(_data_, 'image/png'), _raw_)
        for _url_ in (f'file://{self.secret_png}', 'http://127.0.0.1:9/x.png', 'https://example.com/a.css',
                      'ftp://example.com/a.png', self.secret_png):
            with self.subTest(url=_url_):
                _got_ = _fetchDataURLsOnly_(_url_, 'image/png')
                self.assertNotIn(_raw_, _got_)
                self.assertTrue(_got_.startswith(b'<svg'))

    def test_rasterizer_is_given_bytes_and_the_fetcher(self):
        # The behavioural checks above are the real guarantee, but they only fail once
        # cairosvg's own safe default is gone too.  This one names the invariant directly:
        # the document's bytes with no base url, unsafe=False, and the data:-only fetcher.
        import cairosvg.surface
        _seen_ = []
        _real_ = cairosvg.surface.PNGSurface.convert

        def _spy_(*args, **kwargs):
            _seen_.append(kwargs)
            return _real_(*args, **kwargs)

        with mock.patch.object(cairosvg.surface.PNGSurface, 'convert', _spy_):
            svgToPNGBytes(self.p2s.tile([
                '<svg width="10" height="10" xmlns="http://www.w3.org/2000/svg"/>'
            ])._repr_svg_())

        self.assertEqual(len(_seen_), 1)
        _kw_ = _seen_[0]
        self.assertIsInstance(_kw_.get('bytestring'), bytes)
        self.assertNotIn('url', _kw_)
        self.assertNotIn('file_obj', _kw_)
        self.assertIs(_kw_.get('unsafe'), False)
        self.assertIs(_kw_.get('url_fetcher'), _fetchDataURLsOnly_)


if __name__ == '__main__':
    unittest.main()
