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


@unittest.skipUnless(_svglib_available(), 'svglib/reportlab not installed (optional [export] extra)')
class TestTextPathPNGFlattening(unittest.TestCase):
    '''svglib draws no text that follows a path -- it put those glyphs at (0, 0), off the
    canvas -- so a PNG lacked chordp's circular labels and linkp's curve link labels
    (PLANNING.md C-textpath-png).  svgToPNGBytes() now lays each <textPath> out as
    per-glyph rotated <text> before rendering; it warns only about one it cannot.'''

    _df_ = pl.DataFrame({'fm': list('abcabd'), 'to': list('bcdcda'), 'lbl': list('uvwxyz')})
    _pos_ = {'a': (0.1, 0.1), 'b': (0.9, 0.1), 'c': (0.9, 0.9), 'd': (0.1, 0.9)}

    def _png_(self, svg):
        '''(RGB image, the <textPath> warnings raised while rasterizing svg)'''
        import warnings
        from PIL import Image
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            data = svgToPNGBytes(svg)
        self.assertTrue(data.startswith(_PNG_MAGIC_))
        return (Image.open(io.BytesIO(data)).convert('RGB'),
                [str(x.message) for x in w if '<textPath>' in str(x.message)])

    def assertInkNear(self, with_labels, without_labels, x, y, r):
        '''Some pixel within r of (x, y) differs between the two renders, i.e. a label
        was drawn there.  Everything else in the two renders is the same geometry.'''
        _w_, _h_ = with_labels.size
        _a_, _b_ = with_labels.load(), without_labels.load()
        _hits_ = sum(1 for _px_ in range(max(0, int(x - r)), min(_w_, int(x + r) + 1))
                       for _py_ in range(max(0, int(y - r)), min(_h_, int(y + r) + 1))
                       if (_px_ - x) ** 2 + (_py_ - y) ** 2 <= r * r and _a_[_px_, _py_] != _b_[_px_, _py_])
        self.assertGreater(_hits_, 0, f'no label ink within {r}px of ({x:.1f}, {y:.1f})')

    def _flatten_(self, svg):
        '''Run the flattener over a parsed document; (root, how many were left).'''
        from svglib.svglib import SvgRenderer, load_svg_file
        from reportlab.pdfbase.pdfmetrics import stringWidth
        from polars2svg.export import _flattenTextPaths_
        _root_ = load_svg_file(io.StringIO(svg))
        _left_ = _flattenTextPaths_(_root_, SvgRenderer(io.StringIO(svg)).attrConverter, stringWidth)
        return _root_, _left_

    def _glyphs_(self, root):
        '''Every flattened glyph as (char, x, y, rotation degrees).'''
        import re
        _out_ = []
        for _e_ in root.iter():
            if isinstance(_e_.tag, str) and _e_.tag.endswith('text') and _e_.get('transform'):
                _rot_ = float(re.match(r'rotate\(([-\d.e]+),', _e_.get('transform')).group(1))
                _out_.append((_e_.text, float(_e_.get('x')), float(_e_.get('y')), _rot_))
        return _out_

    # font-size is unitless on purpose: svglib reads '12px' as 9pt (px -> pt), and the
    # flattener measures with that same converter, so a px size would make every
    # expected advance below 0.75x what the arithmetic suggests.
    @staticmethod
    def _doc_(path_d, label, text_attrs='', tp_attrs='startOffset="50%" text-anchor="middle"'):
        return ('<svg xmlns="http://www.w3.org/2000/svg" width="300" height="300">'
                f'<defs><path id="p" d="{path_d}" /></defs>'
                f'<text font-size="12" {text_attrs}><textPath href="#p" {tp_attrs}>{label}</textPath></text>'
                '</svg>')

    # -- the two producers, end to end --------------------------------------------------

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

    def test_a_tiled_rendering_is_flattened_too(self):
        # tile() nests each rendering as its own <svg>; the flattener works on the
        # document, so a label inside a tile is covered without the component's help
        p2s  = Polars2SVG()
        lp   = p2s.linkp(df=self._df_, relationships=[('fm', 'to', 'lbl')], link_shape='curve',
                         pos=self._pos_, draw_link_labels=True)
        cp   = p2s.chordp(df=self._df_, relationships=[('fm', 'to')], draw_labels=True,
                          label_style='circular')
        tile = p2s.tile([lp, cp])
        self.assertEqual(tile._repr_svg_().count('<textPath'), 10)
        _root_, _left_ = self._flatten_(tile._repr_svg_())
        self.assertEqual(_left_, 0)
        self.assertEqual(len(self._glyphs_(_root_)), 6 + len('abcd'))
        self.assertEqual(self._png_(tile._repr_svg_())[1], [])

    def test_straight_labels_do_not_warn(self):
        p2s   = Polars2SVG()
        line  = p2s.linkp(df=self._df_, relationships=[('fm', 'to', 'lbl')], link_shape='line',
                          draw_link_labels=True, draw_node_labels=False)
        radial = p2s.chordp(df=self._df_, relationships=[('fm', 'to')], draw_labels=True)
        self.assertEqual(self._png_(line._repr_svg_())[1], [])
        self.assertEqual(self._png_(radial._repr_svg_())[1], [])

    # -- the layout itself, on paths whose answer is known ------------------------------

    def test_glyphs_step_along_a_straight_path_by_their_advances(self):
        from reportlab.pdfbase.pdfmetrics import stringWidth
        _root_, _left_ = self._flatten_(self._doc_('M 50 100 L 250 100', 'AWi', text_attrs='dy="-3"'))
        self.assertEqual(_left_, 0)
        _g_ = self._glyphs_(_root_)
        self.assertEqual([_c_ for _c_, *_ in _g_], ['A', 'W', 'i'])
        # anchored middle at 50% of 200: the run is centred on x=150, each glyph's own
        # midpoint sits on the path, and dy moves the baseline up off the path
        _x_ = 150.0 - stringWidth('AWi', 'Helvetica', 12) / 2.0
        for _c_, _gx_, _gy_, _rot_ in _g_:
            _w_ = stringWidth(_c_, 'Helvetica', 12)
            self.assertAlmostEqual(_gx_, _x_ + _w_ / 2.0, delta=0.01)
            self.assertAlmostEqual(_gy_, 97.0, delta=0.01)
            self.assertAlmostEqual(_rot_, 0.0, delta=0.01)
            _x_ += _w_

    def test_dy_runs_along_the_paths_own_y_axis(self):
        # a path heading down the page has its +y pointing at screen -x (right of travel)
        _root_, _ = self._flatten_(self._doc_('M 100 50 L 100 250', 'o', text_attrs='dy="5"'))
        (_c_, _gx_, _gy_, _rot_), = self._glyphs_(_root_)
        self.assertAlmostEqual(_gx_, 95.0, delta=0.01)
        self.assertAlmostEqual(_gy_, 150.0, delta=0.01)
        self.assertAlmostEqual(_rot_, 90.0, delta=0.01)

    def test_glyphs_follow_an_arc(self):
        from math import hypot
        # a palindrome, so the run's middle glyph is centred on the run's middle
        _root_, _left_ = self._flatten_(self._doc_('M 0 150 A 100 100 0 0 1 200 150', 'abcdcba'))
        self.assertEqual(_left_, 0)
        _g_ = self._glyphs_(_root_)
        self.assertEqual(len(_g_), 7)
        for _c_, _gx_, _gy_, _rot_ in _g_:
            self.assertAlmostEqual(hypot(_gx_ - 100.0, _gy_ - 150.0), 100.0, delta=0.05)
        # the middle glyph sits at the top of the arc, upright
        _c_, _gx_, _gy_, _rot_ = _g_[3]
        self.assertEqual(_c_, 'd')
        self.assertAlmostEqual(_gx_, 100.0, delta=0.05)
        self.assertAlmostEqual(_gy_, 50.0, delta=0.05)
        self.assertAlmostEqual(_rot_, 0.0, delta=1.0)
        # and the run reads left to right, rotating clockwise as it goes
        self.assertEqual(sorted(_g_, key=lambda _t_: _t_[1]), _g_)
        self.assertEqual(sorted(_g_, key=lambda _t_: _t_[3]), _g_)

    def test_relative_commands_match_absolute_ones(self):
        _abs_, _ = self._flatten_(self._doc_('M 20 40 C 60 0 140 0 180 40 L 180 120', 'curved run'))
        _rel_, _ = self._flatten_(self._doc_('m 20 40 c 40 -40 120 -40 160 0 l 0 80', 'curved run'))
        _ga_, _gr_ = self._glyphs_(_abs_), self._glyphs_(_rel_)
        self.assertEqual(len(_ga_), len('curvedrun'))
        for _a_, _r_ in zip(_ga_, _gr_):
            self.assertEqual(_a_[0], _r_[0])
            for _i_ in (1, 2, 3):
                self.assertAlmostEqual(_a_[_i_], _r_[_i_], delta=0.01)

    def test_start_offset_and_anchor(self):
        from reportlab.pdfbase.pdfmetrics import stringWidth
        _w_ = stringWidth('I', 'Helvetica', 12)
        for _attrs_, _expect_ in [('startOffset="30"', 30.0 + _w_ / 2.0),
                                  ('startOffset="30" text-anchor="end"', 30.0 - _w_ / 2.0),
                                  ('', _w_ / 2.0)]:
            _root_, _ = self._flatten_(self._doc_('M 0 100 L 200 100', 'I', tp_attrs=_attrs_))
            (_c_, _gx_, _gy_, _rot_), = self._glyphs_(_root_)
            self.assertAlmostEqual(_gx_, _expect_, delta=0.01, msg=_attrs_)

    def test_glyphs_off_either_end_of_the_path_are_dropped(self):
        # 'end' at offset 0 puts the whole run before the path starts, as SVG would not draw it
        _root_, _ = self._flatten_(self._doc_('M 0 100 L 200 100', 'gone', tp_attrs='text-anchor="end"'))
        self.assertEqual(self._glyphs_(_root_), [])

    def test_markup_in_a_label_survives_as_text(self):
        _svg_ = self._doc_('M 0 100 L 200 100', 'a&amp;b&lt;c')
        _root_, _ = self._flatten_(_svg_)
        self.assertEqual([_g_[0] for _g_ in self._glyphs_(_root_)], list('a&b<c'))
        self.assertEqual(self._png_(_svg_)[1], [])

    def test_what_cannot_be_flattened_is_left_and_warned_about(self):
        _svg_ = ('<svg xmlns="http://www.w3.org/2000/svg" width="300" height="300"><defs>'
                 '<path id="s" d="M 0 0 S 50 50 100 0" />'
                 '<path id="t" transform="scale(2)" d="M 0 0 L 100 0" />'
                 '<path id="ok" d="M 0 50 L 100 50" /></defs>'
                 '<text><textPath href="#s">smooth curveto</textPath></text>'
                 '<text><textPath href="#t">transformed path</textPath></text>'
                 '<text><textPath href="#nowhere">dangling</textPath></text>'
                 '<text><textPath href="#ok">fine</textPath></text></svg>')
        _root_, _left_ = self._flatten_(_svg_)
        self.assertEqual(_left_, 3)
        self.assertEqual(''.join(_g_[0] for _g_ in self._glyphs_(_root_)), 'fine')
        _msgs_ = self._png_(_svg_)[1]
        self.assertEqual(len(_msgs_), 1)
        self.assertTrue(_msgs_[0].startswith('3 <textPath>'), _msgs_[0])


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
        # refactor to a path fails here with the reason attached even on an svglib
        # that still happens to refuse the resolution.
        # svgToPNGBytes() unrolls svg2rlg() into load_svg_file() + SvgRenderer() so it
        # can flatten <textPath> in between (C-textpath-png); all three take the
        # source, and all are looked up at call time, so patching the module
        # attributes sees whatever is handed over -- whichever route a refactor takes.
        import svglib.svglib
        _seen_ = []

        def _spy_(name):
            _real_ = getattr(svglib.svglib, name)
            def _wrapped_(source, *args, **kwargs):
                _seen_.append((name, source))
                return _real_(source, *args, **kwargs)
            return _wrapped_

        with mock.patch.object(svglib.svglib, 'svg2rlg', _spy_('svg2rlg')), \
             mock.patch.object(svglib.svglib, 'load_svg_file', _spy_('load_svg_file')), \
             mock.patch.object(svglib.svglib, 'SvgRenderer', _spy_('SvgRenderer')):
            svgToPNGBytes(self.p2s.tile([
                '<svg width="10" height="10" xmlns="http://www.w3.org/2000/svg"/>'
            ])._repr_svg_())

        # the renderer is what resolves an href against its source path
        self.assertIn('SvgRenderer', [_n_ for _n_, _src_ in _seen_])
        for _name_, _src_ in _seen_:
            self.assertNotIsInstance(
                _src_, (str, bytes, os.PathLike),
                f'svgToPNGBytes() must hand svglib.{_name_} a stream, not a path: svglib resolves '
                'xlink:href on <image>/<use> against the source document\'s path, so a '
                'path here makes foreign SVG passed to tile() a local file read. '
                'See the comment above svgToPNGBytes() in polars2svg/export.py.',
            )


if __name__ == '__main__':
    unittest.main()
