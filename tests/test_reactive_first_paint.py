"""The contracts the static-view conversion has to keep.

**First paint**: the plot must reach the browser on mount, not one refresh later.

This is the contract that used to require building every view's class per call with
``type('XYZ', (ReactiveHTML,), {...})``, and nothing else in the suite covers it.

``mod_inner`` is bound as ``${mod_inner}`` *between* tags, which makes it a
ReactiveHTML **child**, and children take an asymmetric path:

  * ``ReactiveHTML._init_params()`` drops every child from the data model it builds,
    so ``data.mod_inner`` starts at the bokeh field's default -- taken from the
    param's **class** default;
  * ``ReactiveHTML._update_model()`` does send a child's later values to
    ``data.mod_inner``, and sends them raw.

Every render script does ``mod.innerHTML = data.mod_inner``, so with a class default
of ``''`` the first frame is blank until something writes the param again.  Baking the
SVG into the class default is what the dynamic class was for;
``P2SReactiveHTML._init_params()`` replaces it by seeding the data model with this
instance's child values.

The value must also arrive **raw**: Panel runs ordinary string params through its HTML
sanitizer, which strips an SVG to nothing (measured: 3663 bytes -> 104 on an xyp
render), so a sanitized copy would be as blank as an empty one.
"""
import datetime
import math
import unittest

import polars as pl
from view_js_utils import component_js

from polars2svg import Polars2SVG

try:
    from bokeh.document import Document
    from polars2svg.interactive_controller import linkpi, smallpi
    from polars2svg.spreadlinepi import spreadlinepi
    PANEL_AVAILABLE = True
except ImportError:
    PANEL_AVAILABLE = False


def _df():
    return pl.DataFrame({
        'x':   [1.0, 2.0, 3.0, 4.0],
        'y':   [1.0, 4.0, 2.0, 3.0],
        'cat': ['a', 'b', 'a', 'c'],
        'ts':  [datetime.datetime(2024, 1, d) for d in (1, 2, 3, 4)],
    })


def _link_df():
    return pl.DataFrame({'fm': ['a', 'b', 'c', 'a'], 'to': ['b', 'c', 'a', 'c']})


def _pos():
    return {n: [math.cos(i * 2 * math.pi / 3), math.sin(i * 2 * math.pi / 3)]
            for i, n in enumerate('abc')}


@unittest.skipUnless(PANEL_AVAILABLE, 'panel not installed')
class TestFirstPaintCarriesThePlot(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.p2s = Polars2SVG()

    def _views(self):
        p2s, df = self.p2s, _df()
        _xyp_ = p2s.xyp(df, 'x', 'y')
        return {
            'xypi':         p2s.xypi(_xyp_),
            'histopi':      p2s.histopi(p2s.histop(df, 'cat')),
            'timepi':       p2s.timepi(p2s.timep(df, 'ts')),
            'chordpi':      p2s.chordpi(p2s.chordp(df=_link_df(), relationships=[('fm', 'to')])),
            'piepi':        p2s.piepi(p2s.piep(df, 'cat')),
            'linkpi':       linkpi(p2s.linkp(_link_df(), relationships=[('fm', 'to')], pos=_pos())),
            'smallpi':      smallpi(p2s.smallp(df, 'cat', _xyp_, wxh=(384, 384))),
            'spreadlinepi': spreadlinepi(p2s.spreadlinesp(
                                pl.DataFrame({'fm': ['a', 'a', 'b'], 'to': ['b', 'c', 'c'],
                                              'time': [datetime.datetime(2024, 1, d) for d in (1, 2, 3)]}),
                                [('fm', 'to')], ego='a', time='time')),
            'stack_controli': p2s.stack_controli(p2s.xyp(df, 'x', 'y', wxh=(120, 60))),
        }

    def test_data_model_carries_this_instance_plot(self):
        for _name_, _view_ in self._views().items():
            with self.subTest(view=_name_):
                _model_ = _view_.get_root(Document())
                self.assertEqual(
                    _model_.data.mod_inner, _view_.mod_inner,
                    f'{_name_}: data.mod_inner does not match the instance -- the first '
                    f'frame would paint something other than this view\'s plot')
                self.assertTrue(
                    _model_.data.mod_inner.lstrip().startswith('<svg'),
                    f'{_name_}: data.mod_inner is not an SVG ('
                    f'{_model_.data.mod_inner[:40]!r}) -- blank or sanitized first frame')

    def test_two_views_of_different_plots_do_not_share_a_first_frame(self):
        """The class is shared now, so a class-level default would show up here."""
        _a_ = self.p2s.xypi(self.p2s.xyp(_df(), 'x', 'y', wxh=(256, 256)))
        _b_ = self.p2s.xypi(self.p2s.xyp(_df(), 'x', 'y', wxh=(512, 128)))
        self.assertIs(type(_a_), type(_b_), 'expected one shared class')
        _ma_, _mb_ = _a_.get_root(Document()), _b_.get_root(Document())
        self.assertIn('width="256" height="256"', _ma_.data.mod_inner)
        self.assertIn('width="512" height="128"', _mb_.data.mod_inner)
        self.assertEqual(_ma_.data.svg_w, 256)
        self.assertEqual(_mb_.data.svg_w, 512)


@unittest.skipUnless(PANEL_AVAILABLE, 'panel not installed')
class TestOneClassPerComponent(unittest.TestCase):
    """Views are no longer built with type() per call, so repeated construction must
    reuse one class -- each new ReactiveHTML subclass permanently registers a bokeh
    DataModel that is never reclaimed."""

    def test_repeated_construction_reuses_the_class(self):
        p2s = Polars2SVG()
        _seen_ = {type(p2s.xypi(p2s.xyp(_df(), 'x', 'y'))) for _ in range(5)}
        self.assertEqual(len(_seen_), 1, f'expected one class, got {_seen_}')

    def test_no_new_bokeh_model_classes_per_view(self):
        from bokeh.model import Model
        p2s = Polars2SVG()
        p2s.xypi(p2s.xyp(_df(), 'x', 'y'))            # warm the class
        _before_ = len(Model.model_class_reverse_map)
        for _ in range(10):
            p2s.xypi(p2s.xyp(_df(), 'x', 'y'))
        self.assertEqual(len(Model.model_class_reverse_map), _before_,
                         'constructing views registered new bokeh model classes')


@unittest.skipUnless(PANEL_AVAILABLE, 'panel not installed')
class TestWebGpuRuntimeIsNotOnSvgViews(unittest.TestCase):
    """P2S_GPU_JS is ~14 KB shipped inside every view's render script.  An SVG view
    never calls into it, so it lives on the *_GPU subclasses and must not leak onto
    the SVG classes -- that is the whole reason the split exists."""

    @classmethod
    def setUpClass(cls):
        cls.p2s = Polars2SVG()

    def _pairs(self):
        """name -> (svg view, gpu view)."""
        p2s, df, ldf = self.p2s, _df(), _link_df()
        _xyp_  = p2s.xyp(df, 'x', 'y')
        _sdf_  = pl.DataFrame({'fm': ['a', 'a', 'b'], 'to': ['b', 'c', 'c'],
                               'time': [datetime.datetime(2024, 1, d) for d in (1, 2, 3)]})
        _mk_ = {
            'xypi':         lambda g: p2s.xypi(p2s.xyp(df, 'x', 'y'), use_webgpu=g),
            'histopi':      lambda g: p2s.histopi(p2s.histop(df, 'cat'), use_webgpu=g),
            'timepi':       lambda g: p2s.timepi(p2s.timep(df, 'ts'), use_webgpu=g),
            'chordpi':      lambda g: p2s.chordpi(p2s.chordp(df=ldf, relationships=[('fm', 'to')]), use_webgpu=g),
            'piepi':        lambda g: p2s.piepi(p2s.piep(df, 'cat'), use_webgpu=g),
            'linkpi':       lambda g: linkpi(p2s.linkp(ldf, relationships=[('fm', 'to')], pos=_pos()), use_webgpu=g),
            'smallpi':      lambda g: smallpi(p2s.smallp(df, 'cat', _xyp_, wxh=(384, 384)), use_webgpu=g),
            'spreadlinepi': lambda g: spreadlinepi(p2s.spreadlinesp(_sdf_, [('fm', 'to')], ego='a', time='time'),
                                                   use_webgpu=g),
        }
        return {_n_: (_f_(False), _f_(True)) for _n_, _f_ in _mk_.items()}

    def test_svg_views_ship_no_gpu_runtime(self):
        from polars2svg.p2s_webgpu_runtime import P2S_GPU_JS
        for _name_, (_svg_, _gpu_) in self._pairs().items():
            with self.subTest(component=_name_):
                _svg_js_ = component_js(_svg_)
                _gpu_js_ = component_js(_gpu_)
                self.assertNotIn(P2S_GPU_JS, _svg_js_,
                                 f'{_name_}: the SVG view ships the WebGPU runtime')
                self.assertIn(P2S_GPU_JS, _gpu_js_,
                              f'{_name_}: the GPU view is missing the WebGPU runtime')
                self.assertGreater(len(_gpu_js_) - len(_svg_js_), 10_000,
                                   f'{_name_}: the split saved almost nothing')
                # the GPU params are on the subclass only
                for _p_ in ('gpu_payload', 'gpu_error'):
                    self.assertNotIn(_p_, type(_svg_).param, f'{_name_}: {_p_} on the SVG class')
                    self.assertIn(_p_, type(_gpu_).param, f'{_name_}: {_p_} missing on the GPU class')

    def test_gpu_view_is_a_subclass_of_the_svg_view(self):
        for _name_, (_svg_, _gpu_) in self._pairs().items():
            with self.subTest(component=_name_):
                self.assertIsInstance(_gpu_, type(_svg_),
                                      f'{_name_}: the GPU class must subclass the SVG one, or '
                                      f'every isinstance() check over views misses it')


@unittest.skipUnless(PANEL_AVAILABLE, 'panel not installed')
class TestPanelizeLinksTheGpuSubclasses(unittest.TestCase):
    """panelize() used to decide what a view broadcasts from `type(view).__name__`.
    With the WebGPU variants as subclasses that check silently stops matching, so the
    decision moved to inherited capability flags.  This is what guards that."""

    def _panel(self, use_webgpu):
        from polars2svg.interactive_controller import panelize
        p2s = Polars2SVG()
        _sdf_ = pl.DataFrame({'fm': ['a', 'a', 'b'], 'to': ['b', 'c', 'c'],
                              'time': [datetime.datetime(2024, 1, d) for d in (1, 2, 3)]})
        return panelize([[p2s.linkp(_link_df(), relationships=[('fm', 'to')], pos=_pos()),
                          p2s.spreadlinesp(_sdf_, [('fm', 'to')], ego='a', time='time')],
                         [p2s.xyp(_df(), 'x', 'y')]], use_webgpu=use_webgpu)

    def test_broadcast_flags_survive_both_render_modes(self):
        for _gpu_ in (False, True):
            with self.subTest(use_webgpu=_gpu_):
                _views_ = {type(v).__name__.replace('_GPU', ''): v
                           for v in self._panel(_gpu_).mvc.view_refs.values()}
                self.assertEqual(set(_views_), {'LINKPI', 'SLPI', 'XYPI'})
                self.assertTrue(_views_['LINKPI']._broadcasts_selection_)
                self.assertTrue(_views_['LINKPI']._broadcasts_positions_)
                self.assertTrue(_views_['SLPI']._broadcasts_selection_)
                self.assertFalse(getattr(_views_['XYPI'], '_broadcasts_selection_', False))
                self.assertFalse(getattr(_views_['SLPI'], '_broadcasts_positions_', False))


if __name__ == '__main__':
    unittest.main()
