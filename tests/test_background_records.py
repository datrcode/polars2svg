#
# Background records (PLANNING.md §9.1 / B1-B3).
#
# A background= entry may be a bare shape descriptor (inheriting every background_*
# parameter) or a record carrying its own appearance.  Every case here runs against BOTH
# coordinate-plane components: the contract lives in P2SBackgroundMixin, and only a
# two-component test proves that it is actually shared rather than duplicated.
#
import inspect
import unittest

import polars as pl

import polars2svg
from polars2svg import BackgroundShape, INHERIT
from polars2svg.exceptions import InvalidSpecError
from polars2svg.p2s_background_mixin import P2SBackgroundMixin

from webgpu_test_utils import decode_buffer, hex_to_rgb01, manifest_count


def _tri_vertices_(payload):
    '''Decode the tri VERTEX buffer (6 floats per vertex: x, y, r, g, b, a).  Triangles
    carry a separate index buffer, so they are keyed 'tri_v' rather than 'tri'.'''
    import base64
    import numpy as np
    if 'tri_v' not in payload['buffers']:
        return np.zeros((0, 6), dtype=np.float32)
    return np.frombuffer(base64.b64decode(payload['buffers']['tri_v']), dtype='<f4').reshape(-1, 6)

_COMPONENTS_ = ('xyp', 'linkp')


class _BackgroundCase_(unittest.TestCase):
    '''Renders one background= dict through either component, in a shared world-coordinate
    range (1..5 on both axes) so the same shape fixtures work for both.'''

    def setUp(self):
        self.p2s   = polars2svg.Polars2SVG()
        self.xy_df = pl.DataFrame({'x': [1, 2, 3, 4, 5], 'y': [2, 4, 1, 3, 5]})
        self.ln_df = pl.DataFrame({'fm': ['a', 'b', 'c'], 'to': ['b', 'c', 'a']})
        self.pos   = {'a': (1.0, 1.0), 'b': (5.0, 5.0), 'c': (3.0, 4.0)}
        self.box   = [(1.5, 1.5), (4.5, 1.5), (4.5, 4.5), (1.5, 4.5)]
        self.tri   = 'M 1.5 1.5 L 4.5 1.5 L 3.0 4.5 Z'
        self.open_ = 'M 1.5 1.5 L 3.0 4.5 L 4.5 1.5'

    def render(self, component, background, **sidecars):
        if component == 'xyp':
            return self.p2s.xyp(df=self.xy_df, x='x', y='y', wxh=(256, 256),
                                background=background, **sidecars)
        return self.p2s.linkp(self.ln_df, [('fm', 'to')], pos=self.pos, wxh=(256, 256),
                              background=background, **sidecars)

    def bgPayload(self, chart):
        '''WebGPU payload of the BACKGROUND display list alone, so an assertion about a
        background primitive cannot be satisfied by an axis line or a link.'''
        return chart._dl_background_.webgpu_payload(chart.p2s.glyphAtlas())


# ---------------------------------------------------------------------------------------
# The record type itself
# ---------------------------------------------------------------------------------------
class TestBackgroundShapeType(_BackgroundCase_):
    def test_every_field_defaults_to_inherit(self):
        _r_ = BackgroundShape('M 0 0 L 1 1 Z')
        for _f_ in BackgroundShape._FIELDS_[1:]:
            self.assertIs(getattr(_r_, _f_), INHERIT, f'{_f_} should default to INHERIT')

    def test_inherit_is_a_singleton_and_not_none(self):
        from polars2svg.p2s_background_mixin import _InheritSentinel_
        self.assertIs(_InheritSentinel_(), INHERIT)
        self.assertIsNotNone(INHERIT)
        self.assertNotEqual(INHERIT, None)
        self.assertEqual(repr(INHERIT), 'INHERIT')

    def test_record_is_immutable(self):
        # _copy_mutable_containers_() shares non-container leaves by reference across
        # template clones, so a mutable record would leak edits into its template.
        _r_ = self.p2s.bgShape(self.tri, fill='#ff0000')
        with self.assertRaises(AttributeError):
            _r_.fill = '#00ff00'
        with self.assertRaises(AttributeError):
            del _r_.fill
        self.assertEqual(_r_.fill, '#ff0000')

    def test_template_clone_shares_the_record_but_not_the_dict(self):
        _r_    = self.p2s.bgShape(self.tri, fill='#ff0000')
        _base_ = self.render('xyp', {'t': _r_})
        _clone_ = self.p2s.xyp(template=_base_)
        self.assertIsNot(_clone_.background, _base_.background)   # container copied
        self.assertIs(_clone_.background['t'], _base_.background['t'])   # leaf shared -- safe: frozen

    def test_unknown_field_rejected(self):
        with self.assertRaises(TypeError):
            self.p2s.bgShape(self.tri, filll='#ff0000')

    def test_unknown_key_in_dict_form_rejected(self):
        with self.assertRaises(InvalidSpecError) as _cm_:
            self.render('xyp', {'t': {'shape': self.tri, 'colour': '#ff0000'}})
        self.assertIn('colour', str(_cm_.exception))

    def test_dict_form_requires_a_shape(self):
        with self.assertRaises(InvalidSpecError):
            self.render('xyp', {'t': {'fill': '#ff0000'}})

    def test_repr_shows_only_the_fields_that_were_set(self):
        _r_ = self.p2s.bgShape(self.tri, fill=None, stroke_opacity=0.5)
        self.assertIn('fill=None', repr(_r_))
        self.assertIn('stroke_opacity=0.5', repr(_r_))
        self.assertNotIn('INHERIT', repr(_r_))


# ---------------------------------------------------------------------------------------
# Back-compatibility: bare descriptors, and the two forms mixing
# ---------------------------------------------------------------------------------------
class TestBareDescriptorsStillWork(_BackgroundCase_):
    def test_bare_descriptor_equals_an_all_inherit_record(self):
        for _c_ in _COMPONENTS_:
            with self.subTest(component=_c_):
                _bare_ = self.render(_c_, {'t': self.tri},
                                     background_fill='#3366aa', background_opacity=0.4)
                _rec_  = self.render(_c_, {'t': self.p2s.bgShape(self.tri)},
                                     background_fill='#3366aa', background_opacity=0.4)
                self.assertEqual(_bare_.svg_background, _rec_.svg_background)
                self.assertNotEqual(_bare_.svg_background, '')

    def test_dict_form_equals_the_constructor_form(self):
        for _c_ in _COMPONENTS_:
            with self.subTest(component=_c_):
                _a_ = self.render(_c_, {'t': self.p2s.bgShape(self.tri, fill='#ff0000')})
                _b_ = self.render(_c_, {'t': {'shape': self.tri, 'fill': '#ff0000'}})
                self.assertEqual(_a_.svg_background, _b_.svg_background)

    def test_bare_and_record_mix_in_one_dict(self):
        for _c_ in _COMPONENTS_:
            with self.subTest(component=_c_):
                _chart_ = self.render(_c_, {'bare': self.box,
                                            'rec':  self.p2s.bgShape(self.tri, fill='#ff0000')},
                                      background_fill='#00ff00')
                self.assertIn('fill="#00ff00"', _chart_.svg_background)   # the bare one inherited
                self.assertIn('fill="#ff0000"', _chart_.svg_background)   # the record kept its own


# ---------------------------------------------------------------------------------------
# INHERIT vs None
# ---------------------------------------------------------------------------------------
class TestInheritAndOff(_BackgroundCase_):
    def test_fill_none_emits_fill_none_not_a_missing_attribute(self):
        # Background shapes sit directly under <svg> with no ancestor fill, so omitting the
        # attribute would take SVG's initial value and render solid black.
        for _c_ in _COMPONENTS_:
            with self.subTest(component=_c_):
                _chart_ = self.render(_c_, {'t': self.p2s.bgShape(self.tri, fill=None)},
                                      background_fill='#3366aa')
                self.assertIn('fill="none"', _chart_.svg_background)
                self.assertNotIn('fill-opacity', _chart_.svg_background)
                self.assertNotIn('#3366aa', _chart_.svg_background)

    def test_stroke_none_emits_no_stroke_attributes(self):
        for _c_ in _COMPONENTS_:
            with self.subTest(component=_c_):
                _chart_ = self.render(_c_, {'t': self.p2s.bgShape(self.tri, stroke=None)},
                                      background_stroke='#3366aa', background_stroke_w=2.0)
                self.assertNotIn('stroke=',        _chart_.svg_background)
                self.assertNotIn('stroke-width',   _chart_.svg_background)
                self.assertNotIn('stroke-opacity', _chart_.svg_background)

    def test_inherit_takes_the_sidecar(self):
        for _c_ in _COMPONENTS_:
            with self.subTest(component=_c_):
                _chart_ = self.render(_c_, {'t': self.p2s.bgShape(self.tri, fill=INHERIT)},
                                      background_fill='#123456')
                self.assertIn('fill="#123456"', _chart_.svg_background)

    def test_record_overrides_the_sidecar(self):
        for _c_ in _COMPONENTS_:
            with self.subTest(component=_c_):
                _chart_ = self.render(_c_, {'t': self.p2s.bgShape(self.tri, fill='#abcdef')},
                                      background_fill='#123456')
                self.assertIn('fill="#abcdef"', _chart_.svg_background)
                self.assertNotIn('#123456', _chart_.svg_background)

    def test_vary_resolves_against_the_dict_key(self):
        for _c_ in _COMPONENTS_:
            with self.subTest(component=_c_):
                _chart_ = self.render(_c_, {'zone one': self.p2s.bgShape(self.tri, fill='vary')})
                self.assertIn(f'fill="{self.p2s.color("zone one")}"', _chart_.svg_background)

    def test_fill_and_stroke_resolve_through_the_same_ladder(self):
        # fill used to test dict -> 'vary' -> hex and stroke 'vary' -> hex -> dict; one
        # resolver now serves both (and the label colour).
        for _spec_ in ({'t': '#445566'}, 'vary', '#445566', 'default'):
            with self.subTest(spec=_spec_):
                _f_ = self.render('xyp', {'t': self.tri}, background_fill=_spec_,
                                  background_stroke=None)
                _s_ = self.render('xyp', {'t': self.tri}, background_stroke=_spec_,
                                  background_fill=None)
                _fill_co_   = _f_.svg_background.split('fill="')[1].split('"')[0]
                _stroke_co_ = _s_.svg_background.split('stroke="')[1].split('"')[0]
                self.assertEqual(_fill_co_, _stroke_co_)


# ---------------------------------------------------------------------------------------
# Labels decoupled from the dict key
# ---------------------------------------------------------------------------------------
class TestBackgroundLabels(_BackgroundCase_):
    def test_label_differs_from_the_key(self):
        for _c_ in _COMPONENTS_:
            with self.subTest(component=_c_):
                _chart_ = self.render(_c_, {'flow 1 heads': self.p2s.bgShape(self.tri, label='flow 1')},
                                      background_label_color='#000000')
                self.assertIn('>flow 1</text>', _chart_.svg_background)
                self.assertNotIn('flow 1 heads', _chart_.svg_background)

    def test_label_none_suppresses_the_label(self):
        for _c_ in _COMPONENTS_:
            with self.subTest(component=_c_):
                _chart_ = self.render(_c_, {'t': self.p2s.bgShape(self.tri, label=None)},
                                      background_label_color='#000000')
                self.assertNotIn('<text', _chart_.svg_background)

    def test_label_color_per_record(self):
        for _c_ in _COMPONENTS_:
            with self.subTest(component=_c_):
                _chart_ = self.render(_c_, {'a': self.p2s.bgShape(self.box, label_color='#ff00ff'),
                                            'b': self.tri},
                                      background_label_color='#000000')
                self.assertIn('fill="#ff00ff"', _chart_.svg_background)
                self.assertIn('fill="#000000"', _chart_.svg_background)

    def test_label_color_alone_draws_a_label_the_b_cycle_left_off(self):
        # background_label_color drives the interactive 'b' cycle, so a record picks the
        # label TEXT; to force a label on with labels globally off it supplies its own colour.
        for _c_ in _COMPONENTS_:
            with self.subTest(component=_c_):
                _off_ = self.render(_c_, {'t': self.p2s.bgShape(self.tri, label='X')})
                self.assertNotIn('<text', _off_.svg_background)
                _on_  = self.render(_c_, {'t': self.p2s.bgShape(self.tri, label='X',
                                                                label_color='#202020')})
                self.assertIn('>X</text>', _on_.svg_background)


# ---------------------------------------------------------------------------------------
# B3 -- the attributes the record makes cheap
# ---------------------------------------------------------------------------------------
class TestNewAttributes(_BackgroundCase_):
    def test_stroke_opacity_is_emitted(self):
        for _c_ in _COMPONENTS_:
            with self.subTest(component=_c_):
                _chart_ = self.render(_c_, {'t': self.p2s.bgShape(
                    self.open_, fill=None, stroke='#112233', stroke_opacity=0.35)})
                self.assertIn('stroke-opacity="0.35"', _chart_.svg_background)

    def test_dash_is_emitted(self):
        for _c_ in _COMPONENTS_:
            with self.subTest(component=_c_):
                _chart_ = self.render(_c_, {'t': self.p2s.bgShape(self.tri, dash='4 2')})
                self.assertIn('stroke-dasharray="4 2"', _chart_.svg_background)

    def test_linecap_and_linejoin_are_emitted(self):
        for _c_ in _COMPONENTS_:
            with self.subTest(component=_c_):
                _chart_ = self.render(_c_, {'t': self.p2s.bgShape(
                    self.open_, stroke_linecap='round', stroke_linejoin='bevel')})
                self.assertIn('stroke-linecap="round"',  _chart_.svg_background)
                self.assertIn('stroke-linejoin="bevel"', _chart_.svg_background)

    def test_sub_attributes_are_dropped_when_the_stroke_is_off(self):
        for _c_ in _COMPONENTS_:
            with self.subTest(component=_c_):
                _chart_ = self.render(_c_, {'t': self.p2s.bgShape(
                    self.tri, stroke=None, stroke_opacity=0.5, dash='4 2')})
                self.assertNotIn('stroke-dasharray', _chart_.svg_background)
                self.assertNotIn('stroke-opacity',   _chart_.svg_background)

    def test_no_new_component_parameters(self):
        # The point of B1: new capability arrives as record fields, not as a sixth,
        # seventh and eighth background_* parameter.
        for _cls_, _n_ in ((polars2svg.xyp.XYp, 'XYp'), (polars2svg.linkp.LinkP, 'LinkP')):
            with self.subTest(component=_n_):
                _bg_params_ = sorted(_k_ for _k_ in _cls_._VALID_KWARGS if _k_.startswith('background'))
                self.assertEqual(_bg_params_, ['background', 'background_fill',
                                               'background_label_color', 'background_opacity',
                                               'background_stroke', 'background_stroke_w'])


# ---------------------------------------------------------------------------------------
# B2 -- the GPU path reads the record, not the emitted SVG
# ---------------------------------------------------------------------------------------
class TestDisplayListReadsTheRecord(_BackgroundCase_):
    def test_stroke_opacity_reaches_the_display_list(self):
        # The regex route could never carry this: it only ever read fill, fill-opacity,
        # stroke and stroke-width back out of the SVG.
        for _c_ in _COMPONENTS_:
            with self.subTest(component=_c_):
                _chart_ = self.render(_c_, {'t': self.p2s.bgShape(
                    self.open_, fill=None, stroke='#112233', stroke_opacity=0.25)})
                _lines_ = decode_buffer(self.bgPayload(_chart_), 'line')
                self.assertGreater(len(_lines_), 0)
                for _row_ in _lines_:
                    self.assertAlmostEqual(float(_row_[8]), 0.25, places=4)

    def test_dash_reaches_the_display_list(self):
        for _c_ in _COMPONENTS_:
            with self.subTest(component=_c_):
                _chart_ = self.render(_c_, {'t': self.p2s.bgShape(self.tri, fill=None, dash='4 2')})
                _lines_ = decode_buffer(self.bgPayload(_chart_), 'line')
                self.assertGreater(len(_lines_), 0)
                for _row_ in _lines_:
                    self.assertAlmostEqual(float(_row_[9]),  4.0, places=4)   # dash_on
                    self.assertAlmostEqual(float(_row_[10]), 2.0, places=4)   # dash_off

    def test_fill_colour_and_width_match_the_record(self):
        for _c_ in _COMPONENTS_:
            with self.subTest(component=_c_):
                _chart_ = self.render(_c_, {'t': self.p2s.bgShape(
                    self.box, fill='#204080', fill_opacity=0.5, stroke='#801020', stroke_width=3.0)})
                _payload_ = self.bgPayload(_chart_)
                _tris_ = _tri_vertices_(_payload_)
                self.assertGreater(len(_tris_), 0)
                _r_, _g_, _b_ = hex_to_rgb01('#204080')
                for _v_ in _tris_:
                    self.assertAlmostEqual(float(_v_[2]), _r_, places=3)
                    self.assertAlmostEqual(float(_v_[3]), _g_, places=3)
                    self.assertAlmostEqual(float(_v_[4]), _b_, places=3)
                    self.assertAlmostEqual(float(_v_[5]), 0.5, places=3)
                for _row_ in decode_buffer(_payload_, 'line'):
                    self.assertAlmostEqual(float(_row_[4]), 3.0, places=4)

    def test_fill_none_records_no_triangles(self):
        for _c_ in _COMPONENTS_:
            with self.subTest(component=_c_):
                _chart_ = self.render(_c_, {'t': self.p2s.bgShape(self.box, fill=None)})
                self.assertEqual(manifest_count(self.bgPayload(_chart_), 'tri'), 0)

    def test_stroke_none_records_no_lines(self):
        for _c_ in _COMPONENTS_:
            with self.subTest(component=_c_):
                _chart_ = self.render(_c_, {'t': self.p2s.bgShape(self.box, stroke=None,
                                                                  fill='#204080')})
                self.assertEqual(manifest_count(self.bgPayload(_chart_), 'line'), 0)

    def test_zero_fill_opacity_records_no_triangles(self):
        for _c_ in _COMPONENTS_:
            with self.subTest(component=_c_):
                _chart_ = self.render(_c_, {'t': self.p2s.bgShape(self.box, fill='#204080',
                                                                  fill_opacity=0.0)})
                self.assertEqual(manifest_count(self.bgPayload(_chart_), 'tri'), 0)

    def test_label_reaches_the_display_list_as_glyphs(self):
        for _c_ in _COMPONENTS_:
            with self.subTest(component=_c_):
                _chart_ = self.render(_c_, {'k': self.p2s.bgShape(self.box, label='abc')},
                                      background_label_color='#000000')
                self.assertEqual(manifest_count(self.bgPayload(_chart_), 'glyph'), 3)

    def test_background_primitives_reach_the_composed_component_payload(self):
        # bgPayload() isolates the background display list; this checks it is actually
        # composed into the component's own payload (linkp's background DL had no test
        # at all before -- only xyp's was covered).
        for _c_ in _COMPONENTS_:
            with self.subTest(component=_c_):
                _plain_ = self.render(_c_, None)
                _withbg_ = self.render(_c_, {'t': self.p2s.bgShape(self.box, fill='#204080')})
                self.assertGreater(manifest_count(_withbg_.webgpu(), 'tri'),
                                   manifest_count(_plain_.webgpu(), 'tri'))

    def test_the_display_list_writers_do_not_re_parse_svg(self):
        # PLANNING.md §8 records what the svgToDisplayList() re-parse route cost
        # spreadlinesp; this keeps it from growing back here.
        for _fn_ in (P2SBackgroundMixin.__backgroundShapeToDL__,
                     P2SBackgroundMixin.__backgroundLabelToDL__):
            with self.subTest(fn=_fn_.__name__):
                _src_ = inspect.getsource(_fn_)
                self.assertNotIn('re.search', _src_)
                self.assertNotIn('_attr_',    _src_)


# ---------------------------------------------------------------------------------------
# Draw order is contract
# ---------------------------------------------------------------------------------------
class TestDrawOrder(_BackgroundCase_):
    def test_shapes_follow_dict_insertion_order_then_all_labels(self):
        for _c_ in _COMPONENTS_:
            with self.subTest(component=_c_):
                _chart_ = self.render(_c_, {'one':   self.p2s.bgShape(self.box, fill='#010101'),
                                            'two':   self.p2s.bgShape(self.tri, fill='#020202'),
                                            'three': self.p2s.bgShape(self.open_, fill='#030303')},
                                      background_label_color='#000000')
                _svg_ = _chart_.svg_background
                _i1_, _i2_, _i3_ = (_svg_.index('#010101'), _svg_.index('#020202'), _svg_.index('#030303'))
                self.assertLess(_i1_, _i2_)
                self.assertLess(_i2_, _i3_)
                self.assertLess(_i3_, _svg_.index('<text'))            # every shape before every label
                self.assertLess(_svg_.index('>one</text>'), _svg_.index('>two</text>'))


if __name__ == '__main__':
    unittest.main()
