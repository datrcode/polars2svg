import re
import unittest
import polars as pl
from polars2svg import Polars2SVG
from svg_test_utils import assert_timing_metrics_populated, assert_valid_svg


def _make_df():
    return pl.DataFrame({
        'fm':   ['a', 'b', 'c', 'a', 'd', 'b'],
        'to':   ['b', 'a', 'a', 'c', 'a', 'c'],
        'time': [1,   1,   1,   2,   2,   3  ],
        'w':    [3,   1,   2,   4,   1,   2  ],
    })

def _rels():
    return [('fm', 'to')]


class TestSpreadLinesPBasic(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def test_basic_render(self):
        sp = self.p2s.spreadlinesp(_make_df(), _rels(), ego='a', time='time')
        assert_valid_svg(self, sp.svg)
        self.assertIn('<circle', sp.svg)

    def test_timing_metrics_populated(self):
        sp = self.p2s.spreadlinesp(_make_df(), _rels(), ego='a', time='time')
        assert_timing_metrics_populated(self, sp, ('__parseInput__', '__calculateLayout__', '__renderSVG__'))

    def test_wxh_respected(self):
        sp = self.p2s.spreadlinesp(_make_df(), _rels(), ego='a', time='time', wxh=(600, 300))
        self.assertEqual(sp.wxh, (600, 300))

    def test_count_field(self):
        sp = self.p2s.spreadlinesp(_make_df(), _rels(), ego='a', time='time', count='w')
        assert_valid_svg(self, sp.svg)

    def test_node_color_hex(self):
        sp = self.p2s.spreadlinesp(_make_df(), _rels(), ego='a', time='time', node_color='#ff0000')
        self.assertIn('#ff0000', sp.svg)

    def test_node_color_by_node_name(self):
        sp = self.p2s.spreadlinesp(_make_df(), _rels(), ego='a', time='time',
                                   node_color=self.p2s.COLOR_BY_NODE_NAME)
        assert_valid_svg(self, sp.svg)

    def test_node_color_dict(self):
        sp = self.p2s.spreadlinesp(_make_df(), _rels(), ego='a', time='time',
                                   node_color={'b': '#00ff00', 'c': '#0000ff'})
        self.assertIn('#00ff00', sp.svg)

    def test_set_ego(self):
        sp = self.p2s.spreadlinesp(_make_df(), _rels(), ego={'a', 'b'}, time='time')
        assert_valid_svg(self, sp.svg)

    def test_repr_svg(self):
        sp = self.p2s.spreadlinesp(_make_df(), _rels(), ego='a', time='time')
        self.assertEqual(sp._repr_svg_(), sp.svg)

    def test_no_df_constructor_does_not_crash(self):
        tmpl = self.p2s.spreadlinesp(_rels(), ego='a', time='time')
        self.assertIsNone(tmpl.df)

    def test_render_with_applies_template(self):
        tmpl = self.p2s.spreadlinesp(_rels(), ego='a', time='time')
        sp   = tmpl.render_with(_make_df())
        assert_valid_svg(self, sp.svg)

    def test_render_with_overrides_ego(self):
        tmpl = self.p2s.spreadlinesp(_rels(), ego='a', time='time')
        sp   = tmpl.render_with(_make_df(), ego='b')
        assert_valid_svg(self, sp.svg)

    def test_unknown_kwarg_raises(self):
        with self.assertRaises(TypeError):
            self.p2s.spreadlinesp(_make_df(), _rels(), ego='a', time='time',
                                  _nonexistent_param_=True)

    def test_anno_does_not_crash(self):
        sp = self.p2s.spreadlinesp(_make_df(), _rels(), ego='a', time='time', anno={1: 'Event A'})
        assert_valid_svg(self, sp.svg)

    def test_highlight_nodes(self):
        sp = self.p2s.spreadlinesp(_make_df(), _rels(), ego='a', time='time',
                                   highlight_nodes={'b'})
        assert_valid_svg(self, sp.svg)

    def test_max_rings_1(self):
        sp = self.p2s.spreadlinesp(_make_df(), _rels(), ego='a', time='time', max_rings=1)
        assert_valid_svg(self, sp.svg)

    def test_max_rings_2(self):
        sp = self.p2s.spreadlinesp(_make_df(), _rels(), ego='a', time='time', max_rings=2)
        assert_valid_svg(self, sp.svg)


class TestSpreadLinesPIdScoping(unittest.TestCase):
    '''Two spreadlinesp figures on one page must not collide on SVG ids.

    Every id this component emits lives in the cloud/selection render, which needs a
    set-valued ego: `ego_is_set` puts the focal node behind the cloud glyph, and a
    partial highlight (some of the ego set selected, not all) draws the selection ring
    through a per-bin clipPath rather than as a plain <use>.

    Until 0.2.1 those ids were emitted bare -- `cloud`, `cloud_outline` and `ccl_<bin>`.
    The first two are a fixed <defs> string, so a duplicate resolved to an identical
    definition and only made the markup invalid.  `ccl_<bin>` was a real render bug: the
    clip rect is placed from that figure's own layout, so with two figures in one DOM
    the second figure's rings were clipped by the first figure's window -- and since the
    windows are only 20px tall, two figures of different heights clip to disjoint bands
    and the second figure's rings disappear completely.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def _selection(self, wxh, highlight):
        return self.p2s.spreadlinesp(_make_df(), _rels(), ego=['a', 'b'], time='time',
                                     highlight_nodes=highlight, wxh=wxh)

    # Part of the ego set selected -> the ring is drawn as a <path> through a per-bin
    # clipPath.  All of it selected -> the ring is a <use> of the outline cloud instead.
    # The two branches emit different ids, so both are exercised below.
    def _partial_selection(self, wxh): return self._selection(wxh, {'a'})
    def _full_selection(self, wxh):    return self._selection(wxh, {'a', 'b'})

    def _both(self, wxh): return (self._partial_selection(wxh), self._full_selection(wxh))

    @staticmethod
    def _ids(svg):  return re.findall(r'id="([^"]+)"', svg)

    @staticmethod
    def _refs(svg): return re.findall(r'href="#([^"]+)"|url\(#([^)]+)\)', svg)

    def test_both_selection_branches_are_actually_reached(self):
        # Everything below is vacuous if these renders stop emitting the ids at issue.
        _partial_ = self._partial_selection((700, 300)).svg
        self.assertRegex(_partial_, r'<clipPath id="ccl_\d+_\d+"')
        self.assertRegex(_partial_, r'clip-path="url\(#ccl_\d+_\d+\)"')
        self.assertRegex(_partial_, r'<use href="#cloud_\d+"')
        self.assertRegex(self._full_selection((700, 300)).svg,
                         r'<use href="#cloud_outline_\d+"')

    def test_no_bare_ids_survive(self):
        # The exact strings the audit found: `cloud_outline` and `ccl_<bin>` with no
        # random prefix between the name and the bin index.
        for _name_, _sp_ in zip(('partial', 'full'), self._both((700, 300))):
            with self.subTest(selection=_name_):
                self.assertNotRegex(_sp_.svg, r'id="cloud"')
                self.assertNotRegex(_sp_.svg, r'id="cloud_outline"')
                self.assertNotRegex(_sp_.svg, r'id="ccl_\d+"')      # ccl_<bin>, unprefixed
                self.assertNotRegex(_sp_.svg, r'href="#cloud"')
                self.assertNotRegex(_sp_.svg, r'href="#cloud_outline"')

    def test_two_figures_share_no_ids(self):
        for _name_, _fn_ in (('partial', self._partial_selection),
                             ('full',    self._full_selection)):
            with self.subTest(selection=_name_):
                _a_ = self._ids(_fn_((700, 300)).svg)
                _b_ = self._ids(_fn_((500, 200)).svg)
                self.assertNotEqual(_a_, [], 'no ids emitted -- the render changed')
                self.assertEqual(sorted(set(_a_) & set(_b_)), [])

    def test_two_figures_in_one_document_keep_unique_ids(self):
        # The page case, stated directly: concatenate the figures and every id in the
        # combined document must still be unique.  Mixed branches and mixed sizes,
        # because that is what collided -- clip rects placed from each figure's layout.
        _combined_ = ''.join(sp.svg for sp in self._both((700, 300)) + self._both((500, 200)))
        _ids_   = self._ids(_combined_)
        _dupes_ = sorted({i for i in _ids_ if _ids_.count(i) > 1})
        self.assertEqual(_dupes_, [], f'duplicate ids across figures: {_dupes_}')

    def test_every_reference_resolves_within_its_own_figure(self):
        # A scoped id is only worth anything if the reference moved with it.
        for _wxh_ in ((700, 300), (500, 200)):
            for _name_, _sp_ in zip(('partial', 'full'), self._both(_wxh_)):
                with self.subTest(wxh=_wxh_, selection=_name_):
                    _known_ = set(self._ids(_sp_.svg))
                    _used_  = {r for pair in self._refs(_sp_.svg) for r in pair if r}
                    self.assertNotEqual(_used_, set(), 'no references emitted')
                    self.assertEqual(sorted(_used_ - _known_), [],
                                     'reference to an id this figure does not define')


if __name__ == '__main__':
    unittest.main()
