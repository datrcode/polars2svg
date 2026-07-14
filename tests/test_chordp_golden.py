import unittest
import polars as pl
from polars2svg import Polars2SVG

from svg_test_utils import assert_svg_matches_golden, assert_image_matches_golden


_DF_ = pl.DataFrame({
    'fm':       ['a', 'b', 'c', 'd', 'a', 'c'],
    'to':       ['b', 'c', 'd', 'a', 'c', 'a'],
    'category': ['x', 'y', 'y', 'x', 'x', 'y'],
    'weight':   [2.0, 5.0, 3.0, 1.0, 4.0, 2.0],
})


def _params(**extra):
    return dict(df=_DF_, relationships=[('fm', 'to')], wxh=(128, 128), **extra)


class TestChordPGolden(unittest.TestCase):
    '''Golden-file regression tests for ChP SVG output.

    First run (or UPDATE_GOLDEN=1): golden files are written, tests pass.
    Subsequent runs: SVG must match the golden exactly.
    To regenerate after an intentional visual change: UPDATE_GOLDEN=1 pytest
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def test_default(self):
        cp = self.p2s.chordp(**_params())
        assert_svg_matches_golden(cp.svg, 'chordp_default')
        assert_image_matches_golden(cp.svg, 'chordp_default')

    def test_node_size_vary(self):
        cp = self.p2s.chordp(**_params(node_size='vary'))
        assert_svg_matches_golden(cp.svg, 'chordp_node_size_vary')
        assert_image_matches_golden(cp.svg, 'chordp_node_size_vary')

    def test_node_color_hex(self):
        cp = self.p2s.chordp(**_params(node_color='#ff0000'))
        assert_svg_matches_golden(cp.svg, 'chordp_node_color_hex')
        assert_image_matches_golden(cp.svg, 'chordp_node_color_hex')

    def test_node_color_by_name(self):
        p2s = self.p2s
        cp = self.p2s.chordp(**_params(node_color=p2s.COLOR_BY_NODE_NAME))
        assert_svg_matches_golden(cp.svg, 'chordp_node_color_by_name')
        assert_image_matches_golden(cp.svg, 'chordp_node_color_by_name')

    def test_link_color_src(self):
        cp = self.p2s.chordp(**_params(color='src'))
        assert_svg_matches_golden(cp.svg, 'chordp_link_color_src')
        assert_image_matches_golden(cp.svg, 'chordp_link_color_src')

    def test_link_color_vary(self):
        cp = self.p2s.chordp(**_params(color='category'))
        assert_svg_matches_golden(cp.svg, 'chordp_link_color_vary')
        assert_image_matches_golden(cp.svg, 'chordp_link_color_vary')

    def test_link_shape_bundled(self):
        cp = self.p2s.chordp(**_params(link_shape='bundled'))
        assert_svg_matches_golden(cp.svg, 'chordp_link_shape_bundled')
        assert_image_matches_golden(cp.svg, 'chordp_link_shape_bundled')

    def test_draw_labels_radial(self):
        cp = self.p2s.chordp(**_params(draw_labels=True))
        assert_svg_matches_golden(cp.svg, 'chordp_draw_labels_radial')
        assert_image_matches_golden(cp.svg, 'chordp_draw_labels_radial')

    def test_node_selection(self):
        cp = self.p2s.chordp(**_params(node_selection={'a', 'c'}))
        assert_svg_matches_golden(cp.svg, 'chordp_node_selection')
        assert_image_matches_golden(cp.svg, 'chordp_node_selection')

    def test_count_weight(self):
        cp = self.p2s.chordp(**_params(count='weight'))
        assert_svg_matches_golden(cp.svg, 'chordp_count_weight')
        assert_image_matches_golden(cp.svg, 'chordp_count_weight')


if __name__ == '__main__':
    unittest.main()
