#
# test_dark_palette_golden.py
#
# Golden-file regression tests for the dark palette -- one render per component
# family, which is what pins the whole dark surface without doubling the golden
# corpus (PLANNING.md W3 on per-component tax).
#
# The existing 72 goldens stay light and are untouched: 'light' is still the default,
# so this file ADDS coverage rather than re-baselining any of it.  What these catch
# that tests/test_palettes.py cannot is layout and structure -- a themed color that
# lands on the wrong element still passes a "no light literals" scan.
#
# First run (or UPDATE_GOLDEN=1): golden files are written, tests pass.
# Subsequent runs: SVG must match the golden exactly.
#
import unittest

import polars as pl

from polars2svg import Polars2SVG

from svg_test_utils import assert_svg_matches_golden, assert_image_matches_golden

_XY_DF_ = pl.DataFrame({
    'x':   [1, 2, 3, 4, 5, 6, 7, 8] * 4,
    'y':   [2, 4, 1, 8, 5, 7, 3, 6] * 4,
    'cat': ['a', 'b', 'c', 'a', 'b', 'c', 'd', 'e'] * 4,
    'val': [1.0, 2.5, 3.2, 0.5, 4.4, 2.2, 1.1, 9.9] * 4,
})

# Distinct per-bin row counts -- equal-frequency bins would make the golden
# nondeterministic (histop breaks count ties on group_by order).
_BAR_DF_ = pl.DataFrame({
    'bin': ['alpha'] * 8 + ['beta'] * 6 + ['gamma'] * 4 + ['delta'] * 2,
    'grp': (['x', 'x', 'y', 'z'] * 5),
    'val': ([5.0, 2.0, 8.0, 1.0] * 5),
})

_TS_DF_ = pl.DataFrame({
    'ts': ['2021-01-03', '2021-02-11', '2021-02-27', '2021-05-19',
           '2021-08-01', '2021-11-30', '2021-12-14', '2021-12-29'],
}).with_columns(pl.col('ts').str.to_datetime())

_G_DF_ = pl.DataFrame({
    'fm': ['a', 'b', 'c', 'a', 'd', 'b', 'c', 'a'],
    'to': ['b', 'a', 'a', 'c', 'a', 'c', 'b', 'd'],
})

_SL_DF_ = pl.DataFrame({
    'fm':   ['a', 'b', 'c', 'a', 'd', 'b', 'c', 'a'],
    'to':   ['b', 'a', 'a', 'c', 'a', 'c', 'b', 'd'],
    'time': [1, 1, 1, 2, 2, 2, 3, 3],
})

# Fixed positions: an unpinned graph layout is nondeterministic, and 'b'/'c' share a
# pixel so the cloud <defs> -- the one definition the palette reaches into -- is in
# the golden rather than only in a unit test.
_POS_ = {'a': [0.0, 0.0], 'b': [1.0, 0.0], 'c': [1.0, 0.0], 'd': [0.5, 1.0]}


class TestDarkPaletteGolden(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG(palette='dark')

    def test_xyp_dark(self):
        _r_ = self.p2s.xyp(_XY_DF_, 'x', 'y', color='cat', wxh=(256, 192))
        assert_svg_matches_golden(_r_.svg, 'dark_xyp')
        assert_image_matches_golden(_r_.svg, 'dark_xyp')

    def test_histop_dark(self):
        # draw_labels defaults True here, so this covers label/defaultfg as well.
        _r_ = self.p2s.histop(_BAR_DF_, 'bin', count='val', wxh=(256, 192))
        assert_svg_matches_golden(_r_.svg, 'dark_histop')
        assert_image_matches_golden(_r_.svg, 'dark_histop')

    def test_histop_distribution_dark(self):
        # The grayscale ramp: the one piece of color that had to INVERT rather than
        # just change value, so it gets a golden of its own.
        _r_ = self.p2s.histop(_BAR_DF_, 'bin', count='val', distribution='grp',
                              wxh=(256, 192))
        assert_svg_matches_golden(_r_.svg, 'dark_histop_distribution')
        assert_image_matches_golden(_r_.svg, 'dark_histop_distribution')

    def test_timep_dark(self):
        _r_ = self.p2s.timep(_TS_DF_, ('ts', self.p2s.LT_Y_mp), wxh=(256, 128))
        assert_svg_matches_golden(_r_.svg, 'dark_timep')
        assert_image_matches_golden(_r_.svg, 'dark_timep')

    def test_piep_dark(self):
        # _BAR_DF_ rather than _XY_DF_: its bins have distinct counts.  _XY_DF_['cat']
        # ties a/b/c at 8 and d/e at 4, and piep breaks count ties on group_by order,
        # so that golden did not reproduce between runs.
        _r_ = self.p2s.piep(_BAR_DF_, 'bin', wxh=(192, 192))
        assert_svg_matches_golden(_r_.svg, 'dark_piep')
        assert_image_matches_golden(_r_.svg, 'dark_piep')

    def test_linkp_dark(self):
        _r_ = self.p2s.linkp(_G_DF_, [('fm', 'to')], pos=_POS_,
                             draw_node_labels=True, wxh=(256, 256))
        assert_svg_matches_golden(_r_.svg, 'dark_linkp')
        assert_image_matches_golden(_r_.svg, 'dark_linkp')

    def test_chordp_dark(self):
        _r_ = self.p2s.chordp(_G_DF_, [('fm', 'to')], draw_labels=True, wxh=(256, 256))
        assert_svg_matches_golden(_r_.svg, 'dark_chordp')
        assert_image_matches_golden(_r_.svg, 'dark_chordp')

    def test_spreadlinesp_dark(self):
        # Reaches the direction markers (new/ending) and the ego cloud <defs>.
        _r_ = self.p2s.spreadlinesp(_SL_DF_, [('fm', 'to')], ego='a', time='time',
                                    wxh=(512, 256))
        assert_svg_matches_golden(_r_.svg, 'dark_spreadlinesp')
        assert_image_matches_golden(_r_.svg, 'dark_spreadlinesp')

    def test_smallp_dark(self):
        _r_ = self.p2s.smallp(_XY_DF_, 'cat', self.p2s.xyp(_XY_DF_, 'x', 'y'),
                              wxh=(256, 256))
        assert_svg_matches_golden(_r_.svg, 'dark_smallp')
        assert_image_matches_golden(_r_.svg, 'dark_smallp')

    def test_legend_dark(self):
        # The legend draws its own chrome (fg + muted) rather than reusing the plot's.
        _r_ = self.p2s.xyp(_XY_DF_, 'x', 'y', color='cat', legend='right', wxh=(256, 192))
        assert_svg_matches_golden(_r_.svg, 'dark_legend')
        assert_image_matches_golden(_r_.svg, 'dark_legend')


if __name__ == '__main__':
    unittest.main()
