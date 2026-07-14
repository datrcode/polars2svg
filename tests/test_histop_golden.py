import unittest
import polars as pl
from polars2svg import Polars2SVG

from svg_test_utils import assert_svg_matches_golden, assert_image_matches_golden


# Simple 6-row frame: A=3 rows, B=2 rows, C=1 row.
# val sums:   A=60, B=40, C=50   → val-order: A, C, B
# score mean: A=2.0, B=2.0, C=5.0
_DF = pl.DataFrame({
    'cat':   ['A', 'A', 'A', 'B', 'B', 'C'],
    'grp':   ['x', 'x', 'y', 'x', 'y', 'y'],
    'val':   [10,  20,  30,  15,  25,  50 ],
    'score': [1.0, 2.0, 3.0, 1.5, 2.5, 5.0],
})

# Boxplot needs enough points per bin for meaningful quantiles, and unique
# row counts per bin so the descending sort order is deterministic: A, B, C.
_DF_BOX = pl.DataFrame({
    'cat': ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'C', 'C', 'C'],
    'val': [10,  20,  30,  40,  50,  15,  25,  35,  45,  5,   50,  95 ],
})

# Multi-field bin frame: each (cat, grp) combination has a unique row count
# so the sort order is fully deterministic (no ties).
# A|x=4, A|y=3, B|x=2, B|y=1  → descending: A|x, A|y, B|x, B|y
_DF_MULTI = pl.DataFrame({
    'cat': ['A'] * 4 + ['A'] * 3 + ['B'] * 2 + ['B'] * 1,
    'grp': ['x'] * 4 + ['y'] * 3 + ['x'] * 2 + ['y'] * 1,
})


class TestHistopGolden(unittest.TestCase):
    '''Golden-file regression tests for Histop SVG output.

    Each test renders a small, fully deterministic DataFrame and compares the
    normalized SVG to a stored golden file in tests/golden/.

    First run (or UPDATE_GOLDEN=1): golden files are written, tests pass.
    Subsequent runs: SVG must match the golden exactly.
    To regenerate after an intentional visual change: UPDATE_GOLDEN=1 pytest
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    # ------------------------------------------------------------------
    # Basic bar chart — row counts, default descending order (A=3, B=2, C=1)
    # ------------------------------------------------------------------
    def test_basic_barchart(self):
        hp = self.p2s.histop(_DF, 'cat', wxh=(128, 128))
        assert_svg_matches_golden(hp.svg, 'histop_basic_barchart')
        assert_image_matches_golden(hp.svg, 'histop_basic_barchart')

    # ------------------------------------------------------------------
    # Count by a numeric field (sum of val per bin: A=60, B=40, C=50)
    # ------------------------------------------------------------------
    def test_count_numeric_field(self):
        hp = self.p2s.histop(_DF, 'cat', count='val', wxh=(128, 128))
        assert_svg_matches_golden(hp.svg, 'histop_count_numeric_field')
        assert_image_matches_golden(hp.svg, 'histop_count_numeric_field')

    # ------------------------------------------------------------------
    # Color by a numeric field — spectrum coloring on whole bar
    # ------------------------------------------------------------------
    def test_color_numeric(self):
        hp = self.p2s.histop(_DF, 'cat', color='score', wxh=(128, 128))
        assert_svg_matches_golden(hp.svg, 'histop_color_numeric')
        assert_image_matches_golden(hp.svg, 'histop_color_numeric')

    # ------------------------------------------------------------------
    # Color by a categorical field — stacked bars (grp: x or y)
    # ------------------------------------------------------------------
    def test_color_categorical(self):
        hp = self.p2s.histop(_DF, 'cat', color='grp', wxh=(128, 128))
        assert_svg_matches_golden(hp.svg, 'histop_color_categorical')
        assert_image_matches_golden(hp.svg, 'histop_color_categorical')

    # ------------------------------------------------------------------
    # CROW_MAGNITUDEp — whole bar colored by raw row count (linear spectrum)
    # ------------------------------------------------------------------
    def test_color_crow_magnitude(self):
        hp = self.p2s.histop(_DF, 'cat', count='val',
                              color=self.p2s.CROW_MAGNITUDEp, wxh=(128, 128))
        assert_svg_matches_golden(hp.svg, 'histop_color_crow_magnitude')
        assert_image_matches_golden(hp.svg, 'histop_color_crow_magnitude')

    # ------------------------------------------------------------------
    # CROW_STRETCHEDp — whole bar colored by raw row count (rank-based spectrum)
    # ------------------------------------------------------------------
    def test_color_crow_stretched(self):
        hp = self.p2s.histop(_DF, 'cat', count='val',
                              color=self.p2s.CROW_STRETCHEDp, wxh=(128, 128))
        assert_svg_matches_golden(hp.svg, 'histop_color_crow_stretched')
        assert_image_matches_golden(hp.svg, 'histop_color_crow_stretched')

    # ------------------------------------------------------------------
    # Stacked bar chart style with categorical color
    # ------------------------------------------------------------------
    def test_style_stackedbar(self):
        hp = self.p2s.histop(_DF, 'cat', style=self.p2s.STACKEDBARp,
                              color='grp', wxh=(128, 128))
        assert_svg_matches_golden(hp.svg, 'histop_style_stackedbar')
        assert_image_matches_golden(hp.svg, 'histop_style_stackedbar')

    # ------------------------------------------------------------------
    # Box plot style — requires a numeric count field
    # ------------------------------------------------------------------
    def test_style_boxplot(self):
        hp = self.p2s.histop(_DF_BOX, 'cat', style=self.p2s.BOXPLOTp,
                              count='val', wxh=(128, 128))
        assert_svg_matches_golden(hp.svg, 'histop_style_boxplot')
        assert_image_matches_golden(hp.svg, 'histop_style_boxplot')

    # ------------------------------------------------------------------
    # Ascending order (C=1 row first, A=3 rows last)
    # ------------------------------------------------------------------
    def test_ascending_order(self):
        hp = self.p2s.histop(_DF, 'cat', descending=False, wxh=(128, 128))
        assert_svg_matches_golden(hp.svg, 'histop_ascending_order')
        assert_image_matches_golden(hp.svg, 'histop_ascending_order')

    # ------------------------------------------------------------------
    # Order by a numeric field sum (val sums: A=60, C=50, B=40)
    # ------------------------------------------------------------------
    def test_order_by_field(self):
        hp = self.p2s.histop(_DF, 'cat', order='val', wxh=(128, 128))
        assert_svg_matches_golden(hp.svg, 'histop_order_by_field')
        assert_image_matches_golden(hp.svg, 'histop_order_by_field')

    # ------------------------------------------------------------------
    # draw_context=False — axes and labels suppressed
    # ------------------------------------------------------------------
    def test_no_draw_context(self):
        hp = self.p2s.histop(_DF, 'cat', draw_context=False, wxh=(128, 128))
        assert_svg_matches_golden(hp.svg, 'histop_no_draw_context')
        assert_image_matches_golden(hp.svg, 'histop_no_draw_context')

    # ------------------------------------------------------------------
    # Distribution strip below bars (default is on; explicit here)
    # ------------------------------------------------------------------
    def test_distribution_strip(self):
        hp = self.p2s.histop(_DF, 'cat', distribution=True, wxh=(128, 128))
        assert_svg_matches_golden(hp.svg, 'histop_distribution_strip')
        assert_image_matches_golden(hp.svg, 'histop_distribution_strip')

    # ------------------------------------------------------------------
    # Multi-field bin_by — bins are "cat|grp" combinations with unique counts
    # so the descending sort order is fully deterministic: A|x, A|y, B|x, B|y
    # ------------------------------------------------------------------
    def test_multi_field_bin(self):
        hp = self.p2s.histop(_DF_MULTI, ('cat', 'grp'), wxh=(160, 160))
        assert_svg_matches_golden(hp.svg, 'histop_multi_field_bin')
        assert_image_matches_golden(hp.svg, 'histop_multi_field_bin')


if __name__ == '__main__':
    unittest.main()
