import unittest
import polars as pl
from polars2svg import Polars2SVG

from svg_test_utils import assert_svg_matches_golden, assert_image_matches_golden

class Testxyp_golden(unittest.TestCase):
    '''Golden-file regression tests for XYp SVG output.

    Each test renders a small, fully deterministic dataframe and compares the
    normalized SVG to a stored golden file in tests/golden/.

    First run (or UPDATE_GOLDEN=1): golden files are written, tests pass.
    Subsequent runs: SVG must match the golden exactly.
    To regenerate after an intentional visual change: UPDATE_GOLDEN=1 pytest
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    # ------------------------------------------------------------------
    # Basic scatter
    # ------------------------------------------------------------------
    def test_basic_scatter(self):
        df = pl.DataFrame({
            'x': [1, 2, 3, 4, 5, 6],
            'y': [3, 1, 4, 1, 5, 9],
        })
        _xyp_ = self.p2s.xyp(df, 'x', 'y', wxh=(200, 200))
        assert_svg_matches_golden(_xyp_.svg, 'basic_scatter')
        assert_image_matches_golden(_xyp_.svg, 'basic_scatter')

    # ------------------------------------------------------------------
    # Color by a continuous field
    # ------------------------------------------------------------------
    def test_color_continuous(self):
        df = pl.DataFrame({
            'x':     [1, 2, 3, 4, 5, 6, 7, 8],
            'y':     [4, 2, 7, 1, 6, 3, 8, 5],
            'value': [10, 20, 30, 40, 50, 60, 70, 80],
        })
        _xyp_ = self.p2s.xyp(df, 'x', 'y', color='value',
                              dot_size=4, wxh=(200, 200))
        assert_svg_matches_golden(_xyp_.svg, 'color_continuous')
        assert_image_matches_golden(_xyp_.svg, 'color_continuous')

    # ------------------------------------------------------------------
    # Color by a categorical field
    # ------------------------------------------------------------------
    def test_color_categorical(self):
        df = pl.DataFrame({
            'x':      [1, 2, 3, 4, 5, 6],
            'y':      [3, 1, 4, 1, 5, 9],
            'group':  ['a', 'b', 'a', 'b', 'a', 'b'],
        })
        _xyp_ = self.p2s.xyp(df, 'x', 'y', color='group',
                              dot_size=4, wxh=(200, 200))
        assert_svg_matches_golden(_xyp_.svg, 'color_categorical')
        assert_image_matches_golden(_xyp_.svg, 'color_categorical')

    # ------------------------------------------------------------------
    # Variable dot size
    # ------------------------------------------------------------------
    def test_variable_dot_size(self):
        df = pl.DataFrame({
            'x':    [1, 2, 3, 4, 5],
            'y':    [5, 3, 4, 2, 6],
            'size': [1, 3, 2, 5, 4],
        })
        _xyp_ = self.p2s.xyp(df, 'x', 'y', dot_size='size', wxh=(200, 200))
        assert_svg_matches_golden(_xyp_.svg, 'variable_dot_size')
        assert_image_matches_golden(_xyp_.svg, 'variable_dot_size')

    # ------------------------------------------------------------------
    # Distributions on both axes
    # ------------------------------------------------------------------
    def test_distributions(self):
        df = pl.DataFrame({
            'x': [1, 1, 2, 2, 3, 3, 3, 4, 5, 5],
            'y': [2, 4, 1, 3, 2, 3, 5, 4, 1, 3],
        })
        _xyp_ = self.p2s.xyp(df, 'x', 'y',
                              x_distributions=self.p2s.ROW_COUNTp,
                              y_distributions=self.p2s.ROW_COUNTp,
                              wxh=(200, 200))
        assert_svg_matches_golden(_xyp_.svg, 'distributions')
        assert_image_matches_golden(_xyp_.svg, 'distributions')

    # ------------------------------------------------------------------
    # Lines connecting points
    # ------------------------------------------------------------------
    def test_lines(self):
        df = pl.DataFrame({
            'time':   [0, 1, 2, 3, 4, 5, 6, 7],
            'value':  [2, 3, 1, 4, 2, 5, 3, 4],
            'series': ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b'],
        })
        _xyp_ = self.p2s.xyp(df, 'time', 'value',
                              color='series',
                              line=('series',),
                              wxh=(200, 200))
        assert_svg_matches_golden(_xyp_.svg, 'lines')
        assert_image_matches_golden(_xyp_.svg, 'lines')

    # ------------------------------------------------------------------
    # No context (axis labels / grid suppressed)
    # ------------------------------------------------------------------
    def test_no_context(self):
        df = pl.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': [5, 1, 4, 2, 3],
        })
        _xyp_ = self.p2s.xyp(df, 'x', 'y',
                              draw_context=False, wxh=(100, 100))
        assert_svg_matches_golden(_xyp_.svg, 'no_context')
        assert_image_matches_golden(_xyp_.svg, 'no_context')

if __name__ == '__main__':
    unittest.main()
