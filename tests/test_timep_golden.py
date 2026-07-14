import unittest
import polars as pl
from polars2svg import Polars2SVG

from svg_test_utils import assert_svg_matches_golden, assert_image_matches_golden


# 12 months of 2023, 2 rows per month.  Monthly bins are ordered by time
# (not count), so determinism never depends on tie-breaking.
# value sums per month: Jan=30, Feb=50, Mar=90, ..., Dec=450
# score per row: 1.0, 2.0, ..., 24.0 (monotone → clear spectrum gradient)
_DF = pl.DataFrame({
    'ts': [
        '2023-01-05', '2023-01-20',
        '2023-02-10', '2023-02-25',
        '2023-03-08', '2023-03-22',
        '2023-04-12', '2023-04-27',
        '2023-05-03', '2023-05-18',
        '2023-06-15', '2023-06-30',
        '2023-07-07', '2023-07-21',
        '2023-08-02', '2023-08-16',
        '2023-09-09', '2023-09-24',
        '2023-10-11', '2023-10-26',
        '2023-11-04', '2023-11-19',
        '2023-12-13', '2023-12-28',
    ],
    'category': ['A', 'B'] * 12,
    'value':    list(range(10, 250, 10)),   # 10, 20, 30, ..., 240
    'score':    [float(i) for i in range(1, 25)],
}).with_columns(pl.col('ts').str.to_datetime())

# 3 months × 5 rows each — enough for non-degenerate box-plot quantiles.
# Jan: [10,30,50,70,90] Q1=30 med=50 Q3=70
# Feb: [15,25,55,65,85] Q1=25 med=55 Q3=65
# Mar: [ 5,45,60,75,95] Q1=45 med=60 Q3=75
_DF_BOX = pl.DataFrame({
    'ts': [
        '2023-01-03', '2023-01-09', '2023-01-17', '2023-01-23', '2023-01-28',
        '2023-02-05', '2023-02-11', '2023-02-16', '2023-02-21', '2023-02-26',
        '2023-03-02', '2023-03-08', '2023-03-14', '2023-03-20', '2023-03-27',
    ],
    'value': [10, 30, 50, 70, 90, 15, 25, 55, 65, 85, 5, 45, 60, 75, 95],
}).with_columns(pl.col('ts').str.to_datetime())

# Periodic-monthly frame: one row per month in 2022 + a few extras in 2023.
# PT_mp bins are months 1–12, always ordered 1–12 (no tie-breaking).
_DF_PERIODIC = pl.DataFrame({
    'ts': [
        '2022-01-15', '2022-02-10', '2022-03-20', '2022-04-05',
        '2022-05-25', '2022-06-15', '2022-07-30', '2022-08-10',
        '2022-09-20', '2022-10-05', '2022-11-15', '2022-12-25',
        '2023-02-20', '2023-05-10', '2023-08-30', '2023-11-05',
    ],
    'category': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B',
                 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A'],
    'value': list(range(10, 170, 10)),
}).with_columns(pl.col('ts').str.to_datetime())

# Day-of-week frame: covers all 7 days; rows spread across 3 weeks of Jan 2023.
# PT_DoWp bins are 1–7 (Mon–Sun), ordered 1–7 (no tie-breaking).
# Mon: 3, Tue: 2, Wed: 4, Thu: 1, Fri: 2, Sat: 2, Sun: 3
_DF_DOW = pl.DataFrame({
    'ts': [
        '2023-01-02', '2023-01-09', '2023-01-16',   # Mon ×3
        '2023-01-03', '2023-01-10',                   # Tue ×2
        '2023-01-04', '2023-01-11', '2023-01-18', '2023-01-25',  # Wed ×4
        '2023-01-05',                                 # Thu ×1
        '2023-01-06', '2023-01-13',                   # Fri ×2
        '2023-01-07', '2023-01-14',                   # Sat ×2
        '2023-01-08', '2023-01-15', '2023-01-22',    # Sun ×3
    ],
    'value': list(range(10, 180, 10)),
}).with_columns(pl.col('ts').str.to_datetime())


class TestTimepGolden(unittest.TestCase):
    '''Golden-file regression tests for Timep SVG output.

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
    # Basic monthly bar chart — explicit LT_Y_mp, row counts
    # ------------------------------------------------------------------
    def test_linear_monthly(self):
        tp = self.p2s.timep(_DF, ('ts', self.p2s.LT_Y_mp), wxh=(256, 128))
        assert_svg_matches_golden(tp.svg, 'timep_linear_monthly')
        assert_image_matches_golden(tp.svg, 'timep_linear_monthly')

    # ------------------------------------------------------------------
    # Count by a numeric field (sum of value per month)
    # ------------------------------------------------------------------
    def test_linear_count_numeric(self):
        tp = self.p2s.timep(_DF, ('ts', self.p2s.LT_Y_mp),
                             count='value', wxh=(256, 128))
        assert_svg_matches_golden(tp.svg, 'timep_linear_count_numeric')
        assert_image_matches_golden(tp.svg, 'timep_linear_count_numeric')

    # ------------------------------------------------------------------
    # Color by a numeric field — spectrum coloring on whole bar
    # ------------------------------------------------------------------
    def test_linear_color_numeric(self):
        tp = self.p2s.timep(_DF, ('ts', self.p2s.LT_Y_mp),
                             color='score', wxh=(256, 128))
        assert_svg_matches_golden(tp.svg, 'timep_linear_color_numeric')
        assert_image_matches_golden(tp.svg, 'timep_linear_color_numeric')

    # ------------------------------------------------------------------
    # Color by a categorical field — stacked bars (category: A or B)
    # ------------------------------------------------------------------
    def test_linear_color_categorical(self):
        tp = self.p2s.timep(_DF, ('ts', self.p2s.LT_Y_mp),
                             color='category', wxh=(256, 128))
        assert_svg_matches_golden(tp.svg, 'timep_linear_color_categorical')
        assert_image_matches_golden(tp.svg, 'timep_linear_color_categorical')

    # ------------------------------------------------------------------
    # CROW_MAGNITUDEp — whole bar colored by raw row count (linear spectrum)
    # ------------------------------------------------------------------
    def test_linear_crow_magnitude(self):
        tp = self.p2s.timep(_DF, ('ts', self.p2s.LT_Y_mp),
                             count='value', color=self.p2s.CROW_MAGNITUDEp,
                             wxh=(256, 128))
        assert_svg_matches_golden(tp.svg, 'timep_linear_crow_magnitude')
        assert_image_matches_golden(tp.svg, 'timep_linear_crow_magnitude')

    # ------------------------------------------------------------------
    # CROW_STRETCHEDp — whole bar colored by raw row count (rank-based spectrum)
    # ------------------------------------------------------------------
    def test_linear_crow_stretched(self):
        tp = self.p2s.timep(_DF, ('ts', self.p2s.LT_Y_mp),
                             count='value', color=self.p2s.CROW_STRETCHEDp,
                             wxh=(256, 128))
        assert_svg_matches_golden(tp.svg, 'timep_linear_crow_stretched')
        assert_image_matches_golden(tp.svg, 'timep_linear_crow_stretched')

    # ------------------------------------------------------------------
    # Stacked bar chart style with categorical color
    # ------------------------------------------------------------------
    def test_linear_stackedbar(self):
        tp = self.p2s.timep(_DF, ('ts', self.p2s.LT_Y_mp),
                             style=self.p2s.STACKEDBARp, color='category',
                             wxh=(256, 128))
        assert_svg_matches_golden(tp.svg, 'timep_linear_stackedbar')
        assert_image_matches_golden(tp.svg, 'timep_linear_stackedbar')

    # ------------------------------------------------------------------
    # Box plot style — requires a numeric count field; 3 months × 5 rows
    # ------------------------------------------------------------------
    def test_linear_boxplot(self):
        tp = self.p2s.timep(_DF_BOX, ('ts', self.p2s.LT_Y_mp),
                             style=self.p2s.BOXPLOTp, count='value',
                             wxh=(256, 128))
        assert_svg_matches_golden(tp.svg, 'timep_linear_boxplot')
        assert_image_matches_golden(tp.svg, 'timep_linear_boxplot')

    # ------------------------------------------------------------------
    # draw_context=False — time axis and labels suppressed
    # ------------------------------------------------------------------
    def test_no_draw_context(self):
        tp = self.p2s.timep(_DF, ('ts', self.p2s.LT_Y_mp),
                             draw_context=False, wxh=(256, 128))
        assert_svg_matches_golden(tp.svg, 'timep_no_draw_context')
        assert_image_matches_golden(tp.svg, 'timep_no_draw_context')

    # ------------------------------------------------------------------
    # Periodic mode — monthly (PT_mp): 12 fixed bins for months 1–12
    # ------------------------------------------------------------------
    def test_periodic_monthly(self):
        tp = self.p2s.timep(_DF_PERIODIC, ('ts', self.p2s.PT_mp), wxh=(256, 128))
        assert_svg_matches_golden(tp.svg, 'timep_periodic_monthly')
        assert_image_matches_golden(tp.svg, 'timep_periodic_monthly')

    # ------------------------------------------------------------------
    # Periodic mode — day of week (PT_DoWp): 7 fixed bins for Mon–Sun
    # ------------------------------------------------------------------
    def test_periodic_day_of_week(self):
        tp = self.p2s.timep(_DF_DOW, ('ts', self.p2s.PT_DoWp), wxh=(256, 128))
        assert_svg_matches_golden(tp.svg, 'timep_periodic_day_of_week')
        assert_image_matches_golden(tp.svg, 'timep_periodic_day_of_week')


if __name__ == '__main__':
    unittest.main()
