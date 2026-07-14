"""Exhaustive timep tests: all combinations of count × color.

DataFrame has 10 rows: one datetime column (ts), one int column (value),
one str column (category), one float column (numeric).  Every combination
of count in {None, value, category, numeric} and
color in {None, value, category, numeric} must construct and render
without raising an exception, in both linear and periodic modes.
"""
import unittest
import polars as pl
from polars2svg import Polars2SVG


_DF_ = pl.DataFrame({
    'ts':       ['2023-01-05', '2023-02-10', '2023-03-15', '2023-04-20',
                 '2023-05-25', '2023-06-30', '2023-07-04', '2023-08-08',
                 '2023-09-12', '2023-10-17'],
    'value':    [10, 25, 7, 40, 13, 55, 3, 28, 61, 19],
    'category': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C'],
    'numeric':  [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 0.5],
}).with_columns(pl.col('ts').str.to_datetime())

_COUNTS_ = [None, 'value', 'category', 'numeric']
_COLORS_ = [None, 'value', 'category', 'numeric']


class TestTimepExhaustive(unittest.TestCase):
    """All count × color combinations must not raise, in both linear and periodic modes."""

    @classmethod
    def setUpClass(cls):
        cls.p2s = Polars2SVG()

    def _run_combo(self, time_field, count, color):
        kwargs = {}
        if count is not None:
            kwargs['count'] = count
        if color is not None:
            kwargs['color'] = color
        t = self.p2s.timep(_DF_, time_field, **kwargs)
        t._repr_svg_()

    def test_exhaustive_linear_no_exceptions(self):
        """Collect all failing (count, color) pairs in linear mode and report together."""
        failures = []
        for _count_ in _COUNTS_:
            for _color_ in _COLORS_:
                try:
                    self._run_combo('ts', _count_, _color_)
                except Exception as exc:
                    failures.append(
                        f'count={_count_!r} color={_color_!r} → {type(exc).__name__}: {exc}'
                    )
        if failures:
            self.fail(f'{len(failures)} linear combination(s) raised:\n' + '\n'.join(failures))

    def test_exhaustive_periodic_no_exceptions(self):
        """Collect all failing (count, color) pairs in periodic (month) mode and report together."""
        failures = []
        for _count_ in _COUNTS_:
            for _color_ in _COLORS_:
                try:
                    self._run_combo(('ts', self.p2s.PT_mp), _count_, _color_)
                except Exception as exc:
                    failures.append(
                        f'count={_count_!r} color={_color_!r} → {type(exc).__name__}: {exc}'
                    )
        if failures:
            self.fail(f'{len(failures)} periodic combination(s) raised:\n' + '\n'.join(failures))

    # ── spot-check: a previously-failing combination ─────────────────────────

    def test_color_numeric_linear(self):
        """color='value' (int, spectrum mode) in linear timep — was ColumnNotFoundError."""
        self._run_combo('ts', None, 'value')

    def test_color_numeric_float_linear(self):
        """color='numeric' (float, spectrum mode) in linear timep — was ColumnNotFoundError."""
        self._run_combo('ts', None, 'numeric')

    def test_color_numeric_with_count_linear(self):
        """color='numeric', count='value' in linear timep — spectrum + separate count."""
        self._run_combo('ts', 'value', 'numeric')

    def test_color_same_as_count_linear(self):
        """color='value', count='value' — color consumed by count — was ColumnNotFoundError."""
        self._run_combo('ts', 'value', 'value')


if __name__ == '__main__':
    unittest.main()
