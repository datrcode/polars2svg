"""Exhaustive histop tests: all combinations of bin_by × count × color.

DataFrame mirrors the scratchpad: one int column, one str column, one float
column, 10 rows.  Every combination of bin_by in {a, b, c},
count in {None, a, b, c}, and color in {None, a, b, c} must construct
and render without raising an exception.
"""
import unittest
import polars as pl
from polars2svg import Polars2SVG


_DF_ = pl.DataFrame({
    'a': [1, 2, 3, 1, 2, 3, 4, 5, 1, 2],
    'b': ['p', 'q', 'r', 's', 'p', 'q', 'r', 's', 'p', 'q'],
    'c': [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5],
})

_COLS_   = ['a', 'b', 'c']
_COUNTS_ = [None, 'a', 'b', 'c']
_COLORS_ = [None, 'a', 'b', 'c']


class TestHistopExhaustive(unittest.TestCase):
    """All bin × count × color combinations must not raise."""

    @classmethod
    def setUpClass(cls):
        cls.p2s = Polars2SVG()

    def _run_combo(self, bin_by, count, color):
        kwargs = {}
        if count is not None:
            kwargs['count'] = count
        if color is not None:
            kwargs['color'] = color
        h = self.p2s.histop(_DF_, bin_by, **kwargs)
        h._repr_svg_()

    def test_exhaustive_no_exceptions(self):
        """Collect all failing (bin, count, color) triples and report them together."""
        failures = []
        for _bin_ in _COLS_:
            for _count_ in _COUNTS_:
                for _color_ in _COLORS_:
                    try:
                        self._run_combo(_bin_, _count_, _color_)
                    except Exception as exc:
                        failures.append(
                            f'bin={_bin_!r} count={_count_!r} color={_color_!r} → {type(exc).__name__}: {exc}'
                        )
        if failures:
            self.fail(f'{len(failures)} combination(s) raised:\n' + '\n'.join(failures))

    # ── spot-checks pinning previously-failing combinations ──

    def test_bin_eq_color_numeric_int(self):
        """bin='a', color='a' (int): bin == color, spectrum mode — was SchemaError."""
        self._run_combo('a', None, 'a')

    def test_bin_eq_color_numeric_float(self):
        """bin='c', color='c' (float): bin == color, spectrum mode — was SchemaError."""
        self._run_combo('c', None, 'c')

    def test_count_eq_color_bin_differs(self):
        """bin='a', count='c', color='c': count == color, bin differs — was ColumnNotFoundError."""
        self._run_combo('a', 'c', 'c')

    def test_count_eq_color_int_bin_str(self):
        """bin='b', count='a', color='a': color consumed by count — was ColumnNotFoundError."""
        self._run_combo('b', 'a', 'a')


if __name__ == '__main__':
    unittest.main()
