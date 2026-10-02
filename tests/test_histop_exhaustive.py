"""Exhaustive histop tests: all combinations of bin_by × count × color.

DataFrame mirrors the scratchpad: one int column, one str column, one float
column, 10 rows.  Every combination of bin_by in {a, b, c},
count in {None, a, b, c}, and color in {None, a, b, c} must construct
and render without raising an exception.
"""
import unittest
import polars as pl
from polars2svg import Polars2SVG
from histop_dataframes import HistopAssertions, orderedBins, spectrumColors


_DF_ = pl.DataFrame({
    'a': [1, 2, 3, 1, 2, 3, 4, 5, 1, 2],
    'b': ['p', 'q', 'r', 's', 'p', 'q', 'r', 's', 'p', 'q'],
    'c': [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5],
})

_COLS_   = ['a', 'b', 'c']
_COUNTS_ = [None, 'a', 'b', 'c']
_COLORS_ = [None, 'a', 'b', 'c']


class TestHistopExhaustive(HistopAssertions, unittest.TestCase):
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

    # The bars are the count metric per bin (rows, or the count column's sum), largest
    # first, each coloured by the spectrum of the colour column's per-bin sum -- which
    # still holds when the colour column is also the bin or the count column.
    def assertComboShows(self, bin_by, count, color) -> None:
        _per_ = lambda expr: dict(_DF_.group_by(bin_by).agg(expr.alias('m')).iter_rows())  # noqa: E731
        _h_ = self.p2s.histop(_DF_, bin_by, color=color, **({} if count is None else {'count': count}))
        _length_ = _per_(pl.len() if count is None else pl.col(count).sum())
        self.assertBarsShow(_h_, _length_, orderedBins(_length_))
        self.assertBarColors(_h_, spectrumColors(self.p2s, _per_(pl.col(color).sum())))

    # ── spot-checks pinning previously-failing combinations ──

    def test_bin_eq_color_numeric_int(self):
        """bin='a', color='a' (int): bin == color, spectrum mode — was SchemaError."""
        self.assertComboShows('a', None, 'a')

    def test_bin_eq_color_numeric_float(self):
        """bin='c', color='c' (float): bin == color, spectrum mode — was SchemaError."""
        self.assertComboShows('c', None, 'c')

    def test_count_eq_color_bin_differs(self):
        """bin='a', count='c', color='c': count == color, bin differs — was ColumnNotFoundError."""
        self.assertComboShows('a', 'c', 'c')

    def test_count_eq_color_int_bin_str(self):
        """bin='b', count='a', color='a': color consumed by count — was ColumnNotFoundError."""
        self.assertComboShows('b', 'a', 'a')


if __name__ == '__main__':
    unittest.main()
