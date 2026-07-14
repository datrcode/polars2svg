import unittest
import math
from polars2svg import Polars2SVG


class TestHeckbertNiceNumbers(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def test_standard_range_returns_nice_value(self):
        i = self.p2s.heckbertNiceNumbers(5, 0, 100)
        self.assertIn(i, [1, 2, 5, 10, 20, 25, 50, 100])

    def test_returns_float(self):
        i = self.p2s.heckbertNiceNumbers(10, 0.0, 1.0)
        self.assertIsInstance(i, float)

    def test_n_ticks_zero_clamps_to_one(self):
        i_zero = self.p2s.heckbertNiceNumbers(0, 0, 100)
        i_one  = self.p2s.heckbertNiceNumbers(1, 0, 100)
        self.assertEqual(i_zero, i_one)

    def test_n_ticks_negative_clamps_to_one(self):
        i_neg = self.p2s.heckbertNiceNumbers(-3, 0, 100)
        i_one = self.p2s.heckbertNiceNumbers(1, 0, 100)
        self.assertEqual(i_neg, i_one)

    def test_equal_min_max_zero_expands(self):
        i = self.p2s.heckbertNiceNumbers(5, 0.0, 0.0)
        self.assertIsInstance(i, float)
        self.assertGreater(i, 0.0)

    def test_equal_min_max_nonzero_expands(self):
        i = self.p2s.heckbertNiceNumbers(5, 5.0, 5.0)
        self.assertIsInstance(i, float)
        self.assertGreater(i, 0.0)

    def test_small_range_produces_positive_interval(self):
        i = self.p2s.heckbertNiceNumbers(4, 0.001, 0.002)
        self.assertGreater(i, 0.0)

    def test_large_range_produces_reasonable_interval(self):
        i = self.p2s.heckbertNiceNumbers(10, 0, 1_000_000)
        self.assertGreater(i, 0.0)
        self.assertLessEqual(i, 1_000_000)

    def test_negative_range_produces_positive_interval(self):
        i = self.p2s.heckbertNiceNumbers(5, -100, -10)
        self.assertGreater(i, 0.0)

    def test_interval_divides_range_into_reasonable_ticks(self):
        _min_, _max_, n_ticks = 0, 100, 5
        i = self.p2s.heckbertNiceNumbers(n_ticks, _min_, _max_)
        n_actual = math.ceil((_max_ - _min_) / i)
        self.assertGreaterEqual(n_actual, 1)
        self.assertLessEqual(n_actual, n_ticks * 3)


class TestHeckbertFirstGridLine(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def test_result_is_multiple_of_i_nice(self):
        i_nice = 20.0
        first = self.p2s.heckbertFirstGridLine(i_nice, 3, 97)
        self.assertAlmostEqual(first % i_nice, 0.0, places=9)

    def test_result_is_lte_min(self):
        first = self.p2s.heckbertFirstGridLine(20.0, 3, 97)
        self.assertLessEqual(first, 3)

    def test_exact_min_on_grid_line(self):
        first = self.p2s.heckbertFirstGridLine(10.0, 10.0, 90.0)
        self.assertAlmostEqual(first, 10.0, places=9)

    def test_zero_min(self):
        first = self.p2s.heckbertFirstGridLine(5.0, 0.0, 25.0)
        self.assertAlmostEqual(first, 0.0, places=9)

    def test_negative_min(self):
        first = self.p2s.heckbertFirstGridLine(10.0, -55.0, 100.0)
        self.assertAlmostEqual(first % 10.0, 0.0, places=9)
        self.assertLessEqual(first, -55.0)

    def test_fractional_i_nice(self):
        first = self.p2s.heckbertFirstGridLine(0.5, 1.3, 3.7)
        self.assertAlmostEqual(first % 0.5, 0.0, places=9)
        self.assertLessEqual(first, 1.3)


if __name__ == '__main__':
    unittest.main()
