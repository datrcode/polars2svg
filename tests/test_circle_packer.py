import random
import unittest
from math import sqrt

from polars2svg import Polars2SVG
from polars2svg.circle_packer import CirclePacker

_P2S_ = Polars2SVG()


def _pack(circles, **kwargs):
    return CirclePacker(_P2S_, circles, **kwargs)


class TestCirclePackerCorrectness(unittest.TestCase):

    def test_packed_count_equals_input(self):
        cp = _pack([(0, 0, 1)] * 5)
        self.assertEqual(len(cp.packedCircles()), 5)

    def test_no_overlaps_after_packing(self):
        circles = [(0, 0, 1), (0, 0, 2), (0, 0, 0.5), (0, 0, 1.5), (0, 0, 3)]
        cp = _pack(circles)
        self.assertTrue(cp.__validateNoOverlaps__())

    def test_packed_circles_returns_deep_copy(self):
        cp = _pack([(0, 0, 1)] * 4)
        r1 = cp.packedCircles()
        r2 = cp.packedCircles()
        self.assertIsNot(r1, r2)

    def test_packed_circle_tuples_have_three_elements(self):
        cp = _pack([(0, 0, 1), (0, 0, 2), (0, 0, 0.5)])
        for c in cp.packedCircles():
            self.assertEqual(len(c), 3)

    def test_validate_chains_pass_after_pack(self):
        cp = _pack([(0, 0, 1)] * 6)
        cp.__validateChains__()  # must not raise

    def test_two_circles(self):
        cp = _pack([(0, 0, 1), (0, 0, 2)])
        self.assertEqual(len(cp.packedCircles()), 2)
        self.assertTrue(cp.__validateNoOverlaps__())

    def test_three_circles(self):
        cp = _pack([(0, 0, 1), (0, 0, 1), (0, 0, 1)])
        self.assertEqual(len(cp.packedCircles()), 3)
        self.assertTrue(cp.__validateNoOverlaps__())

    def test_uniform_radii_indices_in_range(self):
        # r_max/r_min == 1.0 <= 4.125: no sort, keys are original indices 0-5
        cp = _pack([(0, 0, 1)] * 6)
        for k in set(cp.fwd) | set(cp.bck):
            self.assertIn(k, range(6))

    def test_varied_radii_keep_order_original_indices(self):
        # r_max/r_min == 10 > 4.125: sort triggered; keep_order=True restores original indices
        circles = [(0, 0, 1), (0, 0, 10), (0, 0, 1), (0, 0, 1), (0, 0, 1)]
        cp = _pack(circles, keep_order=True)
        for k in set(cp.fwd) | set(cp.bck):
            self.assertIn(k, range(5))

    def test_packed_circles_radii_unchanged(self):
        radii = [1.0, 2.0, 0.5, 1.5]
        cp = _pack([(0, 0, r) for r in radii])
        result_radii = sorted(c[2] for c in cp.packedCircles())
        self.assertEqual(result_radii, sorted(radii))


class TestCirclePackerChainExceptions(unittest.TestCase):
    """All three __validateChains__ exception paths, triggered by deliberate state corruption."""

    def setUp(self):
        self.cp = _pack([(0, 0, 1)] * 5)

    def test_length_mismatch_extra_fwd_key_raises(self):
        self.cp.fwd[9999] = 0  # unmatched key in fwd
        with self.assertRaises(Exception) as ctx:
            self.cp.__validateChains__()
        self.assertIn('Chains Are Different Lengths', str(ctx.exception))

    def test_length_mismatch_missing_bck_key_raises(self):
        del self.cp.bck[next(iter(self.cp.bck))]  # remove a key from bck only
        with self.assertRaises(Exception) as ctx:
            self.cp.__validateChains__()
        self.assertIn('Chains Are Different Lengths', str(ctx.exception))

    def test_fwd_key_absent_from_bck_raises(self):
        # Same length but a fwd key is absent from bck:
        # add 9999 to fwd and 8888 to bck so lengths stay equal
        self.cp.fwd[9999] = 0
        self.cp.bck[8888] = 0
        with self.assertRaises(Exception) as ctx:
            self.cp.__validateChains__()
        self.assertIn('Forward Chain Has Key Not In Backward Chain', str(ctx.exception))

    def test_broken_bck_invariant_raises(self):
        # bck[fwd[k]] must equal k; break that invariant for the first key
        k = next(iter(self.cp.fwd))
        self.cp.bck[self.cp.fwd[k]] = 9999
        with self.assertRaises(Exception) as ctx:
            self.cp.__validateChains__()
        self.assertIn('Backward Chain Has Key Not In Forward Chain', str(ctx.exception))


class TestCirclePackerKnownLimitation(unittest.TestCase):

    def test_into_circle_raises_attribute_error(self):
        # threePointCircle is not ported into polars2svg; packedCircles(into_circle=...)
        # calls optimalInscriptionCircle which calls rt_self.threePointCircle
        cp = _pack([(0, 0, 1)] * 4)
        with self.assertRaises(AttributeError):
            cp.packedCircles(into_circle=(0, 0, 10))


class TestCirclePackerEdgeCases(unittest.TestCase):
    """Boundary conditions not exercised by the main correctness suite."""

    def test_one_circle(self):
        cp = _pack([(0, 0, 5)])
        result = cp.packedCircles()
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0][2], 5.0)

    def test_four_circles_count_and_no_overlaps(self):
        # Exercises the full __packFirstCircles__ path (handles up to 4 explicitly).
        cp = _pack([(0, 0, 1), (0, 0, 2), (0, 0, 1.5), (0, 0, 0.8)])
        self.assertEqual(len(cp.packedCircles()), 4)
        self.assertTrue(cp.__validateNoOverlaps__())

    def test_r_min_r_max_set_correctly(self):
        circles = [(0, 0, 1.0), (0, 0, 3.0), (0, 0, 2.0)]
        cp = _pack(circles)
        self.assertAlmostEqual(cp.r_min, 1.0)
        self.assertAlmostEqual(cp.r_max, 3.0)

    def test_packed_bounding_box_centered_near_origin_small(self):
        # __recenterPackedCircles__ runs after __packFirstCircles__ (≤4 circles).
        # For ≤3 inputs circles_left is exhausted before the main loop, so the
        # final packed list IS the centered result.
        cp = _pack([(0, 0, 1), (0, 0, 1), (0, 0, 1)])
        xs = [c[0] for c in cp.packed]
        ys = [c[1] for c in cp.packed]
        rs = [c[2] for c in cp.packed]
        x0 = min(x - r for x, r in zip(xs, rs))
        x1 = max(x + r for x, r in zip(xs, rs))
        y0 = min(y - r for y, r in zip(ys, rs))
        y1 = max(y + r for y, r in zip(ys, rs))
        self.assertAlmostEqual((x0 + x1) / 2.0, 0.0, places=6)
        self.assertAlmostEqual((y0 + y1) / 2.0, 0.0, places=6)

    def test_keep_order_false_wide_range(self):
        # keep_order=False skips the index-restore step; fwd/bck keys are
        # internal packing indices, not original input indices.
        circles = [(0, 0, 1), (0, 0, 10), (0, 0, 1), (0, 0, 1), (0, 0, 1)]
        cp = _pack(circles, keep_order=False)
        self.assertEqual(len(cp.packedCircles()), 5)
        self.assertTrue(cp.__validateNoOverlaps__())

    def test_largest_to_smallest_false_wide_range(self):
        # largest_to_smallest=False sorts small→large when ratio > 4.125.
        circles = [(0, 0, r) for r in [1.0, 10.0, 2.0, 5.0, 1.0]]
        cp = _pack(circles, largest_to_smallest=False)
        self.assertEqual(len(cp.packedCircles()), 5)
        self.assertTrue(cp.__validateNoOverlaps__())


class TestPackCirclesMixin(unittest.TestCase):
    """Exercises the packCircles() convenience wrapper on Polars2SVG."""

    def test_pack_circles_returns_correct_count(self):
        circles = [(0, 0, r) for r in [1.0, 2.0, 0.5, 1.5, 3.0]]
        result = _P2S_.packCircles(circles)
        self.assertEqual(len(result), 5)

    def test_pack_circles_no_overlaps(self):
        circles = [(0, 0, r) for r in [1.0, 2.0, 0.5, 1.5, 3.0]]
        result = _P2S_.packCircles(circles)
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                c0, c1 = result[i], result[j]
                dist = sqrt((c0[0] - c1[0])**2 + (c0[1] - c1[1])**2)
                self.assertGreaterEqual(dist + 1e-9, c0[2] + c1[2],
                                        msg=f'Overlap between circle {i} and {j}')

    def test_pack_circles_radii_preserved(self):
        radii = [1.0, 2.0, 0.5, 1.5]
        result = _P2S_.packCircles([(0, 0, r) for r in radii])
        self.assertEqual(sorted(c[2] for c in result), sorted(radii))

    def test_pack_circles_tuples_have_three_elements(self):
        result = _P2S_.packCircles([(0, 0, 1)] * 6)
        for c in result:
            self.assertEqual(len(c), 3)


class TestCirclePackerRandomized(unittest.TestCase):
    """
    Stress tests adapted from racetrack_svg_framework/tests/test_circle_packer.py.
    Three radius-range regimes to cover both the sort and no-sort code paths.
    """

    ITERATIONS = 100
    N_CIRCLES  = 50

    def _no_overlaps(self, packed, epsilon=0.01):
        for i in range(len(packed)):
            c0 = packed[i]
            for j in range(i + 1, len(packed)):
                c1 = packed[j]
                if _P2S_.circlesOverlap((c0[0], c0[1], c0[2]),
                                        (c1[0], c1[1], c1[2] - epsilon)):
                    return False
        return True

    def test_random_narrow_range_ratio_below_threshold(self):
        # radius in [1.0, 4.0] → max ratio 4.0 ≤ 4.125 → NO sort triggered
        rng = random.Random(42)
        for _ in range(self.ITERATIONS):
            circles = [(0.0, 0.0, 1.0 + 3.0 * rng.random()) for _ in range(self.N_CIRCLES)]
            cp = _pack(circles)
            self.assertTrue(cp.__validateNoOverlaps__(), 'Overlaps found (narrow range)')

    def test_random_small_narrow_range_ratio_below_threshold(self):
        # radius in [0.1, 0.4125] → max ratio 4.125 → NO sort triggered
        rng = random.Random(43)
        for _ in range(self.ITERATIONS):
            circles = [(0.0, 0.0, 0.1 + 0.3125 * rng.random()) for _ in range(self.N_CIRCLES)]
            cp = _pack(circles)
            self.assertTrue(cp.__validateNoOverlaps__(), 'Overlaps found (small narrow range)')

    def test_random_wide_range_ratio_above_threshold(self):
        # radius in [0.1, 10.1] → max ratio ≈ 101 > 4.125 → sort IS triggered
        rng = random.Random(44)
        for _ in range(self.ITERATIONS):
            circles = [(0.0, 0.0, 0.1 + 10.0 * rng.random()) for _ in range(self.N_CIRCLES)]
            cp = _pack(circles)
            self.assertTrue(cp.__validateNoOverlaps__(), 'Overlaps found (wide range)')

    def test_random_pack_circles_mixin_no_overlaps(self):
        # Same randomized check exercised through the mixin wrapper.
        rng = random.Random(45)
        for _ in range(self.ITERATIONS):
            circles = [(0.0, 0.0, 1.0 + 3.0 * rng.random()) for _ in range(self.N_CIRCLES)]
            result = _P2S_.packCircles(circles)
            self.assertTrue(self._no_overlaps(result), 'Mixin packCircles produced overlaps')


if __name__ == '__main__':
    unittest.main()
