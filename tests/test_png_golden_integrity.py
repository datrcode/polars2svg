'''Guards on the PNG golden mechanism itself, rather than on any one render.

The bug these exist to prevent
------------------------------
A PNG golden was compared against a single RMS tolerance of 5.0. That number was
chosen to absorb anti-aliasing variation, but it also absorbed goldens that had
simply gone stale -- and nothing distinguished the two cases. Fourteen goldens in
production drifted that way (2026-08-05 font pinning, 2026-08-07 coordinate
rounding, regenerated in dev and never ported); the suite reported green the whole
time, with one file at 3.37/5.0, having quietly spent 67% of the budget that was
supposed to be there for anti-aliasing.

The invariant that makes this detectable: on the platform that generated them,
with the rasterizer that generated them, rasterization is deterministic. A current
golden re-renders BIT-IDENTICALLY, RMS exactly 0.0. So "visually fine" and "still
current" are separate questions and now have separate thresholds.

These tests are deliberately about the harness, not about pixels, so they run
everywhere -- including the Linux CI that skips every actual PNG-RMS comparison.
'''

import os
import re
import unittest

from PIL import Image

import svg_test_utils
from svg_test_utils import (
    GOLDEN_PNG_DIR,
    GOLDEN_PNG_DRIFT_TOLERANCE,
    GOLDEN_PNG_VISUAL_TOLERANCE,
    assert_image_matches_golden,
    goldenPngDriftTolerance,
    pngRMS,
    pngRMSForFiles,
)

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_TESTS_DIR)

# assert_image_matches_golden(<svg expr>, 'golden_name')  -- the second positional
# argument names the golden file. Matched statically so this does not depend on
# running (or being able to run) the renders themselves.
_GOLDEN_CALL = re.compile(r"assert_image_matches_golden\(\s*[^,]+,\s*['\"]([A-Za-z0-9_]+)['\"]")


def _referencedGoldenNames() -> set:
    '''Every golden name any test file passes to assert_image_matches_golden().

    This file is skipped: its own calls name scratch goldens in a temp dir, not
    stored fixtures, and counting them would make the sweep report itself.
    '''
    names = set()
    _self = os.path.basename(__file__)
    for entry in os.listdir(_TESTS_DIR):
        if not (entry.startswith('test_') and entry.endswith('.py')) or entry == _self:
            continue
        with open(os.path.join(_TESTS_DIR, entry), encoding='utf-8') as f:
            names.update(_GOLDEN_CALL.findall(f.read()))
    return names


def _storedGoldenNames() -> set:
    if not os.path.isdir(GOLDEN_PNG_DIR):
        return set()
    return {f[:-4] for f in os.listdir(GOLDEN_PNG_DIR) if f.endswith('.png')}


def _solidPNG(path: str, color: tuple, size: tuple = (40, 40)) -> None:
    Image.new('RGB', size, color).save(path)


class TestThresholdSeparation(unittest.TestCase):
    '''The two thresholds must stay two thresholds.'''

    def test_drift_tolerance_is_tighter_than_visual_tolerance(self):
        # If these ever converge, staleness becomes invisible again -- that
        # collapse IS the original bug, not a simplification of it.
        self.assertLess(GOLDEN_PNG_DRIFT_TOLERANCE, GOLDEN_PNG_VISUAL_TOLERANCE)

    def test_drift_tolerance_default_is_exact(self):
        # Deterministic rasterization on the generating host means a current
        # golden re-renders bit-identically. Anything above 0.0 here is a tuned
        # fudge factor, and a tuned fudge factor is what hid 14 stale files.
        self.assertEqual(GOLDEN_PNG_DRIFT_TOLERANCE, 0.0)

    def test_env_override_parses(self):
        prior = os.environ.get('P2S_PNG_GOLDEN_DRIFT')
        try:
            os.environ['P2S_PNG_GOLDEN_DRIFT'] = '0.25'
            self.assertEqual(goldenPngDriftTolerance(), 0.25)
            os.environ['P2S_PNG_GOLDEN_DRIFT'] = ''
            self.assertEqual(goldenPngDriftTolerance(), GOLDEN_PNG_DRIFT_TOLERANCE)
            os.environ['P2S_PNG_GOLDEN_DRIFT'] = 'not-a-float'
            with self.assertRaises(ValueError):
                goldenPngDriftTolerance()
        finally:
            os.environ.pop('P2S_PNG_GOLDEN_DRIFT', None)
            if prior is not None:
                os.environ['P2S_PNG_GOLDEN_DRIFT'] = prior

    def test_unset_env_uses_default(self):
        prior = os.environ.pop('P2S_PNG_GOLDEN_DRIFT', None)
        try:
            self.assertEqual(goldenPngDriftTolerance(), GOLDEN_PNG_DRIFT_TOLERANCE)
        finally:
            if prior is not None:
                os.environ['P2S_PNG_GOLDEN_DRIFT'] = prior


class TestPngRMS(unittest.TestCase):
    '''The one RMS implementation the assertion, the comparator and CI all share.'''

    def test_identical_images_score_zero(self):
        a = Image.new('RGB', (10, 10), (10, 20, 30))
        self.assertEqual(pngRMS(a, a.copy()), 0.0)

    def test_known_difference(self):
        # A uniform 4-level difference on every channel of every pixel: RMS = 4.
        a = Image.new('RGB', (8, 8), (100, 100, 100))
        b = Image.new('RGB', (8, 8), (104, 104, 104))
        self.assertAlmostEqual(pngRMS(a, b), 4.0, places=6)

    def test_size_mismatch_raises(self):
        a = Image.new('RGB', (8, 8), (0, 0, 0))
        b = Image.new('RGB', (9, 8), (0, 0, 0))
        with self.assertRaises(ValueError):
            pngRMS(a, b)

    def test_rms_for_files_roundtrip(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p_a, p_b = os.path.join(d, 'a.png'), os.path.join(d, 'b.png')
            _solidPNG(p_a, (100, 100, 100))
            _solidPNG(p_b, (104, 104, 104))
            self.assertAlmostEqual(pngRMSForFiles(p_a, p_b), 4.0, places=6)
            self.assertEqual(pngRMSForFiles(p_a, p_a), 0.0)


class TestStaleGoldenIsReported(unittest.TestCase):
    '''The regression test for the reported bug.

    A golden that differs from the render by LESS than the visual tolerance used
    to pass silently. It must now fail, and say why.
    '''

    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        # Point the assertion at a scratch golden dir and a stub rasterizer, so
        # this exercises the comparison logic without depending on a rasterizer
        # being installed or on the platform gate.
        self._prior_dir  = svg_test_utils.GOLDEN_PNG_DIR
        self._prior_rast = svg_test_utils.rasterize_svg
        svg_test_utils.GOLDEN_PNG_DIR = self._tmp.name
        self.addCleanup(self._restore)
        self._prior_force = os.environ.get('P2S_FORCE_PNG_GOLDEN')
        os.environ['P2S_FORCE_PNG_GOLDEN'] = '1'
        os.environ.pop('UPDATE_GOLDEN', None)

    def _restore(self):
        svg_test_utils.GOLDEN_PNG_DIR = self._prior_dir
        svg_test_utils.rasterize_svg  = self._prior_rast
        if self._prior_force is None:
            os.environ.pop('P2S_FORCE_PNG_GOLDEN', None)
        else:
            os.environ['P2S_FORCE_PNG_GOLDEN'] = self._prior_force

    def _stubRender(self, color: tuple, size: tuple = (40, 40)) -> None:
        svg_test_utils.rasterize_svg = lambda svg: Image.new('RGB', size, color)

    def test_drifted_but_visually_close_golden_fails_as_stale(self):
        # RMS 2.0 -- comfortably under the 5.0 visual tolerance. This is exactly
        # the band production's 14 goldens sat in (0.02 - 3.37) while passing.
        _solidPNG(os.path.join(self._tmp.name, 'case.png'), (100, 100, 100))
        self._stubRender((102, 102, 102))
        with self.assertRaises(AssertionError) as ctx:
            assert_image_matches_golden('<svg/>', 'case')
        msg = str(ctx.exception)
        self.assertIn('STALE', msg)
        self.assertIn('UPDATE_GOLDEN=1', msg)
        # It must not be misreported as a visual regression -- different fix.
        self.assertNotIn('VISUAL regression', msg)

    def test_matching_golden_passes(self):
        _solidPNG(os.path.join(self._tmp.name, 'case.png'), (100, 100, 100))
        self._stubRender((100, 100, 100))
        assert_image_matches_golden('<svg/>', 'case')   # must not raise

    def test_large_difference_reports_visual_regression(self):
        _solidPNG(os.path.join(self._tmp.name, 'case.png'), (0, 0, 0))
        self._stubRender((200, 200, 200))
        with self.assertRaises(AssertionError) as ctx:
            assert_image_matches_golden('<svg/>', 'case')
        self.assertIn('VISUAL regression', str(ctx.exception))

    def test_size_mismatch_fails_with_a_clear_message(self):
        _solidPNG(os.path.join(self._tmp.name, 'case.png'), (100, 100, 100), size=(40, 40))
        self._stubRender((100, 100, 100), size=(41, 40))
        with self.assertRaises(AssertionError) as ctx:
            assert_image_matches_golden('<svg/>', 'case')
        self.assertIn('geometry', str(ctx.exception))

    def test_drift_check_can_be_opted_out_per_call_site(self):
        _solidPNG(os.path.join(self._tmp.name, 'case.png'), (100, 100, 100))
        self._stubRender((102, 102, 102))
        assert_image_matches_golden('<svg/>', 'case', drift_tolerance=5.0)   # must not raise

    def test_update_golden_regenerates_and_clears_the_drift(self):
        # The documented fix for a stale golden. If this path ever broke, the
        # new failure would have no remedy and the pressure would be to widen
        # the tolerance again -- which is how the bug started.
        path = os.path.join(self._tmp.name, 'case.png')
        _solidPNG(path, (100, 100, 100))
        self._stubRender((102, 102, 102))
        with self.assertRaises(AssertionError):
            assert_image_matches_golden('<svg/>', 'case')
        os.environ['UPDATE_GOLDEN'] = '1'
        try:
            assert_image_matches_golden('<svg/>', 'case')
        finally:
            os.environ.pop('UPDATE_GOLDEN', None)
        assert_image_matches_golden('<svg/>', 'case')   # now current: must not raise

    def test_missing_golden_still_fails(self):
        self._stubRender((100, 100, 100))
        with self.assertRaises(AssertionError) as ctx:
            assert_image_matches_golden('<svg/>', 'absent')
        self.assertIn('No golden PNG', str(ctx.exception))


class TestNoOrphanGoldens(unittest.TestCase):
    '''Every stored golden is exercised, and every exercised golden is stored.

    An unreferenced PNG is not harmless: the mirror check treats it as a file
    that must exist in both repos, so it is kept in sync forever while proving
    nothing. `_tmp_test_color_none.png` lived in both repos this way from
    2026-07-14 until it was removed with these tests.
    '''

    def setUp(self):
        if not os.path.isdir(GOLDEN_PNG_DIR):
            self.skipTest('tests/golden_png/ not present (installed from wheel)')

    def test_no_stored_golden_is_unreferenced(self):
        orphans = _storedGoldenNames() - _referencedGoldenNames()
        self.assertEqual(orphans, set(),
                         f'PNG goldens on disk that no test references: {sorted(orphans)}. '
                         f'Delete them, or add the test that was meant to use them.')

    def test_every_referenced_golden_exists(self):
        missing = _referencedGoldenNames() - _storedGoldenNames()
        self.assertEqual(missing, set(),
                         f'Tests reference PNG goldens that are not stored: {sorted(missing)}. '
                         f'Create them with UPDATE_GOLDEN=1 and commit them.')


class TestCrossRepoComparator(unittest.TestCase):
    '''tools/compare_png_goldens.py -- dev-only, so absent in production.'''

    def setUp(self):
        self.tool = os.path.join(_REPO_ROOT, 'tools', 'compare_png_goldens.py')
        if not os.path.exists(self.tool):
            self.skipTest('tools/compare_png_goldens.py not present (dev-only tooling)')
        import importlib.util
        spec = importlib.util.spec_from_file_location('compare_png_goldens', self.tool)
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.a = os.path.join(self._tmp.name, 'a')
        self.b = os.path.join(self._tmp.name, 'b')
        os.makedirs(self.a)
        os.makedirs(self.b)

    def _kinds(self) -> list:
        return [k for k, _, _ in self.mod.comparePNGGoldens(self.a, self.b)]

    def test_identical_directories_report_nothing(self):
        _solidPNG(os.path.join(self.a, 'x.png'), (10, 20, 30))
        _solidPNG(os.path.join(self.b, 'x.png'), (10, 20, 30))
        self.assertEqual(self.mod.comparePNGGoldens(self.a, self.b), [])

    def test_same_name_different_image_is_drift(self):
        # The exact condition the byte-excluded mirror check could not see.
        _solidPNG(os.path.join(self.a, 'x.png'), (100, 100, 100))
        _solidPNG(os.path.join(self.b, 'x.png'), (102, 102, 102))
        findings = self.mod.comparePNGGoldens(self.a, self.b)
        self.assertEqual([k for k, _, _ in findings], ['drift'])
        self.assertIn('RMS=2.0000', findings[0][2])

    def test_presence_gaps_reported_in_both_directions(self):
        _solidPNG(os.path.join(self.a, 'only_a.png'), (0, 0, 0))
        _solidPNG(os.path.join(self.b, 'only_b.png'), (0, 0, 0))
        self.assertEqual(sorted(self._kinds()), ['only_in_a', 'only_in_b'])

    def test_size_mismatch_reported(self):
        _solidPNG(os.path.join(self.a, 'x.png'), (0, 0, 0), size=(10, 10))
        _solidPNG(os.path.join(self.b, 'x.png'), (0, 0, 0), size=(11, 10))
        self.assertEqual(self._kinds(), ['size_mismatch'])

    def test_tolerance_is_honored(self):
        _solidPNG(os.path.join(self.a, 'x.png'), (100, 100, 100))
        _solidPNG(os.path.join(self.b, 'x.png'), (102, 102, 102))
        self.assertEqual(self.mod.comparePNGGoldens(self.a, self.b, tolerance=5.0), [])

    def test_exit_status_is_nonzero_on_a_finding(self):
        _solidPNG(os.path.join(self.a, 'x.png'), (100, 100, 100))
        _solidPNG(os.path.join(self.b, 'x.png'), (102, 102, 102))
        self.assertEqual(self.mod.main([self.a, self.b]), 1)
        _solidPNG(os.path.join(self.b, 'x.png'), (100, 100, 100))
        self.assertEqual(self.mod.main([self.a, self.b]), 0)


if __name__ == '__main__':
    unittest.main()
