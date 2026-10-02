'''Per-file ceiling on test functions that contain no assertion (PLANNING.md V11).

A test that asserts nothing passes whether the code is right or wrong.  All it
proves is that nothing raised.  Several component constructors render eagerly, so
such a test does exercise the render -- and that is exactly how the categorical-
label nondeterminism lived under tests/test_xyp_order.py, which ran both execution
modes over list, tuple and dict orders and asserted nothing about either.

The count is tools/blind_tests.py's.  Read its header for what counts as an
assertion, why the number is an upper bound (assertions inside shared helpers are
not followed), and the bar for fixing one: assert on what the render drew, and
where two paths should agree, assert that they do.  Never `assertIsNotNone(x)`.

The rule: these numbers may fall, never rise.  A test file not listed here must
contain no blind tests.  Adding an assertion to a blind test lowers its file's
count, and test_ceilings_are_not_stale fails until the number here is lowered to
match; delete a file's line when it reaches zero.

Every line has reached zero (PLANNING.md V11), so the table is empty and any blind
test in any file fails the suite.  If a genuine exception is ever wanted -- a test
whose whole claim is that something does not raise -- give it an assertion first
(the tool header lists what counts); list a ceiling here only as a last resort, with
a comment saying why.

tools/ is dev-only, so production skips this, as it skips TestCrossRepoComparator.
'''
import importlib.util
import os
import unittest
from pathlib import Path

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TOOL      = os.path.join(_REPO_ROOT, 'tools', 'blind_tests.py')
_TESTS     = os.path.join(_REPO_ROOT, 'tests')


class TestBlindTestRatchet(unittest.TestCase):

    MAX_BLIND: dict = {}

    @classmethod
    def setUpClass(cls) -> None:
        if not os.path.exists(_TOOL):
            raise unittest.SkipTest('tools/blind_tests.py not present (dev-only tooling)')
        _spec_ = importlib.util.spec_from_file_location('blind_tests', _TOOL)
        assert _spec_ is not None and _spec_.loader is not None
        _mod_  = importlib.util.module_from_spec(_spec_)
        _spec_.loader.exec_module(_mod_)
        _total_, cls.blind = _mod_.scanTests(Path(_TESTS))

    def test_no_file_gains_blind_tests(self) -> None:
        for _file_, _names_ in sorted(self.blind.items()):
            with self.subTest(file=_file_):
                _ceiling_ = self.MAX_BLIND.get(_file_, 0)
                self.assertLessEqual(
                    len(_names_), _ceiling_,
                    f'{_file_} has {len(_names_)} test functions with no assertion, ceiling is '
                    f'{_ceiling_}: {_names_}. A test that asserts nothing passes whether the code '
                    'is right or wrong -- assert on what the render drew (tools/blind_tests.py '
                    'header gives the bar). If you fixed blind tests here, lower the ceiling.')

    def test_ceilings_are_not_stale(self) -> None:
        # A ceiling left above the real count lets a new blind test in unnoticed.
        _slack_ = {f: (c, len(self.blind.get(f, []))) for f, c in self.MAX_BLIND.items()
                   if len(self.blind.get(f, [])) < c}
        self.assertEqual(_slack_, {},
                         'these ceilings are above the real count -- lower them to lock the '
                         'progress in (delete a line that reaches zero) '
                         f'{{file: (ceiling, actual)}}: {_slack_}')


if __name__ == '__main__':
    unittest.main()
