import os
import re
import tomllib
import unittest

import polars2svg

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SECURITY  = os.path.join(_REPO_ROOT, 'SECURITY.md')
_PYPROJECT = os.path.join(_REPO_ROOT, 'pyproject.toml')

# The policy names the supported line three times: a "currently `M.m.x`" sentence and
# two rows of the Supported Versions table.  The row patterns are anchored on the table's
# shape -- a leading pipe, then the version cell, then the mark -- so a version mentioned
# anywhere else in the file can never be read as the policy.
_CURRENTLY   = re.compile(r'currently `(\d+\.\d+)\.x`')
_SUPPORTED   = re.compile(r'^\|\s*(\d+\.\d+)\.x\s*\|\s*✅\s*\|\s*$', re.MULTILINE)
_UNSUPPORTED = re.compile(r'^\|\s*<\s*(\d+\.\d+)\s*\|\s*❌\s*\|\s*$', re.MULTILINE)
_FIX         = 'update SECURITY.md (pypi_instructions.txt step 4, on a major or minor bump)'


def _majorMinor_(version: str) -> str:
    return '.'.join(version.split('.')[:2])


class TestSecurityPolicy(unittest.TestCase):
    """SECURITY.md ships in the sdist and states which versions are supported, and nothing
    in the release path read it: it said "currently `0.2.x`" through both the 0.3.0 and the
    0.3.1 cuts, telling users their own version was unsupported (PLANNING.md R13).  The
    direct analogue of test_citation.py: the policy's supported line must be the package's
    own major.minor."""

    def setUp(self):
        # Mirrors test_citation.py: release hygiene, read from the source tree.
        if not os.path.exists(_SECURITY):
            self.skipTest('SECURITY.md not present (installed from wheel)')
        with open(_SECURITY, encoding='utf-8') as f:
            self.text = f.read()
        self.line = _majorMinor_(polars2svg.__version__)

    def test_currently_names_the_package_line(self):
        self.assertEqual(_CURRENTLY.findall(self.text), [self.line],
                         f'SECURITY.md "currently `{self.line}.x`" -- {_FIX}')

    def test_the_one_supported_row_is_the_package_line(self):
        self.assertEqual(_SUPPORTED.findall(self.text), [self.line],
                         f'SECURITY.md table: exactly one supported row, "{self.line}.x" -- {_FIX}')

    def test_the_unsupported_row_is_everything_before_it(self):
        self.assertEqual(_UNSUPPORTED.findall(self.text), [self.line],
                         f'SECURITY.md table: the unsupported row reads "< {self.line}" -- {_FIX}')

    # Also against pyproject.toml, as test_citation.py does: a stale __version__ (a missed
    # reinstall after a bump) then fails here on its own, rather than looking like a stale
    # SECURITY.md.
    def test_the_package_line_matches_pyproject(self):
        if not os.path.exists(_PYPROJECT):
            self.skipTest('pyproject.toml not present')
        with open(_PYPROJECT, 'rb') as f:
            _declared_ = tomllib.load(f)['project']['version']
        self.assertEqual(self.line, _majorMinor_(_declared_),
                         'polars2svg.__version__ disagrees with pyproject.toml -- reinstall '
                         '(pypi_instructions.txt step 6)')


if __name__ == '__main__':
    unittest.main()
