import os
import re
import tomllib
import unittest

import polars2svg

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CITATION  = os.path.join(_REPO_ROOT, 'CITATION.cff')
_CHANGELOG = os.path.join(_REPO_ROOT, 'CHANGELOG.md')
_PYPROJECT = os.path.join(_REPO_ROOT, 'pyproject.toml')

# Top-level CFF keys only -- no leading whitespace, so the `version:` inside a
# nested `references:` entry can never be picked up by mistake.
_CFF_VERSION = re.compile(r'^version:\s*(\S+)\s*$', re.MULTILINE)
_CFF_DATE    = re.compile(r"^date-released:\s*'?(\d{4}-\d{2}-\d{2})'?\s*$", re.MULTILINE)

# First "## [x.y.z] - YYYY-MM-DD" heading, ignoring "## [Unreleased]". The
# separator is an em dash in every heading written so far; a plain hyphen is
# accepted too so a future entry typed the other way fails on its content
# rather than on its punctuation.
_CHANGELOG_RELEASE = re.compile(
    r'^##\s*\[(\d+\.\d+\.\d+)\]\s*[—-]\s*(\d{4}-\d{2}-\d{2})', re.MULTILINE)


class TestCitation(unittest.TestCase):
    """CITATION.cff carries a version and a release date of its own, and nothing
    else in the release path reads them -- pypi_instructions.txt names only
    pyproject.toml and CHANGELOG.md, so both keys go stale silently. These pin
    them to the two files that are already checked against each other."""

    def setUp(self):
        # Mirrors test_changelog.py: these document release hygiene from the
        # source tree. CITATION.cff ships in the sdist but not the wheel.
        if not os.path.exists(_CITATION):
            self.skipTest('CITATION.cff not present (installed from wheel)')
        with open(_CITATION, encoding='utf-8') as f:
            self.text = f.read()

    def _version(self):
        m = _CFF_VERSION.search(self.text)
        self.assertIsNotNone(m, 'no top-level "version:" key in CITATION.cff')
        return m.group(1)

    def _date_released(self):
        m = _CFF_DATE.search(self.text)
        self.assertIsNotNone(m, 'no top-level "date-released:" key in CITATION.cff')
        return m.group(1)

    def test_version_matches_package_version(self):
        self.assertEqual(
            self._version(), polars2svg.__version__,
            'CITATION.cff version must match polars2svg.__version__ '
            '(bump CITATION.cff when releasing)',
        )

    def test_version_matches_pyproject_version(self):
        if not os.path.exists(_PYPROJECT):
            self.skipTest('pyproject.toml not present')
        with open(_PYPROJECT, 'rb') as f:
            pyproject_version = tomllib.load(f)['project']['version']
        self.assertEqual(self._version(), pyproject_version)

    def test_date_released_matches_newest_changelog_heading(self):
        if not os.path.exists(_CHANGELOG):
            self.skipTest('CHANGELOG.md not present')
        with open(_CHANGELOG, encoding='utf-8') as f:
            m = _CHANGELOG_RELEASE.search(f.read())
        self.assertIsNotNone(m, 'no dated "## [x.y.z]" release heading found')
        changelog_version, changelog_date = m.group(1), m.group(2)
        # Guards the pairing: without this, a CITATION.cff left at the previous
        # release would match that release's date and look consistent.
        self.assertEqual(self._version(), changelog_version)
        self.assertEqual(
            self._date_released(), changelog_date,
            'CITATION.cff date-released must match the date on the newest '
            'CHANGELOG release heading',
        )


if __name__ == '__main__':
    unittest.main()
