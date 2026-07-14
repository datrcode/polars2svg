import os
import re
import tomllib
import unittest

import polars2svg

_REPO_ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CHANGELOG     = os.path.join(_REPO_ROOT, 'CHANGELOG.md')
_PYPROJECT     = os.path.join(_REPO_ROOT, 'pyproject.toml')

# First "## [x.y.z]" heading, ignoring the "## [Unreleased]" section.
_VERSION_HEAD  = re.compile(r'^##\s*\[(\d+\.\d+\.\d+)\]', re.MULTILINE)


class TestChangelog(unittest.TestCase):

    def setUp(self):
        # These tests document release hygiene from the installed source tree.
        # When polars2svg is installed from a wheel (no CHANGELOG alongside),
        # there is nothing to check.
        if not os.path.exists(_CHANGELOG):
            self.skipTest('CHANGELOG.md not present (installed from wheel)')
        with open(_CHANGELOG, encoding='utf-8') as f:
            self.text = f.read()

    def test_changelog_exists_and_nonempty(self):
        self.assertTrue(self.text.strip(), 'CHANGELOG.md is empty')

    def test_keep_a_changelog_and_semver_referenced(self):
        # Format markers Keep-a-Changelog uses; cheap guard against a rewrite
        # that drops the convention.
        self.assertIn('Keep a Changelog', self.text)
        self.assertIn('Semantic Versioning', self.text)

    def test_has_unreleased_section(self):
        self.assertRegex(self.text, r'(?m)^##\s*\[Unreleased\]')

    def test_top_version_matches_package_version(self):
        m = _VERSION_HEAD.search(self.text)
        self.assertIsNotNone(m, 'no "## [x.y.z]" release heading found')
        self.assertEqual(
            m.group(1), polars2svg.__version__,
            'newest CHANGELOG version heading must match polars2svg.__version__ '
            '(bump the CHANGELOG when releasing)',
        )

    def test_top_version_matches_pyproject_version(self):
        if not os.path.exists(_PYPROJECT):
            self.skipTest('pyproject.toml not present')
        with open(_PYPROJECT, 'rb') as f:
            pyproject_version = tomllib.load(f)['project']['version']
        m = _VERSION_HEAD.search(self.text)
        self.assertIsNotNone(m, 'no "## [x.y.z]" release heading found')
        self.assertEqual(m.group(1), pyproject_version)


if __name__ == '__main__':
    unittest.main()
