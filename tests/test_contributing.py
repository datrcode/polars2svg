import os
import unittest

_REPO_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONTRIBUTING = os.path.join(_REPO_ROOT, 'CONTRIBUTING.md')
_PYPROJECT    = os.path.join(_REPO_ROOT, 'pyproject.toml')


class TestContributing(unittest.TestCase):

    def setUp(self):
        # Installed-from-wheel trees don't carry CONTRIBUTING.md; nothing to check.
        if not os.path.exists(_CONTRIBUTING):
            self.skipTest('CONTRIBUTING.md not present (installed from wheel)')
        with open(_CONTRIBUTING, encoding='utf-8') as f:
            self.text = f.read()

    def test_contributing_exists_and_nonempty(self):
        self.assertTrue(self.text.strip(), 'CONTRIBUTING.md is empty')

    def test_documents_dev_setup(self):
        self.assertIn('uv pip install -e .', self.text)
        self.assertIn('--group dev', self.text)

    def test_documents_golden_image_workflow(self):
        self.assertIn('UPDATE_GOLDEN=1', self.text)
        self.assertIn('golden_images.ipynb', self.text)

    def test_documents_perf_baseline_workflow(self):
        self.assertIn('UPDATE_PERF_BASELINE=1', self.text)
        self.assertIn('_make_workloads()', self.text)

    def test_documents_color_modes_notebook(self):
        self.assertIn('color_modes.ipynb', self.text)

    def test_documents_test_file_footer_convention(self):
        self.assertIn("if __name__ == '__main__':", self.text)
        self.assertIn('unittest.main()', self.text)

    def test_documents_ci_tools(self):
        for tool in ('mypy', 'bandit', 'pip-audit', 'ruff'):
            self.assertIn(tool, self.text)

    def test_documents_ruff_rule_set(self):
        # CONTRIBUTING called the rule set "the minimal E9,F ruleset" for a day
        # short of forever after PLANNING.md Q3 widened it, and the only test
        # looking at this checked that the four tool NAMES appeared -- which it
        # still did (20260915_fable_code_audit.md H5).  Every selected rule must
        # be named, in a code span, so widening `select` without documenting it
        # fails here.
        if not os.path.exists(_PYPROJECT):
            self.skipTest('pyproject.toml not present (installed from wheel)')
        import tomllib
        with open(_PYPROJECT, 'rb') as f:
            _select_ = tomllib.load(f)['tool']['ruff']['lint']['select']
        # assertTrue, not assertIn: assertIn's failure message embeds the whole
        # haystack, and CONTRIBUTING.md is ~20 KB -- unreadable in a test report.
        _undocumented_ = [_c_ for _c_ in _select_ if f'`{_c_}`' not in self.text]
        self.assertTrue(not _undocumented_,
                        'ruff rules selected in pyproject.toml but not documented in '
                        f'CONTRIBUTING.md: {", ".join(_undocumented_)}')


if __name__ == '__main__':
    unittest.main()
