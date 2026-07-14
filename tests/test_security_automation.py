#
# test_security_automation.py
#
# Security/quality automation in CI (bandit, pip-audit,
# Dependabot, a minimal ruff pass). The tools themselves run in the ci.yml
# workflow, not here (this repo's convention: mypy is likewise CI-only, not
# invoked from the pytest suite). These are config-consistency guards against
# someone silently dropping a step later -- read/parse the checked-in config
# files and assert the wiring is still present, mirroring test_changelog.py's
# approach to release-hygiene config.
#
import os
import unittest

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

_REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CI_WORKFLOW = os.path.join(_REPO_ROOT, '.github', 'workflows', 'ci.yml')
_DEPENDABOT  = os.path.join(_REPO_ROOT, '.github', 'dependabot.yml')
_PYPROJECT   = os.path.join(_REPO_ROOT, 'pyproject.toml')


def _skip_if_missing(path):
    # These tests document release-tooling hygiene from the repo source tree;
    # a wheel/sdist install carries none of .github/ or dev pyproject config.
    if not os.path.exists(path):
        raise unittest.SkipTest(f'{os.path.relpath(path, _REPO_ROOT)} not present (not a source checkout)')


class TestCIWorkflowSecurityScanning(unittest.TestCase):

    def setUp(self):
        _skip_if_missing(_CI_WORKFLOW)
        with open(_CI_WORKFLOW, encoding='utf-8') as f:
            self.text = f.read()

    def test_bandit_step_present(self):
        self.assertIn('bandit', self.text)

    def test_pip_audit_step_present(self):
        self.assertIn('pip-audit', self.text)

    def test_ruff_step_present(self):
        self.assertRegex(self.text, r'ruff check\b')

    @unittest.skipUnless(YAML_AVAILABLE, 'pyyaml not installed')
    def test_workflow_is_well_formed_yaml_with_jobs(self):
        with open(_CI_WORKFLOW, encoding='utf-8') as f:
            doc = yaml.safe_load(f)
        self.assertIn('jobs', doc)
        self.assertGreaterEqual(len(doc['jobs']), 1)


class TestDependabotConfig(unittest.TestCase):

    def setUp(self):
        _skip_if_missing(_DEPENDABOT)

    @unittest.skipUnless(YAML_AVAILABLE, 'pyyaml not installed')
    def test_covers_pip_and_github_actions_weekly(self):
        with open(_DEPENDABOT, encoding='utf-8') as f:
            doc = yaml.safe_load(f)
        ecosystems = {u['package-ecosystem']: u for u in doc['updates']}
        self.assertIn('pip', ecosystems)
        self.assertIn('github-actions', ecosystems)
        for _eco_, _entry_ in ecosystems.items():
            self.assertEqual(_entry_['schedule']['interval'], 'weekly', _eco_)


class TestRuffConfig(unittest.TestCase):
    # Minimal-ruleset ruff config (E9, F) with __init__.py's intentional
    # public re-exports carved out via per-file-ignores -- see the
    # [tool.ruff] comment block in pyproject.toml for the reasoning.

    def setUp(self):
        _skip_if_missing(_PYPROJECT)
        import tomllib
        with open(_PYPROJECT, 'rb') as f:
            self.pyproject = tomllib.load(f)

    def test_select_is_minimal_e9_f(self):
        _select_ = self.pyproject['tool']['ruff']['lint']['select']
        self.assertEqual(set(_select_), {'E9', 'F'})

    def test_init_py_f401_ignored(self):
        _ignores_ = self.pyproject['tool']['ruff']['lint']['per-file-ignores']
        self.assertIn('F401', _ignores_.get('polars2svg/__init__.py', []))


if __name__ == '__main__':
    unittest.main()
