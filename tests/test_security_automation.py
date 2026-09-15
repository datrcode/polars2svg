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
    def test_covers_uv_and_github_actions_weekly(self):
        with open(_DEPENDABOT, encoding='utf-8') as f:
            doc = yaml.safe_load(f)
        ecosystems = {u['package-ecosystem']: u for u in doc['updates']}
        # `uv`, not `pip`, and the distinction is the point rather than a
        # spelling: pip watches only what pyproject.toml declares, so every
        # transitive pin in uv.lock goes unwatched.  That is how three tornado
        # advisories sat in the lock with a weekly Dependabot reporting nothing
        # (tornado arrives via bokeh via panel, so pyproject.toml never names
        # it).  Asserting the ecosystem by name is what stops a revert to pip
        # from looking like a working config.
        self.assertIn('uv', ecosystems)
        self.assertNotIn('pip', ecosystems)
        self.assertIn('github-actions', ecosystems)
        for _eco_, _entry_ in ecosystems.items():
            self.assertEqual(_entry_['schedule']['interval'], 'weekly', _eco_)


class TestRuffConfig(unittest.TestCase):
    # ruff config: E9 and F are the floor -- see the [tool.ruff] comment block in
    # pyproject.toml for what else is selected and what is deliberately left out.
    # __init__.py's public re-exports used to be carved out via per-file-ignores;
    # they are explicit `Y as Y` aliases now, which ruff reads as re-exports and
    # does not flag, so there are no per-file-ignores at all.  The last test here
    # guards the aliases, since removing them would silently re-break the public
    # API for type checkers.

    def setUp(self):
        _skip_if_missing(_PYPROJECT)
        import tomllib
        with open(_PYPROJECT, 'rb') as f:
            self.pyproject = tomllib.load(f)

    def test_select_includes_e9_f(self):
        # A floor, not an equality: the set grew in 2026-09-14's lint sweep and is
        # expected to grow again.  What must never happen is E9 or F dropping out
        # -- syntax errors and pyflakes are the reason the gate exists at all.
        _select_ = self.pyproject['tool']['ruff']['lint']['select']
        self.assertLessEqual({'E9', 'F'}, set(_select_))

    def test_init_reexports_are_explicit(self):
        # polars2svg ships py.typed, which makes a plain `from .x import Y` in
        # __init__.py PRIVATE under the typing spec: pyright/Pylance reject
        # `from polars2svg import Polars2SVG` with reportPrivateImportUsage and
        # mypy --strict with attr-defined.  Every relative import must therefore
        # use the redundant `Y as Y` form.  [tool.mypy] no_implicit_reexport is
        # the other half of this guard, at the type level.  Audit 20260915 H1.
        import ast
        _init_ = os.path.join(_REPO_ROOT, 'polars2svg', '__init__.py')
        _skip_if_missing(_init_)
        with open(_init_, encoding='utf-8') as f:
            _tree_ = ast.parse(f.read())
        _bad_ = [f'{_a_.name} (line {_n_.lineno})'
                 for _n_ in ast.walk(_tree_)
                 if isinstance(_n_, ast.ImportFrom) and _n_.level > 0
                 for _a_ in _n_.names if _a_.asname != _a_.name]
        self.assertEqual(_bad_, [], 'implicit re-exports in polars2svg/__init__.py: '
                                    + ', '.join(_bad_))


if __name__ == '__main__':
    unittest.main()
