import ast
import glob
import os
import unittest


class TestNoBareFunctions(unittest.TestCase):
    def test_no_bare_test_functions(self):
        # The glob is deliberately NOT recursive.  tests/interaction/ is a browser
        # suite whose tests take pytest fixtures (`page`, and the harness fixtures
        # built on it), and a unittest.TestCase method cannot receive one -- so those
        # files are plain pytest functions by necessity, not by drift.  Making this
        # recursive would flag every one of them; exclude that directory explicitly if
        # you ever do.
        test_dir = os.path.dirname(os.path.abspath(__file__))
        test_files = sorted(glob.glob(os.path.join(test_dir, 'test_*.py')))
        violations = []
        for path in test_files:
            if os.path.basename(path) == 'test_meta.py':
                continue
            with open(path) as f:
                source = f.read()
            tree = ast.parse(source, filename=path)
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith('test_'):
                    violations.append(f"{os.path.basename(path)}: bare function '{node.name}'")
        self.assertEqual(
            violations, [],
            "Bare test_* functions found at module level (must be methods of a unittest.TestCase subclass):\n"
            + "\n".join(violations),
        )




class TestNoDuplicateTestBasenames(unittest.TestCase):
    """No two test modules anywhere under tests/ may share a file name.

    This is not style. pytest imports a test module under a name derived from its
    basename, and with no ``__init__.py`` in the tree two files called ``test_x.py``
    resolve to the same module name -- so collection aborts with an import-file
    mismatch and **the entire suite stops running**, not just the pair. It reads as one
    ERROR line and it is easy to scroll past.

    That is not hypothetical: ``tests/test_tooltip.py`` and
    ``tests/interaction/test_tooltip.py`` disabled ``pytest tests/`` -- the command
    CONTRIBUTING and CLAUDE.md both document -- until the browser half was renamed to
    ``test_tooltip_browser.py``.

    Renaming is the fix rather than the workaround. The two obvious alternatives were
    measured and each breaks the *same* 12 unrelated files, because they reach their
    helpers (``svg_test_utils``, the dataframe builders) through the sys.path entry that
    pytest's default rootdir-relative import provides:

    * ``--import-mode=importlib`` -- no sys.path entry at all;
    * ``__init__.py`` in ``tests/`` and ``tests/interaction/`` -- makes them packages,
      which moves the inserted path up to the repo root.

    conftest.py is exempt: pytest gives per-directory conftests their own handling, and
    one per directory is the intended layout.
    """

    def test_every_test_module_has_a_unique_basename(self):
        _root_ = os.path.dirname(os.path.abspath(__file__))
        _seen_ = {}
        for _path_ in sorted(glob.glob(os.path.join(_root_, '**', 'test_*.py'), recursive=True)):
            _seen_.setdefault(os.path.basename(_path_), []).append(
                os.path.relpath(_path_, _root_))
        _dupes_ = {_n_: _p_ for _n_, _p_ in _seen_.items() if len(_p_) > 1}
        self.assertEqual(
            _dupes_, {},
            'test modules share a basename, which makes pytest abort collection for the '
            'WHOLE suite with an import-file mismatch: '
            + '; '.join(f'{_n_} at {_p_}' for _n_, _p_ in sorted(_dupes_.items()))
            + '. Rename one (see this test docstring for why renaming, and not an '
              'import-mode or __init__.py change, is the fix).')


if __name__ == '__main__':
    unittest.main()
