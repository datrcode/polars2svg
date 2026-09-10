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


if __name__ == '__main__':
    unittest.main()
