import ast
import glob
import os
import unittest


class TestNoBareFunctions(unittest.TestCase):
    def test_no_bare_test_functions(self):
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
