"""Log filtering must allow Numba's first import during a training batch."""
import ast
import builtins
import contextlib
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]


def filter_source():
    # Isolate the actual entrypoint helper from optional rendering dependencies.
    tree = ast.parse((ROOT / "globalsplat/main.py").read_text(encoding="utf-8"))
    helper = next(node for node in tree.body
                  if isinstance(node, ast.FunctionDef) and node.name == "_quiet_upstream_logs")
    return ast.unparse(helper)


class QuietUpstreamLogsTests(unittest.TestCase):
    def test_filter_preserves_other_messages_and_print_options(self):
        namespace = {"DictConfig": dict}
        exec(filter_source(), namespace)
        original = builtins.print
        output = io.StringIO()
        try:
            namespace["_quiet_upstream_logs"]({"quiet_skipped_examples": False})
            self.assertIs(builtins.print, original)
            namespace["_quiet_upstream_logs"]({})
            with contextlib.redirect_stdout(output):
                print("Skipped bad example scene=bad")
                print("step", 10000, sep="=", end="!", flush=True)
                print()
                print(123)
            self.assertEqual(output.getvalue(), "step=10000!\n123\n")
        finally:
            builtins.print = original

    @unittest.skipUnless(importlib.util.find_spec("numba"), "optional numba unavailable")
    def test_first_numba_import_after_filter_and_exact_knn(self):
        # A fresh process is essential: an earlier Numba import hides this bug.
        script = "import sys\nDictConfig = dict\n" + filter_source() + """
assert 'numba' not in sys.modules
_quiet_upstream_logs({})
print('Skipped bad example hidden')
# Import here before optional codec dependencies can initialize Numba for us.
from numba import njit
@njit
def compiled_print():
    print(42)
    return 7
assert compiled_print() == 7
import torch
from globalsplat.compression.token_order import greedy_neighbor_order
values = torch.tensor([[[0.], [2.], [1.], [4.]]])
positions = torch.zeros(1, 4, 3)
expected = greedy_neighbor_order(values, positions).permutation
actual = greedy_neighbor_order(values, positions, backend='knn', candidates=1,
                               distance_device='input').permutation
assert torch.equal(actual, expected)
print('COLD_NUMBA_KNN_OK')
"""
        result = subprocess.run([sys.executable, "-c", script], cwd=ROOT,
                                capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("Skipped bad example", result.stdout)
        self.assertIn("COLD_NUMBA_KNN_OK", result.stdout)


if __name__ == "__main__":
    unittest.main()
