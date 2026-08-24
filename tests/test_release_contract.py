from __future__ import annotations

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ReleaseContractTest(unittest.TestCase):
    def test_prompt_states_current_timeout_contract(self) -> None:
        prompt = " ".join((ROOT / "public" / "prompt.txt").read_text().split())
        self.assertIn("Do not compute gradients", prompt)
        self.assertIn("completed full-suite evaluation", prompt)
        self.assertIn("no active submission command is required", prompt)
        self.assertNotIn("bin/submit", prompt)

    def test_starters_do_not_import_gradient_frameworks(self) -> None:
        forbidden = {"torch", "jax", "tensorflow"}
        for path in (ROOT / "starter").glob("*.py"):
            tree = ast.parse(path.read_text(), filename=str(path))
            imported = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module.split(".")[0])
            self.assertTrue(imported.isdisjoint(forbidden), path.name)

    def test_starters_use_the_public_candidate_format(self) -> None:
        for name in ("randopt.py", "es.py"):
            source = (ROOT / "starter" / name).read_text()
            self.assertIn("zerograd-noise-program-v1", source)
            self.assertNotIn("--submit", source)


if __name__ == "__main__":
    unittest.main()
