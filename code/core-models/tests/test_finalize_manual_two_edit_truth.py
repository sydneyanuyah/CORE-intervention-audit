import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
PATH = ROOT / "scripts" / "finalize_manual_two_edit_truth.py"
SPEC = importlib.util.spec_from_file_location("finalize_manual_two_edit_truth", PATH)
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


class FinalizeManualTruthTests(unittest.TestCase):
    def test_primary_status_requires_adjudication(self):
        row = {field: 1 for field in module.STRUCTURAL}
        row.update({"status": "accepted", "query_answer_after_both_edits": "more"})
        other = dict(row); other["status"] = "needs_adjudication"
        self.assertEqual(module.primary_differences(row, other, "com2"), ["status"])

    def test_variable_names(self):
        self.assertEqual(module.variable_names({"chain": ["a", "b"]}, "com2"), ["event_00", "event_01"])


if __name__ == "__main__":
    unittest.main()
