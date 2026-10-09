import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
PATH = ROOT / "scripts" / "prepare_manual_adjudication.py"
SPEC = importlib.util.spec_from_file_location("prepare_manual_adjudication", PATH)
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


class PrepareManualAdjudicationTests(unittest.TestCase):
    def test_primary_difference_and_blinding(self):
        first = {"source": "wiqa", "gold_final_outputs": {"X": "more"}, "target_variables": ["X"], "actually_changed_variables": ["X"], "preserved_variables": [], "first_edit_valid": True, "second_edit_valid": True, "query_answer_after_both_edits": "more", "status": "accepted"}
        second = {**first, "query_answer_after_both_edits": "less", "annotator_id": "secret"}
        self.assertEqual(module.primary_differences(first, second), ["query_answer_after_both_edits"])
        self.assertNotIn("annotator_id", module.proposal(second))

    def test_com2_normalization_changes_only_query(self):
        queue = {"pair_id": "p", "source": "com2", "chain": ["a", "b"], "split": "validation"}
        annotation = {"pair_id": "p", "source": "com2", "status": "accepted", "gold_final_outputs": {"event_00": "a", "event_01": "b"}, "query_answer_after_both_edits": "explanation"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); q = root / "q.jsonl"; a = root / "a.jsonl"
            out = root / "out.jsonl"; provenance = root / "provenance.json"
            q.write_text(json.dumps(queue) + "\n"); a.write_text(json.dumps(annotation) + "\n")
            result = module.normalize_com2(q, a, out, provenance)
            normalized = json.loads(out.read_text())
            self.assertEqual(normalized["query_answer_after_both_edits"], "b")
            self.assertEqual(result["changed_rows"], 1)
            self.assertFalse(result["other_fields_changed"])


if __name__ == "__main__":
    unittest.main()
