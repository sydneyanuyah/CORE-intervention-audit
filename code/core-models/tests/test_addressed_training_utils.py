import sys
import json
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from core_bert.addressed_training_utils import (
    append_factual_question,
    explicit_target_slot,
    load_group_sidecars,
    sidecar_fingerprints,
)


class AddressedTrainingUtilsTests(unittest.TestCase):
    def test_question_is_appended_without_moving_passage_span(self):
        record = {
            "factual": {"passage": "The sprinkler is off.", "question": "Will the grass be wet?"},
            "intervention": {"target_span": [4, 13]},
        }
        prepared = append_factual_question(record)
        self.assertTrue(prepared["factual"]["passage"].startswith(record["factual"]["passage"]))
        start, end = record["intervention"]["target_span"]
        self.assertEqual(prepared["factual"]["passage"][start:end], "sprinkler")
        self.assertIn("Question: Will the grass be wet?", prepared["factual"]["passage"])
        self.assertEqual(record["factual"]["passage"], "The sprinkler is off.")

    def test_pointer_slot_is_never_inferred(self):
        record = {
            "graph": {"nodes": ["A", "B"]},
            "intervention": {"target": "B"},
        }
        self.assertEqual(explicit_target_slot(record, 16), -1)
        record["intervention"]["target_slot"] = 7
        self.assertEqual(explicit_target_slot(record, 16), 7)
        record["intervention"]["target_slot"] = 16
        self.assertEqual(explicit_target_slot(record, 16), -1)

    def test_json_and_jsonl_group_sidecars(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "a.json").write_text(json.dumps({
                "record-a": {"graph_group_id": "g-a", "world_group_id": "w-a"}
            }))
            (root / "b.jsonl").write_text(json.dumps({
                "record_id": "record-b", "graph_id": "g-b", "world_id": "w-b"
            }) + "\n")
            groups = load_group_sidecars(root)
            self.assertEqual(groups["record-a"], ("g-a", "w-a"))
            self.assertEqual(groups["record-b"], ("g-b", "w-b"))
            self.assertEqual(set(sidecar_fingerprints(root)), {"a.json", "b.jsonl"})


if __name__ == "__main__":
    unittest.main()
