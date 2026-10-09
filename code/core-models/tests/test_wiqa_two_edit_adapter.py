import copy
import json
import tempfile
import unittest
from pathlib import Path

from core_bert.wiqa_two_edit_adapter import load_wiqa_two_edit_bundle


ROOT = Path(__file__).parents[1]
QUEUE = ROOT / "data/two_edit/wiqa/queue.jsonl"
TRUTH = ROOT / "data/two_edit/wiqa/truth.jsonl"


class WiqaTwoEditAdapterTests(unittest.TestCase):
    def test_reexecutes_all_corrected_authoritative_truth(self):
        bundle = load_wiqa_two_edit_bundle(QUEUE, TRUTH, ROOT / "data", split="validation")
        self.assertEqual(len(bundle.examples), 70)
        self.assertEqual(len(bundle.records), 140)

    def test_blocks_test_before_loading(self):
        with self.assertRaisesRegex(ValueError, "test is blocked"):
            load_wiqa_two_edit_bundle(Path("missing"), Path("missing"), Path("missing"), split="test")

    def test_tampered_truth_fails_hash_binding(self):
        rows = [json.loads(line) for line in TRUTH.read_text().splitlines() if line.strip()]
        tampered = copy.deepcopy(rows)
        target = next(row for row in tampered if row["source"] == "wiqa")
        target["gold_final_outputs"]["A"] = (
            "more" if target["gold_final_outputs"]["A"] != "more" else "less"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "truth.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in tampered))
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                load_wiqa_two_edit_bundle(QUEUE, path, ROOT / "data", split="validation")


if __name__ == "__main__":
    unittest.main()
