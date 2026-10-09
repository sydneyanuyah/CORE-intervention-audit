import json
import tempfile
import unittest
from pathlib import Path

from core_bert.com2_two_edit_adapter import load_com2_two_edit_bundle


ROOT = Path(__file__).parents[1]
DATA = ROOT / "data"
AVAILABLE = (DATA / "records" / "com2_intervention.jsonl").exists()


@unittest.skipUnless(AVAILABLE, "accepted benchmark records are installed on small-model cluster")
class Com2TwoEditAdapterTests(unittest.TestCase):
    def test_loads_exact_frozen_validation_truth(self):
        bundle = load_com2_two_edit_bundle(
            DATA / "two_edit/manual/annotation_queue.jsonl",
            DATA / "two_edit/manual/authoritative_manual_truth.jsonl",
            DATA / "two_edit/manual/authoritative_manual_truth.provenance.json",
            DATA, split="validation",
        )
        self.assertEqual(len(bundle.examples), 70)
        self.assertEqual(len(bundle.records), 140)
        self.assertTrue(all(example.source == "com2" for example in bundle.examples))
        self.assertTrue(all(any(example.change_mask) for example in bundle.examples))

    def test_rejects_test_before_loading(self):
        with self.assertRaisesRegex(ValueError, "validation only"):
            load_com2_two_edit_bundle(Path("missing"), Path("missing"), Path("missing"), DATA, split="test")

    def test_rejects_tampered_truth_hash(self):
        source = DATA / "two_edit/manual/authoritative_manual_truth.jsonl"
        with tempfile.TemporaryDirectory() as directory:
            tampered = Path(directory) / "truth.jsonl"
            rows = source.read_text(encoding="utf-8").splitlines()
            row = json.loads(rows[0])
            row["status"] = "needs_adjudication"
            rows[0] = json.dumps(row)
            tampered.write_text("\n".join(rows) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                load_com2_two_edit_bundle(
                    DATA / "two_edit/manual/annotation_queue.jsonl", tampered,
                    DATA / "two_edit/manual/authoritative_manual_truth.provenance.json",
                    DATA, split="validation",
                )


if __name__ == "__main__":
    unittest.main()
