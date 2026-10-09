import copy
import json
import tempfile
import unittest
from pathlib import Path

from core_bert.cladder_two_edit_adapter import load_cladder_two_edit_bundle


ROOT = Path(__file__).parents[1]
ARTIFACT = ROOT / "data" / "two_edit" / "cladder" / "artifact.json"
MANIFEST = ROOT / "data" / "two_edit" / "cladder" / "manifest.jsonl"


class CladderTwoEditAdapterTests(unittest.TestCase):
    def test_reexecutes_complete_validation_truth(self):
        bundle = load_cladder_two_edit_bundle(ARTIFACT, MANIFEST, split="validation")
        self.assertEqual(len(bundle.examples), 615)
        self.assertTrue(bundle.records)

    def test_blocks_test_before_loading(self):
        with self.assertRaisesRegex(ValueError, "test is blocked"):
            load_cladder_two_edit_bundle(Path("missing"), Path("missing"), split="test")

    def test_tampered_pair_truth_fails_closed(self):
        rows = [json.loads(line) for line in MANIFEST.read_text().splitlines() if line.strip()]
        tampered = copy.deepcopy(rows)
        row = next(item for item in tampered if item["split"] == "validation")
        row["gold_outputs"][0] = 1 - row["gold_outputs"][0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.jsonl"
            path.write_text("".join(json.dumps(item) + "\n" for item in tampered))
            with self.assertRaisesRegex(ValueError, "truth disagrees"):
                load_cladder_two_edit_bundle(ARTIFACT, path, split="validation")


if __name__ == "__main__":
    unittest.main()
