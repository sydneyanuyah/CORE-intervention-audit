import copy
import json
import tempfile
import unittest
from pathlib import Path

from core_bert.ccrgb_two_edit_adapter import load_ccrgb_two_edit_bundle


ROOT = Path(__file__).parents[1]
ARTIFACT = ROOT / "data/two_edit/ccrgb/artifact.json"
MANIFEST = ROOT / "data/two_edit/ccrgb/manifest.jsonl"


class CcrgbTwoEditAdapterTests(unittest.TestCase):
    def test_reexecutes_complete_validation_truth(self):
        bundle = load_ccrgb_two_edit_bundle(ARTIFACT, MANIFEST, split="validation")
        self.assertEqual(len(bundle.examples), 2974)
        self.assertEqual(len({example.graph_group_id for example in bundle.examples}), 600)
        self.assertTrue(bundle.records)

    def test_blocks_test_before_loading(self):
        with self.assertRaisesRegex(ValueError, "test is blocked"):
            load_ccrgb_two_edit_bundle(Path("missing"), Path("missing"), split="test")

    def test_tampered_manifest_fails_hash_binding(self):
        rows = [json.loads(line) for line in MANIFEST.read_text().splitlines() if line.strip()]
        tampered = copy.deepcopy(rows)
        tampered[0]["gold_outputs"][0] = 1 - tampered[0]["gold_outputs"][0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in tampered))
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                load_ccrgb_two_edit_bundle(ARTIFACT, path, split="validation")


if __name__ == "__main__":
    unittest.main()
