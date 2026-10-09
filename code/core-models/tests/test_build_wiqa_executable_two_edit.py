import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
PATH = ROOT / "scripts/build_wiqa_executable_two_edit.py"
SPEC = importlib.util.spec_from_file_location("build_wiqa_executable_two_edit", PATH)
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


class BuildWiqaExecutableTwoEditTests(unittest.TestCase):
    def test_corrected_pairs_use_only_train_exposed_distinct_semantics(self):
        queue, truth = module.build(
            ROOT / "data/two_edit/manual/annotation_queue_v3_signed.jsonl", ROOT / "data"
        )
        self.assertEqual(len(queue), 70)
        self.assertEqual(len(truth), 70)
        self.assertTrue(all(row["second_edit"]["target"] not in {"A", "D"} for row in queue))
        self.assertTrue(all(row["first_edit"]["target"] != row["second_edit"]["target"] for row in queue))
        self.assertTrue(all(row["truth_provenance"]["test_evaluated"] is False for row in truth))

    def test_input_hash_is_frozen(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "queue.jsonl"
            path.write_text(json.dumps({"not": "the frozen queue"}) + "\n")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                module.build(path, ROOT / "data")


if __name__ == "__main__":
    unittest.main()
