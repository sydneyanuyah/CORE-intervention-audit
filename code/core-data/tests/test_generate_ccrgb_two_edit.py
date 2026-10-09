import json
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from generate_ccrgb_two_edit import generate, solve  # noqa: E402


ROOT = Path(__file__).parents[1]
RAW = ROOT / "data/raw/ccrgb/feasibility_20_worlds.jsonl"
RECORDS = ROOT / "data/records/ccrgb.jsonl"


class GenerateCcrgbTwoEditTests(unittest.TestCase):
    def test_validation_truth_is_executable_and_test_is_absent(self):
        artifact, manifest = generate(
            RAW,
            RECORDS,
            expected_raw_sha256=hashlib.sha256(RAW.read_bytes()).hexdigest(),
            expected_worlds=20,
        )
        self.assertEqual(len(manifest), 12)
        self.assertTrue(all(row["split"] == "validation" for row in manifest))
        self.assertFalse(artifact["test_evaluated"])
        self.assertNotIn("test", {world["split"] for world in artifact["worlds"]})

    def test_solver_applies_last_clamp_without_mutating_descriptor(self):
        world = json.loads(RAW.read_text().splitlines()[0])
        before = json.dumps(world, sort_keys=True)
        state = solve(world, {world["dag_nodes"][0]: 0, world["dag_nodes"][1]: 1})
        self.assertEqual(state[world["dag_nodes"][0]], 0)
        self.assertEqual(state[world["dag_nodes"][1]], 1)
        self.assertEqual(json.dumps(world, sort_keys=True), before)


if __name__ == "__main__":
    unittest.main()
