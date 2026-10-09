import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from convert_ccrgb import graph_from_world, normalize_item, world_splits  # noqa: E402
from schema import validate_record  # noqa: E402


WORLD = {
    "context_id": 0,
    "sample_id": 0,
    "dag_nodes": ["X", "Y"],
    "dag_adjacency_matrix": [[0, 1], [0, 0]],
    "causal_context": "X causes Y.",
    "sample_context": "X is absent.",
    "factual_queries": [{
        "effect": "Y", "prompt": "Is Y present?", "true_endogenous": {"X": 0, "Y": 0},
        "true_exogenous": {"UX": 0, "UY": 0}, "true_response": 0,
    }],
}
PAIR = {
    "cause": "X",
    "effect": "Y",
    "cause_true": {
        "prompt": "Set X true; is Y present?", "true_endogenous": {"X": 1, "Y": 1},
        "true_exogenous": {"UX": 0, "UY": 0}, "true_response": 1,
    },
    "cause_false": {
        "prompt": "Set X false; is Y present?", "true_endogenous": {"X": 0, "Y": 0},
        "true_exogenous": {"UX": 0, "UY": 0}, "true_response": 0,
    },
}


class CCRGBConverterTests(unittest.TestCase):
    def test_adjacency_conversion(self):
        self.assertEqual(graph_from_world(WORLD), {"nodes": ["X", "Y"], "edges": [["X", "Y"]]})

    def test_changed_intervention_validates(self):
        record = normalize_item(WORLD, PAIR, 0, 1)
        self.assertEqual(validate_record(record), [])
        self.assertEqual(record["intervened"]["answer"], "yes")
        intervention = record["intervention"]
        start, end = intervention["target_span"]
        self.assertEqual(record["factual"]["passage"][start:end], intervention["target_text"])
        self.assertEqual(intervention["value_token"], "1")

    def test_unchanged_intervention_is_screened(self):
        record = normalize_item(WORLD, PAIR, 0, 0)
        self.assertIn(
            "record has no may-change probe with an observed change",
            validate_record(record),
        )

    def test_missing_structural_target_surface_is_rejected(self):
        world = dict(WORLD, causal_context="A causes B.", sample_context="A is absent.")
        with self.assertRaisesRegex(ValueError, "absent from factual passage"):
            normalize_item(world, PAIR, 0, 1)

    def test_world_split_counts(self):
        worlds = [{"context_id": index} for index in range(100)]
        splits = world_splits(worlds)
        self.assertEqual(list(splits.values()).count("train"), 80)
        self.assertEqual(list(splits.values()).count("validation"), 10)
        self.assertEqual(list(splits.values()).count("test"), 10)


if __name__ == "__main__":
    unittest.main()
