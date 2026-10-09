import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from core_bert.two_edit import (  # noqa: E402
    build_ordered_two_edit_examples,
    load_ordered_two_edit_manifest,
    ordered_pair_id,
    two_edit_balanced_metrics,
)


def single(record_id, target, value, source="xor"):
    return {
        "id": record_id,
        "source": source,
        "factual": {"passage": "A fixed world", "question": "What is the state?"},
        "intervention": {"target": target, "value": value, "text": f"Set {target}={value}"},
    }


def truth(first, second, **overrides):
    value = {
        "first_record_id": first,
        "second_record_id": second,
        "factual_outputs": [0, 1, 0, 1],
        "gold_outputs": [1, 1, 0, 0],
        "change_mask": [True, False, False, True],
        "preservation_mask": [False, True, True, False],
        "target_mask": [True, False, False, True],
    }
    value.update(overrides)
    return value


class TwoEditProtocolTests(unittest.TestCase):
    def setUp(self):
        self.records = {
            "x-up": single("x-up", "X", 1),
            "y-down": single("y-down", "Y", 0),
            "x-down": single("x-down", "X", 0),
        }
        self.groups = {key: ("graph-7", "world-3") for key in self.records}
        self.seen = set(self.records)

    def build(self, rows):
        return build_ordered_two_edit_examples(
            self.records, self.groups, rows, seen_separately=self.seen
        )

    def test_ordered_ids_are_deterministic_and_order_sensitive(self):
        forward = self.build([truth("x-up", "y-down")])[0]
        reverse = self.build([truth("y-down", "x-up")])[0]
        self.assertNotEqual(forward.pair_id, reverse.pair_id)
        self.assertEqual(forward.pair_id, self.build([truth("x-up", "y-down")])[0].pair_id)
        expected = ordered_pair_id("xor", "graph-7", "world-3", "x-up", "y-down")
        self.assertEqual(forward.pair_id, expected)

    def test_shared_target_last_write_pair_is_allowed(self):
        example = self.build([truth("x-up", "x-down")])[0]
        self.assertEqual(example.first_intervention["target"], example.second_intervention["target"])

    def test_components_must_be_seen_separately_and_share_world(self):
        with self.assertRaisesRegex(ValueError, "seen separately"):
            build_ordered_two_edit_examples(
                self.records, self.groups, [truth("x-up", "y-down")], seen_separately={"x-up"}
            )
        mismatched = dict(self.groups)
        mismatched["y-down"] = ("graph-7", "world-4")
        with self.assertRaisesRegex(ValueError, "share graph and world"):
            build_ordered_two_edit_examples(
                self.records, mismatched, [truth("x-up", "y-down")], seen_separately=self.seen
            )

    def test_truth_masks_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "change_mask must equal"):
            self.build([truth("x-up", "y-down", change_mask=[False, False, False, True])])
        with self.assertRaisesRegex(ValueError, "preservation_mask must equal"):
            self.build(
                [truth("x-up", "y-down", preservation_mask=[False, True, False, False])]
            )
        with self.assertRaisesRegex(ValueError, "must each be non-empty"):
            self.build([truth("x-up", "y-down", preservation_mask=[False] * 4)])
        canonical = ordered_pair_id("xor", "graph-7", "world-3", "x-up", "y-down")
        with self.assertRaisesRegex(ValueError, "canonical ordered identity"):
            self.build([truth("x-up", "y-down", pair_id=canonical + "-wrong")])

    def test_balanced_metrics_and_graph_units(self):
        first = self.build([truth("x-up", "y-down")])[0]
        second_records = {key: dict(value) for key, value in self.records.items()}
        for value in second_records.values():
            value["factual"] = {"passage": "Another world", "question": "State?"}
        groups = {key: ("graph-8", "world-4") for key in second_records}
        second = build_ordered_two_edit_examples(
            second_records, groups, [truth("x-up", "y-down")], seen_separately=self.seen
        )[0]
        metrics = two_edit_balanced_metrics(
            [first, second],
            {
                first.pair_id: [1, 0, 0, 1],
                second.pair_id: [1, 1, 0, 0],
            },
        )
        self.assertEqual(metrics["two_edit_change_accuracy"], 0.75)
        self.assertEqual(metrics["two_edit_preservation"], 0.75)
        self.assertEqual(metrics["two_edit_balanced"], 0.75)
        self.assertEqual(metrics["graph_count"], 2)
        self.assertFalse(metrics["a1_evidence"])

    def test_prediction_ids_and_jsonl_manifest_are_exact(self):
        row = truth("x-up", "y-down")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pairs.jsonl"
            path.write_text(json.dumps(row) + "\n")
            examples = load_ordered_two_edit_manifest(
                path, self.records, self.groups, seen_separately=self.seen
            )
        with self.assertRaisesRegex(ValueError, "prediction IDs must match exactly"):
            two_edit_balanced_metrics(examples, {})


if __name__ == "__main__":
    unittest.main()
