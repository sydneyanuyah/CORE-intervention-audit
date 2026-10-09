import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from convert_com2 import (  # noqa: E402
    common_prefix_length,
    event_occurrence,
    group_split,
    marked_event_passage,
    normalize_item,
)
from schema import validate_record  # noqa: E402


ITEM = {
    "question": "If rain stops, what happens to the grass?",
    "options": "A) wet B) dry",
    "thinking_process": "fixture",
    "answer": "B) dry",
    "chains": [
        ["clouds gather", "rain begins", "grass becomes wet", "clouds gather"],
        ["clouds gather", "rain stops", "grass becomes dry", "clouds gather"],
    ],
    "scenario": "Clouds gathered and rain began.",
    "type": "intervention",
    "model": "fixture",
}


class Com2ConverterTests(unittest.TestCase):
    def test_common_prefix(self):
        self.assertEqual(common_prefix_length(ITEM["chains"][0], ITEM["chains"][1]), 1)

    def test_occurrence_labels_disambiguate_repeated_text(self):
        self.assertNotEqual(event_occurrence(0, "repeat"), event_occurrence(3, "repeat"))

    def test_normalized_record_validates(self):
        record = normalize_item(copy.deepcopy(ITEM), 17)
        self.assertEqual(validate_record(record), [])
        self.assertEqual(record["intervention"]["target"], "event_01: rain begins")
        self.assertEqual(record["descendants"][-1], "event_03: clouds gather")
        self.assertNotIn(record["intervention"]["target"], record["descendants"])
        intervention = record["intervention"]
        target_start, target_end = intervention["target_span"]
        replacement_start, replacement_end = intervention["replacement_span"]
        self.assertEqual(record["factual"]["passage"][target_start:target_end], "rain begins")
        self.assertEqual(record["factual"]["passage"][replacement_start:replacement_end], "rain stops")
        self.assertIsNone(intervention["value_token"])

    def test_marked_passage_offsets_disambiguate_existing_mentions(self):
        passage, original_span, replacement_span = marked_event_passage(
            "Rain begins in the scenario.", "rain begins", "rain stops"
        )
        self.assertEqual(passage[original_span[0] : original_span[1]], "rain begins")
        self.assertEqual(passage[replacement_span[0] : replacement_span[1]], "rain stops")
        self.assertGreater(original_span[0], passage.find("Rain begins"))

    def test_split_is_stable_and_grouped(self):
        first = group_split("same root")
        self.assertIn(first, {"train", "validation", "test"})
        self.assertEqual(first, group_split("same root"))


if __name__ == "__main__":
    unittest.main()
