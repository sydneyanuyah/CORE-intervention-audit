import copy
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

try:
    import torch
except ImportError:
    torch = None


class FakeFastTokenizer:
    cls_token_id = 101
    pad_token_id = 0
    unk_token_id = 100

    def __init__(self):
        self.vocabulary = {"[DO]": 777}

    def convert_tokens_to_ids(self, token):
        return self.vocabulary.get(token, self.unk_token_id)

    def __call__(self, text, *, add_special_tokens, return_offsets_mapping, truncation):
        assert not add_special_tokens and not truncation
        matches = list(re.finditer(r"\S+", text))
        result = {"input_ids": [1000 + index for index in range(len(matches))]}
        if return_offsets_mapping:
            result["offset_mapping"] = [(match.start(), match.end()) for match in matches]
        return result


def fixture(source, passage, target, target_text, value_token, label):
    start = passage.index(target_text)
    return {
        "id": f"{source}-fixture",
        "source": source,
        "factual": {"passage": passage},
        "intervention": {
            "target": target,
            "target_text": target_text,
            "target_span": [start, start + len(target_text)],
            "value": value_token,
            "value_token": value_token,
            "replacement_span": None,
            "kind": "value_set",
            "text": f"Set {target_text} to {value_token}.",
        },
        "intervened": {"answer": label},
    }


@unittest.skipUnless(torch is not None, "torch is unavailable")
class AddressedDataTests(unittest.TestCase):
    def setUp(self):
        from core_bert.addressed_data import AddressedBatchCollator, CLOSED_VALUE_TOKENS

        self.collator = AddressedBatchCollator(FakeFastTokenizer(), max_length=64)
        self.AddressedBatchCollator = AddressedBatchCollator
        self.CLOSED_VALUE_TOKENS = CLOSED_VALUE_TOKENS

    def test_four_source_batch_layout_masks_values_and_metadata(self):
        records = [
            fixture("cladder", "The treatment is initially absent.", "X", "treatment", "yes", "yes"),
            fixture("wiqa", "[TARGET] The water level controls plant growth.", "node-4", "water level", "less", "less"),
            fixture("ccrgb", "Allele A17 influences the outcome.", "scm:A17", "Allele A17", "1", "yes"),
        ]
        passage = "A story. [ORIG] old event [REPL] new event"
        target_start = passage.index("old event")
        replacement_start = passage.index("new event")
        records.append(
            {
                "id": "com2-fixture",
                "source": "com2",
                "factual": {"passage": passage},
                "intervention": {
                    "target": "event_01: old event",
                    "target_text": "old event",
                    "target_span": [target_start, target_start + len("old event")],
                    "value": "new event",
                    "value_token": None,
                    "replacement_span": [replacement_start, replacement_start + len("new event")],
                    "kind": "event_replace",
                    "text": "Replace old event with new event.",
                },
                "intervened": {"answer": "fixture open answer"},
            }
        )
        batch = self.collator(records)
        self.assertEqual(batch["record_ids"], [record["id"] for record in records])
        self.assertEqual(batch["sources"], ["cladder", "wiqa", "ccrgb", "com2"])
        self.assertEqual(batch["labels"], ["yes", "less", "yes", "fixture open answer"])
        self.assertEqual(batch["label_ids"].tolist(), [1, 2, 1, -1])
        self.assertTrue(torch.all(batch["do_tokens"].sum(1) == 1))
        self.assertTrue(torch.all(batch["command_tokens"].sum(1) >= 1))
        self.assertFalse((batch["do_tokens"] & batch["command_tokens"]).any())
        self.assertEqual(batch["address_marker_tokens"].shape, batch["input_ids"].shape)
        self.assertGreater(batch["address_marker_tokens"][1].sum().item(), 0)
        self.assertEqual(batch["address_marker_tokens"][[0, 2, 3]].sum().item(), 0)
        self.assertFalse((batch["target_tokens"] & ~batch["valid_tokens"]).any())
        self.assertEqual(batch["replacement_tokens"].sum(1).tolist()[:3], [0, 0, 0])
        self.assertGreater(batch["replacement_tokens"][3].sum().item(), 0)
        self.assertEqual(batch["value_ids"][3].item(), -1)
        for row, record in enumerate(records):
            offsets = batch["offset_mapping"][row]
            selected = offsets[batch["target_tokens"][row]]
            start, end = selected[:, 0].min().item(), selected[:, 1].max().item()
            gold_start, gold_end = record["intervention"]["target_span"]
            self.assertLessEqual(start, gold_start)
            self.assertGreaterEqual(end, gold_end)

    def test_rejects_truncation_removing_target_or_replacement(self):
        short = self.AddressedBatchCollator(FakeFastTokenizer(), max_length=8)
        target_record = fixture(
            "wiqa", "zero one two three four five target phrase", "n9", "target phrase", "more", "more"
        )
        with self.assertRaisesRegex(ValueError, "truncation removes an addressed span"):
            short([target_record])

        event = copy.deepcopy(target_record)
        event["id"] = "com2-truncated"
        event["source"] = "com2"
        event["factual"]["passage"] = "old event zero one two three four replacement event"
        event["intervention"].update(
            target="event_00: old event",
            target_text="old event",
            target_span=[0, 9],
            value="replacement event",
            value_token=None,
            replacement_span=[34, 51],
            kind="event_replace",
        )
        with self.assertRaisesRegex(ValueError, "truncation removes an addressed span"):
            short([event])

    def test_closed_vocabulary_and_registered_do_token_are_enforced(self):
        self.assertLessEqual(len(self.CLOSED_VALUE_TOKENS), 16)
        record = fixture("cladder", "The treatment is absent.", "X", "treatment", "unknown", "yes")
        with self.assertRaisesRegex(ValueError, "unknown canonical value token"):
            self.collator([record])
        with self.assertRaisesRegex(ValueError, "register.*special token"):
            self.AddressedBatchCollator(FakeFastTokenizer(), do_token="[MISSING]")


if __name__ == "__main__":
    unittest.main()
