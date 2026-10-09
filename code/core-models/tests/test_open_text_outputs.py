import unittest

import torch

from core_bert.open_text_outputs import (
    OpenTextSlotHead,
    balanced_open_text_transition_loss,
    build_open_text_batch,
    normalize_open_text,
    sequence_cross_entropy,
)


class TinyTokenizer:
    cls_token_id = 1
    sep_token_id = 2
    pad_token_id = 0

    def __call__(self, text, **kwargs):
        return {"input_ids": [3 + (sum(map(ord, word)) % 11) for word in text.split()]}


def record():
    chain = ["first event", "second event", "final event"]
    return {
        "source": "com2", "structure_kind": "chain", "chain": chain,
        "intervention": {"target": "event_01: second event"},
        "probes": [
            {"variable": f"event_{i:02d}: {text}", "question": None,
             "answer_before": text, "answer_after": after,
             "actually_changed": changed}
            for i, (text, after, changed) in enumerate([
                (chain[0], chain[0], False),
                (chain[1], "changed second", True),
                (chain[2], "changed final", True),
            ])
        ],
    }


class OpenTextOutputTests(unittest.TestCase):
    def test_builds_explicit_before_after_batch_and_final_query(self):
        batch = build_open_text_batch([record()], TinyTokenizer(), max_tokens=8)
        self.assertEqual(batch.query_slot.tolist(), [2])
        self.assertEqual(batch.after_text[0][-1], "changed final")
        self.assertEqual(int(batch.label_mask.sum()), 3)
        self.assertTrue(batch.target_mask[0, 1])

    def test_shared_generator_and_balanced_transition_loss(self):
        batch = build_open_text_batch([record()], TinyTokenizer(), max_tokens=8)
        head = OpenTextSlotHead(8, 20, padding_idx=0, bos_token_id=1, eos_token_id=2)
        slots = torch.randn(1, 30, 8)
        after = head(slots, batch.after_inputs, batch.label_mask)
        before = head(slots, batch.before_inputs, batch.before_label_mask)
        after_loss = sequence_cross_entropy(after, batch.after_targets[batch.label_mask])
        before_loss = sequence_cross_entropy(before, batch.before_targets[batch.before_label_mask])
        loss = balanced_open_text_transition_loss(after_loss, before_loss, batch)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(head.generate(slots, batch.variable_mask, max_tokens=5).shape, (3, 5))

    def test_normalization_changes_only_unicode_case_and_spacing(self):
        self.assertEqual(normalize_open_text("  Café   Event "), "café event")
        self.assertNotEqual(normalize_open_text("event."), normalize_open_text("event"))


if __name__ == "__main__":
    unittest.main()
