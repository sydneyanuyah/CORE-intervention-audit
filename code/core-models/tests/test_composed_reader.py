import sys
import unittest
from dataclasses import replace
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = None


def _single(record_id, value):
    return {
        "id": record_id,
        "source": "xor",
        "factual": {"passage": "X controls Y.", "question": "What is Y?"},
        "intervention": {"target": "X", "value": value, "text": f"Set X={value}"},
    }


def _example(first="set-one", second="set-zero"):
    from core_bert.two_edit import build_ordered_two_edit_examples

    records = {
        "set-one": _single("set-one", 1),
        "set-zero": _single("set-zero", 0),
    }
    groups = {key: ("graph-1", "world-1") for key in records}
    row = {
        "first_record_id": first,
        "second_record_id": second,
        "factual_outputs": [0, 1],
        "gold_outputs": [1, 1],
        "change_mask": [True, False],
        "preservation_mask": [False, True],
        "target_mask": [True, False],
    }
    return build_ordered_two_edit_examples(
        records, groups, [row], seen_separately=set(records)
    )[0]


if torch is not None:
    class FakeEmbeddings(nn.Module):
        def __init__(self, vocabulary=128, hidden=4):
            super().__init__()
            self.word_embeddings = nn.Embedding(vocabulary, hidden)

        def forward(self, input_ids):
            return self.word_embeddings(input_ids)


    class FakeLayer(nn.Module):
        def forward(self, hidden, attention_mask):
            allowed = attention_mask[:, 0] == 0
            weights = allowed.to(hidden.dtype)
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1)
            return (hidden + torch.bmm(weights, hidden),)


    class FakeBert(nn.Module):
        def __init__(self, hidden=4):
            super().__init__()
            self.config = type("Config", (), {"hidden_size": hidden})()
            self.embeddings = FakeEmbeddings(hidden=hidden)
            self.encoder = nn.Module()
            self.encoder.layer = nn.ModuleList([FakeLayer() for _ in range(4)])


    class OverwriteEditor(nn.Module):
        """Test operator whose second instruction vector is the new slot value."""

        def forward(self, slots, instruction):
            hidden = slots.shape[-1]
            return instruction[:, None, hidden:].expand_as(slots).clone()


    class IdentityEditor(nn.Module):
        def forward(self, slots, instruction):
            return slots


@unittest.skipUnless(torch is not None, "torch is unavailable")
class ComposedReaderTests(unittest.TestCase):
    def reader(self):
        from core_bert.addressed_reader import AddressedWorldReader
        from core_bert.scientific_reader import ScientificAddressedReader

        torch.manual_seed(41)
        base = AddressedWorldReader(
            FakeBert(), node_count=1, split_layer=2,
            world_slot_count=30, label_count=2,
        )
        return ScientificAddressedReader(base)

    def common(self, input_ids):
        slot_ids = torch.zeros(1, 30, 2, dtype=torch.long)
        slot_ids[0, 0] = torch.tensor([10, 11])
        slot_ids[0, 1] = torch.tensor([12, 13])
        slot_mask = torch.zeros(1, 30, dtype=torch.bool)
        slot_mask[:, :2] = True
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "slot_name_input_ids": slot_ids,
            "slot_name_attention_mask": slot_mask[:, :, None].expand(-1, -1, 2).long(),
            "slot_mask": slot_mask,
            "question_input_ids": torch.tensor([[20, 21]]),
            "question_attention_mask": torch.ones(1, 2, dtype=torch.long),
        }

    def t3_side(self, record_id, value):
        values = self.common(torch.tensor([[1, 2, 3, 9, 30]]))
        values.update({
            "record_ids": [record_id],
            "command_tokens": torch.tensor([[0, 0, 0, 0, 1]], dtype=torch.bool),
            "do_tokens": torch.tensor([[0, 0, 0, 1, 0]], dtype=torch.bool),
            "address_marker_tokens": torch.zeros(1, 5, dtype=torch.bool),
            "target_tokens": torch.tensor([[0, 1, 0, 0, 0]], dtype=torch.bool),
            "replacement_tokens": torch.zeros(1, 5, dtype=torch.bool),
            "target_slot": torch.tensor([0]),
            "slot_support": torch.eye(30, dtype=torch.bool)[None],
            "value_embedding": torch.full((1, 4), float(value)),
        })
        return values

    def t2_side(self, record_id, command_id):
        values = self.common(torch.tensor([[1, 2, 3, 9, command_id]]))
        values.update({
            "record_ids": [record_id],
            "command_tokens": torch.tensor([[0, 0, 0, 0, 1]], dtype=torch.bool),
            "do_tokens": torch.tensor([[0, 0, 0, 1, 0]], dtype=torch.bool),
        })
        return values

    def reference_side(self, record_id, command_id, value_id):
        values = self.t2_side(record_id, command_id)
        values.update({
            "address_marker_tokens": torch.zeros(1, 5, dtype=torch.bool),
            "target_slot": torch.tensor([0]),
            "value_ids": torch.tensor([value_id]),
        })
        return values

    def test_same_target_is_sequential_and_order_sensitive(self):
        from core_bert.composed_reader import ComposedScientificBatch, ComposedScientificExecutor

        forward_example = _example("set-one", "set-zero")
        forward_batch = ComposedScientificBatch(
            (forward_example,), self.t3_side("set-one", 1), self.t3_side("set-zero", 0)
        )
        reverse_example = _example("set-zero", "set-one")
        reverse_batch = ComposedScientificBatch(
            (reverse_example,), self.t3_side("set-zero", 0), self.t3_side("set-one", 1)
        )
        executor = ComposedScientificExecutor(self.reader(), OverwriteEditor(), "t3a")
        forward = executor(forward_batch)
        reverse = executor(reverse_batch)

        self.assertTrue(
            torch.allclose(forward.edited_slots, torch.zeros_like(forward.edited_slots))
        )
        self.assertTrue(
            torch.allclose(reverse.edited_slots, torch.ones_like(reverse.edited_slots))
        )
        self.assertFalse(torch.allclose(forward.edited_slots, reverse.edited_slots))
        self.assertEqual(forward.traces[0].component_record_ids, ("set-one",))
        self.assertEqual(forward.traces[1].component_record_ids, ("set-zero",))
        self.assertTrue(torch.equal(forward.traces[1].slots_before, forward.traces[0].slots_after))
        self.assertIsNotNone(forward.traces[0].address)
        self.assertFalse(forward.a1_evidence)

    def test_t2b_identity_editor_has_no_direct_command_path(self):
        from core_bert.composed_reader import ComposedScientificBatch, ComposedScientificExecutor

        example = _example()
        original = ComposedScientificBatch(
            (example,), self.t2_side("set-one", 30), self.t2_side("set-zero", 31)
        )
        changed_commands = ComposedScientificBatch(
            (example,), self.t2_side("set-one", 40), self.t2_side("set-zero", 41)
        )
        reader = self.reader()
        reader.eval()
        protected = ComposedScientificExecutor(reader, IdentityEditor(), "t2b")
        first = protected(original)
        second = protected(changed_commands)
        self.assertTrue(torch.allclose(first.edited_slots, second.edited_slots))
        self.assertTrue(torch.allclose(first.logits, second.logits))

        # The naive ablation remains command-visible, proving the check is able
        # to detect the path that T2-b removes.
        naive = ComposedScientificExecutor(reader, IdentityEditor(), "t2a")
        self.assertFalse(torch.allclose(naive(original).logits, naive(changed_commands).logits))

    def test_a2_zero_editor_matches_physically_absent_instruction(self):
        from core_bert.composed_reader import ComposedScientificBatch, ComposedScientificExecutor

        example = _example()
        batch = ComposedScientificBatch(
            (example,), self.t2_side("set-one", 30), self.t2_side("set-zero", 31)
        )
        reader = self.reader()
        reader.eval()
        zeroed = ComposedScientificExecutor(
            reader, IdentityEditor(), "t2b",
            editor_enabled=False, a2_control="editor_zeroed",
        )
        absent = ComposedScientificExecutor(
            reader, IdentityEditor(), "t2b",
            editor_enabled=False, a2_control="no_instruction",
        )
        with torch.no_grad():
            zeroed_output = zeroed(batch)
            absent_output = absent(batch)
        self.assertTrue(torch.allclose(zeroed_output.logits, absent_output.logits))
        self.assertTrue(
            torch.allclose(zeroed_output.edited_slots, absent_output.edited_slots)
        )

    def test_a2_prompting_opens_the_ordinary_attention_path(self):
        from core_bert.composed_reader import ComposedScientificBatch, ComposedScientificExecutor

        example = _example()
        first = ComposedScientificBatch(
            (example,), self.t2_side("set-one", 30), self.t2_side("set-zero", 31)
        )
        changed = ComposedScientificBatch(
            (example,), self.t2_side("set-one", 40), self.t2_side("set-zero", 41)
        )
        reader = self.reader()
        reader.eval()
        prompting = ComposedScientificExecutor(
            reader, IdentityEditor(), "t2b",
            editor_enabled=False, a2_control="prompting",
        )
        with torch.no_grad():
            self.assertFalse(torch.allclose(prompting(first).logits, prompting(changed).logits))

    def test_t0_composition_uses_ids_and_hides_command_text(self):
        from core_bert.composed_reader import ComposedScientificBatch, ComposedScientificExecutor
        from core_bert.reference_operators import LearnedInterventionEncoder

        example = _example()
        original = ComposedScientificBatch(
            (example,), self.reference_side("set-one", 30, 13),
            self.reference_side("set-zero", 31, 14),
        )
        changed_words = ComposedScientificBatch(
            (example,), self.reference_side("set-one", 40, 13),
            self.reference_side("set-zero", 41, 14),
        )
        executor = ComposedScientificExecutor(
            self.reader(), IdentityEditor(), "t0",
            intervention_encoder=LearnedInterventionEncoder(450, 4),
        )
        executor.eval()
        first, second = executor(original), executor(changed_words)
        self.assertTrue(torch.allclose(first.logits, second.logits))
        self.assertTrue(torch.equal(first.traces[1].slots_before, first.traces[0].slots_after))

    def test_execution_fails_closed_without_authoritative_truth(self):
        from core_bert.composed_reader import ComposedScientificBatch

        invalid = replace(_example(), gold_outputs=())
        with self.assertRaisesRegex(ValueError, "authoritative final two-edit truth"):
            ComposedScientificBatch(
                (invalid,), self.t2_side("set-one", 30), self.t2_side("set-zero", 31)
            )

    def test_final_slots_feed_existing_per_variable_head(self):
        from core_bert.composed_reader import (
            ComposedScientificBatch,
            ComposedScientificExecutor,
            ComposedVariableModel,
        )
        from core_bert.variable_outputs import PerVariableOutputHead

        example = _example()
        batch = ComposedScientificBatch(
            (example,), self.t3_side("set-one", 1), self.t3_side("set-zero", 0)
        )
        model = ComposedVariableModel(
            ComposedScientificExecutor(self.reader(), OverwriteEditor(), "t3a"),
            PerVariableOutputHead(4, 2),
        )
        output = model(batch)
        self.assertEqual(output.variables.logits.shape, (1, 30, 2))
        self.assertTrue(torch.equal(output.variables.variable_mask, batch.first["slot_mask"]))
        self.assertTrue(
            torch.equal(
                output.composed.edited_slots, output.composed.traces[1].slots_after
            )
        )


if __name__ == "__main__":
    unittest.main()
