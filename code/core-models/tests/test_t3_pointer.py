import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None, "torch is unavailable in the local lightweight test environment")
class T3PointerTests(unittest.TestCase):
    def setUp(self):
        from core_bert.t3_pointer import (
            ClosedValueEmbedding,
            T3BPointer,
            build_span_instruction,
            char_spans_to_token_mask,
            mean_pool_span,
            override_span_mask,
            pointer_metrics,
            pointer_target_cross_entropy,
        )
        self.ClosedValueEmbedding = ClosedValueEmbedding
        self.T3BPointer = T3BPointer
        self.build_span_instruction = build_span_instruction
        self.char_spans_to_token_mask = char_spans_to_token_mask
        self.mean_pool_span = mean_pool_span
        self.override_span_mask = override_span_mask
        self.pointer_metrics = pointer_metrics
        self.pointer_target_cross_entropy = pointer_target_cross_entropy

    def test_value_and_event_instructions(self):
        hidden = torch.arange(2 * 5 * 4, dtype=torch.float32).reshape(2, 5, 4)
        target = torch.tensor([[0, 1, 1, 0, 0], [1, 0, 0, 0, 0]], dtype=torch.bool)
        replacement = torch.tensor([[0, 0, 0, 1, 1], [0, 0, 1, 1, 0]], dtype=torch.bool)
        values = self.ClosedValueEmbedding(4, 4)(torch.tensor([0, 1]))
        value_instruction = self.build_span_instruction(hidden, target, value_embedding=values)
        event_instruction = self.build_span_instruction(hidden, target, replacement_span=replacement)
        self.assertEqual(value_instruction.shape, (2, 8))
        self.assertTrue(torch.equal(event_instruction[:, :4], self.mean_pool_span(hidden, target)))
        self.assertTrue(torch.equal(event_instruction[:, 4:], self.mean_pool_span(hidden, replacement)))

    def test_gold_character_offsets_map_to_tokens(self):
        offsets = torch.tensor([[[0, 0], [0, 4], [5, 14], [15, 18], [0, 0]]])
        mask = self.char_spans_to_token_mask(offsets, torch.tensor([[5, 14]]))
        self.assertTrue(torch.equal(mask, torch.tensor([[0, 0, 1, 0, 0]], dtype=torch.bool)))

    def test_pointer_mass_and_weighted_selection(self):
        pointer = self.T3BPointer(hidden_size=3, rank=2)
        with torch.no_grad():
            pointer.query.weight.copy_(torch.eye(3) * 8)
            pointer.key.weight.copy_(torch.eye(3) * 8)
            pointer.editor.up.weight.fill_(0.5)
            pointer.editor.down.weight.fill_(0.25)
        slots = torch.eye(3).unsqueeze(0)
        address = torch.tensor([[0.0, 1.0, 0.0]])
        instruction = torch.ones(1, 6)
        output = pointer(slots, address, instruction)
        metrics = self.pointer_metrics(output.probabilities, torch.tensor([1]))
        self.assertGreater(metrics["pointer_accuracy"], 0.99)
        self.assertEqual(metrics["pointer_top1_accuracy"], 1.0)
        delta = (output.edited_slots - slots).norm(dim=-1)
        self.assertGreater(delta[0, 1].item(), 100 * delta[0, 0].item())

    def test_pointer_target_cross_entropy_is_exact_and_differentiable(self):
        probabilities = torch.tensor([0.5, 0.25], requires_grad=True)
        loss = self.pointer_target_cross_entropy(probabilities)
        self.assertAlmostEqual(loss.item(), -(math.log(0.5) + math.log(0.25)) / 2)
        loss.backward()
        self.assertTrue(torch.isfinite(probabilities.grad).all())
        with self.assertRaises(ValueError):
            self.pointer_target_cross_entropy(torch.tensor([[0.5]]))

    def test_hard_pointer_edits_exactly_one_slot_and_keeps_gradients(self):
        pointer = self.T3BPointer(hidden_size=3, rank=2, hard_selection=True)
        with torch.no_grad():
            pointer.query.weight.copy_(torch.eye(3))
            pointer.key.weight.copy_(torch.eye(3))
            pointer.editor.up.weight.fill_(0.5)
            pointer.editor.down.weight.fill_(0.25)
        slots = torch.eye(3).unsqueeze(0).requires_grad_()
        output = pointer(slots, torch.tensor([[0.0, 3.0, 0.0]]), torch.ones(1, 6))
        changed = (output.edited_slots.detach() - slots.detach()).norm(dim=-1) > 0
        self.assertEqual(changed.sum().item(), 1)
        self.assertTrue(changed[0, 1])
        output.edited_slots.sum().backward()
        self.assertIsNotNone(pointer.query.weight.grad)

    def test_hard_pointer_edits_selected_structural_support(self):
        pointer = self.T3BPointer(hidden_size=3, rank=2, hard_selection=True)
        with torch.no_grad():
            pointer.query.weight.copy_(torch.eye(3))
            pointer.key.weight.copy_(torch.eye(3))
            pointer.editor.up.weight.fill_(0.5)
            pointer.editor.down.weight.fill_(0.25)
        slots = torch.eye(3).unsqueeze(0)
        support = torch.tensor([[
            [1, 1, 1], [0, 1, 1], [0, 0, 1]
        ]], dtype=torch.bool)
        output = pointer(
            slots, torch.tensor([[0.0, 3.0, 0.0]]), torch.ones(1, 6),
            support_mask=support,
        )
        changed = (output.edited_slots - slots).norm(dim=-1) > 0
        self.assertTrue(torch.equal(changed, torch.tensor([[False, True, True]])))

    def test_pointer_can_use_separate_uncontextualized_name_slots(self):
        pointer = self.T3BPointer(hidden_size=4, rank=2)
        contextual = torch.randn(2, 3, 4)
        names = torch.randn(2, 3, 4)
        address = torch.randn(2, 4)
        instruction = torch.randn(2, 8)
        mask = torch.ones(2, 3, dtype=torch.bool)
        expected = pointer.pointer_probabilities(address, names, mask)
        output = pointer(
            contextual, address, instruction, mask, pointer_slots=names
        )
        self.assertTrue(torch.allclose(output.probabilities, expected))
        with self.assertRaisesRegex(ValueError, "match edited slot shape"):
            pointer(
                contextual, address, instruction, mask,
                pointer_slots=torch.randn(2, 2, 4),
            )

    def test_candidate_grounding_restricts_pointer_and_can_disable_edit(self):
        pointer = self.T3BPointer(hidden_size=3, rank=2, hard_selection=True)
        with torch.no_grad():
            pointer.editor.up.weight.fill_(0.5)
            pointer.editor.down.weight.fill_(0.25)
        slots = torch.eye(3).unsqueeze(0).repeat(2, 1, 1)
        output = pointer(
            slots, torch.ones(2, 3), torch.ones(2, 6),
            slot_mask=torch.ones(2, 3, dtype=torch.bool),
            candidate_mask=torch.tensor([[False, True, False], [False, False, False]]),
            edit_enabled=torch.tensor([True, False]),
        )
        self.assertEqual(output.probabilities[0].argmax().item(), 1)
        self.assertTrue(torch.allclose(output.edited_slots[1], slots[1]))

    def test_correct_adjacent_random_span_overrides(self):
        valid = torch.ones(2, 10, dtype=torch.bool)
        span = torch.zeros_like(valid)
        span[0, 4:6] = True
        span[1, 1:4] = True
        self.assertTrue(torch.equal(self.override_span_mask(span, valid, "correct"), span))
        adjacent = self.override_span_mask(span, valid, "adjacent")
        random = self.override_span_mask(span, valid, "random", generator=torch.Generator().manual_seed(7))
        for changed in (adjacent, random):
            self.assertTrue(torch.equal(changed.sum(1), span.sum(1)))
            self.assertFalse((changed & span).any())
            self.assertTrue((changed <= valid).all())


if __name__ == "__main__":
    unittest.main()
