import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

try:
    import torch
except ImportError:
    torch = None


def dag_record(probes):
    return {
        "structure_kind": "dag",
        "graph": {"nodes": ["A", "B", "C"], "edges": [["A", "B"]]},
        "chain": None,
        "intervention": {"target": "B"},
        "factual": {"question": "What happens?"},
        "probes": probes,
    }


@unittest.skipUnless(torch is not None, "torch is unavailable")
class VariableOutputTests(unittest.TestCase):
    def imports(self):
        from core_bert.variable_outputs import (
            PerVariableOutputHead,
            balanced_variable_cross_entropy,
            build_variable_batch,
            encode_variable_labels,
            per_variable_metrics,
            summarize_variable_records,
            variable_cross_entropy,
        )

        return (
            PerVariableOutputHead,
            balanced_variable_cross_entropy,
            build_variable_batch,
            encode_variable_labels,
            per_variable_metrics,
            summarize_variable_records,
            variable_cross_entropy,
        )

    def test_ids_and_masks_follow_dynamic_slot_order_without_inference(self):
        _, _, build, _, _, _, _ = self.imports()
        graph = dag_record(
            [
                {"variable": "C", "answer_before": "less", "answer_after": "more", "question": "C?"},
                {"variable": "A", "answer_before": None, "answer_after": None},
            ]
        )
        chain = {
            "structure_kind": "chain",
            "graph": None,
            "chain": ["repeat", "middle", "repeat"],
            "intervention": {"target": "event_02: repeat"},
            "factual": {"question": "What follows?"},
            "probes": [{"variable": "event_02: repeat", "answer_after": "changed"}],
        }
        batch = build([graph, chain], max_slots=4)
        self.assertEqual(batch.variable_ids[0], ("A", "B", "C", ""))
        self.assertEqual(
            batch.variable_ids[1],
            ("event_00: repeat", "event_01: middle", "event_02: repeat", ""),
        )
        self.assertEqual(batch.variable_mask.tolist(), [[True, True, True, False]] * 2)
        self.assertEqual(batch.label_mask.tolist(), [
            [True, False, True, False], [False, False, True, False]
        ])
        # Explicit JSON null is supervised; absent B remains masked.
        self.assertIsNone(batch.labels[0][0])
        self.assertFalse(batch.label_mask[0, 1])
        self.assertEqual(batch.before_label_mask.tolist(), [
            [True, False, True, False], [False, False, False, False]
        ])
        self.assertIsNone(batch.before_labels[0][0])
        self.assertEqual(batch.query_slot.tolist(), [2, -1])

    def test_exact_vocabulary_encoding_rejects_open_or_coerced_values(self):
        _, _, build, encode, _, _, _ = self.imports()
        batch = build([dag_record([{"variable": "A", "answer_after": 1}])], max_slots=3)
        self.assertEqual(encode(batch, [0, 1]).tolist(), [[1, -100, -100]])
        with self.assertRaisesRegex(ValueError, "outside the declared vocabulary"):
            encode(batch, ["0", "1"])
        with self.assertRaisesRegex(ValueError, "duplicate JSON values"):
            encode(batch, ["more", "more"])

    def test_balanced_masks_and_loss_weight_groups_equally(self):
        _, balanced, build, encode, _, _, _ = self.imports()
        record = dag_record([
            {"variable": "A", "answer_after": 0, "actually_changed": True},
            {"variable": "B", "answer_after": 1, "actually_changed": True},
            {"variable": "C", "answer_after": 0, "actually_changed": False},
        ])
        batch = build([record], max_slots=3)
        self.assertEqual(batch.target_mask.tolist(), [[False, True, False]])
        self.assertEqual(batch.changed_mask.tolist(), [[True, False, False]])
        self.assertEqual(batch.preservation_mask.tolist(), [[False, False, True]])
        labels = encode(batch, [0, 1])
        logits = torch.tensor([[[8.0, 0.0], [8.0, 0.0], [8.0, 0.0]]])
        loss = balanced(
            logits, labels, batch.label_mask, batch.variable_mask,
            batch.target_mask, batch.changed_mask, batch.preservation_mask,
        )
        expected = torch.stack([
            torch.nn.functional.cross_entropy(logits[:, 1], labels[:, 1]),
            torch.nn.functional.cross_entropy(logits[:, 0], labels[:, 0]),
            torch.nn.functional.cross_entropy(logits[:, 2], labels[:, 2]),
        ]).mean()
        self.assertTrue(torch.allclose(loss, expected))

    def test_transition_loss_pairs_before_and_after_with_shared_groups(self):
        from core_bert.variable_outputs import balanced_transition_cross_entropy

        _, _, build, encode, _, _, _ = self.imports()
        record = dag_record([
            {"variable": "A", "answer_before": 0, "answer_after": 1, "actually_changed": True},
            {"variable": "B", "answer_before": 0, "answer_after": 1, "actually_changed": True},
            {"variable": "C", "answer_before": 1, "answer_after": 1, "actually_changed": False},
        ])
        batch = build([record], max_slots=3)
        before = encode(batch, [0, 1], state="before")
        after = encode(batch, [0, 1])
        pre_logits = torch.tensor([[[8.0, 0.0], [8.0, 0.0], [0.0, 8.0]]], requires_grad=True)
        edited_logits = torch.tensor([[[0.0, 8.0], [0.0, 8.0], [0.0, 8.0]]], requires_grad=True)
        loss = balanced_transition_cross_entropy(
            edited_logits, pre_logits, after, before,
            batch.label_mask, batch.before_label_mask, batch.variable_mask,
            batch.target_mask, batch.changed_mask, batch.preservation_mask,
        )
        self.assertLess(loss.item(), 0.001)
        loss.backward()
        self.assertIsNotNone(pre_logits.grad)
        self.assertIsNotNone(edited_logits.grad)

    def test_before_encoding_rejects_unknown_state(self):
        _, _, build, encode, _, _, _ = self.imports()
        batch = build([
            dag_record([{"variable": "A", "answer_before": 0, "answer_after": 1}])
        ], max_slots=3)
        self.assertEqual(encode(batch, [0, 1], state="before").tolist(), [[0, -100, -100]])
        with self.assertRaisesRegex(ValueError, "state"):
            encode(batch, [0, 1], state="during")

    def test_explicit_query_slot_controls_task_readout_with_fallback(self):
        from core_bert.variable_outputs import task_logits_from_queried_variable

        fallback = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        variables = torch.arange(2 * 3 * 5, dtype=torch.float).reshape(2, 3, 5)
        result = task_logits_from_queried_variable(
            fallback, variables, torch.tensor([2, -1])
        )
        self.assertEqual(result.shape, (2, 5))
        self.assertTrue(torch.equal(result[0], variables[0, 2]))
        self.assertTrue(torch.equal(result[1, :4], fallback[1]))
        self.assertLess(result[1, 4].item(), -1e20)

    def test_multiple_question_bearing_probes_fail_closed(self):
        _, _, build, _, _, _, _ = self.imports()
        with self.assertRaisesRegex(ValueError, "at most one"):
            build([dag_record([
                {"variable": "A", "answer_after": 0, "question": "A?"},
                {"variable": "B", "answer_after": 1, "question": "B?"},
            ])], max_slots=3)

    def test_unknown_and_duplicate_probe_variables_fail_closed(self):
        _, _, build, _, _, _, _ = self.imports()
        with self.assertRaisesRegex(ValueError, "do not map"):
            build([dag_record([{"variable": "Z", "answer_after": "more"}])], max_slots=3)
        with self.assertRaisesRegex(ValueError, "duplicate probe"):
            build(
                [dag_record([
                    {"variable": "A", "answer_after": "more"},
                    {"variable": "A", "answer_after": "less"},
                ])],
                max_slots=3,
            )

    def test_head_loss_and_metrics_respect_label_and_variable_masks(self):
        Head, _, _, _, metrics, _, loss = self.imports()
        head = Head(4, 3)
        slots = torch.randn(2, 3, 4)
        active = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)
        self.assertEqual(head(slots, active).logits.shape, (2, 3, 3))
        logits = torch.tensor([
            [[8.0, 0.0], [0.0, 8.0], [8.0, 0.0]],
            [[0.0, 8.0], [8.0, 0.0], [0.0, 8.0]],
        ])
        labels = torch.tensor([[0, 1, -100], [1, 1, -100]])
        labelled = torch.tensor([[1, 1, 0], [1, 1, 0]], dtype=torch.bool)
        result = metrics(logits, labels, labelled, active)
        self.assertEqual(result["labelled_variable_count"], 4)
        self.assertEqual(result["variable_accuracy"], 0.75)
        self.assertEqual(result["fully_labelled_record_count"], 1)
        self.assertEqual(result["record_exact_match"], 1.0)
        self.assertGreater(loss(logits, labels, labelled, active).item(), 0.0)

    def test_metrics_do_not_report_an_exact_score_without_full_coverage(self):
        _, _, _, _, metrics, _, loss = self.imports()
        logits = torch.zeros(1, 2, 2)
        active = torch.ones(1, 2, dtype=torch.bool)
        one_label = torch.tensor([[True, False]])
        labels = torch.tensor([[0, -100]])
        result = metrics(logits, labels, one_label, active)
        self.assertIsNone(result["record_exact_match"])
        self.assertEqual(result["record_exact_match_coverage"], 0.0)
        self.assertIsNotNone(result["record_exact_match_unavailable_reason"])
        with self.assertRaisesRegex(ValueError, "no explicit"):
            metrics(logits, labels, torch.zeros_like(one_label), active)
        with self.assertRaisesRegex(ValueError, "no explicit"):
            loss(logits, labels, torch.zeros_like(one_label), active)
        with self.assertRaisesRegex(ValueError, "at least one active"):
            metrics(logits, labels, one_label, torch.zeros_like(active))

    def test_exact_gather_summary_preserves_partial_coverage(self):
        _, _, _, _, _, summarize, _ = self.imports()
        result = summarize([
            {
                "active_variable_count": 2,
                "variable_ids": ["A", "B"],
                "variable_gold": [0, 1],
                "variable_predicted": [0, 1],
                "variable_groups": ["target", "change"],
                "variable_loss_sum": 0.4,
            },
            {
                "active_variable_count": 3,
                "variable_ids": ["X"],
                "variable_gold": [1],
                "variable_predicted": [0],
                "variable_groups": ["preservation"],
                "variable_loss_sum": 0.6,
            },
        ])
        self.assertEqual(result["label_coverage"], 3 / 5)
        self.assertEqual(result["variable_accuracy"], 2 / 3)
        self.assertEqual(result["record_exact_match_coverage"], 0.5)
        self.assertEqual(result["record_exact_match"], 1.0)
        self.assertEqual(result["groups"]["target"]["accuracy"], 1.0)
        self.assertEqual(result["groups"]["change"]["accuracy"], 1.0)
        self.assertEqual(result["groups"]["preservation"]["accuracy"], 0.0)


if __name__ == "__main__":
    unittest.main()
