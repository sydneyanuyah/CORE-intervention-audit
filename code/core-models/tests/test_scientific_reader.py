import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = None


def dag_record(target="B", nodes=None):
    return {
        "structure_kind": "dag",
        "graph": {"nodes": nodes or ["A", "B", "C"], "edges": []},
        "chain": None,
        "intervention": {"target": target},
        "factual": {"question": "What happens to C?"},
    }


@unittest.skipUnless(torch is not None, "torch is unavailable")
class ScientificLayoutTests(unittest.TestCase):
    def test_grounded_slot_candidates_use_overlap_and_fail_closed(self):
        from core_bert.scientific_reader import grounded_slot_candidates

        input_ids = torch.tensor([[8, 9, 10, 11], [20, 21, 22, 23]])
        span = torch.tensor([[False, True, True, False], [False, True, False, False]])
        names = torch.tensor([[[9, 10], [9, 30], [40, 41]], [[50, 51], [60, 61], [0, 0]]])
        name_mask = torch.tensor([[[1, 1], [1, 1], [1, 1]], [[1, 1], [1, 1], [0, 0]]])
        slot_mask = torch.tensor([[True, True, True], [True, True, False]])
        candidates, grounded = grounded_slot_candidates(
            input_ids, span, names, name_mask, slot_mask
        )
        self.assertTrue(torch.equal(candidates[0], torch.tensor([True, False, False])))
        self.assertFalse(candidates[1].any())
        self.assertTrue(torch.equal(grounded, torch.tensor([True, False])))

    def test_post_edit_decode_mask_blocks_passage_bypass(self):
        from core_bert.scientific_reader import build_post_edit_decode_mask

        valid = torch.tensor([[True, True, True, True, False]])
        allowed = build_post_edit_decode_mask(valid, passage_length=2)
        self.assertTrue(allowed[0, 0, 1])
        self.assertFalse(allowed[0, 2, 0])
        self.assertFalse(allowed[0, 3, 1])
        self.assertTrue(allowed[0, 3, 2])
        self.assertFalse(allowed[0, :, 4].any())

        isolated = build_post_edit_decode_mask(
            valid, passage_length=2, slot_count=1
        )
        self.assertFalse(isolated[0, 2, 3])
        self.assertTrue(isolated[0, 3, 2])

        slot_isolated = build_post_edit_decode_mask(
            torch.ones(1, 6, dtype=torch.bool), passage_length=2,
            slot_count=2, isolate_slots=True,
        )
        self.assertTrue(slot_isolated[0, 2, 2])
        self.assertTrue(slot_isolated[0, 3, 3])
        self.assertFalse(slot_isolated[0, 2, 3])
        self.assertFalse(slot_isolated[0, 3, 2])
        self.assertTrue(slot_isolated[0, 4, 2])
        with self.assertRaisesRegex(ValueError, "requires slot_count"):
            build_post_edit_decode_mask(
                torch.ones(1, 4, dtype=torch.bool), passage_length=2,
                isolate_slots=True,
            )

    def test_graph_and_repeated_chain_target_indices_are_deterministic(self):
        from core_bert.scientific_reader import variable_slot_layout

        graph = variable_slot_layout(dag_record())
        self.assertEqual(graph.names, ("A", "B", "C"))
        self.assertEqual(graph.display_names, graph.names)
        self.assertEqual(graph.target_slot, 1)

        described = dag_record()
        described["graph"]["node_text"] = {
            "A": ["alpha"], "B": ["beta", "second beta"], "C": ["gamma"]
        }
        described_layout = variable_slot_layout(described)
        self.assertEqual(
            described_layout.display_names, ("alpha", "beta ; second beta", "gamma")
        )
        chain = {
            "structure_kind": "chain",
            "graph": None,
            "chain": ["repeat", "middle", "repeat"],
            "intervention": {"target": "event_02: repeat"},
            "factual": {"question": "What follows?"},
        }
        aligned = variable_slot_layout(chain)
        self.assertEqual(aligned.target_slot, 2)
        self.assertEqual(aligned.names[0], "event_00: repeat")
        self.assertEqual(aligned.names[2], "event_02: repeat")

    def test_missing_duplicate_and_oversized_structures_fail_closed(self):
        from core_bert.scientific_reader import variable_slot_layout

        with self.assertRaisesRegex(ValueError, "exactly one slot"):
            variable_slot_layout(dag_record(target="Z"))
        with self.assertRaisesRegex(ValueError, "unique"):
            variable_slot_layout(dag_record(nodes=["A", "A"]))
        with self.assertRaisesRegex(ValueError, "1..30"):
            variable_slot_layout(dag_record(target="N0", nodes=[f"N{i}" for i in range(31)]))


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


    class ZeroEditor(nn.Module):
        def forward(self, slots, instruction):
            return slots


@unittest.skipUnless(torch is not None, "torch is unavailable")
class ScientificReaderTests(unittest.TestCase):
    def make_reader(self):
        from core_bert.addressed_reader import AddressedWorldReader
        from core_bert.scientific_reader import ScientificAddressedReader

        torch.manual_seed(12)
        base = AddressedWorldReader(
            FakeBert(), node_count=1, split_layer=2,
            world_slot_count=30, label_count=2,
        )
        return ScientificAddressedReader(base)

    def inputs(self):
        slot_ids = torch.zeros(1, 30, 2, dtype=torch.long)
        slot_ids[0, 0] = torch.tensor([10, 11])
        slot_ids[0, 1] = torch.tensor([12, 13])
        slot_mask = torch.zeros(1, 30, dtype=torch.bool)
        slot_mask[:, :2] = True
        slot_name_mask = slot_mask[:, :, None].expand(-1, -1, 2).long()
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
            "target_span": torch.tensor([[0, 1, 0]], dtype=torch.bool),
            "slot_name_input_ids": slot_ids,
            "slot_name_attention_mask": slot_name_mask,
            "slot_mask": slot_mask,
            "target_slot": torch.tensor([1]),
            "question_attention_mask": torch.ones(1, 2, dtype=torch.long),
            "value_embedding": torch.ones(1, 4),
        }

    def test_question_is_post_edit_and_changes_logits_not_pre_edit_slots(self):
        from core_bert.t3_pointer import T3BPointer

        reader = self.make_reader()
        values = self.inputs()
        first = reader.forward_t3(
            **values, question_input_ids=torch.tensor([[20, 21]]),
            editor=T3BPointer(4, 2), mode="t3b",
        )
        second = reader.forward_t3(
            **values, question_input_ids=torch.tensor([[22, 23]]),
            editor=T3BPointer(4, 2), mode="t3b",
        )
        self.assertTrue(torch.allclose(first.pre_edit_slots, second.pre_edit_slots))
        self.assertFalse(torch.allclose(first.logits, second.logits))

    def test_t3b_reports_target_mass_top1_and_masks_padding_slots(self):
        from core_bert.t3_pointer import T3BPointer

        reader = self.make_reader()
        result = reader.forward_t3(
            **self.inputs(), question_input_ids=torch.tensor([[20, 21]]),
            editor=T3BPointer(4, 2), mode="t3b",
        )
        self.assertEqual(result.pointer_mass.shape, (1,))
        self.assertEqual(result.pointer_top1.shape, (1,))
        self.assertTrue(torch.allclose(result.pointer_probabilities[:, 2:], torch.zeros(1, 28)))
        self.assertTrue(torch.allclose(result.pointer_probabilities.sum(1), torch.ones(1)))


@unittest.skipUnless(torch is not None, "torch is unavailable")
class ScientificReaderHFTests(unittest.TestCase):
    def test_random_hf_bert_runs_dynamic_slots_and_post_edit_question(self):
        try:
            from transformers import BertConfig, BertModel
        except ImportError:
            self.skipTest("transformers is unavailable")
        from core_bert.addressed_reader import AddressedWorldReader
        from core_bert.scientific_reader import ScientificAddressedReader
        from core_bert.t3_pointer import T3BPointer

        config = BertConfig(
            vocab_size=64, hidden_size=12, num_hidden_layers=2,
            num_attention_heads=3, intermediate_size=24,
        )
        base = AddressedWorldReader(
            BertModel(config), node_count=1, split_layer=1,
            world_slot_count=30, label_count=2,
        )
        reader = ScientificAddressedReader(base)
        slot_ids = torch.ones(1, 30, 2, dtype=torch.long)
        slot_name_mask = torch.zeros_like(slot_ids)
        slot_name_mask[:, :2] = 1
        slot_mask = torch.zeros(1, 30, dtype=torch.bool)
        slot_mask[:, :2] = True
        output = reader.forward_t3(
            torch.tensor([[1, 2, 3]]), torch.ones(1, 3, dtype=torch.long),
            torch.tensor([[0, 1, 0]], dtype=torch.bool),
            slot_ids, slot_name_mask, slot_mask, torch.tensor([0]),
            torch.tensor([[4, 5]]), torch.ones(1, 2, dtype=torch.long),
            T3BPointer(12, 2), value_embedding=torch.ones(1, 12), mode="t3b",
        )
        self.assertEqual(output.logits.shape, (1, 2))
        self.assertEqual(output.pre_edit_slots.shape, (1, 30, 12))


if __name__ == "__main__":
    unittest.main()
