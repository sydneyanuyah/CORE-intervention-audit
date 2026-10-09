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


if torch is not None:
    class FakeEmbeddings(nn.Module):
        def __init__(self, vocabulary, hidden):
            super().__init__()
            self.word_embeddings = nn.Embedding(vocabulary, hidden)

        def forward(self, input_ids):
            return self.word_embeddings(input_ids)


    class FakeLayer(nn.Module):
        """Attention-only layer whose information flow is easy to inspect."""

        def forward(self, hidden, attention_mask):
            allowed = attention_mask[:, 0] == 0
            weights = allowed.to(hidden.dtype)
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1)
            return (hidden + torch.bmm(weights, hidden),)


    class FakeBert(nn.Module):
        def __init__(self, hidden=4, layer_count=4):
            super().__init__()
            self.config = type("Config", (), {"hidden_size": hidden})()
            self.embeddings = FakeEmbeddings(64, hidden)
            self.encoder = nn.Module()
            self.encoder.layer = nn.ModuleList([FakeLayer() for _ in range(layer_count)])


    class ZeroEditor(nn.Module):
        def forward(self, slots, instruction):
            return slots


@unittest.skipUnless(torch is not None, "torch is unavailable in the local lightweight test environment")
class AddressedReaderTests(unittest.TestCase):
    def make_reader(self):
        from core_bert.addressed_reader import AddressedWorldReader

        torch.manual_seed(4)
        return AddressedWorldReader(
            FakeBert(), node_count=2, split_layer=2, world_slot_count=2
        )

    def test_t2b_zero_editor_blocks_direct_and_relay_leakage(self):
        reader = self.make_reader()
        # [CLS] passage [DO] command; only command changes between inputs.
        first = torch.tensor([[1, 2, 3, 4]])
        second = torch.tensor([[1, 2, 3, 9]])
        valid = torch.ones_like(first)
        command = torch.tensor([[0, 0, 0, 1]], dtype=torch.bool)
        do = torch.tensor([[0, 0, 1, 0]], dtype=torch.bool)
        a = reader.forward_t2(first, valid, command, do, ZeroEditor(), mode="t2b")
        b = reader.forward_t2(second, valid, command, do, ZeroEditor(), mode="t2b")
        self.assertTrue(torch.allclose(a.edited_slots, b.edited_slots))
        self.assertTrue(torch.allclose(a.logits, b.logits))
        self.assertFalse(torch.allclose(a.instruction, b.instruction))

    def test_t2a_naive_leaks_command_to_slots(self):
        reader = self.make_reader()
        first = torch.tensor([[1, 2, 3, 4]])
        second = torch.tensor([[1, 2, 3, 9]])
        valid = torch.ones_like(first)
        command = torch.tensor([[0, 0, 0, 1]], dtype=torch.bool)
        do = torch.tensor([[0, 0, 1, 0]], dtype=torch.bool)
        a = reader.forward_t2(first, valid, command, do, ZeroEditor(), mode="t2a")
        b = reader.forward_t2(second, valid, command, do, ZeroEditor(), mode="t2a")
        self.assertFalse(torch.allclose(a.edited_slots, b.edited_slots))

    def test_t3a_and_t3b_are_wired_to_injected_editors(self):
        from core_bert.t3_pointer import T3AConditioning, T3BPointer

        reader = self.make_reader()
        ids = torch.tensor([[1, 2, 3, 4]])
        valid = torch.ones_like(ids)
        span = torch.tensor([[0, 1, 0, 0]], dtype=torch.bool)
        value = torch.ones(1, reader.hidden_size)
        t3a = reader.forward_t3(
            ids, valid, span, T3AConditioning(reader.hidden_size, 2),
            value_embedding=value, mode="t3a",
        )
        t3b = reader.forward_t3(
            ids, valid, span, T3BPointer(reader.hidden_size, 2),
            value_embedding=value, mode="t3b",
        )
        self.assertEqual(t3a.logits.shape, (1, 2, 2))
        self.assertEqual(t3b.pointer_probabilities.shape, (1, 2))
        self.assertTrue(torch.allclose(t3b.pointer_probabilities.sum(1), torch.ones(1)))


if __name__ == "__main__":
    unittest.main()
