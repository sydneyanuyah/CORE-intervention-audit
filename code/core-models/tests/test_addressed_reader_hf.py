import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

try:
    import torch
    from torch import nn
    from transformers import BertConfig, BertModel
except ImportError:
    torch = None
    nn = None
    BertConfig = None
    BertModel = None


if torch is not None:
    class ZeroEditor(nn.Module):
        def forward(self, slots, instruction):
            return slots


@unittest.skipUnless(
    torch is not None and BertModel is not None,
    "torch/transformers are unavailable in the local lightweight test environment",
)
class HuggingFaceAddressedReaderTests(unittest.TestCase):
    def test_tiny_bert_accepts_layerwise_3d_masks(self):
        from core_bert.addressed_reader import AddressedWorldReader

        torch.manual_seed(11)
        config = BertConfig(
            vocab_size=32,
            hidden_size=16,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=32,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            max_position_embeddings=32,
        )
        bert = BertModel(config, add_pooling_layer=False)
        reader = AddressedWorldReader(
            bert, node_count=2, split_layer=2, world_slot_count=2
        ).eval()
        first = torch.tensor([[1, 2, 3, 4]])
        second = torch.tensor([[1, 2, 3, 9]])
        valid = torch.ones_like(first)
        command = torch.tensor([[0, 0, 0, 1]], dtype=torch.bool)
        do = torch.tensor([[0, 0, 1, 0]], dtype=torch.bool)
        with torch.no_grad():
            a = reader.forward_t2(first, valid, command, do, ZeroEditor(), mode="t2b")
            b = reader.forward_t2(second, valid, command, do, ZeroEditor(), mode="t2b")
        self.assertEqual(a.logits.shape, (1, 2, 2))
        self.assertTrue(torch.allclose(a.edited_slots, b.edited_slots, atol=1e-6))
        self.assertTrue(torch.allclose(a.logits, b.logits, atol=1e-6))
        self.assertFalse(torch.allclose(a.instruction, b.instruction))


if __name__ == "__main__":
    unittest.main()
