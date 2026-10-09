import unittest

import torch
from torch import nn

from train_f2_baselines import LoRALinear


class F2BaselineTests(unittest.TestCase):
    def test_lora_starts_as_exact_identity_over_base(self):
        base = nn.Linear(7, 5)
        layer = LoRALinear(base, rank=3)
        inputs = torch.randn(4, 7)
        self.assertTrue(torch.equal(layer(inputs), base(inputs)))
        self.assertFalse(base.weight.requires_grad)
        self.assertEqual(sum(p.numel() for p in layer.parameters() if p.requires_grad), 36)

    def test_lora_rejects_nonpositive_rank(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            LoRALinear(nn.Linear(2, 2), rank=0)


if __name__ == "__main__":
    unittest.main()
