import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None, "torch is unavailable in the local lightweight test environment")
class T2InlineTests(unittest.TestCase):
    def setUp(self):
        from core_bert.t2_inline import (
            allowed_to_additive,
            assert_zero_editor_no_leakage,
            build_t2a_masks,
            build_t2b_masks,
            pool_t2_instruction,
            zero_editor_leakage_metrics,
        )
        self.allowed_to_additive = allowed_to_additive
        self.assert_zero_editor_no_leakage = assert_zero_editor_no_leakage
        self.build_t2a_masks = build_t2a_masks
        self.build_t2b_masks = build_t2b_masks
        self.pool_t2_instruction = pool_t2_instruction
        self.zero_editor_leakage_metrics = zero_editor_leakage_metrics
        # [CLS] passage passage [DO] command command slot slot padding
        self.valid = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0]], dtype=torch.bool)
        self.do = torch.tensor([[0, 0, 0, 1, 0, 0, 0, 0, 0]], dtype=torch.bool)
        self.command = torch.tensor([[0, 0, 0, 0, 1, 1, 0, 0, 0]], dtype=torch.bool)
        self.slots = torch.tensor([[0, 0, 0, 0, 0, 0, 1, 1, 0]], dtype=torch.bool)

    def test_naive_t2a_keeps_command_visible(self):
        masks = self.build_t2a_masks(self.valid, self.command, self.do)
        self.assertTrue(torch.equal(masks.pre_edit, masks.post_edit))
        self.assertTrue(masks.pre_edit[0, 6, 4])

    def test_encode_only_attention_rules(self):
        masks = self.build_t2b_masks(self.valid, self.command, self.do, self.slots)
        self.assertFalse(masks.pre_edit[0, 6, 3:6].any())
        self.assertTrue(masks.pre_edit[0, 4, 1:3].all())
        self.assertFalse(masks.post_edit[0, :, 3:6].any())
        self.assertFalse(masks.pre_edit[0, :, 8].any())
        additive = self.allowed_to_additive(masks.pre_edit, torch.float32)
        self.assertEqual(additive.shape, (1, 1, 9, 9))
        self.assertLess(additive[0, 0, 6, 4].item(), -1e20)

    def test_mean_and_last_pooling(self):
        hidden = torch.arange(9 * 2, dtype=torch.float32).reshape(1, 9, 2)
        mean = self.pool_t2_instruction(hidden, self.command, self.do, "mean")
        last = self.pool_t2_instruction(hidden, self.command, self.do, "last")
        self.assertTrue(torch.equal(mean, torch.cat([(hidden[:, 4] + hidden[:, 5]) / 2, hidden[:, 3]], 1)))
        self.assertTrue(torch.equal(last, torch.cat([hidden[:, 5], hidden[:, 3]], 1)))

    def test_zero_editor_leakage_check(self):
        baseline = torch.tensor([[1.0, 2.0], [3.0, 1.0]])
        self.assertTrue(self.zero_editor_leakage_metrics(baseline, baseline.clone())["passed"])
        self.assert_zero_editor_no_leakage(baseline, baseline.clone())
        with self.assertRaises(AssertionError):
            self.assert_zero_editor_no_leakage(baseline, baseline + 0.1)


if __name__ == "__main__":
    unittest.main()
