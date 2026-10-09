import sys
import unittest
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from core_bert.reference_operators import (  # noqa: E402
    LearnedInterventionEncoder,
    SeparateTextEncoder,
    StateGatedReferenceEditor,
    compact_masked_tokens,
)


class ReferenceOperatorTests(unittest.TestCase):
    def test_t0_lookup_is_batched_and_range_checked(self):
        encoder = LearnedInterventionEncoder(60, 8)
        output = encoder(torch.tensor([0, 59]))
        self.assertEqual(output.shape, (2, 8))
        with self.assertRaisesRegex(ValueError, "outside"):
            encoder(torch.tensor([60]))

    def test_t1_compaction_removes_passage_position(self):
        ids = torch.tensor([[11, 12, 21, 22, 0], [31, 21, 22, 0, 0]])
        selected = torch.tensor([
            [False, False, True, True, False],
            [False, True, True, False, False],
        ])
        compact, mask = compact_masked_tokens(ids, selected, 0)
        self.assertTrue(torch.equal(compact, torch.tensor([[21, 22], [21, 22]])))
        self.assertTrue(mask.all())

    def test_t1_encoder_and_shared_reference_editor(self):
        encoder = SeparateTextEncoder(torch.randn(32, 8), 0, 8, max_length=4)
        text = encoder(torch.tensor([[3, 4, 0], [5, 6, 7]]),
                       torch.tensor([[1, 1, 0], [1, 1, 1]]))
        self.assertEqual(text.shape, (2, 8))
        editor = StateGatedReferenceEditor(8, rank=2)
        slots = torch.randn(2, 3, 8)
        # Zero-initialized residual projection makes a fresh reference editor a no-op.
        self.assertTrue(torch.equal(editor(slots, text), slots))


if __name__ == "__main__":
    unittest.main()
