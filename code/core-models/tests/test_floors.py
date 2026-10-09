import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))


class FloorTests(unittest.TestCase):
    def test_do_everything(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is unavailable in the local lightweight test environment")
        from core_bert.floors import do_everything
        factual = torch.tensor([[0, 1, 0], [1, 0, 1], [1, 0, 0]])
        result = do_everything(factual, torch.tensor([0, 3, 6]), 3)
        self.assertTrue(torch.equal(result, torch.tensor([[0, 0, 0], [1, 1, 1], [1, 0, 0]])))
