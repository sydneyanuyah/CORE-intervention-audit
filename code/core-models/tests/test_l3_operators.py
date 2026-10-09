import unittest

import torch
from torch import nn

from core_bert.l3_operators import LoReFTOperator, RouterOperator, TaskVectorAddOperator


class _O1(nn.Module):
    def forward(self, hidden, intervention_id): return hidden + 1


class _O2(nn.Module):
    def forward(self, hidden, intervention_id): return hidden + 2


class _O3(nn.Module):
    def forward(self, hidden, intervention_id): return hidden + 3


class _Core:
    O1FixedVector = staticmethod(lambda *args: _O1())
    O2ConditionalLowRank = staticmethod(lambda *args: _O2())
    O3StateGated = staticmethod(lambda *args: _O3())


class L3OperatorTests(unittest.TestCase):
    def test_zero_initialized_residuals_are_identity(self):
        hidden = torch.randn(2, 4, 8); commands = torch.tensor([1, 2])
        for operator in (LoReFTOperator(6, 8, 3), TaskVectorAddOperator(6, 8)):
            self.assertTrue(torch.equal(operator(hidden, commands), hidden))

    def test_router_is_convex_mixture_of_registered_experts(self):
        operator = RouterOperator(_Core(), 6, 4, 8, 3)
        with torch.no_grad():
            operator.router.weight.zero_(); operator.router.bias.zero_()
        hidden = torch.zeros(2, 4, 8)
        result = operator(hidden, torch.tensor([1, 2]))
        self.assertTrue(torch.allclose(result, torch.full_like(result, 2.0)))


if __name__ == "__main__":
    unittest.main()
