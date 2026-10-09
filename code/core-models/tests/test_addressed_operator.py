import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None, "torch is unavailable")
class AddressedOperatorTests(unittest.TestCase):
    def setUp(self):
        from core_bert.addressed_operator import (
            AddressedGatedOperator,
            IdentityAddressedOperator,
            gated_edit_parameter_count,
        )

        self.Operator = AddressedGatedOperator
        self.Identity = IdentityAddressedOperator
        self.count = gated_edit_parameter_count

    def test_registered_dimensions_and_exact_parameter_count(self):
        operator = self.Operator(hidden_size=8, rank=16)
        slots = torch.randn(3, 5, 8)
        instruction = torch.randn(3, 16)
        self.assertEqual(operator(slots, instruction).shape, slots.shape)
        expected = 3 * 8 * 8 + 4 * 8 * 16 + 8
        self.assertEqual(operator.parameter_count, expected)
        self.assertEqual(self.count(8, 16), expected)
        self.assertEqual(
            sum(parameter.numel() for parameter in operator.parameters()), expected
        )

    def test_zero_initialization_is_exact_noop_and_backward_is_finite(self):
        torch.manual_seed(7)
        operator = self.Operator(hidden_size=6, rank=3)
        slots = torch.randn(2, 4, 6, requires_grad=True)
        instruction = torch.randn(2, 12, requires_grad=True)
        output = operator(slots, instruction)
        self.assertTrue(torch.equal(output, slots))
        output.square().mean().backward()
        tensors = [slots.grad, instruction.grad]
        tensors.extend(parameter.grad for parameter in operator.parameters())
        self.assertTrue(all(gradient is not None for gradient in tensors))
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in tensors))

    def test_learned_forward_backward_matches_t2_and_t3a_call_contract(self):
        torch.manual_seed(11)
        operator = self.Operator(hidden_size=5, rank=2)
        with torch.no_grad():
            operator.up.weight.normal_(std=0.1)
        slots = torch.randn(2, 3, 5, requires_grad=True)
        instruction = torch.randn(2, 10, requires_grad=True)
        # AddressedWorldReader.forward_t2 and its T3-a path both make this
        # exact two-argument call; pointer weighting remains outside this module.
        edited = operator(slots, instruction)
        self.assertFalse(torch.equal(edited, slots))
        edited.sum().backward()
        self.assertTrue(torch.isfinite(slots.grad).all())
        self.assertTrue(torch.isfinite(instruction.grad).all())
        self.assertTrue(
            all(
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in operator.parameters()
            )
        )

    def test_identity_modes_are_parameter_counted_noops(self):
        slots = torch.randn(2, 4, 3)
        instruction = torch.randn(2, 6)
        operator = self.Operator(3, rank=2)
        with torch.no_grad():
            operator.up.weight.fill_(1.0)
        operator.set_identity_mode(True)
        self.assertTrue(torch.equal(operator(slots, instruction), slots))
        operator.set_identity_mode(False)
        self.assertFalse(torch.equal(operator(slots, instruction), slots))

        identity = self.Identity()
        self.assertEqual(identity.parameter_count, 0)
        self.assertEqual(sum(p.numel() for p in identity.parameters()), 0)
        self.assertTrue(torch.equal(identity(slots, instruction), slots))

    def test_invalid_shapes_and_configuration_fail_early(self):
        with self.assertRaises(ValueError):
            self.Operator(0)
        operator = self.Operator(4, rank=2)
        with self.assertRaisesRegex(ValueError, "slots must"):
            operator(torch.randn(2, 4), torch.randn(2, 8))
        with self.assertRaisesRegex(ValueError, "instruction shape"):
            operator(torch.randn(2, 3, 4), torch.randn(2, 7))
        with self.assertRaisesRegex(ValueError, "slot hidden"):
            operator(torch.randn(2, 3, 5), torch.randn(2, 8))


if __name__ == "__main__":
    unittest.main()
