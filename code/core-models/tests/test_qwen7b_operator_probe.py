import unittest

import torch

from core_7b.operator_probe import O2ConditionalLowRank, O3StateGated, intervention_id, paired_metrics


class Qwen7BOperatorProbeTests(unittest.TestCase):
    def test_operator_shapes_and_zero_initial_update(self):
        hidden = torch.randn(3, 12); ids = torch.tensor([0, 3, 7])
        for operator in (O2ConditionalLowRank(8, 12, 4), O3StateGated(8, 12, 4)):
            self.assertEqual(operator(hidden, ids).shape, hidden.shape)
            self.assertTrue(torch.equal(operator(hidden, ids), hidden))

    def test_inversion_changes_only_value_bit(self):
        record = {"intervention": {"target": "X", "value": 0}}
        nodes = ["V1", "V2", "X", "Y"]
        self.assertEqual(intervention_id(record, nodes), 4)
        self.assertEqual(intervention_id(record, nodes, invert=True), 5)

    def test_balanced_pair_metric(self):
        pair = {"pair_id": "p", "gold_outputs": [1, 0], "change_mask": [True, False], "preservation_mask": [False, True]}
        self.assertEqual(paired_metrics([pair], {"p": [1, 0]})["two_edit_balanced"], 1.0)


if __name__ == "__main__": unittest.main()

