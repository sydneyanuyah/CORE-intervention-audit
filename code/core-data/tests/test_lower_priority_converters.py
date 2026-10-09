import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from convert_counterbench import rejection_reason as counterbench_reason
from convert_pubmedcausal import long_candidates


class LowerPriorityConverterTests(unittest.TestCase):
    def test_counterbench_multiple_action_gate(self):
        self.assertIn("multiple simultaneous", counterbench_reason({"type": "nested"}))
        self.assertIn("factual outcome", counterbench_reason({"type": "basic"}))

    def test_pubmed_wide_relation_is_reshaped(self):
        row = {
            "s/n": 7,
            "Sentence": "A causes B.",
            "Cause 1": "A",
            "Effect 1": "B",
            "Sententiality 1": "Intra",
            "Causality 1": "Explicit",
        }
        values = long_candidates(row, "train")
        self.assertEqual(len(values), 1)
        self.assertEqual(values[0]["cause"], "A")
        self.assertEqual(values[0]["effect"], "B")

    def test_pubmed_negative_row_remains_one_candidate(self):
        values = long_candidates({"s/n": 8, "Sentence": "No relation."}, "test")
        self.assertEqual(len(values), 1)
        self.assertIsNone(values[0]["relation_index"])


if __name__ == "__main__":
    unittest.main()
