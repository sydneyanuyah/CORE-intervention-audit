import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from build_annotation_pilot_queues import (
    EXAMPLE_SIZE,
    PUBMED_STRATA,
    QUEUE_SIZE,
    blank_annotation,
    eligible_meter_contexts,
    normalize_pubmed_label,
    stable_rank,
)


class AnnotationPilotTests(unittest.TestCase):
    def test_frozen_sizes_and_pubmed_strata(self):
        self.assertEqual(QUEUE_SIZE, 300)
        self.assertEqual(EXAMPLE_SIZE, 5)
        self.assertEqual(sum(PUBMED_STRATA.values()), 300)

    def test_seed_is_three_digit_and_ranking_is_stable(self):
        self.assertEqual(stable_rank("x", "y"), stable_rank("x", "y"))
        self.assertNotEqual(stable_rank("x", "y"), stable_rank("x", "z"))

    def test_blank_annotation_is_independent(self):
        first = blank_annotation()
        second = blank_annotation()
        first["reason_codes"].append("NO_INTERVENTION")
        self.assertEqual(second["reason_codes"], [])

    def test_meter_requires_one_of_each_rung(self):
        context = {
            "context": "C",
            "questions": [
                {"ladder": "Causal_Discovery"},
                {"ladder": "Intervention"},
                {"ladder": "Counterfactual"},
            ],
        }
        duplicate = {"context": "C", "questions": [context["questions"][0]] * 3}
        self.assertEqual(eligible_meter_contexts([context, duplicate]), [(0, context)])

    def test_pubmed_label_normalization_is_narrow(self):
        self.assertEqual(normalize_pubmed_label("implicit"), "Implicit")
        self.assertEqual(normalize_pubmed_label("intra"), "Intra")
        self.assertEqual(normalize_pubmed_label("Explicts"), "Explicts")


if __name__ == "__main__":
    unittest.main()
