import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from core_bert.registry import load_and_validate
from core_bert.statistics import mcnemar_counts, paired_intervals


class RegistryTests(unittest.TestCase):
    def test_complete_scope(self):
        scope = load_and_validate(Path(__file__).parents[1] / "registry" / "bert_scope.json")
        self.assertEqual(len(scope["experiments"]), 23)

    def test_t2_t3_design_is_explicit_and_seed_stages_are_not_conflated(self):
        scope = load_and_validate(Path(__file__).parents[1] / "registry" / "bert_scope.json")
        experiments = {item["id"]: item for item in scope["experiments"]}
        a1 = experiments["A1"]
        self.assertEqual(
            a1["methods"],
            [
                "t0_id",
                "t1_text",
                "t2b_encode_only",
                "t2a_naive",
                "t3b_pointer",
                "t3a_conditioning",
            ],
        )
        self.assertEqual(a1["seeds"], 20)
        self.assertEqual(a1["stages"]["screening"]["seeds_per_family"], 5)
        self.assertEqual(a1["stages"]["confirmatory"]["seeds_per_family"], 20)
        self.assertEqual(a1["stages"]["screening"]["applicable_runs_per_reader"], 130)
        self.assertEqual(a1["stages"]["confirmatory"]["applicable_runs_per_reader"], 520)
        self.assertEqual(a1["applicability"]["t0_id"], ["xor"])

    def test_a2_a3_and_c4_use_revised_arm_names_and_accounting(self):
        scope = load_and_validate(Path(__file__).parents[1] / "registry" / "bert_scope.json")
        experiments = {item["id"]: item for item in scope["experiments"]}
        self.assertEqual(
            experiments["A2"]["methods"],
            [
                "t2b_editor_active",
                "t2b_editor_zeroed",
                "no_instruction_baseline",
                "prompting_baseline",
            ],
        )
        self.assertEqual(
            experiments["A3"]["methods"],
            ["t3b_correct_span", "t3b_adjacent_span", "t3b_random_span"],
        )
        accounting = experiments["C4"]["t3b_accounting"]
        self.assertIn("pointer_query_projection", accounting["include"])
        self.assertIn("pointer_key_projection", accounting["include"])
        self.assertEqual(
            accounting["matched_lora_rule"],
            "recompute_rank_against_total_t3b_trainable_parameters",
        )

    def test_paired_statistics(self):
        result = paired_intervals([0.1, 0.2, 0.3], [0.2, 0.3, 0.4], draws=1000)
        self.assertAlmostEqual(result["mean_difference"], 0.1)

    def test_mcnemar_pairing(self):
        self.assertEqual(mcnemar_counts([True, False], [False, True]), {"baseline_only": 1, "method_only": 1})
