import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
PATH = ROOT / "scripts" / "report_a1_confirmatory.py"
SPEC = importlib.util.spec_from_file_location("a1_confirmatory_report", PATH)
reporter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = reporter
SPEC.loader.exec_module(reporter)


class A1ConfirmatoryReportTests(unittest.TestCase):
    def test_student_interval_collapses_for_identical_values(self):
        self.assertEqual(reporter._student_t_95([0.5] * 20), [0.5, 0.5])

    def test_student_interval_requires_twenty_seeds(self):
        with self.assertRaisesRegex(ValueError, "twenty seeds"):
            reporter._student_t_95([0.5] * 5)

    def test_paired_graph_bootstrap_collapses_for_constant_delta(self):
        arm = {
            seed: {f"g{graph}": 0.6 for graph in range(20)}
            for seed in range(20)
        }
        baseline = {
            seed: {f"g{graph}": 0.5 for graph in range(20)}
            for seed in range(20)
        }
        interval = reporter._paired_graph_bootstrap_95(arm, baseline, draws=100)
        self.assertAlmostEqual(interval[0], 0.1)
        self.assertAlmostEqual(interval[1], 0.1)

    @unittest.skipUnless(
        (ROOT / "outputs" / "a1-confirmatory").exists(),
        "confirmatory outputs live on small-model cluster",
    )
    def test_complete_confirmatory_report(self):
        report = reporter.aggregate(ROOT, ROOT / "registry" / "a1_confirmatory_manifest.json")
        self.assertEqual(report["registered_cells"], 520)
        self.assertEqual(report["completed_cells"], 520)
        self.assertEqual(report["frozen_selected_arm"], "t3b_pointer")
        self.assertTrue(report["confirmatory"])
        self.assertFalse(report["test_evaluated"])
        self.assertEqual(set(report["families"]), {"xor", "ccrgb", "cladder", "wiqa", "com2"})


if __name__ == "__main__":
    unittest.main()
