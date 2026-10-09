import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
PATH = ROOT / "scripts" / "report_a2_measurement.py"
SPEC = importlib.util.spec_from_file_location("a2_measurement_report", PATH)
reporter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = reporter
SPEC.loader.exec_module(reporter)


class A2MeasurementReportTests(unittest.TestCase):
    def test_student_interval_collapses_for_identical_values(self):
        self.assertEqual(reporter.student_t_95([0.5] * 20), [0.5, 0.5])

    def test_student_interval_requires_twenty_seeds(self):
        with self.assertRaisesRegex(ValueError, "twenty paired seeds"):
            reporter.student_t_95([0.5] * 5)

    def test_paired_graph_bootstrap_collapses_for_constant_delta(self):
        arm = {seed: {f"g{graph}": 0.6 for graph in range(3)} for seed in range(20)}
        baseline = {seed: {f"g{graph}": 0.5 for graph in range(3)} for seed in range(20)}
        interval = reporter.paired_graph_bootstrap_95(arm, baseline, bootstrap_seed=7)
        self.assertAlmostEqual(interval[0], 0.1)
        self.assertAlmostEqual(interval[1], 0.1)

    def test_paired_graph_bootstrap_rejects_identity_mismatch(self):
        arm = {seed: {"g1": 0.6} for seed in range(20)}
        baseline = {seed: {"g1": 0.5} for seed in range(20)}
        baseline[0] = {"g2": 0.5}
        with self.assertRaisesRegex(ValueError, "different graph identities"):
            reporter.paired_graph_bootstrap_95(arm, baseline, bootstrap_seed=7)

    @unittest.skipUnless(
        (ROOT / "outputs" / "a2-measurement").exists(),
        "A2 outputs live on small-model cluster",
    )
    def test_complete_a2_report(self):
        report = reporter.aggregate(ROOT, ROOT / "registry" / "a2_measurement_manifest.json")
        self.assertEqual(report["registered_cells"], 400)
        self.assertEqual(report["completed_cells"], 400)
        self.assertFalse(report["test_evaluated"])
        self.assertEqual(set(report["families"]), {"xor", "ccrgb", "cladder", "wiqa", "com2"})


if __name__ == "__main__":
    unittest.main()
