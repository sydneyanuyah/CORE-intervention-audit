import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
PATH = ROOT / "scripts" / "report_a1_screening.py"
SPEC = importlib.util.spec_from_file_location("a1_screening_report", PATH)
reporter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = reporter
SPEC.loader.exec_module(reporter)


class A1ScreeningReportTests(unittest.TestCase):
    def test_student_interval_collapses_for_identical_values(self):
        self.assertEqual(reporter._student_t_95([0.5] * 5), [0.5, 0.5])

    def test_student_interval_requires_five_seeds(self):
        with self.assertRaisesRegex(ValueError, "five seeds"):
            reporter._student_t_95([0.5] * 4)

    @unittest.skipUnless((ROOT / "outputs" / "a1-screen").exists(), "screening outputs live on small-model cluster")
    def test_frozen_complete_report(self):
        report = reporter.aggregate(ROOT, ROOT / "registry" / "a1_screening_manifest.json")
        self.assertEqual(report["registered_cells"], 130)
        self.assertEqual(report["completed_cells"], 130)
        self.assertEqual(report["selected_arm"], "t3b_pointer")
        self.assertFalse(report["confirmatory"])
        self.assertFalse(report["test_evaluated"])
        self.assertEqual(set(report["families"]), {"xor", "ccrgb", "cladder", "wiqa", "com2"})


if __name__ == "__main__":
    unittest.main()
