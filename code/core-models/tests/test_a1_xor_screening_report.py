import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
PATH = ROOT / "scripts" / "report_a1_xor_screening.py"
SPEC = importlib.util.spec_from_file_location("a1_xor_screening_report", PATH)
reporter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = reporter
SPEC.loader.exec_module(reporter)


class A1XorScreeningReportTests(unittest.TestCase):
    def test_student_interval_collapses_for_identical_values(self):
        self.assertEqual(reporter.student_t_95([0.0] * 5), [0.0, 0.0])

    def test_student_interval_contains_sample_mean(self):
        values = [-0.02, -0.01, 0.0, 0.01, 0.02]
        lower, upper = reporter.student_t_95(values)
        self.assertLess(lower, 0.0)
        self.assertGreater(upper, 0.0)

    def test_graph_bootstrap_requires_registered_shape(self):
        with self.assertRaisesRegex(ValueError, "20 graphs"):
            reporter.paired_graph_bootstrap_95({"one": [0.0] * 5})

    def test_graph_bootstrap_is_deterministic(self):
        values = {f"g{i}": [i / 1000] * 5 for i in range(20)}
        first = reporter.paired_graph_bootstrap_95(values, replicates=100)
        second = reporter.paired_graph_bootstrap_95(values, replicates=100)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
