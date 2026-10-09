import importlib.util
import pathlib
import unittest


PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "report_l2.py"
SPEC = importlib.util.spec_from_file_location("report_l2", PATH)
report_l2 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(report_l2)


class L2ReporterTests(unittest.TestCase):
    def test_constant_interval_collapses(self):
        self.assertEqual(report_l2.student_t_95([0.0] * 20), [0.0, 0.0])

    def test_extracts_selected_learned_checkpoint_metrics(self):
        summary = {
            "test_evaluated": False,
            "gpu_policy": {"model_size": "base", "world_size": 4},
            "l1": {"law_profile": "full"},
            "history": [
                {"selected": True, "validation": {"per_variable": {"variable_accuracy": 0.6, "groups": {"change": {"accuracy": 0.4}, "preservation": {"accuracy": 0.8}}}}},
                {"selected": False, "validation": {"per_variable": {"variable_accuracy": 0.1, "groups": {"change": {"accuracy": 0.1}, "preservation": {"accuracy": 0.1}}}}},
            ],
        }
        row = report_l2.extract_metrics(summary, "full")
        self.assertAlmostEqual(row["balanced_intervention_score"], 0.6)
        self.assertAlmostEqual(row["variable_accuracy"], 0.6)

    def test_rejects_test_access(self):
        with self.assertRaisesRegex(ValueError, "validation-only"):
            report_l2.extract_metrics({"test_evaluated": True}, "full")


if __name__ == "__main__":
    unittest.main()
