import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
PATH = ROOT / "scripts" / "validate_manual_annotations.py"
SPEC = importlib.util.spec_from_file_location("manual_annotations", PATH)
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def annotation(name, answer="more", source="wiqa"):
    return {
        "pair_id": "p1", "annotator_id": name, "source": source, "split": "validation",
        "intermediate_outputs": {"X": "more", "Y": "less"},
        "gold_final_outputs": {"X": "more", "Y": answer},
        "query_answer_after_both_edits": answer, "target_variables": ["X", "V2"],
        "actually_changed_variables": ["X"], "preserved_variables": ["Y"],
        "first_edit_valid": True, "second_edit_valid": True, "status": "accepted",
        "confidence": "high", "justification": "grounded",
    }


class ManualAnnotationTests(unittest.TestCase):
    def write(self, path, row):
        path.write_text(json.dumps(row) + "\n")

    def test_validates_and_compares(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); queue = root / "q"; one = root / "one"; two = root / "two"
            self.write(queue, {"pair_id": "p1", "source": "wiqa", "split": "validation", "graph": {"nodes": ["X", "Y"]}})
            self.write(one, annotation("Annotation 1")); self.write(two, annotation("Annotation 2"))
            a = module.validate(queue, one, "Annotation 1")
            b = module.validate(queue, two, "Annotation 2")
            self.assertEqual(module.compare(a, b)["exact_agreement"], 1)

    def test_disagreement_routes_to_adjudication(self):
        result = module.compare({"p1": annotation("Annotation 1")}, {"p1": annotation("Annotation 2", "less")})
        self.assertEqual(result["needs_adjudication"], 1)

    def test_com2_intermediate_wording_is_not_primary_disagreement(self):
        first = annotation("Annotation 1", "end", "com2")
        second = annotation("Annotation 2", "end", "com2")
        first["intermediate_outputs"] = {"event_00": "calm wording"}
        second["intermediate_outputs"] = {"event_00": "different wording"}
        first["gold_final_outputs"] = second["gold_final_outputs"] = {"event_00": "end"}
        first["actually_changed_variables"] = second["actually_changed_variables"] = []
        first["preserved_variables"] = second["preserved_variables"] = ["event_00"]
        first["target_variables"] = second["target_variables"] = ["event_00"]
        result = module.compare({"p1": first}, {"p1": second})
        self.assertEqual(result["exact_agreement"], 1)
        self.assertEqual(result["field_exact_agreement"]["intermediate_outputs"], 0)

    def test_rejects_test_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); queue = root / "q"; one = root / "one"
            self.write(queue, {"pair_id": "p1", "source": "wiqa", "split": "test"})
            row = annotation("Annotation 1"); row["split"] = "test"; self.write(one, row)
            with self.assertRaisesRegex(ValueError, "validation-only"):
                module.validate(queue, one, "Annotation 1")


if __name__ == "__main__":
    unittest.main()
