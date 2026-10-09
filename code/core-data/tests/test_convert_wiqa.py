import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from convert_wiqa import (  # noqa: E402
    TARGET_MARKER,
    append_target_surface,
    graph_components,
    graph_node_text,
    normalize_item,
    normalized_grounding,
    resolve_question,
    signed_paths,
)
from schema import validate_record  # noqa: E402


GRAPH = {
    "graph_id": "1",
    "paragraph": "A small process with less input.",
    "V": ["less input"],
    "Z": ["more input"],
    "X": "more intermediate",
    "U": [],
    "W": ["less result"],
    "Y": "more result",
    "para_outcome_accelerate": "MORE output",
    "para_outcome_decelerate": "LESS output",
    "Y_affects_outcome": "more",
}
ITEM = {
    "question": {
        "stem": "suppose less input happens, how will it affect LESS output.",
        "answer_label": "more",
    },
    "metadata": {
        "ques_id": "fixture:1",
        "graph_id": "1",
        "question_type": "EXOGENOUS_EFFECT",
        "path_len": 4,
    },
}


class WIQAConverterTests(unittest.TestCase):
    def test_source_normalization_matches_apostrophe_variants(self):
        self.assertEqual(normalized_grounding("doesn't"), normalized_grounding("doesnt"))

    def test_signed_path_and_resolution(self):
        _, edges, _ = graph_components(GRAPH)
        self.assertIn((4, 1), signed_paths("V", "D", edges))
        self.assertEqual(resolve_question(ITEM, GRAPH), ("V", "D"))

    def test_normalized_record_validates(self):
        record = normalize_item(ITEM, GRAPH, "train", "fixture.jsonl", "fixture.jsonl", "0" * 64)
        self.assertEqual(validate_record(record), [])
        self.assertEqual(record["factual"]["state"], None)
        self.assertEqual(record["intervened"]["answer"], "more")
        intervention = record["intervention"]
        start, end = intervention["target_span"]
        self.assertEqual(record["factual"]["passage"][start:end], intervention["target_text"])
        self.assertEqual(intervention["value_token"], "less")
        self.assertTrue(record["factual"]["passage"].startswith(GRAPH["paragraph"]))
        self.assertEqual(record["graph"]["node_text"]["V"], ["less input"])
        self.assertEqual(graph_node_text(GRAPH, ["A"]), {"A": ["MORE output"]})
        self.assertEqual(
            record["factual"]["passage"][len(GRAPH["paragraph"]) :],
            TARGET_MARKER + "less input",
        )

    def test_augmentation_preserves_original_paragraph_and_points_to_suffix(self):
        original = "A paragraph with trailing whitespace.  "
        augmented, span = append_target_surface(original, "resolved source")
        self.assertEqual(augmented[: len(original)], original)
        self.assertEqual(augmented[len(original) :], TARGET_MARKER + "resolved source")
        self.assertEqual(augmented[span[0] : span[1]], "resolved source")


if __name__ == "__main__":
    unittest.main()
