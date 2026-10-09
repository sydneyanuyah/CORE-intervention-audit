import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
PATH = ROOT / "scripts" / "build_manual_annotation_queue.py"
SPEC = importlib.util.spec_from_file_location("manual_queue", PATH)
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


class ManualQueueTests(unittest.TestCase):
    def test_wiqa_second_edit_is_distinct_and_label_blinded(self):
        row = {
            "id": "w", "descendants": ["Y"],
            "graph": {"nodes": ["X", "Y"], "edges": [["X", "Y"]], "node_text": {"X": ["more x"], "Y": ["less y"]}},
            "factual": {"passage": "p", "question": "q"},
            "intervention": {"target": "X", "target_text": "more x", "value": "more x", "value_token": "more", "kind": "value_set", "formal": "do(X)", "text": "first"},
            "intervened": {"answer": "less"}, "probes": [{"answer_after": "less"}],
        }
        result = module.wiqa_candidate(row)
        self.assertEqual(result["second_edit"]["target"], "Y")
        self.assertEqual(result["second_edit"]["value_token"], "more")
        self.assertNotIn("intervened", result)
        self.assertNotIn("probes", result)

    def test_signed_edges_are_exposed_without_labels(self):
        row = {
            "id": "w", "descendants": ["Y"],
            "graph": {"nodes": ["X", "Y"], "edges": [["X", "Y"]], "node_text": {"X": ["more x"], "Y": ["more y"]}},
            "factual": {"passage": "p", "question": "q"},
            "intervention": {"target": "X", "target_text": "more x", "value": "more x", "value_token": "more", "kind": "value_set", "formal": "do(X)", "text": "first"},
        }
        edges = [{"source": "X", "target": "Y", "sign": "negative"}]
        result = module.wiqa_candidate(row, edges)
        self.assertEqual(result["graph"]["signed_edges"], edges)
        self.assertNotIn("gold_final_outputs", result)

    def test_com2_second_edit_uses_factual_next_step(self):
        row = {
            "id": "c", "chain": ["a", "b", "c"],
            "factual": {"passage": "story\n[ORIG] b\n[REPL] x", "question": "q"},
            "intervention": {"target": "event_01: b", "target_text": "b", "value": "x", "value_token": None, "kind": "event_replace", "formal": "do", "text": "first"},
        }
        result = module.com2_candidate(row)
        self.assertEqual(result["second_edit"]["target"], "event_02: c")
        self.assertEqual(result["factual_passage"], "story")


if __name__ == "__main__":
    unittest.main()
