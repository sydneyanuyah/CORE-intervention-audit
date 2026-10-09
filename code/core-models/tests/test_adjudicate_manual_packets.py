import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
PATH = ROOT / "scripts" / "adjudicate_manual_packets.py"
SPEC = importlib.util.spec_from_file_location("adjudicate_manual_packets", PATH)
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


class AdjudicateManualPacketsTests(unittest.TestCase):
    def test_exact_wiqa_outcome_grounding(self):
        item = {"factual_question": "suppose x happens, how will it affect LESS rain.", "graph": {"node_text": {"A": ["MORE rain"], "D": ["LESS rain"]}}}
        self.assertEqual(module.queried_wiqa_node(item), "D")

    def test_event_index(self):
        self.assertEqual(module.event_index("event_04: outcome"), 4)


if __name__ == "__main__":
    unittest.main()
