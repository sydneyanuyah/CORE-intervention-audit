import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))
from generate_cladder_two_edit import build_truth_descriptor, build_world, ordered_pair_id  # noqa: E402


class CladderTwoEditTests(unittest.TestCase):
    def test_order_and_last_write_are_recomputed(self):
        model = {
            "model_id": 7, "structure": "X->Y,Z->W", "equation_type": "deterministic",
            "params": {"p(X)": 1, "p(Z)": 0, "p(Y | X)": [0, 1], "p(W | Z)": [0, 1]},
            "variable_mapping": {"Xname": "switch", "X0": "switch off", "X1": "switch on", "Yname": "lamp", "Y0": "lamp off", "Y1": "lamp on", "Zname": "valve", "Z0": "valve off", "Z1": "valve on", "Wname": "pump", "W0": "pump off", "W1": "pump on"},
        }
        question = {"question_id": 1, "meta": {"given_info": {"Z": 0}, "treatment": "X", "action": 0}}
        base = {
            "id": "base", "source_id": 1,
            "graph": {"nodes": ["W", "X", "Y", "Z"], "edges": [["X", "Y"], ["Z", "W"]]},
            "factual": {"passage": "The switch controls the lamp, and the valve controls the pump.", "state": {"W": 0, "X": 1, "Y": 1, "Z": 0}},
        }
        components, rows = build_world(base, question, model, "validation")
        by_id = {row["id"]: row for row in components}
        self.assertTrue(all(row["first_record_id"] != row["second_record_id"] for row in rows))
        self.assertTrue(all(any(row["change_mask"]) and any(row["preservation_mask"]) for row in rows))
        match = next(row for row in rows if by_id[row["first_record_id"]]["intervention"]["target"] == "X" and by_id[row["first_record_id"]]["intervention"]["value"] == 1 and by_id[row["second_record_id"]]["intervention"]["target"] == "X" and by_id[row["second_record_id"]]["intervention"]["value"] == 0)
        self.assertEqual(match["gold_outputs"], [0, 0, 0, 0])
        self.assertEqual(match["target_mask"], [False, True, False, False])
        self.assertEqual(match["pair_id"], ordered_pair_id(
            "cladder", match["graph_group_id"], match["world_group_id"],
            match["first_record_id"], match["second_record_id"],
        ))

        descriptor = build_truth_descriptor(base, question, model, "validation")
        self.assertEqual(descriptor["factual_roots"], {"X": 1, "Z": 0})
        self.assertEqual(descriptor["model"]["params"], model["params"])
        self.assertEqual(descriptor["world_group_id"], "cladder:model:7:world:question:1")


if __name__ == "__main__":
    unittest.main()
