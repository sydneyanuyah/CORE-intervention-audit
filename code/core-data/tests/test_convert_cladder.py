import sys
import unittest
import copy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from convert_cladder import descendants_of, normalize_item, solve_state  # noqa: E402
from schema import validate_record  # noqa: E402


MODEL = {
    "model_id": 9,
    "equation_type": "deterministic",
    "structure": "X->V2,X->Y,V2->Y",
    "params": {
        "p(X)": 0.5,
        "p(V2 | X)": [1, 0],
        "p(Y | X, V2)": [[0, 1], [1, 1]],
    },
    "variable_mapping": {"Xname": "X", "X1": "X", "X0": "X"},
}

ITEM = {
    "question_id": 42,
    "given_info": "X causes not V2. X or V2 causes Y.",
    "question": "Would Y be false if X were false instead of true?",
    "answer": "no",
    "meta": {
        "query_type": "det-counterfactual",
        "model_id": 9,
        "treatment": "X",
        "outcome": "Y",
        "action": 0,
        "polarity": 0,
        "groundtruth": 0,
        "given_info": {},
    },
}


class CLadderConverterTests(unittest.TestCase):
    def test_deterministic_table_solver(self):
        self.assertEqual(solve_state(MODEL, {"X": 0}), {"V2": 1, "X": 0, "Y": 1})
        self.assertEqual(solve_state(MODEL, {"X": 1}), {"V2": 0, "X": 1, "Y": 1})

    def test_graph_reachability(self):
        edges = [["X", "V2"], ["X", "Y"], ["V2", "Y"]]
        self.assertEqual(descendants_of("X", edges), ["V2", "Y"])

    def test_endogenous_treatment_override(self):
        model = {
            "structure": "V1->X,V1->Y,X->Y",
            "params": {
                "p(V1)": 0.5,
                "p(X | V1)": [0, 1],
                "p(Y | V1, X)": [[0, 1], [1, 1]],
            },
        }
        self.assertEqual(
            solve_state(model, {"V1": 0, "X": 1}),
            {"V1": 0, "X": 1, "Y": 1},
        )

    def test_normalized_record_validates_and_matches_oracle(self):
        record = normalize_item(ITEM, MODEL)
        self.assertEqual(validate_record(record), [])
        self.assertEqual(record["factual"]["state"], {"V2": 0, "X": 1, "Y": 1})
        self.assertEqual(record["intervened"]["state"], {"V2": 1, "X": 0, "Y": 1})
        self.assertEqual(record["intervened"]["answer"], "no")
        intervention = record["intervention"]
        start, end = intervention["target_span"]
        self.assertEqual(record["factual"]["passage"][start:end], intervention["target_text"])
        self.assertEqual(intervention["value_token"], "no")

    def test_surface_target_preserves_structural_identifier(self):
        item = copy.deepcopy(ITEM)
        item["given_info"] = "husband causes V2. husband or V2 causes Y."
        model = copy.deepcopy(MODEL)
        model["variable_mapping"] = {"Xname": "husband", "X1": "active husband", "X0": "inactive husband"}
        record = normalize_item(item, model)
        self.assertEqual(record["intervention"]["target"], "X")
        self.assertEqual(record["intervention"]["target_text"], "husband")


if __name__ == "__main__":
    unittest.main()
