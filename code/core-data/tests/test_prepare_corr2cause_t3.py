import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import prepare_corr2cause_t3 as module
from prepare_corr2cause_t3 import prepare, refactor_variables


class Corr2CauseT3Tests(unittest.TestCase):
    def test_reverse_alphabet_refactor(self):
        self.assertEqual(
            refactor_variables("A causes B while C is fixed.", 3),
            "Z causes Y while X is fixed.",
        )

    def test_exact_alignment_and_duplicate_collapse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clean = root / "clean.csv"
            refactor = root / "refactor.csv"
            paraphrase = root / "paraphrase.json"
            with clean.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["input", "label", "num_variables", "template"])
                writer.writeheader(); writer.writerow({
                    "input": "Premise: A is independent of B.\nHypothesis: A causes B.",
                    "label": "0", "num_variables": "2", "template": "parent",
                })
            with refactor.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["", "label", "input"])
                writer.writeheader()
                for index in range(2):
                    writer.writerow({"": index, "label": "0", "input": "Premise: Z is independent of Y.\nHypothesis: Z causes Y."})
            para_row = {
                "premise": "A and B are statistically independent.",
                "hypothesis": "A directly affects B.", "relation": "contradiction",
                "id": "num_nodes=2__mec_id=7__node_i=1__node_j=2__causal_relation=parent__prob=0.00",
            }
            paraphrase.write_text(json.dumps([para_row, para_row]))
            hashes = {
                "clean": module.file_hash(clean),
                "paraphrase": module.file_hash(paraphrase),
                "refactorization": module.file_hash(refactor),
            }
            with patch("prepare_corr2cause_t3.RAW_HASHES", hashes):
                pairs, summary = prepare(
                    clean, paraphrase, refactor, expected_counts=(1, 2, 2)
                )
            self.assertEqual(len(pairs), 1)
            self.assertEqual(pairs[0]["source_rows"]["perturbations"], [0, 1])
            self.assertEqual(pairs[0]["graph_group_id"], "corr2cause:n2:mec7")
            self.assertEqual(summary["duplicate_equivalent_perturbation_groups_collapsed"], 1)
            self.assertFalse(summary["test_evaluated"])


if __name__ == "__main__":
    unittest.main()
