import unittest

from src.generate_g2_sequences import build_rows, scm_state


class G2GenerationTests(unittest.TestCase):
    def test_g2_rows_are_executable_and_graph_disjoint(self):
        rows = list(build_rows(1000, 1))
        self.assertEqual(len(rows), 1000)
        self.assertEqual({row["split"] for row in rows}, {"train", "validation", "test"})
        groups = {split: {row["graph_group_id"] for row in rows if row["split"] == split} for split in ("train", "validation", "test")}
        self.assertFalse(groups["train"] & groups["validation"] or groups["train"] & groups["test"] or groups["validation"] & groups["test"])
        for row in rows:
            self.assertNotEqual(row["ordered_edits"][0]["value"], row["ordered_edits"][-1]["value"])
            expected = scm_state(row["graph"], row["root_values"], {row["target"]: row["ordered_edits"][-1]["value"]})
            self.assertEqual(row["gold_final_state"], expected)
