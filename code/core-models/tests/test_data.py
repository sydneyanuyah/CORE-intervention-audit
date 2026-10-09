import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core_bert.data import LABEL_TO_ID, format_record, load_split


class DataTests(unittest.TestCase):
    def record(self):
        return {
            "id": "wiqa-example",
            "source": "wiqa",
            "structure_kind": "dag",
            "graph": {"nodes": ["X", "Y"], "edges": [["X", "Y"]]},
            "chain": None,
            "factual": {"passage": "Water flows.", "state": None, "question": "What changes?", "answer": None},
            "intervention": {"target": "X", "value": "less X", "kind": "value_set", "formal": "directional_change(X)", "text": "less X"},
            "intervened": {"passage": None, "state": None, "answer": "less"},
            "descendants": ["Y"],
            "non_descendants": [],
            "probes": [],
            "provenance": {},
        }

    def test_formatter_excludes_gold_intervened_answer(self):
        text = format_record(self.record())
        self.assertIn("X -> Y", text)
        self.assertIn("less X", text)
        self.assertNotIn("intervened answer", text)
        self.assertLess(text.index("intervention:"), text.index("passage:"))

    def test_split_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "records").mkdir()
            (root / "splits").mkdir()
            (root / "records/wiqa.jsonl").write_text(json.dumps(self.record()) + "\n")
            (root / "splits/wiqa.train.txt").write_text("wiqa-example\n")
            examples = load_split(root, "train")
            self.assertEqual(len(examples), 1)
            self.assertEqual(examples[0].label, LABEL_TO_ID["less"])

    def test_open_text_source_is_excluded(self):
        record = self.record()
        record["id"] = "com2-example"
        record["source"] = "com2"
        record["intervened"]["answer"] = "an open response"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "records").mkdir()
            (root / "splits").mkdir()
            (root / "records/com2.jsonl").write_text(json.dumps(record) + "\n")
            (root / "splits/com2.train.txt").write_text("com2-example\n")
            with self.assertRaisesRegex(ValueError, "no compatible examples"):
                load_split(root, "train")


if __name__ == "__main__":
    unittest.main()
