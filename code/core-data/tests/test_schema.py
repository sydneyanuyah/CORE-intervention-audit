import copy
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from schema import load_jsonl, validate_record  # noqa: E402
from verify import verify  # noqa: E402


class SchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = load_jsonl(ROOT / "tests" / "fixtures" / "valid_records.jsonl")

    def test_valid_chain_and_dag_fixtures(self):
        self.assertEqual(len(self.records), 2)
        for record in self.records:
            self.assertEqual(validate_record(record), [])

    def test_unexpected_key_is_rejected(self):
        record = copy.deepcopy(self.records[0])
        record["split"] = "test"
        self.assertTrue(any("unexpected keys" in error for error in validate_record(record)))

    def test_descendant_overlap_is_rejected(self):
        record = copy.deepcopy(self.records[1])
        record["non_descendants"].append("wet_grass")
        self.assertTrue(any("overlap" in error for error in validate_record(record)))

    def test_target_in_descendants_is_rejected(self):
        record = copy.deepcopy(self.records[1])
        record["descendants"].append("sprinkler")
        self.assertTrue(any("target" in error for error in validate_record(record)))

    def test_target_span_must_exactly_slice_factual_target(self):
        record = copy.deepcopy(self.records[1])
        record["intervention"]["target_span"] = [19, 28]
        errors = validate_record(record)
        self.assertTrue(any("target_span slices" in error for error in errors))

    def test_new_intervention_addressing_fields_are_required(self):
        for field in ("target_text", "target_span", "value_token", "replacement_span"):
            record = copy.deepcopy(self.records[1])
            del record["intervention"][field]
            self.assertTrue(
                any(
                    "intervention missing keys" in error and field in error
                    for error in validate_record(record)
                )
            )

    def test_span_rejects_bool_offsets_and_invalid_bounds(self):
        record = copy.deepcopy(self.records[1])
        record["intervention"]["target_span"] = [True, 27]
        self.assertTrue(
            any("integer pair" in error for error in validate_record(record))
        )
        record["intervention"]["target_span"] = [27, 18]
        self.assertTrue(
            any("0 <= start < end" in error for error in validate_record(record))
        )
        record["intervention"]["target_span"] = [18, 999]
        self.assertTrue(
            any("outside its passage" in error for error in validate_record(record))
        )

    def test_value_set_requires_value_token_and_no_replacement_span(self):
        record = copy.deepcopy(self.records[1])
        record["intervention"]["value_token"] = None
        self.assertTrue(any("value_token" in error for error in validate_record(record)))
        record = copy.deepcopy(self.records[1])
        record["intervention"]["replacement_span"] = [18, 21]
        self.assertTrue(
            any("replacement_span must be null" in error for error in validate_record(record))
        )

    def test_event_replacement_span_must_exactly_slice_replacement(self):
        record = copy.deepcopy(self.records[0])
        self.assertEqual(validate_record(record), [])
        record["intervention"]["replacement_span"] = [15, 24]
        self.assertTrue(
            any("replacement_span slices" in error for error in validate_record(record))
        )

    def test_target_text_preserves_structural_target_identity(self):
        record = copy.deepcopy(self.records[1])
        record["intervention"]["target"] = "node_17"
        self.assertEqual(validate_record(record), [])
        record["intervention"]["target_text"] = "sprinkler system"
        self.assertTrue(
            any("target_span slices" in error for error in validate_record(record))
        )

    def test_event_replace_rejects_value_token(self):
        record = copy.deepcopy(self.records[0])
        record["intervention"]["value_token"] = "rain stops"
        self.assertTrue(any("value_token must be null" in error for error in validate_record(record)))

    def test_preservation_violation_is_rejected(self):
        record = copy.deepcopy(self.records[0])
        record["probes"][0]["answer_after"] = "no"
        record["probes"][0]["actually_changed"] = True
        self.assertTrue(any("preservation" in error for error in validate_record(record)))

    def test_record_without_observed_effect_can_be_screened_for_quarantine(self):
        record = copy.deepcopy(self.records[0])
        record["probes"][1]["answer_after"] = record["probes"][1]["answer_before"]
        record["probes"][1]["actually_changed"] = False
        self.assertTrue(any("no may-change probe" in error for error in validate_record(record)))
        self.assertEqual(validate_record(record, require_observed_effect=False), [])

    def test_cyclic_graph_is_rejected(self):
        record = copy.deepcopy(self.records[1])
        record["graph"]["edges"].append(["wet_grass", "season"])
        self.assertTrue(any("acyclic" in error for error in validate_record(record)))

    def test_optional_node_text_must_cover_graph_exactly(self):
        record = copy.deepcopy(self.records[1])
        nodes = record["graph"]["nodes"]
        record["graph"]["node_text"] = {node: [f"surface for {node}"] for node in nodes}
        self.assertEqual(validate_record(record), [])
        del record["graph"]["node_text"][nodes[0]]
        self.assertTrue(any("node_text" in error for error in validate_record(record)))

    def test_verifier_rejects_incomplete_split_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records_dir = root / "records"
            splits_dir = root / "splits"
            records_dir.mkdir()
            splits_dir.mkdir()
            (records_dir / "fixture.jsonl").write_text(
                "\n".join(__import__("json").dumps(record) for record in self.records) + "\n",
                encoding="utf-8",
            )
            (splits_dir / "fixture.train.txt").write_text(
                self.records[0]["id"] + "\n", encoding="utf-8"
            )
            with contextlib.redirect_stdout(io.StringIO()) as output:
                status = verify(records_dir, splits_dir)
            self.assertEqual(status, 1)
            self.assertIn("not assigned to a split", output.getvalue())


if __name__ == "__main__":
    unittest.main()
