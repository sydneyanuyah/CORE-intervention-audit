import copy
import json
import tempfile
import unittest
from pathlib import Path

from core_bert.t3_cladder_perturbations import (
    apply_manifest_pair,
    command_variant,
    inverted_instruction,
    load_manifest,
    rename_structural_variables,
    sha256_json,
    structural_aliases,
)


ROOT = Path(__file__).parents[1]
ARTIFACT = ROOT / "data" / "two_edit" / "cladder" / "artifact.json"
MANIFEST = ROOT / "registry" / "t3_cladder_perturbations.json"


class T3CladderPerturbationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifact = json.loads(ARTIFACT.read_text())
        cls.records = {
            record["id"]: record for record in cls.artifact["components"]
            if record["two_edit_metadata"]["split"] == "validation"
        }

    def test_manifest_is_complete_and_replayable(self):
        manifest, pairs = load_manifest(MANIFEST, ARTIFACT)
        self.assertEqual(manifest["world_count"], 20)
        self.assertEqual(manifest["source_record_count"], 144)
        self.assertEqual(manifest["pair_count"], 288)
        self.assertEqual(manifest["counts_by_dataset"], {
            "cladder_variants": 144, "variable_rename": 144,
        })
        for pair in pairs:
            perturbed = apply_manifest_pair(self.records[pair["record_id"]], pair)
            self.assertEqual(sha256_json(perturbed), pair["perturbed_sha256"])
            self.assertFalse(perturbed["t3_perturbation"]["test_evaluated"])

    def test_structural_rename_preserves_text_labels_and_graph_shape(self):
        record = next(iter(self.records.values()))
        aliases = structural_aliases(record["graph"]["nodes"], "world:fixture")
        renamed = rename_structural_variables(record, aliases)
        self.assertEqual(renamed["factual"]["passage"], record["factual"]["passage"])
        self.assertEqual(renamed["factual"]["question"], record["factual"]["question"])
        self.assertEqual(renamed["intervened"]["answer"], record["intervened"]["answer"])
        self.assertEqual(renamed["intervention"]["target_span"], record["intervention"]["target_span"])
        self.assertEqual(len(renamed["graph"]["edges"]), len(record["graph"]["edges"]))
        self.assertTrue(all(name.startswith("N1") for name in renamed["graph"]["nodes"]))

    def test_command_variant_changes_only_command_and_provenance(self):
        record = next(iter(self.records.values()))
        varied = command_variant(record, "assign")
        original = copy.deepcopy(record)
        changed = copy.deepcopy(varied)
        changed["id"] = original["id"]
        changed["intervention"]["text"] = original["intervention"]["text"]
        changed.pop("t3_perturbation")
        self.assertEqual(changed, original)

    def test_inverted_instruction_flips_value_and_retains_gold(self):
        record = next(iter(self.records.values()))
        inverted = inverted_instruction(record)
        self.assertEqual(inverted["intervention"]["value"], 1 - int(record["intervention"]["value"]))
        self.assertNotEqual(inverted["intervention"]["value_token"], record["intervention"]["value_token"])
        self.assertNotEqual(inverted["intervention"]["text"], record["intervention"]["text"])
        self.assertEqual(inverted["intervention"]["target"], record["intervention"]["target"])
        self.assertEqual(inverted["intervened"], record["intervened"])
        self.assertTrue(inverted["t3_positive_control"]["gold_retained_for_sensitivity_only"])

    def test_manifest_tampering_fails_closed(self):
        manifest = json.loads(MANIFEST.read_text())
        manifest["pairs"][0]["perturbed_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(manifest))
            _, pairs = load_manifest(path, ARTIFACT)
            record = self.records[pairs[0]["record_id"]]
            with self.assertRaisesRegex(ValueError, "perturbed record hash mismatch"):
                apply_manifest_pair(record, pairs[0])

    def test_test_split_is_rejected_before_pair_use(self):
        manifest = json.loads(MANIFEST.read_text())
        manifest["split"] = "test"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "validation-only"):
                load_manifest(path, ARTIFACT)


if __name__ == "__main__":
    unittest.main()
