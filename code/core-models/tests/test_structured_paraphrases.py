import importlib.util
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core_bert.paraphrase_protocol import apply_paraphrase, intervention_keys
from core_bert.structured_paraphrases import (
    GENERATOR,
    TEMPLATES,
    apply_structured_paraphrase,
    build_structured_partition,
    structured_catalog_manifest,
    structured_paraphrase_coverage,
    structured_specs_for_record,
)


def value_record(record_id="value-1", target="X", value="low"):
    return {
        "id": record_id,
        "source": "fixture",
        "source_id": record_id,
        "structure_kind": "dag",
        "graph": {"nodes": [target], "edges": []},
        "chain": None,
        "factual": {
            "passage": f"The variable {target} is high.",
            "question": f"What is {target}?",
        },
        "intervention": {
            "kind": "value_set",
            "target": target,
            "target_text": target,
            "value": value,
            "value_token": value,
            "formal": f"do({target}={value})",
            "text": "An original wording that templates must ignore.",
        },
        "intervened": {"answer": value},
        "provenance": {"fixture": True},
    }


def event_record(record_id="event-1"):
    result = value_record(record_id, "event_01: rain starts", "rain stops")
    result.update(
        {
            "structure_kind": "chain",
            "graph": None,
            "chain": ["clouds gather", "rain starts", "ground gets wet"],
        }
    )
    result["intervention"].update(
        {
            "kind": "event_replace",
            "target_text": "rain starts",
            "value_token": None,
            "formal": 'do(event_01 = "rain stops")',
        }
    )
    return result


class StructuredParaphraseTests(unittest.TestCase):
    def test_value_and_event_templates_use_only_explicit_intervention_fields(self):
        value = value_record()
        altered = value_record(record_id="other")
        altered["factual"] = {"passage": "unrelated", "question": "unrelated"}
        altered["intervention"]["text"] = "Completely different original command."
        first = structured_specs_for_record(value, source_split="train")
        second = structured_specs_for_record(altered, source_split="validation")
        self.assertEqual([item.text for item in first], [item.text for item in second])
        self.assertEqual(len(first), 4)
        self.assertEqual(len({item.text for item in first}), 4)
        self.assertTrue(all(item.wording_family_key for item in first))
        event = structured_specs_for_record(event_record(), source_split="train")
        self.assertEqual(len(event), 4)
        self.assertTrue(all('"rain starts"' in item.text for item in event))
        self.assertTrue(all('"rain stops"' in item.text for item in event))
        numeric = value_record(value=0)
        numeric["intervention"]["value_token"] = "0"
        numeric_specs = structured_specs_for_record(numeric, source_split="train")
        self.assertTrue(all(" 0" in item.text for item in numeric_specs))

    def test_template_families_never_cross_train_validation(self):
        records = {
            "train": [value_record("x", "X")],
            "validation": [value_record("y", "Y")],
        }
        partition, provenance = build_structured_partition(records, seed=19)
        by_template = {}
        by_intervention = {}
        for item in partition.assignments:
            by_template.setdefault(item.wording_family_key, set()).add(item.split)
            by_intervention.setdefault(item.intervention_key, set()).add(item.split)
        self.assertTrue(all(len(splits) == 1 for splits in by_template.values()))
        self.assertTrue(all(splits == {"train", "validation"} for splits in by_intervention.values()))
        self.assertEqual(set(by_template), {item[0] for item in TEMPLATES["value_set"]})
        self.assertEqual(provenance["generator"], GENERATOR)

    def test_partition_applies_held_out_wording_and_manifest_is_serializable(self):
        source = value_record()
        partition, provenance = build_structured_partition(
            {"train": [source], "validation": []}, seed=3
        )
        train = apply_paraphrase(source, partition, split="train")
        validation = apply_paraphrase(source, partition, split="validation")
        self.assertNotEqual(
            train["protocol_provenance"]["instruction_paraphrase"]["selected_text_sha256"],
            validation["protocol_provenance"]["instruction_paraphrase"]["selected_text_sha256"],
        )
        manifest = structured_catalog_manifest(partition, provenance)
        self.assertEqual(manifest["partition"]["evidence_status"],
                         "protocol_preparation_only_not_a1_evidence")
        json.dumps(manifest)

    def test_test_source_split_is_hard_blocked(self):
        with self.assertRaisesRegex(PermissionError, "held-out test"):
            structured_specs_for_record(value_record(), source_split="test")
        with self.assertRaises(PermissionError):
            build_structured_partition({"test": [value_record()]})

    def test_surface_target_is_part_of_semantic_intervention_key(self):
        first = value_record("one")
        second = value_record("two")
        second["intervention"]["target_text"] = "a conflicting surface name"
        self.assertNotEqual(intervention_keys(first), intervention_keys(second))
        partition, _ = build_structured_partition(
            {"train": [first], "validation": [second]}
        )
        self.assertEqual(len({item.intervention_key for item in partition.assignments}), 2)

    def test_post_catalog_component_materializes_only_frozen_validation_unit(self):
        source = value_record("source", "X", "low")
        partition, _ = build_structured_partition(
            {"train": [source], "validation": []}, seed=3
        )
        component = value_record("component", "unseen", "high")
        before = partition.to_manifest()
        with self.assertRaises(KeyError):
            apply_structured_paraphrase(component, partition, split="validation")
        prepared, materialized, selected = apply_structured_paraphrase(
            component,
            partition,
            split="validation",
            allow_materialized=True,
        )
        self.assertTrue(materialized)
        self.assertEqual(selected.split, "validation")
        self.assertIn(
            selected.wording_family_key,
            {
                item.wording_family_key
                for item in partition.assignments
                if item.split == "validation"
            },
        )
        audit = prepared["protocol_provenance"]["instruction_paraphrase"]
        self.assertEqual(audit["catalog_sha256"], partition.catalog_sha256)
        self.assertTrue(
            audit["source_provenance"]["materialized_from_frozen_template_unit"]
        )
        coverage = structured_paraphrase_coverage(
            [component],
            partition,
            split="validation",
            allow_materialized=True,
        )
        self.assertEqual(coverage["coverage"], 1.0)
        self.assertEqual(coverage["materialized_count"], 1)
        self.assertEqual(len(coverage["materialized_selections"]), 1)
        self.assertEqual(partition.to_manifest(), before)


class StructuredParaphraseCLITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "build_paraphrase_catalog", ROOT / "scripts" / "build_paraphrase_catalog.py"
        )
        cls.cli = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(cls.cli)

    def test_cli_writes_development_manifest_without_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "records").mkdir()
            (root / "splits").mkdir()
            records = [value_record(), event_record()]
            (root / "records" / "fixture.jsonl").write_text(
                "".join(json.dumps(item) + "\n" for item in records), encoding="utf-8"
            )
            (root / "splits" / "fixture.train.txt").write_text("value-1\n", encoding="utf-8")
            (root / "splits" / "fixture.validation.txt").write_text(
                "event-1\n", encoding="utf-8"
            )
            output = root / "artifacts" / "catalog.json"
            with redirect_stdout(StringIO()):
                self.assertEqual(
                    self.cli.main(
                        ["--data-root", str(root), "--output-json", str(output), "--seed", "11"]
                    ),
                    0,
                )
            manifest = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(manifest["artifact"], "structured_instruction_paraphrase_catalog")
            self.assertEqual(manifest["partition"]["counts"], {"train": 6, "validation": 2})
            self.assertEqual(
                set(manifest["input_provenance"]["source_splits"]),
                {"train", "validation"},
            )

    def test_cli_rejects_test_before_loading_data(self):
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            self.cli.parse_args(
                [
                    "--data-root", "/does/not/matter",
                    "--output-json", "/does/not/matter.json",
                    "--source-splits", "test",
                ]
            )


if __name__ == "__main__":
    unittest.main()
