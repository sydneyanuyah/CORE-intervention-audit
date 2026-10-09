import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from core_bert.paraphrase_protocol import (
    PREPARATION_STATUS,
    ParaphraseSpec,
    apply_paraphrase,
    build_paraphrase_partition,
    intervention_keys,
    normalize_paraphrase,
    paraphrase_spec_from_record,
    load_paraphrase_catalog,
    paraphrase_coverage,
)


def record(record_id="r1", text="Set X low."):
    return {
        "id": record_id,
        "source": "synthetic",
        "structure_kind": "dag",
        "intervention": {
            "kind": "value_set",
            "target": "X",
            "value": "low",
            "value_token": "low",
            "formal": "do(X=low)",
            "text": text,
        },
        "factual": {"passage": "X is high.", "question": "What is X?"},
        "provenance": {"generator": "fixture", "revision": 3},
    }


def catalog():
    base = record()
    return [
        paraphrase_spec_from_record(base | {"intervention": base["intervention"] | {"text": text}})
        for text in (
            "Set X low.",
            "Make X low.",
            "Change X so that it is low.",
            "Intervene to lower X.",
        )
    ]


class ParaphraseProtocolTests(unittest.TestCase):
    def test_partition_is_order_independent_disjoint_and_seeded(self):
        first = build_paraphrase_partition(catalog(), seed=91)
        second = build_paraphrase_partition(list(reversed(catalog())), seed=91)
        self.assertEqual(first.to_manifest(), second.to_manifest())
        operator, intervention = intervention_keys(record())
        train = first.texts_for(operator, intervention, "train")
        validation = first.texts_for(operator, intervention, "validation")
        train_hashes = {item.normalized_sha256 for item in train}
        validation_hashes = {item.normalized_sha256 for item in validation}
        self.assertTrue(train_hashes)
        self.assertTrue(validation_hashes)
        self.assertTrue(train_hashes.isdisjoint(validation_hashes))
        changed = build_paraphrase_partition(catalog(), seed=96)
        self.assertNotEqual(
            [(item.text, item.split) for item in first.assignments],
            [(item.text, item.split) for item in changed.assignments],
        )

    def test_normalized_duplicates_cannot_cross_splits(self):
        specs = catalog() + [
            ParaphraseSpec(
                catalog()[0].operator_key,
                catalog()[0].intervention_key,
                "  SET   x LOW.  ",
                {"generator": "fixture", "revision": 3},
            )
        ]
        partition = build_paraphrase_partition(specs)
        normalized = [normalize_paraphrase(item.text) for item in partition.assignments]
        self.assertEqual(len(normalized), len(set(normalized)))

    def test_conflicting_duplicate_provenance_fails_closed(self):
        duplicate = catalog()[0]
        with self.assertRaisesRegex(ValueError, "conflicting provenance"):
            build_paraphrase_partition(
                catalog()
                + [
                    ParaphraseSpec(
                        duplicate.operator_key,
                        duplicate.intervention_key,
                        duplicate.text.upper(),
                        {"generator": "different"},
                    )
                ]
            )

    def test_application_preserves_record_and_emits_provenance(self):
        partition = build_paraphrase_partition(catalog(), seed=7)
        source = record(text="Original wording.")
        prepared = apply_paraphrase(source, partition, split="validation")
        self.assertEqual(source["intervention"]["text"], "Original wording.")
        self.assertNotEqual(prepared["intervention"]["text"], "Original wording.")
        hook = prepared["protocol_provenance"]["instruction_paraphrase"]
        self.assertEqual(hook["split"], "validation")
        self.assertEqual(hook["evidence_status"], PREPARATION_STATUS)
        self.assertEqual(hook["catalog_sha256"], partition.catalog_sha256)
        restored = json.loads(json.dumps(prepared))
        restored.pop("protocol_provenance")
        restored["intervention"]["text"] = source["intervention"]["text"]
        self.assertEqual(restored, source)
        json.dumps(partition.to_manifest())
        json.dumps(prepared)

    def test_test_split_is_untouched_and_inaccessible(self):
        partition = build_paraphrase_partition(catalog())
        source = record()
        before = json.dumps(source, sort_keys=True)
        with self.assertRaisesRegex(PermissionError, "held-out test"):
            apply_paraphrase(source, partition, split="test")
        with self.assertRaises(PermissionError):
            partition.texts_for(*intervention_keys(source), "test")
        self.assertEqual(json.dumps(source, sort_keys=True), before)

    def test_single_wording_family_cannot_claim_unseen_evaluation(self):
        with self.assertRaisesRegex(ValueError, "at least two"):
            build_paraphrase_partition(catalog()[:1])

    def test_manifest_loader_verifies_hashes_isolation_and_coverage(self):
        partition = build_paraphrase_partition(catalog(), seed=5)
        document = {
            "artifact": "structured_instruction_paraphrase_catalog",
            "input_provenance": {
                "fixture": True,
                "evidence_status": PREPARATION_STATUS,
            },
            "partition": partition.to_manifest(),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            loaded, metadata = load_paraphrase_catalog(path)
            self.assertEqual(loaded, partition)
            self.assertEqual(metadata["catalog_sha256"], partition.catalog_sha256)
            coverage = paraphrase_coverage([record()], loaded, split="validation")
            self.assertEqual(coverage["coverage"], 1.0)
            self.assertEqual(coverage["covered_count"], 1)
            document["partition"]["assignments"][0]["text"] += " tampered"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_paraphrase_catalog(path)

    def test_coverage_rejects_missing_family_and_test(self):
        partition = build_paraphrase_partition(catalog())
        missing = record()
        missing["intervention"]["target"] = "unseen-target"
        with self.assertRaisesRegex(ValueError, "no train bank"):
            paraphrase_coverage([missing], partition, split="train")
        with self.assertRaises(PermissionError):
            paraphrase_coverage([record()], partition, split="test")


if __name__ == "__main__":
    unittest.main()
