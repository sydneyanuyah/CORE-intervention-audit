import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from core_bert.two_edit import ordered_pair_id  # noqa: E402
from core_bert.xor_two_edit_adapter import (  # noqa: E402
    XOR_WORDING_FAMILIES,
    XOR_WORDING_PROTOCOL,
    load_xor_two_edit_bundle,
)
from evaluate_xor_two_edit import (  # noqa: E402
    _summarize_rank_payloads,
    parse_args,
    require_bert_base_world_size,
    shard_graph_bundles,
)


def artifacts():
    nodes = [f"X{index:02d}" for index in range(30)]
    graph = {
        "nodes": nodes,
        "root_nodes": nodes,
        "parents": {node: [] for node in nodes},
        "xor_bias": {node: 0 for node in nodes},
        "edges": [],
        "seed": 7,
    }
    worlds = []
    records = []
    for split, world_id in (("train", "train-0"), ("validation", "validation-0")):
        state = {node: 0 for node in nodes}
        passage = ";".join(f"{node}=0" for node in nodes)
        worlds.append({
            "world_id": world_id, "split": split,
            "root_values": state, "factual_state": state, "passage": passage,
        })
        for node in nodes:
            for value in (0, 1):
                after = dict(state)
                after[node] = value
                records.append({
                    "record_id": f"{world_id}-{node}-{value}",
                    "world_id": world_id,
                    "split": split,
                    "intervention": {
                        "formal": f"do({node}={value})", "text": f"{node}={value}",
                        "target": node, "value": value,
                    },
                    "intervened_state": after,
                })
    graph_group = "xor:graph:7"
    world_group = f"{graph_group}:world:validation-0"
    first, second = "validation-0-X00-1", "validation-0-X07-1"
    factual = [0] * 30
    gold = [1 if index in (0, 7) else 0 for index in range(30)]
    row = {
        "pair_id": ordered_pair_id("xor", graph_group, world_group, first, second),
        "source": "xor", "split": "validation",
        "graph_group_id": graph_group, "world_group_id": world_group,
        "first_record_id": first, "second_record_id": second,
        "gold_outputs": gold, "factual_outputs": factual,
        "change_mask": [value == 1 for value in gold],
        "preservation_mask": [value == 0 for value in gold],
        "target_mask": [index in (0, 7) for index in range(30)],
        "truth_provenance": {
            "generator": "core_bert.xor_two_edit.generate_xor_two_edit_manifest",
            "truth_source": "recomputed executable XOR structural equations",
            "a1_evidence": False,
        },
    }
    return graph, worlds, records, row


class XORAdapterTests(unittest.TestCase):
    def write(self, directory, *, mutate=None):
        graph, worlds, records, row = artifacts()
        if mutate:
            mutate(graph, worlds, records, row)
        path = Path(directory)
        (path / "graph.json").write_text(json.dumps(graph))
        (path / "worlds.json").write_text(json.dumps(worlds))
        (path / "records_real.json").write_text(json.dumps(records))
        manifest = path / "two_edit.jsonl"
        manifest.write_text(json.dumps(row) + "\n")
        return manifest

    def test_loads_validation_truth_and_builds_scientific_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = self.write(temporary)
            bundle = load_xor_two_edit_bundle(
                Path(temporary), manifest, split="validation"
            )
        self.assertEqual(bundle.graph_group_id, "xor:graph:7")
        self.assertEqual(len(bundle.nodes), 30)
        self.assertEqual(len(bundle.examples), 1)
        example = bundle.examples[0]
        self.assertEqual(len(example.gold_outputs), 30)
        record = bundle.records[example.first_record_id]
        start, end = record["intervention"]["target_span"]
        self.assertEqual(record["factual"]["passage"][start:end], "X00")
        self.assertEqual(record["intervention"]["value_token"], "1")
        self.assertEqual(len(record["probes"]), 30)
        self.assertEqual(record["intervened"]["answer"], "yes")
        queried = [probe for probe in record["probes"] if probe["question"]]
        self.assertEqual([probe["variable"] for probe in queried], ["X00"])
        self.assertEqual(queried[0]["answer_before"], 0)
        self.assertEqual(queried[0]["answer_after"], 1)
        self.assertEqual(
            record["intervention"]["wording_family"],
            XOR_WORDING_FAMILIES["validation"],
        )
        self.assertEqual(
            record["intervention"]["wording_protocol"], XOR_WORDING_PROTOCOL
        )
        self.assertNotEqual(
            XOR_WORDING_FAMILIES["train"], XOR_WORDING_FAMILIES["validation"]
        )

    def test_reexecutes_truth_and_rejects_tampering(self):
        def tamper(_graph, _worlds, _records, row):
            row["gold_outputs"][9] = 1
            row["change_mask"][9] = True
            row["preservation_mask"][9] = False

        with tempfile.TemporaryDirectory() as temporary:
            manifest = self.write(temporary, mutate=tamper)
            with self.assertRaisesRegex(ValueError, "disagrees with executable SCM"):
                load_xor_two_edit_bundle(Path(temporary), manifest, split="validation")

    def test_requires_separate_training_exposure_and_blocks_test(self):
        def remove_seen(_graph, _worlds, records, _row):
            records[:] = [
                record for record in records
                if not (
                    record["split"] == "train"
                    and record["intervention"]["target"] == "X07"
                    and record["intervention"]["value"] == 1
                )
            ]

        with tempfile.TemporaryDirectory() as temporary:
            manifest = self.write(temporary, mutate=remove_seen)
            with self.assertRaisesRegex(ValueError, "seen separately"):
                load_xor_two_edit_bundle(Path(temporary), manifest, split="validation")
            with self.assertRaisesRegex(ValueError, "test is blocked"):
                load_xor_two_edit_bundle(Path(temporary), manifest, split="test")

    def test_cli_has_no_test_split(self):
        base = [
            "--bundle", "artifacts", "manifest.jsonl",
            "--checkpoint", "best.pt", "--output-json", "metrics.json",
            "--mode", "t3b",
        ]
        self.assertEqual(parse_args(base).split, "validation")
        with self.assertRaises(SystemExit):
            parse_args([*base, "--split", "test"])

    def test_exact_four_rank_policy_and_whole_graph_sharding(self):
        require_bert_base_world_size(4)
        for invalid in (1, 2, 8):
            with self.assertRaisesRegex(RuntimeError, "exactly 4 ranks"):
                require_bert_base_world_size(invalid)
        bundles = [
            (Path(f"graph-{index}"), Path(f"manifest-{index}"))
            for index in range(2)
        ]
        shards = [shard_graph_bundles(bundles, rank, 4) for rank in range(4)]
        self.assertEqual([len(shard) for shard in shards], [1, 1, 0, 0])
        self.assertEqual(
            sorted(item for shard in shards for item in shard), sorted(bundles)
        )
        self.assertEqual(
            len({item for shard in shards for item in shard}), len(bundles)
        )

    def test_duplicate_bundle_arguments_fail_before_distributed_eval(self):
        bundle = (Path("graph"), Path("manifest"))
        with self.assertRaisesRegex(ValueError, "duplicate artifact"):
            shard_graph_bundles([bundle, bundle], 0, 4)

    def test_rank_gather_rejects_duplicate_pair_ids(self):
        args = parse_args([
            "--bundle", "artifacts", "manifest.jsonl",
            "--checkpoint", "best.pt", "--output-json", "metrics.json",
            "--mode", "t3b",
        ])
        with tempfile.TemporaryDirectory() as temporary:
            manifest = self.write(temporary)
            duplicate = load_xor_two_edit_bundle(
                Path(temporary), manifest, split="validation"
            ).examples[0]
        payloads = [
            {"checkpoint_sha256": "same", "examples": [duplicate]},
            {"checkpoint_sha256": "same", "examples": [duplicate]},
        ]
        with self.assertRaisesRegex(ValueError, "duplicate pair IDs"):
            _summarize_rank_payloads(payloads, args)

    def test_rank_gather_requires_identical_checkpoint_hash(self):
        args = parse_args([
            "--bundle", "artifacts", "manifest.jsonl",
            "--checkpoint", "best.pt", "--output-json", "metrics.json",
            "--mode", "t3b",
        ])
        with self.assertRaisesRegex(RuntimeError, "same checkpoint bytes"):
            _summarize_rank_payloads(
                [
                    {"checkpoint_sha256": "left", "examples": []},
                    {"checkpoint_sha256": "right", "examples": []},
                ],
                args,
            )

    def test_rank_gather_reports_both_composed_pointer_traces(self):
        args = parse_args([
            "--bundle", "artifacts", "manifest.jsonl",
            "--checkpoint", "best.pt", "--output-json", "metrics.json",
            "--mode", "t3b",
        ])
        with tempfile.TemporaryDirectory() as temporary:
            manifest = self.write(temporary)
            example = load_xor_two_edit_bundle(
                Path(temporary), manifest, split="validation"
            ).examples[0]
        gold = list(example.gold_outputs)
        payload = {
            "checkpoint_sha256": "same",
            "bundle_count": 1,
            "examples": [example],
            "predictions": {example.pair_id: gold},
            "variable_rows": [{
                "graph_group_id": example.graph_group_id,
                "pair_id": example.pair_id,
                "active_variable_count": 30,
                "variable_ids": [f"X{index:02d}" for index in range(30)],
                "variable_gold": gold,
                "variable_predicted": gold,
                "variable_loss_sum": 0.0,
            }],
            "pointer_rows": [
                {"pair_id": example.pair_id, "ordinal": 1,
                 "pointer_mass": 0.8, "pointer_top1": 1},
                {"pair_id": example.pair_id, "ordinal": 2,
                 "pointer_mass": 0.6, "pointer_top1": 0},
            ],
            "artifact_sha256": {},
            "graph_ids": [example.graph_group_id],
        }
        result = _summarize_rank_payloads([payload], args)
        self.assertEqual(result["pointer"]["eligible_count"], 2)
        self.assertEqual(result["pointer"]["coverage"], 1.0)
        self.assertAlmostEqual(result["pointer"]["pointer_mass"], 0.7)
        self.assertEqual(result["pointer"]["pointer_top1_accuracy"], 0.5)

    def test_rank_gather_reports_complete_composed_paraphrase_coverage(self):
        args = parse_args([
            "--bundle", "artifacts", "manifest.jsonl",
            "--checkpoint", "best.pt", "--output-json", "metrics.json",
            "--mode", "t3b",
        ])
        with tempfile.TemporaryDirectory() as temporary:
            manifest = self.write(temporary)
            example = load_xor_two_edit_bundle(
                Path(temporary), manifest, split="validation"
            ).examples[0]
        gold = list(example.gold_outputs)
        payload = {
            "checkpoint_sha256": "same",
            "bundle_count": 1,
            "examples": [example],
            "predictions": {example.pair_id: gold},
            "variable_rows": [{
                "graph_group_id": example.graph_group_id,
                "pair_id": example.pair_id,
                "active_variable_count": 30,
                "variable_ids": [f"X{index:02d}" for index in range(30)],
                "variable_gold": gold,
                "variable_predicted": gold,
                "variable_loss_sum": 0.0,
            }],
            "pointer_rows": [
                {"pair_id": example.pair_id, "ordinal": ordinal,
                 "pointer_mass": 1.0, "pointer_top1": 1}
                for ordinal in (1, 2)
            ],
            "artifact_sha256": {},
            "graph_ids": [example.graph_group_id],
            "unseen_paraphrase": {
                "protocol": "xor_intervention_wording_v1",
                "bank": "validation",
                "catalog_sha256": None,
                "eligible_count": 2,
                "covered_count": 2,
            },
        }
        result = _summarize_rank_payloads([payload], args)
        self.assertEqual(result["unseen_paraphrase"]["coverage"], 1.0)
        self.assertEqual(result["unseen_paraphrase"]["bank"], "validation")


if __name__ == "__main__":
    unittest.main()
