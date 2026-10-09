import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from core_bert.benchmark_data import (  # noqa: E402
    BenchmarkDataset,
    BenchmarkExample,
    GroupDistributedSampler,
    deterministic_smoke_subset,
    load_benchmark_split,
)


def record(record_id, source, label, kind="value_set", **extra):
    value = {
        "id": record_id,
        "source": source,
        "source_id": extra.pop("source_id", record_id),
        "structure_kind": "dag",
        "graph": {"nodes": ["A", "B"], "edges": [["A", "B"]]},
        "chain": None,
        "factual": {"passage": f"passage {record_id}", "state": {"A": 0}},
        "intervention": {"kind": kind},
        "intervened": {"answer": label},
    }
    value.update(extra)
    return value


class BenchmarkDataTests(unittest.TestCase):
    def make_root(self):
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        (root / "records").mkdir()
        (root / "splits").mkdir()
        rows = {
            "cladder.jsonl": [
                record("cladder-a", "cladder", "yes", graph_id="g-explicit", world_id="w-explicit"),
                record("cladder-b", "cladder", "no"),
            ],
            "com2_intervention.jsonl": [record("com2-i", "com2", "open-i", "event_replace", chain=["root", "i"], graph=None)],
            "com2_counterfactual.jsonl": [record("com2-c", "com2", "open-c", "event_replace", chain=["root", "c"], graph=None)],
        }
        for filename, values in rows.items():
            (root / "records" / filename).write_text(
                "".join(json.dumps(value) + "\n" for value in values), encoding="utf-8"
            )
            stem = filename.removesuffix(".jsonl")
            for split in ("train", "validation", "test"):
                ids = [value["id"] for value in values] if split in {"train", "test"} else []
                (root / "splits" / f"{stem}.{split}.txt").write_text(
                    "".join(f"{record_id}\n" for record_id in ids), encoding="utf-8"
                )
        return temporary, root

    def test_loads_source_selection_and_both_com2_files(self):
        temporary, root = self.make_root()
        self.addCleanup(temporary.cleanup)
        dataset = load_benchmark_split(root, "train", sources=["com2"])
        self.assertEqual([item.record_id for item in dataset], ["com2-c", "com2-i"])
        self.assertEqual(len(dataset.records), 2)
        self.assertTrue(all(item.graph_group_id.startswith("com2-root:") for item in dataset))
        self.assertEqual(len({item.graph_group_id for item in dataset}), 1)
        self.assertEqual(len({item.world_group_id for item in dataset}), 2)

    def test_test_split_requires_explicit_final_eval(self):
        temporary, root = self.make_root()
        self.addCleanup(temporary.cleanup)
        with self.assertRaisesRegex(PermissionError, "final_eval=True"):
            load_benchmark_split(root, "test")
        dataset = load_benchmark_split(root, "test", sources=["cladder"], final_eval=True)
        self.assertEqual(len(dataset), 2)

    def test_preserves_explicit_graph_and_world_ids(self):
        temporary, root = self.make_root()
        self.addCleanup(temporary.cleanup)
        item = load_benchmark_split(root, "train", sources=["cladder"])[0]
        self.assertEqual(item.graph_group_id, "cladder:graph:g-explicit")
        self.assertEqual(item.world_group_id, "cladder:world:w-explicit")

    def test_smoke_subset_is_order_independent_and_stratified(self):
        examples = []
        for source in ("cladder", "wiqa"):
            for label in ("yes", "no"):
                for suffix in range(3):
                    raw = record(f"{source}-{label}-{suffix}", source, label)
                    examples.append(
                        BenchmarkExample(raw, raw["id"], source, label, "value_set", f"g-{suffix}", f"w-{suffix}")
                    )
        first = deterministic_smoke_subset(BenchmarkDataset(examples), seed=17)
        shuffled = list(examples)
        random.Random(99).shuffle(shuffled)
        second = deterministic_smoke_subset(BenchmarkDataset(shuffled), seed=17)
        self.assertEqual([item.record_id for item in first], [item.record_id for item in second])
        self.assertEqual(len(first), 4)
        self.assertEqual(
            {(item.source, item.label, item.intervention_kind) for item in first},
            {(source, label, "value_set") for source in ("cladder", "wiqa") for label in ("yes", "no")},
        )

    def test_smoke_subset_stratifies_com2_by_choice_not_answer_text(self):
        examples = []
        for index, label in enumerate(("A) first answer", "A) another answer", "B. second")):
            raw = record(f"com2-{index}", "com2", label, "event_replace")
            examples.append(
                BenchmarkExample(raw, raw["id"], "com2", label, "event_replace", f"g-{index}", f"w-{index}")
            )
        smoke = deterministic_smoke_subset(BenchmarkDataset(examples), seed=3)
        self.assertEqual(len(smoke), 2)
        self.assertEqual(
            {item.label[0] for item in smoke},
            {"A", "B"},
        )

    def test_group_sampler_never_splits_groups_and_can_equalize_ranks(self):
        examples = []
        for group, size in (("g0", 3), ("g1", 2), ("g2", 2), ("g3", 1)):
            for offset in range(size):
                raw = record(f"{group}-{offset}", "wiqa", "more")
                examples.append(
                    BenchmarkExample(raw, raw["id"], "wiqa", "more", "value_set", group, f"{group}-w")
                )
        dataset = BenchmarkDataset(examples)
        samplers = [
            GroupDistributedSampler(dataset, rank=rank, world_size=2, seed=7)
            for rank in range(2)
        ]
        shards = [list(sampler) for sampler in samplers]
        self.assertEqual(len(shards[0]), len(shards[1]))
        groups_by_rank = [
            {dataset[index].graph_group_id for index in set(shard)} for shard in shards
        ]
        self.assertFalse(groups_by_rank[0] & groups_by_rank[1])
        self.assertEqual(set.union(*groups_by_rank), {"g0", "g1", "g2", "g3"})
        self.assertEqual(shards[0], list(samplers[0]))
        for sampler in samplers:
            sampler.set_epoch(1)
        epoch_one_groups = [
            {dataset[index].graph_group_id for index in set(list(sampler))}
            for sampler in samplers
        ]
        self.assertFalse(epoch_one_groups[0] & epoch_one_groups[1])
        self.assertEqual(set.union(*epoch_one_groups), {"g0", "g1", "g2", "g3"})

    def test_exact_evaluation_allows_empty_rank_shards(self):
        examples = []
        for group in ("g0", "g1"):
            raw = record(group, "ccrgb", "yes")
            examples.append(
                BenchmarkExample(raw, raw["id"], "ccrgb", "yes", "value_set", group, f"{group}-w")
            )
        dataset = BenchmarkDataset(examples)
        shards = [
            list(GroupDistributedSampler(
                dataset, rank=rank, world_size=4, shuffle=False, pad_to_equal=False
            ))
            for rank in range(4)
        ]
        self.assertEqual(sorted(index for shard in shards for index in shard), [0, 1])
        self.assertEqual(sum(not shard for shard in shards), 2)
        with self.assertRaisesRegex(ValueError, "fewer groups"):
            list(GroupDistributedSampler(
                dataset, rank=0, world_size=4, shuffle=False, pad_to_equal=True
            ))


if __name__ == "__main__":
    unittest.main()
