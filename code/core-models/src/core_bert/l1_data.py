"""Deterministic paired-example construction for the L1 law profiles."""

from __future__ import annotations

import copy
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Mapping

from .benchmark_data import BenchmarkDataset, BenchmarkExample


TRUE_VALUES = frozenset({"1", "on", "high", "more", "present", "increased", "yes"})
FALSE_VALUES = frozenset({"0", "off", "low", "less", "absent", "decreased", "no"})


def binary_intervention(record: Mapping[str, Any]) -> tuple[int, int]:
    intervention, graph = record.get("intervention"), record.get("graph")
    if not isinstance(intervention, Mapping) or not isinstance(graph, Mapping):
        raise ValueError("L1 requires an explicit intervention and graph")
    nodes, target = graph.get("nodes"), intervention.get("target")
    if not isinstance(nodes, list) or target not in nodes:
        raise ValueError("L1 intervention target is not a graph node")
    raw = intervention.get("value")
    if isinstance(raw, bool) or isinstance(raw, int) and raw in (0, 1):
        value = int(raw)
    else:
        token = str(raw).strip().lower()
        if token in TRUE_VALUES:
            value = 1
        elif token in FALSE_VALUES:
            value = 0
        else:
            raise ValueError(f"L1 requires a binary intervention value, found {raw!r}")
    return nodes.index(target), value


@dataclass(frozen=True)
class L1PairedExample:
    base: BenchmarkExample
    commuting: BenchmarkExample
    opposite: BenchmarkExample


class L1PairedDataset(Sequence[L1PairedExample]):
    """Each example carries immutable partners for both paired laws."""

    def __init__(self, dataset: BenchmarkDataset):
        examples = tuple(dataset.examples)
        metadata = [binary_intervention(example.record) for example in examples]
        paired = []
        for index, example in enumerate(examples):
            target, value = metadata[index]
            commute = next((
                candidate for candidate, other in enumerate(examples)
                if other.graph_group_id == example.graph_group_id
                and metadata[candidate][0] != target
            ), None)
            opposite = next((
                candidate for candidate, other in enumerate(examples)
                if other.graph_group_id == example.graph_group_id
                and metadata[candidate] == (target, 1 - value)
            ), None)
            commuting = (
                examples[commute] if commute is not None
                else synthetic_commuting_example(example, target, value)
            )
            opposite_example = (
                examples[opposite] if opposite is not None
                else synthetic_opposite_example(example, target, value)
            )
            paired.append(L1PairedExample(example, commuting, opposite_example))
        self.examples = tuple(paired)

    def __getitem__(self, index: int) -> L1PairedExample:
        return self.examples[index]

    def __len__(self) -> int:
        return len(self.examples)


def synthetic_commuting_example(
    example: BenchmarkExample, target_slot: int, value: int,
) -> BenchmarkExample:
    """Create the preregistered law-only command for a different graph target.

    Some held-out graphs expose observations for only one intervention target.
    Commutation is still defined algebraically on their multi-slot state.  The
    deterministic first other slot supplies an instruction encoding only; its
    answer and post-intervention state are never used as task supervision.
    """

    record = copy.deepcopy(example.record)
    nodes = record["graph"]["nodes"]
    other_slot = next((slot for slot in range(len(nodes)) if slot != target_slot), None)
    if other_slot is None:
        raise ValueError("commutation requires a graph with at least two nodes")
    target = nodes[other_slot]
    intervention = record["intervention"]
    intervention["target"] = target
    intervention["formal"] = f"do({target} = {value})"
    intervention["text"] = f"Set {target} to {value}."
    provenance = dict(record.get("provenance") or {})
    provenance["l1_law_only_augmentation"] = "first_other_slot_same_value_v1"
    provenance["l1_task_supervision"] = False
    record["provenance"] = provenance
    record_id = f"{example.record_id}::l1-commute-{other_slot}-{value}"
    record["id"] = record_id
    return BenchmarkExample(
        record, record_id, example.source, example.label,
        example.intervention_kind, example.graph_group_id, example.world_group_id,
    )


def synthetic_opposite_example(
    example: BenchmarkExample, target_slot: int, value: int,
) -> BenchmarkExample:
    """Create the preregistered law-only opposite-value instruction."""

    record = copy.deepcopy(example.record)
    target = record["graph"]["nodes"][target_slot]
    opposite = 1 - value
    intervention = record["intervention"]
    intervention["target"] = target
    intervention["value"] = opposite
    intervention["value_token"] = "yes" if opposite else "no"
    intervention["formal"] = f"do({target} = {opposite})"
    intervention["text"] = f"Set {target} to {opposite}."
    provenance = dict(record.get("provenance") or {})
    provenance["l1_law_only_augmentation"] = "same_target_opposite_value_v1"
    provenance["l1_task_supervision"] = False
    record["provenance"] = provenance
    record_id = f"{example.record_id}::l1-opposite-{target_slot}-{opposite}"
    record["id"] = record_id
    return BenchmarkExample(
        record, record_id, example.source, example.label,
        example.intervention_kind, example.graph_group_id, example.world_group_id,
    )

class L1PairedCollator:
    """Collate aligned base, different-target, and opposite-value records."""

    def __init__(
        self,
        collator: Callable[[Sequence[BenchmarkExample]], dict[str, Any]],
        law_collator: Callable[[Sequence[BenchmarkExample]], dict[str, Any]] | None = None,
    ):
        self.collator = collator
        self.law_collator = law_collator or collator

    def __call__(self, rows: Sequence[L1PairedExample]) -> dict[str, dict[str, Any]]:
        if not rows:
            raise ValueError("cannot collate an empty L1 batch")
        return {
            "base": self.collator([row.base for row in rows]),
            "commuting": self.law_collator([row.commuting for row in rows]),
            "opposite": self.law_collator([row.opposite for row in rows]),
        }


__all__ = [
    "L1PairedExample", "L1PairedDataset", "L1PairedCollator",
    "binary_intervention", "synthetic_commuting_example", "synthetic_opposite_example",
]
