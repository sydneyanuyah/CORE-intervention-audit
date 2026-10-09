"""Generate authoritative ordered two-edit truth from executable XOR SCM artifacts.

The input graph and worlds are the frozen artifacts emitted by the original
CORE XOR generator. Truth is recomputed from the stored structural equations;
it is never inferred by combining single-edit predictions or labels.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .two_edit import ordered_pair_id


def xor_scm_state(
    graph: Mapping[str, Any],
    roots: Mapping[str, Any],
    interventions: Mapping[str, Any] | None = None,
) -> dict[str, int]:
    nodes = graph.get("nodes")
    root_nodes = graph.get("root_nodes")
    parents = graph.get("parents")
    bias = graph.get("xor_bias")
    if not isinstance(nodes, list) or not nodes or len(nodes) != len(set(nodes)):
        raise ValueError("graph.nodes must be a non-empty unique list")
    if not isinstance(root_nodes, list) or not set(root_nodes) <= set(nodes):
        raise ValueError("graph.root_nodes must be a subset of graph.nodes")
    if not isinstance(parents, Mapping) or not isinstance(bias, Mapping):
        raise ValueError("graph must expose parents and xor_bias mappings")
    interventions = dict(interventions or {})
    if not set(interventions) <= set(nodes):
        raise ValueError("intervention targets must occur in graph.nodes")
    state: dict[str, int] = {}
    roots_set = set(root_nodes)
    for node in nodes:
        if not isinstance(node, str) or not node:
            raise ValueError("graph node names must be non-empty strings")
        if node in interventions:
            value = interventions[node]
        elif node in roots_set:
            if node not in roots:
                raise ValueError(f"missing root assignment for {node!r}")
            value = roots[node]
        else:
            node_parents = parents.get(node)
            if not isinstance(node_parents, list) or any(parent not in state for parent in node_parents):
                raise ValueError(f"parents for {node!r} must precede it in graph.nodes")
            if node not in bias:
                raise ValueError(f"missing xor_bias for {node!r}")
            value = bias[node]
            for parent in node_parents:
                value = int(value) ^ state[parent]
        if value not in (0, 1, False, True):
            raise ValueError(f"XOR value for {node!r} must be binary")
        state[node] = int(value)
    return state


def generate_xor_two_edit_manifest(
    graph: Mapping[str, Any],
    worlds: Sequence[Mapping[str, Any]],
    single_edit_records: Sequence[Mapping[str, Any]],
    *,
    graph_group_id: str,
) -> list[dict[str, Any]]:
    """Reproduce the registered ordered-pair schedule with validated SCM truth."""

    if not isinstance(graph_group_id, str) or not graph_group_id:
        raise ValueError("graph_group_id must be a non-empty string")
    nodes = graph.get("nodes")
    if not isinstance(nodes, list) or len(nodes) < 8:
        raise ValueError("registered XOR pair schedule requires at least eight nodes")
    records: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for record in single_edit_records:
        world_id = record.get("world_id")
        intervention = record.get("intervention")
        if not isinstance(world_id, str) or not isinstance(intervention, Mapping):
            raise ValueError("single-edit records require world_id and intervention")
        target, value = intervention.get("target"), intervention.get("value")
        if not isinstance(target, str) or value not in (0, 1, False, True):
            raise ValueError("single-edit interventions require a target and binary value")
        key = (world_id, target, int(value))
        if key in records:
            raise ValueError(f"duplicate single-edit semantic key {key!r}")
        records[key] = record

    rows: list[dict[str, Any]] = []
    for world in worlds:
        world_id, split = world.get("world_id"), world.get("split")
        roots, stored_factual = world.get("root_values"), world.get("factual_state")
        if not isinstance(world_id, str) or split not in {"train", "validation", "test"}:
            raise ValueError("world requires world_id and a registered split")
        if not isinstance(roots, Mapping) or not isinstance(stored_factual, Mapping):
            raise ValueError("world requires root_values and factual_state")
        factual = xor_scm_state(graph, roots)
        if factual != dict(stored_factual):
            raise ValueError(f"stored factual state disagrees with SCM for {world_id}")
        world_group_id = f"{graph_group_id}:world:{world_id}"
        for first_index in range(0, len(nodes), 3):
            second_index = (first_index + 7) % len(nodes)
            first, second = nodes[first_index], nodes[second_index]
            first_value, second_value = 1 - factual[first], 1 - factual[second]
            first_record = records.get((world_id, first, first_value))
            second_record = records.get((world_id, second, second_value))
            if first_record is None or second_record is None:
                raise ValueError(f"missing scheduled single-edit component for {world_id}")
            first_id, second_id = first_record.get("record_id"), second_record.get("record_id")
            if not isinstance(first_id, str) or not isinstance(second_id, str):
                raise ValueError("single-edit components require record_id")
            for record, target, value in (
                (first_record, first, first_value), (second_record, second, second_value)
            ):
                expected = xor_scm_state(graph, roots, {target: value})
                if record.get("intervened_state") != expected:
                    raise ValueError(f"single-edit truth disagrees with SCM for {record.get('record_id')}")
            gold_state = xor_scm_state(
                graph, roots, {first: first_value, second: second_value}
            )
            gold = [gold_state[node] for node in nodes]
            before = [factual[node] for node in nodes]
            change = [after != prior for after, prior in zip(gold, before)]
            target = [node in {first, second} for node in nodes]
            rows.append(
                {
                    "pair_id": ordered_pair_id(
                        "xor", graph_group_id, world_group_id, first_id, second_id
                    ),
                    "source": "xor",
                    "split": split,
                    "graph_group_id": graph_group_id,
                    "world_group_id": world_group_id,
                    "first_record_id": first_id,
                    "second_record_id": second_id,
                    "gold_outputs": gold,
                    "factual_outputs": before,
                    "change_mask": change,
                    "preservation_mask": [not flag for flag in change],
                    "target_mask": target,
                    "truth_provenance": {
                        "generator": "core_bert.xor_two_edit.generate_xor_two_edit_manifest",
                        "truth_source": "recomputed executable XOR structural equations",
                        "a1_evidence": False,
                    },
                }
            )
    return sorted(rows, key=lambda row: row["pair_id"])


def generate_xor_two_edit_manifest_from_directory(directory: Path) -> list[dict[str, Any]]:
    graph = json.loads((directory / "graph.json").read_text(encoding="utf-8"))
    worlds = json.loads((directory / "worlds.json").read_text(encoding="utf-8"))
    records = json.loads((directory / "records_real.json").read_text(encoding="utf-8"))
    seed = graph.get("seed")
    if not isinstance(seed, int):
        raise ValueError("graph.seed must be an integer")
    return generate_xor_two_edit_manifest(
        graph, worlds, records, graph_group_id=f"xor:graph:{seed}"
    )


__all__ = [
    "generate_xor_two_edit_manifest",
    "generate_xor_two_edit_manifest_from_directory",
    "xor_scm_state",
]
