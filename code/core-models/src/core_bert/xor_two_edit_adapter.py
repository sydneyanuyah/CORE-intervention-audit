"""Fail-closed adapter for authoritative 30-variable XOR two-edit artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .two_edit import OrderedTwoEditExample, build_ordered_two_edit_examples


DEVELOPMENT_SPLITS = frozenset({"train", "validation"})
XOR_WORDING_PROTOCOL = "xor_intervention_wording_v1"
XOR_WORDING_FAMILIES = {
    "train": "set_target_to_value",
    "validation": "assign_value_to_variable",
}


@dataclass(frozen=True)
class XORTwoEditBundle:
    graph_group_id: str
    split: str
    nodes: tuple[str, ...]
    records: Mapping[str, Mapping[str, Any]]
    examples: tuple[OrderedTwoEditExample, ...]


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load required XOR artifact {path}") from error


def _manifest(path: Path) -> list[Mapping[str, Any]]:
    text = path.read_text(encoding="utf-8")
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        decoded = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(decoded, Mapping):
        # A one-line JSONL manifest is also a valid standalone JSON object.
        decoded = [decoded]
    if not isinstance(decoded, list) or not decoded or not all(
        isinstance(row, Mapping) for row in decoded
    ):
        raise ValueError("authoritative two-edit manifest must be a non-empty JSON/JSONL list")
    return list(decoded)


def _xor_state(
    graph: Mapping[str, Any], roots: Mapping[str, Any], interventions: Mapping[str, Any]
) -> dict[str, int]:
    nodes, root_nodes = graph.get("nodes"), graph.get("root_nodes")
    parents, biases = graph.get("parents"), graph.get("xor_bias")
    if (
        not isinstance(nodes, list) or len(nodes) != 30 or len(set(nodes)) != 30
        or not all(isinstance(node, str) and node for node in nodes)
    ):
        raise ValueError("XOR evaluation requires exactly 30 unique named nodes")
    if not isinstance(root_nodes, list) or not set(root_nodes) <= set(nodes):
        raise ValueError("graph.root_nodes must be a subset of graph.nodes")
    if not isinstance(parents, Mapping) or not isinstance(biases, Mapping):
        raise ValueError("graph must contain parents and xor_bias mappings")
    if not set(interventions) <= set(nodes):
        raise ValueError("intervention target is outside graph.nodes")
    state: dict[str, int] = {}
    for node in nodes:
        if node in interventions:
            value = interventions[node]
        elif node in root_nodes:
            if node not in roots:
                raise ValueError(f"missing root value for {node}")
            value = roots[node]
        else:
            node_parents = parents.get(node)
            if not isinstance(node_parents, list) or any(
                parent not in state for parent in node_parents
            ):
                raise ValueError(f"parents for {node} must precede it in graph.nodes")
            if node not in biases:
                raise ValueError(f"missing xor_bias for {node}")
            value = int(biases[node])
            for parent in node_parents:
                value ^= state[parent]
        if value not in (0, 1, False, True):
            raise ValueError(f"non-binary XOR state for {node}")
        state[node] = int(value)
    return state


def _component_record(
    raw: Mapping[str, Any], world: Mapping[str, Any], graph: Mapping[str, Any]
) -> dict[str, Any]:
    record_id = raw.get("record_id")
    intervention = raw.get("intervention")
    passage = world.get("passage")
    if not isinstance(record_id, str) or not isinstance(intervention, Mapping):
        raise ValueError("single-edit record requires record_id and intervention")
    target, value = intervention.get("target"), intervention.get("value")
    if not isinstance(target, str) or value not in (0, 1, False, True):
        raise ValueError(f"{record_id}: intervention must have binary target/value")
    if not isinstance(passage, str) or not passage:
        raise ValueError(f"{record_id}: world passage is missing")
    # Use the earliest exact assignment/parent mention so the gold address is
    # retained under the registered 512-token right-truncation policy.
    start = passage.find(target)
    if start < 0:
        raise ValueError(f"{record_id}: target is absent from the factual passage")
    text = intervention.get("text")
    if not isinstance(text, str) or not text:
        raise ValueError(f"{record_id}: intervention text is missing")
    split = world.get("split")
    if split not in XOR_WORDING_FAMILIES:
        raise ValueError(f"{record_id}: XOR wording protocol allows train/validation only")
    wording_family = XOR_WORDING_FAMILIES[split]
    rendered_text = (
        f"Set {target} to {int(value)}."
        if split == "train" else
        f"Assign value {int(value)} to variable {target}."
    )
    nodes = graph["nodes"]
    edges = graph.get("edges")
    if not isinstance(edges, list):
        raise ValueError("graph.edges must be a list")
    roots = world.get("root_values")
    if not isinstance(roots, Mapping):
        raise ValueError(f"{record_id}: world root_values are missing")
    factual_state = _xor_state(graph, roots, {})
    intervened_state = _xor_state(graph, roots, {target: int(value)})
    if raw.get("intervened_state") != intervened_state:
        raise ValueError(f"{record_id}: stored single-edit truth disagrees with executable SCM")
    # Keep the factual world/query identical across component records. The
    # question-bearing probe identifies the addressed target slot explicitly.
    question = "Is the intervention target equal to one after the intervention?"
    probes = [
        {
            "variable": node,
            "question": question if node == target else None,
            "answer_before": factual_state[node],
            "answer_after": intervened_state[node],
            "actually_changed": intervened_state[node] != factual_state[node],
        }
        for node in nodes
    ]
    return {
        "id": record_id,
        "source": "xor",
        "graph_id": graph.get("seed"),
        "world_id": world.get("world_id"),
        "structure_kind": "dag",
        "graph": {"nodes": list(nodes), "edges": edges},
        "chain": None,
        "factual": {
            "passage": passage,
            "question": question,
            "state": factual_state,
        },
        "intervention": {
            "kind": "value_set",
            "target": target,
            "target_text": target,
            "target_span": [start, start + len(target)],
            "value": int(value),
            "value_token": str(int(value)),
            "replacement_span": None,
            "text": rendered_text,
            "wording_protocol": XOR_WORDING_PROTOCOL,
            "wording_family": wording_family,
            "semantic_key": f"value_set:{target}:{int(value)}",
        },
        "intervened": {
            "answer": "yes" if intervened_state[target] == 1 else "no",
            "state": intervened_state,
        },
        "probes": probes,
    }


def load_xor_two_edit_bundle(
    artifact_directory: Path, manifest_path: Path, *, split: str
) -> XORTwoEditBundle:
    """Load and re-execute truth for one graph without exposing test rows."""

    if split not in DEVELOPMENT_SPLITS:
        raise ValueError(
            "XOR two-edit evaluation is restricted to train/validation; test is blocked"
        )
    graph = _json(artifact_directory / "graph.json")
    worlds = _json(artifact_directory / "worlds.json")
    raw_records = _json(artifact_directory / "records_real.json")
    rows = _manifest(manifest_path)
    if (
        not isinstance(graph, Mapping)
        or not isinstance(worlds, list)
        or not isinstance(raw_records, list)
    ):
        raise ValueError("malformed XOR graph/world/record artifacts")
    nodes = graph.get("nodes")
    # Validate graph independently before trusting stored states.
    _xor_state(graph, {}, {node: 0 for node in nodes} if isinstance(nodes, list) else {})
    seed = graph.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("graph.seed must be an integer")
    graph_group = f"xor:graph:{seed}"

    world_by_id: dict[str, Mapping[str, Any]] = {}
    for world in worlds:
        if not isinstance(world, Mapping) or not isinstance(world.get("world_id"), str):
            raise ValueError("each XOR world requires a world_id")
        world_id = world["world_id"]
        if world_id in world_by_id:
            raise ValueError(f"duplicate XOR world {world_id}")
        if world.get("split") not in {"train", "validation", "test"}:
            raise ValueError(f"{world_id}: invalid split")
        world_by_id[world_id] = world

    raw_by_id: dict[str, Mapping[str, Any]] = {}
    train_semantics: set[tuple[str, int]] = set()
    for raw in raw_records:
        if not isinstance(raw, Mapping) or not isinstance(raw.get("record_id"), str):
            raise ValueError("each single-edit record requires record_id")
        record_id = raw["record_id"]
        if record_id in raw_by_id:
            raise ValueError(f"duplicate single-edit record {record_id}")
        raw_by_id[record_id] = raw
        intervention = raw.get("intervention")
        if raw.get("split") == "train" and isinstance(intervention, Mapping):
            target, value = intervention.get("target"), intervention.get("value")
            if isinstance(target, str) and value in (0, 1, False, True):
                train_semantics.add((target, int(value)))

    selected = [row for row in rows if row.get("split") == split]
    if not selected:
        raise ValueError(f"manifest contains no {split} two-edit rows")
    canonical_records: dict[str, Mapping[str, Any]] = {}
    groups: dict[str, tuple[str, str]] = {}
    for row in selected:
        provenance = row.get("truth_provenance")
        if (
            not isinstance(provenance, Mapping)
            or provenance.get("truth_source") != "recomputed executable XOR structural equations"
            or provenance.get("a1_evidence") is not False
        ):
            raise ValueError("manifest lacks authoritative executable-SCM truth provenance")
        if row.get("source") != "xor" or row.get("graph_group_id") != graph_group:
            raise ValueError("manifest source/graph group does not match original graph")
        first_id, second_id = row.get("first_record_id"), row.get("second_record_id")
        if first_id not in raw_by_id or second_id not in raw_by_id:
            raise ValueError("manifest references an unknown single-edit record")
        first_raw, second_raw = raw_by_id[first_id], raw_by_id[second_id]
        world_id = first_raw.get("world_id")
        if world_id != second_raw.get("world_id") or world_id not in world_by_id:
            raise ValueError("two-edit components must share a known original world")
        world = world_by_id[world_id]
        if (
            world.get("split") != split
            or first_raw.get("split") != split
            or second_raw.get("split") != split
        ):
            raise ValueError("manifest/component/world split mismatch")
        expected_world_group = f"{graph_group}:world:{world_id}"
        if row.get("world_group_id") != expected_world_group:
            raise ValueError("manifest world group does not match original world")
        roots, stored_factual = world.get("root_values"), world.get("factual_state")
        if not isinstance(roots, Mapping) or not isinstance(stored_factual, Mapping):
            raise ValueError("world lacks root values or factual state")
        factual = _xor_state(graph, roots, {})
        if factual != dict(stored_factual):
            raise ValueError("stored factual state disagrees with executable SCM")

        interventions: dict[str, int] = {}
        for raw in (first_raw, second_raw):
            intervention = raw.get("intervention")
            if not isinstance(intervention, Mapping):
                raise ValueError("selected single-edit record lacks an intervention object")
            target, value = intervention.get("target"), intervention.get("value")
            if not isinstance(target, str) or value not in (0, 1, False, True):
                raise ValueError("selected single-edit intervention is not binary")
            semantic = (target, int(value))
            if semantic not in train_semantics:
                raise ValueError(
                    f"component edit {semantic!r} was not seen separately in training"
                )
            expected_single = _xor_state(graph, roots, {target: int(value)})
            if raw.get("intervened_state") != expected_single:
                raise ValueError("stored single-edit truth disagrees with executable SCM")
            # Assignment is intentionally ordered: a repeated target is last-write-wins.
            interventions[target] = int(value)
        gold_state = _xor_state(graph, roots, interventions)
        expected_factual = [factual[node] for node in nodes]
        expected_gold = [gold_state[node] for node in nodes]
        expected_change = [
            after != before for after, before in zip(expected_gold, expected_factual)
        ]
        expected_target = [node in interventions for node in nodes]
        expected = {
            "factual_outputs": expected_factual,
            "gold_outputs": expected_gold,
            "change_mask": expected_change,
            "preservation_mask": [not flag for flag in expected_change],
            "target_mask": expected_target,
        }
        if any(row.get(key) != value for key, value in expected.items()):
            raise ValueError("manifest two-edit truth disagrees with executable SCM")
        for raw in (first_raw, second_raw):
            record_id = raw["record_id"]
            canonical_records[record_id] = _component_record(raw, world, graph)
            groups[record_id] = (graph_group, expected_world_group)

    examples = build_ordered_two_edit_examples(
        canonical_records,
        groups,
        selected,
        # Semantic train exposure was checked above; these are the concrete
        # held-out-world carrier records required by the generic pair builder.
        seen_separately=set(canonical_records),
    )
    return XORTwoEditBundle(
        graph_group, split, tuple(nodes), canonical_records, examples
    )


__all__ = [
    "DEVELOPMENT_SPLITS", "XOR_WORDING_FAMILIES", "XOR_WORDING_PROTOCOL",
    "XORTwoEditBundle", "load_xor_two_edit_bundle",
]
