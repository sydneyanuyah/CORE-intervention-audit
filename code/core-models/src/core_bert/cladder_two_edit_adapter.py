"""Fail-closed adapter for authoritative deterministic CLadder two-edit truth."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .two_edit import OrderedTwoEditExample, build_ordered_two_edit_examples


PROTOCOL = "cladder_deterministic_two_edit_v1"
DEVELOPMENT_SPLITS = frozenset({"train", "validation"})


@dataclass(frozen=True)
class CladderTwoEditBundle:
    split: str
    records: Mapping[str, Mapping[str, Any]]
    examples: tuple[OrderedTwoEditExample, ...]


def _read_manifest(path: Path) -> list[Mapping[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load CLadder manifest {path}") from error
    if not rows or not all(isinstance(row, Mapping) for row in rows):
        raise ValueError("CLadder manifest must be a non-empty JSONL object sequence")
    return rows


def _parents(graph: Mapping[str, Any]) -> tuple[list[str], dict[str, list[str]]]:
    nodes, edges = graph.get("nodes"), graph.get("edges")
    if not isinstance(nodes, list) or not nodes or len(nodes) != len(set(nodes)):
        raise ValueError("CLadder graph requires unique nodes")
    parents = {node: [] for node in nodes}
    if not isinstance(edges, list):
        raise ValueError("CLadder graph edges must be a list")
    for edge in edges:
        if not isinstance(edge, list) or len(edge) != 2 or any(node not in parents for node in edge):
            raise ValueError("CLadder edge references malformed or unknown nodes")
        parents[edge[1]].append(edge[0])
    return nodes, parents


def _conditional(node: str, params: Mapping[str, Any]) -> tuple[Any, list[str]]:
    prefix = f"p({node} |"
    matches = [key for key in params if key.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"expected one conditional table for {node}, found {matches}")
    key = matches[0]
    order = [part.strip() for part in key.split("|", 1)[1].rstrip(")").split(",")]
    return params[key], order


def solve_cladder_world(world: Mapping[str, Any], interventions: Mapping[str, Any]) -> dict[str, int]:
    graph, model = world.get("graph"), world.get("model")
    if not isinstance(graph, Mapping) or not isinstance(model, Mapping):
        raise ValueError("CLadder world requires graph and model descriptors")
    if model.get("equation_type") != "deterministic" or not isinstance(model.get("params"), Mapping):
        raise ValueError("CLadder world must contain deterministic conditional tables")
    nodes, parents = _parents(graph)
    structure = model.get("structure")
    if not isinstance(structure, str):
        raise ValueError("CLadder world lacks a structural equation graph")
    structure_edges = {
        tuple(part.strip() for part in edge.split("->", 1))
        for edge in structure.split(",") if edge.strip()
    }
    if structure_edges != {tuple(edge) for edge in graph["edges"]}:
        raise ValueError("CLadder model structure disagrees with graph edges")
    if not set(interventions) <= set(nodes):
        raise ValueError("CLadder intervention target is outside the graph")
    roots = world.get("factual_roots")
    if not isinstance(roots, Mapping):
        raise ValueError("CLadder world lacks factual roots")
    required_roots = {node for node in nodes if not parents[node]}
    if not required_roots <= set(roots) or not set(roots) <= set(nodes):
        raise ValueError("CLadder factual assignment must cover roots within graph nodes")
    state = {name: int(value) for name, value in roots.items()}
    state.update({name: int(value) for name, value in interventions.items()})
    if any(value not in (0, 1) for value in state.values()):
        raise ValueError("CLadder state and interventions must be binary")
    params = model["params"]
    while len(state) < len(nodes):
        progressed = False
        for node in nodes:
            if node in state or any(parent not in state for parent in parents[node]):
                continue
            value, order = _conditional(node, params)
            if set(order) != set(parents[node]):
                raise ValueError(f"conditional-table parents disagree for {node}")
            for parent in order:
                value = value[state[parent]]
            if value not in (0, 1, False, True):
                raise ValueError(f"non-deterministic conditional value for {node}")
            state[node] = int(value)
            progressed = True
        if not progressed:
            raise ValueError("CLadder graph cannot be solved in topological order")
    return {node: state[node] for node in nodes}


def load_cladder_two_edit_bundle(
    artifact_path: Path, manifest_path: Path, *, split: str,
) -> CladderTwoEditBundle:
    if split not in DEVELOPMENT_SPLITS:
        raise ValueError("CLadder two-edit evaluation is train/validation only; test is blocked")
    try:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load CLadder artifact {artifact_path}") from error
    if (
        not isinstance(artifact, Mapping) or artifact.get("protocol") != PROTOCOL
        or artifact.get("test_evaluated") is not False
    ):
        raise ValueError("CLadder artifact lacks development-only executable-truth provenance")
    rows = _read_manifest(manifest_path)
    worlds_raw, components_raw = artifact.get("worlds"), artifact.get("components")
    if not isinstance(worlds_raw, list) or not isinstance(components_raw, list):
        raise ValueError("CLadder artifact requires world and component lists")
    worlds: dict[tuple[int, int], Mapping[str, Any]] = {}
    for world in worlds_raw:
        if not isinstance(world, Mapping):
            raise ValueError("malformed CLadder world")
        key = (int(world.get("model_id")), int(world.get("source_id")))
        if key in worlds or world.get("split") not in DEVELOPMENT_SPLITS:
            raise ValueError("duplicate or invalid CLadder world")
        factual = solve_cladder_world(world, {})
        if factual != world.get("factual_state"):
            raise ValueError("stored CLadder factual truth disagrees with executable SCM")
        worlds[key] = world
    components: dict[str, Mapping[str, Any]] = {}
    component_world: dict[str, Mapping[str, Any]] = {}
    train_semantics: set[tuple[str, int]] = set()
    for record in components_raw:
        if not isinstance(record, Mapping) or not isinstance(record.get("id"), str):
            raise ValueError("malformed CLadder component")
        record_id = record["id"]
        if record_id in components:
            raise ValueError("duplicate CLadder component ID")
        metadata, intervention = record.get("two_edit_metadata"), record.get("intervention")
        if not isinstance(metadata, Mapping) or not isinstance(intervention, Mapping):
            raise ValueError("CLadder component lacks metadata or intervention")
        key = (int(metadata.get("model_id")), int(record.get("source_id")))
        if key not in worlds or metadata.get("split") != worlds[key].get("split"):
            raise ValueError("CLadder component does not match a world descriptor")
        world = worlds[key]
        if record.get("graph") != world.get("graph") or record.get("factual", {}).get("state") != world.get("factual_state"):
            raise ValueError("CLadder component graph/factual state disagrees with its world")
        target, value = intervention.get("target"), intervention.get("value")
        if not isinstance(target, str) or value not in (0, 1, False, True):
            raise ValueError("CLadder component intervention must be binary")
        expected = solve_cladder_world(world, {target: int(value)})
        if record.get("intervened", {}).get("state") != expected:
            raise ValueError("stored CLadder single-edit truth disagrees with executable SCM")
        if metadata.get("split") == "train":
            train_semantics.add((target, int(value)))
        canonical = dict(record)
        canonical["factual"] = dict(record["factual"])
        canonical_question = "Is the intervention target equal to one after the intervention?"
        canonical["factual"]["question"] = canonical_question
        canonical["probes"] = [
            {**probe, "question": canonical_question if probe.get("variable") == target else None}
            for probe in record.get("probes", [])
        ]
        components[record_id] = canonical
        component_world[record_id] = world
    selected = [row for row in rows if row.get("split") == split]
    if not selected:
        raise ValueError(f"CLadder manifest contains no {split} rows")
    records: dict[str, Mapping[str, Any]] = {}
    groups: dict[str, tuple[str, str]] = {}
    for row in selected:
        provenance = row.get("truth_provenance")
        if (
            row.get("source") != "cladder" or not isinstance(provenance, Mapping)
            or provenance.get("protocol") != PROTOCOL
            or provenance.get("truth_source") != "recomputed deterministic CLadder conditional tables"
            or provenance.get("test_evaluated") is not False
        ):
            raise ValueError("CLadder manifest lacks executable development-truth provenance")
        first_id, second_id = row.get("first_record_id"), row.get("second_record_id")
        if first_id not in components or second_id not in components:
            raise ValueError("CLadder manifest references an unknown component")
        first_world, second_world = component_world[first_id], component_world[second_id]
        if first_world is not second_world or first_world.get("split") != split:
            raise ValueError("CLadder pair components must share one selected world")
        graph_group = f"cladder:model:{first_world['model_id']}"
        if row.get("graph_group_id") != graph_group or row.get("world_group_id") != first_world.get("world_group_id"):
            raise ValueError("CLadder manifest group identity disagrees with its world")
        interventions: dict[str, int] = {}
        for record_id in (first_id, second_id):
            intervention = components[record_id]["intervention"]
            semantic = (intervention["target"], int(intervention["value"]))
            if split == "validation" and semantic not in train_semantics:
                raise ValueError(f"CLadder component semantic {semantic!r} was not seen in training")
            interventions[semantic[0]] = semantic[1]
            records[record_id] = components[record_id]
            groups[record_id] = (graph_group, first_world["world_group_id"])
        factual = solve_cladder_world(first_world, {})
        gold = solve_cladder_world(first_world, interventions)
        nodes = first_world["graph"]["nodes"]
        expected_factual = [factual[node] for node in nodes]
        expected_gold = [gold[node] for node in nodes]
        changed = [after != before for after, before in zip(expected_gold, expected_factual)]
        expected = {
            "factual_outputs": expected_factual, "gold_outputs": expected_gold,
            "change_mask": changed, "preservation_mask": [not flag for flag in changed],
            "target_mask": [node in interventions for node in nodes],
        }
        if any(row.get(field) != value for field, value in expected.items()):
            raise ValueError("stored CLadder two-edit truth disagrees with executable SCM")
    examples = build_ordered_two_edit_examples(
        records, groups, selected, seen_separately=set(records),
    )
    return CladderTwoEditBundle(split, records, examples)


__all__ = ["CladderTwoEditBundle", "load_cladder_two_edit_bundle", "solve_cladder_world"]
