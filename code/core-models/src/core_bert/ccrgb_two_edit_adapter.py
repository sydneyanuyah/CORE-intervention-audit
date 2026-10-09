"""Fail-closed adapter for executable CCR.GB ordered two-edit truth."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .two_edit import OrderedTwoEditExample, build_ordered_two_edit_examples


PROTOCOL = "ccrgb_executable_two_edit_v1"
ARTIFACT_SHA256 = "not-published"
MANIFEST_SHA256 = "not-published"


@dataclass(frozen=True)
class CcrgbTwoEditBundle:
    split: str
    records: Mapping[str, Mapping[str, Any]]
    examples: tuple[OrderedTwoEditExample, ...]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def solve_ccrgb_world(world: Mapping[str, Any], interventions: Mapping[str, Any]) -> dict[str, int]:
    nodes, matrix = world.get("dag_nodes"), world.get("dag_adjacency_matrix")
    names, exogenous = world.get("exogenous_variables"), world.get("true_exogenous")
    if (
        not isinstance(nodes, list) or len(nodes) != 4 or len(nodes) != len(set(nodes))
        or not isinstance(matrix, list) or len(matrix) != len(nodes)
        or any(not isinstance(row, list) or len(row) != len(nodes) for row in matrix)
        or not isinstance(names, list) or len(names) != len(nodes)
        or not isinstance(exogenous, Mapping) or set(exogenous) != set(names)
    ):
        raise ValueError("malformed CCR.GB executable world descriptor")
    if not set(interventions) <= set(nodes):
        raise ValueError("CCR.GB intervention target is outside its world")
    state: dict[str, int] = {}
    for index, node in enumerate(nodes):
        if node in interventions:
            value = interventions[node]
            if value not in (0, 1, False, True):
                raise ValueError("CCR.GB intervention must be binary")
            state[node] = int(value)
            continue
        value = int(exogenous[names[index]])
        parents = [nodes[parent] for parent, row in enumerate(matrix) if row[index] == 1]
        if any(matrix[parent][index] not in (0, 1) for parent in range(len(nodes))):
            raise ValueError("CCR.GB adjacency matrix must be binary")
        if any(parent not in state for parent in parents):
            raise ValueError("CCR.GB nodes are not topologically ordered")
        for parent in parents:
            value = int(value and state[parent]) if index == len(nodes) - 1 else int(value or state[parent])
        state[node] = value
    return state


def _jsonl(path: Path) -> list[Mapping[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load CCR.GB manifest {path}") from error
    if not rows or not all(isinstance(row, Mapping) for row in rows):
        raise ValueError("CCR.GB manifest must be a non-empty JSONL object sequence")
    return rows


def load_ccrgb_two_edit_bundle(
    artifact_path: Path, manifest_path: Path, *, split: str,
) -> CcrgbTwoEditBundle:
    if split != "validation":
        raise ValueError("CCR.GB two-edit evaluation is validation only; test is blocked")
    if _sha256(artifact_path) != ARTIFACT_SHA256 or _sha256(manifest_path) != MANIFEST_SHA256:
        raise ValueError("CCR.GB artifact or manifest SHA-256 mismatch")
    try:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("cannot load CCR.GB executable artifact") from error
    if artifact.get("protocol") != PROTOCOL or artifact.get("test_evaluated") is not False:
        raise ValueError("CCR.GB artifact provenance is invalid")
    worlds: dict[int, Mapping[str, Any]] = {}
    for world in artifact.get("worlds", []):
        context = world.get("context_id") if isinstance(world, Mapping) else None
        if not isinstance(context, int) or context in worlds or world.get("split") not in {"train", "validation"}:
            raise ValueError("duplicate or invalid CCR.GB world descriptor")
        factual = solve_ccrgb_world(world, {})
        if factual != world.get("factual_state"):
            raise ValueError("CCR.GB stored factual truth disagrees with executable equations")
        worlds[context] = world
    if len(worlds) != 5400:
        raise ValueError("CCR.GB artifact must contain exactly 4800 train and 600 validation worlds")
    components: dict[str, Mapping[str, Any]] = {}
    train_semantics: set[tuple[int, int]] = set()
    for component in artifact.get("components", []):
        if not isinstance(component, Mapping) or not isinstance(component.get("id"), str):
            raise ValueError("malformed CCR.GB component")
        metadata, intervention = component.get("two_edit_metadata"), component.get("intervention")
        if not isinstance(metadata, Mapping) or not isinstance(intervention, Mapping):
            raise ValueError("CCR.GB component lacks executable metadata")
        context = metadata.get("context_id")
        if context not in worlds or metadata.get("protocol") != PROTOCOL or metadata.get("test_evaluated") is not False:
            raise ValueError("CCR.GB component provenance disagrees with its world")
        world = worlds[context]
        nodes = world["dag_nodes"]
        target, value = intervention.get("target"), intervention.get("value")
        if target not in nodes or value not in (0, 1, False, True):
            raise ValueError("CCR.GB component intervention is invalid")
        expected = solve_ccrgb_world(world, {target: int(value)})
        if component.get("graph", {}).get("nodes") != nodes or component.get("factual", {}).get("state") != world["factual_state"] or component.get("intervened", {}).get("state") != expected:
            raise ValueError("CCR.GB component truth disagrees with executable equations")
        if metadata.get("split") == "train":
            train_semantics.add((nodes.index(target), int(value)))
        if component["id"] in components:
            raise ValueError("duplicate CCR.GB component ID")
        components[component["id"]] = component
    rows = _jsonl(manifest_path)
    records: dict[str, Mapping[str, Any]] = {}
    groups: dict[str, tuple[str, str]] = {}
    selected = []
    for row in rows:
        provenance = row.get("truth_provenance")
        if row.get("source") != "ccrgb" or row.get("split") != split or not isinstance(provenance, Mapping) or provenance.get("protocol") != PROTOCOL or provenance.get("test_evaluated") is not False:
            raise ValueError("CCR.GB manifest provenance is invalid")
        first_id, second_id = row.get("first_record_id"), row.get("second_record_id")
        if first_id not in components or second_id not in components:
            raise ValueError("CCR.GB manifest references an unknown component")
        first, second = components[first_id], components[second_id]
        context = first["two_edit_metadata"]["context_id"]
        if second["two_edit_metadata"]["context_id"] != context or worlds[context]["split"] != split:
            raise ValueError("CCR.GB pair components do not share one validation world")
        world = worlds[context]
        nodes = world["dag_nodes"]
        interventions = {}
        for component in (first, second):
            intervention = component["intervention"]
            semantic = (nodes.index(intervention["target"]), int(intervention["value"]))
            if semantic not in train_semantics:
                raise ValueError("CCR.GB component role/value semantic was not seen in training")
            interventions[intervention["target"]] = int(intervention["value"])
            records[component["id"]] = component
        expected = solve_ccrgb_world(world, interventions)
        factual = world["factual_state"]
        expected_fields = {
            "factual_outputs": [factual[node] for node in nodes],
            "gold_outputs": [expected[node] for node in nodes],
            "change_mask": [expected[node] != factual[node] for node in nodes],
            "preservation_mask": [expected[node] == factual[node] for node in nodes],
            "target_mask": [node in interventions for node in nodes],
        }
        if any(row.get(key) != value for key, value in expected_fields.items()):
            raise ValueError("CCR.GB composed truth disagrees with executable equations")
        graph_group = f"ccrgb:context:{context}"
        world_group = f"ccrgb:context:{context}:world:{world['sample_id']}"
        if row.get("graph_group_id") != graph_group or row.get("world_group_id") != world_group:
            raise ValueError("CCR.GB manifest group identity is invalid")
        groups[first_id] = groups[second_id] = (graph_group, world_group)
        selected.append(row)
    examples = build_ordered_two_edit_examples(records, groups, selected, seen_separately=set(records))
    return CcrgbTwoEditBundle(split, records, examples)


__all__ = ["CcrgbTwoEditBundle", "load_ccrgb_two_edit_bundle", "solve_ccrgb_world"]
