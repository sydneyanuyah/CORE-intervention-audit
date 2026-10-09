"""Fail-closed adapter for independently adjudicated WIQA two-edit truth."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .two_edit import OrderedTwoEditExample, build_ordered_two_edit_examples, ordered_pair_id


QUEUE_SHA256 = "not-published"
TRUTH_SHA256 = "not-published"
VALUES = {"less": -1, "no_effect": 0, "more": 1}
VALUE_NAMES = {value: name for name, value in VALUES.items()}


@dataclass(frozen=True)
class WiqaTwoEditBundle:
    split: str
    records: Mapping[str, Mapping[str, Any]]
    examples: tuple[OrderedTwoEditExample, ...]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path) -> list[Mapping[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load WIQA two-edit input {path}") from error
    if not rows or not all(isinstance(row, Mapping) for row in rows):
        raise ValueError(f"{path} must contain a non-empty JSONL object sequence")
    return rows


def _solve(row: Mapping[str, Any], edits: list[Mapping[str, Any]]) -> dict[str, str]:
    graph = row.get("graph")
    if not isinstance(graph, Mapping):
        raise ValueError("WIQA queue row lacks a graph")
    nodes, signed = graph.get("nodes"), graph.get("signed_edges")
    if not isinstance(nodes, list) or not nodes or len(nodes) != len(set(nodes)):
        raise ValueError("WIQA graph requires unique nodes")
    incoming: dict[str, list[tuple[str, int]]] = {node: [] for node in nodes}
    if not isinstance(signed, list):
        raise ValueError("WIQA graph lacks signed edges")
    for edge in signed:
        if not isinstance(edge, Mapping):
            raise ValueError("malformed WIQA signed edge")
        source, target, sign = edge.get("source"), edge.get("target"), edge.get("sign")
        if source not in incoming or target not in incoming or sign not in {"positive", "negative"}:
            raise ValueError("WIQA signed edge references invalid nodes or sign")
        incoming[target].append((source, 1 if sign == "positive" else -1))
    clamps: dict[str, int] = {}
    for edit in edits:
        target, value = edit.get("target"), edit.get("value_token")
        if target not in incoming or value not in {"less", "more"}:
            raise ValueError("WIQA edit must clamp a declared node to less or more")
        clamps[target] = VALUES[value]
    state = dict(clamps)
    while len(state) < len(nodes):
        progressed = False
        for node in nodes:
            if node in state or any(parent not in state for parent, _ in incoming[node]):
                continue
            effects = {state[parent] * sign for parent, sign in incoming[node] if state[parent]}
            if len(effects) > 1:
                raise ValueError(f"conflicting signed paths at WIQA node {node}")
            state[node] = next(iter(effects), 0)
            progressed = True
        if not progressed:
            raise ValueError("WIQA graph cannot be solved in topological order")
    return {node: VALUE_NAMES[state[node]] for node in nodes}


def _load_source_records(data_root: Path, split: str) -> tuple[dict[str, Mapping[str, Any]], set[tuple[str, str]]]:
    selected_ids = set((data_root / "splits" / f"wiqa.{split}.txt").read_text().split())
    train_ids = set((data_root / "splits" / "wiqa.train.txt").read_text().split())
    records: dict[str, Mapping[str, Any]] = {}
    train_semantics: set[tuple[str, str]] = set()
    for record in _rows(data_root / "records" / "wiqa.jsonl"):
        record_id = record.get("id")
        if record_id in selected_ids:
            records[record_id] = record
        if record_id in train_ids:
            intervention = record.get("intervention", {})
            train_semantics.add((intervention.get("target"), intervention.get("value_token")))
    if set(records) != selected_ids:
        raise ValueError("WIQA split IDs do not resolve exactly to accepted records")
    return records, train_semantics


def _query_node(record: Mapping[str, Any]) -> str:
    queried = [
        probe.get("variable") for probe in record.get("probes", [])
        if isinstance(probe, Mapping) and isinstance(probe.get("question"), str)
        and probe["question"].strip()
    ]
    if len(queried) != 1 or not isinstance(queried[0], str):
        raise ValueError("WIQA source record must expose exactly one queried probe")
    return queried[0]


def _component(
    queue: Mapping[str, Any], source: Mapping[str, Any], edit: Mapping[str, Any],
    output: Mapping[str, str], ordinal: int, query_node: str,
) -> dict[str, Any]:
    first_text = queue["first_edit"]["target_text"]
    second_text = queue["second_edit"]["target_text"]
    passage = queue["factual_passage"] + f"\n[ADDRESS-1] {first_text}\n[ADDRESS-2] {second_text}"
    marker = f"[ADDRESS-{ordinal}] "
    start = passage.index(marker) + len(marker)
    target_text = edit.get("target_text")
    if passage[start:start + len(target_text)] != target_text:
        raise ValueError("constructed WIQA address does not exactly match target text")
    intervention = dict(edit)
    intervention["target_span"] = [start, start + len(target_text)]
    intervention["replacement_span"] = None
    nodes = queue["graph"]["nodes"]
    return {
        "id": f"manual-wiqa:{queue['pair_id']}:{ordinal}",
        "source": "wiqa", "source_id": source.get("source_id"),
        "structure_kind": "dag", "chain": None,
        "graph": queue["graph"],
        "factual": {"passage": passage, "question": queue["factual_question"], "answer": None, "state": {node: "no_effect" for node in nodes}},
        "intervention": intervention,
        "intervened": {"answer": output[query_node], "passage": None, "state": dict(output)},
        "probes": [
            {
                "variable": node,
                "question": queue["factual_question"] if node == query_node else None,
                "answer_before": "no_effect", "answer_after": output[node],
                "actually_changed": output[node] != "no_effect",
                "required": "may change" if output[node] != "no_effect" else "must not change",
            }
            for node in nodes
        ],
        "provenance": {"protocol": "independent_double_annotation_with_blinded_adjudication_v1", "test_evaluated": False},
    }


def load_wiqa_two_edit_bundle(
    queue_path: Path, truth_path: Path, data_root: Path, *, split: str,
) -> WiqaTwoEditBundle:
    if split != "validation":
        raise ValueError("manual WIQA two-edit evaluation is validation only; test is blocked")
    if _sha256(queue_path) != QUEUE_SHA256 or _sha256(truth_path) != TRUTH_SHA256:
        raise ValueError("manual WIQA queue or authoritative-truth SHA-256 mismatch")
    queue_rows = _rows(queue_path)
    truth_rows = [row for row in _rows(truth_path) if row.get("source") == "wiqa"]
    queue_by_id = {row.get("pair_id"): row for row in queue_rows}
    truth_by_id = {row.get("pair_id"): row for row in truth_rows}
    if len(queue_by_id) != 70 or set(queue_by_id) != set(truth_by_id):
        raise ValueError("WIQA queue and truth must contain the same 70 unique pair IDs")
    sources, train_semantics = _load_source_records(data_root, split)
    group_rows = {row["record_id"]: row for row in _rows(data_root / "groups" / "wiqa.jsonl")}
    records: dict[str, Mapping[str, Any]] = {}
    groups: dict[str, tuple[str, str]] = {}
    manifest = []
    for manual_id in sorted(queue_by_id):
        queue, truth = queue_by_id[manual_id], truth_by_id[manual_id]
        source_id = queue.get("source_record_id")
        if source_id not in sources or queue.get("split") != split:
            raise ValueError(f"{manual_id}: queue source is outside the frozen validation split")
        source = sources[source_id]
        queue_provenance = queue.get("truth_protocol")
        truth_provenance = truth.get("truth_provenance")
        if (
            not isinstance(queue_provenance, Mapping)
            or not isinstance(truth_provenance, Mapping)
            or queue_provenance.get("protocol") != "wiqa_signed_executable_two_edit_v1"
            or truth_provenance.get("protocol") != "wiqa_signed_executable_two_edit_v1"
            or queue_provenance.get("test_evaluated") is not False
            or truth_provenance.get("test_evaluated") is not False
        ):
            raise ValueError(f"{manual_id}: executable WIQA truth provenance is invalid")
        if source.get("factual", {}).get("passage") != queue.get("factual_passage") or source.get("factual", {}).get("question") != queue.get("factual_question"):
            raise ValueError(f"{manual_id}: queue factual world disagrees with its accepted record")
        public_first = {key: source["intervention"].get(key) for key in queue["first_edit"]}
        if public_first != queue["first_edit"]:
            raise ValueError(f"{manual_id}: first edit disagrees with accepted WIQA record")
        for edit in (queue["first_edit"], queue["second_edit"]):
            if (edit.get("target"), edit.get("value_token")) not in train_semantics:
                raise ValueError(f"{manual_id}: edit semantic was not seen separately in training")
        intermediate = _solve(queue, [queue["first_edit"]])
        final = _solve(queue, [queue["first_edit"], queue["second_edit"]])
        if truth.get("status") != "accepted" or truth.get("split") != split or truth.get("intermediate_outputs") != intermediate or truth.get("gold_final_outputs") != final:
            raise ValueError(f"{manual_id}: authoritative WIQA labels disagree with signed propagation")
        nodes = queue["graph"]["nodes"]
        targets = [queue["first_edit"]["target"], queue["second_edit"]["target"]]
        changed = [final[node] != "no_effect" for node in nodes]
        if truth.get("target_variables") != targets or truth.get("actually_changed_variables") != [node for node, flag in zip(nodes, changed) if flag] or truth.get("preserved_variables") != [node for node, flag in zip(nodes, changed) if not flag]:
            raise ValueError(f"{manual_id}: authoritative WIQA masks disagree with final truth")
        query_node = _query_node(source)
        if truth.get("query_answer_after_both_edits") != final[query_node]:
            raise ValueError(f"{manual_id}: authoritative WIQA query answer is not grounded")
        second_only = _solve(queue, [queue["second_edit"]])
        first = _component(queue, source, queue["first_edit"], intermediate, 1, query_node)
        second = _component(queue, source, queue["second_edit"], second_only, 2, query_node)
        graph_group = group_rows[source_id]["graph_group_id"]
        world_group = group_rows[source_id]["world_group_id"]
        for component in (first, second):
            records[component["id"]] = component
            groups[component["id"]] = (graph_group, world_group)
        manifest.append({
            "pair_id": ordered_pair_id("wiqa", graph_group, world_group, first["id"], second["id"]),
            "source": "wiqa", "split": split,
            "graph_group_id": graph_group, "world_group_id": world_group,
            "first_record_id": first["id"], "second_record_id": second["id"],
            "factual_outputs": ["no_effect" for _ in nodes],
            "gold_outputs": [final[node] for node in nodes],
            "change_mask": changed, "preservation_mask": [not flag for flag in changed],
            "target_mask": [node in targets for node in nodes],
        })
    examples = build_ordered_two_edit_examples(records, groups, manifest, seen_separately=set(records))
    return WiqaTwoEditBundle(split, records, examples)


__all__ = ["WiqaTwoEditBundle", "load_wiqa_two_edit_bundle"]
