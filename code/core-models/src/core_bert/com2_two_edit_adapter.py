"""Fail-closed adapter for adjudicated Com2 ordered two-edit truth."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .two_edit import OrderedTwoEditExample, build_ordered_two_edit_examples, ordered_pair_id


QUEUE_SHA256 = "not-published"
TRUTH_SHA256 = "not-published"
PROVENANCE_SHA256 = "not-published"


@dataclass(frozen=True)
class Com2TwoEditBundle:
    split: str
    records: Mapping[str, Mapping[str, Any]]
    examples: tuple[OrderedTwoEditExample, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path}: every JSONL row must be an object")
    return rows


def _accepted_records(data_root: Path, split: str) -> dict[str, dict[str, Any]]:
    ids: set[str] = set()
    for source in ("com2_intervention", "com2_counterfactual"):
        ids.update((data_root / "splits" / f"{source}.{split}.txt").read_text().splitlines())
    records: dict[str, dict[str, Any]] = {}
    for name in ("com2_intervention.jsonl", "com2_counterfactual.jsonl"):
        for row in _rows(data_root / "records" / name):
            if row.get("id") in ids:
                records[row["id"]] = row
    if set(records) != ids:
        raise ValueError(f"accepted Com2 {split} records do not match their split manifest")
    return records


def _event_index(target: Any, chain: list[str]) -> int:
    if not isinstance(target, str):
        raise ValueError("Com2 target must be an occurrence-qualified event")
    for index, event in enumerate(chain):
        if target == f"event_{index:02d}: {event}":
            return index
    raise ValueError(f"Com2 target {target!r} does not identify its chain occurrence")


def _component(
    queue: Mapping[str, Any], source: Mapping[str, Any], edit: Mapping[str, Any],
    outputs: Mapping[str, str], ordinal: int,
) -> dict[str, Any]:
    first, second = queue["first_edit"], queue["second_edit"]
    passage = (
        queue["factual_passage"]
        + f"\n[ADDRESS-1] {first['target_text']}\n[REPLACEMENT-1] {first['value']}"
        + f"\n[ADDRESS-2] {second['target_text']}\n[REPLACEMENT-2] {second['value']}"
    )
    target_marker = f"[ADDRESS-{ordinal}] "
    value_marker = f"[REPLACEMENT-{ordinal}] "
    target_start = passage.index(target_marker) + len(target_marker)
    value_start = passage.index(value_marker) + len(value_marker)
    intervention = dict(edit)
    intervention.update({
        "value_token": None,
        "target_span": [target_start, target_start + len(edit["target_text"])],
        "replacement_span": [value_start, value_start + len(edit["value"])],
    })
    chain = list(queue["chain"])
    names = [f"event_{index:02d}: {event}" for index, event in enumerate(chain)]
    target_index = _event_index(edit["target"], chain)
    return {
        "id": f"manual-com2:{queue['pair_id']}:{ordinal}",
        "source": "com2", "source_id": source.get("source_id"),
        "structure_kind": "chain", "chain": chain, "graph": None,
        "factual": {
            "passage": passage, "question": queue["factual_question"],
            "answer": None, "state": None,
        },
        "intervention": intervention,
        "intervened": {
            "answer": outputs[f"event_{len(chain) - 1:02d}"],
            "passage": None, "state": None,
        },
        "probes": [
            {
                "variable": name, "question": queue["factual_question"] if index == len(chain) - 1 else None,
                "answer_before": chain[index], "answer_after": outputs[f"event_{index:02d}"],
                "actually_changed": outputs[f"event_{index:02d}"] != chain[index],
                "required": "may change" if outputs[f"event_{index:02d}"] != chain[index] else "must not change",
            }
            for index, name in enumerate(names)
        ],
        "provenance": {
            "protocol": "independent_double_annotation_with_blinded_adjudication_v1",
            "component_ordinal": ordinal, "target_index": target_index,
            "test_evaluated": False,
        },
    }


def load_com2_two_edit_bundle(
    queue_path: Path, truth_path: Path, provenance_path: Path, data_root: Path,
    *, split: str,
) -> Com2TwoEditBundle:
    if split != "validation":
        raise ValueError("manual Com2 two-edit evaluation is validation only; test is blocked")
    expected = (
        (queue_path, QUEUE_SHA256), (truth_path, TRUTH_SHA256),
        (provenance_path, PROVENANCE_SHA256),
    )
    for path, digest in expected:
        if _sha256(path) != digest:
            raise ValueError(f"authoritative Com2 artifact SHA-256 mismatch: {path}")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("test_evaluated") is not False or provenance.get("sources", {}).get("com2") != 70:
        raise ValueError("authoritative Com2 provenance is incomplete or test-contaminated")
    queue_rows = [row for row in _rows(queue_path) if row.get("source") == "com2"]
    truth_rows = [row for row in _rows(truth_path) if row.get("source") == "com2"]
    queue = {row.get("pair_id"): row for row in queue_rows}
    truth = {row.get("pair_id"): row for row in truth_rows}
    if len(queue) != 70 or set(queue) != set(truth):
        raise ValueError("Com2 queue and truth must contain the same 70 unique pair IDs")
    validation = _accepted_records(data_root, "validation")
    training = _accepted_records(data_root, "train")
    # The registered trainer adds factual identity/restoration interventions at
    # every position of training chains. This uses no validation text or label.
    exposed_positions = {
        position for row in training.values() for position in range(len(row["chain"]))
    }
    group_rows = {
        row["record_id"]: row for row in _rows(data_root / "groups" / "com2.jsonl")
    }
    records: dict[str, Mapping[str, Any]] = {}
    manifest = []
    groups: dict[str, tuple[str, str]] = {}
    for manual_id in sorted(queue):
        item, gold = queue[manual_id], truth[manual_id]
        source_id = item.get("source_record_id")
        if item.get("split") != split or source_id not in validation:
            raise ValueError(f"{manual_id}: source is outside frozen Com2 validation")
        source = validation[source_id]
        source_passage = source.get("factual", {}).get("passage")
        base_passage = source_passage.split("\n[ORIG]", 1)[0] if isinstance(source_passage, str) else source_passage
        if (
            source.get("chain") != item.get("chain")
            or base_passage != item.get("factual_passage")
            or source.get("factual", {}).get("question") != item.get("factual_question")
        ):
            raise ValueError(f"{manual_id}: queue factual world disagrees with accepted source")
        public_first = {key: source["intervention"].get(key) for key in item["first_edit"]}
        if public_first != item["first_edit"]:
            raise ValueError(f"{manual_id}: first edit disagrees with accepted source record")
        positions = [_event_index(edit["target"], item["chain"]) for edit in (item["first_edit"], item["second_edit"])]
        if positions[0] == positions[1] or any(position not in exposed_positions for position in positions):
            raise ValueError(f"{manual_id}: edit positions must be distinct and train-exposed")
        keys = [f"event_{index:02d}" for index in range(len(item["chain"]))]
        if (
            gold.get("status") != "accepted" or gold.get("split") != split
            or gold.get("first_edit_valid") is not True or gold.get("second_edit_valid") is not True
            or set(gold.get("intermediate_outputs", {})) != set(keys)
            or set(gold.get("gold_final_outputs", {})) != set(keys)
        ):
            raise ValueError(f"{manual_id}: authoritative Com2 truth is incomplete")
        final = gold["gold_final_outputs"]
        factual = dict(zip(keys, item["chain"]))
        changed = [final[key] != factual[key] for key in keys]
        changed_names = [key for key, flag in zip(keys, changed) if flag]
        preserved_names = [key for key, flag in zip(keys, changed) if not flag]
        expected_targets = [f"event_{position:02d}" for position in positions]
        truth_provenance = gold.get("truth_provenance")
        if (
            gold.get("actually_changed_variables") != changed_names
            or gold.get("preserved_variables") != preserved_names
            or gold.get("target_variables") != expected_targets
            or gold.get("query_answer_after_both_edits") != final[keys[-1]]
            or not isinstance(truth_provenance, Mapping)
            or truth_provenance.get("test_evaluated") is not False
        ):
            raise ValueError(f"{manual_id}: masks, targets, query, or provenance disagree")
        first = _component(item, source, item["first_edit"], gold["intermediate_outputs"], 1)
        second = _component(item, source, item["second_edit"], final, 2)
        sidecar = group_rows.get(source_id)
        if not sidecar:
            raise ValueError(f"{manual_id}: missing Com2 group sidecar")
        graph_group, world_group = sidecar["graph_group_id"], sidecar["world_group_id"]
        for component in (first, second):
            records[component["id"]] = component
            groups[component["id"]] = (graph_group, world_group)
        manifest.append({
            "pair_id": ordered_pair_id("com2", graph_group, world_group, first["id"], second["id"]),
            "source": "com2", "split": split,
            "graph_group_id": graph_group, "world_group_id": world_group,
            "first_record_id": first["id"], "second_record_id": second["id"],
            "factual_outputs": [factual[key] for key in keys],
            "gold_outputs": [final[key] for key in keys],
            "change_mask": changed,
            "preservation_mask": [not flag for flag in changed],
            "target_mask": [key in expected_targets for key in keys],
        })
    examples = build_ordered_two_edit_examples(records, groups, manifest, seen_separately=set(records))
    return Com2TwoEditBundle(split, records, examples)


__all__ = ["Com2TwoEditBundle", "load_com2_two_edit_bundle"]
