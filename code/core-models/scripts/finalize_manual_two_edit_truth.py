#!/usr/bin/env python3
"""Merge independent agreement and completed adjudication into final manual truth."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


STRUCTURAL = (
    "gold_final_outputs", "target_variables", "actually_changed_variables",
    "preserved_variables", "first_edit_valid", "second_edit_valid",
)


def read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def keyed(path: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for row in read(path):
        pair_id = row.get("pair_id")
        if not isinstance(pair_id, str) or pair_id in result:
            raise ValueError(f"{path}: missing or duplicate pair ID")
        result[pair_id] = row
    return result


def primary_differences(first: dict[str, Any], second: dict[str, Any], source: str) -> list[str]:
    fields = [field for field in STRUCTURAL if first.get(field) != second.get(field)]
    if source == "wiqa" and first.get("query_answer_after_both_edits") != second.get("query_answer_after_both_edits"):
        fields.append("query_answer_after_both_edits")
    if first.get("status") != "accepted" or second.get("status") != "accepted":
        fields.append("status")
    return fields


def variable_names(queue: dict[str, Any], source: str) -> list[str]:
    if source == "wiqa":
        return list(queue["graph"]["nodes"])
    return [f"event_{index:02d}" for index in range(len(queue["chain"]))]


def validate_final(row: dict[str, Any], queue: dict[str, Any], source: str) -> None:
    pair_id = row["pair_id"]
    variables = variable_names(queue, source)
    expected = set(variables)
    if row.get("status") != "accepted":
        raise ValueError(f"{pair_id}: final truth must be accepted")
    if set(row.get("intermediate_outputs", {})) != expected or set(row.get("gold_final_outputs", {})) != expected:
        raise ValueError(f"{pair_id}: output maps must cover every variable")
    changed, preserved = set(row.get("actually_changed_variables", [])), set(row.get("preserved_variables", []))
    if changed & preserved or changed | preserved != expected:
        raise ValueError(f"{pair_id}: change/preservation must be a complete disjoint partition")
    targets = {
        queue["first_edit"]["target"].split(":", 1)[0],
        queue["second_edit"]["target"].split(":", 1)[0],
    }
    if set(row.get("target_variables", [])) != targets:
        raise ValueError(f"{pair_id}: target variables disagree with queue")
    if row.get("first_edit_valid") is not True or row.get("second_edit_valid") is not True:
        raise ValueError(f"{pair_id}: final truth requires two valid edits")
    if source == "wiqa":
        answer = row.get("query_answer_after_both_edits")
        if answer not in {"more", "less", "no_effect"}:
            raise ValueError(f"{pair_id}: WIQA answer is outside vocabulary")
    else:
        if row.get("query_answer_after_both_edits") != row["gold_final_outputs"][variables[-1]]:
            raise ValueError(f"{pair_id}: Com2 query answer must equal final chain event")


def merge_source(
    queue_path: Path, first_path: Path, second_path: Path,
    adjudicated_path: Path, source: str,
) -> list[dict[str, Any]]:
    queue_all, first, second, adjudicated = (
        keyed(queue_path), keyed(first_path), keyed(second_path), keyed(adjudicated_path)
    )
    queue = {pair_id: row for pair_id, row in queue_all.items() if row.get("source") == source}
    expected = set(queue)
    if not expected <= set(first) or not expected <= set(second):
        raise ValueError(f"{source}: annotations do not cover queue")
    disagreements = {
        pair_id for pair_id in expected
        if primary_differences(first[pair_id], second[pair_id], source)
    }
    if set(adjudicated) != disagreements:
        raise ValueError(
            f"{source}: adjudication IDs must exactly resolve disagreements; "
            f"missing={sorted(disagreements-set(adjudicated))}, extra={sorted(set(adjudicated)-disagreements)}"
        )
    final = []
    for pair_id in sorted(expected):
        selected = dict(adjudicated[pair_id] if pair_id in adjudicated else first[pair_id])
        selected["annotator_id"] = "double-annotation-consensus+independent-adjudication"
        selected["status"] = "accepted"
        selected["truth_provenance"] = {
            "protocol": "independent_double_annotation_with_blinded_adjudication_v1",
            "resolution": "adjudicated" if pair_id in adjudicated else "exact_primary_agreement",
            "source_queue": str(queue_path), "test_evaluated": False,
        }
        validate_final(selected, queue[pair_id], source)
        final.append(selected)
    return final


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("wiqa_queue", "wiqa_1", "wiqa_2", "wiqa_adjudicated", "com2_queue", "com2_1", "com2_2", "com2_adjudicated", "output", "provenance"):
        parser.add_argument("--" + name.replace("_", "-"), dest=name, type=Path, required=True)
    args = parser.parse_args()
    wiqa = merge_source(args.wiqa_queue, args.wiqa_1, args.wiqa_2, args.wiqa_adjudicated, "wiqa")
    com2 = merge_source(args.com2_queue, args.com2_1, args.com2_2, args.com2_adjudicated, "com2")
    rows = sorted(wiqa + com2, key=lambda row: row["pair_id"])
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    output_hash = hashlib.sha256(args.output.read_bytes()).hexdigest()
    provenance = {
        "protocol": "independent_double_annotation_with_blinded_adjudication_v1",
        "rows": len(rows), "sources": {"wiqa": len(wiqa), "com2": len(com2)},
        "adjudicated": {"wiqa": len(keyed(args.wiqa_adjudicated)), "com2": len(keyed(args.com2_adjudicated))},
        "output_sha256": output_hash, "test_evaluated": False,
        "inputs": {name: hashlib.sha256(getattr(args, name).read_bytes()).hexdigest() for name in (
            "wiqa_queue", "wiqa_1", "wiqa_2", "wiqa_adjudicated",
            "com2_queue", "com2_1", "com2_2", "com2_adjudicated",
        )},
    }
    args.provenance.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(provenance, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
