#!/usr/bin/env python3
"""Normalize mechanical Com2 fields and create blinded adjudication packets."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


STRUCTURAL_FIELDS = (
    "gold_final_outputs", "target_variables", "actually_changed_variables",
    "preserved_variables", "first_edit_valid", "second_edit_valid",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def keyed(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        pair_id = row.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id or pair_id in result:
            raise ValueError(f"{label}: missing or duplicate pair_id")
        result[pair_id] = row
    return result


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def normalize_com2(queue_path: Path, input_path: Path, output_path: Path, provenance_path: Path) -> dict[str, Any]:
    queue = keyed(read_jsonl(queue_path), "queue")
    rows = read_jsonl(input_path)
    annotations = keyed(rows, "annotations")
    if set(annotations) != set(queue):
        raise ValueError("annotation IDs must exactly match the queue")
    normalized = copy.deepcopy(rows)
    changed = []
    for row in normalized:
        if row.get("source") != "com2":
            continue
        queued = queue[row["pair_id"]]
        chain = queued.get("chain")
        if not isinstance(chain, list) or not chain:
            raise ValueError(f"{row['pair_id']}: missing Com2 chain")
        final_variable = f"event_{len(chain) - 1:02d}"
        final_outputs = row.get("gold_final_outputs")
        if not isinstance(final_outputs, dict) or final_variable not in final_outputs:
            if row.get("status") == "accepted":
                raise ValueError(f"{row['pair_id']}: accepted row lacks final event")
            continue
        exact = final_outputs[final_variable]
        if row.get("query_answer_after_both_edits") != exact:
            row["query_answer_after_both_edits"] = exact
            changed.append(row["pair_id"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, normalized)
    provenance = {
        "protocol": "mechanical_com2_query_normalization_v1",
        "input": str(input_path), "input_sha256": digest(input_path),
        "output": str(output_path), "output_sha256": digest(output_path),
        "changed_field": "query_answer_after_both_edits",
        "changed_rows": len(changed), "changed_pair_ids": sorted(changed),
        "other_fields_changed": False, "test_evaluated": False,
    }
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return provenance


def primary_differences(first: dict[str, Any], second: dict[str, Any]) -> list[str]:
    fields = [field for field in STRUCTURAL_FIELDS if first.get(field) != second.get(field)]
    if first.get("source") == "wiqa" and first.get("query_answer_after_both_edits") != second.get("query_answer_after_both_edits"):
        fields.append("query_answer_after_both_edits")
    if first.get("status") != "accepted" or second.get("status") != "accepted":
        fields.append("status")
    return fields


def proposal(row: dict[str, Any]) -> dict[str, Any]:
    return {
        field: row.get(field) for field in (
            *STRUCTURAL_FIELDS, "intermediate_outputs", "query_answer_after_both_edits",
            "status", "confidence", "justification",
        )
    }


def build_packet(
    queue_path: Path, first_path: Path, second_path: Path, source: str, output_path: Path,
) -> dict[str, Any]:
    if source not in {"wiqa", "com2"}:
        raise ValueError("source must be wiqa or com2")
    queue = keyed(read_jsonl(queue_path), "queue")
    first = keyed(read_jsonl(first_path), "Annotation 1")
    second = keyed(read_jsonl(second_path), "Annotation 2")
    expected = {pair_id for pair_id, row in queue.items() if row.get("source") == source}
    if set(first) != expected or set(second) != expected:
        # Full v2 submissions are allowed when creating the Com2-only packet.
        if not expected <= set(first) or not expected <= set(second):
            raise ValueError("annotation IDs do not cover the requested source queue")
    packet = []
    for pair_id in sorted(expected):
        differences = primary_differences(first[pair_id], second[pair_id])
        if not differences:
            continue
        proposals = [proposal(first[pair_id]), proposal(second[pair_id])]
        if hashlib.sha256(("blind-v1:" + pair_id).encode()).digest()[0] & 1:
            proposals.reverse()
        packet.append({
            "pair_id": pair_id, "source": source, "split": "validation",
            "annotation_item": queue[pair_id],
            "primary_differing_fields": differences,
            "proposal_A": proposals[0], "proposal_B": proposals[1],
            "adjudication_required": {
                "decision": "A, B, or custom",
                "gold_final_outputs": "required",
                "query_answer_after_both_edits": "required",
                "target_variables": "required",
                "actually_changed_variables": "required",
                "preserved_variables": "required",
                "first_edit_valid": "required", "second_edit_valid": "required",
                "justification": "required", "adjudicator_id": "required",
            },
        })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, packet)
    return {
        "protocol": "blinded_two_edit_adjudication_v1", "source": source,
        "rows": len(packet), "output_sha256": digest(output_path), "test_evaluated": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    normalize = sub.add_parser("normalize-com2")
    normalize.add_argument("--queue", type=Path, required=True)
    normalize.add_argument("--input", type=Path, required=True)
    normalize.add_argument("--output", type=Path, required=True)
    normalize.add_argument("--provenance", type=Path, required=True)
    packet = sub.add_parser("packet")
    packet.add_argument("--queue", type=Path, required=True)
    packet.add_argument("--annotation-1", type=Path, required=True)
    packet.add_argument("--annotation-2", type=Path, required=True)
    packet.add_argument("--source", choices=("wiqa", "com2"), required=True)
    packet.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "normalize-com2":
        result = normalize_com2(args.queue, args.input, args.output, args.provenance)
    else:
        result = build_packet(args.queue, args.annotation_1, args.annotation_2, args.source, args.output)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
