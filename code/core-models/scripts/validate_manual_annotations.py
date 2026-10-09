#!/usr/bin/env python3
"""Validate two independent manual two-edit annotation files and compare them."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED = {
    "pair_id", "annotator_id", "source", "split", "intermediate_outputs",
    "gold_final_outputs", "query_answer_after_both_edits", "target_variables",
    "actually_changed_variables", "preserved_variables", "first_edit_valid",
    "second_edit_valid", "status", "confidence", "justification",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{number}: row must be an object")
        rows.append(value)
    return rows


def keyed(rows: list[dict[str, Any]], path: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        pair_id = row.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id:
            raise ValueError(f"{path}: missing pair_id")
        if pair_id in result:
            raise ValueError(f"{path}: duplicate pair_id {pair_id}")
        result[pair_id] = row
    return result


def validate(queue_path: Path, annotation_path: Path, annotator_id: str) -> dict[str, dict[str, Any]]:
    queue = keyed(read_jsonl(queue_path), queue_path)
    annotations = keyed(read_jsonl(annotation_path), annotation_path)
    if set(annotations) != set(queue):
        missing = sorted(set(queue) - set(annotations))
        extra = sorted(set(annotations) - set(queue))
        raise ValueError(f"{annotation_path}: queue mismatch; missing={missing[:5]}, extra={extra[:5]}")
    for pair_id, row in annotations.items():
        absent = REQUIRED - set(row)
        if absent:
            raise ValueError(f"{pair_id}: missing fields {sorted(absent)}")
        if row["annotator_id"] != annotator_id:
            raise ValueError(f"{pair_id}: wrong annotator_id")
        if row["split"] != "validation" or queue[pair_id].get("split") != "validation":
            raise ValueError(f"{pair_id}: manual annotation is validation-only")
        if row["source"] not in {"wiqa", "com2"} or row["source"] != queue[pair_id].get("source"):
            raise ValueError(f"{pair_id}: source mismatch")
        if row["status"] not in {"accepted", "needs_adjudication"}:
            raise ValueError(f"{pair_id}: invalid status")
        if row["confidence"] not in {"high", "medium", "low"}:
            raise ValueError(f"{pair_id}: invalid confidence")
        if not isinstance(row["intermediate_outputs"], dict) or not isinstance(row["gold_final_outputs"], dict):
            raise ValueError(f"{pair_id}: outputs must be objects")
        for field in ("target_variables", "actually_changed_variables", "preserved_variables"):
            if not isinstance(row[field], list) or len(row[field]) != len(set(row[field])):
                raise ValueError(f"{pair_id}: {field} must be a unique list")
        if set(row["actually_changed_variables"]) & set(row["preserved_variables"]):
            raise ValueError(f"{pair_id}: changed and preserved variables overlap")
        if row["status"] == "accepted":
            if row["source"] == "com2":
                chain = queue[pair_id].get("chain")
                expected_variables = {f"event_{index:02d}" for index in range(len(chain or []))}
            else:
                expected_variables = set((queue[pair_id].get("graph") or {}).get("nodes") or [])
            if not expected_variables:
                raise ValueError(f"{pair_id}: queue has no variables")
            if set(row["intermediate_outputs"]) != expected_variables or set(row["gold_final_outputs"]) != expected_variables:
                raise ValueError(f"{pair_id}: accepted outputs must cover every queue variable")
            if set(row["actually_changed_variables"]) | set(row["preserved_variables"]) != expected_variables:
                raise ValueError(f"{pair_id}: changed/preserved lists must partition every variable")
            if row["source"] == "wiqa" and row["query_answer_after_both_edits"] not in {"more", "less", "no_effect"}:
                raise ValueError(f"{pair_id}: WIQA query answer must use the closed vocabulary")
            if row["source"] == "com2":
                final_variable = f"event_{len(chain) - 1:02d}"
                if row["query_answer_after_both_edits"] != row["gold_final_outputs"][final_variable]:
                    raise ValueError(f"{pair_id}: Com2 query answer must exactly equal the final chain event")
    return annotations


def compare(first: dict[str, dict[str, Any]], second: dict[str, dict[str, Any]]) -> dict[str, Any]:
    structural = ("gold_final_outputs", "target_variables", "actually_changed_variables",
                  "preserved_variables", "first_edit_valid", "second_edit_valid")
    secondary = ("intermediate_outputs", "query_answer_after_both_edits")
    disagreements = []
    agreed = 0
    field_agreement = {field: 0 for field in ("status",) + structural + secondary}
    for pair_id in sorted(first):
        for field in field_agreement:
            field_agreement[field] += int(first[pair_id][field] == second[pair_id][field])
        differing = [field for field in structural if first[pair_id][field] != second[pair_id][field]]
        if first[pair_id]["source"] == "wiqa" and first[pair_id]["query_answer_after_both_edits"] != second[pair_id]["query_answer_after_both_edits"]:
            differing.append("query_answer_after_both_edits")
        if first[pair_id]["status"] != "accepted" or second[pair_id]["status"] != "accepted":
            differing.append("status")
        if differing:
            disagreements.append({
                "pair_id": pair_id, "primary_differing_fields": differing,
                "secondary_differing_fields": [field for field in secondary if first[pair_id][field] != second[pair_id][field]],
            })
        else:
            agreed += 1
    total = len(first)
    return {
        "protocol": "core_manual_two_edit_double_annotation_v1",
        "total": total,
        "exact_agreement": agreed,
        "exact_agreement_rate": agreed / total if total else None,
        "agreement_definition": "final state, targets, change/preservation, edit validity, accepted status; WIQA query label is also primary",
        "field_exact_agreement": field_agreement,
        "com2_intermediate_text_is_diagnostic": True,
        "needs_adjudication": len(disagreements),
        "disagreements": disagreements,
        "test_evaluated": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--annotation-1", type=Path, required=True)
    parser.add_argument("--annotation-2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    first = validate(args.queue, args.annotation_1, "Annotation 1")
    second = validate(args.queue, args.annotation_2, "Annotation 2")
    report = compare(first, second)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("total", "exact_agreement", "needs_adjudication")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
