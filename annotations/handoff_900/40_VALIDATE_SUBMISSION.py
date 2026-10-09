#!/usr/bin/env python3
"""Validate that an annotated JSONL preserves its frozen source queue."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


VALID_DECISIONS = {"accept", "reject", "needs_adjudication"}
VALID_REASONS = {
    "ACCEPT_EXPLICIT_PAIR",
    "ACCEPT_DETERMINISTIC_SCM",
    "NO_INTERVENTION",
    "GRAPH_NOT_IDENTIFIED",
    "EDGE_NOT_SUPPORTED",
    "FACTUAL_STATE_NOT_IDENTIFIED",
    "COUNTERFACTUAL_STATE_NOT_IDENTIFIED",
    "TARGET_NOT_GROUNDED",
    "NO_OBSERVED_MAY_CHANGE_EFFECT",
    "AMBIGUOUS_OR_CONFLICTING_EVIDENCE",
    "LICENCE_NOT_CLEARED",
    "DUPLICATE_OR_LEAKAGE_RISK",
}
IMMUTABLE_FIELDS = ("pilot_id", "source", "source_locator", "graph_group_id", "source_payload")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"{path.name}:{line_number}: invalid JSON: {error}") from error
    return rows


def validate(original_path: Path, annotated_path: Path) -> None:
    original = read_jsonl(original_path)
    annotated = read_jsonl(annotated_path)
    if len(original) != 300 or len(annotated) != 300:
        raise ValueError(f"expected 300 rows in both files; got {len(original)} and {len(annotated)}")
    if len({row.get("pilot_id") for row in annotated}) != 300:
        raise ValueError("annotated file contains duplicate pilot IDs")
    by_id = {row["pilot_id"]: row for row in annotated}
    for source_row in original:
        pilot_id = source_row["pilot_id"]
        row = by_id.get(pilot_id)
        if row is None:
            raise ValueError(f"{pilot_id}: missing from annotated file")
        for field in IMMUTABLE_FIELDS:
            if row.get(field) != source_row.get(field):
                raise ValueError(f"{pilot_id}: immutable field changed: {field}")
        if set(row.get("task_views", {})) != set(source_row.get("task_views", {})):
            raise ValueError(f"{pilot_id}: task-view names changed")
        for task, annotation in row["task_views"].items():
            prefix = f"{pilot_id}/{task}"
            annotator_id = annotation.get("annotator_id")
            if not isinstance(annotator_id, str) or not re.fullmatch(r"ann-\d{3}", annotator_id):
                raise ValueError(f"{prefix}: annotator_id must look like ann-001")
            decision = annotation.get("decision")
            if decision not in VALID_DECISIONS:
                raise ValueError(f"{prefix}: invalid or missing decision")
            reasons = annotation.get("reason_codes")
            if not isinstance(reasons, list) or any(reason not in VALID_REASONS for reason in reasons):
                raise ValueError(f"{prefix}: invalid reason_codes")
            evidence = annotation.get("evidence_spans")
            if not isinstance(evidence, list):
                raise ValueError(f"{prefix}: evidence_spans must be a list")
            if decision == "accept":
                required = ("graph", "factual_state", "intervention", "intervened_state")
                missing = [field for field in required if annotation.get(field) is None]
                if missing or not evidence:
                    raise ValueError(f"{prefix}: accepted row lacks required structures/evidence: {missing}")
            elif not reasons:
                raise ValueError(f"{prefix}: {decision} requires at least one reason code")
            if decision == "needs_adjudication" and not annotation.get("notes"):
                raise ValueError(f"{prefix}: needs_adjudication requires a precise note")
        if row["source"] == "pubmedcausal" and row["source_payload"].get("release_split") == "test":
            raise ValueError(f"{pilot_id}: forbidden PubMedCausal test row")


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: python3 40_VALIDATE_SUBMISSION.py ORIGINAL_QUEUE ANNOTATED_QUEUE", file=sys.stderr)
        return 2
    try:
        validate(Path(sys.argv[1]), Path(sys.argv[2]))
    except (OSError, ValueError) as error:
        print(f"VALIDATION FAIL: {error}", file=sys.stderr)
        return 1
    print("VALIDATION PASS: 300 rows; frozen source fields preserved; all task views completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
