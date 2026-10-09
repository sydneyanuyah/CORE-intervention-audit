#!/usr/bin/env python3
"""Continuously fill eight four-GPU groups with registered WIQA F2 cells."""

from __future__ import annotations

from typing import Any

import dispatch_f2_cladder as base


RECORDS_SHA = "not-published"
TRAIN_SPLIT_SHA = "not-published"
VALIDATION_SPLIT_SHA = "not-published"
GROUPS_SHA = "not-published"

base.FAMILY = "wiqa"
base.ARTIFACT_SHA = "not-published"
base.PAIR_SHA = "not-published"
base.ARTIFACT_NAME = "queue.jsonl"
base.PAIR_NAME = "truth.jsonl"

_base_command = base.command
_base_valid = base.valid


def command(cell: base.Cell) -> list[str]:
    return _base_command(cell) + [
        "--data-root", str(base.ROOT / "data"),
        "--expected-records-sha256", RECORDS_SHA,
        "--expected-train-split-sha256", TRAIN_SPLIT_SHA,
        "--expected-validation-split-sha256", VALIDATION_SPLIT_SHA,
        "--expected-groups-sha256", GROUPS_SHA,
    ]


def valid(cell: base.Cell) -> bool:
    if not _base_valid(cell):
        return False
    try:
        row: dict[str, Any] = base.json.loads((cell.output / "run_summary.json").read_text())
    except (OSError, base.json.JSONDecodeError):
        return False
    return row.get("accepted_data_sha256") == {
        "records": RECORDS_SHA,
        "train_split": TRAIN_SPLIT_SHA,
        "validation_split": VALIDATION_SPLIT_SHA,
        "groups": GROUPS_SHA,
    }


base.command = command
base.valid = valid


if __name__ == "__main__":
    raise SystemExit(base.main())
