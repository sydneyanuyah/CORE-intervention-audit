#!/usr/bin/env python3
"""Continuously fill eight four-GPU groups with registered Com2 F2 cells."""

from __future__ import annotations

from typing import Any

import dispatch_f2_cladder as base


AMENDMENT_SHA = "not-published"
QUEUE_SHA = "not-published"
TRUTH_SHA = "not-published"
PROVENANCE_SHA = "not-published"
INTERVENTION_RECORDS_SHA = "not-published"
COUNTERFACTUAL_RECORDS_SHA = "not-published"
INTERVENTION_TRAIN_SHA = "not-published"
COUNTERFACTUAL_TRAIN_SHA = "not-published"
INTERVENTION_VALIDATION_SHA = "not-published"
COUNTERFACTUAL_VALIDATION_SHA = "not-published"
GROUPS_SHA = "not-published"

base.FAMILY = "com2"
base.ARTIFACT_SHA = QUEUE_SHA
base.PAIR_SHA = TRUTH_SHA
base.ARTIFACT_NAME = "../manual/annotation_queue.jsonl"
base.PAIR_NAME = "../manual/authoritative_manual_truth.jsonl"


def amendment_sha(cell: base.Cell) -> str:
    return AMENDMENT_SHA


def command(cell: base.Cell) -> list[str]:
    source = cell.checkpoint.parent
    root = base.ROOT
    return [
        str(root / ".venv/bin/torchrun"), "--standalone", "--nproc_per_node=4",
        str(root / "src/train_f2_com2.py"), "--method", cell.method,
        "--seed", str(cell.seed), "--checkpoint", str(cell.checkpoint),
        "--expected-checkpoint-sha256", cell.checkpoint_sha,
        "--tokenizer", str(source / "tokenizer"), "--data-root", str(root / "data"),
        "--manifest", str(root / "registry/f2_manifest.json"), "--expected-manifest-sha256", base.MANIFEST_SHA,
        "--cells", str(root / "registry/f2_cells.json"), "--expected-cells-sha256", base.CELLS_SHA,
        "--source-catalog", str(root / "registry/a2_source_catalog.json"), "--expected-source-catalog-sha256", base.CATALOG_SHA,
        "--amendment", str(root / "registry/f2_com2_execution_amendment.json"), "--expected-amendment-sha256", AMENDMENT_SHA,
        "--queue", str(root / "data/two_edit/manual/annotation_queue.jsonl"), "--expected-queue-sha256", QUEUE_SHA,
        "--truth", str(root / "data/two_edit/manual/authoritative_manual_truth.jsonl"), "--expected-truth-sha256", TRUTH_SHA,
        "--provenance", str(root / "data/two_edit/manual/authoritative_manual_truth.provenance.json"), "--expected-provenance-sha256", PROVENANCE_SHA,
        "--expected-intervention-records-sha256", INTERVENTION_RECORDS_SHA,
        "--expected-counterfactual-records-sha256", COUNTERFACTUAL_RECORDS_SHA,
        "--expected-intervention-train-sha256", INTERVENTION_TRAIN_SHA,
        "--expected-counterfactual-train-sha256", COUNTERFACTUAL_TRAIN_SHA,
        "--expected-intervention-validation-sha256", INTERVENTION_VALIDATION_SHA,
        "--expected-counterfactual-validation-sha256", COUNTERFACTUAL_VALIDATION_SHA,
        "--expected-groups-sha256", GROUPS_SHA,
        "--output-dir", str(cell.output),
    ]


_base_valid = base.valid


def valid(cell: base.Cell) -> bool:
    if not _base_valid(cell):
        return False
    try:
        row: dict[str, Any] = base.json.loads((cell.output / "run_summary.json").read_text())
    except (OSError, base.json.JSONDecodeError):
        return False
    return row.get("protocol") == "f2_com2_open_text_cell_v1"


base.amendment_sha = amendment_sha
base.command = command
base.valid = valid


def main() -> int:
    # The shared dispatcher resolves its frozen artifact paths through the
    # family directory.  Com2's adjudicated files live in the sibling manual
    # directory, so ensure the traversal anchor exists without copying data.
    (base.ROOT / "data/two_edit/com2").mkdir(exist_ok=True)
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
