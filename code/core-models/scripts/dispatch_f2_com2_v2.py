#!/usr/bin/env python3
"""Continuously fill eight four-GPU groups with registered fixed Com2 F2 cells."""

from __future__ import annotations

import fcntl
from pathlib import Path
from typing import Any

import dispatch_f2_cladder as base


ROOT = Path(__file__).parents[1].resolve()
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
base.Cell.output = property(lambda self: ROOT / f"outputs/f2-fixed/com2/{self.method}/seed-{self.seed}")


def registered_cells() -> list[base.Cell]:
    """Hash each frozen source checkpoint once, then expand its four methods."""

    catalog_path = ROOT / "registry/a2_source_catalog.json"
    if base.sha256(catalog_path) != base.CATALOG_SHA:
        raise RuntimeError("F2 source catalog hash mismatch")
    catalog = base.json.loads(catalog_path.read_text())
    indexed = {(row["family"], row["seed"]): row for row in catalog["cells"]}
    rows: list[base.Cell] = []
    for seed in range(301, 321):
        source = indexed[("com2", 2026090300 + seed)]
        checkpoint = ROOT / source["source_directory"] / "best.pt"
        if base.sha256(checkpoint) != source["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint hash mismatch for com2 seed {seed}")
        rows.extend(
            base.Cell(method, seed, checkpoint, source["checkpoint_sha256"])
            for method in ("o2", "o3", "prompting", "lora_matched")
        )
    return rows


def amendment_sha(cell: base.Cell) -> str:
    return AMENDMENT_SHA


def command(cell: base.Cell) -> list[str]:
    source = cell.checkpoint.parent
    return [
        str(ROOT / ".venv/bin/torchrun"), "--standalone", "--nproc_per_node=4",
        str(ROOT / "src/train_f2_com2.py"), "--method", cell.method,
        "--seed", str(cell.seed), "--checkpoint", str(cell.checkpoint),
        "--expected-checkpoint-sha256", cell.checkpoint_sha,
        "--tokenizer", str(source / "tokenizer"), "--data-root", str(ROOT / "data"),
        "--manifest", str(ROOT / "registry/f2_manifest.json"), "--expected-manifest-sha256", base.MANIFEST_SHA,
        "--cells", str(ROOT / "registry/f2_cells.json"), "--expected-cells-sha256", base.CELLS_SHA,
        "--source-catalog", str(ROOT / "registry/a2_source_catalog.json"), "--expected-source-catalog-sha256", base.CATALOG_SHA,
        "--amendment", str(ROOT / "registry/f2_com2_execution_amendment_v2.json"), "--expected-amendment-sha256", AMENDMENT_SHA,
        "--queue", str(ROOT / "data/two_edit/manual/annotation_queue.jsonl"), "--expected-queue-sha256", QUEUE_SHA,
        "--truth", str(ROOT / "data/two_edit/manual/authoritative_manual_truth.jsonl"), "--expected-truth-sha256", TRUTH_SHA,
        "--provenance", str(ROOT / "data/two_edit/manual/authoritative_manual_truth.provenance.json"), "--expected-provenance-sha256", PROVENANCE_SHA,
        "--expected-intervention-records-sha256", INTERVENTION_RECORDS_SHA,
        "--expected-counterfactual-records-sha256", COUNTERFACTUAL_RECORDS_SHA,
        "--expected-intervention-train-sha256", INTERVENTION_TRAIN_SHA,
        "--expected-counterfactual-train-sha256", COUNTERFACTUAL_TRAIN_SHA,
        "--expected-intervention-validation-sha256", INTERVENTION_VALIDATION_SHA,
        "--expected-counterfactual-validation-sha256", COUNTERFACTUAL_VALIDATION_SHA,
        "--expected-groups-sha256", GROUPS_SHA,
        "--output-dir", str(cell.output),
    ]


def valid(cell: base.Cell) -> bool:
    path = cell.output / "run_summary.json"
    if not path.is_file():
        return False
    try:
        row: dict[str, Any] = base.json.loads(path.read_text())
    except (OSError, base.json.JSONDecodeError):
        return False
    return (
        row.get("protocol") == "f2_com2_open_text_cell_v2"
        and row.get("cell_id") == cell.cell_id
        and row.get("checkpoint_sha256") == cell.checkpoint_sha
        and row.get("world_size") == 4
        and row.get("test_evaluated") is False
        and row.get("amendment_sha256") == AMENDMENT_SHA
        and isinstance(row.get("two_edit", {}).get("two_edit_balanced"), (int, float))
    )


base.amendment_sha = amendment_sha
base.command = command
base.valid = valid
base.registered_cells = registered_cells


def main() -> int:
    (ROOT / "data/two_edit/com2").mkdir(exist_ok=True)
    lock_path = ROOT / "logs/f2-com2-v2.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another F2 Com2 v2 dispatcher is already running") from error
        lock.write(str(base.os.getpid()) + "\n")
        lock.flush()
        return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
