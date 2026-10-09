#!/usr/bin/env python3
"""One-shot, evaluation-only runner for frozen T2/T4 checkpoints."""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

import train_t2


def final_rows(path: Path, task: str, arm: str | None) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            # The frozen CSuite exporter physically partitioned distinct files
            # but retained its provenance label, ``development_generated``.
            # Disjointness from train/validation is verified below using IDs.
            if row.get("split") not in {"test", "development_generated"} or row.get("test_evaluated") is not False:
                raise ValueError(f"{path}:{line_number}: inadmissible final-test row")
            if task == "t4" and row.get("arm") != arm:
                continue
            rows.append(row)
    if not rows:
        raise ValueError(f"no final-test rows for {task}/{arm or 'all'}")
    return rows


def record_ids(path: Path) -> set[str]:
    values: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            record_id = value.get("id")
            if not isinstance(record_id, str) or not record_id or record_id in values:
                raise ValueError(f"{path}:{line_number}: invalid or duplicate record ID")
            values.add(record_id)
    return values


def prepare(rows: list[dict], task: str, max_nodes: int):
    if task == "t2":
        return train_t2.prepare(rows, max_nodes)
    converted = []
    for row in rows:
        match = re.search(r"do\((x\d+)\s*=\s*(-?\d+(?:\.\d+)?)\)", row["rendered_input"])
        if match is None:
            match = re.search(
                r"Change (x\d+) from (-?\d+(?:\.\d+)?) to (-?\d+(?:\.\d+)?)",
                row["rendered_input"],
            )
        if match is None:
            raise ValueError(f"cannot parse T4 intervention: {row['id']}")
        if len(match.groups()) == 2:
            target, new_value = match.groups()
            reference = row["factual_state"][target]
        else:
            target, reference, new_value = match.groups()
        item = dict(row)
        item["intervention"] = {
            "target": target,
            "value": float(new_value),
            "reference_value": float(reference),
        }
        converted.append(item)
    return train_t2.prepare(converted, max_nodes)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("t2", "t4"), required=True)
    parser.add_argument("--arm", choices=("changing_only", "imagining_only", "joint"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--authorization", required=True)
    args = parser.parse_args()
    if args.authorization != "FINAL_ONE_SHOT":
        raise PermissionError("final test requires the exact FINAL_ONE_SHOT authorization")
    if (args.task == "t4") != (args.arm is not None):
        raise ValueError("--arm is required only for T4")
    if train_t2.sha256(args.manifest) != args.manifest_sha256:
        raise ValueError("training manifest identity mismatch")
    if train_t2.sha256(args.checkpoint) != args.checkpoint_sha256:
        raise ValueError("frozen checkpoint identity mismatch")
    if int(os.environ.get("WORLD_SIZE", "1")) != 4:
        raise RuntimeError("final T2/T4 evaluation requires exactly four ranks")

    manifest = json.loads(args.manifest.read_text())
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("test_evaluated") is not False:
        raise ValueError("checkpoint is not validation-selected/test-clean")
    rows = final_rows(args.test_data, args.task, args.arm)
    test_ids = {row["id"] for row in rows}
    development_ids = record_ids(Path(manifest["train_data"])) | record_ids(Path(manifest["validation_data"]))
    overlap = test_ids & development_ids
    if overlap:
        raise ValueError(f"final-test IDs overlap train/validation: {len(overlap)}")
    parts = prepare(rows, args.task, manifest["max_nodes"])

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    tokenizer = AutoTokenizer.from_pretrained(manifest["model"])
    encoded = tokenizer(parts[0], padding=True, truncation=True, max_length=512, return_tensors="pt")
    encoded["intervention_id"] = parts[4]
    model = train_t2.T2Model(
        train_t2.load_core(Path(manifest["operator_source"])),
        manifest["model"], manifest["split_layer"], manifest["world_tokens"], manifest["max_nodes"],
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    overall, by_family = train_t2.evaluate(model, encoded, parts[1], parts[2], parts[3], parts[5], rank, 4, device)
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
        result = {
            "protocol": "final_one_shot_t2_t4_v1",
            "task": args.task,
            "arm": args.arm,
            "checkpoint_sha256": args.checkpoint_sha256,
            "training_manifest_sha256": args.manifest_sha256,
            "test_data_sha256": train_t2.sha256(args.test_data),
            "test_rows": len(rows),
            "world_size": 4,
            "metrics": overall,
            "by_sem_family": by_family,
            "test_evaluated": True,
        }
        (args.output / "test_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
