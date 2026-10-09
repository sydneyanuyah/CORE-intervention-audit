#!/usr/bin/env python3
"""One-shot, evaluation-only runner for frozen T5 candidate checkpoints."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

import train_candidate_transfer as candidate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("baseline", "o3"), required=True)
    parser.add_argument("--seed", type=int, required=True)
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
    if candidate.sha(args.manifest) != args.manifest_sha256:
        raise ValueError("training manifest identity mismatch")
    if candidate.sha(args.checkpoint) != args.checkpoint_sha256:
        raise ValueError("frozen checkpoint identity mismatch")
    if int(os.environ.get("WORLD_SIZE", "1")) != 4:
        raise RuntimeError("final T5 evaluation requires exactly four ranks")

    manifest = json.loads(args.manifest.read_text())
    source_cells = [
        cell for cell in manifest["cells"]
        if cell["seed"] == args.seed and cell["method"] == args.method
    ]
    if len(source_cells) != 1:
        raise ValueError("unregistered frozen T5 cell")
    source_cell = source_cells[0]
    operator_checkpoint = Path(source_cell["checkpoint"])
    if candidate.sha(operator_checkpoint) != source_cell["checkpoint_sha256"]:
        raise ValueError("frozen operator identity mismatch")
    operator_payload = torch.load(operator_checkpoint, map_location="cpu", weights_only=False)
    if operator_payload.get("test_evaluated") is not False:
        raise ValueError("operator checkpoint is not test-clean")
    trained_payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if (
        trained_payload.get("test_evaluated") is not False
        or trained_payload.get("task") != "t5"
        or trained_payload.get("method") != args.method
        or trained_payload.get("seed") != args.seed
    ):
        raise ValueError("candidate checkpoint provenance mismatch")

    raw_rows = [
        row for row in candidate.rows(args.test_data)
        if str(row.get("source", "")).lower() != "com2"
    ]
    if not raw_rows:
        raise ValueError("T5 final set is empty after the frozen Com2 exclusion")
    if any(row.get("split") != "test" for row in raw_rows):
        raise ValueError("T5 final file contains a non-test row")
    expanded = candidate.expand(raw_rows, "t5")

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    tokenizer = AutoTokenizer.from_pretrained(manifest["model"])
    encoded = tokenizer(
        [row["text"] for row in expanded], padding=True, truncation=True,
        max_length=512, return_tensors="pt",
    )
    encoded["command"] = torch.tensor([row["command"] for row in expanded])
    model = candidate.Model(
        candidate.core_load(Path(manifest["operator_source"])),
        args.method, operator_payload["operator"],
    ).to(device)
    model.load_state_dict(trained_payload["model"], strict=False)
    metrics = candidate.evaluate(model, encoded, expanded, rank, device)
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
        result = {
            "protocol": "final_one_shot_t5_v1",
            "task": "t5",
            "method": args.method,
            "seed": args.seed,
            "checkpoint_sha256": args.checkpoint_sha256,
            "frozen_operator_checkpoint_sha256": source_cell["checkpoint_sha256"],
            "training_manifest_sha256": args.manifest_sha256,
            "test_data_sha256": candidate.sha(args.test_data),
            "test_rows": len(raw_rows),
            "excluded_sources": ["com2"],
            "world_size": 4,
            "metrics": metrics,
            "test_evaluated": True,
        }
        (args.output / "test_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
