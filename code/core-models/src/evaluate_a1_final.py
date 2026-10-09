#!/usr/bin/env python3
"""One-shot, evaluation-only runner for a frozen addressed A1 checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from core_bert.addressed_reader import AddressedWorldReader
from core_bert.benchmark_data import BenchmarkDataset, BenchmarkExample, GroupDistributedSampler, _source_groups
from core_bert.scientific_reader import ScientificAddressedReader
from train_addressed import (
    LABEL_TO_ID,
    ProductionAddressedModel,
    ProductionCollator,
    _task_dataset,
    evaluate_exact,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--source", choices=("ccrgb", "cladder", "wiqa"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--authorization", required=True)
    args = parser.parse_args()
    if args.authorization != "FINAL_ONE_SHOT":
        raise PermissionError("final test requires the exact FINAL_ONE_SHOT authorization")
    if sha256(args.checkpoint) != args.checkpoint_sha256:
        raise ValueError("frozen checkpoint identity mismatch")
    if int(os.environ.get("WORLD_SIZE", "1")) != 4:
        raise RuntimeError("final A1 evaluation requires exactly four ranks")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint.get("configuration", {})
    if config.get("mode") != "t3b" or config.get("model_size") != "base":
        raise ValueError("final A1 accepts only frozen BERT-base T3-b checkpoints")
    if config.get("open_text_output", False):
        raise ValueError("Com2/open-text checkpoints are excluded from final evaluation")

    examples = []
    with args.test_data.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("source") != args.source or not isinstance(record.get("id"), str):
                raise ValueError(f"{args.test_data}:{line_number}: invalid sealed A1 row")
            graph_group, world_group = _source_groups(record)
            examples.append(BenchmarkExample(
                record=record, record_id=record["id"], source=args.source,
                label=record["intervened"].get("answer"),
                intervention_kind=str(record["intervention"].get("kind")),
                graph_group_id=graph_group, world_group_id=world_group,
            ))
    dataset = _task_dataset(BenchmarkDataset(examples), "t3b", open_text_output=False)
    world_size = 4
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    tokenizer = AutoTokenizer.from_pretrained(config["model"], use_fast=True)
    tokenizer.add_special_tokens({"additional_special_tokens": ["[DO]"]})
    collator = ProductionCollator(
        tokenizer, int(config.get("max_length", 512)), int(config.get("world_slots", 30)),
        variable_outputs=float(config.get("variable_loss_weight", 0.0)) > 0,
        paraphrase_partition=None, split="test",
    )
    sampler = GroupDistributedSampler(
        dataset, rank=rank, world_size=world_size, group_by="graph",
        seed=int(config["seed"]), shuffle=False, pad_to_equal=False,
    )
    loader = DataLoader(
        dataset, batch_size=int(config.get("batch_size", 16)), sampler=sampler,
        collate_fn=collator, num_workers=0, pin_memory=True,
    )
    base = AddressedWorldReader.from_pretrained(
        config["model"], node_count=1, split_layer=1,
        world_slot_count=int(config.get("world_slots", 30)), label_count=len(LABEL_TO_ID),
    )
    configured_split = int(config.get("split_layer", 0))
    base.split_layer = configured_split or len(base.bert.encoder.layer) // 2
    base.bert.resize_token_embeddings(len(tokenizer))
    model = ProductionAddressedModel(
        ScientificAddressedReader(base, max_slots=30), "t3b", int(config.get("rank", 16)),
        editor_disabled=False, span_override="correct", seed=int(config["seed"]) + rank,
        padding_idx=tokenizer.pad_token_id,
        variable_outputs=float(config.get("variable_loss_weight", 0.0)) > 0,
        causal_task_readout=bool(config.get("causal_task_readout", False)),
        hard_pointer=bool(config.get("hard_pointer", False)),
        open_text_output=False, tokenizer=tokenizer,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    metrics = evaluate_exact(model, loader, device, rank, world_size, "t3b", args.output, "final_test")
    if rank == 0:
        record_identity = hashlib.sha256(
            "\n".join(example.record_id for example in dataset).encode("utf-8")
        ).hexdigest()
        result = {
            "protocol": "final_one_shot_a1_v1",
            "source": args.source,
            "checkpoint_sha256": args.checkpoint_sha256,
            "test_record_ids_sha256": record_identity,
            "test_rows": len(dataset),
            "world_size": 4,
            "metrics": metrics,
            "excluded_sources": ["com2"],
            "test_evaluated": True,
        }
        (args.output / "test_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
