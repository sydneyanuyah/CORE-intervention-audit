#!/usr/bin/env python3
"""Four-GPU train/validation smoke for T2/T3 addressed BERT readers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoTokenizer
import transformers

from core_bert.addressed_data import AddressedBatchCollator, CLOSED_VALUE_TOKENS
from core_bert.addressed_reader import AddressedWorldReader
from core_bert.data import LABEL_TO_ID
from core_bert.t3_pointer import ClosedValueEmbedding, T3BPointer, override_span_mask


GPU_COUNT = 4
MODES = ("t2a", "t2b", "t3a", "t3b")
SPAN_OVERRIDES = ("correct", "adjacent", "random")


class RecordDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


def _split_ids(data_root: Path, split: str) -> set[str]:
    paths = sorted((data_root / "splits").glob(f"*.{split}.txt"))
    if not paths:
        raise ValueError(f"no {split} manifests found under {data_root / 'splits'}")
    result: set[str] = set()
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            value = line.strip()
            if value and not value.startswith("#"):
                if value in result:
                    raise ValueError(f"duplicate split ID {value!r}")
                result.add(value)
    return result


def load_addressed_records(
    data_root: Path,
    split: str,
    *,
    mode: str,
    limit: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Load addressed, closed-label records from train or validation only."""

    if split not in {"train", "validation"}:
        raise ValueError("addressed smoke never opens the held-out test split")
    selected = _split_ids(data_root, split)
    records: list[dict[str, Any]] = []
    for path in sorted((data_root / "records").glob("*.jsonl")):
        if path.name.endswith(".rejected.jsonl"):
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("id") not in selected:
                    continue
                answer = str(record.get("intervened", {}).get("answer", "")).strip().lower()
                intervention = record.get("intervention", {})
                if answer not in LABEL_TO_ID:
                    continue
                # The current smoke head is closed-label classification. T3's
                # event-replacement path is implemented in the reader, but Com2
                # requires its separate open-text task head before it can train.
                if mode.startswith("t3") and intervention.get("kind") != "value_set":
                    continue
                records.append(record)
    records.sort(key=lambda item: item["id"])
    if len(records) > limit:
        rng = random.Random(seed)
        records = sorted(rng.sample(records, limit), key=lambda item: item["id"])
    if not records:
        raise ValueError(f"no addressed closed-label {split} records are available")
    return records


def data_fingerprint(data_root: Path) -> dict[str, str]:
    result = {}
    paths = sorted((data_root / "records").glob("*.jsonl"))
    paths += sorted((data_root / "splits").glob("*.txt"))
    for path in paths:
        result[str(path.relative_to(data_root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


class AddressedSmokeModel(nn.Module):
    """Mode-specific adapter around the shared addressed reader and operators."""

    def __init__(
        self,
        reader: AddressedWorldReader,
        mode: str,
        rank: int,
        *,
        editor_disabled: bool,
        span_override: str,
        override_seed: int,
    ):
        super().__init__()
        try:
            from core_bert.addressed_operator import InstructionGatedEdit
        except ImportError:
            # The canonical port currently exports the descriptive class name;
            # accept it while retaining the registry adapter name at this seam.
            from core_bert.addressed_operator import AddressedGatedOperator as InstructionGatedEdit

        self.reader = reader
        self.mode = mode
        self.editor_disabled = editor_disabled
        self.span_override = span_override
        self.override_seed = override_seed
        if mode in {"t2a", "t2b", "t3a"}:
            self.editor: nn.Module = InstructionGatedEdit(reader.hidden_size, rank)
        else:
            self.editor = T3BPointer(reader.hidden_size, rank)
        self.value_embedding = (
            ClosedValueEmbedding(len(CLOSED_VALUE_TOKENS), reader.hidden_size)
            if mode.startswith("t3")
            else None
        )

    def forward(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mode.startswith("t2"):
            result = self.reader.forward_t2(
                batch["input_ids"],
                batch["attention_mask"],
                batch["command_tokens"],
                batch["do_tokens"],
                self.editor,
                mode=self.mode,
                editor_enabled=not self.editor_disabled,
            )
        else:
            span = batch["target_tokens"]
            if self.span_override != "correct":
                generator = torch.Generator(device="cpu").manual_seed(self.override_seed)
                span_eligible = batch["offset_mapping"][..., 1] > batch["offset_mapping"][..., 0]
                span = override_span_mask(
                    span.cpu(),
                    span_eligible.cpu(),
                    self.span_override,
                    generator=generator,
                ).to(batch["input_ids"].device)
            if (batch["value_ids"] < 0).any():
                raise ValueError("event replacement requires the Com2 open-text task head")
            value = self.value_embedding(batch["value_ids"])
            # T3 obtains its address/value from the factual passage; appended
            # command tokens are hidden so the arm cannot become prompting.
            passage_attention = (
                batch["attention_mask"].bool()
                & ~batch["command_tokens"]
                & ~batch["do_tokens"]
            ).long()
            result = self.reader.forward_t3(
                batch["input_ids"],
                passage_attention,
                span,
                self.editor,
                value_embedding=value,
                mode=self.mode,
            )
        pointer = result.pointer_probabilities
        if pointer is None:
            pointer = result.logits.new_empty((result.logits.shape[0], 0))
        return result.logits[:, 0, :], pointer


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--model", default="google-bert/bert-base-uncased")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2, help="per-GPU batch size")
    parser.add_argument("--max-train-records", type=int, default=32)
    parser.add_argument("--max-validation-records", type=int, default=16)
    parser.add_argument("--split-layer", type=int, default=0, help="0 selects the midpoint")
    parser.add_argument("--world-slots", type=int, default=16)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--editor-disabled", action="store_true", help="A2 zero-editor control")
    parser.add_argument("--span-override", choices=SPAN_OVERRIDES, default="correct", help="A3 control")
    args = parser.parse_args(argv)
    if args.steps < 1 or args.batch_size < 1:
        parser.error("steps and batch-size must be positive")
    if args.max_length < 4:
        parser.error("max-length must be at least 4")
    if args.editor_disabled and not args.mode.startswith("t2"):
        parser.error("--editor-disabled is the A2 control and requires t2a/t2b")
    if args.span_override != "correct" and not args.mode.startswith("t3"):
        parser.error("--span-override is the A3 control and requires t3a/t3b")
    return args


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    totals = torch.zeros(3, dtype=torch.float64, device=device)
    for batch in loader:
        batch = _move_batch(batch, device)
        logits, _ = model(batch)
        labels = batch["label_ids"]
        totals[0] += F.cross_entropy(logits, labels, reduction="sum").double()
        totals[1] += (logits.argmax(-1) == labels).sum()
        totals[2] += len(labels)
    dist.all_reduce(totals)
    return {
        "loss": float((totals[0] / totals[2]).item()),
        "accuracy": float((totals[1] / totals[2]).item()),
        "count_with_distributed_padding": int(totals[2].item()),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != GPU_COUNT:
        raise RuntimeError(f"addressed BERT smoke requires exactly {GPU_COUNT} GPUs, got {world_size}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    rank_id = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    device = torch.device("cuda", local_rank)
    torch.manual_seed(args.seed + rank_id)
    torch.cuda.manual_seed_all(args.seed + rank_id)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    added_tokens = tokenizer.add_special_tokens({"additional_special_tokens": ["[DO]"]})
    collator = AddressedBatchCollator(tokenizer, max_length=args.max_length)
    train_records = load_addressed_records(
        args.data_root, "train", mode=args.mode,
        limit=args.max_train_records, seed=args.seed,
    )
    validation_records = load_addressed_records(
        args.data_root, "validation", mode=args.mode,
        limit=args.max_validation_records, seed=args.seed + 1,
    )

    # Probe config first so split_layer=0 remains the true midpoint for both
    # BERT-base and BERT-large.
    reader = AddressedWorldReader.from_pretrained(
        args.model,
        node_count=1,
        split_layer=1,
        world_slot_count=args.world_slots,
        label_count=len(LABEL_TO_ID),
    )
    chosen_split = args.split_layer or len(reader.bert.encoder.layer) // 2
    if chosen_split != reader.split_layer:
        if not 0 < chosen_split < len(reader.bert.encoder.layer):
            raise ValueError("split-layer must leave layers on both sides")
        reader.split_layer = chosen_split
    reader.bert.resize_token_embeddings(len(tokenizer))
    model = AddressedSmokeModel(
        reader, args.mode, args.rank,
        editor_disabled=args.editor_disabled,
        span_override=args.span_override,
        override_seed=args.seed + rank_id,
    ).to(device)
    wrapped = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
    )

    train_sampler = DistributedSampler(
        RecordDataset(train_records), num_replicas=world_size, rank=rank_id,
        shuffle=True, seed=args.seed, drop_last=False,
    )
    validation_sampler = DistributedSampler(
        RecordDataset(validation_records), num_replicas=world_size, rank=rank_id,
        shuffle=False, drop_last=False,
    )
    train_loader = DataLoader(
        train_sampler.dataset, batch_size=args.batch_size, sampler=train_sampler,
        collate_fn=collator, num_workers=0,
    )
    validation_loader = DataLoader(
        validation_sampler.dataset, batch_size=args.batch_size, sampler=validation_sampler,
        collate_fn=collator, num_workers=0,
    )
    optimizer = torch.optim.AdamW(wrapped.parameters(), lr=args.learning_rate)
    autocast_enabled = args.precision != "fp32"
    amp_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=args.precision == "fp16")
    iterator = iter(train_loader)
    losses: list[float] = []
    for step in range(args.steps):
        try:
            batch = next(iterator)
        except StopIteration:
            train_sampler.set_epoch(step + 1)
            iterator = iter(train_loader)
            batch = next(iterator)
        batch = _move_batch(batch, device)
        wrapped.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=autocast_enabled, dtype=amp_dtype):
            logits, _ = wrapped(batch)
            loss = F.cross_entropy(logits, batch["label_ids"])
        if args.precision == "fp16":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        reduced = loss.detach().double()
        dist.all_reduce(reduced)
        losses.append(float((reduced / world_size).item()))

    metrics = evaluate(wrapped, validation_loader, device)
    if rank_id == 0:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        fingerprint = hashlib.sha256(
            "\n".join(record["id"] for record in train_records + validation_records).encode()
        ).hexdigest()
        summary = {
            "protocol": "addressed train/validation smoke; held-out test not loaded",
            "mode": args.mode,
            "a2_editor_disabled": args.editor_disabled,
            "a3_span_override": args.span_override,
            "model": args.model,
            "world_size": world_size,
            "split_layer": chosen_split,
            "do_token_id": tokenizer.convert_tokens_to_ids("[DO]"),
            "max_length": args.max_length,
            "steps": args.steps,
            "per_gpu_batch_size": args.batch_size,
            "train_records": len(train_records),
            "validation_records": len(validation_records),
            "record_id_sha256": fingerprint,
            "data_files_sha256": data_fingerprint(args.data_root),
            "train_losses": losses,
            "validation": metrics,
            "model_revision": getattr(reader.bert.config, "_commit_hash", None),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "added_special_tokens": added_tokens,
            "seed": args.seed,
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "cuda_runtime": torch.version.cuda,
                "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "gpu": torch.cuda.get_device_name(local_rank),
            },
        }
        args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
