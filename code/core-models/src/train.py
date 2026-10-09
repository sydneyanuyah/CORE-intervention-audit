"""Distributed BERT training for the closed-label CORE reader baseline."""

from __future__ import annotations

import argparse, hashlib, json, math, os, platform, random, time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from core_bert.data import ID_TO_LABEL, LABEL_TO_ID, CoreExample, load_split


class ExampleDataset(Dataset):
    def __init__(self, examples: list[CoreExample]): self.examples = examples
    def __len__(self) -> int: return len(self.examples)
    def __getitem__(self, index: int) -> CoreExample: return self.examples[index]


class ExactShardSampler(Sampler[int]):
    """Shard evaluation without padding or duplicate records."""
    def __init__(self, size: int, rank: int, world_size: int): self.indices = list(range(rank, size, world_size))
    def __iter__(self) -> Iterator[int]: return iter(self.indices)
    def __len__(self) -> int: return len(self.indices)


def distributed_context() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1")); rank = int(os.environ.get("RANK", "0")); local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if world_size > 1: dist.init_process_group(backend="nccl", device_id=torch.device("cuda", local_rank))
    return rank, local_rank, world_size


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def limit_examples(examples: list[CoreExample], maximum: int | None, seed: int) -> list[CoreExample]:
    if maximum is None or maximum >= len(examples): return examples
    rng = random.Random(seed); strata: dict[tuple[str, int], list[CoreExample]] = defaultdict(list)
    for example in examples: strata[(example.source, example.label)].append(example)
    for values in strata.values(): rng.shuffle(values)
    selected: list[CoreExample] = []; keys = sorted(strata)
    while len(selected) < maximum and keys:
        remaining = []
        for key in keys:
            if strata[key] and len(selected) < maximum: selected.append(strata[key].pop())
            if strata[key]: remaining.append(key)
        keys = remaining
    return sorted(selected, key=lambda item: item.record_id)


def collator(tokenizer: Any, max_length: int):
    def collate(batch: list[CoreExample]) -> dict[str, Any]:
        encoded = tokenizer([item.text for item in batch], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        encoded["labels"] = torch.tensor([item.label for item in batch], dtype=torch.long)
        encoded["record_ids"] = [item.record_id for item in batch]; encoded["sources"] = [item.source for item in batch]
        return encoded
    return collate


def macro_f1(gold: list[int], predicted: list[int], labels: list[int] | None = None) -> float:
    scores = []
    for label in labels if labels is not None else list(range(len(LABEL_TO_ID))):
        tp = sum(g == label and p == label for g, p in zip(gold, predicted)); fp = sum(g != label and p == label for g, p in zip(gold, predicted)); fn = sum(g == label and p != label for g, p in zip(gold, predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0; recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores)


def merge_predictions(parts: list[dict[str, Any]]) -> dict[str, Any]:
    rows = sorted((row for part in parts for row in part["rows"]), key=lambda row: row[0]); gold = [r[2] for r in rows]; predicted = [r[3] for r in rows]
    by_source: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for _, source, actual, guess in rows: by_source[source].append((actual, guess))
    total_items = sum(part["count"] for part in parts); total_loss = sum(part["loss_sum"] for part in parts)
    return {"loss": total_loss / max(total_items, 1), "accuracy": sum(g == p for g, p in zip(gold, predicted)) / max(len(gold), 1), "macro_f1": macro_f1(gold, predicted), "count": total_items,
            "label_counts": {ID_TO_LABEL[k]: v for k, v in sorted(Counter(gold).items())},
            "per_source": {source: {"count": len(pairs), "accuracy": sum(g == p for g, p in pairs) / len(pairs), "macro_f1_present_labels": macro_f1([g for g, _ in pairs], [p for _, p in pairs], sorted({g for g, _ in pairs}))} for source, pairs in sorted(by_source.items())}}


@torch.no_grad()
def evaluate(model: Any, loader: DataLoader, device: torch.device, rank: int, world_size: int) -> dict[str, Any] | None:
    model.eval(); local: dict[str, Any] = {"loss_sum": 0.0, "count": 0, "rows": []}
    for batch in loader:
        sources = batch.pop("sources"); record_ids = batch.pop("record_ids"); batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        output = model(**batch); guesses = output.logits.argmax(dim=-1).cpu().tolist(); labels = batch["labels"].cpu().tolist()
        local["loss_sum"] += float(output.loss) * len(labels); local["count"] += len(labels); local["rows"].extend(zip(record_ids, sources, labels, guesses))
    if world_size == 1: return merge_predictions([local])
    gathered: list[dict[str, Any] | None] = [None] * world_size if rank == 0 else []
    dist.gather_object(local, gathered if rank == 0 else None, dst=0)
    return merge_predictions([part for part in gathered if part is not None]) if rank == 0 else None


def make_loader(examples: list[CoreExample], batch_size: int, collate: Any, workers: int, rank: int, world_size: int, train: bool) -> tuple[DataLoader, Sampler[int] | None]:
    dataset = ExampleDataset(examples)
    sampler: Sampler[int] | None = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=0, drop_last=False) if train and world_size > 1 else (ExactShardSampler(len(dataset), rank, world_size) if not train and world_size > 1 else None)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train and sampler is None, sampler=sampler, num_workers=workers, pin_memory=True, persistent_workers=workers > 0, collate_fn=collate)
    return loader, sampler


def data_fingerprint(data_root: Path) -> dict[str, Any]:
    files = sorted((data_root / "records").glob("*.jsonl")) + sorted((data_root / "splits").glob("*.txt")); digest = hashlib.sha256(); details = {}
    for path in files:
        file_digest = hashlib.sha256(path.read_bytes()).hexdigest(); relative = str(path.relative_to(data_root)); details[relative] = file_digest; digest.update(relative.encode()); digest.update(file_digest.encode())
    return {"combined_sha256": digest.hexdigest(), "files": details}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(); p.add_argument("--data-root", type=Path, required=True); p.add_argument("--output-dir", type=Path, required=True); p.add_argument("--model", default="google-bert/bert-base-uncased")
    p.add_argument("--epochs", type=int, default=3); p.add_argument("--train-batch-size", type=int, default=16, help="Per-GPU batch size"); p.add_argument("--eval-batch-size", type=int, default=32, help="Per-GPU batch size"); p.add_argument("--gradient-accumulation", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=2e-5); p.add_argument("--weight-decay", type=float, default=0.01); p.add_argument("--warmup-ratio", type=float, default=0.1); p.add_argument("--max-length", type=int, default=384); p.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--seed", type=int, default=20260904); p.add_argument("--num-workers", type=int, default=2); p.add_argument("--max-train-samples", type=int); p.add_argument("--max-eval-samples", type=int); p.add_argument("--max-steps", type=int); p.add_argument("--include-test", action="store_true"); p.add_argument("--eval-only", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA GPU is required")
    rank, local_rank, world_size = distributed_context(); primary = rank == 0
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported(): raise RuntimeError("bf16 is unsupported on the selected GPU")
    if args.include_test and (args.max_train_samples or args.max_eval_samples or args.max_steps): raise ValueError("held-out test is disabled for smoke/subsample runs")
    seed_everything(args.seed + rank)
    if primary: args.output_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1: dist.barrier()
    device = torch.device("cuda", local_rank); tokenizer = AutoTokenizer.from_pretrained(args.model)
    raw_model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=len(LABEL_TO_ID), id2label=ID_TO_LABEL, label2id=LABEL_TO_ID).to(device)
    model_revision = getattr(raw_model.config, "_commit_hash", None); parameter_count = sum(p.numel() for p in raw_model.parameters())
    model: Any = DDP(raw_model, device_ids=[local_rank], output_device=local_rank, forward_sync_buffers=False) if world_size > 1 else raw_model
    train_examples = limit_examples(load_split(args.data_root, "train"), args.max_train_samples, args.seed); validation_examples = limit_examples(load_split(args.data_root, "validation"), args.max_eval_samples, args.seed + 1); test_examples = load_split(args.data_root, "test") if args.include_test else []
    collate = collator(tokenizer, args.max_length); train_loader, train_sampler = make_loader(train_examples, args.train_batch_size, collate, args.num_workers, rank, world_size, True); validation_loader, _ = make_loader(validation_examples, args.eval_batch_size, collate, args.num_workers, rank, world_size, False)
    baseline = evaluate(model, validation_loader, device, rank, world_size); summary: dict[str, Any] = {}
    if primary:
        summary = {"arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "environment": {"python": platform.python_version(), "torch": torch.__version__, "cuda_runtime": torch.version.cuda, "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "world_size": world_size, "gpu": torch.cuda.get_device_name(local_rank), "gpu_memory_mib": round(torch.cuda.get_device_properties(local_rank).total_memory / 2**20)}, "model_revision": model_revision, "data_fingerprint": data_fingerprint(args.data_root), "parameter_count": parameter_count, "effective_global_batch_size": args.train_batch_size * world_size * args.gradient_accumulation, "training_sampler": {"kind": "DistributedSampler", "padded_examples_per_epoch": len(train_loader) * args.train_batch_size * world_size - len(train_examples)}, "data_counts": {"train": len(train_examples), "validation": len(validation_examples), "test": len(test_examples)}, "baseline_validation": baseline, "epochs": []}
        print(json.dumps({"world_size": world_size, "parameter_count": parameter_count, "baseline_validation": baseline}, indent=2), flush=True)
    if not args.eval_only:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay); counts = Counter(item.label for item in train_examples); weights = torch.tensor([len(train_examples) / (len(LABEL_TO_ID) * counts[i]) for i in range(len(LABEL_TO_ID))], device=device)
        if primary: summary["class_weights"] = {ID_TO_LABEL[i]: float(v) for i, v in enumerate(weights.cpu())}
        updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation); planned_steps = args.epochs * updates_per_epoch; total_steps = min(planned_steps, args.max_steps) if args.max_steps else planned_steps
        scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * args.warmup_ratio), max(total_steps, 1)); use_autocast = args.precision != "fp32"; amp_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16; scaler = torch.amp.GradScaler("cuda", enabled=args.precision == "fp16")
        best_accuracy, global_step = -1.0, 0; started = time.time(); torch.cuda.reset_peak_memory_stats(device); optimizer.zero_grad(set_to_none=True)
        for epoch in range(1, args.epochs + 1):
            if isinstance(train_sampler, DistributedSampler): train_sampler.set_epoch(epoch)
            model.train(); local_loss = torch.zeros(2, device=device, dtype=torch.float64); epoch_started = time.time()
            for batch_index, batch in enumerate(train_loader, 1):
                batch.pop("sources"); batch.pop("record_ids"); batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                with torch.amp.autocast("cuda", enabled=use_autocast, dtype=amp_dtype): output = model(**batch); weighted_loss = F.cross_entropy(output.logits, batch["labels"], weight=weights); loss = weighted_loss / args.gradient_accumulation
                scaler.scale(loss).backward() if args.precision == "fp16" else loss.backward(); local_loss[0] += weighted_loss.detach().double() * len(batch["labels"]); local_loss[1] += len(batch["labels"])
                if batch_index % args.gradient_accumulation == 0 or batch_index == len(train_loader):
                    if args.precision == "fp16": scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    if args.precision == "fp16": scaler.step(optimizer); scaler.update()
                    else: optimizer.step()
                    optimizer.zero_grad(set_to_none=True); scheduler.step(); global_step += 1
                    if args.max_steps and global_step >= args.max_steps: break
            if world_size > 1: dist.all_reduce(local_loss); dist.barrier()
            epoch_seconds = time.time() - epoch_started; metrics = evaluate(model, validation_loader, device, rank, world_size); memory = torch.tensor([torch.cuda.max_memory_allocated(device)], device=device, dtype=torch.float64)
            if world_size > 1: dist.reduce(memory, dst=0, op=dist.ReduceOp.MAX)
            if primary:
                samples = float(local_loss[1].item()); result = {"epoch": epoch, "global_step": global_step, "train_loss": float(local_loss[0] / local_loss[1]), "validation": metrics, "epoch_seconds": epoch_seconds, "global_examples_per_second": samples / epoch_seconds, "max_gpu_memory_mib": float(memory.item() / 2**20), "elapsed_seconds": time.time() - started}; summary["epochs"].append(result); print(json.dumps(result, indent=2), flush=True)
                if metrics and metrics["accuracy"] > best_accuracy:
                    best_accuracy = metrics["accuracy"]; target = model.module if isinstance(model, DDP) else model; target.save_pretrained(args.output_dir / "best_model"); tokenizer.save_pretrained(args.output_dir / "best_model")
            if world_size > 1: dist.barrier()
            if args.max_steps and global_step >= args.max_steps: break
        if primary: summary["best_validation_accuracy"] = best_accuracy
    if args.include_test:
        if world_size > 1: dist.barrier()
        unwrapped = AutoModelForSequenceClassification.from_pretrained(args.output_dir / "best_model").to(device); test_model: Any = DDP(unwrapped, device_ids=[local_rank], forward_sync_buffers=False) if world_size > 1 else unwrapped; test_loader, _ = make_loader(test_examples, args.eval_batch_size, collate, args.num_workers, rank, world_size, False); held_out = evaluate(test_model, test_loader, device, rank, world_size)
        if primary: summary["held_out_test"] = held_out
    if primary:
        path = args.output_dir / "run_summary.json"; path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n"); print(f"wrote {path}", flush=True)
    if world_size > 1: dist.barrier(); dist.destroy_process_group()
    return 0


if __name__ == "__main__": raise SystemExit(main())
