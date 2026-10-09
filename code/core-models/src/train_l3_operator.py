#!/usr/bin/env python3
"""Train one registered four-rank L3 operator cell and score laws on validation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import transformers
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer

from core_bert.l3_operators import build_operator
from train_f1 import OperatorTrainModule, load_core, load_json, patch_reader_layers, setup


METHODS = {"o1", "loreft", "task_vector_add", "router"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def registered_cell(args) -> dict:
    if sha256(args.manifest) != args.expected_manifest_sha256 or sha256(args.cells) != args.expected_cells_sha256:
        raise ValueError("L3 manifest or cell-registry hash mismatch")
    manifest, registry = load_json(args.manifest), load_json(args.cells)
    if manifest.get("allowed_splits") != ["train", "validation"] or manifest.get("test_evaluated") is not False:
        raise ValueError("L3 split or test lock changed")
    matches = [row for row in registry["cells"] if row["cell_id"] == args.cell_id]
    if len(matches) != 1:
        raise ValueError("L3 cell is not uniquely registered")
    cell = matches[0]
    expected = {"method": args.method, "seed": args.seed, "graph_seed": args.graph_seed, "world_size": 4}
    if any(cell.get(key) != value for key, value in expected.items()) or args.method not in METHODS:
        raise ValueError("L3 cell contract mismatch")
    if sha256(args.reader_checkpoint) != cell["reader_sha256"]:
        raise ValueError("L3 reader checkpoint hash mismatch")
    for name in ("graph.json", "worlds.json", "records_real.json"):
        if sha256(args.artifact_dir / name) != cell["artifact_sha256"][name]:
            raise ValueError(f"L3 artifact hash mismatch: {name}")
    return cell


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--graph-seed", type=int, required=True)
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--reader-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--core-components", type=Path, required=True)
    parser.add_argument("--expected-core-sha256", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--cells", type=Path, required=True)
    parser.add_argument("--expected-cells-sha256", required=True)
    parser.add_argument("--operator-steps", type=int, choices=(1, 500), default=500)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke != (args.operator_steps == 1):
        raise ValueError("L3 smoke must use exactly one optimizer step; production must use 500")
    cell = registered_cell(args)
    if sha256(args.core_components) != args.expected_core_sha256:
        raise ValueError("canonical CORE operator source hash mismatch")

    local_rank, rank = setup("l3", 4)
    device = torch.device("cuda", local_rank)
    core = load_core(args.core_components); core.seed_all(args.seed)
    graph = load_json(args.artifact_dir / "graph.json")
    worlds = load_json(args.artifact_dir / "worlds.json")
    records = load_json(args.artifact_dir / "records_real.json")
    if graph.get("seed") != args.graph_seed or {row["split"] for row in worlds} != {"train", "validation"}:
        raise ValueError("L3 graph identity or world splits changed")
    if any(row["split"] not in {"train", "validation"} for row in records):
        raise ValueError("L3 records include forbidden split")

    original = transformers.AutoModel.from_pretrained
    def eager(*model_args, **model_kwargs):
        model_kwargs.setdefault("attn_implementation", "eager")
        return original(*model_args, **model_kwargs)
    transformers.AutoModel.from_pretrained = eager
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    reader = core.SharedWorldReader("google-bert/bert-base-uncased", 30, split_layer=4, max_world_tokens=16).to(device)
    patch_reader_layers(reader)
    checkpoint = torch.load(args.reader_checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("seed") != args.seed or checkpoint.get("graph_seed") != args.graph_seed:
        raise ValueError("L3 reader seed pairing mismatch")
    reader.load_state_dict(checkpoint["reader"]); reader.eval()
    for parameter in reader.parameters(): parameter.requires_grad_(False)
    cfg = SimpleNamespace(max_length=512, eval_batch_size=8)
    train_worlds, train_hidden, train_mask = core.cache_worlds(reader, tokenizer, worlds, "train", device, cfg)
    val_worlds, val_hidden, val_mask = core.cache_worlds(reader, tokenizer, worlds, "validation", device, cfg)
    train_rows = core.make_operator_rows(graph, train_worlds, records, "train", device)
    val_rows = core.make_operator_rows(graph, val_worlds, records, "validation", device)
    operator = build_operator(args.method, core, 60, 16, reader.hidden_size, 16).to(device)
    random_metrics = core.evaluate_operator(operator, reader, val_hidden, val_mask, val_rows, graph, "world16", cfg)
    wrapped = DDP(OperatorTrainModule(operator, reader, core, graph), device_ids=[local_rank], broadcast_buffers=False)
    optimizer = torch.optim.AdamW(wrapped.module.operator.parameters(), lr=5e-3, weight_decay=1e-4)
    generator = torch.Generator(device=device).manual_seed(args.seed * 100 + rank)
    history = []
    for step in range(args.operator_steps):
        idx = torch.randint(0, len(train_rows["labels"]), (16,), generator=generator, device=device)
        base = train_hidden[train_rows["world_idx"][idx]]
        mask = train_mask[train_rows["world_idx"][idx]]
        logits, laws = wrapped(base, mask, train_rows["intervention_id"][idx])
        per = F.cross_entropy(logits.flatten(0, 1), train_rows["labels"][idx].flatten(), reduction="none").reshape(len(idx), -1)
        changed, invariant = train_rows["changed"][idx], train_rows["causal_invariant"][idx]
        task = 0.5 * (((per * changed).sum(1) / changed.sum(1).clamp_min(1)).mean() + ((per * invariant).sum(1) / invariant.sum(1).clamp_min(1)).mean())
        loss = task + laws
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(wrapped.module.operator.parameters(), 1.0); optimizer.step()
        if step % 100 == 0 or step == args.operator_steps - 1:
            history.append({"step": step + 1, "loss": float(loss.detach()), "task": float(task.detach()), "laws": float(laws.detach())})
    dist.barrier()
    if rank == 0:
        trained = core.evaluate_operator(wrapped.module.operator, reader, val_hidden, val_mask, val_rows, graph, "world16", cfg)
        composed = core.composed_operator_metrics(wrapped.module.operator, reader, val_hidden, val_mask, val_worlds, graph, "world16", cfg)
        args.output_dir.mkdir(parents=True, exist_ok=False)
        torch.save({
            "operator": wrapped.module.operator.state_dict(), "method": args.method,
            "seed": args.seed, "graph_seed": args.graph_seed,
            "reader_sha256": cell["reader_sha256"], "manifest_sha256": args.expected_manifest_sha256,
            "cells_sha256": args.expected_cells_sha256, "test_evaluated": False,
        }, args.output_dir / "operator.pt")
        (args.output_dir / "run_summary.json").write_text(json.dumps({
            "protocol": "l3_operator_smoke_v1" if args.smoke else "l3_eight_method_law_v1", "cell_id": args.cell_id,
            "method": args.method, "seed": args.seed, "graph_seed": args.graph_seed,
            "world_size": 4, "random_init": random_metrics, "trained": trained,
            "composed": composed, "history": history, "reader_sha256": cell["reader_sha256"],
            "manifest_sha256": args.expected_manifest_sha256, "cells_sha256": args.expected_cells_sha256,
            "test_evaluated": False,
        }, indent=2, sort_keys=True) + "\n")
    dist.barrier(); dist.destroy_process_group(); return 0


if __name__ == "__main__":
    raise SystemExit(main())
