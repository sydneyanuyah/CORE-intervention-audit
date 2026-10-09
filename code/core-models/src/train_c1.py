#!/usr/bin/env python3
"""Train one registered fixed-O3 C1 condition on four BERT-base ranks."""

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

from train_f1 import OperatorTrainModule, apply_placebo, load_core, load_json, patch_reader_layers, setup


def sha256(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def balanced(prediction, target, factual):
    changed, unchanged = target != factual, target == factual
    sensitivity = (prediction[changed] == target[changed]).float().mean().item() if changed.any() else 1.0
    preservation = (prediction[unchanged] == target[unchanged]).float().mean().item() if unchanged.any() else 1.0
    return {"accuracy": (prediction == target).float().mean().item(), "change_accuracy": sensitivity, "preservation_accuracy": preservation, "two_edit_balanced": 0.5 * (sensitivity + preservation)}


def composed_metrics(operator, reader, hidden, mask, worlds, graph, records, condition, core, artifact_dir, device):
    selected = [world for world in worlds if world["split"] == "validation"]
    lookup = {(row["world_id"], row["intervention"]["target"], int(row["intervention"]["value"])): row for row in records if row["split"] == "validation"}
    world_indices, first_ids, second_ids, gold, factual = [], [], [], [], []
    spec = load_json(artifact_dir / "placebo_spec.json") if condition == "noncausal_placebo" else None
    nodes = graph["nodes"]
    for world_index, world in enumerate(selected):
        for first_index in range(0, len(nodes), 3):
            second_index = (first_index + 7) % len(nodes); first, second = nodes[first_index], nodes[second_index]
            first_value = 1 - int(world["factual_state"][first])
            if condition == "real":
                second_value = 1 - int(world["factual_state"][second]); final = core.scm_state(graph, world["root_values"], {first: first_value, second: second_value})
            elif condition == "noncausal_placebo":
                intermediate = apply_placebo(world["factual_state"], first, first_value, spec)
                second_value = 1 - int(intermediate[second]); final = apply_placebo(intermediate, second, second_value, spec)
            else:
                second_value = 1 - int(world["factual_state"][second]); final = lookup[(world["world_id"], second, second_value)]["intervened_state"]
            world_indices.append(world_index); first_ids.append(first_index * 2 + first_value); second_ids.append(second_index * 2 + second_value)
            gold.append([final[node] for node in nodes]); factual.append([world["factual_state"][node] for node in nodes])
    world_indices = torch.tensor(world_indices, device=device); first_ids = torch.tensor(first_ids, device=device); second_ids = torch.tensor(second_ids, device=device)
    target = torch.tensor(gold, device=device); factual_tensor = torch.tensor(factual, device=device); predictions = []
    operator.eval(); reader.eval()
    with torch.no_grad():
        for start in range(0, len(target), 8):
            sl = slice(start, start + 8); base = hidden[world_indices[sl]]; edited = operator(base[:, -16:], first_ids[sl]); edited = operator(edited, second_ids[sl])
            full = base.clone(); full[:, -16:] = edited; predictions.append(reader.decode(full, mask[world_indices[sl]]).argmax(-1))
    return {**balanced(torch.cat(predictions), target, factual_tensor), "examples": len(target)}


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--cell-id", required=True); parser.add_argument("--condition", choices=("real", "noncausal_placebo", "shuffled"), required=True)
    parser.add_argument("--seed", type=int, required=True); parser.add_argument("--graph-seed", type=int, required=True); parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True); parser.add_argument("--reader-checkpoint", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--core-components", type=Path, required=True); parser.add_argument("--expected-core-sha256", required=True)
    parser.add_argument("--manifest", type=Path, required=True); parser.add_argument("--expected-manifest-sha256", required=True); parser.add_argument("--cells", type=Path, required=True); parser.add_argument("--expected-cells-sha256", required=True)
    parser.add_argument("--operator-steps", type=int, choices=(1, 500), default=500); parser.add_argument("--smoke", action="store_true"); args = parser.parse_args()
    if args.smoke != (args.operator_steps == 1): raise ValueError("C1 smoke requires one step; production requires 500")
    if sha256(args.manifest) != args.expected_manifest_sha256 or sha256(args.cells) != args.expected_cells_sha256: raise ValueError("C1 registry hash mismatch")
    cells = load_json(args.cells)["cells"]; matches = [cell for cell in cells if cell["cell_id"] == args.cell_id]
    if len(matches) != 1: raise ValueError("C1 cell not uniquely registered")
    cell = matches[0]
    for key, value in {"condition": args.condition, "seed": args.seed, "graph_seed": args.graph_seed, "world_size": 4}.items():
        if cell.get(key) != value: raise ValueError(f"C1 cell mismatch: {key}")
    if sha256(args.records) != cell["records_sha256"] or sha256(args.reader_checkpoint) != cell["reader_sha256"]: raise ValueError("C1 input hash mismatch")
    if sha256(args.core_components) != args.expected_core_sha256: raise ValueError("C1 canonical source hash mismatch")
    local_rank, rank = setup("c1", 4); device = torch.device("cuda", local_rank); core = load_core(args.core_components); core.seed_all(args.seed)
    graph, worlds, records = load_json(args.artifact_dir / "graph.json"), load_json(args.artifact_dir / "worlds.json"), read_jsonl(args.records)
    if graph["seed"] != args.graph_seed or {world["split"] for world in worlds} != {"train", "validation"} or any(row["split"] not in {"train", "validation"} for row in records): raise ValueError("C1 data split/identity mismatch")
    original = transformers.AutoModel.from_pretrained
    def eager(*a, **kw): kw.setdefault("attn_implementation", "eager"); return original(*a, **kw)
    transformers.AutoModel.from_pretrained = eager; tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    reader = core.SharedWorldReader("google-bert/bert-base-uncased", 30, split_layer=4, max_world_tokens=16).to(device); patch_reader_layers(reader)
    checkpoint = torch.load(args.reader_checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("seed") != args.seed or checkpoint.get("graph_seed") != args.graph_seed or checkpoint.get("test_evaluated") is not False:
        raise ValueError("C1 reader pairing/provenance mismatch")
    reader.load_state_dict(checkpoint["reader"]); reader.eval()
    for parameter in reader.parameters(): parameter.requires_grad_(False)
    cfg = SimpleNamespace(max_length=512, eval_batch_size=8); train_worlds, train_hidden, train_mask = core.cache_worlds(reader, tokenizer, worlds, "train", device, cfg); val_worlds, val_hidden, val_mask = core.cache_worlds(reader, tokenizer, worlds, "validation", device, cfg)
    train_rows = core.make_operator_rows(graph, train_worlds, records, "train", device); val_rows = core.make_operator_rows(graph, val_worlds, records, "validation", device)
    operator = core.O3StateGated(60, reader.hidden_size, 16).to(device); wrapped = DDP(OperatorTrainModule(operator, reader, core, graph), device_ids=[local_rank], broadcast_buffers=False)
    optimizer = torch.optim.AdamW(wrapped.module.operator.parameters(), lr=5e-3, weight_decay=1e-4); generator = torch.Generator(device=device).manual_seed(args.seed * 100 + rank); history = []
    for step in range(args.operator_steps):
        idx = torch.randint(0, len(train_rows["labels"]), (16,), generator=generator, device=device); base = train_hidden[train_rows["world_idx"][idx]]; logits, laws = wrapped(base, train_mask[train_rows["world_idx"][idx]], train_rows["intervention_id"][idx])
        per = F.cross_entropy(logits.flatten(0, 1), train_rows["labels"][idx].flatten(), reduction="none").reshape(len(idx), -1); changed, unchanged = train_rows["changed"][idx], train_rows["unchanged"][idx]
        task = 0.5 * (((per * changed).sum(1) / changed.sum(1).clamp_min(1)).mean() + ((per * unchanged).sum(1) / unchanged.sum(1).clamp_min(1)).mean()); loss = task + laws
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(wrapped.module.operator.parameters(), 1.0); optimizer.step()
        if step % 100 == 0 or step == args.operator_steps - 1: history.append({"step": step + 1, "loss": float(loss.detach()), "task": float(task.detach()), "laws": float(laws.detach())})
    dist.barrier()
    if rank == 0:
        single = core.evaluate_operator(wrapped.module.operator, reader, val_hidden, val_mask, val_rows, graph, "world16", cfg); composed = composed_metrics(wrapped.module.operator, reader, val_hidden, val_mask, val_worlds, graph, records, args.condition, core, args.artifact_dir, device)
        args.output_dir.mkdir(parents=True, exist_ok=False); torch.save({"operator": wrapped.module.operator.state_dict(), "method": "o3", "condition": args.condition, "seed": args.seed, "graph_seed": args.graph_seed, "reader_sha256": cell["reader_sha256"], "test_evaluated": False}, args.output_dir / "operator.pt")
        (args.output_dir / "run_summary.json").write_text(json.dumps({"protocol": "c1_smoke_v1" if args.smoke else "c1_fixed_o3_closing_controls_v1", "cell_id": args.cell_id, "condition": args.condition, "method": "o3", "seed": args.seed, "graph_seed": args.graph_seed, "world_size": 4, "single": single, "composed": composed, "history": history, "reader_sha256": cell["reader_sha256"], "records_sha256": cell["records_sha256"], "manifest_sha256": args.expected_manifest_sha256, "cells_sha256": args.expected_cells_sha256, "test_evaluated": False}, indent=2, sort_keys=True) + "\n")
    dist.barrier(); dist.destroy_process_group(); return 0


if __name__ == "__main__": raise SystemExit(main())
