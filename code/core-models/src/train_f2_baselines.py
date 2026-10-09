#!/usr/bin/env python3
"""Four-rank F2 prompting and parameter-matched LoRA cells."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import transformers
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer

from train_f1 import load_core, load_json, patch_reader_layers, setup


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class LoRALinear(nn.Module):
    """Frozen linear layer with a zero-initialized low-rank residual."""

    def __init__(self, base: nn.Linear, rank: int):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        for parameter in base.parameters():
            parameter.requires_grad_(False)
        self.lora_a = nn.Parameter(torch.empty(
            rank, base.in_features, device=base.weight.device, dtype=base.weight.dtype
        ))
        self.lora_b = nn.Parameter(torch.zeros(
            base.out_features, rank, device=base.weight.device, dtype=base.weight.dtype
        ))
        nn.init.kaiming_uniform_(self.lora_a, a=5 ** 0.5)

    def forward(self, inputs):
        return self.base(inputs) + F.linear(F.linear(inputs, self.lora_a), self.lora_b)


def lora_parameter_count(reader, rank: int) -> int:
    return sum(
        rank * (linear.in_features + linear.out_features)
        for layer in reader.bert.encoder.layer
        for linear in (layer.attention.self.query, layer.attention.self.value)
    )


def select_lora_rank(reader, target: int) -> int:
    """Closest rank to O3's count; deterministic smaller-rank tie break."""
    return min(
        range(1, 129),
        key=lambda rank: (abs(lora_parameter_count(reader, rank) - target), rank),
    )


def install_lora(reader, rank: int) -> None:
    for layer in reader.bert.encoder.layer:
        layer.attention.self.query = LoRALinear(layer.attention.self.query, rank)
        layer.attention.self.value = LoRALinear(layer.attention.self.value, rank)


def prompting_examples(core, graph, worlds, records, split):
    selected = [record for record in records if record["split"] == split]
    lookup = {world["world_id"]: world for world in worlds if world["split"] == split}
    descendants = core.descendant_map(graph)
    texts, labels, factual, invariant, targets = [], [], [], [], []
    for record in selected:
        world = lookup[record["world_id"]]
        target = record["intervention"]["target"]
        affected = {target, *descendants[target]}
        texts.append(f"{world['passage']};{record['intervention']['text']}")
        labels.append(record["intervened_state"])
        factual.append(world["factual_state"])
        invariant.append([node not in affected for node in graph["nodes"]])
        targets.append([node == target for node in graph["nodes"]])
    return texts, labels, factual, invariant, targets


def prompting_metrics(reader, tokenizer, core, graph, worlds, records, split, device, cfg):
    texts, states, facts, invariants, targets = prompting_examples(
        core, graph, worlds, records, split
    )
    labels = core.labels_tensor(states, graph["nodes"], device)
    factual = core.labels_tensor(facts, graph["nodes"], device)
    invariant = torch.tensor(invariants, dtype=torch.bool, device=device)
    target_mask = torch.tensor(targets, dtype=torch.bool, device=device)
    predictions = []
    reader.eval()
    with torch.no_grad():
        for start in range(0, len(texts), cfg.eval_batch_size):
            batch = tokenizer(
                texts[start:start + cfg.eval_batch_size], padding=True, truncation=True,
                max_length=cfg.max_length, return_tensors="pt",
            )
            predictions.append(reader(
                batch["input_ids"].to(device), batch["attention_mask"].to(device)
            ).argmax(-1))
    prediction = torch.cat(predictions)
    changed = labels != factual
    unchanged = ~changed
    changed_descendant = changed & ~target_mask
    sensitivity = (prediction[changed] == labels[changed]).float().mean().item()
    specificity = (prediction[invariant] == labels[invariant]).float().mean().item()
    unchanged_accuracy = (prediction[unchanged] == labels[unchanged]).float().mean().item()
    return {
        "accuracy": (prediction == labels).float().mean().item(),
        "sensitivity": sensitivity,
        "unchanged_accuracy": unchanged_accuracy,
        "non_descendant_specificity": specificity,
        "changed_descendant_sensitivity": (
            (prediction[changed_descendant] == labels[changed_descendant]).float().mean().item()
        ),
        "target_variable_success": (
            (prediction[target_mask] == labels[target_mask]).float().mean().item()
        ),
        "balanced_intervention_score": 0.5 * (sensitivity + specificity),
        "balanced_numerical_score": 0.5 * (sensitivity + unchanged_accuracy),
        "examples": len(texts),
    }


def validate_registry(args) -> None:
    if sha256(args.manifest) != args.expected_manifest_sha256:
        raise ValueError("F2 manifest hash mismatch")
    if sha256(args.cells) != args.expected_cells_sha256:
        raise ValueError("F2 cells hash mismatch")
    if sha256(args.amendment) != args.expected_amendment_sha256:
        raise ValueError("F2 baseline execution amendment hash mismatch")
    manifest = load_json(args.manifest)
    amendment = load_json(args.amendment)
    if manifest.get("allowed_splits") != ["train", "validation"]:
        raise ValueError("F2 allowed splits changed")
    if manifest.get("test_evaluated") is not False:
        raise ValueError("F2 test lock changed")
    if amendment.get("base_manifest_sha256") != args.expected_manifest_sha256:
        raise ValueError("F2 amendment base-manifest binding mismatch")
    if amendment.get("base_cells_sha256") != args.expected_cells_sha256:
        raise ValueError("F2 amendment cells binding mismatch")
    if amendment.get("allowed_splits") != ["train", "validation"]:
        raise ValueError("F2 amendment allowed splits changed")
    if amendment.get("test_evaluated") is not False:
        raise ValueError("F2 amendment test lock changed")
    cell_id = f"f2:xor:{args.method}:{args.seed}"
    cells = load_json(args.cells).get("cells", [])
    matches = [cell for cell in cells if cell.get("cell_id") == cell_id]
    if len(matches) != 1 or matches[0].get("world_size") != 4:
        raise ValueError(f"unregistered F2 cell: {cell_id}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=("prompting", "lora_matched"))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--graph-seed", type=int, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--reader-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--core-components", type=Path, required=True)
    parser.add_argument("--expected-core-sha256", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--cells", type=Path, required=True)
    parser.add_argument("--expected-cells-sha256", required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--expected-amendment-sha256", required=True)
    args = parser.parse_args()
    if args.seed not in range(301, 321) or args.graph_seed != args.seed + 2760:
        raise ValueError("unregistered F2 seed pairing")
    validate_registry(args)
    if sha256(args.core_components) != args.expected_core_sha256:
        raise ValueError("canonical O2/O3 source hash mismatch")

    local_rank, rank = setup("f2")
    device = torch.device("cuda", local_rank)
    core = load_core(args.core_components)
    core.seed_all(args.seed)
    graph = load_json(args.artifact_dir / "graph.json")
    worlds = load_json(args.artifact_dir / "worlds.json")
    records = load_json(args.artifact_dir / "records_real.json")
    if graph.get("seed") != args.graph_seed:
        raise ValueError("F2 graph identity mismatch")
    if {world["split"] for world in worlds} != {"train", "validation"}:
        raise ValueError("F2 worlds must contain exactly train and validation")
    if any(record["split"] not in {"train", "validation"} for record in records):
        raise ValueError("F2 records include a forbidden split")

    original = transformers.AutoModel.from_pretrained
    def eager(*model_args, **model_kwargs):
        model_kwargs.setdefault("attn_implementation", "eager")
        return original(*model_args, **model_kwargs)
    transformers.AutoModel.from_pretrained = eager
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    reader = core.SharedWorldReader(
        "google-bert/bert-base-uncased", 30, split_layer=4, max_world_tokens=16
    ).to(device)
    patch_reader_layers(reader)
    checkpoint = torch.load(args.reader_checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("seed") != args.seed or checkpoint.get("graph_seed") != args.graph_seed:
        raise ValueError("reader checkpoint pairing mismatch")
    reader.load_state_dict(checkpoint["reader"])
    for parameter in reader.parameters():
        parameter.requires_grad_(False)
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    cfg = SimpleNamespace(max_length=512, eval_batch_size=8)
    history, extra = [], {}

    if args.method == "lora_matched":
        target_count = sum(
            parameter.numel()
            for parameter in core.O3StateGated(60, reader.hidden_size, 16).parameters()
        )
        lora_rank = select_lora_rank(reader, target_count)
        selected_count = lora_parameter_count(reader, lora_rank)
        install_lora(reader, lora_rank)
        texts, states, facts, invariants, _ = prompting_examples(
            core, graph, worlds, records, "train"
        )
        encoded = tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )
        labels = core.labels_tensor(states, graph["nodes"], torch.device("cpu"))
        factual = core.labels_tensor(facts, graph["nodes"], torch.device("cpu"))
        changed = labels != factual
        invariant = torch.tensor(invariants, dtype=torch.bool)
        model = DDP(reader, device_ids=[local_rank], broadcast_buffers=False)
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if sum(parameter.numel() for parameter in trainable) != selected_count:
            raise RuntimeError("LoRA trainable parameter count mismatch")
        optimizer = torch.optim.AdamW(trainable, lr=1e-3, weight_decay=1e-4)
        generator = torch.Generator().manual_seed(args.seed * 100 + rank)
        for step in range(500):
            idx = torch.randint(0, len(labels), (16,), generator=generator)
            logits = model(
                encoded["input_ids"][idx].to(device),
                encoded["attention_mask"][idx].to(device),
            )
            target = labels[idx].to(device)
            per = F.cross_entropy(
                logits.flatten(0, 1), target.flatten(), reduction="none"
            ).reshape(len(idx), -1)
            changed_mask = changed[idx].to(device)
            invariant_mask = invariant[idx].to(device)
            task = 0.5 * (
                ((per * changed_mask).sum(1) / changed_mask.sum(1).clamp_min(1)).mean()
                + ((per * invariant_mask).sum(1) / invariant_mask.sum(1).clamp_min(1)).mean()
            )
            optimizer.zero_grad(set_to_none=True)
            task.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            if step % 100 == 0 or step == 499:
                mean = task.detach().clone()
                dist.all_reduce(mean)
                mean /= 4
                history.append({"step": step + 1, "task": float(mean)})
        reader = model.module
        extra = {
            "lora_rank": lora_rank,
            "trainable_parameter_count": selected_count,
            "o3_parameter_count": target_count,
            "parameter_count_delta": selected_count - target_count,
        }

    dist.barrier()
    if rank == 0:
        single = prompting_metrics(
            reader, tokenizer, core, graph, worlds, records, "validation", device, cfg
        )
        composed = core.composed_prompting(
            reader, tokenizer, graph, worlds, "validation", device, cfg
        )
        state = {
            name: parameter.detach().cpu()
            for name, parameter in reader.named_parameters()
            if name.endswith(("lora_a", "lora_b"))
        }
        torch.save({
            "method": args.method, "lora": state, "seed": args.seed,
            "graph_seed": args.graph_seed,
            "reader_sha256": sha256(args.reader_checkpoint),
            "manifest_sha256": args.expected_manifest_sha256,
            "cells_sha256": args.expected_cells_sha256,
            "amendment_sha256": args.expected_amendment_sha256,
            "runner_sha256": sha256(Path(__file__)),
            **extra, "test_evaluated": False,
        }, args.output_dir / "operator.pt")
        (args.output_dir / "run_summary.json").write_text(json.dumps({
            "protocol": "f2_cell_v1", "method": args.method, "seed": args.seed,
            "graph_seed": args.graph_seed, "world_size": 4, "single": single,
            "composed": composed, "history": history,
            "reader_sha256": sha256(args.reader_checkpoint),
            "manifest_sha256": args.expected_manifest_sha256,
            "cells_sha256": args.expected_cells_sha256,
            "amendment_sha256": args.expected_amendment_sha256,
            "runner_sha256": sha256(Path(__file__)),
            **extra, "test_evaluated": False,
        }, indent=2, sort_keys=True) + "\n")
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
