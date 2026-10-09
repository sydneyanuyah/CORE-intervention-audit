#!/usr/bin/env python3
"""Four-rank, train/validation-only F1 reader and O2/O3 training."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch
import torch.distributed as dist
import transformers
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import functional as F
from transformers import AutoTokenizer


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_core(path: Path):
    spec = importlib.util.spec_from_file_location("canonical_f1_core", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def patch_reader_layers(reader) -> None:
    """Preserve canonical reader semantics across tuple/tensor BertLayer APIs."""
    def run(layers, hidden, bias):
        for layer in layers:
            output = layer(hidden, attention_mask=bias)
            hidden = output[0] if isinstance(output, (tuple, list)) else output
        return hidden

    def trunk(self, input_ids, attention_mask):
        hidden = self.bert.embeddings(input_ids=input_ids)
        world = self.world_token_embeddings[None].expand(len(hidden), -1, -1)
        hidden = torch.cat([hidden, world], dim=1)
        world_mask = torch.ones(len(hidden), self.max_world_tokens, dtype=attention_mask.dtype, device=attention_mask.device)
        expanded = torch.cat([attention_mask, world_mask], dim=1)
        bias = self.attention_bias(expanded, hidden.dtype)
        return run(self.bert.encoder.layer[:self.split_layer], hidden, bias), expanded

    def decode(self, world_hidden, world_mask):
        queries = self.query_embeddings[None].expand(len(world_hidden), -1, -1)
        hidden = torch.cat([world_hidden, queries], dim=1)
        query_mask = torch.ones(len(hidden), len(self.query_embeddings), dtype=world_mask.dtype, device=world_mask.device)
        mask = torch.cat([world_mask, query_mask], dim=1)
        bias = self.attention_bias(mask, hidden.dtype)
        hidden = run(self.bert.encoder.layer[self.split_layer:], hidden, bias)
        return self.answer_head(hidden[:, -len(self.query_embeddings):])

    reader.trunk = MethodType(trunk, reader)
    reader.decode = MethodType(decode, reader)


class OperatorTrainModule(nn.Module):
    def __init__(self, operator, reader, core, graph):
        super().__init__()
        self.operator = operator
        self.reader = reader
        self.core = core
        self.graph = graph

    def forward(self, base, mask, intervention_id):
        state = base[:, -16:]
        edited_state = self.operator(state, intervention_id)
        edited = base.clone()
        edited[:, -16:] = edited_state
        logits = self.reader.decode(edited, mask)
        laws = self.core.law_losses(
            self.operator, state, intervention_id, 4, len(self.graph["nodes"]) * 2,
            len(self.graph["nodes"]),
        )
        return logits, torch.stack([laws[name] for name in (
            "identity", "idempotence", "commutation", "last_write_wins"
        )]).sum()


def apply_placebo(state, target, value, spec):
    out = dict(state)
    if out[target] == value:
        return out
    for node in spec["targets"][target]["changed_on_target_flip"]:
        out[node] = 1 - out[node]
    out[target] = int(value)
    return out


def placebo_composed_metrics(operator, reader, world_hidden, world_mask, worlds, graph, spec, core, cfg):
    indices, first_ids, second_ids, labels, factual, invariant_masks, target_masks = [], [], [], [], [], [], []
    nodes = graph["nodes"]
    for world_idx, world in enumerate(worlds):
        for first_idx in range(0, len(nodes), 3):
            second_idx = (first_idx + 7) % len(nodes)
            first, second = nodes[first_idx], nodes[second_idx]
            first_value = 1 - world["factual_state"][first]
            first_state = apply_placebo(world["factual_state"], first, first_value, spec)
            second_value = 1 - first_state[second]
            final_state = apply_placebo(first_state, second, second_value, spec)
            indices.append(world_idx); first_ids.append(first_idx * 2 + first_value)
            second_ids.append(second_idx * 2 + second_value); labels.append(final_state)
            factual.append(world["factual_state"])
            affected = set(spec["targets"][first]["support"]) | set(spec["targets"][second]["support"])
            invariant_masks.append([node not in affected for node in nodes])
            target_masks.append([node in (first, second) for node in nodes])
    device = world_hidden.device
    world_indices = torch.tensor(indices, device=device)
    first_ids = torch.tensor(first_ids, device=device); second_ids = torch.tensor(second_ids, device=device)
    labels = core.labels_tensor(labels, nodes, device); factual = core.labels_tensor(factual, nodes, device)
    invariant = torch.tensor(invariant_masks, dtype=torch.bool, device=device)
    targets = torch.tensor(target_masks, dtype=torch.bool, device=device)
    interface_idx = core.interface_indices(world_hidden, "world16", reader.max_world_tokens)
    predictions = []
    operator.eval(); reader.eval()
    with torch.no_grad():
        for start in range(0, len(labels), cfg.eval_batch_size):
            sl = slice(start, start + cfg.eval_batch_size)
            base = world_hidden[world_indices[sl]]; mask = world_mask[world_indices[sl]]
            state = operator(base[:, interface_idx], first_ids[sl]); state = operator(state, second_ids[sl])
            edited = base.clone(); edited[:, interface_idx] = state
            predictions.append(reader.decode(edited, mask).argmax(-1))
    pred = torch.cat(predictions); changed = labels != factual; unchanged = ~changed
    changed_non_target = changed & ~targets
    sensitivity = (pred[changed] == labels[changed]).float().mean().item()
    specificity = (pred[invariant] == labels[invariant]).float().mean().item()
    return {
        "accuracy": (pred == labels).float().mean().item(),
        "sensitivity": sensitivity,
        "changed_non_target_sensitivity": (pred[changed_non_target] == labels[changed_non_target]).float().mean().item(),
        "target_variable_success": (pred[targets] == labels[targets]).float().mean().item(),
        "non_support_specificity": specificity,
        "unchanged_accuracy": (pred[unchanged] == labels[unchanged]).float().mean().item(),
        "balanced_intervention_score": 0.5 * (sensitivity + specificity),
        "examples": len(labels),
    }


def setup(experiment: str, expected_world_size: int):
    if int(os.environ.get("WORLD_SIZE", "1")) != expected_world_size:
        raise RuntimeError(f"{experiment.upper()} requires exactly {expected_world_size} ranks")
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_rank()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", choices=("f1", "f2", "f3", "f4"), default="f1")
    p.add_argument("--phase", choices=("reader", "operator"), required=True)
    p.add_argument("--method", choices=("o2", "o3"))
    p.add_argument("--condition", choices=("real", "placebo"), default="real")
    p.add_argument("--model-size", choices=("base", "large"), default="base")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--graph-seed", type=int, required=True)
    p.add_argument("--artifact-dir", type=Path, required=True)
    p.add_argument("--reader-checkpoint", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--core-components", type=Path, required=True)
    p.add_argument("--expected-core-sha256", required=True)
    args = p.parse_args()
    pairings = {
        "f1": (range(101, 121), 2940),
        "f2": (range(301, 321), 2760),
        "f3": (range(601, 621), 2480),
        "f4": (range(701, 726), 2400),
    }
    allowed, offset = pairings[args.experiment]
    valid = args.seed in allowed and args.graph_seed == args.seed + offset
    if not valid:
        raise ValueError(f"unregistered {args.experiment.upper()} seed pairing")
    if args.experiment != "f3" and args.condition != "real":
        raise ValueError("placebo condition is registered only for F3")
    if args.experiment != "f4" and args.model_size != "base":
        raise ValueError("BERT-large is registered only for F4 in this runner")
    model_name = "google-bert/bert-base-uncased" if args.model_size == "base" else "google-bert/bert-large-uncased"
    expected_world_size = 4 if args.model_size == "base" else 8
    if sha256(args.core_components) != args.expected_core_sha256:
        raise ValueError("canonical O2/O3 source hash mismatch")
    local_rank, rank = setup(args.experiment, expected_world_size)
    device = torch.device("cuda", local_rank)
    core = load_core(args.core_components)
    core.seed_all(args.seed)
    graph = load_json(args.artifact_dir / "graph.json")
    worlds = load_json(args.artifact_dir / "worlds.json")
    records = load_json(args.artifact_dir / f"records_{args.condition}.json")
    if graph.get("seed") != args.graph_seed or {w["split"] for w in worlds} != {"train", "validation"}:
        raise ValueError(f"{args.experiment.upper()} artifact identity or split mismatch")
    if any(r["split"] not in {"train", "validation"} for r in records):
        raise ValueError(f"{args.experiment.upper()} records include a forbidden split")
    cfg = SimpleNamespace(max_length=512, eval_batch_size=8)
    original_from_pretrained = transformers.AutoModel.from_pretrained
    def eager_from_pretrained(*model_args, **model_kwargs):
        model_kwargs.setdefault("attn_implementation", "eager")
        return original_from_pretrained(*model_args, **model_kwargs)
    transformers.AutoModel.from_pretrained = eager_from_pretrained
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    reader = core.SharedWorldReader(
        model_name, 30, split_layer=4, max_world_tokens=16
    ).to(device)
    patch_reader_layers(reader)
    reader.initialize_queries(tokenizer, graph["nodes"])
    reader.initialize_world_tokens()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.phase == "reader":
        texts, states = core.reader_examples(graph, worlds, records, "train")
        encoded = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        labels = core.labels_tensor(states, graph["nodes"], torch.device("cpu"))
        model = DDP(reader, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        history = []
        for epoch in range(8):
            order = torch.randperm(len(labels), generator=torch.Generator().manual_seed(args.seed + epoch))
            order = order[rank::expected_world_size]
            losses = []
            reader_batch = 8 if args.model_size == "base" else 4
            for start in range(0, len(order), reader_batch):
                idx = order[start:start + reader_batch]
                ids = encoded["input_ids"][idx].to(device)
                mask = encoded["attention_mask"][idx].to(device)
                target = labels[idx].to(device)
                loss = F.cross_entropy(model(ids, mask).flatten(0, 1), target.flatten())
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
                losses.append(loss.detach())
            mean = torch.stack(losses).mean(); dist.all_reduce(mean); mean /= expected_world_size
            history.append({"epoch": epoch + 1, "train_loss": float(mean)})
        if rank == 0:
            torch.save({"reader": model.module.state_dict(), "seed": args.seed,
                        "graph_seed": args.graph_seed, "test_evaluated": False},
                       args.output_dir / "reader.pt")
            (args.output_dir / "run_summary.json").write_text(json.dumps({
                "protocol":f"{args.experiment}_reader_v1","seed":args.seed,"graph_seed":args.graph_seed,
                "model_size":args.model_size,"world_size":expected_world_size,"history":history,"test_evaluated":False}, indent=2) + "\n")
    else:
        if args.method is None or args.reader_checkpoint is None:
            raise ValueError("operator phase requires method and reader checkpoint")
        checkpoint = torch.load(args.reader_checkpoint, map_location="cpu", weights_only=True)
        if checkpoint.get("seed") != args.seed or checkpoint.get("graph_seed") != args.graph_seed:
            raise ValueError("reader checkpoint pairing mismatch")
        reader.load_state_dict(checkpoint["reader"]); reader.eval()
        for parameter in reader.parameters(): parameter.requires_grad_(False)
        train_worlds, train_hidden, train_mask = core.cache_worlds(reader, tokenizer, worlds, "train", device, cfg)
        val_worlds, val_hidden, val_mask = core.cache_worlds(reader, tokenizer, worlds, "validation", device, cfg)
        train_rows = core.make_operator_rows(graph, train_worlds, records, "train", device)
        val_rows = core.make_operator_rows(graph, val_worlds, records, "validation", device)
        operator = (core.O2ConditionalLowRank(60, 16, reader.hidden_size, 16)
                    if args.method == "o2" else core.O3StateGated(60, reader.hidden_size, 16)).to(device)
        wrapped = DDP(OperatorTrainModule(operator, reader, core, graph), device_ids=[local_rank], broadcast_buffers=False)
        optimizer = torch.optim.AdamW(wrapped.module.operator.parameters(), lr=5e-3, weight_decay=1e-4)
        history = []
        generator = torch.Generator(device=device).manual_seed(args.seed * 100 + rank)
        operator_batch = 16 if args.model_size == "base" else 8
        for step in range(500):
            idx = torch.randint(0, len(train_rows["labels"]), (operator_batch,), generator=generator, device=device)
            base = train_hidden[train_rows["world_idx"][idx]]
            mask = train_mask[train_rows["world_idx"][idx]]
            intervention_id = train_rows["intervention_id"][idx]
            logits, laws = wrapped(base, mask, intervention_id)
            per = F.cross_entropy(logits.flatten(0, 1), train_rows["labels"][idx].flatten(), reduction="none").reshape(len(idx), -1)
            changed = train_rows["changed"][idx]; invariant = train_rows["causal_invariant"][idx]
            task = 0.5 * ((per * changed).sum(1) / changed.sum(1).clamp_min(1)).mean() + 0.5 * ((per * invariant).sum(1) / invariant.sum(1).clamp_min(1)).mean()
            loss = task + laws
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(wrapped.module.operator.parameters(), 1.0); optimizer.step()
            if step % 100 == 0 or step == 499:
                history.append({"step":step + 1,"loss":float(loss.detach()),"task":float(task.detach()),"laws":float(laws.detach())})
        dist.barrier()
        if rank == 0:
            op = wrapped.module.operator
            single = core.evaluate_operator(op, reader, val_hidden, val_mask, val_rows, graph, "world16", cfg)
            if args.condition == "placebo":
                spec = load_json(args.artifact_dir / "placebo_spec.json")
                if spec.get("is_dag") is not False:
                    raise ValueError("F3 placebo must be explicitly non-DAG")
                composed = placebo_composed_metrics(op, reader, val_hidden, val_mask, val_worlds, graph, spec, core, cfg)
            else:
                composed = core.composed_operator_metrics(op, reader, val_hidden, val_mask, val_worlds, graph, "world16", cfg)
            torch.save({"operator":op.state_dict(),"method":args.method,"seed":args.seed,
                        "graph_seed":args.graph_seed,"reader_sha256":sha256(args.reader_checkpoint),
                        "test_evaluated":False}, args.output_dir / "operator.pt")
            (args.output_dir / "run_summary.json").write_text(json.dumps({
                "protocol":f"{args.experiment}_cell_v1","method":args.method,"seed":args.seed,
                "graph_seed":args.graph_seed,"condition":args.condition,"model_size":args.model_size,
                "world_size":expected_world_size,"single":single,"composed":composed,
                "history":history,"reader_sha256":sha256(args.reader_checkpoint),
                "test_evaluated":False}, indent=2, sort_keys=True) + "\n")
    dist.barrier(); dist.destroy_process_group(); return 0


if __name__ == "__main__":
    raise SystemExit(main())
