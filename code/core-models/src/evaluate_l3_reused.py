#!/usr/bin/env python3
"""Four-rank validation-only L3 scoring for frozen F2 prompting/LoRA/O2/O3 cells."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import transformers
from transformers import AutoTokenizer

from train_f1 import load_core, load_json, patch_reader_layers, setup
from train_f2_baselines import install_lora, prompting_metrics


METHODS = {"prompting", "lora_matched", "o2", "o3"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def predict(reader, tokenizer, texts, device, batch_size=8):
    rows = []
    reader.eval()
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(texts[start:start + batch_size], padding=True, truncation=True, max_length=512, return_tensors="pt")
            rows.append(reader(batch["input_ids"].to(device), batch["attention_mask"].to(device)).argmax(-1))
    return torch.cat(rows)


def prompt_law_metrics(reader, tokenizer, core, graph, worlds, records, device):
    lookup = {world["world_id"]: world for world in worlds if world["split"] == "validation"}
    nodes, node_index = graph["nodes"], {node: index for index, node in enumerate(graph["nodes"])}
    identity_texts, once_texts, twice_texts, fs_texts, sf_texts, seq_texts, last_texts = ([] for _ in range(7))
    factual_states, once_states, commute_states = [], [], []
    for record in records:
        if record["split"] != "validation": continue
        world = lookup[record["world_id"]]; passage = world["passage"]
        target, value = record["intervention"]["target"], int(record["intervention"]["value"])
        command = core.intervention_text(target, value)
        other = nodes[(node_index[target] + 7) % len(nodes)]; other_value = 1 - value
        other_command = core.intervention_text(other, other_value)
        opposite = core.intervention_text(target, 1 - value)
        identity_texts.append(passage)
        once_texts.append(f"{passage};{command}")
        twice_texts.append(f"{passage};{command}&{command}")
        fs_texts.append(f"{passage};{command}&{other_command}")
        sf_texts.append(f"{passage};{other_command}&{command}")
        seq_texts.append(f"{passage};{opposite}&{command}")
        last_texts.append(f"{passage};{command}")
        factual_states.append(world["factual_state"]); once_states.append(record["intervened_state"])
        commute_states.append(core.scm_state(graph, world["root_values"], {target: value, other: other_value}))
    factual = core.labels_tensor(factual_states, nodes, device)
    once_gold = core.labels_tensor(once_states, nodes, device)
    commute_gold = core.labels_tensor(commute_states, nodes, device)
    identity = predict(reader, tokenizer, identity_texts, device)
    once = predict(reader, tokenizer, once_texts, device)
    twice = predict(reader, tokenizer, twice_texts, device)
    fs = predict(reader, tokenizer, fs_texts, device); sf = predict(reader, tokenizer, sf_texts, device)
    seq = predict(reader, tokenizer, seq_texts, device); last = predict(reader, tokenizer, last_texts, device)
    score = lambda pred, gold: (pred == gold).float().mean().item()
    agree = lambda left, right: (left == right).float().mean().item()
    return {
        "identity_answer_agreement": 1.0,
        "identity_scm_correctness": score(identity, factual),
        "idempotence_answer_agreement": agree(twice, once),
        "idempotence_scm_correctness": score(twice, once_gold),
        "commutation_answer_agreement": agree(fs, sf),
        "commutation_scm_correctness": 0.5 * (score(fs, commute_gold) + score(sf, commute_gold)),
        "last_write_wins_answer_agreement": agree(seq, last),
        "last_write_wins_scm_correctness": score(seq, once_gold),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument("--seed", type=int, required=True); parser.add_argument("--graph-seed", type=int, required=True)
    parser.add_argument("--cell-id", required=True); parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--reader-checkpoint", type=Path, required=True); parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--core-components", type=Path, required=True)
    parser.add_argument("--expected-core-sha256", required=True); parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True); parser.add_argument("--cells", type=Path, required=True)
    parser.add_argument("--expected-cells-sha256", required=True)
    args = parser.parse_args()
    if sha256(args.manifest) != args.expected_manifest_sha256 or sha256(args.cells) != args.expected_cells_sha256:
        raise ValueError("L3 manifest or registry hash mismatch")
    registry = load_json(args.cells); matches = [row for row in registry["cells"] if row["cell_id"] == args.cell_id]
    if len(matches) != 1: raise ValueError("L3 cell is not uniquely registered")
    cell = matches[0]
    if any(cell.get(key) != value for key, value in {"method": args.method, "seed": args.seed, "graph_seed": args.graph_seed, "world_size": 4}.items()):
        raise ValueError("L3 cell contract mismatch")
    if sha256(args.reader_checkpoint) != cell["reader_sha256"] or sha256(args.source_checkpoint) != cell["source_checkpoint_sha256"]:
        raise ValueError("L3 frozen checkpoint hash mismatch")
    if sha256(args.core_components) != args.expected_core_sha256: raise ValueError("CORE source hash mismatch")
    for name in ("graph.json", "worlds.json", "records_real.json"):
        if sha256(args.artifact_dir / name) != cell["artifact_sha256"][name]: raise ValueError(f"artifact hash mismatch: {name}")

    local_rank, rank = setup("l3", 4); device = torch.device("cuda", local_rank)
    core = load_core(args.core_components); core.seed_all(args.seed)
    graph, worlds, records = (load_json(args.artifact_dir / name) for name in ("graph.json", "worlds.json", "records_real.json"))
    if graph.get("seed") != args.graph_seed or any(row["split"] not in {"train", "validation"} for row in records): raise ValueError("forbidden L3 data")
    original = transformers.AutoModel.from_pretrained
    def eager(*model_args, **model_kwargs): model_kwargs.setdefault("attn_implementation", "eager"); return original(*model_args, **model_kwargs)
    transformers.AutoModel.from_pretrained = eager
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    reader = core.SharedWorldReader("google-bert/bert-base-uncased", 30, split_layer=4, max_world_tokens=16).to(device); patch_reader_layers(reader)
    reader_state = torch.load(args.reader_checkpoint, map_location="cpu", weights_only=True); reader.load_state_dict(reader_state["reader"])
    for parameter in reader.parameters(): parameter.requires_grad_(False)
    source = torch.load(args.source_checkpoint, map_location="cpu", weights_only=True)
    cfg = SimpleNamespace(max_length=512, eval_batch_size=8)
    if args.method in {"o2", "o3"}:
        cls = core.O2ConditionalLowRank if args.method == "o2" else core.O3StateGated
        make = (lambda: cls(60, 16, reader.hidden_size, 16)) if args.method == "o2" else (lambda: cls(60, reader.hidden_size, 16))
        operator = make().to(device)
        val_worlds, hidden, mask = core.cache_worlds(reader, tokenizer, worlds, "validation", device, cfg); rows = core.make_operator_rows(graph, val_worlds, records, "validation", device)
        random_metrics = core.evaluate_operator(operator, reader, hidden, mask, rows, graph, "world16", cfg)
        operator.load_state_dict(source["operator"])
        trained = core.evaluate_operator(operator, reader, hidden, mask, rows, graph, "world16", cfg)
        composed = core.composed_operator_metrics(operator, reader, hidden, mask, val_worlds, graph, "world16", cfg)
    else:
        if args.method == "lora_matched":
            install_lora(reader, int(source["lora_rank"])); random_metrics = prompt_law_metrics(reader, tokenizer, core, graph, worlds, records, device)
            reader.load_state_dict(source["lora"], strict=False)
        else:
            random_metrics = prompt_law_metrics(reader, tokenizer, core, graph, worlds, records, device)
        trained = prompt_law_metrics(reader, tokenizer, core, graph, worlds, records, device)
        composed = core.composed_prompting(reader, tokenizer, graph, worlds, "validation", device, cfg)
        trained["single"] = prompting_metrics(reader, tokenizer, core, graph, worlds, records, "validation", device, cfg)
    dist.barrier()
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        (args.output_dir / "run_summary.json").write_text(json.dumps({
            "protocol": "l3_eight_method_law_v1", "cell_id": args.cell_id, "method": args.method,
            "seed": args.seed, "graph_seed": args.graph_seed, "execution": "frozen_f2_checkpoint_validation_rescore_v1",
            "world_size": 4, "random_init": random_metrics, "trained": trained, "composed": composed,
            "reader_sha256": cell["reader_sha256"], "source_checkpoint_sha256": cell["source_checkpoint_sha256"],
            "manifest_sha256": args.expected_manifest_sha256, "cells_sha256": args.expected_cells_sha256,
            "prompting_random_init_degenerate": args.method == "prompting", "test_evaluated": False,
        }, indent=2, sort_keys=True) + "\n")
    dist.barrier(); dist.destroy_process_group(); return 0


if __name__ == "__main__":
    raise SystemExit(main())
