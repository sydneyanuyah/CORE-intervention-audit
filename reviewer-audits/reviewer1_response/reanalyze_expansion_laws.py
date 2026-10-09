#!/usr/bin/env python3
"""Measure selective invariance from frozen expansion L1/L2/L3 checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn

ROOT_HINT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_HINT / "src"))
from core_7b.operator_probe import O2ConditionalLowRank, O3StateGated, StateHead


class Add(nn.Module):
    def __init__(self, count, hidden):
        super().__init__(); self.e = nn.Embedding(count, hidden)
    def forward(self, state, intervention):
        return state + self.e(intervention)


class Router(nn.Module):
    def __init__(self, count, hidden):
        super().__init__(); self.e = nn.Embedding(count, hidden); self.g = nn.Linear(hidden, 1)
    def forward(self, state, intervention):
        return state + torch.sigmoid(self.g(state)) * self.e(intervention)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text())


def operator_for(method: str, hidden: int):
    if method in {"do_nothing", "do_everything", "prompting", "task_vector_add", "o1"}:
        return Add(61, hidden)
    if method == "router":
        return Router(61, hidden)
    if method in {"lora_matched", "loreft", "o2"}:
        return O2ConditionalLowRank(61, hidden, 16)
    return O3StateGated(61, hidden, 16)


def t_interval(values: list[float]) -> list[float]:
    if len(values) < 2:
        return [values[0], values[0]]
    critical = {5: 2.7764451051977987, 20: 2.093024054408263}.get(len(values), 1.96)
    mean = statistics.fmean(values)
    half = critical * statistics.stdev(values) / math.sqrt(len(values))
    return [mean - half, mean + half]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--prefix", required=True, choices=("qwen7b", "phi14b", "qwen32", "llama70b"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); root = args.root.resolve()
    rows = []
    for task in ("l1", "l2", "l3"):
        manifest_path = root / "registry" / f"{args.prefix}_{task}_manifest.json"
        manifest = load_json(manifest_path)
        source_manifest_path = Path(manifest["source_manifest"])
        source_amendment_path = Path(manifest["source_amendment"])
        if not source_manifest_path.is_absolute(): source_manifest_path = root / source_manifest_path
        if not source_amendment_path.is_absolute(): source_amendment_path = root / source_amendment_path
        if sha256(source_manifest_path) != manifest["source_manifest_sha256"] or sha256(source_amendment_path) != manifest["source_amendment_sha256"]:
            raise ValueError(f"{task} source binding mismatch")
        source, amendment = load_json(source_manifest_path), load_json(source_amendment_path)
        cache_path = Path(amendment["feature_cache"])
        if not cache_path.is_absolute(): cache_path = root / cache_path
        if sha256(cache_path) != amendment["feature_cache_sha256"]:
            raise ValueError(f"{task} feature-cache mismatch")
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        lookup = {(int(row["graph_seed"]), row["world_id"]): payload["features"][index].float() for index, row in enumerate(payload["records"])}
        nodes = source["operator_contract"]["nodes"]
        source_cells = {int(row["seed"]): row for row in source["cells"]}
        source_graphs = {int(row["graph_seed"]): row for row in source["graphs"]}
        packed = {}
        for source_seed in sorted({int(cell["source_seed"]) for cell in manifest["cells"]}):
            graph_seed = int(source_cells[source_seed]["graph_seed"])
            graph = source_graphs[graph_seed]
            records = load_json(Path(graph["files"]["records_real.json"]["path"]))
            worlds = {row["world_id"]: row for row in load_json(Path(graph["files"]["worlds.json"]["path"]))}
            validation = [row for row in records if row["split"] == "validation"]
            features = torch.stack([lookup[(graph_seed, row["world_id"])] for row in validation])
            interventions = torch.tensor([nodes.index(row["intervention"]["target"]) * 2 + int(row["intervention"]["value"]) for row in validation])
            labels = torch.tensor([[int(row["intervened_state"][node]) for node in nodes] for row in validation])
            factual = torch.tensor([[int(worlds[row["world_id"]]["factual_state"][node]) for node in nodes] for row in validation])
            packed[source_seed] = (features, interventions, labels, factual)
        for cell in sorted(manifest["cells"], key=lambda row: (row["method"], int(row["seed"]))):
            output = root / cell["output"]
            checkpoint_path, summary_path = output / "operator.pt", output / "run_summary.json"
            checkpoint, summary = torch.load(checkpoint_path, map_location="cpu", weights_only=True), load_json(summary_path)
            if sha256(checkpoint_path) != summary["checkpoint_sha256"]:
                raise ValueError(f"checkpoint hash mismatch: {checkpoint_path}")
            operator = operator_for(cell["method"], int(payload["hidden_size"])); head = StateHead(int(payload["hidden_size"]), 30)
            operator.load_state_dict(checkpoint["operator"]); head.load_state_dict(checkpoint["head"])
            operator.eval(); head.eval()
            features, interventions, labels, factual = packed[int(cell["source_seed"])]
            with torch.inference_mode(): predictions = head(operator(features, interventions)).argmax(-1)
            changed = labels != factual; preserved = ~changed
            change = float((predictions[changed] == labels[changed]).float().mean())
            preservation = float((predictions[preserved] == labels[preserved]).float().mean())
            balanced = 0.5 * (change + preservation)
            if abs(balanced - float(summary["validation_balanced_accuracy"])) > 2e-7:
                raise ValueError(f"frozen rescore mismatch: {cell['cell_id']}")
            rows.append({
                "backbone": source["model"], "model_revision": source["model_revision"],
                "experiment": task.upper(), "cell_id": cell["cell_id"], "method": cell["method"],
                "seed": int(cell["seed"]), "source_seed": int(cell["source_seed"]),
                "non_descendant_preservation": preservation,
                "descendant_change_sensitivity": change,
                "balanced_selective_invariance": balanced,
                "checkpoint_path": str(checkpoint_path.relative_to(root)),
                "checkpoint_sha256": summary["checkpoint_sha256"],
                "metric_source_path": str(summary_path.relative_to(root)),
                "metric_source_sha256": sha256(summary_path),
                "manifest_path": str(manifest_path.relative_to(root)),
                "manifest_sha256": sha256(manifest_path),
                "evaluation_split": "validation", "test_evaluated": False,
            })
    grouped = defaultdict(list)
    for row in rows: grouped[(row["backbone"], row["experiment"], row["method"])].append(row)
    macro = []
    for (backbone, experiment, method), group in sorted(grouped.items()):
        item = {"backbone": backbone, "experiment": experiment, "method": method, "n_seeds": len(group)}
        for key in ("non_descendant_preservation", "descendant_change_sensitivity", "balanced_selective_invariance"):
            values = [row[key] for row in group]
            item[key] = {"mean": statistics.fmean(values), "student_t_95": t_interval(values)}
        macro.append(item)
    result = {"protocol": "reviewer1_selective_invariance_frozen_checkpoint_rescore_v1", "per_seed": rows, "macro": macro, "test_evaluated": False}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"backbone": rows[0]["backbone"], "cells": len(rows), "macro_rows": len(macro)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
