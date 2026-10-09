#!/usr/bin/env python3
"""Recompute frozen F2 checkpoint predictions and paired graph-bootstrap uncertainty."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT_HINT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_HINT / "src"))
from core_7b.operator_probe import O2ConditionalLowRank, O3StateGated, StateHead, intervention_id


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text())


def t_interval(values: list[float]) -> list[float]:
    if len(values) < 2:
        return [values[0], values[0]]
    mean = statistics.fmean(values)
    # Exact two-sided 95% Student-t critical for df=19; manifests require 20 seeds.
    critical = 2.093024054408263
    half = critical * statistics.stdev(values) / math.sqrt(len(values))
    return [mean - half, mean + half]


def graph_bootstrap(graph_deltas: dict[str, float], *, draws: int = 10000, seed: int = 917031) -> list[float]:
    keys = sorted(graph_deltas)
    rng = random.Random(seed)
    samples = []
    for _ in range(draws):
        samples.append(statistics.fmean(graph_deltas[rng.choice(keys)] for _ in keys))
    samples.sort()
    return [samples[int(0.025 * draws)], samples[int(0.975 * draws) - 1]]


def score_predictions(validation, predictions):
    change_correct = change_count = preservation_correct = preservation_count = 0
    by_graph = defaultdict(lambda: [0, 0, 0, 0])
    for pair, predicted in zip(validation, predictions):
        gold = pair["gold_outputs"]
        graph = pair["graph_group_id"]
        for index, value in enumerate(predicted):
            if pair["change_mask"][index]:
                change_correct += int(value == gold[index]); change_count += 1
                by_graph[graph][0] += int(value == gold[index]); by_graph[graph][1] += 1
            if pair["preservation_mask"][index]:
                preservation_correct += int(value == gold[index]); preservation_count += 1
                by_graph[graph][2] += int(value == gold[index]); by_graph[graph][3] += 1
    change = change_correct / change_count
    preservation = preservation_correct / preservation_count
    graph_scores = {
        graph: 0.5 * ((counts[0] / counts[1]) + (counts[2] / counts[3]))
        for graph, counts in by_graph.items()
    }
    return {
        "change_accuracy": change,
        "preservation_accuracy": preservation,
        "two_edit_balanced": 0.5 * (change + preservation),
        "graph_scores": graph_scores,
    }


def predict_cell(root: Path, manifest: dict, amendment: dict, cell: dict, validation, records, feature_by_model):
    checkpoint_path = root / cell["output"] / "operator.pt"
    summary_path = root / cell["output"] / "run_summary.json"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    summary = load_json(summary_path)
    if sha256(checkpoint_path) != summary["checkpoint_sha256"]:
        raise ValueError(f"checkpoint hash mismatch: {checkpoint_path}")
    method = cell["method"]
    hidden_size = int(checkpoint["hidden_size"])
    nodes = manifest["operator_contract"]["node_order"]
    intervention_count = len(nodes) * 2
    operator = O2ConditionalLowRank(intervention_count, hidden_size, 16) if method == "o2" else O3StateGated(intervention_count, hidden_size, 16)
    head = StateHead(hidden_size, len(nodes))
    operator.load_state_dict(checkpoint["operator"]); head.load_state_dict(checkpoint["head"])
    operator.eval(); head.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(validation), 128):
            batch = validation[start:start + 128]
            first = [records[row["first_record_id"]] for row in batch]
            second = [records[row["second_record_id"]] for row in batch]
            states = torch.stack([feature_by_model[int(row["two_edit_metadata"]["model_id"])] for row in first]).float()
            first_ids = torch.tensor([intervention_id(row, nodes) for row in first])
            second_ids = torch.tensor([intervention_id(row, nodes) for row in second])
            raw = head(operator(operator(states, first_ids), second_ids)).argmax(-1)
            for index, row in enumerate(first):
                predictions.append([int(raw[index, nodes.index(node)]) for node in row["graph"]["nodes"]])
    scored = score_predictions(validation, predictions)
    if abs(scored["two_edit_balanced"] - float(summary["clean"]["two_edit_balanced"])) > 1e-12:
        raise ValueError(f"frozen rescore mismatch for {cell['cell_id']}")
    return scored, {
        "cell_id": cell["cell_id"],
        "checkpoint_path": str(checkpoint_path.relative_to(root)),
        "checkpoint_sha256": summary["checkpoint_sha256"],
        "metric_source_path": str(summary_path.relative_to(root)),
        "metric_source_sha256": sha256(summary_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--prefix", required=True, choices=("qwen7b", "phi14b", "qwen32", "llama70b"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); root = args.root.resolve()
    manifest_path = root / "registry" / f"{args.prefix}_cladder_extension_manifest.json"
    amendment_path = root / "registry" / f"{args.prefix}_cladder_cache_amendment.json"
    manifest, amendment = load_json(manifest_path), load_json(amendment_path)
    if amendment["base_manifest_sha256"] != sha256(manifest_path):
        raise ValueError("manifest/amendment binding mismatch")
    cache_path = Path(amendment["feature_cache"])
    if not cache_path.is_absolute(): cache_path = root / cache_path
    if sha256(cache_path) != amendment["feature_cache_sha256"]:
        raise ValueError("feature-cache hash mismatch")
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    artifact_path = Path(manifest["artifact"]); pairs_path = Path(manifest["pair_manifest"])
    if not artifact_path.is_absolute(): artifact_path = root / artifact_path
    if not pairs_path.is_absolute(): pairs_path = root / pairs_path
    if sha256(artifact_path) != manifest["artifact_sha256"] or sha256(pairs_path) != manifest["pair_manifest_sha256"]:
        raise ValueError("artifact hash mismatch")
    artifact = load_json(artifact_path); records = {row["id"]: row for row in artifact["components"]}
    pairs = [json.loads(line) for line in pairs_path.read_text().splitlines() if line.strip()]
    validation = [row for row in pairs if row["split"] == "validation"]
    feature_by_model = {int(row["model_id"]): cache["features"][index] for index, row in enumerate(cache["records"])}
    by_method_seed, provenance = defaultdict(dict), []
    for cell in sorted(manifest["cells"], key=lambda row: (row["method"], int(row["seed"]))):
        scored, source = predict_cell(root, manifest, amendment, cell, validation, records, feature_by_model)
        by_method_seed[cell["method"]][int(cell["seed"])] = scored
        provenance.append(source)
    seeds = sorted(set(by_method_seed["o2"]) & set(by_method_seed["o3"]))
    if len(seeds) != 20:
        raise ValueError(f"expected 20 paired seeds, found {len(seeds)}")
    seed_rows, graph_seed_deltas = [], defaultdict(list)
    for seed in seeds:
        o2, o3 = by_method_seed["o2"][seed], by_method_seed["o3"][seed]
        seed_rows.append({
            "seed": seed,
            "o2": o2["two_edit_balanced"],
            "o3": o3["two_edit_balanced"],
            "o2_minus_o3": o2["two_edit_balanced"] - o3["two_edit_balanced"],
        })
        for graph in sorted(set(o2["graph_scores"]) & set(o3["graph_scores"])):
            graph_seed_deltas[graph].append(o2["graph_scores"][graph] - o3["graph_scores"][graph])
    graph_deltas = {graph: statistics.fmean(values) for graph, values in graph_seed_deltas.items()}
    deltas = [row["o2_minus_o3"] for row in seed_rows]
    output = {
        "protocol": "reviewer1_f2_frozen_checkpoint_reanalysis_v1",
        "backbone": manifest["model"],
        "model_revision": manifest["model_revision"],
        "evaluation_split": "validation",
        "test_evaluated": False,
        "paired_seed_count": len(seeds),
        "graph_count": len(graph_deltas),
        "mean_o2_minus_o3": statistics.fmean(deltas),
        "paired_seed_student_t_95": t_interval(deltas),
        "equal_graph_mean_o2_minus_o3": statistics.fmean(graph_deltas.values()),
        "paired_graph_bootstrap_95": graph_bootstrap(graph_deltas),
        "per_seed": seed_rows,
        "per_graph_mean_delta": graph_deltas,
        "provenance": provenance,
        "manifest_path": str(manifest_path.relative_to(root)),
        "manifest_sha256": sha256(manifest_path),
        "amendment_path": str(amendment_path.relative_to(root)),
        "amendment_sha256": sha256(amendment_path),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: output[key] for key in ("backbone", "mean_o2_minus_o3", "paired_seed_student_t_95", "equal_graph_mean_o2_minus_o3", "paired_graph_bootstrap_95")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
