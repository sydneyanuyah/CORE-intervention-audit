#!/usr/bin/env python3
"""Aggregate the completed L2 dropped-law validation cells without reopening test."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path


T_CRITICAL_DF19 = 2.093024054
BOOTSTRAP_DRAWS = 10_000
LEARNED = ("full", "drop_identity", "drop_idempotence", "drop_commutation", "drop_selective_invariance")
FLOORS = ("do_nothing", "do_everything", "random_init")
METRICS = ("balanced_intervention_score", "change_accuracy", "preservation_accuracy", "variable_accuracy")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def student_t_95(values: list[float]) -> list[float]:
    if len(values) != 20:
        raise ValueError("L2 paired inference requires exactly twenty seeds")
    center = statistics.fmean(values)
    if statistics.stdev(values) == 0:
        return [center, center]
    margin = T_CRITICAL_DF19 * statistics.stdev(values) / math.sqrt(len(values))
    return [center - margin, center + margin]


def paired_seed_bootstrap_95(values: list[float], seed: int) -> list[float]:
    if len(values) != 20:
        raise ValueError("L2 paired bootstrap requires exactly twenty seeds")
    rng = random.Random(seed)
    draws = sorted(statistics.fmean(rng.choices(values, k=20)) for _ in range(BOOTSTRAP_DRAWS))
    return [draws[int(0.025 * BOOTSTRAP_DRAWS)], draws[int(0.975 * BOOTSTRAP_DRAWS) - 1]]


def selected_validation(summary: dict) -> dict:
    selected = [row for row in summary.get("history", []) if row.get("selected") is True]
    if not selected:
        raise ValueError("learned L2 summary has no selected validation epoch")
    return selected[-1]["validation"]


def extract_metrics(summary: dict, method: str) -> dict[str, float]:
    if summary.get("test_evaluated") is not False:
        raise ValueError("L2 summary is not validation-only")
    if method in FLOORS:
        if summary.get("protocol") != "l2_dropped_law_v1" or summary.get("world_size") != 4:
            raise ValueError("invalid L2 floor contract")
        validation = summary["validation"]
        return {name: float(validation[name]) for name in METRICS}
    if summary.get("gpu_policy") != {"model_size": "base", "world_size": 4}:
        raise ValueError("invalid L2 learned-cell GPU policy")
    if summary.get("l1", {}).get("law_profile") != method:
        raise ValueError("L2 law profile does not match registered method")
    validation = selected_validation(summary)
    per_variable = validation["per_variable"]
    change = float(per_variable["groups"]["change"]["accuracy"])
    preservation = float(per_variable["groups"]["preservation"]["accuracy"])
    return {
        "balanced_intervention_score": (change + preservation) / 2.0,
        "change_accuracy": change,
        "preservation_accuracy": preservation,
        "variable_accuracy": float(per_variable["variable_accuracy"]),
    }


def summarize(values: list[float]) -> dict:
    return {"n": len(values), "mean": statistics.fmean(values), "student_t_95": student_t_95(values), "values": values}


def comparison(values: list[float], seed: int) -> dict:
    interval = student_t_95(values)
    return {
        "n_paired_seeds": 20,
        "mean_delta": statistics.fmean(values),
        "student_t_95": interval,
        "paired_seed_bootstrap_95": paired_seed_bootstrap_95(values, seed),
        "interval_excludes_zero": interval[0] > 0 or interval[1] < 0,
        "paired_seed_deltas": values,
    }


def aggregate(root: Path, manifest_path: Path, evidence_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    evidence = json.loads(evidence_path.read_text())
    if manifest.get("protocol") != "l2_dropped_law_v1" or manifest.get("test_evaluated") is not False:
        raise ValueError("invalid L2 manifest")
    if evidence.get("complete") != 320 or evidence.get("remaining") != 0 or evidence.get("test_evaluated") is not False:
        raise ValueError("L2 evidence inventory is incomplete")
    seeds = [int(seed) for seed in manifest["seeds"]]
    families = list(manifest["families"])
    methods = list(manifest["methods"]) + list(manifest["floors"])
    if seeds != list(range(201, 221)) or methods != list(LEARNED + FLOORS):
        raise ValueError("unexpected L2 preregistration")
    expected_hashes = {row["cell_id"]: row["summary_sha256"] for row in evidence["cells"]}
    if len(expected_hashes) != 320:
        raise ValueError("L2 completion evidence does not contain 320 unique cell identities")

    cells: dict[str, dict[str, dict[int, dict[str, float]]]] = {}
    source_summaries = []
    for family in families:
        cells[family] = {}
        for method in methods:
            cells[family][method] = {}
            for seed in seeds:
                path = root / family / method / f"seed-{seed}" / "run_summary.json"
                summary = json.loads(path.read_text())
                expected_id = f"l2:{family}:{method}:{seed}"
                digest = sha256(path)
                if expected_hashes.get(expected_id) != digest:
                    raise ValueError(f"summary hash mismatch: {expected_id}")
                if method in FLOORS and summary.get("cell_id") != expected_id:
                    raise ValueError(f"floor cell identity mismatch: {expected_id}")
                metrics = extract_metrics(summary, method)
                cells[family][method][seed] = metrics
                source_summaries.append({"cell_id": expected_id, "sha256": digest})
    if len(source_summaries) != 320:
        raise ValueError("L2 aggregate did not consume exactly 320 summaries")

    output = {
        "protocol": "l2_dropped_law_v1",
        "manifest_sha256": sha256(manifest_path),
        "completion_evidence_sha256": sha256(evidence_path),
        "completed_cells": 320,
        "families": {},
        "primary_metric": "validation balanced_intervention_score",
        "delta_sign": "full minus dropped; positive means the full four-law objective scored higher",
        "inference_unit": "paired seed",
        "graph_level_inference": {
            "available": False,
            "reason": "Frozen L2 summaries retain selected-checkpoint aggregate validation metrics but no per-graph predictions; graph-level intervals cannot be reconstructed without re-evaluation.",
        },
        "source_summaries": source_summaries,
        "test_evaluated": False,
    }
    comparison_index = 0
    for family in families:
        aggregates = {}
        for method in methods:
            aggregates[method] = {
                metric: summarize([cells[family][method][seed][metric] for seed in seeds])
                for metric in METRICS
            }
        dropped = {}
        for method in LEARNED[1:]:
            dropped[method] = {}
            for metric in METRICS:
                deltas = [cells[family]["full"][seed][metric] - cells[family][method][seed][metric] for seed in seeds]
                dropped[method][metric] = comparison(deltas, 20260913 + comparison_index)
                comparison_index += 1
        floors = {}
        for method in FLOORS:
            deltas = [cells[family]["full"][seed]["balanced_intervention_score"] - cells[family][method][seed]["balanced_intervention_score"] for seed in seeds]
            floors[method] = comparison(deltas, 20261013 + comparison_index)
            comparison_index += 1
        output["families"][family] = {"aggregates": aggregates, "full_minus_dropped": dropped, "full_minus_floors_balanced": floors}
    return output


def render(report: dict) -> str:
    lines = [
        "# L2 dropped-law ablation",
        "",
        "All 320 registered validation-only cells completed: two families, twenty paired seeds, five learned law profiles, and three deterministic floors. No held-out test was accessed.",
        "",
        "The primary estimate is `full − dropped` validation balanced intervention score. Positive values would mean that retaining the named law improved the selected-checkpoint score. Intervals are paired across the twenty registered seeds.",
        "",
    ]
    for family, family_row in report["families"].items():
        lines.extend([f"## {family}", "", "| Dropped objective | Full mean | Dropped mean | Full − dropped | Student-t 95% | Paired-seed bootstrap 95% |", "|---|---:|---:|---:|---:|---:|"])
        full_mean = family_row["aggregates"]["full"]["balanced_intervention_score"]["mean"]
        for method, metrics in family_row["full_minus_dropped"].items():
            row = metrics["balanced_intervention_score"]
            dropped_mean = family_row["aggregates"][method]["balanced_intervention_score"]["mean"]
            t_ci, b_ci = row["student_t_95"], row["paired_seed_bootstrap_95"]
            lines.append(f"| {method.removeprefix('drop_')} | {full_mean:.6f} | {dropped_mean:.6f} | {row['mean_delta']:+.6f} | [{t_ci[0]:+.6f}, {t_ci[1]:+.6f}] | [{b_ci[0]:+.6f}, {b_ci[1]:+.6f}] |")
        lines.extend(["", "### Full model versus deterministic floors", "", "| Floor | Full − floor | Student-t 95% |", "|---|---:|---:|"])
        for method, row in family_row["full_minus_floors_balanced"].items():
            t_ci = row["student_t_95"]
            lines.append(f"| {method} | {row['mean_delta']:+.6f} | [{t_ci[0]:+.6f}, {t_ci[1]:+.6f}] |")
        lines.append("")
    primary = [
        row["balanced_intervention_score"]
        for family in report["families"].values()
        for row in family["full_minus_dropped"].values()
    ]
    significant = sum(row["interval_excludes_zero"] for row in primary)
    lines.extend([
        "## Interpretation",
        "",
        f"Across the eight family-by-dropped-law primary comparisons, {significant} paired-seed Student-t intervals excluded zero. "
        + ("L2 therefore identifies at least one validation effect of removing a law objective; the direction and family are shown above." if significant else "L2 supplies no evidence that retaining any single registered law objective improves the selected-checkpoint balanced score."),
        "",
        "Graph-level inference is unavailable because the frozen summaries do not retain per-graph predictions. The report fails closed at paired-seed inference instead of presenting a graph bootstrap that cannot be reconstructed.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parents[1]
    parser.add_argument("--summaries-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=repo / "registry/l2_manifest.json")
    parser.add_argument("--evidence", type=Path, default=repo / "reports/evidence/l2_cells_complete.json")
    parser.add_argument("--json-output", type=Path, default=repo / "reports/L2_RESULTS.json")
    parser.add_argument("--markdown-output", type=Path, default=repo / "reports/L2_RESULTS.md")
    args = parser.parse_args()
    report = aggregate(args.summaries_root, args.manifest, args.evidence)
    args.json_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.markdown_output.write_text(render(report))
    print(json.dumps({"completed_cells": report["completed_cells"], "families": list(report["families"]), "test_evaluated": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
