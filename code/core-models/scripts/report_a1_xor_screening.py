#!/usr/bin/env python3
"""Aggregate the completed five-seed XOR A1 architecture screen."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import sys
from pathlib import Path
from typing import Any


T_CRITICAL_95_DF4 = 2.7764451051977987
METRIC = "two_edit_balanced"


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return sum(values) / len(values)


def student_t_95(values: list[float]) -> list[float]:
    if len(values) != 5:
        raise ValueError("A1 screening Student-t interval requires five seeds")
    center = mean(values)
    variance = sum((value - center) ** 2 for value in values) / 4
    half_width = T_CRITICAL_95_DF4 * math.sqrt(variance / 5)
    return [center - half_width, center + half_width]


def percentile(sorted_values: list[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def paired_graph_bootstrap_95(
    per_graph_deltas: dict[str, list[float]], *, replicates: int = 10_000, seed: int = 20260905
) -> list[float]:
    graph_ids = sorted(per_graph_deltas)
    if len(graph_ids) != 20 or any(len(per_graph_deltas[g]) != 5 for g in graph_ids):
        raise ValueError("paired graph bootstrap requires 20 graphs and five paired seeds")
    graph_values = [mean(per_graph_deltas[g]) for g in graph_ids]
    rng = random.Random(seed)
    draws = sorted(
        mean([graph_values[rng.randrange(20)] for _ in range(20)])
        for _ in range(replicates)
    )
    return [percentile(draws, 0.025), percentile(draws, 0.975)]


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aggregate(repo_root: Path, manifest_path: Path) -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location(
        "a1_screening_runner", repo_root / "scripts" / "run_a1_screening.py"
    )
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)
    manifest = _load(manifest_path)
    cells = [cell for cell in runner.validate_manifest(manifest, repo_root) if cell.family == "xor"]
    by_arm: dict[str, dict[int, tuple[dict[str, Any], dict[str, Any]]]] = {}
    for cell in cells:
        summary_path = cell.output / "cell_summary.json"
        composed_path = cell.output / "composed_validation.json"
        if not summary_path.exists() or not composed_path.exists():
            raise RuntimeError(f"incomplete XOR screening cell: {cell.cell_id}")
        summary = _load(summary_path)
        composed = _load(composed_path)
        if summary["provenance"]["test_evaluated"] is not False or composed["test_evaluated"] is not False:
            raise RuntimeError(f"test isolation violated by {cell.cell_id}")
        by_arm.setdefault(cell.arm, {})[cell.seed] = (summary, composed)

    seeds = sorted(int(seed) for seed in manifest["seeds"])
    baseline = by_arm["t0_id"]
    arms: dict[str, Any] = {}
    for arm in manifest["arms"]:
        runs = by_arm[arm]
        if sorted(runs) != seeds:
            raise RuntimeError(f"{arm} does not have the registered five seeds")
        scores = [runs[seed][0]["composed_validation"]["two_edit"][METRIC] for seed in seeds]
        baseline_scores = [baseline[seed][0]["composed_validation"]["two_edit"][METRIC] for seed in seeds]
        deltas = [score - base for score, base in zip(scores, baseline_scores)]
        per_graph: dict[str, list[float]] = {}
        for seed in seeds:
            arm_graphs = runs[seed][1]["two_edit"]["per_graph"]
            base_graphs = baseline[seed][1]["two_edit"]["per_graph"]
            if set(arm_graphs) != set(base_graphs):
                raise RuntimeError(f"graph pairing mismatch for {arm}, seed {seed}")
            for graph_id in arm_graphs:
                per_graph.setdefault(graph_id, []).append(
                    arm_graphs[graph_id][METRIC] - base_graphs[graph_id][METRIC]
                )
        summaries = [runs[seed][0] for seed in seeds]
        target = mean([row["composed_validation"]["two_edit"]["target_success"] for row in summaries])
        pointer_values = [row["composed_validation"]["pointer"]["pointer_top1_accuracy"] for row in summaries]
        arms[arm] = {
            "seed_scores": scores,
            "mean_two_edit_balanced": mean(scores),
            "mean_target_success": target,
            "mean_pointer_top1": None if any(v is None for v in pointer_values) else mean(pointer_values),
            "paired_delta_vs_t0": mean(deltas),
            "student_t_95_delta_vs_t0": student_t_95(deltas),
            "paired_graph_bootstrap_95_delta_vs_t0": paired_graph_bootstrap_95(per_graph),
            "within_two_point_rule": mean(deltas) >= -0.02,
            "checkpoint_sha256": [row["checkpoint_sha256"] for row in summaries],
        }
    selected = max(
        (arm for arm in arms if arms[arm]["within_two_point_rule"]),
        key=lambda arm: arms[arm]["mean_two_edit_balanced"],
    )
    return {
        "protocol": "a1_xor_architecture_screening_report_v1",
        "scope": "xor_architecture_screening_only",
        "a1_family_complete": False,
        "confirmatory": False,
        "test_evaluated": False,
        "registered_cells": 30,
        "completed_cells": 30,
        "seeds": seeds,
        "graphs_per_seed": 20,
        "selection_rule": "highest mean two_edit_balanced among arms no more than 0.02 below T0",
        "selected_arm": selected,
        "arms": arms,
        "provenance": {"manifest_sha256": _sha256(manifest_path)},
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# A1 XOR architecture screening",
        "",
        "This report covers the complete 30-cell, five-seed XOR architecture screen. It is not the full A1 family or confirmatory evidence. No test split was accessed.",
        "",
        "| Arm | Mean balanced | Delta vs T0 | Student-t 95% CI | Paired-graph bootstrap 95% CI | Target | Pointer top-1 | 2-point rule |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for arm, row in report["arms"].items():
        t_ci = row["student_t_95_delta_vs_t0"]
        b_ci = row["paired_graph_bootstrap_95_delta_vs_t0"]
        pointer = "n/a" if row["mean_pointer_top1"] is None else f'{row["mean_pointer_top1"]:.4f}'
        lines.append(
            f'| {arm} | {row["mean_two_edit_balanced"]:.4f} | {row["paired_delta_vs_t0"]:+.4f} '
            f'| [{t_ci[0]:+.4f}, {t_ci[1]:+.4f}] | [{b_ci[0]:+.4f}, {b_ci[1]:+.4f}] '
            f'| {row["mean_target_success"]:.4f} | {pointer} | {"pass" if row["within_two_point_rule"] else "fail"} |'
        )
    lines.extend([
        "",
        f'Architecture selected on XOR: **{report["selected_arm"]}**.',
        "",
        "Student-t intervals use the five paired seed deltas (df=4). The deterministic paired-graph bootstrap resamples the 20 graph units and averages paired arm-minus-T0 differences across the same five seeds (10,000 replicates).",
        "",
        "The other 100 registered real-family screening cells remain blocked because the normalized sources do not yet provide authoritative executable two-edit truth. Single-edit outcomes are not combined to fabricate that truth, so confirmatory A1 has not started.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path(__file__).parents[1] / "registry" / "a1_screening_manifest.json")
    parser.add_argument("--json-output", type=Path, default=Path(__file__).parents[1] / "reports" / "A1_XOR_SCREENING.json")
    parser.add_argument("--markdown-output", type=Path, default=Path(__file__).parents[1] / "reports" / "A1_XOR_SCREENING.md")
    args = parser.parse_args()
    repo_root = args.manifest.resolve().parents[1]
    report = aggregate(repo_root, args.manifest.resolve())
    args.json_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown_output.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"selected_arm": report["selected_arm"], "completed_cells": 30}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
