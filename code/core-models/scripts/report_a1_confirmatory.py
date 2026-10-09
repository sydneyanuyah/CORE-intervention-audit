#!/usr/bin/env python3
"""Aggregate the complete preregistered 20-seed A1 confirmatory matrix."""

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


T_CRITICAL_95 = {19: 2.093024054408263}
BOOTSTRAP_DRAWS = 10_000


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return sum(values) / len(values)


def _student_t_95(values: list[float]) -> list[float]:
    if len(values) != 20:
        raise ValueError("A1 confirmatory interval requires twenty seeds")
    center = _mean(values)
    variance = sum((value - center) ** 2 for value in values) / 19
    half_width = T_CRITICAL_95[19] * math.sqrt(variance / 20)
    return [center - half_width, center + half_width]


def _percentile(sorted_values: list[float], probability: float) -> float:
    position = probability * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _paired_graph_bootstrap_95(
    arm_by_seed_graph: dict[int, dict[str, float]],
    baseline_by_seed_graph: dict[int, dict[str, float]],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    bootstrap_seed: int = 20260906,
) -> list[float]:
    seeds = sorted(arm_by_seed_graph)
    if seeds != sorted(baseline_by_seed_graph) or len(seeds) != 20:
        raise ValueError("paired graph bootstrap requires the same twenty seeds")
    graph_ids = sorted(arm_by_seed_graph[seeds[0]])
    if len(graph_ids) != 20:
        raise ValueError("paired graph bootstrap requires twenty XOR graphs")
    for seed in seeds:
        if sorted(arm_by_seed_graph[seed]) != graph_ids:
            raise ValueError("arm graph identities differ across seeds")
        if sorted(baseline_by_seed_graph[seed]) != graph_ids:
            raise ValueError("baseline graph identities differ across seeds")
    graph_deltas = [
        _mean([
            arm_by_seed_graph[seed][graph_id]
            - baseline_by_seed_graph[seed][graph_id]
            for seed in seeds
        ])
        for graph_id in graph_ids
    ]
    rng = random.Random(bootstrap_seed)
    samples = sorted(
        _mean([graph_deltas[rng.randrange(len(graph_deltas))] for _ in graph_deltas])
        for _ in range(draws)
    )
    return [_percentile(samples, 0.025), _percentile(samples, 0.975)]


def aggregate(repo_root: Path, manifest_path: Path) -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location(
        "a1_confirmatory_runner", repo_root / "scripts" / "run_a1_screening.py"
    )
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)
    manifest = _load(manifest_path)
    cells = runner.validate_manifest(manifest, repo_root)
    if manifest.get("stage") != "confirmatory":
        raise RuntimeError("confirmatory reporter requires a confirmatory manifest")
    seeds = sorted(int(seed) for seed in manifest["seeds"])
    if len(seeds) != 20:
        raise RuntimeError("confirmatory report requires exactly twenty seeds")
    registry = _load(repo_root / "registry" / "bert_scope.json")
    a1 = next(row for row in registry["experiments"] if row["id"] == "A1")
    expected_cells = int(a1["stages"]["confirmatory"]["applicable_runs_per_reader"])
    if len(cells) != expected_cells:
        raise RuntimeError(f"expected {expected_cells} registered cells, found {len(cells)}")

    grouped: dict[str, dict[str, dict[int, dict[str, Any]]]] = {}
    checkpoints: list[str] = []
    for cell in cells:
        path = cell.output / "cell_summary.json"
        if not path.exists():
            raise RuntimeError(f"incomplete confirmatory cell: {cell.cell_id}")
        summary = _load(path)
        if summary.get("cell_id") != cell.cell_id:
            raise RuntimeError(f"cell identity mismatch: {cell.cell_id}")
        if summary.get("stage") != "confirmatory" or summary.get("confirmatory") is not True:
            raise RuntimeError(f"non-confirmatory summary: {cell.cell_id}")
        provenance = summary.get("provenance", {})
        if provenance.get("test_evaluated") is not False:
            raise RuntimeError(f"test isolation violated: {cell.cell_id}")
        if provenance.get("world_size") != 4:
            raise RuntimeError(f"BERT-base cell did not use exactly four GPUs: {cell.cell_id}")
        if not provenance.get("cell_contract_sha256"):
            raise RuntimeError(f"cell contract is missing: {cell.cell_id}")
        grouped.setdefault(cell.family, {}).setdefault(cell.arm, {})[cell.seed] = summary
        checkpoints.append(summary["checkpoint_sha256"])

    if len(set(checkpoints)) != expected_cells:
        raise RuntimeError("confirmatory cells do not have unique checkpoints")

    families: dict[str, Any] = {}
    graph_scores: dict[str, dict[int, dict[str, float]]] = {}
    for family in manifest["families"]:
        family_arms: dict[str, Any] = {}
        for arm, runs in grouped[family].items():
            if sorted(runs) != seeds:
                raise RuntimeError(f"{family}/{arm} does not have the registered twenty seeds")
            summaries = [runs[seed] for seed in seeds]
            balanced = [row["composed_validation"]["two_edit"]["two_edit_balanced"] for row in summaries]
            change = [row["composed_validation"]["two_edit"]["two_edit_change_accuracy"] for row in summaries]
            preservation = [row["composed_validation"]["two_edit"]["two_edit_preservation"] for row in summaries]
            target = [row["composed_validation"]["two_edit"]["target_success"] for row in summaries]
            unseen_rows = [row["single_edit_validation"]["unseen_paraphrase"] for row in summaries]
            unseen_available = [row.get("available") is True for row in unseen_rows]
            if any(unseen_available) and not all(unseen_available):
                raise RuntimeError(f"mixed unseen-paraphrase availability: {family}/{arm}")
            unseen = [row["accuracy"] for row in unseen_rows] if all(unseen_available) else []
            pointer = [row["composed_validation"]["pointer"]["pointer_top1_accuracy"] for row in summaries]
            family_arms[arm] = {
                "seed_scores": balanced,
                "mean_two_edit_balanced": _mean(balanced),
                "student_t_95_two_edit_balanced": _student_t_95(balanced),
                "mean_change": _mean(change),
                "mean_preservation": _mean(preservation),
                "mean_target_success": _mean(target),
                "mean_unseen_paraphrase_accuracy": _mean(unseen) if unseen else None,
                "mean_pointer_top1": None if any(value is None for value in pointer) else _mean(pointer),
                "checkpoint_sha256": [row["checkpoint_sha256"] for row in summaries],
            }
            if family == "xor":
                graph_scores[arm] = {
                    seed: {
                        graph_id: metrics["two_edit_balanced"]
                        for graph_id, metrics in runs[seed]["composed_validation"]["two_edit"]["per_graph"].items()
                    }
                    for seed in seeds
                }
        ranked = sorted(
            family_arms,
            key=lambda arm: family_arms[arm]["mean_two_edit_balanced"],
            reverse=True,
        )
        families[family] = {"arms": family_arms, "ranked_arms": ranked}

    t0 = families["xor"]["arms"]["t0_id"]
    for arm, row in families["xor"]["arms"].items():
        deltas = [score - baseline for score, baseline in zip(row["seed_scores"], t0["seed_scores"])]
        row["paired_delta_vs_t0"] = _mean(deltas)
        row["student_t_95_delta_vs_t0"] = _student_t_95(deltas)
        row["paired_graph_bootstrap_95_delta_vs_t0"] = _paired_graph_bootstrap_95(
            graph_scores[arm], graph_scores["t0_id"]
        )
        row["within_two_point_rule"] = row["paired_delta_vs_t0"] >= -0.02

    selected = manifest["frozen_selection"]["selected_arm"]
    if selected not in families["xor"]["arms"]:
        raise RuntimeError("frozen confirmatory architecture identity is invalid")
    selected_xor = families["xor"]["arms"][selected]
    return {
        "protocol": "a1_confirmatory_report_v1",
        "scope": "all_five_families",
        "stage": "preregistered_confirmatory_evidence",
        "confirmatory": True,
        "test_evaluated": False,
        "registered_cells": expected_cells,
        "completed_cells": len(cells),
        "seeds": seeds,
        "frozen_selected_arm": selected,
        "primary_rule": "paired XOR two_edit_balanced versus T0; pass when mean delta is at least -0.02",
        "primary_pass": selected_xor["within_two_point_rule"],
        "families": families,
        "provenance": {
            "manifest_sha256": _sha256(manifest_path),
            "frozen_screening_report_sha256": manifest["frozen_selection"]["sha256"],
            "checkpoint_count": len(checkpoints),
            "unique_checkpoint_count": len(set(checkpoints)),
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "experimental_unit": "graph",
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# A1 preregistered confirmatory evidence",
        "",
        "All 520 registered BERT-base confirmatory cells are complete. Every cell used exactly four GPUs, remained validation-only, and reports `test_evaluated=false`. The architecture was frozen before this matrix; these results do not reselect it.",
        "",
        "| Family | Arm | Mean balanced | 95% t interval | Change | Preservation | Target | Unseen paraphrase | Pointer top-1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for family, family_row in report["families"].items():
        for arm, row in family_row["arms"].items():
            interval = row["student_t_95_two_edit_balanced"]
            pointer = "n/a" if row["mean_pointer_top1"] is None else f'{row["mean_pointer_top1"]:.4f}'
            unseen = (
                "n/a" if row["mean_unseen_paraphrase_accuracy"] is None
                else f'{row["mean_unseen_paraphrase_accuracy"]:.4f}'
            )
            lines.append(
                f'| {family} | {arm} | {row["mean_two_edit_balanced"]:.4f} '
                f'| [{interval[0]:.4f}, {interval[1]:.4f}] | {row["mean_change"]:.4f} '
                f'| {row["mean_preservation"]:.4f} | {row["mean_target_success"]:.4f} '
                f'| {unseen} | {pointer} |'
            )
    selected = report["frozen_selected_arm"]
    xor = report["families"]["xor"]["arms"][selected]
    bootstrap = xor["paired_graph_bootstrap_95_delta_vs_t0"]
    t_interval = xor["student_t_95_delta_vs_t0"]
    lines.extend([
        "",
        f'Frozen architecture: **{selected}**. Its confirmatory XOR mean is {xor["mean_two_edit_balanced"]:.4f}; '
        f'the paired mean delta versus T0 is {xor["paired_delta_vs_t0"]:+.4f}.',
        "",
        f'95% intervals for that delta are [{t_interval[0]:+.4f}, {t_interval[1]:+.4f}] by Student t across seeds and '
        f'[{bootstrap[0]:+.4f}, {bootstrap[1]:+.4f}] by deterministic paired bootstrap over graphs. '
        f'The preregistered within-two-points result is **{"PASS" if report["primary_pass"] else "FAIL"}**.',
        "",
        "T0 is the synthetic learned-ID reference and remains undefined on real families. No real-family T0 value was fabricated, and no held-out test record was opened.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--manifest", type=Path, default=root / "registry" / "a1_confirmatory_manifest.json")
    parser.add_argument("--json-output", type=Path, default=root / "reports" / "A1_CONFIRMATORY.json")
    parser.add_argument("--markdown-output", type=Path, default=root / "reports" / "A1_CONFIRMATORY.md")
    args = parser.parse_args()
    report = aggregate(root, args.manifest.resolve())
    args.json_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown_output.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({
        "completed_cells": report["completed_cells"],
        "frozen_selected_arm": report["frozen_selected_arm"],
        "primary_pass": report["primary_pass"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
