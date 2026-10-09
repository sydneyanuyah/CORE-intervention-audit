#!/usr/bin/env python3
"""Aggregate the complete five-family A1 architecture screen."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any


T_CRITICAL_95_DF4 = 2.7764451051977987


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
    if len(values) != 5:
        raise ValueError("A1 screening interval requires five seeds")
    center = _mean(values)
    variance = sum((value - center) ** 2 for value in values) / 4
    half_width = T_CRITICAL_95_DF4 * math.sqrt(variance / 5)
    return [center - half_width, center + half_width]


def aggregate(repo_root: Path, manifest_path: Path) -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location(
        "a1_screening_runner", repo_root / "scripts" / "run_a1_screening.py"
    )
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)
    manifest = _load(manifest_path)
    cells = runner.validate_manifest(manifest, repo_root)
    seeds = sorted(int(seed) for seed in manifest["seeds"])
    registry = _load(repo_root / "registry" / "bert_scope.json")
    a1 = next(row for row in registry["experiments"] if row["id"] == "A1")
    expected_cells = int(a1["stages"]["screening"]["applicable_runs_per_reader"])
    if len(cells) != expected_cells:
        raise RuntimeError(f"expected {expected_cells} registered cells, found {len(cells)}")

    grouped: dict[str, dict[str, dict[int, dict[str, Any]]]] = {}
    checkpoints: list[str] = []
    for cell in cells:
        path = cell.output / "cell_summary.json"
        if not path.exists():
            raise RuntimeError(f"incomplete screening cell: {cell.cell_id}")
        summary = _load(path)
        if summary.get("cell_id") != cell.cell_id:
            raise RuntimeError(f"cell identity mismatch: {cell.cell_id}")
        if summary.get("provenance", {}).get("test_evaluated") is not False:
            raise RuntimeError(f"test isolation violated: {cell.cell_id}")
        if summary.get("provenance", {}).get("world_size") != 4:
            raise RuntimeError(f"BERT-base cell did not use exactly four GPUs: {cell.cell_id}")
        grouped.setdefault(cell.family, {}).setdefault(cell.arm, {})[cell.seed] = summary
        checkpoints.append(summary["checkpoint_sha256"])

    families: dict[str, Any] = {}
    for family in manifest["families"]:
        family_arms: dict[str, Any] = {}
        for arm, runs in grouped[family].items():
            if sorted(runs) != seeds:
                raise RuntimeError(f"{family}/{arm} does not have the registered five seeds")
            summaries = [runs[seed] for seed in seeds]
            balanced = [row["composed_validation"]["two_edit"]["two_edit_balanced"] for row in summaries]
            change = [row["composed_validation"]["two_edit"]["two_edit_change_accuracy"] for row in summaries]
            preservation = [row["composed_validation"]["two_edit"]["two_edit_preservation"] for row in summaries]
            target = [row["composed_validation"]["two_edit"]["target_success"] for row in summaries]
            pointer = [row["composed_validation"]["pointer"]["pointer_top1_accuracy"] for row in summaries]
            family_arms[arm] = {
                "seed_scores": balanced,
                "mean_two_edit_balanced": _mean(balanced),
                "student_t_95_two_edit_balanced": _student_t_95(balanced),
                "mean_change": _mean(change),
                "mean_preservation": _mean(preservation),
                "mean_target_success": _mean(target),
                "mean_pointer_top1": None if any(value is None for value in pointer) else _mean(pointer),
                "checkpoint_sha256": [row["checkpoint_sha256"] for row in summaries],
            }
        families[family] = {"arms": family_arms}

    xor_arms = families["xor"]["arms"]
    t0_scores = xor_arms["t0_id"]["seed_scores"]
    eligible: list[str] = []
    for arm, row in xor_arms.items():
        deltas = [score - baseline for score, baseline in zip(row["seed_scores"], t0_scores)]
        row["paired_delta_vs_t0"] = _mean(deltas)
        row["student_t_95_delta_vs_t0"] = _student_t_95(deltas)
        row["within_two_point_rule"] = row["paired_delta_vs_t0"] >= -0.02
        if row["within_two_point_rule"]:
            eligible.append(arm)
    selected = max(eligible, key=lambda arm: xor_arms[arm]["mean_two_edit_balanced"])

    for family, family_row in families.items():
        ranked = sorted(
            family_row["arms"],
            key=lambda arm: family_row["arms"][arm]["mean_two_edit_balanced"],
            reverse=True,
        )
        family_row["ranked_arms"] = ranked
        family_row["selected_arm_rank"] = ranked.index(selected) + 1 if selected in ranked else None

    return {
        "protocol": "a1_architecture_screening_report_v1",
        "scope": "all_five_families",
        "stage": "architecture_screening_only",
        "confirmatory": False,
        "test_evaluated": False,
        "registered_cells": expected_cells,
        "completed_cells": len(cells),
        "seeds": seeds,
        "selection_rule": "highest XOR mean two_edit_balanced among arms no more than 0.02 below synthetic T0; T0 remains undefined on real families",
        "selected_arm": selected,
        "families": families,
        "provenance": {
            "manifest_sha256": _sha256(manifest_path),
            "checkpoint_count": len(checkpoints),
            "unique_checkpoint_count": len(set(checkpoints)),
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# A1 architecture screening — all families",
        "",
        "All 130 registered BERT-base architecture-screening cells are complete. Every cell used exactly four GPUs, remained validation-only, and accessed no held-out test records. These five-seed results select architecture; they are not confirmatory evidence.",
        "",
        "| Family | Arm | Mean balanced | 95% t interval | Change | Preservation | Target | Pointer top-1 | Rank |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for family, family_row in report["families"].items():
        rank = {arm: index + 1 for index, arm in enumerate(family_row["ranked_arms"])}
        for arm, row in family_row["arms"].items():
            interval = row["student_t_95_two_edit_balanced"]
            pointer = "n/a" if row["mean_pointer_top1"] is None else f'{row["mean_pointer_top1"]:.4f}'
            lines.append(
                f'| {family} | {arm} | {row["mean_two_edit_balanced"]:.4f} '
                f'| [{interval[0]:.4f}, {interval[1]:.4f}] | {row["mean_change"]:.4f} '
                f'| {row["mean_preservation"]:.4f} | {row["mean_target_success"]:.4f} '
                f'| {pointer} | {rank[arm]} |'
            )
    selected = report["selected_arm"]
    xor = report["families"]["xor"]["arms"][selected]
    lines.extend([
        "",
        f'Frozen architecture selection: **{selected}**. On XOR it scores {xor["mean_two_edit_balanced"]:.4f}, '
        f'{xor["paired_delta_vs_t0"]:+.4f} versus T0, and therefore passes the preregistered two-point rule.',
        "",
        "T0 is a synthetic learned-ID reference and is deliberately undefined for the four real families. Real-family results are reported as architecture-screening diagnostics and are not compared to a fabricated T0.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--manifest", type=Path, default=root / "registry" / "a1_screening_manifest.json")
    parser.add_argument("--json-output", type=Path, default=root / "reports" / "A1_SCREENING.json")
    parser.add_argument("--markdown-output", type=Path, default=root / "reports" / "A1_SCREENING.md")
    args = parser.parse_args()
    report = aggregate(root, args.manifest.resolve())
    args.json_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown_output.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"completed_cells": report["completed_cells"], "selected_arm": report["selected_arm"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
