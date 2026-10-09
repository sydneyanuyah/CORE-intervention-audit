#!/usr/bin/env python3
"""Aggregate the complete preregistered A2 frozen-checkpoint measurements."""

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
COMPARISONS = {
    "primary_masking": ("t2b_editor_zeroed", "no_instruction_baseline"),
    "secondary_editor_contribution": ("t2b_editor_active", "t2b_editor_zeroed"),
    "prompting_vs_no_instruction": ("prompting_baseline", "no_instruction_baseline"),
}


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return sum(values) / len(values)


def student_t_95(values: list[float]) -> list[float]:
    if len(values) != 20:
        raise ValueError("A2 Student-t interval requires twenty paired seeds")
    center = mean(values)
    variance = sum((value - center) ** 2 for value in values) / 19
    half_width = T_CRITICAL_95[19] * math.sqrt(variance / 20)
    return [center - half_width, center + half_width]


def percentile(values: list[float], probability: float) -> float:
    position = probability * (len(values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def paired_graph_bootstrap_95(
    arm: dict[int, dict[str, float]],
    baseline: dict[int, dict[str, float]],
    *,
    bootstrap_seed: int,
) -> list[float]:
    seeds = sorted(arm)
    if seeds != sorted(baseline) or len(seeds) != 20:
        raise ValueError("A2 graph bootstrap requires the same twenty seeds")
    graph_ids = sorted(arm[seeds[0]])
    if not graph_ids:
        raise ValueError("A2 graph bootstrap requires at least one graph")
    for seed in seeds:
        if sorted(arm[seed]) != graph_ids or sorted(baseline[seed]) != graph_ids:
            raise ValueError("A2 paired controls have different graph identities")
    graph_deltas = [
        mean([arm[seed][graph_id] - baseline[seed][graph_id] for seed in seeds])
        for graph_id in graph_ids
    ]
    rng = random.Random(bootstrap_seed)
    samples = sorted(
        mean([graph_deltas[rng.randrange(len(graph_deltas))] for _ in graph_deltas])
        for _ in range(BOOTSTRAP_DRAWS)
    )
    return [percentile(samples, 0.025), percentile(samples, 0.975)]


def aggregate(repo_root: Path, manifest_path: Path) -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location(
        "a2_measurement_runner", repo_root / "scripts" / "run_a2_measurement.py"
    )
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)
    manifest = load(manifest_path)
    cells = runner.validate(manifest, repo_root)
    if len(cells) != 400:
        raise RuntimeError("A2 reporter requires exactly 400 registered cells")

    grouped: dict[str, dict[str, dict[int, dict[str, Any]]]] = {}
    summary_hashes: list[str] = []
    checkpoints_by_source: dict[tuple[str, int], str] = {}
    for cell in cells:
        path = cell.output / "cell_summary.json"
        if not path.is_file():
            raise RuntimeError(f"incomplete A2 cell: {cell.cell_id}")
        summary = load(path)
        if (
            summary.get("protocol") != "a2_complete_measurement_cell_v1"
            or summary.get("cell_id") != cell.cell_id
            or summary.get("manifest_sha256") != sha256(manifest_path)
            or summary.get("checkpoint_sha256") != cell.checkpoint_sha256
            or summary.get("source_cell_contract_sha256") != cell.contract_sha256
            or summary.get("source_cell_summary_sha256") != cell.source_summary_sha256
            or summary.get("source_composed_validation_sha256") != cell.source_composed_sha256
            or summary.get("test_evaluated") is not False
            or summary.get("distributed", {}).get("world_size") != 4
        ):
            raise RuntimeError(f"inadmissible A2 summary: {cell.cell_id}")
        source_key = (cell.family, cell.seed)
        previous = checkpoints_by_source.setdefault(source_key, cell.checkpoint_sha256)
        if previous != cell.checkpoint_sha256:
            raise RuntimeError(f"A2 controls do not share a checkpoint: {source_key}")
        grouped.setdefault(cell.family, {}).setdefault(cell.control, {})[cell.seed] = summary
        summary_hashes.append(sha256(path))

    seeds = sorted(int(seed) for seed in manifest["seeds"])
    controls = list(manifest["controls"])
    families: dict[str, Any] = {}
    for family_index, family in enumerate(manifest["families"]):
        family_controls = grouped.get(family, {})
        if set(family_controls) != set(controls):
            raise RuntimeError(f"A2 control coverage mismatch: {family}")
        control_rows: dict[str, Any] = {}
        graph_scores: dict[str, dict[int, dict[str, float]]] = {}
        for control in controls:
            runs = family_controls[control]
            if sorted(runs) != seeds:
                raise RuntimeError(f"A2 seed coverage mismatch: {family}/{control}")
            seed_scores = [float(runs[seed]["two_edit"]["two_edit_balanced"]) for seed in seeds]
            control_rows[control] = {
                "seed_scores": seed_scores,
                "mean_two_edit_balanced": mean(seed_scores),
                "student_t_95_two_edit_balanced": student_t_95(seed_scores),
                "mean_change": mean([float(runs[s]["two_edit"]["two_edit_change_accuracy"]) for s in seeds]),
                "mean_preservation": mean([float(runs[s]["two_edit"]["two_edit_preservation"]) for s in seeds]),
                "mean_target_success": mean([float(runs[s]["two_edit"]["target_success"]) for s in seeds]),
            }
            graph_scores[control] = {
                seed: {
                    graph_id: float(metrics["two_edit_balanced"])
                    for graph_id, metrics in runs[seed]["two_edit"]["per_graph"].items()
                }
                for seed in seeds
            }
        comparisons: dict[str, Any] = {}
        for comparison_index, (name, (arm, baseline)) in enumerate(COMPARISONS.items()):
            deltas = [
                control_rows[arm]["seed_scores"][index]
                - control_rows[baseline]["seed_scores"][index]
                for index in range(len(seeds))
            ]
            comparisons[name] = {
                "arm": arm,
                "baseline": baseline,
                "paired_seed_deltas": deltas,
                "mean_delta": mean(deltas),
                "student_t_95_delta": student_t_95(deltas),
                "paired_graph_bootstrap_95_delta": paired_graph_bootstrap_95(
                    graph_scores[arm], graph_scores[baseline],
                    bootstrap_seed=20260906 + family_index * 10 + comparison_index,
                ),
            }
        families[family] = {"controls": control_rows, "comparisons": comparisons}

    return {
        "protocol": "a2_frozen_checkpoint_report_v1",
        "stage": "preregistered_confirmatory_measurement",
        "test_evaluated": False,
        "registered_cells": 400,
        "completed_cells": len(summary_hashes),
        "source_checkpoint_count": len(checkpoints_by_source),
        "seeds": seeds,
        "families": families,
        "analysis": manifest["analysis"],
        "provenance": {
            "manifest_sha256": sha256(manifest_path),
            "source_catalog_sha256": manifest["source"]["catalog_sha256"],
            "unique_cell_summary_count": len(set(summary_hashes)),
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "experimental_unit": "graph",
            "world_size": 4,
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# A2 frozen-checkpoint measurement",
        "",
        "All 400 registered validation-only measurements are complete. Each fresh control used exactly four BERT-base ranks; active measurements were exact-provenance reuse of the frozen A1 validation outputs. No checkpoint was retrained or mutated, and held-out test remained locked.",
        "",
        "| Family | Control | Mean balanced | 95% t interval | Change | Preservation | Target |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for family, family_row in report["families"].items():
        for control, row in family_row["controls"].items():
            interval = row["student_t_95_two_edit_balanced"]
            lines.append(
                f"| {family} | {control} | {row['mean_two_edit_balanced']:.6f} | "
                f"[{interval[0]:.6f}, {interval[1]:.6f}] | {row['mean_change']:.6f} | "
                f"{row['mean_preservation']:.6f} | {row['mean_target_success']:.6f} |"
            )
    lines.extend(["", "## Paired comparisons", ""])
    for family, family_row in report["families"].items():
        lines.append(f"### {family}")
        lines.append("")
        for name, row in family_row["comparisons"].items():
            t_interval = row["student_t_95_delta"]
            bootstrap = row["paired_graph_bootstrap_95_delta"]
            lines.append(
                f"- `{name}` ({row['arm']} minus {row['baseline']}): {row['mean_delta']:+.6f}; "
                f"95% t [{t_interval[0]:+.6f}, {t_interval[1]:+.6f}]; "
                f"paired graph bootstrap [{bootstrap[0]:+.6f}, {bootstrap[1]:+.6f}]."
            )
        lines.append("")
    lines.append("Prompting is reported separately from the primary masking check and secondary editor-contribution comparison. All results are validation-only; no held-out test record was opened.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=root / "registry" / "a2_measurement_manifest.json")
    parser.add_argument("--json-output", type=Path, default=root / "reports" / "A2_MEASUREMENT.json")
    parser.add_argument("--markdown-output", type=Path, default=root / "reports" / "A2_MEASUREMENT.md")
    args = parser.parse_args()
    report = aggregate(root, args.manifest.resolve())
    args.json_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown_output.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"completed_cells": report["completed_cells"], "test_evaluated": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
