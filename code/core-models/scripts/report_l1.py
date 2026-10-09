#!/usr/bin/env python3
"""Fail-closed aggregate reporter for the preregistered L1 matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any


T_CRITICAL_95_DF19 = 2.093024054408263
BOOTSTRAP_DRAWS = 10_000
PROFILES = ("core_base_1law", "core_i_3law", "core_full_4law")
FLOORS = ("do_nothing", "do_everything", "random_init")


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object in {path}")
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean(values: list[float]) -> float:
    if not values:
        raise RuntimeError("cannot average an empty sequence")
    return sum(values) / len(values)


def student_t_95(values: list[float]) -> list[float]:
    if len(values) != 20:
        raise RuntimeError("L1 inference requires exactly twenty paired seeds")
    center = mean(values)
    variance = sum((value - center) ** 2 for value in values) / 19
    half_width = T_CRITICAL_95_DF19 * math.sqrt(variance / 20)
    return [center - half_width, center + half_width]


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def paired_seed_bootstrap_95(values: list[float], seed: int) -> list[float]:
    if len(values) != 20:
        raise RuntimeError("L1 paired bootstrap requires exactly twenty seeds")
    rng = random.Random(seed)
    samples = [mean([values[rng.randrange(20)] for _ in range(20)]) for _ in range(BOOTSTRAP_DRAWS)]
    return [percentile(samples, 0.025), percentile(samples, 0.975)]


def selected_validation(summary: dict[str, Any]) -> dict[str, Any]:
    if summary.get("validation") is not None:
        return summary["validation"]
    history = summary.get("history")
    if not isinstance(history, list) or not history:
        raise RuntimeError("learned L1 cell has no validation history")
    selected = [row for row in history if row.get("selected") is True]
    if not selected:
        raise RuntimeError("learned L1 cell has no selected validation epoch")
    best = max(selected, key=lambda row: (float(row["selection_value"]), -int(row["epoch"])))
    if not math.isclose(float(best["selection_value"]), float(summary["best_validation_metric"]), abs_tol=1e-12):
        raise RuntimeError("selected epoch does not match the frozen checkpoint metric")
    return best["validation"]


def metrics(summary: dict[str, Any]) -> dict[str, float | int | None]:
    validation = selected_validation(summary)
    if "balanced_intervention_score" in validation:
        return {
            "balanced_intervention_score": float(validation["balanced_intervention_score"]),
            "change_accuracy": float(validation["change_accuracy"]),
            "preservation_accuracy": float(validation["preservation_accuracy"]),
            "target_accuracy": None,
            "variable_accuracy": float(validation["variable_accuracy"]),
            "task_accuracy": None,
            "macro_f1": None,
            "record_exact_match": None,
            "graph_count": int(validation["graph_count"]),
        }
    variable = validation["per_variable"]
    groups = variable["groups"]
    change = float(groups["change"]["accuracy"])
    preservation = float(groups["preservation"]["accuracy"])
    return {
        "balanced_intervention_score": 0.5 * (change + preservation),
        "change_accuracy": change,
        "preservation_accuracy": preservation,
        "target_accuracy": float(groups["target"]["accuracy"]),
        "variable_accuracy": float(variable["variable_accuracy"]),
        "task_accuracy": float(validation["accuracy"]),
        "macro_f1": float(validation["macro_f1"]),
        "record_exact_match": float(variable["record_exact_match"]),
        "graph_count": int(validation["graph_count"]),
    }


def aggregate(root: Path, inventory_path: Path) -> dict[str, Any]:
    manifest_path = root / "registry" / "l1_manifest.json"
    cells_path = root / "registry" / "l1_cells.json"
    manifest, registry, inventory = load(manifest_path), load(cells_path), load(inventory_path)
    cells = registry["cells"]
    expected_ids = {cell["cell_id"] for cell in cells}
    inventory_ids = {cell["cell_id"] for cell in inventory.get("cells", [])}
    if len(cells) != 240 or inventory.get("registered") != 240:
        raise RuntimeError("L1 requires exactly 240 registered cells")
    if inventory.get("complete") != 240 or inventory.get("remaining") != 0 or inventory.get("rejected"):
        raise RuntimeError("L1 provenance inventory is not complete")
    if expected_ids != inventory_ids or inventory.get("manifest_sha256") != registry.get("manifest_sha256"):
        raise RuntimeError("L1 inventory does not match the immutable cell registry")
    if manifest.get("test_evaluated") is not False:
        raise RuntimeError("L1 manifest does not explicitly lock test")

    grouped: dict[str, dict[str, dict[int, dict[str, Any]]]] = {}
    provenance = {row["cell_id"]: row for row in inventory["cells"]}
    for cell in cells:
        cell_id = cell["cell_id"]
        summary_path = root / provenance[cell_id]["summary"]
        if sha256(summary_path) != provenance[cell_id]["summary_sha256"]:
            raise RuntimeError(f"summary changed after provenance validation: {cell_id}")
        summary = load(summary_path)
        if summary.get("test_evaluated") is not False:
            raise RuntimeError(f"test isolation is not explicit: {cell_id}")
        grouped.setdefault(cell["family"], {}).setdefault(cell["method"], {})[int(cell["seed"])] = {
            "metrics": metrics(summary),
            "summary_sha256": provenance[cell_id]["summary_sha256"],
            "checkpoint_sha256": provenance[cell_id].get("checkpoint_sha256"),
        }

    seeds = [int(seed) for seed in manifest["optimization_seeds"]]
    methods = (*PROFILES, *FLOORS)
    families: dict[str, Any] = {}
    for family in ("xor", manifest["held_out_family"]["name"]):
        family_methods: dict[str, Any] = {}
        for method in methods:
            runs = grouped.get(family, {}).get(method, {})
            if sorted(runs) != seeds:
                raise RuntimeError(f"{family}/{method} does not have the registered twenty seeds")
            score_names = (
                "balanced_intervention_score", "change_accuracy", "preservation_accuracy",
                "target_accuracy", "variable_accuracy", "task_accuracy", "macro_f1", "record_exact_match",
            )
            row: dict[str, Any] = {"seed_scores": {str(seed): runs[seed]["metrics"] for seed in seeds}}
            for name in score_names:
                values = [runs[seed]["metrics"][name] for seed in seeds]
                row[f"mean_{name}"] = None if any(value is None for value in values) else mean([float(value) for value in values])
                row[f"student_t_95_{name}"] = None if any(value is None for value in values) else student_t_95([float(value) for value in values])
            row["summary_sha256"] = [runs[seed]["summary_sha256"] for seed in seeds]
            row["checkpoint_sha256"] = [runs[seed]["checkpoint_sha256"] for seed in seeds]
            family_methods[method] = row

        comparisons: dict[str, Any] = {}
        comparison_pairs = (
            ("core_i_3law", "core_base_1law"),
            ("core_full_4law", "core_i_3law"),
            ("core_full_4law", "core_base_1law"),
            ("core_full_4law", "do_nothing"),
            ("core_full_4law", "do_everything"),
            ("core_full_4law", "random_init"),
        )
        for arm, baseline in comparison_pairs:
            deltas = [
                float(grouped[family][arm][seed]["metrics"]["balanced_intervention_score"])
                - float(grouped[family][baseline][seed]["metrics"]["balanced_intervention_score"])
                for seed in seeds
            ]
            comparisons[f"{arm}_minus_{baseline}"] = {
                "mean_delta_balanced_intervention_score": mean(deltas),
                "paired_seed_student_t_95": student_t_95(deltas),
                "paired_seed_bootstrap_95": paired_seed_bootstrap_95(
                    deltas, 20260906 + sum(ord(char) for char in family + arm + baseline)
                ),
            }
        families[family] = {"methods": family_methods, "paired_comparisons": comparisons}

    return {
        "protocol": "l1_law_retention_aggregate_v1",
        "manifest_sha256": sha256(manifest_path),
        "cell_registry_manifest_sha256": registry["manifest_sha256"],
        "inventory_sha256": sha256(inventory_path),
        "registered_cells": 240,
        "complete_cells": 240,
        "rejected_cells": 0,
        "test_evaluated": False,
        "families": families,
        "inference": {
            "paired_seed_student_t_95": True,
            "paired_seed_bootstrap_95": True,
            "graph_level_bootstrap": {
                "available": False,
                "reason": "cell summaries retain aggregate validation metrics but not per-graph predictions; no graph-level interval is fabricated",
            },
        },
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# L1 Law-Retention Results", "",
        "All 240 registered cells passed provenance validation; no held-out test rows were evaluated.", "",
        "| Family | Method | Balanced intervention | Change | Preservation | Variable accuracy |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for family, family_row in report["families"].items():
        for method, row in family_row["methods"].items():
            lines.append(
                f"| {family} | {method} | {row['mean_balanced_intervention_score']:.6f} | "
                f"{row['mean_change_accuracy']:.6f} | {row['mean_preservation_accuracy']:.6f} | "
                f"{row['mean_variable_accuracy']:.6f} |"
            )
    lines.extend(["", "## Paired retention comparisons", ""])
    for family, family_row in report["families"].items():
        lines.extend([f"### {family}", "", "| Comparison | Mean delta | Student-t 95% | Paired-seed bootstrap 95% |", "|---|---:|---:|---:|"])
        for name, row in family_row["paired_comparisons"].items():
            t = row["paired_seed_student_t_95"]
            b = row["paired_seed_bootstrap_95"]
            lines.append(f"| {name} | {row['mean_delta_balanced_intervention_score']:+.6f} | [{t[0]:+.6f}, {t[1]:+.6f}] | [{b[0]:+.6f}, {b[1]:+.6f}] |")
        lines.append("")
    lines.extend([
        "## Inference scope", "",
        "The preregistered paired-seed Student-t intervals and a paired-seed bootstrap are reported. "
        "Graph-level bootstrap inference is unavailable because the frozen cell summaries do not retain per-graph predictions; the reporter fails closed rather than inventing graph-level evidence.", "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--inventory", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    inventory = args.inventory or root / "reports" / "l1_inventory.json"
    report = aggregate(root, inventory)
    json_path, md_path = root / "reports" / "L1_RESULTS.json", root / "reports" / "L1_RESULTS.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(markdown(report), encoding="utf-8")
    status = load(inventory)
    status["aggregate_report"] = str(json_path.relative_to(root))
    status["aggregate_report_sha256"] = sha256(json_path)
    (root / "outputs" / "l1" / "status.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"complete": report["complete_cells"], "report": str(json_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
