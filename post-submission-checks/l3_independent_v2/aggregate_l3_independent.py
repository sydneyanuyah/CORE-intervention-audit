#!/usr/bin/env python3
"""Aggregate corrected L3 results and emit JSON, CSV, and Markdown."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


T_CRITICAL = {5: 2.7764451051977987, 20: 2.093024054408263}
METHODS = ("lora_matched", "loreft", "o2")
METRICS = ("balanced_accuracy", "identity", "idempotence", "commutation", "last_write_wins")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interval(values: list[float]) -> dict:
    mean = statistics.fmean(values)
    half = T_CRITICAL[len(values)] * statistics.stdev(values) / math.sqrt(len(values))
    return {"mean": mean, "student_t_95": [mean - half, mean + half], "n": len(values)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.input_dir.glob("*_L3_INDEPENDENT_V2.json"))
    if len(paths) != 4:
        raise ValueError(f"expected four backbone reports, found {len(paths)}")
    rows = []
    sources = {}
    for path in paths:
        payload = json.loads(path.read_text())
        if payload["completed_cells"] != payload["expected_cells"] or payload["checkpoint_collisions"]:
            raise ValueError(f"invalid source report: {path}")
        backbone = path.name.split("_L3_", 1)[0]
        sources[backbone] = {"path": str(path), "sha256": sha(path), "manifest_sha256": payload["manifest_sha256"]}
        rows.extend(dict(row, backbone=backbone) for row in payload["results"])
    if len(rows) != 60:
        raise ValueError(f"expected 60 cells, found {len(rows)}")
    for backbone in sorted({row["backbone"] for row in rows}):
        for seed in sorted({row["seed"] for row in rows if row["backbone"] == backbone}):
            group = [row for row in rows if row["backbone"] == backbone and row["seed"] == seed]
            if len({row["checkpoint_sha256"] for row in group}) != 3:
                raise ValueError(f"checkpoint collision for {backbone} seed {seed}")
            if len({row["operator_class"] for row in group}) != 3:
                raise ValueError(f"operator-class collision for {backbone} seed {seed}")
    scopes = {"ALL": rows}
    scopes.update({backbone: [row for row in rows if row["backbone"] == backbone] for backbone in sorted({row["backbone"] for row in rows})})
    aggregates = []
    for scope, scope_rows in scopes.items():
        for method in METHODS:
            group = [row for row in scope_rows if row["method"] == method]
            item = {
                "scope": scope, "method": method, "operator_class": group[0]["operator_class"],
                "operator_parameter_counts": sorted({row["operator_trainable_parameters"] for row in group}),
            }
            item["balanced_accuracy"] = interval([row["validation_balanced_accuracy"] for row in group])
            for law in METRICS[1:]:
                item[law] = interval([row["law_residuals"][law] for row in group])
            aggregates.append(item)
    result = {
        "protocol": "l3_independent_methods_cross_backbone_aggregate_v2",
        "source_reports": sources, "cells": len(rows), "checkpoint_collisions": [],
        "evaluation_split": "validation", "test_evaluated": False,
        "comparison_scope": "frozen_feature_operator_probe",
        "reporting_caveat": "lora_matched is a parameter-matched frozen-feature adapter, not Q/V LoRA on backbone weights",
        "aggregates": aggregates,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "L3_INDEPENDENT_V2_AGGREGATE.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with (args.output_dir / "L3_INDEPENDENT_V2_AGGREGATE.csv").open("w", newline="") as handle:
        fields = ["scope", "method", "operator_class", "metric", "mean", "ci_low", "ci_high", "n"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in aggregates:
            for metric in METRICS:
                value = item[metric]
                writer.writerow({
                    "scope": item["scope"], "method": item["method"], "operator_class": item["operator_class"],
                    "metric": metric, "mean": value["mean"], "ci_low": value["student_t_95"][0],
                    "ci_high": value["student_t_95"][1], "n": value["n"],
                })
    lines = [
        "# Corrected independent L3 expansion", "",
        "The invalid shared-checkpoint row is superseded by 60 validation-only cells: four frozen backbones, three distinct operator classes, and five seeds per class. All 20 backbone-seed triplets have three distinct checkpoint hashes.", "",
        "| Method | Balanced accuracy | Identity residual | Idempotence residual | Commutation residual | Last-write residual |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in [row for row in aggregates if row["scope"] == "ALL"]:
        values = []
        for metric in METRICS:
            value = item[metric]
            values.append(f"{value['mean']:.4f} [{value['student_t_95'][0]:.4f}, {value['student_t_95'][1]:.4f}]")
        lines.append(f"| `{item['method']}` | " + " | ".join(values) + " |")
    lines.extend(["", "Lower law residual is better. Intervals are Student-t 95% intervals over 20 backbone-seed cells.", "", "Important: `lora_matched` is the corrected frozen-feature parameter-matched adapter. It must not be described as Q/V LoRA on the frozen backbone."])
    (args.output_dir / "L3_INDEPENDENT_V2_AGGREGATE.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"cells": len(rows), "aggregate_rows": len(aggregates)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
