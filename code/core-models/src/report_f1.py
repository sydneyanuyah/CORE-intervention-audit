#!/usr/bin/env python3
"""Fail-closed paired graph inference for the completed F1 matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] if low == high else ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    seeds = manifest["optimization_seeds"]
    if seeds != list(range(101, 121)) or manifest.get("data", {}).get("test_evaluated") is not False:
        raise ValueError("invalid F1 manifest or test lock")
    rows = []
    for seed, graph_seed in zip(seeds, manifest["graph_seeds"]):
        reader = args.outputs / "readers" / f"seed-{seed}" / "reader.pt"
        reader_summary = args.outputs / "readers" / f"seed-{seed}" / "run_summary.json"
        if not reader.is_file() or not reader_summary.is_file():
            raise RuntimeError(f"F1 reader seed {seed} is incomplete")
        reader_meta = json.loads(reader_summary.read_text())
        if reader_meta.get("test_evaluated") is not False or reader_meta.get("world_size") != 4:
            raise RuntimeError(f"F1 reader seed {seed} violates provenance")
        methods = {}
        for method in ("o2", "o3"):
            directory = args.outputs / method / f"seed-{seed}"
            summary_path, checkpoint = directory / "run_summary.json", directory / "operator.pt"
            if not summary_path.is_file() or not checkpoint.is_file():
                raise RuntimeError(f"F1 {method} seed {seed} is incomplete")
            summary = json.loads(summary_path.read_text())
            if (
                summary.get("protocol") != "f1_cell_v1"
                or summary.get("method") != method
                or summary.get("seed") != seed
                or summary.get("graph_seed") != graph_seed
                or summary.get("world_size") != 4
                or summary.get("test_evaluated") is not False
                or summary.get("reader_sha256") != sha256(reader)
            ):
                raise RuntimeError(f"F1 {method} seed {seed} violates provenance")
            methods[method] = summary
        o2 = float(methods["o2"]["composed"]["balanced_intervention_score"])
        o3 = float(methods["o3"]["composed"]["balanced_intervention_score"])
        rows.append({"seed": seed, "graph_seed": graph_seed, "o2": o2, "o3": o3, "delta": o3 - o2})
    deltas = [row["delta"] for row in rows]
    mean = statistics.mean(deltas)
    sem = statistics.stdev(deltas) / math.sqrt(len(deltas))
    critical = 2.093024054408263  # two-sided 95%, df=19
    student = [mean - critical * sem, mean + critical * sem]
    rng = random.Random(301)
    boot = [statistics.mean(rng.choices(deltas, k=len(deltas))) for _ in range(10000)]
    report = {
        "protocol": "f1_paired_graph_report_v1",
        "manifest_sha256": sha256(args.manifest),
        "complete_cells": 40,
        "reader_count": 20,
        "primary": "composed.balanced_intervention_score:o3-o2",
        "o2_mean": statistics.mean(row["o2"] for row in rows),
        "o3_mean": statistics.mean(row["o3"] for row in rows),
        "paired_delta": mean,
        "student_t_95": student,
        "paired_graph_bootstrap_95": [quantile(boot, 0.025), quantile(boot, 0.975)],
        "passed": student[0] > 0 and quantile(boot, 0.025) > 0,
        "rows": rows,
        "test_evaluated": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in ("paired_delta", "student_t_95", "paired_graph_bootstrap_95", "passed")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
