#!/usr/bin/env python3
"""Validate all L3 cells and emit preregistered paired graph-level inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path


LAWS = ("identity", "idempotence", "commutation", "last_write_wins")


def sha256(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()


def interval(values: list[float]) -> list[float]:
    mean = sum(values) / len(values)
    sd = math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
    half = 2.7764451051977987 * sd / math.sqrt(len(values))
    return [mean - half, mean + half]


def bootstrap(values: list[float]) -> list[float]:
    rng = random.Random(31003); n = len(values)
    draws = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(10000))
    return [draws[249], draws[9749]]


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--cells", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True); parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True); parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(); registry, manifest = json.loads(args.cells.read_text()), json.loads(args.manifest.read_text())
    manifest_sha, cells_sha = sha256(args.manifest), sha256(args.cells)
    if registry.get("manifest_sha256") != manifest_sha or len(registry.get("cells", [])) != 40: raise ValueError("L3 registry binding mismatch")
    values = {method: {law: [] for law in LAWS} for method in manifest["methods"]}; provenance = []
    for cell in registry["cells"]:
        directory = args.workdir / "outputs" / "l3" / cell["method"] / f"seed-{cell['seed']}"
        summary_path = directory / "run_summary.json"; summary = json.loads(summary_path.read_text())
        required = {
            "protocol": "l3_eight_method_law_v1", "cell_id": cell["cell_id"], "method": cell["method"],
            "seed": cell["seed"], "graph_seed": cell["graph_seed"], "world_size": 4,
            "manifest_sha256": manifest_sha, "cells_sha256": cells_sha, "reader_sha256": cell["reader_sha256"],
            "test_evaluated": False,
        }
        if any(summary.get(key) != value for key, value in required.items()): raise ValueError(f"L3 summary contract mismatch: {cell['cell_id']}")
        if cell["execution"] == "fresh_operator_training":
            checkpoint = directory / "operator.pt"
            if not checkpoint.is_file(): raise ValueError(f"missing L3 checkpoint: {cell['cell_id']}")
            checkpoint_sha = sha256(checkpoint)
        else:
            if summary.get("source_checkpoint_sha256") != cell["source_checkpoint_sha256"]: raise ValueError(f"source provenance mismatch: {cell['cell_id']}")
            checkpoint_sha = cell["source_checkpoint_sha256"]
        for law in LAWS:
            key = f"{law}_scm_correctness"
            values[cell["method"]][law].append(float(summary["trained"][key]) - float(summary["random_init"][key]))
        provenance.append({"cell_id": cell["cell_id"], "summary_sha256": sha256(summary_path), "checkpoint_sha256": checkpoint_sha, "test_evaluated": False})
    results = {}
    for method in manifest["methods"]:
        results[method] = {}
        for law in LAWS:
            row = values[method][law]
            if len(row) != 5: raise ValueError(f"L3 graph coverage mismatch: {method}/{law}")
            results[method][law] = {"mean_delta_vs_random_init": sum(row) / 5, "paired_seed_student_t_95": interval(row), "paired_graph_bootstrap_95": bootstrap(row), "graph_deltas": row}
    payload = {"protocol": manifest["protocol"], "manifest_sha256": manifest_sha, "cells_sha256": cells_sha, "registered": 40, "complete": 40, "experimental_unit": "graph", "results": results, "provenance": provenance, "test_evaluated": False}
    args.evidence.parent.mkdir(parents=True, exist_ok=True); args.evidence.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = ["# L3 eight-method law comparison", "", "All 40 registered XOR cells completed with five graph seeds, four ranks per BERT-base cell, method-matched random-initialization floors, and no test access.", "", "| Method | Law | Mean trained − random | Student-t 95% | Graph bootstrap 95% |", "|---|---|---:|---:|---:|"]
    for method in manifest["methods"]:
        for law in LAWS:
            row = results[method][law]; t, b = row["paired_seed_student_t_95"], row["paired_graph_bootstrap_95"]
            lines.append(f"| {method} | {law} | {row['mean_delta_vs_random_init']:+.6f} | [{t[0]:+.6f}, {t[1]:+.6f}] | [{b[0]:+.6f}, {b[1]:+.6f}] |")
    args.report.parent.mkdir(parents=True, exist_ok=True); args.report.write_text("\n".join(lines) + "\n")
    print(json.dumps({"registered": 40, "complete": 40, "test_evaluated": False}, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
