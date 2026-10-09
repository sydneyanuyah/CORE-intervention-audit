#!/usr/bin/env python3
"""Provenance-bound T1 aggregation over frozen F2 composed measurements."""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "registry/t1_manifest.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean_ci(values: list[float]) -> dict:
    mean = statistics.fmean(values)
    if len(values) < 2:
        return {"mean": mean, "ci95": [mean, mean], "n": len(values)}
    half = 2.093024054 * statistics.stdev(values) / math.sqrt(len(values))
    return {"mean": mean, "ci95": [mean - half, mean + half], "n": len(values)}


def source_summary(family: str, method: str, seed: int) -> Path:
    candidates = [
        ROOT / f"outputs/f2-fixed/{family}/{method}/seed-{seed}/run_summary.json",
        ROOT / f"outputs/f2/{family}/{method}/seed-{seed}/run_summary.json",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"missing F2 summary: {family}/{method}/{seed}")


def metric_and_examples(data: dict) -> tuple[float, int]:
    block = data.get("two_edit", data.get("composed", {}))
    if "two_edit_balanced" in block:
        return float(block["two_edit_balanced"]), int(block["example_count"])
    if "balanced_numerical_score" in block:
        return float(block["balanced_numerical_score"]), int(block["examples"])
    raise ValueError("source summary lacks registered composed metric")


def main() -> int:
    manifest = load(MANIFEST)
    data_path = ROOT / manifest["data"]
    if sha256(data_path) != manifest["data_sha256"]:
        raise ValueError("T1 data hash mismatch")
    observed_counts = {family: 0 for family in manifest["family_pair_counts"]}
    for line in data_path.open():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("split") != "validation":
            raise ValueError("T1 encountered a non-validation row")
        observed_counts[row["family"]] += 1
    if observed_counts != manifest["family_pair_counts"]:
        raise ValueError(f"T1 pair inventory mismatch: {observed_counts}")

    families = {}
    source_cells = []
    for family in manifest["family_pair_counts"]:
        values = {method: [] for method in manifest["methods"]}
        observed_examples = set()
        for seed in manifest["seeds"]:
            for method in manifest["methods"]:
                path = source_summary(family, method, seed)
                data = load(path)
                if data.get("method") != method or int(data.get("seed")) != seed:
                    raise ValueError(f"F2 identity mismatch: {path}")
                if data.get("test_evaluated") is not False:
                    raise ValueError(f"test provenance failed: {path}")
                if int(data.get("world_size", 0)) != 4:
                    raise ValueError(f"world size provenance failed: {path}")
                metric, examples = metric_and_examples(data)
                values[method].append(metric)
                observed_examples.add(examples)
                source_cells.append({
                    "family": family,
                    "method": method,
                    "seed": seed,
                    "summary": str(path.relative_to(ROOT)),
                    "summary_sha256": sha256(path),
                    "examples": examples,
                    "two_edit_balanced": metric,
                })
        expected_examples = manifest["expected_summary_examples_per_seed"][family]
        eligible = observed_examples == {expected_examples}
        method_stats = {method: mean_ci(scores) for method, scores in values.items()}
        o3_delta = [a - b for a, b in zip(values["o3"], values["prompting"])]
        o2_delta = [a - b for a, b in zip(values["o2"], values["prompting"])]
        families[family] = {
            "eligible": eligible,
            "expected_examples_per_seed": expected_examples,
            "observed_examples_per_seed": sorted(observed_examples),
            "methods": method_stats,
            "o3_minus_prompting": mean_ci(o3_delta),
            "o2_minus_prompting": mean_ci(o2_delta),
            "quarantine_reason": None if eligible else "F2 summary does not cover the registered T1 pair inventory",
        }

    eligible = [name for name, result in families.items() if result["eligible"]]
    output = {
        "protocol": manifest["protocol"],
        "manifest_sha256": sha256(MANIFEST),
        "data_sha256": sha256(data_path),
        "eligible_families": eligible,
        "quarantined_families": [name for name in families if name not in eligible],
        "families": families,
        "source_cells": source_cells,
        "test_evaluated": False,
    }
    evidence = ROOT / "reports/T1_EVIDENCE.json"
    evidence.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")

    lines = [
        "# T1 Composed-pair measurement",
        "",
        f"T1 aggregates {len(source_cells)} frozen F2 validation summaries; no training or test access occurred.",
        "",
        "| Family | Status | Prompting | O2 | O3 | O3 - prompting (95% CI) | O2 - prompting (95% CI) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for family, result in families.items():
        methods = result["methods"]
        d3 = result["o3_minus_prompting"]
        d2 = result["o2_minus_prompting"]
        status = "eligible" if result["eligible"] else "quarantined"
        lines.append(
            f"| {family} | {status} | {methods['prompting']['mean']:.6f} | {methods['o2']['mean']:.6f} | "
            f"{methods['o3']['mean']:.6f} | {d3['mean']:+.6f} [{d3['ci95'][0]:+.6f}, {d3['ci95'][1]:+.6f}] | "
            f"{d2['mean']:+.6f} [{d2['ci95'][0]:+.6f}, {d2['ci95'][1]:+.6f}] |"
        )
    lines += ["", "Quarantined families remain visible but are excluded from valid T1 claims."]
    (ROOT / "reports/T1_REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"eligible": eligible, "quarantined": output["quarantined_families"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
