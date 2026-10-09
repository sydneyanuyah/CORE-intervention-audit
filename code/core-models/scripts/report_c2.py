#!/usr/bin/env python3
"""Generate fail-closed C2 floor calibration from C1 real cells."""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path: Path): return json.loads(path.read_text())


def interval(values: list[float]) -> list[float]:
    mean = statistics.fmean(values)
    if len(values) < 2: return [mean, mean]
    # Fixed n=20 two-sided t critical value, df=19.
    half = 2.093024054 * statistics.stdev(values) / math.sqrt(len(values))
    return [mean - half, mean + half]


def bootstrap(values: list[float]) -> list[float]:
    rng = random.Random(402); means = []
    for _ in range(10000): means.append(statistics.fmean(rng.choice(values) for _ in values))
    means.sort(); return [means[249], means[9749]]


def main() -> int:
    manifest_path = ROOT / "registry/c2_manifest.json"; manifest = load(manifest_path)
    suite = ROOT / manifest["floor_suite"]
    if sha(suite) != manifest["floor_suite_sha256"]: raise RuntimeError("C2 floor suite changed")
    rows = [json.loads(line) for line in suite.read_text().splitlines() if line.strip()]
    if len(rows) != 7200 or any(r.get("split") != "validation" or r.get("test_evaluated") is not False for r in rows): raise RuntimeError("C2 suite contract failed")
    per_graph = {}
    for graph in manifest["graph_seeds"]:
        subset = [r for r in rows if r["graph_group_id"] == f"xor:graph:{graph}"]
        if len(subset) != 360: raise RuntimeError(f"C2 graph {graph} incomplete")
        scores = {}
        for method in ("do_nothing", "do_everything", "random_init"):
            correct = total = 0
            for row in subset:
                for key, value in row["gold_state"].items(): correct += row["floor_predictions"][method][key] == value; total += 1
            scores[method] = correct / total
        c1 = load(ROOT / f"outputs/c1/real/seed-{graph - 2480}/run_summary.json")
        if c1.get("protocol") != "c1_fixed_o3_closing_controls_v1" or c1.get("graph_seed") != graph or c1.get("test_evaluated") is not False: raise RuntimeError("C2 source C1 cell invalid")
        scores["trained_o3"] = float(c1["single"]["accuracy"]); per_graph[str(graph)] = scores
    methods = manifest["methods"]; aggregates = {}
    for method in methods:
        values = [per_graph[str(g)][method] for g in manifest["graph_seeds"]]
        aggregates[method] = {"mean": statistics.fmean(values), "student_t_95": interval(values)}
    comparisons = {}
    for floor in methods[:-1]:
        values = [per_graph[str(g)]["trained_o3"] - per_graph[str(g)][floor] for g in manifest["graph_seeds"]]
        comparisons[f"trained_o3-minus-{floor}"] = {"mean": statistics.fmean(values), "student_t_95": interval(values), "paired_graph_bootstrap_95": bootstrap(values)}
    evidence = {"protocol": manifest["protocol"], "manifest_sha256": sha(manifest_path), "graphs": 20, "cases": 7200,
                "per_graph": per_graph, "aggregates": aggregates, "comparisons": comparisons, "test_evaluated": False}
    (ROOT / "reports/C2_EVIDENCE.json").write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    lines = ["# C2 Law-floor calibration", "", "Validation-only, 20 graph units; no held-out test access.", "", "| Method | Mean variable accuracy | 95% t interval |", "|---|---:|---:|"]
    for method in methods:
        row = aggregates[method]; lines.append(f"| {method} | {row['mean']:.6f} | [{row['student_t_95'][0]:.6f}, {row['student_t_95'][1]:.6f}] |")
    lines += ["", "| Paired comparison | Mean delta | 95% t interval | 95% graph bootstrap |", "|---|---:|---:|---:|"]
    for name, row in comparisons.items(): lines.append(f"| {name} | {row['mean']:+.6f} | [{row['student_t_95'][0]:+.6f}, {row['student_t_95'][1]:+.6f}] | [{row['paired_graph_bootstrap_95'][0]:+.6f}, {row['paired_graph_bootstrap_95'][1]:+.6f}] |")
    (ROOT / "reports/C2_REPORT.md").write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__": raise SystemExit(main())
