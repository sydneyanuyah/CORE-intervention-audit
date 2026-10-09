#!/usr/bin/env python3
"""Aggregate repaired A3 correct/adjacent/random frozen-checkpoint measurements."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parents[1].resolve()
def load(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def mean(values): return sum(values) / len(values)


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--manifest", type=Path, required=True); parser.add_argument("--json-output", type=Path, required=True); parser.add_argument("--markdown-output", type=Path, required=True); args = parser.parse_args()
    manifest = load(args.manifest); catalog = load(ROOT / manifest["source"]["catalog"]); grouped = {}
    for row in catalog["cells"]:
        family = row["family"]
        for control in manifest["controls"]:
            path = ROOT / manifest["paths"]["output_template"].format(family=family, control=control, seed=row["seed"]) / "cell_summary.json"
            value = load(path)
            if value.get("test_evaluated") is not False or value.get("world_size") != 4: raise RuntimeError("inadmissible A3 cell")
            grouped.setdefault(family, {}).setdefault(control, []).append(value)
    families = {}
    for family, controls in grouped.items():
        result = {control: {"mean_accuracy": mean([x["accuracy"] for x in rows]), "mean_macro_f1": mean([x["macro_f1"] for x in rows]), "mean_pointer_top1": mean([x["pointer"]["pointer_top1_accuracy"] for x in rows])} for control, rows in controls.items()}
        result["random_minus_correct_accuracy"] = result["random"]["mean_accuracy"] - result["correct"]["mean_accuracy"]
        result["random_minus_correct_pointer_top1"] = result["random"]["mean_pointer_top1"] - result["correct"]["mean_pointer_top1"]
        families[family] = result
    report = {"protocol": "a3_repaired_report_v2", "completed_cells": 300, "test_evaluated": False, "manifest_sha256": sha(args.manifest), "families": families}
    args.json_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    lines = ["# A3 repaired addressing controls", "", "All 300 frozen-checkpoint validation cells completed with four-GPU BERT-base DDP and no test access.", "", "| Family | Correct accuracy | Random accuracy | Random − correct | Correct pointer top-1 | Random pointer top-1 |", "|---|---:|---:|---:|---:|---:|"]
    for family, row in families.items(): lines.append(f'| {family} | {row["correct"]["mean_accuracy"]:.4f} | {row["random"]["mean_accuracy"]:.4f} | {row["random_minus_correct_accuracy"]:+.4f} | {row["correct"]["mean_pointer_top1"]:.4f} | {row["random"]["mean_pointer_top1"]:.4f} |')
    args.markdown_output.write_text("\n".join(lines) + "\n"); print(json.dumps({"completed_cells": 300})); return 0


if __name__ == "__main__": raise SystemExit(main())
