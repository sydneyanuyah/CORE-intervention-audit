#!/usr/bin/env python3
"""Aggregate the registered Llama-3.3-70B CLadder O2/O3 extension."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "registry/llama70b_cladder_extension_manifest.json"
AMENDMENT = ROOT / "registry/llama70b_cladder_cache_amendment.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interval(values: list[float]) -> dict:
    mean = statistics.fmean(values)
    sem = statistics.stdev(values) / math.sqrt(len(values))
    margin = 2.093024054 * sem  # df=19, two-sided 95%
    return {"mean": mean, "student_t_95": [mean - margin, mean + margin], "n_paired_seeds": len(values)}


def main() -> int:
    manifest, amendment = json.loads(MANIFEST.read_text()), json.loads(AMENDMENT.read_text())
    manifest_sha, amendment_sha = sha(MANIFEST), sha(AMENDMENT)
    if amendment.get("base_manifest_sha256") != manifest_sha:
        raise ValueError("amendment binding mismatch")
    rows = {}
    for cell in manifest["cells"]:
        path = ROOT / cell["output"] / "run_summary.json"
        row = json.loads(path.read_text())
        if row.get("cell_id") != cell["cell_id"] or row.get("manifest_sha256") != manifest_sha or row.get("amendment_sha256") != amendment_sha or row.get("test_evaluated") is not False:
            raise ValueError(f"invalid cell: {cell['cell_id']}")
        rows[(cell["method"], int(cell["seed"]))] = row
    metrics = {}
    for method in ("o2", "o3"):
        selected = [rows[(method, seed)] for seed in manifest["seeds"]]
        metrics[method] = {
            "clean_two_edit_balanced": interval([row["clean"]["two_edit_balanced"] for row in selected]),
            "changed_pair_fraction": interval([row["changed_pair_fraction"] for row in selected]),
            "inverted_two_edit_balanced": interval([row["inverted_against_retained_gold"]["two_edit_balanced"] for row in selected]),
            "inversion_retained_gold_delta": interval([row["inverted_against_retained_gold"]["two_edit_balanced"] - row["clean"]["two_edit_balanced"] for row in selected]),
            "do_nothing_two_edit_balanced": interval([row["do_nothing"]["two_edit_balanced"] for row in selected]),
            "operator_parameter_count": selected[0]["operator_parameter_count"],
        }
    def contrast(fn):
        return interval([fn(rows[("o2", seed)]) - fn(rows[("o3", seed)]) for seed in manifest["seeds"]])
    contrasts = {
        "o2_minus_o3_clean": contrast(lambda row: row["clean"]["two_edit_balanced"]),
        "o2_minus_o3_changed_pair_fraction": contrast(lambda row: row["changed_pair_fraction"]),
        "o2_minus_o3_inversion_drop": contrast(lambda row: row["inverted_against_retained_gold"]["two_edit_balanced"] - row["clean"]["two_edit_balanced"]),
    }
    result = {
        "protocol": manifest["protocol"], "model": manifest["model"], "model_revision": manifest["model_revision"],
        "completed_cells": len(rows), "expected_cells": len(manifest["cells"]),
        "manifest_sha256": manifest_sha, "amendment_sha256": amendment_sha,
        "feature_cache_sha256": amendment["feature_cache_sha256"],
        "metrics": metrics, "paired_contrasts": contrasts,
        "inference_unit": "paired training seed", "evaluation_split": "validation", "test_evaluated": False,
    }
    json_path = ROOT / "reports/LLAMA70B_EXTENSION.json"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    def fmt(x): return f"{x['mean']:+.6f} [{x['student_t_95'][0]:+.6f}, {x['student_t_95'][1]:+.6f}]"
    markdown = f"""# Llama-3.3-70B CLadder operator extension

All {len(rows)}/{len(manifest['cells'])} registered cells completed on validation only. The Llama-3.3-70B-Instruct backbone was frozen and evaluated in BF16 for one-time feature extraction; the canonical O2 and O3 operators and a state head were trained independently for seeds 401–420.

| Measurement | O2 | O3 |
|---|---:|---:|
| Clean composed balanced score | {metrics['o2']['clean_two_edit_balanced']['mean']:.6f} | {metrics['o3']['clean_two_edit_balanced']['mean']:.6f} |
| Pair vectors moved by inversion | {metrics['o2']['changed_pair_fraction']['mean']:.3%} | {metrics['o3']['changed_pair_fraction']['mean']:.3%} |
| Inverted score against retained gold | {metrics['o2']['inverted_two_edit_balanced']['mean']:.6f} | {metrics['o3']['inverted_two_edit_balanced']['mean']:.6f} |
| Do-nothing floor | {metrics['o2']['do_nothing_two_edit_balanced']['mean']:.6f} | {metrics['o3']['do_nothing_two_edit_balanced']['mean']:.6f} |

Paired O2−O3 clean contrast: {fmt(contrasts['o2_minus_o3_clean'])}. Paired O2−O3 movement contrast: {fmt(contrasts['o2_minus_o3_changed_pair_fraction'])}. These are paired-seed intervals; no held-out test data were read.
"""
    (ROOT / "reports/LLAMA70B_EXTENSION.md").write_text(markdown)
    print(json.dumps({"completed_cells": len(rows), "o2_minus_o3_clean": contrasts["o2_minus_o3_clean"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
