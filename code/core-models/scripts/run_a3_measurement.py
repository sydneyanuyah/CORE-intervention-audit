#!/usr/bin/env python3
"""Execute and validate one repaired A3 frozen-checkpoint control."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1].resolve()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path):
    return json.loads(path.read_text())


def cells(manifest):
    catalog = load(ROOT / manifest["source"]["catalog"])
    return [(row, control) for row in catalog["cells"] for control in manifest["controls"]]


def cell_id(row, control):
    return f'{row["family"]}:{control}:{row["seed"]}'


def output(manifest, row, control):
    return ROOT / manifest["paths"]["output_template"].format(family=row["family"], control=control, seed=row["seed"])


def validate(manifest):
    if manifest.get("protocol") != "a3_frozen_checkpoint_addressing_v2" or manifest.get("world_size") != 4 or manifest.get("model_size") != "base" or manifest.get("test_evaluated") is not False:
        raise RuntimeError("invalid A3 top-level contract")
    for key, hash_key in (("report", "report_sha256"), ("manifest", "manifest_sha256"), ("catalog", "catalog_sha256")):
        path = ROOT / manifest["source"][key]
        if not path.is_file() or sha(path) != manifest["source"][hash_key]:
            raise RuntimeError(f"A3 frozen source mismatch: {key}")
    expanded = cells(manifest)
    if len(expanded) != 300 or len({cell_id(*item) for item in expanded}) != 300:
        raise RuntimeError("A3 must contain 300 unique cells")
    return expanded


def command(manifest, row, control, out):
    paths = manifest["paths"]
    source = ROOT / row["source_directory"]
    cmd = [str(ROOT / ".venv/bin/python"), "scripts/launch_addressed_training.py", "--mode", "t3b", "--model-size", "base", "--data-root", paths["data_root"], "--output-dir", str(out), "--sources", row["family"], "--evidence-capable", "--measure-checkpoint", str(source / "best.pt"), "--span-override", control, "--seed", str(row["seed"]), "--variable-loss-weight", "1.0", "--balanced-variable-loss", "--transition-variable-loss", "--causal-task-readout", "--hard-pointer", "--pointer-loss-weight", "1.0"]
    if row["family"] == "xor":
        for graph in paths["xor_graphs"]:
            cmd += ["--xor-bundle", paths["xor_artifact_template"].format(graph=graph), paths["xor_manifest_template"].format(graph=graph)]
    else:
        cmd += ["--groups-dir", "data/groups", "--paraphrase-catalog", paths["paraphrase_catalog"]]
    if row["family"] == "com2":
        cmd += ["--open-text-output", "--selection-metric", "loss", "--batch-size", "8"]
    return cmd


def valid(manifest_path, manifest, row, control):
    out = output(manifest, row, control)
    final = out / "cell_summary.json"
    if not final.is_file():
        return False
    value = load(final)
    return value.get("cell_id") == cell_id(row, control) and value.get("checkpoint_sha256") == row["checkpoint_sha256"] and value.get("test_evaluated") is False and value.get("world_size") == 4 and value.get("manifest_sha256") == sha(manifest_path)


def finalize(manifest_path, manifest, row, control):
    out = output(manifest, row, control)
    measurement = load(out / "measurement_summary.json")
    if measurement.get("checkpoint_sha256") != row["checkpoint_sha256"] or measurement.get("span_override") != control or measurement.get("split") != "validation" or measurement.get("test_evaluated") is not False or measurement.get("distributed", {}).get("world_size") != 4:
        raise RuntimeError("A3 measurement violates frozen contract")
    value = {"protocol": "a3_complete_measurement_cell_v2", "cell_id": cell_id(row, control), "family": row["family"], "control": control, "seed": row["seed"], "checkpoint_sha256": row["checkpoint_sha256"], "measurement_sha256": sha(out / "measurement_summary.json"), "manifest_sha256": sha(manifest_path), "graph_count": measurement["validation"]["graph_count"], "accuracy": measurement["validation"]["accuracy"], "macro_f1": measurement["validation"]["macro_f1"], "pointer": measurement["validation"].get("pointer"), "test_evaluated": False, "world_size": 4}
    (out / "cell_summary.json").write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--action", choices=("validate", "status", "launch"), default="status")
    parser.add_argument("--cell-id")
    args = parser.parse_args()
    manifest_path = args.manifest.resolve(); manifest = load(manifest_path); expanded = validate(manifest)
    if args.action == "validate":
        print(json.dumps({"cells": len(expanded)})); return 0
    if args.action == "status":
        complete = sum(valid(manifest_path, manifest, *item) for item in expanded)
        print(json.dumps({"complete": complete, "pending": len(expanded) - complete})); return 0
    match = [item for item in expanded if cell_id(*item) == args.cell_id]
    if len(match) != 1: raise RuntimeError("unknown A3 cell")
    row, control = match[0]
    if valid(manifest_path, manifest, row, control): raise RuntimeError("refusing duplicate A3 cell")
    out = output(manifest, row, control)
    if out.exists() and any(out.iterdir()): raise RuntimeError("refusing partial A3 output")
    out.mkdir(parents=True)
    subprocess.run(command(manifest, row, control, out), cwd=ROOT, env={**os.environ, "PYTHONUNBUFFERED": "1"}, check=True)
    print(json.dumps(finalize(manifest_path, manifest, row, control), sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
