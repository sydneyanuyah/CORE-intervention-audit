#!/usr/bin/env python3
"""Run corrected L3 manifests concurrently on assigned GPUs and aggregate them."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_manifest(root: Path, prefix: str, gpu_count: int) -> None:
    python = str(root / ".venv/bin/python")
    source = root / f"registry/{prefix}_l3_manifest.json"
    manifest = root / f"registry/{prefix}_l3_independent_v2_manifest.json"
    subprocess.run([
        python, "scripts/prepare_l3_independent.py", "--source", str(source), "--output", str(manifest)
    ], cwd=root, check=True)
    manifest_hash = sha(manifest)
    payload = json.loads(manifest.read_text())
    pending = []
    for cell in payload["cells"]:
        summary_path = root / cell["output"] / "run_summary.json"
        try:
            summary = json.loads(summary_path.read_text())
            valid = (
                summary["cell_id"] == cell["cell_id"]
                and summary["manifest_sha256"] == manifest_hash
                and summary["operator_class"] == payload["operator_contract"][cell["method"]]
                and summary["test_evaluated"] is False
            )
        except Exception:
            valid = False
        if not valid:
            pending.append(cell)
    free = list(range(gpu_count))
    active = {}
    logs = root / f"logs/{prefix}-l3-independent-v2"
    logs.mkdir(parents=True, exist_ok=True)
    while pending or active:
        while pending and free:
            gpu = free.pop(0)
            cell = pending.pop(0)
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            log = (logs / f"{cell['method']}-{cell['seed']}.log").open("a")
            command = [
                python, "src/train_l3_independent_cell.py", "--cell-id", cell["cell_id"],
                "--manifest", str(manifest), "--manifest-sha256", manifest_hash,
            ]
            active[gpu] = (cell, subprocess.Popen(command, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT), log)
        time.sleep(1)
        for gpu, (cell, process, log) in list(active.items()):
            status = process.poll()
            if status is None:
                continue
            log.close()
            del active[gpu]
            free.append(gpu)
            free.sort()
            if status:
                raise RuntimeError(f"failed corrected L3 cell {cell['cell_id']}")
    results = [json.loads((root / cell["output"] / "run_summary.json").read_text()) for cell in payload["cells"]]
    hashes = {}
    for result in results:
        key = (result["seed"], result["checkpoint_sha256"])
        hashes.setdefault(key, []).append(result["method"])
    collisions = [methods for methods in hashes.values() if len(methods) > 1]
    if collisions:
        raise RuntimeError(f"cross-method checkpoint collision: {collisions}")
    report = {
        "protocol": payload["protocol"], "completed_cells": len(results),
        "expected_cells": len(payload["cells"]), "manifest_sha256": manifest_hash,
        "operator_contract": payload["operator_contract"], "checkpoint_collisions": collisions,
        "results": results, "evaluation_split": "validation", "test_evaluated": False,
    }
    (root / f"reports/{prefix.upper()}_L3_INDEPENDENT_V2.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--gpu-count", type=int, required=True)
    parser.add_argument("prefixes", nargs="+")
    args = parser.parse_args()
    for prefix in args.prefixes:
        run_manifest(args.root.resolve(), prefix, args.gpu_count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
