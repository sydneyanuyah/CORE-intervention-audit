#!/usr/bin/env python3
"""Dispatch every registered A2 or A3 cell over eight four-GPU slots."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).parents[1].resolve()
PYTHON = str(ROOT / ".venv/bin/python")
GROUPS = [list(range(start, start + 4)) for start in range(0, 32, 4)]


def load(path): return json.loads(Path(path).read_text())


def tasks(experiment, manifest_path):
    manifest = load(manifest_path); catalog = load(ROOT / manifest["source"]["catalog"])
    if experiment == "a2":
        return [(f'{row["family"]}:{control}:{row["seed"]}', ROOT / manifest["paths"]["output_template"].format(family=row["family"], control=control, seed=row["seed"])) for row in catalog["cells"] for control in manifest["controls"]]
    return [(f'{row["family"]}:{control}:{row["seed"]}', ROOT / manifest["paths"]["output_template"].format(family=row["family"], control=control, seed=row["seed"])) for row in catalog["cells"] for control in manifest["controls"]]


def valid(task): return (task[1] / "cell_summary.json").is_file()


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--experiment", choices=("a2", "a3"), required=True); parser.add_argument("--manifest", type=Path, required=True); parser.add_argument("--log-dir", type=Path, required=True); args = parser.parse_args()
    manifest = args.manifest.resolve(); log_dir = args.log_dir.resolve(); log_dir.mkdir(parents=True, exist_ok=True); registered = tasks(args.experiment, manifest)
    runner = "scripts/run_a2_measurement.py" if args.experiment == "a2" else "scripts/run_a3_measurement.py"
    subprocess.run([PYTHON, runner, "--manifest", str(manifest), "--action", "validate"], cwd=ROOT, check=True)
    active = {}
    try:
        while True:
            active_ids = {entry[0][0] for entry in active.values()}
            pending = [task for task in registered if not valid(task) and task[0] not in active_ids]
            if not pending and not active: break
            for slot, gpus in enumerate(GROUPS):
                if slot in active or not pending: continue
                task = pending.pop(0); cell, out = task
                if out.exists() and any(out.iterdir()): raise RuntimeError(f"refusing partial output: {out}")
                stream = (log_dir / (cell.replace(":", "-") + ".log")).open("w")
                env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
                proc = subprocess.Popen([PYTHON, runner, "--manifest", str(manifest), "--action", "launch", "--cell-id", cell], cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
                active[slot] = (task, proc, stream, gpus)
            status = {"state": "running", "updated_epoch": time.time(), "registered_cells": len(registered), "complete_cells": sum(valid(x) for x in registered), "active": {str(slot): {"cell_id": item[0][0], "pid": proc.pid, "gpus": gpus} for slot, (item, proc, _, gpus) in active.items()}}
            (log_dir / "dispatcher-status.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
            time.sleep(5)
            for slot, (task, proc, stream, _) in list(active.items()):
                code = proc.poll()
                if code is None: continue
                stream.close(); del active[slot]
                if code != 0 or not valid(task): raise RuntimeError(f"{args.experiment.upper()} cell failed: {task[0]}, exit={code}")
        (log_dir / "dispatcher-status.json").write_text(json.dumps({"state": "complete", "updated_epoch": time.time(), "registered_cells": len(registered), "complete_cells": len(registered), "active": {}}, indent=2, sort_keys=True) + "\n")
        return 0
    except Exception as exc:
        for _, proc, stream, _ in active.values(): proc.terminate(); stream.close()
        (log_dir / "dispatcher-status.json").write_text(json.dumps({"state": "failed", "updated_epoch": time.time(), "error": repr(exc)}, indent=2) + "\n")
        raise


if __name__ == "__main__": raise SystemExit(main())
