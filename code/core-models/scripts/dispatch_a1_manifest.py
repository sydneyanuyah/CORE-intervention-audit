#!/usr/bin/env python3
"""Fill all eight four-GPU slots with cells from one immutable A1 manifest."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).parents[1].resolve()
GPU_GROUPS = [list(range(start, start + 4)) for start in range(0, 32, 4)]


def load_runner():
    spec = importlib.util.spec_from_file_location("a1_runner", ROOT / "scripts/run_a1_screening.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_status(path: Path, state: str, cells, runner, active, error: str | None = None) -> None:
    counts = {}
    for cell in cells:
        cell_state = runner.cell_status(cell)["state"]
        counts[cell_state] = counts.get(cell_state, 0) + 1
    payload = {
        "state": state,
        "updated_epoch": time.time(),
        "registered_cells": len(cells),
        "counts": counts,
        "active": {
            str(slot): {"cell_id": cell.cell_id, "pid": proc.pid, "gpus": gpus}
            for slot, (cell, proc, _, gpus) in active.items()
        },
    }
    if error:
        payload["error"] = error
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    log_dir = args.log_dir.resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    status_path = log_dir / "dispatcher-status.json"
    runner = load_runner()
    manifest = runner._load(manifest_path)
    cells = runner.validate_manifest(manifest, ROOT)
    active = {}
    try:
        while True:
            active_ids = {entry[0].cell_id for entry in active.values()}
            pending = [
                cell for cell in cells
                if runner.cell_status(cell)["state"] != "complete" and cell.cell_id not in active_ids
            ]
            if not pending and not active:
                break
            for slot, gpus in enumerate(GPU_GROUPS):
                if slot in active or not pending:
                    continue
                cell = pending.pop(0)
                state = runner.cell_status(cell)["state"]
                if state not in {"pending", "trained_pending_composed", "measured_pending_finalization"}:
                    raise RuntimeError(f"refusing non-resumable cell state {state}: {cell.cell_id}")
                if state != "pending":
                    command = [str(ROOT / ".venv/bin/python"), "scripts/run_a1_screening.py", "--manifest", str(manifest_path), "--action", "resume", "--cell-id", cell.cell_id]
                else:
                    if cell.output.exists() and any(cell.output.iterdir()):
                        raise RuntimeError(f"refusing partial output before launch: {cell.output}")
                    command = [str(ROOT / ".venv/bin/python"), "scripts/run_a1_screening.py", "--manifest", str(manifest_path), "--action", "launch", "--cell-id", cell.cell_id]
                stream = (log_dir / f"{cell.family}-{cell.arm}-{cell.seed}.log").open("w")
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
                proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
                active[slot] = (cell, proc, stream, gpus)
            write_status(status_path, "running", cells, runner, active)
            time.sleep(5)
            for slot, (cell, proc, stream, gpus) in list(active.items()):
                code = proc.poll()
                if code is None:
                    continue
                stream.close()
                del active[slot]
                if code != 0 or runner.cell_status(cell)["state"] != "complete":
                    raise RuntimeError(f"A1 cell failed provenance validation: {cell.cell_id}, exit={code}")
                # The selected checkpoint is best.pt.  Once provenance finalization
                # succeeds, last.pt is a redundant full model/optimizer copy and
                # must not consume the shared filesystem for hundreds of cells.
                redundant = cell.output / "last.pt"
                if redundant.is_file():
                    redundant.unlink()
        write_status(status_path, "complete", cells, runner, {})
        return 0
    except Exception as exc:
        for _, proc, stream, _ in active.values():
            proc.terminate()
            stream.close()
        write_status(status_path, "failed", cells, runner, {}, repr(exc))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
