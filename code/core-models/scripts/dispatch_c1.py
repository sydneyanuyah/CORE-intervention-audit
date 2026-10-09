#!/usr/bin/env python3
"""Run C1 readers, a one-step smoke, then all 60 fixed-O3 cells."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORE = Path("${PRIVATE_STORAGE_ROOT}/CORE_closing_controls/core_components.py")
GROUPS = tuple(",".join(str(i) for i in range(start, start + 4)) for start in range(0, 32, 4))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path):
    return json.loads(path.read_text())


def reader_dir(seed: int) -> Path:
    return ROOT / f"outputs/c1/readers/seed-{seed}"


def reader_ok(seed: int) -> bool:
    summary, checkpoint = reader_dir(seed) / "run_summary.json", reader_dir(seed) / "reader.pt"
    if not summary.is_file() or not checkpoint.is_file(): return False
    try: row = load(summary)
    except (OSError, json.JSONDecodeError): return False
    return row.get("protocol") == "f3_reader_v1" and row.get("seed") == seed and row.get("graph_seed") == seed + 2480 and row.get("world_size") == 4 and row.get("test_evaluated") is False


def reader_command(seed: int) -> list[str]:
    graph = seed + 2480
    return [str(ROOT / ".venv/bin/torchrun"), "--standalone", "--nproc_per_node=4", str(ROOT / "src/train_f1.py"),
            "--experiment", "f3", "--phase", "reader", "--seed", str(seed), "--graph-seed", str(graph),
            "--condition", "real", "--model-size", "base", "--artifact-dir", f"${PRIVATE_STORAGE_ROOT}/CORE_f3/runs/graph_{graph}",
            "--output-dir", str(reader_dir(seed)), "--core-components", str(CORE),
            "--expected-core-sha256", "not-published"]


def freeze_cells(manifest: Path) -> Path:
    target = ROOT / "registry/c1_cells.json"
    cells = []
    for seed in range(601, 621):
        graph = seed + 2480
        reader = reader_dir(seed) / "reader.pt"
        if not reader_ok(seed): raise RuntimeError(f"invalid C1 reader {seed}")
        for condition in ("real", "noncausal_placebo", "shuffled"):
            records = ROOT / f"data/experiments/c1/{condition}/graph_{graph}.jsonl"
            if not records.is_file(): raise FileNotFoundError(records)
            cells.append({"cell_id": f"c1:{condition}:{seed}", "condition": condition, "method": "o3", "seed": seed,
                          "graph_seed": graph, "world_size": 4, "records": str(records.relative_to(ROOT)),
                          "records_sha256": sha(records), "reader_checkpoint": str(reader.relative_to(ROOT)),
                          "reader_sha256": sha(reader), "artifact_dir": f"${PRIVATE_STORAGE_ROOT}/CORE_f3/runs/graph_{graph}",
                          "test_evaluated": False})
    payload = {"protocol": "c1_fixed_o3_cells_v1", "manifest_sha256": sha(manifest), "cell_count": 60,
               "runner_sha256": sha(ROOT / "src/train_c1.py"), "cells": cells, "test_evaluated": False}
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if target.exists() and target.read_text() != rendered: raise RuntimeError("refusing to mutate frozen C1 cell registry")
    target.write_text(rendered)
    return target


def output(cell: dict) -> Path:
    return ROOT / f"outputs/c1/{cell['condition']}/seed-{cell['seed']}"


def cell_ok(cell: dict, manifest_sha: str, cells_sha: str) -> bool:
    summary, checkpoint = output(cell) / "run_summary.json", output(cell) / "operator.pt"
    if not summary.is_file() or not checkpoint.is_file(): return False
    try: row = load(summary)
    except (OSError, json.JSONDecodeError): return False
    return row.get("protocol") == "c1_fixed_o3_closing_controls_v1" and row.get("cell_id") == cell["cell_id"] and row.get("condition") == cell["condition"] and row.get("seed") == cell["seed"] and row.get("graph_seed") == cell["graph_seed"] and row.get("world_size") == 4 and row.get("reader_sha256") == cell["reader_sha256"] and row.get("records_sha256") == cell["records_sha256"] and row.get("manifest_sha256") == manifest_sha and row.get("cells_sha256") == cells_sha and isinstance(row.get("composed", {}).get("two_edit_balanced"), (int, float)) and row.get("test_evaluated") is False


def cell_command(cell: dict, manifest: Path, cells: Path, smoke: bool = False) -> list[str]:
    command = [str(ROOT / ".venv/bin/torchrun"), "--standalone", "--nproc_per_node=4", str(ROOT / "src/train_c1.py"),
               "--cell-id", cell["cell_id"], "--condition", cell["condition"], "--seed", str(cell["seed"]),
               "--graph-seed", str(cell["graph_seed"]), "--artifact-dir", cell["artifact_dir"],
               "--records", str(ROOT / cell["records"]), "--reader-checkpoint", str(ROOT / cell["reader_checkpoint"]),
               "--output-dir", str(ROOT / (f"outputs/c1-smoke/{cell['condition']}/seed-{cell['seed']}" if smoke else f"outputs/c1/{cell['condition']}/seed-{cell['seed']}")),
               "--core-components", str(CORE), "--expected-core-sha256", "not-published",
               "--manifest", str(manifest), "--expected-manifest-sha256", sha(manifest), "--cells", str(cells), "--expected-cells-sha256", sha(cells)]
    if smoke: command += ["--operator-steps", "1", "--smoke"]
    return command


def run_queue(tasks, command_for, valid, log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    active = {}
    def task_id(task) -> str:
        return task["cell_id"] if isinstance(task, dict) else str(task)
    while True:
        for group, (process, stream, task) in list(active.items()):
            code = process.poll()
            if code is None: continue
            stream.close(); del active[group]
            if code != 0 or not valid(task): raise RuntimeError(f"C1 task failed: {task} exit={code}")
        active_ids = {task_id(item[2]) for item in active.values()}
        pending = [task for task in tasks if not valid(task) and task_id(task) not in active_ids]
        for group in GROUPS:
            if group in active or not pending: continue
            task = pending.pop(0); log = log_dir / (task_id(task).replace(":", "_") + ".log")
            stream = log.open("ab", buffering=0); env = {**os.environ, "CUDA_VISIBLE_DEVICES": group, "PYTHONUNBUFFERED": "1"}
            process = subprocess.Popen(command_for(task), cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            active[group] = (process, stream, task)
        (log_dir / "status.json").write_text(json.dumps({"registered": len(tasks), "complete": sum(valid(t) for t in tasks), "active": {g: str(x[2]) for g, x in active.items()}, "test_evaluated": False}, indent=2, sort_keys=True) + "\n")
        if not active and not pending: return
        time.sleep(5)


def main() -> int:
    manifest = ROOT / "registry/c1_manifest.json"; log_dir = ROOT / "logs/c1"; log_dir.mkdir(parents=True, exist_ok=True)
    lock = (log_dir / "dispatcher.lock").open("w"); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if sha(ROOT / "data/experiments/manifests/c1_controls_summary.json") != load(manifest)["data_summary_sha256"]: raise RuntimeError("C1 data summary changed")
    if sha(CORE) != load(manifest)["canonical_core_sha256"]: raise RuntimeError("C1 core changed")
    seeds = list(range(601, 621)); run_queue(seeds, reader_command, reader_ok, log_dir / "readers")
    cells_path = freeze_cells(manifest); registry = load(cells_path); cells = registry["cells"]
    smoke_dir = ROOT / "outputs/c1-smoke/real/seed-601"; smoke_summary = smoke_dir / "run_summary.json"
    if not smoke_summary.is_file():
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": GROUPS[0], "PYTHONUNBUFFERED": "1"}
        subprocess.run(cell_command(cells[0], manifest, cells_path, True), cwd=ROOT, env=env, check=True,
                       stdout=(log_dir / "smoke.log").open("ab"), stderr=subprocess.STDOUT)
    smoke = load(smoke_summary)
    if smoke.get("protocol") != "c1_smoke_v1" or smoke.get("world_size") != 4 or smoke.get("test_evaluated") is not False: raise RuntimeError("C1 smoke failed validation")
    run_queue(cells, lambda c: cell_command(c, manifest, cells_path), lambda c: cell_ok(c, sha(manifest), sha(cells_path)), log_dir / "production")
    return 0


if __name__ == "__main__": raise SystemExit(main())
