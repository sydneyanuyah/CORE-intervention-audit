#!/usr/bin/env python3
"""Run the immutable F3 and F4 queues with fixed four/eight-GPU slots."""

from __future__ import annotations

import hashlib
import argparse
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).parents[1].resolve()
CORE = Path("${PRIVATE_STORAGE_ROOT}/CORE_closing_controls/core_components.py")
CORE_SHA = "not-published"
F3_MANIFEST_SHA = "not-published"
F4_MANIFEST_SHA = "not-published"
F3_PROVENANCE_SHA = "not-published"
F4_PROVENANCE_SHA = "not-published"
OUTPUT_VARIANT = ""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class Task:
    experiment: str
    model_size: str
    phase: str
    seed: int
    graph_seed: int
    condition: str = "real"
    method: str | None = None

    @property
    def world_size(self) -> int:
        return 8 if self.model_size == "large" else 4

    @property
    def name(self) -> str:
        pieces = [self.experiment, self.model_size, self.phase, self.condition]
        if self.method: pieces.append(self.method)
        pieces.append(str(self.seed))
        return "-".join(pieces)

    @property
    def artifact(self) -> Path:
        return Path(f"${PRIVATE_STORAGE_ROOT}/CORE_{self.experiment}/runs/graph_{self.graph_seed}")

    @property
    def reader_output(self) -> Path:
        family = self.experiment if not OUTPUT_VARIANT else f"{self.experiment}-{OUTPUT_VARIANT}"
        if self.experiment == "f3": return ROOT / f"outputs/{family}/readers/seed-{self.seed}"
        return ROOT / f"outputs/{family}/{self.model_size}/readers/seed-{self.seed}"

    @property
    def output(self) -> Path:
        if self.phase == "reader": return self.reader_output
        family = self.experiment if not OUTPUT_VARIANT else f"{self.experiment}-{OUTPUT_VARIANT}"
        if self.experiment == "f3": return ROOT / f"outputs/{family}/{self.condition}/{self.method}/seed-{self.seed}"
        return ROOT / f"outputs/{family}/{self.model_size}/{self.method}/seed-{self.seed}"


def registered_tasks() -> list[Task]:
    tasks = []
    for seed in range(601, 621):
        graph = seed + 2480
        tasks.append(Task("f3", "base", "reader", seed, graph))
        for condition in ("real", "placebo"):
            for method in ("o2", "o3"):
                tasks.append(Task("f3", "base", "operator", seed, graph, condition, method))
    for model_size in ("base", "large"):
        for seed in range(701, 726):
            graph = seed + 2400
            tasks.append(Task("f4", model_size, "reader", seed, graph))
            for method in ("o2", "o3"):
                tasks.append(Task("f4", model_size, "operator", seed, graph, "real", method))
    return tasks


def valid(task: Task) -> bool:
    summary = task.output / "run_summary.json"
    if not summary.is_file(): return False
    try: row = json.loads(summary.read_text())
    except (OSError, json.JSONDecodeError): return False
    common = row.get("seed") == task.seed and row.get("graph_seed") == task.graph_seed and row.get("world_size") == task.world_size and row.get("model_size") == task.model_size and row.get("test_evaluated") is False
    if task.phase == "reader":
        return common and row.get("protocol") == f"{task.experiment}_reader_v1" and (task.output / "reader.pt").is_file()
    return common and row.get("protocol") == f"{task.experiment}_cell_v1" and row.get("method") == task.method and row.get("condition") == task.condition and isinstance(row.get("composed", {}).get("balanced_intervention_score"), (int, float)) and (task.output / "operator.pt").is_file()


def ready(task: Task) -> bool:
    return task.phase == "reader" or valid(Task(task.experiment, task.model_size, "reader", task.seed, task.graph_seed))


def command(task: Task) -> list[str]:
    cmd = [str(ROOT / ".venv/bin/torchrun"), "--standalone", f"--nproc_per_node={task.world_size}", str(ROOT / "src/train_f1.py"), "--experiment", task.experiment, "--phase", task.phase, "--seed", str(task.seed), "--graph-seed", str(task.graph_seed), "--condition", task.condition, "--model-size", task.model_size, "--artifact-dir", str(task.artifact), "--output-dir", str(task.output), "--core-components", str(CORE), "--expected-core-sha256", CORE_SHA]
    if task.phase == "operator": cmd.extend(["--method", str(task.method), "--reader-checkpoint", str(task.reader_output / "reader.pt")])
    return cmd


def verify_frozen_inputs() -> None:
    checks = [
        (ROOT / "registry/f3_manifest.json", F3_MANIFEST_SHA),
        (ROOT / "registry/f4_manifest.json", F4_MANIFEST_SHA),
        (Path("${PRIVATE_STORAGE_ROOT}/CORE_f3/runs/provenance.json"), F3_PROVENANCE_SHA),
        (Path("${PRIVATE_STORAGE_ROOT}/CORE_f4/runs/provenance.json"), F4_PROVENANCE_SHA),
        (CORE, CORE_SHA),
    ]
    for path, expected in checks:
        if sha256(path) != expected: raise RuntimeError(f"frozen input mismatch: {path}")


def write_status(state, tasks, active, error=None):
    path = ROOT / "logs/f3-f4/dispatcher-status.json"
    payload = {
        "state": state, "updated_epoch": time.time(), "registered_tasks": len(tasks),
        "complete_tasks": sum(valid(task) for task in tasks),
        "formal_cells_complete": sum(valid(task) for task in tasks if task.phase == "operator"),
        "formal_cells_registered": sum(task.phase == "operator" for task in tasks),
        "active": {slot: {"task": task.name, "pid": process.pid, "gpus": gpus} for slot, (task, process, _, gpus) in active.items()},
    }
    if error: payload["error"] = error
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=("f3", "f4", "both"), default="both")
    parser.add_argument("--output-variant", default="")
    args = parser.parse_args()
    global OUTPUT_VARIANT
    OUTPUT_VARIANT = args.output_variant
    verify_frozen_inputs(); tasks = registered_tasks()
    if args.experiment != "both":
        tasks = [task for task in tasks if task.experiment == args.experiment]
    log_name = f"{args.experiment}-{OUTPUT_VARIANT}" if OUTPUT_VARIANT else args.experiment
    logs = ROOT / f"logs/f3-f4/{log_name}"; logs.mkdir(parents=True, exist_ok=True)
    if args.experiment == "f3":
        slots = {f"base{group}": list(range(group * 4, group * 4 + 4)) for group in range(8)}
    else:
        slots = {
            "base0": [0,1,2,3], "base1": [4,5,6,7], "base2": [8,9,10,11], "base3": [12,13,14,15],
            "large0": [16,17,18,19,20,21,22,23], "large1": [24,25,26,27,28,29,30,31],
        }
    active = {}
    while True:
        pending = [task for task in tasks if not valid(task) and all(task != item[0] for item in active.values())]
        if not pending and not active: break
        for slot, gpus in slots.items():
            if slot in active: continue
            size = "large" if len(gpus) == 8 else "base"
            task = next((item for item in pending if item.model_size == size and ready(item)), None)
            if task is None: continue
            if task.output.exists() and not valid(task): raise RuntimeError(f"refusing partial output: {task.output}")
            stream = (logs / f"{task.name}.log").open("w")
            env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
            process = subprocess.Popen(command(task), cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            active[slot] = (task, process, stream, gpus); pending.remove(task)
        write_status("running", tasks, active); time.sleep(5)
        for slot, (task, process, stream, gpus) in list(active.items()):
            code = process.poll()
            if code is None: continue
            stream.close(); del active[slot]
            if code != 0 or not valid(task):
                error = f"task failed provenance validation: {task.name}, exit={code}"
                write_status("failed", tasks, active, error)
                for _, running, open_stream, _ in active.values(): running.terminate(); open_stream.close()
                raise RuntimeError(error)
    write_status("complete", tasks, {}); return 0


if __name__ == "__main__": raise SystemExit(main())
