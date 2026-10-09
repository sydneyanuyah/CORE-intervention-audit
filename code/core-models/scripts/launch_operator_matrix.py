#!/usr/bin/env python3
"""Run independent graph/seed jobs concurrently across the approved GPU set."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


MODELS = {
    "base": {"name": "google-bert/bert-base-uncased", "gpus": 4, "split_layer": 6, "batch": 16},
    "large": {"name": "google-bert/bert-large-uncased", "gpus": 8, "split_layer": 12, "batch": 8},
}


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def command(args: argparse.Namespace, run_id: int, output: Path) -> list[str]:
    spec = MODELS[args.model_size]
    compatibility_entry = Path(__file__).parents[1] / "src" / "core_bert" / "operator_compat.py"
    if args.phase == "operator-smoke":
        cmd = [args.python, str(compatibility_entry), str(Path(__file__).parents[1] / "src" / "operator_smoke.py"), "--model", spec["name"], "--split-layer", str(spec["split_layer"]), "--seed", str(run_id), "--output", str(output / "smoke.json")]
    elif args.phase == "foundation":
        cmd = [
            args.python, str(compatibility_entry), str(args.server_source / "CORE_balanced_multitoken_experiment" / "run_balanced_multitoken_experiment.py"),
            "--output-dir", str(output), "--model", spec["name"], "--seed", str(run_id),
            "--split-layer", str(spec["split_layer"]), "--reader-batch-size", str(spec["batch"]),
            "--eval-batch-size", str(spec["batch"]), "--reader-epochs", str(args.reader_epochs),
            "--operator-steps", str(args.operator_steps), "--operator-batch-size", str(spec["batch"]),
            "--interfaces", "world16", "--condition-limit", str(args.condition_limit),
        ]
    else:
        if not args.reader_init:
            raise ValueError("--reader-init is required for closing-controls")
        cmd = [
            args.python, str(compatibility_entry), str(args.server_source / "CORE_closing_controls" / "run_closing_controls.py"),
            "--output-dir", str(output), "--reader-init", str(args.reader_init), "--model", spec["name"],
            "--graph-seed", str(run_id), "--split-seed", str(run_id + 10_000),
            "--optimization-seed", str(run_id + 20_000), "--split-layer", str(spec["split_layer"]),
            "--reader-batch-size", str(spec["batch"]), "--eval-batch-size", str(spec["batch"]),
            "--reader-epochs", str(args.reader_epochs), "--operator-steps", str(args.operator_steps),
            "--operator-batch-size", str(spec["batch"]),
        ]
        if args.development_only: cmd.append("--development-only")
    return cmd


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("operator-smoke", "foundation", "closing-controls"), required=True)
    p.add_argument("--model-size", choices=tuple(MODELS), required=True)
    p.add_argument("--gpus", type=parse_ints, required=True)
    p.add_argument("--run-ids", type=parse_ints, required=True, help="Seeds or graph seeds")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--server-source", type=Path, default=Path("${PRIVATE_STORAGE_ROOT}"))
    p.add_argument("--reader-init", type=Path)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--reader-epochs", type=int, default=8)
    p.add_argument("--operator-steps", type=int, default=500)
    p.add_argument("--condition-limit", type=int, default=0)
    p.add_argument("--development-only", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    required = MODELS[args.model_size]["gpus"]
    if len(args.gpus) != required:
        raise SystemExit(f"{args.model_size} requires exactly {required} GPUs; received {len(args.gpus)}")
    if len(args.run_ids) < 1: raise SystemExit("at least one run ID is required")
    args.output_root.mkdir(parents=True, exist_ok=True)
    pending = list(args.run_ids); running: dict[int, tuple[subprocess.Popen[str], object, int, Path, float]] = {}; results = []
    while pending or running:
        for gpu in args.gpus:
            if not pending or gpu in running: continue
            run_id = pending.pop(0); output = args.output_root / f"{args.model_size}_{args.phase}_{run_id}"
            cmd = command(args, run_id, output)
            if args.dry_run:
                print(json.dumps({"gpu": gpu, "run_id": run_id, "command": cmd})); results.append({"gpu": gpu, "run_id": run_id, "dry_run": True}); continue
            output.mkdir(parents=True, exist_ok=True); log = (output / "console.log").open("w", encoding="utf-8")
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1"}
            process = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
            running[gpu] = (process, log, run_id, output, time.time()); print(f"started run={run_id} gpu={gpu} pid={process.pid}", flush=True)
        if args.dry_run: continue
        time.sleep(1)
        for gpu, (process, log, run_id, output, started) in list(running.items()):
            code = process.poll()
            if code is None: continue
            log.close(); result = {"gpu": gpu, "run_id": run_id, "returncode": code, "seconds": time.time() - started, "output": str(output)}; results.append(result); del running[gpu]
            print(json.dumps(result), flush=True)
    summary = {"phase": args.phase, "model_size": args.model_size, "gpu_count": len(args.gpus), "results": results}
    (args.output_root / "launcher_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0 if all(item.get("returncode", 0) == 0 for item in results) else 1


if __name__ == "__main__": raise SystemExit(main())
