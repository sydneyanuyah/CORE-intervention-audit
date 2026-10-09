#!/usr/bin/env python3
"""Run all registered T3 inverted controls in four-GPU cells."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "registry/t3_inverted_control_manifest.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def valid(cell: dict) -> bool:
    summary = ROOT / cell["output"] / "measurement_summary.json"
    predictions = ROOT / cell["output"] / "predictions.jsonl"
    try:
        data = json.loads(summary.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        data.get("test_evaluated") is False
        and data.get("checkpoint_sha256") == cell["checkpoint_sha256"]
        and data.get("distributed", {}).get("world_size") == 4
        and predictions.is_file()
    )


def main() -> int:
    manifest = json.loads(MANIFEST.read_text())
    manifest_sha = sha(MANIFEST)
    pending = [cell for cell in manifest["cells"] if not valid(cell)]
    groups = [f"{i},{i+1},{i+2},{i+3}" for i in range(0, 32, 4)]
    log_dir = ROOT / "logs/t3-positive-control"
    log_dir.mkdir(parents=True, exist_ok=True)
    while pending:
        wave, pending = pending[:8], pending[8:]
        processes = []
        for cell, gpus in zip(wave, groups):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpus
            log = (log_dir / f"inverted-seed-{cell['seed']}.log").open("a")
            command = [
                str(ROOT / ".venv/bin/torchrun"), "--standalone", "--nproc_per_node=4",
                str(ROOT / "src/evaluate_t3_inverted_control.py"),
                "--cell-id", cell["cell_id"], "--manifest", str(MANIFEST),
                "--manifest-sha256", manifest_sha, "--output", str(ROOT / cell["output"]),
            ]
            processes.append((cell, subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT), log))
        failed = []
        for cell, process, log in processes:
            code = process.wait()
            log.close()
            if code or not valid(cell):
                failed.append((cell["cell_id"], code))
        if failed:
            raise RuntimeError(f"T3 positive-control cells failed: {failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
