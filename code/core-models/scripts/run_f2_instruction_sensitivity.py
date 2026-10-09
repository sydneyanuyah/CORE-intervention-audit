#!/usr/bin/env python3
"""Fill eight four-GPU groups with registered F2 instruction-sensitivity cells."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "registry/f2_cladder_instruction_sensitivity_manifest.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def valid(cell: dict, manifest_sha: str) -> bool:
    path = ROOT / cell["output"] / "sensitivity_summary.json"
    try:
        row = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        row.get("cell_id") == cell["cell_id"]
        and row.get("method") == cell["method"]
        and row.get("reader_sha256") == cell["reader_sha256"]
        and row.get("operator_sha256") == cell["operator_sha256"]
        and row.get("manifest_sha256") == manifest_sha
        and row.get("world_size") == 4
        and row.get("test_evaluated") is False
    )


def main() -> int:
    manifest = json.loads(MANIFEST.read_text())
    manifest_sha = sha256(MANIFEST)
    pending = [cell for cell in manifest["cells"] if not valid(cell, manifest_sha)]
    log_dir = ROOT / "logs/f2-instruction-sensitivity"
    log_dir.mkdir(parents=True, exist_ok=True)
    groups = [f"{index},{index+1},{index+2},{index+3}" for index in range(0, 32, 4)]
    while pending:
        wave, pending = pending[:8], pending[8:]
        active = []
        for cell, devices in zip(wave, groups):
            output = ROOT / cell["output"]
            if output.exists() and not valid(cell, manifest_sha):
                raise RuntimeError(f"refusing partial output: {output}")
            log = (log_dir / f"{cell['method']}-{cell['seed']}.log").open("w")
            env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = devices
            command = [
                str(ROOT / ".venv/bin/torchrun"), "--standalone", "--nproc_per_node=4",
                str(ROOT / "src/evaluate_f2_instruction_sensitivity.py"),
                "--cell-id", cell["cell_id"], "--manifest", str(MANIFEST),
                "--manifest-sha256", manifest_sha, "--output", str(output),
            ]
            active.append((cell, subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT), log))
        failed = []
        while active:
            time.sleep(3)
            for item in list(active):
                cell, process, log = item
                code = process.poll()
                if code is None:
                    continue
                log.close(); active.remove(item)
                if code or not valid(cell, manifest_sha):
                    failed.append((cell["cell_id"], code))
            if failed:
                for _, process, log in active:
                    process.terminate(); log.close()
                raise RuntimeError(f"sensitivity cells failed: {failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
