#!/usr/bin/env python3
"""Keep small-model cluster's 32 GPUs occupied with registered one-GPU Phi-14B operator cells."""

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "registry/phi14b_cladder_extension_manifest.json"
AMENDMENT = ROOT / "registry/phi14b_cladder_cache_amendment.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def valid(cell, manifest_sha, amendment_sha):
    path = ROOT / cell["output"] / "run_summary.json"
    try: row = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError): return False
    return row.get("cell_id") == cell["cell_id"] and row.get("manifest_sha256") == manifest_sha and row.get("amendment_sha256") == amendment_sha and row.get("gpu_count") == 1 and row.get("test_evaluated") is False


def main() -> int:
    manifest = json.loads(MANIFEST.read_text()); manifest_sha = sha(MANIFEST); amendment_sha = sha(AMENDMENT)
    pending = [cell for cell in manifest["cells"] if not valid(cell, manifest_sha, amendment_sha)]
    logs = ROOT / "logs/phi14b-f2"; logs.mkdir(parents=True, exist_ok=True)
    while pending:
        wave, pending = pending[:32], pending[32:]
        active = []
        for gpu, cell in enumerate(wave):
            output = ROOT / cell["output"]
            if output.exists() and not valid(cell, manifest_sha, amendment_sha):
                raise RuntimeError(f"refusing partial output: {output}")
            stream = (logs / f"{cell['method']}-{cell['seed']}.log").open("w")
            env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            command = [str(ROOT / ".venv/bin/python"), str(ROOT / "src/train_phi14b_f2.py"), "--cell-id", cell["cell_id"], "--manifest", str(MANIFEST), "--manifest-sha256", manifest_sha, "--amendment", str(AMENDMENT), "--amendment-sha256", amendment_sha]
            active.append((cell, subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT), stream))
        while active:
            time.sleep(3)
            for item in list(active):
                cell, process, stream = item
                code = process.poll()
                if code is None: continue
                stream.close(); active.remove(item)
                if code or not valid(cell, manifest_sha, amendment_sha):
                    for _, other, other_stream in active: other.terminate(); other_stream.close()
                    raise RuntimeError(f"failed Phi-14B cell: {cell['cell_id']} exit={code}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

