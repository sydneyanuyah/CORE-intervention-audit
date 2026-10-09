#!/usr/bin/env python3
"""Keep eight disjoint four-GPU groups filled with the 40 registered L3 cells."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GROUPS = tuple(",".join(str(index) for index in range(start, start + 4)) for start in range(0, 32, 4))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def output(cell: dict) -> Path:
    return ROOT / "outputs" / "l3" / cell["method"] / f"seed-{cell['seed']}"


def complete(cell: dict, manifest_sha: str, cells_sha: str) -> bool:
    summary_path = output(cell) / "run_summary.json"
    if not summary_path.is_file(): return False
    try: summary = json.loads(summary_path.read_text())
    except (OSError, json.JSONDecodeError): return False
    valid = (
        summary.get("protocol") == "l3_eight_method_law_v1"
        and summary.get("cell_id") == cell["cell_id"]
        and summary.get("method") == cell["method"]
        and int(summary.get("seed", -1)) == int(cell["seed"])
        and int(summary.get("graph_seed", -1)) == int(cell["graph_seed"])
        and summary.get("world_size") == 4 and summary.get("test_evaluated") is False
        and summary.get("manifest_sha256") == manifest_sha and summary.get("cells_sha256") == cells_sha
        and summary.get("reader_sha256") == cell["reader_sha256"]
        and isinstance(summary.get("random_init"), dict) and isinstance(summary.get("trained"), dict)
    )
    if cell["execution"] == "fresh_operator_training": valid = valid and (output(cell) / "operator.pt").is_file()
    else: valid = valid and summary.get("source_checkpoint_sha256") == cell["source_checkpoint_sha256"]
    return bool(valid)


def command(cell: dict, manifest: Path, manifest_sha: str, cells: Path, cells_sha: str) -> list[str]:
    shared = [
        "--method", cell["method"], "--seed", str(cell["seed"]), "--graph-seed", str(cell["graph_seed"]),
        "--cell-id", cell["cell_id"], "--artifact-dir", cell["artifact_dir"],
        "--reader-checkpoint", str(ROOT / cell["reader_checkpoint"]), "--output-dir", str(output(cell)),
        "--core-components", "${PRIVATE_STORAGE_ROOT}/CORE_closing_controls/core_components.py",
        "--expected-core-sha256", "not-published",
        "--manifest", str(manifest), "--expected-manifest-sha256", manifest_sha,
        "--cells", str(cells), "--expected-cells-sha256", cells_sha,
    ]
    runner = "src/train_l3_operator.py" if cell["execution"] == "fresh_operator_training" else "src/evaluate_l3_reused.py"
    if cell["execution"] != "fresh_operator_training":
        shared.extend(["--source-checkpoint", str(ROOT / cell["source_checkpoint"])])
    return [str(ROOT / ".venv/bin/torchrun"), "--standalone", "--nproc_per_node=4", runner, *shared]


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--cells", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "registry/l3_manifest.json")
    parser.add_argument("--log-dir", type=Path, default=ROOT / "logs/l3")
    args = parser.parse_args(); args.cells = args.cells.resolve(); args.manifest = args.manifest.resolve()
    registry, manifest = json.loads(args.cells.read_text()), json.loads(args.manifest.read_text())
    manifest_sha, cells_sha = sha256(args.manifest), sha256(args.cells)
    if registry.get("manifest_sha256") != manifest_sha or registry.get("cell_count") != 40 or len(registry.get("cells", [])) != 40:
        raise ValueError("L3 registry/manifest binding mismatch")
    for path, expected in manifest["runner_sha256"].items():
        if sha256(ROOT / path) != expected: raise ValueError(f"L3 runner source changed: {path}")
    args.log_dir.mkdir(parents=True, exist_ok=True)
    lock = (args.log_dir / "dispatcher.lock").open("w"); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cells, active = registry["cells"], {}
    while True:
        for group, (process, stream, cell) in list(active.items()):
            code = process.poll()
            if code is None: continue
            stream.close(); del active[group]
            if code != 0 or not complete(cell, manifest_sha, cells_sha): raise RuntimeError(f"L3 cell failed: {cell['cell_id']} (exit {code})")
        active_ids = {value[2]["cell_id"] for value in active.values()}
        pending = [cell for cell in cells if not complete(cell, manifest_sha, cells_sha) and cell["cell_id"] not in active_ids]
        for group in GROUPS:
            if group in active or not pending: continue
            cell = pending.pop(0); log = args.log_dir / (cell["cell_id"].replace(":", "_") + ".log")
            stream = log.open("ab", buffering=0); env = {**os.environ, "CUDA_VISIBLE_DEVICES": group, "PYTHONUNBUFFERED": "1"}
            process = subprocess.Popen(command(cell, args.manifest, manifest_sha, args.cells, cells_sha), cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            active[group] = (process, stream, cell)
        status = {"registered": 40, "complete": sum(complete(cell, manifest_sha, cells_sha) for cell in cells), "remaining": sum(not complete(cell, manifest_sha, cells_sha) for cell in cells), "active": {group: value[2]["cell_id"] for group, value in active.items()}, "test_evaluated": False}
        (args.log_dir / "status.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
        if not active and not pending: return 0
        time.sleep(5)


if __name__ == "__main__":
    raise SystemExit(main())
