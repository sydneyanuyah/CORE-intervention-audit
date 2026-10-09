#!/usr/bin/env python3
"""Continuously execute the repaired L1 learned cells on eight four-GPU groups."""

from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GPU_GROUPS = tuple(",".join(str(i) for i in range(start, start + 4)) for start in range(0, 32, 4))
PROFILES = {"core_base_1law", "core_i_3law", "core_full_4law"}
OUTPUT_TEMPLATE = "outputs/l1-repaired/cells/{family}/{method}/seed-{seed}"
READER_TEMPLATE = "outputs/a1-repaired/confirmatory/{family}/t2b_encode_only/seed-{seed}/best.pt"


def output_for(cell: dict) -> Path:
    return ROOT / OUTPUT_TEMPLATE.format(**cell)


def complete(cell: dict) -> bool:
    directory = output_for(cell)
    summary_path = directory / "run_summary.json"
    checkpoint_path = directory / "best.pt"
    if not summary_path.is_file() or not checkpoint_path.is_file():
        return False
    try:
        summary = json.loads(summary_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    config = summary.get("configuration", {})
    l1 = summary.get("l1", {})
    return (
        summary.get("protocol") == "l1_law_retention_v1"
        and summary.get("test_evaluated") is False
        and summary.get("gpu_policy") == {"model_size": "base", "world_size": 4}
        and int(config.get("seed", -1)) == int(cell["seed"])
        and int(config.get("split_layer", -1)) == 4
        and l1.get("law_profile") == cell["method"]
        and l1.get("reader_frozen") is True
    )


def command(cells_path: Path, cell: dict, gpus: str) -> list[str]:
    return [
        str(ROOT / ".venv/bin/python"), str(ROOT / "src/launch_l1_cell.py"),
        "--cells", str(cells_path), "--workdir", str(ROOT),
        "--cell-id", cell["cell_id"], "--gpus", gpus,
        "--output-template", OUTPUT_TEMPLATE,
        "--reader-template", READER_TEMPLATE,
        "--paraphrase-catalog", "data/paraphrases/structured_catalog_repaired.json",
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, default=ROOT / "logs/l1-repaired")
    args = parser.parse_args()
    cells_path = args.cells.resolve()
    registry = json.loads(cells_path.read_text())
    cells = [c for c in registry["cells"] if c["method"] in PROFILES]
    if len(cells) != 120 or len({c["cell_id"] for c in cells}) != 120:
        raise RuntimeError("repaired L1 requires exactly 120 unique learned cells")
    for cell in cells:
        reader = ROOT / READER_TEMPLATE.format(**cell)
        if not reader.is_file():
            raise FileNotFoundError(f"missing repaired A1 reader: {reader}")
    args.log_dir.mkdir(parents=True, exist_ok=True)
    lock = (args.log_dir / "dispatcher.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    active: dict[str, tuple[subprocess.Popen, object, dict]] = {}
    while True:
        for gpus, (process, stream, cell) in list(active.items()):
            status = process.poll()
            if status is None:
                continue
            stream.close()
            del active[gpus]
            if status != 0 or not complete(cell):
                raise RuntimeError(f"L1 cell failed validation: {cell['cell_id']} (exit {status})")
            last = output_for(cell) / "last.pt"
            if last.exists():
                last.unlink()
        pending = [c for c in cells if not complete(c) and c["cell_id"] not in {x[2]["cell_id"] for x in active.values()}]
        for gpus in GPU_GROUPS:
            if gpus in active or not pending:
                continue
            cell = pending.pop(0)
            log = args.log_dir / (cell["cell_id"].replace(":", "_") + ".log")
            stream = log.open("ab", buffering=0)
            process = subprocess.Popen(command(cells_path, cell, gpus), cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            active[gpus] = (process, stream, cell)
        status_payload = {
            "registered_learned": len(cells),
            "complete": sum(complete(c) for c in cells),
            "active": {g: x[2]["cell_id"] for g, x in active.items()},
            "remaining": sum(not complete(c) for c in cells),
            "test_evaluated": False,
        }
        (args.log_dir / "status.json").write_text(json.dumps(status_payload, indent=2, sort_keys=True) + "\n")
        if not active and not pending:
            return 0
        time.sleep(5)


if __name__ == "__main__":
    raise SystemExit(main())
