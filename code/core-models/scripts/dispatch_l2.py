#!/usr/bin/env python3
"""Complete all registered L2 learned cells and deterministic floors."""

from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GROUPS = tuple(",".join(str(i) for i in range(s, s + 4)) for s in range(0, 32, 4))
METHODS = {"full", "drop_identity", "drop_idempotence", "drop_commutation", "drop_selective_invariance"}
FLOORS = {"do_nothing", "do_everything", "random_init"}
OUTPUT = "outputs/l2/cells/{family}/{method}/seed-{seed}"
READER = "outputs/a1-repaired/confirmatory/{family}/t2b_encode_only/seed-{seed}/best.pt"


def directory(cell: dict) -> Path:
    return ROOT / OUTPUT.format(**cell)


def complete(cell: dict) -> bool:
    summary_path = directory(cell) / "run_summary.json"
    checkpoint = directory(cell) / "best.pt"
    if not summary_path.is_file():
        return False
    try:
        summary = json.loads(summary_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if cell["method"] in FLOORS:
        return (
            summary.get("protocol") == "l2_dropped_law_v1"
            and summary.get("cell_id") == cell["cell_id"]
            and summary.get("execution") == "deterministic_symbolic_validation_floor_v1"
            and summary.get("world_size") == 4
            and summary.get("test_evaluated") is False
            and not checkpoint.exists()
        )
    if not checkpoint.is_file():
        return False
    config, law = summary.get("configuration", {}), summary.get("l1", {})
    return (
        summary.get("test_evaluated") is False
        and summary.get("gpu_policy") == {"model_size": "base", "world_size": 4}
        and int(config.get("seed", -1)) == int(cell["seed"])
        and int(config.get("split_layer", -1)) == 4
        and law.get("law_profile") == cell["method"]
        and law.get("reader_frozen") is True
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, default=ROOT / "logs/l2")
    args = parser.parse_args()
    cells_path = args.cells.resolve()
    registry = json.loads(cells_path.read_text())
    cells = [c for c in registry["cells"] if c["method"] in METHODS | FLOORS]
    if len(cells) != 320:
        raise RuntimeError("L2 requires exactly 320 registered cells")
    learned = [c for c in cells if c["method"] in METHODS]
    for cell in learned:
        if not (ROOT / READER.format(**cell)).is_file():
            raise FileNotFoundError(f"missing L2 initialization for {cell['cell_id']}")
    subprocess.run([
        str(ROOT / ".venv/bin/python"), str(ROOT / "src/materialize_l2_floors.py"),
        "--cells", str(cells_path), "--workdir", str(ROOT),
    ], cwd=ROOT, check=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    lock = (args.log_dir / "dispatcher.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    active: dict[str, tuple[subprocess.Popen, object, dict]] = {}
    while True:
        for group, (process, stream, cell) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            stream.close(); del active[group]
            if code != 0 or not complete(cell):
                raise RuntimeError(f"L2 cell failed: {cell['cell_id']} (exit {code})")
            last = directory(cell) / "last.pt"
            if last.exists(): last.unlink()
        active_ids = {item[2]["cell_id"] for item in active.values()}
        pending = [c for c in learned if not complete(c) and c["cell_id"] not in active_ids]
        for group in GROUPS:
            if group in active or not pending: continue
            cell = pending.pop(0)
            log = args.log_dir / (cell["cell_id"].replace(":", "_") + ".log")
            stream = log.open("ab", buffering=0)
            command = [
                str(ROOT / ".venv/bin/python"), str(ROOT / "src/launch_l1_cell.py"),
                "--cells", str(cells_path), "--workdir", str(ROOT),
                "--cell-id", cell["cell_id"], "--gpus", group,
                "--output-template", OUTPUT, "--reader-template", READER,
                "--paraphrase-catalog", "data/paraphrases/structured_catalog_repaired.json",
            ]
            process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            active[group] = (process, stream, cell)
        payload = {"registered": 320, "registered_learned": 200, "registered_floors": 120, "complete": sum(complete(c) for c in cells), "remaining": sum(not complete(c) for c in cells), "active": {g: x[2]["cell_id"] for g, x in active.items()}, "test_evaluated": False}
        (args.log_dir / "status.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        if not active and not pending: return 0
        time.sleep(5)


if __name__ == "__main__":
    raise SystemExit(main())
