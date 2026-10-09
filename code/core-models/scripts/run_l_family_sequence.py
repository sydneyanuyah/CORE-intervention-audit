#!/usr/bin/env python3
"""Persistent fail-closed handoff from repaired L1 through L2 and L3."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "logs/l-family-sequence/status.json"


def write(stage: str, state: str, detail: str) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps({
        "stage": stage, "state": state, "detail": detail,
        "updated_epoch": time.time(), "test_evaluated": False,
    }, indent=2, sort_keys=True) + "\n")


def wait_complete(path: Path, expected: int, stage: str) -> None:
    while True:
        try:
            status = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            status = {}
        complete = int(status.get("complete", 0))
        active = status.get("active", {})
        write(stage, "waiting", f"{complete}/{expected} registered cells complete")
        if complete == expected and not active:
            return
        time.sleep(30)


def run(command: list[str], stage: str) -> None:
    write(stage, "running", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    wait_complete(ROOT / "logs/l1-repaired/status.json", 120, "L1")
    run([
        str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/dispatch_l2.py"),
        "--cells", str(ROOT / "registry/l2_cells.json"),
        "--log-dir", str(ROOT / "logs/l2"),
    ], "L2")
    # L3 is deliberately resolved only after L2. Its dispatcher must validate
    # the frozen eight-method registry before this supervisor will execute it.
    l3_dispatcher = ROOT / "scripts/dispatch_l3.py"
    l3_cells = ROOT / "registry/l3_cells.json"
    while not (l3_dispatcher.is_file() and l3_cells.is_file()):
        write(
            "L3",
            "waiting_for_validated_runner",
            "L3 dispatcher/cell registry not yet installed; LoReFT, task-vector-add, and router implementations must be completed",
        )
        time.sleep(30)
    run([
        str(ROOT / ".venv/bin/python"), str(l3_dispatcher),
        "--cells", str(l3_cells), "--log-dir", str(ROOT / "logs/l3"),
    ], "L3")
    write("sequence", "complete", "L1, L2, and L3 execution complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
