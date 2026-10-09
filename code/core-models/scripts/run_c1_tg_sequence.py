#!/usr/bin/env python3
"""Persistent fail-closed C1..C4 -> T1..T5 -> G1..G4 queue."""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "registry/c1_tg_sequence.json"
STATUS = ROOT / "logs/c1-tg-sequence/status.json"


def write(stage: str, state: str, detail: str) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(
        json.dumps(
            {
                "stage": stage,
                "state": state,
                "detail": detail,
                "updated_epoch": time.time(),
                "test_evaluated": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def c1_counts() -> tuple[int, int, int]:
    try:
        data = json.loads((ROOT / "logs/c1/production/status.json").read_text())
    except (OSError, json.JSONDecodeError):
        return 0, 0, 0
    completed = data.get("complete", data.get("completed", []))
    active = data.get("active", {})
    failed = data.get("failed", {})
    completed_count = completed if isinstance(completed, int) else len(completed)
    failed_count = failed if isinstance(failed, int) else len(failed)
    return completed_count, len(active), failed_count


def wait_for_c_family() -> None:
    while True:
        complete, active, failed = c1_counts()
        if failed:
            write("C1", "failed", f"C1 has {failed} failed cells; downstream queue stopped")
            raise SystemExit(1)
        try:
            family = json.loads((ROOT / "logs/c-family-sequence/status.json").read_text())
        except (OSError, json.JSONDecodeError):
            family = {}
        stage = family.get("stage", "C1")
        state = family.get("state", "waiting")
        detail = family.get("detail", f"C1 {complete}/60 complete; {active} active")
        write(stage, state, detail)
        if state == "failed":
            raise SystemExit(1)
        if stage == "sequence" and state == "complete":
            return
        time.sleep(30)


def wait_for_launcher(stage: str, launcher: Path) -> None:
    while not launcher.is_file():
        write(stage, "waiting_for_launcher", str(launcher.relative_to(ROOT)))
        time.sleep(60)


def run_stage(stage: str, launcher: Path) -> None:
    wait_for_launcher(stage, launcher)
    write(stage, "running", str(launcher.relative_to(ROOT)))
    proc = subprocess.run([str(ROOT / ".venv/bin/python"), str(launcher)], cwd=ROOT)
    if proc.returncode != 0:
        write(stage, "failed", f"launcher exited {proc.returncode}; downstream queue stopped")
        raise SystemExit(proc.returncode)
    write(stage, "complete", "launcher exited successfully")


def main() -> int:
    spec = json.loads(QUEUE.read_text())
    expected = ["C1", "C2", "C3", "C4", "T1", "T2", "T3", "T4", "T5", "G1", "G2", "G3", "G4"]
    if spec["sequence"] != expected:
        raise ValueError("unexpected queue order")
    wait_for_c_family()
    for stage in spec["sequence"][4:]:
        run_stage(stage, ROOT / spec["launchers"][stage])
    write("sequence", "complete", "C1-C4, T1-T5, and G1-G4 complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
