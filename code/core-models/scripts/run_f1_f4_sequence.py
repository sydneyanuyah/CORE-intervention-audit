#!/usr/bin/env python3
"""Fail-closed sequential supervisor: F1 -> F2 -> F3 -> F4."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).parents[1].resolve()
STATUS = ROOT / "logs/f1-f4-sequence/status.json"


def write(stage: str, state: str, detail: str = "") -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps({"stage": stage, "state": state, "detail": detail, "updated_epoch": time.time()}, indent=2) + "\n")


def run(stage: str, command: list[str]) -> None:
    write(stage, "running", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)
    write(stage, "complete")


def validate_f1() -> None:
    summaries = list((ROOT / "outputs/f1").glob("**/run_summary.json"))
    if len(summaries) != 60:
        raise RuntimeError(f"F1 expected 60 reader/cell summaries, found {len(summaries)}")
    rows = [json.loads(path.read_text()) for path in summaries]
    if any(row.get("test_evaluated") is not False for row in rows):
        raise RuntimeError("F1 contains a non-validation-only summary")
    write("F1", "complete", "validated 60 existing registered reader/cell summaries")


def main() -> int:
    python = str(ROOT / ".venv/bin/python")
    try:
        validate_f1()
        run("F2-CCR.GB-operators", [python, "scripts/dispatch_f2_ccrgb_v2.py", "--phase", "operators"])
        run("F2-CCR.GB-baselines", [python, "scripts/dispatch_f2_ccrgb_v2.py", "--phase", "baselines"])
        for name in ("dispatch_f2_cladder.py", "dispatch_f2_wiqa.py", "dispatch_f2_com2_v2.py"):
            run(f"F2-{name}", [python, f"scripts/{name}"])
        run("F3", [python, "scripts/dispatch_f3_f4.py", "--experiment", "f3", "--output-variant", "sequential-v2"])
        run("F4", [python, "scripts/dispatch_f3_f4.py", "--experiment", "f4", "--output-variant", "sequential-v2"])
    except Exception as exc:
        write("sequence", "failed", repr(exc))
        raise
    write("sequence", "complete", "F1, F2, F3, and F4 complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
