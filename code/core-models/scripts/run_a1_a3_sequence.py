#!/usr/bin/env python3
"""Wait for F4, then execute repaired A1 and prepare its A2/A3 dependants."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).parents[1].resolve()
PYTHON = str(ROOT / ".venv/bin/python")
STATUS = ROOT / "logs/a1-a3-sequence/status.json"


def write(stage: str, state: str, detail: str = "") -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps({"stage": stage, "state": state, "detail": detail, "updated_epoch": time.time()}, indent=2) + "\n")


def run(stage: str, command: list[str]) -> None:
    write(stage, "running", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)
    write(stage, "complete")


def wait_for_f4() -> None:
    path = ROOT / "logs/f3-f4/dispatcher-status.json"
    write("F4", "waiting", "waiting for the already-running sequential-v2 F4 dispatcher")
    while True:
        try:
            value = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            time.sleep(5)
            continue
        if value.get("state") == "failed":
            raise RuntimeError(f"upstream F4 failed: {value.get('error')}")
        if value.get("state") == "complete" and value.get("formal_cells_complete") == 100 and value.get("complete_tasks") == 150:
            write("F4", "complete", "validated 150/150 tasks and 100/100 formal cells")
            return
        time.sleep(5)


def main() -> int:
    try:
        wait_for_f4()
        screening = ROOT / "registry/a1_repaired_screening_manifest.json"
        run("A1-screening", [PYTHON, "scripts/dispatch_a1_manifest.py", "--manifest", str(screening), "--log-dir", "logs/a1-repaired/screening"])
        run("A1-screening-report", [PYTHON, "scripts/report_a1_screening.py", "--manifest", str(screening), "--json-output", "reports/A1_REPAIRED_SCREENING.json", "--markdown-output", "reports/A1_REPAIRED_SCREENING.md"])
        run("A1-confirmatory-freeze", [PYTHON, "scripts/build_repaired_downstream_manifests.py", "confirmatory"])
        confirmatory = ROOT / "registry/a1_repaired_confirmatory_manifest.json"
        run("A1-confirmatory", [PYTHON, "scripts/dispatch_a1_manifest.py", "--manifest", str(confirmatory), "--log-dir", "logs/a1-repaired/confirmatory"])
        run("A1-confirmatory-report", [PYTHON, "scripts/report_a1_confirmatory.py", "--manifest", str(confirmatory), "--json-output", "reports/A1_REPAIRED_CONFIRMATORY.json", "--markdown-output", "reports/A1_REPAIRED_CONFIRMATORY.md"])
        run("A2-A3-freeze", [PYTHON, "scripts/build_repaired_a2_a3_manifests.py"])
        a2 = ROOT / "logs/a1-repaired/a2_repaired_manifest.json"
        a3 = ROOT / "logs/a1-repaired/a3_repaired_manifest.json"
        run("A2", [PYTHON, "scripts/dispatch_repaired_measurements.py", "--experiment", "a2", "--manifest", str(a2), "--log-dir", "logs/a2-repaired"])
        run("A2-report", [PYTHON, "scripts/report_a2_measurement.py", "--manifest", str(a2), "--json-output", "reports/A2_REPAIRED.json", "--markdown-output", "reports/A2_REPAIRED.md"])
        run("A3", [PYTHON, "scripts/dispatch_repaired_measurements.py", "--experiment", "a3", "--manifest", str(a3), "--log-dir", "logs/a3-repaired"])
        run("A3-report", [PYTHON, "scripts/report_a3_repaired.py", "--manifest", str(a3), "--json-output", "reports/A3_REPAIRED.json", "--markdown-output", "reports/A3_REPAIRED.md"])
        write("sequence", "complete", "F4, repaired A1, repaired A2, and repaired A3 complete")
    except Exception as exc:
        write("sequence", "failed", repr(exc))
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
