#!/usr/bin/env python3
"""Wait for L3 completion and run its fail-closed reporter exactly once."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    status_path = ROOT / "logs/l3/status.json"
    report = ROOT / "reports/L3_RESULTS.md"
    while not report.is_file():
        try: status = json.loads(status_path.read_text())
        except (OSError, json.JSONDecodeError): status = {}
        complete, remaining, active = int(status.get("complete", 0)), int(status.get("remaining", 40)), status.get("active", {})
        if complete == 40 and remaining == 0 and not active:
            subprocess.run([
                str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/report_l3.py"),
                "--cells", str(ROOT / "registry/l3_cells.json"), "--manifest", str(ROOT / "registry/l3_manifest.json"),
                "--workdir", str(ROOT), "--evidence", str(ROOT / "reports/evidence/l3_complete.json"),
                "--report", str(report),
            ], cwd=ROOT, check=True)
            return 0
        if not active and complete > 0 and remaining > 0: raise RuntimeError(f"L3 stopped incomplete at {complete}/40")
        time.sleep(30)
    return 0


if __name__ == "__main__": raise SystemExit(main())
