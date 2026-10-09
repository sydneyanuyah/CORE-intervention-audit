#!/usr/bin/env python3
"""Run the registered measurement-only T1 aggregation."""
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    return subprocess.run(
        [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/report_t1.py")],
        cwd=ROOT,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
