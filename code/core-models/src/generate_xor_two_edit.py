#!/usr/bin/env python3
"""Write a validated XOR ordered two-edit JSONL manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from core_bert.xor_two_edit import generate_xor_two_edit_manifest_from_directory


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = generate_xor_two_edit_manifest_from_directory(args.artifact_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(json.dumps({"rows": len(rows), "output": str(args.output), "a1_evidence": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
