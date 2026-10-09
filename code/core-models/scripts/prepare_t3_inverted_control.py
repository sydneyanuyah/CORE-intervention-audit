#!/usr/bin/env python3
"""Register T3 inverted-instruction sensitivity controls on frozen CLadder checkpoints."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    source = ROOT / "data/two_edit/cladder/artifact.json"
    a1 = json.loads((ROOT / "registry/a1_confirmatory_manifest.json").read_text())
    cells = []
    for seed in a1["seeds"]:
        checkpoint = ROOT / f"outputs/a1-confirmatory/cladder/t3b_pointer/seed-{seed}/best.pt"
        summary = checkpoint.with_name("cell_summary.json")
        contract = checkpoint.with_name("cell_contract.json")
        if not checkpoint.is_file() or not summary.is_file() or not contract.is_file():
            raise ValueError(f"missing frozen T3 source: {seed}")
        if json.loads(contract.read_text()).get("test_evaluated") is not False:
            raise ValueError(f"T3 source is not validation-only: {seed}")
        cells.append({
            "cell_id": f"t3-positive-control:cladder:inverted_instruction:seed-{seed}",
            "seed": seed,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha(checkpoint),
            "source_summary": str(summary),
            "source_summary_sha256": sha(summary),
            "source_contract": str(contract),
            "source_contract_sha256": sha(contract),
            "world_size": 4,
            "output": f"outputs/t3-positive-control/cladder/inverted_instruction/seed-{seed}",
            "test_evaluated": False,
        })
    manifest = {
        "protocol": "t3_inverted_instruction_positive_control_v1",
        "status": "registered",
        "purpose": "decision-sensitivity control; retained gold is not a task-accuracy target",
        "records": 144,
        "source_artifact": str(source),
        "source_artifact_sha256": sha(source),
        "cells": cells,
        "test_evaluated": False,
    }
    path = ROOT / "registry/t3_inverted_control_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"cells": len(cells), "manifest_sha256": sha(path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
