#!/usr/bin/env python3
"""Freeze the final held-out evaluation queue without opening test records."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def entry(task: str, cell_id: str, checkpoint: Path, output: str, **extra):
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return {
        "task": task,
        "cell_id": cell_id,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": sha(checkpoint),
        "output": output,
        "world_size": 4,
        **extra,
    }


def main() -> int:
    manifests = {
        name: ROOT / f"registry/{name}_manifest.json"
        for name in ("a1_repaired_confirmatory", "t2", "t4", "t5")
    }
    source = {name: json.loads(path.read_text()) for name, path in manifests.items()}
    cells = []
    a1 = source["a1_repaired_confirmatory"]
    for family in ("xor", "ccrgb", "cladder", "wiqa"):
        for seed in a1["seeds"]:
            cp = ROOT / a1["paths"]["output_template"].format(
                family=family, arm="t3b_pointer", seed=seed
            ) / "best.pt"
            cells.append(entry(
                "a1", f"a1:{family}:t3b_pointer:{seed}", cp,
                f"outputs/final-one-shot/a1/{family}/seed-{seed}",
                family=family, seed=seed,
            ))
    t2 = source["t2"]
    for seed in t2["seeds"]:
        cells.append(entry(
            "t2", f"t2:{seed}", ROOT / f"outputs/t2/seed-{seed}/best.pt",
            f"outputs/final-one-shot/t2/seed-{seed}", seed=seed,
        ))
    t4 = source["t4"]
    for arm in t4["arms"]:
        for seed in t4["seeds"]:
            cells.append(entry(
                "t4", f"t4:{arm}:{seed}", ROOT / f"outputs/t4/{arm}/seed-{seed}/best.pt",
                f"outputs/final-one-shot/t4/{arm}/seed-{seed}", arm=arm, seed=seed,
            ))
    t5 = source["t5"]
    for method in t5["methods"]:
        for seed in t5["seeds"]:
            cells.append(entry(
                "t5", f"t5:{method}:{seed}", ROOT / f"outputs/t5/{method}/seed-{seed}/best.pt",
                f"outputs/final-one-shot/t5/{method}/seed-{seed}", method=method, seed=seed,
            ))

    runners = {
        "a1": ROOT / "src/evaluate_a1_final.py",
        "t2_t4": ROOT / "src/evaluate_t2_t4_final.py",
        "t5": ROOT / "src/evaluate_candidate_final.py",
        "dispatcher": ROOT / "scripts/run_final_one_shot.py",
    }
    audit = ROOT / "experiments/DATA_SPLIT_AUDIT.json"
    payload = {
        "protocol": "core_final_one_shot_v1",
        "status": "frozen_prelaunch",
        "authorization_literal": "FINAL_ONE_SHOT",
        "world_size_per_cell": 4,
        "max_parallel_cells": 8,
        "test_policy": "consume lock before first test read; no retry or reselection",
        "excluded_sources": {"com2": "removed by user decision before final freeze"},
        "test_sets": {
            "a1": {"data_root": "data", "families": ["xor", "ccrgb", "cladder", "wiqa"], "expected_rows": 6706, "lock": "experiments/Experiment-5-A1/data/test/LOCKED_DO_NOT_EVALUATE.txt"},
            "t2": {"path": "experiments/Experiment-16-T2/data/test/csuite_t2.jsonl", "expected_rows": 2000, "lock": "experiments/Experiment-16-T2/data/test/LOCKED_DO_NOT_EVALUATE.txt"},
            "t4": {"path": "experiments/Experiment-18-T4/data/test/csuite_t4.jsonl", "expected_rows": 6000, "lock": "experiments/Experiment-18-T4/data/test/LOCKED_DO_NOT_EVALUATE.txt"},
            "t5": {"path": "experiments/Experiment-19-T5/data/test/native_t5.jsonl", "expected_rows_before_com2_exclusion": 590, "lock": "experiments/Experiment-19-T5/data/test/LOCKED_DO_NOT_EVALUATE.txt"},
        },
        "data_split_audit": str(audit.relative_to(ROOT)),
        "data_split_audit_sha256": sha(audit),
        "source_manifests": {name: {"path": str(path.relative_to(ROOT)), "sha256": sha(path)} for name, path in manifests.items()},
        "runners": {name: {"path": str(path.relative_to(ROOT)), "sha256": sha(path)} for name, path in runners.items()},
        "cell_count": len(cells),
        "cells": cells,
        "test_evaluated": False,
    }
    if len(cells) != 140 or len({cell["cell_id"] for cell in cells}) != 140:
        raise RuntimeError("final queue must contain exactly 140 unique cells")
    output = ROOT / "registry/final_one_shot_manifest.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(output), "sha256": sha(output), "cells": 140, "test_evaluated": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
