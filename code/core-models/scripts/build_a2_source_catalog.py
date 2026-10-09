#!/usr/bin/env python3
"""Freeze exact A1 confirmatory T2-b sources for preregistered A2 measurement."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


FAMILIES = ("ccrgb", "cladder", "com2", "wiqa", "xor")
SEEDS = tuple(range(2026090601, 2026090621))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def build(root: Path) -> dict:
    rows = []
    for family in FAMILIES:
        for seed in SEEDS:
            relative = Path("outputs/a1-confirmatory") / family / "t2b_encode_only" / f"seed-{seed}"
            directory = root / relative
            paths = {
                name: directory / name
                for name in ("best.pt", "cell_contract.json", "cell_summary.json", "composed_validation.json")
            }
            missing = [str(path) for path in paths.values() if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"missing frozen A2 source artifacts: {missing}")
            summary = load(paths["cell_summary.json"])
            contract = load(paths["cell_contract.json"])
            composed = load(paths["composed_validation.json"])
            expected_id = f"{family}:t2b_encode_only:{seed}"
            if (
                summary.get("cell_id") != expected_id
                or summary.get("family") != family
                or summary.get("arm") != "t2b_encode_only"
                or summary.get("mode") != "t2b"
                or summary.get("seed") != seed
                or summary.get("confirmatory") is not True
                or summary.get("provenance", {}).get("test_evaluated") is not False
                or summary.get("provenance", {}).get("world_size") != 4
            ):
                raise ValueError(f"inadmissible frozen A1 source cell {expected_id}")
            checkpoint_hash = summary.get("checkpoint_sha256")
            if not isinstance(checkpoint_hash, str) or len(checkpoint_hash) != 64:
                raise ValueError(f"source checkpoint identity is missing for {expected_id}")
            contract_hash = sha256(paths["cell_contract.json"])
            composed_hash = sha256(paths["composed_validation.json"])
            if (
                checkpoint_hash != composed.get("checkpoint_sha256")
                or contract_hash != summary.get("provenance", {}).get("cell_contract_sha256")
                or composed_hash != summary.get("provenance", {}).get("composed_summary_sha256")
                or contract.get("test_evaluated") is not False
                or composed.get("test_evaluated") is not False
            ):
                raise ValueError(f"frozen A1 hashes disagree for {expected_id}")
            rows.append({
                "cell_id": expected_id,
                "family": family,
                "seed": seed,
                "source_directory": str(relative),
                "checkpoint_sha256": checkpoint_hash,
                "cell_contract_sha256": contract_hash,
                "cell_summary_sha256": sha256(paths["cell_summary.json"]),
                "composed_validation_sha256": composed_hash,
            })
    if len(rows) != 100 or len({row["cell_id"] for row in rows}) != 100:
        raise RuntimeError("A2 source catalog must contain exactly 100 unique cells")
    return {
        "protocol": "a2_frozen_t2b_source_catalog_v1",
        "source_stage": "a1_confirmatory",
        "source_arm": "t2b_encode_only",
        "model_size": "base",
        "world_size": 4,
        "test_evaluated": False,
        "cells": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.repo_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"cells": len(result["cells"]), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
