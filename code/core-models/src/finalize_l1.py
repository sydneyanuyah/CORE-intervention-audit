#!/usr/bin/env python3
"""Fail-closed L1 cell inventory and provenance validator."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PROFILES = {"core_base_1law", "core_i_3law", "core_full_4law"}
FLOORS = {"do_nothing", "do_everything", "random_init"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(cell: dict, root: Path) -> dict:
    family, method, seed = cell["family"], cell["method"], int(cell["seed"])
    directory = root / "outputs" / "l1" / "cells" / family / method / f"seed-{seed}"
    summary_path, checkpoint_path = directory / "run_summary.json", directory / "best.pt"
    if not summary_path.is_file():
        raise ValueError("missing summary")
    summary = json.loads(summary_path.read_text())
    if summary.get("protocol") != "l1_law_retention_v1":
        raise ValueError("non-L1 runner protocol")
    if summary.get("test_evaluated") is not False:
        raise ValueError("test lock is not explicit")
    if method in FLOORS:
        if summary.get("cell_id") != cell["cell_id"]:
            raise ValueError("registered cell identity mismatch")
        if checkpoint_path.exists():
            raise ValueError("symbolic floor must not have a checkpoint")
        if summary.get("execution") != "deterministic_symbolic_validation_floor_v1":
            raise ValueError("symbolic floor execution mismatch")
        if summary.get("family") != family or summary.get("method") != method:
            raise ValueError("symbolic floor family or method mismatch")
        if int(summary.get("seed", -1)) != seed or int(summary.get("world_size", -1)) != 4:
            raise ValueError("symbolic floor seed or four-rank contract mismatch")
        return {
            "cell_id": cell["cell_id"],
            "summary": str(summary_path.relative_to(root)),
            "summary_sha256": sha256(summary_path),
            "checkpoint": None,
            "checkpoint_sha256": None,
            "test_evaluated": False,
        }
    if not checkpoint_path.is_file():
        raise ValueError("missing best checkpoint")
    if summary.get("gpu_policy") != {"model_size": "base", "world_size": 4}:
        raise ValueError("L1 requires exactly four BERT-base ranks")
    config, l1 = summary.get("configuration", {}), summary.get("l1", {})
    if str(config.get("seed")) != str(seed) or int(config.get("split_layer", -1)) != 4:
        raise ValueError("seed or frozen edit layer mismatch")
    if config.get("selection_metric") != "macro_f1":
        raise ValueError("checkpoint selection mismatch")
    if method in PROFILES:
        if l1.get("law_profile") != method or l1.get("reader_frozen") is not True:
            raise ValueError("law profile or reader-freeze mismatch")
        init = Path(l1.get("initialization_checkpoint", ""))
        if not init.is_absolute():
            init = root / init
        if not init.is_file() or sha256(init) != l1.get("initialization_checkpoint_sha256"):
            raise ValueError("reader initialization identity mismatch")
    return {
        "cell_id": cell["cell_id"],
        "summary": str(summary_path.relative_to(root)),
        "summary_sha256": sha256(summary_path),
        "checkpoint": str(checkpoint_path.relative_to(root)),
        "checkpoint_sha256": sha256(checkpoint_path),
        "test_evaluated": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    registry = json.loads(args.cells.read_text())
    complete, rejected = [], {}
    for cell in registry["cells"]:
        try:
            complete.append(validate(cell, args.workdir))
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            rejected[cell["cell_id"]] = str(error)
    payload = {
        "protocol": registry["protocol"],
        "manifest_sha256": registry["manifest_sha256"],
        "registered": len(registry["cells"]),
        "complete": len(complete),
        "remaining": len(registry["cells"]) - len(complete),
        "cells": complete,
        "rejected": rejected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: payload[key] for key in ("registered", "complete", "remaining")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
