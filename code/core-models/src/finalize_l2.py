#!/usr/bin/env python3
"""Fail-closed L2 inventory and provenance validator."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


LEARNED = {"full", "drop_identity", "drop_idempotence", "drop_commutation", "drop_selective_invariance"}
FLOORS = {"do_nothing", "do_everything", "random_init"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(cell: dict, root: Path) -> dict:
    directory = root / "outputs" / "l2" / "cells" / cell["family"] / cell["method"] / f"seed-{cell['seed']}"
    summary_path, checkpoint_path = directory / "run_summary.json", directory / "best.pt"
    summary = json.loads(summary_path.read_text())
    if summary.get("test_evaluated") is not False:
        raise ValueError("test lock missing")
    method = cell["method"]
    if method in FLOORS:
        if summary.get("protocol") != "l2_dropped_law_v1" or summary.get("cell_id") != cell["cell_id"]:
            raise ValueError("floor protocol or cell identity mismatch")
        if summary.get("execution") != "deterministic_symbolic_validation_floor_v1":
            raise ValueError("floor execution mismatch")
        if checkpoint_path.exists():
            raise ValueError("symbolic floor must not create a checkpoint")
        if int(summary.get("world_size", -1)) != 4:
            raise ValueError("four-rank contract missing")
        checkpoint = checkpoint_sha = None
    elif method in LEARNED:
        if not checkpoint_path.is_file():
            raise ValueError("missing learned checkpoint")
        config, law = summary.get("configuration", {}), summary.get("l1", {})
        if summary.get("gpu_policy") != {"model_size": "base", "world_size": 4}:
            raise ValueError("learned cell GPU policy mismatch")
        if int(config.get("seed", -1)) != int(cell["seed"]) or int(config.get("split_layer", -1)) != 4:
            raise ValueError("seed or frozen layer mismatch")
        if law.get("law_profile") != method or law.get("reader_frozen") is not True:
            raise ValueError("law profile or reader freeze mismatch")
        checkpoint, checkpoint_sha = str(checkpoint_path.relative_to(root)), sha256(checkpoint_path)
    else:
        raise ValueError("unregistered L2 method")
    return {
        "cell_id": cell["cell_id"], "summary": str(summary_path.relative_to(root)),
        "summary_sha256": sha256(summary_path), "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_sha, "test_evaluated": False,
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
        "protocol": registry["protocol"], "manifest_sha256": registry["manifest_sha256"],
        "registered": len(registry["cells"]), "complete": len(complete),
        "remaining": len(registry["cells"]) - len(complete), "cells": complete,
        "rejected": rejected, "test_evaluated": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: payload[key] for key in ("registered", "complete", "remaining")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
