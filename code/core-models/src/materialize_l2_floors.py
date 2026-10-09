#!/usr/bin/env python3
"""Materialize every preregistered L2 prediction-level floor on validation only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from core_bert.benchmark_data import load_benchmark_split
from materialize_l1_floors import FLOORS, summarize
from train_addressed import load_xor_training_split


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    args = parser.parse_args()
    registry = json.loads(args.cells.read_text())
    if registry.get("protocol") != "l2_dropped_law_v1" or registry.get("test_evaluated") is not False:
        raise ValueError("invalid L2 registry or test lock")

    cladder = load_benchmark_split(args.workdir / "data", "validation", sources=["cladder"])
    xor_cache: dict[int, object] = {}
    materialized = 0
    for cell in registry["cells"]:
        method = cell["method"]
        if method not in FLOORS:
            continue
        family, seed = cell["family"], int(cell["seed"])
        if family == "xor":
            graph = int(cell["graph_seed"])
            if graph not in xor_cache:
                xor_cache[graph] = load_xor_training_split([[
                    f"${PRIVATE_STORAGE_ROOT}/CORE_f1/runs/graph_{graph}",
                    str(args.workdir / f"data/two_edit/f1/xor_graph_{graph}.jsonl"),
                ]], "validation")
            examples = xor_cache[graph]
        elif family == "cladder":
            examples = cladder
        else:
            raise ValueError(f"unsupported L2 family: {family}")
        output = args.workdir / "outputs" / "l2" / "cells" / family / method / f"seed-{seed}"
        summary_path = output / "run_summary.json"
        if summary_path.is_file():
            continue
        output.mkdir(parents=True, exist_ok=True)
        payload = {
            "protocol": "l2_dropped_law_v1",
            "cell_id": cell["cell_id"],
            "family": family,
            "method": method,
            "seed": seed,
            "execution": "deterministic_symbolic_validation_floor_v1",
            "world_size": 4,
            "validation": summarize(method, seed, examples),
            "test_evaluated": False,
        }
        summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        materialized += 1
    print(json.dumps({"materialized_floor_cells": materialized}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
