#!/usr/bin/env python3
"""Expand the immutable F2 family/method/seed queue."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.manifest.read_bytes()
    manifest = json.loads(raw)
    seeds = manifest["optimization_seeds"]
    families = manifest["families"]
    methods = list(manifest["methods"])
    if seeds != list(range(301, 321)) or len(families) != 5 or len(methods) != 4:
        raise ValueError("F2 frozen axes changed")
    if manifest.get("allowed_splits") != ["train", "validation"] or manifest.get("test_evaluated") is not False:
        raise ValueError("F2 split lock changed")
    for raw_path, expected in manifest["artifact_sha256"].items():
        path = Path(raw_path)
        if not path.is_absolute():
            path = args.workdir / path
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"F2 artifact identity mismatch: {path}")
    xor_template = manifest["paths"]["xor_artifact_template"]
    for graph_seed in manifest["xor_graph_seeds"]:
        directory = Path(xor_template.format(graph_seed=graph_seed))
        for name in ("graph.json", "worlds.json", "records_real.json"):
            if not (directory / name).is_file():
                raise ValueError(f"missing F2 XOR artifact: {directory / name}")
    cells = []
    for family in families:
        for seed in seeds:
            for method in methods:
                cell = {
                    "cell_id": f"f2:{family}:{method}:{seed}",
                    "family": family,
                    "method": method,
                    "seed": seed,
                    "split_layer": 4,
                    "world_size": 4,
                    "allowed_splits": ["train", "validation"],
                    "test_evaluated": False,
                }
                if family == "xor":
                    cell["graph_seed"] = seed + 2760
                cells.append(cell)
    if len(cells) != 400 or len({cell["cell_id"] for cell in cells}) != 400:
        raise RuntimeError("F2 must contain exactly 400 unique cells")
    payload = {
        "protocol": manifest["protocol"],
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "cell_count": len(cells),
        "cells": cells,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"cell_count": len(cells), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
