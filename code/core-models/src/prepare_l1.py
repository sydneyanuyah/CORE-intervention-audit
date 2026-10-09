#!/usr/bin/env python3
"""Validate L1 development artifacts and expand the immutable cell queue."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if (
        manifest.get("protocol") != "l1_law_retention_v1"
        or manifest.get("test_evaluated") is not False
        or manifest.get("world_size") != 4
        or manifest.get("split_layer") != 4
    ):
        raise ValueError("L1 manifest violates the frozen production contract")
    seeds = manifest["optimization_seeds"]
    if seeds != list(range(201, 221)):
        raise ValueError("L1 requires the frozen ordinary seeds 201-220")
    train = manifest["training_family"]
    held = manifest["held_out_family"]
    if train.get("allowed_splits") != ["train", "validation"] or held.get("allowed_splits") != ["train", "validation"]:
        raise ValueError("L1 must be train/validation only")
    artifact = args.workdir / held["artifact"]
    pair_manifest = args.workdir / held["manifest"]
    for path, expected in ((artifact, held["artifact_sha256"]), (pair_manifest, held["manifest_sha256"])):
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"L1 held-out artifact identity mismatch: {path}")
    held_rows = rows(pair_manifest)
    if not held_rows or {row.get("split") for row in held_rows} - {"train", "validation"}:
        raise ValueError("L1 held-out manifest contains a forbidden or missing split")
    graph_seeds = train["graph_seeds"]
    if graph_seeds != list(range(3041, 3061)):
        raise ValueError("L1 XOR graph seeds must remain frozen at 3041-3060")
    xor_hashes = {}
    for graph_seed in graph_seeds:
        directory = Path(train["artifact_template"].format(graph_seed=graph_seed))
        for name in ("graph.json", "worlds.json", "records_real.json"):
            path = directory / name
            if not path.is_file():
                raise ValueError(f"missing L1 XOR artifact {path}")
            xor_hashes[str(path)] = digest(path)
        worlds = json.loads((directory / "worlds.json").read_text())
        records = json.loads((directory / "records_real.json").read_text())
        if {row.get("split") for row in worlds + records} - {"train", "validation"}:
            raise ValueError(f"L1 XOR graph {graph_seed} contains a forbidden split")
    methods = [*manifest["methods"].keys()]
    methods.remove("floors")
    methods.extend(manifest["methods"]["floors"])
    cells = []
    for family in ("xor", held["name"]):
        for seed, graph_seed in zip(seeds, graph_seeds):
            for method in methods:
                cells.append({
                    "cell_id": f"l1:{family}:{method}:{seed}",
                    "family": family,
                    "method": method,
                    "seed": seed,
                    "graph_seed": graph_seed if family == "xor" else None,
                    "world_size": 4,
                    "split_layer": 4,
                    "allowed_splits": ["train", "validation"],
                    "test_evaluated": False,
                })
    if len(cells) != 240 or len({cell["cell_id"] for cell in cells}) != 240:
        raise RuntimeError("L1 must expand to exactly 240 unique cells")
    payload = {
        "protocol": manifest["protocol"],
        "manifest_sha256": digest(args.manifest),
        "held_out_sha256": {str(artifact): digest(artifact), str(pair_manifest): digest(pair_manifest)},
        "xor_sha256": xor_hashes,
        "cell_count": len(cells),
        "cells": cells,
        "test_evaluated": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"cell_count": len(cells), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
