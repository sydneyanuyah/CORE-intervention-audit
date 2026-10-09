#!/usr/bin/env python3
"""Expand the immutable L2 dropped-law cell registry."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("protocol") != "l2_dropped_law_v1" or manifest.get("test_evaluated") is not False:
        raise ValueError("invalid L2 protocol or test lock")
    if manifest.get("world_size") != 4 or manifest.get("split_layer") != 4:
        raise ValueError("L2 requires four-GPU BERT-base cells at edit layer 4")
    seeds = manifest["seeds"]
    if seeds != list(range(201, 221)):
        raise ValueError("L2 seeds must be 201-220")
    methods = [*manifest["methods"], *manifest["floors"]]
    cells = []
    for family in manifest["families"]:
        for offset, seed in enumerate(seeds):
            for method in methods:
                cells.append({
                    "cell_id": f"l2:{family}:{method}:{seed}",
                    "family": family, "method": method, "seed": seed,
                    "graph_seed": manifest["xor_graph_seeds"][offset] if family == "xor" else None,
                    "world_size": 4, "split_layer": 4,
                    "allowed_splits": ["train", "validation"],
                    "test_evaluated": False,
                })
    if len(cells) != 320 or len({c["cell_id"] for c in cells}) != 320:
        raise RuntimeError("L2 requires exactly 320 unique cells")
    payload = {
        "protocol": manifest["protocol"],
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "cell_count": len(cells), "cells": cells, "test_evaluated": False,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"cell_count": len(cells), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
