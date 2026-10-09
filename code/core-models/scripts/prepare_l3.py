#!/usr/bin/env python3
"""Expand L3 into 40 hash-bound registered cells on the execution host."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("protocol") != "l3_eight_method_law_v1" or manifest.get("test_evaluated") is not False:
        raise ValueError("invalid L3 manifest or test lock")
    if manifest["optimization_seeds"] != list(range(301, 306)) or manifest["graph_seeds"] != list(range(3061, 3066)):
        raise ValueError("L3 requires the five frozen F2 XOR seed pairs")
    if manifest.get("world_size") != 4 or manifest.get("allowed_splits") != ["train", "validation"]:
        raise ValueError("L3 requires four ranks and train/validation only")
    if sha256(Path(manifest["canonical_core_source"])) != manifest["canonical_core_sha256"]:
        raise ValueError("canonical CORE source changed")
    for path, expected in manifest["runner_sha256"].items():
        if sha256(args.workdir / path) != expected: raise ValueError(f"L3 runner changed: {path}")
    reused = set(manifest["execution"]["frozen_f2_validation_rescore"])
    cells = []
    for seed, graph_seed in zip(manifest["optimization_seeds"], manifest["graph_seeds"]):
        artifact = Path(manifest["paths"]["artifact_template"].format(graph_seed=graph_seed))
        reader = args.workdir / manifest["paths"]["reader_template"].format(seed=seed)
        worlds = json.loads((artifact / "worlds.json").read_text())
        if {row["split"] for row in worlds} != {"train", "validation"}:
            raise ValueError(f"L3 artifact {graph_seed} includes a forbidden or missing split")
        artifact_hashes = {name: sha256(artifact / name) for name in ("graph.json", "worlds.json", "records_real.json")}
        for method in manifest["methods"]:
            source = args.workdir / manifest["paths"]["f2_source_template"].format(method=method, seed=seed) if method in reused else None
            if source is not None and not source.is_file(): raise FileNotFoundError(source)
            cells.append({
                "cell_id": f"l3:xor:{method}:{seed}", "family": "xor", "method": method,
                "seed": seed, "graph_seed": graph_seed, "world_size": 4, "split_layer": 4,
                "execution": "frozen_f2_validation_rescore" if method in reused else "fresh_operator_training",
                "artifact_dir": str(artifact), "artifact_sha256": artifact_hashes,
                "reader_checkpoint": str(reader.relative_to(args.workdir)), "reader_sha256": sha256(reader),
                "source_checkpoint": str(source.relative_to(args.workdir)) if source else None,
                "source_checkpoint_sha256": sha256(source) if source else None,
                "allowed_splits": ["train", "validation"], "test_evaluated": False,
            })
    if len(cells) != 40 or len({row["cell_id"] for row in cells}) != 40:
        raise RuntimeError("L3 requires exactly 40 unique cells")
    payload = {"protocol": manifest["protocol"], "manifest_sha256": sha256(args.manifest), "cell_count": 40, "cells": cells, "test_evaluated": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"cell_count": 40, "manifest_sha256": payload["manifest_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
