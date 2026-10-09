#!/usr/bin/env python3
"""Materialize F2's preregistered XOR train/validation artifacts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_core(path: Path):
    spec = importlib.util.spec_from_file_location("canonical_f2_core", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load canonical source {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--core-components", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    expected = manifest["operator"]["source_sha256"]
    if sha256(args.core_components) != expected:
        raise ValueError("canonical O2/O3 source hash mismatch")
    seeds = manifest["xor_graph_seeds"]
    if seeds != list(range(3061, 3081)):
        raise ValueError("F2 XOR graph seeds changed")
    core = load_core(args.core_components)
    provenance = {}
    for graph_seed in seeds:
        graph = core.make_graph(seed=graph_seed)
        worlds = [row for row in core.make_worlds(graph, graph_seed) if row["split"] in {"train", "validation"}]
        records = core.make_records(graph, worlds)
        if {row["split"] for row in worlds + records} != {"train", "validation"}:
            raise ValueError(f"graph {graph_seed} does not contain exactly train/validation")
        directory = args.output_root / f"graph_{graph_seed}"
        directory.mkdir(parents=True, exist_ok=True)
        for name, payload in (("graph.json", graph), ("worlds.json", worlds), ("records_real.json", records)):
            path = directory / name
            path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            provenance[str(path)] = sha256(path)
    provenance_path = args.output_root / "provenance.json"
    provenance_path.write_text(json.dumps({
        "protocol": manifest["protocol"],
        "canonical_source_sha256": expected,
        "allowed_splits": ["train", "validation"],
        "artifacts": provenance,
        "test_evaluated": False,
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"graphs": len(seeds), "provenance_sha256": sha256(provenance_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
