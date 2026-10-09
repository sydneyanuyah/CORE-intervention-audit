#!/usr/bin/env python3
"""Build disjoint CPU-only XOR artifacts for A1 confirmation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_generator(root: Path):
    path = root / "core_components.py"
    spec = importlib.util.spec_from_file_location("frozen_xor_core_components", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load frozen XOR generator {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, path


def build(generator_root: Path, output_root: Path, manifest_dir: Path, seeds: list[int]) -> dict:
    if len(seeds) != 20 or len(set(seeds)) != 20:
        raise ValueError("confirmatory XOR construction requires exactly 20 unique graph seeds")
    if set(seeds) & set(range(3001, 3021)):
        raise ValueError("confirmatory XOR graph seeds overlap architecture screening")
    core, generator_path = _load_generator(generator_root)
    from core_bert.xor_two_edit import generate_xor_two_edit_manifest

    artifacts = []
    for seed in seeds:
        directory = output_root / "runs" / f"graph_{seed}"
        graph = core.make_graph(seed=seed)
        worlds = [
            row for row in core.make_worlds(graph, seed=seed + 100_000)
            if row["split"] in {"train", "validation"}
        ]
        records = core.make_records(graph, worlds)
        if len(worlds) != 26 or len(records) != 1560:
            raise RuntimeError(f"graph {seed} has an unexpected frozen-generator shape")
        _write(directory / "graph.json", graph)
        _write(directory / "worlds.json", worlds)
        _write(directory / "records_real.json", records)
        rows = generate_xor_two_edit_manifest(
            graph, worlds, records, graph_group_id=f"xor:graph:{seed}"
        )
        manifest_path = manifest_dir / f"xor_graph_{seed}.jsonl"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        if not rows or {row["split"] for row in rows} != {"train", "validation"}:
            raise RuntimeError(f"graph {seed} manifest violated train/validation isolation")
        artifacts.append({
            "seed": seed,
            "graph_sha256": _sha256(directory / "graph.json"),
            "worlds_sha256": _sha256(directory / "worlds.json"),
            "records_sha256": _sha256(directory / "records_real.json"),
            "manifest_sha256": _sha256(manifest_path),
            "pair_count": len(rows),
        })
    provenance = {
        "protocol": "a1_confirmatory_xor_artifacts_v1",
        "generator": str(generator_path),
        "generator_sha256": _sha256(generator_path),
        "screening_graphs": list(range(3001, 3021)),
        "confirmatory_graphs": seeds,
        "test_evaluated": False,
        "artifacts": artifacts,
    }
    _write(manifest_dir / "provenance.json", provenance)
    return provenance


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--first-seed", type=int, default=3021)
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()
    seeds = list(range(args.first_seed, args.first_seed + args.count))
    result = build(args.generator_root, args.output_root, args.manifest_dir, seeds)
    print(json.dumps({
        "graph_count": len(result["artifacts"]),
        "test_evaluated": result["test_evaluated"],
        "provenance_sha256": _sha256(args.manifest_dir / "provenance.json"),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
