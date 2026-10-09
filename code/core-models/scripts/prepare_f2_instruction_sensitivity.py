#!/usr/bin/env python3
"""Register F2 CLadder O2/O3 inverted-instruction sensitivity cells."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    source_catalog_path = ROOT / "registry/a2_source_catalog.json"
    source_catalog = json.loads(source_catalog_path.read_text())
    indexed = {(row["family"], int(row["seed"])): row for row in source_catalog["cells"]}
    artifact = ROOT / "data/two_edit/cladder/artifact.json"
    pairs = ROOT / "data/two_edit/cladder/manifest.jsonl"
    core = Path("${PRIVATE_STORAGE_ROOT}/CORE_closing_controls/core_components.py")
    cells = []
    for seed in range(301, 321):
        source = indexed[("cladder", 2026090300 + seed)]
        reader = ROOT / source["source_directory"] / "best.pt"
        if sha256(reader) != source["checkpoint_sha256"]:
            raise ValueError(f"reader hash mismatch for seed {seed}")
        for method in ("o2", "o3"):
            directory = ROOT / f"outputs/f2/cladder/{method}/seed-{seed}"
            operator = directory / "operator.pt"
            summary = directory / "run_summary.json"
            row = json.loads(summary.read_text())
            if row.get("test_evaluated") is not False or row.get("world_size") != 4:
                raise ValueError(f"invalid F2 source cell: {method} {seed}")
            if row.get("checkpoint_sha256") != sha256(reader):
                raise ValueError(f"F2 reader binding mismatch: {method} {seed}")
            cells.append({
                "cell_id": f"f2-instruction-sensitivity:cladder:{method}:{seed}",
                "method": method,
                "seed": seed,
                "reader": str(reader),
                "reader_sha256": sha256(reader),
                "operator": str(operator),
                "operator_sha256": sha256(operator),
                "source_summary": str(summary),
                "source_summary_sha256": sha256(summary),
                "tokenizer": str(reader.parent / "tokenizer"),
                "output": f"outputs/f2-instruction-sensitivity/cladder/{method}/seed-{seed}",
                "world_size": 4,
                "test_evaluated": False,
            })
    manifest = {
        "protocol": "f2_cladder_inverted_instruction_positive_control_v1",
        "status": "registered",
        "purpose": "decision sensitivity only; original gold retained and not redefined",
        "family": "cladder",
        "methods": ["o2", "o3"],
        "seeds": list(range(301, 321)),
        "artifact": str(artifact),
        "artifact_sha256": sha256(artifact),
        "pair_manifest": str(pairs),
        "pair_manifest_sha256": sha256(pairs),
        "source_catalog": str(source_catalog_path),
        "source_catalog_sha256": sha256(source_catalog_path),
        "core_components": str(core),
        "core_components_sha256": sha256(core),
        "cells": cells,
        "allowed_splits": ["validation"],
        "test_evaluated": False,
    }
    output = ROOT / "registry/f2_cladder_instruction_sensitivity_manifest.json"
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"cells": len(cells), "manifest_sha256": sha256(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
