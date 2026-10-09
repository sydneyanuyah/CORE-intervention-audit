#!/usr/bin/env python3
"""Register the validation-only Phi-4-14B CLadder operator extension."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL = Path.home() / ".cache/huggingface/hub/models--microsoft--phi-4/snapshots/not-published"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    artifact = ROOT / "data/two_edit/cladder/artifact.json"
    pairs = ROOT / "data/two_edit/cladder/manifest.jsonl"
    required = [artifact, pairs, MODEL / "config.json", MODEL / "model.safetensors.index.json"]
    if any(not path.is_file() for path in required):
        raise FileNotFoundError("Phi snapshot or CLadder inputs are incomplete")
    artifact_value = json.loads(artifact.read_text())
    node_order = sorted({node for row in artifact_value["components"] for node in row["graph"]["nodes"]})
    cells = [
        {
            "cell_id": f"phi14b-f2:cladder:{method}:{seed}",
            "method": method,
            "seed": seed,
            "output": f"outputs/phi14b-f2/cladder/{method}/seed-{seed}",
            "gpu_count": 1,
        }
        for seed in range(401, 421) for method in ("o2", "o3")
    ]
    manifest = {
        "protocol": "phi4_14b_cladder_o2_o3_extension_v1",
        "status": "preregistered_prelaunch",
        "scope": ["F2 composed operator comparison", "inverted-command sensitivity", "do-nothing floor"],
        "model": "microsoft/phi-4",
        "model_revision": MODEL.name,
        "model_snapshot": str(MODEL),
        "model_config_sha256": sha(MODEL / "config.json"),
        "model_index_sha256": sha(MODEL / "model.safetensors.index.json"),
        "feature_contract": {
            "backbone_frozen": True,
            "quantization": "NF4 feature extraction only",
            "prompt": "factual passage followed by a fixed downstream-editing cue",
            "pooling": "last non-padding final-layer hidden state",
            "max_length": 512,
        },
        "operator_contract": {"methods": ["o2", "o3"], "rank": 16, "node_order": node_order, "variable_graph_width": True},
        "training_contract": {"steps": 1200, "learning_rate": 0.0003, "full_batch": True, "selection": "final preregistered step"},
        "inference_contract": {
            "unit": "paired training seed",
            "primary": "O2 minus O3 validation two_edit_balanced",
            "secondary": ["clean score versus inversion movement", "each method minus do-nothing"],
            "interval": "two-sided paired Student-t 95%",
        },
        "artifact": str(artifact), "artifact_sha256": sha(artifact),
        "pair_manifest": str(pairs), "pair_manifest_sha256": sha(pairs),
        "feature_cache": "outputs/phi14b-f2/cache/cladder_features.pt",
        "feature_cache_summary": "outputs/phi14b-f2/cache/cache_summary.json",
        "seeds": list(range(401, 421)), "cells": cells,
        "allowed_splits": ["train", "validation"], "evaluation_split": "validation",
        "test_evaluated": False,
    }
    output = ROOT / "registry/phi14b_cladder_extension_manifest.json"
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"cells": len(cells), "manifest_sha256": sha(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
