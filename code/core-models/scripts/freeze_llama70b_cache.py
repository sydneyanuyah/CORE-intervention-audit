#!/usr/bin/env python3
"""Bind the deterministic Phi feature cache before operator training."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    manifest_path = ROOT / "registry/llama70b_cladder_extension_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    cache = Path(manifest["feature_cache"]); summary = Path(manifest["feature_cache_summary"])
    row = json.loads(summary.read_text())
    if row.get("manifest_sha256") != sha(manifest_path) or row.get("test_evaluated") is not False:
        raise ValueError("feature cache summary is not bound to the manifest")
    amendment = {
        "protocol": "llama3_3_70b_cladder_feature_cache_binding_v1",
        "base_manifest": str(manifest_path), "base_manifest_sha256": sha(manifest_path),
        "feature_cache": str(cache), "feature_cache_sha256": sha(cache),
        "feature_cache_summary": str(summary), "feature_cache_summary_sha256": sha(summary),
        "cells": manifest["cells"], "allowed_splits": ["train", "validation"],
        "test_evaluated": False,
    }
    output = ROOT / "registry/llama70b_cladder_cache_amendment.json"
    output.write_text(json.dumps(amendment, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"cells": len(amendment["cells"]), "amendment_sha256": sha(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
