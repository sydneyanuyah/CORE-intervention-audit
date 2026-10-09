#!/usr/bin/env python3
"""Freeze A2 and A3 source catalogs/manifests from repaired A1 bytes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parents[1].resolve()
CONF = ROOT / "registry/a1_repaired_confirmatory_manifest.json"
REPORT = ROOT / "reports/A1_REPAIRED_CONFIRMATORY.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path):
    return json.loads(path.read_text())


def dump_once(path: Path, value) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != encoded:
        raise RuntimeError(f"refusing to mutate frozen artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded)


def catalog(arm: str, output: Path) -> dict:
    manifest = load(CONF)
    rows = []
    for family in sorted(manifest["families"]):
        for seed in manifest["seeds"]:
            relative = Path(manifest["paths"]["output_template"].format(family=family, arm=arm, seed=seed))
            directory = ROOT / relative
            files = {name: directory / name for name in ("best.pt", "cell_contract.json", "cell_summary.json", "composed_validation.json")}
            if not all(path.is_file() for path in files.values()):
                raise RuntimeError(f"missing repaired A1 source: {family}:{arm}:{seed}")
            summary = load(files["cell_summary.json"])
            if summary.get("cell_id") != f"{family}:{arm}:{seed}" or summary.get("confirmatory") is not True or summary.get("provenance", {}).get("test_evaluated") is not False or summary.get("provenance", {}).get("world_size") != 4:
                raise RuntimeError(f"inadmissible repaired A1 source: {family}:{arm}:{seed}")
            rows.append({
                "cell_id": summary["cell_id"], "family": family, "seed": seed,
                "source_directory": str(relative),
                "checkpoint_sha256": sha(files["best.pt"]),
                "cell_contract_sha256": sha(files["cell_contract.json"]),
                "cell_summary_sha256": sha(files["cell_summary.json"]),
                "composed_validation_sha256": sha(files["composed_validation.json"]),
            })
    value = {"protocol": f"repaired_a1_{arm}_source_catalog_v2", "source_arm": arm, "model_size": "base", "world_size": 4, "test_evaluated": False, "cells": rows}
    dump_once(output, value)
    return value


def main() -> int:
    if load(REPORT).get("completed_cells") != 520:
        raise RuntimeError("repaired A1 confirmatory report is incomplete")
    a2_catalog_path = ROOT / "logs/a1-repaired/a2_source_catalog.json"
    a3_catalog_path = ROOT / "logs/a1-repaired/a3_source_catalog.json"
    a2_catalog = catalog("t2b_encode_only", a2_catalog_path)
    a3_catalog = catalog("t3b_pointer", a3_catalog_path)
    old = load(ROOT / "registry/a2_measurement_manifest.json")
    old["source"] = {
        "report": str(REPORT.relative_to(ROOT)), "report_sha256": sha(REPORT),
        "manifest": str(CONF.relative_to(ROOT)), "manifest_sha256": sha(CONF),
        "catalog": str(a2_catalog_path.relative_to(ROOT)), "catalog_sha256": sha(a2_catalog_path),
        "arm": "t2b_encode_only", "checkpoint_count": 100,
    }
    old["seeds"] = list(range(201, 221))
    old["paths"]["output_template"] = "outputs/a2-repaired/{family}/{control}/seed-{seed}"
    old["paths"]["paraphrase_catalog"] = "data/paraphrases/structured_catalog_repaired.json"
    a2_path = ROOT / "logs/a1-repaired/a2_repaired_manifest.json"
    dump_once(a2_path, old)
    a3 = {
        "protocol": "a3_frozen_checkpoint_addressing_v2", "registry": "registry/bert_scope.json",
        "registry_sha256": sha(ROOT / "registry/bert_scope.json"), "model_size": "base", "world_size": 4,
        "source": {"report": str(REPORT.relative_to(ROOT)), "report_sha256": sha(REPORT), "manifest": str(CONF.relative_to(ROOT)), "manifest_sha256": sha(CONF), "catalog": str(a3_catalog_path.relative_to(ROOT)), "catalog_sha256": sha(a3_catalog_path), "arm": "t3b_pointer", "checkpoint_count": 100},
        "families": sorted({row["family"] for row in a3_catalog["cells"]}), "seeds": list(range(201, 221)),
        "controls": ["correct", "adjacent", "random"], "registered_cell_count": 300,
        "data_access": {"allowed_split": "validation", "held_out_test": "forbidden", "final_eval": False},
        "paths": {**old["paths"], "output_template": "outputs/a3-repaired/{family}/{control}/seed-{seed}"},
        "test_evaluated": False,
    }
    a3_path = ROOT / "logs/a1-repaired/a3_repaired_manifest.json"
    dump_once(a3_path, a3)
    print(json.dumps({"a2_cells": 400, "a3_cells": 300, "a2_manifest": str(a2_path), "a3_manifest": str(a3_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
