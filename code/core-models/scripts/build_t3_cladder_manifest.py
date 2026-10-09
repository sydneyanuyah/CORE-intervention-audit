#!/usr/bin/env python3
"""Freeze compact, hash-bound CLadder robustness-pair instructions for T3."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from core_bert.t3_cladder_perturbations import (
    MANIFEST_VERSION,
    PROTOCOL,
    command_variant,
    rename_structural_variables,
    selected_template,
    sha256_file,
    sha256_json,
    structural_aliases,
)


def pair_id(dataset: str, record_id: str) -> str:
    digest = hashlib.sha256(f"{PROTOCOL}:{dataset}:{record_id}".encode()).hexdigest()[:24]
    return f"t3-cladder-pair:{digest}"


def build_manifest(artifact_path: Path) -> dict:
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    if artifact.get("test_evaluated") is not False:
        raise ValueError("source CLadder artifact must explicitly exclude test")
    worlds = {
        (int(world["model_id"]), int(world["source_id"])): world
        for world in artifact.get("worlds", [])
        if world.get("split") == "validation"
    }
    components = [
        record for record in artifact.get("components", [])
        if record.get("two_edit_metadata", {}).get("split") == "validation"
    ]
    if not components or not worlds:
        raise ValueError("source CLadder artifact has no validation data")
    pairs = []
    for record in sorted(components, key=lambda item: item["id"]):
        key = (int(record["two_edit_metadata"]["model_id"]), int(record["source_id"]))
        world = worlds.get(key)
        if world is None:
            raise ValueError(f"{record['id']}: missing validation world")
        world_group_id = world["world_group_id"]
        aliases = structural_aliases(record["graph"]["nodes"], world_group_id)
        renamed = rename_structural_variables(record, aliases)
        template_id = selected_template(record["id"])
        varied = command_variant(record, template_id)
        common = {
            "protocol": PROTOCOL,
            "record_id": record["id"],
            "source_id": record["source_id"],
            "model_id": key[0],
            "world_group_id": world_group_id,
            "split": "validation",
            "clean_sha256": sha256_json(record),
            "test_evaluated": False,
        }
        pairs.append({
            **common,
            "pair_id": pair_id("variable_rename", record["id"]),
            "dataset": "variable_rename",
            "structural_aliases": dict(sorted(aliases.items())),
            "perturbed_sha256": sha256_json(renamed),
        })
        pairs.append({
            **common,
            "pair_id": pair_id("cladder_variants", record["id"]),
            "dataset": "cladder_variants",
            "template_id": template_id,
            "perturbed_sha256": sha256_json(varied),
        })
    return {
        "protocol": PROTOCOL,
        "version": MANIFEST_VERSION,
        "split": "validation",
        "source_artifact": str(artifact_path),
        "source_artifact_sha256": sha256_file(artifact_path),
        "construction_seed": 317,
        "world_count": len(worlds),
        "source_record_count": len(components),
        "pair_count": len(pairs),
        "counts_by_dataset": {
            dataset: sum(pair["dataset"] == dataset for pair in pairs)
            for dataset in ("variable_rename", "cladder_variants")
        },
        "test_evaluated": False,
        "pairs": pairs,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_manifest(args.artifact)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in (
        "world_count", "source_record_count", "pair_count", "counts_by_dataset",
        "source_artifact_sha256", "test_evaluated",
    )}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
