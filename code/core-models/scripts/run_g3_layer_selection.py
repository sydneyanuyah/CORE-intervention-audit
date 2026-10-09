#!/usr/bin/env python3
"""Validate, execute, and finalize the preregistered development-only G3 sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Cell:
    layer: int
    seed: int
    output: Path

    @property
    def cell_id(self) -> str:
        return f"layer-{self.layer:02d}:seed-{self.seed}"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expand_cells(manifest: dict[str, Any], repo_root: Path) -> list[Cell]:
    seed = int(manifest["seed"])
    template = manifest["paths"]["output_template"]
    return [
        Cell(int(layer), seed, repo_root / template.format(layer=layer, seed=seed))
        for layer in manifest["candidate_layers"]
    ]


def validate_manifest(manifest: dict[str, Any], repo_root: Path) -> list[Cell]:
    if manifest.get("protocol") != "g3_layer_selection_v1":
        raise ValueError("unknown G3 protocol")
    if manifest.get("stage") != "development_selection" or manifest.get("evidence_status") != "development_only":
        raise ValueError("G3 must remain development-only")
    registry_path = repo_root / manifest["registry"]
    if _sha256(registry_path) != manifest.get("registry_sha256"):
        raise ValueError("G3 registry hash mismatch")
    registry = _load(registry_path)
    g3 = next(row for row in registry["experiments"] if row["id"] == "G3")
    if g3["datasets"] != ["development_only"] or g3["methods"] != ["edit_layer_sweep"] or g3["seeds"] != 1:
        raise ValueError("G3 manifest disagrees with registry")
    if manifest.get("model_size") != "base":
        raise ValueError("G3 selection is fixed to BERT-base")
    if manifest.get("candidate_layers") != list(range(1, 12)):
        raise ValueError("G3 must sweep every valid BERT-base split boundary")
    if manifest.get("method") != {"name": "edit_layer_sweep", "implementation": "t3b_pointer"}:
        raise ValueError("G3 method mapping is invalid")
    if manifest.get("selection") != {
        "metric": "two_edit_balanced", "direction": "maximize", "tie_break": "smallest_layer"
    }:
        raise ValueError("G3 selection rule is invalid")
    if manifest.get("data_access") != {
        "allowed_splits": ["train", "validation"], "held_out_test": "forbidden", "final_eval": False
    }:
        raise ValueError("G3 data-access boundary is invalid")
    paths = manifest["paths"]
    graphs = paths["xor_graphs"]
    if len(graphs) != 20 or len(set(graphs)) != 20:
        raise ValueError("G3 requires twenty unique development graphs")
    screening = _load(repo_root / "registry" / "a1_screening_manifest.json")
    confirmatory = _load(repo_root / "registry" / "a1_confirmatory_manifest.json")
    if graphs != screening["paths"]["xor_graphs"]:
        raise ValueError("G3 is frozen to the architecture-screening development graphs")
    if set(graphs) & set(confirmatory["paths"]["xor_graphs"]):
        raise ValueError("G3 cannot use A1 confirmatory graphs")
    if "g3-development" not in paths["output_template"]:
        raise ValueError("G3 outputs must be isolated")
    cells = expand_cells(manifest, repo_root)
    if len({cell.cell_id for cell in cells}) != 11:
        raise ValueError("G3 cell identities are not unique")
    for cell in cells:
        rendered = commands(manifest, repo_root, cell)
        if "test" in " ".join(rendered["train"] + rendered["validation"]):
            raise ValueError("G3 commands cannot request test")
    return cells


def commands(manifest: dict[str, Any], repo_root: Path, cell: Cell) -> dict[str, list[str]]:
    training = manifest["training"]
    paths = manifest["paths"]
    train = [
        sys.executable, str(repo_root / "scripts" / "launch_addressed_training.py"),
        "--mode", "t3b", "--model-size", "base", "--data-root", paths["data_root"],
        "--output-dir", str(cell.output), "--sources", "xor", "--evidence-capable",
        "--epochs", str(training["epochs"]), "--batch-size", str(training["batch_size"]),
        "--seed", str(cell.seed), "--split-layer", str(cell.layer),
        "--variable-loss-weight", str(training["variable_loss_weight"]),
        "--balanced-variable-loss", "--transition-variable-loss", "--causal-task-readout",
        "--hard-pointer", "--pointer-loss-weight", str(training["pointer_loss_weight"]),
    ]
    validation = [
        sys.executable, str(repo_root / "scripts" / "launch_xor_two_edit_eval.py"),
        "--split", "validation", "--checkpoint", str(cell.output / "best.pt"),
        "--output-json", str(cell.output / "composed_validation.json"),
        "--mode", "t3b", "--batch-size", "8",
    ]
    for graph in paths["xor_graphs"]:
        artifact = paths["xor_artifact_template"].format(graph=graph)
        pair_manifest = paths["xor_manifest_template"].format(graph=graph)
        train.extend(["--xor-bundle", artifact, pair_manifest])
        validation.extend(["--bundle", artifact, pair_manifest])
    return {"train": train, "validation": validation}


def status(cell: Cell) -> str:
    if (cell.output / "g3_cell_summary.json").exists():
        return "complete"
    if (cell.output / "composed_validation.json").exists():
        return "measured_pending_finalization"
    if (cell.output / "run_summary.json").exists():
        return "trained_pending_validation"
    return "pending"


def finalize(manifest: dict[str, Any], manifest_path: Path, cell: Cell) -> dict[str, Any]:
    training_path = cell.output / "run_summary.json"
    validation_path = cell.output / "composed_validation.json"
    checkpoint_path = cell.output / "best.pt"
    if not training_path.exists() or not validation_path.exists() or not checkpoint_path.exists():
        raise RuntimeError("G3 training, validation, and checkpoint must all exist")
    training = _load(training_path)
    validation = _load(validation_path)
    config = training["configuration"]
    expected = {
        "mode": "t3b", "model_size": "base", "sources": ["xor"],
        "seed": cell.seed, "split_layer": cell.layer,
        "epochs": manifest["training"]["epochs"],
        "batch_size": manifest["training"]["batch_size"],
        "evidence_capable": True,
        "hard_pointer": True, "variable_loss_weight": manifest["training"]["variable_loss_weight"],
        "balanced_variable_loss": True, "transition_variable_loss": True,
        "causal_task_readout": True, "pointer_loss_weight": manifest["training"]["pointer_loss_weight"],
    }
    mismatched = {key: (config.get(key), value) for key, value in expected.items() if config.get(key) != value}
    if mismatched:
        raise RuntimeError(f"G3 training configuration mismatch: {mismatched}")
    checkpoint_hash = _sha256(checkpoint_path)
    if validation.get("checkpoint_sha256") != checkpoint_hash:
        raise RuntimeError("G3 validation checkpoint hash mismatch")
    if validation.get("test_evaluated") is not False or validation.get("distributed", {}).get("world_size") != 4:
        raise RuntimeError("G3 validation violates test or GPU policy")
    if validation.get("graph_count") != 20:
        raise RuntimeError("G3 validation must cover all twenty registered graphs")
    contract_path = cell.output / "cell_contract.json"
    contract = _load(contract_path)
    if (
        contract.get("cell_id") != cell.cell_id
        or contract.get("layer") != cell.layer
        or contract.get("seed") != cell.seed
        or contract.get("manifest_sha256") != _sha256(manifest_path)
        or contract.get("test_evaluated") is not False
    ):
        raise RuntimeError("G3 cell contract mismatch")
    result = {
        "protocol": "g3_layer_selection_cell_v1", "stage": "development_selection",
        "evidence_status": "development_only", "test_evaluated": False,
        "cell_id": cell.cell_id, "layer": cell.layer, "seed": cell.seed,
        "checkpoint_sha256": checkpoint_hash,
        "two_edit_balanced": validation["two_edit"]["two_edit_balanced"],
        "graph_count": validation["graph_count"], "pair_count": validation["pair_count"],
        "provenance": {
            "manifest_sha256": _sha256(manifest_path),
            "training_summary_sha256": _sha256(training_path),
            "validation_summary_sha256": _sha256(validation_path),
            "cell_contract_sha256": _sha256(contract_path),
            "world_size": 4,
        },
    }
    (cell.output / "g3_cell_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def report(cells: list[Cell]) -> dict[str, Any]:
    rows = [_load(cell.output / "g3_cell_summary.json") for cell in cells]
    if len(rows) != 11 or any(row.get("test_evaluated") is not False for row in rows):
        raise RuntimeError("G3 sweep is incomplete or invalid")
    ranked = sorted(rows, key=lambda row: (-row["two_edit_balanced"], row["layer"]))
    return {
        "protocol": "g3_layer_selection_report_v1", "stage": "development_selection",
        "test_evaluated": False, "completed_cells": 11,
        "selected_layer": ranked[0]["layer"], "selection_metric": "two_edit_balanced",
        "ranked_layers": [{"layer": row["layer"], "two_edit_balanced": row["two_edit_balanced"]} for row in ranked],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path(__file__).parents[1] / "registry" / "g3_layer_selection_manifest.json")
    parser.add_argument("--action", choices=("validate", "status", "command", "launch", "finalize", "report"), default="status")
    parser.add_argument("--cell-id")
    args = parser.parse_args(argv)
    manifest_path = args.manifest.resolve()
    repo_root = manifest_path.parents[1]
    manifest = _load(manifest_path)
    cells = validate_manifest(manifest, repo_root)
    if args.action == "validate":
        print(json.dumps({"valid": True, "cell_count": len(cells)}, sort_keys=True)); return 0
    if args.action == "status":
        counts: dict[str, int] = {}
        for cell in cells: counts[status(cell)] = counts.get(status(cell), 0) + 1
        print(json.dumps({"counts": counts}, sort_keys=True)); return 0
    if args.action == "report":
        value = report(cells)
        output = repo_root / "reports" / "G3_LAYER_SELECTION.json"
        output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(value, sort_keys=True)); return 0
    matches = [cell for cell in cells if cell.cell_id == args.cell_id]
    if len(matches) != 1:
        raise ValueError("action requires one registered --cell-id")
    cell = matches[0]
    if args.action == "command":
        print(json.dumps(commands(manifest, repo_root, cell), indent=2)); return 0
    if args.action == "finalize":
        print(json.dumps(finalize(manifest, manifest_path, cell), sort_keys=True)); return 0
    visible = [item for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item]
    if len(visible) != 4:
        raise RuntimeError("BERT-base G3 launch requires exactly four visible GPUs")
    if cell.output.exists() and any(cell.output.iterdir()):
        raise RuntimeError("G3 output directory is nonempty; refusing reuse")
    cell.output.mkdir(parents=True)
    contract = {
        "protocol": "g3_layer_selection_cell_contract_v1", "cell_id": cell.cell_id,
        "layer": cell.layer, "seed": cell.seed, "manifest_sha256": _sha256(manifest_path),
        "test_evaluated": False,
    }
    (cell.output / "cell_contract.json").write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rendered = commands(manifest, repo_root, cell)
    subprocess.run(rendered["train"], cwd=repo_root, check=True)
    subprocess.run(rendered["validation"], cwd=repo_root, check=True)
    print(json.dumps(finalize(manifest, manifest_path, cell), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
