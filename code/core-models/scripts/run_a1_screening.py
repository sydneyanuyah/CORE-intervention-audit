#!/usr/bin/env python3
"""Validate, inspect, or execute one manifest-defined A1 cell."""

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
    family: str
    arm: str
    mode: str
    seed: int
    source: str
    pointer: bool
    output: Path
    two_edit: str | None
    stage: str

    @property
    def cell_id(self) -> str:
        return f"{self.family}:{self.arm}:{self.seed}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path,
        default=Path(__file__).parents[1] / "registry" / "a1_screening_manifest.json",
    )
    parser.add_argument(
        "--action", choices=("validate", "status", "command", "finalize", "resume", "launch"),
        default="status",
    )
    parser.add_argument("--cell-id")
    args = parser.parse_args(argv)
    if args.action in {"command", "finalize", "resume", "launch"} and not args.cell_id:
        parser.error(f"--action {args.action} requires --cell-id")
    return args


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("screening manifest must be a JSON object")
    return value


def expand_cells(manifest: dict[str, Any], repo_root: Path) -> list[Cell]:
    paths = manifest["paths"]
    overrides = manifest.get("completed_cell_overrides", {})
    cells = []
    for family, family_spec in manifest["families"].items():
        for arm, arm_spec in manifest["arms"].items():
            for seed in manifest["seeds"]:
                cell_id = f"{family}:{arm}:{seed}"
                if arm == "t0_id" and family != "xor":
                    continue
                output = overrides.get(
                    cell_id,
                    paths["output_template"].format(
                        family=family, arm=arm, seed=seed
                    ),
                )
                cells.append(Cell(
                    family, arm, arm_spec["mode"], int(seed),
                    family_spec["source"], bool(arm_spec["pointer"]),
                    repo_root / output, family_spec.get("two_edit"),
                    manifest["stage"],
                ))
    return cells


def validate_manifest(manifest: dict[str, Any], repo_root: Path) -> list[Cell]:
    protocols = {
        "a1_architecture_screening_v1": "screening",
        "a1_confirmatory_v1": "confirmatory",
    }
    if manifest.get("protocol") not in protocols:
        raise ValueError("unknown A1 manifest protocol")
    if manifest.get("stage") != protocols[manifest["protocol"]]:
        raise ValueError("A1 manifest protocol and stage disagree")
    registry = _load(repo_root / manifest["registry"])
    a1 = next(item for item in registry["experiments"] if item["id"] == "A1")
    stage = a1["stages"][manifest["stage"]]
    if len(manifest["seeds"]) != stage["seeds_per_family"]:
        raise ValueError("manifest seed count disagrees with the A1 registry")
    if set(manifest["families"]) != set(a1["datasets"]):
        raise ValueError("manifest families disagree with the A1 registry")
    if set(manifest["arms"]) != set(a1["methods"]):
        raise ValueError("manifest arms disagree with the A1 registry")
    expected_arms = {
        "t0_id": ("t0", False),
        "t1_text": ("t1", False),
        "t2b_encode_only": ("t2b", False),
        "t2a_naive": ("t2a", False),
        "t3b_pointer": ("t3b", True),
        "t3a_conditioning": ("t3a", False),
    }
    for arm, (mode, pointer) in expected_arms.items():
        if (manifest["arms"][arm]["mode"], manifest["arms"][arm]["pointer"]) != (
            mode, pointer
        ):
            raise ValueError(f"manifest implementation mapping disagrees for {arm}")
    for arm, families in a1["applicability"].items():
        actual = {
            cell.family for cell in expand_cells(manifest, repo_root)
            if cell.arm == arm
        }
        if actual != set(families):
            raise ValueError(f"manifest applicability disagrees for {arm}")
    cells = expand_cells(manifest, repo_root)
    if len(cells) != stage["applicable_runs_per_reader"]:
        raise ValueError("expanded cell count disagrees with the A1 registry")
    if len({cell.cell_id for cell in cells}) != len(cells):
        raise ValueError("expanded A1 cell IDs are not unique")
    if len(set(manifest["seeds"])) != len(manifest["seeds"]):
        raise ValueError("manifest seeds must be unique")
    if len(manifest["paths"]["xor_graphs"]) != 20 or len(
        set(manifest["paths"]["xor_graphs"])
    ) != 20:
        raise ValueError("A1 XOR evaluation requires exactly 20 unique graphs")
    if manifest["model_size"] != "base":
        raise ValueError("this A1 manifest is fixed to BERT-base")
    if manifest["stage"] == "confirmatory":
        if manifest.get("evidence_status") != "preregistered_evidence":
            raise ValueError("confirmatory evidence status is missing")
        registry_path = repo_root / manifest["registry"]
        if _sha256(registry_path) != manifest.get("registry_sha256"):
            raise ValueError("confirmatory registry hash mismatch")
        access = manifest.get("data_access")
        if access != {
            "allowed_splits": ["train", "validation"],
            "held_out_test": "forbidden",
            "final_eval": False,
        }:
            raise ValueError("confirmatory data-access boundary is invalid")
        namespace = manifest.get("seed_namespace")
        if not isinstance(namespace, str) or not namespace.startswith("a1-"):
            raise ValueError("confirmatory seed namespace is invalid")
        if manifest.get("completed_cell_overrides"):
            raise ValueError("confirmatory manifests cannot reuse screening outputs")
        if "confirmatory" not in manifest["paths"]["output_template"]:
            raise ValueError("confirmatory outputs must use an isolated directory")
        frozen = manifest.get("frozen_selection")
        if not isinstance(frozen, dict):
            raise ValueError("confirmatory manifest requires frozen screening selection")
        report_path = repo_root / frozen.get("report", "")
        if not report_path.is_file() or _sha256(report_path) != frozen.get("sha256"):
            raise ValueError("frozen screening report hash mismatch")
        report = _load(report_path)
        if (
            report.get("completed_cells") != 130
            or report.get("stage") != "architecture_screening_only"
            or report.get("confirmatory") is not False
            or report.get("test_evaluated") is not False
            or report.get("selected_arm") != frozen.get("selected_arm")
        ):
            raise ValueError("frozen screening selection is not admissible")
        if set(manifest["seeds"]) & set(report.get("seeds", [])):
            raise ValueError("confirmatory seeds must be disjoint from screening seeds")
        screening_manifest_path = repo_root / frozen.get("manifest", "")
        if (
            not screening_manifest_path.is_file()
            or _sha256(screening_manifest_path) != frozen.get("manifest_sha256")
        ):
            raise ValueError("frozen screening manifest hash mismatch")
        screening_manifest = _load(screening_manifest_path)
        if set(manifest["paths"]["xor_graphs"]) & set(screening_manifest["paths"]["xor_graphs"]):
            raise ValueError("confirmatory XOR graphs must be disjoint from screening graphs")
        for cell in cells:
            evaluation = commands(manifest, repo_root, cell)["composed_validation"]
            if evaluation is None or "validation" not in evaluation:
                raise ValueError("confirmatory evaluation must be validation-only")
            if "test" in evaluation:
                raise ValueError("confirmatory manifest cannot request held-out test")
    return cells


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cell(manifest: dict[str, Any], repo_root: Path, cell_id: str) -> Cell:
    matches = [cell for cell in validate_manifest(manifest, repo_root) if cell.cell_id == cell_id]
    if len(matches) != 1:
        raise ValueError(f"unknown manifest cell {cell_id!r}")
    return matches[0]


def commands(manifest: dict[str, Any], repo_root: Path, cell: Cell) -> dict[str, Any]:
    training = manifest["training"]
    paths = manifest["paths"]
    launch = [
        sys.executable, str(repo_root / "scripts" / "launch_addressed_training.py"),
        "--mode", cell.mode, "--model-size", manifest["model_size"],
        "--data-root", paths["data_root"], "--output-dir", str(cell.output),
        "--sources", cell.source, "--evidence-capable",
        "--epochs", str(training["epochs"]), "--batch-size", str(training["batch_size"]),
        "--seed", str(cell.seed),
        "--variable-loss-weight", str(training["variable_loss_weight"]),
        "--balanced-variable-loss", "--transition-variable-loss", "--causal-task-readout",
    ]
    if cell.pointer:
        launch.extend([
            "--hard-pointer", "--pointer-loss-weight",
            str(training["pointer_loss_weight"]),
        ])
    if cell.family == "com2":
        launch.extend([
            "--open-text-output", "--open-text-max-tokens", "32",
            "--selection-metric", "loss",
        ])
        batch_index = launch.index("--batch-size") + 1
        launch[batch_index] = "8"
    evaluation = None
    if cell.family == "xor":
        for graph in paths["xor_graphs"]:
            launch.extend([
                "--xor-bundle",
                paths["xor_artifact_template"].format(graph=graph),
                paths["xor_manifest_template"].format(graph=graph),
            ])
        evaluation = [
            sys.executable, str(repo_root / "scripts" / "launch_xor_two_edit_eval.py"),
            "--split", "validation", "--checkpoint", str(cell.output / "best.pt"),
            "--output-json", str(cell.output / "composed_validation.json"),
            "--mode", cell.mode, "--batch-size", "8",
        ]
        for graph in paths["xor_graphs"]:
            evaluation.extend([
                "--bundle", paths["xor_artifact_template"].format(graph=graph),
                paths["xor_manifest_template"].format(graph=graph),
            ])
    else:
        launch.extend([
            "--groups-dir", paths["groups_dir"],
            "--paraphrase-catalog", paths["paraphrase_catalog"],
        ])
        if cell.family == "cladder":
            evaluation = [
                sys.executable, str(repo_root / "scripts" / "launch_xor_two_edit_eval.py"),
                "--family", "cladder", "--split", "validation",
                "--checkpoint", str(cell.output / "best.pt"),
                "--output-json", str(cell.output / "composed_validation.json"),
                "--mode", cell.mode, "--batch-size", "8",
                "--bundle", paths["cladder_artifact"], paths["cladder_manifest"],
            ]
        elif cell.family == "wiqa":
            evaluation = [
                sys.executable, str(repo_root / "scripts" / "launch_xor_two_edit_eval.py"),
                "--family", "wiqa", "--data-root", paths["data_root"],
                "--split", "validation", "--checkpoint", str(cell.output / "best.pt"),
                "--output-json", str(cell.output / "composed_validation.json"),
                "--mode", cell.mode, "--batch-size", "8",
                "--bundle", paths["wiqa_queue"], paths["wiqa_truth"],
            ]
        elif cell.family == "ccrgb":
            evaluation = [
                sys.executable, str(repo_root / "scripts" / "launch_xor_two_edit_eval.py"),
                "--family", "ccrgb", "--split", "validation",
                "--checkpoint", str(cell.output / "best.pt"),
                "--output-json", str(cell.output / "composed_validation.json"),
                "--mode", cell.mode, "--batch-size", "8",
                "--bundle", paths["ccrgb_artifact"], paths["ccrgb_manifest"],
            ]
        elif cell.family == "com2":
            evaluation = [
                sys.executable, str(repo_root / "scripts" / "launch_xor_two_edit_eval.py"),
                "--family", "com2", "--data-root", paths["data_root"],
                "--split", "validation", "--checkpoint", str(cell.output / "best.pt"),
                "--output-json", str(cell.output / "composed_validation.json"),
                "--mode", cell.mode, "--batch-size", "4",
                "--bundle", paths["com2_queue"], paths["com2_truth"],
                "--provenance", paths["com2_provenance"],
            ]
        if cell.stage == "confirmatory":
            evaluation.extend(["--paraphrase-catalog", paths["paraphrase_catalog"]])
    return {"train": launch, "composed_validation": evaluation}


def cell_status(cell: Cell) -> dict[str, Any]:
    train = cell.output / "run_summary.json"
    composed = cell.output / "composed_validation.json"
    if cell.two_edit is None:
        state = "blocked_missing_authoritative_two_edit"
    elif not train.exists():
        state = "pending"
    elif not composed.exists() or composed.stat().st_size == 0:
        state = "trained_pending_composed"
    else:
        try:
            _load(composed)
        except (OSError, ValueError, json.JSONDecodeError):
            state = "trained_pending_composed"
        else:
            state = "measured_pending_finalization"
    final = cell.output / "cell_summary.json"
    if final.exists():
        value = _load(final)
        expected = cell.stage == "confirmatory"
        state = "complete" if value.get("confirmatory") is expected else "invalid_finalization"
    return {"cell_id": cell.cell_id, "state": state, "output": str(cell.output)}


def finalize(manifest: dict[str, Any], manifest_path: Path, cell: Cell) -> dict[str, Any]:
    if cell.two_edit is None:
        raise RuntimeError("real-family cell lacks authoritative composed truth")
    training_path = cell.output / "run_summary.json"
    composed_path = cell.output / "composed_validation.json"
    if not training_path.exists() or not composed_path.exists():
        raise RuntimeError("cell training and composed validation must both exist")
    training = _load(training_path)
    composed = _load(composed_path)
    configuration = training["configuration"]
    if configuration["mode"] != cell.mode or int(configuration["seed"]) != cell.seed:
        raise RuntimeError("training configuration does not match manifest cell")
    training_spec = manifest["training"]
    expected_configuration = {
        "model_size": "base",
        "sources": [cell.source],
        "epochs": training_spec["epochs"],
        "variable_loss_weight": training_spec["variable_loss_weight"],
        "balanced_variable_loss": training_spec["balanced_variable_loss"],
        "transition_variable_loss": training_spec["transition_variable_loss"],
        "causal_task_readout": training_spec["causal_task_readout"],
        "hard_pointer": cell.pointer,
        "pointer_loss_weight": training_spec["pointer_loss_weight"] if cell.pointer else 0.0,
        "evidence_capable": True,
    }
    mismatched = {
        key: (configuration.get(key), expected)
        for key, expected in expected_configuration.items()
        if configuration.get(key) != expected
    }
    if mismatched:
        raise RuntimeError(f"training configuration violates manifest: {mismatched}")
    checkpoint = cell.output / "best.pt"
    checkpoint_hash = _sha256(checkpoint)
    if composed["checkpoint_sha256"] != checkpoint_hash:
        raise RuntimeError("composed evaluation did not use the selected checkpoint bytes")
    if composed.get("test_evaluated") is not False:
        raise RuntimeError("cell report refuses any test evaluation")
    selected_rows = [row for row in training["history"] if row["selected"]]
    selected_row = next(
        row for row in reversed(selected_rows)
        if row["selection_value"] == training["best_validation_metric"]
    )
    confirmatory = cell.stage == "confirmatory"
    if confirmatory:
        contract_path = cell.output / "cell_contract.json"
        contract = _load(contract_path)
        if (
            contract.get("manifest_sha256") != _sha256(manifest_path)
            or contract.get("cell_id") != cell.cell_id
            or contract.get("stage") != "confirmatory"
        ):
            raise RuntimeError("confirmatory cell contract mismatch")
        unseen = composed.get("unseen_paraphrase")
        if (
            not isinstance(unseen, dict)
            or unseen.get("bank") != "validation"
            or unseen.get("coverage") != 1.0
        ):
            raise RuntimeError("confirmatory composed unseen-paraphrase coverage is incomplete")
        if cell.family != "xor":
            catalog_path = manifest_path.resolve().parents[1] / manifest["paths"]["paraphrase_catalog"]
            catalog_ref = manifest["paths"]["paraphrase_catalog"]
            artifact_hashes = composed.get("artifact_sha256", {})
            if artifact_hashes.get(catalog_ref) != _sha256(catalog_path):
                raise RuntimeError("composed paraphrase catalog artifact hash mismatch")
            if configuration.get("paraphrase_catalog_sha256") != unseen.get("catalog_sha256"):
                raise RuntimeError("training/composed paraphrase catalog semantic hash mismatch")
    result = {
        "protocol": "a1_complete_confirmatory_cell_v1" if confirmatory else "a1_complete_screening_cell_v1",
        "a1_evidence": False,
        "confirmatory": confirmatory,
        "evidence_status": (
            "preregistered_confirmatory_cell_pending_complete_matrix"
            if confirmatory else "architecture_screening_only"
        ),
        "stage": cell.stage,
        "cell_id": cell.cell_id,
        "family": cell.family,
        "arm": cell.arm,
        "mode": cell.mode,
        "seed": cell.seed,
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_selected_epoch": selected_row["epoch"],
        "single_edit_validation": selected_row["validation"],
        "composed_validation": {
            "graph_count": composed["graph_count"],
            "pair_count": composed["pair_count"],
            "two_edit": {
                key: value for key, value in composed["two_edit"].items()
                if key != "per_example" and (confirmatory or key != "per_graph")
            },
            "pointer": composed["pointer"],
            "unseen_paraphrase": composed.get("unseen_paraphrase"),
            "variables": composed["variables"]["overall"],
        },
        "provenance": {
            "training_summary_sha256": _sha256(training_path),
            "composed_summary_sha256": _sha256(composed_path),
            "test_evaluated": False,
            "world_size": composed["distributed"]["world_size"],
            "artifact_sha256": composed["artifact_sha256"],
            "cell_contract_sha256": (
                _sha256(cell.output / "cell_contract.json") if confirmatory else None
            ),
        },
    }
    output = cell.output / "cell_summary.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = _load(args.manifest)
    repo_root = args.manifest.resolve().parents[1]
    cells = validate_manifest(manifest, repo_root)
    if args.action == "validate":
        print(json.dumps({"valid": True, "cell_count": len(cells)}, sort_keys=True))
        return 0
    if args.action == "status":
        rows = [cell_status(cell) for cell in cells]
        counts: dict[str, int] = {}
        for row in rows:
            counts[row["state"]] = counts.get(row["state"], 0) + 1
        print(json.dumps({"counts": counts, "cells": rows}, indent=2, sort_keys=True))
        return 0
    cell = _cell(manifest, repo_root, args.cell_id)
    if args.action == "command":
        print(json.dumps(commands(manifest, repo_root, cell), indent=2))
        return 0
    if args.action == "finalize":
        print(json.dumps(finalize(manifest, args.manifest.resolve(), cell), indent=2, sort_keys=True))
        return 0
    visible = [item for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item]
    if len(visible) != 4:
        raise RuntimeError("BERT-base A1 launch requires exactly four visible GPUs")
    cell_commands = commands(manifest, repo_root, cell)
    if args.action == "resume":
        state = cell_status(cell)["state"]
        if state == "trained_pending_composed":
            subprocess.run(cell_commands["composed_validation"], cwd=repo_root, check=True)
        elif state != "measured_pending_finalization":
            raise RuntimeError(f"A1 resume requires a resumable partial cell, found {state}")
        print(json.dumps(finalize(manifest, args.manifest.resolve(), cell), indent=2, sort_keys=True))
        return 0
    if cell.stage == "confirmatory":
        if cell.output.exists() and any(cell.output.iterdir()):
            raise RuntimeError("confirmatory output directory is nonempty; refusing reuse")
        cell.output.mkdir(parents=True, exist_ok=True)
        contract = {
            "protocol": "a1_confirmatory_cell_contract_v1",
            "stage": "confirmatory",
            "model_size": manifest["model_size"],
            "seed_namespace": manifest["seed_namespace"],
            "cell_id": cell.cell_id,
            "family": cell.family,
            "arm": cell.arm,
            "seed": cell.seed,
            "manifest_sha256": _sha256(args.manifest.resolve()),
            "frozen_selection_sha256": manifest["frozen_selection"]["sha256"],
            "commands_sha256": hashlib.sha256(
                json.dumps(cell_commands, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "test_evaluated": False,
        }
        (cell.output / "cell_contract.json").write_text(
            json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    else:
        cell.output.mkdir(parents=True, exist_ok=True)
    subprocess.run(cell_commands["train"], cwd=repo_root, check=True)
    if cell_commands["composed_validation"] is None:
        raise RuntimeError("real-family launch stops before absent authoritative two-edit truth")
    subprocess.run(cell_commands["composed_validation"], cwd=repo_root, check=True)
    print(json.dumps(finalize(manifest, args.manifest.resolve(), cell), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
