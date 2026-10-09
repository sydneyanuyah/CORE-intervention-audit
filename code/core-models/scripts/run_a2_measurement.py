#!/usr/bin/env python3
"""Validate, execute, and finalize manifest-defined A2 measurement cells."""

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
    control: str
    a2_control: str
    seed: int
    source: Path
    output: Path
    checkpoint_sha256: str
    contract_sha256: str
    source_summary_sha256: str
    source_composed_sha256: str

    @property
    def cell_id(self) -> str:
        return f"{self.family}:{self.control}:{self.seed}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def expand(manifest: dict[str, Any], root: Path) -> list[Cell]:
    catalog = load(root / manifest["source"]["catalog"])
    controls = manifest["controls"]
    cells = []
    for source in catalog["cells"]:
        for control, spec in controls.items():
            cells.append(Cell(
                family=source["family"], control=control,
                a2_control=spec["a2_control"], seed=int(source["seed"]),
                source=root / source["source_directory"],
                output=root / manifest["paths"]["output_template"].format(
                    family=source["family"], control=control, seed=source["seed"]
                ),
                checkpoint_sha256=source["checkpoint_sha256"],
                contract_sha256=source["cell_contract_sha256"],
                source_summary_sha256=source["cell_summary_sha256"],
                source_composed_sha256=source["composed_validation_sha256"],
            ))
    return cells


def validate(manifest: dict[str, Any], root: Path) -> list[Cell]:
    if manifest.get("protocol") != "a2_frozen_checkpoint_measurement_v1":
        raise ValueError("unknown A2 manifest protocol")
    registry_path = root / manifest["registry"]
    if sha256(registry_path) != manifest["registry_sha256"]:
        raise ValueError("A2 registry hash mismatch")
    experiment = next(x for x in load(registry_path)["experiments"] if x["id"] == "A2")
    if set(manifest["controls"]) != set(experiment["methods"]):
        raise ValueError("A2 controls disagree with registry")
    source = manifest["source"]
    for key, hash_key in (
        ("report", "report_sha256"), ("manifest", "manifest_sha256"),
        ("catalog", "catalog_sha256"),
    ):
        path = root / source[key]
        if not path.is_file() or sha256(path) != source[hash_key]:
            raise ValueError(f"frozen A2 source {key} hash mismatch")
    report = load(root / source["report"])
    if (
        report.get("completed_cells") != 520
        or report.get("confirmatory") is not True
        or report.get("test_evaluated") is not False
    ):
        raise ValueError("A2 source report is not complete no-test A1 evidence")
    if manifest.get("world_size") != 4 or manifest.get("model_size") != "base":
        raise ValueError("A2 requires exactly four-GPU BERT-base evaluation")
    if manifest.get("mode") != "t2b" or manifest.get("test_evaluated") is not False:
        raise ValueError("A2 must be T2-b and no-test")
    if manifest.get("data_access") != {
        "allowed_split": "validation", "held_out_test": "forbidden", "final_eval": False,
    }:
        raise ValueError("A2 data-access boundary is invalid")
    catalog = load(root / source["catalog"])
    if (
        catalog.get("test_evaluated") is not False
        or catalog.get("world_size") != 4
        or len(catalog.get("cells", [])) != 100
    ):
        raise ValueError("A2 source catalog is inadmissible")
    cells = expand(manifest, root)
    if len(cells) != manifest.get("registered_cell_count") or len(cells) != 400:
        raise ValueError("A2 must expand to exactly 400 cells")
    if len({cell.cell_id for cell in cells}) != 400:
        raise ValueError("A2 cell IDs are not unique")
    if set(manifest["families"]) != {cell.family for cell in cells}:
        raise ValueError("A2 family coverage mismatch")
    if set(manifest["seeds"]) != {cell.seed for cell in cells}:
        raise ValueError("A2 seed coverage mismatch")
    if manifest["analysis"] != {
        "experimental_unit": "graph",
        "primary": "t2b_editor_zeroed-minus-no_instruction_baseline",
        "primary_expectation": "mechanical_equivalence",
        "secondary": "t2b_editor_active-minus-t2b_editor_zeroed",
        "prompting": "reported_separately",
        "intervals": ["paired_graph_bootstrap_95", "student_t_95"],
    }:
        raise ValueError("A2 analysis contract changed")
    return cells


def command(manifest: dict[str, Any], root: Path, cell: Cell) -> list[str]:
    paths = manifest["paths"]
    cmd = [
        sys.executable, str(root / "scripts/launch_xor_two_edit_eval.py"),
        "--family", cell.family, "--data-root", paths["data_root"],
        "--split", "validation", "--checkpoint", str(cell.source / "best.pt"),
        "--tokenizer", str(cell.source / "tokenizer"),
        "--output-json", str(cell.output / "measurement.json"),
        "--mode", "t2b", "--a2-control", cell.a2_control,
        "--batch-size", "4" if cell.family == "com2" else "8",
    ]
    if cell.family == "xor":
        for graph in paths["xor_graphs"]:
            cmd.extend(["--bundle", paths["xor_artifact_template"].format(graph=graph),
                        paths["xor_manifest_template"].format(graph=graph)])
    elif cell.family == "cladder":
        cmd.extend(["--bundle", paths["cladder_artifact"], paths["cladder_manifest"]])
    elif cell.family == "wiqa":
        cmd.extend(["--bundle", paths["wiqa_queue"], paths["wiqa_truth"]])
    elif cell.family == "ccrgb":
        cmd.extend(["--bundle", paths["ccrgb_artifact"], paths["ccrgb_manifest"]])
    elif cell.family == "com2":
        cmd.extend(["--bundle", paths["com2_queue"], paths["com2_truth"],
                    "--provenance", paths["com2_provenance"]])
    else:
        raise ValueError(f"unsupported A2 family {cell.family}")
    if cell.family != "xor":
        cmd.extend(["--paraphrase-catalog", paths["paraphrase_catalog"]])
    if "test" in cmd:
        raise RuntimeError("A2 command cannot request test")
    return cmd


def verify_source(cell: Cell) -> dict[str, Any]:
    contract = cell.source / "cell_contract.json"
    summary = cell.source / "cell_summary.json"
    composed = cell.source / "composed_validation.json"
    if sha256(contract) != cell.contract_sha256:
        raise RuntimeError("A2 source contract hash mismatch")
    if sha256(summary) != cell.source_summary_sha256:
        raise RuntimeError("A2 source summary hash mismatch")
    if sha256(composed) != cell.source_composed_sha256:
        raise RuntimeError("A2 source active measurement hash mismatch")
    source = load(summary)
    if (
        source.get("cell_id") != f"{cell.family}:t2b_encode_only:{cell.seed}"
        or source.get("checkpoint_sha256") != cell.checkpoint_sha256
        or source.get("provenance", {}).get("world_size") != 4
        or source.get("provenance", {}).get("test_evaluated") is not False
    ):
        raise RuntimeError("A2 source cell provenance is inadmissible")
    return source


def materialize_active(cell: Cell) -> None:
    verify_source(cell)
    result = load(cell.source / "composed_validation.json")
    result.update({
        "protocol": "a2_frozen_checkpoint_measurement_v1",
        "a2_control": "active", "a2_reused_source": True,
        "source_composed_validation_sha256": cell.source_composed_sha256,
    })
    cell.output.mkdir(parents=True, exist_ok=True)
    (cell.output / "measurement.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def finalize(manifest_path: Path, cell: Cell) -> dict[str, Any]:
    source = verify_source(cell)
    measurement_path = cell.output / "measurement.json"
    if not measurement_path.is_file():
        raise RuntimeError("A2 measurement is missing")
    result = load(measurement_path)
    if (
        result.get("checkpoint_sha256") != cell.checkpoint_sha256
        or result.get("a2_control") != cell.a2_control
        or result.get("split") != "validation"
        or result.get("test_evaluated") is not False
        or result.get("distributed", {}).get("world_size") != 4
        or result.get("graph_count", 0) < 1
    ):
        raise RuntimeError("A2 measurement violates its cell contract")
    final = {
        "protocol": "a2_complete_measurement_cell_v1",
        "cell_id": cell.cell_id, "family": cell.family,
        "control": cell.control, "a2_control": cell.a2_control, "seed": cell.seed,
        "source_a1_cell_id": source["cell_id"],
        "checkpoint_sha256": cell.checkpoint_sha256,
        "source_cell_contract_sha256": cell.contract_sha256,
        "source_cell_summary_sha256": cell.source_summary_sha256,
        "source_composed_validation_sha256": cell.source_composed_sha256,
        "measurement_sha256": sha256(measurement_path),
        "manifest_sha256": sha256(manifest_path),
        "graph_count": result["graph_count"], "pair_count": result["pair_count"],
        "two_edit": result["two_edit"], "variables": result["variables"],
        "test_evaluated": False,
        "distributed": result["distributed"],
    }
    (cell.output / "cell_summary.json").write_text(
        json.dumps(final, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return final


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path(__file__).parents[1] / "registry/a2_measurement_manifest.json")
    parser.add_argument(
        "--action",
        choices=("validate", "status", "command", "launch", "finalize", "materialize-active"),
        default="status",
    )
    parser.add_argument("--cell-id")
    args = parser.parse_args(argv)
    root = Path(__file__).parents[1].resolve()
    manifest_path = args.manifest.resolve()
    manifest = load(manifest_path)
    cells = validate(manifest, root)
    if args.action == "validate":
        print(json.dumps({"cells": len(cells), "sources": len(cells) // 4}))
        return 0
    if args.action == "status":
        counts: dict[str, int] = {}
        for cell in cells:
            state = "complete" if (cell.output / "cell_summary.json").is_file() else "measured_pending_finalization" if (cell.output / "measurement.json").is_file() else "pending"
            counts[state] = counts.get(state, 0) + 1
        print(json.dumps({"counts": counts}, sort_keys=True))
        return 0
    if args.action == "materialize-active":
        completed = 0
        for cell in cells:
            if cell.a2_control != "active" or (cell.output / "cell_summary.json").exists():
                continue
            materialize_active(cell)
            finalize(manifest_path, cell)
            completed += 1
        print(json.dumps({"materialized_active_cells": completed}))
        return 0
    matches = [cell for cell in cells if cell.cell_id == args.cell_id]
    if len(matches) != 1:
        raise ValueError(f"unknown A2 cell {args.cell_id!r}")
    cell = matches[0]
    if args.action == "command":
        print(json.dumps(command(manifest, root, cell)))
    elif args.action == "finalize":
        print(json.dumps(finalize(manifest_path, cell), sort_keys=True))
    else:
        if (cell.output / "cell_summary.json").exists():
            raise RuntimeError("refusing duplicate completed A2 cell")
        if cell.a2_control == "active":
            materialize_active(cell)
        else:
            cell.output.mkdir(parents=True, exist_ok=True)
            subprocess.run(command(manifest, root, cell), check=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})
        print(json.dumps(finalize(manifest_path, cell), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
