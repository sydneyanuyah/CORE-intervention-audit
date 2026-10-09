#!/usr/bin/env python3
"""Four-rank, validation-only T3 inverted-instruction sensitivity measurement."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import train_addressed
from core_bert.benchmark_data import BenchmarkDataset, BenchmarkExample
from core_bert.t3_cladder_perturbations import inverted_instruction


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sha(args.manifest) != args.manifest_sha256:
        raise ValueError("T3 positive-control manifest hash mismatch")
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("protocol") != "t3_inverted_instruction_positive_control_v1":
        raise ValueError("wrong T3 positive-control protocol")
    matches = [row for row in manifest["cells"] if row["cell_id"] == args.cell_id]
    if len(matches) != 1:
        raise ValueError("T3 positive-control cell is not uniquely registered")
    cell = matches[0]
    checkpoint = Path(cell["checkpoint"])
    artifact_path = Path(manifest["source_artifact"])
    for path, expected in ((checkpoint, cell["checkpoint_sha256"]), (artifact_path, manifest["source_artifact_sha256"])):
        if sha(path) != expected:
            raise ValueError(f"T3 positive-control input hash mismatch: {path}")
    if json.loads(Path(cell["source_contract"]).read_text()).get("test_evaluated") is not False:
        raise ValueError("T3 positive-control source is not validation-only")
    artifact = json.loads(artifact_path.read_text())
    if artifact.get("test_evaluated") is not False:
        raise ValueError("T3 positive-control artifact is not validation-only")
    clean = [row for row in artifact["components"] if row.get("two_edit_metadata", {}).get("split") == "validation"]
    examples = []
    for record in clean:
        value = inverted_instruction(record)
        model_id = record["two_edit_metadata"]["model_id"]
        examples.append(BenchmarkExample(
            record=value,
            record_id=value["id"],
            source="cladder",
            label=value["intervened"]["answer"],
            intervention_kind=value["intervention"]["kind"],
            graph_group_id=f"cladder:model:{model_id}",
            world_group_id=f"cladder:model:{model_id}",
        ))
    dataset = BenchmarkDataset(sorted(examples, key=lambda row: row.record_id))
    if len(dataset) != manifest["records"]:
        raise ValueError("T3 positive-control record count mismatch")
    train_addressed.load_benchmark_split = lambda *unused_args, **unused_kwargs: dataset
    original_check = train_addressed._validate_checkpoint_configuration

    def architectural_check(checkpoint_data, parsed):
        stored = checkpoint_data.get("configuration", {})
        required = {"mode":"t3b", "model_size":"base", "model":"google-bert/bert-base-uncased", "world_slots":30, "rank":16, "variable_loss_weight":1.0, "balanced_variable_loss":True, "transition_variable_loss":True, "causal_task_readout":True, "hard_pointer":True, "pointer_loss_weight":1.0}
        for key, expected in required.items():
            if str(stored.get(key)) != str(expected) or str(getattr(parsed, key)) != str(expected):
                raise ValueError(f"T3 positive-control checkpoint architecture mismatch: {key}")

    train_addressed._validate_checkpoint_configuration = architectural_check
    try:
        return train_addressed.main([
            "--mode", "t3b", "--model-size", "base", "--data-root", "unused",
            "--output-dir", str(args.output), "--sources", "cladder", "--seed", str(cell["seed"]),
            "--batch-size", "16", "--max-length", "512", "--rank", "16",
            "--variable-loss-weight", "1", "--balanced-variable-loss", "--transition-variable-loss",
            "--causal-task-readout", "--hard-pointer", "--pointer-loss-weight", "1",
            "--measure-checkpoint", str(checkpoint),
            "--prediction-jsonl", str(args.output / "predictions.jsonl"),
        ])
    finally:
        train_addressed._validate_checkpoint_configuration = original_check


if __name__ == "__main__":
    raise SystemExit(main())
