#!/usr/bin/env python3
"""Four-rank frozen-checkpoint T3 CLadder robustness measurement."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import train_addressed
from core_bert.benchmark_data import BenchmarkDataset, BenchmarkExample
from core_bert.t3_cladder_perturbations import apply_manifest_pair, sha256_json


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cell-id", required=True); p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--manifest-sha256", required=True); p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(); manifest = json.loads(args.manifest.read_text())
    if sha(args.manifest) != args.manifest_sha256: raise ValueError("T3 manifest hash mismatch")
    matches = [row for row in manifest["cells"] if row["cell_id"] == args.cell_id]
    if len(matches) != 1: raise ValueError("T3 cell is not uniquely registered")
    cell = matches[0]; checkpoint = Path(cell["checkpoint"]); artifact_path = Path(manifest["source_artifact"]); pairs_path = Path(manifest["perturbation_manifest"])
    for path, expected in ((checkpoint, cell["checkpoint_sha256"]), (artifact_path, manifest["source_artifact_sha256"]), (pairs_path, manifest["perturbation_manifest_sha256"])):
        if sha(path) != expected: raise ValueError(f"T3 input hash mismatch: {path}")
    artifact = json.loads(artifact_path.read_text()); pair_manifest = json.loads(pairs_path.read_text())
    if artifact.get("test_evaluated") is not False or pair_manifest.get("test_evaluated") is not False: raise ValueError("T3 test isolation failed")
    clean = {row["id"]: row for row in artifact["components"] if row.get("two_edit_metadata", {}).get("split") == "validation"}
    selected = [row for row in pair_manifest["pairs"] if row["dataset"] == "variable_rename"]
    if cell["view"] != "clean": selected = [row for row in pair_manifest["pairs"] if row["dataset"] == cell["view"]]
    examples = []
    for pair in selected:
        record = clean[pair["record_id"]]
        if sha256_json(record) != pair["clean_sha256"]: raise ValueError("T3 clean record identity mismatch")
        value = record if cell["view"] == "clean" else apply_manifest_pair(record, pair)
        examples.append(BenchmarkExample(record=value, record_id=value["id"], source="cladder", label=value["intervened"]["answer"], intervention_kind=value["intervention"]["kind"], graph_group_id=f"cladder:model:{pair['model_id']}", world_group_id=pair["world_group_id"]))
    dataset = BenchmarkDataset(sorted(examples, key=lambda row: row.record_id))
    if len(dataset) != manifest["records_per_view"]: raise ValueError("T3 view count mismatch")
    train_addressed.load_benchmark_split = lambda *unused_args, **unused_kwargs: dataset
    original_check = train_addressed._validate_checkpoint_configuration
    def architectural_check(checkpoint_data, parsed):
        stored = checkpoint_data.get("configuration", {})
        required = {"mode":"t3b", "model_size":"base", "model":"google-bert/bert-base-uncased", "world_slots":30, "rank":16, "variable_loss_weight":1.0, "balanced_variable_loss":True, "transition_variable_loss":True, "causal_task_readout":True, "hard_pointer":True, "pointer_loss_weight":1.0}
        for key, expected in required.items():
            if str(stored.get(key)) != str(expected) or str(getattr(parsed, key)) != str(expected): raise ValueError(f"T3 checkpoint architecture mismatch: {key}")
    train_addressed._validate_checkpoint_configuration = architectural_check
    try:
        return train_addressed.main(["--mode","t3b","--model-size","base","--data-root","unused","--output-dir",str(args.output),"--sources","cladder","--seed",str(cell["seed"]),"--batch-size","16","--max-length","512","--rank","16","--variable-loss-weight","1","--balanced-variable-loss","--transition-variable-loss","--causal-task-readout","--hard-pointer","--pointer-loss-weight","1","--measure-checkpoint",str(checkpoint),"--prediction-jsonl",str(args.output/"predictions.jsonl")])
    finally:
        train_addressed._validate_checkpoint_configuration = original_check


if __name__ == "__main__": raise SystemExit(main())
