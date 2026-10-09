#!/usr/bin/env python3
"""Four-rank F2 CLadder inverted-instruction decision-sensitivity control."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from core_bert.cladder_two_edit_adapter import CladderTwoEditBundle, load_cladder_two_edit_bundle
from core_bert.composed_reader import ComposedScientificCollator
from core_bert.t3_cladder_perturbations import inverted_instruction
from core_bert.two_edit import two_edit_balanced_metrics
from evaluate_xor_two_edit import VARIABLE_LABEL_VOCABULARY, _load_model
from train_f1 import load_core, setup
from train_f2_real import (
    CanonicalScientificOperator,
    binary_interventions,
    encode_factual,
    intervention_ids,
    move,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inverted_records(records: Mapping[str, Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    result = {}
    for record_id, record in records.items():
        changed = inverted_instruction(record)
        changed["id"] = record_id
        result[record_id] = changed
    return result


def inverted_bundle(bundle: CladderTwoEditBundle) -> CladderTwoEditBundle:
    records = inverted_records(bundle.records)
    examples = tuple(
        replace(
            example,
            first_intervention=dict(records[example.first_record_id]["intervention"]),
            second_intervention=dict(records[example.second_record_id]["intervention"]),
        )
        for example in bundle.examples
    )
    return CladderTwoEditBundle(bundle.split, records, examples)


@torch.no_grad()
def predict_shard(reader: Any, variable_head: Any, operator: Any, tokenizer: Any,
                  bundle: CladderTwoEditBundle, records: Mapping[str, Mapping[str, Any]],
                  device: torch.device, rank_id: int) -> dict[str, list[str]]:
    examples = bundle.examples[rank_id::4]
    outputs: dict[str, list[str]] = {}
    for start in range(0, len(examples), 8):
        group = examples[start:start + 8]
        batch = ComposedScientificCollator(tokenizer, records, max_length=512)(group)
        first, second = move(batch.first, device), move(batch.second, device)
        first["intervention_binary"] = binary_interventions([records[x.first_record_id] for x in group], device)
        second["intervention_binary"] = binary_interventions([records[x.second_record_id] for x in group], device)
        first_trunk, _, slots = encode_factual(reader, first)
        after_first = operator(slots, intervention_ids(first))
        second_trunk, second_valid, _ = encode_factual(reader, second)
        after_second = operator(after_first, intervention_ids(second))
        _, decoded = reader._decode_question(
            second_trunk, second_valid, after_second,
            second["question_input_ids"], second["question_attention_mask"],
            return_slots=True, isolate_slots=True,
        )
        predicted = variable_head(decoded, second["slot_mask"]).logits.argmax(-1)
        for index, example in enumerate(group):
            width = len(example.gold_outputs)
            outputs[example.pair_id] = [VARIABLE_LABEL_VOCABULARY[x] for x in predicted[index, :width].tolist()]
    return outputs


def gather(local: dict[str, list[str]]) -> dict[str, list[str]]:
    shards: list[dict[str, list[str]] | None] = [None] * 4
    dist.all_gather_object(shards, local)
    merged: dict[str, list[str]] = {}
    for shard in shards:
        if shard is None or set(merged).intersection(shard):
            raise ValueError("invalid or overlapping sensitivity shard")
        merged.update(shard)
    return merged


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sha256(args.manifest) != args.manifest_sha256:
        raise ValueError("sensitivity manifest hash mismatch")
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("protocol") != "f2_cladder_inverted_instruction_positive_control_v1":
        raise ValueError("wrong sensitivity protocol")
    if manifest.get("allowed_splits") != ["validation"] or manifest.get("test_evaluated") is not False:
        raise ValueError("sensitivity split lock changed")
    matches = [row for row in manifest["cells"] if row["cell_id"] == args.cell_id]
    if len(matches) != 1 or matches[0].get("world_size") != 4:
        raise ValueError("cell is not uniquely registered for four ranks")
    cell = matches[0]
    for key, hash_key in (
        ("reader", "reader_sha256"), ("operator", "operator_sha256"),
        ("source_summary", "source_summary_sha256"),
    ):
        if sha256(Path(cell[key])) != cell[hash_key]:
            raise ValueError(f"source hash mismatch: {key}")
    for key in ("artifact", "pair_manifest", "source_catalog", "core_components"):
        if sha256(Path(manifest[key])) != manifest[f"{key}_sha256"]:
            raise ValueError(f"manifest input hash mismatch: {key}")
    source_summary = json.loads(Path(cell["source_summary"]).read_text())
    if source_summary.get("test_evaluated") is not False or source_summary.get("method") != cell["method"]:
        raise ValueError("F2 source summary is invalid")

    local_rank, rank_id = setup("f2_instruction_sensitivity", 4)
    device = torch.device("cuda", local_rank)
    tokenizer = AutoTokenizer.from_pretrained(cell["tokenizer"])
    checkpoint = torch.load(cell["reader"], map_location="cpu", weights_only=False)
    loaded = _load_model(SimpleNamespace(family="cladder", mode="t2b", a2_control="active", model=None), checkpoint, tokenizer, device)
    reader, variable_head = loaded.executor.reader, loaded.variable_head
    reader.eval(); variable_head.eval()
    core = load_core(Path(manifest["core_components"]))
    raw = core.O2ConditionalLowRank(32, 16, reader.hidden_size, 16) if cell["method"] == "o2" else core.O3StateGated(32, reader.hidden_size, 16)
    operator_state = torch.load(cell["operator"], map_location="cpu", weights_only=False)
    if operator_state.get("method") != cell["method"] or operator_state.get("test_evaluated") is not False:
        raise ValueError("F2 operator artifact metadata mismatch")
    raw.load_state_dict(operator_state["operator"], strict=True)
    operator = CanonicalScientificOperator(raw).to(device).eval()

    bundle = load_cladder_two_edit_bundle(Path(manifest["artifact"]), Path(manifest["pair_manifest"]), split="validation")
    inverse = inverted_bundle(bundle)
    clean = gather(predict_shard(reader, variable_head, operator, tokenizer, bundle, bundle.records, device, rank_id))
    inverted = gather(predict_shard(reader, variable_head, operator, tokenizer, inverse, inverse.records, device, rank_id))
    if set(clean) != set(inverted) or len(clean) != len(bundle.examples):
        raise ValueError("sensitivity predictions do not cover the registered pairs")
    if rank_id == 0:
        changed_pairs = sum(clean[key] != inverted[key] for key in clean)
        changed_variables = sum(a != b for key in clean for a, b in zip(clean[key], inverted[key]))
        total_variables = sum(len(row) for row in clean.values())
        clean_metrics = two_edit_balanced_metrics(bundle.examples, clean)
        inverted_metrics = two_edit_balanced_metrics(bundle.examples, inverted)
        args.output.mkdir(parents=True, exist_ok=False)
        summary = {
            "protocol": manifest["protocol"], "cell_id": cell["cell_id"],
            "method": cell["method"], "seed": cell["seed"], "family": "cladder",
            "pair_count": len(clean), "variable_decision_count": total_variables,
            "changed_pair_vectors": changed_pairs, "changed_variable_decisions": changed_variables,
            "clean_two_edit_balanced": clean_metrics["two_edit_balanced"],
            "inverted_two_edit_balanced_against_retained_gold": inverted_metrics["two_edit_balanced"],
            "retained_gold_accuracy_delta": inverted_metrics["two_edit_balanced"] - clean_metrics["two_edit_balanced"],
            "reader_sha256": cell["reader_sha256"], "operator_sha256": cell["operator_sha256"],
            "manifest_sha256": args.manifest_sha256, "world_size": 4,
            "allowed_splits": ["validation"], "test_evaluated": False,
        }
        (args.output / "sensitivity_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        with (args.output / "predictions.jsonl").open("w") as stream:
            for key in sorted(clean):
                stream.write(json.dumps({"pair_id": key, "clean": clean[key], "inverted": inverted[key]}, sort_keys=True) + "\n")
        print(json.dumps(summary, sort_keys=True))
    dist.barrier(); dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
