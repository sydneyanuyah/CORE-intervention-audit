#!/usr/bin/env python3
"""Four-rank real-family F2 prompting and parameter-matched LoRA cells."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer

from core_bert.two_edit import two_edit_balanced_metrics
from evaluate_xor_two_edit import VARIABLE_LABEL_VOCABULARY, _load_model
from train_f1 import load_core, setup
from train_f2_baselines import install_lora, lora_parameter_count, select_lora_rank
from train_f2_real import (
    balanced_loss, collate_records, labels_and_masks, load_family_bundle,
    load_json, move, sha256, source_seed, wiqa_training_records,
)


class PromptingReadout(nn.Module):
    def __init__(self, reader: nn.Module, variable_head: nn.Module):
        super().__init__()
        self.reader = reader
        self.variable_head = variable_head

    def forward(self, side: Mapping[str, Any]) -> torch.Tensor:
        trunk, valid = self.reader._trunk(
            side["input_ids"], side["attention_mask"],
            side["slot_name_input_ids"], side["slot_name_attention_mask"],
            side["slot_mask"],
        )
        slots = trunk[:, -30:]
        _, decoded = self.reader._decode_question(
            trunk, valid, slots, side["question_input_ids"],
            side["question_attention_mask"], return_slots=True, isolate_slots=True,
        )
        return self.variable_head(decoded, side["slot_mask"]).logits


def validate_contract(args: argparse.Namespace) -> Mapping[str, Any]:
    expected = (
        (args.manifest, args.expected_manifest_sha256),
        (args.cells, args.expected_cells_sha256),
        (args.amendment, args.expected_amendment_sha256),
        (args.source_catalog, args.expected_source_catalog_sha256),
        (args.artifact, args.expected_artifact_sha256),
        (args.two_edit_manifest, args.expected_two_edit_manifest_sha256),
        (args.core_components, args.expected_core_sha256),
    )
    if any(sha256(path) != digest for path, digest in expected):
        raise ValueError("F2 real baseline frozen-file hash mismatch")
    amendment = load_json(args.amendment)
    if (
        amendment.get("protocol") != "f2_real_baseline_execution_amendment_v1"
        or amendment.get("base_manifest_sha256") != args.expected_manifest_sha256
        or amendment.get("base_cells_sha256") != args.expected_cells_sha256
        or amendment.get("source_seed_rule") != "2026090300 + f2_seed"
        or amendment.get("world_size") != 4
        or amendment.get("test_evaluated") is not False
    ):
        raise ValueError("F2 real baseline amendment changed")
    cell_id = f"f2:{args.family}:{args.method}:{args.seed}"
    matches = [row for row in load_json(args.cells)["cells"] if row.get("cell_id") == cell_id]
    if len(matches) != 1 or matches[0].get("world_size") != 4:
        raise ValueError(f"unregistered F2 cell {cell_id}")
    wanted = source_seed(args.seed)
    sources = [row for row in load_json(args.source_catalog)["cells"] if row.get("family") == args.family and row.get("seed") == wanted]
    if len(sources) != 1 or sources[0].get("checkpoint_sha256") != args.expected_checkpoint_sha256:
        raise ValueError("F2 real baseline source mapping changed")
    if sha256(args.checkpoint) != args.expected_checkpoint_sha256:
        raise ValueError("F2 reader checkpoint hash mismatch")
    if args.family == "wiqa":
        if args.data_root is None:
            raise ValueError("WIQA F2 requires --data-root")
        frozen_data = (
            (args.data_root / "records/wiqa.jsonl", args.expected_records_sha256),
            (args.data_root / "splits/wiqa.train.txt", args.expected_train_split_sha256),
            (args.data_root / "splits/wiqa.validation.txt", args.expected_validation_split_sha256),
            (args.data_root / "groups/wiqa.jsonl", args.expected_groups_sha256),
        )
        if any(expected is None or sha256(path) != expected for path, expected in frozen_data):
            raise ValueError("WIQA F2 accepted-data identity mismatch")
    return sources[0]


def combined_records(bundle: Any) -> list[Mapping[str, Any]]:
    rows = []
    for example in bundle.examples:
        first = bundle.records[example.first_record_id]
        second = copy.deepcopy(bundle.records[example.second_record_id])
        second["id"] = f"{example.pair_id}:combined-prompt"
        second["intervention"]["text"] = (
            first["intervention"]["text"] + " Then " + second["intervention"]["text"]
        )
        rows.append(second)
    return rows


@torch.no_grad()
def evaluate(model: nn.Module, tokenizer: Any, bundle: Any, device: torch.device) -> dict[str, Any]:
    outputs = {}
    for start in range(0, len(bundle.examples), 8):
        examples = bundle.examples[start:start + 8]
        subset = SimpleNamespace(examples=examples, records=bundle.records)
        logits = model(collate_records(tokenizer, combined_records(subset), device))
        predicted = logits.argmax(-1)
        for index, example in enumerate(examples):
            outputs[example.pair_id] = [
                VARIABLE_LABEL_VOCABULARY[value]
                for value in predicted[index, :len(example.gold_outputs)].tolist()
            ]
    return two_edit_balanced_metrics(bundle.examples, outputs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("ccrgb", "cladder", "wiqa"), default="ccrgb")
    parser.add_argument("--method", choices=("prompting", "lora_matched"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--expected-artifact-sha256", required=True)
    parser.add_argument("--two-edit-manifest", type=Path, required=True)
    parser.add_argument("--expected-two-edit-manifest-sha256", required=True)
    parser.add_argument("--source-catalog", type=Path, required=True)
    parser.add_argument("--expected-source-catalog-sha256", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--cells", type=Path, required=True)
    parser.add_argument("--expected-cells-sha256", required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--expected-amendment-sha256", required=True)
    parser.add_argument("--core-components", type=Path, required=True)
    parser.add_argument("--expected-core-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--expected-records-sha256")
    parser.add_argument("--expected-train-split-sha256")
    parser.add_argument("--expected-validation-split-sha256")
    parser.add_argument("--expected-groups-sha256")
    args = parser.parse_args()
    source = validate_contract(args)
    local_rank, rank_id = setup("f2", 4)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(args.seed + rank_id)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    loaded = _load_model(SimpleNamespace(family=args.family, mode="t2b", a2_control="active", model=None), checkpoint, tokenizer, device)
    reader, variable_head = loaded.executor.reader, loaded.variable_head
    for parameter in loaded.parameters():
        parameter.requires_grad_(False)
    history: list[dict[str, float | int]] = []
    extra: dict[str, Any] = {}
    readout = PromptingReadout(reader, variable_head).to(device)
    if args.method == "lora_matched":
        core = load_core(args.core_components)
        target_count = sum(parameter.numel() for parameter in core.O3StateGated(32, reader.hidden_size, 16).parameters())
        lora_rank = select_lora_rank(reader.reader, target_count)
        selected_count = lora_parameter_count(reader.reader, lora_rank)
        install_lora(reader.reader, lora_rank)
        model: nn.Module = DDP(readout, device_ids=[local_rank], broadcast_buffers=False)
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if sum(parameter.numel() for parameter in trainable) != selected_count:
            raise RuntimeError("real-family LoRA parameter count mismatch")
        if args.family == "wiqa":
            records = wiqa_training_records(args.data_root)
        else:
            artifact = load_json(args.artifact)
            records = [row for row in artifact["components"] if row.get("two_edit_metadata", {}).get("split") == "train"]
        optimizer = torch.optim.AdamW(trainable, lr=1e-3, weight_decay=1e-4)
        generator = torch.Generator().manual_seed(args.seed * 100 + rank_id)
        for step in range(500):
            indices = torch.randint(0, len(records), (8,), generator=generator).tolist()
            selected = [records[index] for index in indices]
            side = collate_records(tokenizer, selected, device)
            logits = model(side)
            labels, changed, preserved = labels_and_masks(selected, device)
            loss = balanced_loss(logits, labels, changed, preserved)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0); optimizer.step()
            if step % 100 == 0 or step == 499:
                mean = loss.detach().clone(); dist.all_reduce(mean); mean /= 4
                history.append({"step": step + 1, "loss": float(mean)})
        readout = model.module
        extra = {"lora_rank": lora_rank, "trainable_parameter_count": selected_count, "o3_parameter_count": target_count, "parameter_count_delta": selected_count - target_count}
    dist.barrier()
    if rank_id == 0:
        if args.output_dir.exists():
            raise FileExistsError(args.output_dir)
        args.output_dir.mkdir(parents=True)
        bundle = load_family_bundle(
            args.family, args.artifact, args.two_edit_manifest,
            split="validation", data_root=args.data_root,
        )
        metrics = evaluate(readout, tokenizer, bundle, device)
        state = {name: parameter.detach().cpu() for name, parameter in readout.named_parameters() if name.endswith(("lora_a", "lora_b"))}
        torch.save({"method": args.method, "lora": state, "seed": args.seed, **extra, "test_evaluated": False}, args.output_dir / "operator.pt")
        summary = {
            "protocol": "f2_real_baseline_cell_v1", "cell_id": f"f2:{args.family}:{args.method}:{args.seed}",
            "family": args.family, "method": args.method, "seed": args.seed,
            "source_a1_seed": source_seed(args.seed), "source_a1_cell_id": source["cell_id"],
            "checkpoint_sha256": args.expected_checkpoint_sha256,
            "manifest_sha256": args.expected_manifest_sha256, "cells_sha256": args.expected_cells_sha256,
            "amendment_sha256": args.expected_amendment_sha256,
            "artifact_sha256": args.expected_artifact_sha256, "two_edit_manifest_sha256": args.expected_two_edit_manifest_sha256,
            "accepted_data_sha256": {
                "records": args.expected_records_sha256,
                "train_split": args.expected_train_split_sha256,
                "validation_split": args.expected_validation_split_sha256,
                "groups": args.expected_groups_sha256,
            } if args.family == "wiqa" else None,
            "world_size": 4, "allowed_splits": ["train", "validation"], "history": history,
            "two_edit": metrics, **extra, "runner_sha256": sha256(Path(__file__)), "test_evaluated": False,
        }
        (args.output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"cell_id": summary["cell_id"], "two_edit_balanced": metrics["two_edit_balanced"]}))
    dist.barrier(); dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
