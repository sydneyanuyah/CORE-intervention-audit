#!/usr/bin/env python3
"""Fail-closed four-rank F2 execution for open-text Com2 interventions."""

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

from core_bert.addressed_data import AddressedBatchCollator
from core_bert.addressed_operator import AddressedGatedOperator
from core_bert.benchmark_data import load_benchmark_split
from core_bert.com2_two_edit_adapter import load_com2_two_edit_bundle
from core_bert.composed_reader import ComposedOpenTextModel, ComposedScientificCollator, ComposedScientificExecutor
from core_bert.open_text_outputs import build_open_text_batch, normalize_open_text, sequence_cross_entropy
from core_bert.scientific_reader import tokenize_scientific_fields
from core_bert.two_edit import two_edit_balanced_metrics
from evaluate_xor_two_edit import _load_model
from train_addressed import _task_dataset, augment_com2_identity_exposure
from train_f1 import setup
from train_f2_baselines import install_lora, lora_parameter_count, select_lora_rank
from train_f2_real import load_json, move, sha256, source_seed


class TextO2Additive(nn.Module):
    """Ungated rank-r additive residual from a textual T2 instruction."""

    def __init__(self, hidden_size: int, rank: int = 16) -> None:
        super().__init__()
        self.down = nn.Linear(3 * hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, slots: torch.Tensor, instruction: torch.Tensor) -> torch.Tensor:
        expanded = instruction[:, None, :].expand(-1, slots.shape[1], -1)
        return slots + self.up(torch.tanh(self.down(torch.cat([slots, expanded], dim=-1))))


def validate_contract(args: argparse.Namespace) -> Mapping[str, Any]:
    frozen = (
        (args.manifest, args.expected_manifest_sha256, "manifest"),
        (args.cells, args.expected_cells_sha256, "cells"),
        (args.source_catalog, args.expected_source_catalog_sha256, "source catalog"),
        (args.amendment, args.expected_amendment_sha256, "Com2 amendment"),
        (args.queue, args.expected_queue_sha256, "annotation queue"),
        (args.truth, args.expected_truth_sha256, "authoritative truth"),
        (args.provenance, args.expected_provenance_sha256, "truth provenance"),
    )
    for path, expected, name in frozen:
        if sha256(path) != expected:
            raise ValueError(f"F2 Com2 {name} hash mismatch")
    data_files = (
        ("records/com2_intervention.jsonl", args.expected_intervention_records_sha256),
        ("records/com2_counterfactual.jsonl", args.expected_counterfactual_records_sha256),
        ("splits/com2_intervention.train.txt", args.expected_intervention_train_sha256),
        ("splits/com2_counterfactual.train.txt", args.expected_counterfactual_train_sha256),
        ("splits/com2_intervention.validation.txt", args.expected_intervention_validation_sha256),
        ("splits/com2_counterfactual.validation.txt", args.expected_counterfactual_validation_sha256),
        ("groups/com2.jsonl", args.expected_groups_sha256),
    )
    for relative, expected in data_files:
        if sha256(args.data_root / relative) != expected:
            raise ValueError(f"F2 Com2 accepted-data hash mismatch: {relative}")
    amendment = load_json(args.amendment)
    if (
        amendment.get("protocol") != "f2_com2_open_text_execution_amendment_v2"
        or amendment.get("status") != "preregistered_prelaunch"
        or amendment.get("base_manifest_sha256") != args.expected_manifest_sha256
        or amendment.get("base_cells_sha256") != args.expected_cells_sha256
        or amendment.get("world_size") != 4
        or amendment.get("allowed_splits") != ["train", "validation"]
        or amendment.get("test_evaluated") is not False
    ):
        raise ValueError("F2 Com2 execution amendment changed")
    cell_id = f"f2:com2:{args.method}:{args.seed}"
    cells = [row for row in load_json(args.cells)["cells"] if row.get("cell_id") == cell_id]
    if len(cells) != 1 or cells[0].get("world_size") != 4:
        raise ValueError(f"unregistered F2 cell {cell_id}")
    sources = [
        row for row in load_json(args.source_catalog)["cells"]
        if row.get("family") == "com2" and row.get("seed") == source_seed(args.seed)
    ]
    if len(sources) != 1 or sources[0].get("checkpoint_sha256") != args.expected_checkpoint_sha256:
        raise ValueError("F2 Com2 frozen reader mapping changed")
    if sha256(args.checkpoint) != args.expected_checkpoint_sha256:
        raise ValueError("F2 Com2 checkpoint hash mismatch")
    return sources[0]


def training_records(data_root: Path) -> list[Mapping[str, Any]]:
    dataset = load_benchmark_split(data_root, "train", sources=["com2"])
    dataset = _task_dataset(dataset, "t2b", open_text_output=True)
    dataset = augment_com2_identity_exposure(dataset)
    rows = [example.record for example in dataset]
    if not rows or any(row.get("source") != "com2" for row in rows):
        raise ValueError("F2 Com2 training records failed closed")
    return rows


def collate(tokenizer: Any, records: Sequence[Mapping[str, Any]], device: torch.device) -> dict[str, Any]:
    batch = AddressedBatchCollator(tokenizer, max_length=512)(records)
    batch.update(tokenize_scientific_fields(tokenizer, records, max_slots=30))
    text = build_open_text_batch(records, tokenizer, max_slots=30, max_tokens=32)
    batch.update({
        "variable_mask": text.variable_mask,
        "variable_label_mask": text.label_mask,
        "variable_changed_mask": text.changed_mask,
        "variable_preservation_mask": text.preservation_mask,
        "open_after_inputs": text.after_inputs,
        "open_after_targets": text.after_targets,
    })
    return move(batch, device)


def open_text_loss(head: nn.Module, slots: torch.Tensor, side: Mapping[str, Any]) -> torch.Tensor:
    mask = side["variable_label_mask"].bool()
    logits = head(slots, side["open_after_inputs"], mask)
    sequence = sequence_cross_entropy(logits, side["open_after_targets"][mask])
    matrix = sequence.new_zeros(mask.shape)
    matrix[mask] = sequence
    groups = []
    for name in ("variable_changed_mask", "variable_preservation_mask"):
        selected = matrix[side[name].bool() & mask]
        if len(selected):
            groups.append(selected.mean())
    if not groups:
        raise ValueError("F2 Com2 batch has no supervised event")
    return torch.stack(groups).mean()


def t2_slots(reader: nn.Module, editor: nn.Module, side: Mapping[str, Any]) -> torch.Tensor:
    result = reader.forward_t2(
        input_ids=side["input_ids"], attention_mask=side["attention_mask"],
        command_tokens=side["command_tokens"], do_tokens=side["do_tokens"],
        slot_name_input_ids=side["slot_name_input_ids"],
        slot_name_attention_mask=side["slot_name_attention_mask"], slot_mask=side["slot_mask"],
        question_input_ids=side["question_input_ids"],
        question_attention_mask=side["question_attention_mask"],
        editor=editor, mode="t2b",
    )
    return result.decoded_slots if result.decoded_slots is not None else result.edited_slots


def prompting_slots(reader: nn.Module, side: Mapping[str, Any]) -> torch.Tensor:
    trunk, valid = reader._trunk(
        side["input_ids"], side["attention_mask"], side["slot_name_input_ids"],
        side["slot_name_attention_mask"], side["slot_mask"],
    )
    _, decoded = reader._decode_question(
        trunk, valid, trunk[:, -30:], side["question_input_ids"],
        side["question_attention_mask"], return_slots=True, isolate_slots=True,
    )
    return decoded


class LoRAReadout(nn.Module):
    def __init__(self, reader: nn.Module) -> None:
        super().__init__(); self.reader = reader

    def forward(self, side: Mapping[str, Any]) -> torch.Tensor:
        return prompting_slots(self.reader, side)


@torch.no_grad()
def evaluate_operator(reader: nn.Module, head: nn.Module, editor: nn.Module, tokenizer: Any, bundle: Any, device: torch.device) -> dict[str, Any]:
    batch = ComposedScientificCollator(tokenizer, bundle.records, max_length=512)(bundle.examples)
    batch = type(batch)(batch.examples, move(batch.first, device), move(batch.second, device))
    model = ComposedOpenTextModel(ComposedScientificExecutor(reader, editor, "t2b"), head, max_tokens=32)
    output = model(batch)
    generated = tokenizer.batch_decode(output.generated_token_ids.cpu(), skip_special_tokens=True)
    predictions, cursor = {}, 0
    for example in bundle.examples:
        width = len(example.gold_outputs)
        predictions[example.pair_id] = [normalize_open_text(x) for x in generated[cursor:cursor + width]]
        cursor += width
    return two_edit_balanced_metrics(bundle.examples, predictions)


def combined_records(bundle: Any) -> list[Mapping[str, Any]]:
    rows = []
    for example in bundle.examples:
        first, second = bundle.records[example.first_record_id], copy.deepcopy(bundle.records[example.second_record_id])
        second["id"] = f"{example.pair_id}:combined-prompt"
        second["intervention"] = dict(second["intervention"])
        second["intervention"]["text"] = first["intervention"]["text"] + " Then " + second["intervention"]["text"]
        rows.append(second)
    return rows


@torch.no_grad()
def evaluate_prompting(reader: nn.Module, head: nn.Module, tokenizer: Any, bundle: Any, device: torch.device) -> dict[str, Any]:
    side = collate(tokenizer, combined_records(bundle), device)
    generated = head.generate(prompting_slots(reader, side), side["slot_mask"], max_tokens=32)
    text = tokenizer.batch_decode(generated.cpu(), skip_special_tokens=True)
    predictions, cursor = {}, 0
    for example in bundle.examples:
        width = len(example.gold_outputs)
        predictions[example.pair_id] = [normalize_open_text(x) for x in text[cursor:cursor + width]]
        cursor += width
    return two_edit_balanced_metrics(bundle.examples, predictions)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("o2", "o3", "prompting", "lora_matched"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    for name in ("checkpoint", "manifest", "cells", "source-catalog", "amendment", "queue", "truth", "provenance"):
        parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument(f"--expected-{name.replace('-', '_')}-sha256".replace("_", "-"), required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    for name in ("intervention-records", "counterfactual-records", "intervention-train", "counterfactual-train", "intervention-validation", "counterfactual-validation", "groups"):
        parser.add_argument(f"--expected-{name}-sha256", required=True)
    args = parser.parse_args()
    source = validate_contract(args)
    local_rank, rank_id = setup("f2", 4)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(args.seed + rank_id)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    loaded = _load_model(SimpleNamespace(family="com2", mode="t2b", a2_control="active", model=None), checkpoint, tokenizer, device)
    reader, head = loaded.executor.reader, loaded.open_text_head
    for parameter in loaded.parameters(): parameter.requires_grad_(False)
    for parameter in head.parameters(): parameter.requires_grad_(True)
    head_wrapped = DDP(head.to(device), device_ids=[local_rank], broadcast_buffers=False)
    head_parameters = list(head_wrapped.parameters())
    rows = training_records(args.data_root)
    history: list[dict[str, Any]] = []
    extra: dict[str, Any] = {}
    editor: nn.Module | None = None
    if args.method in {"o2", "o3"}:
        raw = TextO2Additive(reader.hidden_size, 16) if args.method == "o2" else AddressedGatedOperator(reader.hidden_size, 16)
        wrapped = DDP(raw.to(device), device_ids=[local_rank], broadcast_buffers=False)
        trainable = list(wrapped.parameters()) + head_parameters
        optimizer = torch.optim.AdamW(trainable, lr=5e-3, weight_decay=1e-4)
        mode = "operator"
    elif args.method == "lora_matched":
        target_count = sum(p.numel() for p in AddressedGatedOperator(reader.hidden_size, 16).parameters())
        lora_rank = select_lora_rank(reader.reader, target_count)
        selected_count = lora_parameter_count(reader.reader, lora_rank)
        install_lora(reader.reader, lora_rank)
        wrapped = DDP(LoRAReadout(reader).to(device), device_ids=[local_rank], broadcast_buffers=False)
        method_parameters = [p for p in wrapped.parameters() if p.requires_grad]
        if sum(p.numel() for p in method_parameters) != selected_count:
            raise RuntimeError("F2 Com2 matched-LoRA parameter count mismatch")
        trainable = method_parameters + head_parameters
        optimizer = torch.optim.AdamW(trainable, lr=1e-3, weight_decay=1e-4)
        extra = {"lora_rank": lora_rank, "trainable_parameter_count": selected_count, "o3_parameter_count": target_count, "parameter_count_delta": selected_count - target_count}
        mode = "lora"
    else:
        wrapped = None; trainable = head_parameters
        optimizer = torch.optim.AdamW(trainable, lr=1e-3, weight_decay=1e-4)
        mode = "prompting"
    if optimizer is not None:
        generator = torch.Generator().manual_seed(args.seed * 100 + rank_id)
        for step in range(500):
            selected = [rows[i] for i in torch.randint(0, len(rows), (8,), generator=generator).tolist()]
            side = collate(tokenizer, selected, device)
            if mode == "operator":
                slots = t2_slots(reader, wrapped, side)
            elif mode == "lora":
                slots = wrapped(side)
            else:
                slots = prompting_slots(reader, side)
            loss = open_text_loss(head_wrapped, slots, side)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0); optimizer.step()
            if step % 100 == 0 or step == 499:
                mean = loss.detach().clone(); dist.all_reduce(mean); mean /= 4
                history.append({"step": step + 1, "loss": float(mean)})
        if mode == "operator": editor = wrapped.module
    dist.barrier()
    if rank_id == 0:
        if args.output_dir.exists(): raise FileExistsError(args.output_dir)
        args.output_dir.mkdir(parents=True)
        bundle = load_com2_two_edit_bundle(args.queue, args.truth, args.provenance, args.data_root, split="validation")
        metrics = evaluate_operator(reader, head, editor, tokenizer, bundle, device) if editor is not None else evaluate_prompting(reader, head, tokenizer, bundle, device)
        state = {}
        if mode == "operator": state = editor.state_dict()
        elif mode == "lora": state = {n: p.detach().cpu() for n, p in wrapped.module.named_parameters() if n.endswith(("lora_a", "lora_b"))}
        torch.save({
            "method": args.method,
            "state": state,
            "open_text_head": head.state_dict(),
            "seed": args.seed,
            **extra,
            "test_evaluated": False,
        }, args.output_dir / "operator.pt")
        summary = {
            "protocol": "f2_com2_open_text_cell_v2", "cell_id": f"f2:com2:{args.method}:{args.seed}",
            "family": "com2", "method": args.method, "seed": args.seed,
            "source_a1_seed": source_seed(args.seed), "source_a1_cell_id": source["cell_id"],
            "checkpoint_sha256": args.expected_checkpoint_sha256,
            "manifest_sha256": args.expected_manifest_sha256, "cells_sha256": args.expected_cells_sha256,
            "amendment_sha256": args.expected_amendment_sha256, "history": history,
            "world_size": 4, "allowed_splits": ["train", "validation"], "two_edit": metrics,
            **extra, "runner_sha256": sha256(Path(__file__)), "test_evaluated": False,
        }
        (args.output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"cell_id": summary["cell_id"], "two_edit_balanced": metrics["two_edit_balanced"]}))
    dist.barrier(); dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
