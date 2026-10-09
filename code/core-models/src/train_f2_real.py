#!/usr/bin/env python3
"""Four-rank F2 O2/O3 training on executable real-family records.

The first production adapter is CCR.GB.  It freezes one A1 T2-b reader per
registered F2 seed, trains only the canonical discrete O2/O3 operator on the
artifact's train worlds, and evaluates authoritative ordered pairs on
validation.  Held-out test is neither accepted nor opened.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer

from core_bert.addressed_data import AddressedBatchCollator
from core_bert.ccrgb_two_edit_adapter import load_ccrgb_two_edit_bundle
from core_bert.cladder_two_edit_adapter import load_cladder_two_edit_bundle
from core_bert.wiqa_two_edit_adapter import load_wiqa_two_edit_bundle
from core_bert.composed_reader import ComposedScientificCollator
from core_bert.scientific_reader import tokenize_scientific_fields, variable_slot_layout
from core_bert.two_edit import two_edit_balanced_metrics
from evaluate_xor_two_edit import VARIABLE_LABEL_VOCABULARY, _load_model
from train_f1 import load_core, setup


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def source_seed(seed: int) -> int:
    if seed not in range(301, 321):
        raise ValueError("F2 seed must be 301..320")
    return 2026090300 + seed


def load_family_bundle(
    family: str, artifact: Path, manifest: Path, split: str,
    data_root: Path | None = None,
) -> Any:
    if family == "ccrgb":
        return load_ccrgb_two_edit_bundle(artifact, manifest, split=split)
    if family == "cladder":
        return load_cladder_two_edit_bundle(artifact, manifest, split=split)
    if family == "wiqa":
        if data_root is None:
            raise ValueError("WIQA F2 requires the frozen accepted-data root")
        return load_wiqa_two_edit_bundle(artifact, manifest, data_root, split=split)
    raise ValueError(f"unsupported executable F2 family: {family}")


def validate_contract(args: argparse.Namespace) -> Mapping[str, Any]:
    for path, expected, name in (
        (args.manifest, args.expected_manifest_sha256, "manifest"),
        (args.cells, args.expected_cells_sha256, "cells"),
        (args.amendment, args.expected_amendment_sha256, "real-family amendment"),
        (args.source_catalog, args.expected_source_catalog_sha256, "source catalog"),
        (args.artifact, args.expected_artifact_sha256, "CCR.GB artifact"),
        (args.two_edit_manifest, args.expected_two_edit_manifest_sha256, "CCR.GB two-edit manifest"),
        (args.core_components, args.expected_core_sha256, "canonical operators"),
    ):
        if sha256(path) != expected:
            raise ValueError(f"F2 {name} hash mismatch")
    manifest, amendment = load_json(args.manifest), load_json(args.amendment)
    if manifest.get("allowed_splits") != ["train", "validation"] or manifest.get("test_evaluated") is not False:
        raise ValueError("F2 split/test lock changed")
    if amendment.get("protocol") != "f2_real_family_execution_amendment_v1":
        raise ValueError("unknown F2 real-family amendment")
    if amendment.get("base_manifest_sha256") != args.expected_manifest_sha256 or amendment.get("base_cells_sha256") != args.expected_cells_sha256:
        raise ValueError("F2 amendment is not bound to the registered sweep")
    if amendment.get("source_seed_rule") != "2026090300 + f2_seed":
        raise ValueError("F2 source-seed mapping changed")
    cell_id = f"f2:{args.family}:{args.method}:{args.seed}"
    matches = [row for row in load_json(args.cells).get("cells", []) if row.get("cell_id") == cell_id]
    if len(matches) != 1 or matches[0].get("world_size") != 4:
        raise ValueError(f"unregistered F2 cell {cell_id}")
    wanted = source_seed(args.seed)
    sources = [row for row in load_json(args.source_catalog).get("cells", []) if row.get("family") == args.family and row.get("seed") == wanted]
    if len(sources) != 1 or sources[0].get("checkpoint_sha256") != args.expected_checkpoint_sha256:
        raise ValueError("F2 frozen reader mapping is invalid")
    if sha256(args.checkpoint) != args.expected_checkpoint_sha256:
        raise ValueError("F2 frozen reader checkpoint hash mismatch")
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


def move(values: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in values.items()}


def collate_records(tokenizer: Any, records: Sequence[Mapping[str, Any]], device: torch.device) -> dict[str, Any]:
    batch = AddressedBatchCollator(tokenizer, max_length=512)(records)
    batch.update(tokenize_scientific_fields(tokenizer, records, max_slots=30))
    moved = move(batch, device)
    moved["intervention_binary"] = binary_interventions(records, device)
    return moved


def binary_interventions(records: Sequence[Mapping[str, Any]], device: torch.device) -> torch.Tensor:
    values = []
    for record in records:
        intervention = record.get("intervention", {})
        value = intervention.get("value")
        if value not in (0, 1):
            value = {"less": 0, "more": 1}.get(intervention.get("value_token"))
        values.append(value)
    if any(value not in (0, 1) for value in values):
        raise ValueError("F2 accepts binary value-set interventions only")
    return torch.tensor(values, dtype=torch.long, device=device)


class CanonicalScientificOperator(nn.Module):
    """Apply a frozen canonical 16-token operator inside a 30-slot reader."""

    def __init__(self, operator: nn.Module):
        super().__init__()
        self.operator = operator

    def forward(self, slots: torch.Tensor, intervention_id: torch.Tensor) -> torch.Tensor:
        if slots.shape[1] != 30:
            raise ValueError("scientific reader must expose 30 slots")
        return torch.cat([self.operator(slots[:, :16], intervention_id), slots[:, 16:]], dim=1)


def intervention_ids(side: Mapping[str, Any]) -> torch.Tensor:
    binary = side["intervention_binary"]
    targets = side["target_slot"]
    if bool((targets >= 16).any()):
        raise ValueError("F2 canonical operator is frozen to 16 active world tokens")
    return targets * 2 + binary


def encode_factual(reader: Any, side: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    attention = side["attention_mask"].bool() & ~side["command_tokens"].bool() & ~side["do_tokens"].bool() & ~side["address_marker_tokens"].bool()
    trunk, valid = reader._trunk(
        side["input_ids"], attention.long(), side["slot_name_input_ids"],
        side["slot_name_attention_mask"], side["slot_mask"],
    )
    return trunk, valid, trunk[:, -30:]


def predict_single(reader: Any, variable_head: Any, operator: CanonicalScientificOperator, side: Mapping[str, Any]) -> torch.Tensor:
    trunk, valid, slots = encode_factual(reader, side)
    edited = operator(slots, intervention_ids(side))
    _, decoded = reader._decode_question(
        trunk, valid, edited, side["question_input_ids"], side["question_attention_mask"],
        return_slots=True, isolate_slots=True,
    )
    return variable_head(decoded, side["slot_mask"]).logits


def labels_and_masks(records: Sequence[Mapping[str, Any]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    labels = torch.full((len(records), 30), -100, dtype=torch.long, device=device)
    changed = torch.zeros((len(records), 30), dtype=torch.bool, device=device)
    preserved = torch.zeros_like(changed)
    for row, record in enumerate(records):
        names = variable_slot_layout(record).names
        before, after = record["factual"].get("state"), record["intervened"].get("state")
        if isinstance(before, Mapping) and isinstance(after, Mapping):
            for column, name in enumerate(names):
                labels[row, column] = VARIABLE_LABEL_VOCABULARY.index(after[name])
                changed[row, column] = after[name] != before[name]
                preserved[row, column] = not changed[row, column]
        else:
            probes = {probe["variable"]: probe for probe in record.get("probes", [])}
            if not probes or set(probes) - set(names):
                raise ValueError("F2 record has missing or out-of-graph probe supervision")
            for column, name in enumerate(names):
                if name not in probes:
                    continue
                probe = probes[name]
                labels[row, column] = VARIABLE_LABEL_VOCABULARY.index(probe["answer_after"])
                changed[row, column] = bool(probe["actually_changed"])
                preserved[row, column] = not changed[row, column]
    return labels, changed, preserved


def wiqa_training_records(data_root: Path) -> list[Mapping[str, Any]]:
    selected = set((data_root / "splits/wiqa.train.txt").read_text().split())
    records = [
        json.loads(line) for line in (data_root / "records/wiqa.jsonl").read_text().splitlines()
        if line.strip()
    ]
    result = [row for row in records if row.get("id") in selected]
    if not result or {row.get("id") for row in result} != selected:
        raise ValueError("WIQA F2 train split does not resolve exactly")
    if any(row.get("source") != "wiqa" for row in result):
        raise ValueError("WIQA F2 training records contain another family")
    return result


def balanced_loss(logits: torch.Tensor, labels: torch.Tensor, changed: torch.Tensor, preserved: torch.Tensor) -> torch.Tensor:
    per = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=-100, reduction="none").reshape_as(labels)
    return 0.5 * ((per * changed).sum(1) / changed.sum(1).clamp_min(1)).mean() + 0.5 * ((per * preserved).sum(1) / preserved.sum(1).clamp_min(1)).mean()


@torch.no_grad()
def evaluate_pairs(reader: Any, variable_head: Any, operator: CanonicalScientificOperator, tokenizer: Any, bundle: Any, device: torch.device) -> dict[str, Any]:
    outputs: dict[str, list[Any]] = {}
    # The regenerated CCR.GB validation set is intentionally large.  Keep the
    # exact example order and aggregate once, but bound accelerator memory.
    for start in range(0, len(bundle.examples), 8):
        examples = bundle.examples[start:start + 8]
        batch = ComposedScientificCollator(tokenizer, bundle.records, max_length=512)(examples)
        first, second = move(batch.first, device), move(batch.second, device)
        first["intervention_binary"] = binary_interventions(
            [bundle.records[example.first_record_id] for example in examples], device
        )
        second["intervention_binary"] = binary_interventions(
            [bundle.records[example.second_record_id] for example in examples], device
        )
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
        for index, example in enumerate(examples):
            width = len(example.gold_outputs)
            outputs[example.pair_id] = [VARIABLE_LABEL_VOCABULARY[item] for item in predicted[index, :width].tolist()]
    return two_edit_balanced_metrics(bundle.examples, outputs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("ccrgb", "cladder", "wiqa"), default="ccrgb")
    parser.add_argument("--method", choices=("o2", "o3"), required=True)
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
    reader.eval(); variable_head.eval()
    for parameter in loaded.parameters():
        parameter.requires_grad_(False)
    core = load_core(args.core_components)
    raw = core.O2ConditionalLowRank(32, 16, reader.hidden_size, 16) if args.method == "o2" else core.O3StateGated(32, reader.hidden_size, 16)
    wrapped = DDP(CanonicalScientificOperator(raw).to(device), device_ids=[local_rank], broadcast_buffers=False)
    if args.family == "wiqa":
        train_records = wiqa_training_records(args.data_root)
    else:
        artifact = load_json(args.artifact)
        train_records = [row for row in artifact.get("components", []) if row.get("two_edit_metadata", {}).get("split") == "train"]
    if not train_records or any(row.get("source") != args.family for row in train_records):
        raise ValueError(f"{args.family} F2 requires executable training components from the selected family")
    optimizer = torch.optim.AdamW(wrapped.parameters(), lr=5e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(args.seed * 100 + rank_id)
    history = []
    for step in range(500):
        selected = torch.randint(0, len(train_records), (8,), generator=generator).tolist()
        records = [train_records[index] for index in selected]
        side = collate_records(tokenizer, records, device)
        logits = predict_single(reader, variable_head, wrapped, side)
        labels, changed, preserved = labels_and_masks(records, device)
        loss = balanced_loss(logits, labels, changed, preserved)
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(wrapped.parameters(), 1.0); optimizer.step()
        if step % 100 == 0 or step == 499:
            mean = loss.detach().clone(); dist.all_reduce(mean); mean /= 4
            history.append({"step": step + 1, "loss": float(mean)})
    dist.barrier()
    if rank_id == 0:
        if args.output_dir.exists():
            raise FileExistsError(args.output_dir)
        args.output_dir.mkdir(parents=True)
        bundle = load_family_bundle(
            args.family, args.artifact, args.two_edit_manifest,
            split="validation", data_root=args.data_root,
        )
        metrics = evaluate_pairs(reader, variable_head, wrapped.module, tokenizer, bundle, device)
        torch.save({"operator": wrapped.module.operator.state_dict(), "method": args.method, "seed": args.seed, "test_evaluated": False}, args.output_dir / "operator.pt")
        summary = {
            "protocol": "f2_real_family_cell_v1", "cell_id": f"f2:{args.family}:{args.method}:{args.seed}",
            "family": args.family, "method": args.method, "seed": args.seed,
            "source_a1_seed": source_seed(args.seed), "source_a1_cell_id": source["cell_id"],
            "checkpoint_sha256": args.expected_checkpoint_sha256,
            "manifest_sha256": args.expected_manifest_sha256, "cells_sha256": args.expected_cells_sha256,
            "amendment_sha256": args.expected_amendment_sha256,
            "artifact_sha256": args.expected_artifact_sha256,
            "two_edit_manifest_sha256": args.expected_two_edit_manifest_sha256,
            "accepted_data_sha256": {
                "records": args.expected_records_sha256,
                "train_split": args.expected_train_split_sha256,
                "validation_split": args.expected_validation_split_sha256,
                "groups": args.expected_groups_sha256,
            } if args.family == "wiqa" else None,
            "core_components_sha256": args.expected_core_sha256,
            "world_size": 4, "allowed_splits": ["train", "validation"], "history": history,
            "two_edit": metrics, "test_evaluated": False,
        }
        (args.output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"cell_id": summary["cell_id"], "two_edit_balanced": metrics["two_edit_balanced"]}))
    dist.barrier(); dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
