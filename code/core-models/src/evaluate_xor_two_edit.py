#!/usr/bin/env python3
"""Development-only evaluation of authoritative composed-edit artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Any


VARIABLE_LABEL_VOCABULARY = (0, 1, "less", "more", "no_effect")
REQUIRED_BERT_BASE_WORLD_SIZE = 4


def require_bert_base_world_size(world_size: int) -> None:
    if world_size != REQUIRED_BERT_BASE_WORLD_SIZE:
        raise RuntimeError(
            "BERT-base composed evaluation requires torchrun with exactly 4 ranks; "
            f"received world_size={world_size}"
        )


def shard_graph_bundles(
    bundles: list[tuple[Path, Path]], rank: int, world_size: int
) -> tuple[tuple[Path, Path], ...]:
    """Assign each whole graph bundle to exactly one rank without padding."""

    if world_size < 1 or rank < 0 or rank >= world_size:
        raise ValueError("invalid distributed rank/world size")
    ordered = sorted(bundles, key=lambda item: (str(item[0]), str(item[1])))
    if len(ordered) != len(set(ordered)):
        raise ValueError("duplicate artifact/manifest bundle arguments")
    return tuple(ordered[rank::world_size])


def _assigned_bundles(args: argparse.Namespace, rank: int, world_size: int) -> tuple[tuple[Path, Path], ...]:
    """Keep XOR graphs intact; CLadder is sharded by graph inside its bundle."""

    if args.family == "xor":
        return shard_graph_bundles(args.bundle, rank, world_size)
    return tuple(sorted(args.bundle, key=lambda item: (str(item[0]), str(item[1]))))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("xor", "cladder", "wiqa", "ccrgb", "com2"), default="xor")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--bundle", action="append", nargs=2, metavar=("ARTIFACT_DIR", "MANIFEST"),
        required=True, help="repeat once per authoritative artifact/manifest bundle",
    )
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--paraphrase-catalog", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("t0", "t1", "t2a", "t2b", "t3a", "t3b"), required=True
    )
    parser.add_argument(
        "--a2-control",
        choices=("active", "editor_zeroed", "no_instruction", "prompting"),
        default="active",
    )
    parser.add_argument("--model")
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.max_length < 4:
        parser.error("batch-size must be positive and max-length must be at least four")
    args.bundle = [(Path(directory), Path(manifest)) for directory, manifest in args.bundle]
    if (args.family == "com2") != (args.provenance is not None):
        parser.error("--provenance is required exactly for family com2")
    if args.family == "xor" and args.paraphrase_catalog is not None:
        parser.error("XOR uses its intrinsic split-isolated wording protocol")
    if args.a2_control != "active" and args.mode != "t2b":
        parser.error("A2 controls require --mode t2b")
    if args.a2_control != "active" and args.split != "validation":
        parser.error("A2 controls are validation-only")
    return args


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _move_composed(batch: Any, device: Any) -> Any:
    import torch
    from core_bert.composed_reader import ComposedScientificBatch

    def side(values: Any) -> dict[str, Any]:
        return {
            key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in values.items()
        }

    return ComposedScientificBatch(batch.examples, side(batch.first), side(batch.second))


def _load_model(
    args: argparse.Namespace, checkpoint: dict[str, Any], tokenizer: Any, device: Any
) -> Any:
    import torch
    from core_bert.addressed_operator import AddressedGatedOperator
    from core_bert.addressed_reader import AddressedWorldReader
    from core_bert.composed_reader import (
        ComposedOpenTextModel, ComposedScientificExecutor, ComposedVariableModel,
    )
    from core_bert.open_text_outputs import OpenTextSlotHead
    from core_bert.reference_operators import (
        LearnedInterventionEncoder,
        SeparateTextEncoder,
        StateGatedReferenceEditor,
    )
    from core_bert.scientific_reader import ScientificAddressedReader
    from core_bert.t3_pointer import ClosedValueEmbedding, T3BPointer
    from core_bert.variable_outputs import PerVariableOutputHead

    configuration = checkpoint.get("configuration")
    if not isinstance(configuration, dict):
        raise ValueError("checkpoint lacks its training configuration")
    if configuration.get("model_size") != "base":
        raise ValueError("this exact 4-GPU evaluator accepts BERT-base checkpoints only")
    if configuration.get("mode") != args.mode:
        raise ValueError("checkpoint mode does not match requested composed mode")
    if args.family == "com2":
        if configuration.get("open_text_output") is not True:
            raise ValueError("Com2 evaluation requires an open-text training checkpoint")
        if configuration.get("com2_identity_exposure") is not True:
            raise ValueError(
                "Com2 evaluation requires training-only identity exposure for every chain position"
            )
    if args.mode == "t3b":
        required_architecture = (
            "post_decode_variable_slots",
            "mask_address_markers",
            "structural_support_propagation",
            "slot_question_isolation",
            "target_span_isolated_to_address_channel",
            "grounded_slot_descriptions",
            "task_readout_preserves_no_effect_abstention",
            "post_edit_structural_slot_isolation",
            "pointer_uses_uncontextualized_slot_names",
            "pointer_requires_lexical_grounding",
        )
        missing = [key for key in required_architecture if configuration.get(key) is not True]
        if missing:
            raise ValueError(
                "T3-b composed evaluation requires the frozen A3-passing architecture; "
                f"checkpoint is missing {missing}"
            )
    model_name = args.model or configuration.get("model")
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("model name is absent from CLI and checkpoint")
    world_slots = int(configuration.get("world_slots", 30))
    if world_slots != 30:
        raise ValueError("XOR composed evaluation requires exactly 30 world slots")
    rank = int(configuration.get("rank", 16))
    base = AddressedWorldReader.from_pretrained(
        model_name, node_count=1, split_layer=1,
        world_slot_count=30, label_count=4,
    )
    configured_split = int(configuration.get("split_layer", 0))
    base.split_layer = configured_split or len(base.bert.encoder.layer) // 2
    base.bert.resize_token_embeddings(len(tokenizer))
    reader = ScientificAddressedReader(base)
    intervention_encoder = (
        LearnedInterventionEncoder(30 * 15, reader.hidden_size)
        if args.mode == "t0" else
        SeparateTextEncoder(
            base.bert.embeddings.word_embeddings.weight,
            tokenizer.pad_token_id, reader.hidden_size,
        ) if args.mode == "t1" else None
    )
    editor = (
        StateGatedReferenceEditor(reader.hidden_size, rank)
        if args.mode in {"t0", "t1"} else
        T3BPointer(
            reader.hidden_size, rank,
            hard_selection=bool(configuration.get("hard_pointer", False)),
        )
        if args.mode == "t3b"
        else AddressedGatedOperator(reader.hidden_size, rank)
    )
    values = ClosedValueEmbedding(15, reader.hidden_size) if args.mode.startswith("t3") else None
    executor = ComposedScientificExecutor(
        reader, editor, args.mode, value_embedding=values,
        intervention_encoder=intervention_encoder,
        padding_idx=tokenizer.pad_token_id,
        editor_enabled=args.a2_control == "active",
        a2_control=args.a2_control,
    )
    if args.family == "com2":
        max_tokens = int(configuration.get("open_text_max_tokens", 32))
        open_text_head = OpenTextSlotHead(
            reader.hidden_size, len(tokenizer), padding_idx=tokenizer.pad_token_id,
            bos_token_id=tokenizer.cls_token_id, eos_token_id=tokenizer.sep_token_id,
        )
        model = ComposedOpenTextModel(executor, open_text_head, max_tokens=max_tokens)
    else:
        variable_head = PerVariableOutputHead(
            reader.hidden_size, len(VARIABLE_LABEL_VOCABULARY)
        )
        model = ComposedVariableModel(executor, variable_head)

    state = checkpoint.get("model")
    if not isinstance(state, dict):
        raise ValueError("checkpoint lacks model parameters")
    executor_state = {}
    for key, value in state.items():
        if key.startswith(("reader.", "editor.", "value_embedding.")):
            executor_state[key] = value
        elif key.startswith("reference_encoder."):
            executor_state[key.replace("reference_encoder.", "intervention_encoder.", 1)] = value
    executor.load_state_dict(executor_state, strict=True)
    if args.family == "com2":
        open_state = {
            key.removeprefix("open_text_head."): value
            for key, value in state.items() if key.startswith("open_text_head.")
        }
        if not open_state:
            raise ValueError("checkpoint lacks the trained Com2 open-text output head")
        open_text_head.load_state_dict(open_state, strict=True)
    else:
        variable_state = {
            key.removeprefix("variable_head."): value
            for key, value in state.items() if key.startswith("variable_head.")
        }
        if set(variable_state) != {"classifier.weight", "classifier.bias"}:
            raise ValueError("checkpoint lacks the trained per-variable output head")
        variable_head.load_state_dict(variable_state, strict=True)
    return model.to(device)


def _com2_teacher_forced_losses(
    model: Any, output: Any, batch: Any, tokenizer: Any, max_tokens: int, device: Any,
) -> Any:
    """Return genuine per-slot token losses for the composed Com2 state."""

    import torch
    from core_bert.open_text_outputs import IGNORE_INDEX, loss_matrix, sequence_cross_entropy

    mask = batch.first.get("slot_mask")
    if not isinstance(mask, torch.Tensor):
        raise ValueError("Com2 composed batch lacks its structural slot mask")
    slots = (
        output.composed.decoded_slots
        if output.composed.decoded_slots is not None else output.composed.edited_slots
    )
    inputs = torch.full(
        (*mask.shape, max_tokens), tokenizer.pad_token_id,
        dtype=torch.long, device=device,
    )
    targets = torch.full_like(inputs, IGNORE_INDEX)
    for row, example in enumerate(batch.examples):
        active_count = int(mask[row].sum().item())
        if active_count != len(example.gold_outputs):
            raise ValueError("Com2 gold chain length does not match active structural slots")
        for slot, text in enumerate(example.gold_outputs):
            ids = list(tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"])
            if not ids or len(ids) + 1 > max_tokens:
                raise ValueError(
                    f"Com2 composed label must contain 1..{max_tokens - 1} tokens; "
                    f"found {len(ids)}"
                )
            decoder_input = [int(tokenizer.cls_token_id), *map(int, ids)]
            decoder_target = [*map(int, ids), int(tokenizer.sep_token_id)]
            inputs[row, slot, :len(decoder_input)] = torch.tensor(
                decoder_input, dtype=torch.long, device=device
            )
            targets[row, slot, :len(decoder_target)] = torch.tensor(
                decoder_target, dtype=torch.long, device=device
            )
    logits = model.module.open_text_head(slots, inputs, mask)
    return loss_matrix(sequence_cross_entropy(logits, targets[mask.bool()]), mask)


def _summarize_rank_payloads(
    payloads: list[dict[str, Any]], args: argparse.Namespace
) -> dict[str, Any]:
    from core_bert.two_edit import two_edit_balanced_metrics

    checkpoint_hashes = {payload["checkpoint_sha256"] for payload in payloads}
    if len(checkpoint_hashes) != 1:
        raise RuntimeError("distributed ranks did not evaluate the same checkpoint bytes")
    all_examples = [
        example for payload in payloads for example in payload["examples"]
    ]
    pair_ids = [example.pair_id for example in all_examples]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("distributed gather received duplicate pair IDs")
    from core_bert.variable_outputs import summarize_variable_records

    predictions: dict[str, list[int]] = {}
    variable_rows = []
    pointer_rows = []
    artifact_hashes: dict[str, str] = {}
    graph_ids: set[str] = set()
    for payload in payloads:
        for pair_id, prediction in payload["predictions"].items():
            if pair_id in predictions:
                raise ValueError(f"distributed gather duplicated prediction {pair_id}")
            predictions[pair_id] = prediction
        variable_rows.extend(payload["variable_rows"])
        pointer_rows.extend(payload.get("pointer_rows", []))
        for path, digest in payload["artifact_sha256"].items():
            if path in artifact_hashes and artifact_hashes[path] != digest:
                raise RuntimeError(f"artifact hash disagrees across ranks for {path}")
            artifact_hashes[path] = digest
        graph_ids.update(payload["graph_ids"])
    row_ids = [row["pair_id"] for row in variable_rows]
    if len(row_ids) != len(set(row_ids)) or set(row_ids) != set(pair_ids):
        raise ValueError("variable rows must contain every pair exactly once")
    if args.mode == "t3b":
        expected_pointer_ids = {
            (pair_id, ordinal) for pair_id in pair_ids for ordinal in (1, 2)
        }
        pointer_ids = [(row["pair_id"], row["ordinal"]) for row in pointer_rows]
        if len(pointer_ids) != len(set(pointer_ids)) or set(pointer_ids) != expected_pointer_ids:
            raise ValueError("pointer rows must contain both edits for every pair exactly once")

    two_edit = two_edit_balanced_metrics(all_examples, predictions)
    by_graph: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in variable_rows:
        by_graph[row["graph_group_id"]].append(row)
    pointer = {
        "available": bool(pointer_rows),
        "eligible_count": len(pointer_rows),
        "coverage": len(pointer_rows) / max(2 * len(all_examples), 1),
        "pointer_mass": (
            sum(row["pointer_mass"] for row in pointer_rows) / len(pointer_rows)
            if pointer_rows else None
        ),
        "pointer_top1_accuracy": (
            sum(row["pointer_top1"] for row in pointer_rows) / len(pointer_rows)
            if pointer_rows else None
        ),
        "unavailable_reason": None if pointer_rows else "composed mode has no pointer traces",
    }
    paraphrase_rows = [payload.get("unseen_paraphrase") for payload in payloads]
    if any(value is not None for value in paraphrase_rows):
        if any(value is None for value in paraphrase_rows):
            raise RuntimeError("paraphrase provenance is missing from a distributed rank")
        protocols = {value["protocol"] for value in paraphrase_rows}
        banks = {value["bank"] for value in paraphrase_rows}
        hashes = {value.get("catalog_sha256") for value in paraphrase_rows}
        if len(protocols) != 1 or banks != {"validation"} or len(hashes) != 1:
            raise RuntimeError("distributed paraphrase provenance disagrees")
        eligible = sum(value["eligible_count"] for value in paraphrase_rows)
        covered = sum(value["covered_count"] for value in paraphrase_rows)
        materialized_rows = {
            json.dumps(row, sort_keys=True, separators=(",", ":"))
            for value in paraphrase_rows
            for row in value.get("materialized_selections", [])
        }
        unseen_paraphrase = {
            "protocol": protocols.pop(),
            "bank": "validation",
            "catalog_sha256": hashes.pop(),
            "eligible_count": eligible,
            "covered_count": covered,
            "coverage": covered / eligible if eligible else 0.0,
            "materialized_record_count": len(materialized_rows),
            "materialized_selections_sha256": (
                hashlib.sha256(
                    json.dumps(
                        sorted(materialized_rows), separators=(",", ":")
                    ).encode("utf-8")
                ).hexdigest()
                if materialized_rows else None
            ),
        }
    else:
        unseen_paraphrase = None
    return {
        "protocol": f"{args.family} ordered two-edit 4-GPU evaluation; not A1 evidence",
        "a1_evidence": False,
        "split": args.split,
        "test_evaluated": False,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_hashes.pop(),
        "mode": args.mode,
        "a2_control": args.a2_control,
        "variable_label_vocabulary": (
            "open_text_wordpiece_generation" if args.family == "com2"
            else list(VARIABLE_LABEL_VOCABULARY)
        ),
        "distributed": {
            "backend": "nccl",
            "world_size": REQUIRED_BERT_BASE_WORLD_SIZE,
            "sharding": (
                "whole graph bundles, deterministic, no padding"
                if args.family == "xor" else
                "ordered pairs, deterministic, no padding"
            ),
            "rank_bundle_counts": [payload["bundle_count"] for payload in payloads],
        },
        "graph_count": len(graph_ids),
        "pair_count": len(all_examples),
        "two_edit": two_edit,
        "pointer": pointer,
        "unseen_paraphrase": unseen_paraphrase,
        "variables": {
            "overall": summarize_variable_records(variable_rows),
            "per_graph": {
                graph: summarize_variable_records(rows)
                for graph, rows in sorted(by_graph.items())
            },
        },
        "artifact_sha256": artifact_hashes,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any] | None:
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    from core_bert.composed_reader import ComposedScientificCollator
    from core_bert.cladder_two_edit_adapter import load_cladder_two_edit_bundle
    from core_bert.com2_two_edit_adapter import load_com2_two_edit_bundle
    from core_bert.ccrgb_two_edit_adapter import load_ccrgb_two_edit_bundle
    from core_bert.wiqa_two_edit_adapter import load_wiqa_two_edit_bundle
    from core_bert.xor_two_edit_adapter import load_xor_two_edit_bundle
    from core_bert.paraphrase_protocol import load_paraphrase_catalog, paraphrase_coverage
    from core_bert.structured_paraphrases import structured_paraphrase_coverage

    if args.split not in {"train", "validation"}:
        raise ValueError("test evaluation is hard-blocked")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    require_bert_base_world_size(world_size)
    if not torch.cuda.is_available():
        raise RuntimeError("4-GPU composed evaluation requires CUDA")
    rank_id = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if not 0 <= rank_id < world_size:
        raise RuntimeError("RANK is outside WORLD_SIZE")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(hours=2))

    checkpoint_hash = _sha256(args.checkpoint)
    rank_checkpoint_hashes: list[str | None] = [None] * world_size
    dist.all_gather_object(rank_checkpoint_hashes, checkpoint_hash)
    if len(set(rank_checkpoint_hashes)) != 1:
        raise RuntimeError("distributed ranks did not load the same checkpoint bytes")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    tokenizer_source = args.tokenizer
    if tokenizer_source is None:
        saved = args.checkpoint.parent / "tokenizer"
        tokenizer_source = saved if saved.exists() else None
    configuration = checkpoint.get("configuration", {})
    fallback_model = args.model or configuration.get("model")
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_source or fallback_model), use_fast=True
    )
    tokenizer.add_special_tokens({"additional_special_tokens": ["[DO]"]})
    model = DDP(
        _load_model(args, checkpoint, tokenizer, device),
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
    )
    model.eval()

    paraphrase_partition = None
    paraphrase_metadata = None
    if args.paraphrase_catalog is not None:
        paraphrase_partition, paraphrase_metadata = load_paraphrase_catalog(
            args.paraphrase_catalog
        )

    local_examples = []
    predictions: dict[str, list[int]] = {}
    variable_rows: list[dict[str, Any]] = []
    pointer_rows: list[dict[str, Any]] = []
    artifact_hashes: dict[str, str] = {}
    graph_ids: set[str] = set()
    local_paraphrase = (
        {
            "protocol": "xor_intervention_wording_v1",
            "bank": "validation",
            "catalog_sha256": None,
            "eligible_count": 0,
            "covered_count": 0,
        }
        if args.family == "xor" else
        {
            "protocol": paraphrase_metadata["protocol_version"],
            "bank": args.split,
            "catalog_sha256": paraphrase_partition.catalog_sha256,
            "eligible_count": 0,
            "covered_count": 0,
            "materialized_selections": [],
        }
        if paraphrase_partition is not None else None
    )
    with torch.no_grad():
        assigned_bundles = _assigned_bundles(args, rank_id, world_size)
        for artifact_directory, manifest_path in assigned_bundles:
            if args.family == "xor":
                bundle = load_xor_two_edit_bundle(artifact_directory, manifest_path, split=args.split)
            elif args.family == "cladder":
                bundle = load_cladder_two_edit_bundle(artifact_directory, manifest_path, split=args.split)
            elif args.family == "wiqa":
                bundle = load_wiqa_two_edit_bundle(
                    artifact_directory, manifest_path, args.data_root, split=args.split
                )
            elif args.family == "ccrgb":
                bundle = load_ccrgb_two_edit_bundle(
                    artifact_directory, manifest_path, split=args.split
                )
            else:
                bundle = load_com2_two_edit_bundle(
                    artifact_directory, manifest_path, args.provenance,
                    args.data_root, split=args.split,
                )
            selected_examples = list(bundle.examples)
            if args.family != "xor":
                selected_examples = sorted(
                    selected_examples, key=lambda example: example.pair_id
                )[rank_id::world_size]
            bundle_graph_ids = {example.graph_group_id for example in selected_examples}
            overlap = graph_ids & bundle_graph_ids
            if overlap:
                raise ValueError(f"duplicate graph bundles {sorted(overlap)}")
            graph_ids.update(bundle_graph_ids)
            component_ids = sorted({
                record_id for example in selected_examples
                for record_id in (example.first_record_id, example.second_record_id)
            })
            component_records = [bundle.records[record_id] for record_id in component_ids]
            if args.family == "xor":
                if any(
                    record.get("intervention", {}).get("wording_family")
                    != "assign_value_to_variable"
                    for record in component_records
                ):
                    raise ValueError("XOR composed validation did not use its unseen wording family")
                local_paraphrase["eligible_count"] += len(component_records)
                local_paraphrase["covered_count"] += len(component_records)
            elif paraphrase_partition is not None:
                coverage = (
                    structured_paraphrase_coverage(
                        component_records,
                        paraphrase_partition,
                        split=args.split,
                        allow_materialized=True,
                    )
                    if args.family in {"cladder", "wiqa", "com2"}
                    else paraphrase_coverage(
                        component_records, paraphrase_partition, split=args.split
                    )
                )
                if coverage["coverage"] != 1.0:
                    raise ValueError("composed paraphrase coverage must be complete")
                if local_paraphrase is None:
                    local_paraphrase = {
                        "protocol": paraphrase_metadata["protocol_version"],
                        "bank": args.split,
                        "catalog_sha256": paraphrase_partition.catalog_sha256,
                        "eligible_count": 0,
                        "covered_count": 0,
                        "materialized_selections": [],
                    }
                local_paraphrase["eligible_count"] += coverage["eligible_count"]
                local_paraphrase["covered_count"] += coverage["covered_count"]
                local_paraphrase["materialized_selections"].extend(
                    coverage.get("materialized_selections", [])
                )
            collator = ComposedScientificCollator(
                tokenizer, bundle.records, max_length=args.max_length,
                paraphrase_partition=paraphrase_partition, split=args.split,
                allow_materialized_paraphrases=args.family
                in {"cladder", "wiqa", "com2"},
            )
            loader = DataLoader(
                selected_examples, batch_size=args.batch_size,
                shuffle=False, collate_fn=collator,
            )
            for batch in loader:
                batch = _move_composed(batch, device)
                output = model(batch)
                if args.family == "com2":
                    from core_bert.open_text_outputs import normalize_open_text
                    from core_bert.scientific_reader import variable_slot_layout
                    composed_losses = _com2_teacher_forced_losses(
                        model, output, batch, tokenizer,
                        int(configuration.get("open_text_max_tokens", 32)), device,
                    )
                    generated = tokenizer.batch_decode(
                        output.generated_token_ids.detach().cpu(), skip_special_tokens=True
                    )
                    row_generated = []
                    cursor = 0
                    for example in batch.examples:
                        count = len(example.gold_outputs)
                        row_generated.append(generated[cursor:cursor + count])
                        cursor += count
                else:
                    predicted = output.variables.logits.argmax(-1)
                for index, example in enumerate(batch.examples):
                    if example.pair_id in predictions:
                        raise ValueError(f"duplicate pair prediction {example.pair_id}")
                    active_count = len(example.gold_outputs)
                    if args.family == "com2":
                        row_prediction = [normalize_open_text(value) for value in row_generated[index]]
                    else:
                        row_prediction_ids = predicted[index, :active_count].detach().cpu().tolist()
                        row_prediction = [VARIABLE_LABEL_VOCABULARY[value] for value in row_prediction_ids]
                    predictions[example.pair_id] = row_prediction
                    gold = (
                        [normalize_open_text(value) for value in example.gold_outputs]
                        if args.family == "com2" else list(example.gold_outputs)
                    )
                    if args.family == "com2":
                        losses = composed_losses[index, :active_count]
                    else:
                        try:
                            gold_ids = [VARIABLE_LABEL_VOCABULARY.index(value) for value in gold]
                        except ValueError as error:
                            raise ValueError("composed gold value is outside the checkpoint vocabulary") from error
                        losses = F.cross_entropy(
                            output.variables.logits[index, :active_count],
                            torch.tensor(gold_ids, dtype=torch.long, device=device),
                            reduction="none",
                        )
                    source_record = bundle.records[example.first_record_id]
                    variable_names = (
                        list(variable_slot_layout(source_record).names)
                        if args.family == "com2" else list(
                            bundle.nodes if hasattr(bundle, "nodes")
                            else source_record["graph"]["nodes"]
                        )
                    )
                    variable_rows.append({
                        "graph_group_id": example.graph_group_id,
                        "pair_id": example.pair_id,
                        "active_variable_count": active_count,
                        "variable_ids": variable_names,
                        "variable_gold": gold,
                        "variable_predicted": row_prediction,
                        "variable_loss_sum": float(losses.sum().item()),
                    })
                    for trace in output.composed.traces:
                        if trace.pointer_mass is None or trace.pointer_top1 is None:
                            continue
                        pointer_rows.append({
                            "graph_group_id": example.graph_group_id,
                            "pair_id": example.pair_id,
                            "ordinal": trace.ordinal,
                            "pointer_mass": float(trace.pointer_mass[index].item()),
                            "pointer_top1": int(trace.pointer_top1[index].item()),
                        })
            local_examples.extend(selected_examples)
            artifact_paths = (
                [artifact_directory / name for name in ("graph.json", "worlds.json", "records_real.json")]
                if artifact_directory.is_dir() else [artifact_directory]
            )
            for path in artifact_paths:
                artifact_hashes[str(path)] = _sha256(path)
            artifact_hashes[str(manifest_path)] = _sha256(manifest_path)
            if args.family == "com2":
                artifact_hashes[str(args.provenance)] = _sha256(args.provenance)
            if args.paraphrase_catalog is not None:
                artifact_hashes[str(args.paraphrase_catalog)] = _sha256(args.paraphrase_catalog)

    local_payload = {
        "rank": rank_id,
        "checkpoint_sha256": checkpoint_hash,
        "bundle_count": len(assigned_bundles),
        "examples": local_examples,
        "predictions": predictions,
        "variable_rows": variable_rows,
        "pointer_rows": pointer_rows,
        "artifact_sha256": artifact_hashes,
        "graph_ids": sorted(graph_ids),
        "unseen_paraphrase": local_paraphrase,
    }
    gathered: list[dict[str, Any] | None] = [None] * world_size if rank_id == 0 else []
    dist.gather_object(local_payload, gathered if rank_id == 0 else None, dst=0)
    result = (
        _summarize_rank_payloads(
            [payload for payload in gathered if payload is not None], args
        )
        if rank_id == 0 else None
    )
    return result


def main(argv: list[str] | None = None) -> int:
    import torch.distributed as dist

    args = parse_args(argv)
    try:
        result = evaluate(args)
        if result is not None:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            print(json.dumps(result, indent=2, sort_keys=True))
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
