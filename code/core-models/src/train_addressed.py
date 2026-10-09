#!/usr/bin/env python3
"""Production train/validation runner for addressed BERT experiments.

This runner is deliberately not labeled an A1 result: it does not yet measure
two-edit composition or unseen paraphrases, both required by the registry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F
import transformers
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from addressed_smoke import _move_batch, data_fingerprint
from core_bert.addressed_data import AddressedBatchCollator, CLOSED_VALUE_TOKENS
from core_bert.addressed_operator import AddressedGatedOperator
from core_bert.addressed_reader import AddressedWorldReader
from core_bert.addressed_training_utils import load_group_sidecars, sidecar_fingerprints
from core_bert.benchmark_data import BenchmarkDataset, BenchmarkExample, GroupDistributedSampler, load_benchmark_split
from core_bert.addressed_laws import addressed_law_losses
from core_bert.l1_data import L1PairedCollator, L1PairedDataset
from core_bert.data import ID_TO_LABEL, LABEL_TO_ID
from core_bert.paraphrase_protocol import (
    ParaphrasePartition,
    apply_paraphrase,
    load_paraphrase_catalog,
    paraphrase_coverage,
)
from core_bert.open_text_outputs import (
    OpenTextBatch,
    OpenTextSlotHead,
    balanced_open_text_transition_loss,
    build_open_text_batch,
    normalize_open_text,
    sequence_cross_entropy,
)
from core_bert.reference_operators import (
    LearnedInterventionEncoder,
    SeparateTextEncoder,
    StateGatedReferenceEditor,
    compact_masked_tokens,
)
from core_bert.scientific_reader import ScientificAddressedReader, tokenize_scientific_fields
from core_bert.t3_pointer import (
    ClosedValueEmbedding,
    T3BPointer,
    override_span_mask,
    pointer_target_cross_entropy,
)
from core_bert.variable_outputs import (
    balanced_transition_cross_entropy,
    balanced_variable_cross_entropy,
    PerVariableOutputHead,
    build_variable_batch,
    encode_variable_labels,
    summarize_variable_records,
    task_logits_from_queried_variable,
    variable_cross_entropy,
)
from core_bert.xor_two_edit_adapter import (
    XOR_WORDING_FAMILIES,
    XOR_WORDING_PROTOCOL,
    load_xor_two_edit_bundle,
)


MODES = ("t0", "t1", "t2a", "t2b", "t3a", "t3b")
MODEL_POLICIES = {
    "base": {"gpus": 4, "model": "google-bert/bert-base-uncased"},
    "large": {"gpus": 8, "model": "google-bert/bert-large-uncased"},
}
VARIABLE_LABEL_VOCABULARY = (0, 1, "less", "more", "no_effect")
L1_LAW_PROFILES = {
    "core_base_1law": ("identity",),
    "core_i_3law": ("identity", "idempotence", "commutation"),
    "core_full_4law": ("identity", "idempotence", "commutation", "last_write_wins"),
    "full": ("identity", "idempotence", "commutation", "last_write_wins"),
    "drop_identity": ("idempotence", "commutation", "last_write_wins"),
    "drop_idempotence": ("identity", "commutation", "last_write_wins"),
    "drop_commutation": ("identity", "idempotence", "last_write_wins"),
    "drop_selective_invariance": ("identity", "idempotence", "commutation"),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--model-size", choices=tuple(MODEL_POLICIES), required=True)
    parser.add_argument("--model")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--groups-dir", type=Path)
    parser.add_argument("--evidence-capable", action="store_true")
    parser.add_argument("--sources", nargs="+", default=["cladder", "wiqa", "ccrgb"])
    parser.add_argument(
        "--xor-bundle", action="append", nargs=2,
        metavar=("ARTIFACT_DIR", "MANIFEST"),
        help="authoritative XOR artifact/ordered-pair manifest; repeat per graph",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8, help="per-GPU batch size")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--split-layer", type=int, default=0, help="0 selects model midpoint")
    parser.add_argument("--world-slots", type=int, default=30)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--train-shard-by", choices=("graph", "world"), default="graph",
        help="DDP training shard unit; does not change reported graph identities",
    )
    parser.add_argument(
        "--validation-shard-by", choices=("graph", "world"), default="graph",
        help="DDP validation shard unit; gathered inference remains graph-level",
    )
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--selection-metric", choices=("macro_f1", "accuracy", "loss"), default="macro_f1")
    parser.add_argument(
        "--variable-loss-weight", type=float, default=0.0,
        help="optional explicit-probe auxiliary loss; zero disables the variable head",
    )
    parser.add_argument(
        "--balanced-variable-loss", action="store_true",
        help="average target, changed-nontarget, and preservation losses by group",
    )
    parser.add_argument(
        "--transition-variable-loss", action="store_true",
        help="supervise shared-head pre-edit answer_before and edited answer_after states",
    )
    parser.add_argument(
        "--causal-task-readout", action="store_true",
        help="read task logits from the explicit question-bearing variable when available",
    )
    parser.add_argument(
        "--hard-pointer", action="store_true",
        help="use straight-through one-slot selection for T3-b edits",
    )
    parser.add_argument(
        "--pointer-loss-weight", type=float, default=0.0,
        help="T3-b correct-slot cross-entropy weight; zero disables pointer supervision",
    )
    parser.add_argument(
        "--open-text-output", action="store_true",
        help="use the shared autoregressive Com2 event-state head",
    )
    parser.add_argument("--open-text-max-tokens", type=int, default=32)
    parser.add_argument("--editor-disabled", action="store_true")
    parser.add_argument("--span-override", choices=("correct", "adjacent", "random"), default="correct")
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--initialize-checkpoint", type=Path,
        help="load a frozen reader/task initialization while resetting optimization state",
    )
    parser.add_argument("--law-profile", choices=tuple(L1_LAW_PROFILES))
    parser.add_argument("--law-weight", type=float, default=1.0)
    parser.add_argument("--freeze-reader", action="store_true")
    parser.add_argument(
        "--measure-checkpoint", type=Path,
        help="evaluate one frozen validation-selected checkpoint without training",
    )
    parser.add_argument(
        "--prediction-jsonl", type=Path,
        help="write exact row predictions during frozen-checkpoint validation measurement",
    )
    parser.add_argument(
        "--paraphrase-catalog", type=Path,
        help="optional development-only structured paraphrase manifest",
    )
    args = parser.parse_args(argv)
    args.model = args.model or MODEL_POLICIES[args.model_size]["model"]
    if args.epochs < 1 or args.batch_size < 1 or args.max_length < 4:
        parser.error("epochs/batch-size must be positive and max-length must be at least 4")
    if args.editor_disabled and not args.mode.startswith("t2"):
        parser.error("--editor-disabled requires t2a/t2b")
    if args.span_override != "correct" and not args.mode.startswith("t3"):
        parser.error("wrong-span controls require t3a/t3b")
    if args.world_slots != 30:
        parser.error("scientific addressed reader fixes --world-slots=30")
    if args.evidence_capable and args.groups_dir is None and not args.xor_bundle:
        parser.error("--evidence-capable requires --groups-dir sidecars or --xor-bundle")
    if args.xor_bundle and args.sources != ["xor"]:
        parser.error("--xor-bundle requires --sources xor")
    if args.mode == "t0" and (args.sources != ["xor"] or not args.xor_bundle):
        parser.error("T0 learned IDs are synthetic-only and require authoritative XOR bundles")
    if args.xor_bundle and args.paraphrase_catalog:
        parser.error("XOR bundles carry their own isolated wording protocol")
    if sum(value is not None for value in (args.resume, args.measure_checkpoint, args.initialize_checkpoint)) > 1:
        parser.error("--resume, --measure-checkpoint, and --initialize-checkpoint are mutually exclusive")
    if args.prediction_jsonl is not None and args.measure_checkpoint is None:
        parser.error("--prediction-jsonl requires --measure-checkpoint")
    if args.law_profile and not (
        args.mode == "t2b" and args.initialize_checkpoint and args.freeze_reader
    ):
        parser.error("--law-profile requires T2-b, --initialize-checkpoint, and --freeze-reader")
    if not math.isfinite(args.law_weight) or args.law_weight < 0:
        parser.error("--law-weight must be finite and non-negative")
    if not math.isfinite(args.variable_loss_weight) or args.variable_loss_weight < 0:
        parser.error("--variable-loss-weight must be finite and non-negative")
    if args.balanced_variable_loss and args.variable_loss_weight == 0:
        parser.error("--balanced-variable-loss requires positive --variable-loss-weight")
    if args.transition_variable_loss and not args.balanced_variable_loss:
        parser.error("--transition-variable-loss requires --balanced-variable-loss")
    if args.causal_task_readout and not args.transition_variable_loss:
        parser.error("--causal-task-readout requires --transition-variable-loss")
    if args.hard_pointer and (args.mode != "t3b" or not args.causal_task_readout):
        parser.error("--hard-pointer requires T3-b with --causal-task-readout")
    if not math.isfinite(args.pointer_loss_weight) or args.pointer_loss_weight < 0:
        parser.error("--pointer-loss-weight must be finite and non-negative")
    if args.pointer_loss_weight > 0 and args.mode != "t3b":
        parser.error("--pointer-loss-weight requires --mode t3b")
    if args.open_text_output and args.sources != ["com2"]:
        parser.error("--open-text-output is defined only for --sources com2")
    if args.open_text_output and not (
        args.variable_loss_weight > 0 and args.balanced_variable_loss
        and args.transition_variable_loss and args.causal_task_readout
    ):
        parser.error("Com2 open-text output requires balanced transition supervision and causal readout")
    if args.open_text_max_tokens < 2:
        parser.error("--open-text-max-tokens must be at least two")
    args.post_decode_variable_slots = True
    args.mask_address_markers = True
    args.structural_support_propagation = True
    args.slot_question_isolation = True
    args.target_span_isolated_to_address_channel = True
    args.grounded_slot_descriptions = True
    args.task_readout_preserves_no_effect_abstention = True
    args.post_edit_structural_slot_isolation = True
    args.pointer_uses_uncontextualized_slot_names = True
    args.pointer_requires_lexical_grounding = True
    args.com2_identity_exposure = bool(args.open_text_output)
    return args


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_checkpoint_configuration(
    checkpoint: Mapping[str, Any], args: argparse.Namespace
) -> None:
    stored = checkpoint.get("configuration", {})
    for key in (
        "mode", "model_size", "model", "world_slots", "rank",
        "variable_loss_weight", "balanced_variable_loss", "transition_variable_loss", "causal_task_readout", "hard_pointer", "post_decode_variable_slots", "mask_address_markers", "structural_support_propagation", "slot_question_isolation", "target_span_isolated_to_address_channel", "grounded_slot_descriptions", "task_readout_preserves_no_effect_abstention", "post_edit_structural_slot_isolation", "pointer_uses_uncontextualized_slot_names", "pointer_requires_lexical_grounding", "pointer_loss_weight",
        "paraphrase_catalog_sha256",
    ):
        stored_value = stored.get(
            key,
            False if key in {"balanced_variable_loss", "transition_variable_loss", "causal_task_readout", "hard_pointer"} else
            0.0 if key in {"variable_loss_weight", "pointer_loss_weight"} else None,
        )
        if str(stored_value) != str(getattr(args, key)):
            raise ValueError(f"checkpoint configuration mismatch for {key}")


def _closed_label_dataset(dataset: BenchmarkDataset, mode: str) -> BenchmarkDataset:
    selected = []
    for example in dataset:
        answer = str(example.label).strip().lower()
        if answer not in LABEL_TO_ID:
            continue
        if mode.startswith("t3") and example.intervention_kind != "value_set":
            continue
        selected.append(example)
    if not selected:
        raise ValueError("no addressed closed-label examples remain after task filtering")
    return BenchmarkDataset(selected)


def _task_dataset(
    dataset: BenchmarkDataset, mode: str, *, open_text_output: bool,
) -> BenchmarkDataset:
    if not open_text_output:
        return _closed_label_dataset(dataset, mode)
    selected = []
    for example in dataset:
        if example.source != "com2" or example.intervention_kind != "event_replace":
            continue
        record = dict(example.record)
        chain = record.get("chain")
        probes = [dict(probe) for probe in record.get("probes", [])]
        by_variable = {probe.get("variable"): probe for probe in probes}
        expected = [f"event_{index:02d}: {event}" for index, event in enumerate(chain)]
        missing = [name for name in expected if name not in by_variable]
        if missing:
            if missing != [expected[-1]]:
                raise ValueError(f"{example.record_id}: only a missing final Com2 probe can be source-recovered")
            released = record.get("intervened", {}).get("answer")
            if not isinstance(released, str):
                raise ValueError(f"{example.record_id}: missing final probe and released answer")
            after = re.sub(r"^[A-Z]\)\s*", "", released).strip()
            if after.endswith("."):
                after = after[:-1]
            probes.append({
                "variable": expected[-1], "question": record["factual"]["question"],
                "answer_before": chain[-1], "answer_after": after,
                "actually_changed": normalize_open_text(after) != normalize_open_text(chain[-1]),
                "required": "may change" if normalize_open_text(after) != normalize_open_text(chain[-1]) else "must not change",
                "source_recovery": "released_intervened_answer",
            })
        for probe in probes:
            probe["question"] = record["factual"]["question"] if probe["variable"] == expected[-1] else None
        record["probes"] = probes
        selected.append(BenchmarkExample(
            record, example.record_id, example.source, example.label,
            example.intervention_kind, example.graph_group_id, example.world_group_id,
        ))
    if not selected:
        raise ValueError("no Com2 open-text event-replacement examples remain")
    return BenchmarkDataset(selected)


def apply_group_sidecars(
    dataset: BenchmarkDataset,
    groups: dict[str, tuple[str, str]],
    *,
    required: bool,
) -> BenchmarkDataset:
    missing = [example.record_id for example in dataset if example.record_id not in groups]
    if required and missing:
        raise ValueError(f"group sidecars missing {len(missing)} records; first={missing[0]!r}")
    examples = []
    for example in dataset:
        graph_id, world_id = groups.get(
            example.record_id, (example.graph_group_id, example.world_group_id)
        )
        examples.append(
            BenchmarkExample(
                example.record, example.record_id, example.source, example.label,
                example.intervention_kind, graph_id, world_id,
            )
        )
    return BenchmarkDataset(examples)


def augment_com2_identity_exposure(dataset: BenchmarkDataset) -> BenchmarkDataset:
    """Add training-only factual restoration edits for every Com2 chain position."""

    examples = list(dataset)
    for example in list(dataset):
        record = example.record
        if example.source != "com2" or record.get("structure_kind") != "chain":
            continue
        chain = record.get("chain")
        if not isinstance(chain, list) or not chain:
            raise ValueError("Com2 exposure augmentation requires a non-empty chain")
        base_passage = record["factual"]["passage"].split("\n[ORIG]", 1)[0]
        for position, event in enumerate(chain):
            passage = base_passage + f"\n[ORIG] {event}\n[REPL] {event}"
            target_start = passage.index("\n[ORIG] ") + len("\n[ORIG] ")
            replacement_start = passage.index("\n[REPL] ") + len("\n[REPL] ")
            record_id = f"{example.record_id}:identity-exposure:{position:02d}"
            derived = {
                **record,
                "id": record_id,
                "factual": {**record["factual"], "passage": passage},
                "intervention": {
                    "formal": f'do(event_{position:02d} = {json.dumps(event)})',
                    "kind": "event_replace",
                    "target": f"event_{position:02d}: {event}",
                    "target_text": event,
                    "target_span": [target_start, target_start + len(event)],
                    "text": f"Restore {event} as {event}.",
                    "value": event, "value_token": None,
                    "replacement_span": [replacement_start, replacement_start + len(event)],
                },
                "intervened": {"answer": chain[-1], "passage": None, "state": None},
                "probes": [
                    {
                        "variable": f"event_{index:02d}: {value}",
                        "question": record["factual"]["question"] if index == len(chain) - 1 else None,
                        "answer_before": value, "answer_after": value,
                        "actually_changed": False, "required": "must not change",
                    }
                    for index, value in enumerate(chain)
                ],
                "provenance": {
                    **record.get("provenance", {}),
                    "training_augmentation": "com2_factual_identity_exposure_v1",
                    "test_evaluated": False,
                },
            }
            examples.append(BenchmarkExample(
                derived, record_id, "com2", chain[-1], "event_replace",
                example.graph_group_id, example.world_group_id,
            ))
    return BenchmarkDataset(examples)


def load_xor_training_split(
    bundle_specs: Sequence[Sequence[str]], split: str
) -> BenchmarkDataset:
    """Load unique single edits from authoritative XOR pair bundles."""

    examples: dict[str, BenchmarkExample] = {}
    for artifact, manifest in bundle_specs:
        bundle = load_xor_two_edit_bundle(Path(artifact), Path(manifest), split=split)
        world_by_record: dict[str, str] = {}
        for pair in bundle.examples:
            for record_id in (pair.first_record_id, pair.second_record_id):
                previous = world_by_record.setdefault(record_id, pair.world_group_id)
                if previous != pair.world_group_id:
                    raise ValueError(f"XOR record {record_id!r} crosses factual worlds")
        for record_id, record in bundle.records.items():
            if record_id not in world_by_record:
                raise ValueError(f"XOR record {record_id!r} lacks a composed-world group")
            # Original graph generators reuse world/record names across graph
            # seeds. Namespace only the training identity; authoritative
            # composed manifests retain their original component IDs.
            namespaced_id = f"{bundle.graph_group_id}:{record_id}"
            training_record = dict(record)
            training_record["id"] = namespaced_id
            candidate = BenchmarkExample(
                record=training_record, record_id=namespaced_id, source="xor",
                label=record["intervened"]["answer"], intervention_kind="value_set",
                graph_group_id=bundle.graph_group_id,
                world_group_id=world_by_record[record_id],
            )
            if namespaced_id in examples and examples[namespaced_id] != candidate:
                raise ValueError(
                    f"duplicate XOR record {namespaced_id!r} disagrees across bundles"
                )
            examples[namespaced_id] = candidate
    if not examples:
        raise ValueError(f"no XOR single-edit records found for split={split!r}")
    return BenchmarkDataset([examples[key] for key in sorted(examples)])


def xor_paraphrase_protocol_summary(
    train_data: BenchmarkDataset, validation_data: BenchmarkDataset
) -> dict[str, Any]:
    """Verify split-isolated XOR wording families and semantic coverage."""

    def collect(dataset: BenchmarkDataset) -> tuple[set[str], set[str]]:
        families, semantics = set(), set()
        for example in dataset:
            intervention = example.record.get("intervention", {})
            if intervention.get("wording_protocol") != XOR_WORDING_PROTOCOL:
                raise ValueError("XOR record lacks the registered wording protocol")
            family, semantic = (
                intervention.get("wording_family"), intervention.get("semantic_key")
            )
            if not isinstance(family, str) or not isinstance(semantic, str):
                raise ValueError("XOR record lacks wording family or semantic key")
            families.add(family)
            semantics.add(semantic)
        return families, semantics

    train_families, train_semantics = collect(train_data)
    validation_families, validation_semantics = collect(validation_data)
    if train_families != {XOR_WORDING_FAMILIES["train"]}:
        raise ValueError("XOR train records use an unexpected wording family")
    if validation_families != {XOR_WORDING_FAMILIES["validation"]}:
        raise ValueError("XOR validation records use an unexpected wording family")
    if train_families & validation_families:
        raise ValueError("XOR train/validation wording families overlap")
    missing = validation_semantics - train_semantics
    if missing:
        raise ValueError(f"XOR validation has {len(missing)} unseen intervention semantics")
    return {
        "protocol": XOR_WORDING_PROTOCOL,
        "train_wording_families": sorted(train_families),
        "validation_wording_families": sorted(validation_families),
        "semantic_coverage": 1.0,
        "validation_semantic_count": len(validation_semantics),
    }


class ProductionCollator:
    """Combine passage/command fields with dynamic slots and post-edit questions."""

    def __init__(
        self,
        tokenizer: Any,
        max_length: int,
        world_slots: int,
        *,
        variable_outputs: bool,
        open_text_output: bool = False,
        open_text_max_tokens: int = 32,
        paraphrase_partition: ParaphrasePartition | None = None,
        split: str | None = None,
    ):
        if paraphrase_partition is not None and split not in {"train", "validation"}:
            if split == "test":
                raise PermissionError("held-out test cannot use a preparation paraphrase catalog")
            raise ValueError("paraphrase-enabled collator requires train or validation split")
        self.addressed = AddressedBatchCollator(tokenizer, max_length=max_length)
        self.world_slots = world_slots
        self.variable_outputs = variable_outputs
        self.open_text_output = open_text_output
        self.open_text_max_tokens = open_text_max_tokens
        self.paraphrase_partition = paraphrase_partition
        self.split = split

    def __call__(self, examples: Sequence[Any]) -> dict[str, Any]:
        records = [example.record for example in examples]
        if self.paraphrase_partition is not None:
            assert self.split is not None
            records = [
                record if record.get("provenance", {}).get("training_augmentation")
                == "com2_factual_identity_exposure_v1"
                else apply_paraphrase(record, self.paraphrase_partition, split=self.split)
                for record in records
            ]
        batch = self.addressed(records)
        batch.update(tokenize_scientific_fields(self.addressed.tokenizer, records, max_slots=self.world_slots))
        if self.variable_outputs:
            variables = build_variable_batch(records, max_slots=self.world_slots)
            if not torch.equal(variables.variable_mask, batch["slot_mask"]):
                raise RuntimeError("variable and scientific-reader slot masks disagree")
            batch["variable_ids"] = [list(row) for row in variables.variable_ids]
            batch["variable_mask"] = variables.variable_mask
            batch["variable_label_mask"] = variables.label_mask
            batch["variable_before_label_mask"] = variables.before_label_mask
            batch["variable_query_slot"] = variables.query_slot
            batch["variable_target_mask"] = variables.target_mask
            batch["variable_changed_mask"] = variables.changed_mask
            batch["variable_preservation_mask"] = variables.preservation_mask
            if self.open_text_output:
                text = build_open_text_batch(
                    records, self.addressed.tokenizer, max_slots=self.world_slots,
                    max_tokens=self.open_text_max_tokens,
                )
                batch.update({
                    "open_after_inputs": text.after_inputs,
                    "open_after_targets": text.after_targets,
                    "open_before_inputs": text.before_inputs,
                    "open_before_targets": text.before_targets,
                    "open_after_text": [list(row) for row in text.after_text],
                    "open_before_text": [list(row) for row in text.before_text],
                })
                batch["variable_query_slot"] = text.query_slot
            else:
                batch["variable_label_ids"] = encode_variable_labels(
                    variables, VARIABLE_LABEL_VOCABULARY
                )
                batch["variable_before_label_ids"] = encode_variable_labels(
                    variables, VARIABLE_LABEL_VOCABULARY, state="before"
                )
        batch["graph_groups"] = [example.graph_group_id for example in examples]
        return batch


class ProductionAddressedModel(torch.nn.Module):
    """Scientific reader plus the registered T0/T1/T2/T3 adapters."""

    def __init__(self, reader: ScientificAddressedReader, mode: str, rank: int, *, editor_disabled: bool, span_override: str, seed: int, padding_idx: int = 0, variable_outputs: bool = False, causal_task_readout: bool = False, hard_pointer: bool = False, open_text_output: bool = False, tokenizer: Any | None = None, open_text_max_tokens: int = 32):
        super().__init__()
        self.reader = reader
        self.mode = mode
        self.editor_disabled = editor_disabled
        self.span_override = span_override
        self.seed = seed
        self.causal_task_readout = causal_task_readout
        self.open_text_output = open_text_output
        self.open_text_tokenizer = tokenizer
        self.open_text_max_tokens = open_text_max_tokens
        self.reference_encoder = (
            LearnedInterventionEncoder(
                reader.max_slots * len(CLOSED_VALUE_TOKENS), reader.hidden_size
            ) if mode == "t0" else
            SeparateTextEncoder(
                reader.reader.bert.embeddings.word_embeddings.weight,
                padding_idx, reader.hidden_size,
            ) if mode == "t1" else None
        )
        self.editor = (
            StateGatedReferenceEditor(reader.hidden_size, rank)
            if mode in {"t0", "t1"} else
            T3BPointer(reader.hidden_size, rank, hard_selection=hard_pointer)
            if mode == "t3b" else AddressedGatedOperator(reader.hidden_size, rank)
        )
        self.value_embedding = ClosedValueEmbedding(len(CLOSED_VALUE_TOKENS), reader.hidden_size) if mode.startswith("t3") else None
        self.variable_head = (
            PerVariableOutputHead(reader.hidden_size, len(VARIABLE_LABEL_VOCABULARY))
            if variable_outputs and not open_text_output else None
        )
        self.open_text_head = (
            OpenTextSlotHead(
                reader.hidden_size, len(tokenizer), padding_idx=tokenizer.pad_token_id,
                bos_token_id=tokenizer.cls_token_id, eos_token_id=tokenizer.sep_token_id,
            ) if open_text_output and tokenizer is not None else None
        )
        if open_text_output and self.open_text_head is None:
            raise ValueError("open-text output requires a tokenizer")
        if self.open_text_head is not None:
            with torch.no_grad():
                self.open_text_head.embedding.weight.copy_(
                    reader.reader.bert.embeddings.word_embeddings.weight
                )

    def forward(
        self, batch: dict[str, Any], *, return_law_context: bool = False
    ):
        common = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "slot_name_input_ids": batch["slot_name_input_ids"],
            "slot_name_attention_mask": batch["slot_name_attention_mask"],
            "slot_mask": batch["slot_mask"],
            "question_input_ids": batch["question_input_ids"],
            "question_attention_mask": batch["question_attention_mask"],
        }
        if self.mode in {"t0", "t1"}:
            passage_attention = (
                batch["attention_mask"].bool() & ~batch["command_tokens"]
                & ~batch["do_tokens"] & ~batch["address_marker_tokens"]
            ).long()
            if self.mode == "t0":
                if (batch["value_ids"] < 0).any():
                    raise ValueError("T0 requires a closed value-set intervention ID")
                intervention_id = (
                    batch["target_slot"] * len(CLOSED_VALUE_TOKENS)
                    + batch["value_ids"]
                )
                intervention = self.reference_encoder(intervention_id)
            else:
                command_ids, command_mask = compact_masked_tokens(
                    batch["input_ids"], batch["command_tokens"],
                    self.reference_encoder.word_embeddings.padding_idx,
                    max_length=self.reference_encoder.max_length,
                )
                intervention = self.reference_encoder(command_ids, command_mask)
            result = self.reader.forward_reference(
                **{**common, "attention_mask": passage_attention},
                intervention=intervention, editor=self.editor,
            )
        elif self.mode.startswith("t2"):
            result = self.reader.forward_t2(
                **common,
                command_tokens=batch["command_tokens"], do_tokens=batch["do_tokens"],
                editor=self.editor, mode=self.mode,
                editor_enabled=not self.editor_disabled,
            )
        else:
            target = batch["target_tokens"]
            if self.span_override != "correct":
                eligible = batch["offset_mapping"][..., 1] > batch["offset_mapping"][..., 0]
                target = override_span_mask(
                    target.cpu(), eligible.cpu(), self.span_override,
                    generator=torch.Generator(device="cpu").manual_seed(self.seed),
                ).to(target.device)
            event_replacement = bool((batch["value_ids"] < 0).all())
            if bool((batch["value_ids"] < 0).any()) and not event_replacement:
                raise ValueError("a T3 batch must not mix event replacements and value sets")
            if event_replacement and not self.open_text_output:
                raise ValueError("event replacement requires its open-text task head")
            passage_attention = (
                batch["attention_mask"].bool() & ~batch["command_tokens"]
                & ~batch["do_tokens"] & ~batch["address_marker_tokens"]
                & ~batch["target_tokens"]
            ).long()
            result = self.reader.forward_t3(
                **{**common, "attention_mask": passage_attention},
                target_span=target, target_slot=batch["target_slot"],
                slot_support=batch["slot_support"],
                editor=self.editor,
                value_embedding=None if event_replacement else self.value_embedding(batch["value_ids"]),
                replacement_span=batch["replacement_tokens"] if event_replacement else None,
                mode=self.mode,
            )
        empty_float = result.logits.new_empty((len(result.logits), 0))
        empty_bool = torch.empty((len(result.logits), 0), dtype=torch.bool, device=result.logits.device)
        variable_logits = (
            self.variable_head(
                result.decoded_slots
                if result.decoded_slots is not None else result.edited_slots,
                batch["variable_mask"],
            ).logits
            if self.variable_head is not None
            else result.logits.new_empty((len(result.logits), 0, 0))
        )
        pre_variable_logits = (
            self.variable_head(result.pre_edit_slots, batch["variable_mask"]).logits
            if self.variable_head is not None
            else result.logits.new_empty((len(result.logits), 0, 0))
        )
        task_logits = (
            task_logits_from_queried_variable(
                result.logits, variable_logits, batch["variable_query_slot"]
            )
            if self.causal_task_readout and not self.open_text_output else result.logits
        )
        open_after_logits = open_before_logits = result.logits.new_empty((0, 0, 0))
        if self.open_text_head is not None:
            open_after_logits = self.open_text_head(
                result.decoded_slots if result.decoded_slots is not None else result.edited_slots,
                batch["open_after_inputs"], batch["variable_label_mask"],
            )
            open_before_logits = self.open_text_head(
                result.pre_edit_slots, batch["open_before_inputs"],
                batch["variable_before_label_mask"],
            )
        output = (
            task_logits,
            result.pointer_mass if result.pointer_mass is not None else empty_float,
            result.pointer_top1 if result.pointer_top1 is not None else empty_bool,
            variable_logits,
            pre_variable_logits,
            open_after_logits,
            open_before_logits,
            result.decoded_slots if result.decoded_slots is not None else result.edited_slots,
        )
        if return_law_context:
            if self.mode != "t2b":
                raise ValueError("L1 law context is defined only for the T2-b path")
            return output, {
                "pre_edit_slots": result.pre_edit_slots,
                "instruction": result.instruction,
                "target_slot": batch["target_slot"],
                "value_ids": batch["value_ids"],
            }
        return output


def macro_f1(gold: list[int], predicted: list[int]) -> float:
    if gold and isinstance(gold[0], str):
        return sum(a == b for a, b in zip(gold, predicted)) / len(gold)
    scores = []
    for label in range(len(LABEL_TO_ID)):
        tp = sum(g == label and p == label for g, p in zip(gold, predicted))
        fp = sum(g != label and p == label for g, p in zip(gold, predicted))
        fn = sum(g == label and p != label for g, p in zip(gold, predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores)


def summarize_rows(parts: list[list[dict[str, Any]]], mode: str) -> dict[str, Any]:
    rows = sorted((row for part in parts for row in part), key=lambda row: row["record_id"])
    ids = [row["record_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("exact validation received duplicate record IDs")
    gold = [row["gold"] for row in rows]
    predicted = [row["predicted"] for row in rows]

    def scores(items: list[dict[str, Any]]) -> dict[str, Any]:
        item_gold = [item["gold"] for item in items]
        item_pred = [item["predicted"] for item in items]
        return {
            "count": len(items),
            "accuracy": sum(a == b for a, b in zip(item_gold, item_pred)) / max(len(items), 1),
            "macro_f1": macro_f1(item_gold, item_pred),
        }

    result = scores(rows)
    result["loss"] = sum(row["loss"] for row in rows) / max(len(rows), 1)
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[row["source"]].append(row)
    result["per_source"] = {source: scores(items) for source, items in sorted(by_source.items())}
    result["graph_count"] = len({row["graph_group"] for row in rows})
    if mode == "t3b":
        pointer_rows = [row for row in rows if row["correct_slot"] >= 0]
        result["pointer"] = {
            "available": bool(pointer_rows),
            "eligible_count": len(pointer_rows),
            "coverage": len(pointer_rows) / max(len(rows), 1),
            "pointer_mass": (
                sum(row["pointer_mass"] for row in pointer_rows) / len(pointer_rows)
                if pointer_rows else None
            ),
            "unavailable_reason": (
                None if pointer_rows else
                "no T3-b validation rows were gathered"
            ),
            "pointer_top1_accuracy": (
                sum(row["pointer_top1"] for row in pointer_rows) / len(pointer_rows)
                if pointer_rows else None
            ),
        }
    if rows and "variable_ids" in rows[0]:
        result["per_variable"] = summarize_variable_records(rows)
        result["per_source_variable"] = {
            source: summarize_variable_records(items)
            for source, items in sorted(by_source.items())
        }
    return result


@torch.no_grad()
def evaluate_exact(
    model: ProductionAddressedModel,
    loader: DataLoader,
    device: torch.device,
    rank_id: int,
    world_size: int,
    mode: str,
    output_dir: Path,
    evaluation_id: str,
    prediction_jsonl: Path | None = None,
) -> dict[str, Any]:
    model.eval()
    local = []
    for batch in loader:
        sources = batch.pop("sources")
        record_ids = batch.pop("record_ids")
        graph_groups = batch.pop("graph_groups")
        variable_ids = batch.pop("variable_ids", None)
        batch = _move_batch(batch, device)
        (
            logits, pointer_mass, pointer_top1, variable_logits, _,
            open_after_logits, _, open_slots,
        ) = model(batch)
        if model.open_text_output:
            slot_losses = sequence_cross_entropy(
                open_after_logits,
                batch["open_after_targets"][batch["variable_label_mask"].bool()],
            )
            slot_loss_matrix = slot_losses.new_zeros(batch["variable_label_mask"].shape)
            slot_loss_matrix[batch["variable_label_mask"].bool()] = slot_losses
            generated_ids = model.open_text_head.generate(
                open_slots, batch["variable_mask"], max_tokens=model.open_text_max_tokens
            )
            generated_text = model.open_text_tokenizer.batch_decode(
                generated_ids.detach().cpu(), skip_special_tokens=True
            )
            generated_by_row = []
            cursor = 0
            for row_mask in batch["variable_mask"]:
                count = int(row_mask.sum().item())
                generated_by_row.append(generated_text[cursor:cursor + count])
                cursor += count
            losses = None
            predictions = None
        else:
            losses = F.cross_entropy(logits, batch["label_ids"], reduction="none")
            predictions = logits.argmax(-1)
        variable_losses = variable_predictions = None
        if variable_ids is not None and not model.open_text_output:
            variable_losses = F.cross_entropy(
                variable_logits.transpose(1, 2), batch["variable_label_ids"],
                ignore_index=-100, reduction="none",
            )
            variable_predictions = variable_logits.argmax(-1)
        for index, record_id in enumerate(record_ids):
            if model.open_text_output:
                query_slot = int(batch["variable_query_slot"][index].item())
                gold_task = normalize_open_text(batch["open_after_text"][index][query_slot])
                predicted_task = normalize_open_text(generated_by_row[index][query_slot])
                task_loss = float(slot_loss_matrix[index, query_slot].item())
            else:
                gold_task = int(batch["label_ids"][index].item())
                predicted_task = int(predictions[index].item())
                task_loss = float(losses[index].item())
            row = {
                "record_id": record_id,
                "source": sources[index],
                "graph_group": graph_groups[index],
                "gold": gold_task,
                "predicted": predicted_task,
                "loss": task_loss,
                "correct_slot": int(batch["target_slot"][index].item()),
            }
            if mode == "t3b":
                row["pointer_mass"] = float(pointer_mass[index].item())
                row["pointer_top1"] = int(pointer_top1[index].item())
            if variable_ids is not None and model.open_text_output:
                observed = batch["variable_label_mask"][index].bool()
                slots = observed.nonzero(as_tuple=False).flatten().tolist()
                group_masks = {
                    "target": batch["variable_target_mask"][index].bool(),
                    "change": batch["variable_changed_mask"][index].bool(),
                    "preservation": batch["variable_preservation_mask"][index].bool(),
                }
                row.update({
                    "active_variable_count": int(batch["variable_mask"][index].sum().item()),
                    "variable_ids": [variable_ids[index][slot] for slot in slots],
                    "variable_gold": [normalize_open_text(batch["open_after_text"][index][slot]) for slot in slots],
                    "variable_predicted": [normalize_open_text(generated_by_row[index][slot]) for slot in slots],
                    "variable_groups": [
                        next((name for name, mask in group_masks.items() if mask[slot]), "unclassified")
                        for slot in slots
                    ],
                    "variable_loss_sum": float(slot_loss_matrix[index][observed].sum().item()),
                })
            elif variable_ids is not None:
                observed = batch["variable_label_mask"][index].bool()
                selected_ids = [
                    variable_ids[index][slot]
                    for slot in observed.nonzero(as_tuple=False).flatten().tolist()
                ]
                group_masks = {
                    "target": batch["variable_target_mask"][index].bool(),
                    "change": batch["variable_changed_mask"][index].bool(),
                    "preservation": batch["variable_preservation_mask"][index].bool(),
                }
                variable_groups = []
                for slot in observed.nonzero(as_tuple=False).flatten().tolist():
                    matches = [name for name, mask in group_masks.items() if mask[slot]]
                    variable_groups.append(matches[0] if matches else "unclassified")
                row.update({
                    "active_variable_count": int(batch["variable_mask"][index].sum().item()),
                    "variable_ids": selected_ids,
                    "variable_gold": batch["variable_label_ids"][index][observed].tolist(),
                    "variable_predicted": variable_predictions[index][observed].tolist(),
                    "variable_groups": variable_groups,
                    "variable_loss_sum": float(variable_losses[index][observed].sum().item()),
                })
            local.append(row)
    # NCCL implements gather_object by serializing into CUDA byte tensors.  A
    # complete exact-validation payload can therefore exhaust rank 0's VRAM
    # after training.  The small-model cluster node exposes a shared filesystem, so exchange
    # the large row payloads there and reserve NCCL for the small summary.
    shard_dir = output_dir / ".evaluation_shards" / evaluation_id
    if rank_id == 0:
        shard_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    shard_path = shard_dir / f"rank-{rank_id:03d}.json"
    temporary_path = shard_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(local, separators=(",", ":")))
    os.replace(temporary_path, shard_path)
    dist.barrier()
    if rank_id == 0:
        gathered = [
            json.loads((shard_dir / f"rank-{rank:03d}.json").read_text())
            for rank in range(world_size)
        ]
        if prediction_jsonl is not None:
            rows = sorted(
                (row for shard in gathered for row in shard),
                key=lambda row: row["record_id"],
            )
            record_ids = [row["record_id"] for row in rows]
            if len(record_ids) != len(set(record_ids)):
                raise ValueError("row-level prediction export contains duplicate record IDs")
            prediction_jsonl.parent.mkdir(parents=True, exist_ok=True)
            temporary = prediction_jsonl.with_suffix(prediction_jsonl.suffix + ".tmp")
            temporary.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )
            os.replace(temporary, prediction_jsonl)
        payload = [summarize_rows(gathered, mode)]
    else:
        payload = [None]
    dist.broadcast_object_list(payload, src=0)
    dist.barrier()
    if rank_id == 0:
        for rank in range(world_size):
            (shard_dir / f"rank-{rank:03d}.json").unlink()
        shard_dir.rmdir()
    return payload[0]


def save_checkpoint(
    path: Path,
    model: ProductionAddressedModel,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    best_metric: float,
    args: argparse.Namespace,
    history: list[dict[str, Any]],
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "best_metric": best_metric,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "history": history,
            "configuration": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        },
        path,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    expected_gpus = MODEL_POLICIES[args.model_size]["gpus"]
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != expected_gpus:
        raise RuntimeError(f"{args.model_size} policy requires exactly {expected_gpus} GPUs, got {world_size}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is not supported by the selected GPUs")
    rank_id = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    device = torch.device("cuda", local_rank)
    torch.manual_seed(args.seed + rank_id)
    torch.cuda.manual_seed_all(args.seed + rank_id)
    primary = rank_id == 0
    if primary:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    if args.xor_bundle:
        train_data = _task_dataset(
            load_xor_training_split(args.xor_bundle, "train"), args.mode,
            open_text_output=args.open_text_output,
        )
        validation_data = _task_dataset(
            load_xor_training_split(args.xor_bundle, "validation"), args.mode,
            open_text_output=args.open_text_output,
        )
    else:
        train_data = _task_dataset(
            load_benchmark_split(args.data_root, "train", sources=args.sources), args.mode,
            open_text_output=args.open_text_output,
        )
        validation_data = _task_dataset(
            load_benchmark_split(args.data_root, "validation", sources=args.sources), args.mode,
            open_text_output=args.open_text_output,
        )
        sidecars = load_group_sidecars(args.groups_dir) if args.groups_dir else {}
        train_data = apply_group_sidecars(
            train_data, sidecars, required=args.evidence_capable
        )
        validation_data = apply_group_sidecars(
            validation_data, sidecars, required=args.evidence_capable
        )
        if args.open_text_output:
            train_data = augment_com2_identity_exposure(train_data)
    xor_paraphrase_metadata = (
        xor_paraphrase_protocol_summary(train_data, validation_data)
        if args.xor_bundle else None
    )
    paraphrase_partition = None
    paraphrase_metadata = None
    paraphrase_coverage_by_split = None
    if args.paraphrase_catalog is not None:
        paraphrase_partition, paraphrase_metadata = load_paraphrase_catalog(
            args.paraphrase_catalog
        )
        paraphrase_coverage_by_split = {
            "train": paraphrase_coverage(
                [
                    record for record in train_data.records
                    if record.get("provenance", {}).get("training_augmentation")
                    != "com2_factual_identity_exposure_v1"
                ],
                paraphrase_partition, split="train"
            ),
            "validation": paraphrase_coverage(
                validation_data.records, paraphrase_partition, split="validation"
            ),
        }
        args.paraphrase_catalog_sha256 = paraphrase_partition.catalog_sha256
    else:
        args.paraphrase_catalog_sha256 = None
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    added_tokens = tokenizer.add_special_tokens({"additional_special_tokens": ["[DO]"]})
    variable_outputs = args.variable_loss_weight > 0
    pointer_supervision = args.pointer_loss_weight > 0
    train_collator = ProductionCollator(
        tokenizer, args.max_length, args.world_slots,
        variable_outputs=variable_outputs,
        open_text_output=args.open_text_output,
        open_text_max_tokens=args.open_text_max_tokens,
        paraphrase_partition=paraphrase_partition, split="train",
    )
    validation_collator = ProductionCollator(
        tokenizer, args.max_length, args.world_slots,
        variable_outputs=variable_outputs,
        open_text_output=args.open_text_output,
        open_text_max_tokens=args.open_text_max_tokens,
        paraphrase_partition=paraphrase_partition, split="validation",
    )
    law_collator = ProductionCollator(
        tokenizer, args.max_length, args.world_slots,
        variable_outputs=variable_outputs,
        open_text_output=args.open_text_output,
        open_text_max_tokens=args.open_text_max_tokens,
        paraphrase_partition=None, split="train",
    )
    train_sampler = GroupDistributedSampler(
        train_data, rank=rank_id, world_size=world_size, group_by=args.train_shard_by,
        seed=args.seed, shuffle=True, pad_to_equal=True,
    )
    validation_sampler = GroupDistributedSampler(
        validation_data, rank=rank_id, world_size=world_size, group_by=args.validation_shard_by,
        seed=args.seed, shuffle=False, pad_to_equal=False,
    )
    paired_train_data = L1PairedDataset(train_data) if args.law_profile else train_data
    train_loader = DataLoader(
        paired_train_data, batch_size=args.batch_size, sampler=train_sampler,
        collate_fn=(
            L1PairedCollator(train_collator, law_collator)
            if args.law_profile else train_collator
        ),
        num_workers=args.num_workers,
        pin_memory=True, persistent_workers=args.num_workers > 0,
    )
    validation_loader = DataLoader(
        validation_data, batch_size=args.batch_size, sampler=validation_sampler,
        collate_fn=validation_collator, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=args.num_workers > 0,
    )

    base_reader = AddressedWorldReader.from_pretrained(
        args.model, node_count=1, split_layer=1,
        world_slot_count=args.world_slots, label_count=len(LABEL_TO_ID),
    )
    split_layer = args.split_layer or len(base_reader.bert.encoder.layer) // 2
    if not 0 < split_layer < len(base_reader.bert.encoder.layer):
        raise ValueError("split-layer must leave layers on both sides")
    base_reader.split_layer = split_layer
    base_reader.bert.resize_token_embeddings(len(tokenizer))
    reader = ScientificAddressedReader(base_reader, max_slots=30)
    raw_model = ProductionAddressedModel(
        reader, args.mode, args.rank,
        editor_disabled=args.editor_disabled,
        span_override=args.span_override,
        seed=args.seed + rank_id,
        padding_idx=tokenizer.pad_token_id,
        variable_outputs=variable_outputs,
        causal_task_readout=args.causal_task_readout,
        hard_pointer=args.hard_pointer,
        open_text_output=args.open_text_output,
        tokenizer=tokenizer,
        open_text_max_tokens=args.open_text_max_tokens,
    ).to(device)
    if args.initialize_checkpoint:
        checkpoint = torch.load(args.initialize_checkpoint, map_location=device, weights_only=False)
        _validate_checkpoint_configuration(checkpoint, args)
        raw_model.load_state_dict(checkpoint["model"])
    if args.freeze_reader:
        for parameter in raw_model.parameters():
            parameter.requires_grad_(False)
        for parameter in raw_model.editor.parameters():
            parameter.requires_grad_(True)
    if args.measure_checkpoint:
        checkpoint = torch.load(args.measure_checkpoint, map_location=device, weights_only=False)
        _validate_checkpoint_configuration(checkpoint, args)
        raw_model.load_state_dict(checkpoint["model"])
        validation = evaluate_exact(
            raw_model, validation_loader, device, rank_id, world_size, args.mode,
            args.output_dir, "measurement", args.prediction_jsonl,
        )
        if paraphrase_partition is not None:
            coverage = paraphrase_coverage_by_split["validation"]
            validation["unseen_paraphrase"] = {
                "available": True,
                "a1_evidence": False,
                "catalog_sha256": paraphrase_partition.catalog_sha256,
                **coverage,
                "evaluated_count": validation["count"],
                "accuracy": validation["accuracy"],
                "macro_f1": validation["macro_f1"],
                "loss": validation["loss"],
                "per_source": validation["per_source"],
            }
        elif args.mode == "t0":
            validation["unseen_paraphrase"] = {
                "available": False,
                "a1_evidence": False,
                "unavailable_reason": "not applicable to the learned-ID reference ceiling",
            }
        elif xor_paraphrase_metadata is not None:
            validation["unseen_paraphrase"] = {
                "available": True,
                "a1_evidence": False,
                **xor_paraphrase_metadata,
                "eligible_count": validation["count"],
                "covered_count": validation["count"],
                "coverage": 1.0,
                "evaluated_count": validation["count"],
                "accuracy": validation["accuracy"],
                "macro_f1": validation["macro_f1"],
                "loss": validation["loss"],
                "per_source": validation["per_source"],
            }
        else:
            validation["unseen_paraphrase"] = {
                "available": False,
                "a1_evidence": False,
                "eligible_count": validation["count"],
                "covered_count": 0,
                "coverage": 0.0,
                "unavailable_reason": "--paraphrase-catalog was not supplied",
            }
        if primary:
            measurement = {
                "protocol": "frozen-checkpoint validation measurement; not A1 evidence",
                "a1_evidence": False,
                "trained_during_measurement": False,
                "checkpoint": str(args.measure_checkpoint),
                "checkpoint_sha256": _sha256_file(args.measure_checkpoint),
                "checkpoint_selected_epoch": checkpoint.get("epoch"),
                "mode": args.mode,
                "editor_disabled": args.editor_disabled,
                "span_override": args.span_override,
                "split": "validation",
                "test_evaluated": False,
                "distributed": {"world_size": world_size},
                "validation_sampler": "graph-grouped, exact, no padding",
                "validation": validation,
                "prediction_jsonl": str(args.prediction_jsonl) if args.prediction_jsonl else None,
                "prediction_jsonl_sha256": (
                    _sha256_file(args.prediction_jsonl) if args.prediction_jsonl else None
                ),
            }
            (args.output_dir / "measurement_summary.json").write_text(
                json.dumps(measurement, indent=2, sort_keys=True) + "\n"
            )
            print(json.dumps(measurement, indent=2, sort_keys=True), flush=True)
        dist.barrier()
        dist.destroy_process_group()
        return 0
    wrapped = DDP(
        raw_model, device_ids=[local_rank], output_device=local_rank,
        broadcast_buffers=False, find_unused_parameters=True,
    )
    trainable_parameters = [parameter for parameter in wrapped.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("training configuration exposes no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable_parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    start_epoch = 1
    best_metric = math.inf if args.selection_metric == "loss" else -math.inf
    history: list[dict[str, Any]] = []
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        _validate_checkpoint_configuration(checkpoint, args)
        raw_model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_metric = float(checkpoint["best_metric"])
        history = list(checkpoint.get("history", []))

    if primary:
        tokenizer.save_pretrained(args.output_dir / "tokenizer")
    started = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        # Epoch-derived RNG makes resumed runs reproduce the same dropout and
        # A3 sampling stream without requiring rank-specific RNG checkpoints.
        torch.manual_seed(args.seed + rank_id + epoch * 1_000_003)
        torch.cuda.manual_seed_all(args.seed + rank_id + epoch * 1_000_003)
        train_sampler.set_epoch(epoch)
        wrapped.train()
        totals = torch.zeros(4, dtype=torch.float64, device=device)
        variable_totals = torch.zeros(2, dtype=torch.float64, device=device)
        pointer_totals = torch.zeros(2, dtype=torch.float64, device=device)
        for packed_batch in train_loader:
            if args.law_profile:
                batch = packed_batch["base"]
                commuting_batch = packed_batch["commuting"]
                opposite_batch = packed_batch["opposite"]
            else:
                batch = packed_batch
                commuting_batch = opposite_batch = None
            batch.pop("graph_groups")
            batch.pop("variable_ids", None)
            batch = _move_batch(batch, device)
            if args.law_profile:
                for paired in (commuting_batch, opposite_batch):
                    paired.pop("graph_groups")
                    paired.pop("variable_ids", None)
                commuting_batch = _move_batch(commuting_batch, device)
                opposite_batch = _move_batch(opposite_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                "cuda", enabled=args.precision == "bf16", dtype=torch.bfloat16
            ):
                model_result = wrapped(batch, return_law_context=bool(args.law_profile))
                if args.law_profile:
                    model_output, law_context = model_result
                    with torch.no_grad():
                        _, commuting_context = raw_model(
                            commuting_batch, return_law_context=True
                        )
                        _, opposite_context = raw_model(
                            opposite_batch, return_law_context=True
                        )
                    law_parts = addressed_law_losses(
                        raw_model.editor,
                        law_context["pre_edit_slots"], law_context["instruction"],
                        enabled=L1_LAW_PROFILES[args.law_profile],
                        commuting_instruction=commuting_context["instruction"],
                        opposite_instruction=opposite_context["instruction"],
                    )
                    law_loss = torch.stack(tuple(law_parts.values())).sum()
                else:
                    model_output = model_result
                    law_loss = None
                (
                    logits, pointer_mass, _, variable_logits, pre_variable_logits,
                    open_after_logits, open_before_logits, _,
                ) = model_output
                if args.open_text_output:
                    after_losses = sequence_cross_entropy(
                        open_after_logits,
                        batch["open_after_targets"][batch["variable_label_mask"].bool()],
                    )
                    before_losses = sequence_cross_entropy(
                        open_before_logits,
                        batch["open_before_targets"][batch["variable_before_label_mask"].bool()],
                    )
                    after_matrix = after_losses.new_zeros(batch["variable_label_mask"].shape)
                    after_matrix[batch["variable_label_mask"].bool()] = after_losses
                    rows = torch.arange(len(batch["variable_query_slot"]), device=device)
                    scalar_loss = after_matrix[rows, batch["variable_query_slot"]].mean()
                    text_batch = OpenTextBatch(
                        batch["open_after_inputs"], batch["open_after_targets"],
                        batch["open_before_inputs"], batch["open_before_targets"],
                        batch["variable_label_mask"], batch["variable_before_label_mask"],
                        batch["variable_mask"], batch["variable_target_mask"],
                        batch["variable_changed_mask"], batch["variable_preservation_mask"],
                        batch["variable_query_slot"], tuple(), tuple(),
                    )
                    auxiliary_loss = balanced_open_text_transition_loss(
                        after_losses, before_losses, text_batch
                    )
                else:
                    scalar_loss = F.cross_entropy(logits, batch["label_ids"])
                if variable_outputs and not args.open_text_output:
                    if args.transition_variable_loss:
                        auxiliary_loss = balanced_transition_cross_entropy(
                            variable_logits, pre_variable_logits,
                            batch["variable_label_ids"], batch["variable_before_label_ids"],
                            batch["variable_label_mask"], batch["variable_before_label_mask"],
                            batch["variable_mask"], batch["variable_target_mask"],
                            batch["variable_changed_mask"], batch["variable_preservation_mask"],
                        )
                    elif args.balanced_variable_loss:
                        auxiliary_loss = balanced_variable_cross_entropy(
                            variable_logits, batch["variable_label_ids"],
                            batch["variable_label_mask"], batch["variable_mask"],
                            batch["variable_target_mask"], batch["variable_changed_mask"],
                            batch["variable_preservation_mask"],
                        )
                    else:
                        auxiliary_loss = variable_cross_entropy(
                            variable_logits, batch["variable_label_ids"],
                            batch["variable_label_mask"], batch["variable_mask"],
                        )
                elif not args.open_text_output:
                    auxiliary_loss = scalar_loss.new_zeros(())
                pointer_loss = (
                    pointer_target_cross_entropy(pointer_mass)
                    if pointer_supervision else scalar_loss.new_zeros(())
                )
                loss = (
                    scalar_loss
                    + args.variable_loss_weight * auxiliary_loss
                    + args.pointer_loss_weight * pointer_loss
                    + (args.law_weight * law_loss if law_loss is not None else 0.0)
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(wrapped.parameters(), 1.0)
            optimizer.step()
            batch_count = len(batch["label_ids"])
            totals[0] += loss.detach().double() * batch_count
            totals[1] += batch_count
            totals[2] += scalar_loss.detach().double() * batch_count
            if law_loss is not None:
                totals[3] += law_loss.detach().double() * batch_count
            if variable_outputs:
                variable_weight = (
                    batch_count
                    if args.balanced_variable_loss else batch["variable_label_mask"].sum()
                )
                variable_totals[0] += auxiliary_loss.detach().double() * variable_weight
                variable_totals[1] += variable_weight
            if pointer_supervision:
                pointer_totals[0] += pointer_loss.detach().double() * batch_count
                pointer_totals[1] += batch_count
        dist.all_reduce(totals)
        if variable_outputs:
            dist.all_reduce(variable_totals)
        if pointer_supervision:
            dist.all_reduce(pointer_totals)
        validation = evaluate_exact(
            raw_model, validation_loader, device, rank_id, world_size, args.mode,
            args.output_dir, f"epoch-{epoch:03d}",
        )
        if paraphrase_partition is not None:
            coverage = paraphrase_coverage_by_split["validation"]
            validation["unseen_paraphrase"] = {
                "available": True,
                "a1_evidence": False,
                "catalog_sha256": paraphrase_partition.catalog_sha256,
                **coverage,
                "evaluated_count": validation["count"],
                "accuracy": validation["accuracy"],
                "macro_f1": validation["macro_f1"],
                "loss": validation["loss"],
                "per_source": validation["per_source"],
            }
        elif args.mode == "t0":
            validation["unseen_paraphrase"] = {
                "available": False,
                "a1_evidence": False,
                "unavailable_reason": "not applicable to the learned-ID reference ceiling",
            }
        elif xor_paraphrase_metadata is not None:
            validation["unseen_paraphrase"] = {
                "available": True,
                "a1_evidence": False,
                **xor_paraphrase_metadata,
                "eligible_count": validation["count"],
                "covered_count": validation["count"],
                "coverage": 1.0,
                "evaluated_count": validation["count"],
                "accuracy": validation["accuracy"],
                "macro_f1": validation["macro_f1"],
                "loss": validation["loss"],
                "per_source": validation["per_source"],
            }
        else:
            validation["unseen_paraphrase"] = {
                "available": False,
                "a1_evidence": False,
                "eligible_count": validation["count"],
                "covered_count": 0,
                "coverage": 0.0,
                "unavailable_reason": "--paraphrase-catalog was not supplied",
            }
        selection_value = float(validation[args.selection_metric])
        improved = (
            selection_value < best_metric if args.selection_metric == "loss"
            else selection_value > best_metric
        )
        if improved:
            best_metric = selection_value
        epoch_result = {
            "epoch": epoch,
            "train_loss": float((totals[0] / totals[1]).item()),
            "train_scalar_loss": float((totals[2] / totals[1]).item()),
            "train_law_loss": (
                float((totals[3] / totals[1]).item()) if args.law_profile else None
            ),
            "train_variable_loss": (
                float((variable_totals[0] / variable_totals[1]).item())
                if variable_outputs else None
            ),
            "train_pointer_loss": (
                float((pointer_totals[0] / pointer_totals[1]).item())
                if pointer_supervision else None
            ),
            "validation": validation,
            "selection_metric": args.selection_metric,
            "selection_value": selection_value,
            "selected": improved,
        }
        history.append(epoch_result)
        if primary:
            save_checkpoint(
                args.output_dir / "last.pt", raw_model, optimizer,
                epoch=epoch, best_metric=best_metric, args=args, history=history,
            )
            if improved:
                save_checkpoint(
                    args.output_dir / "best.pt", raw_model, optimizer,
                    epoch=epoch, best_metric=best_metric, args=args, history=history,
                )
            summary = {
                "protocol": (
                    "l1_law_retention_v1" if args.law_profile else
                    "scalar addressed pilot, train/validation only; not A1 evidence"
                ),
                "a1_evidence": False,
                "test_evaluated": False,
                "l1": (
                    {
                        "law_profile": args.law_profile,
                        "enabled_laws": list(L1_LAW_PROFILES[args.law_profile]),
                        "law_weight": args.law_weight,
                        "reader_frozen": args.freeze_reader,
                        "trainable_module": "addressed_editor_only",
                        "initialization_checkpoint": str(args.initialize_checkpoint),
                        "initialization_checkpoint_sha256": _sha256_file(args.initialize_checkpoint),
                    }
                    if args.law_profile else None
                ),
                "missing_registry_metrics": [
                    "two_edit",
                    *([] if args.mode == "t0" or paraphrase_partition is not None or xor_paraphrase_metadata is not None else ["unseen_paraphrase"]),
                    *([] if variable_outputs else ["per_variable_outputs"]),
                ],
                "per_variable_outputs": {
                    "enabled": variable_outputs,
                    "loss_weight": args.variable_loss_weight,
                    "label_vocabulary": list(VARIABLE_LABEL_VOCABULARY),
                    "gold_source": (
                        "record.probes[].answer_before and answer_after only"
                        if args.transition_variable_loss
                        else "record.probes[].answer_after only"
                    ),
                    "balanced_target_change_preservation": args.balanced_variable_loss,
                    "explicit_before_after_transition": args.transition_variable_loss,
                    "causal_task_readout": args.causal_task_readout,
                    "straight_through_hard_pointer": args.hard_pointer,
                    "after_state_from_post_decode_slots": args.post_decode_variable_slots,
                    "address_marker_masked_from_trunk": args.mask_address_markers,
                    "target_descendant_support_from_structure": args.structural_support_propagation,
                    "post_edit_slots_cannot_read_question": args.slot_question_isolation,
                    "target_span_isolated_to_address_channel": args.target_span_isolated_to_address_channel,
                    "released_graph_node_groundings": args.grounded_slot_descriptions,
                    "task_readout_preserves_no_effect_abstention": args.task_readout_preserves_no_effect_abstention,
                    "post_edit_structural_slot_isolation": args.post_edit_structural_slot_isolation,
                    "pointer_uses_uncontextualized_slot_names": args.pointer_uses_uncontextualized_slot_names,
                    "pointer_requires_lexical_grounding": args.pointer_requires_lexical_grounding,
                },
                "pointer_supervision": {
                    "enabled": pointer_supervision,
                    "loss_weight": args.pointer_loss_weight,
                    "gold_source": "authoritative target_slot",
                    "objective": "negative log correct-slot probability",
                },
                "paraphrase_catalog": (
                    {
                        "enabled": True,
                        "path": str(args.paraphrase_catalog),
                        **paraphrase_metadata,
                        "coverage": paraphrase_coverage_by_split,
                    }
                    if paraphrase_partition is not None else
                    {
                        "enabled": False,
                        "evidence_status": "not_applicable_to_learned_id",
                        "a1_evidence": False,
                    }
                    if args.mode == "t0" else
                    {
                        "enabled": True,
                        "source": "authoritative XOR adapter",
                        **xor_paraphrase_metadata,
                    }
                    if xor_paraphrase_metadata is not None else
                    {
                        "enabled": False,
                        "evidence_status": "not_measured",
                        "a1_evidence": False,
                    }
                ),
                "configuration": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "gpu_policy": {"model_size": args.model_size, "world_size": world_size},
                "evidence_capable_grouping": args.evidence_capable,
                "group_id_source": (
                    "authoritative XOR bundles" if args.xor_bundle else
                    "sidecar" if args.groups_dir else "derived fallback"
                ),
                "model_revision": getattr(base_reader.bert.config, "_commit_hash", None),
                "parameter_count": sum(parameter.numel() for parameter in raw_model.parameters()),
                "added_special_tokens": added_tokens,
                "do_token_id": tokenizer.convert_tokens_to_ids("[DO]"),
                "data_files_sha256": data_fingerprint(args.data_root),
                "groups_sidecar_files_sha256": (
                    sidecar_fingerprints(args.groups_dir) if args.groups_dir else None
                ),
                "data_counts": {"train": len(train_data), "validation": len(validation_data)},
                "group_counts": {
                    "train": len({example.graph_group_id for example in train_data}),
                    "validation": len({example.graph_group_id for example in validation_data}),
                },
                "validation_sampler": f"{args.validation_shard_by}-grouped, exact, no padding",
                "best_validation_metric": best_metric,
                "history": history,
                "elapsed_seconds": time.time() - started,
                "environment": {
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                    "transformers": transformers.__version__,
                    "cuda_runtime": torch.version.cuda,
                    "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "gpu": torch.cuda.get_device_name(local_rank),
                },
            }
            (args.output_dir / "run_summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n"
            )
            print(json.dumps(epoch_result, indent=2, sort_keys=True), flush=True)
        dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
