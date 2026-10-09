"""Ordered two-edit collation and execution for the scientific reader.

This is protocol plumbing, not A1 evidence.  Final two-edit labels are supplied
by an authoritative manifest through :mod:`core_bert.two_edit`; this module
never derives a composed answer from the two single-edit answers.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import nn

from .addressed_data import AddressedBatchCollator
from .addressed_data import CLOSED_VALUE_TOKENS
from .reference_operators import compact_masked_tokens
from .scientific_reader import (
    ScientificAddressedReader,
    grounded_slot_candidates,
    tokenize_scientific_fields,
)
from .t2_inline import build_t2b_masks, pool_t2_instruction
from .t3_pointer import PointerOutput, build_span_instruction, mean_pool_span
from .two_edit import OrderedTwoEditExample
from .variable_outputs import PerVariableOutputHead, VariableOutput
from .open_text_outputs import OpenTextSlotHead
from .paraphrase_protocol import ParaphrasePartition, apply_paraphrase
from .structured_paraphrases import apply_structured_paraphrase


ComposedMode = Literal["t0", "t1", "t2a", "t2b", "t3a", "t3b"]
A2Control = Literal["active", "editor_zeroed", "no_instruction", "prompting"]


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _validate_truth(example: OrderedTwoEditExample) -> None:
    size = len(example.gold_outputs)
    if size < 1 or len(example.factual_outputs) != size:
        raise ValueError(f"{example.pair_id}: authoritative final two-edit truth is required")
    masks = (example.change_mask, example.preservation_mask, example.target_mask)
    if any(len(mask) != size for mask in masks):
        raise ValueError(f"{example.pair_id}: truth masks must match final output width")
    if not all(isinstance(flag, bool) for mask in masks for flag in mask):
        raise ValueError(f"{example.pair_id}: truth masks must contain booleans")
    if not all(any(mask) for mask in masks):
        raise ValueError(f"{example.pair_id}: truth masks must each contain an evaluation cell")
    expected_change = tuple(
        after != before
        for after, before in zip(example.gold_outputs, example.factual_outputs)
    )
    if example.change_mask != expected_change:
        raise ValueError(f"{example.pair_id}: change mask disagrees with authoritative truth")
    if example.preservation_mask != tuple(not flag for flag in expected_change):
        raise ValueError(f"{example.pair_id}: preservation mask must cover every unchanged cell")


@dataclass(frozen=True)
class ComposedScientificBatch:
    """Two aligned single-edit batches plus explicit composed ground truth."""

    examples: tuple[OrderedTwoEditExample, ...]
    first: Mapping[str, Any]
    second: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.examples:
            raise ValueError("composed batch cannot be empty")
        pair_ids = [example.pair_id for example in self.examples]
        if len(pair_ids) != len(set(pair_ids)):
            raise ValueError("composed batch pair IDs must be unique")
        for example in self.examples:
            _validate_truth(example)
        expected_first = [example.first_record_id for example in self.examples]
        expected_second = [example.second_record_id for example in self.examples]
        if list(self.first.get("record_ids", ())) != expected_first:
            raise ValueError("first batch record IDs do not match declared edit order")
        if list(self.second.get("record_ids", ())) != expected_second:
            raise ValueError("second batch record IDs do not match declared edit order")

    @property
    def pair_ids(self) -> tuple[str, ...]:
        return tuple(example.pair_id for example in self.examples)

    @property
    def a1_evidence(self) -> bool:
        return False


class ComposedScientificCollator:
    """Resolve ordered pairs into two aligned scientific-reader batches."""

    def __init__(
        self,
        tokenizer: Any,
        records: Mapping[str, Mapping[str, Any]],
        *,
        max_length: int = 512,
        max_slots: int = 30,
        paraphrase_partition: ParaphrasePartition | None = None,
        split: str | None = None,
        allow_materialized_paraphrases: bool = False,
    ) -> None:
        if max_slots != 30:
            raise ValueError("scientific two-edit protocol fixes max_slots at 30")
        self.records = records
        self.addressed = AddressedBatchCollator(tokenizer, max_length=max_length)
        self.max_slots = max_slots
        self.paraphrase_partition = paraphrase_partition
        self.split = split
        self.allow_materialized_paraphrases = allow_materialized_paraphrases
        if paraphrase_partition is not None and split not in {"train", "validation"}:
            raise ValueError("composed paraphrases require a train or validation bank")

    def _resolve(
        self, examples: Sequence[OrderedTwoEditExample], ordinal: int
    ) -> list[Mapping[str, Any]]:
        rows = []
        for example in examples:
            _validate_truth(example)
            record_id = (
                example.first_record_id if ordinal == 1 else example.second_record_id
            )
            if record_id not in self.records:
                raise ValueError(f"{example.pair_id}: missing component record {record_id!r}")
            record = self.records[record_id]
            if record.get("source") != example.source:
                raise ValueError(f"{example.pair_id}: component source changed after validation")
            expected_intervention = (
                example.first_intervention if ordinal == 1 else example.second_intervention
            )
            if _stable_json(record.get("intervention")) != _stable_json(expected_intervention):
                raise ValueError(
                    f"{example.pair_id}: component intervention changed after manifest validation"
                )
            if self.paraphrase_partition is None:
                rows.append(record)
            elif self.allow_materialized_paraphrases:
                prepared, _, _ = apply_structured_paraphrase(
                    record,
                    self.paraphrase_partition,
                    split=self.split,
                    allow_materialized=True,
                )
                rows.append(prepared)
            else:
                rows.append(
                    apply_paraphrase(record, self.paraphrase_partition, split=self.split)
                )
        return rows

    def __call__(
        self, examples: Sequence[OrderedTwoEditExample]
    ) -> ComposedScientificBatch:
        if not examples:
            raise ValueError("cannot collate an empty composed batch")
        examples = tuple(examples)
        first_records = self._resolve(examples, 1)
        second_records = self._resolve(examples, 2)
        for example, first, second in zip(examples, first_records, second_records):
            shared_fields = ("source", "factual", "structure_kind", "graph", "chain")
            if any(
                _stable_json(first.get(field)) != _stable_json(second.get(field))
                for field in shared_fields
            ):
                raise ValueError(
                    f"{example.pair_id}: component records do not share one scientific world"
                )

        def collate(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
            batch = self.addressed(records)
            batch.update(
                tokenize_scientific_fields(
                    self.addressed.tokenizer, records, max_slots=self.max_slots
                )
            )
            return batch

        return ComposedScientificBatch(examples, collate(first_records), collate(second_records))


@dataclass(frozen=True)
class ComposedEditTrace:
    """Address and state transition retained for one declared edit position."""

    ordinal: int
    component_record_ids: tuple[str, ...]
    target_slot: torch.Tensor | None
    address: torch.Tensor | None
    instruction: torch.Tensor
    slots_before: torch.Tensor
    slots_after: torch.Tensor
    pointer_probabilities: torch.Tensor | None = None
    pointer_mass: torch.Tensor | None = None
    pointer_top1: torch.Tensor | None = None


@dataclass(frozen=True)
class ComposedScientificOutput:
    logits: torch.Tensor
    pre_edit_slots: torch.Tensor
    edited_slots: torch.Tensor
    traces: tuple[ComposedEditTrace, ComposedEditTrace]
    pair_ids: tuple[str, ...]
    decoded_slots: torch.Tensor | None = None
    protocol: str = "ordered two-edit execution; not A1 evidence"
    a1_evidence: bool = False


@dataclass(frozen=True)
class ComposedVariableOutput:
    composed: ComposedScientificOutput
    variables: VariableOutput


class ComposedVariableModel(nn.Module):
    """Decode every active final slot after ordered composed execution."""

    def __init__(
        self,
        executor: "ComposedScientificExecutor",
        variable_head: PerVariableOutputHead,
    ) -> None:
        super().__init__()
        self.executor = executor
        self.variable_head = variable_head

    def forward(self, batch: ComposedScientificBatch) -> ComposedVariableOutput:
        composed = self.executor(batch)
        first_mask = batch.first.get("slot_mask")
        second_mask = batch.second.get("slot_mask")
        if not isinstance(first_mask, torch.Tensor) or not torch.equal(first_mask, second_mask):
            raise ValueError("composed variable output requires one shared structural slot mask")
        variables = self.variable_head(
            composed.decoded_slots
            if composed.decoded_slots is not None else composed.edited_slots,
            first_mask,
        )
        return ComposedVariableOutput(composed, variables)


@dataclass(frozen=True)
class ComposedOpenTextOutput:
    composed: ComposedScientificOutput
    generated_token_ids: torch.Tensor


class ComposedOpenTextModel(nn.Module):
    """Execute two edits and generate arbitrary text for every active final slot."""

    def __init__(
        self, executor: "ComposedScientificExecutor", open_text_head: OpenTextSlotHead,
        *, max_tokens: int = 32,
    ) -> None:
        super().__init__()
        self.executor = executor
        self.open_text_head = open_text_head
        self.max_tokens = max_tokens

    def forward(self, batch: ComposedScientificBatch) -> ComposedOpenTextOutput:
        composed = self.executor(batch)
        first_mask = batch.first.get("slot_mask")
        second_mask = batch.second.get("slot_mask")
        if not isinstance(first_mask, torch.Tensor) or not torch.equal(first_mask, second_mask):
            raise ValueError("composed open-text output requires one shared structural slot mask")
        slots = composed.decoded_slots if composed.decoded_slots is not None else composed.edited_slots
        generated = self.open_text_head.generate(slots, first_mask, max_tokens=self.max_tokens)
        return ComposedOpenTextOutput(composed, generated)


@dataclass(frozen=True)
class _EncodedEdit:
    trunk: torch.Tensor
    valid: torch.Tensor
    pre_edit_slots: torch.Tensor
    instruction: torch.Tensor
    address: torch.Tensor | None = None
    hidden_instruction_tokens: torch.Tensor | None = None
    pointer_slots: torch.Tensor | None = None
    pointer_candidates: torch.Tensor | None = None
    address_grounded: torch.Tensor | None = None
    slot_support: torch.Tensor | None = None


class ComposedScientificExecutor(nn.Module):
    """Apply two addressed edits sequentially, then decode exactly once.

    Each instruction/address is independently encoded against the same factual
    world.  The second editor receives the first editor's output, which makes
    declared order real and permits a shared-target operator to implement
    last-write-wins.  No single-edit answer is used by this forward path.
    """

    def __init__(
        self,
        reader: ScientificAddressedReader,
        editor: nn.Module,
        mode: ComposedMode,
        *,
        editor_enabled: bool = True,
        a2_control: A2Control = "active",
        pooling: Literal["mean", "last"] = "mean",
        value_embedding: nn.Module | None = None,
        intervention_encoder: nn.Module | None = None,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        if mode not in {"t0", "t1", "t2a", "t2b", "t3a", "t3b"}:
            raise ValueError(f"unsupported composed mode: {mode}")
        if not editor_enabled and not mode.startswith("t2"):
            raise ValueError("editor-disabled control is defined only for T2")
        if a2_control not in {"active", "editor_zeroed", "no_instruction", "prompting"}:
            raise ValueError(f"unsupported A2 control: {a2_control}")
        if a2_control != "active" and mode != "t2b":
            raise ValueError("A2 controls are defined only for T2-b")
        if a2_control == "active" and not editor_enabled:
            a2_control = "editor_zeroed"
        if a2_control != "active" and editor_enabled:
            raise ValueError("A2 baseline controls must bypass the editor")
        self.reader = reader
        self.editor = editor
        self.mode = mode
        self.editor_enabled = editor_enabled
        self.a2_control = a2_control
        self.pooling = pooling
        self.value_embedding = value_embedding
        self.intervention_encoder = intervention_encoder
        self.padding_idx = int(padding_idx)

    def _encode_reference(self, side: Mapping[str, Any]) -> _EncodedEdit:
        if self.intervention_encoder is None:
            raise ValueError(f"{self.mode.upper()} requires its registered intervention encoder")
        required = ("command_tokens", "do_tokens", "address_marker_tokens", "target_slot", "value_ids")
        missing = [key for key in required if key not in side]
        if missing:
            raise ValueError(f"reference composed batch missing fields: {missing}")
        common = self._common(side)
        common["attention_mask"] = (
            common["attention_mask"].bool()
            & ~side["command_tokens"].bool()
            & ~side["do_tokens"].bool()
            & ~side["address_marker_tokens"].bool()
        ).long()
        self.reader._validate_structural_inputs(**common)
        trunk, valid = self.reader._trunk(
            common["input_ids"], common["attention_mask"],
            common["slot_name_input_ids"], common["slot_name_attention_mask"],
            common["slot_mask"],
        )
        if self.mode == "t0":
            if bool((side["value_ids"] < 0).any()):
                raise ValueError("T0 composed execution requires closed value-set IDs")
            ids = side["target_slot"] * len(CLOSED_VALUE_TOKENS) + side["value_ids"]
            instruction = self.intervention_encoder(ids)
        else:
            ids, mask = compact_masked_tokens(
                common["input_ids"], side["command_tokens"], self.padding_idx,
                max_length=self.intervention_encoder.max_length,
            )
            instruction = self.intervention_encoder(ids, mask)
        return _EncodedEdit(
            trunk, valid, trunk[:, -self.reader.max_slots :], instruction,
        )

    def _reference(self, batch: ComposedScientificBatch) -> ComposedScientificOutput:
        first = self._encode_reference(batch.first)
        second = self._encode_reference(batch.second)
        after_first = self.editor(first.pre_edit_slots, first.instruction)
        after_second = self.editor(after_first, second.instruction)
        logits, decoded_slots = self.reader._decode_question(
            second.trunk, second.valid, after_second,
            batch.second["question_input_ids"], batch.second["question_attention_mask"],
            return_slots=True, isolate_slots=True,
        )
        traces = (
            self._trace(1, batch.first["record_ids"], batch.first["target_slot"],
                        first, first.pre_edit_slots, after_first),
            self._trace(2, batch.second["record_ids"], batch.second["target_slot"],
                        second, after_first, after_second),
        )
        return ComposedScientificOutput(
            logits, first.pre_edit_slots, after_second, traces, batch.pair_ids,
            decoded_slots=decoded_slots,
        )

    @staticmethod
    def _common(batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        required = (
            "input_ids", "attention_mask", "slot_name_input_ids",
            "slot_name_attention_mask", "slot_mask", "question_input_ids",
            "question_attention_mask",
        )
        missing = [key for key in required if key not in batch]
        if missing:
            raise ValueError(f"composed reader batch missing fields: {missing}")
        return {key: batch[key] for key in required}

    @staticmethod
    def _validate_shared_world(batch: ComposedScientificBatch) -> None:
        """Reject two sides that do not encode the same world and query."""

        exact_fields = (
            "slot_name_input_ids", "slot_name_attention_mask", "slot_mask",
            "question_input_ids", "question_attention_mask",
        )
        for key in exact_fields:
            first, second = batch.first.get(key), batch.second.get(key)
            if not isinstance(first, torch.Tensor) or not isinstance(second, torch.Tensor):
                raise ValueError(f"composed reader batch missing tensor field {key}")
            if not torch.equal(first, second):
                raise ValueError(f"composed edit sides disagree on shared field {key}")

        passages = []
        for side in (batch.first, batch.second):
            for key in ("input_ids", "attention_mask", "command_tokens", "do_tokens"):
                if key not in side:
                    raise ValueError(f"composed reader batch missing field {key}")
            if side["input_ids"].ndim != 2 or side["input_ids"].shape[0] != len(batch.examples):
                raise ValueError("composed tensor batch size must match protocol examples")
            factual_mask = (
                side["attention_mask"].bool()
                & ~side["command_tokens"].bool()
                & ~side["do_tokens"].bool()
            )
            passages.append(
                [
                    side["input_ids"][row][factual_mask[row]].detach().cpu().tolist()
                    for row in range(side["input_ids"].shape[0])
                ]
            )
        if passages[0] != passages[1]:
            raise ValueError("composed edit sides do not encode the same factual passage")

    @staticmethod
    def _trace(
        ordinal: int,
        record_ids: Sequence[str],
        target_slot: torch.Tensor | None,
        encoded: _EncodedEdit,
        slots_before: torch.Tensor,
        slots_after: torch.Tensor,
        *,
        probabilities: torch.Tensor | None = None,
    ) -> ComposedEditTrace:
        mass = top1 = None
        if probabilities is not None:
            if target_slot is None:
                raise ValueError("pointer traces require authoritative target slots")
            rows = torch.arange(len(target_slot), device=target_slot.device)
            mass = probabilities[rows, target_slot]
            top1 = probabilities.argmax(-1).eq(target_slot)
        return ComposedEditTrace(
            ordinal=ordinal,
            component_record_ids=tuple(record_ids),
            target_slot=target_slot,
            address=encoded.address,
            instruction=encoded.instruction,
            slots_before=slots_before,
            slots_after=slots_after,
            pointer_probabilities=probabilities,
            pointer_mass=mass,
            pointer_top1=top1,
        )

    def _encode_t2(self, side: Mapping[str, Any]) -> _EncodedEdit:
        common = self._common(side)
        for key in ("command_tokens", "do_tokens"):
            if key not in side:
                raise ValueError(f"T2 composed batch missing {key}")
        self.reader._validate_structural_inputs(**common)
        input_ids = common["input_ids"]
        attention_mask = common["attention_mask"]
        command_tokens = side["command_tokens"]
        do_tokens = side["do_tokens"]
        if command_tokens.shape != input_ids.shape or do_tokens.shape != input_ids.shape:
            raise ValueError("T2 token masks must match passage input IDs")
        passage = self.reader.reader.bert.embeddings(input_ids=input_ids)
        slots = self.reader._dynamic_slots(
            common["slot_name_input_ids"], common["slot_name_attention_mask"]
        )
        initial = torch.cat([passage, slots], dim=1)
        slot_mask = common["slot_mask"]
        valid = torch.cat([attention_mask.bool(), slot_mask.bool()], dim=1)
        zeros = torch.zeros_like(slot_mask, dtype=torch.bool)
        command = torch.cat([command_tokens.bool(), zeros], dim=1)
        do = torch.cat([do_tokens.bool(), zeros], dim=1)
        slot_tokens = torch.cat(
            [torch.zeros_like(attention_mask, dtype=torch.bool), slot_mask.bool()], dim=1
        )
        hidden_instruction = None
        if self.mode == "t2a" or self.a2_control == "prompting":
            allowed = self.reader.reader._full_attention(valid)
        else:
            masks = build_t2b_masks(valid, command, do, slot_tokens)
            hidden_instruction = masks.instruction_tokens
            allowed = masks.pre_edit
            allowed &= ~(
                (~hidden_instruction)[:, :, None] & hidden_instruction[:, None, :]
            )
            if self.a2_control == "no_instruction":
                valid = valid & ~command & ~do
                allowed &= valid[:, :, None] & valid[:, None, :]
                hidden_instruction = None
        trunk = self.reader.reader._run_layers(
            self.reader.reader.bert.encoder.layer[: self.reader.reader.split_layer],
            initial,
            allowed,
        )
        instruction = (
            torch.zeros(
                trunk.shape[0], trunk.shape[-1], dtype=trunk.dtype, device=trunk.device
            )
            if self.a2_control == "no_instruction"
            else pool_t2_instruction(
                trunk[:, : input_ids.shape[1]], command_tokens, do_tokens, self.pooling
            )
        )
        return _EncodedEdit(
            trunk, valid, trunk[:, -self.reader.max_slots :], instruction,
            hidden_instruction_tokens=hidden_instruction,
        )

    def _t2(self, batch: ComposedScientificBatch) -> ComposedScientificOutput:
        first = self._encode_t2(batch.first)
        second = self._encode_t2(batch.second)
        after_first = (
            self.editor(first.pre_edit_slots, first.instruction)
            if self.editor_enabled else first.pre_edit_slots
        )
        after_second = (
            self.editor(after_first, second.instruction)
            if self.editor_enabled else after_first
        )
        if (
            after_first.shape != first.pre_edit_slots.shape
            or after_second.shape != after_first.shape
        ):
            raise ValueError("each T2 edit must preserve the scientific slot tensor shape")
        logits = self.reader._decode_question(
            second.trunk, second.valid, after_second,
            batch.second["question_input_ids"], batch.second["question_attention_mask"],
            second.hidden_instruction_tokens,
        )
        traces = (
            self._trace(
                1, batch.first["record_ids"], None, first,
                first.pre_edit_slots, after_first,
            ),
            self._trace(
                2, batch.second["record_ids"], None, second,
                after_first, after_second,
            ),
        )
        return ComposedScientificOutput(
            logits, first.pre_edit_slots, after_second,
            traces, batch.pair_ids,
        )

    def _encode_t3(self, side: Mapping[str, Any]) -> _EncodedEdit:
        required = (
            "target_tokens", "target_slot", "command_tokens", "do_tokens",
            "address_marker_tokens", "slot_support",
        )
        missing = [key for key in required if key not in side]
        if missing:
            raise ValueError(f"T3 composed batch missing fields: {missing}")
        embedded_value = side.get("value_embedding")
        replacement = side.get("replacement_tokens")
        has_value = embedded_value is not None
        has_replacement = replacement is not None and bool(replacement.bool().any())
        if not has_value and not has_replacement and "value_ids" in side:
            value_ids = side["value_ids"]
            if value_ids.ndim != 1 or not bool((value_ids >= 0).all()):
                raise ValueError("a T3 batch must not mix value-set and event-replacement edits")
            if self.value_embedding is None:
                raise ValueError("value-set T3 execution requires a closed value embedding")
            embedded_value = self.value_embedding(value_ids)
            has_value = True
        if has_replacement and not bool(replacement.bool().any(1).all()):
            raise ValueError("a T3 batch must not mix value-set and event-replacement edits")
        if has_value == has_replacement:
            raise ValueError(
                "each T3 edit must provide exactly one of value_embedding or replacement span"
            )
        common = self._common(side)
        common["attention_mask"] = (
            common["attention_mask"].bool()
            & ~side["command_tokens"].bool()
            & ~side["do_tokens"].bool()
            & ~side["address_marker_tokens"].bool()
            & ~side["target_tokens"].bool()
        ).long()
        self.reader._validate_structural_inputs(
            **common, target_slot=side["target_slot"]
        )
        span_hidden = self.reader.reader.bert.embeddings(input_ids=common["input_ids"])
        pointer_slots = self.reader._dynamic_slots(
            common["slot_name_input_ids"], common["slot_name_attention_mask"]
        )
        trunk, valid = self.reader._trunk(
            common["input_ids"], common["attention_mask"],
            common["slot_name_input_ids"], common["slot_name_attention_mask"],
            common["slot_mask"], initial_slots=pointer_slots,
        )
        address = mean_pool_span(span_hidden, side["target_tokens"])
        pointer_candidates, address_grounded = grounded_slot_candidates(
            common["input_ids"], side["target_tokens"],
            common["slot_name_input_ids"], common["slot_name_attention_mask"],
            common["slot_mask"],
        )
        instruction = build_span_instruction(
            span_hidden, side["target_tokens"], value_embedding=embedded_value,
            replacement_span=replacement if has_replacement else None,
        )
        return _EncodedEdit(
            trunk, valid, trunk[:, -self.reader.max_slots :], instruction,
            address=address, pointer_slots=pointer_slots,
            pointer_candidates=pointer_candidates,
            address_grounded=address_grounded, slot_support=side["slot_support"],
        )

    def _apply_t3(
        self, slots: torch.Tensor, encoded: _EncodedEdit, slot_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.mode == "t3a":
            edited = self.editor(slots, encoded.instruction)
            probabilities = None
        else:
            result = self.editor(
                slots, encoded.address, encoded.instruction, slot_mask.bool(),
                encoded.slot_support.bool(), pointer_slots=encoded.pointer_slots,
                candidate_mask=encoded.pointer_candidates,
                edit_enabled=encoded.address_grounded,
            )
            if isinstance(result, PointerOutput):
                edited, probabilities = result.edited_slots, result.probabilities
            else:
                try:
                    edited, probabilities = result.edited_slots, result.probabilities
                except AttributeError as error:
                    raise TypeError("T3-b editor must return pointer output") from error
        if edited.shape != slots.shape:
            raise ValueError("each T3 edit must preserve the scientific slot tensor shape")
        return edited, probabilities

    def _t3(self, batch: ComposedScientificBatch) -> ComposedScientificOutput:
        first = self._encode_t3(batch.first)
        second = self._encode_t3(batch.second)
        after_first, first_probabilities = self._apply_t3(
            first.pre_edit_slots, first, batch.first["slot_mask"]
        )
        after_second, second_probabilities = self._apply_t3(
            after_first, second, batch.second["slot_mask"]
        )
        logits, decoded_slots = self.reader._decode_question(
            second.trunk, second.valid, after_second,
            batch.second["question_input_ids"], batch.second["question_attention_mask"],
            return_slots=True, isolate_slots=(self.mode == "t3b"),
        )
        traces = (
            self._trace(
                1, batch.first["record_ids"], batch.first["target_slot"],
                first, first.pre_edit_slots, after_first,
                probabilities=first_probabilities,
            ),
            self._trace(
                2, batch.second["record_ids"], batch.second["target_slot"],
                second, after_first, after_second,
                probabilities=second_probabilities,
            ),
        )
        return ComposedScientificOutput(
            logits, first.pre_edit_slots, after_second,
            traces, batch.pair_ids, decoded_slots=decoded_slots,
        )

    def forward(self, batch: ComposedScientificBatch) -> ComposedScientificOutput:
        if not isinstance(batch, ComposedScientificBatch):
            raise TypeError("composed execution requires a validated ComposedScientificBatch")
        # Recheck truth at the execution boundary in case a caller constructed
        # a protocol dataclass without using the manifest builder.
        for example in batch.examples:
            _validate_truth(example)
        self._validate_shared_world(batch)
        if self.mode in {"t0", "t1"}:
            return self._reference(batch)
        return self._t2(batch) if self.mode.startswith("t2") else self._t3(batch)


__all__ = [
    "ComposedEditTrace",
    "ComposedScientificBatch",
    "ComposedScientificCollator",
    "ComposedScientificExecutor",
    "ComposedScientificOutput",
    "ComposedVariableModel",
    "ComposedVariableOutput",
]
