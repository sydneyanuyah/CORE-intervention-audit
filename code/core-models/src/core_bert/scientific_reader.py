"""Question-conditioned, variable-aligned reader for scientific T2/T3 runs.

This module wraps :class:`AddressedWorldReader` without changing its engineering
smoke interface.  Structural names become dynamic world slots before the edit;
question text is embedded and appended only after the edit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

import torch
from torch import nn

from .addressed_reader import AddressedWorldReader
from .t2_inline import build_t2b_masks, pool_t2_instruction
from .t3_pointer import PointerOutput, build_span_instruction, mean_pool_span


T2Mode = Literal["t2a", "t2b"]
T3Mode = Literal["t3a", "t3b"]


def grounded_slot_candidates(
    input_ids: torch.Tensor,
    target_span: torch.Tensor,
    slot_name_input_ids: torch.Tensor,
    slot_name_attention_mask: torch.Tensor,
    slot_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Restrict a textual address to slots with maximal token grounding."""

    if input_ids.shape != target_span.shape:
        raise ValueError("input_ids and target_span must share [batch, sequence]")
    if slot_name_input_ids.shape != slot_name_attention_mask.shape:
        raise ValueError("slot name IDs and attention mask must match")
    if slot_name_input_ids.ndim != 3 or slot_mask.shape != slot_name_input_ids.shape[:2]:
        raise ValueError("slot names must have shape [batch, slots, length]")
    selected = target_span.bool()
    if not torch.all(selected.any(1)):
        raise ValueError("every textual address must contain at least one token")
    matches = (
        input_ids[:, None, :, None].eq(slot_name_input_ids[:, :, None, :])
        & selected[:, None, :, None]
        & slot_name_attention_mask[:, :, None, :].bool()
    )
    overlap = matches.any(-1).sum(-1).float() / selected.sum(1)[:, None]
    overlap = overlap.masked_fill(~slot_mask.bool(), -1.0)
    best = overlap.max(1).values
    grounded = best > 0
    candidates = overlap.eq(best[:, None]) & slot_mask.bool() & grounded[:, None]
    return candidates, grounded


def build_post_edit_decode_mask(
    valid: torch.Tensor, passage_length: int, *, slot_count: int | None = None,
    isolate_slots: bool = False,
) -> torch.Tensor:
    """Make edited slots the exclusive passage-to-answer bottleneck."""

    if valid.ndim != 2 or valid.dtype != torch.bool:
        raise ValueError("valid must be a boolean [batch, sequence] tensor")
    if not 0 < passage_length < valid.shape[1]:
        raise ValueError("passage_length must leave post-edit tokens")
    allowed = valid[:, :, None] & valid[:, None, :]
    positions = torch.arange(valid.shape[1], device=valid.device)
    passage_keys = positions < passage_length
    post_edit_queries = ~passage_keys
    allowed &= ~(
        post_edit_queries[None, :, None] & passage_keys[None, None, :]
    )
    if slot_count is not None:
        if not 0 < slot_count < valid.shape[1] - passage_length:
            raise ValueError("slot_count must leave at least one question token")
        slot_queries = (
            (positions >= passage_length)
            & (positions < passage_length + slot_count)
        )
        question_keys = positions >= passage_length + slot_count
        allowed &= ~(
            slot_queries[None, :, None] & question_keys[None, None, :]
        )
        if isolate_slots:
            slot_keys = slot_queries
            same_slot = positions[:, None].eq(positions[None, :])
            allowed &= ~(
                slot_queries[None, :, None]
                & slot_keys[None, None, :]
                & ~same_slot[None, :, :]
            )
    elif isolate_slots:
        raise ValueError("isolate_slots requires slot_count")
    return allowed


@dataclass(frozen=True)
class VariableSlotLayout:
    names: tuple[str, ...]
    display_names: tuple[str, ...]
    target_slot: int


@dataclass(frozen=True)
class ScientificReaderOutput:
    logits: torch.Tensor
    pre_edit_slots: torch.Tensor
    edited_slots: torch.Tensor
    instruction: torch.Tensor
    pointer_probabilities: torch.Tensor | None = None
    pointer_mass: torch.Tensor | None = None
    pointer_top1: torch.Tensor | None = None
    decoded_slots: torch.Tensor | None = None


def variable_slot_layout(
    record: Mapping[str, Any], *, max_slots: int = 30
) -> VariableSlotLayout:
    """Derive deterministic structural slot names and the unique target slot."""

    if max_slots < 1:
        raise ValueError("max_slots must be positive")
    intervention = record.get("intervention")
    if not isinstance(intervention, Mapping):
        raise ValueError("record.intervention must be an object")
    target = intervention.get("target")
    if not isinstance(target, str) or not target:
        raise ValueError("intervention.target must be a non-empty structural ID")

    kind = record.get("structure_kind")
    if kind == "dag":
        graph = record.get("graph")
        raw_names = graph.get("nodes") if isinstance(graph, Mapping) else None
        if not isinstance(raw_names, list) or not all(
            isinstance(name, str) and name for name in raw_names
        ):
            raise ValueError("DAG graph.nodes must contain non-empty strings")
        names = tuple(raw_names)
    elif kind == "chain":
        chain = record.get("chain")
        if not isinstance(chain, list) or not all(
            isinstance(event, str) and event for event in chain
        ):
            raise ValueError("chain must contain non-empty event strings")
        # Occurrence-qualified names preserve deterministic order even when
        # identical event text occurs more than once in a chain.
        names = tuple(f"event_{index:02d}: {event}" for index, event in enumerate(chain))
    else:
        raise ValueError("structure_kind must be 'dag' or 'chain'")

    if not names or len(names) > max_slots:
        raise ValueError(f"record must contain 1..{max_slots} structural variables")
    if len(set(names)) != len(names):
        raise ValueError("structural slot names must be unique")
    display_names = names
    if kind == "dag" and isinstance(graph, Mapping) and graph.get("node_text") is not None:
        node_text = graph["node_text"]
        if not isinstance(node_text, Mapping) or set(node_text) != set(names):
            raise ValueError("graph.node_text must describe every structural slot")
        descriptions = []
        for name in names:
            surfaces = node_text[name]
            if not isinstance(surfaces, list) or not surfaces or not all(
                isinstance(surface, str) and surface.strip() for surface in surfaces
            ):
                raise ValueError("graph.node_text descriptions must be non-empty string lists")
            descriptions.append(" ; ".join(surfaces))
        display_names = tuple(descriptions)
    matches = [index for index, name in enumerate(names) if name == target]
    if len(matches) != 1:
        raise ValueError(
            f"intervention target must map to exactly one slot, found {len(matches)}"
        )
    return VariableSlotLayout(names, display_names, matches[0])


def tokenize_scientific_fields(
    tokenizer: Any,
    records: Sequence[Mapping[str, Any]],
    *,
    max_slots: int = 30,
    max_name_length: int = 24,
    max_question_length: int = 128,
) -> dict[str, torch.Tensor]:
    """Tokenize dynamic slot names and post-edit questions for a record batch."""

    if not records:
        raise ValueError("cannot tokenize an empty record batch")
    layouts = [variable_slot_layout(record, max_slots=max_slots) for record in records]
    padded_names = [
        [*layout.display_names, *([""] * (max_slots - len(layout.display_names)))]
        for layout in layouts
    ]
    flat_names = [name for row in padded_names for name in row]
    encoded_names = tokenizer(
        flat_names,
        add_special_tokens=False,
        padding="max_length",
        truncation=True,
        max_length=max_name_length,
        return_tensors="pt",
    )
    questions = []
    for record in records:
        factual = record.get("factual")
        question = factual.get("question") if isinstance(factual, Mapping) else None
        if not isinstance(question, str) or not question:
            raise ValueError("factual.question must be non-empty for scientific runs")
        questions.append(question)
    encoded_questions = tokenizer(
        questions,
        padding=True,
        truncation=True,
        max_length=max_question_length,
        return_tensors="pt",
    )
    batch_size = len(records)
    slot_mask = torch.zeros(batch_size, max_slots, dtype=torch.bool)
    slot_support = torch.zeros(
        batch_size, max_slots, max_slots, dtype=torch.bool
    )
    for row, layout in enumerate(layouts):
        slot_mask[row, : len(layout.names)] = True
        children = {name: [] for name in layout.names}
        record = records[row]
        if record.get("structure_kind") == "dag":
            for edge in record["graph"]["edges"]:
                if not isinstance(edge, list) or len(edge) != 2 or any(
                    node not in children for node in edge
                ):
                    raise ValueError("graph edges must connect declared structural slots")
                children[edge[0]].append(edge[1])
        else:
            for left, right in zip(layout.names, layout.names[1:]):
                children[left].append(right)
        index = {name: position for position, name in enumerate(layout.names)}
        for source in layout.names:
            seen, pending = {source}, [source]
            while pending:
                current = pending.pop()
                for child in children[current]:
                    if child not in seen:
                        seen.add(child)
                        pending.append(child)
            for target in seen:
                slot_support[row, index[source], index[target]] = True
    return {
        "slot_name_input_ids": encoded_names["input_ids"].reshape(
            batch_size, max_slots, -1
        ),
        "slot_name_attention_mask": encoded_names["attention_mask"].reshape(
            batch_size, max_slots, -1
        ),
        "slot_mask": slot_mask,
        "slot_support": slot_support,
        "target_slot": torch.tensor(
            [layout.target_slot for layout in layouts], dtype=torch.long
        ),
        "question_input_ids": encoded_questions["input_ids"],
        "question_attention_mask": encoded_questions["attention_mask"],
    }


class ScientificAddressedReader(nn.Module):
    """Wrap an addressed reader with dynamic slots and post-edit text queries."""

    def __init__(self, reader: AddressedWorldReader, *, max_slots: int = 30):
        super().__init__()
        if max_slots != 30:
            raise ValueError("scientific protocol fixes max_slots at 30")
        self.reader = reader
        self.max_slots = max_slots

    @property
    def hidden_size(self) -> int:
        return self.reader.hidden_size

    def _validate_structural_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        slot_name_input_ids: torch.Tensor,
        slot_name_attention_mask: torch.Tensor,
        slot_mask: torch.Tensor,
        question_input_ids: torch.Tensor,
        question_attention_mask: torch.Tensor,
        target_slot: torch.Tensor | None = None,
    ) -> None:
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("input IDs and attention mask must share [batch, sequence]")
        expected_slot_prefix = (input_ids.shape[0], self.max_slots)
        if slot_name_input_ids.ndim != 3 or slot_name_input_ids.shape[:2] != expected_slot_prefix:
            raise ValueError("slot name IDs must have shape [batch, 30, name_length]")
        if slot_name_attention_mask.shape != slot_name_input_ids.shape:
            raise ValueError("slot name mask must match slot name IDs")
        if slot_mask.shape != expected_slot_prefix:
            raise ValueError("slot_mask must have shape [batch, 30]")
        if not torch.all(slot_mask.bool().any(1)):
            raise ValueError("each record must expose at least one structural slot")
        if question_input_ids.ndim != 2 or question_input_ids.shape[0] != input_ids.shape[0]:
            raise ValueError("question IDs must have shape [batch, question_length]")
        if question_attention_mask.shape != question_input_ids.shape:
            raise ValueError("question mask must match question IDs")
        if not torch.all(question_attention_mask.bool().any(1)):
            raise ValueError("each record must contain question tokens")
        if target_slot is not None:
            if target_slot.shape != (input_ids.shape[0],):
                raise ValueError("target_slot must have shape [batch]")
            rows = torch.arange(input_ids.shape[0], device=target_slot.device)
            valid_target = (target_slot >= 0) & (target_slot < self.max_slots)
            safe = target_slot.clamp(0, self.max_slots - 1)
            valid_target &= slot_mask.bool()[rows, safe]
            if not torch.all(valid_target):
                raise ValueError("target_slot must identify an active structural slot")

    def _dynamic_slots(
        self,
        slot_name_input_ids: torch.Tensor,
        slot_name_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, slots, length = slot_name_input_ids.shape
        flat_ids = slot_name_input_ids.reshape(batch * slots, length)
        embedded = self.reader.bert.embeddings(input_ids=flat_ids)
        mask = slot_name_attention_mask.reshape(batch * slots, length).to(embedded.dtype)
        pooled = (embedded * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        return pooled.reshape(batch, slots, self.hidden_size)

    def _trunk(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        slot_name_input_ids: torch.Tensor,
        slot_name_attention_mask: torch.Tensor,
        slot_mask: torch.Tensor,
        allowed: torch.Tensor | None = None,
        initial_slots: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        passage = self.reader.bert.embeddings(input_ids=input_ids)
        slots = (
            self._dynamic_slots(slot_name_input_ids, slot_name_attention_mask)
            if initial_slots is None else initial_slots
        )
        if slots.shape != (input_ids.shape[0], self.max_slots, self.hidden_size):
            raise ValueError("initial_slots must have shape [batch, 30, hidden]")
        hidden = torch.cat([passage, slots], dim=1)
        valid = torch.cat([attention_mask.bool(), slot_mask.bool()], dim=1)
        if allowed is None:
            allowed = self.reader._full_attention(valid)
        hidden = self.reader._run_layers(
            self.reader.bert.encoder.layer[: self.reader.split_layer], hidden, allowed
        )
        return hidden, valid

    def _decode_question(
        self,
        trunk: torch.Tensor,
        trunk_valid: torch.Tensor,
        edited_slots: torch.Tensor,
        question_input_ids: torch.Tensor,
        question_attention_mask: torch.Tensor,
        hidden_instruction_tokens: torch.Tensor | None = None,
        return_slots: bool = False,
        isolate_slots: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        hidden = trunk.clone()
        hidden[:, -self.max_slots :] = edited_slots
        question = self.reader.bert.embeddings(input_ids=question_input_ids)
        hidden = torch.cat([hidden, question], dim=1)
        question_valid = question_attention_mask.bool()
        valid = torch.cat([trunk_valid, question_valid], dim=1)
        passage_length = trunk.shape[1] - self.max_slots
        allowed = build_post_edit_decode_mask(
            valid, passage_length, slot_count=self.max_slots,
            isolate_slots=isolate_slots,
        )
        if hidden_instruction_tokens is not None:
            hidden_keys = torch.cat(
                [hidden_instruction_tokens.bool(), torch.zeros_like(question_valid)], dim=1
            )
            allowed &= ~hidden_keys[:, None, :]
        hidden = self.reader._run_layers(
            self.reader.bert.encoder.layer[self.reader.split_layer :], hidden, allowed
        )
        question_hidden = hidden[:, -question_input_ids.shape[1] :]
        weights = question_valid.to(question_hidden.dtype)
        pooled = (question_hidden * weights[..., None]).sum(1) / weights.sum(
            1, keepdim=True
        ).clamp_min(1)
        logits = self.reader.answer_head(pooled)
        if return_slots:
            decoded_slots = hidden[
                :, passage_length : passage_length + self.max_slots
            ]
            return logits, decoded_slots
        return logits

    def forward_reference(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        slot_name_input_ids: torch.Tensor,
        slot_name_attention_mask: torch.Tensor,
        slot_mask: torch.Tensor,
        question_input_ids: torch.Tensor,
        question_attention_mask: torch.Tensor,
        intervention: torch.Tensor,
        editor: nn.Module,
    ) -> ScientificReaderOutput:
        """Execute a T0/T1 edit while keeping command text outside the reader."""

        self._validate_structural_inputs(
            input_ids, attention_mask, slot_name_input_ids,
            slot_name_attention_mask, slot_mask, question_input_ids,
            question_attention_mask,
        )
        if intervention.shape != (input_ids.shape[0], self.hidden_size):
            raise ValueError("reference intervention must have shape [batch, hidden]")
        trunk, valid = self._trunk(
            input_ids, attention_mask, slot_name_input_ids,
            slot_name_attention_mask, slot_mask,
        )
        pre_edit_slots = trunk[:, -self.max_slots :]
        edited = editor(pre_edit_slots, intervention)
        logits, decoded_slots = self._decode_question(
            trunk, valid, edited, question_input_ids, question_attention_mask,
            return_slots=True, isolate_slots=True,
        )
        return ScientificReaderOutput(
            logits, pre_edit_slots, edited, intervention,
            decoded_slots=decoded_slots,
        )

    def forward_t2(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        command_tokens: torch.Tensor,
        do_tokens: torch.Tensor,
        slot_name_input_ids: torch.Tensor,
        slot_name_attention_mask: torch.Tensor,
        slot_mask: torch.Tensor,
        question_input_ids: torch.Tensor,
        question_attention_mask: torch.Tensor,
        editor: nn.Module,
        *,
        mode: T2Mode = "t2b",
        pooling: Literal["mean", "last"] = "mean",
        editor_enabled: bool = True,
    ) -> ScientificReaderOutput:
        self._validate_structural_inputs(
            input_ids, attention_mask, slot_name_input_ids,
            slot_name_attention_mask, slot_mask, question_input_ids,
            question_attention_mask,
        )
        if command_tokens.shape != input_ids.shape or do_tokens.shape != input_ids.shape:
            raise ValueError("T2 token masks must match passage input IDs")
        passage_hidden = self.reader.bert.embeddings(input_ids=input_ids)
        slots = self._dynamic_slots(slot_name_input_ids, slot_name_attention_mask)
        initial = torch.cat([passage_hidden, slots], dim=1)
        valid = torch.cat([attention_mask.bool(), slot_mask.bool()], dim=1)
        zeros = torch.zeros_like(slot_mask, dtype=torch.bool)
        command = torch.cat([command_tokens.bool(), zeros], dim=1)
        do = torch.cat([do_tokens.bool(), zeros], dim=1)
        slot_tokens = torch.cat(
            [torch.zeros_like(attention_mask, dtype=torch.bool), slot_mask.bool()], dim=1
        )
        if mode == "t2a":
            allowed = self.reader._full_attention(valid)
            hidden_instruction = None
        elif mode == "t2b":
            masks = build_t2b_masks(valid, command, do, slot_tokens)
            hidden_instruction = masks.instruction_tokens
            allowed = masks.pre_edit
            non_instruction_queries = ~hidden_instruction
            allowed &= ~(
                non_instruction_queries[:, :, None] & hidden_instruction[:, None, :]
            )
        else:
            raise ValueError(f"unknown T2 mode: {mode}")
        trunk = self.reader._run_layers(
            self.reader.bert.encoder.layer[: self.reader.split_layer], initial, allowed
        )
        instruction = pool_t2_instruction(
            trunk[:, : input_ids.shape[1]], command_tokens, do_tokens, pooling
        )
        pre_edit_slots = trunk[:, -self.max_slots :]
        edited = editor(pre_edit_slots, instruction) if editor_enabled else pre_edit_slots
        logits, decoded_slots = self._decode_question(
            trunk, valid, edited, question_input_ids, question_attention_mask,
            hidden_instruction, return_slots=True,
        )
        return ScientificReaderOutput(
            logits, pre_edit_slots, edited, instruction, decoded_slots=decoded_slots
        )

    def forward_t3(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_span: torch.Tensor,
        slot_name_input_ids: torch.Tensor,
        slot_name_attention_mask: torch.Tensor,
        slot_mask: torch.Tensor,
        target_slot: torch.Tensor,
        question_input_ids: torch.Tensor,
        question_attention_mask: torch.Tensor,
        editor: nn.Module,
        *,
        slot_support: torch.Tensor | None = None,
        value_embedding: torch.Tensor | None = None,
        replacement_span: torch.Tensor | None = None,
        mode: T3Mode = "t3b",
    ) -> ScientificReaderOutput:
        self._validate_structural_inputs(
            input_ids, attention_mask, slot_name_input_ids,
            slot_name_attention_mask, slot_mask, question_input_ids,
            question_attention_mask, target_slot,
        )
        if slot_support is None:
            slot_support = torch.eye(
                self.max_slots, dtype=torch.bool, device=input_ids.device
            )[None].expand(input_ids.shape[0], -1, -1)
        elif slot_support.shape != (input_ids.shape[0], self.max_slots, self.max_slots):
            raise ValueError("slot_support must have shape [batch, 30, 30]")
        # The addressed phrase is excluded from trunk attention by the caller.
        # Encode it independently so target identity can enter only through the
        # explicit address/instruction path, including wrong-span A3 controls.
        span_hidden = self.reader.bert.embeddings(input_ids=input_ids)
        pointer_slots = self._dynamic_slots(
            slot_name_input_ids, slot_name_attention_mask
        )
        trunk, valid = self._trunk(
            input_ids, attention_mask, slot_name_input_ids,
            slot_name_attention_mask, slot_mask, initial_slots=pointer_slots,
        )
        address = mean_pool_span(span_hidden, target_span)
        pointer_candidates, address_grounded = grounded_slot_candidates(
            input_ids, target_span, slot_name_input_ids,
            slot_name_attention_mask, slot_mask,
        )
        instruction = build_span_instruction(
            span_hidden, target_span, value_embedding=value_embedding,
            replacement_span=replacement_span,
        )
        pre_edit_slots = trunk[:, -self.max_slots :]
        probabilities = mass = top1 = None
        if mode == "t3a":
            edited = editor(pre_edit_slots, instruction)
        elif mode == "t3b":
            result = editor(
                pre_edit_slots, address, instruction, slot_mask.bool(), slot_support.bool(),
                pointer_slots=pointer_slots, candidate_mask=pointer_candidates,
                edit_enabled=address_grounded,
            )
            if not isinstance(result, PointerOutput):
                try:
                    edited = result.edited_slots
                    probabilities = result.probabilities
                except AttributeError as error:
                    raise TypeError("T3-b editor must return pointer output") from error
            else:
                edited = result.edited_slots
                probabilities = result.probabilities
            rows = torch.arange(len(target_slot), device=target_slot.device)
            mass = probabilities[rows, target_slot]
            top1 = probabilities.argmax(-1).eq(target_slot)
        else:
            raise ValueError(f"unknown T3 mode: {mode}")
        if edited.shape != pre_edit_slots.shape:
            raise ValueError("editor must preserve the [batch, 30, hidden] slot shape")
        logits, decoded_slots = self._decode_question(
            trunk, valid, edited, question_input_ids, question_attention_mask,
            return_slots=True, isolate_slots=(mode == "t3b"),
        )
        return ScientificReaderOutput(
            logits, pre_edit_slots, edited, instruction, probabilities, mass, top1,
            decoded_slots,
        )


__all__ = [
    "build_post_edit_decode_mask",
    "grounded_slot_candidates",
    "ScientificAddressedReader",
    "ScientificReaderOutput",
    "VariableSlotLayout",
    "tokenize_scientific_fields",
    "variable_slot_layout",
]
