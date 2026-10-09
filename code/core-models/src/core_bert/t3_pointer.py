"""T3 textual-span addressing and pointer-selection modules."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn


SpanOverride = Literal["correct", "adjacent", "random"]


def char_spans_to_token_mask(
    offset_mapping: torch.Tensor,
    char_spans: torch.Tensor,
    valid_tokens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Map gold ``[start, end)`` character offsets to overlapping input tokens."""

    if offset_mapping.ndim != 3 or offset_mapping.shape[-1] != 2:
        raise ValueError("offset_mapping must have shape [batch, sequence, 2]")
    if char_spans.shape != (offset_mapping.shape[0], 2):
        raise ValueError("char_spans must have shape [batch, 2]")
    if not torch.all(char_spans[:, 0] < char_spans[:, 1]):
        raise ValueError("character spans must be non-empty [start, end) ranges")
    token_start, token_end = offset_mapping[..., 0], offset_mapping[..., 1]
    real_text = token_end > token_start  # excludes special/padding offsets such as [0, 0]
    overlaps = (
        (token_end > char_spans[:, None, 0])
        & (token_start < char_spans[:, None, 1])
        & real_text
    )
    if valid_tokens is not None:
        if valid_tokens.shape != overlaps.shape:
            raise ValueError("valid_tokens must have shape [batch, sequence]")
        overlaps &= valid_tokens.bool()
    if not torch.all(overlaps.any(1)):
        raise ValueError("at least one character span maps to no valid input token")
    return overlaps


def mean_pool_span(hidden: torch.Tensor, span_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool hidden states selected by a non-empty token-span mask."""

    if hidden.ndim != 3 or span_mask.ndim != 2 or hidden.shape[:2] != span_mask.shape:
        raise ValueError("expected hidden [batch, sequence, hidden] and mask [batch, sequence]")
    mask = span_mask.bool()
    counts = mask.sum(1)
    if not torch.all(counts >= 1):
        raise ValueError("every span must contain at least one token")
    return (hidden * mask.unsqueeze(-1)).sum(1) / counts.unsqueeze(-1)


def build_span_instruction(
    hidden: torch.Tensor,
    target_span: torch.Tensor,
    *,
    value_embedding: torch.Tensor | None = None,
    replacement_span: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build ``[target address; value]`` or event-replacement ``[old; new]``."""

    address = mean_pool_span(hidden, target_span)
    if (value_embedding is None) == (replacement_span is None):
        raise ValueError("provide exactly one of value_embedding or replacement_span")
    second = (
        value_embedding
        if value_embedding is not None
        else mean_pool_span(hidden, replacement_span)  # type: ignore[arg-type]
    )
    if second.shape != address.shape:
        raise ValueError("value/replacement representation must match the address shape")
    return torch.cat([address, second], dim=-1)


class ClosedValueEmbedding(nn.Module):
    """Small closed-vocabulary value table; variable names never enter this table."""

    def __init__(self, value_count: int, hidden_size: int):
        super().__init__()
        if not 1 <= value_count <= 16:
            raise ValueError("the preregistered closed value vocabulary supports 1..16 values")
        self.embedding = nn.Embedding(value_count, hidden_size)

    def forward(self, value_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(value_ids)


class T3GatedEdit(nn.Module):
    """Shared gated low-rank edit O(slot, [address; value/replacement])."""

    def __init__(self, hidden_size: int, rank: int = 16):
        super().__init__()
        joint_size = 3 * hidden_size
        self.down = nn.Linear(joint_size, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=False)
        self.gate = nn.Linear(joint_size, hidden_size)
        nn.init.zeros_(self.up.weight)
        nn.init.constant_(self.gate.bias, -2.0)

    def forward(self, slots: torch.Tensor, instruction: torch.Tensor) -> torch.Tensor:
        if slots.ndim != 3 or instruction.ndim != 2:
            raise ValueError("expected slots [batch, slots, hidden] and instruction [batch, 2*hidden]")
        if instruction.shape != (slots.shape[0], 2 * slots.shape[2]):
            raise ValueError("instruction must concatenate two hidden-size vectors")
        expanded = instruction[:, None, :].expand(-1, slots.shape[1], -1)
        joint = torch.cat([slots, expanded], dim=-1)
        return slots + torch.sigmoid(self.gate(joint)) * self.up(torch.tanh(self.down(joint)))


class T3AConditioning(nn.Module):
    """T3-a ablation: condition the existing edit and apply it to every slot."""

    def __init__(self, hidden_size: int, rank: int = 16):
        super().__init__()
        self.editor = T3GatedEdit(hidden_size, rank)

    def forward(self, slots: torch.Tensor, instruction: torch.Tensor) -> torch.Tensor:
        return self.editor(slots, instruction)


@dataclass(frozen=True)
class PointerOutput:
    edited_slots: torch.Tensor
    probabilities: torch.Tensor


class T3BPointer(nn.Module):
    """T3-b: one-head span-to-slot selection followed by a weighted edit."""

    def __init__(self, hidden_size: int, rank: int = 16, *, hard_selection: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.hard_selection = hard_selection
        self.query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.editor = T3GatedEdit(hidden_size, rank)

    def pointer_probabilities(
        self,
        address: torch.Tensor,
        slots: torch.Tensor,
        slot_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if address.shape != (slots.shape[0], slots.shape[2]):
            raise ValueError("address shape must be [batch, hidden]")
        scores = torch.einsum("bd,bsd->bs", self.query(address), self.key(slots))
        scores = scores / math.sqrt(self.hidden_size)
        if slot_mask is not None:
            if slot_mask.shape != scores.shape or not torch.all(slot_mask.bool().any(1)):
                raise ValueError("slot_mask must leave at least one slot per example")
            scores = scores.masked_fill(~slot_mask.bool(), torch.finfo(scores.dtype).min)
        return torch.softmax(scores, dim=-1)

    def forward(
        self,
        slots: torch.Tensor,
        address: torch.Tensor,
        instruction: torch.Tensor,
        slot_mask: torch.Tensor | None = None,
        support_mask: torch.Tensor | None = None,
        pointer_slots: torch.Tensor | None = None,
        candidate_mask: torch.Tensor | None = None,
        edit_enabled: torch.Tensor | None = None,
    ) -> PointerOutput:
        if pointer_slots is None:
            pointer_slots = slots
        elif pointer_slots.shape != slots.shape:
            raise ValueError("pointer_slots must match edited slot shape")
        effective_mask = slot_mask
        if candidate_mask is not None:
            if candidate_mask.shape != slots.shape[:2]:
                raise ValueError("candidate_mask must match the slot dimensions")
            rows_with_candidates = candidate_mask.bool().any(1)
            effective_mask = torch.where(
                rows_with_candidates[:, None], candidate_mask.bool(),
                torch.ones_like(candidate_mask, dtype=torch.bool)
                if slot_mask is None else slot_mask.bool(),
            )
        probabilities = self.pointer_probabilities(address, pointer_slots, effective_mask)
        candidate = self.editor(slots, instruction)
        weights = probabilities
        if self.hard_selection:
            hard = torch.zeros_like(probabilities).scatter_(
                1, probabilities.argmax(-1, keepdim=True), 1.0
            )
            # Discrete forward selection with soft-pointer gradients.
            weights = hard + probabilities - probabilities.detach()
        if support_mask is not None:
            if support_mask.shape != (
                slots.shape[0], slots.shape[1], slots.shape[1]
            ):
                raise ValueError("support_mask must have shape [batch, slots, slots]")
            weights = torch.einsum(
                "bs,bst->bt", weights, support_mask.to(weights.dtype)
            ).clamp(max=1.0)
        if edit_enabled is not None:
            if edit_enabled.shape != (slots.shape[0],):
                raise ValueError("edit_enabled must have shape [batch]")
            weights = weights * edit_enabled.to(weights.dtype)[:, None]
        edited = slots + weights.unsqueeze(-1) * (candidate - slots)
        return PointerOutput(edited_slots=edited, probabilities=probabilities)


def pointer_metrics(
    probabilities: torch.Tensor, correct_slot: torch.Tensor
) -> dict[str, float]:
    """Report correct-slot probability mass and the secondary top-1 diagnostic."""

    if probabilities.ndim != 2 or correct_slot.shape != (probabilities.shape[0],):
        raise ValueError("invalid pointer metric shapes")
    if not torch.all((correct_slot >= 0) & (correct_slot < probabilities.shape[1])):
        raise ValueError("correct slot index out of range")
    rows = torch.arange(len(probabilities), device=probabilities.device)
    mass = probabilities[rows, correct_slot]
    return {
        "pointer_accuracy": float(mass.mean().item()),
        "pointer_top1_accuracy": float(
            (probabilities.argmax(-1) == correct_slot).to(torch.float32).mean().item()
        ),
    }


def pointer_target_cross_entropy(correct_slot_probability: torch.Tensor) -> torch.Tensor:
    """Supervise T3-b with the authoritative correct-slot probability mass."""

    if correct_slot_probability.ndim != 1 or not torch.is_floating_point(
        correct_slot_probability
    ):
        raise ValueError("correct-slot probability must be a floating [batch] tensor")
    if not torch.isfinite(correct_slot_probability).all() or not torch.all(
        (correct_slot_probability >= 0) & (correct_slot_probability <= 1)
    ):
        raise ValueError("correct-slot probabilities must be finite and in [0, 1]")
    probability = correct_slot_probability.float().clamp_min(
        torch.finfo(torch.float32).tiny
    )
    return -probability.log().mean()


def override_span_mask(
    span_mask: torch.Tensor,
    valid_tokens: torch.Tensor,
    mode: SpanOverride,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Create A3 correct, adjacent, or random same-length contiguous spans."""

    span = span_mask.bool()
    valid = valid_tokens.bool()
    if span.shape != valid.shape or span.ndim != 2:
        raise ValueError("span and valid masks must share shape [batch, sequence]")
    if (span & ~valid).any():
        raise ValueError("span must be contained in valid tokens")
    if mode == "correct":
        return span.clone()
    if mode not in {"adjacent", "random"}:
        raise ValueError(f"unknown span override: {mode}")

    result = torch.zeros_like(span)
    for row in range(span.shape[0]):
        selected = span[row].nonzero(as_tuple=False).flatten()
        if selected.numel() == 0:
            raise ValueError("every span must be non-empty")
        start, length = int(selected[0]), int(selected.numel())
        if not torch.equal(selected, torch.arange(start, start + length, device=selected.device)):
            raise ValueError("span masks must be contiguous")
        candidates = []
        for candidate in range(0, span.shape[1] - length + 1):
            indices = torch.arange(candidate, candidate + length, device=span.device)
            if valid[row, indices].all() and not span[row, indices].any():
                candidates.append(candidate)
        if not candidates:
            raise ValueError("no non-overlapping same-length override span exists")
        if mode == "adjacent":
            new_start = min(candidates, key=lambda item: (abs(item - start), item))
        else:
            choice = int(torch.randint(len(candidates), (), generator=generator).item())
            new_start = candidates[choice]
        result[row, new_start : new_start + length] = True
    return result
