"""Open-vocabulary generation and metrics for Com2 chain-state outputs."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .variable_outputs import build_variable_batch


IGNORE_INDEX = -100


@dataclass(frozen=True)
class OpenTextBatch:
    after_inputs: torch.Tensor
    after_targets: torch.Tensor
    before_inputs: torch.Tensor
    before_targets: torch.Tensor
    label_mask: torch.Tensor
    before_label_mask: torch.Tensor
    variable_mask: torch.Tensor
    target_mask: torch.Tensor
    changed_mask: torch.Tensor
    preservation_mask: torch.Tensor
    query_slot: torch.Tensor
    after_text: tuple[tuple[str, ...], ...]
    before_text: tuple[tuple[str, ...], ...]


def normalize_open_text(value: str) -> str:
    """Canonicalize case/spacing only; never map text into a closed class."""

    if not isinstance(value, str):
        raise ValueError("open-text output must be a string")
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip().casefold()


def _encode_text(tokenizer: Any, text: str, max_tokens: int) -> tuple[list[int], list[int]]:
    ids = list(tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"])
    if not ids or len(ids) + 1 > max_tokens:
        raise ValueError(
            f"open-text label must contain 1..{max_tokens - 1} tokens; found {len(ids)}"
        )
    return [int(tokenizer.cls_token_id), *map(int, ids)], [*map(int, ids), int(tokenizer.sep_token_id)]


def build_open_text_batch(
    records: Sequence[Mapping[str, Any]], tokenizer: Any, *, max_slots: int = 30,
    max_tokens: int = 32,
) -> OpenTextBatch:
    """Tokenize explicit before/after probe strings for one shared slot head."""

    if tokenizer.cls_token_id is None or tokenizer.sep_token_id is None or tokenizer.pad_token_id is None:
        raise ValueError("open-text generation requires CLS, SEP, and PAD token IDs")
    variables = build_variable_batch(records, max_slots=max_slots)
    shape = (len(records), max_slots, max_tokens)
    after_inputs = torch.full(shape, tokenizer.pad_token_id, dtype=torch.long)
    before_inputs = torch.full_like(after_inputs, tokenizer.pad_token_id)
    after_targets = torch.full(shape, IGNORE_INDEX, dtype=torch.long)
    before_targets = torch.full_like(after_targets, IGNORE_INDEX)
    after_text: list[tuple[str, ...]] = []
    before_text: list[tuple[str, ...]] = []
    query_slot = torch.full((len(records),), -1, dtype=torch.long)
    for row, record in enumerate(records):
        if record.get("source") != "com2" or record.get("structure_kind") != "chain":
            raise ValueError("open-text batches are defined only for Com2 chain records")
        active = int(variables.variable_mask[row].sum().item())
        if active < 1:
            raise ValueError("Com2 chain must contain at least one event")
        query_slot[row] = active - 1
        row_after: list[str] = []
        row_before: list[str] = []
        for slot in range(active):
            if not variables.label_mask[row, slot] or not variables.before_label_mask[row, slot]:
                raise ValueError("every Com2 event requires explicit before and after text")
            after = variables.labels[row][slot]
            before = variables.before_labels[row][slot]
            if not isinstance(after, str) or not isinstance(before, str):
                raise ValueError("Com2 before/after labels must be strings")
            a_in, a_out = _encode_text(tokenizer, after, max_tokens)
            b_in, b_out = _encode_text(tokenizer, before, max_tokens)
            after_inputs[row, slot, : len(a_in)] = torch.tensor(a_in)
            after_targets[row, slot, : len(a_out)] = torch.tensor(a_out)
            before_inputs[row, slot, : len(b_in)] = torch.tensor(b_in)
            before_targets[row, slot, : len(b_out)] = torch.tensor(b_out)
            row_after.append(after)
            row_before.append(before)
        after_text.append(tuple(row_after))
        before_text.append(tuple(row_before))
    return OpenTextBatch(
        after_inputs, after_targets, before_inputs, before_targets,
        variables.label_mask, variables.before_label_mask, variables.variable_mask,
        variables.target_mask, variables.changed_mask, variables.preservation_mask,
        query_slot, tuple(after_text), tuple(before_text),
    )


class OpenTextSlotHead(nn.Module):
    """Autoregressively generate arbitrary event text from each dynamic slot."""

    def __init__(
        self, hidden_size: int, vocab_size: int, *, padding_idx: int,
        bos_token_id: int, eos_token_id: int,
    ) -> None:
        super().__init__()
        if hidden_size < 1 or vocab_size < 2:
            raise ValueError("hidden size and vocabulary size must be positive")
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=padding_idx)
        self.initial = nn.Linear(hidden_size, hidden_size)
        self.decoder = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))
        self.padding_idx = int(padding_idx)
        self.bos_token_id = int(bos_token_id)
        self.eos_token_id = int(eos_token_id)

    def forward(
        self, slots: torch.Tensor, decoder_inputs: torch.Tensor,
        label_mask: torch.Tensor,
    ) -> torch.Tensor:
        if slots.ndim != 3 or decoder_inputs.ndim != 3:
            raise ValueError("slots and decoder inputs must be [batch, slots, ...]")
        if slots.shape[:2] != decoder_inputs.shape[:2] or label_mask.shape != slots.shape[:2]:
            raise ValueError("open-text slot, token, and label masks must align")
        active_slots = slots[label_mask.bool()]
        active_inputs = decoder_inputs[label_mask.bool()]
        if not len(active_slots):
            raise ValueError("open-text head requires at least one labelled slot")
        embedded = self.embedding(active_inputs)
        initial = torch.tanh(self.initial(active_slots)).unsqueeze(0)
        hidden, _ = self.decoder(embedded, initial)
        return F.linear(hidden, self.embedding.weight, self.output_bias)

    @torch.no_grad()
    def generate(
        self, slots: torch.Tensor, variable_mask: torch.Tensor, *, max_tokens: int = 32,
    ) -> torch.Tensor:
        if variable_mask.shape != slots.shape[:2] or max_tokens < 2:
            raise ValueError("invalid open-text generation shape or length")
        active = slots[variable_mask.bool()]
        state = torch.tanh(self.initial(active)).unsqueeze(0)
        token = torch.full(
            (len(active), 1), self.bos_token_id, dtype=torch.long, device=slots.device
        )
        finished = torch.zeros(len(active), dtype=torch.bool, device=slots.device)
        generated = []
        for _ in range(max_tokens):
            hidden, state = self.decoder(self.embedding(token), state)
            next_token = F.linear(hidden[:, -1], self.embedding.weight, self.output_bias).argmax(-1)
            # Each slot is an independent sequence. Once a slot emits EOS it
            # must remain padded while longer peers finish; continuing to
            # decode it appends non-special garbage that destroys exact match.
            emitted = torch.where(
                finished,
                torch.full_like(next_token, self.padding_idx),
                next_token,
            )
            generated.append(emitted)
            finished |= emitted.eq(self.eos_token_id)
            token = torch.where(
                finished,
                torch.full_like(next_token, self.eos_token_id),
                next_token,
            ).unsqueeze(1)
            if bool(finished.all()):
                break
        result = torch.full(
            (len(active), max_tokens), self.padding_idx, dtype=torch.long, device=slots.device
        )
        if generated:
            values = torch.stack(generated, dim=1)
            result[:, : values.shape[1]] = values
        return result


def sequence_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Return one mean token loss per labelled slot in row-major mask order."""

    if logits.ndim != 3 or targets.shape != logits.shape[:2]:
        raise ValueError("open-text logits/targets must be [labelled slots, tokens, vocab]")
    losses = F.cross_entropy(
        logits.transpose(1, 2), targets, ignore_index=IGNORE_INDEX, reduction="none"
    )
    observed = targets.ne(IGNORE_INDEX)
    if not bool(observed.any(1).all()):
        raise ValueError("every open-text sequence must contain a target token")
    return (losses * observed).sum(1) / observed.sum(1)


def loss_matrix(
    sequence_losses: torch.Tensor, label_mask: torch.Tensor,
) -> torch.Tensor:
    result = sequence_losses.new_zeros(label_mask.shape)
    if len(sequence_losses) != int(label_mask.sum().item()):
        raise ValueError("sequence losses do not match labelled slot count")
    result[label_mask.bool()] = sequence_losses
    return result


def balanced_open_text_transition_loss(
    after_losses: torch.Tensor, before_losses: torch.Tensor,
    batch: OpenTextBatch,
) -> torch.Tensor:
    """Balance target/change/preservation after-state and factual reconstruction."""

    after = loss_matrix(after_losses, batch.label_mask)
    before = loss_matrix(before_losses, batch.before_label_mask)
    groups = []
    for mask in (batch.target_mask, batch.changed_mask, batch.preservation_mask):
        selected = after[mask.bool() & batch.label_mask.bool()]
        if len(selected):
            groups.append(selected.mean())
    factual = before[batch.before_label_mask.bool()]
    if len(factual):
        groups.append(factual.mean())
    if len(groups) < 2:
        raise ValueError("balanced open-text transition loss requires multiple groups")
    return torch.stack(groups).mean()


def open_text_exact_metrics(gold: Sequence[str], predicted: Sequence[str]) -> dict[str, float | int]:
    if len(gold) != len(predicted) or not gold:
        raise ValueError("open-text metrics require aligned non-empty outputs")
    exact = [normalize_open_text(a) == normalize_open_text(b) for a, b in zip(gold, predicted)]
    return {"count": len(exact), "exact_match": sum(exact) / len(exact)}


__all__ = [
    "OpenTextBatch", "OpenTextSlotHead", "balanced_open_text_transition_loss",
    "build_open_text_batch", "loss_matrix", "normalize_open_text",
    "open_text_exact_metrics", "sequence_cross_entropy",
]
