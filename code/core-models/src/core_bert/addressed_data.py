"""Collation for T2/T3 passage-addressed benchmark records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from core_bert.data import LABEL_TO_ID


# Fifteen preregistered canonical values. Event replacements use span content
# and receive the sentinel value ID -1 rather than occupying this table.
CLOSED_VALUE_TOKENS = (
    "on",
    "off",
    "high",
    "low",
    "more",
    "less",
    "no_effect",
    "present",
    "absent",
    "increased",
    "decreased",
    "yes",
    "no",
    "1",
    "0",
)


@dataclass(frozen=True)
class _EncodedRecord:
    input_ids: list[int]
    offsets: list[tuple[int, int]]
    valid: list[bool]
    command: list[bool]
    do: list[bool]
    target: list[bool]
    marker: list[bool]
    replacement: list[bool]
    token_types: list[int]
    value_id: int
    label: Any
    label_id: int
    record_id: str
    source: str


def _required_text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path} must be a non-empty string")
    return value


def _checked_span(value: Any, passage: str, expected: Any, path: str) -> tuple[int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(isinstance(item, int) and not isinstance(item, bool) for item in value)
    ):
        raise ValueError(f"{path} must be a [start, end] integer pair")
    start, end = value
    if start < 0 or end <= start or end > len(passage):
        raise ValueError(f"{path} is outside the factual passage")
    if not isinstance(expected, str) or passage[start:end] != expected:
        raise ValueError(f"{path} does not exactly slice its expected text")
    return start, end


def _span_mask(offsets: Sequence[tuple[int, int]], span: tuple[int, int]) -> list[bool]:
    start, end = span
    mask = [token_end > start and token_start < end for token_start, token_end in offsets]
    if not any(mask):
        raise ValueError(f"character span {span} maps to no passage token")
    return mask


def _literal_mask(
    offsets: Sequence[tuple[int, int]], text: str, literal: str
) -> list[bool]:
    spans = []
    start = 0
    while (found := text.find(literal, start)) >= 0:
        spans.append((found, found + len(literal)))
        start = found + len(literal)
    return [
        any(token_end > left and token_start < right for left, right in spans)
        for token_start, token_end in offsets
    ]


class AddressedBatchCollator:
    """Build ``[CLS] passage [DO] command`` batches with gold span masks.

    Passage offsets remain relative to ``factual.passage``. Special tokens,
    command tokens, and padding carry ``[0, 0]`` offsets. Right truncation may
    shorten the passage or command, but it is rejected if it removes any token
    overlapping a gold target/replacement span.
    """

    def __init__(
        self,
        tokenizer: Any,
        *,
        max_length: int = 512,
        do_token: str = "[DO]",
        value_tokens: Sequence[str] = CLOSED_VALUE_TOKENS,
    ) -> None:
        if max_length < 4:
            raise ValueError("max_length must leave room for [CLS], passage, [DO], and command")
        if not 1 <= len(value_tokens) <= 16 or len(set(value_tokens)) != len(value_tokens):
            raise ValueError("value vocabulary must contain 1..16 unique tokens")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.value_to_id = {token: index for index, token in enumerate(value_tokens)}
        self.cls_id = tokenizer.cls_token_id
        self.pad_id = tokenizer.pad_token_id
        self.do_id = tokenizer.convert_tokens_to_ids(do_token)
        if self.cls_id is None or self.pad_id is None:
            raise ValueError("tokenizer must define CLS and padding token IDs")
        if self.do_id is None or self.do_id == getattr(tokenizer, "unk_token_id", None):
            raise ValueError(f"tokenizer must register {do_token!r} as a special token")

    def _tokenize(self, text: str, *, offsets: bool) -> tuple[list[int], list[tuple[int, int]]]:
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=offsets,
            truncation=False,
        )
        ids = encoded["input_ids"]
        if ids and isinstance(ids[0], list):
            raise ValueError("collator expects tokenizer output for one unbatched string")
        token_ids = [int(item) for item in ids]
        if offsets:
            raw_offsets = encoded.get("offset_mapping")
            if raw_offsets is None:
                raise ValueError("a fast tokenizer with offset_mapping support is required")
            mapped = [(int(start), int(end)) for start, end in raw_offsets]
            if len(mapped) != len(token_ids):
                raise ValueError("token IDs and offset mapping have different lengths")
        else:
            mapped = [(0, 0)] * len(token_ids)
        return token_ids, mapped

    def _encode_record(self, record: Mapping[str, Any]) -> _EncodedRecord:
        record_id = _required_text(record.get("id"), "record.id")
        source = _required_text(record.get("source"), "record.source")
        factual = record.get("factual")
        intervention = record.get("intervention")
        intervened = record.get("intervened")
        if not all(isinstance(item, Mapping) for item in (factual, intervention, intervened)):
            raise ValueError(f"{record_id}: factual, intervention, and intervened must be objects")
        passage = _required_text(factual.get("passage"), f"{record_id}.factual.passage")
        command_text = _required_text(intervention.get("text"), f"{record_id}.intervention.text")
        target_text = _required_text(
            intervention.get("target_text"), f"{record_id}.intervention.target_text"
        )
        target_span = _checked_span(
            intervention.get("target_span"),
            passage,
            target_text,
            f"{record_id}.intervention.target_span",
        )
        kind = intervention.get("kind")
        replacement_span = None
        if kind == "event_replace":
            replacement_span = _checked_span(
                intervention.get("replacement_span"),
                passage,
                intervention.get("value"),
                f"{record_id}.intervention.replacement_span",
            )
            value_id = -1
        elif kind == "value_set":
            token = intervention.get("value_token")
            if token not in self.value_to_id:
                raise ValueError(f"{record_id}: unknown canonical value token {token!r}")
            if intervention.get("replacement_span") is not None:
                raise ValueError(f"{record_id}: value_set replacement_span must be null")
            value_id = self.value_to_id[token]
        else:
            raise ValueError(f"{record_id}: unsupported intervention kind {kind!r}")

        passage_ids, passage_offsets = self._tokenize(passage, offsets=True)
        command_ids, _ = self._tokenize(command_text, offsets=False)
        if not passage_ids or not command_ids:
            raise ValueError(f"{record_id}: passage and command must each tokenize to at least one token")
        target_full = _span_mask(passage_offsets, target_span)
        marker_full = _literal_mask(passage_offsets, passage, "[TARGET]")
        replacement_full = (
            _span_mask(passage_offsets, replacement_span)
            if replacement_span is not None
            else [False] * len(passage_ids)
        )

        # Reserve one command token. Passage tokens are retained from the left;
        # the full set of addressed tokens must survive that truncation.
        passage_keep = min(len(passage_ids), self.max_length - 3)
        removed_address = any(target_full[passage_keep:]) or any(
            replacement_full[passage_keep:]
        )
        if removed_address:
            raise ValueError(f"{record_id}: truncation removes an addressed span")
        command_keep = min(len(command_ids), self.max_length - passage_keep - 2)
        if command_keep < 1:
            raise ValueError(f"{record_id}: truncation removes the command")

        passage_ids = passage_ids[:passage_keep]
        passage_offsets = passage_offsets[:passage_keep]
        target_full = target_full[:passage_keep]
        marker_full = marker_full[:passage_keep]
        replacement_full = replacement_full[:passage_keep]
        command_ids = command_ids[:command_keep]
        input_ids = [self.cls_id, *passage_ids, self.do_id, *command_ids]
        do_index = 1 + passage_keep
        length = len(input_ids)
        label = intervened.get("answer")
        normalized_label = str(label).strip().lower() if label is not None else ""
        return _EncodedRecord(
            input_ids=input_ids,
            offsets=[(0, 0), *passage_offsets, (0, 0), *([(0, 0)] * command_keep)],
            valid=[True] * length,
            command=[False] * (do_index + 1) + [True] * command_keep,
            do=[False] * do_index + [True] + [False] * command_keep,
            target=[False, *target_full, False, *([False] * command_keep)],
            marker=[False, *marker_full, False, *([False] * command_keep)],
            replacement=[False, *replacement_full, False, *([False] * command_keep)],
            token_types=[0] * (1 + passage_keep) + [1] * (1 + command_keep),
            value_id=value_id,
            label=label,
            label_id=LABEL_TO_ID.get(normalized_label, -1),
            record_id=record_id,
            source=source,
        )

    def __call__(self, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not records:
            raise ValueError("cannot collate an empty batch")
        rows = [self._encode_record(record) for record in records]
        width = max(len(row.input_ids) for row in rows)

        def padded(name: str, fill: Any, dtype: torch.dtype) -> torch.Tensor:
            values = [getattr(row, name) + [fill] * (width - len(row.input_ids)) for row in rows]
            return torch.tensor(values, dtype=dtype)

        offsets = [row.offsets + [(0, 0)] * (width - len(row.input_ids)) for row in rows]
        valid = padded("valid", False, torch.bool)
        return {
            "input_ids": padded("input_ids", self.pad_id, torch.long),
            "attention_mask": valid.to(torch.long),
            "token_type_ids": padded("token_types", 0, torch.long),
            "offset_mapping": torch.tensor(offsets, dtype=torch.long),
            "valid_tokens": valid,
            "command_tokens": padded("command", False, torch.bool),
            "do_tokens": padded("do", False, torch.bool),
            "target_tokens": padded("target", False, torch.bool),
            "address_marker_tokens": padded("marker", False, torch.bool),
            "replacement_tokens": padded("replacement", False, torch.bool),
            "value_ids": torch.tensor([row.value_id for row in rows], dtype=torch.long),
            "label_ids": torch.tensor([row.label_id for row in rows], dtype=torch.long),
            "labels": [row.label for row in rows],
            "record_ids": [row.record_id for row in rows],
            "sources": [row.source for row in rows],
        }
