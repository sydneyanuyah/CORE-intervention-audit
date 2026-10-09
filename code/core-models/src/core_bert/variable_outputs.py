"""Per-variable supervision and metrics for dynamic scientific slots.

This is infrastructure for future per-variable experiments, not an A1 result.
Gold values come only from explicit probe ``answer_after`` fields. Missing probe
labels remain masked; they are never inferred from the record-level answer or
from another world state.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .scientific_reader import variable_slot_layout


IGNORE_INDEX = -100


@dataclass(frozen=True)
class VariableBatch:
    """Deterministically aligned variable identities and explicit gold values."""

    variable_ids: tuple[tuple[str, ...], ...]
    variable_mask: torch.Tensor
    labels: tuple[tuple[Any, ...], ...]
    label_mask: torch.Tensor
    before_labels: tuple[tuple[Any, ...], ...]
    before_label_mask: torch.Tensor
    query_slot: torch.Tensor
    target_mask: torch.Tensor
    changed_mask: torch.Tensor
    preservation_mask: torch.Tensor


@dataclass(frozen=True)
class VariableOutput:
    """Class logits plus the structural-slot mask that gives them meaning."""

    logits: torch.Tensor
    variable_mask: torch.Tensor


def _canonical_value(value: Any) -> str:
    try:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"variable label is not a finite JSON value: {value!r}") from exc


def build_variable_batch(
    records: Sequence[Mapping[str, Any]], *, max_slots: int = 30
) -> VariableBatch:
    """Align explicit probe labels to the reader's deterministic dynamic slots.

    ``variable_mask`` marks structural variables. ``label_mask`` is narrower:
    it marks only slots with an explicit ``answer_after`` in ``record.probes``.
    A JSON null is an available label and is therefore distinguished from a
    missing label by ``label_mask``.
    """

    if not records:
        raise ValueError("cannot build variable outputs for an empty batch")
    layouts = [variable_slot_layout(record, max_slots=max_slots) for record in records]
    ids: list[tuple[str, ...]] = []
    labels: list[tuple[Any, ...]] = []
    before_labels: list[tuple[Any, ...]] = []
    variable_mask = torch.zeros(len(records), max_slots, dtype=torch.bool)
    label_mask = torch.zeros_like(variable_mask)
    before_label_mask = torch.zeros_like(variable_mask)
    target_mask = torch.zeros_like(variable_mask)
    changed_mask = torch.zeros_like(variable_mask)
    preservation_mask = torch.zeros_like(variable_mask)
    query_slot = torch.full((len(records),), -1, dtype=torch.long)

    for row_index, (record, layout) in enumerate(zip(records, layouts)):
        probes = record.get("probes")
        if not isinstance(probes, list):
            raise ValueError("record.probes must be a list for per-variable supervision")
        by_variable: dict[str, Mapping[str, Any]] = {}
        for probe_index, probe in enumerate(probes):
            if not isinstance(probe, Mapping):
                raise ValueError(f"record.probes[{probe_index}] must be an object")
            variable = probe.get("variable")
            if not isinstance(variable, str) or not variable:
                raise ValueError(
                    f"record.probes[{probe_index}].variable must be a non-empty string"
                )
            if variable in by_variable:
                raise ValueError(f"duplicate probe label for structural variable {variable!r}")
            by_variable[variable] = probe
        unknown = sorted(set(by_variable) - set(layout.names))
        if unknown:
            raise ValueError(f"probe variables do not map to dynamic slots: {unknown}")
        queried = [
            variable for variable, probe in by_variable.items()
            if isinstance(probe.get("question"), str) and probe["question"].strip()
        ]
        if len(queried) > 1:
            raise ValueError("record must identify at most one question-bearing probe")
        if queried:
            query_slot[row_index] = layout.names.index(queried[0])

        padded_ids = (*layout.names, *("" for _ in range(max_slots - len(layout.names))))
        row_labels: list[Any] = [None] * max_slots
        row_before_labels: list[Any] = [None] * max_slots
        variable_mask[row_index, : len(layout.names)] = True
        for slot, variable in enumerate(layout.names):
            probe = by_variable.get(variable)
            if probe is not None and "answer_after" in probe:
                # Validate serializability now; null remains a valid explicit label.
                _canonical_value(probe["answer_after"])
                row_labels[slot] = probe["answer_after"]
                label_mask[row_index, slot] = True
                if slot == layout.target_slot:
                    target_mask[row_index, slot] = True
                elif probe.get("actually_changed") is True:
                    changed_mask[row_index, slot] = True
                elif probe.get("actually_changed") is False:
                    preservation_mask[row_index, slot] = True
            if probe is not None and "answer_before" in probe:
                _canonical_value(probe["answer_before"])
                row_before_labels[slot] = probe["answer_before"]
                before_label_mask[row_index, slot] = True
        ids.append(tuple(padded_ids))
        labels.append(tuple(row_labels))
        before_labels.append(tuple(row_before_labels))

    return VariableBatch(
        tuple(ids), variable_mask, tuple(labels), label_mask,
        tuple(before_labels), before_label_mask, query_slot,
        target_mask, changed_mask, preservation_mask,
    )


def encode_variable_labels(
    batch: VariableBatch,
    vocabulary: Sequence[Any],
    *,
    ignore_index: int = IGNORE_INDEX,
    state: str = "after",
) -> torch.Tensor:
    """Encode explicit labels using a caller-declared, exact JSON vocabulary.

    An observed value outside the declared vocabulary is an error. This keeps
    open-text Com2 targets from being silently coerced into closed classes.
    """

    if not vocabulary:
        raise ValueError("variable label vocabulary must not be empty")
    encoded_vocabulary: dict[str, int] = {}
    for index, value in enumerate(vocabulary):
        key = _canonical_value(value)
        if key in encoded_vocabulary:
            raise ValueError("variable label vocabulary contains duplicate JSON values")
        encoded_vocabulary[key] = index
    if state == "after":
        labels, label_mask = batch.labels, batch.label_mask
    elif state == "before":
        labels, label_mask = batch.before_labels, batch.before_label_mask
    else:
        raise ValueError("variable label state must be 'before' or 'after'")
    label_ids = torch.full(label_mask.shape, ignore_index, dtype=torch.long)
    for row in range(label_mask.shape[0]):
        for slot in range(label_mask.shape[1]):
            if not bool(label_mask[row, slot]):
                continue
            key = _canonical_value(labels[row][slot])
            if key not in encoded_vocabulary:
                raise ValueError(
                    f"observed variable label {labels[row][slot]!r} is outside "
                    "the declared vocabulary"
                )
            label_ids[row, slot] = encoded_vocabulary[key]
    return label_ids


class PerVariableOutputHead(nn.Module):
    """Apply one shared classifier to each dynamic structural slot."""

    def __init__(self, hidden_size: int, label_count: int):
        super().__init__()
        if hidden_size < 1 or label_count < 2:
            raise ValueError("hidden_size must be positive and label_count must be at least two")
        self.classifier = nn.Linear(hidden_size, label_count)

    def forward(self, edited_slots: torch.Tensor, variable_mask: torch.Tensor) -> VariableOutput:
        if edited_slots.ndim != 3:
            raise ValueError("edited_slots must have shape [batch, variables, hidden]")
        if variable_mask.shape != edited_slots.shape[:2]:
            raise ValueError("variable_mask must match the first two edited-slot dimensions")
        if not torch.all(variable_mask.bool().any(1)):
            raise ValueError("each record must contain at least one active variable")
        return VariableOutput(self.classifier(edited_slots), variable_mask.bool())


def variable_cross_entropy(
    logits: torch.Tensor,
    label_ids: torch.Tensor,
    label_mask: torch.Tensor,
    variable_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean cross-entropy over explicitly labelled active variables only."""

    _validate_metric_inputs(logits, label_ids, label_mask, variable_mask)
    selected = label_mask.bool()
    return F.cross_entropy(logits[selected], label_ids[selected])


def balanced_variable_cross_entropy(
    logits: torch.Tensor,
    label_ids: torch.Tensor,
    label_mask: torch.Tensor,
    variable_mask: torch.Tensor,
    target_mask: torch.Tensor,
    changed_mask: torch.Tensor,
    preservation_mask: torch.Tensor,
) -> torch.Tensor:
    """Average target, changed-nontarget, and preservation group losses."""

    _validate_metric_inputs(logits, label_ids, label_mask, variable_mask)
    masks = (target_mask.bool(), changed_mask.bool(), preservation_mask.bool())
    for mask in masks:
        if mask.shape != label_mask.shape or torch.any(mask & ~label_mask.bool()):
            raise ValueError("balanced supervision masks must select explicit labels only")
    if torch.any(masks[0] & masks[1]) or torch.any(masks[0] & masks[2]) or torch.any(
        masks[1] & masks[2]
    ):
        raise ValueError("balanced supervision masks must be disjoint")
    losses = [F.cross_entropy(logits[mask], label_ids[mask]) for mask in masks if mask.any()]
    if not losses:
        raise ValueError("balanced supervision requires a target, change, or preservation label")
    return torch.stack(losses).mean()


def balanced_transition_cross_entropy(
    edited_logits: torch.Tensor,
    pre_edit_logits: torch.Tensor,
    after_ids: torch.Tensor,
    before_ids: torch.Tensor,
    after_mask: torch.Tensor,
    before_mask: torch.Tensor,
    variable_mask: torch.Tensor,
    target_mask: torch.Tensor,
    changed_mask: torch.Tensor,
    preservation_mask: torch.Tensor,
) -> torch.Tensor:
    """Supervise each group on both its explicit before and after state."""

    _validate_metric_inputs(edited_logits, after_ids, after_mask, variable_mask)
    _validate_metric_inputs(pre_edit_logits, before_ids, before_mask, variable_mask)
    paired = after_mask.bool() & before_mask.bool()
    groups = (target_mask.bool(), changed_mask.bool(), preservation_mask.bool())
    selected_groups = []
    for group in groups:
        if group.shape != paired.shape or torch.any(group & ~after_mask.bool()):
            raise ValueError("transition groups must select explicit after labels only")
        selected_groups.append(group & paired)
    if any(torch.any(selected_groups[i] & selected_groups[j]) for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("transition supervision groups must be disjoint")
    losses = []
    for selected in selected_groups:
        if selected.any():
            losses.append(0.5 * (
                F.cross_entropy(pre_edit_logits[selected], before_ids[selected])
                + F.cross_entropy(edited_logits[selected], after_ids[selected])
            ))
    if not losses:
        raise ValueError("transition supervision requires paired before/after labels")
    return torch.stack(losses).mean()


def task_logits_from_queried_variable(
    fallback_logits: torch.Tensor,
    variable_logits: torch.Tensor,
    query_slot: torch.Tensor,
) -> torch.Tensor:
    """Use an explicit question-bearing probe slot where one is available."""

    if fallback_logits.ndim != 2 or variable_logits.ndim != 3:
        raise ValueError("task and variable logits must be rank 2 and 3")
    if fallback_logits.shape[0] != variable_logits.shape[0] or query_slot.shape != (
        fallback_logits.shape[0],
    ):
        raise ValueError("queried-slot readout batch dimensions must agree")
    task_labels = fallback_logits.shape[1]
    variable_labels = variable_logits.shape[2]
    if variable_labels < task_labels:
        raise ValueError("variable vocabulary must cover all task labels")
    covered = query_slot >= 0
    if torch.any(query_slot >= variable_logits.shape[1]):
        raise ValueError("query_slot is outside the variable dimension")
    safe = query_slot.clamp_min(0)
    rows = torch.arange(len(query_slot), device=query_slot.device)
    queried_logits = variable_logits[rows, safe]
    fallback_padding = fallback_logits.new_full(
        (len(fallback_logits), variable_labels - task_labels),
        torch.finfo(fallback_logits.dtype).min,
    )
    padded_fallback = torch.cat([fallback_logits, fallback_padding], dim=1)
    return torch.where(covered[:, None], queried_logits, padded_fallback)


def per_variable_metrics(
    logits: torch.Tensor,
    label_ids: torch.Tensor,
    label_mask: torch.Tensor,
    variable_mask: torch.Tensor,
) -> dict[str, Any]:
    """Compute labelled-slot accuracy and fully-observed record exact match.

    Record exact match excludes partially labelled records and reports its
    eligibility coverage. If no record is fully labelled, its metric is null
    rather than a fabricated score.
    """

    _validate_metric_inputs(logits, label_ids, label_mask, variable_mask)
    active = variable_mask.bool()
    observed = label_mask.bool()
    predicted = logits.argmax(dim=-1)
    correct = predicted.eq(label_ids) & observed
    labelled_count = int(observed.sum().item())
    active_count = int(active.sum().item())
    fully_observed = torch.all(observed | ~active, dim=1)
    eligible_count = int(fully_observed.sum().item())
    if eligible_count:
        record_correct = torch.all(correct | ~active, dim=1)
        exact_match = float(record_correct[fully_observed].float().mean().item())
    else:
        exact_match = None
    return {
        "available": True,
        "active_variable_count": active_count,
        "labelled_variable_count": labelled_count,
        "label_coverage": labelled_count / active_count,
        "variable_accuracy": float(correct.sum().item() / labelled_count),
        "fully_labelled_record_count": eligible_count,
        "record_count": int(logits.shape[0]),
        "record_exact_match_coverage": eligible_count / logits.shape[0],
        "record_exact_match": exact_match,
        "record_exact_match_unavailable_reason": (
            None if eligible_count else "no record has labels for every active variable"
        ),
    }


def summarize_variable_records(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize exact, once-per-record variable predictions after DDP gather."""

    if not rows:
        raise ValueError("no per-variable evaluation rows are available")
    active_count = 0
    labelled_count = 0
    correct_count = 0
    loss_sum = 0.0
    fully_labelled = 0
    exactly_correct = 0
    group_totals: dict[str, int] = {}
    group_correct: dict[str, int] = {}
    for index, row in enumerate(rows):
        active = row.get("active_variable_count")
        ids = row.get("variable_ids")
        gold = row.get("variable_gold")
        predicted = row.get("variable_predicted")
        groups = row.get("variable_groups")
        if not isinstance(active, int) or isinstance(active, bool) or active < 1:
            raise ValueError(f"variable evaluation row {index} has invalid active count")
        if not all(isinstance(values, list) for values in (ids, gold, predicted)):
            raise ValueError(f"variable evaluation row {index} must contain prediction lists")
        if len(ids) != len(gold) or len(ids) != len(predicted):
            raise ValueError(f"variable evaluation row {index} has misaligned predictions")
        if groups is not None and (
            not isinstance(groups, list) or len(groups) != len(ids)
            or not all(isinstance(group, str) and group for group in groups)
        ):
            raise ValueError(f"variable evaluation row {index} has invalid groups")
        if len(ids) > active or len(set(ids)) != len(ids) or not all(
            isinstance(variable_id, str) and variable_id for variable_id in ids
        ):
            raise ValueError(f"variable evaluation row {index} has invalid variable IDs")
        row_loss = row.get("variable_loss_sum")
        if not isinstance(row_loss, (int, float)) or isinstance(row_loss, bool):
            raise ValueError(f"variable evaluation row {index} has invalid loss sum")
        active_count += active
        labelled_count += len(ids)
        loss_sum += float(row_loss)
        matches = [left == right for left, right in zip(gold, predicted)]
        correct_count += sum(matches)
        if groups is not None:
            for group, match in zip(groups, matches):
                group_totals[group] = group_totals.get(group, 0) + 1
                group_correct[group] = group_correct.get(group, 0) + int(match)
        if len(ids) == active:
            fully_labelled += 1
            exactly_correct += int(all(matches))
    if labelled_count == 0:
        raise ValueError("no explicit per-variable labels are available")
    result = {
        "available": True,
        "active_variable_count": active_count,
        "labelled_variable_count": labelled_count,
        "label_coverage": labelled_count / active_count,
        "variable_loss": loss_sum / labelled_count,
        "variable_accuracy": correct_count / labelled_count,
        "fully_labelled_record_count": fully_labelled,
        "record_count": len(rows),
        "record_exact_match_coverage": fully_labelled / len(rows),
        "record_exact_match": (
            exactly_correct / fully_labelled if fully_labelled else None
        ),
        "record_exact_match_unavailable_reason": (
            None if fully_labelled else "no record has labels for every active variable"
        ),
    }
    if group_totals:
        result["groups"] = {
            group: {
                "count": group_totals[group],
                "accuracy": group_correct[group] / group_totals[group],
            }
            for group in sorted(group_totals)
        }
    return result


def _validate_metric_inputs(
    logits: torch.Tensor,
    label_ids: torch.Tensor,
    label_mask: torch.Tensor,
    variable_mask: torch.Tensor,
) -> None:
    if logits.ndim != 3 or logits.shape[-1] < 2:
        raise ValueError("logits must have shape [batch, variables, at-least-two-labels]")
    expected = logits.shape[:2]
    if (
        label_ids.shape != expected
        or label_mask.shape != expected
        or variable_mask.shape != expected
    ):
        raise ValueError("label IDs and masks must match logits [batch, variables]")
    active = variable_mask.bool()
    observed = label_mask.bool()
    if not torch.all(active.any(dim=1)):
        raise ValueError("each record must contain at least one active variable")
    if torch.any(observed & ~active):
        raise ValueError("label_mask may only select active structural variables")
    if not torch.any(observed):
        raise ValueError("no explicit per-variable labels are available")
    selected_ids = label_ids[observed]
    if torch.any((selected_ids < 0) | (selected_ids >= logits.shape[-1])):
        raise ValueError("observed variable label ID is outside the logit vocabulary")


__all__ = [
    "IGNORE_INDEX",
    "PerVariableOutputHead",
    "VariableBatch",
    "VariableOutput",
    "build_variable_batch",
    "balanced_variable_cross_entropy",
    "encode_variable_labels",
    "per_variable_metrics",
    "summarize_variable_records",
    "variable_cross_entropy",
]
