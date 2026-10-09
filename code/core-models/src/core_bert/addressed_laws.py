"""Explicit algebraic losses for L1's addressed O3 editor."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


LAW_NAMES = ("identity", "idempotence", "commutation", "last_write_wins")


@dataclass(frozen=True)
class LawPairIndex:
    commuting: tuple[int, ...]
    opposite: tuple[int, ...]


def build_law_pair_index(
    graph_groups: Iterable[str], target_slots: Iterable[int], value_ids: Iterable[int]
) -> LawPairIndex:
    """Choose deterministic within-graph partners for the paired L1 laws."""

    groups, targets, values = tuple(graph_groups), tuple(target_slots), tuple(value_ids)
    if not groups or not (len(groups) == len(targets) == len(values)):
        raise ValueError("law-pair metadata must be non-empty and aligned")
    commuting, opposite = [], []
    for row, (group, target, value) in enumerate(zip(groups, targets, values)):
        if not isinstance(group, str) or not group or target < 0 or value < 0:
            raise ValueError(f"invalid law-pair metadata at row {row}")
        commute = next((
            candidate for candidate, (other_group, other_target) in
            enumerate(zip(groups, targets))
            if other_group == group and other_target != target
        ), None)
        overwrite = next((
            candidate for candidate, (other_group, other_target, other_value) in
            enumerate(zip(groups, targets, values))
            if other_group == group and other_target == target and other_value != value
        ), None)
        if commute is None or overwrite is None:
            raise ValueError(f"row {row} lacks a complete within-graph law pair")
        commuting.append(commute)
        opposite.append(overwrite)
    return LawPairIndex(tuple(commuting), tuple(opposite))


def addressed_law_losses(
    operator: nn.Module,
    slots: torch.Tensor,
    instruction: torch.Tensor,
    *,
    enabled: Iterable[str],
    commuting_instruction: torch.Tensor | None = None,
    opposite_instruction: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Return only the preregistered addressed-law losses.

    ``commuting_instruction`` must describe a different target.  The caller
    constructs that pairing from immutable record metadata.  Likewise,
    ``opposite_instruction`` must describe the opposite value for the same
    target.  Requiring those tensors explicitly prevents accidental random
    pairings from changing the scientific contract.
    """

    selected = tuple(enabled)
    unknown = set(selected) - set(LAW_NAMES)
    if unknown or len(selected) != len(set(selected)):
        raise ValueError(f"invalid addressed law set: {selected!r}")
    if slots.ndim != 3 or instruction.shape != (
        slots.shape[0], 2 * slots.shape[2]
    ):
        raise ValueError("slots/instruction shapes violate the addressed contract")

    losses: dict[str, torch.Tensor] = {}
    once: torch.Tensor | None = None
    if "identity" in selected:
        losses["identity"] = F.mse_loss(
            operator(slots, torch.zeros_like(instruction)), slots
        )
    if "idempotence" in selected:
        once = operator(slots, instruction)
        losses["idempotence"] = F.mse_loss(operator(once, instruction), once)
    if "commutation" in selected:
        if commuting_instruction is None or commuting_instruction.shape != instruction.shape:
            raise ValueError("commutation requires one validated different-target instruction per row")
        losses["commutation"] = F.mse_loss(
            operator(operator(slots, instruction), commuting_instruction),
            operator(operator(slots, commuting_instruction), instruction),
        )
    if "last_write_wins" in selected:
        if opposite_instruction is None or opposite_instruction.shape != instruction.shape:
            raise ValueError("last-write-wins requires one validated opposite-value instruction per row")
        once = once if once is not None else operator(slots, instruction)
        losses["last_write_wins"] = F.mse_loss(
            operator(operator(slots, opposite_instruction), instruction), once
        )
    return losses


__all__ = [
    "LAW_NAMES", "LawPairIndex", "build_law_pair_index", "addressed_law_losses"
]
