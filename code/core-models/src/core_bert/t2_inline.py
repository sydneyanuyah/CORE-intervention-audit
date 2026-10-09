"""T2-b inline-command masking, pooling, and leakage-check primitives.

Masks use ``True`` for an allowed attention edge.  They are intentionally
model-agnostic so a split BERT reader can translate them to its preferred
boolean or additive attention-mask representation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


Pooling = Literal["mean", "last"]


@dataclass(frozen=True)
class T2Masks:
    """Attention masks for the two halves of a T2-b split reader."""

    pre_edit: torch.Tensor
    post_edit: torch.Tensor
    instruction_tokens: torch.Tensor


def _as_bool_mask(name: str, mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim != 2:
        raise ValueError(f"{name} must have shape [batch, sequence]")
    return mask.bool()


def build_t2a_masks(
    valid_tokens: torch.Tensor,
    command_tokens: torch.Tensor,
    do_tokens: torch.Tensor,
) -> T2Masks:
    """Build the naive T2-a ablation with command visibility throughout."""

    valid = _as_bool_mask("valid_tokens", valid_tokens)
    command = _as_bool_mask("command_tokens", command_tokens)
    do = _as_bool_mask("do_tokens", do_tokens)
    if not (valid.shape == command.shape == do.shape):
        raise ValueError("all masks must have the same shape")
    if ((command | do) & ~valid).any() or (command & do).any():
        raise ValueError("command and [DO] positions must be valid and disjoint")
    if not torch.all(do.sum(dim=1) == 1):
        raise ValueError("each example must contain exactly one [DO] token")
    if not torch.all(command.sum(dim=1) >= 1):
        raise ValueError("each example must contain at least one command token")
    allowed = valid[:, :, None] & valid[:, None, :]
    return T2Masks(
        pre_edit=allowed,
        post_edit=allowed.clone(),
        instruction_tokens=command | do,
    )


def build_t2b_masks(
    valid_tokens: torch.Tensor,
    command_tokens: torch.Tensor,
    do_tokens: torch.Tensor,
    slot_tokens: torch.Tensor,
) -> T2Masks:
    """Build the exact encode-only masks specified for T2-b.

    Before the edit, slots cannot read ``[DO]`` or command tokens.  After the
    edit, neither ``[DO]`` nor command tokens can be read by any live token.
    Hiding ``[DO]`` post-edit is necessary because its layer-2 state is part of
    the instruction vector and could otherwise leak the instruction.
    """

    valid = _as_bool_mask("valid_tokens", valid_tokens)
    command = _as_bool_mask("command_tokens", command_tokens)
    do = _as_bool_mask("do_tokens", do_tokens)
    slots = _as_bool_mask("slot_tokens", slot_tokens)
    if not (valid.shape == command.shape == do.shape == slots.shape):
        raise ValueError("all masks must have the same shape")
    if ((command | do | slots) & ~valid).any():
        raise ValueError("command, [DO], and slot positions must be valid tokens")
    if (command & do).any() or (command & slots).any() or (do & slots).any():
        raise ValueError("command, [DO], and slot masks must be disjoint")
    if not torch.all(do.sum(dim=1) == 1):
        raise ValueError("each example must contain exactly one [DO] token")
    if not torch.all(command.sum(dim=1) >= 1):
        raise ValueError("each example must contain at least one command token")

    allowed = valid[:, :, None] & valid[:, None, :]
    instruction = command | do

    pre_edit = allowed & ~(slots[:, :, None] & instruction[:, None, :])
    post_edit = allowed & ~instruction[:, None, :]
    return T2Masks(
        pre_edit=pre_edit,
        post_edit=post_edit,
        instruction_tokens=instruction,
    )


def allowed_to_additive(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Convert an allowed-edge mask to a BERT-style additive attention bias."""

    if mask.ndim != 3:
        raise ValueError("mask must have shape [batch, query, key]")
    bias = torch.zeros(mask.shape, dtype=dtype, device=mask.device)
    return bias.masked_fill(~mask.bool(), torch.finfo(dtype).min)[:, None, :, :]


def pool_t2_instruction(
    layer2_hidden: torch.Tensor,
    command_tokens: torch.Tensor,
    do_tokens: torch.Tensor,
    pooling: Pooling = "mean",
) -> torch.Tensor:
    """Return ``[pooled command ; h([DO])]`` without adding parameters."""

    command = _as_bool_mask("command_tokens", command_tokens)
    do = _as_bool_mask("do_tokens", do_tokens)
    if layer2_hidden.ndim != 3 or layer2_hidden.shape[:2] != command.shape:
        raise ValueError("hidden must have shape [batch, sequence, hidden]")
    if do.shape != command.shape or not torch.all(do.sum(1) == 1):
        raise ValueError("each example must have exactly one [DO] token")
    counts = command.sum(1)
    if not torch.all(counts >= 1):
        raise ValueError("each example must have at least one command token")

    if pooling == "mean":
        pooled = (layer2_hidden * command.unsqueeze(-1)).sum(1) / counts.unsqueeze(-1)
    elif pooling == "last":
        positions = torch.arange(command.shape[1], device=command.device)
        last = (command * positions).argmax(1)
        pooled = layer2_hidden[torch.arange(len(layer2_hidden), device=command.device), last]
    else:
        raise ValueError(f"unsupported pooling: {pooling}")
    do_index = do.to(torch.int64).argmax(1)
    do_hidden = layer2_hidden[
        torch.arange(len(layer2_hidden), device=layer2_hidden.device), do_index
    ]
    return torch.cat([pooled, do_hidden], dim=-1)


def zero_editor_leakage_metrics(
    no_instruction_logits: torch.Tensor,
    zero_editor_logits: torch.Tensor,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> dict[str, float | bool]:
    """Mechanically check that a zeroed T2-b editor cannot leak a command."""

    if no_instruction_logits.shape != zero_editor_logits.shape:
        raise ValueError("logit tensors must have identical shape")
    delta = (no_instruction_logits - zero_editor_logits).abs()
    return {
        "passed": bool(torch.allclose(no_instruction_logits, zero_editor_logits, atol=atol, rtol=rtol)),
        "max_abs_logit_delta": float(delta.max().item()) if delta.numel() else 0.0,
        "mean_abs_logit_delta": float(delta.mean().item()) if delta.numel() else 0.0,
        "prediction_agreement": float(
            (no_instruction_logits.argmax(-1) == zero_editor_logits.argmax(-1))
            .to(torch.float32)
            .mean()
            .item()
        ) if delta.numel() else 1.0,
    }


def assert_zero_editor_no_leakage(
    no_instruction_logits: torch.Tensor,
    zero_editor_logits: torch.Tensor,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> None:
    metrics = zero_editor_leakage_metrics(
        no_instruction_logits, zero_editor_logits, atol=atol, rtol=rtol
    )
    if not metrics["passed"]:
        raise AssertionError(f"T2-b masking leakage detected: {metrics}")
