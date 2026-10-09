"""Shared textual-conditioning editor for T2 and the T3-a ablation.

The instruction is already encoded by the reader as two hidden-size vectors.
Consequently this module has no intervention-ID table and owns no addressing or
pointer-selection logic.
"""

from __future__ import annotations

import torch
from torch import nn


def gated_edit_parameter_count(hidden_size: int, rank: int = 16) -> int:
    """Return the exact trainable parameter count of the registered gated edit."""

    if hidden_size < 1 or rank < 1:
        raise ValueError("hidden_size and rank must be positive")
    # down: (3d)r; up: rd; gate: (3d)d + d bias.
    return 3 * hidden_size * hidden_size + 4 * hidden_size * rank + hidden_size


class AddressedGatedOperator(nn.Module):
    """Rank-r gated residual edit ``O(slot, [address; value/replacement])``.

    It consumes slots shaped ``[B, S, d]`` and a textual instruction shaped
    ``[B, 2d]``.  The zero-initialized up-projection makes a fresh operator an
    exact no-op.  ``identity_mode`` provides the mechanical zero-editor control
    without changing the module passed to :class:`AddressedWorldReader`.
    """

    def __init__(
        self, hidden_size: int, rank: int = 16, *, identity_mode: bool = False
    ) -> None:
        super().__init__()
        if hidden_size < 1 or rank < 1:
            raise ValueError("hidden_size and rank must be positive")
        self.hidden_size = int(hidden_size)
        self.rank = int(rank)
        self.identity_mode = bool(identity_mode)
        joint_size = 3 * self.hidden_size
        self.down = nn.Linear(joint_size, self.rank, bias=False)
        self.up = nn.Linear(self.rank, self.hidden_size, bias=False)
        self.gate = nn.Linear(joint_size, self.hidden_size)
        nn.init.zeros_(self.up.weight)
        nn.init.constant_(self.gate.bias, -2.0)

    @property
    def parameter_count(self) -> int:
        """Exact number of trainable scalar parameters."""

        return gated_edit_parameter_count(self.hidden_size, self.rank)

    def set_identity_mode(self, enabled: bool = True) -> None:
        """Enable or disable the zero-editor control for subsequent forwards."""

        self.identity_mode = bool(enabled)

    def _validate(self, slots: torch.Tensor, instruction: torch.Tensor) -> None:
        if slots.ndim != 3:
            raise ValueError("slots must have shape [batch, slots, hidden]")
        if instruction.ndim != 2:
            raise ValueError("instruction must have shape [batch, 2*hidden]")
        if slots.shape[2] != self.hidden_size:
            raise ValueError(
                f"slot hidden dimension must be {self.hidden_size}, found {slots.shape[2]}"
            )
        expected = (slots.shape[0], 2 * self.hidden_size)
        if tuple(instruction.shape) != expected:
            raise ValueError(
                f"instruction shape must be {expected}, found {tuple(instruction.shape)}"
            )
        if not slots.is_floating_point() or not instruction.is_floating_point():
            raise TypeError("slots and instruction must be floating-point tensors")

    def forward(
        self, slots: torch.Tensor, instruction: torch.Tensor
    ) -> torch.Tensor:
        self._validate(slots, instruction)
        if self.identity_mode:
            return slots
        expanded = instruction[:, None, :].expand(-1, slots.shape[1], -1)
        joint = torch.cat([slots, expanded], dim=-1)
        update = self.up(torch.tanh(self.down(joint)))
        return slots + torch.sigmoid(self.gate(joint)) * update

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, rank={self.rank}, "
            f"identity_mode={self.identity_mode}, parameters={self.parameter_count}"
        )


class IdentityAddressedOperator(nn.Module):
    """Parameter-free editor implementing the T2 zero-editor floor."""

    parameter_count = 0

    def forward(
        self, slots: torch.Tensor, instruction: torch.Tensor
    ) -> torch.Tensor:
        if slots.ndim != 3 or instruction.ndim != 2:
            raise ValueError(
                "expected slots [batch, slots, hidden] and instruction [batch, 2*hidden]"
            )
        if instruction.shape != (slots.shape[0], 2 * slots.shape[2]):
            raise ValueError("instruction must concatenate two hidden-size vectors")
        return slots


# Public name used by the addressed training entry point. Keeping this as an
# alias preserves a single parameterization and state-dict contract.
InstructionGatedEdit = AddressedGatedOperator


__all__ = [
    "AddressedGatedOperator",
    "IdentityAddressedOperator",
    "InstructionGatedEdit",
    "gated_edit_parameter_count",
]
