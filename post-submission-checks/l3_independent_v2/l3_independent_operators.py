"""Independent frozen-feature operator classes for the corrected L3 expansion.

These classes intentionally expose different parameterizations.  The previous
expansion mapped all three registered labels to ``O2ConditionalLowRank``.
"""

from __future__ import annotations

import torch
from torch import nn


class ParameterMatchedFeatureAdapter(nn.Module):
    """Command-blind low-rank adapter over frozen backbone features.

    This is the faithful PEFT comparator available in a frozen-feature probe;
    it is not a Q/V LoRA update to the inaccessible frozen backbone weights.
    Rank 46 nearly matches O2's parameter count when there are 60 commands plus
    the no-op command: 92H versus O2's 93H parameters.
    """

    implementation_id = "parameter_matched_frozen_feature_adapter_v1"

    def __init__(self, intervention_count: int, hidden_size: int, rank: int = 46):
        super().__init__()
        del intervention_count
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden: torch.Tensor, intervention_id: torch.Tensor) -> torch.Tensor:
        del intervention_id
        return hidden + self.up(torch.tanh(self.down(hidden)))


class CommandConditionedLoReFT(nn.Module):
    """State- and command-conditioned low-rank representation residual."""

    implementation_id = "command_conditioned_loreft_style_residual_v1"

    def __init__(self, intervention_count: int, hidden_size: int, rank: int = 16):
        super().__init__()
        self.command = nn.Embedding(intervention_count + 1, rank)
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden: torch.Tensor, intervention_id: torch.Tensor) -> torch.Tensor:
        coordinates = self.down(hidden) + self.command(intervention_id)
        return hidden + self.up(torch.tanh(coordinates))


class StateIndependentO2(nn.Module):
    """Command-indexed low-rank residual that does not inspect hidden state."""

    implementation_id = "state_independent_command_indexed_o2_v1"

    def __init__(self, intervention_count: int, hidden_size: int, rank: int = 16):
        super().__init__()
        self.embedding = nn.Embedding(intervention_count + 1, hidden_size)
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden: torch.Tensor, intervention_id: torch.Tensor) -> torch.Tensor:
        update = self.up(torch.tanh(self.down(self.embedding(intervention_id))))
        return hidden + update


def build_independent_operator(method: str, intervention_count: int, hidden_size: int) -> nn.Module:
    if method == "lora_matched":
        return ParameterMatchedFeatureAdapter(intervention_count, hidden_size)
    if method == "loreft":
        return CommandConditionedLoReFT(intervention_count, hidden_size)
    if method == "o2":
        return StateIndependentO2(intervention_count, hidden_size)
    raise ValueError(f"unsupported independent L3 method: {method}")


def trainable_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
