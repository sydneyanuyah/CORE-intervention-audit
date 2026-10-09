"""Registered L3 operator alternatives with one shared hidden-state interface."""

from __future__ import annotations

import torch
from torch import nn


class LoReFTOperator(nn.Module):
    """Intervention-conditioned low-rank representation finetuning residual."""

    def __init__(self, intervention_count: int, hidden_size: int, rank: int):
        super().__init__()
        self.command = nn.Embedding(intervention_count + 1, rank)
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden: torch.Tensor, intervention_id: torch.Tensor) -> torch.Tensor:
        coordinates = self.down(hidden) + self.command(intervention_id)[:, None, :]
        return hidden + self.up(torch.tanh(coordinates))


class TaskVectorAddOperator(nn.Module):
    """Add one shared task vector to every editable world token."""

    def __init__(self, intervention_count: int, hidden_size: int):
        super().__init__()
        self.vectors = nn.Embedding(intervention_count + 1, hidden_size)
        nn.init.zeros_(self.vectors.weight)

    def forward(self, hidden: torch.Tensor, intervention_id: torch.Tensor) -> torch.Tensor:
        return hidden + self.vectors(intervention_id)[:, None, :]


class RouterOperator(nn.Module):
    """State/command-conditioned mixture of fixed-vector, low-rank, and gated experts."""

    def __init__(self, core, intervention_count: int, token_count: int, hidden_size: int, rank: int):
        super().__init__()
        self.experts = nn.ModuleList([
            core.O1FixedVector(intervention_count, token_count, hidden_size),
            core.O2ConditionalLowRank(intervention_count, token_count, hidden_size, rank),
            core.O3StateGated(intervention_count, hidden_size, rank),
        ])
        self.command = nn.Embedding(intervention_count + 1, hidden_size)
        self.router = nn.Linear(2 * hidden_size, len(self.experts))

    def forward(self, hidden: torch.Tensor, intervention_id: torch.Tensor) -> torch.Tensor:
        features = torch.cat([hidden.mean(dim=1), self.command(intervention_id)], dim=-1)
        weights = torch.softmax(self.router(features), dim=-1)
        candidates = torch.stack([expert(hidden, intervention_id) for expert in self.experts], dim=1)
        return (candidates * weights[:, :, None, None]).sum(dim=1)


def build_operator(method: str, core, intervention_count: int, token_count: int, hidden_size: int, rank: int):
    if method == "o1":
        return core.O1FixedVector(intervention_count, token_count, hidden_size)
    if method == "o2":
        return core.O2ConditionalLowRank(intervention_count, token_count, hidden_size, rank)
    if method == "o3":
        return core.O3StateGated(intervention_count, hidden_size, rank)
    if method == "loreft":
        return LoReFTOperator(intervention_count, hidden_size, rank)
    if method == "task_vector_add":
        return TaskVectorAddOperator(intervention_count, hidden_size)
    if method == "router":
        return RouterOperator(core, intervention_count, token_count, hidden_size, rank)
    raise ValueError(f"unsupported L3 operator method: {method}")
