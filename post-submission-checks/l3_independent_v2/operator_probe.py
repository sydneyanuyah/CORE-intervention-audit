"""Frozen-feature O2/O3 probe used by the Qwen-7B CLadder extension."""

from __future__ import annotations

from typing import Iterable, Mapping

import torch
from torch import nn
from torch.nn import functional as F


class O2ConditionalLowRank(nn.Module):
    def __init__(self, intervention_count: int, hidden_size: int, rank: int):
        super().__init__()
        self.embedding = nn.Embedding(intervention_count + 1, hidden_size)
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden: torch.Tensor, intervention_id: torch.Tensor) -> torch.Tensor:
        update = self.up(torch.tanh(self.down(self.embedding(intervention_id))))
        return hidden + update


class O3StateGated(nn.Module):
    def __init__(self, intervention_count: int, hidden_size: int, rank: int):
        super().__init__()
        self.embedding = nn.Embedding(intervention_count + 1, hidden_size)
        self.down = nn.Linear(2 * hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=False)
        self.gate = nn.Linear(2 * hidden_size, hidden_size)
        nn.init.zeros_(self.up.weight)
        nn.init.constant_(self.gate.bias, -2.0)

    def forward(self, hidden: torch.Tensor, intervention_id: torch.Tensor) -> torch.Tensor:
        joint = torch.cat([hidden, self.embedding(intervention_id)], dim=-1)
        return hidden + torch.sigmoid(self.gate(joint)) * self.up(torch.tanh(self.down(joint)))


class StateHead(nn.Module):
    def __init__(self, hidden_size: int, node_count: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.output = nn.Linear(hidden_size, node_count * 2)
        self.node_count = node_count

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.output(self.norm(hidden)).reshape(-1, self.node_count, 2)


def intervention_id(record: Mapping, nodes: list[str], *, invert: bool = False) -> int:
    intervention = record["intervention"]
    value = int(intervention["value"])
    if invert:
        value = 1 - value
    return nodes.index(intervention["target"]) * 2 + value


def state_labels(record: Mapping, nodes: list[str]) -> list[int]:
    state = record["intervened"]["state"]
    return [int(state[node]) for node in nodes]


def balanced_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    changed: torch.Tensor,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    if valid is None:
        valid = labels != -100
    per = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=-100, reduction="none").reshape_as(labels)
    changed = changed & valid
    preserved = (~changed) & valid
    change_loss = (per * changed).sum(1) / changed.sum(1).clamp_min(1)
    preserve_loss = (per * preserved).sum(1) / preserved.sum(1).clamp_min(1)
    return 0.5 * (change_loss.mean() + preserve_loss.mean())


def paired_metrics(pairs: Iterable[Mapping], predictions: Mapping[str, list[int]]) -> dict:
    change_correct = change_total = preserve_correct = preserve_total = 0
    for pair in pairs:
        predicted = predictions[pair["pair_id"]]
        for index, gold in enumerate(pair["gold_outputs"]):
            if pair["change_mask"][index]:
                change_correct += int(predicted[index] == int(gold)); change_total += 1
            if pair["preservation_mask"][index]:
                preserve_correct += int(predicted[index] == int(gold)); preserve_total += 1
    change = change_correct / change_total
    preserve = preserve_correct / preserve_total
    return {
        "change_accuracy": change,
        "preservation_accuracy": preserve,
        "two_edit_balanced": 0.5 * (change + preserve),
        "change_decisions": change_total,
        "preservation_decisions": preserve_total,
    }
