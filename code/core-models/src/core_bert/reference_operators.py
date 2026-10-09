"""Registered T0 learned-ID and T1 separate-text reference operators."""

from __future__ import annotations

import torch
from torch import nn


class LearnedInterventionEncoder(nn.Module):
    """T0's synthetic-only lookup-table intervention representation."""

    def __init__(self, intervention_count: int, hidden_size: int) -> None:
        super().__init__()
        if intervention_count < 1 or hidden_size < 1:
            raise ValueError("intervention_count and hidden_size must be positive")
        self.embedding = nn.Embedding(intervention_count, hidden_size)

    def forward(self, intervention_id: torch.Tensor) -> torch.Tensor:
        if intervention_id.ndim != 1:
            raise ValueError("intervention_id must have shape [batch]")
        if (intervention_id < 0).any() or (intervention_id >= self.embedding.num_embeddings).any():
            raise ValueError("intervention_id is outside the registered T0 inventory")
        return self.embedding(intervention_id)


class SeparateTextEncoder(nn.Module):
    """T1 sentence encoder with no intervention-ID lookup table."""

    def __init__(
        self,
        initial_word_embeddings: torch.Tensor,
        padding_idx: int,
        hidden_size: int,
        *,
        max_length: int = 64,
    ) -> None:
        super().__init__()
        if initial_word_embeddings.ndim != 2 or initial_word_embeddings.shape[1] != hidden_size:
            raise ValueError("initial word embeddings must have shape [vocabulary, hidden]")
        if max_length < 1:
            raise ValueError("max_length must be positive")
        self.max_length = int(max_length)
        self.word_embeddings = nn.Embedding.from_pretrained(
            initial_word_embeddings.detach().clone(), freeze=True, padding_idx=padding_idx
        )
        self.position_embeddings = nn.Parameter(
            torch.randn(self.max_length, hidden_size) * 0.02
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=4,
            dim_feedforward=2 * hidden_size,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.output_norm = nn.LayerNorm(hidden_size)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("T1 text IDs and mask must share [batch, sequence]")
        if input_ids.shape[1] > self.max_length:
            raise ValueError("T1 intervention text exceeds configured maximum length")
        if not torch.all(attention_mask.bool().any(1)):
            raise ValueError("every T1 intervention must contain a text token")
        hidden = self.word_embeddings(input_ids)
        hidden = hidden + self.position_embeddings[: input_ids.shape[1]][None]
        hidden = self.encoder(hidden, src_key_padding_mask=~attention_mask.bool())
        weights = attention_mask.to(hidden.dtype).unsqueeze(-1)
        pooled = (hidden * weights).sum(1) / weights.sum(1).clamp_min(1.0)
        return self.output_norm(pooled)


class StateGatedReferenceEditor(nn.Module):
    """Apply the same state-gated edit to T0 or T1 intervention embeddings."""

    def __init__(self, hidden_size: int, rank: int = 16) -> None:
        super().__init__()
        if hidden_size < 1 or rank < 1:
            raise ValueError("hidden_size and rank must be positive")
        self.hidden_size = int(hidden_size)
        joint_size = 2 * self.hidden_size
        self.down = nn.Linear(joint_size, rank, bias=False)
        self.up = nn.Linear(rank, self.hidden_size, bias=False)
        self.gate = nn.Linear(joint_size, self.hidden_size)
        nn.init.zeros_(self.up.weight)
        nn.init.constant_(self.gate.bias, -2.0)

    def forward(
        self, slots: torch.Tensor, intervention: torch.Tensor
    ) -> torch.Tensor:
        if slots.ndim != 3 or slots.shape[2] != self.hidden_size:
            raise ValueError("slots must have shape [batch, slots, hidden]")
        if intervention.shape != (slots.shape[0], self.hidden_size):
            raise ValueError("intervention must have shape [batch, hidden]")
        expanded = intervention[:, None].expand(-1, slots.shape[1], -1)
        joint = torch.cat([slots, expanded], dim=-1)
        update = self.up(torch.tanh(self.down(joint)))
        return slots + torch.sigmoid(self.gate(joint)) * update


def compact_masked_tokens(
    input_ids: torch.Tensor, selected: torch.Tensor, padding_idx: int,
    *, max_length: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compact selected tokens so T1 positions cannot encode passage length."""

    if input_ids.ndim != 2 or selected.shape != input_ids.shape:
        raise ValueError("input IDs and selected mask must share [batch, sequence]")
    counts = selected.bool().sum(1)
    if not torch.all((counts > 0) & (counts <= max_length)):
        raise ValueError("each selected intervention must contain 1..max_length tokens")
    width = int(counts.max().item())
    compact = input_ids.new_full((input_ids.shape[0], width), int(padding_idx))
    mask = torch.zeros_like(compact, dtype=torch.bool)
    for row in range(input_ids.shape[0]):
        values = input_ids[row][selected[row].bool()]
        compact[row, : len(values)] = values
        mask[row, : len(values)] = True
    return compact, mask


__all__ = [
    "LearnedInterventionEncoder",
    "SeparateTextEncoder",
    "StateGatedReferenceEditor",
    "compact_masked_tokens",
]
