"""Deterministic prediction-level floors for CORE law tables."""

from __future__ import annotations

import torch


def do_everything(factual: torch.Tensor, intervention_ids: torch.Tensor, node_count: int) -> torch.Tensor:
    """Set every variable to the requested target value; no-op preserves factual state."""
    if factual.ndim != 2 or factual.shape[1] != node_count:
        raise ValueError("factual must have shape [batch, node_count]")
    if intervention_ids.shape != (len(factual),):
        raise ValueError("intervention_ids must have shape [batch]")
    values = (intervention_ids % 2).unsqueeze(1).expand(-1, node_count)
    noop = intervention_ids == node_count * 2
    return torch.where(noop.unsqueeze(1), factual, values)
