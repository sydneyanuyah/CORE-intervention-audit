import pytest
import torch
from torch import nn

from core_bert.addressed_laws import addressed_law_losses, build_law_pair_index


class AdditiveOperator(nn.Module):
    def forward(self, slots, instruction):
        return slots + instruction[:, None, : slots.shape[-1]]


def tensors():
    return torch.zeros(2, 3, 4), torch.ones(2, 8)


def test_identity_and_commutation_are_exact_for_additive_operator():
    slots, instruction = tensors()
    losses = addressed_law_losses(
        AdditiveOperator(), slots, instruction,
        enabled=("identity", "commutation"),
        commuting_instruction=instruction * 2,
    )
    assert set(losses) == {"identity", "commutation"}
    assert losses["identity"].item() == 0
    assert losses["commutation"].item() == 0


def test_idempotence_and_last_write_wins_detect_additive_violation():
    slots, instruction = tensors()
    losses = addressed_law_losses(
        AdditiveOperator(), slots, instruction,
        enabled=("idempotence", "last_write_wins"),
        opposite_instruction=-instruction,
    )
    assert losses["idempotence"].item() > 0
    assert losses["last_write_wins"].item() > 0


def test_pair_dependent_laws_fail_closed_without_validated_pair():
    slots, instruction = tensors()
    with pytest.raises(ValueError, match="different-target"):
        addressed_law_losses(
            AdditiveOperator(), slots, instruction, enabled=("commutation",)
        )
    with pytest.raises(ValueError, match="opposite-value"):
        addressed_law_losses(
            AdditiveOperator(), slots, instruction, enabled=("last_write_wins",)
        )


def test_pair_index_is_deterministic_and_stays_within_graph():
    pairs = build_law_pair_index(
        ["g1", "g1", "g1", "g1"], [0, 0, 1, 1], [0, 1, 0, 1]
    )
    assert pairs.commuting == (2, 2, 0, 0)
    assert pairs.opposite == (1, 0, 3, 2)


def test_pair_index_rejects_incomplete_graph_groups():
    with pytest.raises(ValueError, match="complete within-graph"):
        build_law_pair_index(["g1", "g1"], [0, 1], [0, 0])
