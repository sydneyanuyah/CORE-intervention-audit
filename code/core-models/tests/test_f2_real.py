from __future__ import annotations

import pytest
import torch
from torch import nn

from train_f2_real import (
    CanonicalScientificOperator,
    binary_interventions,
    intervention_ids,
    source_seed,
)


class AddOne(nn.Module):
    def forward(self, slots, intervention_id):
        assert intervention_id.tolist() == [3]
        return slots + 1


def test_source_seed_mapping_is_frozen_and_ordinary() -> None:
    assert source_seed(301) == 2026090601
    assert source_seed(320) == 2026090620
    with pytest.raises(ValueError):
        source_seed(2026090601)


def test_canonical_operator_changes_only_first_sixteen_slots() -> None:
    slots = torch.zeros(1, 30, 2)
    result = CanonicalScientificOperator(AddOne())(slots, torch.tensor([3]))
    assert torch.equal(result[:, :16], torch.ones(1, 16, 2))
    assert torch.equal(result[:, 16:], torch.zeros(1, 14, 2))


def test_binary_intervention_ids_use_dynamic_slot_and_value() -> None:
    side = {"intervention_binary": torch.tensor([1, 0]), "target_slot": torch.tensor([2, 7])}
    assert intervention_ids(side).tolist() == [5, 14]
    with pytest.raises(ValueError):
        intervention_ids({"intervention_binary": torch.tensor([1]), "target_slot": torch.tensor([16])})


def test_binary_interventions_use_authoritative_record_values() -> None:
    records = [
        {"intervention": {"value": 1, "value_token": "yes"}},
        {"intervention": {"value": 0, "value_token": "no"}},
    ]
    assert binary_interventions(records, torch.device("cpu")).tolist() == [1, 0]
    directional = [
        {"intervention": {"value": "surface wording", "value_token": "more"}},
        {"intervention": {"value": "surface wording", "value_token": "less"}},
    ]
    assert binary_interventions(directional, torch.device("cpu")).tolist() == [1, 0]
    with pytest.raises(ValueError):
        binary_interventions([{"intervention": {"value": "yes"}}], torch.device("cpu"))


def test_partial_probe_supervision_leaves_unobserved_slots_masked() -> None:
    from train_f2_real import labels_and_masks

    record = {
        "structure_kind": "dag",
        "graph": {"nodes": ["A", "B"]},
        "intervention": {"target": "A"},
        "factual": {"state": None},
        "intervened": {"state": None},
        "probes": [{"variable": "B", "answer_after": "more", "actually_changed": True}],
    }
    labels, changed, preserved = labels_and_masks([record], torch.device("cpu"))
    assert labels[0, 0].item() == -100
    assert labels[0, 1].item() >= 0
    assert changed[0, :2].tolist() == [False, True]
    assert preserved[0, :2].tolist() == [False, False]
