import pytest

from core_bert.benchmark_data import BenchmarkDataset, BenchmarkExample
from core_bert.l1_data import L1PairedCollator, L1PairedDataset, binary_intervention


def example(record_id, target, value):
    record = {
        "id": record_id,
        "graph": {"nodes": ["x", "y"], "edges": []},
        "intervention": {"target": target, "value": value},
    }
    return BenchmarkExample(record, record_id, "xor", 0, "value_set", "g", "w")


def test_l1_pairing_uses_same_target_opposite_value_and_other_target():
    rows = [example("x0", "x", 0), example("x1", "x", 1), example("y0", "y", 0), example("y1", "y", 1)]
    paired = L1PairedDataset(BenchmarkDataset(rows))
    assert paired[0].opposite.record_id == "x1"
    assert paired[0].commuting.record_id == "y0"


@pytest.mark.parametrize("value,expected", [(True, 1), (False, 0), ("yes", 1), ("off", 0)])
def test_binary_intervention_normalizes_registered_values(value, expected):
    assert binary_intervention(example("r", "x", value).record) == (0, expected)


def test_l1_pairing_synthesizes_missing_opposite_value_without_task_supervision():
    paired = L1PairedDataset(BenchmarkDataset([example("x0", "x", 0), example("y0", "y", 0)]))
    opposite = paired[0].opposite
    assert opposite.record["intervention"]["target"] == "x"
    assert opposite.record["intervention"]["value"] == 1
    assert opposite.record["provenance"]["l1_task_supervision"] is False


def test_l1_pairing_synthesizes_law_only_different_target_when_family_has_one_observed_target():
    paired = L1PairedDataset(BenchmarkDataset([
        example("x0", "x", 0), example("x1", "x", 1),
    ]))
    synthetic = paired[0].commuting
    assert synthetic.record["intervention"]["target"] == "y"
    assert synthetic.record["intervention"]["text"] == "Set y to 0."
    assert synthetic.record["provenance"]["l1_task_supervision"] is False


def test_l1_collator_preserves_alignment_of_all_three_paths():
    rows = [example("x0", "x", 0), example("x1", "x", 1), example("y0", "y", 0), example("y1", "y", 1)]
    paired = L1PairedDataset(BenchmarkDataset(rows))
    collate = L1PairedCollator(lambda items: {"ids": [item.record_id for item in items]})
    batch = collate([paired[0], paired[3]])
    assert batch == {
        "base": {"ids": ["x0", "y1"]},
        "commuting": {"ids": ["y0", "x0"]},
        "opposite": {"ids": ["x1", "y0"]},
    }


def test_l1_collator_uses_unaugmented_law_path():
    rows = [example("x0", "x", 0), example("x1", "x", 1)]
    paired = L1PairedDataset(BenchmarkDataset(rows))
    collate = L1PairedCollator(
        lambda items: {"kind": "task", "ids": [item.record_id for item in items]},
        lambda items: {"kind": "law", "ids": [item.record_id for item in items]},
    )
    batch = collate([paired[0]])
    assert batch["base"]["kind"] == "task"
    assert batch["commuting"]["kind"] == "law"
    assert batch["opposite"]["kind"] == "law"
