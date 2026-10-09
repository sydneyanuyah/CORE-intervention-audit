from __future__ import annotations

from train_f2_real_baselines import combined_records


class Bundle:
    examples = [type("E", (), {"pair_id": "p", "first_record_id": "a", "second_record_id": "b"})()]
    records = {
        "a": {"intervention": {"text": "set A"}},
        "b": {"id": "b", "intervention": {"text": "set B"}},
    }


def test_combined_prompt_preserves_declared_order_without_mutation() -> None:
    rows = combined_records(Bundle())
    assert rows[0]["id"] == "p:combined-prompt"
    assert rows[0]["intervention"]["text"] == "set A Then set B"
    assert Bundle.records["b"]["intervention"]["text"] == "set B"
