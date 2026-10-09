"""Fail-closed ordered two-edit protocol data and balanced metrics.

This module is protocol infrastructure, not an A1 result.  It requires final
two-edit ground truth explicitly; it never invents composition by combining
two single-edit labels.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def ordered_pair_id(
    source: str,
    graph_group_id: str,
    world_group_id: str,
    first_record_id: str,
    second_record_id: str,
) -> str:
    """Return a deterministic, order-sensitive identifier for a two-edit row."""

    values = (source, graph_group_id, world_group_id, first_record_id, second_record_id)
    if not all(isinstance(value, str) and value for value in values):
        raise ValueError("ordered pair identity fields must be non-empty strings")
    digest = hashlib.sha256(_stable_json(values).encode("utf-8")).hexdigest()[:24]
    return f"two-edit:{digest}"


@dataclass(frozen=True)
class OrderedTwoEditExample:
    pair_id: str
    source: str
    graph_group_id: str
    world_group_id: str
    first_record_id: str
    second_record_id: str
    first_intervention: Mapping[str, Any]
    second_intervention: Mapping[str, Any]
    gold_outputs: tuple[Any, ...]
    factual_outputs: tuple[Any, ...]
    change_mask: tuple[bool, ...]
    preservation_mask: tuple[bool, ...]
    target_mask: tuple[bool, ...]


def _bool_mask(value: Any, length: int, field: str) -> tuple[bool, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{field} must be a boolean list of length {length}")
    if not all(isinstance(item, bool) for item in value):
        raise ValueError(f"{field} must contain only booleans")
    return tuple(value)


def _same_factual(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    return _stable_json(first.get("factual")) == _stable_json(second.get("factual"))


def build_ordered_two_edit_examples(
    records: Mapping[str, Mapping[str, Any]],
    groups: Mapping[str, tuple[str, str]],
    manifest_rows: Sequence[Mapping[str, Any]],
    *,
    seen_separately: Set[str],
) -> tuple[OrderedTwoEditExample, ...]:
    """Validate and construct ordered pairs from explicit final-state truth.

    ``seen_separately`` is the set of single-edit record IDs available during
    fitting. Both component edits must be present there, while the ordered pair
    itself remains an unseen composition.
    """

    if not manifest_rows:
        raise ValueError("two-edit manifest must contain at least one row")
    examples = []
    seen_pairs: set[str] = set()
    for row_number, row in enumerate(manifest_rows, 1):
        if not isinstance(row, Mapping):
            raise ValueError(f"row {row_number}: manifest row must be an object")
        first_id = row.get("first_record_id")
        second_id = row.get("second_record_id")
        if not isinstance(first_id, str) or not isinstance(second_id, str):
            raise ValueError(f"row {row_number}: component record IDs must be strings")
        if first_id == second_id:
            raise ValueError(f"row {row_number}: component records must be distinct")
        if first_id not in records or second_id not in records:
            raise ValueError(f"row {row_number}: unknown component record ID")
        if first_id not in seen_separately or second_id not in seen_separately:
            raise ValueError(f"row {row_number}: both edits must be seen separately in training")
        if first_id not in groups or second_id not in groups:
            raise ValueError(f"row {row_number}: both component records require group IDs")
        first, second = records[first_id], records[second_id]
        source = first.get("source")
        if not isinstance(source, str) or not source or second.get("source") != source:
            raise ValueError(f"row {row_number}: component sources must match")
        if groups[first_id] != groups[second_id]:
            raise ValueError(f"row {row_number}: components must share graph and world groups")
        if not _same_factual(first, second):
            raise ValueError(f"row {row_number}: components must share an identical factual world")
        first_intervention = first.get("intervention")
        second_intervention = second.get("intervention")
        if not isinstance(first_intervention, Mapping) or not isinstance(second_intervention, Mapping):
            raise ValueError(f"row {row_number}: component interventions must be objects")

        gold = row.get("gold_outputs")
        factual = row.get("factual_outputs")
        if not isinstance(gold, list) or not gold or not isinstance(factual, list):
            raise ValueError(f"row {row_number}: explicit gold/factual output lists are required")
        if len(gold) != len(factual):
            raise ValueError(f"row {row_number}: gold and factual outputs must have equal length")
        change = _bool_mask(row.get("change_mask"), len(gold), "change_mask")
        preservation = _bool_mask(
            row.get("preservation_mask"), len(gold), "preservation_mask"
        )
        target = _bool_mask(row.get("target_mask"), len(gold), "target_mask")
        if not any(change) or not any(preservation) or not any(target):
            raise ValueError(
                f"row {row_number}: change, preservation, and target masks must each be non-empty"
            )
        if any(a and b for a, b in zip(change, preservation)):
            raise ValueError(f"row {row_number}: change and preservation masks must be disjoint")
        if any(flag != (after != before) for flag, after, before in zip(change, gold, factual)):
            raise ValueError(f"row {row_number}: change_mask must equal gold != factual")
        if any(
            flag != (after == before)
            for flag, after, before in zip(preservation, gold, factual)
        ):
            raise ValueError(
                f"row {row_number}: preservation_mask must equal gold == factual"
            )

        graph_group, world_group = groups[first_id]
        canonical_id = ordered_pair_id(source, graph_group, world_group, first_id, second_id)
        provided_id = row.get("pair_id", canonical_id)
        if provided_id != canonical_id:
            raise ValueError(f"row {row_number}: pair_id does not match canonical ordered identity")
        if canonical_id in seen_pairs:
            raise ValueError(f"row {row_number}: duplicate ordered pair")
        seen_pairs.add(canonical_id)
        examples.append(
            OrderedTwoEditExample(
                canonical_id,
                source,
                graph_group,
                world_group,
                first_id,
                second_id,
                dict(first_intervention),
                dict(second_intervention),
                tuple(gold),
                tuple(factual),
                change,
                preservation,
                target,
            )
        )
    return tuple(sorted(examples, key=lambda example: example.pair_id))


def load_ordered_two_edit_manifest(
    path: Path,
    records: Mapping[str, Mapping[str, Any]],
    groups: Mapping[str, tuple[str, str]],
    *,
    seen_separately: Set[str],
) -> tuple[OrderedTwoEditExample, ...]:
    """Load a JSONL manifest and apply all ordered-pair validations."""

    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise ValueError(f"{path}:{line_number}: manifest row must be an object")
        rows.append(value)
    return build_ordered_two_edit_examples(
        records, groups, rows, seen_separately=seen_separately
    )


def _masked_accuracy(
    predicted: Sequence[Any], gold: Sequence[Any], mask: Sequence[bool]
) -> tuple[int, int]:
    total = sum(mask)
    correct = sum(p == g for p, g, keep in zip(predicted, gold, mask) if keep)
    return correct, total


def _aggregate_counts(rows: Sequence[dict[str, int]]) -> dict[str, float | int]:
    change_correct = sum(row["change_correct"] for row in rows)
    change_total = sum(row["change_total"] for row in rows)
    preservation_correct = sum(row["preservation_correct"] for row in rows)
    preservation_total = sum(row["preservation_total"] for row in rows)
    target_correct = sum(row["target_correct"] for row in rows)
    target_total = sum(row["target_total"] for row in rows)
    all_correct = sum(row["all_correct"] for row in rows)
    all_total = sum(row["all_total"] for row in rows)
    if not change_total or not preservation_total or not target_total:
        raise ValueError("balanced aggregation requires change, preservation, and target cells")
    change_accuracy = change_correct / change_total
    preservation_accuracy = preservation_correct / preservation_total
    return {
        "two_edit_change_accuracy": change_accuracy,
        "two_edit_preservation": preservation_accuracy,
        "two_edit_balanced": 0.5 * (change_accuracy + preservation_accuracy),
        "target_success": target_correct / target_total,
        "overall_accuracy": all_correct / all_total,
        "change_cells": change_total,
        "preservation_cells": preservation_total,
        "target_cells": target_total,
    }


def two_edit_balanced_metrics(
    examples: Sequence[OrderedTwoEditExample],
    predictions: Mapping[str, Sequence[Any]],
) -> dict[str, Any]:
    """Compute micro-balanced scores and graph-level analysis units."""

    if not examples:
        raise ValueError("two-edit metrics require at least one example")
    expected = {example.pair_id for example in examples}
    if len(expected) != len(examples):
        raise ValueError("two-edit examples must have unique pair IDs")
    if set(predictions) != expected:
        missing, extra = expected - set(predictions), set(predictions) - expected
        raise ValueError(f"prediction IDs must match exactly; missing={sorted(missing)}, extra={sorted(extra)}")
    counts = []
    graph_rows: dict[str, list[dict[str, int]]] = defaultdict(list)
    per_example = {}
    for example in examples:
        predicted = tuple(predictions[example.pair_id])
        if len(predicted) != len(example.gold_outputs):
            raise ValueError(f"{example.pair_id}: prediction length mismatch")
        change_correct, change_total = _masked_accuracy(
            predicted, example.gold_outputs, example.change_mask
        )
        preserve_correct, preserve_total = _masked_accuracy(
            predicted, example.gold_outputs, example.preservation_mask
        )
        target_correct, target_total = _masked_accuracy(
            predicted, example.gold_outputs, example.target_mask
        )
        row = {
            "change_correct": change_correct,
            "change_total": change_total,
            "preservation_correct": preserve_correct,
            "preservation_total": preserve_total,
            "target_correct": target_correct,
            "target_total": target_total,
            "all_correct": sum(a == b for a, b in zip(predicted, example.gold_outputs)),
            "all_total": len(predicted),
        }
        counts.append(row)
        graph_rows[example.graph_group_id].append(row)
        per_example[example.pair_id] = _aggregate_counts([row])
    overall = _aggregate_counts(counts)
    overall.update(
        {
            "protocol": "ordered two-edit foundation; not A1 evidence",
            "a1_evidence": False,
            "example_count": len(examples),
            "graph_count": len(graph_rows),
            "per_graph": {
                graph_id: _aggregate_counts(rows)
                for graph_id, rows in sorted(graph_rows.items())
            },
            "per_example": per_example,
        }
    )
    return overall


__all__ = [
    "OrderedTwoEditExample",
    "build_ordered_two_edit_examples",
    "load_ordered_two_edit_manifest",
    "ordered_pair_id",
    "two_edit_balanced_metrics",
]
