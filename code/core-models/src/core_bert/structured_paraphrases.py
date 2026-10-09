"""Structured, explicit-field-only instruction paraphrase catalog generation."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from .paraphrase_protocol import (
    PREPARATION_STATUS,
    ParaphraseAssignment,
    ParaphrasePartition,
    ParaphraseSpec,
    apply_paraphrase,
    build_paraphrase_partition,
    intervention_keys,
    normalize_paraphrase,
)


GENERATOR = "core_bert.structured_paraphrases"
GENERATOR_VERSION = 1


TEMPLATES = {
    "value_set": (
        ("value_set_direct", 'Set the variable {target} to {value}.'),
        ("value_set_intervene", 'Intervene on {target} so its value becomes {value}.'),
        ("value_set_assign", 'Apply an intervention assigning {value} to {target}.'),
        ("value_set_suppose", 'Suppose the value of {target} is changed to {value}.'),
    ),
    "event_replace": (
        ("event_replace_direct", 'Replace the event {target} with {value}.'),
        ("event_replace_substitute", 'In the event sequence, substitute {value} for {target}.'),
        ("event_replace_intervene", 'Intervene by changing the event {target} to {value}.'),
        ("event_replace_counterfactual", 'Consider the sequence with {value} instead of {target}.'),
    ),
}


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _quoted(value: Any) -> str:
    """Quote a field deterministically so text boundaries remain unambiguous."""

    if not isinstance(value, (str, int, float, bool)) or (
        isinstance(value, str) and not value
    ):
        raise ValueError("template fields must be non-empty JSON scalars")
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _development_split(split: str) -> None:
    if split == "test":
        raise PermissionError("held-out test cannot be used to build a paraphrase catalog")
    if split not in {"train", "validation"}:
        raise ValueError(f"unsupported catalog source split {split!r}")


def structured_specs_for_record(
    record: Mapping[str, Any], *, source_split: str
) -> tuple[ParaphraseSpec, ...]:
    """Instantiate registered templates using intervention fields only."""

    _development_split(source_split)
    record_id = record.get("id")
    if not isinstance(record_id, str) or not record_id:
        raise ValueError("record.id must be a non-empty string")
    intervention = record.get("intervention")
    if not isinstance(intervention, Mapping):
        raise ValueError("record.intervention must be an object")
    operator_key, intervention_key = intervention_keys(record)
    if operator_key not in TEMPLATES:
        raise ValueError(f"unsupported structured paraphrase operator {operator_key!r}")

    target = intervention.get("target_text") or intervention.get("target")
    value = intervention.get("value")
    if not isinstance(target, str) or not target:
        raise ValueError("intervention target/target_text must be non-empty")
    if not isinstance(value, (str, int, float, bool)) or (
        isinstance(value, str) and not value
    ):
        raise ValueError("intervention.value must be a non-empty JSON scalar")
    fields = {"target": _quoted(target), "value": _quoted(value)}
    explicit_fields_sha256 = _digest(
        {
            "kind": operator_key,
            "target": intervention.get("target"),
            "target_text": intervention.get("target_text"),
            "value": value,
            "value_token": intervention.get("value_token"),
        }
    )
    return tuple(
        ParaphraseSpec(
            operator_key=operator_key,
            intervention_key=intervention_key,
            text=template.format(**fields),
            provenance={
                "generator": GENERATOR,
                "generator_version": GENERATOR_VERSION,
                "template_id": template_id,
                "construction": "explicit_intervention_fields_only",
                "explicit_fields_sha256": explicit_fields_sha256,
                "source_split": source_split,
                "source_record_id_sha256": _digest(record_id),
            },
            wording_family_key=template_id,
        )
        for template_id, template in TEMPLATES[operator_key]
    )


def _materialized_assignment(
    record: Mapping[str, Any],
    partition: ParaphrasePartition,
    *,
    split: str,
) -> ParaphraseAssignment:
    """Instantiate a frozen template unit for a post-catalog component record."""

    specs = structured_specs_for_record(record, source_split=split)
    operator_key, intervention_key = intervention_keys(record)
    unit_splits: dict[str, str] = {}
    for item in partition.assignments:
        if item.operator_key != operator_key:
            continue
        existing = unit_splits.setdefault(item.wording_family_key, item.split)
        if existing != item.split:
            raise ValueError("frozen paraphrase template unit crosses development splits")
    expected_units = {template_id for template_id, _ in TEMPLATES[operator_key]}
    if set(unit_splits) != expected_units:
        raise ValueError(
            f"frozen paraphrase catalog lacks the complete template registry for {operator_key!r}"
        )
    candidates = sorted(
        (spec for spec in specs if unit_splits[spec.wording_family_key] == split),
        key=lambda spec: (normalize_paraphrase(spec.text), spec.wording_family_key),
    )
    if not candidates:
        raise ValueError(f"frozen paraphrase catalog has no {split} template unit")
    record_id = record["id"]
    digest = hashlib.sha256(
        f"{partition.protocol_version}:{partition.seed}:{split}:{record_id}".encode("utf-8")
    ).digest()
    selected = candidates[int.from_bytes(digest[:8], "big") % len(candidates)]
    provenance = dict(selected.provenance)
    provenance.update(
        {
            "materialized_from_frozen_template_unit": True,
            "base_catalog_sha256": partition.catalog_sha256,
        }
    )
    normalized = normalize_paraphrase(selected.text)
    return ParaphraseAssignment(
        operator_key,
        intervention_key,
        split,
        selected.text,
        hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
        provenance,
        selected.wording_family_key,
    )


def apply_structured_paraphrase(
    record: Mapping[str, Any],
    partition: ParaphrasePartition,
    *,
    split: str,
    allow_materialized: bool = False,
) -> tuple[dict[str, Any], bool, ParaphraseAssignment]:
    """Apply a catalog wording, optionally materializing a frozen template unit."""

    operator_key, intervention_key = intervention_keys(record)
    try:
        selected = partition.select_for_record(
            record["id"], operator_key, intervention_key, split
        )
        return apply_paraphrase(record, partition, split=split), False, selected
    except KeyError:
        if not allow_materialized:
            raise
    selected = _materialized_assignment(record, partition, split=split)
    extended = ParaphrasePartition(
        partition.assignments + (selected,),
        partition.seed,
        partition.validation_fraction,
        partition.catalog_sha256,
        partition.protocol_version,
        partition.evidence_status,
    )
    return apply_paraphrase(record, extended, split=split), True, selected


def structured_paraphrase_coverage(
    records: Sequence[Mapping[str, Any]],
    partition: ParaphrasePartition,
    *,
    split: str,
    allow_materialized: bool = False,
) -> dict[str, Any]:
    """Validate coverage and bind deterministic materialized selections."""

    selections = []
    operators: dict[str, int] = {}
    selected_units: set[str] = set()
    selected_hashes: set[str] = set()
    for record in records:
        _, materialized, selected = apply_structured_paraphrase(
            record, partition, split=split, allow_materialized=allow_materialized
        )
        operator_key, intervention_key = intervention_keys(record)
        operators[operator_key] = operators.get(operator_key, 0) + 1
        selected_units.add(selected.wording_family_key)
        selected_hashes.add(selected.normalized_sha256)
        if materialized:
            selections.append(
                {
                    "record_id_sha256": _digest(record["id"]),
                    "intervention_key": intervention_key,
                    "wording_family_key": selected.wording_family_key,
                    "selected_text_sha256": selected.normalized_sha256,
                }
            )
    return {
        "eligible_count": len(records),
        "covered_count": len(records),
        "coverage": 1.0 if records else 0.0,
        "selected_wording_count": len(selected_hashes),
        "selected_wording_family_count": len(selected_units),
        "records_by_operator": dict(sorted(operators.items())),
        "materialized_count": len(selections),
        "materialized_selections": sorted(
            selections, key=lambda row: row["record_id_sha256"]
        ),
    }


def build_structured_partition(
    split_records: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    seed: int = 0,
    validation_fraction: float = 0.25,
) -> tuple[ParaphrasePartition, dict[str, Any]]:
    """Build a partition and its input provenance from development records only."""

    if not split_records:
        raise ValueError("split_records cannot be empty")
    gathered: dict[tuple[str, str], list[tuple[str, Mapping[str, Any]]]] = defaultdict(list)
    split_ids: dict[str, list[str]] = defaultdict(list)
    for split, records in split_records.items():
        _development_split(split)
        for record in records:
            record_id = record.get("id")
            if not isinstance(record_id, str) or not record_id:
                raise ValueError("record.id must be a non-empty string")
            keys = intervention_keys(record)
            gathered[keys].append((split, record))
            split_ids[split].append(record_id)

    specs: list[ParaphraseSpec] = []
    for family, members in sorted(gathered.items()):
        # Repeated records for one semantic intervention must instantiate the
        # exact same templates; otherwise the catalog is semantically ambiguous.
        generated = [structured_specs_for_record(record, source_split=split) for split, record in members]
        surfaces = [tuple(item.text for item in group) for group in generated]
        if any(surface != surfaces[0] for surface in surfaces[1:]):
            raise ValueError(
                "records sharing an intervention key disagree on explicit semantic fields: "
                f"operator={family[0]!r}, intervention={family[1]!r}"
            )
        source_record_hashes = sorted(_digest(record["id"]) for _, record in members)
        source_splits = sorted({split for split, _ in members})
        for item in generated[0]:
            provenance = dict(item.provenance)
            provenance.pop("source_split", None)
            provenance.pop("source_record_id_sha256", None)
            provenance.update(
                {
                    "source_splits": source_splits,
                    "source_record_count": len(members),
                    "source_record_ids_sha256": _digest(source_record_hashes),
                }
            )
            specs.append(
                ParaphraseSpec(
                    item.operator_key,
                    item.intervention_key,
                    item.text,
                    provenance,
                    item.wording_family_key,
                )
            )

    partition = build_paraphrase_partition(
        specs, seed=seed, validation_fraction=validation_fraction
    )
    input_provenance = {
        "generator": GENERATOR,
        "generator_version": GENERATOR_VERSION,
        "evidence_status": PREPARATION_STATUS,
        "source_splits": {
            split: {
                "record_count": len(ids),
                "record_ids_sha256": _digest(sorted(ids)),
            }
            for split, ids in sorted(split_ids.items())
        },
        "template_registry_sha256": _digest(TEMPLATES),
    }
    return partition, input_provenance


def structured_catalog_manifest(
    partition: ParaphrasePartition, input_provenance: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "artifact": "structured_instruction_paraphrase_catalog",
        "input_provenance": dict(input_provenance),
        "partition": partition.to_manifest(),
    }


__all__ = [
    "GENERATOR",
    "GENERATOR_VERSION",
    "TEMPLATES",
    "apply_structured_paraphrase",
    "build_structured_partition",
    "structured_paraphrase_coverage",
    "structured_catalog_manifest",
    "structured_specs_for_record",
]
