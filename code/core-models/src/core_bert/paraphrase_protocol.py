"""Leakage-safe preparation for unseen-instruction-paraphrase evaluation.

This module partitions an explicitly supplied paraphrase catalog.  It does not
generate paraphrases, open the held-out benchmark test split, or produce A1
evidence.  Callers may use the resulting train/validation banks to replace only
``intervention.text`` while leaving the benchmark record and its split intact.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Mapping, Sequence


PROTOCOL_VERSION = "unseen_instruction_paraphrase_v1"
PREPARATION_STATUS = "protocol_preparation_only_not_a1_evidence"
DEVELOPMENT_SPLITS = frozenset({"train", "validation"})


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_paraphrase(text: str) -> str:
    """Return the surface-family key used for leakage checks."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError("paraphrase text must be a non-empty string")
    normalized = unicodedata.normalize("NFKC", text).casefold().strip()
    return re.sub(r"\s+", " ", normalized)


@dataclass(frozen=True)
class ParaphraseSpec:
    """One candidate wording and its externally supplied provenance."""

    operator_key: str
    intervention_key: str
    text: str
    provenance: Mapping[str, Any]
    wording_family_key: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.operator_key, str) or not self.operator_key:
            raise ValueError("operator_key must be a non-empty string")
        if not isinstance(self.intervention_key, str) or not self.intervention_key:
            raise ValueError("intervention_key must be a non-empty string")
        normalize_paraphrase(self.text)
        if not isinstance(self.provenance, Mapping):
            raise ValueError("provenance must be a mapping")
        if self.wording_family_key is not None and (
            not isinstance(self.wording_family_key, str) or not self.wording_family_key
        ):
            raise ValueError("wording_family_key must be a non-empty string when supplied")


@dataclass(frozen=True)
class ParaphraseAssignment:
    operator_key: str
    intervention_key: str
    split: str
    text: str
    normalized_sha256: str
    provenance: Mapping[str, Any]
    wording_family_key: str


@dataclass(frozen=True)
class ParaphrasePartition:
    """Immutable train/validation wording banks plus reproducibility metadata."""

    assignments: tuple[ParaphraseAssignment, ...]
    seed: int
    validation_fraction: float
    catalog_sha256: str
    protocol_version: str = PROTOCOL_VERSION
    evidence_status: str = PREPARATION_STATUS

    @cached_property
    def _bank_index(
        self,
    ) -> dict[tuple[str, str, str], tuple[ParaphraseAssignment, ...]]:
        grouped: dict[tuple[str, str, str], list[ParaphraseAssignment]] = {}
        for item in self.assignments:
            grouped.setdefault(
                (item.operator_key, item.intervention_key, item.split), []
            ).append(item)
        return {key: tuple(values) for key, values in grouped.items()}

    def texts_for(
        self, operator_key: str, intervention_key: str, split: str
    ) -> tuple[ParaphraseAssignment, ...]:
        _require_development_split(split)
        matches = self._bank_index.get(
            (operator_key, intervention_key, split), ()
        )
        if not matches:
            raise KeyError(
                f"no {split} paraphrases for operator={operator_key!r}, "
                f"intervention={intervention_key!r}"
            )
        return matches

    def select_for_record(
        self,
        record_id: str,
        operator_key: str,
        intervention_key: str,
        split: str,
    ) -> ParaphraseAssignment:
        """Select a stable wording without depending on process hash order."""

        if not isinstance(record_id, str) or not record_id:
            raise ValueError("record_id must be a non-empty string")
        choices = self.texts_for(operator_key, intervention_key, split)
        digest = hashlib.sha256(
            f"{self.protocol_version}:{self.seed}:{split}:{record_id}".encode("utf-8")
        ).digest()
        return choices[int.from_bytes(digest[:8], "big") % len(choices)]

    def to_manifest(self) -> dict[str, Any]:
        """Return JSON-serializable provenance for a run artifact."""

        counts = {"train": 0, "validation": 0}
        families: set[tuple[str, str]] = set()
        values: list[dict[str, Any]] = []
        for item in self.assignments:
            counts[item.split] += 1
            families.add((item.operator_key, item.intervention_key))
            values.append(
                {
                    "operator_key": item.operator_key,
                    "intervention_key": item.intervention_key,
                    "split": item.split,
                    "text": item.text,
                    "normalized_sha256": item.normalized_sha256,
                    "wording_family_key": item.wording_family_key,
                    "source_provenance": dict(item.provenance),
                }
            )
        return {
            "protocol_version": self.protocol_version,
            "evidence_status": self.evidence_status,
            "seed": self.seed,
            "validation_fraction": self.validation_fraction,
            "catalog_sha256": self.catalog_sha256,
            "family_count": len(families),
            "counts": counts,
            "assignments": values,
        }


def _require_development_split(split: str) -> None:
    if split == "test":
        raise PermissionError(
            "held-out test is outside paraphrase preparation; use it only after "
            "validation selection under the final-evaluation gate"
        )
    if split not in DEVELOPMENT_SPLITS:
        raise ValueError(f"unsupported paraphrase split {split!r}")


def intervention_keys(record: Mapping[str, Any]) -> tuple[str, str]:
    """Derive wording-independent operator and semantic-intervention keys."""

    intervention = record.get("intervention")
    if not isinstance(intervention, Mapping):
        raise ValueError("record.intervention must be an object")
    kind = intervention.get("kind")
    if not isinstance(kind, str) or not kind:
        raise ValueError("intervention.kind must be a non-empty operator key")
    target = intervention.get("target")
    if not isinstance(target, str) or not target:
        raise ValueError("intervention.target must be non-empty")
    value = intervention.get("value_token")
    if value is None:
        value = intervention.get("value")
    if value is None or isinstance(value, (dict, list)):
        raise ValueError("intervention must expose a scalar value/value_token")
    signature = {
        "source": record.get("source"),
        "structure_kind": record.get("structure_kind"),
        "operator": kind,
        "target": target,
        "target_text": intervention.get("target_text"),
        "value": value,
        "formal": intervention.get("formal"),
    }
    return kind, f"intervention:{_sha256(_stable_json(signature))[:20]}"


def paraphrase_spec_from_record(
    record: Mapping[str, Any], *, provenance: Mapping[str, Any] | None = None
) -> ParaphraseSpec:
    """Adapt a record or generated record variant into a catalog entry."""

    operator_key, intervention_key = intervention_keys(record)
    intervention = record["intervention"]
    text = intervention.get("text")
    if not isinstance(text, str):
        raise ValueError("intervention.text must be a string")
    source_provenance = provenance if provenance is not None else record.get("provenance", {})
    if not isinstance(source_provenance, Mapping):
        raise ValueError("record provenance must be a mapping")
    return ParaphraseSpec(operator_key, intervention_key, text, source_provenance)


def build_paraphrase_partition(
    specs: Sequence[ParaphraseSpec],
    *,
    seed: int = 0,
    validation_fraction: float = 0.25,
) -> ParaphrasePartition:
    """Partition normalized wording families within each intervention/operator.

    All spelling/case/whitespace-equivalent candidates are kept in one unit.
    Every intervention/operator family must provide at least two distinct
    normalized wordings so each development split receives at least one.
    """

    if not specs:
        raise ValueError("paraphrase catalog cannot be empty")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie strictly between zero and one")

    grouped: dict[tuple[str, str], dict[str, ParaphraseSpec]] = {}
    for spec in specs:
        family = (spec.operator_key, spec.intervention_key)
        normalized = normalize_paraphrase(spec.text)
        existing = grouped.setdefault(family, {}).get(normalized)
        if existing is not None:
            if _stable_json(dict(existing.provenance)) != _stable_json(dict(spec.provenance)):
                raise ValueError(
                    "duplicate normalized paraphrase has conflicting provenance for "
                    f"operator={spec.operator_key!r}, intervention={spec.intervention_key!r}"
                )
            continue
        grouped[family][normalized] = spec

    # A structured wording/template is one leakage unit across every semantic
    # intervention using the same operator.  Unstructured catalogs fall back
    # to their normalized surface form, which is still safe but less powerful.
    operator_units: dict[str, set[str]] = {}
    family_units: dict[tuple[str, str], dict[str, str]] = {}
    for family, candidates in grouped.items():
        units: dict[str, str] = {}
        for normalized, spec in candidates.items():
            unit = spec.wording_family_key or f"surface:{normalized}"
            if unit in units:
                raise ValueError(
                    "one intervention has multiple wordings in the same leakage unit: "
                    f"operator={family[0]!r}, intervention={family[1]!r}, unit={unit!r}"
                )
            units[unit] = normalized
            operator_units.setdefault(family[0], set()).add(unit)
        family_units[family] = units

    unit_splits: dict[tuple[str, str], str] = {}
    for operator, units in sorted(operator_units.items()):
        if len(units) < 2:
            raise ValueError(
                f"operator {operator!r} needs at least two distinct wording leakage units"
            )
        ranked_units = sorted(
            units,
            key=lambda unit: (
                _sha256(f"{PROTOCOL_VERSION}:{seed}:{operator}:{unit}"), unit
            ),
        )
        validation_count = min(
            len(ranked_units) - 1,
            max(1, int(math.floor(len(ranked_units) * validation_fraction + 0.5))),
        )
        validation_units = set(ranked_units[:validation_count])
        for unit in ranked_units:
            unit_splits[(operator, unit)] = (
                "validation" if unit in validation_units else "train"
            )

    assignments: list[ParaphraseAssignment] = []
    catalog_rows: list[dict[str, Any]] = []
    for family in sorted(grouped):
        candidates = grouped[family]
        if len(candidates) < 2:
            raise ValueError(
                "unseen-paraphrase family needs at least two distinct normalized "
                f"wordings: operator={family[0]!r}, intervention={family[1]!r}"
            )
        ranked = sorted(candidates)
        splits_present = {
            unit_splits[(family[0], unit)] for unit in family_units[family]
        }
        if splits_present != DEVELOPMENT_SPLITS:
            raise ValueError(
                "wording leakage-unit partition leaves an intervention without both "
                f"train and validation forms: operator={family[0]!r}, "
                f"intervention={family[1]!r}"
            )
        normalized_to_unit = {
            normalized: unit for unit, normalized in family_units[family].items()
        }
        for normalized in ranked:
            spec = candidates[normalized]
            unit = normalized_to_unit[normalized]
            split = unit_splits[(family[0], unit)]
            assignment = ParaphraseAssignment(
                family[0], family[1], split, spec.text, _sha256(normalized),
                dict(spec.provenance), unit,
            )
            assignments.append(assignment)
            catalog_rows.append(
                {
                    "operator_key": family[0],
                    "intervention_key": family[1],
                    "normalized": normalized,
                    "wording_family_key": unit,
                    "provenance": dict(spec.provenance),
                }
            )

    assignments.sort(
        key=lambda item: (
            item.operator_key, item.intervention_key, item.split,
            item.normalized_sha256,
        )
    )
    return ParaphrasePartition(
        tuple(assignments), seed, validation_fraction,
        _sha256(_stable_json(catalog_rows)),
    )


def apply_paraphrase(
    record: Mapping[str, Any],
    partition: ParaphrasePartition,
    *,
    split: str,
) -> dict[str, Any]:
    """Return a copied record with a split-appropriate instruction and audit hook."""

    _require_development_split(split)
    record_id = record.get("id")
    if not isinstance(record_id, str) or not record_id:
        raise ValueError("record.id must be a non-empty string")
    operator_key, intervention_key = intervention_keys(record)
    selected = partition.select_for_record(
        record_id, operator_key, intervention_key, split
    )
    prepared = copy.deepcopy(dict(record))
    intervention = prepared.get("intervention")
    if not isinstance(intervention, dict):
        raise ValueError("record.intervention must be a mutable object after copying")
    original_text = intervention.get("text")
    if not isinstance(original_text, str) or not original_text:
        raise ValueError("intervention.text must be non-empty")
    intervention["text"] = selected.text
    protocol = prepared.setdefault("protocol_provenance", {})
    if not isinstance(protocol, dict):
        raise ValueError("record.protocol_provenance must be an object when present")
    protocol["instruction_paraphrase"] = {
        "protocol_version": partition.protocol_version,
        "evidence_status": partition.evidence_status,
        "split": split,
        "seed": partition.seed,
        "catalog_sha256": partition.catalog_sha256,
        "operator_key": operator_key,
        "intervention_key": intervention_key,
        "original_text_sha256": _sha256(normalize_paraphrase(original_text)),
        "selected_text_sha256": selected.normalized_sha256,
        "source_provenance": dict(selected.provenance),
    }
    return prepared


def _catalog_digest(assignments: Sequence[ParaphraseAssignment]) -> str:
    rows = sorted(
        (
            {
                "operator_key": item.operator_key,
                "intervention_key": item.intervention_key,
                "normalized": normalize_paraphrase(item.text),
                "wording_family_key": item.wording_family_key,
                "provenance": dict(item.provenance),
            }
            for item in assignments
        ),
        key=lambda row: (
            row["operator_key"], row["intervention_key"], row["normalized"]
        ),
    )
    return _sha256(_stable_json(rows))


def load_paraphrase_catalog(
    path: Path,
) -> tuple[ParaphrasePartition, dict[str, Any]]:
    """Load and fully validate a generated partition manifest."""

    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("paraphrase catalog must be a JSON object")
    if document.get("artifact") != "structured_instruction_paraphrase_catalog":
        raise ValueError("paraphrase catalog has an unsupported artifact type")
    input_provenance = document.get("input_provenance")
    if not isinstance(input_provenance, dict):
        raise ValueError("paraphrase catalog must carry input provenance")
    if input_provenance.get("evidence_status") != PREPARATION_STATUS:
        raise ValueError("catalog provenance must not claim A1 evidence")
    raw = document.get("partition")
    if not isinstance(raw, dict):
        raise ValueError("paraphrase catalog partition must be an object")
    if raw.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("unsupported paraphrase protocol version")
    if raw.get("evidence_status") != PREPARATION_STATUS:
        raise ValueError("catalog must remain protocol preparation, not A1 evidence")
    seed = raw.get("seed")
    validation_fraction = raw.get("validation_fraction")
    catalog_sha256 = raw.get("catalog_sha256")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("catalog seed must be an integer")
    if not isinstance(validation_fraction, (int, float)) or not 0 < validation_fraction < 1:
        raise ValueError("catalog validation_fraction is invalid")
    if not isinstance(catalog_sha256, str) or len(catalog_sha256) != 64:
        raise ValueError("catalog_sha256 must be a full SHA-256 hex digest")
    values = raw.get("assignments")
    if not isinstance(values, list) or not values:
        raise ValueError("catalog assignments must be a non-empty list")

    assignments: list[ParaphraseAssignment] = []
    for index, value in enumerate(values):
        if not isinstance(value, dict):
            raise ValueError(f"catalog assignment {index} must be an object")
        split = value.get("split")
        _require_development_split(split)
        operator = value.get("operator_key")
        intervention = value.get("intervention_key")
        text = value.get("text")
        digest = value.get("normalized_sha256")
        unit = value.get("wording_family_key")
        provenance = value.get("source_provenance")
        if not all(isinstance(item, str) and item for item in (operator, intervention, unit)):
            raise ValueError(f"catalog assignment {index} has invalid family keys")
        if not isinstance(text, str) or _sha256(normalize_paraphrase(text)) != digest:
            raise ValueError(f"catalog assignment {index} has a text hash mismatch")
        if not isinstance(provenance, dict):
            raise ValueError(f"catalog assignment {index} provenance must be an object")
        assignments.append(
            ParaphraseAssignment(
                operator, intervention, split, text, digest, provenance, unit
            )
        )

    partition = ParaphrasePartition(
        tuple(assignments), seed, float(validation_fraction), catalog_sha256
    )
    if _catalog_digest(partition.assignments) != catalog_sha256:
        raise ValueError("catalog content does not match catalog_sha256")
    unit_splits: dict[tuple[str, str], set[str]] = {}
    family_splits: dict[tuple[str, str], set[str]] = {}
    for item in assignments:
        unit_splits.setdefault(
            (item.operator_key, item.wording_family_key), set()
        ).add(item.split)
        family_splits.setdefault(
            (item.operator_key, item.intervention_key), set()
        ).add(item.split)
    if any(len(splits) != 1 for splits in unit_splits.values()):
        raise ValueError("wording leakage unit appears in both train and validation")
    if any(splits != DEVELOPMENT_SPLITS for splits in family_splits.values()):
        raise ValueError("semantic intervention is missing a train or validation bank")
    expected_counts = raw.get("counts")
    observed_counts = {
        split: sum(item.split == split for item in assignments)
        for split in sorted(DEVELOPMENT_SPLITS)
    }
    if expected_counts != observed_counts:
        raise ValueError("catalog assignment counts do not match manifest")
    if raw.get("family_count") != len(family_splits):
        raise ValueError("catalog family_count does not match assignments")
    metadata = {
        "artifact": document.get("artifact"),
        "input_provenance": input_provenance,
        "protocol_version": partition.protocol_version,
        "evidence_status": partition.evidence_status,
        "seed": partition.seed,
        "validation_fraction": partition.validation_fraction,
        "catalog_sha256": partition.catalog_sha256,
        "family_count": len(family_splits),
        "counts": observed_counts,
    }
    return partition, metadata


def paraphrase_coverage(
    records: Sequence[Mapping[str, Any]],
    partition: ParaphrasePartition,
    *,
    split: str,
) -> dict[str, Any]:
    """Validate complete bank coverage and summarize deterministic selections."""

    _require_development_split(split)
    selected_hashes: set[str] = set()
    selected_units: set[str] = set()
    operators: dict[str, int] = {}
    for record in records:
        record_id = record.get("id")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError("record.id must be a non-empty string")
        operator, intervention = intervention_keys(record)
        try:
            selected = partition.select_for_record(
                record_id, operator, intervention, split
            )
        except KeyError as error:
            raise ValueError(
                f"paraphrase catalog has no {split} bank for record {record_id!r}"
            ) from error
        selected_hashes.add(selected.normalized_sha256)
        selected_units.add(selected.wording_family_key)
        operators[operator] = operators.get(operator, 0) + 1
    return {
        "eligible_count": len(records),
        "covered_count": len(records),
        "coverage": 1.0 if records else 0.0,
        "selected_wording_count": len(selected_hashes),
        "selected_wording_family_count": len(selected_units),
        "records_by_operator": dict(sorted(operators.items())),
    }


__all__ = [
    "DEVELOPMENT_SPLITS",
    "PREPARATION_STATUS",
    "PROTOCOL_VERSION",
    "ParaphraseAssignment",
    "ParaphrasePartition",
    "ParaphraseSpec",
    "apply_paraphrase",
    "build_paraphrase_partition",
    "intervention_keys",
    "load_paraphrase_catalog",
    "normalize_paraphrase",
    "paraphrase_coverage",
    "paraphrase_spec_from_record",
]
