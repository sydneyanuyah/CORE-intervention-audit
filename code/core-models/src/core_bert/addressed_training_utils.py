"""Pure-data helpers for the scalar addressed-training pilot."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from collections.abc import Mapping
from typing import Any


def append_factual_question(record: Mapping[str, Any]) -> dict[str, Any]:
    """Append the factual question while preserving passage-relative spans.

    The passage remains an exact prefix, so all existing target/replacement
    character offsets stay valid.
    """

    result = copy.deepcopy(dict(record))
    factual = result.get("factual")
    if not isinstance(factual, dict):
        raise ValueError("record.factual must be an object")
    passage = factual.get("passage")
    if not isinstance(passage, str) or not passage:
        raise ValueError("record.factual.passage must be non-empty")
    question = factual.get("question")
    if isinstance(question, str) and question.strip():
        factual["passage"] = f"{passage}\nQuestion: {question.strip()}"
    return result


def explicit_target_slot(record: Mapping[str, Any], world_slots: int) -> int:
    """Return only a recorded slot alignment; never infer one from node order."""

    intervention = record.get("intervention")
    value = intervention.get("target_slot") if isinstance(intervention, Mapping) else None
    if isinstance(value, bool) or not isinstance(value, int):
        return -1
    return value if 0 <= value < world_slots else -1


def load_group_sidecars(groups_dir: Path) -> dict[str, tuple[str, str]]:
    """Load record→(graph, world) IDs from JSON or JSONL sidecars."""

    if not groups_dir.is_dir():
        raise ValueError(f"groups directory does not exist: {groups_dir}")
    result: dict[str, tuple[str, str]] = {}

    def add(record_id: Any, value: Any, origin: str) -> None:
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"{origin}: sidecar record ID must be non-empty")
        if isinstance(value, str):
            graph_id, world_id = value, value
        elif isinstance(value, Mapping):
            graph_id = value.get("graph_group_id") or value.get("graph_id")
            world_id = value.get("world_group_id") or value.get("world_id") or graph_id
        else:
            raise ValueError(f"{origin}: sidecar value must be a string or object")
        if not isinstance(graph_id, str) or not graph_id or not isinstance(world_id, str) or not world_id:
            raise ValueError(f"{origin}: graph/world group IDs must be non-empty strings")
        pair = (graph_id, world_id)
        if record_id in result and result[record_id] != pair:
            raise ValueError(f"{origin}: conflicting sidecar entry for {record_id!r}")
        result[record_id] = pair

    paths = sorted(groups_dir.rglob("*.json")) + sorted(groups_dir.rglob("*.jsonl"))
    if not paths:
        raise ValueError(f"no JSON/JSONL group sidecars found in {groups_dir}")
    for path in paths:
        if path.suffix == ".jsonl":
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise ValueError(f"{path}:{line_number}: sidecar row must be an object")
                record_id = value.get("record_id") or value.get("id")
                add(record_id, value, f"{path}:{line_number}")
        else:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, Mapping):
                raise ValueError(f"{path}: sidecar JSON must be an object mapping")
            for record_id, groups in value.items():
                add(record_id, groups, str(path))
    return result


def sidecar_fingerprints(groups_dir: Path) -> dict[str, str]:
    return {
        str(path.relative_to(groups_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(groups_dir.rglob("*"))
        if path.is_file() and path.suffix in {".json", ".jsonl"}
    }
