"""Deterministic record-to-group sidecar helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


GROUP_KEYS = {"record_id", "graph_group_id", "world_group_id"}


def stable_group_digest(namespace: str, value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return f"{namespace}:sha256:{hashlib.sha256(payload).hexdigest()}"


def group_entry(record_id: str, graph_group_id: str, world_group_id: str) -> dict[str, str]:
    return {
        "record_id": record_id,
        "graph_group_id": graph_group_id,
        "world_group_id": world_group_id,
    }


def write_group_manifest(path: Path, entries: Iterable[dict[str, str]]) -> None:
    rows = sorted(entries, key=lambda row: row["record_id"])
    seen: set[str] = set()
    for row in rows:
        if set(row) != GROUP_KEYS or not all(
            isinstance(row[key], str) and row[key] for key in GROUP_KEYS
        ):
            raise ValueError("group entries must contain exactly three non-empty string fields")
        if row["record_id"] in seen:
            raise ValueError(f"duplicate group record ID {row['record_id']!r}")
        seen.add(row["record_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)
