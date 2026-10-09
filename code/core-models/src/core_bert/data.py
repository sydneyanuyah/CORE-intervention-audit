"""Load and format the closed-label CORE intervention subset."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LABEL_TO_ID = {"no": 0, "yes": 1, "less": 2, "more": 3}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}
SUPPORTED_SOURCES = {"cladder", "wiqa", "ccrgb"}


@dataclass(frozen=True)
class CoreExample:
    record_id: str
    source: str
    text: str
    label: int


def _compact(value: Any) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, dict):
        return "; ".join(f"{key}={value[key]}" for key in sorted(value)) or "none"
    if isinstance(value, list):
        return "; ".join(str(item) for item in value) or "none"
    return str(value).strip()


def format_record(record: dict[str, Any]) -> str:
    factual = record["factual"]
    intervention = record["intervention"]
    graph = record.get("graph")
    if graph:
        structure = "; ".join(f"{parent} -> {child}" for parent, child in graph["edges"])
    else:
        structure = " -> ".join(record.get("chain") or [])
    fields = [
        f"source: {record['source']}",
        f"intervention: {_compact(intervention.get('text'))}",
        f"formal intervention: {_compact(intervention.get('formal'))}",
        f"target: {_compact(intervention.get('target'))}",
        f"intervention value: {_compact(intervention.get('value'))}",
        f"factual question: {_compact(factual.get('question'))}",
        f"factual answer: {_compact(factual.get('answer'))}",
        f"factual state: {_compact(factual.get('state'))}",
        f"structure: {structure or 'no edges'}",
        f"passage: {_compact(factual.get('passage'))}",
    ]
    return " [SEP] ".join(fields)


def _split_ids(splits_dir: Path, split: str) -> set[str]:
    paths = sorted(splits_dir.glob(f"*.{split}.txt"))
    if not paths:
        raise ValueError(f"no split manifests found for {split!r} in {splits_dir}")
    values: set[str] = set()
    for path in paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            record_id = line.strip()
            if not record_id or record_id.startswith("#"):
                continue
            if record_id in values:
                raise ValueError(f"duplicate split id {record_id!r} at {path}:{line_number}")
            values.add(record_id)
    return values


def load_split(data_root: Path, split: str) -> list[CoreExample]:
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"unsupported split {split!r}")
    records_dir = data_root / "records"
    selected_ids = _split_ids(data_root / "splits", split)
    examples: list[CoreExample] = []
    seen: set[str] = set()
    for path in sorted(records_dir.glob("*.jsonl")):
        if path.name.endswith(".rejected.jsonl"):
            continue
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("source") not in SUPPORTED_SOURCES or record.get("id") not in selected_ids:
                    continue
                record_id = record["id"]
                if record_id in seen:
                    raise ValueError(f"duplicate accepted record id {record_id!r}")
                answer = str(record["intervened"]["answer"]).strip().lower()
                if answer not in LABEL_TO_ID:
                    raise ValueError(f"unsupported answer {answer!r} at {path}:{line_number}")
                seen.add(record_id)
                examples.append(CoreExample(record_id, record["source"], format_record(record), LABEL_TO_ID[answer]))
    if not examples:
        raise ValueError(f"no compatible examples found for split {split!r}")
    return sorted(examples, key=lambda item: item.record_id)
