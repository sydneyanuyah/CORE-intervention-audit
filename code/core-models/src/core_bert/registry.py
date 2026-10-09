"""Validation for the machine-readable experiment scope."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_and_validate(path: Path) -> dict[str, Any]:
    scope = json.loads(path.read_text(encoding="utf-8"))
    if scope.get("experimental_unit") != "graph":
        raise ValueError("experimental unit must be graph")
    if set(scope.get("models", {})) != {"bert_base", "bert_large"}:
        raise ValueError("scope must contain exactly BERT-base and BERT-large")
    if scope["models"]["bert_base"]["gpus"] != [4]:
        raise ValueError("BERT-base GPU policy must be [4]")
    if scope["models"]["bert_large"]["gpus"] != [8, 16]:
        raise ValueError("BERT-large GPU policy must be [8, 16]")
    ids = [item["id"] for item in scope.get("experiments", [])]
    if len(ids) != len(set(ids)):
        raise ValueError("experiment IDs must be unique")
    if len(ids) != 23:
        raise ValueError(f"expected all 23 registry experiments, found {len(ids)}")
    for item in scope["experiments"]:
        if not item.get("primary"):
            raise ValueError(f"{item['id']} lacks a primary comparison")
        if item.get("requires_floors") and not scope["global_rules"].get("laws_require_floors"):
            raise ValueError(f"{item['id']} requires law floors")
    return scope
