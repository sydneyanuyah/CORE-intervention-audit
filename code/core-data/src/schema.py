"""Schema and validation helpers for normalized CORE intervention records."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


ROOT_KEYS = {
    "id",
    "source",
    "source_id",
    "structure_kind",
    "graph",
    "chain",
    "factual",
    "intervention",
    "intervened",
    "descendants",
    "non_descendants",
    "probes",
    "provenance",
}
GRAPH_KEYS = {"nodes", "edges"}
GRAPH_OPTIONAL_KEYS = {"node_text"}
FACTUAL_KEYS = {"passage", "state", "question", "answer"}
INTERVENTION_KEYS = {
    "target",
    "target_text",
    "target_span",
    "value",
    "value_token",
    "replacement_span",
    "kind",
    "formal",
    "text",
}
INTERVENED_KEYS = {"passage", "state", "answer"}
PROBE_KEYS = {
    "variable",
    "question",
    "answer_before",
    "answer_after",
    "required",
    "actually_changed",
}
PROVENANCE_KEYS = {
    "fetched_utc",
    "url",
    "file",
    "sha256",
    "converter",
    "converter_version",
}

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


class RecordValidationError(ValueError):
    """Raised when a CORE record fails one or more validation rules."""

    def __init__(self, errors: Iterable[str]):
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


def _exact_keys(value: Any, expected: set[str], path: str, errors: List[str]) -> bool:
    if not isinstance(value, dict):
        errors.append(f"{path} must be an object")
        return False
    actual = set(value)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing:
        errors.append(f"{path} missing keys: {missing}")
    if unexpected:
        errors.append(f"{path} unexpected keys: {unexpected}")
    return not missing and not unexpected


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _string_list(value: Any, path: str, errors: List[str]) -> bool:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        errors.append(f"{path} must be a list of strings")
        return False
    if len(value) != len(set(value)):
        errors.append(f"{path} must not contain duplicates")
        return False
    return True


def _validate_state(value: Any, path: str, errors: List[str]) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        errors.append(f"{path} must be an object or null")
        return
    for key, item in value.items():
        if not isinstance(key, str) or not _is_scalar(item):
            errors.append(f"{path} must map string variables to JSON scalar values")
            return


def _validate_span(
    value: Any,
    path: str,
    passage: Any,
    expected_text: Any,
    errors: List[str],
) -> None:
    """Validate a required half-open character span and its exact text."""

    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(isinstance(item, int) and not isinstance(item, bool) for item in value)
    ):
        errors.append(f"{path} must be a [start, end] integer pair")
        return
    start, end = value
    if start < 0 or end <= start:
        errors.append(f"{path} must satisfy 0 <= start < end")
        return
    if not isinstance(passage, str):
        errors.append(f"{path} requires its passage to be a string")
        return
    if end > len(passage):
        errors.append(f"{path} is outside its passage")
        return
    if not isinstance(expected_text, str):
        errors.append(f"{path} requires the expected text to be a string")
        return
    actual = passage[start:end]
    if actual != expected_text:
        errors.append(
            f"{path} slices {actual!r}, expected {expected_text!r}"
        )


def _validate_dag(nodes: List[str], edges: List[List[str]], errors: List[str]) -> None:
    children = {node: [] for node in nodes}
    indegree = {node: 0 for node in nodes}
    for edge in edges:
        if not isinstance(edge, list) or len(edge) != 2 or not all(
            isinstance(item, str) for item in edge
        ):
            errors.append("graph.edges entries must be [parent, child] string pairs")
            continue
        parent, child = edge
        if parent not in children or child not in children:
            errors.append(f"graph edge references unknown node: {edge}")
            continue
        if parent == child:
            errors.append(f"graph contains self-edge: {edge}")
            continue
        children[parent].append(child)
        indegree[child] += 1
    queue = [node for node, degree in indegree.items() if degree == 0]
    visited = 0
    while queue:
        node = queue.pop()
        visited += 1
        for child in children[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    if visited != len(nodes):
        errors.append("graph must be acyclic")


def _find_nonfinite(value: Any, path: str, errors: List[str]) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        errors.append(f"{path} contains NaN or infinity")
    elif isinstance(value, dict):
        for key, item in value.items():
            _find_nonfinite(item, f"{path}.{key}", errors)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _find_nonfinite(item, f"{path}[{index}]", errors)


def validate_record(record: Mapping[str, Any], *, require_observed_effect: bool = True) -> List[str]:
    """Return all validation errors for one normalized record."""

    errors: List[str] = []
    if not _exact_keys(record, ROOT_KEYS, "record", errors):
        return errors

    for key in ("id", "source"):
        if not isinstance(record[key], str) or not record[key].strip():
            errors.append(f"{key} must be a non-empty string")
    if not isinstance(record["source_id"], (str, int)) or isinstance(record["source_id"], bool):
        errors.append("source_id must be a string or integer")

    kind = record["structure_kind"]
    if kind not in {"dag", "chain"}:
        errors.append("structure_kind must be 'dag' or 'chain'")

    graph_nodes: List[str] = []
    if kind == "dag":
        if record["chain"] is not None:
            errors.append("chain must be null for dag records")
        graph = record["graph"]
        graph_keys_valid = isinstance(graph, dict)
        if not graph_keys_valid:
            errors.append("graph must be an object")
        else:
            missing = sorted(GRAPH_KEYS - set(graph))
            unexpected = sorted(set(graph) - GRAPH_KEYS - GRAPH_OPTIONAL_KEYS)
            if missing:
                errors.append(f"graph missing keys: {missing}")
            if unexpected:
                errors.append(f"graph unexpected keys: {unexpected}")
            graph_keys_valid = not missing and not unexpected
        if graph_keys_valid:
            graph_nodes = record["graph"]["nodes"]
            edges = record["graph"]["edges"]
            if _string_list(graph_nodes, "graph.nodes", errors) and graph_nodes:
                if not isinstance(edges, list):
                    errors.append("graph.edges must be a list")
                else:
                    _validate_dag(graph_nodes, edges, errors)
                node_text = record["graph"].get("node_text")
                if node_text is not None:
                    if not isinstance(node_text, dict) or set(node_text) != set(graph_nodes):
                        errors.append("graph.node_text must map every graph node exactly once")
                    else:
                        for node, surfaces in node_text.items():
                            if (
                                not isinstance(surfaces, list)
                                or not surfaces
                                or len(surfaces) != len(set(surfaces))
                                or not all(isinstance(surface, str) and surface.strip() for surface in surfaces)
                            ):
                                errors.append(
                                    f"graph.node_text[{node!r}] must be a non-empty unique list of strings"
                                )
            elif graph_nodes == []:
                errors.append("graph.nodes must not be empty")
    elif kind == "chain":
        if record["graph"] is not None:
            errors.append("graph must be null for chain records")
        if not isinstance(record["chain"], list) or not all(
            isinstance(item, str) for item in record["chain"]
        ):
            errors.append("chain must be a list of event strings")
        elif len(record["chain"]) < 2:
            errors.append("chain must contain at least two events")

    if _exact_keys(record["factual"], FACTUAL_KEYS, "factual", errors):
        for key in ("passage", "question"):
            if record["factual"][key] is not None and not isinstance(record["factual"][key], str):
                errors.append(f"factual.{key} must be a string or null")
        if not _is_scalar(record["factual"]["answer"]):
            errors.append("factual.answer must be a JSON scalar or null")
        _validate_state(record["factual"]["state"], "factual.state", errors)

    target = None
    if _exact_keys(record["intervention"], INTERVENTION_KEYS, "intervention", errors):
        intervention = record["intervention"]
        target = intervention["target"]
        if not isinstance(target, str) or not target:
            errors.append("intervention.target must be a non-empty string")
        target_text = intervention["target_text"]
        if not isinstance(target_text, str) or not target_text:
            errors.append("intervention.target_text must be a non-empty string")
        if not _is_scalar(intervention["value"]):
            errors.append("intervention.value must be a JSON scalar or null")
        if intervention["kind"] not in {"value_set", "event_replace"}:
            errors.append("intervention.kind must be 'value_set' or 'event_replace'")
        for key in ("formal", "text"):
            if intervention[key] is not None and not isinstance(intervention[key], str):
                errors.append(f"intervention.{key} must be a string or null")

        factual_passage = record["factual"].get("passage") if isinstance(
            record["factual"], dict
        ) else None
        _validate_span(
            intervention["target_span"],
            "intervention.target_span",
            factual_passage,
            target_text,
            errors,
        )

        if intervention["kind"] == "value_set":
            if intervention["value_token"] is None or not _is_scalar(
                intervention["value_token"]
            ):
                errors.append(
                    "intervention.value_token must be a non-null JSON scalar for value_set"
                )
            if intervention["replacement_span"] is not None:
                errors.append(
                    "intervention.replacement_span must be null for value_set"
                )
        elif intervention["kind"] == "event_replace":
            if intervention["value_token"] is not None:
                errors.append(
                    "intervention.value_token must be null for event_replace"
                )
            _validate_span(
                intervention["replacement_span"],
                "intervention.replacement_span",
                factual_passage,
                intervention["value"],
                errors,
            )

    if _exact_keys(record["intervened"], INTERVENED_KEYS, "intervened", errors):
        if record["intervened"]["passage"] is not None and not isinstance(
            record["intervened"]["passage"], str
        ):
            errors.append("intervened.passage must be a string or null")
        if not _is_scalar(record["intervened"]["answer"]):
            errors.append("intervened.answer must be a JSON scalar or null")
        _validate_state(record["intervened"]["state"], "intervened.state", errors)

    descendants_ok = _string_list(record["descendants"], "descendants", errors)
    non_descendants_ok = _string_list(record["non_descendants"], "non_descendants", errors)
    if descendants_ok and non_descendants_ok:
        overlap = sorted(set(record["descendants"]) & set(record["non_descendants"]))
        if overlap:
            errors.append(f"descendants and non_descendants overlap: {overlap}")
        if target in record["descendants"] or target in record["non_descendants"]:
            errors.append("intervention target must appear in neither descendants list")
        if kind == "dag" and graph_nodes:
            unknown = sorted(
                (set(record["descendants"]) | set(record["non_descendants"])) - set(graph_nodes)
            )
            if unknown:
                errors.append(f"descendant lists reference unknown graph nodes: {unknown}")

    probes = record["probes"]
    observed_effect = False
    if not isinstance(probes, list):
        errors.append("probes must be a list")
    else:
        for index, probe in enumerate(probes):
            path = f"probes[{index}]"
            if not _exact_keys(probe, PROBE_KEYS, path, errors):
                continue
            if not isinstance(probe["variable"], str) or not probe["variable"]:
                errors.append(f"{path}.variable must be a non-empty string")
            if probe["question"] is not None and not isinstance(probe["question"], str):
                errors.append(f"{path}.question must be a string or null")
            if not _is_scalar(probe["answer_before"]) or not _is_scalar(probe["answer_after"]):
                errors.append(f"{path} answers must be JSON scalars or null")
            if probe["required"] not in {"must not change", "may change"}:
                errors.append(f"{path}.required has an invalid value")
            if not isinstance(probe["actually_changed"], bool):
                errors.append(f"{path}.actually_changed must be boolean")
            answers_known = probe["answer_before"] is not None and probe["answer_after"] is not None
            if answers_known:
                changed = probe["answer_before"] != probe["answer_after"]
                if probe["actually_changed"] != changed:
                    errors.append(f"{path}.actually_changed disagrees with its answers")
            if probe["required"] == "must not change":
                if probe["answer_before"] != probe["answer_after"]:
                    errors.append(f"{path} violates required preservation")
                if probe["actually_changed"]:
                    errors.append(f"{path} is marked changed despite required preservation")
            if probe["required"] == "may change" and probe["actually_changed"]:
                observed_effect = True
        if require_observed_effect and not observed_effect:
            errors.append("record has no may-change probe with an observed change")

    if _exact_keys(record["provenance"], PROVENANCE_KEYS, "provenance", errors):
        provenance = record["provenance"]
        if not isinstance(provenance["fetched_utc"], str) or not UTC_RE.fullmatch(
            provenance["fetched_utc"]
        ):
            errors.append("provenance.fetched_utc must be an ISO-8601 UTC timestamp")
        for key in ("url", "file", "converter"):
            if not isinstance(provenance[key], str) or not provenance[key]:
                errors.append(f"provenance.{key} must be a non-empty string")
        if not isinstance(provenance["sha256"], str) or not SHA256_RE.fullmatch(
            provenance["sha256"]
        ):
            errors.append("provenance.sha256 must be 64 lowercase hexadecimal characters")
        if not isinstance(provenance["converter_version"], (str, int)) or isinstance(
            provenance["converter_version"], bool
        ):
            errors.append("provenance.converter_version must be a string or integer")

    _find_nonfinite(record, "record", errors)
    return errors


def require_valid_record(record: Mapping[str, Any], *, require_observed_effect: bool = True) -> None:
    errors = validate_record(record, require_observed_effect=require_observed_effect)
    if errors:
        raise RecordValidationError(errors)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: record must be a JSON object")
            records.append(value)
    return records
