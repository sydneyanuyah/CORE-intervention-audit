"""Deterministic, validation-only CLadder perturbations for T3 robustness."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


PROTOCOL = "t3_cladder_robustness_v1"
MANIFEST_VERSION = 1
DATASETS = frozenset({"variable_rename", "cladder_variants"})
COMMAND_TEMPLATES = (
    ("assign", "Apply an intervention assigning {value} to {target}."),
    ("intervene", "Intervene on {target} so its value becomes {value}."),
    ("suppose", "Suppose {target} is set to {value}."),
)


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def structural_aliases(nodes: Sequence[str], world_group_id: str) -> dict[str, str]:
    """Assign opaque three-digit identifiers without using date-like seeds."""

    if not nodes or len(set(nodes)) != len(nodes):
        raise ValueError("CLadder nodes must be non-empty and unique")
    if not isinstance(world_group_id, str) or not world_group_id:
        raise ValueError("world_group_id must be non-empty")
    ordered = sorted(
        nodes,
        key=lambda node: hashlib.sha256(
            f"{PROTOCOL}:rename:317:{world_group_id}:{node}".encode("utf-8")
        ).hexdigest(),
    )
    return {node: f"N{101 + index:03d}" for index, node in enumerate(ordered)}


def _rename_mapping(mapping: Mapping[str, Any], aliases: Mapping[str, str]) -> dict[str, Any]:
    if set(mapping) != set(aliases):
        raise ValueError("state keys do not match the structural rename map")
    return {aliases[key]: value for key, value in mapping.items()}


def rename_structural_variables(
    record: Mapping[str, Any], aliases: Mapping[str, str]
) -> dict[str, Any]:
    """Rename only structural IDs; natural-language evidence and spans stay fixed."""

    result = copy.deepcopy(record)
    graph = result.get("graph")
    if not isinstance(graph, dict):
        raise ValueError("CLadder record requires a graph")
    nodes = graph.get("nodes")
    if not isinstance(nodes, list) or set(nodes) != set(aliases):
        raise ValueError("rename map must cover every graph node exactly")
    if len(set(aliases.values())) != len(aliases) or any(
        not isinstance(value, str) or not value for value in aliases.values()
    ):
        raise ValueError("renamed structural IDs must be unique non-empty strings")
    graph["nodes"] = [aliases[node] for node in nodes]
    graph["edges"] = [[aliases[left], aliases[right]] for left, right in graph["edges"]]
    result["factual"]["state"] = _rename_mapping(result["factual"]["state"], aliases)
    result["intervened"]["state"] = _rename_mapping(result["intervened"]["state"], aliases)
    intervention = result["intervention"]
    old_target = intervention["target"]
    intervention["target"] = aliases[old_target]
    intervention["formal"] = f"do({aliases[old_target]} = {int(intervention['value'])})"
    result["descendants"] = [aliases[node] for node in result["descendants"]]
    result["non_descendants"] = [aliases[node] for node in result["non_descendants"]]
    for probe in result["probes"]:
        probe["variable"] = aliases[probe["variable"]]
    result["id"] = f"t3-variable-rename:{record['id']}"
    result["t3_perturbation"] = {
        "protocol": PROTOCOL,
        "dataset": "variable_rename",
        "seed": 317,
        "structural_aliases": dict(sorted(aliases.items())),
        "natural_language_changed": False,
        "test_evaluated": False,
    }
    return result


def command_variant(record: Mapping[str, Any], template_id: str) -> dict[str, Any]:
    templates = dict(COMMAND_TEMPLATES)
    if template_id not in templates:
        raise ValueError(f"unknown CLadder command template {template_id!r}")
    result = copy.deepcopy(record)
    intervention = result.get("intervention")
    if not isinstance(intervention, dict) or intervention.get("kind") != "value_set":
        raise ValueError("CLadder T3 variants require a value_set intervention")
    target = intervention.get("target_text")
    value = intervention.get("value")
    if not isinstance(target, str) or not target or value not in (0, 1, False, True):
        raise ValueError("CLadder command requires target text and a binary value")
    intervention["text"] = templates[template_id].format(
        target=json.dumps(target, ensure_ascii=False), value=int(value)
    )
    result["id"] = f"t3-cladder-variant:{record['id']}"
    result["t3_perturbation"] = {
        "protocol": PROTOCOL,
        "dataset": "cladder_variants",
        "template_id": template_id,
        "test_evaluated": False,
    }
    return result


def inverted_instruction(record: Mapping[str, Any]) -> dict[str, Any]:
    """Flip a binary value-set command without changing its target or labels.

    This is a decision-sensitivity positive control, not a scored task example: the
    original gold outcome is intentionally retained so paired predictions can show
    whether the model responds when the requested value is reversed.
    """

    result = copy.deepcopy(record)
    intervention = result.get("intervention")
    if not isinstance(intervention, dict) or intervention.get("kind") != "value_set":
        raise ValueError("T3 inverted controls require a value_set intervention")
    value = intervention.get("value")
    target = intervention.get("target")
    target_text = intervention.get("target_text")
    if value not in (0, 1, False, True) or not isinstance(target, str) or not target:
        raise ValueError("T3 inverted controls require a binary value and target")
    if not isinstance(target_text, str) or not target_text:
        raise ValueError("T3 inverted controls require target text")
    inverted = 1 - int(value)
    intervention["value"] = inverted
    intervention["value_token"] = "yes" if inverted else "no"
    intervention["formal"] = f"do({target} = {inverted})"
    intervention["text"] = f"Set {target_text} to {inverted}."
    result["id"] = f"t3-inverted-instruction:{record['id']}"
    result["t3_positive_control"] = {
        "protocol": "t3_inverted_instruction_positive_control_v1",
        "original_value": int(value),
        "inverted_value": inverted,
        "gold_retained_for_sensitivity_only": True,
        "test_evaluated": False,
    }
    return result


def selected_template(record_id: str) -> str:
    if not isinstance(record_id, str) or not record_id:
        raise ValueError("record ID must be non-empty")
    digest = hashlib.sha256(f"{PROTOCOL}:template:317:{record_id}".encode()).digest()
    return COMMAND_TEMPLATES[int.from_bytes(digest[:8], "big") % len(COMMAND_TEMPLATES)][0]


def apply_manifest_pair(record: Mapping[str, Any], pair: Mapping[str, Any]) -> dict[str, Any]:
    if pair.get("protocol") != PROTOCOL or pair.get("split") != "validation":
        raise ValueError("T3 CLadder pair is not frozen validation-only protocol data")
    if pair.get("test_evaluated") is not False or pair.get("record_id") != record.get("id"):
        raise ValueError("T3 CLadder pair identity or test isolation is invalid")
    if pair.get("clean_sha256") != sha256_json(record):
        raise ValueError("T3 CLadder clean record hash mismatch")
    dataset = pair.get("dataset")
    if dataset == "variable_rename":
        perturbed = rename_structural_variables(record, pair.get("structural_aliases", {}))
    elif dataset == "cladder_variants":
        perturbed = command_variant(record, pair.get("template_id"))
    else:
        raise ValueError(f"unsupported T3 CLadder dataset {dataset!r}")
    if pair.get("perturbed_sha256") != sha256_json(perturbed):
        raise ValueError("T3 CLadder perturbed record hash mismatch")
    return perturbed


def load_manifest(path: Path, artifact_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != PROTOCOL or manifest.get("version") != MANIFEST_VERSION:
        raise ValueError("unsupported T3 CLadder manifest")
    if manifest.get("split") != "validation" or manifest.get("test_evaluated") is not False:
        raise ValueError("T3 CLadder manifest must be validation-only")
    if manifest.get("source_artifact_sha256") != sha256_file(artifact_path):
        raise ValueError("T3 CLadder source artifact hash mismatch")
    pairs = manifest.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("T3 CLadder manifest has no pairs")
    pair_ids = [pair.get("pair_id") for pair in pairs]
    if len(pair_ids) != len(set(pair_ids)) or any(not isinstance(item, str) for item in pair_ids):
        raise ValueError("T3 CLadder pair IDs must be unique strings")
    return manifest, pairs


__all__ = [
    "DATASETS", "MANIFEST_VERSION", "PROTOCOL", "apply_manifest_pair",
    "command_variant", "inverted_instruction", "load_manifest", "rename_structural_variables",
    "selected_template", "sha256_file", "sha256_json", "structural_aliases",
]
