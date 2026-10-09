"""Convert eligible CLadder deterministic counterfactuals to CORE DAG records."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from inspect_cladder import PINNED_REVISION, SOURCE_SHA256, load_release, parse_structure
from groups import group_entry, stable_group_digest, write_group_manifest
from schema import validate_record


SOURCE_URL = f"https://github.com/causalNLP/cladder/archive/{PINNED_REVISION}.tar.gz"
FETCHED_UTC = "2026-09-03T18:10:38Z"
CONVERTER_VERSION = 2
ELIGIBLE_QUERY_TYPE = "det-counterfactual"


def graph_parts(structure: str) -> tuple[list[str], list[list[str]], dict[str, list[str]]]:
    nodes, edges = parse_structure(structure)
    parents = {node: [] for node in nodes}
    for parent, child in edges:
        parents[child].append(parent)
    return nodes, edges, parents


def descendants_of(target: str, edges: list[list[str]]) -> list[str]:
    children: dict[str, list[str]] = collections.defaultdict(list)
    for parent, child in edges:
        children[parent].append(child)
    seen = set()
    stack = list(children[target])
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(children[node])
    return sorted(seen)


def conditional_key(variable: str, params: dict[str, Any]) -> tuple[str, list[str]]:
    prefix = f"p({variable} |"
    matches = [key for key in params if key.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"expected one conditional table for {variable}, found {matches}")
    key = matches[0]
    order = [part.strip() for part in key.split("|", 1)[1].rstrip(")").split(",")]
    return key, order


def solve_state(model: dict[str, Any], fixed_values: dict[str, int]) -> dict[str, int]:
    """Evaluate a deterministic SCM with all roots fixed and optional node overrides."""

    nodes, _, parents = graph_parts(model["structure"])
    sources = {node for node in nodes if not parents[node]}
    if not sources <= set(fixed_values) or not set(fixed_values) <= set(nodes):
        raise ValueError(
            f"fixed assignment mismatch: requires roots {sorted(sources)} within nodes {nodes}, "
            f"found {sorted(fixed_values)}"
        )
    state = {name: int(value) for name, value in fixed_values.items()}
    while len(state) < len(nodes):
        progressed = False
        for node in nodes:
            if node in state or any(parent not in state for parent in parents[node]):
                continue
            key, table_order = conditional_key(node, model["params"])
            value: Any = model["params"][key]
            for parent in table_order:
                value = value[state[parent]]
            if value not in (0, 1):
                raise ValueError(f"non-deterministic value for {node}: {value!r}")
            state[node] = int(value)
            progressed = True
        if not progressed:
            unresolved = sorted(set(nodes) - set(state))
            raise ValueError(f"could not solve nodes: {unresolved}")
    return {node: state[node] for node in nodes}


def model_group_split(model_id: int) -> str:
    key = f"cladder-split-v1:{model_id}".encode("utf-8")
    bucket = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


def target_surface_and_span(
    passage: str, target: str, variable_mapping: dict[str, Any]
) -> tuple[str, list[int]]:
    """Resolve a structural CLadder variable to text actually present in the story."""

    candidates = [
        variable_mapping.get(f"{target}name"),
        variable_mapping.get(f"{target}1"),
        variable_mapping.get(f"{target}0"),
    ]
    for surface in candidates:
        if not isinstance(surface, str) or not surface:
            continue
        start = passage.find(surface)
        if start >= 0:
            return surface, [start, start + len(surface)]
    raise ValueError(f"no verbalized surface for structural target {target!r} in passage")


def normalize_item(item: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    meta = item["meta"]
    if meta["query_type"] != ELIGIBLE_QUERY_TYPE:
        raise ValueError(f"ineligible query type {meta['query_type']!r}")
    if model["equation_type"] != "deterministic":
        raise ValueError(f"expected deterministic model, found {model['equation_type']!r}")

    nodes, edges, parents = graph_parts(model["structure"])
    target = meta["treatment"]
    action = int(meta["action"])
    if action not in (0, 1):
        raise ValueError(f"action is not Boolean: {action!r}")
    sources = {node for node in nodes if not parents[node]}
    evidence = {name: int(value) for name, value in meta["given_info"].items()}
    if set(evidence) != sources - {target}:
        raise ValueError(
            f"evidence is not exactly the non-treatment roots: {sorted(evidence)}"
        )

    factual_state = solve_state(model, {**evidence, target: 1 - action})
    intervened_state = solve_state(model, {**evidence, target: action})
    outcome = meta["outcome"]
    oracle_answer = "yes" if intervened_state[outcome] == int(meta["polarity"]) else "no"
    if oracle_answer != item["answer"] or int(oracle_answer == "yes") != int(meta["groundtruth"]):
        raise ValueError(
            f"reconstructed oracle mismatch: computed {oracle_answer}, "
            f"source answer={item['answer']}, groundtruth={meta['groundtruth']}"
        )

    descendants = descendants_of(target, edges)
    non_descendants = sorted(set(nodes) - {target} - set(descendants))
    probes = []
    for variable in nodes:
        required = "must not change" if variable in non_descendants else "may change"
        before = factual_state[variable]
        after = intervened_state[variable]
        probes.append(
            {
                "variable": variable,
                "question": None,
                "answer_before": before,
                "answer_after": after,
                "required": required,
                "actually_changed": before != after,
            }
        )

    question_id = int(item["question_id"])
    passage = item["given_info"]
    target_text, target_span = target_surface_and_span(
        passage, target, model["variable_mapping"]
    )
    return {
        "id": f"cladder-det-counterfactual-{question_id:06d}",
        "source": "cladder",
        "source_id": question_id,
        "structure_kind": "dag",
        "graph": {"nodes": nodes, "edges": edges},
        "chain": None,
        "factual": {
            "passage": passage,
            "state": factual_state,
            "question": item["question"],
            "answer": None,
        },
        "intervention": {
            "target": target,
            "target_text": target_text,
            "target_span": target_span,
            "value": action,
            "value_token": "yes" if action else "no",
            "replacement_span": None,
            "kind": "value_set",
            "formal": f"do({target} = {action})",
            "text": item["question"],
        },
        "intervened": {
            "passage": None,
            "state": intervened_state,
            "answer": item["answer"],
        },
        "descendants": descendants,
        "non_descendants": non_descendants,
        "probes": probes,
        "provenance": {
            "fetched_utc": FETCHED_UTC,
            "url": SOURCE_URL,
            "file": "data/raw/cladder/cladder-3d2d1169.tar.gz",
            "sha256": SOURCE_SHA256,
            "converter": "src/convert_cladder.py",
            "converter_version": CONVERTER_VERSION,
        },
    }


def write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n")
    temporary.replace(path)


def write_lines(path: Path, values: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(f"{value}\n" for value in values), encoding="utf-8")
    temporary.replace(path)


def rejection_reason(query_type: str) -> str:
    categories = {
        "marginal": "observational marginal query has no intervention action",
        "correlation": "observational association query has no intervention action",
        "backadj": "adjustment-set selection query is not a state intervention",
        "ate": "population effect contrasts two actions rather than one before/after world",
        "ett": "effect-on-treated estimand contrasts counterfactual populations",
        "nde": "nested direct-effect estimand cannot be represented by one value_set action",
        "nie": "nested indirect-effect estimand cannot be represented by one value_set action",
        "exp_away": "conditioning/explaining-away query is not a state intervention",
        "collider_bias": "collider-bias query does not supply a single before/after world",
    }
    return categories.get(query_type, f"unsupported query type {query_type!r}")


def convert(
    input_path: Path,
    records_dir: Path,
    splits_dir: Path,
    summary_path: Path,
    groups_dir: Path = Path("data/groups"),
) -> dict:
    questions, models, actual_hash = load_release(input_path)
    by_model = {model["model_id"]: model for model in models}
    accepted = []
    rejected = []
    rejected_reasons = collections.Counter()
    split_ids: dict[str, list[str]] = collections.defaultdict(list)
    accepted_models: dict[str, set[int]] = collections.defaultdict(set)
    group_entries = []

    for item in questions:
        query_type = item["meta"]["query_type"]
        if query_type != ELIGIBLE_QUERY_TYPE:
            reason = rejection_reason(query_type)
            rejected_reasons[reason] += 1
            rejected.append({"reason": reason, "record": item})
            continue
        model_id = int(item["meta"]["model_id"])
        try:
            record = normalize_item(item, by_model[model_id])
        except (KeyError, TypeError, ValueError) as exc:
            reason = f"deterministic reconstruction error: {exc}"
            rejected_reasons[reason] += 1
            rejected.append({"reason": reason, "record": item})
            continue
        errors = validate_record(record)
        if errors:
            reason = "; ".join(errors)
            rejected_reasons[reason] += 1
            rejected.append({"reason": reason, "record": record})
            continue
        accepted.append(record)
        group_entries.append(
            group_entry(
                record["id"],
                f"cladder:model:{model_id}",
                stable_group_digest(
                    f"cladder:model:{model_id}:world", record["factual"]["state"]
                ),
            )
        )
        split = model_group_split(model_id)
        split_ids[split].append(record["id"])
        accepted_models[split].add(model_id)

    if len(accepted) != 1422 or len(rejected) != 8690:
        raise ValueError(f"unexpected conversion counts: accepted={len(accepted)}, rejected={len(rejected)}")
    if any("reconstruction error" in reason for reason in rejected_reasons):
        raise ValueError(f"eligible deterministic rows failed reconstruction: {rejected_reasons}")
    model_sets = [accepted_models[name] for name in ("train", "validation", "test")]
    if any(model_sets[i] & model_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("model leakage across splits")

    write_jsonl(records_dir / "cladder.jsonl", accepted)
    write_jsonl(records_dir / "cladder.rejected.jsonl", rejected)
    write_group_manifest(groups_dir / "cladder.jsonl", group_entries)
    for split in ("train", "validation", "test"):
        write_lines(splits_dir / f"cladder.{split}.txt", sorted(split_ids[split]))

    summary = {
        "source": "cladder",
        "raw_sha256": actual_hash,
        "balanced_questions": len(questions),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "rejected_query_type_counts": dict(
            sorted(collections.Counter(
                item["record"]["meta"]["query_type"] for item in rejected
                if "meta" in item["record"]
            ).items())
        ),
        "rejection_reason_counts": dict(sorted(rejected_reasons.items())),
        "split_counts": {split: len(split_ids[split]) for split in ("train", "validation", "test")},
        "split_model_counts": {
            split: len(accepted_models[split]) for split in ("train", "validation", "test")
        },
        "split_group": "model_id",
        "group_sidecar_entries": len(group_entries),
        "split_algorithm": "sha256('cladder-split-v1:' + model_id) modulo 100; 0-79 train, 80-89 validation, 90-99 test",
        "oracle_check": "all accepted answers matched deterministic SCM reconstruction",
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(summary_path)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/cladder/cladder-3d2d1169.tar.gz"),
    )
    parser.add_argument("--records-dir", type=Path, default=Path("data/records"))
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--groups-dir", type=Path, default=Path("data/groups"))
    parser.add_argument("--summary", type=Path, default=Path("reports/cladder_conversion_summary.json"))
    args = parser.parse_args()
    print(json.dumps(convert(args.input, args.records_dir, args.splits_dir, args.summary, args.groups_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
