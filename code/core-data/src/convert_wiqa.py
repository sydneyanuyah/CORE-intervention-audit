"""Convert resolvable WIQA signed influence questions into CORE DAG records."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

from schema import validate_record
from groups import group_entry, write_group_manifest


FETCHED_UTC = "2026-09-03T18:10:38Z"
CONVERTER_VERSION = 3
GRAPH_FILE = "wiqa_influence_graphs.jsonl"
GRAPH_SHA256 = "not-published"
QUESTION_FILES = {
    "train": ("no_explanation_v2_train.jsonl", "train.jsonl", "not-published"),
    "validation": ("no_explanation_v2_dev.jsonl", "dev.jsonl", "not-published"),
    "test": ("no_explanation_v2_test.jsonl", "test.jsonl", "not-published"),
}
QUESTION_URL_ROOT = "https://public-aristo-processes.s3-us-west-2.amazonaws.com/wiqa_dataset_no_explanation_v2/"
STEM_RE = re.compile(
    r"\s*suppose\s+(.*?)\s+happens,\s+how\s+will\s+it\s+affect\s+(.+?)\.?\s*",
    re.IGNORECASE,
)
VALUE_TOKEN_BY_NODE = {
    "V": "less",
    "Z": "more",
    "X": "more",
    "U": "less",
    "W": "less",
    "Y": "more",
    "A": "more",
    "D": "less",
}
TARGET_MARKER = " [TARGET] "


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_grounding(text: str) -> str:
    """Match WIQA's source normalizer: letters only, case-insensitive."""

    return "".join(character.lower() for character in text if character.isalpha())


def parse_stem(stem: str) -> tuple[str, str] | None:
    match = STEM_RE.fullmatch(stem)
    return match.groups() if match else None


def append_target_surface(passage: str, surface: str) -> tuple[str, list[int]]:
    """Append one unambiguous pointer target without modifying the source paragraph."""

    if not isinstance(passage, str) or not isinstance(surface, str) or not surface:
        raise ValueError("paragraph and resolved source text must be non-empty strings")
    start = len(passage) + len(TARGET_MARKER)
    return passage + TARGET_MARKER + surface, [start, start + len(surface)]


def graph_components(graph: dict[str, Any]) -> tuple[list[str], list[tuple[str, str, int]], dict[str, set[str]]]:
    groundings: dict[str, set[str]] = collections.defaultdict(set)
    active = set()
    for node in ("V", "Z", "X", "U", "W", "Y"):
        values = graph.get(node, [])
        if not isinstance(values, list):
            values = [values]
        for value in values:
            if value:
                groundings[normalized_grounding(value)].add(node)
                active.add(node)
    for node, key in (("A", "para_outcome_accelerate"), ("D", "para_outcome_decelerate")):
        value = graph.get(key, "")
        if value:
            groundings[normalized_grounding(value)].add(node)
            active.add(node)

    y_positive = graph.get("Y_affects_outcome") in ("a", "more", True)
    edges = [
        ("V", "X", -1),
        ("Z", "X", 1),
        ("X", "W", -1),
        ("X", "Y", 1),
        ("U", "Y", -1),
        ("W", "A", -1 if y_positive else 1),
        ("W", "D", 1 if y_positive else -1),
        ("Y", "A", 1 if y_positive else -1),
        ("Y", "D", -1 if y_positive else 1),
    ]
    edges = [edge for edge in edges if edge[0] in active and edge[1] in active]
    return sorted(active), edges, groundings


def graph_node_text(graph: dict[str, Any], nodes: list[str]) -> dict[str, list[str]]:
    """Preserve released natural-language groundings for each symbolic node."""

    result: dict[str, list[str]] = {}
    for node in nodes:
        key = {
            "A": "para_outcome_accelerate",
            "D": "para_outcome_decelerate",
        }.get(node, node)
        raw = graph.get(key, [])
        values = raw if isinstance(raw, list) else [raw]
        surfaces = list(dict.fromkeys(
            value.strip() for value in values if isinstance(value, str) and value.strip()
        ))
        if not surfaces:
            raise ValueError(f"active WIQA node {node!r} has no text grounding")
        result[node] = surfaces
    return result


def signed_paths(source: str, target: str, edges: list[tuple[str, str, int]]) -> list[tuple[int, int]]:
    children: dict[str, list[tuple[str, int]]] = collections.defaultdict(list)
    for parent, child, sign in edges:
        children[parent].append((child, sign))
    paths = []

    def visit(node: str, seen: list[str], sign: int) -> None:
        if node == target:
            paths.append((len(seen), sign))
            return
        for child, edge_sign in children[node]:
            if child not in seen:
                visit(child, seen + [child], sign * edge_sign)

    visit(source, [source], 1)
    return paths


def resolve_question(item: dict[str, Any], graph: dict[str, Any]) -> tuple[str, str] | None:
    parsed = parse_stem(item["question"]["stem"])
    if parsed is None:
        return None
    source_text, target_text = parsed
    _, edges, groundings = graph_components(graph)
    source_nodes = groundings[normalized_grounding(source_text)]
    target_nodes = groundings[normalized_grounding(target_text)]
    expected_length = int(item["metadata"]["path_len"])
    expected_sign = 1 if item["question"]["answer_label"] == "more" else -1
    pairs = set()
    for source in source_nodes:
        for target in target_nodes:
            if any(
                path_length == expected_length and sign == expected_sign
                for path_length, sign in signed_paths(source, target, edges)
            ):
                pairs.add((source, target))
    if len(pairs) != 1:
        return None
    return next(iter(pairs))


def descendants_of(target: str, edges: list[tuple[str, str, int]]) -> list[str]:
    children: dict[str, list[str]] = collections.defaultdict(list)
    for parent, child, _ in edges:
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


def normalize_item(
    item: dict[str, Any], graph: dict[str, Any], split: str, filename: str, remote_filename: str, file_hash: str
) -> dict[str, Any]:
    resolved = resolve_question(item, graph)
    if resolved is None:
        raise ValueError("question does not resolve to exactly one signed graph source/target pair")
    source_node, queried_node = resolved
    nodes, signed_edges, _ = graph_components(graph)
    descendants = descendants_of(source_node, signed_edges)
    if queried_node not in descendants:
        raise ValueError("resolved query target is not a graph descendant")
    non_descendants = sorted(set(nodes) - {source_node} - set(descendants))
    parsed = parse_stem(item["question"]["stem"])
    if parsed is None:
        raise ValueError("question stem could not be parsed")
    source_text, _ = parsed
    original_passage = graph["paragraph"]
    passage, target_span = append_target_surface(original_passage, source_text)
    node_text = graph_node_text(graph, nodes)
    node_text[source_node] = [
        source_text,
        *(surface for surface in node_text[source_node] if surface != source_text),
    ]
    answer = item["question"]["answer_label"]
    if answer not in {"more", "less"}:
        raise ValueError(f"accepted causal answer must be more/less, found {answer!r}")

    probes = [
        {
            "variable": queried_node,
            "question": item["question"]["stem"],
            "answer_before": "no_effect",
            "answer_after": answer,
            "required": "may change",
            "actually_changed": True,
        }
    ]
    probes.extend(
        {
            "variable": node,
            "question": None,
            "answer_before": "no_effect",
            "answer_after": "no_effect",
            "required": "must not change",
            "actually_changed": False,
        }
        for node in non_descendants
    )
    question_id = item["metadata"]["ques_id"]
    return {
        "id": f"wiqa-{question_id}",
        "source": "wiqa",
        "source_id": question_id,
        "structure_kind": "dag",
        "graph": {
            "nodes": nodes,
            "edges": [[parent, child] for parent, child, _ in signed_edges],
            "node_text": node_text,
        },
        "chain": None,
        "factual": {
            "passage": passage,
            "state": None,
            "question": item["question"]["stem"],
            "answer": None,
        },
        "intervention": {
            "target": source_node,
            "target_text": source_text,
            "target_span": target_span,
            "value": source_text,
            "value_token": VALUE_TOKEN_BY_NODE[source_node],
            "replacement_span": None,
            "kind": "value_set",
            "formal": f"directional_change({source_node} = {json.dumps(source_text, ensure_ascii=False)})",
            "text": source_text,
        },
        "intervened": {"passage": None, "state": None, "answer": answer},
        "descendants": descendants,
        "non_descendants": non_descendants,
        "probes": probes,
        "provenance": {
            "fetched_utc": FETCHED_UTC,
            "url": QUESTION_URL_ROOT + remote_filename,
            "file": f"data/raw/wiqa/{filename}",
            "sha256": file_hash,
            "converter": "src/convert_wiqa.py",
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


def convert(
    raw_dir: Path,
    records_dir: Path,
    splits_dir: Path,
    summary_path: Path,
    groups_dir: Path = Path("data/groups"),
) -> dict:
    graph_path = raw_dir / GRAPH_FILE
    if sha256_file(graph_path) != GRAPH_SHA256:
        raise ValueError("WIQA influence-graph SHA-256 mismatch")
    graphs = {
        graph["graph_id"]: graph
        for graph in (json.loads(line) for line in graph_path.read_text(encoding="utf-8").splitlines())
    }
    if len(graphs) != 2107:
        raise ValueError(f"expected 2107 influence graphs, found {len(graphs)}")

    accepted = []
    rejected = []
    split_ids: dict[str, list[str]] = collections.defaultdict(list)
    reason_counts = collections.Counter()
    raw_counts = {}
    question_type_counts = collections.Counter()
    group_entries = []

    for split, (filename, remote_filename, expected_hash) in QUESTION_FILES.items():
        path = raw_dir / filename
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(f"{filename} SHA-256 mismatch: {actual_hash}")
        items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        raw_counts[split] = len(items)
        for item in items:
            metadata = item["metadata"]
            question_type = metadata["question_type"]
            question_type_counts[question_type] += 1
            if question_type == "OUTOFPARA_DISTRACTOR":
                reason = "out-of-paragraph no-effect distractor has no in-graph intervention and no observed change"
                reason_counts[reason] += 1
                rejected.append({"reason": reason, "record": item})
                continue
            graph = graphs.get(metadata["graph_id"])
            if graph is None:
                reason = "question references a missing influence graph"
                reason_counts[reason] += 1
                rejected.append({"reason": reason, "record": item})
                continue
            resolved = resolve_question(item, graph)
            if resolved is None:
                parsed = parse_stem(item["question"]["stem"])
                if parsed is None:
                    reason = "question stem cannot be parsed"
                else:
                    _, edges, groundings = graph_components(graph)
                    source_text, target_text = parsed
                    source_nodes = groundings[normalized_grounding(source_text)]
                    target_nodes = groundings[normalized_grounding(target_text)]
                    expected_length = int(metadata["path_len"])
                    expected_sign = 1 if item["question"]["answer_label"] == "more" else -1
                    pairs = {
                        (source, target)
                        for source in source_nodes
                        for target in target_nodes
                        if any(
                            length == expected_length and sign == expected_sign
                            for length, sign in signed_paths(source, target, edges)
                        )
                    }
                    reason = (
                        "question maps ambiguously to multiple signed graph node pairs"
                        if len(pairs) > 1
                        else "question text does not resolve to a signed graph path"
                    )
                reason_counts[reason] += 1
                rejected.append({"reason": reason, "record": item})
                continue
            try:
                record = normalize_item(item, graph, split, filename, remote_filename, expected_hash)
            except (KeyError, TypeError, ValueError) as exc:
                reason = f"normalization error: {exc}"
                reason_counts[reason] += 1
                rejected.append({"reason": reason, "record": item})
                continue
            errors = validate_record(record)
            if errors:
                reason = "; ".join(errors)
                reason_counts[reason] += 1
                rejected.append({"reason": reason, "record": record})
                continue
            accepted.append(record)
            graph_id = str(metadata["graph_id"])
            question_id = str(metadata["ques_id"])
            group_entries.append(
                group_entry(
                    record["id"],
                    f"wiqa:graph:{graph_id}",
                    f"wiqa:graph:{graph_id}:question:{question_id}",
                )
            )
            split_ids[split].append(record["id"])

    if raw_counts != {"train": 29808, "validation": 6894, "test": 3003}:
        raise ValueError(f"unexpected raw counts: {raw_counts}")
    if len(accepted) != 26045 or len(rejected) != 13660:
        raise ValueError(f"unexpected conversion counts: accepted={len(accepted)}, rejected={len(rejected)}")
    if len({record["id"] for record in accepted}) != len(accepted):
        raise ValueError("duplicate accepted WIQA IDs")

    write_jsonl(records_dir / "wiqa.jsonl", accepted)
    write_jsonl(records_dir / "wiqa.rejected.jsonl", rejected)
    write_group_manifest(groups_dir / "wiqa.jsonl", group_entries)
    for split in ("train", "validation", "test"):
        write_lines(splits_dir / f"wiqa.{split}.txt", sorted(split_ids[split]))

    summary = {
        "source": "wiqa",
        "raw_question_counts": raw_counts,
        "raw_total": sum(raw_counts.values()),
        "influence_graphs": len(graphs),
        "question_type_counts": dict(sorted(question_type_counts.items())),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "rejection_reason_counts": dict(sorted(reason_counts.items())),
        "split_counts": {split: len(split_ids[split]) for split in ("train", "validation", "test")},
        "split_source": "official no-explanation-v2 train/dev/test files; dev renamed validation",
        "graph_sha256": GRAPH_SHA256,
        "metric_semantics": "direction of change, not state-value match",
        "group_sidecar_entries": len(group_entries),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(summary_path)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/wiqa"))
    parser.add_argument("--records-dir", type=Path, default=Path("data/records"))
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--groups-dir", type=Path, default=Path("data/groups"))
    parser.add_argument("--summary", type=Path, default=Path("reports/wiqa_conversion_summary.json"))
    args = parser.parse_args()
    print(json.dumps(convert(args.raw_dir, args.records_dir, args.splits_dir, args.summary, args.groups_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
