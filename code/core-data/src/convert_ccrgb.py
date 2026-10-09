"""Convert the isolated CCR.GB 20-world feasibility output to CORE records."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from groups import group_entry, write_group_manifest
from schema import validate_record


SOURCE_URL = "https://github.com/jmaasch/compositional_causal_reasoning/tree/not-published"
SOURCE_SHA256 = "not-published"
SOURCE_FILE = "ccrgb_6000_worlds.jsonl"
FETCHED_UTC = "2026-09-08T00:00:00Z"
CONVERTER_VERSION = 3


def exact_span(passage: str, surface: str) -> list[int]:
    """Return deterministic character offsets for an exact surface occurrence."""

    start = passage.find(surface)
    if start < 0:
        raise ValueError(f"target surface {surface!r} is absent from factual passage")
    return [start, start + len(surface)]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def graph_from_world(world: dict[str, Any]) -> dict[str, list]:
    nodes = world["dag_nodes"]
    matrix = world["dag_adjacency_matrix"]
    if len(matrix) != len(nodes) or any(len(row) != len(nodes) for row in matrix):
        raise ValueError("DAG adjacency dimensions do not match node count")
    edges = [
        [nodes[parent], nodes[child]]
        for parent, row in enumerate(matrix)
        for child, value in enumerate(row)
        if int(value) == 1
    ]
    if any(value not in (0, 1) for row in matrix for value in row):
        raise ValueError("DAG adjacency matrix is not binary")
    return {"nodes": list(nodes), "edges": edges}


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


def world_splits(worlds: list[dict[str, Any]]) -> dict[int, str]:
    ranked = sorted(
        (int(world["context_id"]) for world in worlds),
        key=lambda context_id: hashlib.sha256(
            f"ccrgb-split-v1:{context_id}".encode("utf-8")
        ).digest(),
    )
    train_cut = len(ranked) * 8 // 10
    validation_cut = len(ranked) * 9 // 10
    if train_cut < 1 or validation_cut <= train_cut or validation_cut >= len(ranked):
        raise ValueError("CCR.GB needs at least ten worlds for an 80/10/10 split")
    return {
        context_id: "train" if rank < train_cut else "validation" if rank < validation_cut else "test"
        for rank, context_id in enumerate(ranked)
    }


def bool_answer(value: int) -> str:
    return "yes" if int(value) == 1 else "no"


def normalize_item(
    world: dict[str, Any], pair: dict[str, Any], pair_index: int, action: int
) -> dict[str, Any]:
    graph = graph_from_world(world)
    target = pair["cause"]
    effect = pair["effect"]
    query = pair["cause_true"] if action else pair["cause_false"]
    factual_query = next(
        (item for item in world["factual_queries"] if item["effect"] == effect), None
    )
    if factual_query is None:
        raise ValueError(f"missing factual query for effect {effect!r}")
    factual_state = factual_query["true_endogenous"]
    intervened_state = query["true_endogenous"]
    if set(factual_state) != set(graph["nodes"]) or set(intervened_state) != set(graph["nodes"]):
        raise ValueError("state variables do not match DAG nodes")

    descendants = descendants_of(target, graph["edges"])
    non_descendants = sorted(set(graph["nodes"]) - {target} - set(descendants))
    probes = []
    for variable in graph["nodes"]:
        before = int(factual_state[variable])
        after = int(intervened_state[variable])
        probes.append(
            {
                "variable": variable,
                "question": query["prompt"] if variable == effect else None,
                "answer_before": before,
                "answer_after": after,
                "required": "must not change" if variable in non_descendants else "may change",
                "actually_changed": before != after,
            }
        )
    context_id = int(world["context_id"])
    sample_id = int(world["sample_id"])
    passage = " ".join((world["causal_context"].strip(), world["sample_context"].strip()))
    return {
        "id": f"ccrgb-clinical-{context_id:03d}-{sample_id:03d}-{pair_index:02d}-do{action}",
        "source": "ccrgb",
        "source_id": f"{context_id}:{sample_id}:{pair_index}:{action}",
        "structure_kind": "dag",
        "graph": graph,
        "chain": None,
        "factual": {
            "passage": passage,
            "state": {name: int(value) for name, value in factual_state.items()},
            "question": factual_query["prompt"],
            "answer": bool_answer(factual_query["true_response"]),
        },
        "intervention": {
            "target": target,
            "target_text": target,
            "target_span": exact_span(passage, target),
            "value": action,
            "value_token": str(action),
            "replacement_span": None,
            "kind": "value_set",
            "formal": f"do({target} = {action})",
            "text": query["prompt"],
        },
        "intervened": {
            "passage": None,
            "state": {name: int(value) for name, value in intervened_state.items()},
            "answer": bool_answer(query["true_response"]),
        },
        "descendants": descendants,
        "non_descendants": non_descendants,
        "probes": probes,
        "provenance": {
            "fetched_utc": FETCHED_UTC,
            "url": SOURCE_URL,
            "file": SOURCE_FILE,
            "sha256": SOURCE_SHA256,
            "converter": "src/convert_ccrgb.py",
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
    input_path: Path,
    records_dir: Path,
    splits_dir: Path,
    summary_path: Path,
    groups_dir: Path = Path("data/groups"),
) -> dict:
    actual_hash = sha256_file(input_path)
    if actual_hash != SOURCE_SHA256:
        raise ValueError(f"CCR.GB raw SHA-256 mismatch: {actual_hash}")
    worlds = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line]
    if len(worlds) != 6000 or len({world["context_id"] for world in worlds}) != 6000:
        raise ValueError("expected 6000 unique CCR.GB worlds")
    if any(world["generator_commit"] != "not-published" for world in worlds):
        raise ValueError("unexpected CCR.GB generator commit")
    split_by_world = world_splits(worlds)
    accepted = []
    rejected = []
    split_ids: dict[str, list[str]] = collections.defaultdict(list)
    group_entries = []
    generated = 0
    for world in worlds:
        for pair_index, pair in enumerate(world["counterfactual_queries"]):
            for action in (0, 1):
                generated += 1
                try:
                    record = normalize_item(world, pair, pair_index, action)
                except (KeyError, StopIteration, TypeError, ValueError) as exc:
                    rejected.append({"reason": f"normalization error: {exc}", "record": pair})
                    continue
                errors = validate_record(record)
                if errors:
                    rejected.append({"reason": "; ".join(errors), "record": record})
                    continue
                accepted.append(record)
                context_id = int(world["context_id"])
                sample_id = int(world["sample_id"])
                group_entries.append(
                    group_entry(
                        record["id"],
                        f"ccrgb:context:{context_id}",
                        f"ccrgb:context:{context_id}:world:{sample_id}",
                    )
                )
                split_ids[split_by_world[int(world["context_id"])]].append(record["id"])

    expected_pairs = sum(len(world["counterfactual_queries"]) for world in worlds)
    if generated != expected_pairs * 2 or len(accepted) != expected_pairs or len(rejected) != expected_pairs:
        raise ValueError(
            f"unexpected CCR.GB counts: generated={generated}, accepted={len(accepted)}, rejected={len(rejected)}"
        )
    rejection_counts = collections.Counter(item["reason"] for item in rejected)
    expected_reason = "record has no may-change probe with an observed change"
    if rejection_counts != {expected_reason: expected_pairs}:
        raise ValueError(f"unexpected rejection reasons: {rejection_counts}")

    write_jsonl(records_dir / "ccrgb.jsonl", accepted)
    write_jsonl(records_dir / "ccrgb.rejected.jsonl", rejected)
    write_group_manifest(groups_dir / "ccrgb.jsonl", group_entries)
    for split in ("train", "validation", "test"):
        write_lines(splits_dir / f"ccrgb.{split}.txt", sorted(split_ids[split]))

    summary = {
        "source": "ccrgb",
        "generator_commit": worlds[0]["generator_commit"],
        "generator_domain": worlds[0]["generator_domain"],
        "generator_seed": worlds[0]["generator_seed"],
        "raw_sha256": actual_hash,
        "worlds": len(worlds),
        "counterfactual_pairs": sum(len(world["counterfactual_queries"]) for world in worlds),
        "candidate_interventions": generated,
        "accepted": len(accepted),
        "rejected": len(rejected),
        "rejection_reason_counts": dict(rejection_counts),
        "split_counts": {split: len(split_ids[split]) for split in ("train", "validation", "test")},
        "split_world_counts": dict(collections.Counter(split_by_world.values())),
        "split_group": "context_id world",
        "group_sidecar_entries": len(group_entries),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(summary_path)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--records-dir", type=Path, default=Path("data/records"))
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--groups-dir", type=Path, default=Path("data/groups"))
    parser.add_argument("--summary", type=Path, default=Path("reports/ccrgb_conversion_summary.json"))
    args = parser.parse_args()
    print(json.dumps(convert(args.input, args.records_dir, args.splits_dir, args.summary, args.groups_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
