#!/usr/bin/env python3
"""Build a deterministic, label-blinded WIQA/Com2 two-edit annotation queue."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


WIQA_GRAPH_SHA256 = "not-published"
WIQA_VALUE_TOKEN_BY_NODE = {
    "V": "less", "Z": "more", "X": "more", "U": "less",
    "W": "less", "Y": "more", "A": "more", "D": "less",
}


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def validation_ids(data_root: Path, names: list[str]) -> set[str]:
    ids: set[str] = set()
    for name in names:
        path = data_root / "splits" / f"{name}.validation.txt"
        ids.update(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return ids


def pair_id(source: str, record_id: str, second_target: str, second_value: Any) -> str:
    payload = json.dumps([source, record_id, second_target, second_value], sort_keys=True)
    return f"manual-two-edit:{source}:{hashlib.sha256(payload.encode()).hexdigest()[:24]}"


def public_first_edit(record: dict[str, Any]) -> dict[str, Any]:
    edit = record["intervention"]
    return {
        "target": edit["target"], "target_text": edit["target_text"],
        "value": edit["value"], "value_token": edit["value_token"],
        "kind": edit["kind"], "formal": edit["formal"], "text": edit["text"],
    }


def wiqa_candidate(
    record: dict[str, Any], signed_edges: list[dict[str, str]] | None = None,
) -> dict[str, Any] | None:
    graph = record.get("graph") or {}
    node_text = graph.get("node_text") or {}
    first_target = record["intervention"]["target"]
    ordered = list(record.get("descendants") or []) + sorted(graph.get("nodes") or [])
    seen = set()
    choices = []
    for target in ordered:
        if target == first_target or target in seen:
            continue
        seen.add(target)
        surfaces = sorted({value.strip() for value in node_text.get(target, []) if isinstance(value, str) and value.strip()})
        if surfaces:
            choices.append((target, surfaces[0]))
    if not choices:
        return None
    second_target, second_value = choices[0]
    public_graph = dict(graph)
    if signed_edges is not None:
        public_graph["signed_edges"] = signed_edges
    result = {
        "pair_id": pair_id("wiqa", record["id"], second_target, second_value),
        "source": "wiqa", "split": "validation", "source_record_id": record["id"],
        "structure_kind": "dag", "factual_passage": record["factual"]["passage"],
        "factual_question": record["factual"]["question"], "graph": public_graph,
        "first_edit": public_first_edit(record),
        "second_edit": {
            "target": second_target, "target_text": second_value, "value": second_value,
            "value_token": WIQA_VALUE_TOKEN_BY_NODE[second_target],
            "kind": "value_set", "formal": f'do({second_target} = {json.dumps(second_value)})',
            "text": f'After the first change, also suppose {second_value}.',
        },
        "annotation_requirements": {
            "output_vocabulary": ["more", "less", "no_effect"],
            "apply_order": ["first_edit", "second_edit"],
            "label_intermediate_and_final": True,
            "signed_edge_semantics": {
                "positive": "preserve more/less direction",
                "negative": "reverse more/less direction",
                "no_directed_effect": "no_effect",
                "conflicting_paths": "needs_adjudication",
            },
            "baseline": "All node outputs are directional changes relative to the factual world; unchanged nodes are no_effect.",
        },
    }
    return result


def load_wiqa_signed_edges(data_root: Path) -> dict[str, list[dict[str, str]]]:
    """Reconstruct released WIQA edge signs and bind them to normalized record IDs."""
    graph_path = data_root / "raw" / "wiqa" / "wiqa_influence_graphs.jsonl"
    if hashlib.sha256(graph_path.read_bytes()).hexdigest() != WIQA_GRAPH_SHA256:
        raise ValueError("WIQA influence-graph SHA-256 mismatch")
    raw_graphs = {str(row["graph_id"]): row for row in read_jsonl(graph_path)}
    record_to_graph = {}
    for row in read_jsonl(data_root / "groups" / "wiqa.jsonl"):
        prefix = "wiqa:graph:"
        group = row["graph_group_id"]
        if not group.startswith(prefix):
            raise ValueError(f"unexpected WIQA graph group {group!r}")
        record_to_graph[row["record_id"]] = group[len(prefix):]
    result = {}
    for record_id, graph_id in record_to_graph.items():
        graph = raw_graphs[graph_id]
        y_positive = graph.get("Y_affects_outcome") in ("a", "more", True)
        triples = [
            ("V", "X", -1), ("Z", "X", 1), ("X", "W", -1),
            ("X", "Y", 1), ("U", "Y", -1),
            ("W", "A", -1 if y_positive else 1),
            ("W", "D", 1 if y_positive else -1),
            ("Y", "A", 1 if y_positive else -1),
            ("Y", "D", -1 if y_positive else 1),
        ]
        result[record_id] = [
            {"source": parent, "target": child, "sign": "positive" if sign == 1 else "negative"}
            for parent, child, sign in triples
        ]
    return result


def com2_candidate(record: dict[str, Any]) -> dict[str, Any] | None:
    chain = record.get("chain")
    if not isinstance(chain, list) or len(chain) < 3:
        return None
    first_target = record["intervention"]["target"]
    try:
        index = int(first_target.split(":", 1)[0].split("_", 1)[1])
    except (IndexError, ValueError):
        return None
    second_index = index + 1 if index + 1 < len(chain) else index - 1
    if second_index < 0 or second_index == index:
        return None
    second_value = chain[second_index]
    second_target = f"event_{second_index:02d}: {second_value}"
    passage = record["factual"]["passage"].split("\n[ORIG]", 1)[0].rstrip()
    return {
        "pair_id": pair_id("com2", record["id"], second_target, second_value),
        "source": "com2", "split": "validation", "source_record_id": record["id"],
        "structure_kind": "chain", "factual_passage": passage,
        "factual_question": record["factual"]["question"], "chain": chain,
        "first_edit": public_first_edit(record),
        "second_edit": {
            "target": second_target, "target_text": second_value, "value": second_value,
            "kind": "event_replace", "formal": f'do(event_{second_index:02d} = {json.dumps(second_value)})',
            "text": f'After the first change, force step {second_index + 1} to occur as: {second_value}.',
        },
        "annotation_requirements": {
            "output_vocabulary": "concise event text", "apply_order": ["first_edit", "second_edit"],
            "label_intermediate_and_final": True,
        },
    }


def select(rows: Iterable[dict[str, Any]], valid_ids: set[str], builder, count: int) -> list[dict[str, Any]]:
    candidates = [item for row in rows if row["id"] in valid_ids if (item := builder(row)) is not None]
    candidates.sort(key=lambda row: hashlib.sha256(("manual-queue-v1:" + row["pair_id"]).encode()).hexdigest())
    if len(candidates) < count:
        raise ValueError(f"only {len(candidates)} eligible validation candidates; requested {count}")
    return candidates[:count]


def build(data_root: Path, per_source: int) -> list[dict[str, Any]]:
    signs = load_wiqa_signed_edges(data_root)
    def signed_wiqa(record: dict[str, Any]) -> dict[str, Any] | None:
        row = wiqa_candidate(record, signs[record["id"]])
        if row is not None:
            unsigned = {tuple(edge) for edge in row["graph"]["edges"]}
            signed = {(edge["source"], edge["target"]) for edge in row["graph"]["signed_edges"]}
            if unsigned != signed:
                raise ValueError(f"{record['id']}: signed edges do not exactly match normalized graph")
        return row
    wiqa = select(read_jsonl(data_root / "records" / "wiqa.jsonl"), validation_ids(data_root, ["wiqa"]), signed_wiqa, per_source)
    com2_ids = validation_ids(data_root, ["com2_intervention", "com2_counterfactual"])
    com2_rows = list(read_jsonl(data_root / "records" / "com2_intervention.jsonl")) + list(read_jsonl(data_root / "records" / "com2_counterfactual.jsonl"))
    com2 = select(com2_rows, com2_ids, com2_candidate, per_source)
    rows = wiqa + com2
    rows.sort(key=lambda row: row["pair_id"])
    if len({row["pair_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate annotation pair IDs")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-source", type=int, default=70)
    parser.add_argument("--wiqa-only", action="store_true")
    args = parser.parse_args()
    rows = build(args.data_root, args.per_source)
    if args.wiqa_only:
        rows = [row for row in rows if row["source"] == "wiqa"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(json.dumps({"rows": len(rows), "per_source": args.per_source, "sha256": digest, "test_evaluated": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
