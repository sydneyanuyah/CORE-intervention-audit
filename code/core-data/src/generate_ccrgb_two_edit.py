"""Generate executable ordered two-edit truth from pinned CCR.GB worlds."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any


RAW_SHA256 = "not-published"
PROTOCOL = "ccrgb_executable_two_edit_v1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def pair_id(source: str, graph: str, world: str, first: str, second: str) -> str:
    digest = hashlib.sha256(stable((source, graph, world, first, second)).encode()).hexdigest()[:24]
    return f"two-edit:{digest}"


def split_worlds(worlds: list[dict[str, Any]]) -> dict[int, str]:
    ranked = sorted(
        (int(world["context_id"]) for world in worlds),
        key=lambda context: hashlib.sha256(f"ccrgb-split-v1:{context}".encode()).digest(),
    )
    train_cut = len(ranked) * 8 // 10
    validation_cut = len(ranked) * 9 // 10
    if train_cut < 1 or validation_cut <= train_cut or validation_cut >= len(ranked):
        raise ValueError("CCR.GB needs at least ten worlds for an 80/10/10 split")
    return {
        context: "train" if rank < train_cut else "validation" if rank < validation_cut else "test"
        for rank, context in enumerate(ranked)
    }


def solve(world: dict[str, Any], interventions: dict[str, int]) -> dict[str, int]:
    nodes = world["dag_nodes"]
    matrix = world["dag_adjacency_matrix"]
    exogenous = world["factual_queries"][0]["true_exogenous"]
    names = world["exogenous_variables"]
    state: dict[str, int] = {}
    for index, node in enumerate(nodes):
        if node in interventions:
            state[node] = int(interventions[node])
            continue
        value = int(exogenous[names[index]])
        parents = [nodes[parent] for parent, row in enumerate(matrix) if int(row[index]) == 1]
        for parent in parents:
            value = int(value and state[parent]) if index == len(nodes) - 1 else int(value or state[parent])
        state[node] = value
    return state


def generate(
    raw_path: Path,
    records_path: Path,
    *,
    expected_raw_sha256: str = RAW_SHA256,
    expected_worlds: int = 6000,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if sha256(raw_path) != expected_raw_sha256:
        raise ValueError("CCR.GB raw SHA-256 mismatch")
    worlds = read_jsonl(raw_path)
    if len(worlds) != expected_worlds or any(world.get("generator_commit") != "not-published" for world in worlds):
        raise ValueError("unexpected CCR.GB world inventory or generator commit")
    splits = split_worlds(worlds)
    normalized = read_jsonl(records_path)
    by_context: dict[int, list[dict[str, Any]]] = {}
    for record in normalized:
        context = int(record["source_id"].split(":", 1)[0])
        by_context.setdefault(context, []).append(record)
    artifact_worlds, components, manifest = [], [], []
    for world in worlds:
        split = splits[int(world["context_id"])]
        if split == "test":
            continue
        factual = solve(world, {})
        for query in world["factual_queries"]:
            if query["true_endogenous"] != factual:
                raise ValueError("CCR.GB factual truth disagrees with executable equations")
        raw_single: dict[tuple[str, int], dict[str, int]] = {}
        for query in world["counterfactual_queries"]:
            for value, key in ((0, "cause_false"), (1, "cause_true")):
                expected = solve(world, {query["cause"]: value})
                if query[key]["true_endogenous"] != expected:
                    raise ValueError("CCR.GB single-edit truth disagrees with executable equations")
                if expected != factual:
                    raw_single[(query["cause"], value)] = expected
        records = by_context[int(world["context_id"])]
        component_by_semantic = {}
        leaf = world["dag_nodes"][-1]
        factual_question = next(query["prompt"] for query in world["factual_queries"] if query["effect"] == leaf)
        for semantic, after in sorted(raw_single.items()):
            target, value = semantic
            candidates = [r for r in records if (r["intervention"]["target"], int(r["intervention"]["value"])) == semantic]
            if not candidates:
                raise ValueError("normalized CCR.GB records omit an executable intervention")
            base = sorted(candidates, key=lambda row: row["id"])[0]
            component = dict(base)
            component["id"] = f"ccrgb-two-edit:{world['context_id']}:{target}:{value}"
            component["factual"] = {**base["factual"], "question": factual_question, "answer": "yes" if factual[leaf] else "no"}
            component["intervened"] = {**base["intervened"], "state": after, "answer": "yes" if after[leaf] else "no"}
            component["probes"] = [
                {"variable": node, "question": factual_question if node == leaf else None,
                 "answer_before": factual[node], "answer_after": after[node],
                 "actually_changed": after[node] != factual[node],
                 "required": "may change" if after[node] != factual[node] else "must not change"}
                for node in world["dag_nodes"]
            ]
            component["two_edit_metadata"] = {"protocol": PROTOCOL, "split": split, "context_id": world["context_id"], "test_evaluated": False}
            components.append(component)
            component_by_semantic[semantic] = component
        graph_group = f"ccrgb:context:{world['context_id']}"
        world_group = f"ccrgb:context:{world['context_id']}:world:{world['sample_id']}"
        if split == "validation":
            for first_semantic, second_semantic in itertools.permutations(sorted(raw_single), 2):
                final = solve(world, {first_semantic[0]: first_semantic[1], second_semantic[0]: second_semantic[1]})
                changed = [final[node] != factual[node] for node in world["dag_nodes"]]
                if not any(changed) or all(changed):
                    continue
                first, second = component_by_semantic[first_semantic], component_by_semantic[second_semantic]
                manifest.append({
                    "pair_id": pair_id("ccrgb", graph_group, world_group, first["id"], second["id"]),
                    "source": "ccrgb", "split": "validation", "graph_group_id": graph_group,
                    "world_group_id": world_group, "first_record_id": first["id"], "second_record_id": second["id"],
                    "factual_outputs": [factual[node] for node in world["dag_nodes"]],
                    "gold_outputs": [final[node] for node in world["dag_nodes"]],
                    "change_mask": changed, "preservation_mask": [not flag for flag in changed],
                    "target_mask": [node in {first_semantic[0], second_semantic[0]} for node in world["dag_nodes"]],
                    "truth_provenance": {"protocol": PROTOCOL, "raw_sha256": expected_raw_sha256, "test_evaluated": False},
                })
        artifact_worlds.append({
            "context_id": world["context_id"], "sample_id": world["sample_id"], "split": split,
            "dag_nodes": world["dag_nodes"], "dag_adjacency_matrix": world["dag_adjacency_matrix"],
            "exogenous_variables": world["exogenous_variables"],
            "true_exogenous": world["factual_queries"][0]["true_exogenous"], "factual_state": factual,
        })
    artifact = {"protocol": PROTOCOL, "raw_sha256": expected_raw_sha256, "test_evaluated": False,
                "worlds": artifact_worlds, "components": sorted(components, key=lambda row: row["id"])}
    manifest.sort(key=lambda row: row["pair_id"])
    if not manifest or any(row["split"] != "validation" for row in manifest):
        raise ValueError(f"expected non-empty validation-only eligible pairs, found {len(manifest)}")
    return artifact, manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--records", type=Path, default=Path("data/records/ccrgb.jsonl"))
    parser.add_argument("--artifact", type=Path, default=Path("data/two_edit/ccrgb/artifact.json"))
    parser.add_argument("--manifest", type=Path, default=Path("data/two_edit/ccrgb/manifest.jsonl"))
    args = parser.parse_args()
    artifact, manifest = generate(args.raw, args.records)
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    args.artifact.write_text(json.dumps(artifact, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    args.manifest.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in manifest), encoding="utf-8")
    print(json.dumps({"pairs": len(manifest), "artifact_sha256": sha256(args.artifact),
                      "manifest_sha256": sha256(args.manifest), "test_evaluated": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
