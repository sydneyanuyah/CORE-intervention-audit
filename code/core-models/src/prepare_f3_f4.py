#!/usr/bin/env python3
"""Materialize fresh validation-only F3/F4 XOR artifacts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_core(path: Path):
    spec = importlib.util.spec_from_file_location("canonical_f3_f4_core", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def is_dag(nodes, support_map):
    children = {node: set() for node in nodes}
    for target, support in support_map.items():
        for source in support:
            if source != target:
                children[source].add(target)
    indegree = {node: 0 for node in nodes}
    for source in nodes:
        for target in children[source]: indegree[target] += 1
    queue = [node for node in nodes if indegree[node] == 0]; seen = 0
    while queue:
        node = queue.pop(); seen += 1
        for child in children[node]:
            indegree[child] -= 1
            if indegree[child] == 0: queue.append(child)
    return seen == len(nodes)


def make_placebo_spec(core, graph, train_worlds, seed):
    nodes = graph["nodes"]; descendants = core.descendant_map(graph)
    for attempt in range(1000):
        rng = random.Random(seed + attempt); targets = {}
        for target in nodes:
            real_support_size = 1 + len(descendants[target])
            pool = [node for node in nodes if node != target]; rng.shuffle(pool)
            support = [target, *pool[:real_support_size - 1]]
            flips = []
            for world in train_worlds:
                value = 1 - world["factual_state"][target]
                after = core.scm_state(graph, world["root_values"], {target: value})
                flips.append(sum(after[node] != world["factual_state"][node] for node in nodes))
            changed_size = max(1, min(real_support_size, round(sum(flips) / len(flips))))
            changed_pool = [node for node in support if node != target]; rng.shuffle(changed_pool)
            targets[target] = {
                "support": support,
                "changed_on_target_flip": [target, *changed_pool[:changed_size - 1]],
                "real_support_size": real_support_size,
                "real_mean_flip_count": sum(flips) / len(flips),
            }
        if not is_dag(nodes, {node: targets[node]["support"] for node in nodes}):
            return {"seed": seed + attempt, "construction": "fixed cyclic random target-to-support mapping; exact support-size match and rounded mean flip-count match per target", "is_dag": False, "targets": targets}
    raise RuntimeError("failed to construct non-DAG placebo")


def apply_placebo(state, target, value, spec):
    out = dict(state)
    if out[target] != value:
        for node in spec["targets"][target]["changed_on_target_flip"]: out[node] = 1 - out[node]
        out[target] = int(value)
    return out


def make_placebo_records(core, graph, worlds, spec):
    rows = []; nodes = graph["nodes"]
    for world in worlds:
        for target in nodes:
            support = set(spec["targets"][target]["support"])
            for value in (0, 1):
                after = apply_placebo(world["factual_state"], target, value, spec)
                rows.append({
                    "record_id": f"placebo-{world['world_id']}-{target}-{value}", "world_id": world["world_id"],
                    "split": world["split"], "intervention": {"formal": f"edit({target}={value})", "text": core.intervention_text(target, value), "target": target, "value": value},
                    "intervened_state": after,
                    "changed_variables": [node for node in nodes if after[node] != world["factual_state"][node]],
                    "required_invariant_variables": [node for node in nodes if node not in support],
                })
    return rows


def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--experiment", choices=("f3", "f4"), required=True)
    p.add_argument("--core-components", type=Path, required=True); p.add_argument("--expected-core-sha256", required=True)
    p.add_argument("--output-root", type=Path, required=True); args = p.parse_args()
    if sha256(args.core_components) != args.expected_core_sha256: raise ValueError("core source hash mismatch")
    core = load_core(args.core_components)
    graph_seeds = range(3081, 3101) if args.experiment == "f3" else range(3101, 3126)
    provenance = {}
    for graph_seed in graph_seeds:
        graph = core.make_graph(seed=graph_seed)
        worlds = [row for row in core.make_worlds(graph, graph_seed) if row["split"] in {"train", "validation"}]
        real = core.make_records(graph, worlds)
        directory = args.output_root / f"graph_{graph_seed}"; directory.mkdir(parents=True, exist_ok=True)
        payloads = {"graph.json": graph, "worlds.json": worlds, "records_real.json": real}
        if args.experiment == "f3":
            spec = make_placebo_spec(core, graph, [w for w in worlds if w["split"] == "train"], graph_seed + 7919)
            payloads.update({"placebo_spec.json": spec, "records_placebo.json": make_placebo_records(core, graph, worlds, spec)})
        for name, payload in payloads.items():
            path = directory / name; path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            provenance[str(path)] = sha256(path)
    out = args.output_root / "provenance.json"
    out.write_text(json.dumps({"protocol": f"{args.experiment}_artifacts_v1", "allowed_splits": ["train", "validation"], "artifacts": provenance, "test_evaluated": False}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"experiment": args.experiment, "graphs": len(graph_seeds), "provenance_sha256": sha256(out)}))
    return 0


if __name__ == "__main__": raise SystemExit(main())
