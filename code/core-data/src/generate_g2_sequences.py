"""Generate the large deterministic G2 same-target sequence corpus.

Each row contains an executable binary XOR SCM, a factual world, an ordered
sequence of repeated writes to one target, and truth recomputed from the SCM
under the final write.  Graphs, rather than rows, define the split boundary.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
from pathlib import Path


PROTOCOL = "g2_same_target_last_write_wins_v1"
GENERATOR_SEED = 271


def stable_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def scm_state(graph: dict, roots: dict[str, int], interventions: dict[str, int] | None = None) -> dict[str, int]:
    interventions = interventions or {}
    state: dict[str, int] = {}
    for node in graph["nodes"]:
        if node in interventions:
            state[node] = int(interventions[node])
        elif node in graph["root_nodes"]:
            state[node] = int(roots[node])
        else:
            value = int(graph["xor_bias"][node])
            for parent in graph["parents"][node]:
                value ^= state[parent]
            state[node] = value
    return state


def make_graph(graph_id: int) -> dict:
    rng = random.Random(GENERATOR_SEED * 1_000_003 + graph_id)
    nodes = [f"v{i}" for i in range(8)]
    parents: dict[str, list[str]] = {}
    for index, node in enumerate(nodes):
        candidates = nodes[:index]
        rng.shuffle(candidates)
        parents[node] = sorted(candidates[: min(len(candidates), rng.randint(0, 3))])
    root_nodes = [node for node in nodes if not parents[node]]
    return {
        "graph_id": f"g2-graph-{graph_id:04d}",
        "nodes": nodes,
        "parents": parents,
        "root_nodes": root_nodes,
        "xor_bias": {node: rng.randrange(2) for node in nodes},
    }


def split_for_graph(graph_id: int) -> str:
    if graph_id <= 800:
        return "train"
    if graph_id <= 900:
        return "validation"
    return "test"


def build_rows(graph_count: int, rows_per_graph: int):
    for graph_id in range(1, graph_count + 1):
        graph = make_graph(graph_id)
        split = split_for_graph(graph_id)
        for sequence_id in range(1, rows_per_graph + 1):
            rng = random.Random(GENERATOR_SEED * 10**9 + graph_id * 10_000 + sequence_id)
            roots = {node: rng.randrange(2) for node in graph["root_nodes"]}
            factual = scm_state(graph, roots)
            target = graph["nodes"][rng.randrange(len(graph["nodes"]))]
            length = rng.randint(2, 8)
            values = [rng.randrange(2) for _ in range(length)]
            # Ensure the row actually contains an overwrite, not repeated copies.
            values[0] = 1 - values[-1]
            edits = [{"step": i + 1, "target": target, "value": value} for i, value in enumerate(values)]
            final_state = scm_state(graph, roots, {target: values[-1]})
            yield {
                "id": f"g2-{graph_id:04d}-{sequence_id:03d}",
                "protocol": PROTOCOL,
                "split": split,
                "graph_group_id": graph["graph_id"],
                "world_group_id": f"{graph['graph_id']}:world:{sequence_id:03d}",
                "graph": graph,
                "root_values": roots,
                "factual_state": factual,
                "target": target,
                "ordered_edits": edits,
                "gold_final_state": final_state,
                "law": "last_write_wins",
                "test_evaluated": False,
            }


def generate(output: Path, summary_path: Path, graph_count: int = 1000, rows_per_graph: int = 120) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    counts: collections.Counter[str] = collections.Counter()
    graph_counts: collections.defaultdict[str, set[str]] = collections.defaultdict(set)
    digest = hashlib.sha256()
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in build_rows(graph_count, rows_per_graph):
            line = json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            handle.write(line)
            digest.update(line.encode())
            counts[row["split"]] += 1
            graph_counts[row["split"]].add(row["graph_group_id"])
    temporary.replace(output)
    summary = {
        "protocol": PROTOCOL,
        "generator_seed": GENERATOR_SEED,
        "rows": sum(counts.values()),
        "rows_by_split": dict(counts),
        "graphs_by_split": {key: len(value) for key, value in graph_counts.items()},
        "output_sha256": digest.hexdigest(),
        "split_unit": "graph_group_id",
        "test_rows_generated_but_not_evaluated": counts["test"],
        "test_evaluated": False,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/g2/sequences.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("reports/g2_generation_summary.json"))
    parser.add_argument("--graphs", type=int, default=1000)
    parser.add_argument("--rows-per-graph", type=int, default=120)
    args = parser.parse_args()
    print(json.dumps(generate(args.output, args.summary, args.graphs, args.rows_per_graph), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
