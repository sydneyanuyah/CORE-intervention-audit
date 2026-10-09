"""Freeze executable C2 floor predictions on the C1 validation law suite."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path


PROTOCOL = "c2_law_floor_calibration_v1"
RANDOM_SEED = 271


def random_state(record: dict) -> dict[str, int]:
    nodes = sorted(record["factual_state"])
    key = f"{RANDOM_SEED}:{record['graph_group_id']}:{record['record_id']}".encode()
    raw = hashlib.shake_256(key).digest((len(nodes) + 7) // 8)
    return {node: (raw[index // 8] >> (index % 8)) & 1 for index, node in enumerate(nodes)}


def floors(record: dict) -> dict[str, dict[str, int]]:
    factual = record["factual_state"]
    value = int(record["intervention"]["value"])
    return {
        "do_nothing": dict(factual),
        "do_everything": {node: value for node in factual},
        "random_init": random_state(record),
    }


def generate(c1_root: Path, output: Path, summary_path: Path) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(); count = 0; graphs = set(); correct = collections.Counter(); total = collections.Counter()
    with output.open("w", encoding="utf-8") as handle:
        for path in sorted((c1_root / "real").glob("graph_*.jsonl")):
            for line in path.open(encoding="utf-8"):
                source = json.loads(line)
                if source["split"] != "validation": continue
                predictions = floors(source); gold = source["intervened_state"]
                for method, state in predictions.items():
                    for node in gold:
                        correct[method] += state[node] == gold[node]; total[method] += 1
                row = {
                    "id": f"c2:{source['graph_group_id']}:{source['record_id']}",
                    "protocol": PROTOCOL, "split": "validation",
                    "graph_group_id": source["graph_group_id"],
                    "source_record_id": source["record_id"],
                    "factual_state": source["factual_state"],
                    "intervention": source["intervention"],
                    "gold_state": gold, "floor_predictions": predictions,
                    "random_init_seed": RANDOM_SEED, "test_evaluated": False,
                }
                encoded = json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                handle.write(encoded); digest.update(encoded.encode()); count += 1; graphs.add(row["graph_group_id"])
    summary = {
        "protocol": PROTOCOL, "cases": count, "graphs": len(graphs),
        "methods": ["do_nothing", "do_everything", "random_init", "trained_o3"],
        "materialized_methods": ["do_nothing", "do_everything", "random_init"],
        "trained_o3_status": "requires checkpoint-bound measurement; not a data-generation operation",
        "floor_variable_accuracy": {method: correct[method] / total[method] for method in sorted(total)},
        "output_sha256": digest.hexdigest(), "test_evaluated": False,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c1-root", type=Path, default=Path("data/c1/controls"))
    parser.add_argument("--output", type=Path, default=Path("data/c2/floor_suite.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("reports/c2_floor_suite_summary.json"))
    args = parser.parse_args()
    print(json.dumps(generate(args.c1_root, args.output, args.summary), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
