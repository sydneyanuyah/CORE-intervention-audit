"""Build C1 real/placebo/shuffled controls from frozen F3 development artifacts."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path


PROTOCOL = "c1_fixed_o3_closing_controls_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shuffled(rows: list[dict], graph_seed: int) -> list[dict]:
    """Derange outcomes within split/target/value strata, preserving marginals."""
    groups: collections.defaultdict[tuple, list[int]] = collections.defaultdict(list)
    for index, row in enumerate(rows):
        intervention = row["intervention"]
        groups[(row["split"], intervention["target"], int(intervention["value"]))].append(index)
    output = [dict(row) for row in rows]
    for key, indices in groups.items():
        if len(indices) < 2:
            raise ValueError(f"cannot derange singleton stratum {key}")
        ordered = sorted(indices, key=lambda index: hashlib.sha256(
            f"c1:{graph_seed}:{key}:{rows[index]['world_id']}".encode()
        ).hexdigest())
        donors = ordered[1:] + ordered[:1]
        for recipient_index, donor_index in zip(ordered, donors):
            recipient, donor = rows[recipient_index], rows[donor_index]
            if recipient["world_id"] == donor["world_id"]:
                raise ValueError("shuffle produced a fixed point")
            factual = recipient.get("factual_state")
            # F3 records store factual states in worlds.json, attached by generate().
            if not isinstance(factual, dict):
                raise ValueError("recipient factual state is missing")
            state = dict(donor["intervened_state"])
            out = dict(recipient)
            out["record_id"] = f"shuffled-{recipient['record_id']}"
            out["intervened_state"] = state
            out["changed_variables"] = sorted(node for node in state if state[node] != factual[node])
            out["required_invariant_variables"] = []
            out["shuffle_provenance"] = {
                "donor_record_id": donor["record_id"],
                "donor_world_id": donor["world_id"],
                "stratum": {"split": key[0], "target": key[1], "value": key[2]},
                "construction": "one-position cyclic derangement after deterministic SHA-256 ordering",
            }
            output[recipient_index] = out
    return output


def write_jsonl(path: Path, rows: list[dict]) -> str:
    digest = hashlib.sha256(); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            line = json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            handle.write(line); digest.update(line.encode())
    return digest.hexdigest()


def generate(source_root: Path, output_root: Path, summary_path: Path) -> dict:
    condition_counts: collections.Counter[str] = collections.Counter()
    split_counts: collections.Counter[str] = collections.Counter()
    artifacts = {}
    for graph_seed in range(3081, 3101):
        directory = source_root / f"graph_{graph_seed}"
        worlds = json.loads((directory / "worlds.json").read_text())
        factual = {world["world_id"]: world["factual_state"] for world in worlds}
        real = json.loads((directory / "records_real.json").read_text())
        placebo = json.loads((directory / "records_placebo.json").read_text())
        for rows in (real, placebo):
            for row in rows:
                row["factual_state"] = factual[row["world_id"]]
        conditions = {"real": real, "noncausal_placebo": placebo, "shuffled": shuffled(real, graph_seed)}
        for condition, rows in conditions.items():
            for row in rows:
                row["condition"] = condition
                row["graph_group_id"] = f"xor:graph:{graph_seed}"
                row["protocol"] = PROTOCOL
                row["test_evaluated"] = False
                condition_counts[condition] += 1; split_counts[f"{condition}:{row['split']}"] += 1
            path = output_root / condition / f"graph_{graph_seed}.jsonl"
            artifacts[str(path.relative_to(output_root))] = write_jsonl(path, rows)
    summary = {
        "protocol": PROTOCOL,
        "graph_seeds": list(range(3081, 3101)),
        "graph_count": 20,
        "condition_counts": dict(condition_counts),
        "split_counts": dict(split_counts),
        "total_records": sum(condition_counts.values()),
        "artifacts": artifacts,
        "shuffled_semantics": "outcomes deranged within graph/split/target/value; all outcome marginals preserved",
        "test_evaluated": False,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/c1/controls"))
    parser.add_argument("--summary", type=Path, default=Path("reports/c1_controls_summary.json"))
    args = parser.parse_args()
    print(json.dumps(generate(args.source_root, args.output_root, args.summary), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
