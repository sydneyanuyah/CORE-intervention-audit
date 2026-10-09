#!/usr/bin/env python3
"""Export the exact validation composed-pair suite required by T1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from core_bert.xor_two_edit import generate_xor_two_edit_manifest_from_directory


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--f2-artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = args.root / "data"
    rows: list[dict] = []
    sources: dict[str, str] = {}

    for graph_seed in range(3061, 3081):
        directory = args.f2_artifacts / f"graph_{graph_seed}"
        for row in generate_xor_two_edit_manifest_from_directory(directory):
            if row["split"] == "validation":
                rows.append({"protocol": "t1_f2_composed_pairs_v1", "family": "xor", "split": "validation", "pair_id": row["pair_id"], "pair": row})
    sources["xor_f2_provenance"] = sha256(args.f2_artifacts / "provenance.json")

    for family, relative in (
        ("ccrgb", "two_edit/ccrgb/manifest.jsonl"),
        ("cladder", "two_edit/cladder/manifest.jsonl"),
    ):
        path = data / relative
        sources[family] = sha256(path)
        for row in read_jsonl(path):
            if row["split"] == "validation":
                rows.append({"protocol": "t1_f2_composed_pairs_v1", "family": family, "split": "validation", "pair_id": row["pair_id"], "pair": row})

    queue_path = data / "two_edit/manual/annotation_queue.jsonl"
    truth_path = data / "two_edit/manual/authoritative_manual_truth.jsonl"
    queue = {row["pair_id"]: row for row in read_jsonl(queue_path)}
    truths = read_jsonl(truth_path)
    sources["manual_queue"] = sha256(queue_path)
    sources["manual_truth"] = sha256(truth_path)
    for truth in truths:
        pair_id = truth["pair_id"]
        if pair_id not in queue:
            raise ValueError(f"truth without pair input: {pair_id}")
        family = truth["source"]
        if family not in {"wiqa", "com2"}:
            raise ValueError(f"unexpected manual family: {family}")
        rows.append({"protocol": "t1_f2_composed_pairs_v1", "family": family, "split": "validation", "pair_id": pair_id, "pair": queue[pair_id], "truth": truth})

    identities = [(row["family"], row["pair_id"]) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate family/pair identity")
    expected = {"xor": 1200, "ccrgb": 2974, "cladder": 615, "wiqa": 70, "com2": 70}
    observed = {family: sum(row["family"] == family for row in rows) for family in expected}
    if observed != expected:
        raise ValueError(f"unexpected T1 family counts: {observed}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temp, args.output)
    provenance = {
        "protocol": "t1_f2_composed_pairs_v1",
        "split": "validation",
        "test_evaluated": False,
        "records": len(rows),
        "family_counts": observed,
        "source_sha256": sources,
        "output_sha256": sha256(args.output),
    }
    args.output.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(provenance, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
