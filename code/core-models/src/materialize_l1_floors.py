#!/usr/bin/env python3
"""Materialize preregistered symbolic L1 floors on validation states only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from core_bert.benchmark_data import load_benchmark_split
from core_bert.l1_data import binary_intervention
from train_addressed import load_xor_training_split


FLOORS = {"do_nothing", "do_everything", "random_init"}


def random_state(seed: int, state: tuple[int, ...], target: int, value: int) -> tuple[int, ...]:
    raw = hashlib.sha256(f"{seed}:{state}:{target}:{value}".encode()).digest()
    return tuple((raw[index // 8] >> (index % 8)) & 1 for index in range(len(state)))


def apply_floor(method: str, seed: int, state: tuple[int, ...], target: int, value: int) -> tuple[int, ...]:
    if method == "do_nothing":
        return state
    if method == "do_everything":
        return state if target < 0 else tuple(value for _ in state)
    return random_state(seed, state, target, value)


def summarize(method: str, seed: int, examples) -> dict:
    task_correct = task_total = changed_correct = changed_total = preserved_correct = preserved_total = 0
    laws = {name: [0, 0] for name in ("identity", "idempotence", "commutation", "last_write_wins")}
    graphs = set()
    for example in examples:
        record, nodes = example.record, example.record["graph"]["nodes"]
        before_map, after_map = record["factual"].get("state"), record["intervened"].get("state")
        if not isinstance(before_map, dict) or not isinstance(after_map, dict):
            raise ValueError(f"{example.record_id}: floors require explicit before/after state")
        before = tuple(int(before_map[node]) for node in nodes)
        after = tuple(int(after_map[node]) for node in nodes)
        if set(before + after) - {0, 1}:
            raise ValueError(f"{example.record_id}: floors require binary states")
        target, value = binary_intervention(record)
        predicted = apply_floor(method, seed, before, target, value)
        graphs.add(example.graph_group_id)
        for index, (gold, guess) in enumerate(zip(after, predicted)):
            task_total += 1; task_correct += gold == guess
            if before[index] != gold:
                changed_total += 1; changed_correct += gold == guess
            else:
                preserved_total += 1; preserved_correct += gold == guess
        identity = apply_floor(method, seed, before, -1, 0)
        once = apply_floor(method, seed, before, target, value)
        twice = apply_floor(method, seed, once, target, value)
        other = next(index for index in range(len(nodes)) if index != target)
        commute_lr = apply_floor(method, seed, once, other, value)
        commute_rl = apply_floor(method, seed, apply_floor(method, seed, before, other, value), target, value)
        overwrite = apply_floor(method, seed, apply_floor(method, seed, before, target, 1 - value), target, value)
        for name, passed in (
            ("identity", identity == before), ("idempotence", twice == once),
            ("commutation", commute_lr == commute_rl), ("last_write_wins", overwrite == once),
        ):
            laws[name][1] += 1; laws[name][0] += passed
    change = changed_correct / max(changed_total, 1)
    preservation = preserved_correct / max(preserved_total, 1)
    return {
        "variable_accuracy": task_correct / task_total,
        "change_accuracy": change,
        "preservation_accuracy": preservation,
        "balanced_intervention_score": 0.5 * (change + preservation),
        "graph_count": len(graphs),
        "record_count": len(examples),
        "law_correctness": {name: passed / total for name, (passed, total) in laws.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    args = parser.parse_args()
    registry = json.loads(args.cells.read_text())
    cladder = load_benchmark_split(args.workdir / "data", "validation", sources=["cladder"])
    count = 0
    for cell in registry["cells"]:
        method = cell["method"]
        if method not in FLOORS:
            continue
        family, seed = cell["family"], int(cell["seed"])
        if family == "xor":
            graph = int(cell["graph_seed"])
            examples = load_xor_training_split([[
                str(args.workdir / f"${PRIVATE_STORAGE_ROOT}/CORE_f1/runs/graph_{graph}"),
                str(args.workdir / f"data/two_edit/f1/xor_graph_{graph}.jsonl"),
            ]], "validation")
        else:
            examples = cladder
        metrics = summarize(method, seed, examples)
        output = args.workdir / "outputs" / "l1" / "cells" / family / method / f"seed-{seed}"
        output.mkdir(parents=True, exist_ok=True)
        payload = {
            "protocol": "l1_law_retention_v1", "cell_id": cell["cell_id"],
            "family": family, "method": method, "seed": seed,
            "execution": "deterministic_symbolic_validation_floor_v1",
            "world_size": 4, "validation": metrics, "test_evaluated": False,
        }
        (output / "run_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        count += 1
    print(json.dumps({"materialized_floor_cells": count}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
