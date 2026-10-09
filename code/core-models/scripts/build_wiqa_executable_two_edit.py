#!/usr/bin/env python3
"""Correct the WIQA two-edit artifact using train-exposed signed interventions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


INPUT_SHA256 = "not-published"
VALUE_BY_NODE = {
    "U": "less", "V": "less", "W": "less",
    "X": "more", "Y": "more", "Z": "more",
}
NUMBER = {"less": -1, "no_effect": 0, "more": 1}
NAME = {value: name for name, value in NUMBER.items()}


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pair_id(record_id: str, target: str, value: str) -> str:
    payload = json.dumps(["wiqa", record_id, target, value], sort_keys=True)
    return f"manual-two-edit:wiqa:{hashlib.sha256(payload.encode()).hexdigest()[:24]}"


def solve(row: Mapping[str, Any], edits: list[Mapping[str, Any]]) -> dict[str, str]:
    nodes = row["graph"]["nodes"]
    incoming: dict[str, list[tuple[str, int]]] = {node: [] for node in nodes}
    for edge in row["graph"]["signed_edges"]:
        incoming[edge["target"]].append((edge["source"], 1 if edge["sign"] == "positive" else -1))
    state = {edit["target"]: NUMBER[edit["value_token"]] for edit in edits}
    while len(state) < len(nodes):
        progress = False
        for node in nodes:
            if node in state or any(parent not in state for parent, _ in incoming[node]):
                continue
            effects = {state[parent] * sign for parent, sign in incoming[node] if state[parent]}
            if len(effects) > 1:
                raise ValueError(f"{row['pair_id']}: conflicting signed effects at {node}")
            state[node] = next(iter(effects), 0)
            progress = True
        if not progress:
            raise ValueError(f"{row['pair_id']}: graph is cyclic")
    return {node: NAME[state[node]] for node in nodes}


def query_node(record: Mapping[str, Any]) -> str:
    found = [
        probe["variable"] for probe in record["probes"]
        if isinstance(probe.get("question"), str) and probe["question"].strip()
    ]
    if len(found) != 1:
        raise ValueError(f"{record['id']}: expected exactly one queried probe")
    return found[0]


def build(input_path: Path, data_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if digest(input_path) != INPUT_SHA256:
        raise ValueError("signed WIQA input queue SHA-256 mismatch")
    source_rows = {row["id"]: row for row in rows(data_root / "records" / "wiqa.jsonl")}
    train_ids = set((data_root / "splits" / "wiqa.train.txt").read_text().split())
    validation_ids = set((data_root / "splits" / "wiqa.validation.txt").read_text().split())
    exposure = {
        (row["intervention"]["target"], row["intervention"]["value_token"])
        for record_id, row in source_rows.items() if record_id in train_ids
    }
    if exposure != set(VALUE_BY_NODE.items()):
        raise ValueError(f"unexpected WIQA train intervention semantics: {sorted(exposure)}")
    queue_out, truth_out = [], []
    for original in rows(input_path):
        record_id = original["source_record_id"]
        if record_id not in validation_ids or record_id not in source_rows:
            raise ValueError(f"{record_id}: source is outside validation")
        first_target = original["first_edit"]["target"]
        ordered = list(source_rows[record_id].get("descendants") or []) + sorted(original["graph"]["nodes"])
        seen: set[str] = set()
        eligible = []
        for target in ordered:
            semantic = (target, VALUE_BY_NODE.get(target))
            if target == first_target or target in seen or semantic not in exposure:
                continue
            seen.add(target)
            surfaces = sorted({text.strip() for text in original["graph"]["node_text"][target] if text.strip()})
            if surfaces:
                eligible.append((target, surfaces[0], semantic[1]))
        if not eligible:
            raise ValueError(f"{record_id}: no distinct train-exposed second edit")
        target, target_text, value_token = eligible[0]
        second = {
            "target": target, "target_text": target_text, "value": target_text,
            "value_token": value_token, "kind": "value_set",
            "formal": f"do({target} = {json.dumps(target_text)})",
            "text": f"After the first change, also suppose {target_text}.",
        }
        corrected = dict(original)
        corrected["pair_id"] = pair_id(record_id, target, target_text)
        corrected["second_edit"] = second
        corrected["truth_protocol"] = {
            "protocol": "wiqa_signed_executable_two_edit_v1",
            "selection": "distinct second target with train-exposed target/value semantics",
            "input_queue_sha256": INPUT_SHA256,
            "test_evaluated": False,
        }
        intermediate = solve(corrected, [corrected["first_edit"]])
        final = solve(corrected, [corrected["first_edit"], second])
        nodes = corrected["graph"]["nodes"]
        changed = [node for node in nodes if final[node] != "no_effect"]
        preserved = [node for node in nodes if final[node] == "no_effect"]
        if not changed or not preserved:
            raise ValueError(f"{record_id}: corrected pair lacks a balanced evaluation cell")
        queried = query_node(source_rows[record_id])
        truth_out.append({
            "pair_id": corrected["pair_id"], "source": "wiqa", "split": "validation",
            "status": "accepted", "confidence": "high",
            "annotator_id": "executable-signed-wiqa-solver",
            "first_edit_valid": True, "second_edit_valid": True,
            "intermediate_outputs": intermediate, "gold_final_outputs": final,
            "target_variables": [corrected["first_edit"]["target"], target],
            "actually_changed_variables": changed, "preserved_variables": preserved,
            "query_answer_after_both_edits": final[queried],
            "justification": "Deterministic signed-edge propagation with interventions applied in declared order and incoming edges severed at clamped targets.",
            "truth_provenance": {
                "protocol": "wiqa_signed_executable_two_edit_v1",
                "input_queue_sha256": INPUT_SHA256, "test_evaluated": False,
            },
        })
        queue_out.append(corrected)
    queue_out.sort(key=lambda row: row["pair_id"])
    truth_out.sort(key=lambda row: row["pair_id"])
    if len(queue_out) != 70 or len({row["pair_id"] for row in queue_out}) != 70:
        raise ValueError("corrected WIQA artifact must contain exactly 70 unique rows")
    return queue_out, truth_out


def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in values), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--queue-output", type=Path, required=True)
    parser.add_argument("--truth-output", type=Path, required=True)
    args = parser.parse_args()
    queue, truth = build(args.input, args.data_root)
    write_jsonl(args.queue_output, queue)
    write_jsonl(args.truth_output, truth)
    print(json.dumps({
        "rows": len(queue), "queue_sha256": digest(args.queue_output),
        "truth_sha256": digest(args.truth_output), "test_evaluated": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
