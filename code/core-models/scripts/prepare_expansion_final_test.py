#!/usr/bin/env python3
"""Freeze normalized held-out registries for the four-model expansion."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASKS = ("t2", "t4", "t5")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n" for value in values))


def directions(row: dict) -> list[int]:
    answer = []
    for key in sorted(row["factual_state"]):
        factual = float(row["factual_state"][key])
        intervened = float(row["intervened_state"][key])
        answer.append(0 if intervened < factual else 2 if intervened > factual else 1)
    return answer


def normalize(task: str) -> list[dict]:
    base = ROOT / "data/expansion-test"
    if task == "t2":
        source = rows(base / "t2/csuite_t2.jsonl")
        return [{"id": r["id"], "group": r["graph_group_id"], "split": "test", "text": r["rendered_input"],
                 "label": directions(r), "command": sorted(r["factual_state"]).index(r["intervention"]["target"]),
                 "source": r["sem_family"], "test_evaluated": True} for r in source]
    if task == "t4":
        source = rows(base / "t4/csuite_t4.jsonl")
        arms = {"changing_only": 0, "imagining_only": 1, "joint": 2}
        return [{"id": r["id"], "group": r["graph_group_id"], "split": "test", "text": r["rendered_input"],
                 "label": directions(r), "command": arms[r["arm"]], "source": r["arm"],
                 "test_evaluated": True} for r in source]
    if task == "t5":
        output = []
        for r in rows(base / "t5/native_t5.jsonl"):
            if r["source"].lower() == "com2":
                continue
            prompt = r.get("input") or " ".join(str(r.get(key, "")) for key in ("premise", "context", "question"))
            answer = str(r["answer"])
            if r["source"] == "counterbench":
                options = ["no", "yes"]
                answer = answer.lower()
            elif r["source"] == "corr2cause":
                options = ["0", "1"]
            else:
                options = [str(value) for value in r.get("options", [])]
            if answer not in options and r["source"] == "crass":
                options.append(answer)
            for index, option in enumerate(options):
                output.append({"id": f"{r['id']}:{index}", "item_id": r["id"], "group": r.get("group_id", r["id"]),
                               "split": "test", "text": f"{prompt} Candidate answer: {option}",
                               "label": int(option == answer or str(index) == answer), "command": index,
                               "source": r["source"], "test_evaluated": True})
        return output
    raise ValueError(task)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", choices=("qwen7b", "phi14b", "qwen32", "llama70b"), required=True)
    args = parser.parse_args()
    for task in TASKS:
        training_path = ROOT / f"registry/{args.model_key}_{task}_manifest.json"
        training = json.loads(training_path.read_text())
        values = normalize(task)
        data_path = ROOT / f"data/expansion-test/normalized/{task}.jsonl"
        write_jsonl(data_path, values)
        source_cells = []
        for cell in training["cells"]:
            checkpoint = ROOT / cell["output"] / "model.pt"
            summary = ROOT / cell["output"] / "run_summary.json"
            if not checkpoint.is_file() or not summary.is_file():
                raise FileNotFoundError(f"incomplete frozen source cell: {cell['cell_id']}")
            run = json.loads(summary.read_text())
            if run.get("test_evaluated") is not False or run.get("checkpoint_sha256") != sha(checkpoint):
                raise ValueError(f"invalid frozen source cell: {cell['cell_id']}")
            source_cells.append({**cell, "checkpoint": str(checkpoint), "checkpoint_sha256": sha(checkpoint),
                                 "summary": str(summary), "summary_sha256": sha(summary)})
        shards = 32 if args.model_key in {"qwen7b", "phi14b"} else 1
        manifest = {
            "protocol": f"{args.model_key}_{task}_final_one_shot_v1", "model_key": args.model_key, "task": task,
            "model": training["model"], "model_revision": training["model_revision"],
            "model_snapshot": training["model_snapshot"], "training_manifest": str(training_path),
            "training_manifest_sha256": sha(training_path), "data": str(data_path), "data_sha256": sha(data_path),
            "test_rows": len(values), "feature_shards": shards, "source_cells": source_cells,
            "excluded_sources": ["com2"], "evaluation_split": "test", "test_evaluated": True,
            "authorization": "EXPANSION_FINAL_ONE_SHOT",
        }
        output = ROOT / f"registry/{args.model_key}_{task}_final_test_manifest.json"
        output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"model": args.model_key, "task": task, "rows": len(values), "manifest_sha256": sha(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
