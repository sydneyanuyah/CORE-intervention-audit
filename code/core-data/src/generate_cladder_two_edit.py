"""Generate executable-SCM CLadder single edits and ordered two-edit truth."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

from convert_cladder import (
    ELIGIBLE_QUERY_TYPE, descendants_of, model_group_split, solve_state,
    target_surface_and_span,
)
from inspect_cladder import SOURCE_SHA256, load_release


PROTOCOL = "cladder_deterministic_two_edit_v1"


def stable_id(prefix: str, payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return f"{prefix}:{hashlib.sha256(encoded).hexdigest()[:24]}"


def ordered_pair_id(
    source: str, graph_group: str, world_group: str, first_id: str, second_id: str,
) -> str:
    """Match the model repository's shared order-sensitive pair identity."""
    encoded = json.dumps(
        [source, graph_group, world_group, first_id, second_id],
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return f"two-edit:{hashlib.sha256(encoded).hexdigest()[:24]}"


def build_component(
    base: dict[str, Any], model: dict[str, Any], model_id: int,
    factual: dict[str, int], roots: dict[str, int], target: str, value: int, split: str,
) -> dict[str, Any]:
    nodes = list(base["graph"]["nodes"])
    edges = list(base["graph"]["edges"])
    after = solve_state(model, {**roots, target: value})
    target_text, target_span = target_surface_and_span(
        base["factual"]["passage"], target, model["variable_mapping"]
    )
    value_text = model["variable_mapping"].get(f"{target}{value}", str(value))
    record_id = stable_id("cladder-two-edit-component", [model_id, base["id"], target, value])
    descendants = descendants_of(target, edges)
    non_descendants = sorted(set(nodes) - {target} - set(descendants))
    question = f"After this intervention, is {value_text} true?"
    return {
        "id": record_id, "source": "cladder", "source_id": base["source_id"],
        "structure_kind": "dag", "graph": base["graph"], "chain": None,
        "factual": {"passage": base["factual"]["passage"], "state": factual, "question": question, "answer": None},
        "intervention": {
            "target": target, "target_text": target_text, "target_span": target_span,
            "value": value, "value_token": "yes" if value else "no", "replacement_span": None,
            "kind": "value_set", "formal": f"do({target} = {value})",
            "text": f"Set {target_text} so that {value_text}.",
        },
        "intervened": {"passage": None, "state": after, "answer": "yes" if after[target] == value else "no"},
        "descendants": descendants, "non_descendants": non_descendants,
        "probes": [{
            "variable": node, "question": question if node == target else None,
            "answer_before": factual[node], "answer_after": after[node],
            "required": "must not change" if node in non_descendants else "may change",
            "actually_changed": factual[node] != after[node],
        } for node in nodes],
        "two_edit_metadata": {"protocol": PROTOCOL, "model_id": model_id, "split": split},
    }


def build_world(
    base: dict[str, Any], question: dict[str, Any], model: dict[str, Any], split: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    model_id = int(model["model_id"])
    meta = question["meta"]
    factual_roots = {name: int(value) for name, value in meta["given_info"].items()}
    factual_roots[meta["treatment"]] = 1 - int(meta["action"])
    factual = solve_state(model, factual_roots)
    if factual != base["factual"]["state"]:
        raise ValueError(f"{base['id']}: normalized factual state mismatch")
    nodes = list(base["graph"]["nodes"])
    components = [
        build_component(base, model, model_id, factual, factual_roots, target, value, split)
        for target in nodes for value in (0, 1)
    ]
    by_edit = {(r["intervention"]["target"], r["intervention"]["value"]): r for r in components}
    graph_group = f"cladder:model:{model_id}"
    world_group = f"{graph_group}:world:question:{base['source_id']}"
    rows = []
    for first, second in itertools.product(by_edit, repeat=2):
        # A composed example must contain two distinct component records.
        # Same-target/different-value pairs remain valid and exercise order;
        # repeating the identical assignment is only one semantic edit.
        if first == second:
            continue
        interventions: dict[str, int] = {}
        interventions[first[0]] = first[1]
        interventions[second[0]] = second[1]
        gold = solve_state(model, {**factual_roots, **interventions})
        factual_outputs = [factual[node] for node in nodes]
        gold_outputs = [gold[node] for node in nodes]
        changed = [a != b for a, b in zip(gold_outputs, factual_outputs)]
        targets = [node in interventions for node in nodes]
        preservation = [not flag for flag in changed]
        # The registered balanced protocol requires both causal-change and
        # preservation cells in every retained composition. Factual-equivalent
        # pairs and all-node-changing pairs cannot yield that paired score.
        if not any(changed) or not any(preservation):
            continue
        first_id, second_id = by_edit[first]["id"], by_edit[second]["id"]
        rows.append({
            "pair_id": ordered_pair_id("cladder", graph_group, world_group, first_id, second_id),
            "source": "cladder", "split": split, "graph_group_id": graph_group,
            "world_group_id": world_group,
            "first_record_id": first_id, "second_record_id": second_id,
            "factual_outputs": factual_outputs, "gold_outputs": gold_outputs,
            "change_mask": changed, "preservation_mask": preservation,
            "target_mask": targets,
            "truth_provenance": {
                "protocol": PROTOCOL, "truth_source": "recomputed deterministic CLadder conditional tables",
                "raw_sha256": SOURCE_SHA256, "model_id": model_id, "test_evaluated": False,
            },
        })
    return components, rows


def build_truth_descriptor(
    base: dict[str, Any], question: dict[str, Any], model: dict[str, Any], split: str
) -> dict[str, Any]:
    """Keep the minimal raw SCM needed for independent downstream re-execution."""
    meta = question["meta"]
    roots = {name: int(value) for name, value in meta["given_info"].items()}
    roots[meta["treatment"]] = 1 - int(meta["action"])
    model_id = int(model["model_id"])
    return {
        "world_group_id": f"cladder:model:{model_id}:world:question:{base['source_id']}",
        "source_id": base["source_id"],
        "model_id": model_id,
        "split": split,
        "graph": base["graph"],
        "factual_roots": roots,
        "factual_state": base["factual"]["state"],
        "model": {
            "structure": model["structure"],
            "equation_type": model["equation_type"],
            "params": model["params"],
        },
    }


def generate(raw_path: Path, records_path: Path, per_split_models: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    questions, models, _ = load_release(raw_path)
    accepted = {int(row["source_id"]): row for row in map(json.loads, records_path.read_text(encoding="utf-8").splitlines())}
    eligible = {
        int(q["question_id"]): q for q in questions
        if q["meta"]["query_type"] == ELIGIBLE_QUERY_TYPE and int(q["question_id"]) in accepted
    }
    by_model = {int(model["model_id"]): model for model in models}
    selected: dict[str, list[int]] = {}
    for split in ("train", "validation"):
        candidates = sorted(
            eligible,
            key=lambda qid: hashlib.sha256(f"cladder-two-edit-selection-v1:{qid}".encode()).hexdigest(),
        )
        chosen = []
        used_models = set()
        for qid in candidates:
            model_id = int(eligible[qid]["meta"]["model_id"])
            if model_group_split(model_id) != split or model_id in used_models:
                continue
            chosen.append(qid); used_models.add(model_id)
            if len(chosen) == per_split_models:
                break
        if len(chosen) != per_split_models:
            raise ValueError(f"not enough {split} deterministic CLadder models")
        selected[split] = chosen
    components, manifests, worlds = [], [], []
    for split, qids in selected.items():
        for qid in qids:
            question = eligible[qid]; model = by_model[int(question["meta"]["model_id"])]
            world_components, world_rows = build_world(accepted[qid], question, model, split)
            components.extend(world_components); manifests.extend(world_rows)
            worlds.append(build_truth_descriptor(accepted[qid], question, model, split))
    artifact = {
        "protocol": PROTOCOL, "raw_sha256": SOURCE_SHA256, "test_evaluated": False,
        "selected_question_ids": selected, "worlds": worlds, "components": components,
    }
    return artifact, manifests


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--per-split-models", type=int, default=20)
    args = parser.parse_args()
    artifact, rows = generate(args.raw, args.records, args.per_split_models)
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    args.artifact.write_text(json.dumps(artifact, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    args.manifest.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({
        "components": len(artifact["components"]), "pairs": len(rows),
        "splits": {split: sum(row["split"] == split for row in rows) for split in ("train", "validation")},
        "test_evaluated": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
