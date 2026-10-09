#!/usr/bin/env python3
"""Train one registered O2/O3 cell on frozen Phi-4-14B CLadder features."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch

from core_7b.operator_probe import (
    O2ConditionalLowRank, O3StateGated, StateHead, balanced_loss,
    intervention_id, paired_metrics, state_labels,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--amendment-sha256", required=True)
    args = parser.parse_args()
    if sha(args.manifest) != args.manifest_sha256 or sha(args.amendment) != args.amendment_sha256:
        raise ValueError("registry hash mismatch")
    manifest, amendment = load_json(args.manifest), load_json(args.amendment)
    if amendment.get("base_manifest_sha256") != args.manifest_sha256:
        raise ValueError("cache amendment is not bound to manifest")
    if manifest.get("test_evaluated") is not False or amendment.get("test_evaluated") is not False:
        raise ValueError("test lock changed")
    matches = [row for row in manifest["cells"] if row["cell_id"] == args.cell_id]
    if len(matches) != 1 or matches[0].get("gpu_count") != 1:
        raise ValueError("unregistered cell")
    cell = matches[0]; method = cell["method"]; seed = int(cell["seed"])
    cache_path = Path(amendment["feature_cache"])
    if sha(cache_path) != amendment["feature_cache_sha256"]:
        raise ValueError("feature cache hash mismatch")
    artifact_path, pairs_path = Path(manifest["artifact"]), Path(manifest["pair_manifest"])
    if sha(artifact_path) != manifest["artifact_sha256"] or sha(pairs_path) != manifest["pair_manifest_sha256"]:
        raise ValueError("data hash mismatch")
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda:0")
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    if cache.get("manifest_sha256") != args.manifest_sha256 or cache.get("test_evaluated") is not False:
        raise ValueError("invalid feature cache")
    hidden_size = int(cache["hidden_size"]); nodes = manifest["operator_contract"]["node_order"]
    feature_by_model = {int(row["model_id"]): cache["features"][i].float() for i, row in enumerate(cache["records"])}
    split_by_model = {int(row["model_id"]): row["split"] for row in cache["records"]}
    artifact = load_json(artifact_path)
    records = {row["id"]: row for row in artifact["components"]}
    train = [row for row in artifact["components"] if row["two_edit_metadata"]["split"] == "train"]
    if not train or any(split_by_model[int(row["two_edit_metadata"]["model_id"])] != "train" for row in train):
        raise ValueError("training split mismatch")
    features = torch.stack([feature_by_model[int(row["two_edit_metadata"]["model_id"])] for row in train]).to(device)
    ids = torch.tensor([intervention_id(row, nodes) for row in train], device=device)
    labels = torch.full((len(train), len(nodes)), -100, dtype=torch.long, device=device)
    factual = torch.full_like(labels, -100)
    valid = torch.zeros_like(labels, dtype=torch.bool)
    for row_index, row in enumerate(train):
        for node in row["graph"]["nodes"]:
            column = nodes.index(node); valid[row_index, column] = True
            labels[row_index, column] = int(row["intervened"]["state"][node])
            factual[row_index, column] = int(row["factual"]["state"][node])
    changed = (labels != factual) & valid
    intervention_count = len(nodes) * 2
    operator = (O2ConditionalLowRank(intervention_count, hidden_size, 16) if method == "o2" else O3StateGated(intervention_count, hidden_size, 16)).to(device)
    head = StateHead(hidden_size, len(nodes)).to(device)
    parameters = list(operator.parameters()) + list(head.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=float(manifest["training_contract"]["learning_rate"]), weight_decay=1e-4)
    steps = int(manifest["training_contract"]["steps"]); history = []
    for step in range(steps):
        logits = head(operator(features, ids))
        loss = balanced_loss(logits, labels, changed, valid)
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0); optimizer.step()
        if step in {0, steps - 1} or (step + 1) % 200 == 0:
            history.append({"step": step + 1, "loss": float(loss.detach())})
    pairs = [json.loads(line) for line in pairs_path.read_text().splitlines() if line.strip()]
    validation = [row for row in pairs if row["split"] == "validation"]
    if not validation or any(row["split"] != "validation" for row in validation):
        raise ValueError("validation pair selection failed")

    def predict(*, invert: bool = False, do_nothing: bool = False):
        outputs = {}
        with torch.no_grad():
            for pair in validation:
                first, second = records[pair["first_record_id"]], records[pair["second_record_id"]]
                model_id = int(first["two_edit_metadata"]["model_id"])
                if model_id != int(second["two_edit_metadata"]["model_id"]):
                    raise ValueError("composed pair crosses factual worlds")
                state = feature_by_model[model_id].to(device).unsqueeze(0)
                if not do_nothing:
                    first_id = torch.tensor([intervention_id(first, nodes, invert=invert)], device=device)
                    second_id = torch.tensor([intervention_id(second, nodes, invert=invert)], device=device)
                    state = operator(operator(state, first_id), second_id)
                graph_nodes = first["graph"]["nodes"]
                global_prediction = head(state).argmax(-1)[0]
                outputs[pair["pair_id"]] = [int(global_prediction[nodes.index(node)]) for node in graph_nodes]
        return outputs

    operator.eval(); head.eval()
    clean, inverted, do_nothing = predict(), predict(invert=True), predict(do_nothing=True)
    changed_pairs = sum(clean[key] != inverted[key] for key in clean)
    changed_variables = sum(a != b for key in clean for a, b in zip(clean[key], inverted[key]))
    total_variables = sum(len(row) for row in clean.values())
    output = Path(cell["output"])
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    checkpoint = output / "operator.pt"
    torch.save({
        "operator": operator.state_dict(), "head": head.state_dict(), "method": method, "seed": seed,
        "hidden_size": hidden_size, "manifest_sha256": args.manifest_sha256,
        "amendment_sha256": args.amendment_sha256, "test_evaluated": False,
    }, checkpoint)
    summary = {
        "protocol": manifest["protocol"], "cell_id": cell["cell_id"], "method": method, "seed": seed,
        "model": manifest["model"], "model_revision": manifest["model_revision"],
        "backbone_frozen": True, "clean": paired_metrics(validation, clean),
        "inverted_against_retained_gold": paired_metrics(validation, inverted),
        "do_nothing": paired_metrics(validation, do_nothing),
        "changed_pair_vectors": changed_pairs, "pair_count": len(clean),
        "changed_pair_fraction": changed_pairs / len(clean),
        "changed_variable_decisions": changed_variables, "variable_decision_count": total_variables,
        "changed_variable_fraction": changed_variables / total_variables,
        "operator_parameter_count": sum(p.numel() for p in operator.parameters()),
        "head_parameter_count": sum(p.numel() for p in head.parameters()),
        "history": history, "checkpoint_sha256": sha(checkpoint),
        "feature_cache_sha256": amendment["feature_cache_sha256"],
        "manifest_sha256": args.manifest_sha256, "amendment_sha256": args.amendment_sha256,
        "gpu_count": 1, "evaluation_split": "validation", "test_evaluated": False,
    }
    (output / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
