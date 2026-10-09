#!/usr/bin/env python3
"""Train one registered corrected L3 frozen-feature cell."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch

from core_7b.operator_probe import StateHead, balanced_loss
from core_7b.l3_independent_operators import build_independent_operator, trainable_parameter_count


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args()
    if sha(args.manifest) != args.manifest_sha256:
        raise ValueError("manifest mismatch")
    manifest = json.loads(args.manifest.read_text())
    matches = [cell for cell in manifest["cells"] if cell["cell_id"] == args.cell_id]
    if len(matches) != 1 or manifest["test_evaluated"] is not False:
        raise ValueError("cell is not uniquely registered or test lock changed")
    cell = matches[0]
    if cell["method"] not in {"lora_matched", "loreft", "o2"}:
        raise ValueError("corrected L3 manifest contains an unsupported method")

    source_manifest_path = Path(manifest["source_manifest"])
    source_amendment_path = Path(manifest["source_amendment"])
    if sha(source_manifest_path) != manifest["source_manifest_sha256"]:
        raise ValueError("source manifest changed")
    if sha(source_amendment_path) != manifest["source_amendment_sha256"]:
        raise ValueError("source amendment changed")
    source = json.loads(source_manifest_path.read_text())
    amendment = json.loads(source_amendment_path.read_text())
    cache_path = Path(amendment["feature_cache"])
    if sha(cache_path) != amendment["feature_cache_sha256"]:
        raise ValueError("feature cache changed")

    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    lookup = {
        (row["graph_seed"], row["world_id"]): payload["features"][index].float()
        for index, row in enumerate(payload["records"])
    }
    nodes = source["operator_contract"]["nodes"]
    source_cell = next(row for row in source["cells"] if row["seed"] == cell["source_seed"])
    graph = next(row for row in source["graphs"] if row["graph_seed"] == source_cell["graph_seed"])
    records = json.loads(Path(graph["files"]["records_real.json"]["path"]).read_text())
    worlds = {
        row["world_id"]: row
        for row in json.loads(Path(graph["files"]["worlds.json"]["path"]).read_text())
    }

    def pack(split: str):
        selected = [row for row in records if row["split"] == split]
        features = torch.stack([lookup[(graph["graph_seed"], row["world_id"])] for row in selected])
        interventions = torch.tensor([
            nodes.index(row["intervention"]["target"]) * 2 + int(row["intervention"]["value"])
            for row in selected
        ])
        labels = torch.tensor([[int(row["intervened_state"][node]) for node in nodes] for row in selected])
        factual = torch.tensor([[int(worlds[row["world_id"]]["factual_state"][node]) for node in nodes] for row in selected])
        return features, interventions, labels, factual

    device = torch.device("cuda:0")
    random.seed(cell["seed"])
    torch.manual_seed(cell["seed"])
    train = tuple(value.to(device) for value in pack("train"))
    validation = tuple(value.to(device) for value in pack("validation"))
    hidden_size = int(payload["hidden_size"])
    operator = build_independent_operator(cell["method"], 60, hidden_size).to(device)
    head = StateHead(hidden_size, 30).to(device)
    operator_class = operator.implementation_id
    operator_parameters = trainable_parameter_count(operator)
    optimizer = torch.optim.AdamW(
        list(operator.parameters()) + list(head.parameters()), lr=3e-4, weight_decay=1e-4
    )
    features, interventions, labels, factual = train
    changed = labels != factual
    noop = torch.full_like(interventions, 60)
    generator = torch.Generator(device=device).manual_seed(cell["seed"])
    batch_size = int(manifest.get("batch_size", 256))
    for _ in range(int(manifest["training_steps"])):
        indices = torch.randint(0, len(features), (min(batch_size, len(features)),), generator=generator, device=device)
        batch_features = features[indices]
        batch_interventions = interventions[indices]
        batch_labels = labels[indices]
        batch_changed = changed[indices]
        edited = operator(batch_features, batch_interventions)
        loss = balanced_loss(head(edited), batch_labels, batch_changed)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    with torch.inference_mode():
        features, interventions, labels, factual = validation
        edited = operator(features, interventions)
        predictions = head(edited).argmax(-1)
        changed = labels != factual
        preserved = ~changed
        score = 0.5 * (
            (predictions[changed] == labels[changed]).float().mean()
            + (predictions[preserved] == labels[preserved]).float().mean()
        )
        reversed_ids = interventions.roll(1)
        residuals = {
            "identity": float((operator(features, torch.full_like(interventions, 60)) - features).square().mean()),
            "idempotence": float((operator(edited, interventions) - edited).square().mean()),
            "commutation": float((
                operator(operator(features, interventions), reversed_ids)
                - operator(operator(features, reversed_ids), interventions)
            ).square().mean()),
            "last_write_wins": float((
                operator(operator(features, interventions ^ 1), interventions)
                - operator(features, interventions)
            ).square().mean()),
        }

    output = Path(cell["output"])
    output.mkdir(parents=True, exist_ok=False)
    checkpoint_path = output / "operator.pt"
    torch.save({
        "operator": operator.state_dict(), "head": head.state_dict(), "method": cell["method"],
        "operator_class": operator_class, "operator_trainable_parameters": operator_parameters,
        "test_evaluated": False,
    }, checkpoint_path)
    summary = {
        "protocol": manifest["protocol"], "cell_id": cell["cell_id"], "task": "l3",
        "method": cell["method"], "operator_class": operator_class,
        "operator_trainable_parameters": operator_parameters, "seed": cell["seed"],
        "source_seed": cell["source_seed"], "validation_balanced_accuracy": float(score),
        "law_residuals": residuals, "manifest_sha256": args.manifest_sha256,
        "checkpoint_sha256": sha(checkpoint_path), "gpu_count": 1,
        "evaluation_split": "validation", "test_evaluated": False,
    }
    (output / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
