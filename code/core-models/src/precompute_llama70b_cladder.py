#!/usr/bin/env python3
"""Create one frozen Llama-3.3-70B representation per CLadder world."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args()
    if sha(args.manifest) != args.manifest_sha256:
        raise ValueError("manifest hash mismatch")
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("test_evaluated") is not False or manifest.get("allowed_splits") != ["train", "validation"]:
        raise ValueError("split lock changed")
    artifact_path = Path(manifest["artifact"])
    if sha(artifact_path) != manifest["artifact_sha256"]:
        raise ValueError("artifact hash mismatch")
    artifact = json.loads(artifact_path.read_text())
    records = {}
    for row in artifact["components"]:
        split = row["two_edit_metadata"]["split"]
        if split not in {"train", "validation"}:
            raise ValueError("forbidden split")
        model_id = int(row["two_edit_metadata"]["model_id"])
        value = {
            "model_id": model_id, "split": split, "passage": row["factual"]["passage"],
            "nodes": row["graph"]["nodes"], "factual_state": row["factual"]["state"],
        }
        if model_id in records and records[model_id] != value:
            raise ValueError("model ID maps to inconsistent factual worlds")
        records[model_id] = value
    model_path = Path(manifest["model_snapshot"])
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModel.from_pretrained(model_path, local_files_only=True, dtype=torch.bfloat16, device_map="balanced", max_memory={0:"76GiB",1:"76GiB"}, attn_implementation="eager", low_cpu_mem_usage=True)
    model.eval()
    ordered = [records[key] for key in sorted(records)]
    features = []
    with torch.no_grad():
        for start in range(0, len(ordered), 1):
            prompts = [f"Causal system description:\n{x['passage']}\nEncode this factual world for downstream intervention editing." for x in ordered[start:start + 1]]
            batch = tokenizer(prompts, padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda:0")
            hidden = model(**batch).last_hidden_state
            final = batch["attention_mask"].sum(1) - 1
            features.extend(hidden[torch.arange(len(final), device=hidden.device), final].float().cpu())
    payload = {
        "protocol": manifest["protocol"], "manifest_sha256": args.manifest_sha256,
        "hidden_size": int(model.config.hidden_size), "model_revision": manifest["model_revision"],
        "records": ordered, "features": torch.stack(features).half(), "test_evaluated": False,
    }
    output = Path(manifest["feature_cache"]); output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    torch.save(payload, output)
    summary = {
        "protocol": manifest["protocol"], "manifest_sha256": args.manifest_sha256,
        "feature_cache_sha256": sha(output), "world_count": len(ordered),
        "train_worlds": sum(x["split"] == "train" for x in ordered),
        "validation_worlds": sum(x["split"] == "validation" for x in ordered),
        "hidden_size": payload["hidden_size"], "test_evaluated": False,
    }
    Path(manifest["feature_cache_summary"]).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
