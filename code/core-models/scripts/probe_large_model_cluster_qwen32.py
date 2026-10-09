#!/usr/bin/env python3
"""Load the frozen Qwen2.5-32B backbone on one large-model cluster H100 and verify inference."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer


MODEL = Path(
    "${PRIVATE_STORAGE_ROOT}/hf_cache/"
    "models--Qwen--Qwen2.5-32B-Instruct/snapshots/"
    "not-published"
)
OUTPUT = Path(
    "${CORE_PROJECT_ROOT}/"
    "reports/evidence/large_model_cluster-qwen32-probe.json"
)


def main() -> None:
    if not (MODEL / "model.safetensors.index.json").is_file():
        raise FileNotFoundError(f"incomplete frozen model snapshot: {MODEL}")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"probe requires exactly one visible GPU, got {torch.cuda.device_count()}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = AutoModel.from_pretrained(
        MODEL,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).eval()
    batch = tokenizer(
        ["Intervene on X, then predict Y."],
        return_tensors="pt",
        truncation=True,
        max_length=64,
    ).to("cuda:0")
    with torch.inference_mode():
        hidden = model(**batch).last_hidden_state

    result = {
        "model": "Qwen/Qwen2.5-32B-Instruct",
        "revision": MODEL.name,
        "dtype": str(hidden.dtype),
        "hidden_shape": list(hidden.shape),
        "gpu": torch.cuda.get_device_name(0),
        "peak_memory_bytes": torch.cuda.max_memory_allocated(0),
        "status": "passed",
        "test_evaluated": False,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
