"""Fast BERT-scale forward/backward smoke for the canonical CORE operators."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer


def load_core(path: Path):
    spec = importlib.util.spec_from_file_location("canonical_core_components", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--model", required=True); p.add_argument("--split-layer", type=int, required=True)
    p.add_argument("--core-components", type=Path, default=Path("${PRIVATE_STORAGE_ROOT}/CORE_closing_controls/core_components.py")); p.add_argument("--output", type=Path, required=True); p.add_argument("--seed", type=int, default=1)
    args = p.parse_args(); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed); device = torch.device("cuda:0"); core = load_core(args.core_components)
    tokenizer = AutoTokenizer.from_pretrained(args.model); reader = core.SharedWorldReader(args.model, 30, split_layer=args.split_layer, max_world_tokens=16).to(device)
    reader.initialize_queries(tokenizer, [f"X{i:02d}" for i in range(30)]); reader.initialize_world_tokens(); reader.eval()
    for parameter in reader.parameters(): parameter.requires_grad_(False)
    batch = tokenizer(["X00 causes X01. X01 causes X02."] * 2, padding=True, return_tensors="pt"); batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad(): hidden, mask = reader.trunk(batch["input_ids"], batch["attention_mask"])
    state = hidden[:, -16:].detach(); ids = torch.tensor([0, 3], device=device); results = {}
    for name, operator in (("o2", core.O2ConditionalLowRank(60, 16, reader.hidden_size, 16)), ("o3", core.O3StateGated(60, reader.hidden_size, 16))):
        operator = operator.to(device); edited = operator(state, ids); world = hidden.clone(); world[:, -16:] = edited; logits = reader.decode(world, mask)
        laws = core.law_losses(operator, state, ids, 4, 60, 30); loss = logits.square().mean() + sum(laws.values()); loss.backward()
        results[name] = {"parameters": sum(p.numel() for p in operator.parameters()), "output_shape": list(edited.shape), "logit_shape": list(logits.shape), "loss": float(loss.detach()), "laws": {k: float(v.detach()) for k, v in laws.items()}, "gradient_finite": all(p.grad is None or torch.isfinite(p.grad).all().item() for p in operator.parameters())}
    payload = {"model": args.model, "hidden_size": reader.hidden_size, "split_layer": args.split_layer, "gpu": torch.cuda.get_device_name(0), "max_memory_mib": torch.cuda.max_memory_allocated() / 2**20, "operators": results}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(payload, indent=2) + "\n"); print(json.dumps(payload), flush=True); return 0


if __name__ == "__main__": raise SystemExit(main())
