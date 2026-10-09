#!/usr/bin/env python3
"""Validate one registered L1 law-profile cell and execute its four-rank job."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


PROFILES = {
    "core_base_1law", "core_i_3law", "core_full_4law",
    "full", "drop_identity", "drop_idempotence", "drop_commutation",
    "drop_selective_invariance",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--gpus", required=True)
    parser.add_argument(
        "--output-template",
        default="outputs/l1/cells/{family}/{method}/seed-{seed}",
    )
    parser.add_argument(
        "--reader-template",
        default="outputs/l1/readers/{family}/seed-{seed}/best.pt",
    )
    parser.add_argument(
        "--paraphrase-catalog",
        default="data/paraphrases/structured_catalog.json",
    )
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()
    registry = json.loads(args.cells.read_text())
    matches = [cell for cell in registry["cells"] if cell["cell_id"] == args.cell_id]
    if len(matches) != 1:
        raise ValueError("cell ID is not uniquely registered")
    cell = matches[0]
    if cell["method"] not in PROFILES or cell["world_size"] != 4:
        raise ValueError("launcher accepts registered four-rank law profiles only")
    gpu_ids = [int(value) for value in args.gpus.split(",")]
    if len(gpu_ids) != 4 or len(set(gpu_ids)) != 4 or min(gpu_ids) < 0 or max(gpu_ids) > 31:
        raise ValueError("--gpus must name four distinct server devices")
    family, method, seed = cell["family"], cell["method"], int(cell["seed"])
    output = Path(args.output_template.format(
        family=family, method=method, seed=seed,
    ))
    if (args.workdir / output / "run_summary.json").is_file():
        raise FileExistsError(f"refusing duplicate completed cell {cell['cell_id']}")
    reader = Path(args.reader_template.format(
        family=family, method=method, seed=seed,
    ))
    if not (args.workdir / reader).is_file():
        raise FileNotFoundError(f"missing reader checkpoint {reader}")
    command = [
        str(args.workdir / ".venv/bin/torchrun"), "--standalone", "--nproc_per_node=4",
        "src/train_addressed.py", "--mode", "t2b", "--model-size", "base",
        "--data-root", "data", "--output-dir", str(output), "--evidence-capable",
        "--sources", family, "--epochs", "3", "--batch-size", "16",
        "--max-length", "512", "--world-slots", "30", "--rank", "16",
        "--learning-rate", "2e-5", "--weight-decay", "0.01", "--seed", str(seed),
        "--num-workers", "2", "--precision", "bf16", "--selection-metric", "macro_f1",
        "--variable-loss-weight", "1.0", "--balanced-variable-loss",
        "--transition-variable-loss", "--causal-task-readout", "--split-layer", "4",
        "--train-shard-by", "world" if family == "xor" else "graph",
        "--validation-shard-by", "world" if family == "xor" else "graph",
        "--initialize-checkpoint", str(reader), "--freeze-reader",
        "--law-profile", method, "--law-weight", "1.0",
    ]
    if family == "xor":
        graph = int(cell["graph_seed"])
        command.extend([
            "--xor-bundle", f"${PRIVATE_STORAGE_ROOT}/CORE_f1/runs/graph_{graph}",
            f"data/two_edit/f1/xor_graph_{graph}.jsonl",
        ])
    else:
        command.extend([
            "--groups-dir", "data/groups",
            "--paraphrase-catalog", args.paraphrase_catalog,
        ])
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    if args.print_only:
        print(json.dumps({"cell": cell, "command": command, "visible_devices": env["CUDA_VISIBLE_DEVICES"]}))
        return 0
    os.chdir(args.workdir)
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    raise SystemExit(main())
