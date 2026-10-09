#!/usr/bin/env python3
"""Policy-preserving launcher for production addressed BERT training."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path


MODELS = {
    "base": {"gpus": 4, "model": "google-bert/bert-base-uncased"},
    "large": {"gpus": 8, "model": "google-bert/bert-large-uncased"},
}
MODES = ("t0", "t1", "t2a", "t2b", "t3a", "t3b")


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--model-size", choices=tuple(MODELS), required=True)
    parser.add_argument("--model")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sources", nargs="+", default=None)
    parser.add_argument(
        "--xor-bundle", action="append", nargs=2,
        metavar=("ARTIFACT_DIR", "MANIFEST"),
    )
    parser.add_argument("--groups-dir", type=Path)
    parser.add_argument("--evidence-capable", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--split-layer", type=int, default=0, help="0 selects the model midpoint")
    parser.add_argument("--selection-metric", choices=("macro_f1", "accuracy", "loss"), default="macro_f1")
    parser.add_argument("--variable-loss-weight", type=float, default=0.0)
    parser.add_argument("--balanced-variable-loss", action="store_true")
    parser.add_argument("--transition-variable-loss", action="store_true")
    parser.add_argument("--causal-task-readout", action="store_true")
    parser.add_argument("--hard-pointer", action="store_true")
    parser.add_argument("--pointer-loss-weight", type=float, default=0.0)
    parser.add_argument("--open-text-output", action="store_true")
    parser.add_argument("--open-text-max-tokens", type=int, default=32)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--measure-checkpoint", type=Path)
    parser.add_argument("--prediction-jsonl", type=Path)
    parser.add_argument("--paraphrase-catalog", type=Path)
    parser.add_argument("--editor-disabled", action="store_true")
    parser.add_argument("--span-override", choices=("correct", "adjacent", "random"), default="correct")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.editor_disabled and not args.mode.startswith("t2"):
        parser.error("--editor-disabled requires t2a/t2b")
    if args.span_override != "correct" and not args.mode.startswith("t3"):
        parser.error("wrong-span controls require t3a/t3b")
    if args.evidence_capable and args.groups_dir is None and not args.xor_bundle:
        parser.error("--evidence-capable requires --groups-dir or --xor-bundle")
    if args.xor_bundle and args.sources != ["xor"]:
        parser.error("--xor-bundle requires --sources xor")
    if args.mode == "t0" and (args.sources != ["xor"] or not args.xor_bundle):
        parser.error("T0 learned IDs require --sources xor and authoritative XOR bundles")
    if args.resume and args.measure_checkpoint:
        parser.error("--resume and --measure-checkpoint are mutually exclusive")
    if args.prediction_jsonl is not None and args.measure_checkpoint is None:
        parser.error("--prediction-jsonl requires --measure-checkpoint")
    if not math.isfinite(args.variable_loss_weight) or args.variable_loss_weight < 0:
        parser.error("--variable-loss-weight must be finite and non-negative")
    if args.balanced_variable_loss and args.variable_loss_weight == 0:
        parser.error("--balanced-variable-loss requires positive --variable-loss-weight")
    if args.transition_variable_loss and not args.balanced_variable_loss:
        parser.error("--transition-variable-loss requires --balanced-variable-loss")
    if args.causal_task_readout and not args.transition_variable_loss:
        parser.error("--causal-task-readout requires --transition-variable-loss")
    if args.hard_pointer and (args.mode != "t3b" or not args.causal_task_readout):
        parser.error("--hard-pointer requires T3-b with --causal-task-readout")
    if not math.isfinite(args.pointer_loss_weight) or args.pointer_loss_weight < 0:
        parser.error("--pointer-loss-weight must be finite and non-negative")
    if args.pointer_loss_weight > 0 and args.mode != "t3b":
        parser.error("--pointer-loss-weight requires --mode t3b")
    if args.open_text_output and args.sources != ["com2"]:
        parser.error("--open-text-output requires --sources com2")
    if args.split_layer < 0:
        parser.error("--split-layer must be non-negative")
    return args


def build_command(args) -> list[str]:
    policy = MODELS[args.model_size]
    command = [
        str(Path(sys.executable).with_name("torchrun")),
        "--standalone", f"--nproc_per_node={policy['gpus']}",
        "src/train_addressed.py",
        "--mode", args.mode,
        "--model-size", args.model_size,
        "--model", args.model or policy["model"],
        "--data-root", str(args.data_root),
        "--output-dir", str(args.output_dir),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--seed", str(args.seed),
        "--split-layer", str(args.split_layer),
        "--selection-metric", args.selection_metric,
        "--variable-loss-weight", str(args.variable_loss_weight),
        "--pointer-loss-weight", str(args.pointer_loss_weight),
        "--open-text-max-tokens", str(args.open_text_max_tokens),
        "--span-override", args.span_override,
    ]
    if args.editor_disabled:
        command.append("--editor-disabled")
    if args.balanced_variable_loss:
        command.append("--balanced-variable-loss")
    if args.transition_variable_loss:
        command.append("--transition-variable-loss")
    if args.causal_task_readout:
        command.append("--causal-task-readout")
    if args.hard_pointer:
        command.append("--hard-pointer")
    if args.open_text_output:
        command.append("--open-text-output")
    if args.resume:
        command.extend(["--resume", str(args.resume)])
    if args.measure_checkpoint:
        command.extend(["--measure-checkpoint", str(args.measure_checkpoint)])
    if args.prediction_jsonl:
        command.extend(["--prediction-jsonl", str(args.prediction_jsonl)])
    if args.paraphrase_catalog:
        command.extend(["--paraphrase-catalog", str(args.paraphrase_catalog)])
    if args.sources:
        command.extend(["--sources", *args.sources])
    if args.xor_bundle:
        for artifact, manifest in args.xor_bundle:
            command.extend(["--xor-bundle", artifact, manifest])
    if args.groups_dir:
        command.extend(["--groups-dir", str(args.groups_dir)])
    if args.evidence_capable:
        command.append("--evidence-capable")
    return command


def main(argv=None) -> int:
    args = parse_args(argv)
    command = build_command(args)
    if args.dry_run:
        print(json.dumps(command))
        return 0
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path.cwd() / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    return subprocess.run(command, env=environment, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
