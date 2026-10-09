#!/usr/bin/env python3
"""Launch exact nonpadding composed two-edit evaluation on four BERT-base GPUs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


GPU_COUNT = 4
MODES = ("t0", "t1", "t2a", "t2b", "t3a", "t3b")


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("xor", "cladder", "wiqa", "ccrgb", "com2"), default="xor")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--bundle", action="append", nargs=2,
        metavar=("ARTIFACT_DIR", "MANIFEST"), required=True,
    )
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--paraphrase-catalog", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument(
        "--a2-control",
        choices=("active", "editor_zeroed", "no_instruction", "prompting"),
        default="active",
    )
    parser.add_argument("--model")
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.max_length < 4:
        parser.error("batch-size must be positive and max-length must be at least four")
    if (args.family == "com2") != (args.provenance is not None):
        parser.error("--provenance is required exactly for family com2")
    if args.a2_control != "active" and args.mode != "t2b":
        parser.error("A2 controls require --mode t2b")
    if args.a2_control != "active" and args.split != "validation":
        parser.error("A2 controls are validation-only")
    return args


def build_command(args) -> list[str]:
    command = [
        str(Path(sys.executable).with_name("torchrun")),
        "--standalone", f"--nproc_per_node={GPU_COUNT}",
        "src/evaluate_xor_two_edit.py",
        "--family", args.family,
        "--data-root", str(args.data_root),
        "--split", args.split,
        "--checkpoint", str(args.checkpoint),
        "--output-json", str(args.output_json),
        "--mode", args.mode,
        "--a2-control", args.a2_control,
        "--batch-size", str(args.batch_size),
        "--max-length", str(args.max_length),
    ]
    for artifact, manifest in args.bundle:
        command.extend(["--bundle", artifact, manifest])
    if args.model:
        command.extend(["--model", args.model])
    if args.tokenizer:
        command.extend(["--tokenizer", str(args.tokenizer)])
    if args.provenance:
        command.extend(["--provenance", str(args.provenance)])
    if args.paraphrase_catalog:
        command.extend(["--paraphrase-catalog", str(args.paraphrase_catalog)])
    return command


def main(argv=None) -> int:
    args = parse_args(argv)
    command = build_command(args)
    if args.dry_run:
        print(json.dumps(command))
        return 0
    environment = dict(os.environ)
    environment["PYTHONPATH"] = (
        str(Path.cwd() / "src")
        + os.pathsep
        + environment.get("PYTHONPATH", "")
    )
    return subprocess.run(command, env=environment, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
