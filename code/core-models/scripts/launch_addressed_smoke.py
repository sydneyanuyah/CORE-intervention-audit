#!/usr/bin/env python3
"""Launch an addressed BERT smoke on exactly four GPUs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


GPU_COUNT = 4
MODES = ("t2a", "t2b", "t3a", "t3b")
SPAN_OVERRIDES = ("correct", "adjacent", "random")


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--model", default="google-bert/bert-base-uncased")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--editor-disabled", action="store_true")
    parser.add_argument("--span-override", choices=SPAN_OVERRIDES, default="correct")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.editor_disabled and not args.mode.startswith("t2"):
        parser.error("--editor-disabled requires t2a/t2b")
    if args.span_override != "correct" and not args.mode.startswith("t3"):
        parser.error("non-correct --span-override requires t3a/t3b")
    return args


def build_command(args) -> list[str]:
    command = [
        "torchrun", "--standalone", f"--nproc_per_node={GPU_COUNT}",
        "src/addressed_smoke.py",
        "--mode", args.mode,
        "--data-root", str(args.data_root),
        "--output-json", str(args.output_json),
        "--model", args.model,
        "--max-length", str(args.max_length),
        "--steps", str(args.steps),
        "--batch-size", str(args.batch_size),
        "--span-override", args.span_override,
    ]
    if args.editor_disabled:
        command.append("--editor-disabled")
    return command


def main(argv=None) -> int:
    args = parse_args(argv)
    command = build_command(args)
    if args.dry_run:
        print(json.dumps(command))
        return 0
    environment = dict(os.environ)
    source = str(Path.cwd() / "src")
    environment["PYTHONPATH"] = source + os.pathsep + environment.get("PYTHONPATH", "")
    return subprocess.run(command, env=environment, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
