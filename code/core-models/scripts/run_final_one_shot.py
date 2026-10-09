#!/usr/bin/env python3
"""Fail-closed eight-way dispatcher for the frozen final held-out queue."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GROUPS = [f"{i},{i+1},{i+2},{i+3}" for i in range(0, 32, 4)]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(manifest_path: Path, expected_sha: str) -> dict:
    if sha(manifest_path) != expected_sha:
        raise ValueError("final manifest identity mismatch")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "frozen_prelaunch" or manifest.get("test_evaluated") is not False:
        raise ValueError("final manifest is not frozen/test-clean")
    if manifest.get("cell_count") != 140 or len(manifest.get("cells", [])) != 140:
        raise ValueError("final cell count mismatch")
    for section in ("source_manifests", "runners"):
        for item in manifest[section].values():
            path = ROOT / item["path"]
            if sha(path) != item["sha256"]:
                raise ValueError(f"frozen input identity mismatch: {path}")
    if sha(ROOT / manifest["data_split_audit"]) != manifest["data_split_audit_sha256"]:
        raise ValueError("data split audit identity mismatch")
    for spec in manifest["test_sets"].values():
        if not (ROOT / spec["lock"]).is_file():
            raise FileNotFoundError(spec["lock"])
        if "path" in spec and not (ROOT / spec["path"]).is_file():
            raise FileNotFoundError(spec["path"])
    for cell in manifest["cells"]:
        if cell.get("world_size") != 4 or sha(ROOT / cell["checkpoint"]) != cell["checkpoint_sha256"]:
            raise ValueError(f"checkpoint provenance mismatch: {cell['cell_id']}")
        if (ROOT / cell["output"]).exists():
            raise FileExistsError(f"final output already exists: {cell['output']}")
    return manifest


def command(cell: dict, manifest: dict) -> list[str]:
    torchrun = str(ROOT / ".venv/bin/torchrun")
    common = [torchrun, "--standalone", "--nproc_per_node=4"]
    cp = str(ROOT / cell["checkpoint"])
    out = str(ROOT / cell["output"])
    if cell["task"] == "a1":
        return common + [str(ROOT / manifest["runners"]["a1"]["path"]), "--checkpoint", cp, "--checkpoint-sha256", cell["checkpoint_sha256"], "--data-root", str(ROOT / manifest["test_sets"]["a1"]["data_root"]), "--source", cell["family"], "--output", out, "--authorization", "FINAL_ONE_SHOT"]
    if cell["task"] in {"t2", "t4"}:
        task = cell["task"]
        args = common + [str(ROOT / manifest["runners"]["t2_t4"]["path"]), "--task", task, "--checkpoint", cp, "--checkpoint-sha256", cell["checkpoint_sha256"], "--test-data", str(ROOT / manifest["test_sets"][task]["path"]), "--manifest", str(ROOT / manifest["source_manifests"][task]["path"]), "--manifest-sha256", manifest["source_manifests"][task]["sha256"], "--output", out, "--authorization", "FINAL_ONE_SHOT"]
        if task == "t4":
            args += ["--arm", cell["arm"]]
        return args
    return common + [str(ROOT / manifest["runners"]["t5"]["path"]), "--method", cell["method"], "--seed", str(cell["seed"]), "--checkpoint", cp, "--checkpoint-sha256", cell["checkpoint_sha256"], "--test-data", str(ROOT / manifest["test_sets"]["t5"]["path"]), "--manifest", str(ROOT / manifest["source_manifests"]["t5"]["path"]), "--manifest-sha256", manifest["source_manifests"]["t5"]["sha256"], "--output", out, "--authorization", "FINAL_ONE_SHOT"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--authorization")
    args = parser.parse_args()
    manifest = validate(args.manifest, args.manifest_sha256)
    if not args.execute:
        print(json.dumps({"ready": True, "cells": 140, "test_opened": False}, sort_keys=True))
        return 0
    if args.authorization != manifest["authorization_literal"]:
        raise PermissionError("exact final authorization is required")

    output_root = ROOT / "outputs/final-one-shot"
    output_root.mkdir(parents=True, exist_ok=False)
    consumed = output_root / "ONE_SHOT_CONSUMED.json"
    consumed.write_text(json.dumps({"manifest_sha256": args.manifest_sha256, "consumed_epoch": time.time(), "reason": "authorized final held-out evaluation"}, indent=2, sort_keys=True) + "\n")
    logs = ROOT / "logs/final-one-shot"
    logs.mkdir(parents=True, exist_ok=True)
    pending = iter(manifest["cells"])
    active = {}
    failures = []
    while True:
        for group in GROUPS:
            if group in active:
                continue
            try:
                cell = next(pending)
            except StopIteration:
                break
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = group
            handle = (logs / f"{cell['cell_id'].replace(':', '-')}.log").open("a")
            process = subprocess.Popen(command(cell, manifest), cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
            active[group] = (process, handle, cell)
        if not active:
            break
        time.sleep(2)
        for group, (process, handle, cell) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            handle.close()
            summary = ROOT / cell["output"] / "test_summary.json"
            if code != 0 or not summary.is_file():
                failures.append({"cell_id": cell["cell_id"], "exit_code": code})
            del active[group]
        if failures:
            for process, handle, _ in active.values():
                process.terminate()
                handle.close()
            break
    status = {"manifest_sha256": args.manifest_sha256, "complete": 140 - len(failures) if not failures else None, "failures": failures, "test_evaluated": True}
    (output_root / "STATUS.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    if failures:
        raise RuntimeError(f"final one-shot failed closed: {failures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
