#!/usr/bin/env python3
"""Keep eight four-GPU groups occupied with registered CCR.GB O2/O3 cells."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).parents[1].resolve()
MANIFEST_SHA = "not-published"
CELLS_SHA = "not-published"
AMENDMENT_SHA = "not-published"
CATALOG_SHA = "not-published"
ARTIFACT_SHA = "not-published"
PAIR_SHA = "not-published"
CORE_SHA = "not-published"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class Cell:
    method: str
    seed: int
    checkpoint: Path
    checkpoint_sha: str

    @property
    def cell_id(self) -> str:
        return f"f2:ccrgb:{self.method}:{self.seed}"

    @property
    def output(self) -> Path:
        return ROOT / f"outputs/f2/ccrgb/{self.method}/seed-{self.seed}"


def cells() -> list[Cell]:
    catalog_path = ROOT / "registry/a2_source_catalog.json"
    if sha256(catalog_path) != CATALOG_SHA:
        raise RuntimeError("F2 source catalog hash mismatch")
    catalog = json.loads(catalog_path.read_text())
    indexed = {(row["family"], row["seed"]): row for row in catalog["cells"]}
    result = []
    for seed in range(301, 321):
        row = indexed[("ccrgb", 2026090300 + seed)]
        checkpoint = ROOT / row["source_directory"] / "best.pt"
        if sha256(checkpoint) != row["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint hash mismatch for F2 seed {seed}")
        for method in ("o2", "o3"):
            result.append(Cell(method, seed, checkpoint, row["checkpoint_sha256"]))
    return result


def valid(cell: Cell) -> bool:
    path = cell.output / "run_summary.json"
    if not path.is_file():
        return False
    try:
        row = json.loads(path.read_text())
    except json.JSONDecodeError:
        return False
    return (
        row.get("cell_id") == cell.cell_id
        and row.get("checkpoint_sha256") == cell.checkpoint_sha
        and row.get("world_size") == 4
        and row.get("test_evaluated") is False
        and row.get("amendment_sha256") == AMENDMENT_SHA
        and isinstance(row.get("two_edit", {}).get("two_edit_balanced"), (int, float))
    )


def command(cell: Cell) -> list[str]:
    source = cell.checkpoint.parent
    return [
        str(ROOT / ".venv/bin/torchrun"), "--standalone", "--nproc_per_node=4",
        str(ROOT / "src/train_f2_real.py"), "--method", cell.method, "--seed", str(cell.seed),
        "--checkpoint", str(cell.checkpoint), "--expected-checkpoint-sha256", cell.checkpoint_sha,
        "--tokenizer", str(source / "tokenizer"),
        "--artifact", str(ROOT / "data/two_edit/ccrgb/artifact.json"), "--expected-artifact-sha256", ARTIFACT_SHA,
        "--two-edit-manifest", str(ROOT / "data/two_edit/ccrgb/manifest.jsonl"), "--expected-two-edit-manifest-sha256", PAIR_SHA,
        "--source-catalog", str(ROOT / "registry/a2_source_catalog.json"), "--expected-source-catalog-sha256", CATALOG_SHA,
        "--manifest", str(ROOT / "registry/f2_manifest.json"), "--expected-manifest-sha256", MANIFEST_SHA,
        "--cells", str(ROOT / "registry/f2_cells.json"), "--expected-cells-sha256", CELLS_SHA,
        "--amendment", str(ROOT / "registry/f2_real_family_execution_amendment.json"), "--expected-amendment-sha256", AMENDMENT_SHA,
        "--core-components", "${PRIVATE_STORAGE_ROOT}/CORE_closing_controls/core_components.py", "--expected-core-sha256", CORE_SHA,
        "--output-dir", str(cell.output),
    ]


def write_status(state: str, pending: int, active: dict[int, tuple[Cell, subprocess.Popen]], registered: list[Cell]) -> None:
    path = ROOT / "logs/f2-real/dispatcher-status.json"
    path.write_text(json.dumps({
        "state": state, "updated_epoch": time.time(), "pending": pending,
        "active": {str(group): {"cell_id": cell.cell_id, "pid": process.pid} for group, (cell, process) in active.items()},
        "complete": sum(valid(cell) for cell in registered), "registered": 40,
    }, indent=2, sort_keys=True) + "\n")


def main() -> int:
    for path, expected in ((ROOT / "registry/f2_manifest.json", MANIFEST_SHA), (ROOT / "registry/f2_cells.json", CELLS_SHA), (ROOT / "registry/f2_real_family_execution_amendment.json", AMENDMENT_SHA)):
        if sha256(path) != expected:
            raise RuntimeError(f"frozen file hash mismatch: {path}")
    registered = cells()
    queue = [cell for cell in registered if not valid(cell)]
    logs = ROOT / "logs/f2-real"
    logs.mkdir(parents=True, exist_ok=True)
    active: dict[int, tuple[Cell, subprocess.Popen, object]] = {}
    while queue or active:
        for group in range(8):
            if group in active or not queue:
                continue
            cell = queue.pop(0)
            if cell.output.exists() and not valid(cell):
                raise RuntimeError(f"refusing partial output {cell.output}")
            stream = (logs / f"ccrgb-{cell.method}-{cell.seed}.log").open("w")
            env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = ",".join(str(group * 4 + offset) for offset in range(4))
            process = subprocess.Popen(command(cell), cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            active[group] = (cell, process, stream)
        write_status("running", len(queue), {group: (cell, process) for group, (cell, process, _) in active.items()}, registered)
        time.sleep(5)
        for group, (cell, process, stream) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            stream.close(); del active[group]
            if code != 0 or not valid(cell):
                write_status("failed", len(queue), {key: (value[0], value[1]) for key, value in active.items()}, registered)
                for _, running, open_stream in active.values():
                    running.terminate(); open_stream.close()
                raise RuntimeError(f"cell failed provenance validation: {cell.cell_id}, exit={code}")
    write_status("complete", 0, {}, registered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
