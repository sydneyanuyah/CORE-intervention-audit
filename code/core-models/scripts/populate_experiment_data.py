#!/usr/bin/env python3
"""Materialize experiment-local train/validation/test data bundles.

This script copies or partitions already-authoritative data.  It never trains,
scores, or evaluates a model.  Test rows are materialized only because the
experiment workspace owner requested complete split bundles; every populated
test directory receives an explicit lock marker.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Optional


ROOT = Path("${CORE_PROJECT_ROOT}")
CENTRAL = ROOT / "data"
EXPERIMENTS = ROOT / "experiments"
SPLITS = ("train", "validation", "test")
CSUITE_GROUPS = sorted([
    "csuite:cat_chain", "csuite:cat_collider", "csuite:cat_to_cts",
    "csuite:cts_to_cat", "csuite:large_backdoor",
    "csuite:large_backdoor_binary_t", "csuite:linexp", "csuite:lingauss",
    "csuite:mixed_confounding", "csuite:mixed_simpson",
    "csuite:nonlin_simpson", "csuite:nonlingauss",
    "csuite:symprod_simpson", "csuite:weak_arrows",
    "csuite:weak_arrows_binary_t",
])

EXPERIMENT_NAMES = {
    "F1": "Experiment-1-F1", "F2": "Experiment-2-F2",
    "F3": "Experiment-3-F3", "F4": "Experiment-4-F4",
    "A1": "Experiment-5-A1", "A2": "Experiment-6-A2",
    "A3": "Experiment-7-A3", "L1": "Experiment-8-L1",
    "L2": "Experiment-9-L2", "L3": "Experiment-10-L3",
    "C1": "Experiment-11-C1", "C2": "Experiment-12-C2",
    "C3": "Experiment-13-C3", "C4": "Experiment-14-C4",
    "T1": "Experiment-15-T1", "T2": "Experiment-16-T2",
    "T3": "Experiment-17-T3", "T4": "Experiment-18-T4",
    "T5": "Experiment-19-T5", "G1": "Experiment-20-G1",
    "G2": "Experiment-21-G2", "G3": "Experiment-22-G3",
    "G4": "Experiment-23-G4",
}

# Yes = the formal experiment needs this split.  Training families receive a
# locked test bundle for the later one-shot evaluation; validation-only
# diagnostics intentionally do not.
REQUIREMENTS = {
    "F1": (1, 1, 0), "F2": (1, 1, 0), "F3": (1, 1, 0),
    "F4": (1, 1, 0), "A1": (1, 1, 1), "A2": (0, 1, 0),
    "A3": (0, 1, 0), "L1": (1, 1, 0), "L2": (1, 1, 0),
    "L3": (1, 1, 0), "C1": (1, 1, 0), "C2": (0, 1, 0),
    "C3": (0, 1, 0), "C4": (0, 0, 0), "T1": (0, 1, 0),
    "T2": (1, 1, 1), "T3": (0, 1, 0), "T4": (1, 1, 1),
    "T5": (0, 1, 1), "G1": (0, 1, 0), "G2": (0, 1, 0),
    "G3": (1, 1, 0), "G4": (1, 1, 0),
}

# Dataset assignments.  The symbolic names are resolved below.
ASSIGNMENTS = {
    "F1": ["xor_f1"],
    "F2": ["xor_f1", "ccrgb", "cladder", "wiqa", "com2"],
    "F3": ["c1_real", "c1_noncausal_placebo"],
    "F4": ["xor_f1"],
    "A1": ["xor_a1", "ccrgb", "cladder", "wiqa", "com2"],
    "A2": ["xor_a1", "ccrgb", "cladder", "wiqa", "com2"],
    "A3": ["xor_a1"],
    "L1": ["xor_f1", "cladder"],
    "L2": ["xor_f1", "cladder"],
    "L3": ["xor_f1", "c2_floor_suite"],
    "C1": ["c1_real", "c1_noncausal_placebo", "c1_shuffled"],
    "C2": ["c2_floor_suite"],
    "C3": ["xor_f1", "ccrgb", "cladder", "wiqa", "com2"],
    "C4": [],
    "T1": ["f2_composed_pairs"],
    "T2": ["csuite_t2"],
    "T3": ["corr2cause_t3"],
    "T4": ["csuite_t4"],
    "T5": ["native_t5"],
    "G1": ["cladder"],
    "G2": ["shared_target_sequences"],
    "G3": ["xor_g3"],
    "G4": ["evidence_inference_g4"],
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def assigned_split(dataset: str, row: dict) -> Optional[str]:
    """Return the explicit split, or deterministically split generated CSuite dev."""
    source_split = row.get("split")
    if source_split != "development_generated":
        return source_split
    if dataset not in {"csuite_t2", "csuite_t4"}:
        return source_split
    group = str(row.get("graph_group_id") or row.get("world_group_id") or row.get("id"))
    # Domain-level split: all examples from one SEM family remain together.
    # Eleven domains train, two validate, and two form the locked test set.
    rank = CSUITE_GROUPS.index(group)
    return "train" if rank < 11 else ("validation" if rank < 13 else "test")


def atomic_jsonl_partition(dataset: str, inputs: list[Path], split: str, destination: Path) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(destination.suffix + ".tmp")
    count = 0
    with temp.open("w", encoding="utf-8") as out:
        for source in inputs:
            with source.open("r", encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    if assigned_split(dataset, row) == split:
                        out.write(line if line.endswith("\n") else line + "\n")
                        count += 1
    if count:
        os.replace(temp, destination)
    else:
        temp.unlink(missing_ok=True)
        destination.unlink(missing_ok=True)
    return count


def atomic_indexed_partition(record_sources: list[Path], ids_file: Path, destination: Path) -> int:
    wanted = {line.strip() for line in ids_file.open(encoding="utf-8") if line.strip()}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(destination.suffix + ".tmp")
    count = 0
    with temp.open("w", encoding="utf-8") as out:
        for source in record_sources:
            with source.open("r", encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    if row.get("id") in wanted:
                        out.write(line if line.endswith("\n") else line + "\n")
                        count += 1
    if count != len(wanted):
        temp.unlink(missing_ok=True)
        raise RuntimeError(f"{destination}: found {count} records for {len(wanted)} IDs")
    os.replace(temp, destination)
    shutil.copy2(ids_file, destination.with_name(destination.stem + ".ids.txt"))
    return count


def internal_sources(dataset: str) -> list[Path]:
    patterns = {
        "xor_f1": [CENTRAL / "two_edit/f1/xor_graph_*.jsonl"],
        "xor_a1": [CENTRAL / "two_edit/xor_graph_*.jsonl", CENTRAL / "two_edit/confirmatory/xor_graph_*.jsonl"],
        "xor_g3": [CENTRAL / "two_edit/xor_graph_*.jsonl"],
        "c1_real": [CENTRAL / "experiments/c1/real/graph_*.jsonl"],
        "c1_noncausal_placebo": [CENTRAL / "experiments/c1/noncausal_placebo/graph_*.jsonl"],
        "c1_shuffled": [CENTRAL / "experiments/c1/shuffled/graph_*.jsonl"],
        "c2_floor_suite": [CENTRAL / "experiments/c2/floor_suite.jsonl"],
        "f2_composed_pairs": [CENTRAL / "experiments/t1/f2_composed_pairs.jsonl"],
        "csuite_t2": [CENTRAL / "experiments/t2/csuite_pairs.jsonl"],
        "corr2cause_t3": [CENTRAL / "experiments/t3/corr2cause_pairs.jsonl"],
        "csuite_t4": [CENTRAL / "experiments/t4/csuite_rung_views.jsonl"],
        "native_t5": [CENTRAL / "experiments/t5/native_suite.jsonl"],
        "shared_target_sequences": [CENTRAL / "experiments/g2/sequences.jsonl"],
        "evidence_inference_g4": [CENTRAL / "experiments/g4/evidence_inference/adjudicated_prompts.jsonl"],
    }
    found: list[Path] = []
    for pattern in patterns[dataset]:
        found.extend(sorted(pattern.parent.glob(pattern.name)))
    return found


def indexed_sources(dataset: str) -> tuple[list[Path], str]:
    if dataset == "com2":
        return ([CENTRAL / "records/com2_counterfactual.jsonl", CENTRAL / "records/com2_intervention.jsonl"], "com2")
    return ([CENTRAL / f"records/{dataset}.jsonl"], dataset)


def populate_dataset(exp: str, dataset: str, split: str, destination: Path) -> tuple[str, int, str]:
    if dataset in {"ccrgb", "cladder", "wiqa", "com2"}:
        sources, prefix = indexed_sources(dataset)
        if dataset == "com2":
            counts = 0
            for variant, source in (("com2_counterfactual", sources[0]), ("com2_intervention", sources[1])):
                ids_file = CENTRAL / f"splits/{variant}.{split}.txt"
                if ids_file.exists() and ids_file.stat().st_size:
                    counts += atomic_indexed_partition([source], ids_file, destination / f"{variant}.jsonl")
            return ("present", counts, "") if counts else ("missing", 0, f"no {split} Com2 IDs")
        ids_file = CENTRAL / f"splits/{prefix}.{split}.txt"
        if not ids_file.exists() or not ids_file.stat().st_size:
            return "missing", 0, f"missing {ids_file.relative_to(ROOT)}"
        return "present", atomic_indexed_partition(sources, ids_file, destination / f"{dataset}.jsonl"), ""
    sources = internal_sources(dataset)
    if not sources:
        return "missing", 0, f"no source artifact for {dataset}"
    count = atomic_jsonl_partition(dataset, sources, split, destination / f"{dataset}.jsonl")
    return ("present", count, "") if count else ("missing", 0, f"source has no {split} rows")


def datasets_for(exp: str, split: str) -> list[str]:
    # The executable floor suite calibrates L3 on validation; it is not a
    # training corpus and therefore must not make L3 train/test look missing.
    if exp == "L3":
        return ["xor_f1", "c2_floor_suite"] if split == "validation" else ["xor_f1"]
    return ASSIGNMENTS[exp]


def main() -> None:
    audit = []
    for exp, directory_name in EXPERIMENT_NAMES.items():
        base = EXPERIMENTS / directory_name / "data"
        base.mkdir(parents=True, exist_ok=True)
        required = dict(zip(SPLITS, REQUIREMENTS[exp]))
        row = {"experiment": exp, "directory": directory_name, "splits": {}}
        for split in SPLITS:
            split_dir = base / split
            split_dir.mkdir(parents=True, exist_ok=True)
            # Remove only files generated by this script so a changed mapping
            # cannot leave stale experiment-local placeholders behind.
            for pattern in ("*.jsonl", "*.ids.txt", "NOT_REQUIRED.txt", "LOCKED_DO_NOT_EVALUATE.txt"):
                for stale in split_dir.glob(pattern):
                    stale.unlink()
            if split == "test" and required[split]:
                (split_dir / "LOCKED_DO_NOT_EVALUATE.txt").write_text(
                    "Held-out test data. Do not evaluate until the separately authorized one-shot final evaluation.\n",
                    encoding="utf-8",
                )
            if not required[split]:
                (split_dir / "NOT_REQUIRED.txt").write_text(
                    f"The registered {exp} protocol does not require the {split} split.\n",
                    encoding="utf-8",
                )
                row["splits"][split] = {"required": False, "status": "not_required", "records": 0, "datasets": []}
                continue
            details = []
            total = 0
            all_present = True
            assigned = datasets_for(exp, split)
            for dataset in assigned:
                status, count, note = populate_dataset(exp, dataset, split, split_dir)
                details.append({"dataset": dataset, "status": status, "records": count, "note": note})
                total += count
                all_present &= status == "present"
            status = "present" if all_present and assigned else "missing"
            row["splits"][split] = {"required": True, "status": status, "records": total, "datasets": details}
        manifest = base / "DATA_MANIFEST.json"
        manifest.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        row["manifest_sha256"] = sha256(manifest)
        audit.append(row)

    audit_path = EXPERIMENTS / "DATA_SPLIT_AUDIT.json"
    audit_path.write_text(json.dumps({"experiments": audit}, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Experiment data split audit", "",
        "Any required test bundles are copied and locked; they have not been scored or evaluated. Train/validation-only protocols are marked `Not required` rather than incomplete.", "",
        "| # | Experiment | Training | Validation | Test | Overall data status | Missing / note |",
        "|---:|---|---|---|---|---|---|",
    ]
    present = []
    missing = []
    for index, (exp, directory_name) in enumerate(EXPERIMENT_NAMES.items(), 1):
        row = audit[index - 1]
        cells = []
        for split in SPLITS:
            info = row["splits"][split]
            if not info["required"]:
                cells.append("Not required")
            elif info["status"] == "present":
                cells.append(f"Present ({info['records']:,} rows)")
            else:
                cells.append("Missing")
        required_infos = [row["splits"][s] for s in SPLITS if row["splits"][s]["required"]]
        overall = "Complete" if required_infos and all(i["status"] == "present" for i in required_infos) else ("No dataset required" if not required_infos else "Incomplete")
        notes = []
        for split in SPLITS:
            info = row["splits"][split]
            if info["required"] and info["status"] != "present":
                missing_sets = [d["dataset"] for d in info["datasets"] if d["status"] != "present"]
                notes.append(f"{split}: {', '.join(missing_sets)}")
        note = "; ".join(notes) if notes else ("No dataset is used" if overall == "No dataset required" else "—")
        complete_marker = EXPERIMENTS / directory_name / "DATA_COMPLETE.txt"
        incomplete_marker = EXPERIMENTS / directory_name / "DATA_INCOMPLETE.txt"
        if overall in {"Complete", "No dataset required"}:
            incomplete_marker.unlink(missing_ok=True)
            complete_marker.write_text(f"{exp} data status: {overall}. See data/DATA_MANIFEST.json.\n", encoding="utf-8")
        else:
            complete_marker.unlink(missing_ok=True)
            incomplete_marker.write_text(f"{exp} data status: Incomplete. Missing {note}. See data/DATA_MANIFEST.json.\n", encoding="utf-8")
        (present if overall in {"Complete", "No dataset required"} else missing).append(exp)
        lines.append(f"| {index} | {exp} | {cells[0]} | {cells[1]} | {cells[2]} | **{overall}** | {note} |")
    lines += ["", f"**Complete / no dataset required:** {', '.join(present)}", "", f"**Still missing required data:** {', '.join(missing) if missing else 'None'}", ""]
    (EXPERIMENTS / "DATA_SPLIT_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    print(EXPERIMENTS / "DATA_SPLIT_AUDIT.md")


if __name__ == "__main__":
    main()
