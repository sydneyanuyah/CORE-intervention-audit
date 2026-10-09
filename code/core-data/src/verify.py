"""Verify generated CORE JSONL records and split manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from schema import load_jsonl, validate_record


PRIORITY_SOURCES = {"cladder", "wiqa", "ccrgb", "com2"}


def verify(
    records_dir: Path,
    splits_dir: Path,
    allow_empty: bool = False,
    groups_dir: Optional[Path] = None,
) -> int:
    accepted_files = sorted(
        path for path in records_dir.glob("*.jsonl") if not path.name.endswith(".rejected.jsonl")
    )
    if not accepted_files and not allow_empty:
        print("VERIFY FAIL")
        print("- no accepted JSONL files found")
        return 1

    errors = []
    ids = {}
    record_sources = {}
    record_count = 0
    probe_count = 0
    non_descendant_count = 0

    for path in accepted_files:
        try:
            records = load_jsonl(path)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        for index, record in enumerate(records, 1):
            record_count += 1
            record_id = record.get("id")
            if record_id in ids:
                errors.append(f"duplicate id {record_id!r}: {ids[record_id]} and {path}:{index}")
            else:
                ids[record_id] = f"{path}:{index}"
                record_sources[record_id] = record.get("source")
            for error in validate_record(record):
                errors.append(f"{path}:{index}: {error}")
            probe_count += len(record.get("probes", [])) if isinstance(record.get("probes"), list) else 0
            non_descendant_count += (
                len(record.get("non_descendants", []))
                if isinstance(record.get("non_descendants"), list)
                else 0
            )
            # Explicit in-memory round trip catches non-standard JSON values and coercions.
            try:
                round_trip = json.loads(json.dumps(record, allow_nan=False, sort_keys=True))
            except (TypeError, ValueError) as exc:
                errors.append(f"{path}:{index}: round-trip serialization failed: {exc}")
            else:
                if round_trip != record:
                    errors.append(f"{path}:{index}: round-trip value mismatch")

    assigned = {}
    assigned_split = {}
    split_files = sorted(splits_dir.glob("*.txt"))
    for path in split_files:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            record_id = line.strip()
            if not record_id or record_id.startswith("#"):
                continue
            if record_id not in ids:
                errors.append(f"{path}:{line_number}: unknown record id {record_id!r}")
            if record_id in assigned:
                errors.append(
                    f"record id {record_id!r} appears in multiple split entries: "
                    f"{assigned[record_id]} and {path}:{line_number}"
                )
            else:
                assigned[record_id] = f"{path}:{line_number}"
                split = path.stem.rsplit(".", 1)[-1]
                if split not in {"train", "validation", "test"}:
                    errors.append(f"{path}: split filename must end in train, validation, or test")
                else:
                    assigned_split[record_id] = split

    if split_files:
        unassigned = sorted(set(ids) - set(assigned))
        if unassigned:
            preview = unassigned[:10]
            suffix = "..." if len(unassigned) > len(preview) else ""
            errors.append(f"{len(unassigned)} accepted record ids are not assigned to a split: {preview}{suffix}")

    groups_dir = groups_dir or records_dir.parent / "groups"
    group_mappings = {}
    graph_splits = {}
    world_splits = {}
    for path in sorted(groups_dir.glob("*.jsonl")):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{path}:{line_number}: invalid group JSON: {exc}")
                continue
            if not isinstance(item, dict) or set(item) != {
                "record_id", "graph_group_id", "world_group_id"
            }:
                errors.append(
                    f"{path}:{line_number}: group mapping must contain exactly "
                    "record_id, graph_group_id, and world_group_id"
                )
                continue
            if not all(isinstance(item[key], str) and item[key] for key in item):
                errors.append(f"{path}:{line_number}: group mapping values must be non-empty strings")
                continue
            record_id = item["record_id"]
            if record_id not in ids:
                errors.append(f"{path}:{line_number}: group mapping references unknown record {record_id!r}")
                continue
            if record_id in group_mappings:
                errors.append(
                    f"record id {record_id!r} has multiple group mappings: "
                    f"{group_mappings[record_id]['location']} and {path}:{line_number}"
                )
                continue
            group_mappings[record_id] = {**item, "location": f"{path}:{line_number}"}
            split = assigned_split.get(record_id)
            if split is not None:
                graph_splits.setdefault(item["graph_group_id"], set()).add(split)
                world_splits.setdefault(item["world_group_id"], set()).add(split)

    priority_ids = {
        record_id for record_id, source in record_sources.items() if source in PRIORITY_SOURCES
    }
    missing_groups = sorted(priority_ids - set(group_mappings))
    if missing_groups:
        preview = missing_groups[:10]
        suffix = "..." if len(missing_groups) > len(preview) else ""
        errors.append(
            f"{len(missing_groups)} accepted priority record ids lack a group mapping: "
            f"{preview}{suffix}"
        )
    for group_kind, memberships in (("graph", graph_splits), ("world", world_splits)):
        crossing = sorted(
            (group_id, sorted(splits))
            for group_id, splits in memberships.items()
            if len(splits) > 1
        )
        if crossing:
            preview = crossing[:10]
            suffix = "..." if len(crossing) > len(preview) else ""
            errors.append(
                f"{len(crossing)} {group_kind} group ids cross splits: {preview}{suffix}"
            )

    rejected_count = 0
    for path in sorted(records_dir.glob("*.rejected.jsonl")):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            rejected_count += 1
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{path}:{line_number}: invalid rejected-record JSON: {exc}")
                continue
            if set(item) != {"reason", "record"} or not isinstance(item["reason"], str):
                errors.append(f"{path}:{line_number}: rejection must contain exactly reason and record")

    if errors:
        print("VERIFY FAIL")
        for error in errors:
            print(f"- {error}")
        return 1

    mean_probes = probe_count / record_count if record_count else 0.0
    mean_non_descendants = non_descendant_count / record_count if record_count else 0.0
    print("VERIFY PASS")
    print(f"accepted_files={len(accepted_files)}")
    print(f"records={record_count}")
    print(f"rejected_records={rejected_count}")
    print(f"unique_ids={len(ids)}")
    print(f"split_entries={len(assigned)}")
    print(f"split_coverage={'complete' if len(assigned) == len(ids) else 'not_applicable'}")
    print(f"group_entries={len(group_mappings)}")
    print(f"graph_groups={len(graph_splits)}")
    print(f"world_groups={len(world_splits)}")
    print(f"priority_group_coverage={'complete' if priority_ids <= set(group_mappings) else 'incomplete'}")
    print(f"mean_probes_per_record={mean_probes:.4f}")
    print(f"mean_non_descendants_per_record={mean_non_descendants:.4f}")
    print("round_trip=pass")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records-dir", type=Path, default=Path("data/records"))
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--groups-dir", type=Path)
    parser.add_argument("--allow-empty", action="store_true")
    args = parser.parse_args()
    return verify(
        args.records_dir,
        args.splits_dir,
        allow_empty=args.allow_empty,
        groups_dir=args.groups_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
