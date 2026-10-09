"""Assess CRASS fixed-target counterfactual questions against the CORE contract."""

from __future__ import annotations

import argparse
import csv
import io
import json
import tarfile
from pathlib import Path

from lower_priority_common import require_hash, write_empty_outputs, write_jsonl, write_summary


SHA256 = "not-published"
REASON = "counterfactual multiple-choice item has no explicit causal graph or paired state maps"


def read_rows(path: Path) -> list[dict]:
    with tarfile.open(path, "r:gz") as archive:
        member = next(m for m in archive.getmembers() if m.name.endswith("/CRASS_FTM_main_data_set.csv"))
        raw = archive.extractfile(member)
        if raw is None:
            raise ValueError("CRASS CSV is unreadable")
        return list(csv.DictReader(io.StringIO(raw.read().decode("utf-8-sig")), delimiter=";"))


def convert(input_path: Path, records_dir: Path, splits_dir: Path, summary_path: Path) -> dict:
    require_hash(input_path, SHA256)
    rows = read_rows(input_path)
    if len(rows) != 274:
        raise ValueError(f"expected 274 CRASS rows, got {len(rows)}")
    rejected_count = write_jsonl(
        records_dir / "crass.rejected.jsonl",
        ({"reason": REASON, "record": row} for row in rows),
    )
    write_empty_outputs("crass", records_dir, splits_dir)
    summary = {
        "source": "crass",
        "pinned_revision": "not-published",
        "raw_sha256": SHA256,
        "raw_questions": len(rows),
        "accepted": 0,
        "rejected": rejected_count,
        "rejection_reason_counts": {REASON: rejected_count},
        "status": "evaluation-only; contract-incompatible",
    }
    write_summary(summary_path, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/raw/crass/crass-3944517b.tar.gz"))
    parser.add_argument("--records-dir", type=Path, default=Path("data/records"))
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--summary", type=Path, default=Path("reports/crass_conversion_summary.json"))
    args = parser.parse_args()
    print(json.dumps(convert(args.input, args.records_dir, args.splits_dir, args.summary), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
