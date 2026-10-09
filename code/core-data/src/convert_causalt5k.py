"""Assess the ten CausalT5K domains against the CORE intervention contract."""

from __future__ import annotations

import argparse
import collections
import json
import tarfile
from pathlib import Path

from lower_priority_common import require_hash, write_empty_outputs, write_jsonl, write_summary


SHA256 = "not-published"
NO_INTERVENTION = "L1 association item has no intervention"
NO_STATE_PAIR = "item has no machine-readable edge list or paired factual/intervened states"


def read_rows(path: Path) -> list[dict]:
    rows = []
    with tarfile.open(path, "r:gz") as archive:
        for domain in range(1, 11):
            for level in ("L1", "L2", "L3"):
                suffix = f"/final_dataset/D{domain}/D{domain}_{level}.json"
                member = next(m for m in archive.getmembers() if m.name.endswith(suffix))
                raw = archive.extractfile(member)
                if raw is None:
                    raise ValueError(f"unreadable archive member {member.name}")
                for item in json.loads(raw.read().decode("utf-8")):
                    rows.append({"release_domain": f"D{domain}", "release_level": level, **item})
    return rows


def rejection_reason(item: dict) -> str:
    return NO_INTERVENTION if item["release_level"] == "L1" else NO_STATE_PAIR


def convert(input_path: Path, records_dir: Path, splits_dir: Path, summary_path: Path) -> dict:
    require_hash(input_path, SHA256)
    rows = read_rows(input_path)
    reasons = collections.Counter(rejection_reason(row) for row in rows)
    rejected_count = write_jsonl(
        records_dir / "causalt5k.rejected.jsonl",
        ({"reason": rejection_reason(row), "record": row} for row in rows),
    )
    write_empty_outputs("causalt5k", records_dir, splits_dir)
    summary = {
        "source": "causalt5k",
        "pinned_revision": "not-published",
        "raw_sha256": SHA256,
        "raw_questions": len(rows),
        "domain_counts": dict(collections.Counter(row["release_domain"] for row in rows)),
        "level_counts": dict(collections.Counter(row["release_level"] for row in rows)),
        "accepted": 0,
        "rejected": rejected_count,
        "rejection_reason_counts": dict(reasons),
        "status": "contract-incompatible",
        "future_split_policy": "leave one complete named domain out per evaluation fold if a compatible representation is added",
    }
    write_summary(summary_path, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/raw/causalt5k/causalt5k-fd358e95.tar.gz"))
    parser.add_argument("--records-dir", type=Path, default=Path("data/records"))
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--summary", type=Path, default=Path("reports/causalt5k_conversion_summary.json"))
    args = parser.parse_args()
    print(json.dumps(convert(args.input, args.records_dir, args.splits_dir, args.summary), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
