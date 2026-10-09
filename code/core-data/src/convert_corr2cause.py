"""Assess the Corr2Cause paraphrased test set against the CORE contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lower_priority_common import require_hash, write_empty_outputs, write_jsonl, write_summary


SHA256 = "not-published"
REASON = "correlation-to-causal-relation classification has no intervention or paired world states"


def convert(input_path: Path, records_dir: Path, splits_dir: Path, summary_path: Path) -> dict:
    require_hash(input_path, SHA256)
    rows = json.loads(input_path.read_text(encoding="utf-8"))
    rejected_count = write_jsonl(
        records_dir / "corr2cause.rejected.jsonl",
        ({"reason": REASON, "record": row} for row in rows),
    )
    write_empty_outputs("corr2cause", records_dir, splits_dir)
    summary = {
        "source": "corr2cause",
        "pinned_revision": "not-published",
        "raw_sha256": SHA256,
        "raw_questions": len(rows),
        "accepted": 0,
        "rejected": rejected_count,
        "rejection_reason_counts": {REASON: rejected_count},
        "status": "contract-incompatible",
    }
    write_summary(summary_path, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/raw/corr2cause/perturbation_by_paraphrasing_test.json"))
    parser.add_argument("--records-dir", type=Path, default=Path("data/records"))
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--summary", type=Path, default=Path("reports/corr2cause_conversion_summary.json"))
    args = parser.parse_args()
    print(json.dumps(convert(args.input, args.records_dir, args.splits_dir, args.summary), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
