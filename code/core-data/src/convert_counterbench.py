"""Assess CounterBench against the CORE intervention record contract."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from lower_priority_common import require_hash, write_empty_outputs, write_jsonl, write_summary


FILES = {
    "data_balanced_alpha_V1.json": "not-published",
    "data_balanced_backdoor_V2.json": "not-published",
    "meta_model_alpha_V1.json": "not-published",
    "meta_model_backdoor_V2.json": "not-published",
}


def rejection_reason(item: dict) -> str:
    if item.get("type") in {"joint", "nested"}:
        return "multiple simultaneous interventions cannot be represented by one intervention.target"
    return "counterfactual answer is supplied without a factual outcome/state pair needed to prove an observed change"


def convert(raw_dir: Path, records_dir: Path, splits_dir: Path, summary_path: Path) -> dict:
    for name, digest in FILES.items():
        require_hash(raw_dir / name, digest)
    alpha = json.loads((raw_dir / "data_balanced_alpha_V1.json").read_text(encoding="utf-8"))
    backdoor = json.loads((raw_dir / "data_balanced_backdoor_V2.json").read_text(encoding="utf-8"))
    candidates = [("alpha", item) for item in alpha] + [("backdoor", item) for item in backdoor]
    reasons = collections.Counter(rejection_reason(item) for _, item in candidates)
    rejected = (
        {"reason": rejection_reason(item), "record": {"release": release, **item}}
        for release, item in candidates
    )
    rejected_count = write_jsonl(records_dir / "counterbench.rejected.jsonl", rejected)
    write_empty_outputs("counterbench", records_dir, splits_dir)
    summary = {
        "source": "counterbench",
        "pinned_revision": "not-published",
        "raw_files": FILES,
        "raw_questions": len(candidates),
        "accepted": 0,
        "rejected": rejected_count,
        "rejection_reason_counts": dict(reasons),
        "status": "contract-incompatible",
    }
    write_summary(summary_path, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/counterbench"))
    parser.add_argument("--records-dir", type=Path, default=Path("data/records"))
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--summary", type=Path, default=Path("reports/counterbench_conversion_summary.json"))
    args = parser.parse_args()
    print(json.dumps(convert(args.raw_dir, args.records_dir, args.splits_dir, args.summary), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
