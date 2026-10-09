"""Reshape PubMedCausal annotations to long candidates and assess CORE compatibility."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from lower_priority_common import require_hash, write_empty_outputs, write_jsonl, write_summary


FILES = {
    "30k_train.json": "not-published",
    "30k_test.json": "not-published",
    "validation_set.json": "not-published",
}
RELATION_REASON = "cause/effect span relation has no intervention or paired world states"
NEGATIVE_REASON = "non-causal sentence has no intervention candidate"


def long_candidates(row: dict, release_split: str) -> list[dict]:
    relations = []
    for index in range(1, 17):
        cause = str(row.get(f"Cause {index}", "")).strip()
        effect = str(row.get(f"Effect {index}", "")).strip()
        if cause or effect:
            relations.append({
                "source_id": row.get("s/n"),
                "release_split": release_split,
                "sentence": row.get("Sentence"),
                "relation_index": index,
                "cause": cause or None,
                "effect": effect or None,
                "sententiality": str(row.get(f"Sententiality {index}", "")).strip() or None,
                "causality": str(row.get(f"Causality {index}", "")).strip() or None,
                "is_causal": row.get("is_causal"),
            })
    if relations:
        return relations
    return [{
        "source_id": row.get("s/n"),
        "release_split": release_split,
        "sentence": row.get("Sentence"),
        "relation_index": None,
        "cause": None,
        "effect": None,
        "sententiality": None,
        "causality": None,
        "is_causal": row.get("is_causal"),
    }]


def convert(raw_dir: Path, records_dir: Path, splits_dir: Path, summary_path: Path) -> dict:
    for name, digest in FILES.items():
        require_hash(raw_dir / name, digest)
    candidates = []
    raw_counts = {}
    relation_counts = {}
    for name in FILES:
        split = {"30k_train.json": "train", "30k_test.json": "test", "validation_set.json": "validation"}[name]
        rows = json.loads((raw_dir / name).read_text(encoding="utf-8"))
        raw_counts[split] = len(rows)
        before = len(candidates)
        for row in rows:
            candidates.extend(long_candidates(row, split))
        relation_counts[split] = sum(item["relation_index"] is not None for item in candidates[before:])
    def reason(item: dict) -> str:
        return RELATION_REASON if item["relation_index"] is not None else NEGATIVE_REASON
    reasons = collections.Counter(reason(item) for item in candidates)
    rejected_count = write_jsonl(
        records_dir / "pubmedcausal.rejected.jsonl",
        ({"reason": reason(item), "record": item} for item in candidates),
    )
    write_empty_outputs("pubmedcausal", records_dir, splits_dir)
    summary = {
        "source": "pubmedcausal",
        "pinned_revision": "not-published",
        "raw_files": FILES,
        "raw_row_counts": raw_counts,
        "long_relation_counts": relation_counts,
        "long_candidates": len(candidates),
        "accepted": 0,
        "rejected": rejected_count,
        "rejection_reason_counts": dict(reasons),
        "status": "wide-to-long reshape complete; CORE contract-incompatible",
    }
    write_summary(summary_path, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/pubmedcausal"))
    parser.add_argument("--records-dir", type=Path, default=Path("data/records"))
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--summary", type=Path, default=Path("reports/pubmedcausal_conversion_summary.json"))
    args = parser.parse_args()
    print(json.dumps(convert(args.raw_dir, args.records_dir, args.splits_dir, args.summary), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
