"""Assess METER question triplets against the CORE intervention contract."""

from __future__ import annotations

import argparse
import collections
import json
import tarfile
from pathlib import Path

from lower_priority_common import require_hash, write_empty_outputs, write_jsonl, write_summary


SHA256 = "not-published"
NO_INTERVENTION = "causal-discovery question has no intervention"
NO_STATE_PAIR = "natural-language intervention/counterfactual has no explicit graph or paired world states"


def read_contexts(path: Path) -> list[dict]:
    with tarfile.open(path, "r:gz") as archive:
        member = next(m for m in archive.getmembers() if m.name.endswith("/dataset.jsonl"))
        raw = archive.extractfile(member)
        if raw is None:
            raise ValueError("METER dataset.jsonl is unreadable")
        return [json.loads(line) for line in raw.read().decode("utf-8").splitlines() if line]


def rejection_reason(question: dict) -> str:
    return NO_INTERVENTION if question.get("ladder") == "Causal_Discovery" else NO_STATE_PAIR


def convert(input_path: Path, records_dir: Path, splits_dir: Path, summary_path: Path) -> dict:
    require_hash(input_path, SHA256)
    contexts = read_contexts(input_path)
    candidates = []
    for context_index, context in enumerate(contexts):
        for question_index, question in enumerate(context["questions"]):
            candidates.append({
                "context_index": context_index,
                "question_index": question_index,
                "context": context["context"],
                **question,
            })
    reasons = collections.Counter(rejection_reason(row) for row in candidates)
    rejected_count = write_jsonl(
        records_dir / "meter.rejected.jsonl",
        ({"reason": rejection_reason(row), "record": row} for row in candidates),
    )
    write_empty_outputs("meter", records_dir, splits_dir)
    summary = {
        "source": "meter",
        "pinned_revision": "not-published",
        "raw_sha256": SHA256,
        "raw_contexts": len(contexts),
        "raw_questions": len(candidates),
        "ladder_counts": dict(collections.Counter(row["ladder"] for row in candidates)),
        "accepted": 0,
        "rejected": rejected_count,
        "rejection_reason_counts": dict(reasons),
        "status": "contract-incompatible; no licence stated",
    }
    write_summary(summary_path, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/raw/meter/meter-0d2a53b8.tar.gz"))
    parser.add_argument("--records-dir", type=Path, default=Path("data/records"))
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--summary", type=Path, default=Path("reports/meter_conversion_summary.json"))
    args = parser.parse_args()
    print(json.dumps(convert(args.input, args.records_dir, args.splits_dir, args.summary), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
