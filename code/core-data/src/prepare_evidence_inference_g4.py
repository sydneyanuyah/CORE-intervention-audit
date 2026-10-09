"""Convert Evidence Inference 2.0 development labels into G4 native records."""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import io
import json
import tarfile
from pathlib import Path

from inspect_evidence_inference import (
    PINNED_REVISION, SOURCE_SHA256, TRAIN_IDS, VALIDATION_IDS, read_member, sha256_file,
)


PROTOCOL = "g4_evidence_inference_article_grouped_v1"
LABELS = {
    "no significant difference": "no_change",
    "significantly decreased": "decreased",
    "significantly increase": "increased",
    "significantly increased": "increased",
}


def ids(payload: bytes) -> set[str]:
    return {line.strip() for line in payload.decode("utf-8-sig").splitlines() if line.strip()}


def write_jsonl(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            line = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            handle.write(line); digest.update(line.encode())
    return digest.hexdigest()


def prepare(source: Path, output_root: Path, summary_path: Path) -> dict:
    if sha256_file(source) != SOURCE_SHA256:
        raise ValueError("Evidence Inference source SHA-256 mismatch")
    with tarfile.open(source, "r:gz") as archive:
        train, validation = ids(read_member(archive, TRAIN_IDS)), ids(read_member(archive, VALIDATION_IDS))
        prompts = list(csv.DictReader(io.StringIO(read_member(archive, "annotations/prompts_merged.csv").decode("utf-8-sig"))))
        annotations = list(csv.DictReader(io.StringIO(read_member(archive, "annotations/v2_annotations.csv").decode("utf-8-sig"))))
    split_of = {article: "train" for article in train} | {article: "validation" for article in validation}
    prompt_by_id = {row["PromptID"]: row for row in prompts if row["PMCID"] in split_of}
    doctor_rows: list[dict] = []
    grouped: collections.defaultdict[str, list[dict]] = collections.defaultdict(list)
    for source_row, row in enumerate(annotations):
        if row.get("Valid Label", "").strip().lower() != "true":
            continue
        prompt = prompt_by_id.get(row.get("PromptID", ""))
        label = LABELS.get(row.get("Label", "").strip().lower())
        if prompt is None or label is None or row.get("PMCID") != prompt.get("PMCID"):
            continue
        item = {
            "id": f"ei2-{prompt['PMCID']}-{prompt['PromptID']}-doctor-{row['UserID']}-row-{source_row}",
            "protocol": PROTOCOL,
            "split": split_of[prompt["PMCID"]],
            "article_group_id": f"pmcid:{prompt['PMCID']}",
            "prompt_id": prompt["PromptID"],
            "intervention": prompt["Intervention"].strip(),
            "comparator": prompt["Comparator"].strip(),
            "outcome": prompt["Outcome"].strip(),
            "doctor_id": row["UserID"],
            "label": label,
            "evidence_text": row["Annotations"].strip(),
            "evidence_start": row["Evidence Start"].strip() or None,
            "evidence_end": row["Evidence End"].strip() or None,
            "valid_reasoning": row["Valid Reasoning"].strip().lower() == "true",
            "test_evaluated": False,
        }
        doctor_rows.append(item); grouped[prompt["PromptID"]].append(item)
    adjudicated, ties = [], 0
    for prompt_id, rows in sorted(grouped.items(), key=lambda item: int(item[0])):
        counts = collections.Counter(row["label"] for row in rows)
        top = max(counts.values())
        winners = sorted(label for label, count in counts.items() if count == top)
        if len(winners) != 1:
            ties += 1
            continue
        first = rows[0]
        adjudicated.append({
            "id": f"ei2-{first['article_group_id'].split(':', 1)[1]}-{prompt_id}",
            "protocol": PROTOCOL,
            "split": first["split"],
            "article_group_id": first["article_group_id"],
            "prompt_id": prompt_id,
            "intervention": first["intervention"],
            "comparator": first["comparator"],
            "outcome": first["outcome"],
            "label": winners[0],
            "doctor_label_counts": dict(sorted(counts.items())),
            "doctor_annotation_ids": [row["id"] for row in rows],
            "aggregation": "unique plurality over all valid doctor labels; tied prompts quarantined",
            "test_evaluated": False,
        })
    doctor_hash = write_jsonl(output_root / "doctor_annotations.jsonl", doctor_rows)
    aggregate_hash = write_jsonl(output_root / "adjudicated_prompts.jsonl", adjudicated)
    article_manifest = collections.defaultdict(lambda: {"train": [], "validation": []})
    for row in adjudicated:
        article_manifest[row["article_group_id"]][row["split"]].append(row["id"])
    manifest_rows = [
        {"article_group_id": article, "split": "train" if values["train"] else "validation", "prompt_ids": values["train"] or values["validation"]}
        for article, values in sorted(article_manifest.items())
    ]
    manifest_hash = write_jsonl(output_root / "article_manifest.jsonl", manifest_rows)
    summary = {
        "protocol": PROTOCOL, "pinned_revision": PINNED_REVISION, "raw_sha256": SOURCE_SHA256,
        "valid_doctor_annotations": len(doctor_rows), "adjudicated_prompts": len(adjudicated),
        "tied_prompts_quarantined": ties,
        "articles": len(manifest_rows),
        "articles_by_split": dict(collections.Counter(row["split"] for row in manifest_rows)),
        "labels": dict(collections.Counter(row["label"] for row in adjudicated)),
        "doctor_annotations_sha256": doctor_hash, "adjudicated_prompts_sha256": aggregate_hash,
        "article_manifest_sha256": manifest_hash, "test_evaluated": False,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("data/raw/evidence_inference/evidence-inference-a661e8c1-development-source.tar.gz"))
    parser.add_argument("--output-root", type=Path, default=Path("data/g4/evidence_inference"))
    parser.add_argument("--summary", type=Path, default=Path("reports/g4_evidence_inference_summary.json"))
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output_root, args.summary), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
