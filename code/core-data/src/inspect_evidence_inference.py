"""Inspect Evidence Inference 2.0 train/validation annotations only."""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import io
import json
import tarfile
from pathlib import Path


PINNED_REVISION = "not-published"
SOURCE_SHA256 = "not-published"
TRAIN_IDS = "annotations/splits/ev2_train_article_ids.txt"
VALIDATION_IDS = "annotations/splits/ev2_validation_article_ids.txt"
TEST_IDS = "annotations/splits/ev2_test_article_ids.txt"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_member(archive: tarfile.TarFile, name: str) -> bytes:
    if name == TEST_IDS or "test_article_ids" in name:
        raise ValueError(f"held-out Evidence Inference split is sealed: {name}")
    handle = archive.extractfile(name)
    if handle is None:
        raise ValueError(f"missing archive member: {name}")
    return handle.read()


def _ids(payload: bytes) -> set[str]:
    return {line.strip() for line in payload.decode("utf-8-sig").splitlines() if line.strip()}


def inspect(path: Path) -> dict:
    actual_hash = sha256_file(path)
    if actual_hash != SOURCE_SHA256:
        raise ValueError("Evidence Inference source SHA-256 mismatch")
    with tarfile.open(path, "r:gz") as archive:
        train_ids = _ids(read_member(archive, TRAIN_IDS))
        validation_ids = _ids(read_member(archive, VALIDATION_IDS))
        if train_ids & validation_ids:
            raise ValueError("Evidence Inference train and validation articles overlap")
        prompts = list(csv.DictReader(io.StringIO(
            read_member(archive, "annotations/prompts_merged.csv").decode("utf-8-sig")
        )))
        annotations = list(csv.DictReader(io.StringIO(
            read_member(archive, "annotations/v2_annotations.csv").decode("utf-8-sig")
        )))
    development_ids = train_ids | validation_ids
    dev_prompts = {row["PromptID"]: row for row in prompts if row["PMCID"] in development_ids}
    dev_annotations = [
        row for row in annotations
        if row["PMCID"] in development_ids and row["PromptID"] in dev_prompts
    ]
    valid = [row for row in dev_annotations if row["Valid Label"].strip().lower() == "true"]
    label_counts = collections.Counter(row["Label"].strip().lower() for row in valid)
    return {
        "source": "evidence_inference_2",
        "repository_revision": PINNED_REVISION,
        "raw_sha256": actual_hash,
        "train_articles": len(train_ids),
        "validation_articles": len(validation_ids),
        "development_prompts": len(dev_prompts),
        "development_annotation_rows": len(dev_annotations),
        "valid_label_rows": len(valid),
        "valid_label_counts": dict(sorted(label_counts.items())),
        "task": "comparative intervention/comparator outcome direction",
        "protocol_boundary": "trial-level comparative evidence; not a complete DAG world state",
        "test_evaluated": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path,
        default=Path("data/raw/evidence_inference/evidence-inference-a661e8c1-development-source.tar.gz"),
    )
    parser.add_argument(
        "--summary", type=Path,
        default=Path("reports/evidence_inference_schema_summary.json"),
    )
    args = parser.parse_args()
    summary = inspect(args.input)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
