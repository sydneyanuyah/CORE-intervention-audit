#!/usr/bin/env python3
"""Generate a development-only structured instruction paraphrase manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from core_bert.benchmark_data import load_benchmark_split
from core_bert.structured_paraphrases import (
    build_structured_partition,
    structured_catalog_manifest,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--sources", nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument(
        "--source-splits", nargs="+", default=["train", "validation"],
        help="Development splits used for catalog coverage; test is forbidden.",
    )
    args = parser.parse_args(argv)
    if "test" in args.source_splits:
        parser.error("held-out test cannot be used to build a paraphrase catalog")
    unsupported = set(args.source_splits) - {"train", "validation"}
    if unsupported:
        parser.error(f"unsupported source splits: {sorted(unsupported)}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    records = {
        split: load_benchmark_split(
            args.data_root, split, sources=args.sources
        ).records
        for split in args.source_splits
    }
    partition, provenance = build_structured_partition(
        records, seed=args.seed, validation_fraction=args.validation_fraction
    )
    manifest = structured_catalog_manifest(partition, provenance)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output_json),
                "catalog_sha256": partition.catalog_sha256,
                "counts": partition.to_manifest()["counts"],
                "evidence_status": partition.evidence_status,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
