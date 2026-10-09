"""Prepare validation-only native Corr2Cause robustness pairs for T3.

This does not coerce Corr2Cause into the CORE intervention schema.  It retains
the source task as binary causal-relation classification and aligns the clean,
variable-refactorized, and paraphrased development releases.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any


REVISION = "not-published"
PROTOCOL = "corr2cause_native_robustness_v1"
RAW_HASHES = {
    "clean": "not-published",
    "paraphrase": "not-published",
    "refactorization": "not-published",
}
ID_PATTERN = re.compile(
    r"^num_nodes=(?P<n>\d+)__mec_id=(?P<mec>\d+)__node_i=\d+__node_j=\d+"
    r"__causal_relation=[a-z_-]+__prob=[0-9.]+$"
)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def refactor_variables(text: str, count: int) -> str:
    if not 2 <= count <= 26:
        raise ValueError(f"unsupported Corr2Cause variable count {count}")
    aliases = {chr(65 + index): chr(90 - index) for index in range(count)}
    return re.sub(r"\b[A-Z]\b", lambda match: aliases.get(match.group(), match.group()), text)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _native_paraphrase(row: dict[str, Any]) -> str:
    return f"Premise: {row['premise']}\nHypothesis: {row['hypothesis']}"


def prepare(
    clean_path: Path,
    paraphrase_path: Path,
    refactor_path: Path,
    *,
    expected_counts: tuple[int, int, int] = (1076, 2246, 2246),
) -> tuple[list[dict], dict]:
    actual_hashes = {
        "clean": file_hash(clean_path),
        "paraphrase": file_hash(paraphrase_path),
        "refactorization": file_hash(refactor_path),
    }
    if actual_hashes != RAW_HASHES:
        raise ValueError(f"Corr2Cause raw hash mismatch: {actual_hashes}")
    clean = _read_csv(clean_path)
    refactored = _read_csv(refactor_path)
    paraphrased = json.loads(paraphrase_path.read_text(encoding="utf-8"))
    if (len(clean), len(refactored), len(paraphrased)) != expected_counts:
        raise ValueError("unexpected Corr2Cause development counts")

    by_refactored: dict[str, list[int]] = collections.defaultdict(list)
    for index, row in enumerate(refactored):
        if row.get("label") not in {"0", "1"} or not row.get("input"):
            raise ValueError(f"invalid refactorized row {index}")
        by_refactored[row["input"]].append(index)

    output = []
    duplicate_collapses = 0
    for clean_index, row in enumerate(clean):
        if row.get("label") not in {"0", "1"}:
            raise ValueError(f"invalid clean label at row {clean_index}")
        count = int(row["num_variables"])
        expected_refactor = refactor_variables(row["input"], count)
        candidates = by_refactored.get(expected_refactor, [])
        if not candidates:
            raise ValueError(f"clean row {clean_index} has no exact refactorized counterpart")
        if len(candidates) > 1:
            duplicate_collapses += 1
        candidate_views = {
            (
                refactored[index]["label"],
                _native_paraphrase(paraphrased[index]),
                paraphrased[index]["relation"],
            )
            for index in candidates
        }
        if len(candidate_views) != 1:
            raise ValueError(f"ambiguous perturbation alignment for clean row {clean_index}")
        source_index = min(candidates)
        para = paraphrased[source_index]
        binary_label = "1" if para["relation"] == "entailment" else "0"
        if binary_label != row["label"] or refactored[source_index]["label"] != row["label"]:
            raise ValueError(f"label mismatch at clean row {clean_index}")
        match = ID_PATTERN.fullmatch(para.get("id", ""))
        if match is None or int(match.group("n")) != count:
            raise ValueError(f"invalid paraphrase identity at source row {source_index}")
        clean_sha = stable_hash(row["input"])
        pair = {
            "pair_id": f"corr2cause-dev:{clean_sha[:24]}",
            "protocol": PROTOCOL,
            "split": "validation",
            "task": "binary_causal_relation_classification",
            "label": int(row["label"]),
            "template": row["template"],
            "num_variables": count,
            "graph_group_id": f"corr2cause:n{count}:mec{match.group('mec')}",
            "source_rows": {"clean": clean_index, "perturbations": candidates},
            "clean": {"input": row["input"], "sha256": clean_sha},
            "refactorization": {
                "input": expected_refactor,
                "sha256": stable_hash(expected_refactor),
                "construction": "official reverse-alphabet variable refactorization",
            },
            "paraphrase": {
                "input": _native_paraphrase(para),
                "sha256": stable_hash(_native_paraphrase(para)),
            },
            "test_evaluated": False,
        }
        output.append(pair)
    if len({row["pair_id"] for row in output}) != len(output):
        raise ValueError("Corr2Cause pair IDs are not unique")
    summary = {
        "source": "corr2cause",
        "protocol": PROTOCOL,
        "pinned_revision": REVISION,
        "raw_sha256": actual_hashes,
        "clean_validation_rows": len(clean),
        "paired_validation_rows": len(output),
        "official_perturbation_rows": len(refactored),
        "duplicate_equivalent_perturbation_groups_collapsed": duplicate_collapses,
        "labels": dict(sorted(collections.Counter(row["label"] for row in output).items())),
        "graph_groups": len({row["graph_group_id"] for row in output}),
        "pair_manifest_sha256": stable_hash(output),
        "test_evaluated": False,
        "scope": "T3 native classification robustness only; not CORE intervention-state evidence",
    }
    return output, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    root = Path("data/raw/corr2cause")
    parser.add_argument("--clean", type=Path, default=root / "dev.csv")
    parser.add_argument("--paraphrase", type=Path, default=root / "perturbation_by_paraphrasing_dev.json")
    parser.add_argument("--refactorization", type=Path, default=root / "perturbation_by_refactorization_dev.csv")
    parser.add_argument("--output", type=Path, default=Path("data/t3/corr2cause/pairs.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("reports/corr2cause_t3_summary.json"))
    args = parser.parse_args()
    pairs, summary = prepare(args.clean, args.paraphrase, args.refactorization)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in pairs), encoding="utf-8")
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
