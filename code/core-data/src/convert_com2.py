"""Convert the pinned Com2 release into normalized CORE chain records."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from groups import group_entry, stable_group_digest, write_group_manifest
from schema import validate_record


SOURCE_URL = (
    "https://raw.githubusercontent.com/Waste-Wood/Com2/"
    "not-published/benckmark/com2/main.json"
)
SOURCE_SHA256 = "not-published"
FETCHED_UTC = "2026-09-03T18:10:38Z"
CONVERTER_VERSION = 2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def common_prefix_length(before: List[str], after: List[str]) -> int:
    length = 0
    while length < min(len(before), len(after)) and before[length] == after[length]:
        length += 1
    return length


def event_occurrence(index: int, event: str) -> str:
    """Give repeated event text a unique identity while preserving it verbatim."""

    return f"event_{index:02d}: {event}"


def group_split(root_event: str) -> str:
    """Deterministically assign exact root-event groups to an 80/10/10 split."""

    key = f"com2-split-v1:{root_event}".encode("utf-8")
    bucket = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


def marked_event_passage(scenario: str, original: str, replacement: str) -> tuple[str, list[int], list[int]]:
    """Expose both divergent events as marked, non-asserted pointer candidates."""

    prefix = scenario.rstrip()
    if prefix:
        prefix += "\n"
    original_prefix = f"{prefix}[ORIG] "
    replacement_prefix = f"{original_prefix}{original}\n[REPL] "
    passage = f"{replacement_prefix}{replacement}"
    original_start = len(original_prefix)
    replacement_start = len(replacement_prefix)
    return (
        passage,
        [original_start, original_start + len(original)],
        [replacement_start, replacement_start + len(replacement)],
    )


def normalize_item(item: Dict[str, Any], source_index: int) -> Dict[str, Any]:
    chains = item.get("chains")
    if not isinstance(chains, list) or len(chains) != 2:
        raise ValueError("chains must contain exactly two paths")
    before, after = chains
    if not isinstance(before, list) or not isinstance(after, list):
        raise ValueError("each chain must be a list")
    if not before or not after or not all(isinstance(event, str) for event in before + after):
        raise ValueError("chains must contain non-empty event strings")

    k = common_prefix_length(before, after)
    if k >= min(len(before), len(after)):
        raise ValueError("chains do not contain a replacement event")

    record_type = item["type"]
    record_id = f"com2-{record_type}-{source_index:06d}"
    target = event_occurrence(k, before[k])
    non_descendants = [event_occurrence(index, before[index]) for index in range(k)]
    descendants = [
        event_occurrence(index, before[index]) for index in range(k + 1, len(before))
    ]

    probes = []
    for index in range(k):
        probes.append(
            {
                "variable": event_occurrence(index, before[index]),
                "question": None,
                "answer_before": before[index],
                "answer_after": after[index],
                "required": "must not change",
                "actually_changed": False,
            }
        )
    probes.append(
        {
            "variable": target,
            "question": None,
            "answer_before": before[k],
            "answer_after": after[k],
            "required": "may change",
            "actually_changed": True,
        }
    )
    for index in range(k + 1, min(len(before), len(after))):
        probes.append(
            {
                "variable": event_occurrence(index, before[index]),
                "question": None,
                "answer_before": before[index],
                "answer_after": after[index],
                "required": "may change",
                "actually_changed": before[index] != after[index],
            }
        )

    question = item.get("question")
    answer = item.get("answer")
    scenario = item.get("scenario")
    if not isinstance(scenario, str):
        raise ValueError("scenario must be a string for T3 span construction")
    passage, target_span, replacement_span = marked_event_passage(
        scenario, before[k], after[k]
    )
    record = {
        "id": record_id,
        "source": "com2",
        "source_id": source_index,
        "structure_kind": "chain",
        "graph": None,
        "chain": before,
        "factual": {
            "passage": passage,
            "state": None,
            "question": question if isinstance(question, str) else None,
            "answer": None,
        },
        "intervention": {
            "target": target,
            "target_text": before[k],
            "target_span": target_span,
            "value": after[k],
            "value_token": None,
            "replacement_span": replacement_span,
            "kind": "event_replace",
            "formal": f"do(event_{k:02d} = {json.dumps(after[k], ensure_ascii=False)})",
            "text": f"Replace {before[k]} with {after[k]}.",
        },
        "intervened": {
            "passage": passage,
            "state": None,
            "answer": answer if isinstance(answer, (str, int, float, bool)) else None,
        },
        "descendants": descendants,
        "non_descendants": non_descendants,
        "probes": probes,
        "provenance": {
            "fetched_utc": FETCHED_UTC,
            "url": SOURCE_URL,
            "file": "data/raw/com2/main.json",
            "sha256": SOURCE_SHA256,
            "converter": "src/convert_com2.py",
            "converter_version": CONVERTER_VERSION,
        },
    }
    return record


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False, sort_keys=True))
            handle.write("\n")
    temporary.replace(path)


def write_lines(path: Path, values: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(value + "\n")
    temporary.replace(path)


def convert(
    input_path: Path,
    records_dir: Path,
    splits_dir: Path,
    summary_path: Path,
    groups_dir: Path = Path("data/groups"),
) -> Dict[str, Any]:
    actual_hash = sha256_file(input_path)
    if actual_hash != SOURCE_SHA256:
        raise ValueError(f"raw SHA-256 mismatch: expected {SOURCE_SHA256}, found {actual_hash}")
    with input_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, list) or len(raw) != 2500:
        raise ValueError(f"expected 2500 raw items, found {len(raw) if isinstance(raw, list) else type(raw)}")

    type_counts = collections.Counter(item.get("type") for item in raw)
    expected_types = {
        "counterfactual": 500,
        "decision": 500,
        "direct": 500,
        "intervention": 500,
        "reco": 500,
    }
    if dict(type_counts) != expected_types:
        raise ValueError(f"unexpected Com2 type counts: {dict(type_counts)}")

    selected = {
        "intervention": [
            (index, item)
            for index, item in enumerate(raw)
            if item.get("type") == "intervention" and "chains" in item
        ],
        "counterfactual": [
            (index, item)
            for index, item in enumerate(raw)
            if item.get("type") == "counterfactual" and "chains" in item
        ],
    }
    if len(selected["intervention"]) != 270 or len(selected["counterfactual"]) != 500:
        raise ValueError(
            f"unexpected selected counts: intervention={len(selected['intervention'])}, "
            f"counterfactual={len(selected['counterfactual'])}"
        )

    accepted: Dict[str, List[Dict[str, Any]]] = {key: [] for key in selected}
    rejected: Dict[str, List[Dict[str, Any]]] = {key: [] for key in selected}
    split_ids: Dict[Tuple[str, str], List[str]] = collections.defaultdict(list)
    prefix_counts: Dict[str, collections.Counter] = {}
    length_counts: Dict[str, collections.Counter] = {}
    group_entries = []

    for record_type, items in selected.items():
        prefix_counts[record_type] = collections.Counter()
        length_counts[record_type] = collections.Counter()
        for source_index, item in items:
            before, after = item["chains"]
            prefix_counts[record_type][common_prefix_length(before, after)] += 1
            length_counts[record_type][(len(before), len(after))] += 1
            try:
                record = normalize_item(item, source_index)
            except (KeyError, TypeError, ValueError) as exc:
                rejected[record_type].append({"reason": f"normalization error: {exc}", "record": item})
                continue
            errors = validate_record(record)
            if errors:
                rejected[record_type].append(
                    {"reason": "; ".join(errors), "record": record}
                )
                continue
            accepted[record_type].append(record)
            group_entries.append(
                group_entry(
                    record["id"],
                    stable_group_digest("com2-root", before[0]),
                    stable_group_digest("com2-chain", before),
                )
            )
            split = group_split(before[0])
            split_ids[(record_type, split)].append(record["id"])

    if prefix_counts["intervention"] != {1: 270}:
        raise ValueError(f"unexpected intervention prefix counts: {prefix_counts['intervention']}")
    if length_counts["intervention"] != {(5, 5): 258, (6, 6): 12}:
        raise ValueError(f"unexpected intervention length counts: {length_counts['intervention']}")
    if prefix_counts["counterfactual"] != {0: 448, 1: 52}:
        raise ValueError(f"unexpected counterfactual prefix counts: {prefix_counts['counterfactual']}")

    for record_type in ("intervention", "counterfactual"):
        write_jsonl(records_dir / f"com2_{record_type}.jsonl", accepted[record_type])
        write_jsonl(records_dir / f"com2_{record_type}.rejected.jsonl", rejected[record_type])
        for split in ("train", "validation", "test"):
            write_lines(
                splits_dir / f"com2_{record_type}.{split}.txt",
                sorted(split_ids[(record_type, split)]),
            )
    write_group_manifest(groups_dir / "com2.jsonl", group_entries)

    summary = {
        "source": "com2",
        "raw_sha256": actual_hash,
        "raw_count": len(raw),
        "type_counts": dict(sorted(type_counts.items())),
        "accepted_counts": {key: len(value) for key, value in accepted.items()},
        "rejected_counts": {key: len(value) for key, value in rejected.items()},
        "prefix_counts": {
            key: {str(k): v for k, v in sorted(value.items())}
            for key, value in prefix_counts.items()
        },
        "chain_length_counts": {
            key: {f"{a}/{b}": count for (a, b), count in sorted(value.items())}
            for key, value in length_counts.items()
        },
        "split_counts": {
            record_type: {
                split: len(split_ids[(record_type, split)])
                for split in ("train", "validation", "test")
            }
            for record_type in ("intervention", "counterfactual")
        },
        "split_group": "exact first event in the factual chain",
        "group_sidecar_entries": len(group_entries),
        "split_algorithm": "sha256('com2-split-v1:' + root_event) modulo 100; 0-79 train, 80-89 validation, 90-99 test",
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(summary_path)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/raw/com2/main.json"))
    parser.add_argument("--records-dir", type=Path, default=Path("data/records"))
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--groups-dir", type=Path, default=Path("data/groups"))
    parser.add_argument(
        "--summary", type=Path, default=Path("reports/com2_conversion_summary.json")
    )
    args = parser.parse_args()
    summary = convert(args.input, args.records_dir, args.splits_dir, args.summary, args.groups_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
