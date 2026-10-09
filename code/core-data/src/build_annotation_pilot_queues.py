"""Build frozen, blind 300-item annotation queues and separate worked examples.

This is a CPU-only sampling utility. It never reads the PubMedCausal test file and
does not emit normalized CORE records; annotations must pass the later contract
and adjudication workflow before conversion.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from convert_causalt5k import SHA256 as CAUSALT5K_SHA256, read_rows
from convert_meter import SHA256 as METER_SHA256, read_contexts
from convert_pubmedcausal import FILES as PUBMED_FILES, long_candidates
from lower_priority_common import require_hash


SELECTION_SEED = 317
QUEUE_SIZE = 300
EXAMPLE_SIZE = 5
CAUSALT5K_PATH = Path("data/raw/causalt5k/causalt5k-fd358e95.tar.gz")
METER_PATH = Path("data/raw/meter/meter-0d2a53b8.tar.gz")
PUBMED_DIR = Path("data/raw/pubmedcausal")
PUBMED_ALLOWED_FILES = {"30k_train.json": "train", "validation_set.json": "validation"}
PUBMED_STRATA = {
    ("Implicit", "Intra"): 120,
    ("Explicit", "Intra"): 120,
    ("Implicit", "Inter"): 30,
    ("Explicit", "Inter"): 30,
}


def stable_rank(namespace: str, value: str) -> str:
    return hashlib.sha256(f"{SELECTION_SEED}|{namespace}|{value}".encode()).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def blank_annotation() -> dict[str, Any]:
    return {
        "annotator_id": None,
        "decision": None,
        "reason_codes": [],
        "graph": None,
        "factual_state": None,
        "intervention": None,
        "intervened_state": None,
        "evidence_spans": [],
        "notes": None,
    }


def select_ranked(rows: list[dict[str, Any]], count: int, namespace: str, key) -> list[dict[str, Any]]:
    if len(rows) < count:
        raise ValueError(f"{namespace}: requested {count} from {len(rows)} candidates")
    return sorted(rows, key=lambda row: stable_rank(namespace, str(key(row))))[:count]


def unique_rows(rows: Iterable[dict[str, Any]], key) -> list[dict[str, Any]]:
    unique = {}
    for row in rows:
        unique.setdefault(str(key(row)), row)
    return list(unique.values())


def causalt5k_payload(row: dict[str, Any]) -> dict[str, Any]:
    case_id = row.get("case_id", row["id"])
    return {
        "source_id": row["id"],
        "case_id": case_id,
        "release_domain": row["release_domain"],
        "release_level": row["release_level"],
        "domain": row.get("domain"),
        "subdomain": row.get("subdomain"),
        "scenario": row.get("scenario"),
        "claim": row.get("claim"),
        "variables": row.get("variables"),
        "causal_structure": row.get("causal_structure"),
        "conditional_answers": row.get("conditional_answers"),
    }


def build_causalt5k(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    chosen: list[dict[str, Any]] = []
    for domain in [f"D{i}" for i in range(1, 11)]:
        for level in ("L1", "L2", "L3"):
            stratum = unique_rows(
                [r for r in rows if r["release_domain"] == domain and r["release_level"] == level],
                lambda r: r["id"],
            )
            chosen.extend(select_ranked(stratum, 10, f"causalt5k:{domain}:{level}:queue", lambda r: r["id"]))
    chosen = sorted(chosen, key=lambda r: stable_rank("causalt5k:queue-order", r["id"]))
    used = {(r["release_domain"], r["release_level"], r["id"]) for r in chosen}
    remaining = unique_rows(
        [r for r in rows if (r["release_domain"], r["release_level"], r["id"]) not in used],
        lambda r: (r["release_domain"], r["release_level"], r["id"]),
    )
    example_specs = [("D1", "L1"), ("D3", "L2"), ("D5", "L3"), ("D7", "L2"), ("D9", "L3")]
    examples = [
        select_ranked(
            [r for r in remaining if (r["release_domain"], r["release_level"]) == spec],
            1,
            f"causalt5k:{spec}:example",
            lambda r: r["id"],
        )[0]
        for spec in example_specs
    ]
    queue = []
    for index, row in enumerate(chosen, 1):
        case_id = row.get("case_id", row["id"])
        queue.append({
            "pilot_id": f"C5K-{index:03d}",
            "source": "causalt5k",
            "source_locator": f"{row['release_domain']}/{row['release_level']}/{row['id']}",
            "graph_group_id": f"causalt5k:case:{case_id}",
            "task_views": {"T2": blank_annotation(), "T4": blank_annotation()},
            "source_payload": causalt5k_payload(row),
        })
    return queue, examples


def meter_rung(question: dict[str, Any]) -> str:
    return {
        "Causal_Discovery": "discovery",
        "Intervention": "intervention",
        "Counterfactual": "counterfactual",
    }[question["ladder"]]


def meter_payload(context_index: int, context: dict[str, Any], question: dict[str, Any]) -> dict[str, Any]:
    return {
        "context_index": context_index,
        "context": context["context"],
        "rung": meter_rung(question),
        "question": question["question"],
        "options": [question.get(f"option_{i}") for i in range(5)],
    }


def eligible_meter_contexts(contexts: list[dict[str, Any]]) -> list[tuple[int, dict[str, Any]]]:
    required = {"discovery", "intervention", "counterfactual"}
    eligible = []
    for index, context in enumerate(contexts):
        rungs = [meter_rung(q) for q in context["questions"]]
        if len(context["questions"]) == 3 and set(rungs) == required and len(set(rungs)) == 3:
            eligible.append((index, context))
    return eligible


def build_meter(contexts: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[tuple[int, dict[str, Any]]]]:
    eligible = eligible_meter_contexts(contexts)
    selected = sorted(eligible, key=lambda item: stable_rank("meter:queue", str(item[0])))[:100]
    used = {index for index, _ in selected}
    remaining = [(i, c) for i, c in eligible if i not in used]
    examples = sorted(remaining, key=lambda item: stable_rank("meter:example", str(item[0])))[:EXAMPLE_SIZE]
    queue = []
    counter = 0
    for context_index, context in selected:
        for question in sorted(context["questions"], key=lambda q: ("discovery", "intervention", "counterfactual").index(meter_rung(q))):
            counter += 1
            queue.append({
                "pilot_id": f"MTR-{counter:03d}",
                "source": "meter",
                "source_locator": f"dataset.jsonl/context/{context_index}/{meter_rung(question)}",
                "graph_group_id": f"meter:context:{context_index}",
                "task_views": {"T4": blank_annotation()},
                "source_payload": meter_payload(context_index, context, question),
            })
    return queue, examples


def normalize_pubmed_label(value: Any) -> Any:
    if isinstance(value, str) and value.lower() in {"implicit", "explicit", "intra", "inter"}:
        return value.capitalize()
    return value


def read_pubmed_development(raw_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for name, split in PUBMED_ALLOWED_FILES.items():
        require_hash(raw_dir / name, PUBMED_FILES[name])
        raw_rows = json.loads((raw_dir / name).read_text(encoding="utf-8"))
        for raw in raw_rows:
            for row in long_candidates(raw, split):
                row["causality"] = normalize_pubmed_label(row.get("causality"))
                row["sententiality"] = normalize_pubmed_label(row.get("sententiality"))
                rows.append(row)
    return rows


def pubmed_key(row: dict[str, Any]) -> str:
    return f"{row['release_split']}:{row['source_id']}:{row['relation_index']}"


def build_pubmed(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    relation_rows = [r for r in rows if r["relation_index"] is not None]
    chosen = []
    for stratum, count in PUBMED_STRATA.items():
        pool = [r for r in relation_rows if (r["causality"], r["sententiality"]) == stratum]
        chosen.extend(select_ranked(pool, count, f"pubmed:{stratum}:queue", pubmed_key))
    chosen = sorted(chosen, key=lambda r: stable_rank("pubmed:queue-order", pubmed_key(r)))
    used = {pubmed_key(r) for r in chosen}
    example_pools = [
        [r for r in relation_rows if (r["causality"], r["sententiality"]) == stratum and pubmed_key(r) not in used]
        for stratum in PUBMED_STRATA
    ]
    noncausal = [r for r in rows if r["relation_index"] is None]
    examples = [select_ranked(pool, 1, f"pubmed:{i}:example", pubmed_key)[0] for i, pool in enumerate(example_pools)]
    examples.append(select_ranked(noncausal, 1, "pubmed:noncausal:example", pubmed_key)[0])
    queue = []
    for index, row in enumerate(chosen, 1):
        queue.append({
            "pilot_id": f"PMC-{index:03d}",
            "source": "pubmedcausal",
            "source_locator": f"{row['release_split']}/{row['source_id']}/relation/{row['relation_index']}",
            "graph_group_id": f"pubmedcausal:{row['release_split']}:sentence:{row['source_id']}",
            "task_views": {"G4": blank_annotation()},
            "source_payload": row,
        })
    return queue, examples


def senior_examples(c5_rows, meter_contexts, pubmed_rows) -> list[dict[str, Any]]:
    output = []
    for i, row in enumerate(c5_rows, 1):
        level = row["release_level"]
        t2_reasons = ["NO_INTERVENTION"] if level == "L1" else ["COUNTERFACTUAL_STATE_NOT_IDENTIFIED"]
        t4_reasons = ["NO_INTERVENTION"] if level == "L1" else ["FACTUAL_STATE_NOT_IDENTIFIED", "COUNTERFACTUAL_STATE_NOT_IDENTIFIED"]
        output.append({
            "example_id": f"C5K-EX-{i:03d}", "source": "causalt5k", "task_views": ["T2", "T4"],
            "source_locator": f"{row['release_domain']}/{level}/{row['id']}",
            "source_payload": causalt5k_payload(row),
            "senior_annotation_by_task": {
                "T2": {
                    "decision": "reject", "reason_codes": t2_reasons, "graph": None,
                    "factual_state": None, "intervention": None, "intervened_state": None,
                    "evidence_spans": [{"field": "causal_structure", "text": row.get("causal_structure")} ] if row.get("causal_structure") else [],
                    "explanation": "The held-out-domain task needs a complete source-grounded intervention pair. This released item does not supply both complete worlds; a claim, rationale, or answer cannot be promoted into missing state values.",
                },
                "T4": {
                    "decision": "reject", "reason_codes": t4_reasons, "graph": None,
                    "factual_state": None, "intervention": None, "intervened_state": None,
                    "evidence_spans": [{"field": "release_level", "text": level}],
                    "explanation": "Preserve the released level for rung analysis, but the rung label alone is not a paired causal world. Keep the item in the ledger/sidecar and out of formal T4 evidence.",
                },
            },
        })
    for i, (context_index, context) in enumerate(meter_contexts, 1):
        questions = sorted(context["questions"], key=lambda q: q["ladder"])
        output.append({
            "example_id": f"MTR-EX-{i:03d}", "source": "meter", "task_views": ["T4"],
            "source_locator": f"dataset.jsonl/context/{context_index}",
            "source_payload": {
                "context_index": context_index, "context": context["context"],
                "questions": [meter_payload(context_index, context, q) for q in questions],
            },
            "senior_annotation": {
                "decision": "reject", "reason_codes": ["GRAPH_NOT_IDENTIFIED", "FACTUAL_STATE_NOT_IDENTIFIED", "COUNTERFACTUAL_STATE_NOT_IDENTIFIED"],
                "graph": None, "factual_state": None, "intervention": None, "intervened_state": None,
                "evidence_spans": [{"field": "questions", "rungs": [meter_rung(q) for q in questions]}],
                "explanation": "The three questions form a useful ladder-aligned group, but answer options are not an explicit graph or complete paired worlds. Preserve the context group and reject it from formal CORE evidence unless an authoritative attached structure is located.",
            },
        })
    for i, row in enumerate(pubmed_rows, 1):
        has_relation = row["relation_index"] is not None
        sentence = row.get("sentence") or ""
        span_evidence = []
        for role in ("cause", "effect"):
            text = row.get(role)
            start = sentence.find(text) if text else -1
            if start >= 0:
                span_evidence.append({"role": role, "text": text, "span": [start, start + len(text)]})
        output.append({
            "example_id": f"PMC-EX-{i:03d}", "source": "pubmedcausal", "task_views": ["G4"],
            "source_locator": f"{row['release_split']}/{row['source_id']}/relation/{row['relation_index']}",
            "source_payload": row,
            "senior_annotation": {
                "decision": "reject", "reason_codes": ["NO_INTERVENTION"] if has_relation else ["GRAPH_NOT_IDENTIFIED"],
                "graph": None, "factual_state": None, "intervention": None, "intervened_state": None,
                "evidence_spans": span_evidence,
                "explanation": "A released cause/effect span identifies causal language, not a surgical intervention or paired experimental worlds. The annotator must locate an authoritative linked experiment; without it, this item remains a rejection.",
            },
        })
    return output


def queue_summary(queue: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(queue),
        "id_first": queue[0]["pilot_id"],
        "id_last": queue[-1]["pilot_id"],
        "unique_ids": len({r["pilot_id"] for r in queue}),
        "unique_locators": len({r["source_locator"] for r in queue}),
    }


def validate_outputs(queues: dict[str, list[dict[str, Any]]], examples: list[dict[str, Any]]) -> None:
    prefixes = {"causalt5k": "C5K", "meter": "MTR", "pubmedcausal": "PMC"}
    for source, queue in queues.items():
        expected_ids = [f"{prefixes[source]}-{i:03d}" for i in range(1, QUEUE_SIZE + 1)]
        if len(queue) != QUEUE_SIZE or [row["pilot_id"] for row in queue] != expected_ids:
            raise ValueError(f"{source}: queue is not the exact frozen 300-ID sequence")
        if len({row["source_locator"] for row in queue}) != QUEUE_SIZE:
            raise ValueError(f"{source}: duplicate source locator in production queue")
    c5_counts = collections.Counter(
        (row["source_payload"]["release_domain"], row["source_payload"]["release_level"])
        for row in queues["causalt5k"]
    )
    if set(c5_counts.values()) != {10} or len(c5_counts) != 30:
        raise ValueError("causalt5k: expected 10 items in each of 30 domain-level strata")
    meter_groups: dict[str, set[str]] = collections.defaultdict(set)
    for row in queues["meter"]:
        meter_groups[row["graph_group_id"]].add(row["source_payload"]["rung"])
    required_rungs = {"discovery", "intervention", "counterfactual"}
    if len(meter_groups) != 100 or any(rungs != required_rungs for rungs in meter_groups.values()):
        raise ValueError("meter: expected 100 intact three-rung context groups")
    pubmed_counts = collections.Counter(
        (row["source_payload"]["causality"], row["source_payload"]["sententiality"])
        for row in queues["pubmedcausal"]
    )
    if pubmed_counts != collections.Counter(PUBMED_STRATA):
        raise ValueError("pubmedcausal: frozen causal-language strata do not match")
    if any(row["source_payload"]["release_split"] not in {"train", "validation"} for row in queues["pubmedcausal"]):
        raise ValueError("pubmedcausal: forbidden split in annotation queue")
    example_counts = collections.Counter(row["source"] for row in examples)
    if example_counts != collections.Counter({source: EXAMPLE_SIZE for source in queues}):
        raise ValueError("worked examples: expected five examples per dataset")
    for source, queue in queues.items():
        queue_locators = {row["source_locator"] for row in queue}
        example_locators = {row["source_locator"] for row in examples if row["source"] == source}
        if queue_locators & example_locators:
            raise ValueError(f"{source}: worked example leaks into production queue")


def build(output_dir: Path, manifest_path: Path) -> dict[str, Any]:
    require_hash(CAUSALT5K_PATH, CAUSALT5K_SHA256)
    require_hash(METER_PATH, METER_SHA256)
    c5_queue, c5_examples = build_causalt5k(read_rows(CAUSALT5K_PATH))
    meter_queue, meter_examples = build_meter(read_contexts(METER_PATH))
    pubmed_queue, pubmed_examples = build_pubmed(read_pubmed_development(PUBMED_DIR))
    examples = senior_examples(c5_examples, meter_examples, pubmed_examples)
    queues = {"causalt5k": c5_queue, "meter": meter_queue, "pubmedcausal": pubmed_queue}
    validate_outputs(queues, examples)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for source, queue in queues.items():
        path = output_dir / f"{source}.queue.jsonl"
        write_jsonl(path, queue)
        paths[source] = path
    example_path = output_dir / "worked_examples.jsonl"
    write_jsonl(example_path, examples)
    manifest = {
        "protocol": "T2-T4-G4 annotation pilot",
        "selection_seed": SELECTION_SEED,
        "production_items_per_dataset": QUEUE_SIZE,
        "worked_examples_per_dataset": EXAMPLE_SIZE,
        "production_items_total": 900,
        "worked_examples_total": 15,
        "examples_excluded_from_production": True,
        "validation_passed": True,
        "pubmed_test_accessed": False,
        "source_hashes": {
            "causalt5k": CAUSALT5K_SHA256,
            "meter": METER_SHA256,
            "pubmedcausal_train": PUBMED_FILES["30k_train.json"],
            "pubmedcausal_validation": PUBMED_FILES["validation_set.json"],
        },
        "queues": {
            source: {**queue_summary(queues[source]), "path": str(path), "sha256": sha256_path(path)}
            for source, path in paths.items()
        },
        "worked_examples": {
            "path": str(example_path), "count": len(examples), "sha256": sha256_path(example_path),
            "source_counts": dict(collections.Counter(e["source"] for e in examples)),
        },
        "stratification": {
            "causalt5k": "10 records per D1-D10 x L1-L3 stratum",
            "meter": "100 intact contexts x discovery/intervention/counterfactual",
            "pubmedcausal": {f"{a}_{b}": n for (a, b), n in PUBMED_STRATA.items()},
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/annotation_pilots"))
    parser.add_argument("--manifest", type=Path, default=Path("reports/annotation_pilot_300_manifest.json"))
    args = parser.parse_args()
    print(json.dumps(build(args.output_dir, args.manifest), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
