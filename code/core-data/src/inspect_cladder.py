"""Inspect the pinned CLadder archive and render its pre-conversion schema report."""

from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import tarfile
import zipfile
from pathlib import Path


SOURCE_SHA256 = "not-published"
PINNED_REVISION = "not-published"
EXPLICIT_SINGLE_ACTION = {"det-counterfactual"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_structure(structure: str) -> tuple[list[str], list[list[str]]]:
    edges = []
    nodes = set()
    for token in structure.split(","):
        parent, child = token.split("->")
        nodes.update((parent, child))
        edges.append([parent, child])
    return sorted(nodes), edges


def load_release(path: Path) -> tuple[list[dict], list[dict], str]:
    actual_hash = sha256_file(path)
    if actual_hash != SOURCE_SHA256:
        raise ValueError(f"raw SHA-256 mismatch: expected {SOURCE_SHA256}, found {actual_hash}")
    with tarfile.open(path, "r:gz") as archive:
        member = next(
            item for item in archive.getmembers() if item.name.endswith("data/cladder-v1.zip")
        )
        extracted = archive.extractfile(member)
        if extracted is None:
            raise ValueError("nested cladder-v1.zip could not be read")
        nested = extracted.read()
    with zipfile.ZipFile(io.BytesIO(nested)) as archive:
        questions = json.loads(archive.read("cladder-v1-q-balanced.json"))
        models = json.loads(archive.read("cladder-v1-meta-models.json"))
    return questions, models, actual_hash


def inspect(path: Path) -> tuple[dict, dict]:
    questions, models, actual_hash = load_release(path)
    if len(questions) != 10112 or len(models) != 7064:
        raise ValueError(f"unexpected counts: questions={len(questions)}, models={len(models)}")
    by_model = {model["model_id"]: model for model in models}
    missing = sorted({item["meta"]["model_id"] for item in questions} - set(by_model))
    if missing:
        raise ValueError(f"question model IDs missing from metadata: {missing[:10]}")
    for model in models:
        parse_structure(model["structure"])

    query_counts = collections.Counter(item["meta"]["query_type"] for item in questions)
    equation_counts = collections.Counter(model["equation_type"] for model in models)
    explicit = sum(query_counts[name] for name in EXPLICIT_SINGLE_ACTION)
    summary = {
        "source": "cladder",
        "raw_sha256": actual_hash,
        "balanced_questions": len(questions),
        "meta_models": len(models),
        "question_model_links_missing": len(missing),
        "query_type_counts": dict(sorted(query_counts.items())),
        "model_equation_type_counts": dict(sorted(equation_counts.items())),
        "explicit_deterministic_single_action_questions": explicit,
        "not_directly_representable_by_single_intervention_contract": len(questions) - explicit,
        "graph_lookup": "question.meta.model_id -> meta-model.model_id -> structure",
        "split_metadata_present": False,
    }
    return summary, questions[0]


def render_card(summary: dict, first_item: dict) -> str:
    counts = summary["query_type_counts"]
    rows = "\n".join(f"| `{name}` | {count:,} |" for name, count in counts.items())
    return f"""# CLadder data card

Status: schema discovery complete; conversion is gated on the record-contract decision below.

## Source and integrity

- Pinned revision: `{PINNED_REVISION}`
- Snapshot URL: `https://github.com/causalNLP/cladder/archive/{PINNED_REVISION}.tar.gz`
- Local raw file: `data/raw/cladder/cladder-3d2d1169.tar.gz`
- Bytes: 9,970,314
- SHA-256: `{SOURCE_SHA256}`
- Licence: MIT License, included as `LICENSE` in the pinned snapshot
- Balanced questions: {summary['balanced_questions']:,}
- Meta-models: {summary['meta_models']:,}

The snapshot contains `data/cladder-v1.zip`. The selected `cladder-v1-q-balanced.json` rows link to `cladder-v1-meta-models.json` by `question.meta.model_id == model.model_id`; all links resolve. The DAG is not stored directly on each question. It is parsed from the linked model's `structure` string, such as `X->V2,X->Y,V2->Y`. Human-readable node labels come from that model's `variable_mapping`.

## Query inventory

| Query type | Rows |
|---|---:|
{rows}

The 1,422 `det-counterfactual` rows use deterministic models and expose one action value in `meta.action`. The other 8,690 rows do not all describe one world-state intervention: they include observational marginals/correlations, adjustment-set selection, population treatment contrasts, mediation estimands, and collider/explaining-away questions.

## Conversion gate

The exact CORE schema requires one `intervention.target`, one `intervention.value`, before/after probe values, and an observed changed probe. Mapping all 10,112 questions would require inventing a single action for observational questions or collapsing two/nested interventions into one. That violates the no-invention rule.

The defensible next implementation is to convert the 1,422 explicit deterministic counterfactuals and quarantine the remaining 8,690 with query-type-specific reasons. This produces fewer accepted CLadder rows than the approximate 10,112 expectation, so it is recorded as a surfaced source/schema finding, not a silent fix. No source-provided train/validation/test metadata was found in the pinned release; a later converter must document a deterministic model-group split.

## First balanced raw item, verbatim fields

```json
{json.dumps(first_item, ensure_ascii=False, indent=2)}
```

Required after the gate is resolved: deterministic state reconstruction and oracle tests, field mapping, accepted/rejected outputs, five normalized spot checks, model-group splits, and full verification.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/cladder/cladder-3d2d1169.tar.gz"),
    )
    parser.add_argument("--summary", type=Path, default=Path("reports/cladder_schema_summary.json"))
    parser.add_argument("--card", type=Path)
    args = parser.parse_args()
    summary, first_item = inspect(args.input)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.card is not None:
        args.card.write_text(render_card(summary, first_item), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
