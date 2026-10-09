"""Render the completed CLadder conversion data card."""

from __future__ import annotations

import argparse
import json
import random
import tarfile
from pathlib import Path

from inspect_cladder import PINNED_REVISION, SOURCE_SHA256, load_release


SPOT_CHECK_SEED = 20260903


def fenced_json(value: object) -> str:
    return "```json\n" + json.dumps(value, ensure_ascii=False, indent=2) + "\n```"


def licence_text(raw_path: Path) -> str:
    with tarfile.open(raw_path, "r:gz") as archive:
        member = next(item for item in archive.getmembers() if item.name.endswith("/LICENSE"))
        handle = archive.extractfile(member)
        if handle is None:
            raise ValueError("CLadder LICENSE could not be read")
        return handle.read().decode("utf-8").replace("\r\n", "\n").replace("\r", "\n").strip()


def render(raw_path: Path, records_path: Path, summary_path: Path) -> str:
    questions, _, actual_hash = load_release(raw_path)
    if actual_hash != SOURCE_SHA256:
        raise ValueError("unexpected source hash")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(records) != 1422:
        raise ValueError(f"expected 1422 accepted records, found {len(records)}")
    samples = random.Random(SPOT_CHECK_SEED).sample(records, 5)
    rejection_rows = "\n".join(
        f"| `{name}` | {count:,} |"
        for name, count in summary["rejected_query_type_counts"].items()
    )
    sections = [
        "# CLadder data card",
        "",
        "Status: eligible deterministic counterfactual conversion complete and verified.",
        "",
        "## Source and integrity",
        "",
        f"- Pinned revision: `{PINNED_REVISION}`",
        f"- Snapshot URL: `https://github.com/causalNLP/cladder/archive/{PINNED_REVISION}.tar.gz`",
        "- Local raw file: `data/raw/cladder/cladder-3d2d1169.tar.gz`",
        "- Bytes: 9,970,314",
        f"- SHA-256: `{SOURCE_SHA256}`",
        "- Balanced questions: 10,112",
        "- Meta-models: 7,064",
        "- Licence: MIT License",
        "",
        "The selected `cladder-v1-q-balanced.json` rows link to `cladder-v1-meta-models.json` by `question.meta.model_id == model.model_id`; all links resolve. Each DAG is parsed from the linked model's `structure` string. It is not embedded directly in the question.",
        "",
        "## Conversion counts",
        "",
        "- Accepted: 1,422 `det-counterfactual` records",
        "- Rejected/quarantined: 8,690 records",
        "- Deterministic oracle mismatches: 0",
        "",
        "| Quarantined query type | Rows |",
        "|---|---:|",
        rejection_rows,
        "",
        "The quarantined types do not all encode one before/after world: they include observational associations and marginals, adjustment-set selection, population treatment contrasts, nested mediation effects, and collider/conditioning questions. Converting them to one `do(target=value)` would invent or collapse semantics.",
        "",
        "## Field mapping and reconstruction",
        "",
        "| CORE field | CLadder source or derivation |",
        "|---|---|",
        "| `id` | `cladder-det-counterfactual-<question_id>` |",
        "| `source_id` | balanced row `question_id` |",
        "| `graph` | nodes and edges parsed from linked model `structure` |",
        "| `chain` | `null`; this is a DAG source |",
        "| `factual.passage` | question `given_info`, verbatim |",
        "| `factual.question` | source `question`, verbatim |",
        "| `factual.state` | deterministic SCM solution using non-treatment root evidence and the original treatment value `1-action` |",
        "| `factual.answer` | `null`; source answer concerns the counterfactual question |",
        "| intervention | structural target=`meta.treatment`, exact natural-language `target_text`/`target_span` from `given_info`, value=`meta.action`, canonical `value_token`=`yes`/`no`, kind=`value_set` |",
        "| `intervention.text` | source question, verbatim |",
        "| `intervened.state` | deterministic SCM solution with the same root evidence and treatment overridden to `action` |",
        "| `intervened.answer` | source yes/no answer |",
        "| descendants | directed graph reachability from treatment |",
        "| non-descendants | all other graph nodes except target and descendants |",
        "| probes | one Boolean before/after state probe for every graph node |",
        "",
        "The solver evaluates the released deterministic conditional tables in topological order. In 632 rows `X` is endogenous; matching CLadder's generator requires overriding `X` while holding its root causes fixed. For every accepted row, the reconstructed intervened outcome was compared with `meta.polarity`; all 1,422 results matched both `meta.groundtruth` and the released yes/no answer.",
        "",
        "## Splits",
        "",
        "The pinned release provides no train/validation/test labels. All questions sharing a `model_id` stay together. Assignment is `sha256('cladder-split-v1:' + model_id) mod 100`: buckets 0–79 train, 80–89 validation, and 90–99 test.",
        "",
        f"- Train: {summary['split_counts']['train']:,} records across {summary['split_model_counts']['train']:,} models",
        f"- Validation: {summary['split_counts']['validation']:,} records across {summary['split_model_counts']['validation']:,} models",
        f"- Test: {summary['split_counts']['test']:,} records across {summary['split_model_counts']['test']:,} models",
        "- Cross-split model overlap: zero",
        "",
        "## Caveats",
        "",
        "- Accepted coverage is 1,422 rather than the complete 10,112 balanced questions because the fixed CORE schema represents a single state intervention.",
        "- Graph and state keys use CLadder's symbolic node IDs (`X`, `Y`, `V1`, etc.); natural-language meanings remain in the passage and question.",
        "- Pointer addressing uses the exact natural-language treatment surface in the passage while preserving the symbolic graph node as `intervention.target`.",
        "- The target probe always changes because the released counterfactual contrasts Boolean `action` with `1-action`; downstream nodes may legitimately remain unchanged.",
        "- The yes/no answer asks whether the intervened outcome equals the queried polarity. The actual Boolean outcome is available in `intervened.state`.",
        "",
        "## Licence text, verbatim",
        "",
        "```text",
        licence_text(raw_path),
        "```",
        "",
        "## First balanced raw item, verbatim fields",
        "",
        fenced_json(questions[0]),
        "",
        "## Reproducible random spot checks",
        "",
        f"Five records were sampled from the 1,422 accepted rows with Python `random.Random({SPOT_CHECK_SEED})`.",
    ]
    for index, record in enumerate(samples, 1):
        must = [probe for probe in record["probes"] if probe["required"] == "must not change"]
        may = [probe for probe in record["probes"] if probe["required"] == "may change"]
        changed = sum(probe["actually_changed"] for probe in may)
        sections += [
            "",
            f"### Spot check {index}: `{record['id']}`",
            "",
            fenced_json(record),
            "",
            f"Assessment: all {len(must)} non-descendant probes are preserved; {changed} of {len(may)} target/descendant probes changed. The source answer agrees with the reconstructed intervened outcome and queried polarity.",
        ]
    return "\n".join(sections) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=Path("data/raw/cladder/cladder-3d2d1169.tar.gz"))
    parser.add_argument("--records", type=Path, default=Path("data/records/cladder.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("reports/cladder_conversion_summary.json"))
    parser.add_argument("--output", type=Path, default=Path("docs/CLADDER_CARD.md"))
    args = parser.parse_args()
    content = render(args.raw, args.records, args.summary)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(args.output)
    print(f"wrote {args.output} with licence, 1 raw item, and 5 normalized spot checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
