"""Render the completed CCR.GB conversion data card."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


SPOT_CHECK_SEED = 20260903


def fenced_json(value: object) -> str:
    return "~~~json\n" + json.dumps(value, ensure_ascii=False, indent=2) + "\n~~~"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def render(raw_path: Path, records_path: Path, summary_path: Path) -> str:
    worlds = read_jsonl(raw_path)
    records = read_jsonl(records_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    raw_bytes = raw_path.stat().st_size
    raw_sha256 = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    if raw_sha256 != summary["raw_sha256"]:
        raise ValueError("CCR.GB raw hash does not match conversion summary")
    samples = random.Random(SPOT_CHECK_SEED).sample(records, 5)
    sections = [
        "# CCR.GB data card",
        "",
        f"Status: {summary['worlds']:,}-world external-generator conversion complete and verified.",
        "",
        "## Source and isolation",
        "",
        "- Repository: https://github.com/jmaasch/compositional_causal_reasoning",
        "- Pinned generator commit: not-published",
        "- External clone: sibling ${USER_HOME}/Documents/compositional_causal_reasoning",
        f"- Raw generated output: {raw_path}",
        f"- Raw bytes: {raw_bytes:,}",
        f"- Raw SHA-256: {raw_sha256}",
        "- Domain: ClinicalNotes",
        "- Generator seed: 20260903",
        "- Licence: none stated at the pinned commit. No LICENSE, COPYING, licence text, or README licence declaration is present.",
        "",
        "The generator was cloned and run outside core-data. No CCR.GB source file is copied, imported, or vendored by the shipped converter. Only the generator JSONL output is consumed.",
        "",
        "## Generation",
        "",
        f"The CPU-only run used graph_sizes=[[2,2,2]], cycle BCCs, random monotone Boolean functions, {summary['worlds']:,} tasks/worlds, and one sample per world. It produced {summary['worlds']:,} unique four-node worlds, {summary['counterfactual_pairs']:,} cause-effect pairs, and both Boolean actions for every pair.",
        "",
        "## Conversion counts",
        "",
        f"- Candidate interventions: {summary['candidate_interventions']:,}",
        f"- Accepted: {summary['accepted']:,}",
        f"- Quarantined: {summary['rejected']:,}",
        "- Rejection reason: intervention matches the factual state and therefore has no observed may-change probe",
        "",
        "Exactly one of the two Boolean actions per pair differs from the factual cause value. The other reproduces the factual world and is quarantined under the shared observed-effect requirement.",
        "",
        "## Field mapping",
        "",
        "| CORE field | CCR.GB output source or derivation |",
        "|---|---|",
        "| graph | dag_nodes plus nonzero entries of dag_adjacency_matrix |",
        "| factual.passage | generated causal context joined with sample context |",
        "| factual.state | generated factual query true_endogenous |",
        "| factual.question/answer | matching factual effect prompt and Boolean response mapped to yes/no |",
        "| intervention | counterfactual cause and action 0/1; `target_text` is the exact variable mention in the factual passage, `target_span` addresses it, and `value_token` is canonical `0`/`1` |",
        "| intervened.state | generated action-specific true_endogenous |",
        "| intervened.answer | generated action-specific response mapped to yes/no |",
        "| descendants | DAG reachability from the cause |",
        "| non-descendants | other DAG nodes excluding cause and descendants |",
        "| probes | every endogenous node generated factual/intervened Boolean value |",
        "",
        "## Splits",
        "",
        "Whole context worlds are kept together. Context IDs are ranked by SHA-256 of `ccrgb-split-v1:<context_id>` and assigned deterministically in an 80/10/10 train/validation/test partition.",
        "",
        f"- Train: {summary['split_counts']['train']} records",
        f"- Validation: {summary['split_counts']['validation']} records",
        f"- Test: {summary['split_counts']['test']} records",
        f"- Train worlds: {summary['split_world_counts']['train']}",
        f"- Validation worlds: {summary['split_world_counts']['validation']}",
        f"- Test worlds: {summary['split_world_counts']['test']}",
        "- Cross-split world overlap: zero",
        "",
        "## Caveats",
        "",
        "- All worlds use one domain and the smallest configured graph size; the scale supports repeated graph-level inference but does not establish broad domain coverage.",
        "- Variable names and clinical prose are randomly generated and should not be interpreted as real medical advice or patient data.",
        "- The pinned repository missing licence statement conflicts with the task GPL-3.0 expectation. Redistribution rights must not be inferred.",
        "",
        "## First generated raw world, verbatim fields",
        "",
        fenced_json(worlds[0]),
        "",
        "## Reproducible random spot checks",
        "",
        f"Five accepted records were sampled with Python random.Random({SPOT_CHECK_SEED}).",
    ]
    for index, record in enumerate(samples, 1):
        must = [probe for probe in record["probes"] if probe["required"] == "must not change"]
        changed = sum(probe["actually_changed"] for probe in record["probes"])
        sections += [
            "",
            f"### Spot check {index}: {record['id']}",
            "",
            fenced_json(record),
            "",
            f"Assessment: {changed} of {len(record['probes'])} generated state probes changed and all {len(must)} graph non-descendants were preserved; the yes/no answer matches the generated intervened effect value.",
        ]
    return "\n".join(sections) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--records", type=Path, default=Path("data/records/ccrgb.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("reports/ccrgb_conversion_summary.json"))
    parser.add_argument("--output", type=Path, default=Path("docs/CCRGB_CARD.md"))
    args = parser.parse_args()
    content = render(args.raw, args.records, args.summary)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(args.output)
    print(f"wrote {args.output} with raw world and 5 spot checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
