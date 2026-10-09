"""Render the senior-annotated few-shot Bible from frozen example JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def clipped(value, limit: int = 700) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def fenced(value) -> str:
    return "```json\n" + json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n```"


def render(examples_path: Path, manifest_path: Path, output_path: Path) -> None:
    examples = [json.loads(line) for line in examples_path.read_text(encoding="utf-8").splitlines() if line]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_source = {source: [row for row in examples if row["source"] == source] for source in ("causalt5k", "meter", "pubmedcausal")}
    parts = [
        "# T2/T4/G4 Annotation Bible: 300-per-dataset Pilot",
        "",
        "Version: 1.0",
        "",
        "This Bible governs three separate blind production queues: **300 CausalT5K items, 300 METER items, and 300 PubMedCausal items**. Five additional worked examples from each dataset are excluded from production. The total frozen selection is therefore 900 production items plus 15 teaching items.",
        "",
        "The examples below are completed senior-annotator decisions against the current CORE evidence contract. A rejection is a valid annotation result. It means the released row does not itself justify a formal graph/intervention pair; annotators must not fill missing worlds from common sense.",
        "",
        "## Frozen sampling contract",
        "",
        f"- Selection seed: `{manifest['selection_seed']}` (ordinary three-digit seed).",
        f"- CausalT5K: {manifest['stratification']['causalt5k']}.",
        f"- METER: {manifest['stratification']['meter']}.",
        "- PubMedCausal: 120 implicit/intra, 120 explicit/intra, 30 implicit/inter, and 30 explicit/inter relation candidates.",
        "- PubMedCausal inputs are train and validation only. The held-out test file is neither read nor sampled.",
        "- Every teaching example is disjoint from its 300-item production queue.",
        "- CausalT5K rows contain independent T2 and T4 annotation forms, so one source review can support both tasks without duplicate labor.",
        "",
        "## Senior decision rule",
        "",
        "Accept only when the authoritative source identifies the graph (or deterministic SCM), the complete factual state, a surgical intervention, and the complete intervened state. If any element is missing, mark `reject` or `needs_adjudication`, cite the available evidence, and leave unsupported structured fields null. Multiple-choice options, causal wording, and published cause/effect spans are not substitutes for paired worlds.",
        "",
        "## CausalT5K: five completed examples",
        "",
        "For every example, annotate T2 and T4 separately. T2 asks whether the item can support held-out-domain evaluation. T4 asks whether its released level can support rung-transfer evidence.",
    ]
    for i, row in enumerate(by_source["causalt5k"], 1):
        payload = row["source_payload"]
        parts += [
            "", f"### CausalT5K example {i}: {row['source_locator']}", "",
            f"Scenario: {clipped(payload.get('scenario'))}", "",
            f"Claim: {clipped(payload.get('claim'))}", "",
            f"Released causal structure: {clipped(payload.get('causal_structure'))}", "",
            "Completed annotation:", "", fenced(row["senior_annotation_by_task"]), "",
            "Senior walkthrough: Preserve the domain, level, case grouping, and quoted structural hint. Do not infer unspecified values. The item stays in the rejection ledger unless a frozen authoritative attachment supplies the missing graph/state evidence.",
        ]
    parts += ["", "## METER: five completed examples", "", "Each worked example is one intact context with its discovery, intervention, and counterfactual questions. Keeping the triplet together prevents rung leakage."]
    for i, row in enumerate(by_source["meter"], 1):
        payload = row["source_payload"]
        questions = "\n".join(f"- {q['rung']}: {clipped(q['question'], 350)}" for q in payload["questions"])
        parts += [
            "", f"### METER example {i}: context {payload['context_index']}", "",
            f"Context: {clipped(payload['context'])}", "", questions, "",
            "Completed annotation:", "", fenced(row["senior_annotation"]), "",
            "Senior walkthrough: The three rung labels and answer choices are preserved as released evidence, but they do not identify a complete common graph or paired states. Do not choose an option and reverse-engineer it into an SCM.",
        ]
    parts += ["", "## PubMedCausal: five completed examples", "", "The first four examples cover the released implicit/explicit and intra/inter strata; the fifth is a non-relation control. PubMedCausal sentence labels can identify candidates for source-linking, but cannot themselves establish an intervention pair."]
    for i, row in enumerate(by_source["pubmedcausal"], 1):
        payload = row["source_payload"]
        parts += [
            "", f"### PubMedCausal example {i}: {row['source_locator']}", "",
            f"Sentence: {clipped(payload.get('sentence'))}", "",
            f"Released relation: cause=`{clipped(payload.get('cause'), 180)}`, effect=`{clipped(payload.get('effect'), 180)}`, causality=`{payload.get('causality')}`, sententiality=`{payload.get('sententiality')}`.", "",
            "Completed annotation:", "", fenced(row["senior_annotation"]), "",
            "Senior walkthrough: Copy exact released spans when present. Then search only the frozen authoritative attachment for a manipulated variable, comparator/factual arm, outcome, and paired results. Without all four, reject; do not translate causal language into `do(X)` by intuition.",
        ]
    parts += [
        "", "## Annotator handoff checklist", "",
        "1. Work only from the assigned `pilot_id`; never replace a sampled item.",
        "2. Keep `graph_group_id` intact and use anonymous three-digit annotator IDs such as `ann-001`.",
        "3. Annotate CausalT5K T2 and T4 independently on the same row.",
        "4. Cite exact evidence spans and locators. Unsupported fields remain null.",
        "5. Do not inspect released answer keys while independently labelling; the production queues omit CausalT5K labels/rationales and METER correct-answer indices.",
        "6. Send `needs_adjudication` rows to a second senior reviewer; do not force an accept/reject guess.",
        "7. Never open or annotate PubMedCausal test data in this phase.",
        "8. Accepted rows still require mechanical schema, graph, span, observed-effect, group-leakage, and provenance validation before they become experimental evidence.",
        "",
        "## Files and integrity", "",
        f"- CausalT5K queue SHA-256: `{manifest['queues']['causalt5k']['sha256']}`",
        f"- METER queue SHA-256: `{manifest['queues']['meter']['sha256']}`",
        f"- PubMedCausal queue SHA-256: `{manifest['queues']['pubmedcausal']['sha256']}`",
        f"- Worked examples SHA-256: `{manifest['worked_examples']['sha256']}`",
        "- Machine-readable manifest: `reports/annotation_pilot_300_manifest.json`",
        "- Local generated queues: `data/annotation_pilots/` (intentionally ignored by Git because they contain third-party records).",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", type=Path, default=Path("data/annotation_pilots/worked_examples.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("reports/annotation_pilot_300_manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("docs/ANNOTATION_BIBLE_T2_T4_G4.md"))
    args = parser.parse_args()
    render(args.examples, args.manifest, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
