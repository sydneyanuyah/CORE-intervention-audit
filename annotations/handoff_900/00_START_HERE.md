# CORE annotation handoff — start here

This folder is the complete annotation handoff. It contains **three separate 300-item production queues** (900 production annotations total) and **five additional worked examples per dataset** (15 teaching examples, excluded from production).

## File order

1. Read `01_ANNOTATION_BIBLE.md` first. It contains the completed real-data examples and senior-annotator walkthroughs.
2. Use `02_FULL_ANNOTATION_GUIDE.md` for the complete T2/T4/G4 evidence contract, reason codes, graph rules, world-state rules, and adjudication rules.
3. Review `20_WORKED_EXAMPLES_15.jsonl`, the machine-readable form of the five examples per dataset.
4. Annotate an assigned copy of the appropriate production file:
   - `10_CAUSALT5K_PRODUCTION_300.jsonl`
   - `11_METER_PRODUCTION_300.jsonl`
   - `12_PUBMEDCAUSAL_PRODUCTION_300.jsonl`
5. Run `python3 40_VALIDATE_SUBMISSION.py ORIGINAL_QUEUE ANNOTATED_QUEUE` before submission.

## What annotators edit

Only edit objects inside `task_views`. Do not change `pilot_id`, `source`, `source_locator`, `graph_group_id`, or `source_payload`.

- CausalT5K: complete **both** `task_views.T2` and `task_views.T4` independently.
- METER: complete `task_views.T4`.
- PubMedCausal: complete `task_views.G4`.

Use an anonymous three-digit annotator ID such as `ann-001`. Valid decisions are `accept`, `reject`, and `needs_adjudication`.

For `accept`, provide a source-supported graph, factual state, surgical intervention, intervened state, and exact evidence spans. For `reject`, leave unsupported structures null and provide one or more approved reason codes. For `needs_adjudication`, provide the reason codes, evidence already found, and a precise note describing the unresolved question.

## Non-negotiable evidence rule

Do not invent edges, values, interventions, counterfactual outcomes, or missing evidence. Causal wording, answer options, and cause/effect spans do not by themselves define paired worlds. A rejection is a valid annotation outcome.

PubMedCausal test data are outside this package and must not be opened. The production queue contains train/validation candidates only.

## Independent annotation and adjudication

Each production item must be completed independently by two annotators. Give each annotator a fresh copy of the unchanged production queue. Do not let the second annotator see the first annotation. Differences are reconciled in the adjudication template after both submissions pass the validator.

Do not overwrite the three original production queues. Recommended submission names are:

- `causalt5k.ann-001.jsonl`
- `meter.ann-001.jsonl`
- `pubmedcausal.ann-001.jsonl`

## Integrity

`90_SAMPLE_MANIFEST.json` records the selection design, counts, and three-digit selection seed `317`.
