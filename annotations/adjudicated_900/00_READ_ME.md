# CORE adjudicated annotation package

This folder contains the senior adjudication of the two independent annotation result packages found in Downloads.

## Start here

- `ADJUDICATION_REPORT.md`: concise outcome, agreement, accepted IDs, and mandatory repair.
- `ADJUDICATION_REPORT.json`: machine-readable report, input/output hashes, and exact counts.
- `adjudication_audit.jsonl`: every decision disagreement, every accepted view whose structure was canonicalized, and every same-decision acceptance overridden by the senior adjudicator.
- `accepted_task_views.jsonl`: the 12 accepted annotation candidates only.
- `causalt5k.adjudicated.jsonl`, `meter.adjudicated.jsonl`, and `pubmedcausal.adjudicated.jsonl`: all 900 source rows with final consensus task views.
- `REPRODUCE_ADJUDICATION.py`: deterministic script used to build this package from the frozen handoff and two result folders.

## Outcome

- CausalT5K T2: 9 accepted, 291 rejected.
- CausalT5K T4: 3 accepted, 297 rejected.
- METER T4: 0 accepted, 300 rejected.
- PubMedCausal G4: 0 accepted, 300 rejected.
- Total: 12 accepted task views across 9 unique source rows.

These are accepted annotation candidates, not yet experiment-ready CORE records. They still require normalization, exact target-span construction, schema validation, leakage-group repair, and split generation.

## Important audit notes

The two CausalT5K result files both identify themselves as `ann-001`. They are treated as distinct submissions by package path and SHA-256, not by the colliding annotator label. Final consensus rows use `ann-999`; the senior adjudicator is recorded as `adj-001` in the report.

The first result folder's pre-existing self-adjudicated files were not treated as an independent annotation. Only its raw `ann-001` submissions and the second result folder's raw submissions were compared.

`C5K-200` and `C5K-208` share the frozen group ID `causalt5k:case:0148` despite belonging to different domains and unrelated cases. Before normalization or splitting, the group identity must include the release domain. The frozen adjudicated source rows are intentionally not silently mutated.

No PubMedCausal held-out test data were accessed.
