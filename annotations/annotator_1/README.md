# CORE annotation pilot — ann-001 results

Annotation of the 900-item production handoff in `CORE_Annotation_Handoff_900`:
300 CausalT5K rows (independent T2 and T4 forms), 300 METER rows (T4), 300 PubMedCausal
rows (G4). 1,200 task views in total, all completed.

## Files

- `ANNOTATOR_BRIEF.md` — the evidence contract the pass was worked to, drawn from
  `01_ANNOTATION_BIBLE.md` and `02_FULL_ANNOTATION_GUIDE.md`.
- `submissions/*.ann-001.jsonl` — the annotator's pass as submitted, unmodified. Use these
  as the first of the two independent annotations the protocol requires.
- `submissions/*.adjudicated.jsonl` — the same rows with the senior adjudicator's rulings
  applied. The ruling is appended to the note of each affected task view.
- `annotations/<source>.ledger.jsonl` — one ledger row per task view, including every
  rejection, with decision, reason codes, evidence spans and notes.
- `annotations/adjudication.jsonl` — the 44 contested task views, in the shape of
  `31_ADJUDICATION_TEMPLATE.json`, with the auditor's position and the final ruling.
- `reports/annotation_summary.json` — counts, reason-code distributions, mechanical checks.
- `reports/QC_AUDIT_REPORT.md` — what the audit found, what was overturned, and the data
  defects worth referring back to the queue owner.
- `reports/SHA256SUMS.txt` — checksums for everything here.

## Outcome

| | task views | accept | reject |
|---|---|---|---|
| CausalT5K (T2 + T4) | 600 | 12 | 588 |
| METER (T4) | 300 | 0 | 300 |
| PubMedCausal (G4) | 300 | 0 | 300 |

No unresolved adjudications remain. `40_VALIDATE_SUBMISSION.py` passes on all six queue files;
the frozen source fields are byte-identical to the originals, and no PubMedCausal test row was
opened.

## Before these rows become experimental evidence

This is one independent pass. Every accepted record still needs a second blind annotation and
adjudication of any disagreement, then schema, span, observed-effect, leakage and provenance
validation, per Section 8 of the full guide. Give the second annotator a fresh copy of the
original production queue, not these files.
