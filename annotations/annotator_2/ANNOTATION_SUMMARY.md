# CORE Annotation Results — 900 Production Rows

## Completed files

- `causalt5k.ann-001.jsonl`: 300 rows; both T2 and T4 annotated per row (600 task-view decisions).
- `meter.ann-002.jsonl`: 300 rows; T4 annotated per row, preserving 100 three-row context groups.
- `pubmedcausal.ann-003.jsonl`: 300 rows; G4 annotated per row. Held-out test data was not accessed.

## Decision summary

- CausalT5K: 3 source rows accepted in both T2 and T4 (6 accepted task views); 297 rows rejected in both views (594 rejected task views).
- METER: 300 rejected T4 views.
- PubMedCausal: 300 rejected G4 views.

Strict evidence rules from the handoff were applied: causal wording, options, association language, or released cause/effect spans were not promoted into missing graphs or paired worlds. Unsupported structured fields remain null.

## Manual review records

The three `.notes.md` files record row-range coverage and difficult or borderline decisions for each dataset.
