# A1 canonical-result reconciliation

**Disposition:** resolved reporting-provenance defect.

The former A1 table in `CORE_RESEARCH_REPORT.md` did not match the package-local canonical aggregate. It has been replaced from `A1_CONFIRMATORY.json` (SHA-256 `not-published`). The repaired confirmatory template is `registry/a1_repaired_confirmatory_template.json` (SHA-256 `not-published`).

| Family | Canonical T3-b balanced | Change | Preservation | Target success | Pointer top-1 |
|---|---:|---:|---:|---:|---:|
| XOR | 0.531522 | 0.565569 | 0.497475 | 0.895063 | 1.000000 |
| CLadder | 0.499019 | 0.486372 | 0.511667 | 0.496124 | 0.263577 |
| WIQA | 0.702173 | 0.425135 | 0.979211 | 0.230000 | 0.966071 |

The canonical XOR paired delta versus T0 is +0.010427, with Student-t 95% interval [+0.007658, +0.013195] and paired-graph bootstrap [+0.007676, +0.013067].

The validation CCR.GB T3-b row is omitted from the scientific A1 table. Its balanced score is 0.503147, but change accuracy is 0.015385 against preservation 0.990909 and target success 0.016667. This is a no-op signature, not efficacy evidence.

## CLadder pointer protocols

The 0.263577 confirmatory value is computed by the composed two-edit evaluator over 615 composed examples and 20 graph units. The 0.996988 held-out value is computed by `src/evaluate_a1_final.py`, which calls the single-edit `evaluate_exact` path over 166 sealed rows per seed. A3's 1.000000 correct-address value comes from a third 147-row single-view control artifact. These are different evaluation protocols; the discrepancy is evaluator-path sensitivity and must not be interpreted as ordinary validation-to-test improvement.

## Open-text pointer audit

The perfect pointer diagnostic in the excluded open-text arm is not an empty-denominator calculation: the hash-bound extract in `reports/evidence/a1_open_text_pointer_audit_seed218.json` records 73 eligible pointer rows with full coverage. However, every task and variable score is zero, so the arm remains outside scientific interpretation and is absent from the research report's A1 table.
