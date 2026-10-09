# CORE experiments after measurement audit

**Corrected evidence snapshot:** 2026-09-07  
**Companion workbook:** `reports/CORE_RESULTS_TO_DATE.xlsx`  
**Current scientific verdict:** **The operator hypothesis lost where the measurement worked, and where the laws were tested the measurement was broken.**

This report supersedes the earlier positive interpretation of L1, G3, A2, and F1. Those runs produced real files and numeric outputs, but they do not provide valid evidence for the intended scientific claims. Their raw results remain in the Excel workbook solely for auditability and are explicitly marked `VOID — DO NOT REPORT`.

## Defensible findings

| Finding | Evidence | Result | Interpretation |
|---|---|---|---|
| **A3 pointer mechanism** | 486 graphs; correct, adjacent, and random address conditions | Pointer top-1 fell from **0.9707** with the correct address to **0.1702** with a random address. Task accuracy fell from **0.4981** to **0.3164**. | The pointer address causally affects the prediction path. This mechanism result stands. |
| **F2 on CLadder** | 615 examples; O2 and O3 comparison | O2 scored **0.756197**, O3 scored **0.627018**. O2 beat O3 by **0.129180**, and the interval excludes zero. | The working measurement contradicts the operator hypothesis: the HyperSteer-style O2 baseline is better. |

There is currently no result supporting the claim that the proposed operator obeys the laws and therefore wins.

## Void results — do not report as findings

| Evidence | Disposition | One-line reason |
|---|---|---|
| **L1** | **VOID** | It measured constant-predictor behavior, so it says nothing about law use or law retention. |
| **G3** | **VOID; layer 4 unfrozen** | Layers 2, 7, 8, 9, and 10 tied at `0.5385110419881652`; the selected layer was arbitrary because the edit was not affecting the prediction. |
| **A2** | **VOID as a positive control result** | Editor-on and editor-off were identical in four of five datasets; this diagnoses an inactive editor path, not a successful masking result. |
| **F1** | **VOID** | It was an XOR-only noise result, and the broader F2 result reversed its apparent direction. |
| **All CCR.GB outputs** | **VOID** | The evaluation has only 12 examples across two graphs, which is not adequate evidence. |
| **All Com2 outputs** | **VOID** | Every method scored exactly zero, indicating a broken loader or evaluation path. |

The numeric outputs from these runs remain in the workbook as an audit record. Their presence is not an endorsement, a pass, or a scientific conclusion.

## Why the editor-path results are invalid

The one-law, three-law, and four-law runs produced different checkpoint files, but 39 of 40 comparisons produced exactly the same score to eight decimal places. G3 likewise returned the identical 16-digit score for five different edit layers. A2 independently showed that disabling the editor changed nothing in four of five datasets.

The editor-output amplification test below shows a more precise failure mode: the edited tensor reaches the logits, but even a 1000× perturbation does not change a single task or variable decision. Changing laws, checkpoints, or edit layers therefore does not produce a meaningful measured intervention under the current path.

## Editor-output amplification diagnostic

One frozen L1 XOR four-law checkpoint was scored twice on the same 120 validation examples: once normally and once with the emitted editor tensor multiplied by **1000**. The checkpoint and validation rows were unchanged.

| Quantity | Baseline | 1000× editor output | Change |
|---|---:|---:|---:|
| Task accuracy | 0.508333 | 0.508333 | 0.000000 |
| Task macro-F1 | 0.168508 | 0.168508 | 0.000000 |
| Task predictions changed | — | — | **0 / 120** |
| Variable predictions changed | — | — | **0 / 3,600** |
| Task loss | 0.773001 | 0.769100 | −0.003902 |
| Variable loss | 0.792714 | 0.787780 | −0.004934 |

The small loss movement proves that the scaled tensor is technically connected to downstream logits. The complete absence of changed decisions proves that the editor is decision-inert under this evaluation, even at 1000× scale. This confirms the substantive measurement failure while refining “disconnected” to “connected but ineffective at the decision boundary.” No additional operator or law experiment should be interpreted until this path is repaired. Layer 4 remains unfrozen.

## Experiment ledger

| Experiment | Execution state | Scientific state |
|---|---|---|
| A1 | Architecture screening and confirmatory cells completed | Retained as execution context only; it does not establish a working editor path. |
| A2 | 400/400 measurements completed | Void as a positive result; retained as evidence that the editor path was inactive. |
| A3 | Three mechanism controls completed | **Stands.** |
| F1 | 40/40 formal cells completed | Void; XOR noise. |
| F2 | 400/400 formal cells completed | **Only CLadder stands.** CCR.GB and Com2 are void; no other F2 claim is made here. |
| F3 | Formal result not established | Pause interpretation pending the editor-output amplification test. |
| F4 | Formal result not established | Pause interpretation pending the editor-output amplification test. |
| G3 | 11/11 layer measurements completed | Void; layer 4 unfrozen. |
| L1 | 240/240 formal cells completed; 1000× diagnostic completed | Void; the scale test changed zero task and variable decisions. |
| L2 | 320/320 registered cells and aggregate completed | Across eight family-by-dropped-law comparisons, 0/8 paired-seed intervals exclude zero; no positive law-retention evidence. |
| L3 | 40/40 registered cells completed | No method showed a per-law improvement over matched random initialization whose Student-t 95% interval excluded zero. |
| C1, C2, C3, C4 | Not completed as formal evidence | No result. |
| T1, T2, T3, T4, T5 | Not completed | No result. |
| G1, G2, G4 | Not completed | No result. |

## Excel workbook contents

The workbook contains the full evidence archive plus the amplification diagnostic: 1,842 canonical source artifacts, 1,834 artifact-level summaries, 190,347 scalar metric records, 290,220 graph/example records, and 5,942 aggregate-report metrics.

| Sheet | Contents |
|---|---|
| `Summary` | The corrected standing/void disposition and the two defensible findings. |
| `Cell Results` | One row per result artifact, with `evidence_disposition` and `disposition_reason` preceding the raw fields. |
| `Unit Evidence` | Every available per-graph/per-example record in lossless NDJSON chunks. |
| `Scalar Evidence` | Every scalar JSON leaf in lossless NDJSON chunks. |
| `Aggregate Metrics` | Scalar values imported from the canonical aggregate reports. Void-report values are retained only for auditing. |
| `Sources` | Artifact path, SHA-256, size, protocol, test flag, and corrected evidence disposition. |
| `ReadMe` | Definitions, scope, and reconstruction instructions. |

For either NDJSON evidence sheet, group rows by `artifact_id`, order by `chunk_index`, concatenate `ndjson_chunk` values with a newline, and parse one JSON object per line. This preserves every recorded field while keeping the Excel file usable.

## Scope and provenance

- No selected artifact records `test_evaluated=true`.
- Raw values are never silently deleted or rewritten because a result is void. Instead, the workbook adds an explicit scientific disposition while preserving the source value, source path, checksum, protocol, seed, checkpoint, and cell identity.
- Superseded plumbing smokes, rejected unregistered launches, and legacy nonformal C1 artifacts remain excluded.
- The defensible publication options are now: fix the editor path and re-test the laws, or pivot the submission around A3 and the evaluation critique.
