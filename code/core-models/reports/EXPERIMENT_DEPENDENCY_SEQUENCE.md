# CORE experiment sequence and direct dependencies

This table follows the registered Experiment 1–23 numbering. “Yes” means the experiment can start without waiting for another experiment’s output, provided its own runner and data are ready. Dependencies are direct checkpoint, measurement, or registered queue dependencies; ordinary shared infrastructure is not counted.

| # | Experiment | Can run without a previous experiment? | Direct prerequisite | Run and works well? | Why / current evidence |
|---:|---|---|---|---|---|
| 1 | F1 | Yes | None | **Run — No; void** | XOR-only noise; broader F2 reversed its apparent direction |
| 2 | F2 | No | F1 | **Run — Partly; CLadder stands** | On CLadder, O2 0.756197 beat O3 0.627018 by 0.129180; CCR.GB and Com2 results are void |
| 3 | F3 | No | F1 | Partially run — no formal result | Only partial artifacts exist; no defensible aggregate result |
| 4 | F4 | No | F1 | Partially run — no formal result | Only partial artifacts exist; no defensible aggregate result |
| 5 | A1 | Yes | None | Run — not accepted as a working-editor result | Execution completed, but it does not establish an effective editor path |
| 6 | A2 | No | A1 | **Run — No; void** | Editor-on and editor-off were identical in four of five datasets |
| 7 | A3 | No | A1 | **Run — Yes; stands** | Pointer top-1 fell 0.9707→0.1702 and task accuracy 0.4981→0.3164 under random addressing |
| 8 | L1 | Yes | None | **Run — No; void** | Constant-predictor behavior; 1000× editor scaling changed zero decisions |
| 9 | L2 | No | L1 | **Complete — aggregate null** | 320/320 cells complete; 0/8 dropped-law primary intervals exclude zero; no test access |
| 10 | L3 | No | F1 | **Run — complete; no significant law gain** | 40/40 cells complete; no method/law Student-t 95% interval excludes zero versus matched random initialization |
| 11 | C1 | No | F1 | Development/legacy runs only — no formal result | Formal fixed-O3 real/noncausal/shuffled execution has not completed |
| 12 | C2 | No | F1 | Not completed — no result | Floor data are ready; trained-O3 calibration is not completed |
| 13 | C3 | No | F2 | Not completed — no result | F2 checkpoint-bound editor-zeroed measurement remains |
| 14 | C4 | No | F2 | Not completed — no result | Parameter-accounting measurement remains |
| 15 | T1 | No | F2 | Not completed — no result | F2 composed-pair export/measurement remains |
| 16 | T2 | Yes | None | Not run — no result | CSuite data are ready; model execution has not started |
| 17 | T3 | No | A1 | Not completed — no result | Perturbation data exist; checkpoint-bound measurement remains |
| 18 | T4 | Yes | None | Not run — no result | Rung-view data are ready; model execution has not started |
| 19 | T5 | No | F1 | Not run — no result | Native-task data are ready; baseline/O3 evaluation has not started |
| 20 | G1 | No | F1 | Not completed — no result | CLadder estimand evaluator remains |
| 21 | G2 | No | F1 | Not run — no result | 120,000 sequences are ready; operator measurement has not started |
| 22 | G3 | Yes | None | **Run — No; void** | Five layers tied exactly; layer 4 is unfrozen |
| 23 | G4 | Yes | None | Not run — no result | Evidence Inference data are ready; baseline/O3 training has not started |

## Runnable waves

| Wave | Experiments | Condition |
|---:|---|---|
| 0 | F1, A1, L1, T2, T4, G3, G4 | No previous experiment output required |
| 1 | F2, F3, F4, A2, A3, L2, L3, C1, C2, T3, T5, G1, G2 | Their Wave 0 prerequisite is complete |
| 2 | C3, C4, T1 | F2 is complete |

## small-model cluster folder mapping

The server root is `${CORE_PROJECT_ROOT}/experiments`. It contains exactly these numbered folders:

`Experiment-1-F1` through `Experiment-23-G4`, following the table above.

Each experiment folder now contains `data/train`, `data/validation`, and
`data/test`, plus a machine-readable `data/DATA_MANIFEST.json`. Splits not used
by the registered experiment carry `NOT_REQUIRED.txt`; copied held-out test
splits carry `LOCKED_DO_NOT_EVALUATE.txt`. Each experiment root also carries
exactly one of `DATA_COMPLETE.txt` or `DATA_INCOMPLETE.txt`.

The row counts and outstanding data gaps are recorded in
`reports/EXPERIMENT_DATA_SPLIT_AUDIT.md` and mirrored on small-model cluster at
`experiments/DATA_SPLIT_AUDIT.md`.
