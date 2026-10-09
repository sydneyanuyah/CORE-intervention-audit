# BERT-base and BERT-large execution scope

This translates `EXPERIMENT_REGISTRY_v2` into the two encoder readers requested for the next CORE stage. The attached registry is treated as the scientific specification; this document defines how it will be executed. Yesterday's single-GPU classifier runs are engineering baselines only and do not count as staged evidence.

## Compute policy

| Reader | Default | Alternative | Decision |
|---|---:|---:|---|
| BERT-base, 110M | 4 × A16 | none | Always DDP on four GPUs |
| BERT-large, 340M | 8 × A16 | 16 × A16 | Use 16 only when a fixed-workload benchmark beats 8-GPU throughput |

Initial validation selected 8 GPUs: 22.52 examples/s versus 12.81 on 16 GPUs at the same global batch and fixed workload. See `reports/DISTRIBUTED_VALIDATION.md`.

VRAM allocation is tuned through per-GPU batch size. The optimization target is sustained examples/second without OOM, with enough headroom for variable sequence lengths—not an artificial 100% allocation reading.

## Complete scientific scope

Both readers cover all 23 registry entries: F1–F4, A1–A3, L1–L3, C1–C4, T1–T5, and G1–G4. Training entries create new checkpoints; `MEAS` entries reuse validation-selected checkpoints and therefore do not create redundant training runs. `registry/bert_scope.json` is the authoritative machine-readable matrix.

The eight paper-carrying experiments remain F2, L3, T1, A1, L1, C1, C2, and G1. The launch order is: infrastructure smoke; G3 layer selection on development only; C1; F1; F2; A1; L1; L3; then measurements T1/C2/G1 and the remaining controls, transfer, and diagnostics.

## T2/T3 addressing protocol

A1 has six explicit arms. T2-b and T3-b are the primary T2 and T3 designs;
T2-a and T3-a are named ablations and cannot be pooled with their primary arm.

| Registry name | Meaning | Applicability |
|---|---|---|
| `t0_id` | Learned intervention ID reference ceiling | XOR only; real-family cells are undefined |
| `t1_text` | Separate text encoder | All five families |
| `t2b_encode_only` | Encode-only inline command with post-edit command masking | All five families |
| `t2a_naive` | Naive inline-command ablation | All five families |
| `t3b_pointer` | Span-to-slot pointer-weighted edit | All five families |
| `t3a_conditioning` | Span-conditioned edit applied to every slot | All five families |

The primary A1 comparison remains two-edit balanced performance against T0,
including unseen paraphrases and the preregistered within-two-points rule. Since
T0 is synthetic-only, it is not fabricated for CCR.GB, CLadder, WIQA, or Com²;
those table cells stay undefined.

### Seed-count reconciliation

`T2_T3_DESIGN` gives a five-seed × five-family S1 calculation after expanding
from four to six arms. That is the architecture-screening stage, not a change
to the 20-seed confirmatory field retained in the machine-readable registry.
Counts below are per reader, before replication across BERT-base/BERT-large.

| Stage | Seeds/family | Six-arm Cartesian envelope | Defined runs | Why different |
|---|---:|---:|---:|---|
| Architecture screening | 5 | 150 | 130 | 20 T0-on-real cells are undefined |
| Confirmatory evidence | 20 | 600 | 520 | 80 T0-on-real cells are undefined |

No five-seed screen is promoted to confirmatory evidence. It may select and
mechanically validate an architecture; the frozen choice is rerun with 20
seeds per family for registered inference.

A2 now names `t2b_editor_active`, `t2b_editor_zeroed`,
`no_instruction_baseline`, and `prompting_baseline`. Its primary mechanical
masking check is zeroed T2-b versus the no-instruction baseline; active versus
zeroed is the secondary estimate of the edit's contribution.

A3 is explicitly a T3-b control: `t3b_correct_span`,
`t3b_adjacent_span`, and `t3b_random_span`. It reports correct-slot probability
mass and top-1 accuracy alongside target success, change accuracy, and
preservation. The random-span condition must collapse.

C4 counts the complete T3-b method, including its pointer query projection,
pointer key projection, and editor. Counts are measured from instantiated
BERT-base and BERT-large models rather than hard-coded, and the matched LoRA
rank is recomputed against that total instead of the older O3 total.

## Non-negotiable analysis rules

- The experimental unit is the graph, never individual records.
- F1/F2 and other paired comparisons use paired bootstrap over graphs and Student-t intervals; both are reported when they disagree.
- One primary comparison is recorded for every experiment before launch.
- Every law result includes do-nothing, do-everything, and random-init floors.
- CRASS uses question-paired McNemar accounting.
- Test sets are opened once, after validation selection; development and smoke runs cannot request test evaluation.

## Integration status

The operators are known and available on the small-model cluster server. The canonical implementation sources are:

- `${PRIVATE_STORAGE_ROOT}/CORE_closing_controls/core_components.py`: O1, O2, O3, text-conditioned O3, identity control, law losses, operator training, and evaluation.
- `${PRIVATE_STORAGE_ROOT}/CORE_joint_text_20graph/core_components.py`: joint text conditioning and addressing experiments.
- `${PRIVATE_STORAGE_ROOT}/CORE_o3g/core_components.py`: graph-masked O3 variants.
- `${PRIVATE_STORAGE_ROOT}/CORE_final_law_gate/core_components.py`: final law-gate implementation and results.

There is therefore no operator-definition or operator-availability blocker. The remaining engineering step is to consolidate those already-working server modules into the shared BERT-base/BERT-large runner and add DDP around their reader/operator training loops. Existing graph, real/placebo, law, selection, validation-candidate, and checkpoint artifacts on the server can seed that consolidation. Benchmark-specific manifests still need to preserve stable graph IDs so the registry analysis operates on graphs rather than treating records as independent observations.
