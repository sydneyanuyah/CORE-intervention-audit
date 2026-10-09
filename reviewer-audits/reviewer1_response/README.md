# CORE Reviewer 1 response artifacts

Generated 2026-09-14 from frozen checkpoints and existing result artifacts. No model was trained and no held-out test data was opened in this pass.

## Deliverables

- `CORE_REVIEWER1_SELECTIVE_INVARIANCE_PER_SEED.csv`: 1,640 checkpoint/method/seed rows spanning BERT-base, Qwen2.5-7B, Phi-4, Qwen2.5-32B, and Llama-3.3-70B. It reports non-descendant preservation, descendant/change sensitivity, and their balanced mean.
- `CORE_REVIEWER1_SELECTIVE_INVARIANCE.json`: the same per-seed records plus macro means and Student-t 95% intervals. L1 floors and every registered L2/L3 arm are included.
- `CORE_REVIEWER1_F2_UNCERTAINTY.csv` and `.json`: matched O2−O3 seed deltas, Student-t 95% intervals, and 10,000-draw equal-graph bootstrap intervals. The JSON retains every source cell and checkpoint hash.
- `CORE_REVIEWER1_DUPLICATE_PROVENANCE_AUDIT.csv` and `.json`: F2→T1/C3/C4 lineage, metric paths, cell IDs, checkpoint hashes, and L3 equality checks.
- `CORE_REVIEWER1_TRAINING_CONFIGURATION.json`: source-bound registry snapshots, normalized recipes, and the exact locked-test baseline definition.
- `raw/`: unchanged machine outputs from small-model cluster and large-model cluster.
- `*.py`: executable extraction/rescoring scripts used for this response.

## Findings to use in the revision

1. Selective invariance is now measured explicitly. For BERT-base L1 full-four-law, the balanced score is 0.4906 on XOR (95% CI 0.4829–0.4983) and 0.5187 on CLadder (0.5030–0.5344). In L2, removing selective invariance is essentially indistinguishable from the full arm: XOR 0.49377 versus 0.49383 and CLadder exactly 0.51356 versus 0.51356 at exported precision. This fills the missing column but does not support a beneficial selective-invariance law effect.
2. F2 CLadder O2−O3 is positive for BERT-base: +0.12918, Student-t 95% CI [0.10466, 0.15370], graph-bootstrap [0.11645, 0.16520]. Across larger backbones: Qwen-7B +0.02774 with both intervals above zero; Phi-4 +0.00451 with both intervals crossing zero; Qwen-32B +0.00962 with the graph-bootstrap crossing zero; Llama-70B +0.02456 with both intervals above zero. Accordingly, only BERT-base, Qwen-7B, and Llama-70B support a statistically resolved CLadder advantage under both requested intervals.
3. T1, C3 full-O3, and C4 performance are derived from F2 cells and are not independent confirmations. Every non-Com2 source hash verifies. The 100 Com2 C3/C4 lineage hashes are stale after the later repair/replacement and remain excluded.
4. The claimed exact L3 identities are not present at artifact precision. LoRA/LoReFT/O2 and O1/task-vector share some exact scalar values, especially degenerate law metrics, but not all common scalars across seeds. Describe displayed repetitions as rounded numerical ties, not exact duplicate runs or independent evidence.
5. The exact locked-test terminology is now resolved: T2 has no separately registered baseline arm; T4 has three O3 arms and no baseline; T5 `baseline` is a trained BERT-base candidate scorer with learned state tokens and head that bypasses O3. It is not a zero-shot language-model baseline.

## Scope and interpretation

The BERT-base F2 CSV retains CCR.GB, WIQA, and Com2 re-analysis for audit completeness, but only CLadder is marked paper-eligible. Com2 remains excluded; CCR.GB remains quarantined; WIQA lacks the graph count needed for a strong graph-level headline. Selective-invariance values are descriptive checkpoint measurements, not independent new training runs.
