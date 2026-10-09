# CORE Reviewer 2 response artifacts

Generated 2026-09-15. This pass reused frozen checkpoints and validation artifacts only. It performed no model training and opened no held-out test data.

large-model cluster GPU execution was repeated under Slurm after the cluster administrator rejected direct GPU-process execution. Formal Qwen-32B and Llama-70B results come from completed `LocalQ` jobs 6207 and 6208. Their output files are byte-identical to the preliminary copies; only the Slurm-produced copies are retained as evidence.

## Already complete — not rerun

- Selective-invariance export: the Reviewer 1 package contains 1,640 per-seed/model/method rows and macro Student-t 95% intervals.
- Training metadata: the Reviewer 1 package contains the source-bound configuration export and exact T2/T4/T5 baseline definitions.
- F2 and law-residual confidence intervals: already extracted previously; Reviewer 2 explicitly said not to redo them.

Those files remain in `${USER_HOME}/Downloads/CORE_REVIEWER1_RESPONSE_2026-09-14/`.

## New measurement 1 — oracle residual calibration

`CORE_REVIEWER2_ORACLE_RESIDUAL_CALIBRATION.json` executes the frozen XOR and CCR.GB structural equations on validation data. For both families:

- identity residual = 0; exact correctness = 1.000
- idempotence residual = 0; exact correctness = 1.000
- commutation residual = 0; exact correctness = 1.000

Thus the executable truth and law-measurement machinery are capable of returning the expected zero-residual oracle result. Model nulls cannot be attributed to a non-zero oracle floor.

## New measurement 2 — A3 address inversion

`CORE_REVIEWER2_A3_ADDRESS_INVERSION_PER_SEED.csv` contains 80 frozen-checkpoint rows: 20 seeds for each of Qwen-7B, Phi-4, Qwen-32B, and Llama-70B. Two positive controls were measured: a deterministic wrong target and an inverted command value on the correct target.

Mean perturbation minus correct-address results:

| Backbone | Wrong-target balanced Δ | Wrong-target target-success Δ | Inverted-value balanced Δ | Inverted-value target-success Δ |
|---|---:|---:|---:|---:|
| Qwen-7B | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| Phi-4 | -0.005095 | -0.032639 | -0.005639 | -0.066667 |
| Qwen-32B | -0.000468 | -0.009444 | -0.001336 | -0.018889 |
| Llama-70B | +0.000018 | -0.000833 | -0.000106 | -0.001667 |

Only Phi-4 has target-success intervals excluding zero for both perturbations; its inverted-value balanced-accuracy interval also excludes zero. Qwen-7B is exactly decision-inert. Qwen-32B and Llama-70B effects are unresolved and near zero. This does not support a general four-backbone claim that address changes reliably control decisions.

## Files

- `CORE_REVIEWER2_ORACLE_RESIDUAL_CALIBRATION.json`
- `CORE_REVIEWER2_A3_ADDRESS_INVERSION.json`
- `CORE_REVIEWER2_A3_ADDRESS_INVERSION_PER_SEED.csv`
- `raw/`: unmodified per-server outputs
- measurement and consolidation scripts
- `SLURM_EXECUTION.json` and `slurm-logs/`: scheduler provenance for large-model cluster jobs 6207 and 6208
- `SHA256SUMS.json`
