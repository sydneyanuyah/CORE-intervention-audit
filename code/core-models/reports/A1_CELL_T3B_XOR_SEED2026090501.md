# A1 screening cell: T3-b / XOR / seed 2026090501

## Status

This is the first complete end-to-end A1 screening **cell**. It is a validation-only architecture-screening measurement, not confirmatory A1 evidence and not completion of the A1 family. T0/T1 and the other registered arms/seeds are still required before an A1 comparison can be made.

## Frozen run identity

- Model: BERT-base, T3-b, 112,524,297 parameters
- Training seed: 2026090501
- GPUs: exactly four NVIDIA A16 GPUs
- Training: 8,000 authoritative XOR single-edit records; validation: 2,400
- Selected checkpoint: epoch 2 by validation macro-F1
- Checkpoint SHA-256: `not-published`
- Training elapsed time: 537.27 seconds
- Test records accessed: no

## Single-edit validation

The train wording family (`set_target_to_value`) and validation wording family (`assign_value_to_variable`) are disjoint under `xor_intervention_wording_v1`, with 100% semantic and evaluation coverage.

| Metric | Result | Count |
| --- | ---: | ---: |
| Task accuracy | 1.0000 | 2,400 |
| Macro-F1 | 0.5000 | 2,400 |
| Unseen-paraphrase accuracy | 1.0000 | 2,400 |
| Pointer top-1 | 1.0000 | 2,400 |
| Pointer mass | 1.0000 | 2,400 |
| Variable accuracy | 0.5212 | 72,000 |
| Target accuracy | 1.0000 | 2,400 |
| Changed-nontarget accuracy | 0.5077 | 19,680 |
| Preservation accuracy | 0.5035 | 49,920 |

Macro-F1 is computed over the fixed four-class task vocabulary while this XOR cell supplies only its two scalar answer classes; it must not be read as a contradiction of the 100% binary task accuracy.

## Frozen authoritative two-edit validation

The selected checkpoint was reused without training on 1,200 ordered pairs from 20 independently generated XOR graphs. Every pair and all 36,000 final variable labels were evaluated exactly once. Both address traces were measured for every pair.

| Metric | Result | Count |
| --- | ---: | ---: |
| Two-edit balanced | 0.5386 | 36,000 cells |
| Two-edit change accuracy | 0.5731 | 14,478 cells |
| Two-edit preservation | 0.5041 | 21,522 cells |
| Target success | 0.8938 | 2,400 target cells |
| Overall variable accuracy | 0.5319 | 36,000 cells |
| Pointer top-1 | 1.0000 | 2,400 edits |
| Pointer mass | 1.0000 | 2,400 edits |

The cell is mechanically complete: training, frozen composed evaluation, unseen wording, target/change/preservation groups, pointer metrics, artifact hashes, distributed provenance, and checkpoint identity are all present. Its result does not yet pass or fail A1 because the registered T0 baseline and remaining architecture cells have not been run.

