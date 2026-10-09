# CORE BERT baseline results

Date: 2026-09-04

## Task

The encoder baseline predicts four closed labels after an intervention: `no`, `yes`, `less`, and `more`. It uses CLadder, WIQA, and CCR.GB. Com2 is excluded because its 770 records use open-text answers with 763 distinct strings, which are not a valid fixed-class BERT target.

| Split | Records | CCR.GB | CLadder | WIQA |
|---|---:|---:|---:|---:|
| Train | 20,963 | 96 | 1,109 | 19,758 |
| Validation | 4,706 | 12 | 147 | 4,547 |
| Test | 1,918 | 12 | 166 | 1,740 |

Combined data fingerprint: `not-published`.

## Environment

- Host: `[private compute host]`
- GPU: NVIDIA A16, 15,356 MiB
- Python: 3.12.3
- PyTorch: 2.14.0+cu130
- Transformers: 5.16.1
- Precision: BF16
- Seed: 20260904
- Maximum sequence length: 384
- Class weighting: inverse-frequency across the four training labels

## BERT-base

- Model: `google-bert/bert-base-uncased`
- Pinned model revision: `not-published`
- Parameters: 109,485,316
- Batch size: 16
- Learning rate: 2e-5
- Epochs: 3
- Training time: 1,719.5 seconds
- Best checkpoint: epoch 3
- Checkpoint size: 437,964,776 bytes
- Checkpoint SHA-256: `not-published`

| Stage | Accuracy | Macro-F1 |
|---|---:|---:|
| Untouched validation baseline | 10.39% | 0.0756 |
| Epoch 1 validation | 54.23% | 0.4561 |
| Epoch 2 validation | 49.98% | 0.3367 |
| Epoch 3 validation | **56.23%** | **0.4642** |
| Held-out test, selected checkpoint | **56.26%** | **0.4919** |

Held-out test by source:

| Source | Records | Accuracy | Present-label macro-F1 |
|---|---:|---:|---:|
| CCR.GB | 12 | 75.00% | 0.4286 |
| CLadder | 166 | 52.41% | 0.3439 |
| WIQA | 1,740 | 56.49% | 0.5540 |

The test set was evaluated once, after the checkpoint was selected entirely on validation.

## BERT-large scale check

- Model: `google-bert/bert-large-uncased`
- Pinned model revision: `not-published`
- Parameters: 335,145,988
- Batch size: 16
- Learning rate: 1e-5
- Epochs completed: 1
- Training/evaluation time: 1,772.9 seconds
- Peak observed training memory: approximately 11.2 GiB
- Checkpoint size: 1,340,630,912 bytes
- Checkpoint SHA-256: `not-published`

| Stage | Accuracy | Macro-F1 |
|---|---:|---:|
| Untouched validation baseline | 46.35% | 0.1623 |
| Epoch 1 validation | 50.11% | 0.3358 |

BERT-large did not beat the BERT-base validation result, so its held-out test set was not opened. Additional raw parameter scaling is paused until the task design is improved.

## Judgment and next experiment

The first real model-training implementation is complete. BERT-base is the current selected baseline. The major limitation is that the label space is source-partitioned and highly imbalanced: `yes/no` occur in CLadder and CCR.GB, while `more/less` occur in WIQA. The next useful experiment is a shared encoder with source-specific classification heads or balanced source batches, compared on validation. Only after that design beats the BERT-base validation macro-F1 should BERT-large receive additional epochs or the test set be evaluated.

Remote project: `${CORE_PROJECT_ROOT}`

Local project: `${USER_HOME}/Documents/core-models`
